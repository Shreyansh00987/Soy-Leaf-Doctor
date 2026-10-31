"""
Automated Experiment Training & Cross-Validation Runner for Soy Leaf Doctor (HARITVA).
Evaluates architectures, resolutions, losses, and schedules strictly against the 5-fold CV protocol.
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models_extended import CLASSES, NUM_CLASSES, GenericSoyClassifier, save_fp16_weights, load_weights_safe
from src.dataset import SoyLeafDataset, get_train_transforms, get_eval_transforms


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(df: pd.DataFrame, beta: float = 0.999) -> torch.Tensor:
    counts = df["label"].value_counts().to_dict()
    weights = []
    for cls in CLASSES:
        n_c = counts.get(cls, 1)
        w = (1.0 - beta) / (1.0 - (beta ** n_c))
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def eval_fold(model, loader, device, use_tta: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            imgs = images.to(device)
            if use_tta:
                p = model.predict_tta(imgs).cpu().numpy()
            else:
                p = model.predict_probs(imgs).cpu().numpy()
            preds = np.argmax(p, axis=1)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.append(p)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)
    mf1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    return float(mf1), float(acc), all_probs, all_targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, required=True, help="e.g. Exp1_Res288")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large")
    parser.add_argument("--resolution", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss-type", type=str, default="class_balanced", choices=["ce", "class_balanced", "focal"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--quick-folds", type=int, default=5, help="Number of folds to evaluate (1 to 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"=== Starting Experiment: {args.exp_id} ===", flush=True)
    print(f"Backbone: {args.backbone} | Res: {args.resolution}x{args.resolution} | Epochs: {args.epochs} | Loss: {args.loss_type} | Smooth: {args.label_smoothing}", flush=True)

    df_splits = pd.read_csv("data/splits_5fold.csv")
    img_dir = "data/p4-soy-leaf-health/images"

    train_tf = get_train_transforms(image_size=args.resolution)
    eval_tf = get_eval_transforms(image_size=args.resolution)

    cv_folds = [f"fold_{i}" for i in range(args.quick_folds)]
    fold_metrics = []
    all_oof_records = []

    os.makedirs("models/experiments", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    for fold in cv_folds:
        f_val = df_splits[df_splits["split"] == fold].reset_index(drop=True)
        # Training data = all other CV folds (holdout is NEVER included!)
        f_trn = df_splits[(df_splits["split"] != "holdout") & (df_splits["split"] != fold)].reset_index(drop=True)

        trn_ds = SoyLeafDataset(f_trn, img_dir, transform=train_tf)
        val_ds = SoyLeafDataset(f_val, img_dir, transform=eval_tf)

        # Sampler or standard loader
        if args.use_sampler:
            cls_counts = f_trn["label"].value_counts().to_dict()
            sample_weights = [1.0 / cls_counts[row["label"]] for _, row in f_trn.iterrows()]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        else:
            trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Loss setup
        if args.loss_type == "class_balanced":
            class_weights = compute_class_weights(f_trn).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        elif args.loss_type == "focal":
            class_weights = compute_class_weights(f_trn).to(device)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Initialize model
        model = GenericSoyClassifier(backbone_name=args.backbone, num_classes=NUM_CLASSES, pretrained=True)
        model.to(device)

        # Stage 1: Warmup with frozen feature extractor (1 epoch)
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
        opt_warmup = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 2.0)
        train_epoch(model, trn_loader, opt_warmup, criterion, device)

        # Stage 2: Unfreeze full network with layer-wise LR
        for p in model.feature_extractor.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                {"params": model.feature_extractor.parameters(), "lr": args.lr * 0.25},
                {"params": model.model.classifier.parameters() if hasattr(model.model, "classifier") else model.model.fc.parameters(), "lr": args.lr},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 1), eta_min=1e-6)

        best_val_f1 = 0.0
        best_state = None

        for ep in range(1, args.epochs):
            trn_loss = train_epoch(model, trn_loader, optimizer, criterion, device)
            scheduler.step()
            val_f1, val_acc, _, _ = eval_fold(model, val_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Load best checkpoint for final evaluation with 4-view TTA
        model.load_state_dict(best_state)
        f_mf1, f_acc, oof_probs, oof_targets = eval_fold(model, val_loader, device, use_tta=True)
        fold_metrics.append({"fold": fold, "macro_f1": f_mf1, "accuracy": f_acc})
        print(f"[{args.exp_id}] {fold} -> Accuracy: {f_acc*100:.2f}%, Macro-F1: {f_mf1:.4f}", flush=True)

        for i in range(len(f_val)):
            rec = {
                "image_id": f_val.iloc[i]["image_id"],
                "fold": fold,
                "true_label": f_val.iloc[i]["label"],
                "pred_label": CLASSES[int(np.argmax(oof_probs[i]))],
            }
            for c_idx, c_name in enumerate(CLASSES):
                rec[f"p_{c_name}"] = float(round(oof_probs[i][c_idx], 5))
            all_oof_records.append(rec)

    # Aggregate CV results
    mean_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
    std_f1 = float(np.std([m["macro_f1"] for m in fold_metrics]))
    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    std_acc = float(np.std([m["accuracy"] for m in fold_metrics]))

    df_oof = pd.DataFrame(all_oof_records)
    oof_path = f"reports/{args.exp_id}_oof_predictions.csv"
    df_oof.to_csv(oof_path, index=False)

    # Per class F1
    true_labels = df_oof["true_label"].values
    pred_labels = df_oof["pred_label"].values
    per_class_f1 = f1_score(true_labels, pred_labels, labels=CLASSES, average=None)
    f1_dict = {cls: round(float(per_class_f1[i]), 4) for i, cls in enumerate(CLASSES)}

    # Measure CPU latency per image under 2 threads
    torch.set_num_threads(2)
    sample_x = torch.randn(1, 3, args.resolution, args.resolution)
    for _ in range(5):
        _ = model.predict_tta(sample_x)
    t0 = time.time()
    for _ in range(15):
        _ = model.predict_tta(sample_x)
    t1 = time.time()
    latency_ms = ((t1 - t0) / 15.0) * 1000.0

    # Save FP16 weights of the final model
    weights_path = f"models/experiments/{args.exp_id}.pt"
    save_fp16_weights(model, weights_path)
    model_size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2)

    print("\n" + "=" * 65, flush=True)
    print(f"EXPERIMENT RESULTS: {args.exp_id}", flush=True)
    print(f"CV Macro-F1: {mean_f1:.4f} +- {std_f1:.4f}", flush=True)
    print(f"CV Accuracy: {mean_acc*100:.2f}% +- {std_acc*100:.2f}%", flush=True)
    print(f"Per-Class F1: {f1_dict}", flush=True)
    print(f"FP16 Model Size: {model_size_mb} MB (Budget <= 25 MB)", flush=True)
    print(f"CPU Latency (2 threads): {latency_ms:.1f} ms/image", flush=True)
    print("=" * 65, flush=True)

    # Append to experiments.md
    f1_str = " / ".join([f"{f1_dict[c]:.4f}" for c in CLASSES])
    log_line = f"| **{args.exp_id}** | {args.backbone}, Res {args.resolution}, {args.loss_type}, smooth={args.label_smoothing} | **{mean_f1:.4f} ± {std_f1:.4f}** | {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | {f1_str} | {model_size_mb} MB | {latency_ms:.1f} ms | Evaluated |"
    
    with open("experiments.md", "a", encoding="utf-8") as f:
        f.write("\n" + log_line)
    print(f"Logged {args.exp_id} to experiments.md", flush=True)


if __name__ == "__main__":
    main()
