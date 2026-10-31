"""
Training and Conformal Calibration Pipeline for Soy Leaf Doctor (P4).
Implements class-balanced focal weighting, domain-shift augmentations, and split-conformal calibration.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import CLASSES, NUM_CLASSES, SoyLeafClassifier, CLASS_TO_IDX
from src.dataset import SoyLeafDataset, get_train_transforms, get_eval_transforms
from src.conformal import ConformalCalibrator


def compute_class_weights(df: pd.DataFrame, beta: float = 0.999) -> torch.Tensor:
    """
    Computes Effective Number of Samples weights to balance rare classes.
    w_c = (1 - beta) / (1 - beta^N_c)
    """
    counts = df["label"].value_counts().to_dict()
    weights = []
    for cls in CLASSES:
        n_c = counts.get(cls, 1)
        effective_num = 1.0 - (beta ** n_c)
        w = (1.0 - beta) / effective_num
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.mean(weights)  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc = np.mean(all_preds == all_targets)

    return float(macro_f1), float(acc), all_probs, all_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Train Soy Leaf Doctor")
    parser.add_argument("--data-dir", type=str, default="data/p4-soy-leaf-health")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # Locate labels and images
    labels_path = os.path.join(args.data_dir, "train_labels.csv")
    if not os.path.exists(labels_path):
        # Fallback to local train_labels.csv if present
        labels_path = "train_labels.csv"

    images_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(images_dir):
        # Search parent or current dir
        if os.path.exists("images"):
            images_dir = "images"

    print(f"Loading labels from: {labels_path}", file=sys.stderr)
    print(f"Loading images from: {images_dir}", file=sys.stderr)

    df = pd.read_csv(labels_path)
    print(f"Total dataset records: {len(df)} across {NUM_CLASSES} classes.", file=sys.stderr)

    # Stratified 80/20 train/validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, val_idx = next(sss.split(df, df["label"]))
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    print(f"Stratified Split: {len(df_train)} train, {len(df_val)} validation.", file=sys.stderr)

    train_dataset = SoyLeafDataset(df_train, images_dir, transform=get_train_transforms())
    val_dataset = SoyLeafDataset(df_val, images_dir, transform=get_eval_transforms())

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Model and Loss
    class_weights = compute_class_weights(df_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SoyLeafClassifier(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    # Optimizer with differential learning rate
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.features.parameters(), "lr": args.lr * 0.2},
            {"params": model.backbone.classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_weights_path = os.path.join(args.out, "mobilenet_v3.pt")
    ep7_path = os.path.join(args.out, "mobilenet_v3_ep7.pt")
    ep13_path = os.path.join(args.out, "mobilenet_v3_ep13.pt")
    ep7_metrics = {"val_f1": 0.0, "val_acc": 0.0}
    ep13_metrics = {"val_f1": 0.0, "val_acc": 0.0}
    start_time = time.time()

    print("\n--- Beginning Two-Stage Transfer Learning ---", file=sys.stderr)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_f1, val_acc, val_probs, val_targets = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}",
            file=sys.stderr,
        )

        if epoch == 7:
            torch.save(model.state_dict(), ep7_path)
            ep7_metrics = {"val_f1": val_f1, "val_acc": val_acc}
            print(f">>> [CHECKPOINT] Saved Epoch 7 Model: Val Acc: {val_acc*100:.2f}%, Val Macro-F1: {val_f1:.4f}", file=sys.stderr)

        if epoch == 13:
            torch.save(model.state_dict(), ep13_path)
            ep13_metrics = {"val_f1": val_f1, "val_acc": val_acc}
            print(f">>> [CHECKPOINT] Saved Epoch 13 Model: Val Acc: {val_acc*100:.2f}%, Val Macro-F1: {val_f1:.4f}", file=sys.stderr)

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {elapsed_min:.2f} mins. Peak Val Macro-F1: {best_macro_f1:.4f}", file=sys.stderr)

    # Compare Epoch 7 vs Epoch 13 if both were reached
    if os.path.exists(ep7_path) and os.path.exists(ep13_path):
        print("\n" + "=" * 60, file=sys.stderr)
        print("       EPOCH 7 vs EPOCH 13 COMPARISON REPORT", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Epoch 07 -> Val Acc: {ep7_metrics['val_acc']*100:.2f}% | Val Macro-F1: {ep7_metrics['val_f1']:.4f}", file=sys.stderr)
        print(f"Epoch 13 -> Val Acc: {ep13_metrics['val_acc']*100:.2f}% | Val Macro-F1: {ep13_metrics['val_f1']:.4f}", file=sys.stderr)
        if ep13_metrics["val_f1"] >= ep7_metrics["val_f1"]:
            print(">>> WINNER: Epoch 13 performed better or equal! Selected Epoch 13.", file=sys.stderr)
            winner_weights = ep13_path
        else:
            print(">>> WINNER: Epoch 7 performed better (Epoch 13 likely overfitted)! Selected Epoch 7.", file=sys.stderr)
            winner_weights = ep7_path
        print("=" * 60, file=sys.stderr)
        # Deploy winning weights
        import shutil
        shutil.copy2(winner_weights, best_weights_path)

    # Load winning checkpoint for conformal calibration
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    val_f1_win, val_acc_win, best_probs, val_targets = evaluate(model, val_loader, device)

    # Conformal Calibration (90% coverage)
    calibrator = ConformalCalibrator(alpha=0.10)
    q_hat = calibrator.calibrate(best_probs, val_targets)
    print(f"Conformal non-conformity threshold q_hat (alpha=0.10): {q_hat:.4f}", file=sys.stderr)

    # Export classes and conformal config
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(CLASSES, f, indent=2)

    conformal_config = {
        "alpha": 0.10,
        "q_hat": round(q_hat, 4),
        "target_coverage": 0.90,
        "classes": CLASSES,
        "val_macro_f1": round(val_f1_win, 4),
        "val_accuracy": round(val_acc_win, 4),
        "weights_file": "mobilenet_v3.pt",
        "file_size_mb": round(os.path.getsize(best_weights_path) / (1024 * 1024), 2),
    }
    with open(os.path.join(args.out, "conformal.json"), "w") as f:
        json.dump(conformal_config, f, indent=2)

    print(f"Successfully exported models to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
