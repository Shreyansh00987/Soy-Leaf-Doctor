"""
Main entry point for automated evaluator scoring (P4 Soy Leaf Doctor).
Contract: python predict.py --images {image_dir} --output {output_csv}
Offline, CPU-only, deterministic, fail-soft.
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.model import CLASSES, build_model
from src.dataset import get_eval_transforms

# Evaluator required probability column prefix
PROB_COLS = [f"p_{cls}" for cls in CLASSES]
CSV_HEADER = ["image_id", "label"] + PROB_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Soy Leaf Doctor Offline Inference")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder containing .jpg images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output CSV submission file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mobilenet_v3.pt",
        help="Path to model weights checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    return parser.parse_args()


def load_image_failsoft(img_path: str, transform) -> torch.Tensor:
    """Loads image safely; falls back to neutral gray canvas on corruption."""
    try:
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            return transform(rgb_img)
    except Exception as exc:
        print(f"[WARN] Failed to read {img_path}: {exc}. Using neutral fallback.", file=sys.stderr)
        blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return transform(blank)


def main():
    args = parse_args()

    # Enforce deterministic CPU execution for 2-core sandbox
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    image_dir = args.images
    output_path = args.output
    model_path = args.model_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Scan directory for images (deduplicate for case-insensitive filesystems)
    supported_exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths_set = set()
    for ext in supported_exts:
        image_paths_set.update(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(list(image_paths_set))

    if not image_paths:
        print(f"[WARN] No image files found in {image_dir}. Writing empty header.", file=sys.stderr)
        df_empty = pd.DataFrame(columns=CSV_HEADER)
        df_empty.to_csv(output_path, index=False)
        return

    # Load model
    device = torch.device("cpu")
    model = build_model(pretrained=False, checkpoint_path=model_path if os.path.exists(model_path) else None)
    model.to(device)
    model.eval()

    transform = get_eval_transforms()
    records = []

    print(f"Scoring {len(image_paths)} images with MobileNetV3-Large (CPU 2-threads)...", file=sys.stderr)

    with torch.no_grad():
        batch_tensors = []
        batch_ids = []

        for idx, img_p in enumerate(image_paths):
            file_stem = os.path.splitext(os.path.basename(img_p))[0]
            tensor = load_image_failsoft(img_p, transform)
            batch_tensors.append(tensor)
            batch_ids.append(file_stem)

            # Process batch
            if len(batch_tensors) == args.batch_size or idx == len(image_paths) - 1:
                batch_x = torch.stack(batch_tensors, dim=0).to(device)

                # 4-View Test Time Augmentation (TTA)
                probs = model.predict_tta(batch_x).cpu().numpy()

                for b_i in range(len(batch_ids)):
                    p_vec = probs[b_i]

                    # Normalize to guarantee exact sum = 1.0
                    p_sum = float(np.sum(p_vec))
                    if p_sum > 0:
                        p_vec = p_vec / p_sum
                    else:
                        p_vec = np.ones(len(CLASSES)) / len(CLASSES)

                    top_idx = int(np.argmax(p_vec))
                    top_label = CLASSES[top_idx]

                    row = {
                        "image_id": batch_ids[b_i],
                        "label": top_label,
                    }
                    for c_idx, cls in enumerate(CLASSES):
                        row[f"p_{cls}"] = round(float(p_vec[c_idx]), 6)

                    records.append(row)

                batch_tensors = []
                batch_ids = []

    df_out = pd.DataFrame(records, columns=CSV_HEADER)
    df_out.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path} with {len(df_out)} predictions.", file=sys.stderr)


if __name__ == "__main__":
    main()
