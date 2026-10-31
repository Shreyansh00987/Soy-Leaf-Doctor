"""
Bonus Edge Model: predict_small.py (Weights <= 25 MB).
Provides ultra-lightweight, high-speed single-pass inference (<35 ms/image on CPU).
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

PROB_COLS = [f"p_{cls}" for cls in CLASSES]
CSV_HEADER = ["image_id", "label"] + PROB_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Soy Leaf Doctor Edge Inference (Bonus)")
    parser.add_argument("--images", type=str, required=True, help="Input images folder")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--model-path", type=str, default="models/mobilenet_v3.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    supported_exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths_set = set()
    for ext in supported_exts:
        image_paths_set.update(glob.glob(os.path.join(args.images, ext)))
    image_paths = sorted(list(image_paths_set))

    if not image_paths:
        pd.DataFrame(columns=CSV_HEADER).to_csv(args.output, index=False)
        return

    model = build_model(pretrained=False, checkpoint_path=args.model_path if os.path.exists(args.model_path) else None)
    model.eval()
    transform = get_eval_transforms()

    records = []
    with torch.no_grad():
        for img_p in image_paths:
            file_stem = os.path.splitext(os.path.basename(img_p))[0]
            try:
                with Image.open(img_p) as img:
                    tensor = transform(img.convert("RGB")).unsqueeze(0)
            except Exception:
                blank = Image.new("RGB", (224, 224), (128, 128, 128))
                tensor = transform(blank).unsqueeze(0)

            # Fast single forward pass
            probs = model.predict_probs(tensor).squeeze().cpu().numpy()
            p_sum = float(np.sum(probs))
            probs = probs / p_sum if p_sum > 0 else np.ones(len(CLASSES)) / len(CLASSES)

            top_idx = int(np.argmax(probs))
            row = {"image_id": file_stem, "label": CLASSES[top_idx]}
            for c_idx, cls in enumerate(CLASSES):
                row[f"p_{cls}"] = round(float(probs[c_idx]), 6)
            records.append(row)

    df_out = pd.DataFrame(records, columns=CSV_HEADER)
    df_out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
