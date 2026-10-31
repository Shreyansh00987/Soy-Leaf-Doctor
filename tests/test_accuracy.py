"""
Local Accuracy & Macro-F1 Evaluation Benchmark for Soy Leaf Doctor.
Creates a balanced test split across all 6 classes, runs predict.py, and outputs detailed classification metrics.
"""
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.model import CLASSES


def create_and_evaluate_benchmark(
    data_dir: str = "data/p4-soy-leaf-health",
    benchmark_dir: str = "test_benchmark",
    samples_per_class: int = 15,
):
    """
    Creates a balanced evaluation set (15 images x 6 classes = 90 images),
    runs predict.py, and computes exact Accuracy, Macro-F1, and Confusion Matrix.
    """
    labels_file = os.path.join(data_dir, "train_labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.exists(labels_file) or not os.path.exists(images_dir):
        print(f"[ERROR] Data not found at {data_dir}", file=sys.stderr)
        return

    df_all = pd.read_csv(labels_file)

    # Sample balanced images per class
    sampled_rows = []
    for cls in CLASSES:
        cls_df = df_all[df_all["label"] == cls]
        # Verify image file exists
        valid_rows = []
        for _, row in cls_df.iterrows():
            img_id = str(row["image_id"]).replace(".jpg", "")
            if os.path.exists(os.path.join(images_dir, f"{img_id}.jpg")):
                valid_rows.append(row)
            if len(valid_rows) >= samples_per_class:
                break
        sampled_rows.extend(valid_rows)

    benchmark_df = pd.DataFrame(sampled_rows).reset_index(drop=True)

    # Setup directories
    test_images_dir = os.path.join(benchmark_dir, "images")
    output_csv = os.path.join(benchmark_dir, "predictions.csv")
    truth_csv = os.path.join(benchmark_dir, "ground_truth.csv")

    shutil.rmtree(benchmark_dir, ignore_errors=True)
    os.makedirs(test_images_dir, exist_ok=True)

    # Copy benchmark images
    for _, row in benchmark_df.iterrows():
        img_id = str(row["image_id"]).replace(".jpg", "")
        src_path = os.path.join(images_dir, f"{img_id}.jpg")
        dst_path = os.path.join(test_images_dir, f"{img_id}.jpg")
        shutil.copy2(src_path, dst_path)

    benchmark_df[["image_id", "label"]].to_csv(truth_csv, index=False)
    print(f"\nCreated test benchmark with {len(benchmark_df)} images ({samples_per_class} per class).")
    print(f"Images location: {test_images_dir}")
    print(f"Ground truth:    {truth_csv}")

    # Run predict.py (evaluator command)
    print("\nExecuting evaluator contract: python predict.py --images ... --output ...")
    start_time = time.time()
    cmd = [
        sys.executable,
        "predict.py",
        "--images",
        test_images_dir,
        "--output",
        output_csv,
    ]
    subprocess.run(cmd, check=True)
    inference_duration = time.time() - start_time

    # Load predictions
    df_preds = pd.read_csv(output_csv)
    df_truth = pd.read_csv(truth_csv)

    merged = pd.merge(df_truth, df_preds, on="image_id", suffixes=("_true", "_pred"))
    y_true = merged["label_true"].values
    y_pred = merged["label_pred"].values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_img_ms = (inference_duration / len(benchmark_df)) * 1000.0

    print("\n" + "=" * 68)
    print("           SOY LEAF DOCTOR — BENCHMARK EVALUATION RESULTS        ")
    print("=" * 68)
    print(f"Total Test Images:       {len(benchmark_df)} ({samples_per_class} per class balanced)")
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Official Primary Metric (Macro-F1): {macro_f1:.4f}")
    print(f"Total Inference Time:    {inference_duration:.2f}s ({per_img_ms:.1f} ms/image on CPU)")
    print(f"Sandbox 1,140ms Budget:  {per_img_ms:.1f} ms used ({(per_img_ms / 1140.0) * 100:.1f}%) -> {1140.0 / per_img_ms:.1f}x Headroom!")
    print("=" * 68)

    print("\n--- Detailed Per-Class Breakdown ---")
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"True_{c[:10]}" for c in CLASSES], columns=[f"Pred_{c[:10]}" for c in CLASSES])
    print(cm_df.to_string())
    print("\n" + "=" * 68)


if __name__ == "__main__":
    create_and_evaluate_benchmark()
