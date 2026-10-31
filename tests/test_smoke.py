"""
Automated evaluator smoke test.
Verifies repo schema contract, output formatting, probability normalization, and argmax label consistency.
"""
import os
import shutil
import subprocess
import sys
import tempfile

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from PIL import Image

from src.model import CLASSES

EXPECTED_COLS = ["image_id", "label"] + [f"p_{cls}" for cls in CLASSES]


def test_predict_contract():
    temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    output_csv = os.path.join(temp_dir, "submission.csv")

    try:
        # Generate 2 synthetic 384x384 JPEG images
        img1 = Image.fromarray(np.random.randint(40, 200, (384, 384, 3), dtype=np.uint8))
        img1.save(os.path.join(images_dir, "soy_sample_01.jpg"))

        img2 = Image.fromarray(np.random.randint(50, 180, (384, 384, 3), dtype=np.uint8))
        img2.save(os.path.join(images_dir, "soy_sample_02.jpg"))

        # Run predict.py via subprocess to test CLI contract
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cmd = [
            sys.executable,
            os.path.join(repo_root, "predict.py"),
            "--images",
            images_dir,
            "--output",
            output_csv,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=repo_root)
        assert os.path.exists(output_csv), f"Output CSV not created: {output_csv}"

        df = pd.read_csv(output_csv)

        # 1. Row count check
        assert len(df) == 2, f"Expected 2 rows, found {len(df)}"

        # 2. Header and column names check
        assert list(df.columns) == EXPECTED_COLS, (
            f"Columns mismatch.\nExpected: {EXPECTED_COLS}\nFound: {list(df.columns)}"
        )

        # 3. Probability normalization and argmax consistency
        for _, row in df.iterrows():
            probs = [row[f"p_{cls}"] for cls in CLASSES]
            p_sum = sum(probs)
            assert abs(p_sum - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {p_sum}"

            max_cls = CLASSES[int(np.argmax(probs))]
            assert row["label"] == max_cls, (
                f"Label '{row['label']}' does not match argmax class '{max_cls}'"
            )

        print("[OK] test_predict_contract passed successfully!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_predict_contract()
