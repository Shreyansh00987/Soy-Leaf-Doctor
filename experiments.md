# Soybean Foliar Disease Classifier (HARITVA) — Experiment Log

**Goal:** Elevate 5-Fold Cross-Validation Macro-F1 from baseline ~0.81 towards 0.90–0.92 under competition constraints.  
**Hard Constraints:**
1. Offline CPU-only inference (2 threads).
2. Packaged model weights $\le 25\text{ MB}$ (FP16 permitted).
3. 1,050 image batch inference runtime $\le 570\text{ ms/image}$ ($<10\text{ min}$ target).
4. CLI contract: `predict.py --images <dir> --output submission.csv`.
5. Fixed 295-sample Holdout Set strictly untouched until final evaluation.

---

## 📈 Experiment Tracking Log

| Exp # | Description / Change | CV Macro-F1 (mean ± std) | CV Accuracy | Per-Class F1 (Pest / Frog / Health / Mos / Rust / Spect) | Model Size | CPU Latency / img | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 0 (Baseline)** | MobileNetV3-Large, Res 224, 4-view TTA, Effective-Num Loss | **0.8084 ± 0.0201** | 81.95% ± 1.91% | 0.7915 / 0.6540 / 0.9963 / 0.8547 / 0.8332 / 0.7273 | 13.42 MB | 222.9 ms | **Baseline Anchor** |

---

## 🧪 Experiment Notes & Progression

### Experiment 0: Baseline Evaluation
- **Architecture:** MobileNetV3-Large
- **Resolution:** 224x224 (downsampled from 384x384 native)
- **Loss:** Effective-Number Weighted Cross Entropy ($\beta=0.999$)
- **Findings:** Healthy foliage is separated with near-perfect accuracy (F1: 0.9963). The primary performance bottlenecks are **Frog_Leaf_Eye (F1: 0.6540)** and **Spectoria_Brown_Spot (F1: 0.7273)** due to confusion with the majority **Rust** class on downsampled 224x224 patches. Native dataset images are 384x384.
