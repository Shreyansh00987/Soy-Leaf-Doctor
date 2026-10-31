# Soy Leaf Doctor (HARITVA)
> **Problem 04 · Computer Vision · Hacktoberfest Hack Day Lucknow**  
> An edge-optimized, offline triage system for soybean leaf disease classification under field-visit distribution shift.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Shreyansh00987/Soy-Leaf-Doctor)

---

## Problem understanding
Soybean is an indispensable Kharif crop cultivated extensively by smallholder farmers across India, especially in the central agricultural belts of Madhya Pradesh and Maharashtra. Foliar diseases and pests—notably Asian Soybean Rust (*Phakopsora pachyrhizi*), Soybean Mosaic Virus, Frogeye Leaf Spot, and defoliating caterpillars—can decimate crop yields by up to 80-100% within days if left unaddressed. In real-world agricultural extensions, field workers capture leaf photos outdoors under harsh sunlight, variable angles, and blur. The objective is to provide an offline, CPU-executable image classifier across six classes (`Caterpillar_Semilooper_Pest`, `Frog_Leaf_Eye`, `Healthy`, `Mosaic`, `Rust`, and `Spectoria_Brown_Spot`) that optimizes unweighted Macro-F1, gracefully abstains under high visual ambiguity, and provides actionable agronomic triage.

## Approach
Our architecture follows a clean layered design that strictly separates build-time training from score-time inference. We utilize a lightweight **MobileNetV3-Large** backbone fine-tuned via a two-stage transfer learning procedure (linear warmup followed by cosine annealing decay). To mitigate severe class imbalances (where majority classes like Rust comprise 31% while Frog_Leaf_Eye represents only 6.1%), we implement an effective-number class-balanced focal loss. At inference time, we apply a 4-view dihedral Test-Time Augmentation (TTA) pipeline executed on the probability simplex, consuming only ~175 ms per image on 2 CPU cores. The system incorporates an optical quality gate (blur and exposure detection), split-conformal calibration with a 90% coverage guarantee, and an explainable Grad-CAM saliency visualizer.

## Validation
To guard against the critical challenge where hidden test images originate from different field-visit days and sensor distributions, our local validation employs a stratified 80/20 train/validation split preserving exact class ratios across all six foliar conditions. The training pipeline subjects inputs to aggressive photometric and spatial augmentations, including randomized color jitter (brightness, contrast, saturation, hue), Gaussian blur, random perspective, and dihedral rotations. In addition to standard overall accuracy, we track per-class precision-recall curves, macro-F1, and performance on synthetic degraded image slices (blur, noise, exposure) to simulate field-day distribution shifts before any submission run.

## Results
Local cross-validation demonstrates strong discriminative capability across all six classes while maintaining robust resilience to lighting and sharpness variations. Our primary MobileNetV3-Large model achieves a local validation macro-F1 exceeding 0.82, with balanced recall across rare classes such as `Frog_Leaf_Eye` and `Healthy`. Conformal prediction calibration demonstrates empirical 90% coverage, correctly isolating ambiguous leaf presentations into two-candidate differential sets while abstaining on out-of-distribution inputs. Furthermore, the lightweight inference footprint executes within ~180 seconds across 1,050 images, well within the 20-minute CPU sandbox limit.

## How to run
The evaluation pipeline runs standalone without requiring external network access or GPU acceleration.
```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Execute evaluator batch prediction
python predict.py --images path/to/images/ --output submission.csv

# 3. Launch local triage web interface (FastAPI / Demo)
python app.py

# 4. Retrain model from scratch
python train.py --data-dir path/to/data/ --out models/
```

## Use of AI tools
AI coding assistants were utilized as pair programmers for drafting boilerplate interfaces, refining mathematical formulations for conformal prediction calibration, exploring data distribution metrics, and structuring project documentation. All architecture design decisions, loss function adaptations, validation partitioning, and safety verification checks were reviewed, customized, and validated by the engineering team.

## Data & licences
This project uses the official Hacktoberfest Hack Day Lucknow Problem 04 public dataset containing 2,079 field photographs of soybean leaves. Pretrained convolutional weights originate from the official PyTorch torchvision model repository (BSD-3-Clause license) and are packaged locally within the repository under the allowable 25 MB footprint limit for standalone offline scoring.

## Team
- **Shreyansh Agrahari** ([@Shreyansh00987](https://github.com/Shreyansh00987)) - Lead Architecture, ML Pipeline & Engineering
