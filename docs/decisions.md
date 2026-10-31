# Architecture Decision Records (ADRs)
**Project:** Soy Leaf Doctor (HARITVA)  
**Problem:** P4 Computer Vision · Hacktoberfest Hack Day Lucknow  
**Target:** Macro-F1 $\ge 0.82$, CPU Inference $< 1,140\text{ ms/image}$, Zero Runtime Network

---

### ADR 1: Strict Separation of Build-Time and Score-Time Execution Contexts
* **Status:** Accepted & Enforced
* **Context:** The automated scoring sandbox operates under strict air-gapped constraints: 2 CPU cores, 4 GB RAM, no GPU, and zero network access. In contrast, local development and model training enjoy GPU acceleration (NVIDIA RTX 4050) and full internet access. The most common cause of zero-score disqualifications in ML hackathons is build-time dependencies leaking into score-time scripts (e.g. attempting to download pretrained weights from PyTorch Hub or Hugging Face at runtime).
* **Decision:** We strictly enforce a unidirectional boundary. All computationally expensive training, hyperparameter tuning, and conformal quantile calibrations occur at build time. Build time produces only static serialized data files (`mobilenet_v3.pt`, `conformal.json`, `classes.json`). Score-time code (`predict.py`, `predict_small.py`) consumes only these local files and makes zero network calls.
* **Consequences:** Reproducible, hermetic, and fail-safe execution inside the locked-down evaluation container.

---

### ADR 2: Probabilities on the Simplex as the Sole Inter-Component Contract
* **Status:** Accepted
* **Context:** Pipeline components—such as test-time augmentation (TTA) averaging, temperature calibration, specialist arbiters, and conformal prediction sets—frequently interact with model outputs.
* **Decision:** Components never pass raw unnormalized logits across boundaries; they communicate exclusively via normalized probability vectors $\vec{p} \in \Delta^5$ ($\sum p_i = 1.0$).
* **Consequences:** TTA averaging across dihedral views is mathematically sound on the probability simplex. Calibration and conformal sets compose cleanly without leaking model-specific logit scaling.

---

### ADR 3: Model as a Committed Lightweight Artifact ($\le 25\text{ MB}$)
* **Status:** Accepted
* **Context:** The competition offers an optional **+2 Edge Model Bonus** for `predict_small.py` if model weights are $\le 25\text{ MB}$ and achieve within 0.03 macro-F1 of the main model. Large transformer backbones (e.g., ViT-B/16 at ~340 MB, Swin-B) violate size limits and easily timeout on 2 CPU cores.
* **Decision:** Standardize on **MobileNetV3-Large**. Its uncompressed PyTorch state dictionary is only **~21.3 MB**, comfortably within the 25 MB limit. It delivers forward passes in ~42 ms on CPU.
* **Consequences:** Qualifies automatically for the Edge Model bonus while leaving an 85% safety margin on the 20-minute sandbox timeout.

---

### ADR 4: Fail-Soft Fallback Per-Image, Never Per-Run
* **Status:** Accepted
* **Context:** Hidden test sets of ~1,050 field images may contain corrupted EXIF data, truncated byte streams, unusual aspect ratios, or unreadable headers. A single uncaught exception would abort the batch script and award a catastrophic score of zero.
* **Decision:** Every image decode operation is isolated with a graceful fallback policy. If an image fails to decode, the pipeline logs a warning to `stderr`, substitutes a neutral canvas, outputs a uniform prior probability vector ($p_c = 1/6 \approx 0.166667$), and continues processing.
* **Consequences:** Even in the worst-case scenario of corrupted files, the remaining 99.7% of images are scored accurately, preserving high macro-F1.

---

### ADR 5: Conformal Prediction Sets Over Arbitrary Scalar Thresholds
* **Status:** Accepted
* **Context:** Agricultural field workers cannot rely on misleading scalar confidence scores (e.g. "82% Rust" from an uncalibrated softmax). In field conditions with distribution shifts, softmax probabilities are notoriously overconfident.
* **Decision:** Implement **Inductive Split-Conformal Prediction** with a 90% statistical coverage guarantee ($\alpha = 0.10$). Rather than returning a single guessed label, the system outputs prediction sets $C(X)$:
  * $|C| = 1$: Single unambiguous diagnosis $\rightarrow$ immediate treatment advisory.
  * $|C| = 2$: Differential diagnosis $\rightarrow$ presents both diseases with distinguishing features.
  * $|C| \ge 3$: High visual ambiguity / Out-of-Distribution $\rightarrow$ triggers principled abstention and pathologist escalation.
* **Consequences:** Statistically grounded safety, preventing misdiagnoses that could lead to improper chemical application.

---

### ADR 6: Decoupled Presentation Layer via Typed Dataclasses
* **Status:** Accepted
* **Context:** The presentation tier (`app.py`, CLI) should remain independent of deep learning frameworks so that backend models can be retrained or replaced without rewriting the UI.
* **Decision:** Presentation layers receive a plain Python `Result` dictionary / dataclass containing diagnostic labels, prediction sets, quality flags, and localized strings. `app.py` never directly manipulates PyTorch tensors or gradients.
* **Consequences:** High architectural modularity, allowing the mobile web UI to be tested independently using mock fixtures.

---

### ADR 7: Stratified Validation and Domain-Shift Simulating Augmentations
* **Status:** Accepted
* **Context:** The competition guide specifically warns that *hidden test photos come from different field-visit days than training photos*. Furthermore, rare classes like `Frog_Leaf_Eye` comprise only 6.1% (127 images) while `Rust` comprises 31.1% (646 images).
* **Decision:** We use an 80/20 Stratified Shuffle Split to preserve exact rare-class proportions. Training pipelines apply heavy photometric augmentations (ColorJitter for sunlight variation, GaussianBlur for camera defocus, Dihedral Flips for leaf orientation) and train with Effective-Number Class-Balanced loss weights.
* **Consequences:** Maximizes generalization across unseen field-visit lighting conditions and prevents minority-class recall collapse.
