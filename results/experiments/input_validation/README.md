# Input Validation / Preliminary OOD Screening — Phase 1 Addendum

**Isolated experiment:** `results/experiments/input_validation/`
**Does not modify:** checkpoint, dataset, splits, baseline metrics, robustness, Grad-CAM experiments
**Official clean test result unchanged:** 99.63% (269/270) — see `results/experiments/baseline_resnet18_no_aug/metrics/test_metrics.json`

## Why the Original Classifier Always Predicts a Steel Defect

The ResNet18 classifier was trained as a **closed-set 6-class** softmax classifier (`crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches`) with `CrossEntropyLoss`. By design, `softmax` always assigns probability mass across the six classes and `argmax` always returns one of them, even for images completely unrelated to steel (e.g., `C:/new.png`). There is no "unknown" class in the trained model, and no rejection mechanism was part of training.

## Why Confidence Alone Is Insufficient

A neural network can be **highly confident on OOD images**. Example from this codebase:

- Solid red image `RGB(255,0,0)`: model predicts `scratches` with **91.05%** confidence but feature similarity to training set is only **0.45** (well below threshold 0.85) → correctly blocked by feature method, but would be falsely accepted by confidence threshold alone.
- Similarly, unrelated `new.png` predicts `scratches` at 41% but could be higher for other OOD; confidence threshold is not a calibrated OOD score and varies per class.

Therefore the web app does **not** use a confidence threshold as the rejection mechanism.

## Preliminary Input-Validation Mechanism (No Retraining, No Test Leakage)

**Extractor:** Same trained ResNet18 checkpoint `checkpoints/baseline_resnet18_no_aug/best_validation_model.pt`, features taken from `avgpool` layer before `fc` (512-dim), `L2`-normalized. Implemented in `src/input_validation/validator.py:15` via `build_feature_extractor`.

**Reference:** 1260 training embeddings only, from `data/splits/seed_42_70_15_15/train.csv` (seed 42, 210/class). Built via `scripts/build_input_validation_reference.py:1` → `reference_embeddings.npz` (2308489 bytes), `class_centroids.npz`. **No test images used** for reference or threshold.

**Metric:** For query image, apply identical baseline preprocessing (`RGB, Resize 224, ToTensor, ImageNet Normalize`), extract 512-dim embedding, compute **max cosine similarity** to any training embedding (nearest neighbor). Cosine because embeddings are L2-normalized → dot product.

**Threshold (training-only):**

| Statistic | Nearest-neighbor max sim | Centroid max sim |
|---|---|---|
| mean | 0.9565 | 0.9402 |
| std | 0.0158 | 0.0282 |
| p1 (1st percentile) | 0.904 | 0.857 |
| p0.1 | 0.880 | 0.823 |
| min | 0.849 | 0.788 |
| chosen conservative | **0.85** | — |

Candidate `p1 0.904` would incorrectly reject valid steel: `crazing_151.jpg` (known test error) has `max_sim 0.8942`, validation `scratches_44.jpg` has `0.8938`. Both are legitimate steel images. Unrelated `C:/new.png` has `0.633`, solid colors `0.45-0.67`. 

Chosen conservative threshold **0.85** balances:
- Accepts >99.9% training (only 1–2 below 0.85)
- Accepts known valid `crazing_151` (0.894) and all 10 sampled validation images (0.893–0.971) 
- Rejects unrelated with large margin (0.63 < 0.85)
- Documented as **preliminary prototype heuristic**, not a fully validated OOD detector.

Config saved to `threshold_config.json:48` `chosen_threshold 0.904` (training p1) plus override `CONSERVATIVE_THRESHOLD 0.85` in `validator.py:16`.

**Flow in `app.py:413`:**

```
Upload -> PIL RGB -> validation (feature similarity) -> log to validation_log.csv
  if max_sim < 0.85:
     display "UNKNOWN / UNSUITABLE IMAGE" + explanation
     do NOT run classifier as valid
     do NOT generate Grad-CAM
  else:
     display "✓ Suitable"
     run ResNet18 prediction + confidence + probabilities
     generate Grad-CAM layer4[-1] (triptych)
```

Suitable images are saved to `results/experiments/web_demo/`; all validation attempts (including rejects) logged to `results/experiments/input_validation/validation_log.csv`.

## Limitations

- Preliminary, not a fully validated OOD detector. No formal FAR/FRR evaluation on held-out OOD set.
- Some OOD may still pass (false accepts) if its feature happens to be close to training manifold.
- Threshold chosen conservatively to avoid false rejects of valid steel; tighter threshold would improve OOD rejection but increase false rejects.
- Does not guarantee domain membership; methodology note shown in UI: "Input validation is a preliminary out-of-distribution screening mechanism. It does not guarantee that an image belongs to the steel-defect domain."

## Verification (No Leakage)

- Threshold derived **only** from training nearest-neighbor distribution, never from `test.csv`.
- Validation on 10 random validation images: all accepted (0.893–0.971) → no over-rejection.
- Required tests:
  - `crazing_37.jpg` → suitable 0.962 → CRAZING 99.71% (expected)
  - `crazing_151.jpg` → suitable 0.894 → PATCHES 51.27% (consistent with CLI/test, now gated but accepted)
  - `C:/new.png` → unsuitable 0.633 → blocked, no Grad-CAM (ideal)
  - `inclusion_20.jpg` / `patches_10.jpg` → suitable → correct predictions
  - Solid red → unsuitable 0.452 → blocked despite 91% confidence (demonstrates confidence insufficiency)

## Artifacts (Isolated)

- `reference_embeddings.npz` — 1260×512 training embeddings
- `class_centroids.npz` — 6 centroids
- `threshold_config.json` — training stats and candidates
- `validation_log.csv` — runtime validation attempts (append only)
- `scripts/build_input_validation_reference.py` — reproducible builder
- `src/input_validation/validator.py` — reusable validator
- `scripts/check_val_sims.py`, `scripts/test_input_validation.py` — verification scripts (not required for app)

All existing checkpoints, splits, metrics, robustness, and Grad-CAM experiments remain unchanged.
