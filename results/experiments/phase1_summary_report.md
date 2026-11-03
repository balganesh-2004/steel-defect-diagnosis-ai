# Phase 1 Experiment Summary Report

## Project

**AI-Driven Multi-Stage Steel Defect Diagnosis and Evidence-Based Corrective Action Recommendation System**

This report summarizes the completed Phase 1 experiments using existing saved results only. No new training, model tuning, dataset modification, or additional experiment was performed while preparing this report.

## Phase 1 Scope

Phase 1 focuses on AI-driven steel surface defect diagnosis with:

- dataset inspection and reproducible splitting
- baseline defect classification
- visual evidence generation using Grad-CAM
- controlled robustness evaluation

Phase 2 activities such as manufacturing-cause analysis and corrective-action recommendation are not included.

## Dataset

Primary dataset: **NEU-DET steel surface defect dataset**

Dataset inspection found:

| Item | Result |
|---|---:|
| Total valid images | 1,800 |
| Classes | 6 |
| Images per class | 300 |
| Image size | 200 x 200 |
| Image mode | RGB |
| File type | JPG |
| XML annotations | 4,189 |
| Corrupted files | 0 |
| Duplicate content groups | 1 |

Classes:

1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled_in_scale
6. scratches

One duplicate-content group was detected in the patches class. It remained inside the training split and did not cross into validation or test splits.

## Reproducible Split

A fixed stratified 70/15/15 manifest-based split was created using seed `42`. Original images and annotations were not moved, modified, or duplicated.

| Split | Total | Per Class |
|---|---:|---:|
| Train | 1,260 | 210 |
| Validation | 270 | 45 |
| Test | 270 | 45 |

Verification:

- image-path overlap across splits: 0
- duplicate content across splits: 0
- missing annotations after pairing: 0
- orphan annotations after pairing: 0

Dataset note: `crazing_240.jpg` physically appears under the original downloaded train image folder while its XML annotation appears under the original validation annotation folder. The manifest links the correct image and annotation without modifying the dataset.

## Preprocessing

Input size: `224 x 224`

Baseline preprocessing:

- convert image to RGB
- resize to `224 x 224`
- convert to tensor
- ImageNet normalization
- no random augmentation for the first baseline experiment

Normalization:

| Channel | Mean | Std |
|---|---:|---:|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

Verified tensor shape: `[3, 224, 224]`

## Baseline Model

Model: **torchvision ResNet18**

Configuration:

| Item | Value |
|---|---|
| Architecture | ResNet18 |
| Pretrained weights | ImageNet |
| Output classes | 6 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Epochs | 5 |
| Device | CPU |
| Random seed | 42 |

The final classification layer was replaced to output six steel-defect classes.

Class mapping:

| Class | Index |
|---|---:|
| crazing | 0 |
| inclusion | 1 |
| patches | 2 |
| pitted_surface | 3 |
| rolled_in_scale | 4 |
| scratches | 5 |

## Training History

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.2609 | 92.94% | 0.0828 | 98.15% |
| 2 | 0.0256 | 99.37% | 0.0219 | 99.26% |
| 3 | 0.0074 | 99.92% | 0.0125 | 99.26% |
| 4 | 0.0105 | 99.60% | 0.0135 | 99.63% |
| 5 | 0.0060 | 99.84% | 0.0150 | 99.63% |

Best checkpoint selection used validation performance only. The best epoch was epoch `4`.

## Clean Test Evaluation

Best checkpoint: `checkpoints/baseline_resnet18_no_aug/best_validation_model.pt`

| Metric | Result |
|---|---:|
| Test accuracy | 99.63% |
| Macro precision | 99.64% |
| Macro recall | 99.63% |
| Macro F1-score | 99.63% |
| Correct | 269 / 270 |
| Incorrect | 1 / 270 |
| Parameters | 11,179,590 |
| Checkpoint size | 42.72 MB |
| Average inference time | 0.0422 s/image |

Per-class results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| crazing | 1.0000 | 0.9778 | 0.9888 | 45 |
| inclusion | 1.0000 | 1.0000 | 1.0000 | 45 |
| patches | 0.9783 | 1.0000 | 0.9890 | 45 |
| pitted_surface | 1.0000 | 1.0000 | 1.0000 | 45 |
| rolled_in_scale | 1.0000 | 1.0000 | 1.0000 | 45 |
| scratches | 1.0000 | 1.0000 | 1.0000 | 45 |

Confusion matrix class order:

`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, `scratches`

```text
[[44, 0, 1, 0, 0, 0],
 [ 0,45, 0, 0, 0, 0],
 [ 0, 0,45, 0, 0, 0],
 [ 0, 0, 0,45, 0, 0],
 [ 0, 0, 0, 0,45, 0],
 [ 0, 0, 0, 0, 0,45]]
```

Known clean test error:

- image: `crazing_151.jpg`
- true class: `crazing`
- predicted class: `patches`
- confidence: `0.512681`

## Grad-CAM Evidence Analysis

Grad-CAM was generated using:

- checkpoint: `checkpoints/baseline_resnet18_no_aug/best_validation_model.pt`
- target layer: `layer4[-1]`
- preprocessing: baseline test preprocessing
- device: CPU

The finalized test-set Grad-CAM experiment scored all 270 test images and generated 18 visualizations, with 3 representative samples per class.

| Class | Grad-CAM Samples |
|---|---:|
| crazing | 3 |
| inclusion | 3 |
| patches | 3 |
| pitted_surface | 3 |
| rolled_in_scale | 3 |
| scratches | 3 |

The known incorrect test image `crazing_151.jpg` was included in the Grad-CAM outputs:

`results/experiments/testset_gradcam/visualizations/crazing/crazing_151_true-crazing_pred-patches_incorrect.png`

Important limitation: Grad-CAM is model-attribution evidence. It should not be described as ground-truth localization.

## Robustness Evaluation

The robustness experiment used the same best validation checkpoint and evaluated all 270 untouched test images under controlled in-memory degradations. No degraded images were saved over the original dataset and no retraining was performed.

Clean control reproduced the baseline result:

- accuracy: 99.63%
- correct: 269 / 270
- incorrect: 1 / 270

Robustness results:

| Condition | Severity | Parameter | Accuracy | Macro F1 | Accuracy Drop |
|---|---:|---|---:|---:|---:|
| clean | 0 | none | 99.63% | 99.63% | 0.00 pp |
| gaussian_noise | 1 | sigma=10 | 30.00% | 24.25% | 69.63 pp |
| gaussian_noise | 2 | sigma=25 | 16.67% | 4.76% | 82.96 pp |
| gaussian_noise | 3 | sigma=40 | 16.67% | 4.76% | 82.96 pp |
| gaussian_blur | 1 | kernel=3x3, sigma=1.0 | 64.81% | 55.95% | 34.81 pp |
| gaussian_blur | 2 | kernel=5x5, sigma=2.0 | 57.41% | 48.83% | 42.22 pp |
| gaussian_blur | 3 | kernel=7x7, sigma=3.0 | 52.22% | 44.35% | 47.41 pp |
| brightness | 1 | factor=0.7 | 99.26% | 99.26% | 0.37 pp |
| brightness | 2 | factor=1.3 | 98.89% | 98.89% | 0.74 pp |
| brightness | 3 | factor=1.5 | 90.37% | 90.31% | 9.26 pp |
| contrast | 1 | factor=0.7 | 98.52% | 98.52% | 1.11 pp |
| contrast | 2 | factor=1.3 | 99.63% | 99.63% | 0.00 pp |
| contrast | 3 | factor=1.5 | 99.63% | 99.63% | 0.00 pp |

Key robustness findings:

- Best robustness condition: contrast factor `1.3`, no performance drop from clean.
- Worst robustness condition: Gaussian noise sigma `25`, accuracy `16.67%`.
- Largest performance drop: `82.96` percentage points under Gaussian noise sigma `25` and sigma `40`.
- Under Gaussian noise sigma `25` and `40`, the model predicted `pitted_surface` for all 270 test images, producing exactly `45/270 = 16.67%` accuracy.
- Most common degraded confusion: `crazing -> pitted_surface`, total count `252` across degraded error rows.

Breakdown of `crazing -> pitted_surface`:

| Condition | Severity | Count |
|---|---:|---:|
| brightness | 2 | 1 |
| brightness | 3 | 15 |
| gaussian_blur | 1 | 45 |
| gaussian_blur | 2 | 38 |
| gaussian_blur | 3 | 18 |
| gaussian_noise | 1 | 45 |
| gaussian_noise | 2 | 45 |
| gaussian_noise | 3 | 45 |

## Discussion

The ResNet18 baseline achieved very high clean test performance on the balanced NEU-DET split, with only one clean test error. This establishes a strong classification baseline.

However, the robustness evaluation shows that clean benchmark performance alone is not sufficient for industrial reliability. Gaussian noise caused a severe performance collapse, especially at sigma `25` and `40`, where predictions collapsed to a single class. Gaussian blur also caused substantial degradation. Brightness and contrast changes were less damaging, except for strong brightness increase.

Grad-CAM outputs provide useful model-attribution evidence for inspection, including the known incorrect `crazing -> patches` case. These visualizations support the evidence-linked diagnosis objective, but they should be interpreted carefully and not treated as ground-truth localization.

## Limitations

- Only one baseline model was trained.
- Training used CPU, not GPU.
- Training duration was not recorded during the baseline run.
- Robustness degradations are controlled synthetic perturbations.
- Grad-CAM provides attribution evidence, not verified segmentation.
- No improved model, robustness-aware training, Streamlit interface, or Phase 2 recommendation system has been implemented yet.

## Saved Artifacts

Important files:

- `results/metrics/dataset_inspection/dataset_summary.json`
- `data/splits/seed_42_70_15_15/split_summary.json`
- `results/metrics/preprocessing/preprocessing_summary.json`
- `results/experiments/baseline_resnet18_no_aug/metrics/test_metrics.json`
- `results/experiments/baseline_resnet18_no_aug/metrics/training_history.csv`
- `checkpoints/baseline_resnet18_no_aug/best_validation_model.pt`
- `results/experiments/testset_gradcam/summary.json`
- `results/experiments/testset_gradcam/predictions.csv`
- `results/experiments/robustness_resnet18/metrics/robustness_results.csv`
- `results/experiments/robustness_resnet18/metrics/error_analysis.csv`
- `results/experiments/robustness_resnet18/metrics/robustness_summary.json`

## Phase 1 Status

Completed:

- environment setup
- dataset inspection
- reproducible train/validation/test split
- preprocessing verification
- baseline ResNet18 experiment
- clean test evaluation
- Grad-CAM evidence generation
- test-set Grad-CAM verification
- robustness evaluation

Not yet implemented:

- improved model comparison
- quantitative Grad-CAM overlap evaluation with bounding boxes
- robustness-aware training
- Streamlit prototype
- Phase 2 cause/corrective-action recommendation

