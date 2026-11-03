"""CLI live prediction prototype for Phase 1 review.

Steel Image -> Baseline Preprocessing -> ResNet18 Checkpoint -> Prediction + Grad-CAM

Uses existing checkpoint:
checkpoints/baseline_resnet18_no_aug/best_validation_model.pt

No training, no dataset modification, no experiment overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Keep cache dirs consistent with other scripts
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms, build_visual_transform


DEFAULT_CHECKPOINT = Path("checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
DEFAULT_CLASS_MAP = Path("results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json")
DEFAULT_OUTPUT_DIR = Path("results/experiments/predict_demo")
DEFAULT_CONFIG = Path("configs/baseline_resnet18_no_aug.json")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_resnet18_classifier(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_preprocessing_transform(preprocessing_config: dict):
    """Reuse existing baseline preprocessing utility (no augmentation)."""
    return build_transforms(preprocessing_config, "test")


def resized_rgb_float(image: Image.Image, visual_transform) -> np.ndarray:
    tensor = visual_transform(image)
    return tensor.permute(1, 2, 0).numpy().astype(np.float32)


def save_triptych(original_rgb: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    original_uint8 = np.uint8(255 * original_rgb)
    combined = np.concatenate([original_uint8, heatmap_color, overlay], axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 CLI Steel Defect Diagnosis - Image -> Prediction -> Grad-CAM")
    parser.add_argument("--image", type=Path, required=True, help="Path to steel defect image (JPG/PNG)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to baseline checkpoint")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP, help="Path to class_to_index.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to baseline config for preprocessing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for predictions")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu or cuda (auto-detected if not set)")
    args = parser.parse_args()

    # 1. Validate image exists
    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    if not args.image.is_file():
        print(f"ERROR: Not a file: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not args.class_map.exists():
        print(f"ERROR: Class map not found: {args.class_map}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load configs
    class_to_index = load_json(args.class_map)
    index_to_class = {int(v): k for k, v in class_to_index.items()}
    config = load_json(args.config)
    preprocessing_config = config.get("preprocessing", {})
    # Fallback if preprocessing missing (should not happen)
    if "input_size" not in preprocessing_config:
        preprocessing_config = {
            "input_size": 224,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }

    # Device selection (baseline was CPU)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build transforms reusing existing utility
    transform = build_preprocessing_transform(preprocessing_config)
    visual_transform = build_visual_transform(int(preprocessing_config["input_size"]))

    # Load model
    try:
        model = load_model(args.checkpoint, len(class_to_index), device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load image and run inference
    try:
        with Image.open(args.image) as pil_image:
            pil_image = pil_image.convert("RGB")
            # Keep copy for visualization after transform
            image_for_vis = pil_image.copy()
            model_input = transform(pil_image).unsqueeze(0).to(device)
            rgb_float = resized_rgb_float(image_for_vis, visual_transform)
    except Exception as e:
        print(f"ERROR: Failed to load/process image: {e}", file=sys.stderr)
        sys.exit(1)

    # Inference with model.eval() and torch.no_grad()
    start = time.perf_counter()
    with torch.no_grad():
        model.eval()
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        pred_class = index_to_class[pred_idx]
        # Also get full distribution for JSON
        prob_dict = {index_to_class[i]: float(probs[i].item()) for i in range(len(index_to_class))}
    elapsed = time.perf_counter() - start

    # Grad-CAM generation
    gradcam_path = None
    gradcam_status = "Failed"
    try:
        target_layers = [model.layer4[-1]]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=model_input, targets=[ClassifierOutputTarget(pred_idx)])[0]
            overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            # Save triptych
            args.output_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = args.output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            stem = args.image.stem.replace("-", "_")
            gradcam_filename = f"{stem}_pred-{pred_class}_gradcam.png"
            gradcam_path = vis_dir / gradcam_filename
            save_triptych(rgb_float, grayscale_cam, overlay, gradcam_path)
            gradcam_status = "Generated successfully"
    except Exception as e:
        print(f"WARNING: Grad-CAM generation failed: {e}", file=sys.stderr)
        gradcam_status = f"Failed: {e}"

    # Prepare outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.output_dir / "predictions.csv"
    # Record for CSV
    record = {
        "timestamp": datetime.now().isoformat(),
        "image_path": str(args.image),
        "image_name": args.image.name,
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "inference_time_s": elapsed,
        "gradcam_path": str(gradcam_path) if gradcam_path else "",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
    }

    # Append to predictions.csv (create header if not exists)
    write_header = not predictions_csv.exists()
    with predictions_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    # Also write per-image JSON summary
    json_path = args.output_dir / f"{args.image.stem}_result.json"
    json_summary = {
        "image_path": str(args.image),
        "image_name": args.image.name,
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "inference_time_seconds": elapsed,
        "gradcam_path": str(gradcam_path) if gradcam_path else None,
        "gradcam_status": gradcam_status,
        "checkpoint_path": str(args.checkpoint),
        "class_to_index": class_to_index,
        "preprocessing": preprocessing_config,
        "device": str(device),
        "target_layer": "layer4[-1]",
        "note": "Grad-CAM is model-attribution evidence, not ground-truth localization.",
    }
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(json_summary, jf, indent=2)

    # CLI output - clean review format
    print("")
    print("========================================")
    print(" STEEL DEFECT DIAGNOSIS")
    print("========================================")
    print("")
    print("Image:")
    print(f"{args.image.name}")
    print(f"({args.image})")
    print("")
    print("Predicted defect:")
    print(f"{pred_class.upper()}")
    print("")
    print("Confidence:")
    print(f"{confidence*100:.2f}% ({confidence:.6f})")
    print("")
    print("Grad-CAM:")
    print(f"{gradcam_status}")
    if gradcam_path:
        print(f"{gradcam_path}")
    print("")
    print("Output:")
    print(f"{args.output_dir}")
    print(f"  - {predictions_csv}")
    if gradcam_path:
        print(f"  - {gradcam_path}")
    print(f"  - {json_path}")
    print("")
    print("Inference time:")
    print(f"{elapsed:.4f} s")
    print(f"Device: {device}")
    print("")
    print("========================================")
    print("")


if __name__ == "__main__":
    main()
