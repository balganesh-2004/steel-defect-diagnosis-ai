"""Generate Grad-CAM evidence visualizations for the ResNet18 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms, build_visual_transform


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_resnet18_classifier(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def normalized_label(label: str, aliases: dict[str, str]) -> str:
    return aliases.get(label, label)


def select_samples(manifest: pd.DataFrame, samples_per_class: int, seed: int, aliases: dict[str, str]) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["normalized_class"] = manifest["class"].apply(lambda label: normalized_label(label, aliases))
    samples = []
    for class_label, group in manifest.groupby("normalized_class"):
        sample_count = min(samples_per_class, len(group))
        samples.append(group.sample(n=sample_count, random_state=seed).sort_values("image_path"))
    return pd.concat(samples).sort_values(["normalized_class", "image_path"]).reset_index(drop=True)


def tensor_to_model_input(image: Image.Image, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(device)


def resized_rgb_float(image: Image.Image, visual_transform: transforms.Compose) -> np.ndarray:
    tensor = visual_transform(image)
    return tensor.permute(1, 2, 0).numpy().astype(np.float32)


def save_triptych(original_rgb: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_uint8 = np.uint8(255 * original_rgb)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    combined = np.concatenate([original_uint8, heatmap_color, overlay_rgb], axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def safe_name(path: Path) -> str:
    return path.stem.replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for baseline ResNet18.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_resnet18_gradcam.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    class_to_index = load_json(Path(config["class_to_index_path"]))
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}
    aliases = {"rolled-in_scale": "rolled_in_scale"}
    output_dir = Path(config["output_dir"])
    visualization_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(config["checkpoint_path"]), len(class_to_index), device)
    transform = build_transforms(config["preprocessing"], "test")
    visual_transform = build_visual_transform(config["preprocessing"]["input_size"])
    target_layers = [model.layer4[-1]]
    manifest = pd.read_csv(config["test_manifest"])
    samples = select_samples(manifest, config["samples_per_class"], 42, aliases)

    prediction_records = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for _, record in samples.iterrows():
            image_path = Path(record["image_path"])
            true_class = normalized_label(record["class"], aliases)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                model_input = tensor_to_model_input(image, transform, device)
                rgb_image = resized_rgb_float(image, visual_transform)

            with torch.no_grad():
                logits = model(model_input)
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_index = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted_index].item())
                predicted_class = index_to_class[predicted_index]

            grayscale_cam = cam(
                input_tensor=model_input,
                targets=[ClassifierOutputTarget(predicted_index)],
            )[0]
            overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            class_dir = visualization_dir / true_class
            output_path = class_dir / f"{safe_name(image_path)}_true-{true_class}_pred-{predicted_class}.png"
            save_triptych(rgb_image, grayscale_cam, overlay, output_path)

            prediction_records.append(
                {
                    "image_path": str(image_path),
                    "annotation_path": record["annotation_path"],
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "correct": true_class == predicted_class,
                    "gradcam_path": str(output_path),
                }
            )

    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "experiment_name": config["experiment_name"],
        "source_experiment": config["source_experiment"],
        "checkpoint_path": config["checkpoint_path"],
        "device": str(device),
        "target_layer": config["target_layer"],
        "samples_per_class": config["samples_per_class"],
        "total_visualizations": int(len(predictions)),
        "class_counts": predictions["true_class"].value_counts().sort_index().to_dict(),
        "correct_visualized_predictions": int(predictions["correct"].sum()),
        "incorrect_visualized_predictions": int((~predictions["correct"]).sum()),
        "mean_confidence": float(predictions["confidence"].mean()),
        "note": "Grad-CAM is model-attribution evidence and not ground-truth localization.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("Grad-CAM Evidence Generation")
    print("=" * 32)
    print(f"Device: {device}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Visualizations: {len(predictions)}")
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Visualization dir: {visualization_dir}")


if __name__ == "__main__":
    main()
