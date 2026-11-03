"""Generate Grad-CAM visualizations for selected untouched test-set images."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms, build_visual_transform


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def normalize_label(label: str) -> str:
    return {"rolled-in_scale": "rolled_in_scale"}.get(label, label)


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_resnet18_classifier(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_test_manifest(
    manifest: pd.DataFrame,
    model: torch.nn.Module,
    transform,
    index_to_class: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    prediction_records = []
    for _, record in manifest.iterrows():
        image_path = Path(record["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            model_input = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(model_input)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities.argmax().item())

        true_class = normalize_label(record["class"])
        predicted_class = index_to_class[predicted_index]
        prediction_records.append(
            {
                "image_path": str(image_path),
                "annotation_path": record["annotation_path"],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": float(probabilities[predicted_index].item()),
                "correct": true_class == predicted_class,
            }
        )
    return pd.DataFrame(prediction_records)


def select_visualization_samples(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    selected_indexes: set[int] = set()
    samples_per_class = int(config["samples_per_class"])
    minimum_samples_per_class = int(config["minimum_samples_per_class"])
    seed = int(config["random_seed"])

    if config.get("include_incorrect_predictions", True):
        for index in predictions.index[~predictions["correct"]]:
            selected_indexes.add(int(index))

    for class_label, group in predictions.groupby("true_class"):
        already_selected = group[group.index.isin(selected_indexes)]
        remaining_needed = max(samples_per_class - len(already_selected), minimum_samples_per_class - len(already_selected))
        remaining_needed = max(0, min(remaining_needed, len(group) - len(already_selected)))
        if remaining_needed:
            candidates = group[~group.index.isin(selected_indexes)]
            sampled = candidates.sample(n=remaining_needed, random_state=seed)
            selected_indexes.update(int(index) for index in sampled.index)

    return predictions.loc[sorted(selected_indexes)].sort_values(["true_class", "image_path"]).reset_index(drop=True)


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


def visualization_filename(record: pd.Series) -> str:
    image_stem = Path(record["image_path"]).stem.replace("-", "_")
    status = "correct" if record["correct"] else "incorrect"
    return f"{image_stem}_true-{record['true_class']}_pred-{record['predicted_class']}_{status}.png"


def generate_visualizations(
    selected: pd.DataFrame,
    model: torch.nn.Module,
    transform,
    visual_transform,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    records = []
    visualization_dir = output_dir / "visualizations"
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for _, record in selected.iterrows():
            image_path = Path(record["image_path"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                model_input = transform(image).unsqueeze(0).to(device)
                rgb_image = resized_rgb_float(image, visual_transform)

            target_index = int(record["predicted_index"])
            grayscale_cam = cam(input_tensor=model_input, targets=[ClassifierOutputTarget(target_index)])[0]
            overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            gradcam_path = visualization_dir / record["true_class"] / visualization_filename(record)
            save_triptych(rgb_image, grayscale_cam, overlay, gradcam_path)
            output_record = record.to_dict()
            output_record["gradcam_path"] = str(gradcam_path)
            records.append(output_record)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for untouched test samples.")
    parser.add_argument("--config", type=Path, default=Path("configs/testset_gradcam.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    class_to_index = load_json(Path(config["class_to_index_path"]))
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms(config["preprocessing"], "test")
    visual_transform = build_visual_transform(config["preprocessing"]["input_size"])
    model = load_model(Path(config["checkpoint_path"]), len(class_to_index), device)

    test_manifest = pd.read_csv(config["test_manifest"])
    predictions = predict_test_manifest(test_manifest, model, transform, index_to_class, device)
    predictions["predicted_index"] = predictions["predicted_class"].map(class_to_index)
    selected = select_visualization_samples(predictions, config)
    visualized = generate_visualizations(selected, model, transform, visual_transform, device, output_dir)

    output_columns = [
        "image_path",
        "annotation_path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
        "gradcam_path",
    ]
    visualized[output_columns].to_csv(output_dir / "predictions.csv", index=False)

    incorrect = visualized[~visualized["correct"]]
    summary = {
        "experiment_name": config["experiment_name"],
        "source_experiment": config["source_experiment"],
        "checkpoint_path": config["checkpoint_path"],
        "test_manifest": config["test_manifest"],
        "device": str(device),
        "target_layer": config["target_layer"],
        "total_test_images_scored": int(len(predictions)),
        "total_visualizations": int(len(visualized)),
        "visualizations_per_class": visualized["true_class"].value_counts().sort_index().to_dict(),
        "incorrect_visualized_predictions": incorrect[output_columns].to_dict(orient="records"),
        "all_test_incorrect_predictions": predictions[~predictions["correct"]][
            ["image_path", "annotation_path", "true_class", "predicted_class", "confidence", "correct"]
        ].to_dict(orient="records"),
        "mean_visualized_confidence": float(visualized["confidence"].mean()),
        "note": "Grad-CAM is model-attribution evidence and not ground-truth localization.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("Test-Set Grad-CAM Evaluation")
    print("=" * 32)
    print(f"Device: {device}")
    print(f"Test images scored: {len(predictions)}")
    print(f"Visualizations generated: {len(visualized)}")
    print(f"Visualizations per class: {summary['visualizations_per_class']}")
    print(f"All test incorrect predictions: {len(summary['all_test_incorrect_predictions'])}")
    print(f"Incorrect visualized predictions: {len(summary['incorrect_visualized_predictions'])}")
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
