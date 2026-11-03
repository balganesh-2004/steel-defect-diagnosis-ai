"""Evaluate ResNet18 baseline robustness on degraded NEU-DET test images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18_classifier


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def normalize_label(label: str) -> str:
    return {"rolled-in_scale": "rolled_in_scale"}.get(label, label)


def parameter_text(condition: str, spec: dict) -> str:
    if condition == "clean":
        return "none"
    if condition == "gaussian_noise":
        return f"sigma={spec['sigma']}"
    if condition == "gaussian_blur":
        return f"kernel={spec['kernel']}x{spec['kernel']}, sigma={spec['sigma']}"
    if condition in {"brightness", "contrast"}:
        return f"factor={spec['factor']}"
    raise ValueError(f"Unsupported condition: {condition}")


def build_preprocess(config: dict) -> transforms.Compose:
    input_size = int(config["preprocessing"]["input_size"])
    mean = config["preprocessing"]["normalization"]["mean"]
    std = config["preprocessing"]["normalization"]["std"]
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def apply_degradation(image: Image.Image, condition: str, spec: dict, rng: np.random.Generator) -> Image.Image:
    image = image.convert("RGB")
    if condition == "clean":
        return image
    if condition == "gaussian_noise":
        array = np.asarray(image).astype(np.float32)
        noise = rng.normal(0.0, float(spec["sigma"]), size=array.shape)
        degraded = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(degraded, mode="RGB")
    if condition == "gaussian_blur":
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        kernel = int(spec["kernel"])
        degraded = cv2.GaussianBlur(array, (kernel, kernel), float(spec["sigma"]))
        return Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB), mode="RGB")
    if condition == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(spec["factor"]))
    if condition == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(spec["factor"]))
    raise ValueError(f"Unsupported condition: {condition}")


def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    class_to_index = load_json(Path(config["class_to_index_path"]))
    model = build_resnet18_classifier(len(class_to_index))
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_condition(
    manifest: pd.DataFrame,
    model: torch.nn.Module,
    preprocess,
    class_to_index: dict[str, int],
    condition: str,
    spec: dict,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict], np.ndarray]:
    index_to_class = {index: label for label, index in class_to_index.items()}
    y_true: list[int] = []
    y_pred: list[int] = []
    error_rows: list[dict] = []
    severity = int(spec["severity"])

    with torch.no_grad():
        for row_index, record in manifest.iterrows():
            image_path = Path(record["image_path"])
            true_class = normalize_label(record["class"])
            true_index = class_to_index[true_class]
            rng = np.random.default_rng(seed + severity * 100000 + int(row_index))

            with Image.open(image_path) as image:
                degraded_image = apply_degradation(image, condition, spec, rng)
                tensor = preprocess(degraded_image).unsqueeze(0).to(device)

            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities.argmax().item())
            predicted_class = index_to_class[predicted_index]
            confidence = float(probabilities[predicted_index].item())

            y_true.append(true_index)
            y_pred.append(predicted_index)
            if predicted_index != true_index:
                error_rows.append(
                    {
                        "image_path": str(image_path),
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "condition": condition,
                        "severity": severity,
                    }
                )

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_to_index))))
    metrics = {
        "condition": condition,
        "severity": severity,
        "parameter": parameter_text(condition, spec),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "correct": int((np.array(y_true) == np.array(y_pred)).sum()),
        "incorrect": int((np.array(y_true) != np.array(y_pred)).sum()),
        "evaluated_images": len(y_true),
    }
    return metrics, error_rows, matrix


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(classes)))
    axis.set_yticks(np.arange(len(classes)))
    axis.set_xticklabels(classes, rotation=30, ha="right")
    axis.set_yticklabels(classes)
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_accuracy(results: pd.DataFrame, condition: str, output_path: Path) -> None:
    subset = results[results["condition"] == condition].sort_values("severity")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(subset["severity"], subset["accuracy"], marker="o")
    axis.set_title(f"Accuracy vs Severity: {condition.replace('_', ' ').title()}")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined(results: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for condition in ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]:
        subset = results[results["condition"] == condition].sort_values("severity")
        axis.plot(subset["severity"], subset["accuracy"], marker="o", label=condition.replace("_", " ").title())
    axis.set_title("Robustness Accuracy Comparison")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def most_common_new_confusion(error_df: pd.DataFrame) -> dict | None:
    degraded = error_df[error_df["condition"] != "clean"]
    if degraded.empty:
        return None
    counts = degraded.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False)
    (true_class, predicted_class), count = counts.index[0], int(counts.iloc[0])
    return {"true_class": true_class, "predicted_class": predicted_class, "count": count}


def verify_test_manifest(manifest: pd.DataFrame) -> dict:
    normalized_classes = manifest["class"].apply(normalize_label)
    return {
        "total": int(len(manifest)),
        "per_class": normalized_classes.value_counts().sort_index().to_dict(),
        "unique_image_paths": int(manifest["image_path"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 robustness on NEU-DET test split.")
    parser.add_argument("--config", type=Path, default=Path("configs/robustness_resnet18.json"))
    args = parser.parse_args()

    start_time = time.perf_counter()
    config = load_json(args.config)
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    copied_config_dir = output_dir / "configs"
    copied_script_dir = output_dir / "scripts"
    for directory in (metrics_dir, figures_dir, copied_config_dir, copied_script_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, copied_config_dir / args.config.name)
    shutil.copy2(Path(__file__), copied_script_dir / Path(__file__).name)

    manifest = pd.read_csv(config["test_manifest"])
    manifest_verification = verify_test_manifest(manifest)
    expected_counts = {
        "crazing": 45,
        "inclusion": 45,
        "patches": 45,
        "pitted_surface": 45,
        "rolled_in_scale": 45,
        "scratches": 45,
    }
    if manifest_verification["total"] != 270 or manifest_verification["per_class"] != expected_counts:
        raise RuntimeError(f"Unexpected test split counts: {manifest_verification}")
    if manifest_verification["unique_image_paths"] != 270:
        raise RuntimeError(f"Duplicate image paths in test manifest: {manifest_verification}")

    class_to_index = load_json(Path(config["class_to_index_path"]))
    classes = [class_name for class_name, _index in sorted(class_to_index.items(), key=lambda item: item[1])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    preprocess = build_preprocess(config)

    result_rows = []
    all_errors = []
    confusion_dir = figures_dir / "confusion_matrices"
    for condition, specs in config["degradations"].items():
        for spec in specs:
            metrics, error_rows, matrix = evaluate_condition(
                manifest,
                model,
                preprocess,
                class_to_index,
                condition,
                spec,
                int(config["random_seed"]),
                device,
            )
            result_rows.append(metrics)
            all_errors.extend(error_rows)
            filename = f"{condition}_severity_{metrics['severity']}.png"
            plot_confusion_matrix(matrix, classes, f"{condition.replace('_', ' ').title()} Severity {metrics['severity']}", confusion_dir / filename)
            print(
                f"{condition} severity {metrics['severity']} | "
                f"accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f} | "
                f"incorrect {metrics['incorrect']}"
            )

    results = pd.DataFrame(result_rows)
    clean_accuracy = float(results.loc[results["condition"].eq("clean"), "accuracy"].iloc[0])
    clean_correct = int(results.loc[results["condition"].eq("clean"), "correct"].iloc[0])
    clean_incorrect = int(results.loc[results["condition"].eq("clean"), "incorrect"].iloc[0])
    expected_clean = config["expected_clean"]
    if (
        abs(clean_accuracy - float(expected_clean["accuracy"])) > 1e-12
        or clean_correct != int(expected_clean["correct"])
        or clean_incorrect != int(expected_clean["incorrect"])
    ):
        raise RuntimeError(
            "Clean baseline mismatch. "
            f"Observed accuracy={clean_accuracy}, correct={clean_correct}, incorrect={clean_incorrect}; "
            f"expected={expected_clean}"
        )
    results["accuracy_drop"] = clean_accuracy - results["accuracy"]
    results["relative_accuracy_drop"] = results["accuracy_drop"] / clean_accuracy
    results.to_csv(metrics_dir / "robustness_results.csv", index=False)

    error_df = pd.DataFrame(
        all_errors,
        columns=["image_path", "true_class", "predicted_class", "confidence", "condition", "severity"],
    )
    error_df.to_csv(metrics_dir / "error_analysis.csv", index=False)

    for condition in ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]:
        plot_accuracy(results, condition, figures_dir / f"accuracy_vs_severity_{condition}.png")
    plot_combined(results, figures_dir / "accuracy_comparison_all_conditions.png")

    degraded_results = results[results["condition"] != "clean"]
    best_row = degraded_results.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
    worst_row = degraded_results.sort_values(["accuracy", "macro_f1"], ascending=True).iloc[0].to_dict()
    summary = {
        "experiment_name": config["experiment_name"],
        "checkpoint_path": config["checkpoint_path"],
        "test_split_path": config["test_manifest"],
        "random_seed": config["random_seed"],
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "preprocessing": config["preprocessing"],
        "degradations": config["degradations"],
        "test_manifest_verification": manifest_verification,
        "clean_accuracy_verified": True,
        "number_of_conditions_evaluated": int(len(results)),
        "evaluated_images_per_condition": int(results["evaluated_images"].min()),
        "all_conditions_evaluated_270_images": bool(results["evaluated_images"].eq(270).all()),
        "best_robustness_condition": best_row,
        "worst_robustness_condition": worst_row,
        "largest_performance_drop": worst_row,
        "most_common_new_confusion": most_common_new_confusion(error_df),
        "training_performed": False,
        "duration_seconds": time.perf_counter() - start_time,
    }
    with (metrics_dir / "robustness_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("Robustness evaluation complete")
    print(f"Results: {metrics_dir / 'robustness_results.csv'}")
    print(f"Error analysis: {metrics_dir / 'error_analysis.csv'}")
    print(f"Summary: {metrics_dir / 'robustness_summary.json'}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
