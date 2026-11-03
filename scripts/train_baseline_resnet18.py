"""Train and evaluate the first Phase 1 ResNet18 baseline experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.neu_dataset import NEUDefectDataset
from src.models.resnet import build_resnet18_classifier, count_parameters
from src.preprocessing.transforms import build_transforms


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_batch(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels


def make_loaders(config: dict) -> tuple[dict[str, DataLoader], dict[str, NEUDefectDataset], dict[str, int]]:
    split_dir = Path(config["split_dir"])
    class_to_index = {class_name: index for index, class_name in enumerate(config["classes"])}
    label_aliases = config.get("label_aliases", {})
    training_config = config["training"]

    datasets = {
        "train": NEUDefectDataset(
            split_dir / "train.csv",
            class_to_index,
            build_transforms(config["preprocessing"], "train"),
            label_aliases,
        ),
        "validation": NEUDefectDataset(
            split_dir / "validation.csv",
            class_to_index,
            build_transforms(config["preprocessing"], "validation"),
            label_aliases,
        ),
        "test": NEUDefectDataset(
            split_dir / "test.csv",
            class_to_index,
            build_transforms(config["preprocessing"], "test"),
            label_aliases,
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(config["random_seed"])
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            collate_fn=collate_batch,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            collate_fn=collate_batch,
        ),
    }
    return loaders, datasets, class_to_index


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
    return y_true, y_pred


def average_inference_time(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_images = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            _ = model(images)
            total_images += images.size(0)
    elapsed = time.perf_counter() - start_time
    return elapsed / total_images


def plot_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]

    for key, title, filename in (
        ("train_loss", "Training Loss vs Epoch", "training_loss.png"),
        ("validation_loss", "Validation Loss vs Epoch", "validation_loss.png"),
        ("train_accuracy", "Training Accuracy vs Epoch", "training_accuracy.png"),
        ("validation_accuracy", "Validation Accuracy vs Epoch", "validation_accuracy.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(epochs, [record[key] for record in history], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(key.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
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
    axis.set_title("Test Confusion Matrix")

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


def save_checkpoint(path: Path, model: nn.Module, config: dict, class_to_index: dict[str, int], epoch: int, metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_to_index": class_to_index,
            "config": config,
            "validation_accuracy": metric,
        },
        path,
    )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def verify_before_training(
    config: dict,
    loaders: dict[str, DataLoader],
    datasets: dict[str, NEUDefectDataset],
    class_to_index: dict[str, int],
    model: nn.Module,
    device: torch.device,
) -> None:
    print("Pre-Training Verification")
    print("=" * 32)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device selected: {device}")
    print(f"Class mapping: {class_to_index}")
    for split_name, dataset in datasets.items():
        counts = dataset.records["class"].replace(config.get("label_aliases", {})).value_counts().sort_index().to_dict()
        print(f"{split_name} count: {len(dataset)} | per class: {counts}")

    images, _labels = next(iter(loaders["train"]))
    print(f"One batch image shape: {list(images.shape)}")
    model.to(device)
    with torch.no_grad():
        outputs = model(images.to(device))
    print(f"Model architecture:\n{model}")
    print(f"Final output shape: {list(outputs.shape)}")
    if outputs.shape[1] != config["model"]["num_classes"]:
        raise RuntimeError(f"Expected output shape [batch_size, 6], got {list(outputs.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet18 no-augmentation baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_resnet18_no_aug.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    set_reproducibility(config["random_seed"])

    results_dir = Path(config["outputs"]["results_dir"])
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders, datasets, class_to_index = make_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_classifier(config["model"]["num_classes"])
    verify_before_training(config, loaders, datasets, class_to_index, model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    history = []

    print("\nTraining")
    print("=" * 32)
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_metrics = run_epoch(model, loaders["validation"], criterion, device)
        history_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(history_record)
        print(
            f"Epoch {epoch}/{config['training']['epochs']} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} acc {validation_metrics['accuracy']:.4f}"
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                checkpoint_dir / "best_validation_model.pt",
                model,
                config,
                class_to_index,
                epoch,
                best_validation_accuracy,
            )

    save_checkpoint(
        checkpoint_dir / "final_model.pt",
        model,
        config,
        class_to_index,
        config["training"]["epochs"],
        history[-1]["validation_accuracy"],
    )

    pd.DataFrame(history).to_csv(metrics_dir / "training_history.csv", index=False)
    with (metrics_dir / "experiment_config_used.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    with (metrics_dir / "class_to_index.json").open("w", encoding="utf-8") as mapping_file:
        json.dump(class_to_index, mapping_file, indent=2)

    plot_history(history, figures_dir)

    model.load_state_dict(best_model_state)
    y_true, y_pred = predict(model, loaders["test"], device)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=config["classes"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(config["classes"]))))
    plot_confusion_matrix(matrix, config["classes"], figures_dir / "test_confusion_matrix.png")

    inference_time = average_inference_time(model, loaders["test"], device)
    best_checkpoint_path = checkpoint_dir / "best_validation_model.pt"
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    metrics = {
        "experiment_name": config["experiment_name"],
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "parameter_count": count_parameters(model),
        "best_checkpoint_size_mb": file_size_mb(best_checkpoint_path),
        "final_checkpoint_size_mb": file_size_mb(final_checkpoint_path),
        "average_inference_time_seconds_per_image": inference_time,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_weights": config["model"]["pretrained_weights"],
        "epochs_completed": config["training"]["epochs"],
        "checkpoints": {
            "best_validation": str(best_checkpoint_path),
            "final": str(final_checkpoint_path),
        },
        "figures_dir": str(figures_dir),
    }
    with (metrics_dir / "test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("\nFinal Test Evaluation")
    print("=" * 32)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Parameters: {metrics['parameter_count']}")
    print(f"Average inference time/image: {inference_time:.6f} seconds")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Metrics: {metrics_dir / 'test_metrics.json'}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
