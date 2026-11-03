"""Verify Phase 1 preprocessing transforms without modifying dataset files."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.neu_dataset import NEUDefectDataset
from src.preprocessing.transforms import build_transforms, build_visual_transform


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def representative_records(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Select one reproducible representative image per class."""
    records = []
    for class_label, group in manifest.groupby("class"):
        records.append(group.sample(n=1, random_state=seed).iloc[0])
    return pd.DataFrame(records).sort_values("class").reset_index(drop=True)


def denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std_tensor + mean_tensor, 0.0, 1.0)


def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(1, 2, 0)


def save_preprocessing_grid(
    records: pd.DataFrame,
    train_transform,
    visual_transform,
    preprocessing_config: dict,
    output_path: Path,
) -> None:
    """Save original/resized and preprocessed examples for all classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean = preprocessing_config["normalization"]["mean"]
    std = preprocessing_config["normalization"]["std"]

    fig, axes = plt.subplots(len(records), 2, figsize=(8, 3 * len(records)))
    for row_index, record in records.iterrows():
        image_path = Path(record["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            visual_tensor = visual_transform(image)
            processed_tensor = train_transform(image)

        axes[row_index][0].imshow(tensor_to_image(visual_tensor))
        axes[row_index][0].set_title(f"{record['class']} - resized RGB")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(tensor_to_image(denormalize(processed_tensor, mean, std)))
        axes[row_index][1].set_title(f"{record['class']} - train transform")
        axes[row_index][1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def tensor_stats(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "channel_means": [float(value) for value in tensor.mean(dim=(1, 2)).tolist()],
        "channel_stds": [float(value) for value in tensor.std(dim=(1, 2)).tolist()],
    }


def write_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify NEU-DET preprocessing transforms.")
    parser.add_argument("--preprocessing-config", type=Path, default=Path("configs/preprocessing_config.json"))
    parser.add_argument("--split-config", type=Path, default=Path("configs/split_config.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/seed_42_70_15_15"))
    parser.add_argument("--figure-path", type=Path, default=Path("results/figures/preprocessing/preprocessed_samples.png"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/metrics/preprocessing/preprocessing_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessing_config = load_json(args.preprocessing_config)
    split_config = load_json(args.split_config)
    seed = int(split_config["random_seed"])

    random.seed(seed)
    torch.manual_seed(seed)

    train_manifest_path = args.split_dir / "train.csv"
    validation_manifest_path = args.split_dir / "validation.csv"
    test_manifest_path = args.split_dir / "test.csv"

    train_manifest = pd.read_csv(train_manifest_path)
    classes = sorted(train_manifest["class"].unique().tolist())
    class_to_index = {class_label: index for index, class_label in enumerate(classes)}

    train_transform = build_transforms(preprocessing_config, "train")
    validation_transform = build_transforms(preprocessing_config, "validation")
    test_transform = build_transforms(preprocessing_config, "test")
    visual_transform = build_visual_transform(preprocessing_config["input_size"])

    train_dataset = NEUDefectDataset(train_manifest_path, class_to_index, train_transform)
    validation_dataset = NEUDefectDataset(validation_manifest_path, class_to_index, validation_transform)
    test_dataset = NEUDefectDataset(test_manifest_path, class_to_index, test_transform)

    sample = train_dataset[0]
    validation_sample = validation_dataset[0]
    test_sample = test_dataset[0]

    sample_records = representative_records(train_manifest, seed)
    save_preprocessing_grid(
        sample_records,
        train_transform,
        visual_transform,
        preprocessing_config,
        args.figure_path,
    )

    summary = {
        "random_seed": seed,
        "input_size": preprocessing_config["input_size"],
        "normalization": preprocessing_config["normalization"],
        "classes": classes,
        "class_to_index": class_to_index,
        "dataset_lengths": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "sample_tensor_stats": {
            "train": tensor_stats(sample["image"]),
            "validation": tensor_stats(validation_sample["image"]),
            "test": tensor_stats(test_sample["image"]),
        },
        "train_transforms": preprocessing_config["train_transforms"],
        "validation_transforms": preprocessing_config["validation_transforms"],
        "test_transforms": preprocessing_config["test_transforms"],
        "figure_path": str(args.figure_path),
        "manifests_used": {
            "train": str(train_manifest_path),
            "validation": str(validation_manifest_path),
            "test": str(test_manifest_path),
        },
    }
    write_summary(summary, args.summary_path)

    print("Preprocessing Verification")
    print("=" * 32)
    print(f"Input size: {preprocessing_config['input_size']}x{preprocessing_config['input_size']}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(validation_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    print(f"Sample tensor shape: {list(sample['image'].shape)}")
    print(f"Sample tensor dtype: {sample['image'].dtype}")
    print(f"Summary: {args.summary_path}")
    print(f"Figure: {args.figure_path}")


if __name__ == "__main__":
    main()
