"""Inspect the local NEU-DET dataset without modifying original data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_images(dataset_root: Path) -> list[Path]:
    """Return all image files under the dataset root."""
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_dataset_root(dataset_root: Path) -> Path:
    """Handle a common extra nested NEU-DET folder after extraction."""
    nested_root = dataset_root / "NEU-DET"
    if (nested_root / "train").exists() or (nested_root / "validation").exists():
        return nested_root
    return dataset_root


def infer_class_name(image_path: Path, dataset_root: Path) -> str:
    """Infer class from NEU-DET layouts or a simple class-folder layout."""
    relative_parts = image_path.relative_to(dataset_root).parts
    if "images" in relative_parts:
        image_index = relative_parts.index("images")
        if len(relative_parts) > image_index + 2:
            return relative_parts[image_index + 1]
    if len(relative_parts) >= 2:
        return relative_parts[0]
    return "unclassified"


def infer_split_name(image_path: Path, dataset_root: Path) -> str:
    """Infer dataset split name when train/validation folders are present."""
    relative_parts = image_path.relative_to(dataset_root).parts
    if relative_parts and relative_parts[0].lower() in {"train", "validation", "valid", "val", "test"}:
        return relative_parts[0]
    return "unsplit"


def file_sha256(path: Path) -> str:
    """Calculate a SHA-256 hash for duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_images(dataset_root: Path, check_duplicates: bool) -> tuple[pd.DataFrame, dict[str, list[str]], list[list[str]]]:
    """Inspect image metadata and report corrupted or duplicate files."""
    records = []
    corrupted: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, list[str]] = defaultdict(list)

    for image_path in iter_images(dataset_root):
        class_name = infer_class_name(image_path, dataset_root)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError) as exc:
            corrupted[class_name].append(f"{image_path}: {exc}")
            continue

        if check_duplicates:
            hashes[file_sha256(image_path)].append(str(image_path))

        records.append(
            {
                "path": str(image_path),
                "class": class_name,
                "split": infer_split_name(image_path, dataset_root),
                "width": width,
                "height": height,
                "mode": mode,
                "suffix": image_path.suffix.lower(),
            }
        )

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return pd.DataFrame.from_records(records), dict(corrupted), duplicates


def inspect_annotations(dataset_root: Path) -> pd.DataFrame:
    """Inspect Pascal VOC style XML annotations when available."""
    records = []
    for annotation_path in sorted(dataset_root.rglob("annotations/*.xml")):
        split_name = annotation_path.relative_to(dataset_root).parts[0]
        try:
            root = ET.parse(annotation_path).getroot()
        except ET.ParseError as exc:
            records.append(
                {
                    "annotation_path": str(annotation_path),
                    "split": split_name,
                    "filename": "",
                    "class": "parse_error",
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None,
                    "error": str(exc),
                }
            )
            continue

        filename = root.findtext("filename", default="")
        for object_node in root.findall("object"):
            box_node = object_node.find("bndbox")
            records.append(
                {
                    "annotation_path": str(annotation_path),
                    "split": split_name,
                    "filename": filename,
                    "class": object_node.findtext("name", default="unknown"),
                    "xmin": int(box_node.findtext("xmin")) if box_node is not None else None,
                    "ymin": int(box_node.findtext("ymin")) if box_node is not None else None,
                    "xmax": int(box_node.findtext("xmax")) if box_node is not None else None,
                    "ymax": int(box_node.findtext("ymax")) if box_node is not None else None,
                    "error": "",
                }
            )
    return pd.DataFrame.from_records(records)


def save_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Save a class-distribution bar chart for report use."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = df["class"].value_counts().sort_index()

    fig, axis = plt.subplots(figsize=(9, 5))
    class_counts.plot(kind="bar", ax=axis, color="#4c78a8")
    axis.set_title("NEU-DET Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of Images")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close(fig)


def save_sample_grid(df: pd.DataFrame, output_dir: Path, samples_per_class: int) -> None:
    """Save representative samples from each inferred class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = (
        df.groupby("class", group_keys=False)
        .head(samples_per_class)
        .reset_index(drop=True)
    )
    if sampled.empty:
        return

    rows = sampled["class"].nunique()
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    for axis in axes.flat:
        axis.axis("off")

    for row_index, (class_name, group) in enumerate(sampled.groupby("class")):
        for col_index, (_, record) in enumerate(group.iterrows()):
            with Image.open(record["path"]) as image:
                axes[row_index][col_index].imshow(image.convert("RGB"))
            axes[row_index][col_index].set_title(class_name)
            axes[row_index][col_index].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "representative_samples.png", dpi=200)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    corrupted: dict[str, list[str]],
    duplicates: list[list[str]],
    output_dir: Path,
) -> None:
    """Write dataset statistics to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        summary = {
            "total_valid_images": 0,
            "classes": [],
            "class_counts": {},
            "split_counts": {},
            "annotation_count": int(len(annotation_df)),
            "image_dimensions": {},
            "corrupted_files": corrupted,
            "duplicate_groups": duplicates,
        }
    else:
        dimension_counts = Counter(zip(df["width"], df["height"]))
        summary = {
            "total_valid_images": int(len(df)),
            "classes": sorted(df["class"].unique().tolist()),
            "class_counts": df["class"].value_counts().sort_index().to_dict(),
            "split_counts": df["split"].value_counts().sort_index().to_dict(),
            "annotation_count": int(len(annotation_df)),
            "annotation_class_counts": annotation_df["class"].value_counts().sort_index().to_dict()
            if not annotation_df.empty
            else {},
            "image_dimensions": {f"{width}x{height}": count for (width, height), count in dimension_counts.items()},
            "image_modes": df["mode"].value_counts().to_dict(),
            "file_extensions": df["suffix"].value_counts().to_dict(),
            "corrupted_files": corrupted,
            "duplicate_groups": duplicates,
        }

    df.to_csv(output_dir / "dataset_inventory.csv", index=False)
    annotation_df.to_csv(output_dir / "annotation_inventory.csv", index=False)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the NEU-DET dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/NEU-DET"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/dataset_inspection"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/dataset_inspection"))
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--check-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)

    print("NEU-DET Dataset Inspection")
    print("=" * 32)
    print(f"Dataset root: {dataset_root.resolve()}")

    if not dataset_root.exists():
        print("\nDataset not found.")
        print("Place the NEU-DET dataset at: data/raw/NEU-DET")
        print("Expected structure: data/raw/NEU-DET/<class_name>/<image_files>")
        print("Detected annotation layout is also supported: train/images/<class> and train/annotations")
        print("Alternative structures can be inspected by passing --dataset-root PATH.")
        return

    df, corrupted, duplicates = inspect_images(dataset_root, args.check_duplicates)
    annotation_df = inspect_annotations(dataset_root)
    write_summary(df, annotation_df, corrupted, duplicates, args.output_dir)

    if not df.empty:
        save_class_distribution(df, args.figure_dir)
        save_sample_grid(df, args.figure_dir, args.samples_per_class)

    print(f"Valid images: {len(df)}")
    print(f"Classes: {', '.join(sorted(df['class'].unique())) if not df.empty else 'none'}")
    print(f"Splits: {', '.join(sorted(df['split'].unique())) if not df.empty else 'none'}")
    print(f"Annotations: {len(annotation_df)}")
    print(f"Corrupted files: {sum(len(items) for items in corrupted.values())}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Inventory: {args.output_dir / 'dataset_inventory.csv'}")
    print(f"Summary: {args.output_dir / 'dataset_summary.json'}")
    if not df.empty:
        print(f"Figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
