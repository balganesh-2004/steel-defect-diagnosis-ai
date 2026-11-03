"""Create reproducible stratified NEU-DET train/validation/test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG_PATH = Path("configs/split_config.json")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_dataset_root(dataset_root: Path) -> Path:
    nested_root = dataset_root / "NEU-DET"
    if (nested_root / "train").exists() or (nested_root / "validation").exists():
        return nested_root
    return dataset_root


def relative_path(path: Path) -> str:
    return path.as_posix()


def annotation_class(annotation_path: Path) -> str:
    root = ET.parse(annotation_path).getroot()
    first_object = root.find("object")
    if first_object is None:
        return ""
    return first_object.findtext("name", default="")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_verified_records(dataset_root: Path, image_extensions: set[str]) -> tuple[list[dict], dict]:
    records = []
    pairing_issues = {
        "missing_annotations": [],
        "cross_split_annotations": [],
        "annotation_class_mismatches": [],
        "orphan_annotations": [],
    }
    paired_annotation_paths = set()

    image_paths = sorted(
        path
        for split_dir in ("train", "validation")
        for path in (dataset_root / split_dir / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    all_annotation_paths = set(dataset_root.glob("*/annotations/*.xml"))
    annotations_by_stem: dict[str, list[Path]] = defaultdict(list)
    for annotation_path in all_annotation_paths:
        annotations_by_stem[annotation_path.stem].append(annotation_path)

    for image_path in image_paths:
        relative_parts = image_path.relative_to(dataset_root).parts
        source_split = relative_parts[0]
        class_label = relative_parts[2]
        expected_annotation_path = dataset_root / source_split / "annotations" / f"{image_path.stem}.xml"
        annotation_path = expected_annotation_path

        if not expected_annotation_path.exists() and annotations_by_stem.get(image_path.stem):
            annotation_path = sorted(annotations_by_stem[image_path.stem])[0]
            pairing_issues["cross_split_annotations"].append(
                {
                    "image_path": relative_path(image_path),
                    "expected_annotation_path": relative_path(expected_annotation_path),
                    "used_annotation_path": relative_path(annotation_path),
                }
            )

        if not annotation_path.exists():
            pairing_issues["missing_annotations"].append(relative_path(image_path))
            continue

        xml_class = annotation_class(annotation_path)
        if xml_class and xml_class != class_label:
            pairing_issues["annotation_class_mismatches"].append(
                {
                    "image_path": relative_path(image_path),
                    "folder_class": class_label,
                    "annotation_class": xml_class,
                    "annotation_path": relative_path(annotation_path),
                }
            )
            continue

        paired_annotation_paths.add(annotation_path)
        records.append(
            {
                "image_path": relative_path(image_path),
                "class": class_label,
                "annotation_path": relative_path(annotation_path),
                "source_split": source_split,
                "sha256": file_sha256(image_path),
            }
        )

    orphan_annotations = sorted(all_annotation_paths - paired_annotation_paths)
    pairing_issues["orphan_annotations"] = [relative_path(path) for path in orphan_annotations]
    return records, pairing_issues


def split_class_records(records: list[dict], ratios: dict[str, float], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    grouped_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[record["class"]].append(record)

    split_records = {"train": [], "validation": [], "test": []}
    for class_label in sorted(grouped_records):
        hash_groups: dict[str, list[dict]] = defaultdict(list)
        for record in grouped_records[class_label]:
            hash_groups[record["sha256"]].append(record)

        grouped_items = list(hash_groups.values())
        rng.shuffle(grouped_items)
        grouped_items.sort(key=len, reverse=True)

        total = len(grouped_records[class_label])
        train_count = round(total * ratios["train"])
        validation_count = round(total * ratios["validation"])
        test_count = total - train_count - validation_count
        target_counts = {"train": train_count, "validation": validation_count, "test": test_count}
        current_counts = {"train": 0, "validation": 0, "test": 0}

        for item_group in grouped_items:
            candidate_splits = [
                split_name
                for split_name, target_count in target_counts.items()
                if current_counts[split_name] + len(item_group) <= target_count
            ]
            if not candidate_splits:
                raise RuntimeError(
                    f"Cannot assign duplicate group of size {len(item_group)} for class {class_label} "
                    "without exceeding split targets."
                )

            selected_split = max(
                candidate_splits,
                key=lambda split_name: target_counts[split_name] - current_counts[split_name],
            )
            split_records[selected_split].extend(item_group)
            current_counts[selected_split] += len(item_group)

    for split_name in split_records:
        split_records[split_name] = sorted(split_records[split_name], key=lambda item: item["image_path"])

    return split_records


def verify_no_overlap(split_records: dict[str, list[dict]]) -> dict:
    image_to_splits: dict[str, list[str]] = defaultdict(list)
    hash_to_records: dict[str, list[dict]] = defaultdict(list)

    for split_name, records in split_records.items():
        for record in records:
            image_to_splits[record["image_path"]].append(split_name)
            hash_to_records[record["sha256"]].append(
                {"split": split_name, "image_path": record["image_path"], "class": record["class"]}
            )

    overlapping_images = {
        image_path: splits
        for image_path, splits in image_to_splits.items()
        if len(splits) > 1
    }
    duplicate_groups = [records for records in hash_to_records.values() if len(records) > 1]
    duplicate_groups_across_splits = [
        records
        for records in duplicate_groups
        if len({record["split"] for record in records}) > 1
    ]

    return {
        "overlapping_image_paths": overlapping_images,
        "duplicate_groups": duplicate_groups,
        "duplicate_groups_across_splits": duplicate_groups_across_splits,
    }


def build_summary(
    split_records: dict[str, list[dict]],
    pairing_issues: dict,
    overlap_report: dict,
    config: dict,
    dataset_root: Path,
) -> dict:
    all_records = [record for records in split_records.values() for record in records]
    summary = {
        "dataset_root": relative_path(dataset_root),
        "random_seed": config["random_seed"],
        "split_ratios": config["split_ratios"],
        "total_images_in_manifests": len(all_records),
        "images_per_class": pd.DataFrame(all_records)["class"].value_counts().sort_index().to_dict(),
        "split_counts": {},
        "pairing_issues": pairing_issues,
        "overlap_report": overlap_report,
    }

    for split_name, records in split_records.items():
        split_df = pd.DataFrame(records)
        summary["split_counts"][split_name] = {
            "total": int(len(split_df)),
            "per_class": split_df["class"].value_counts().sort_index().to_dict() if not split_df.empty else {},
        }

    return summary


def write_outputs(split_records: dict[str, list[dict]], summary: dict, config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_columns = config["manifest_columns"]

    for split_name, records in split_records.items():
        df = pd.DataFrame(records)
        df[manifest_columns].to_csv(output_dir / f"{split_name}.csv", index=False)

    all_records = pd.DataFrame([record for records in split_records.values() for record in records])
    all_records[["image_path", "class", "annotation_path", "source_split", "sha256"]].to_csv(
        output_dir / "all_splits_with_hashes.csv",
        index=False,
    )

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with (output_dir / "split_config_used.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified NEU-DET split manifests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(Path(config["dataset_root"]))
    output_dir = Path(config["output_dir"])
    image_extensions = set(config["image_extensions"])

    print("Creating NEU-DET stratified split manifests")
    print("=" * 48)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Output dir: {output_dir}")
    print(f"Random seed: {config['random_seed']}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    records, pairing_issues = collect_verified_records(dataset_root, image_extensions)
    if not records:
        raise RuntimeError("No verified image/annotation pairs found.")

    split_records = split_class_records(records, config["split_ratios"], config["random_seed"])
    overlap_report = verify_no_overlap(split_records)
    summary = build_summary(split_records, pairing_issues, overlap_report, config, dataset_root)
    write_outputs(split_records, summary, config, output_dir)

    print(f"Verified image/annotation pairs used: {summary['total_images_in_manifests']}")
    for split_name, split_summary in summary["split_counts"].items():
        print(f"{split_name}: {split_summary['total']} images")
        for class_label, count in split_summary["per_class"].items():
            print(f"  {class_label}: {count}")
    print(f"Missing annotations: {len(pairing_issues['missing_annotations'])}")
    print(f"Orphan annotations: {len(pairing_issues['orphan_annotations'])}")
    print(f"Overlapping image paths: {len(overlap_report['overlapping_image_paths'])}")
    print(f"Duplicate content groups: {len(overlap_report['duplicate_groups'])}")
    print(f"Duplicate content groups across splits: {len(overlap_report['duplicate_groups_across_splits'])}")


if __name__ == "__main__":
    main()
