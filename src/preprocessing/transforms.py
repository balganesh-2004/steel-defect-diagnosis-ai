"""Configurable torchvision preprocessing transforms for Phase 1."""

from __future__ import annotations

from torchvision import transforms


def build_transforms(config: dict, split: str) -> transforms.Compose:
    """Build separate preprocessing pipelines for train, validation, and test."""
    input_size = int(config["input_size"])
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]

    if split == "train":
        train_config = config["train_transforms"]
        transform_steps = [transforms.Resize((input_size, input_size))]

        if train_config.get("random_rotation_degrees", 0) > 0:
            transform_steps.append(transforms.RandomRotation(degrees=train_config["random_rotation_degrees"]))
        if train_config.get("horizontal_flip_probability", 0) > 0:
            transform_steps.append(transforms.RandomHorizontalFlip(p=train_config["horizontal_flip_probability"]))
        if train_config.get("vertical_flip_probability", 0) > 0:
            transform_steps.append(transforms.RandomVerticalFlip(p=train_config["vertical_flip_probability"]))

        jitter_config = train_config.get("color_jitter", {})
        if any(float(jitter_config.get(key, 0.0)) > 0 for key in ("brightness", "contrast", "saturation", "hue")):
            transform_steps.append(
                transforms.ColorJitter(
                    brightness=jitter_config["brightness"],
                    contrast=jitter_config["contrast"],
                    saturation=jitter_config["saturation"],
                    hue=jitter_config["hue"],
                )
            )

        transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose(transform_steps)

    if split in {"validation", "test"}:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    raise ValueError(f"Unsupported split: {split}")


def build_visual_transform(input_size: int) -> transforms.Compose:
    """Build a deterministic transform for before/after visual comparison."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
