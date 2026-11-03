"""PyTorch Dataset for NEU-DET split manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NEUDefectDataset(Dataset):
    """Load NEU-DET images from split manifest CSV files."""

    def __init__(
        self,
        manifest_path: Path,
        class_to_index: dict[str, int],
        transform=None,
        label_aliases: dict[str, str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = pd.read_csv(self.manifest_path)
        self.class_to_index = class_to_index
        self.transform = transform
        self.label_aliases = label_aliases or {}

        required_columns = {"image_path", "class", "annotation_path"}
        missing_columns = required_columns - set(self.records.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records.iloc[index]
        image_path = Path(record["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            transformed_image = self.transform(image) if self.transform else image

        class_label = self.label_aliases.get(record["class"], record["class"])
        return {
            "image": transformed_image,
            "label": self.class_to_index[class_label],
            "class": class_label,
            "image_path": str(image_path),
            "annotation_path": str(record["annotation_path"]),
        }
