from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .crops import CROP_MODES, crop_vessel
from .resample import resample_image


def _label_indices(labels, class_to_idx) -> list[int]:
    unknown = sorted(set(labels) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Labels missing from the class list: {unknown}")
    return [class_to_idx[label] for label in labels]


class ShipImageDataset(Dataset):
    """Whole-image classification dataset built from a filename/label dataframe.

    If `classes` is not given the class index is built from this dataframe
    alone, as in the original study. Pass the train classes to share one index.
    """

    def __init__(self, df, image_dir, transform=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = list(classes) if classes is not None else sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = _label_indices(self.df["label"], self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class VesselCropDataset(Dataset):
    """One sample per annotated vessel, cut out of its scene.

    `records` comes from `coco_to_vessel_records`, or from `resampled_records`
    for a coarser resolution: those carry a `factor` by which each scene is
    shrunk before the vessel is cut out. `classes` is the shared class list; it
    is never derived from the records of a single split. With a `cache_dir`
    each crop is written to disk the first time it is used.
    """

    def __init__(self, records, image_dir, classes, transform=None, crop="rotated", margin=0.1, cache_dir=None,
                 window=56):
        if crop not in CROP_MODES:
            raise ValueError(f"Unknown crop mode '{crop}'")
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.crop = crop
        self.margin = margin
        self.window = window
        variant = f"fixed_w{window}" if crop == "fixed" else f"{crop}_m{margin}"
        self.cache_dir = Path(cache_dir) / variant if cache_dir else None
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = _label_indices(self.records["label"], self.class_to_idx)
        self.areas = self.records["area"].tolist()
        self.lengths_m = self.records["length_m"].tolist() if "length_m" in self.records else None

    def __len__(self):
        return len(self.records)

    def _cache_path(self, row) -> Path:
        # Keyed by geometry, so a crop is shared across label levels
        digest = hashlib.md5(repr(row["polygon"]).encode()).hexdigest()[:12]
        return self.cache_dir / f"{Path(row['filename']).stem}_{digest}.png"

    def load_crop(self, idx) -> Image.Image:
        row = self.records.iloc[idx]
        cached = self._cache_path(row) if self.cache_dir else None
        if cached is not None and cached.exists():
            return Image.open(cached).convert("RGB")

        img = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        if "factor" in row:
            img = resample_image(img, row["factor"])
        crop = crop_vessel(img, row["polygon"], self.crop, self.margin, self.window)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so a loader worker never reads a half-written file
            tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp")
            crop.save(tmp, format="PNG")
            os.replace(tmp, cached)
        return crop

    def __getitem__(self, idx):
        img = self.load_crop(idx)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
