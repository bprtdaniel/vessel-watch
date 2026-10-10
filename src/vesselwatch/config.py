"""Experiment configuration: one YAML file per run, paths overridable by environment."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

DATA_ENV = "VESSELWATCH_DATA"
RUNS_ENV = "VESSELWATCH_RUNS"
CACHE_ENV = "VESSELWATCH_CACHE"


@dataclass
class Config:
    level: int                      # ShipRSImageNet label level: 1, 2 or 3
    model: str                      # key in models.registry
    name: str | None = None         # run folder name, defaults to level{n}_{model}
    epochs: int = 30
    batch_size: int = 32
    optimizer: str = "adam"         # "adam" or "sgd"
    lr: float = 1e-3
    momentum: float = 0.9           # sgd only
    pretrained: bool = False        # ImageNet weights, torchvision models only
    seed: int = 0
    num_workers: int = 2
    # "vessel_crops": one sample per annotated vessel, cut out by its box.
    # "image_majority": the original study's one majority-vote label per image.
    labels: str = "vessel_crops"
    # Original study only: each split builds its own class index.
    per_split_classes: bool = False
    # vessel_crops only. The dataset's test split has no labels, so the official
    # val split is the held-out test set and validation is carved out of train.
    val_fraction: float = 0.15      # share of train images used for validation
    split_seed: int = 0
    # "rotated" follows the oriented box, "upright" its bounding rectangle; both fill the crop with the
    # vessel. "fixed" cuts a crop_window x crop_window pixel window at true scale, so size stays visible.
    crop: str = "rotated"
    crop_margin: float = 0.1        # rotated and upright: context added on each side, as a share of the box
    crop_window: int = 56           # fixed: side of the window in pixels
    # Resample every scene to this many metres per pixel before cropping (None keeps the original images)
    resample_to: float | None = None
    min_length_m: float = 0.0       # with resample_to: leave out vessels shorter than this
    data_root: str | None = None    # folder holding COCO_Format/ and VOC_Format/
    runs_dir: str | None = None
    cache_dir: str | None = None    # extracted crops are kept here between epochs and runs

    @property
    def run_name(self) -> str:
        return self.name or f"level{self.level}_{self.model}"

    def resolved_data_root(self) -> Path:
        root = self.data_root or os.environ.get(DATA_ENV)
        if not root:
            raise ValueError(f"Set data_root in the config or the {DATA_ENV} environment variable")
        return Path(root)

    def run_dir(self) -> Path:
        return Path(self.runs_dir or os.environ.get(RUNS_ENV, "runs")) / self.run_name

    def resolved_cache_dir(self) -> Path | None:
        cache = self.cache_dir or os.environ.get(CACHE_ENV)
        return Path(cache) if cache else None

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path, **overrides) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return Config(**raw)
