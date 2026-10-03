"""Experiment configuration: one YAML file per run, paths overridable by environment."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

DATA_ENV = "VESSELWATCH_DATA"
RUNS_ENV = "VESSELWATCH_RUNS"


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
    # Legacy behaviour switches, kept so the original numbers can be reproduced.
    labels: str = "image_majority"  # one majority-vote label per image
    per_split_classes: bool = True  # each split builds its own class index
    data_root: str | None = None    # folder holding COCO_Format/ and VOC_Format/
    runs_dir: str | None = None

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
