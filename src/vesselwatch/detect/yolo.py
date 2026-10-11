"""Train and score a YOLO oriented-box detector on ShipRSImageNet.

    python -m vesselwatch.detect.yolo --config configs/detect/yolo11s_obb.yaml

Exports the dataset, trains, and scores the best weights once on the held-out
test split. Everything lands in <runs>/<name>/; a run that already has a
metrics_test.json is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from ..config import CACHE_ENV, DATA_ENV, RUNS_ENV
from .canvases import export_canvases
from .export import export_yolo_obb

S2_ENV = "VESSELWATCH_S2"


@dataclass
class DetectConfig:
    name: str
    weights: str = "yolo11s-obb.pt"  # pretrained starting point, downloaded by ultralytics
    level: int = 0                   # label level the detector's classes come from
    epochs: int = 100
    imgsz: int = 1024
    batch: int = 16
    seed: int = 0
    val_fraction: float = 0.15       # keep equal to the classification study, so the splits match
    split_seed: int = 0
    # Train at a coarser resolution: scenes are resampled to this many metres per pixel and composed
    # into canvas x canvas pixel images, see detect/canvases.py. None trains on the original scenes.
    resample_to: float | None = None
    min_length_m: float = 20.0       # with resample_to: vessels shorter than this are not labelled
    canvas: int = 320
    passes: int = 20                 # passes over the train scenes when composing canvases
    # "shiprsimagenet", or "finland": chips of the Finnish Sentinel-2 scenes with their vessel boxes,
    # see detect/finland_export.py. Those scenes are kept in s2_dir (or VESSELWATCH_S2).
    dataset: str = "shiprsimagenet"
    s2_dir: str | None = None
    empty_ratio: float = 2.0         # finland: chips without a vessel kept per chip with one
    data_root: str | None = None
    runs_dir: str | None = None
    cache_dir: str | None = None

    def resolved_data_root(self) -> Path:
        root = self.data_root or os.environ.get(DATA_ENV)
        if not root:
            raise ValueError(f"Set data_root in the config or the {DATA_ENV} environment variable")
        return Path(root)

    def runs_root(self) -> Path:
        return Path(self.runs_dir or os.environ.get(RUNS_ENV, "runs"))

    def run_dir(self) -> Path:
        return self.runs_root() / self.name

    def dataset_dir(self) -> Path:
        base = self.cache_dir or os.environ.get(CACHE_ENV) or self.runs_root()
        if self.dataset == "finland":
            return Path(base) / f"finland_chips_empty{self.empty_ratio:g}_seed{self.seed}"
        name = f"yolo_obb_level{self.level}_split{self.split_seed}"
        if self.resample_to:
            name += f"_at_{self.resample_to:g}m_canvas{self.canvas}_passes{self.passes}"
        return Path(base) / name

    def export_dataset(self) -> Path:
        """Write the training data in YOLO's layout and return its dataset.yaml."""
        if self.dataset == "finland":
            from .finland_export import export_finland_chips

            s2_dir = self.s2_dir or os.environ.get(S2_ENV)
            if not s2_dir:
                raise ValueError(f"Set s2_dir in the config or the {S2_ENV} environment variable")
            return export_finland_chips(s2_dir, self.dataset_dir(), empty_ratio=self.empty_ratio, seed=self.seed)
        if self.dataset != "shiprsimagenet":
            raise ValueError(f"Unknown dataset '{self.dataset}'")
        if self.resample_to:
            return export_canvases(self.resolved_data_root(), self.dataset_dir(), self.resample_to, self.level,
                                   self.val_fraction, self.split_seed, self.min_length_m, self.canvas,
                                   self.passes, self.seed)
        return export_yolo_obb(self.resolved_data_root(), self.dataset_dir(), self.level,
                               self.val_fraction, self.split_seed)


def load_detect_config(path, **overrides) -> DetectConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(raw) - {f.name for f in fields(DetectConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return DetectConfig(**raw)


def detection_metrics(results) -> dict:
    """The headline numbers of an ultralytics validation run, in percent."""
    box = results.box
    return {
        "precision": 100 * float(box.mp),
        "recall": 100 * float(box.mr),
        "map50": 100 * float(box.map50),
        "map50_95": 100 * float(box.map),
    }


def epochs_done(run_dir: Path) -> int:
    """Epochs ultralytics has logged for this run. Resuming a finished run would start a new training."""
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return 0
    return max(0, len(results.read_text().strip().splitlines()) - 1)


def train_detector(cfg: DetectConfig) -> dict:
    from ultralytics import YOLO  # imported here so the rest of the package works without it

    run_dir = cfg.run_dir()
    metrics_file = run_dir / "metrics_test.json"
    if metrics_file.exists():
        print(f"{cfg.name}: already scored on test, skipping")
        return json.loads(metrics_file.read_text())

    dataset_yaml = cfg.export_dataset()
    last = run_dir / "weights" / "last.pt"
    if epochs_done(run_dir) >= cfg.epochs:
        print(f"{cfg.name}: training already finished, scoring only")
    elif last.exists():
        # An interrupted session: continue from the last epoch on disk
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(cfg.weights).train(data=str(dataset_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
                                seed=cfg.seed, project=str(cfg.runs_root().resolve()), name=cfg.name,
                                exist_ok=True)

    best = run_dir / "weights" / "best.pt"
    results = YOLO(str(best)).val(data=str(dataset_yaml), split="test", imgsz=cfg.imgsz, batch=cfg.batch,
                                  project=str(run_dir.resolve()), name="test", exist_ok=True)
    metrics = {"split": "test", "weights": str(best), **detection_metrics(results)}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args(argv)
    cfg = load_detect_config(args.config, data_root=args.data_root, runs_dir=args.runs_dir,
                             epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    metrics = train_detector(cfg)
    print(f"{cfg.name} | test | Precision: {metrics['precision']:.2f}% | Recall: {metrics['recall']:.2f}% "
          f"| mAP50: {metrics['map50']:.2f}% | mAP50-95: {metrics['map50_95']:.2f}%")


if __name__ == "__main__":
    main()
