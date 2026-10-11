"""The Finnish Sentinel-2 vessel annotations as a YOLO training set.

    <out>/images/{train,val,test}/<tile>_<date>_x<x>_y<y>.png   320 x 320 pixel chips
    <out>/labels/{train,val,test}/<same name>.txt               class x_centre y_centre width height
    <out>/dataset.yaml

Each scene is cut into chips. A vessel belongs to the chip that holds the
centre of its box, and its box is clipped to that chip. Most chips show no
vessel at all; for training and validation only some of those are kept, so
that the model still sees plenty of empty water and land without spending
most of each epoch on it. The test chips are all kept.

Train and validation scenes are different dates of the same three tiles, so
validation shares its coastline with training and is only good for choosing a
checkpoint. The test scenes are two other tiles.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

from ..monitor.tiles import chip_origins, cut_chip
from .finland import CHIP, TEST_SCENES, TRAIN_SCENES, VAL_SCENES, load_scenes

CLASS_NAME = "vessel"


def chip_labels(truth: np.ndarray, x: int, y: int, chip: int) -> list[str]:
    """YOLO label lines for the vessels whose box centre lies in the chip at (x, y)."""
    lines = []
    for x0, y0, x1, y1 in truth:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (x <= cx < x + chip and y <= cy < y + chip):
            continue
        left, right = max(x0, x), min(x1, x + chip)
        top, bottom = max(y0, y), min(y1, y + chip)
        lines.append(f"0 {((left + right) / 2 - x) / chip:.6f} {((top + bottom) / 2 - y) / chip:.6f} "
                     f"{(right - left) / chip:.6f} {(bottom - top) / chip:.6f}")
    return lines


def export_scene(name, img, truth, out_dir: Path, split: str, chip: int, empty_ratio: float | None,
                 rng: random.Random) -> tuple[int, int, int]:
    """Write one scene's chips. Returns (chips with vessels, empty chips kept, vessels).

    `empty_ratio` empty chips are kept per chip with vessels; None keeps all.
    Chips that are entirely black lie outside the satellite's swath and are skipped.
    """
    with_vessels, empty = [], []
    for x, y in chip_origins(img.width, img.height, chip):
        lines = chip_labels(truth, x, y, chip)
        (with_vessels if lines else empty).append((x, y, lines))
    if empty_ratio is not None:
        rng.shuffle(empty)
        empty = empty[:round(len(with_vessels) * empty_ratio)]

    kept_empty = vessels = 0
    for x, y, lines in with_vessels + empty:
        image = cut_chip(img, x, y, chip)
        if not lines and image.getbbox() is None:
            continue
        stem = f"{name}_x{x}_y{y}"
        image.save(out_dir / "images" / split / f"{stem}.png")
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        kept_empty += not lines
        vessels += len(lines)
    return len(with_vessels), kept_empty, vessels


def export_finland_chips(data_dir, out_dir, chip: int = CHIP, empty_ratio: float = 2.0, seed: int = 0,
                         splits: dict | None = None) -> Path:
    """Write the chip dataset and return the path of its dataset.yaml. Downloads scenes not in `data_dir` yet."""
    out_dir = Path(out_dir)
    dataset_yaml = out_dir / "dataset.yaml"
    if dataset_yaml.exists():
        return dataset_yaml

    splits = splits or {"train": TRAIN_SCENES, "val": VAL_SCENES, "test": TEST_SCENES}
    rng = random.Random(seed)
    for split, scenes in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        totals = np.zeros(3, dtype=int)
        for name, img, truth in load_scenes(data_dir, scenes):
            totals += export_scene(name, img, truth, out_dir, split, chip, None if split == "test" else empty_ratio, rng)
        print(f"{split}: {len(scenes)} scenes -> {totals[0]} chips with vessels, {totals[1]} without, "
              f"{totals[2]} labelled vessels", flush=True)

    with open(dataset_yaml, "w") as f:
        yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {0: CLASS_NAME}}, f, sort_keys=False)
    return dataset_yaml
