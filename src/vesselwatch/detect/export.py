"""Write ShipRSImageNet in the YOLO oriented-box layout.

    <out>/images/{train,val,test}/<image>      links to the original files
    <out>/labels/{train,val,test}/<image>.txt  one line per vessel: class x1 y1 x2 y2 x3 y3 x4 y4
    <out>/dataset.yaml

Corner coordinates are divided by the image size. The splits are those of the
classification study: validation is carved out of the official train split by
image, and the official val split is the held-out test set.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from ..data import annotation_path, category_names, coco_to_vessel_records, image_dir, split_images


def _link(src: Path, dst: Path) -> None:
    """Symlink where the system allows it, otherwise copy."""
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _label_line(class_index: int, polygon, width: float, height: float) -> str:
    coords = []
    for i, value in enumerate(polygon):
        size = width if i % 2 == 0 else height
        coords.append(min(max(value / size, 0.0), 1.0))
    return f"{class_index} " + " ".join(f"{c:.6f}" for c in coords)


def export_yolo_obb(data_root, out_dir, level: int = 0, val_fraction: float = 0.15, split_seed: int = 0) -> Path:
    """Write the dataset and return the path of its dataset.yaml."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    train_json = annotation_path(data_root, "train", level)
    classes = category_names(train_json)

    records = coco_to_vessel_records(train_json)
    train_ids, val_ids = split_images(records["image_id"], val_fraction, split_seed)
    splits = {
        "train": records[records["image_id"].isin(train_ids)],
        "val": records[records["image_id"].isin(val_ids)],
        "test": coco_to_vessel_records(annotation_path(data_root, "val", level)),
    }

    for split, split_records in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for filename, vessels in split_records.groupby("filename"):
            _link(image_dir(data_root) / filename, out_dir / "images" / split / filename)
            lines = [_label_line(classes.index(v.label), v.polygon, v.width, v.height)
                     for v in vessels.itertuples()]
            (out_dir / "labels" / split / f"{Path(filename).stem}.txt").write_text("\n".join(lines) + "\n")

    dataset_yaml = out_dir / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        yaml.safe_dump({
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": dict(enumerate(classes)),
        }, f, sort_keys=False)
    return dataset_yaml
