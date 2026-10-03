"""Reading the ShipRSImageNet COCO annotation files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def annotation_path(data_root: str | Path, split: str, level: int) -> Path:
    return Path(data_root) / "COCO_Format" / f"ShipRSImageNet_bbox_{split}_level_{level}.json"


def image_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "VOC_Format" / "JPEGImages"


def coco_to_image_labels(coco_json: str | Path) -> pd.DataFrame:
    """One row per image: filename and the most frequent category among its boxes.

    This is the labelling of the original study. Images with several vessel
    types get a single label, and images without annotations are dropped.
    """
    with open(coco_json) as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    img_to_cats = {}
    for ann in data["annotations"]:
        img_to_cats.setdefault(ann["image_id"], []).append(ann["category_id"])

    records = []
    for img_id, fname in images.items():
        if img_id in img_to_cats:
            cat_ids = img_to_cats[img_id]
            main_cat = max(set(cat_ids), key=cat_ids.count)
            records.append({"filename": fname, "label": categories[main_cat]})
    return pd.DataFrame(records, columns=["filename", "label"])
