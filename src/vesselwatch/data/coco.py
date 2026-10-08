"""Reading the ShipRSImageNet COCO annotation files."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def annotation_path(data_root: str | Path, split: str, level: int) -> Path:
    return Path(data_root) / "COCO_Format" / f"ShipRSImageNet_bbox_{split}_level_{level}.json"


def image_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "VOC_Format" / "JPEGImages"


def category_names(coco_json: str | Path) -> list[str]:
    """Sorted names of every category the file defines, used as the class list."""
    with open(coco_json) as f:
        return sorted(cat["name"] for cat in json.load(f)["categories"])


def coco_to_vessel_records(coco_json: str | Path) -> pd.DataFrame:
    """One row per annotated vessel: filename, image_id, label, polygon, area.

    `polygon` is the oriented box as (x1, y1, ..., x4, y4), taken from the
    annotation's `segmentation`. Annotations without four corner points fall
    back to the corners of the upright `bbox`.
    """
    with open(coco_json) as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    records = []
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        seg = ann.get("segmentation") or []
        if seg and len(seg[0]) == 8:
            polygon = tuple(float(v) for v in seg[0])
        else:
            polygon = (x, y, x, y + h, x + w, y + h, x + w, y)
        records.append({
            "filename": images[ann["image_id"]],
            "image_id": ann["image_id"],
            "label": categories[ann["category_id"]],
            "polygon": polygon,
            "area": float(ann.get("area", w * h)),
        })
    return pd.DataFrame(records, columns=["filename", "image_id", "label", "polygon", "area"])


def split_images(image_ids, val_fraction: float, seed: int) -> tuple[set, set]:
    """Split image ids into (train, val). Splitting by image keeps every vessel of a scene on one side."""
    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    return set(ids[n_val:]), set(ids[:n_val])


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
