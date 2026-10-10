"""Detector training images at a coarse resolution, composed from resampled scenes.

Resampled to 10 m a ShipRSImageNet scene is only some 30 to 60 pixels wide,
while the detector is to see 320 pixel chips of Sentinel-2 imagery. Enlarging
a scene would change the size of its vessels, so instead many scenes are laid
side by side on a canvas at true scale, like a collage. Several passes over
the scenes, each in a new order and with new flips, give varied canvases.

    <out>/images/{train,val,test}/canvas_00000.png
    <out>/labels/{train,val,test}/canvas_00000.txt   class x1 y1 x2 y2 x3 y3 x4 y4
    <out>/dataset.yaml
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml
from PIL import Image

from ..data import annotation_path, category_names, coco_to_vessel_records, image_dir, split_images
from ..data.resample import resample_image, resampled_records
from ..data.resolution import assigned_resolutions
from .export import _label_line


def flip_scene(img: Image.Image, vessels, horizontal: bool, vertical: bool):
    """Mirror a scene and its (class, polygon) vessels."""
    if horizontal:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        vessels = [(c, tuple(img.width - v if i % 2 == 0 else v for i, v in enumerate(p))) for c, p in vessels]
    if vertical:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        vessels = [(c, tuple(img.height - v if i % 2 == 1 else v for i, v in enumerate(p))) for c, p in vessels]
    return img, vessels


def pack(scenes, size: int, background=(20, 40, 45)):
    """Lay scenes on size x size canvases, row by row, each at its own scale.

    `scenes` is a list of (image, vessels) with vessels as (class, polygon) in
    the scene's pixels. Yields (canvas, vessels in canvas pixels). Scenes
    larger than a canvas are skipped.
    """
    canvas, placed, x, y, row_height = None, [], 0, 0, 0
    for img, vessels in scenes:
        if img.width > size or img.height > size:
            continue
        if canvas is not None and x + img.width > size:      # next row
            x, y, row_height = 0, y + row_height, 0
        if canvas is not None and y + img.height > size:     # canvas full
            yield canvas, placed
            canvas = None
        if canvas is None:
            canvas, placed, x, y, row_height = Image.new("RGB", (size, size), background), [], 0, 0, 0
        canvas.paste(img, (x, y))
        placed.extend((c, tuple(v + (x if i % 2 == 0 else y) for i, v in enumerate(p))) for c, p in vessels)
        x, row_height = x + img.width, max(row_height, img.height)
    if canvas is not None:
        yield canvas, placed


def load_scenes(data_root, records, all_records, classes):
    """One (image, vessels) per scene of `all_records`, resampled; only vessels in `records` are labelled.

    `records` is `all_records` without the vessels too short to label, so a
    scene whose vessels are all too short still comes along as background.
    """
    labelled = {name: group for name, group in records.groupby("filename")}
    scenes = []
    for name, factor in all_records.drop_duplicates("filename")[["filename", "factor"]].itertuples(index=False):
        img = resample_image(Image.open(image_dir(data_root) / name).convert("RGB"), factor)
        vessels = [(classes.index(v.label), v.polygon) for v in labelled[name].itertuples()] if name in labelled else []
        scenes.append((img, vessels))
    return scenes


def export_canvases(data_root, out_dir, target: float = 10.0, level: int = 0, val_fraction: float = 0.15,
                    split_seed: int = 0, min_length_m: float = 20.0, size: int = 320, passes: int = 20,
                    seed: int = 0) -> Path:
    """Write the canvas dataset and return the path of its dataset.yaml.

    Train canvases come from `passes` passes over the train scenes, validation
    from a fifth as many, test from a single pass without flips.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    dataset_yaml = out_dir / "dataset.yaml"
    if dataset_yaml.exists():
        return dataset_yaml

    train_json = annotation_path(data_root, "train", level)
    classes = category_names(train_json)
    resolutions = assigned_resolutions(str(data_root))
    records = coco_to_vessel_records(train_json)
    train_ids, val_ids = split_images(records["image_id"], val_fraction, split_seed)
    parts = {
        "train": (records[records["image_id"].isin(train_ids)], passes, True),
        "val": (records[records["image_id"].isin(val_ids)], max(1, passes // 5), True),
        "test": (coco_to_vessel_records(annotation_path(data_root, "val", level)), 1, False),
    }

    rng = random.Random(seed)
    for split, (split_records, split_passes, flips) in parts.items():
        everything = resampled_records(split_records, resolutions, target)
        labelled = resampled_records(split_records, resolutions, target, min_length_m)
        scenes = load_scenes(data_root, labelled, everything, classes)
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        count = vessels_written = 0
        for _ in range(split_passes):
            order = scenes[:]
            if flips:
                rng.shuffle(order)
                order = [flip_scene(img, vessels, rng.random() < 0.5, rng.random() < 0.5) for img, vessels in order]
            for canvas, placed in pack(order, size):
                name = f"canvas_{count:05d}"
                canvas.save(out_dir / "images" / split / f"{name}.png")
                lines = [_label_line(c, polygon, size, size) for c, polygon in placed]
                (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                count += 1
                vessels_written += len(placed)
        print(f"{split}: {len(scenes)} scenes at {target:g} m, {len(labelled)} vessels of at least {min_length_m:g} m "
              f"(of {len(split_records)} annotated) -> {count} canvases, {vessels_written} labelled boxes", flush=True)

    with open(dataset_yaml, "w") as f:
        yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": dict(enumerate(classes))}, f, sort_keys=False)
    return dataset_yaml
