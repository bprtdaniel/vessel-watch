"""Score detectors and the two-stage pipeline on real Sentinel-2 imagery.

    python -m vesselwatch.detect.finland --name step_a --data-dir <folder> \\
        --detector high_res=runs/yolo11s_obb/weights/best.pt --reference \\
        --classifier configs/crops/level1_resnet101.yaml=runs/crops_level1_resnet101/best.pt

The benchmark is the test part of the Finnish Environment Institute's vessel
annotations (Mäyrä et al., zenodo.org/records/15019034, CC BY 4.0): seven
Sentinel-2 scenes of the Finnish coast with a box around every vessel and its
wake. The scenes are downloaded from Copernicus and cut into 320 x 320 pixel
chips, the size the published reference model was trained on.

Detection is scored against the boxes. The boxes carry no vessel type, so the
classifier cannot be scored here: its predictions are only tallied, and those
naming a ship class far longer than the box are counted as contradictions.
Writes <runs>/sentinel2_<name>.json and example images next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import RUNS_ENV, load_config
from ..data import crop_eval_transform, rotated_crop
from ..monitor.cdse import fetch_true_colour
from ..monitor.tiles import chip_origins, cut_chip, load_scene
from .pipeline import Detection, classify_crops, load_classifier
from .sizes import contradicts

ANNOTATIONS_URL = "https://zenodo.org/records/15019034/files/{tile}.gpkg?download=1"
REFERENCE_WEIGHTS_URL = "https://huggingface.co/mayrajeo/marine-vessel-yolo/resolve/main/yolo11s_tci.pt"
TEST_SCENES = [("34VEN", "20210714"), ("34VEN", "20220619"), ("34VEN", "20220624"), ("34VEN", "20220813"),
               ("34VER", "20220617"), ("34VER", "20220712"), ("34VER", "20220826")]
CHIP = 320            # pixels, 3.2 km at 10 m
CHIPS_AT_ONCE = 256   # chips cut and held in memory together
# By the long side of the annotated box, which includes the wake
LENGTH_BUCKETS = (("under 50 m", 0, 50), ("50 to 100 m", 50, 100), ("over 100 m", 100, float("inf")))
COLOURS = [(255, 70, 70), (70, 170, 255), (255, 200, 60)]


# --- data ---------------------------------------------------------------------------------------

def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def read_boxes(gpkg: Path, layer: str) -> list[tuple[float, float, float, float]]:
    """Annotated boxes of one acquisition as (min x, min y, max x, max y) in map coordinates.

    A GeoPackage is an SQLite file; each geometry is a short header followed by
    a well-known-binary polygon, read here directly to avoid a GIS dependency.
    """
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    boxes = []
    with sqlite3.connect(gpkg) as con:
        for (blob,) in con.execute(f'select geom from "{layer}"'):
            wkb = blob[8 + envelope_bytes[(blob[3] >> 1) & 7]:]
            order = "<" if wkb[0] == 1 else ">"
            n_points = struct.unpack(order + "I", wkb[9:13])[0]
            points = struct.unpack(order + f"{2 * n_points}d", wkb[13:13 + 16 * n_points])
            xs, ys = points[0::2], points[1::2]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def to_pixels(box, info: dict) -> tuple[float, float, float, float]:
    """A map-coordinate box as (x0, y0, x1, y1) in image pixels. Map y points north, image y down."""
    min_x, min_y, max_x, max_y = box
    res = info["resolution"]
    return ((min_x - info["ulx"]) / res, (info["uly"] - max_y) / res,
            (max_x - info["ulx"]) / res, (info["uly"] - min_y) / res)


def load_test_scenes(data_dir, scenes=TEST_SCENES):
    """Yield (name, image, truth boxes in pixels), downloading what is not in `data_dir` yet."""
    data_dir = Path(data_dir)
    for tile, date in scenes:
        gpkg = download(ANNOTATIONS_URL.format(tile=tile), data_dir / f"{tile}.gpkg")
        image, info = fetch_true_colour(tile, date, data_dir)
        truth = np.array([to_pixels(box, info) for box in read_boxes(gpkg, date)]).reshape(-1, 4)
        yield f"{tile}_{date}", load_scene(image), truth


# --- detection ----------------------------------------------------------------------------------

def chip_detector(weights, imgsz: int = 640, conf: float = 0.25, drop_classes=(), batch: int = 32):
    """A function mapping a list of chips to one list of Detections per chip.

    Works for oriented-box and ordinary box models; an ordinary box comes back
    as its four corners. Chips are enlarged to `imgsz` by the model.
    """
    from ultralytics import YOLO  # imported here so the rest of the package works without it

    model = YOLO(str(weights))

    def detect(chips):
        found = []
        for start in range(0, len(chips), batch):
            for result in model.predict(chips[start:start + batch], imgsz=imgsz, conf=conf, verbose=False):
                if result.obb is not None:
                    corners = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
                    boxes = result.obb
                else:
                    x0, y0, x1, y1 = result.boxes.xyxy.cpu().numpy().T
                    corners = np.stack([x0, y0, x0, y1, x1, y1, x1, y0], axis=1)
                    boxes = result.boxes
                labels = [result.names[int(c)] for c in boxes.cls.cpu().numpy()]
                found.append([Detection(tuple(map(float, c)), float(s), name)
                              for c, s, name in zip(corners, boxes.conf.cpu().numpy(), labels)
                              if name not in drop_classes])
        return found

    return detect


def detect_scene(img: Image.Image, detect, chip: int = CHIP) -> list[Detection]:
    """Run a chip detector over a whole scene; polygons come back in scene pixels."""
    origins = chip_origins(img.width, img.height, chip)
    detections = []
    for start in range(0, len(origins), CHIPS_AT_ONCE):
        group = origins[start:start + CHIPS_AT_ONCE]
        for (x, y), found in zip(group, detect([cut_chip(img, x, y, chip) for x, y in group])):
            for d in found:
                shifted = tuple(v + (x if i % 2 == 0 else y) for i, v in enumerate(d.polygon))
                detections.append(Detection(shifted, d.score, d.label))
    return detections


def bounds(detections) -> np.ndarray:
    """Upright (x0, y0, x1, y1) rectangle around each detection."""
    if not detections:
        return np.zeros((0, 4))
    corners = np.array([d.polygon for d in detections]).reshape(-1, 4, 2)
    return np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)


def match(truth: np.ndarray, found: np.ndarray, scores, min_iou: float | None) -> np.ndarray:
    """For each truth box the index of the detection that claims it, or -1.

    Detections go in order of confidence; each takes the free truth box it
    overlaps most. With `min_iou` the overlap must reach that IoU. With None
    the looser rule applies: the detection's centre lies inside the truth box,
    which suits vessels of a few pixels where IoU is brittle.
    """
    matched = np.full(len(truth), -1)
    if not len(truth) or not len(found):
        return matched
    truth_area = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
    for d in np.argsort(-np.asarray(scores)):
        box = found[d]
        width = np.clip(np.minimum(truth[:, 2], box[2]) - np.maximum(truth[:, 0], box[0]), 0, None)
        height = np.clip(np.minimum(truth[:, 3], box[3]) - np.maximum(truth[:, 1], box[1]), 0, None)
        inter = width * height
        iou = inter / (truth_area + (box[2] - box[0]) * (box[3] - box[1]) - inter + 1e-9)
        if min_iou is None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            ok = (truth[:, 0] <= cx) & (cx <= truth[:, 2]) & (truth[:, 1] <= cy) & (cy <= truth[:, 3])
        else:
            ok = iou >= min_iou
        ok &= matched == -1
        if ok.any():
            matched[np.where(ok, iou, -1).argmax()] = d
    return matched


def long_side_m(boxes: np.ndarray, resolution: float = 10) -> np.ndarray:
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * resolution


def empty_score() -> dict:
    return {"vessels": 0, "detections": 0, "found_iou50": 0, "found_centre": 0,
            "by_length": {name: {"vessels": 0, "found_centre": 0} for name, _, _ in LENGTH_BUCKETS}}


def add_scene(score: dict, truth: np.ndarray, detections) -> None:
    found = bounds(detections)
    scores = [d.score for d in detections]
    by_centre = match(truth, found, scores, None) >= 0
    score["vessels"] += len(truth)
    score["detections"] += len(detections)
    score["found_iou50"] += int((match(truth, found, scores, 0.5) >= 0).sum())
    score["found_centre"] += int(by_centre.sum())
    lengths = long_side_m(truth)
    for name, low, high in LENGTH_BUCKETS:
        mask = (lengths >= low) & (lengths < high)
        score["by_length"][name]["vessels"] += int(mask.sum())
        score["by_length"][name]["found_centre"] += int(by_centre[mask].sum())


def finish(score: dict) -> dict:
    def share(part, whole):
        return 100 * part / whole if whole else None

    score["recall_iou50"] = share(score["found_iou50"], score["vessels"])
    score["recall_centre"] = share(score["found_centre"], score["vessels"])
    score["precision_centre"] = share(score["found_centre"], score["detections"])
    for bucket in score["by_length"].values():
        bucket["recall_centre"] = share(bucket["found_centre"], bucket["vessels"])
    return score


# --- classification -----------------------------------------------------------------------------

def classify_boxes(img, boxes: np.ndarray, classifier, device) -> list[str]:
    """Class the classifier assigns to each (x0, y0, x1, y1) box of a scene."""
    model, classes, margin = classifier
    crops = [rotated_crop(img, (x0, y0, x0, y1, x1, y1, x1, y0), margin) for x0, y0, x1, y1 in boxes]
    return [classes[i] for i in classify_crops(model, crops, crop_eval_transform(), device)]


def tally(labels, lengths_m) -> dict:
    """What a classifier said about a set of boxes, without knowing the truth."""
    verdicts = [contradicts(label, length) for label, length in zip(labels, lengths_m)]
    named = [v for v in verdicts if v is not None]
    counts = Counter(labels)
    return {
        "crops": len(labels),
        "top_classes": [{"class": name, "share": 100 * n / len(labels)} for name, n in counts.most_common(5)],
        "named_ship_class": len(named),
        "named_class_far_too_long": sum(named),
    }


# --- everything together ------------------------------------------------------------------------

def save_examples(img, truth, detections_by_name: dict, out_dir: Path, scene: str, count: int = 4, zoom: int = 3):
    """The chips of a scene with the most vessels, enlarged, with truth (green) and each detector's boxes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = Counter((int(b[0] // CHIP), int(b[1] // CHIP)) for b in truth)
    for (cx, cy), _ in cells.most_common(count):
        x, y = cx * CHIP, cy * CHIP
        view = cut_chip(img, x, y, CHIP).resize((CHIP * zoom, CHIP * zoom), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(view)

        def outline(box, colour, grow):
            draw.rectangle(((box[0] - x) * zoom - grow, (box[1] - y) * zoom - grow,
                            (box[2] - x) * zoom + grow, (box[3] - y) * zoom + grow), outline=colour)

        for box in truth:
            outline(box, (60, 230, 90), 4)
        for i, (name, detections) in enumerate(detections_by_name.items()):
            colour = COLOURS[i % len(COLOURS)]
            for box in bounds(detections):
                if x - CHIP < box[0] < x + CHIP and y - CHIP < box[1] < y + CHIP:
                    outline(box, colour, 1)
            draw.text((6, 6 + 14 * (i + 1)), name, fill=colour)
        draw.text((6, 6), "labelled vessels", fill=(60, 230, 90))
        view.save(out_dir / f"{scene}_x{x}_y{y}.png")


def evaluate(scenes, detectors: dict, classifiers: dict | None = None, examples_dir: Path | None = None) -> dict:
    """Score every detector on every scene and tally every classifier.

    `scenes` yields (name, image, truth boxes in pixels). `detectors` maps a
    name to a chip detector; the first one is the pipeline's own detector, and
    its detections are what the classifiers are run on, next to the true boxes.
    `classifiers` maps a name to (model, class list, crop margin).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifiers = classifiers or {}
    scores = {name: empty_score() for name in detectors}
    labels = {name: {"true_boxes": ([], []), "pipeline_detections": ([], [])} for name in classifiers}
    scene_names = []

    for scene, img, truth in scenes:
        scene_names.append(scene)
        found = {}
        for name, detect in detectors.items():
            found[name] = detect_scene(img, detect)
            add_scene(scores[name], truth, found[name])
            print(f"{scene} | {name}: {len(truth)} labelled vessels, {len(found[name])} detections", flush=True)

        pipeline_boxes = bounds(next(iter(found.values()))) if found else np.zeros((0, 4))
        for name, classifier in classifiers.items():
            for key, boxes in (("true_boxes", truth), ("pipeline_detections", pipeline_boxes)):
                labels[name][key][0].extend(classify_boxes(img, boxes, classifier, device))
                labels[name][key][1].extend(long_side_m(boxes).tolist())
        if examples_dir is not None:
            save_examples(img, truth, found, examples_dir, scene)

    return {
        "scenes": scene_names,
        "chip": CHIP,
        "detectors": {name: finish(score) for name, score in scores.items()},
        "classifiers": {name: {key: tally(*pair) for key, pair in sets.items() if pair[0]}
                        for name, sets in labels.items()},
    }


def print_report(report: dict) -> None:
    print(f"\n{len(report['scenes'])} scenes, cut into {report['chip']} x {report['chip']} pixel chips")
    print(f"\n{'detector':<22} {'vessels':>8} {'detections':>11} {'found':>8} {'found (IoU 0.5)':>16} {'hits':>7}")
    print("-" * 78)
    for name, s in report["detectors"].items():
        def pct(v):
            return "-" if v is None else f"{v:.1f}%"
        print(f"{name:<22} {s['vessels']:>8} {s['detections']:>11} {pct(s['recall_centre']):>8} "
              f"{pct(s['recall_iou50']):>16} {pct(s['precision_centre']):>7}")
    print("\nfound: a detection's centre lies inside the vessel's box. hits: share of detections that found a vessel.")
    print("\nFound, by length of the labelled box (wake included):")
    for name, s in report["detectors"].items():
        parts = [f"{bucket} {b['found_centre']}/{b['vessels']}" for bucket, b in s["by_length"].items()]
        print(f"  {name:<20} " + " | ".join(parts))
    for name, sets in report["classifiers"].items():
        for key, t in sets.items():
            top = ", ".join(f"{c['class']} {c['share']:.0f}%" for c in t["top_classes"])
            print(f"\n{name} on {key.replace('_', ' ')} ({t['crops']} crops): {top}")
            if t["named_ship_class"]:
                print(f"  named a specific ship class {t['named_ship_class']} times; "
                      f"in {t['named_class_far_too_long']} the box is under half that class's length")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to <runs>/sentinel2_<name>.json")
    parser.add_argument("--data-dir", required=True, help="where scenes and annotations are kept between runs")
    parser.add_argument("--detector", action="append", default=[], metavar="NAME=WEIGHTS",
                        help="detector to score; repeat for several, the first one feeds the classifiers")
    parser.add_argument("--reference", action="store_true", help="also score the published Finnish model")
    parser.add_argument("--classifier", action="append", default=[], metavar="CONFIG=WEIGHTS")
    parser.add_argument("--drop-class", action="append", default=[], help="detector class to ignore, e.g. Dock")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scenes", type=int, help="use only the first N scenes, for a quick trial")
    parser.add_argument("--runs-dir")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    runs = Path(args.runs_dir or os.environ.get(RUNS_ENV, "runs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detectors = {}
    for spec in args.detector:
        name, weights = spec.split("=", 1)
        detectors[name] = chip_detector(weights, conf=args.conf, drop_classes=args.drop_class)
    if args.reference:
        weights = download(REFERENCE_WEIGHTS_URL, data_dir / "yolo11s_tci.pt")
        detectors["finnish_reference"] = chip_detector(weights, conf=args.conf)

    classifiers = {}
    for spec in args.classifier:
        config, weights = spec.split("=", 1)
        cfg = load_config(config, runs_dir=args.runs_dir)
        model, classes = load_classifier(cfg, weights, device)
        classifiers[cfg.run_name] = (model, classes, cfg.crop_margin)

    scenes = load_test_scenes(data_dir, TEST_SCENES[:args.scenes] if args.scenes else TEST_SCENES)
    report = evaluate(scenes, detectors, classifiers, examples_dir=runs / f"sentinel2_{args.name}_examples")
    report = {"name": args.name, "conf": args.conf, **report}

    runs.mkdir(parents=True, exist_ok=True)
    out = runs / f"sentinel2_{args.name}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
