"""Two-stage pipeline: detect vessels in a scene, cut each one out, classify the crop.

    python -m vesselwatch.detect.pipeline --name two_stage_level2 \\
        --detector runs/yolo11s_obb/weights/best.pt \\
        --classifier-config configs/crops/level2_resnet101.yaml \\
        --classifier-weights runs/crops_level2_resnet101/best.pt

Scored on the held-out test split against the ground-truth boxes, one stage at
a time: how many vessels the detector finds, how many of the found ones the
classifier types correctly, and how many vessels end up both found and typed
correctly. Writes <runs>/pipeline_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..checkpoint import load_checkpoint
from ..config import RUNS_ENV, load_config
from ..data import annotation_path, coco_to_vessel_records, crop_eval_transform, image_dir
from ..data.crops import CROPPERS
from ..data.transforms import squash_eval_transform
from ..metrics import accuracy
from ..models import build_model
from ..train import build_datasets

PREPROCESS = {"pad": crop_eval_transform, "squash": squash_eval_transform}


@dataclass
class Detection:
    polygon: tuple  # (x1, y1, ..., x4, y4) in image pixels
    score: float
    label: str      # the detector's own class name


def polygon_iou(a, b) -> float:
    """Intersection over union of two convex quadrilaterals."""
    a = cv2.convexHull(np.asarray(a, dtype=np.float32).reshape(4, 2))
    b = cv2.convexHull(np.asarray(b, dtype=np.float32).reshape(4, 2))
    area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / (area_a + area_b - inter))


def match_detections(truth, detections, iou_threshold: float = 0.5) -> list:
    """For each ground-truth polygon the index of its detection, or None.

    Detections are taken in order of confidence and each claims the free
    ground-truth box it overlaps most, if that overlap reaches the threshold.
    """
    matched = [None] * len(truth)
    for d in sorted(range(len(detections)), key=lambda i: -detections[i].score):
        best, best_iou = None, iou_threshold
        for t, polygon in enumerate(truth):
            if matched[t] is None:
                iou = polygon_iou(polygon, detections[d].polygon)
                if iou >= best_iou:
                    best, best_iou = t, iou
        if best is not None:
            matched[best] = d
    return matched


def yolo_detector(weights, imgsz: int = 1024, conf: float = 0.25, keep_classes=None):
    """A function mapping image paths to one list of Detections per image."""
    from ultralytics import YOLO  # imported here so the rest of the package works without it

    model = YOLO(str(weights))

    def detect(image_paths):
        for path in image_paths:
            result = model.predict(str(path), imgsz=imgsz, conf=conf, verbose=False)[0]
            corners = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            scores = result.obb.conf.cpu().numpy()
            labels = [result.names[int(c)] for c in result.obb.cls.cpu().numpy()]
            yield [Detection(tuple(map(float, c)), float(s), name)
                   for c, s, name in zip(corners, scores, labels)
                   if keep_classes is None or name in keep_classes]

    return detect


def load_classifier(cfg, weights, device):
    """(model in eval mode, class list). The original study's weights carry no class list."""
    ckpt = load_checkpoint(weights, map_location=device)
    classes = ckpt["classes"] or build_datasets(cfg)["train"].classes
    model = build_model(cfg.model, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), classes


@torch.no_grad()
def classify_crops(model, crops, transform, device, batch_size: int = 64) -> list[int]:
    predictions = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack([transform(c) for c in crops[start:start + batch_size]]).to(device)
        predictions.extend(model(batch).argmax(dim=1).tolist())
    return predictions


def run_pipeline(detect, classifier_cfg, classifier_weights, crop: str = "rotated", margin: float | None = None,
                 preprocess: str = "pad", iou_threshold: float = 0.5) -> dict:
    """Run both stages over the test split and score them against the ground truth.

    `detect` maps image paths to Detections, see `yolo_detector`. All figures
    are in percent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_classifier(classifier_cfg, classifier_weights, device)
    transform = PREPROCESS[preprocess]()
    margin = classifier_cfg.crop_margin if margin is None else margin

    root = classifier_cfg.resolved_data_root()
    truth = coco_to_vessel_records(annotation_path(root, "val", classifier_cfg.level))
    filenames = sorted(truth["filename"].unique())

    n_truth = n_detections = n_found = n_found_and_typed = 0
    for filename, detections in zip(filenames, detect([image_dir(root) / f for f in filenames])):
        vessels = truth[truth["filename"] == filename]
        n_truth += len(vessels)
        n_detections += len(detections)
        if not detections:
            continue

        img = Image.open(image_dir(root) / filename).convert("RGB")
        crops = [CROPPERS[crop](img, d.polygon, margin) for d in detections]
        predicted = [classes[i] for i in classify_crops(model, crops, transform, device)]

        for label, match in zip(vessels["label"], match_detections(list(vessels["polygon"]), detections, iou_threshold)):
            if match is not None:
                n_found += 1
                n_found_and_typed += predicted[match] == label

    def share(part, whole):
        return 100 * part / whole if whole else None

    return {
        "images": len(filenames),
        "vessels": n_truth,
        "detections": n_detections,
        "iou_threshold": iou_threshold,
        "detection_recall": share(n_found, n_truth),
        "detection_precision": share(n_found, n_detections),
        "classification_accuracy_on_found": share(n_found_and_typed, n_found),
        "end_to_end_recall": share(n_found_and_typed, n_truth),
    }


def classifier_reference(classifier_cfg) -> float | None:
    """Test accuracy of the classifier on ground-truth crops, if its run was scored."""
    metrics = classifier_cfg.run_dir() / "metrics_test.json"
    return json.loads(metrics.read_text())["accuracy"] if metrics.exists() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to <runs>/pipeline_<name>.json")
    parser.add_argument("--detector", required=True, help="YOLO-OBB weights")
    parser.add_argument("--classifier-config", required=True)
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--crop", default="rotated", choices=sorted(CROPPERS))
    parser.add_argument("--margin", type=float, help="defaults to the classifier's crop_margin")
    parser.add_argument("--preprocess", default="pad", choices=sorted(PREPROCESS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keep-class", action="append", help="detector class to keep; repeat for several")
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.classifier_config, data_root=args.data_root, runs_dir=args.runs_dir)
    detect = yolo_detector(args.detector, args.imgsz, args.conf, args.keep_class)
    report = run_pipeline(detect, cfg, args.classifier_weights, args.crop, args.margin, args.preprocess)
    report = {"name": args.name, "detector": str(args.detector), "classifier": str(args.classifier_weights),
              "level": cfg.level, "crop": args.crop, "preprocess": args.preprocess, "conf": args.conf,
              "classifier_accuracy_on_true_boxes": classifier_reference(cfg), **report}

    out = Path(args.runs_dir or os.environ.get(RUNS_ENV, "runs")) / f"pipeline_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{args.name} | level {cfg.level} | vessels: {report['vessels']} | detections: {report['detections']}")
    print(f"  detection recall:               {report['detection_recall']:6.2f}%")
    print(f"  detection precision:            {report['detection_precision']:6.2f}%")
    print(f"  classified correctly, if found: {report['classification_accuracy_on_found']:6.2f}%")
    print(f"  found and classified correctly: {report['end_to_end_recall']:6.2f}%")
    if report["classifier_accuracy_on_true_boxes"] is not None:
        print(f"  classifier on true boxes:       {report['classifier_accuracy_on_true_boxes']:6.2f}%")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
