"""Ground resolution of the ShipRSImageNet images, and what is left of them at a coarser one.

    python -m vesselwatch.data.resolution --target 10

The dataset mixes sources between about 0.12 and 6 m per pixel. Each image's
resolution is recorded in its VOC annotation file as `Img_Resolution`. This
prints how the images spread over resolutions, how small the scenes become
when resampled to the target resolution, and how long the vessels are in
metres, which decides how many remain visible.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import DATA_ENV
from .coco import annotation_path, coco_to_vessel_records
from .crops import box_sides

LENGTH_STEPS_M = (10, 20, 30, 50, 100, 150, 200)


def annotation_dir(data_root) -> Path:
    return Path(data_root) / "VOC_Format" / "Annotations"


def image_resolutions(data_root) -> dict[str, float]:
    """Metres per pixel of each image, keyed by file stem. Images without a usable value are left out."""
    resolutions = {}
    for xml in annotation_dir(data_root).glob("*.xml"):
        match = re.search(r"<Img_Resolution>\s*([\d.]+)\s*</Img_Resolution>", xml.read_text(errors="ignore"))
        if match and float(match.group(1)) > 0:
            resolutions[xml.stem] = float(match.group(1))
    return resolutions


def vessel_table(data_root, level: int = 1, splits=("train", "val")):
    """Every annotated vessel with its image's resolution and its length and width in metres."""
    import pandas as pd

    resolutions = image_resolutions(data_root)
    frames = []
    for split in splits:
        records = coco_to_vessel_records(annotation_path(data_root, split, level))
        records["split"] = split
        frames.append(records)
    vessels = pd.concat(frames, ignore_index=True)
    vessels["resolution"] = [resolutions.get(Path(name).stem) for name in vessels["filename"]]
    sides = np.array([box_sides(polygon) for polygon in vessels["polygon"]]).reshape(-1, 2)
    vessels["length_m"] = sides[:, 0] * vessels["resolution"]
    vessels["width_m"] = sides[:, 1] * vessels["resolution"]
    return vessels


def summary(data_root, target: float = 10.0, level: int = 1) -> dict:
    vessels = vessel_table(data_root, level)
    known = vessels.dropna(subset=["resolution"])
    images = known.drop_duplicates("filename")
    scene_px = np.maximum(images["width"], images["height"]) * images["resolution"] / target
    lengths = known["length_m"].to_numpy()

    def percentiles(values):
        return {f"p{p}": float(np.percentile(values, p)) for p in (0, 10, 50, 90, 100)}

    by_resolution = Counter(round(r, 2) for r in images["resolution"])
    vessels_by_resolution = Counter(round(r, 2) for r in known["resolution"])
    return {
        "target_resolution": target,
        "vessels": int(len(vessels)),
        "vessels_without_resolution": int(len(vessels) - len(known)),
        "images": int(len(images)),
        "by_resolution": [{"resolution": r, "images": n, "vessels": vessels_by_resolution[r]}
                          for r, n in sorted(by_resolution.items())],
        "scene_long_side_px_at_target": percentiles(scene_px),
        "vessel_length_m": percentiles(lengths),
        "vessels_at_least": {f"{m} m": int((lengths >= m).sum()) for m in LENGTH_STEPS_M},
        "by_class": [
            {"class": label, "vessels": int(len(group)), "median_length_m": float(group["length_m"].median()),
             "at_least_30_m": int((group["length_m"] >= 30).sum())}
            for label, group in sorted(known.groupby("label"), key=lambda item: -len(item[1]))
        ],
    }


def print_summary(report: dict) -> None:
    target = report["target_resolution"]
    print(f"{report['images']} images, {report['vessels']} annotated vessels "
          f"({report['vessels_without_resolution']} in images without a recorded resolution)")

    print(f"\n{'m per pixel':>12} {'images':>8} {'vessels':>9}")
    for row in report["by_resolution"]:
        print(f"{row['resolution']:>12.2f} {row['images']:>8} {row['vessels']:>9}")

    px = report["scene_long_side_px_at_target"]
    print(f"\nLong side of a scene resampled to {target:g} m, in pixels: "
          f"smallest {px['p0']:.0f}, 10% {px['p10']:.0f}, median {px['p50']:.0f}, 90% {px['p90']:.0f}, largest {px['p100']:.0f}")

    m = report["vessel_length_m"]
    print(f"\nVessel length in metres: shortest {m['p0']:.0f}, 10% {m['p10']:.0f}, median {m['p50']:.0f}, "
          f"90% {m['p90']:.0f}, longest {m['p100']:.0f}")
    total = report["vessels"] - report["vessels_without_resolution"]
    for step, n in report["vessels_at_least"].items():
        metres = float(step.split()[0])
        print(f"  at least {step:>6} ({metres / target:>4.1f} px at {target:g} m): {n:>6} vessels, {100 * n / total:5.1f}%")

    print(f"\n{'class':<14} {'vessels':>8} {'median length':>14} {'at least 30 m':>14}")
    for row in report["by_class"]:
        print(f"{row['class']:<14} {row['vessels']:>8} {row['median_length_m']:>12.0f} m {row['at_least_30_m']:>14}")


def main(argv=None):
    import os

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get(DATA_ENV))
    parser.add_argument("--target", type=float, default=10.0, help="resolution to resample to, in metres per pixel")
    parser.add_argument("--level", type=int, default=1, help="label level used for the per-class table")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error(f"give --data-root or set {DATA_ENV}")
    print_summary(summary(args.data_root, args.target, args.level))


if __name__ == "__main__":
    main()
