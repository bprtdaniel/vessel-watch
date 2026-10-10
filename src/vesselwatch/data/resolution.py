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
from functools import lru_cache
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


def _tag(xml_text: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", xml_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def image_sources(data_root) -> dict[str, dict]:
    """Per image, keyed by file stem: where it comes from and the resolution text exactly as recorded."""
    sources = {}
    for xml in annotation_dir(data_root).glob("*.xml"):
        text = xml.read_text(errors="ignore")
        sources[xml.stem] = {"source": _tag(text, "dataset_source"), "sensor": _tag(text, "database"),
                             "recorded": _tag(text, "Img_Resolution")}
    return sources


def name_pattern(stem: str) -> str:
    """Rough family of a file name, with digits replaced by 9: `1472__1840_0` becomes `9__9_9`."""
    return re.sub(r"\d+", "9", stem)


def source_report(data_root, level: int = 3) -> list[dict]:
    """Recorded against measured resolution, per group of images with the same origin.

    The measured value uses vessels of a named ship class: the class's known
    hull length divided by the length of its box in pixels. Where the two
    disagree, the recorded value is wrong, or missing.
    """
    import pandas as pd

    from ..detect.sizes import CLASS_LENGTH_M

    sources = image_sources(data_root)
    vessels = pd.concat([coco_to_vessel_records(annotation_path(data_root, split, level)) for split in ("train", "val")],
                        ignore_index=True)
    stems = [Path(name).stem for name in vessels["filename"]]
    empty = {"source": "", "sensor": "", "recorded": ""}
    for key in ("source", "sensor", "recorded"):
        vessels[key] = [sources.get(stem, empty)[key] or "-" for stem in stems]
    vessels["pattern"] = [name_pattern(stem) for stem in stems]
    vessels["length_px"] = [box_sides(polygon)[0] for polygon in vessels["polygon"]]
    vessels["measured"] = [CLASS_LENGTH_M[label] / px if label in CLASS_LENGTH_M and px > 0 else np.nan
                           for label, px in zip(vessels["label"], vessels["length_px"])]

    rows = []
    for (source, sensor, recorded, pattern), group in vessels.groupby(["source", "sensor", "recorded", "pattern"]):
        measured = group["measured"].dropna().to_numpy()
        rows.append({
            "source": source, "sensor": sensor, "recorded": recorded, "file_names": pattern,
            "images": int(group["filename"].nunique()), "vessels": int(len(group)),
            "named_class_vessels": int(len(measured)),
            "measured_median": float(np.median(measured)) if len(measured) else None,
            "measured_p10": float(np.percentile(measured, 10)) if len(measured) else None,
            "measured_p90": float(np.percentile(measured, 90)) if len(measured) else None,
            "median_box_length_px": float(group["length_px"].median()),
        })
    return sorted(rows, key=lambda row: -row["vessels"])


def print_source_report(rows: list[dict]) -> None:
    print(f"{'source':<14} {'sensor':<16} {'recorded':>9} {'file names':<12} {'images':>7} {'vessels':>8} "
          f"{'named':>6} {'measured m/px (10% - median - 90%)':>36} {'box px':>7}")
    print("-" * 124)
    for r in rows:
        if r["measured_median"] is None:
            measured = "-"
        else:
            measured = f"{r['measured_p10']:.2f} - {r['measured_median']:.2f} - {r['measured_p90']:.2f}"
        print(f"{r['source'][:14]:<14} {r['sensor'][:16]:<16} {r['recorded'][:9]:>9} {r['file_names'][:12]:<12} "
              f"{r['images']:>7} {r['vessels']:>8} {r['named_class_vessels']:>6} {measured:>36} {r['median_box_length_px']:>7.0f}")
    print("\nrecorded: the Img_Resolution text in the annotation file. named: vessels of a ship class with a known hull")
    print("length. measured: that length divided by the box length in pixels. box px: median box length in the group.")


# Sources that record nothing and hold no vessels of a named class: their documented nominal resolution
NOMINAL_M = {"Airbus ship": 1.5}
MIN_NAMED_IMAGES = 20  # images with a measurement before a source's median is trusted


def measured_by_image(data_root, level: int = 3) -> dict[str, float]:
    """Metres per pixel of each image that holds a vessel of a named ship class, keyed by file stem.

    Per vessel: the class's hull length divided by the box length in pixels;
    per image the median over its named vessels.
    """
    from ..detect.sizes import CLASS_LENGTH_M

    per_image: dict[str, list[float]] = {}
    for split in ("train", "val"):
        path = annotation_path(data_root, split, level)
        if not path.exists():
            continue
        records = coco_to_vessel_records(path)
        for filename, label, polygon in zip(records["filename"], records["label"], records["polygon"]):
            length_px = box_sides(polygon)[0]
            if label in CLASS_LENGTH_M and length_px > 0:
                per_image.setdefault(Path(filename).stem, []).append(CLASS_LENGTH_M[label] / length_px)
    return {stem: float(np.median(values)) for stem, values in per_image.items()}


@lru_cache(maxsize=4)
def assigned_resolutions(data_root: str) -> dict[str, float]:
    """The resolution to use for each image, keyed by file stem.

    The recorded `Img_Resolution` is missing for half the images and wrong for
    one source, so it is the last resort. In order: the image's own measured
    value, held within a factor of two of its source's median; that median; the
    recorded value; the source's nominal value. Images with none are left out.
    """
    sources = image_sources(data_root)
    measured = measured_by_image(data_root)
    by_source: dict[str, list[float]] = {}
    for stem, value in measured.items():
        by_source.setdefault(sources.get(stem, {}).get("source", ""), []).append(value)
    medians = {source: float(np.median(values)) for source, values in by_source.items()
               if len(values) >= MIN_NAMED_IMAGES}

    assigned = {}
    for stem, meta in sources.items():
        median = medians.get(meta["source"])
        recorded = re.fullmatch(r"[\d.]+", meta["recorded"])
        if stem in measured and median:
            assigned[stem] = float(np.clip(measured[stem], median / 2, median * 2))
        elif median:
            assigned[stem] = median
        elif recorded and float(meta["recorded"]) > 0:
            assigned[stem] = float(meta["recorded"])
        elif meta["source"] in NOMINAL_M:
            assigned[stem] = NOMINAL_M[meta["source"]]
    return assigned


def main(argv=None):
    import os

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get(DATA_ENV))
    parser.add_argument("--target", type=float, default=10.0, help="resolution to resample to, in metres per pixel")
    parser.add_argument("--level", type=int, help="label level: default 1 for the summary, 3 with --sources")
    parser.add_argument("--sources", action="store_true",
                        help="instead, compare recorded with measured resolution per image source")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error(f"give --data-root or set {DATA_ENV}")
    if args.sources:
        print_source_report(source_report(args.data_root, args.level or 3))
    else:
        print_summary(summary(args.data_root, args.target, args.level or 1))


if __name__ == "__main__":
    main()
