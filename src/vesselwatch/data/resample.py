"""Bringing images and their vessel boxes to a coarser ground resolution."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .crops import box_sides


def resampled_size(width: int, height: int, factor: float) -> tuple[int, int]:
    return max(1, round(width * factor)), max(1, round(height * factor))


def resample_image(img: Image.Image, factor: float) -> Image.Image:
    """Shrink by `factor` with area averaging, as a sensor with larger pixels would see the scene."""
    return img.resize(resampled_size(img.width, img.height, factor), Image.Resampling.BOX)


def resampled_records(records, resolutions: dict[str, float], target: float, min_length_m: float = 0.0):
    """Vessel records as they are at `target` metres per pixel.

    `resolutions` maps a file stem to the image's metres per pixel. Polygons,
    areas and image sizes are scaled; `factor` (the shrink factor of the image),
    `resolution` and `length_m` are added. Vessels in images without a
    resolution and vessels shorter than `min_length_m` are dropped.
    """
    rows = []
    for row in records.to_dict("records"):
        resolution = resolutions.get(Path(row["filename"]).stem)
        if resolution is None:
            continue
        length_m = box_sides(row["polygon"])[0] * resolution
        if length_m < min_length_m:
            continue
        factor = resolution / target
        width, height = resampled_size(row["width"], row["height"], factor)
        fx, fy = width / row["width"], height / row["height"]
        polygon = tuple(v * (fx if i % 2 == 0 else fy) for i, v in enumerate(row["polygon"]))
        rows.append({**row, "polygon": polygon, "area": row["area"] * fx * fy, "width": width, "height": height,
                     "factor": factor, "resolution": resolution, "length_m": length_m})
    columns = [*records.columns, "factor", "resolution", "length_m"]
    return type(records)(rows, columns=columns)
