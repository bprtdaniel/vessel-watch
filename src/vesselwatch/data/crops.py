"""Cutting single vessels out of a scene."""
from __future__ import annotations

import numpy as np
from PIL import Image

MIN_SIDE = 2  # pixels; keeps degenerate boxes loadable


def _corners(polygon, margin: float) -> np.ndarray:
    """The four corners in order around the box, grown by `margin` on each side."""
    pts = np.asarray(polygon, dtype=float).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    return centre + (pts - centre) * (1 + 2 * margin)


def box_sides(polygon) -> tuple[float, float]:
    """Length of the long and of the short side of an oriented box, in pixels."""
    pts = _corners(polygon, 0.0)
    a, b = float(np.linalg.norm(pts[1] - pts[0])), float(np.linalg.norm(pts[2] - pts[1]))
    return max(a, b), min(a, b)


def rotated_crop(img: Image.Image, polygon, margin: float = 0.0) -> Image.Image:
    """Crop along the oriented box, so the vessel lies horizontally and fills the crop.

    Which end of the vessel points left is not defined by the annotation.
    Parts of the box outside the image come back black.
    """
    pts = _corners(polygon, margin)
    if np.linalg.norm(pts[1] - pts[0]) > np.linalg.norm(pts[2] - pts[1]):
        pts = np.roll(pts, -1, axis=0)  # make the first edge the short one
    height = max(MIN_SIDE, round(float(np.linalg.norm(pts[1] - pts[0]))))
    width = max(MIN_SIDE, round(float(np.linalg.norm(pts[2] - pts[1]))))
    # QUAD takes the source corners as upper left, lower left, lower right, upper right
    return img.transform((width, height), Image.Transform.QUAD, data=tuple(pts.flatten()),
                         resample=Image.Resampling.BILINEAR)


def upright_crop(img: Image.Image, polygon, margin: float = 0.0) -> Image.Image:
    """Crop the axis-aligned rectangle around the oriented box."""
    pts = _corners(polygon, margin)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    return img.crop((x1, y1, max(x2, x1 + MIN_SIDE), max(y2, y1 + MIN_SIDE)))


CROPPERS = {"rotated": rotated_crop, "upright": upright_crop}
