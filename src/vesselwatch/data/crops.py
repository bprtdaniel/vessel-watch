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


def fixed_crop(img: Image.Image, polygon, window: int) -> Image.Image:
    """A window x window pixel crop centred on the box and turned so the vessel lies horizontally.

    Unlike `rotated_crop` nothing is rescaled: a longer vessel fills more of
    the window. At coarse resolution size is most of what tells vessels apart.
    """
    pts = _corners(polygon, 0.0)
    centre = pts.mean(axis=0)
    first, second = pts[1] - pts[0], pts[2] - pts[1]
    along = first if np.linalg.norm(first) >= np.linalg.norm(second) else second
    norm = np.linalg.norm(along)
    along = along / norm if norm > 0 else np.array([1.0, 0.0])
    across = np.array([-along[1], along[0]])
    half = window / 2
    # QUAD takes the source corners as upper left, lower left, lower right, upper right
    quad = [centre - along * half - across * half, centre - along * half + across * half,
            centre + along * half + across * half, centre + along * half - across * half]
    return img.transform((window, window), Image.Transform.QUAD, data=tuple(np.concatenate(quad)),
                         resample=Image.Resampling.BILINEAR)


CROPPERS = {"rotated": rotated_crop, "upright": upright_crop}
CROP_MODES = (*CROPPERS, "fixed")


def crop_vessel(img: Image.Image, polygon, crop: str, margin: float = 0.0, window: int = 56) -> Image.Image:
    """Cut one vessel out: `rotated` and `upright` scale to the box plus margin, `fixed` uses a fixed window."""
    if crop == "fixed":
        return fixed_crop(img, polygon, window)
    return CROPPERS[crop](img, polygon, margin)
