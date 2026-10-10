"""Cutting a large scene into square chips for inference."""
from __future__ import annotations

from PIL import Image

# A Sentinel-2 tile is 10980 x 10980 pixels, more than Pillow accepts by default
Image.MAX_IMAGE_PIXELS = None


def load_scene(path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def chip_origins(width: int, height: int, size: int) -> list[tuple[int, int]]:
    """Upper-left corners of the chips covering an image, row by row, without overlap."""
    return [(x, y) for y in range(0, height, size) for x in range(0, width, size)]


def cut_chip(img: Image.Image, x: int, y: int, size: int) -> Image.Image:
    """A size x size chip. Chips at the right and bottom edge are filled up with black."""
    chip = img.crop((x, y, min(x + size, img.width), min(y + size, img.height)))
    if chip.size == (size, size):
        return chip
    padded = Image.new("RGB", (size, size))
    padded.paste(chip, (0, 0))
    return padded
