import json

import numpy as np
import pytest
from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "Other Ship"},
    {"id": 2, "name": "Warship"},
    {"id": 3, "name": "Merchant"},
    {"id": 4, "name": "Dock"},
]


def _write_split(root, split, level, image_cats, start_id):
    """image_cats: one list of category ids per image; an empty list means no annotations."""
    rng = np.random.default_rng(start_id)
    images, annotations = [], []
    for i, cats in enumerate(image_cats):
        img_id = start_id + i
        fname = f"{img_id:06d}.bmp"
        pixels = rng.integers(0, 255, size=(300, 360, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "VOC_Format" / "JPEGImages" / fname)
        images.append({"id": img_id, "file_name": fname, "width": 360, "height": 300})
        for cat in cats:
            annotations.append({"id": len(annotations) + 1, "image_id": img_id,
                                "category_id": cat, "bbox": [10, 10, 50, 20], "area": 1000.0,
                                "segmentation": [[10, 10, 10, 30, 60, 30, 60, 10]]})
    path = root / "COCO_Format" / f"ShipRSImageNet_bbox_{split}_level_{level}.json"
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, f)


@pytest.fixture
def data_root(tmp_path):
    """Tiny synthetic dataset laid out like ShipRSImageNet_V1, level 1."""
    root = tmp_path / "ShipRSImageNet_V1"
    (root / "COCO_Format").mkdir(parents=True)
    (root / "VOC_Format" / "JPEGImages").mkdir(parents=True)
    train = [[1], [2, 2, 3], [3], [4], [1, 1], [2], [3, 3, 1], [4], []]
    val = [[1], [2], [3], [2, 2, 1]]  # no Dock in val
    _write_split(root, "train", 1, train, start_id=1)
    _write_split(root, "val", 1, val, start_id=101)
    return root
