import numpy as np
import pytest
from PIL import Image, ImageDraw

from vesselwatch.data import (
    VesselCropDataset,
    annotation_path,
    category_names,
    coco_to_vessel_records,
    crop_eval_transform,
    image_dir,
    rotated_crop,
    split_images,
    upright_crop,
)

# A 100 x 20 box rotated by 30 degrees around (150, 150)
ANGLE = np.deg2rad(30)
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])
TILTED = tuple((np.array([[-50, -10], [50, -10], [50, 10], [-50, 10]]) @ ROTATION.T + 150).flatten())


def _scene():
    """Black scene with one white tilted vessel."""
    img = Image.new("RGB", (300, 300))
    ImageDraw.Draw(img).polygon([tuple(p) for p in np.reshape(TILTED, (4, 2))], fill="white")
    return img


def test_rotated_crop_lays_the_vessel_flat_and_fills_the_crop():
    crop = rotated_crop(_scene(), TILTED)
    assert crop.size == (100, 20)
    assert np.asarray(crop).mean() > 0.9 * 255


def test_rotated_crop_ignores_corner_order():
    shuffled = tuple(np.reshape(TILTED, (4, 2))[[2, 0, 3, 1]].flatten())
    assert rotated_crop(_scene(), shuffled).size == (100, 20)


def test_margin_adds_context_on_every_side():
    assert rotated_crop(_scene(), TILTED, margin=0.1).size == (120, 24)


def test_upright_crop_is_mostly_background_for_a_tilted_vessel():
    crop = upright_crop(_scene(), TILTED)
    assert crop.size[0] > 90 and crop.size[1] > 60
    assert np.asarray(crop).mean() < 0.5 * 255


def test_vessel_records_one_row_per_annotation(data_root):
    records = coco_to_vessel_records(annotation_path(data_root, "train", 1))
    assert len(records) == 13
    assert list(records[records["filename"] == "000002.bmp"]["label"]) == ["Warship", "Warship", "Merchant"]
    assert records.iloc[0]["polygon"] == (10, 10, 10, 30, 60, 30, 60, 10)
    assert category_names(annotation_path(data_root, "val", 1)) == ["Dock", "Merchant", "Other Ship", "Warship"]


def test_split_images_is_disjoint_and_seeded():
    train, val = split_images(range(100), 0.15, seed=0)
    assert len(val) == 15 and len(train) == 85 and not train & val
    assert split_images(range(100), 0.15, seed=0) == (train, val)
    assert split_images(range(100), 0.15, seed=1)[1] != val


def test_crop_dataset_items_and_cache(data_root, tmp_path):
    path = annotation_path(data_root, "train", 1)
    ds = VesselCropDataset(coco_to_vessel_records(path), image_dir(data_root), category_names(path),
                           crop_eval_transform(), margin=0.0, cache_dir=tmp_path / "cache")
    assert ds.load_crop(0).size == (50, 20)
    img, label = ds[1]
    assert img.shape == (3, 224, 224)
    assert ds.classes[label] == "Warship"

    cached = list((tmp_path / "cache").rglob("*.png"))
    assert len(cached) == 2
    assert np.array_equal(np.asarray(ds.load_crop(0)), np.asarray(Image.open(cached[0]).convert("RGB"))) or \
        np.array_equal(np.asarray(ds.load_crop(0)), np.asarray(Image.open(cached[1]).convert("RGB")))


def test_crop_dataset_rejects_labels_outside_the_class_list(data_root):
    records = coco_to_vessel_records(annotation_path(data_root, "train", 1))
    with pytest.raises(ValueError, match="Dock"):
        VesselCropDataset(records, image_dir(data_root), ["Merchant", "Other Ship", "Warship"])
