import torch

from vesselwatch.data import (
    ShipImageDataset,
    annotation_path,
    coco_to_image_labels,
    eval_transform,
    image_dir,
    legacy_train_transform,
)


def test_majority_label_per_image(data_root):
    df = coco_to_image_labels(annotation_path(data_root, "train", 1))
    labels = dict(zip(df["filename"], df["label"]))
    assert len(df) == 8                       # image without annotations is dropped
    assert labels["000002.bmp"] == "Warship"  # [2, 2, 3]
    assert labels["000007.bmp"] == "Merchant" # [3, 3, 1]


def test_dataset_item_shape_and_label(data_root):
    df = coco_to_image_labels(annotation_path(data_root, "train", 1))
    for transform in (legacy_train_transform(), eval_transform()):
        ds = ShipImageDataset(df, image_dir(data_root), transform)
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert label.dtype == torch.long
    assert ds.classes == ["Dock", "Merchant", "Other Ship", "Warship"]


def test_per_split_classes_shift_indices(data_root):
    """Documents the legacy defect: a class missing from val shifts every later index."""
    train_df = coco_to_image_labels(annotation_path(data_root, "train", 1))
    val_df = coco_to_image_labels(annotation_path(data_root, "val", 1))
    train_ds = ShipImageDataset(train_df, image_dir(data_root))

    legacy_val = ShipImageDataset(val_df, image_dir(data_root))
    assert legacy_val.class_to_idx["Warship"] != train_ds.class_to_idx["Warship"]

    shared_val = ShipImageDataset(val_df, image_dir(data_root), classes=train_ds.classes)
    assert shared_val.class_to_idx == train_ds.class_to_idx
