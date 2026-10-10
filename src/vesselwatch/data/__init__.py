from .coco import (
    annotation_path,
    category_names,
    coco_to_image_labels,
    coco_to_vessel_records,
    image_dir,
    split_images,
)
from .crops import crop_vessel, fixed_crop, rotated_crop, upright_crop
from .datasets import ShipImageDataset, VesselCropDataset
from .transforms import (
    crop_eval_transform,
    crop_train_transform,
    eval_transform,
    fixed_train_transform,
    legacy_train_transform,
    squash_eval_transform,
)


def crop_transforms(crop: str):
    """(train, eval) transforms for a crop mode. Fixed-scale crops must not be padded or stretched unevenly."""
    if crop == "fixed":
        return fixed_train_transform(), squash_eval_transform()
    return crop_train_transform(), crop_eval_transform()

__all__ = [
    "annotation_path",
    "category_names",
    "coco_to_image_labels",
    "coco_to_vessel_records",
    "image_dir",
    "split_images",
    "crop_transforms",
    "crop_vessel",
    "fixed_crop",
    "rotated_crop",
    "upright_crop",
    "ShipImageDataset",
    "VesselCropDataset",
    "crop_eval_transform",
    "crop_train_transform",
    "eval_transform",
    "legacy_train_transform",
]
