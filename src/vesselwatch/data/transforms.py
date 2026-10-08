from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def legacy_train_transform():
    """Augmentation of the original study, unchanged (rotation is +-0.5 degrees)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomInvert(p=0.1),
        transforms.RandomRotation(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PadToSquare:
    """Centre the image on a black square, so resizing keeps the vessel's proportions."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def crop_train_transform():
    """Vessel crops: flips only. Overhead imagery has no preferred orientation."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def crop_eval_transform():
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
