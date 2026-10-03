from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ShipImageDataset(Dataset):
    """Whole-image classification dataset built from a filename/label dataframe.

    If `classes` is not given the class index is built from this dataframe
    alone, as in the original study. Pass the train classes to share one index.
    """

    def __init__(self, df, image_dir, transform=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = list(classes) if classes is not None else sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
