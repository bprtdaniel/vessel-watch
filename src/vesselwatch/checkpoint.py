"""Saving and loading model weights together with their class list."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, classes, config: dict, epoch: int, val_acc: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": list(classes),
            "config": config,
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load a checkpoint written by `save_checkpoint` or a bare state_dict.

    The original study saved bare state_dicts; those come back with
    `classes` and `config` set to None.
    """
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj
    return {"model_state": obj, "classes": None, "config": None, "epoch": None, "val_acc": None}
