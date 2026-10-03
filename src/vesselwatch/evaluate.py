"""Score saved weights on the validation split.

    python -m vesselwatch.evaluate --config configs/legacy/level1_resnet101.yaml --weights best.pt

Accepts checkpoints from vesselwatch.train and the bare state_dict files of the
original study.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .checkpoint import load_checkpoint
from .config import Config, load_config
from .models import build_model
from .train import build_loaders, run_epoch


def evaluate(cfg: Config, weights) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = build_loaders(cfg)
    ckpt = load_checkpoint(weights, map_location=device)

    classes = train_dl.dataset.classes
    if ckpt["classes"] is not None and ckpt["classes"] != classes:
        raise ValueError("Class list in the checkpoint differs from the one built from the train split")

    model = build_model(cfg.model, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    return run_epoch(model, val_dl, nn.CrossEntropyLoss(), device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_root=args.data_root, num_workers=args.num_workers)
    val_loss, val_acc = evaluate(cfg, args.weights)
    print(f"{cfg.run_name} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
