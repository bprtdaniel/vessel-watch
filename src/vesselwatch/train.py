"""Train one model at one label level.

    python -m vesselwatch.train --config configs/legacy/level1_net.yaml
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import Config, load_config
from .data import (
    ShipImageDataset,
    VesselCropDataset,
    annotation_path,
    category_names,
    coco_to_image_labels,
    coco_to_vessel_records,
    crop_eval_transform,
    crop_train_transform,
    eval_transform,
    image_dir,
    legacy_train_transform,
    split_images,
)
from .metrics import macro_f1
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _image_datasets(cfg: Config) -> dict:
    """The original study: whole images, majority label, official train and val splits."""
    root = cfg.resolved_data_root()
    img_dir = image_dir(root)
    train_df = coco_to_image_labels(annotation_path(root, "train", cfg.level))
    val_df = coco_to_image_labels(annotation_path(root, "val", cfg.level))

    train_ds = ShipImageDataset(train_df, img_dir, legacy_train_transform())
    val_classes = None if cfg.per_split_classes else train_ds.classes
    val_ds = ShipImageDataset(val_df, img_dir, eval_transform(), classes=val_classes)
    return {"train": train_ds, "val": val_ds}


def _crop_datasets(cfg: Config) -> dict:
    """Per-vessel crops. Train and val share the official train images; test is the official val split."""
    if cfg.per_split_classes:
        raise ValueError("per_split_classes belongs to the original study and needs labels: image_majority")
    root = cfg.resolved_data_root()
    train_json = annotation_path(root, "train", cfg.level)
    classes = category_names(train_json)

    records = coco_to_vessel_records(train_json)
    train_ids, val_ids = split_images(records["image_id"], cfg.val_fraction, cfg.split_seed)
    splits = {
        "train": (records[records["image_id"].isin(train_ids)], crop_train_transform()),
        "val": (records[records["image_id"].isin(val_ids)], crop_eval_transform()),
        "test": (coco_to_vessel_records(annotation_path(root, "val", cfg.level)), crop_eval_transform()),
    }
    return {
        split: VesselCropDataset(split_records, image_dir(root), classes, transform, crop=cfg.crop,
                                 margin=cfg.crop_margin, cache_dir=cfg.resolved_cache_dir())
        for split, (split_records, transform) in splits.items()
    }


def build_datasets(cfg: Config) -> dict:
    """Datasets by split name. Only vessel_crops has a held-out "test" split."""
    if cfg.labels == "image_majority":
        return _image_datasets(cfg)
    if cfg.labels == "vessel_crops":
        return _crop_datasets(cfg)
    raise ValueError(f"Unsupported labels mode '{cfg.labels}'")


def build_loaders(cfg: Config) -> dict:
    return {
        split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=cfg.num_workers)
        for split, ds in build_datasets(cfg).items()
    }


def build_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.

    Returns (mean batch loss, accuracy in percent, true labels, predicted labels).
    """
    training = optimizer is not None
    model.train(training)
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return running_loss / len(loader), 100 * correct / total, y_true, y_pred


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(cfg)
    train_dl, val_dl = loaders["train"], loaders["val"]
    classes = train_dl.dataset.classes
    model = build_model(cfg.model, len(classes), cfg.pretrained).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    with open(run_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    print(f"Run: {run_dir} | device: {device} | classes: {len(classes)} "
          f"| train: {len(train_dl.dataset)} | val: {len(val_dl.dataset)}")

    # The keys of the original study's history files, plus val_macro_f1
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "val_macro_f1": [],
        "best_val_acc": 0.0,
    }

    for epoch in range(cfg.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_dl, loss_function, device, optimizer)
        val_loss, val_acc, y_true, y_pred = run_epoch(model, val_dl, loss_function, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accuracies"].append(train_acc)
        history["val_accuracies"].append(val_acc)
        history["val_macro_f1"].append(macro_f1(y_true, y_pred, len(classes)))

        print(f"Epoch [{epoch+1:2d}/{cfg.epochs}]"
              f"  || Train Loss: {train_loss:.4f} || Train Acc: {train_acc:6.2f}%"
              f"  ||  Val Loss: {val_loss:.4f} || Val Acc: {val_acc:6.2f}%", end="")

        # The first epoch always counts, so a run that never beats 0% still leaves a best.pt
        if epoch == 0 or val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            save_checkpoint(run_dir / "best.pt", model, classes, cfg.to_dict(), epoch, val_acc)
            print("  Current Best Model")
        else:
            print()

        # Written every epoch so a dropped Colab session keeps its curves
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f)

    save_checkpoint(run_dir / "final.pt", model, classes, cfg.to_dict(), cfg.epochs - 1, val_acc)
    print(f"Training done. Best Val Acc: {history['best_val_acc']:.2f}%")
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_root=args.data_root, runs_dir=args.runs_dir,
                      epochs=args.epochs, num_workers=args.num_workers)
    train(cfg)


if __name__ == "__main__":
    main()
