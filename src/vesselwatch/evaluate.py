"""Score saved weights on one split.

    python -m vesselwatch.evaluate --config configs/crops/level1_resnet101.yaml --weights best.pt --split test

Writes metrics_<split>.json and confusion_<split>.png to the run folder. The
test split is meant to be scored once per experiment, after model selection on
val, so an existing metrics_test.json is not overwritten without --force.

Accepts checkpoints from vesselwatch.train and the bare state_dict files of the
original study.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from .checkpoint import load_checkpoint
from .config import Config, load_config
from .metrics import classification_metrics, save_confusion_matrix
from .models import build_model
from .train import build_loaders, run_epoch


def score(cfg: Config, weights, split: str = "val") -> dict:
    """Loss and metrics of `weights` on `split`.

    With the original study's per-split class index only loss and accuracy are
    returned, because the class names of the two splits do not line up.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg)
    if split not in loaders:
        raise ValueError(f"No '{split}' split with labels: {cfg.labels}")
    ckpt = load_checkpoint(weights, map_location=device)

    train_ds, eval_ds = loaders["train"].dataset, loaders[split].dataset
    classes = train_ds.classes
    if ckpt["classes"] is not None and ckpt["classes"] != classes:
        raise ValueError("Class list in the checkpoint differs from the one built from the train split")

    model = build_model(cfg.model, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    loss, acc, y_true, y_pred = run_epoch(model, loaders[split], nn.CrossEntropyLoss(), device)

    if cfg.per_split_classes:
        return {"split": split, "loss": loss, "n": len(y_true), "accuracy": acc}
    report = classification_metrics(y_true, y_pred, classes, train_labels=train_ds.labels,
                                    areas=getattr(eval_ds, "areas", None),
                                    lengths_m=getattr(eval_ds, "lengths_m", None))
    return {"split": split, "loss": loss, **report}


def evaluate(cfg: Config, weights, split: str = "val") -> tuple[float, float]:
    """(loss, accuracy in percent) of `weights` on `split`."""
    report = score(cfg, weights, split)
    return report["loss"], report["accuracy"]


def write_report(cfg: Config, report: dict) -> None:
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    split = report["split"]
    with open(run_dir / f"metrics_{split}.json", "w") as f:
        json.dump(report, f, indent=2)
    if "confusion_matrix" in report:
        classes = [row["class"] for row in report["per_class"]]
        save_confusion_matrix(report["confusion_matrix"], classes, run_dir / f"confusion_{split}.png",
                              title=f"{cfg.run_name}, {split} split")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--force", action="store_true", help="score the test split again")
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_root=args.data_root, runs_dir=args.runs_dir, num_workers=args.num_workers)

    if args.split == "test" and (cfg.run_dir() / "metrics_test.json").exists() and not args.force:
        parser.error(f"{cfg.run_dir() / 'metrics_test.json'} exists; the test split is scored once per experiment")

    report = score(cfg, args.weights, args.split)
    line = f"{cfg.run_name} | {args.split} | Loss: {report['loss']:.4f} | Acc: {report['accuracy']:.2f}%"
    if "macro_f1" in report:
        line += f" | Macro-F1: {report['macro_f1']:.2f}%"
        write_report(cfg, report)
    print(line)


if __name__ == "__main__":
    main()
