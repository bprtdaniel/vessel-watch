"""Check that this package reproduces the original study.

    python -m vesselwatch.verify_legacy --models-dir /content/drive/MyDrive/5.Projects/models

Loads the nine `best_model_*_Level*.pth` files of the original study into the
ported models, scores them on the validation split with the original labelling,
and compares with the accuracy reported in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluate import evaluate
from .inventory import PAPER_BEST_VAL_ACC

LEGACY_MODEL_TOKEN = {"net": "Net", "net_max": "Net_Max", "resnet101": "ResNet"}
TOLERANCE = 0.5  # percentage points


def legacy_weights_name(run: str) -> str:
    level, model = run.split("_", 1)
    return f"best_model_{LEGACY_MODEL_TOKEN[model]}_Level{level.removeprefix('level')}.pth"


def verify(models_dir, configs_dir="configs/legacy", data_root=None, num_workers=None) -> list[dict]:
    rows = []
    for run, paper_acc in PAPER_BEST_VAL_ACC.items():
        row = {"run": run, "paper": paper_acc, "val_acc": None, "diff": None}
        weights = Path(models_dir) / legacy_weights_name(run)
        if not weights.exists():
            row["status"] = f"weights missing: {weights.name}"
        else:
            try:
                cfg = load_config(Path(configs_dir) / f"{run}.yaml", data_root=data_root, num_workers=num_workers)
                _, val_acc = evaluate(cfg, weights)
                row["val_acc"] = round(val_acc, 2)
                row["diff"] = round(val_acc - paper_acc, 2)
                row["status"] = "match" if abs(row["diff"]) <= TOLERANCE else "DIFFERS"
            except Exception as e:  # keep going so one bad run does not hide the others
                row["status"] = f"error: {type(e).__name__}: {e}"[:200]
        rows.append(row)
        print_row(row)
    return rows


def print_row(row: dict) -> None:
    acc = f"{row['val_acc']:.2f}" if row["val_acc"] is not None else "-"
    diff = f"{row['diff']:+.2f}" if row["diff"] is not None else "-"
    print(f"{row['run']:<18} {row['paper']:>7.2f} {acc:>9} {diff:>7}   {row['status']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--configs-dir", default="configs/legacy")
    parser.add_argument("--data-root")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--out", help="write the comparison as JSON to this file")
    args = parser.parse_args(argv)

    print(f"{'run':<18} {'paper':>7} {'measured':>9} {'diff':>7}   status")
    print("-" * 60)
    rows = verify(args.models_dir, args.configs_dir, args.data_root, args.num_workers)

    matched = sum(r["status"] == "match" for r in rows)
    print(f"\n{matched} of {len(rows)} runs reproduce the paper within {TOLERANCE} points.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
