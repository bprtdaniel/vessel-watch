"""Train and score every experiment in a config folder, one after the other.

    python -m vesselwatch.study --configs configs/crops

Each run is trained, its best checkpoint (chosen on val) is scored on val and
once on the held-out test split, and a summary table is written to
study_summary.json in the runs folder. Runs that already have a metrics_test.json are skipped, so the command
can be started again after an interrupted session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluate import score, write_report
from .train import train

SUMMARY_FILE = "study_summary.json"


def summary_row(cfg, val: dict, test: dict) -> dict:
    row = {
        "run": cfg.run_name,
        "level": cfg.level,
        "model": cfg.model,
        "classes": len(test["per_class"]),
        "val_acc": val["accuracy"],
        "test_acc": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "baseline_acc": test["majority_baseline"]["accuracy"],
    }
    if "vessels_only" in test:
        row["test_acc_vessels_only"] = test["vessels_only"]["accuracy"]
    return row


def run_study(configs_dir, data_root=None, runs_dir=None, epochs=None, num_workers=None) -> list[dict]:
    rows = []
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        cfg = load_config(path, data_root=data_root, runs_dir=runs_dir, epochs=epochs, num_workers=num_workers)
        run_dir = cfg.run_dir()

        if (run_dir / "metrics_test.json").exists():
            print(f"\n=== {cfg.run_name}: already scored on test, skipping ===", flush=True)
        else:
            print(f"\n=== {cfg.run_name} ===", flush=True)
            train(cfg)
            for split in ("val", "test"):
                write_report(cfg, score(cfg, run_dir / "best.pt", split))

        val = json.loads((run_dir / "metrics_val.json").read_text())
        test = json.loads((run_dir / "metrics_test.json").read_text())
        rows.append(summary_row(cfg, val, test))
        # Rewritten after every run, so an interrupted session keeps what it finished
        with open(run_dir.parent / SUMMARY_FILE, "w") as f:
            json.dump(rows, f, indent=2)
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'run':<26} {'classes':>7} {'val acc':>8} {'test acc':>9} {'macro-F1':>9} {'baseline':>9}")
    print("-" * 73)
    for r in rows:
        print(f"{r['run']:<26} {r['classes']:>7} {r['val_acc']:>8.2f} {r['test_acc']:>9.2f} "
              f"{r['test_macro_f1']:>9.2f} {r['baseline_acc']:>9.2f}")
    print("\nAll values in percent. Baseline: always predicting the most frequent train class.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", required=True, help="folder with one YAML per run")
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)

    rows = run_study(args.configs, args.data_root, args.runs_dir, args.epochs, args.num_workers)
    print_summary(rows)
    print(f"Also written to {SUMMARY_FILE} in the runs folder.")


if __name__ == "__main__":
    main()
