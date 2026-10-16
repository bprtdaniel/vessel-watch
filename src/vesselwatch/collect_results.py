"""Gather every result of the project into one file, for the record in docs/pathway.md.

    python -m vesselwatch.collect_results --runs <runs folder> --models <models folder> --out results.json

Reads what the steps left on Drive and changes nothing there: the verification
of the 2025 weights, the summaries of both classifier studies, each run's
config, training curve summary and scores, each detector's training log, and
the pipeline and Sentinel-2 reports. Missing files are listed, not guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

MODEL_FILES = ("legacy_verification.json", "legacy_verification_shared_classes.json")
SUMMARY_FILES = ("study_summary.json", "study_summary_10m.json")
# Files the record expects; any not found are reported as missing
EXPECTED = (
    "models/legacy_verification.json", "models/legacy_verification_shared_classes.json",
    "runs/study_summary.json", "runs/study_summary_10m.json",
    "runs/yolo11s_obb/metrics_test.json", "runs/yolo11s_obb_10m/metrics_test.json",
    "runs/yolo11s_finland/metrics_test.json",
    "runs/sentinel2_step_a.json", "runs/sentinel2_step_d.json", "runs/sentinel2_step_e.json",
    "runs/sentinel2_step_10.json",
    "runs/crops10m_level1_resnet101/by_source.json", "runs/crops_level1_resnet101/by_source.json",
)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def history_summary(history: dict) -> dict:
    """Length of training, where val accuracy peaked, and how far train and val ended apart."""
    val = history.get("val_accuracies") or []
    train = history.get("train_accuracies") or []
    if not val:
        return {}
    best = max(range(len(val)), key=val.__getitem__)
    return {"epochs": len(val), "best_epoch": best + 1, "best_val_acc": val[best],
            "train_acc_at_best": train[best] if best < len(train) else None,
            "final_val_acc": val[-1], "final_train_acc": train[-1] if train else None,
            "val_macro_f1_at_best": (history.get("val_macro_f1") or [None] * len(val))[best]}


def yolo_log_summary(results_csv: Path) -> dict:
    """Epochs run, the last epoch's val scores and the epoch with the best mAP50-95, from ultralytics' log."""
    with open(results_csv, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        return {}
    keys = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
            "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)"}

    def scores(row):
        return {"epoch": int(float(row["epoch"])),
                **{name: 100 * float(row[col]) for name, col in keys.items() if col in row}}

    best = max(rows, key=lambda r: float(r.get(keys["map50_95"], 0) or 0))
    return {"epochs": len(rows), "last": scores(rows[-1]), "best": scores(best)}


def run_folder(run_dir: Path) -> dict | None:
    """Everything a classifier or detector run folder says about itself; None if it holds no results."""
    out = {}
    if (run_dir / "config.yaml").exists():
        out["config"] = yaml.safe_load((run_dir / "config.yaml").read_text())
    for name in ("metrics_val.json", "metrics_test.json", "by_source.json"):
        data = read_json(run_dir / name)
        if data is not None:
            out[name.removesuffix(".json")] = data
    history = read_json(run_dir / "history.json")
    if history:
        out["training"] = history_summary(history)
    if (run_dir / "results.csv").exists():
        out["training"] = yolo_log_summary(run_dir / "results.csv")
    if (run_dir / "args.yaml").exists():
        args = yaml.safe_load((run_dir / "args.yaml").read_text())
        out["train_args"] = {k: args.get(k) for k in ("model", "epochs", "imgsz", "batch", "optimizer", "lr0",
                                                      "seed", "data", "patience", "mosaic", "close_mosaic")}
    return out or None


def collect(runs: Path, models: Path | None) -> dict:
    runs = Path(runs)
    result = {"models": {}, "summaries": {}, "runs": {}, "pipeline": {}, "sentinel2": {}, "missing": []}
    if models is not None:
        for name in MODEL_FILES:
            data = read_json(Path(models) / name)
            if data is not None:
                result["models"][name.removesuffix(".json")] = data
    for name in SUMMARY_FILES:
        data = read_json(runs / name)
        if data is not None:
            result["summaries"][name.removesuffix(".json")] = data
    for path in sorted(runs.glob("pipeline_*.json")):
        result["pipeline"][path.stem.removeprefix("pipeline_")] = read_json(path)
    for path in sorted(runs.glob("sentinel2_*.json")):
        result["sentinel2"][path.stem.removeprefix("sentinel2_")] = read_json(path)
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        data = run_folder(run_dir)
        if data is not None:
            result["runs"][run_dir.name] = data
    for expected in EXPECTED:
        base = Path(models) if expected.startswith("models/") and models is not None else runs
        if not (base / expected.split("/", 1)[1]).exists():
            result["missing"].append(expected)
    return result


def print_overview(result: dict) -> None:
    print(f"2025 verification files: {', '.join(result['models']) or 'none'}")
    print(f"Study summaries: {', '.join(result['summaries']) or 'none'}")
    print(f"Run folders with results: {len(result['runs'])}")
    for name, run in result["runs"].items():
        test = run.get("metrics_test", {})
        score = (f"test acc {test['accuracy']:.1f}" if "accuracy" in test else
                 f"test mAP50 {test['map50']:.1f}" if "map50" in test else "no test score")
        epochs = run.get("training", {}).get("epochs", "-")
        print(f"  {name:<32} epochs {epochs:>4} | {score}")
    print(f"Pipeline reports: {', '.join(result['pipeline']) or 'none'}")
    print(f"Sentinel-2 reports: {', '.join(result['sentinel2']) or 'none'}")
    print("Missing: " + (", ".join(result["missing"]) if result["missing"] else "nothing"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="the runs folder")
    parser.add_argument("--models", help="the folder with the 2025 weights and their verification")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = collect(Path(args.runs), Path(args.models) if args.models else None)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print_overview(result)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
