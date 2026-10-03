"""Work out what the saved files of the original study are.

    python -m vesselwatch.inventory /content/drive/MyDrive/5.Projects/models --out inventory.json

Weights are identified from their contents (layer names and output size), not
from their file names. Read-only unless --out or --copy-to is given; nothing in
the scanned folder is renamed or deleted.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

WEIGHT_SUFFIXES = {".pth", ".pt"}
CLASSES_TO_LEVEL = {4: 1, 25: 2, 48: 3}
RESNET_BY_LAYER3_BLOCKS = {(2, 512): "resnet18", (6, 512): "resnet34", (6, 2048): "resnet50",
                           (23, 2048): "resnet101", (36, 2048): "resnet152"}

# Best validation accuracy per run as reported in the paper
PAPER_BEST_VAL_ACC = {
    "level1_net": 70.18, "level1_net_max": 68.73, "level1_resnet101": 85.64,
    "level2_net": 19.27, "level2_net_max": 17.82, "level2_resnet101": 50.55,
    "level3_net": 9.64, "level3_net_max": 10.36, "level3_resnet101": 23.09,
}
HISTORY_KEYS = {"train_losses", "val_losses", "train_accuracies", "val_accuracies"}
NAME_MODELS = (("net_max", "net_max"), ("resnet", "resnet101"), ("net", "net"))


def identify_state_dict(state: dict) -> tuple[str | None, int | None]:
    """(model name, number of classes) from the layer names of a state_dict."""
    if "fc3.weight" in state and "conv4c.weight" in state:
        return "net_max", state["fc3.weight"].shape[0]
    if "fc1.weight" in state and "conv3.weight" in state and "fc2.weight" not in state:
        return "net", state["fc1.weight"].shape[0]
    if "fc.weight" in state and "layer3.0.conv1.weight" in state:
        blocks = len({k.split(".")[1] for k in state if k.startswith("layer3.")})
        out, in_features = state["fc.weight"].shape
        return RESNET_BY_LAYER3_BLOCKS.get((blocks, in_features), "resnet?"), out
    return None, None


def from_filename(name: str) -> dict:
    """What the file name claims: level, model, best/final. Any may be missing."""
    lower = name.lower()
    level = re.search(r"level[_ ]?(\d)", lower)
    model = next((m for token, m in NAME_MODELS if token in lower), None)
    kind = "best" if "best" in lower else "final" if "final" in lower else None
    return {"level": int(level.group(1)) if level else None, "model": model, "kind": kind}


def inspect_weights(path: Path) -> dict:
    row = {"file": path.name, "type": "weights", "size_mb": round(path.stat().st_size / 1e6, 1)}
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # e.g. an Ultralytics checkpoint, which pickles a full model object
        row.update(model=None, note=f"not a plain state_dict ({type(e).__name__})")
        return row
    state = obj.get("model_state", obj) if isinstance(obj, dict) else {}
    model, num_classes = identify_state_dict(state)
    level = CLASSES_TO_LEVEL.get(num_classes)
    claimed = from_filename(path.name)
    row.update(model=model, num_classes=num_classes, level=level, kind=claimed["kind"])

    notes = []
    if model is None:
        notes.append("unrecognised layers")
    if model and level is None:
        notes.append(f"{num_classes} classes matches no level")
    if claimed["level"] and level and claimed["level"] != level:
        notes.append(f"name says level {claimed['level']}")
    if claimed["model"] and model and claimed["model"] != model:
        notes.append(f"name says {claimed['model']}")
    row["note"] = "; ".join(notes)
    return row


def inspect_history(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not HISTORY_KEYS <= set(data):
        return None
    claimed = from_filename(path.name)
    best = data.get("best_val_acc")
    if best is None and data["val_accuracies"]:
        best = max(data["val_accuracies"])
    paper = [run for run, acc in PAPER_BEST_VAL_ACC.items() if best is not None and abs(acc - best) < 0.01]
    return {
        "file": path.name, "type": "history", "model": claimed["model"], "level": claimed["level"],
        "epochs": len(data["val_accuracies"]),
        "best_val_acc": round(best, 2) if best is not None else None,
        "note": f"matches paper: {paper[0]}" if paper else "no match with paper numbers",
    }


def scan(models_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(models_dir).rglob("*")):
        if not path.is_file():
            continue
        row = None
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            row = inspect_weights(path)
        elif path.suffix.lower() == ".json":
            row = inspect_history(path)
        if row is None:
            row = {"file": path.name, "type": "other", "note": ""}
        row["path"] = str(path)
        rows.append(row)
    return rows


def run_name(row: dict) -> str | None:
    if row.get("model") and row.get("level"):
        return f"level{row['level']}_{row['model']}"
    return None


def copy_identified(rows: list[dict], dest: str | Path) -> int:
    """Copy identified weights and histories into dest/<run>/, keeping file names."""
    dest = Path(dest)
    copied = 0
    for row in rows:
        run = run_name(row)
        if row["type"] == "other" or run is None:
            continue
        target = dest / run / row["file"]
        if Path(row["path"]).resolve() == target.resolve() or dest.resolve() in Path(row["path"]).resolve().parents:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(row["path"], target)
            copied += 1
    return copied


def print_report(rows: list[dict]) -> None:
    print(f"{'file':<45} {'type':<8} {'run':<18} {'kind':<6} {'detail':<22} note")
    print("-" * 125)
    for r in sorted(rows, key=lambda r: (r["type"] == "other", run_name(r) or "~", r["type"], r["file"])):
        if r["type"] == "weights":
            classes = f"{r['num_classes']} classes, " if r.get("num_classes") else ""
            detail = f"{classes}{r['size_mb']} MB"
        elif r["type"] == "history":
            detail = f"{r['epochs']} ep, best {r['best_val_acc']}"
        else:
            detail = ""
        print(f"{r['file'][:44]:<45} {r['type']:<8} {run_name(r) or '-':<18} "
              f"{r.get('kind') or '-':<6} {detail:<22} {r.get('note', '')}")

    print("\nCoverage of the nine runs of the original study:")
    for run in PAPER_BEST_VAL_ACC:
        weights = [r for r in rows if r["type"] == "weights" and run_name(r) == run]
        best = sum(r.get("kind") == "best" for r in weights)
        history = any(r["type"] == "history" and run_name(r) == run for r in rows)
        print(f"  {run:<18} weights: {len(weights)} ({best} best)   history: {'yes' if history else 'MISSING'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models_dir")
    parser.add_argument("--out", help="write the inventory as JSON to this file")
    parser.add_argument("--copy-to", help="copy identified files into this folder, one subfolder per run")
    args = parser.parse_args(argv)

    rows = scan(args.models_dir)
    print_report(rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nInventory written to {args.out}")
    if args.copy_to:
        print(f"\nCopied {copy_identified(rows, args.copy_to)} files to {args.copy_to}")


if __name__ == "__main__":
    main()
