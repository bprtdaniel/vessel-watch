"""Does a classifier recognise vessels, or only where an image comes from?

    python -m vesselwatch.by_source --config configs/crops_10m/level1_resnet101.yaml --weights <best.pt>

ShipRSImageNet is assembled from several sources that differ in what they
show: one is mostly small civilian boats, others mostly warships at naval
bases. If a crop gives its source away, by its size or its look, a classifier
can score well without looking at the vessel. Two numbers test for that:

- the accuracy of a rule that knows only the source and always answers with
  that source's most frequent class in the train split;
- the classifier's accuracy within each source, next to that rule's.

A classifier that beats the rule clearly within each source is telling vessels
apart. One that matches it has learned the source.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import load_checkpoint
from .config import load_config
from .data.resolution import image_sources
from .models import build_model
from .train import build_loaders, run_epoch


def source_rule(train_sources, train_labels, test_sources) -> list:
    """For each test sample the most frequent train label of its source; the overall most frequent for unseen sources."""
    overall = Counter(train_labels).most_common(1)[0][0]
    by_source: dict = {}
    for source, label in zip(train_sources, train_labels):
        by_source.setdefault(source, Counter())[label] += 1
    return [by_source[source].most_common(1)[0][0] if source in by_source else overall for source in test_sources]


def breakdown(test_sources, y_true, y_pred, rule_pred, classes) -> dict:
    """Accuracy of the classifier and of the source rule, overall and per source. All in percent."""
    test_sources, y_true = np.asarray(test_sources), np.asarray(y_true)
    y_pred, rule_pred = np.asarray(y_pred), np.asarray(rule_pred)

    def row(mask):
        return {"n": int(mask.sum()),
                "classes_present": int(len(set(y_true[mask].tolist()))),
                "source_rule_answer": classes[Counter(rule_pred[mask].tolist()).most_common(1)[0][0]],
                "source_rule_accuracy": 100 * float((rule_pred[mask] == y_true[mask]).mean()),
                "classifier_accuracy": 100 * float((y_pred[mask] == y_true[mask]).mean())}

    sources = sorted(set(test_sources.tolist()), key=lambda s: -(test_sources == s).sum())
    return {"overall": row(np.ones(len(y_true), dtype=bool)),
            "by_source": {source: row(test_sources == source) for source in sources}}


def by_source(cfg, weights) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg)
    train_ds, test_ds = loaders["train"].dataset, loaders["test"].dataset
    sources = image_sources(cfg.resolved_data_root())

    def sources_of(dataset):
        return [sources.get(Path(name).stem, {}).get("source") or "-" for name in dataset.records["filename"]]

    ckpt = load_checkpoint(weights, map_location=device)
    model = build_model(cfg.model, len(train_ds.classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    _, _, y_true, y_pred = run_epoch(model, loaders["test"], nn.CrossEntropyLoss(), device)

    test_sources = sources_of(test_ds)
    rule = source_rule(sources_of(train_ds), train_ds.labels, test_sources)
    return {"run": cfg.run_name, **breakdown(test_sources, y_true, y_pred, rule, train_ds.classes)}


def print_report(report: dict) -> None:
    print(f"\n{report['run']}")
    print(f"{'source':<14} {'vessels':>8} {'classes':>8} {'source rule says':<22} {'source rule':>12} {'classifier':>11}")
    print("-" * 80)
    for name, r in [("all", report["overall"]), *report["by_source"].items()]:
        answer = "(per source)" if name == "all" else r["source_rule_answer"]
        print(f"{name[:14]:<14} {r['n']:>8} {r['classes_present']:>8} {answer[:22]:<22} "
              f"{r['source_rule_accuracy']:>11.1f}% {r['classifier_accuracy']:>10.1f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--runs-dir")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_root=args.data_root, runs_dir=args.runs_dir, num_workers=args.num_workers)
    print_report(by_source(cfg, args.weights))


if __name__ == "__main__":
    main()
