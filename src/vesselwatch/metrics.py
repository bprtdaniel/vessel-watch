"""Classification metrics beyond accuracy.

Accuracy, precision, recall and F1 are all reported in percent. Macro-F1 is the
unweighted mean of the per-class F1 over the classes that occur in the split.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NON_VESSEL = ("Dock",)  # labelled in the dataset but not a vessel
# COCO object size buckets, by box area in pixels
SIZE_BUCKETS = (("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf")))


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 100 * float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    _, _, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0)
    return 100 * float(f1[support > 0].mean()) if (support > 0).any() else float("nan")


def classification_metrics(y_true, y_pred, classes, train_labels=None, areas=None) -> dict:
    """Full report for one split.

    `train_labels` adds the baseline of always predicting the most frequent
    train class; `areas` (box area per sample, in pixels) adds accuracy by size.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n), zero_division=0)

    report = {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n),
    }

    if train_labels is not None:
        majority = int(np.bincount(train_labels, minlength=n).argmax())
        guess = np.full_like(y_true, majority)
        report["majority_baseline"] = {
            "class": classes[majority],
            "accuracy": accuracy(y_true, guess),
            "macro_f1": macro_f1(y_true, guess, n),
        }

    vessels = np.ones(len(y_true), dtype=bool)
    for name in NON_VESSEL:
        if name in classes:
            vessels &= y_true != classes.index(name)
    if not vessels.all():
        report["vessels_only"] = {
            "n": int(vessels.sum()),
            "accuracy": accuracy(y_true[vessels], y_pred[vessels]),
            "macro_f1": macro_f1(y_true[vessels], y_pred[vessels], n),
        }

    if areas is not None:
        areas = np.asarray(areas)
        report["by_size"] = {}
        for name, low, high in SIZE_BUCKETS:
            mask = (areas >= low) & (areas < high)
            report["by_size"][name] = {"n": int(mask.sum()), "accuracy": accuracy(y_true[mask], y_pred[mask])}

    report["per_class"] = [
        {"class": classes[i], "support": int(support[i]), "precision": 100 * float(precision[i]),
         "recall": 100 * float(recall[i]), "f1": 100 * float(f1[i])}
        for i in range(n)
    ]
    # Rows are the true class, columns the predicted class
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(n)).tolist()
    return report


def save_confusion_matrix(matrix, classes, path, title="") -> None:
    """Heatmap of the confusion matrix, each row scaled to the share of its true class."""
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    n = len(classes)
    side = max(5, 0.32 * n + 2.5)
    fig = Figure(figsize=(side + 1, side), dpi=150)
    ax = fig.subplots()
    image = ax.imshow(100 * shares, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n), classes, rotation=90)
    ax.set_yticks(range(n), classes)
    ax.tick_params(length=0, labelsize=9 if n <= 10 else 7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    if title:
        ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if n <= 10:
        for i in range(n):
            for j in range(n):
                if totals[i, 0] > 0:
                    ax.text(j, i, f"{100 * shares[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if shares[i, j] > 0.5 else "#1a1a1a")

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Share of the true class (%)")
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
