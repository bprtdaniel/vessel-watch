import pytest

from vesselwatch.metrics import classification_metrics, macro_f1, save_confusion_matrix

CLASSES = ["Dock", "Merchant", "Warship"]
#          Dock, Merchant x4, Warship x2
Y_TRUE = [0, 1, 1, 1, 1, 2, 2]
Y_PRED = [0, 1, 1, 1, 2, 2, 1]


def test_report_values():
    report = classification_metrics(Y_TRUE, Y_PRED, CLASSES, train_labels=[1, 1, 1, 2, 0],
                                    areas=[10, 10, 2000, 2000, 2000, 20000, 20000])

    assert report["n"] == 7
    assert report["accuracy"] == pytest.approx(100 * 5 / 7)
    # F1: Dock 1.0, Merchant 0.75, Warship 0.5
    assert report["macro_f1"] == pytest.approx(75.0)
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 3, 1], [0, 1, 1]]

    merchant = report["per_class"][1]
    assert (merchant["class"], merchant["support"]) == ("Merchant", 4)
    assert merchant["precision"] == pytest.approx(75.0) and merchant["recall"] == pytest.approx(75.0)

    baseline = report["majority_baseline"]
    assert baseline["class"] == "Merchant"
    assert baseline["accuracy"] == pytest.approx(100 * 4 / 7)
    assert baseline["macro_f1"] < report["macro_f1"]

    assert report["vessels_only"]["n"] == 6
    assert report["vessels_only"]["accuracy"] == pytest.approx(100 * 4 / 6)

    assert report["by_size"]["small"] == {"n": 2, "accuracy": pytest.approx(100.0)}
    assert report["by_size"]["medium"]["n"] == 3
    assert report["by_size"]["large"] == {"n": 2, "accuracy": pytest.approx(50.0)}


def test_macro_f1_skips_classes_absent_from_the_split():
    # Only two of five classes occur; the three empty ones must not pull the mean to 40
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1], num_classes=5) == pytest.approx(100.0)


def test_confusion_matrix_figure(tmp_path):
    report = classification_metrics(Y_TRUE, Y_PRED, CLASSES)
    path = tmp_path / "confusion.png"
    save_confusion_matrix(report["confusion_matrix"], CLASSES, path, title="test")
    assert path.stat().st_size > 0
