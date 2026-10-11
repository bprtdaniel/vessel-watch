import pytest

from vesselwatch.by_source import breakdown, by_source, main, source_rule
from vesselwatch.config import Config
from vesselwatch.data.resolution import annotation_dir, assigned_resolutions
from vesselwatch.train import train

CLASSES = ["Merchant", "Warship"]


def test_source_rule_answers_with_each_sources_most_frequent_class():
    train_sources = ["xView", "xView", "xView", "HRSC", "HRSC"]
    train_labels = [0, 0, 1, 1, 1]
    assert source_rule(train_sources, train_labels, ["HRSC", "xView", "Unseen"]) == [1, 0, 1]


def test_breakdown_separates_knowing_the_source_from_knowing_the_vessel():
    sources = ["xView"] * 4 + ["HRSC"] * 4
    y_true = [0, 0, 0, 1, 1, 1, 1, 0]
    rule = [0, 0, 0, 0, 1, 1, 1, 1]          # right six times out of eight from the source alone
    only_source = breakdown(sources, y_true, rule, rule, CLASSES)
    assert only_source["overall"]["source_rule_accuracy"] == pytest.approx(75)
    assert only_source["overall"]["classifier_accuracy"] == pytest.approx(75)      # no better than the rule

    report = breakdown(sources, y_true, y_true, rule, CLASSES)
    assert report["overall"]["classifier_accuracy"] == pytest.approx(100)
    xview = report["by_source"]["xView"]
    assert (xview["n"], xview["classes_present"], xview["source_rule_answer"]) == (4, 2, "Merchant")
    assert xview["source_rule_accuracy"] == pytest.approx(75) and xview["classifier_accuracy"] == pytest.approx(100)


def test_by_source_on_a_trained_run(data_root, tmp_path, capsys):
    for xml in annotation_dir(data_root).glob("*.xml"):
        source = "xView" if int(xml.stem) % 2 else "HRSC"
        resolution = 0.5 if xml.stem.startswith("0000") else 2
        xml.write_text(f"<annotation><source><dataset_source>{source}</dataset_source></source>"
                       f"<Img_Resolution>{resolution}</Img_Resolution></annotation>")
    assigned_resolutions.cache_clear()
    cfg = Config(level=1, model="net", epochs=1, batch_size=4, num_workers=0, val_fraction=0.25,
                 resample_to=10, min_length_m=20, crop="fixed", data_root=str(data_root),
                 runs_dir=str(tmp_path / "runs"))
    train(cfg)

    report = by_source(cfg, cfg.run_dir() / "best.pt")
    assert report["overall"]["n"] == 6
    assert set(report["by_source"]) == {"xView", "HRSC"}
    assert sum(r["n"] for r in report["by_source"].values()) == 6
    assert all(0 <= r["classifier_accuracy"] <= 100 for r in report["by_source"].values())

    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("level: 1\nmodel: net\nbatch_size: 4\nnum_workers: 0\nval_fraction: 0.25\n"
                           "resample_to: 10\nmin_length_m: 20\ncrop: fixed\n")
    main(["--config", str(config_file), "--weights", str(cfg.run_dir() / "best.pt"), "--data-root", str(data_root)])
    assert "source rule" in capsys.readouterr().out
    assigned_resolutions.cache_clear()
