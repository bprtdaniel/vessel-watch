import json

from vesselwatch.collect_results import collect, main


def test_collect_reads_classifier_and_detector_runs_and_lists_what_is_missing(tmp_path):
    runs, models = tmp_path / "runs", tmp_path / "models"
    clf = runs / "crops_level1_net"
    det = runs / "yolo11s_obb"
    for folder in (clf, det, models):
        folder.mkdir(parents=True)
    (clf / "config.yaml").write_text("level: 1\nmodel: net\nlr: 0.001\n")
    (clf / "metrics_test.json").write_text(json.dumps({"accuracy": 85.1}))
    (clf / "history.json").write_text(json.dumps({"train_accuracies": [50, 70, 90], "val_accuracies": [40, 80, 75],
                                                  "val_macro_f1": [10, 60, 55]}))
    (det / "results.csv").write_text("  epoch,  metrics/mAP50(B),  metrics/mAP50-95(B)\n"
                                     "1, 0.5, 0.2\n2, 0.8, 0.4\n3, 0.81, 0.35\n")
    (det / "metrics_test.json").write_text(json.dumps({"map50": 87.8}))
    (runs / "study_summary.json").write_text("[]")
    (runs / "sentinel2_step_a.json").write_text(json.dumps({"name": "step_a"}))
    (models / "legacy_verification.json").write_text("[]")

    result = collect(runs, models)

    training = result["runs"]["crops_level1_net"]["training"]
    assert (training["epochs"], training["best_epoch"], training["best_val_acc"]) == (3, 2, 80)
    assert training["train_acc_at_best"] == 70 and training["val_macro_f1_at_best"] == 60
    assert result["runs"]["crops_level1_net"]["config"]["lr"] == 0.001
    log = result["runs"]["yolo11s_obb"]["training"]
    assert log["epochs"] == 3 and log["best"]["epoch"] == 2 and abs(log["last"]["map50"] - 81) < 1e-9
    assert result["sentinel2"]["step_a"]["name"] == "step_a"
    assert "runs/study_summary_10m.json" in result["missing"]
    assert "models/legacy_verification.json" not in result["missing"]

    out = tmp_path / "all.json"
    main(["--runs", str(runs), "--models", str(models), "--out", str(out)])
    assert json.loads(out.read_text())["summaries"]["study_summary"] == []
