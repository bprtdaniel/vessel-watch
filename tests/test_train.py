import json
from pathlib import Path

import pytest
import torch

from vesselwatch.checkpoint import load_checkpoint
from vesselwatch.config import Config, load_config
from vesselwatch.evaluate import evaluate, score, write_report
from vesselwatch.evaluate import main as evaluate_main
from vesselwatch.models import build_model
from vesselwatch.study import run_study
from vesselwatch.train import build_datasets, train
from vesselwatch.train import main as train_main

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "legacy"
CROP_CONFIG_DIR = Path(__file__).parents[1] / "configs" / "crops"


def _cfg(data_root, tmp_path, **kw):
    """The original study's setup."""
    return Config(level=1, model="net", optimizer="sgd", epochs=2, batch_size=4, num_workers=0,
                  labels="image_majority", per_split_classes=True,
                  data_root=str(data_root), runs_dir=str(tmp_path / "runs"), **kw)


def _crop_cfg(data_root, tmp_path, **kw):
    return Config(level=1, model="net", epochs=2, batch_size=4, num_workers=0, val_fraction=0.25,
                  data_root=str(data_root), runs_dir=str(tmp_path / "runs"),
                  cache_dir=str(tmp_path / "cache"), **kw)


def test_crop_splits_share_classes_and_no_image(data_root, tmp_path):
    datasets = build_datasets(_crop_cfg(data_root, tmp_path))
    assert set(datasets) == {"train", "val", "test"}
    assert len(datasets["train"]) + len(datasets["val"]) == 13  # every annotated train vessel, once
    assert len(datasets["test"]) == 6                           # the official val split

    files = {split: set(ds.records["filename"]) for split, ds in datasets.items()}
    assert not files["train"] & files["val"] and not files["train"] & files["test"]
    for ds in datasets.values():
        assert ds.classes == ["Dock", "Merchant", "Other Ship", "Warship"]

    again = build_datasets(_crop_cfg(data_root, tmp_path))
    assert files["val"] == set(again["val"].records["filename"])
    other_seed = build_datasets(_crop_cfg(data_root, tmp_path, split_seed=1))
    assert files["val"] != set(other_seed["val"].records["filename"])


def test_crop_run_reports_test_metrics(data_root, tmp_path):
    cfg = _crop_cfg(data_root, tmp_path)
    history = train(cfg)
    assert len(history["val_macro_f1"]) == 2

    report = score(cfg, cfg.run_dir() / "best.pt", "test")
    assert report["n"] == 6
    assert report["majority_baseline"]["class"] in ("Other Ship", "Warship", "Merchant")
    assert report["by_size"]["small"]["n"] == 6
    assert len(report["confusion_matrix"]) == 4
    assert "vessels_only" not in report  # the test split has no Dock

    write_report(cfg, report)
    assert json.loads((cfg.run_dir() / "metrics_test.json").read_text())["accuracy"] == report["accuracy"]
    assert (cfg.run_dir() / "confusion_test.png").stat().st_size > 0
    assert any((tmp_path / "cache").rglob("*.png"))


def test_test_split_is_not_scored_twice_by_accident(data_root, tmp_path):
    cfg_file = tmp_path / "crops.yaml"
    cfg_file.write_text("name: crops\nlevel: 1\nmodel: net\nepochs: 1\nbatch_size: 4\nnum_workers: 0\n")
    args = ["--config", str(cfg_file), "--data-root", str(data_root), "--runs-dir", str(tmp_path / "runs")]
    train_main(args)
    weights = str(tmp_path / "runs" / "crops" / "best.pt")

    evaluate_main(args + ["--weights", weights, "--split", "test"])
    with pytest.raises(SystemExit):
        evaluate_main(args + ["--weights", weights, "--split", "test"])
    evaluate_main(args + ["--weights", weights, "--split", "test", "--force"])


def test_study_runs_every_config_and_skips_finished_runs(data_root, tmp_path, capsys):
    configs = tmp_path / "configs"
    configs.mkdir()
    for model in ("net", "net_max"):
        (configs / f"{model}.yaml").write_text(
            f"name: crops_{model}\nlevel: 1\nmodel: {model}\nepochs: 1\nbatch_size: 4\nnum_workers: 0\n")
    kwargs = dict(data_root=str(data_root), runs_dir=str(tmp_path / "runs"))

    rows = run_study(configs, **kwargs)
    assert [r["run"] for r in rows] == ["crops_net", "crops_net_max"]
    assert all(0 <= r["test_acc"] <= 100 and r["classes"] == 4 for r in rows)
    assert json.loads((tmp_path / "runs" / "study_summary.json").read_text()) == rows

    capsys.readouterr()
    assert run_study(configs, **kwargs) == rows
    assert capsys.readouterr().out.count("skipping") == 2


def test_train_writes_run_and_evaluate_matches(data_root, tmp_path):
    cfg = _cfg(data_root, tmp_path)
    history = train(cfg)

    run_dir = cfg.run_dir()
    for name in ("best.pt", "final.pt", "history.json", "config.yaml", "classes.json"):
        assert (run_dir / name).exists()
    assert len(history["val_accuracies"]) == 2
    assert json.loads((run_dir / "history.json").read_text())["best_val_acc"] == history["best_val_acc"]

    ckpt = load_checkpoint(run_dir / "final.pt")
    assert ckpt["classes"] == ["Dock", "Merchant", "Other Ship", "Warship"]

    _, val_acc = evaluate(cfg, run_dir / "final.pt")
    assert val_acc == pytest.approx(history["val_accuracies"][-1])


def test_evaluate_accepts_bare_state_dict(data_root, tmp_path):
    """The original study saved `model.state_dict()` directly."""
    path = tmp_path / "best_model_Net_Level1.pth"
    torch.save(build_model("net", 4).state_dict(), path)
    val_loss, val_acc = evaluate(_cfg(data_root, tmp_path), path)
    assert 0 <= val_acc <= 100


def test_legacy_configs_load():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(files) == 9
    for f in files:
        cfg = load_config(f)
        assert cfg.run_name == f.stem
        assert cfg.epochs == 30 and cfg.batch_size == 32
        assert (cfg.labels, cfg.per_split_classes) == ("image_majority", True)

    resnet = load_config(CONFIG_DIR / "level3_resnet101.yaml")
    assert (resnet.pretrained, resnet.optimizer, resnet.lr) == (True, "adam", 0.0001)
    net = load_config(CONFIG_DIR / "level1_net.yaml")
    assert (net.optimizer, net.lr, net.momentum) == ("sgd", 0.001, 0.9)


def test_crop_configs_differ_only_in_model_and_learning_rate():
    files = sorted(CROP_CONFIG_DIR.glob("*.yaml"))
    assert len(files) == 9
    shared = set()
    for f in files:
        cfg = load_config(f)
        assert cfg.run_name == f"crops_{f.stem}"
        settings = cfg.to_dict()
        for key in ("name", "level", "model", "pretrained", "lr"):
            settings.pop(key)
        shared.add(json.dumps(settings, sort_keys=True))
    assert len(shared) == 1
    assert load_config(files[0]).labels == "vessel_crops"


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("level: 1\nmodel: net\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError):
        load_config(p)
