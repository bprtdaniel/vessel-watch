import json
from pathlib import Path

import pytest
import torch

from vesselwatch.checkpoint import load_checkpoint
from vesselwatch.config import Config, load_config
from vesselwatch.evaluate import evaluate
from vesselwatch.models import build_model
from vesselwatch.train import train

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "legacy"


def _cfg(data_root, tmp_path, **kw):
    return Config(level=1, model="net", optimizer="sgd", epochs=2, batch_size=4, num_workers=0,
                  data_root=str(data_root), runs_dir=str(tmp_path / "runs"), **kw)


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

    resnet = load_config(CONFIG_DIR / "level3_resnet101.yaml")
    assert (resnet.pretrained, resnet.optimizer, resnet.lr) == (True, "adam", 0.0001)
    net = load_config(CONFIG_DIR / "level1_net.yaml")
    assert (net.optimizer, net.lr, net.momentum) == ("sgd", 0.001, 0.9)


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("level: 1\nmodel: net\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError):
        load_config(p)
