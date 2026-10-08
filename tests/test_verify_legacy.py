import json
from pathlib import Path

import torch

from vesselwatch.models import build_model
from vesselwatch.verify_legacy import legacy_weights_name, main, verify

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "legacy"


def test_legacy_weights_names():
    assert legacy_weights_name("level1_net") == "best_model_Net_Level1.pth"
    assert legacy_weights_name("level2_net_max") == "best_model_Net_Max_Level2.pth"
    assert legacy_weights_name("level3_resnet101") == "best_model_ResNet_Level3.pth"


def test_verify_scores_present_runs_and_reports_the_rest(data_root, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    torch.save(build_model("net", 4).state_dict(), models / "best_model_Net_Level1.pth")
    # Wrong class count for level 1: must be reported, not crash the whole check
    torch.save(build_model("net_max", 25).state_dict(), models / "best_model_Net_Max_Level1.pth")

    rows = {r["run"]: r for r in verify(models, CONFIG_DIR, data_root=str(data_root), num_workers=0)}

    assert len(rows) == 9
    net = rows["level1_net"]
    assert 0 <= net["val_acc"] <= 100
    assert net["diff"] == round(net["val_acc"] - 70.18, 2)
    assert net["status"] in ("match", "DIFFERS")
    assert rows["level1_net_max"]["status"].startswith("error: RuntimeError")
    assert rows["level3_resnet101"]["status"] == "weights missing: best_model_ResNet_Level3.pth"


def test_shared_classes_rescores_with_the_train_index(data_root, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    torch.save(build_model("net", 4).state_dict(), models / "best_model_Net_Level1.pth")
    kwargs = dict(data_root=str(data_root), num_workers=0)

    legacy = verify(models, CONFIG_DIR, **kwargs)[0]
    shared = verify(models, CONFIG_DIR, shared_classes=True, **kwargs)[0]
    assert shared["val_acc"] is not None and shared["status"] in ("match", "DIFFERS")
    assert legacy["val_acc"] is not None

    main(["--models-dir", str(models), "--configs-dir", str(CONFIG_DIR), "--data-root", str(data_root),
          "--num-workers", "0", "--shared-classes"])
    assert "shared class index" in capsys.readouterr().out


def test_cli_writes_json(data_root, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    out = tmp_path / "verification.json"
    main(["--models-dir", str(models), "--configs-dir", str(CONFIG_DIR),
          "--data-root", str(data_root), "--out", str(out)])
    assert len(json.loads(out.read_text())) == 9
    assert "0 of 9 runs reproduce" in capsys.readouterr().out
