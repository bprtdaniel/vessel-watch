import json
import pickle

import torch

from vesselwatch.inventory import copy_identified, identify_state_dict, main, run_name, scan
from vesselwatch.models import build_model


def test_identify_state_dict():
    assert identify_state_dict(build_model("net", 4).state_dict()) == ("net", 4)
    assert identify_state_dict(build_model("net_max", 25).state_dict()) == ("net_max", 25)
    assert identify_state_dict(build_model("resnet101", 48).state_dict()) == ("resnet101", 48)
    assert identify_state_dict(build_model("resnet50", 4).state_dict()) == ("resnet50", 4)
    assert identify_state_dict({"something.weight": torch.zeros(1)}) == (None, None)


def _messy_folder(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    torch.save(build_model("net", 4).state_dict(), d / "best_model_Net_Level1.pth")
    # Name claims level 1, contents are level 2
    torch.save(build_model("net_max", 25).state_dict(), d / "final_model_Net_Max_Level1aug.pth")
    torch.save(build_model("resnet18", 48).state_dict(), d / "untitled (3).pth")
    with open(d / "yolo11n-obb.pt", "wb") as f:
        pickle.dump({"model": object}, f)  # not loadable as plain tensors
    history = {"train_losses": [1.0, 0.9], "val_losses": [1.0, 0.9],
               "train_accuracies": [60.0, 65.0], "val_accuracies": [66.0, 70.18], "best_val_acc": 70.18}
    (d / "Net_history_Level1.json").write_text(json.dumps(history))
    (d / "notes.json").write_text(json.dumps({"hello": 1}))
    (d / "accuracy_comparison.png").write_bytes(b"\x89PNG")
    return d


def test_scan_identifies_by_content(tmp_path):
    rows = {r["file"]: r for r in scan(_messy_folder(tmp_path))}

    net = rows["best_model_Net_Level1.pth"]
    assert (run_name(net), net["kind"], net["note"]) == ("level1_net", "best", "")

    mislabelled = rows["final_model_Net_Max_Level1aug.pth"]
    assert run_name(mislabelled) == "level2_net_max"
    assert "name says level 1" in mislabelled["note"]

    assert run_name(rows["untitled (3).pth"]) == "level3_resnet18"
    assert rows["yolo11n-obb.pt"]["model"] is None

    history = rows["Net_history_Level1.json"]
    assert (history["type"], history["epochs"], history["note"]) == ("history", 2, "matches paper: level1_net")
    assert rows["notes.json"]["type"] == "other"
    assert rows["accuracy_comparison.png"]["type"] == "other"


def test_copy_and_cli_leave_source_untouched(tmp_path, capsys):
    src = _messy_folder(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    dest = tmp_path / "legacy_v1"

    main([str(src), "--out", str(tmp_path / "inventory.json"), "--copy-to", str(dest)])

    assert sorted(p.name for p in src.iterdir()) == before
    assert (dest / "level1_net" / "best_model_Net_Level1.pth").exists()
    assert (dest / "level1_net" / "Net_history_Level1.json").exists()
    assert (dest / "level2_net_max" / "final_model_Net_Max_Level1aug.pth").exists()
    assert not list(dest.rglob("yolo11n-obb.pt"))
    assert len(json.loads((tmp_path / "inventory.json").read_text())) == 7
    assert "level1_net" in capsys.readouterr().out

    # Running again copies nothing new
    assert copy_identified(scan(src), dest) == 0
