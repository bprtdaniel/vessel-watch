import pytest
import yaml

from vesselwatch.config import Config
from vesselwatch.detect import export_yolo_obb
from vesselwatch.detect.pipeline import Detection, match_detections, polygon_iou, run_pipeline
from vesselwatch.detect.yolo import load_detect_config
from vesselwatch.train import build_datasets, train

SQUARE = (0, 0, 0, 10, 10, 10, 10, 0)


def _shift(polygon, dx):
    return tuple(v + dx if i % 2 == 0 else v for i, v in enumerate(polygon))


def test_export_writes_yolo_obb_layout(data_root, tmp_path):
    dataset_yaml = export_yolo_obb(data_root, tmp_path / "yolo", level=1, val_fraction=0.25, split_seed=0)
    spec = yaml.safe_load(dataset_yaml.read_text())
    assert spec["names"] == {0: "Dock", 1: "Merchant", 2: "Other Ship", 3: "Warship"}

    out = tmp_path / "yolo"
    counts = {split: len(list((out / "images" / split).iterdir())) for split in ("train", "val", "test")}
    assert counts == {"train": 6, "val": 2, "test": 4}
    for split in counts:
        assert {p.stem for p in (out / "images" / split).iterdir()} == {p.stem for p in (out / "labels" / split).iterdir()}

    # Image 101 holds one "Other Ship" with corners (10,10) (10,30) (60,30) (60,10) in a 360 x 300 image
    fields = (out / "labels" / "test" / "000101.txt").read_text().split()
    assert fields[0] == "2"
    assert [float(v) for v in fields[1:]] == pytest.approx([10 / 360, 10 / 300, 10 / 360, 30 / 300,
                                                            60 / 360, 30 / 300, 60 / 360, 10 / 300], abs=1e-6)


def test_export_uses_the_classification_splits(data_root, tmp_path):
    export_yolo_obb(data_root, tmp_path / "yolo", level=1, val_fraction=0.25, split_seed=0)
    crops = build_datasets(Config(level=1, model="net", val_fraction=0.25, split_seed=0, data_root=str(data_root)))
    for split in ("train", "val", "test"):
        exported = {p.name for p in (tmp_path / "yolo" / "images" / split).iterdir()}
        assert exported == set(crops[split].records["filename"])


def test_polygon_iou():
    assert polygon_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    assert polygon_iou(SQUARE, _shift(SQUARE, 5)) == pytest.approx(50 / 150)
    assert polygon_iou(SQUARE, _shift(SQUARE, 20)) == 0.0
    assert polygon_iou(SQUARE, (10, 0, 10, 10, 0, 10, 0, 0)) == pytest.approx(1.0)  # corner order does not matter


def test_matching_is_one_to_one_and_prefers_confident_detections():
    truth = [SQUARE, _shift(SQUARE, 100)]
    detections = [Detection(_shift(SQUARE, 1), 0.6, "ship"),   # overlaps the first vessel
                  Detection(SQUARE, 0.9, "ship"),              # overlaps it better and is more confident
                  Detection(_shift(SQUARE, 50), 0.8, "ship")]  # overlaps nothing
    assert match_detections(truth, detections) == [1, None]


def test_pipeline_with_perfect_detections_equals_the_classifier(data_root, tmp_path):
    cfg = Config(level=1, model="net", epochs=1, batch_size=4, num_workers=0, crop_margin=0.0,
                 data_root=str(data_root), runs_dir=str(tmp_path / "runs"))
    train(cfg)
    truth = build_datasets(cfg)["test"].records

    def perfect(image_paths):
        for path in image_paths:
            yield [Detection(p, 1.0, "ship") for p in truth[truth["filename"] == path.name]["polygon"]]

    report = run_pipeline(perfect, cfg, cfg.run_dir() / "best.pt")
    assert (report["images"], report["vessels"], report["detections"]) == (4, 6, 6)
    assert report["detection_recall"] == 100 and report["detection_precision"] == 100
    assert report["end_to_end_recall"] == report["classification_accuracy_on_found"]

    def blind(image_paths):
        return ([] for _ in image_paths)

    missed = run_pipeline(blind, cfg, cfg.run_dir() / "best.pt")
    assert missed["detection_recall"] == 0 and missed["end_to_end_recall"] == 0


def test_detect_config(tmp_path):
    from pathlib import Path
    cfg = load_detect_config(Path(__file__).parents[1] / "configs" / "detect" / "yolo11s_obb.yaml",
                             runs_dir=str(tmp_path))
    assert (cfg.name, cfg.level, cfg.imgsz) == ("yolo11s_obb", 0, 1024)
    assert cfg.run_dir() == tmp_path / "yolo11s_obb"
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nimage_size: 640\n")
    with pytest.raises(ValueError):
        load_detect_config(bad)
