import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from vesselwatch.config import Config
from vesselwatch.data import annotation_path, coco_to_vessel_records, fixed_crop
from vesselwatch.data.resample import resample_image, resampled_records
from vesselwatch.data.resolution import annotation_dir, assigned_resolutions
from vesselwatch.detect.canvases import export_canvases, flip_scene, pack
from vesselwatch.detect.yolo import DetectConfig
from vesselwatch.evaluate import score
from vesselwatch.train import build_datasets, train


def _xml(source, recorded):
    return (f"<annotation><source><database>x</database><dataset_source>{source}</dataset_source></source>"
            f"<Img_Resolution>{recorded}</Img_Resolution></annotation>")


def test_assigned_resolution_falls_back_to_the_recorded_value(data_root):
    assigned_resolutions.cache_clear()
    resolutions = assigned_resolutions(str(data_root))
    assert resolutions["000001"] == 0.5 and resolutions["000101"] == 2


def test_assigned_resolution_prefers_measured_over_recorded(data_root, monkeypatch):
    import vesselwatch.data.resolution as resolution
    import vesselwatch.detect.sizes as sizes

    # A Warship is 100 m and its boxes are 50 px long, so images holding one measure 2 m per pixel
    monkeypatch.setattr(sizes, "CLASS_LENGTH_M", {"Warship": 100})
    monkeypatch.setattr(resolution, "MIN_NAMED_IMAGES", 1)
    monkeypatch.setattr(resolution, "measured_by_image",
                        lambda root: {"000002": 2.0, "000006": 2.0, "000007": 9.0})
    for xml in annotation_dir(data_root).glob("0000*.xml"):
        xml.write_text(_xml("HRSC", "1.07"))
    (annotation_dir(data_root) / "000101.xml").write_text(_xml("Airbus ship", ""))
    (annotation_dir(data_root) / "000102.xml").write_text(_xml("Unknown", ""))

    assigned_resolutions.cache_clear()
    resolutions = assigned_resolutions(str(data_root))
    assert resolutions["000002"] == 2.0     # its own measurement
    assert resolutions["000007"] == 4.0     # an outlier is held within a factor of two of the source median
    assert resolutions["000001"] == 2.0     # no named vessel: the source median, not the recorded 1.07
    assert resolutions["000101"] == 1.5     # nothing recorded or measured: the source's nominal value
    assert "000102" not in resolutions      # nothing to go on
    assert resolutions["000103"] == 2       # an untouched source keeps its recorded value
    assigned_resolutions.cache_clear()


def test_records_are_scaled_and_short_vessels_dropped(data_root):
    records = coco_to_vessel_records(annotation_path(data_root, "train", 1))
    resolutions = {f"{i:06d}": 0.5 for i in range(1, 10)}

    at_10m = resampled_records(records, resolutions, target=10)
    assert len(at_10m) == 13
    first = at_10m.iloc[0]
    assert (first["width"], first["height"], first["factor"]) == (18, 15, 0.05)
    assert first["polygon"] == pytest.approx((0.5, 0.5, 0.5, 1.5, 3.0, 1.5, 3.0, 0.5))   # 50 x 20 px becomes 2.5 x 1 px
    assert first["length_m"] == pytest.approx(25) and first["area"] == pytest.approx(2.5)

    assert len(resampled_records(records, resolutions, target=10, min_length_m=30)) == 0
    assert len(resampled_records(records, {"000001": 0.5}, target=10)) == 1    # images without a resolution drop out
    assert list(resampled_records(records, {}, target=10).columns) == list(at_10m.columns)


def test_resampling_averages_pixels():
    img = Image.new("RGB", (40, 20), "black")
    ImageDraw.Draw(img).rectangle((0, 0, 19, 19), fill="white")
    small = resample_image(img, 0.1)
    assert small.size == (4, 2)
    assert np.asarray(small)[0, :, 0].tolist() == [255, 255, 0, 0]


def test_fixed_crop_keeps_the_vessel_at_true_scale():
    # A 30 x 4 px vessel turned by 30 degrees around (100, 100)
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    polygon = tuple((np.array([[-15, -2], [15, -2], [15, 2], [-15, 2]]) @ rotation.T + 100).flatten())
    img = Image.new("L", (200, 200))
    ImageDraw.Draw(img).polygon([tuple(p) for p in np.reshape(polygon, (4, 2))], fill=255)

    crop = np.asarray(fixed_crop(img.convert("RGB"), polygon, window=56))[:, :, 0] > 127
    assert crop.shape == (56, 56)
    rows, cols = np.where(crop)
    assert 100 <= crop.sum() <= 170                    # about 30 x 4 pixels, not stretched to fill the window
    assert rows.min() >= 24 and rows.max() <= 31       # lying horizontally through the centre
    assert cols.min() <= 14 and cols.max() >= 41       # 30 px long, centred


def _cfg_10m(data_root, tmp_path, min_length_m=20):
    return Config(level=1, model="net", epochs=1, batch_size=4, num_workers=0, val_fraction=0.25,
                  resample_to=10, min_length_m=min_length_m, crop="fixed", crop_window=56,
                  data_root=str(data_root), runs_dir=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"))


def test_crop_study_at_10m(data_root, tmp_path):
    assigned_resolutions.cache_clear()
    cfg = _cfg_10m(data_root, tmp_path)
    datasets = build_datasets(cfg)
    assert len(datasets["train"]) + len(datasets["val"]) == 13 and len(datasets["test"]) == 6
    assert set(datasets["test"].lengths_m) == {100.0}          # 50 px at 2 m per pixel
    assert datasets["train"][0][0].shape == (3, 224, 224)
    assert datasets["train"].load_crop(0).size == (56, 56)

    # The split does not depend on the resolution or on which vessels are long enough
    original = build_datasets(Config(level=1, model="net", val_fraction=0.25, data_root=str(data_root)))
    assert set(datasets["val"].records["filename"]) == set(original["val"].records["filename"])

    # Nothing in the train images reaches 30 m, so nothing is left to train on
    assert len(build_datasets(_cfg_10m(data_root, tmp_path, min_length_m=30))["train"]) == 0

    train(cfg)
    report = score(cfg, cfg.run_dir() / "best.pt", "test")
    assert report["by_length"]["50 to 150 m"]["n"] == 6 and report["by_length"]["under 50 m"]["n"] == 0
    assert any((tmp_path / "cache" / "at_10m" / "fixed_w56").glob("*.png"))


def test_scenes_are_packed_row_by_row_at_true_scale():
    def scene(w, h, colour):
        return Image.new("RGB", (w, h), colour), [(0, (1, 1, 1, 3, 5, 3, 5, 1))]

    scenes = [scene(60, 40, "red"), scene(50, 30, "green"), scene(40, 50, "blue"), scene(200, 20, "white"), scene(90, 90, "yellow")]
    canvases = list(pack(scenes, size=100))
    assert len(canvases) == 2                                  # the 200 px wide scene fits nowhere and is skipped
    first, placed = canvases[0]
    assert first.size == (100, 100)
    assert first.getpixel((10, 10)) == (255, 0, 0) and first.getpixel((10, 50)) == (0, 128, 0)
    assert first.getpixel((60, 50)) == (0, 0, 255)             # the third scene sits beside the second, in row two
    assert first.getpixel((80, 10)) == (20, 40, 45)            # background where nothing fits
    assert [polygon[:2] for _, polygon in placed] == [(1, 1), (1, 41), (51, 41)]
    assert canvases[1][0].getpixel((45, 45)) == (255, 255, 0)


def test_flipping_a_scene_moves_its_vessels():
    img = Image.new("RGB", (20, 10))
    _, vessels = flip_scene(img, [(0, (1, 2, 1, 4, 6, 4, 6, 2))], horizontal=True, vertical=False)
    assert vessels == [(0, (19, 2, 19, 4, 14, 4, 14, 2))]
    _, vessels = flip_scene(img, [(0, (1, 2, 1, 4, 6, 4, 6, 2))], horizontal=False, vertical=True)
    assert vessels == [(0, (1, 8, 1, 6, 6, 6, 6, 8))]


def test_canvas_dataset_export(data_root, tmp_path):
    assigned_resolutions.cache_clear()
    out = tmp_path / "canvases"
    dataset_yaml = export_canvases(data_root, out, target=10, level=1, val_fraction=0.25, min_length_m=20,
                                   size=96, passes=3)
    spec = yaml.safe_load(dataset_yaml.read_text())
    assert spec["names"] == {0: "Dock", 1: "Merchant", 2: "Other Ship", 3: "Warship"}

    for split in ("train", "val", "test"):
        images = sorted((out / "images" / split).iterdir())
        labels = sorted((out / "labels" / split).iterdir())
        assert images and [p.stem for p in images] == [p.stem for p in labels]
        assert Image.open(images[0]).size == (96, 96)

    # Train scenes are 18 x 15 px at 10 m, 6 of them, 3 passes: every vessel of the split appears three times
    datasets = build_datasets(Config(level=1, model="net", val_fraction=0.25, data_root=str(data_root)))
    train_lines = [line for p in (out / "labels" / "train").iterdir() for line in p.read_text().splitlines()]
    assert len(train_lines) == 3 * len(datasets["train"])
    assert all(0 <= float(v) <= 1 for line in train_lines for v in line.split()[1:])
    # Test: one pass over four 72 x 60 px scenes, one per canvas, six vessels in all
    test_lines = [line for p in (out / "labels" / "test").iterdir() for line in p.read_text().splitlines()]
    assert len(list((out / "images" / "test").iterdir())) == 4 and len(test_lines) == 6

    # A second call reuses what is there
    assert export_canvases(data_root, out, size=96) == dataset_yaml


def test_detector_config_selects_the_canvas_export(tmp_path):
    plain = DetectConfig(name="a", runs_dir=str(tmp_path))
    coarse = DetectConfig(name="b", runs_dir=str(tmp_path), resample_to=10)
    assert plain.dataset_dir().name == "yolo_obb_level0_split0"
    assert coarse.dataset_dir().name == "yolo_obb_level0_split0_at_10m_canvas320_passes20"
