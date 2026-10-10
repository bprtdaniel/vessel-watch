import sqlite3
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from vesselwatch.config import Config
from vesselwatch.detect import finland
from vesselwatch.detect.pipeline import Detection, load_classifier
from vesselwatch.detect.sizes import contradicts
from vesselwatch.monitor import cdse
from vesselwatch.monitor.tiles import chip_origins, cut_chip
from vesselwatch.train import train

INFO = {"ulx": 499980.0, "uly": 6800040.0, "resolution": 10}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _gpkg(path, layer, boxes, envelope=True):
    """A minimal GeoPackage layer: one polygon per (min x, min y, max x, max y) box."""
    con = sqlite3.connect(path)
    con.execute(f'create table "{layer}" (fid integer primary key, geom blob, id text)')
    for min_x, min_y, max_x, max_y in boxes:
        ring = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
        wkb = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", 5) + b"".join(struct.pack("<2d", *p) for p in ring)
        header = b"GP" + bytes([0, 3 if envelope else 1]) + struct.pack("<i", 32634)
        if envelope:
            header += struct.pack("<4d", min_x, max_x, min_y, max_y)
        con.execute(f'insert into "{layer}" (geom, id) values (?, ?)', (header + wkb, "boat"))
    con.commit()
    con.close()


def _corners(box):
    x0, y0, x1, y1 = box
    return (x0, y0, x0, y1, x1, y1, x1, y0)


def test_boxes_are_read_from_a_geopackage_with_or_without_envelope(tmp_path):
    boxes = [(500000.0, 6799000.0, 500060.0, 6799040.0), (500500.0, 6798000.0, 500520.0, 6798030.0)]
    for envelope in (True, False):
        path = tmp_path / f"tile_{envelope}.gpkg"
        _gpkg(path, "20220813", boxes, envelope)
        assert finland.read_boxes(path, "20220813") == boxes


def test_map_coordinates_become_pixels_with_y_flipped():
    # 20 m east and 1,000 m south of the upper-left corner, 60 m wide and 40 m tall
    assert finland.to_pixels((500000.0, 6799000.0, 500060.0, 6799040.0), INFO) == (2.0, 100.0, 8.0, 104.0)


def test_chips_cover_the_scene_and_edge_chips_are_padded():
    assert chip_origins(700, 330, 320) == [(0, 0), (320, 0), (640, 0), (0, 320), (320, 320), (640, 320)]
    img = Image.new("RGB", (700, 330), "white")
    assert np.asarray(cut_chip(img, 0, 0, 320)).min() == 255
    edge = np.asarray(cut_chip(img, 640, 320, 320))
    assert edge.shape == (320, 320, 3)
    assert edge[:10, :60].min() == 255 and edge[10:].max() == 0 and edge[:, 60:].max() == 0


def test_matching_by_iou_and_by_centre():
    truth = np.array([[10.0, 10, 20, 20], [100, 100, 104, 104]])
    found = np.array([[11.0, 11, 21, 21],      # overlaps the first well
                      [102, 102, 110, 110],    # touches the second: IoU too low, centre outside
                      [101, 101, 105, 105],    # centre inside the second, IoU 0.39
                      [300, 300, 310, 310]])   # nothing there
    scores = [0.9, 0.8, 0.7, 0.6]
    assert finland.match(truth, found, scores, 0.5).tolist() == [0, -1]
    assert finland.match(truth, found, scores, None).tolist() == [0, 2]
    assert finland.match(truth, np.zeros((0, 4)), [], None).tolist() == [-1, -1]

    # Two detections on one vessel: only the more confident one counts
    twice = np.array([[11.0, 11, 21, 21], [10, 10, 20, 20]])
    assert finland.match(truth[:1], twice, [0.5, 0.9], 0.5).tolist() == [1]


def test_size_contradiction():
    assert contradicts("Nimitz", 40) is True
    assert contradicts("Nimitz", 300) is False
    assert contradicts("Nimitz", 900) is False     # a wake makes boxes longer, never shorter
    assert contradicts("Other Warship", 40) is None


def _scene():
    """A 700 x 400 scene with three vessels, one of them straddling two chips."""
    truth = np.array([[50.0, 60, 54, 63], [315, 100, 325, 104], [600, 350, 630, 356]])
    return [("scene", Image.new("RGB", (700, 400), (10, 40, 30)), truth)], truth


def test_evaluate_scores_detectors_in_scene_coordinates():
    scenes, truth = _scene()

    def perfect(chips):
        """Reports each vessel from the chip that holds its upper-left corner, in chip coordinates."""
        origins = chip_origins(700, 400, finland.CHIP)
        out = []
        for x, y in origins[:len(chips)]:
            inside = [b for b in truth if x <= b[0] < x + finland.CHIP and y <= b[1] < y + finland.CHIP]
            out.append([Detection(_corners((b[0] - x, b[1] - y, b[2] - x, b[3] - y)), 0.9, "ship") for b in inside])
        return out

    def noisy(chips):
        return [[Detection(_corners((5, 5, 9, 9)), 0.5, "ship")] for _ in chips]

    report = finland.evaluate(scenes, {"perfect": perfect, "noisy": noisy, "blind": lambda chips: [[] for _ in chips]})
    perfect_score, noisy_score, blind_score = (report["detectors"][k] for k in ("perfect", "noisy", "blind"))

    assert (perfect_score["vessels"], perfect_score["detections"]) == (3, 3)
    assert perfect_score["recall_iou50"] == 100 and perfect_score["precision_centre"] == 100
    assert perfect_score["by_length"]["under 50 m"] == {"vessels": 1, "found_centre": 1, "recall_centre": 100.0}
    assert perfect_score["by_length"]["over 100 m"]["vessels"] == 2

    assert noisy_score["detections"] == 6 and noisy_score["recall_centre"] == 0 and noisy_score["precision_centre"] == 0
    assert blind_score["detections"] == 0 and blind_score["recall_centre"] == 0 and blind_score["precision_centre"] is None


def test_evaluate_tallies_the_classifier_and_saves_examples(data_root, tmp_path):
    cfg = Config(level=1, model="net", epochs=1, batch_size=4, num_workers=0,
                 data_root=str(data_root), runs_dir=str(tmp_path / "runs"))
    train(cfg)
    model, classes = load_classifier(cfg, cfg.run_dir() / "best.pt", DEVICE)
    scenes, truth = _scene()

    report = finland.evaluate(scenes, {"blind": lambda chips: [[] for _ in chips]},
                              {"level1": (model.to(DEVICE), classes, cfg)},
                              examples_dir=tmp_path / "examples")

    tallies = report["classifiers"]["level1"]
    assert set(tallies) == {"true_boxes"}                 # the blind detector gave the pipeline nothing to classify
    assert tallies["true_boxes"]["crops"] == 3
    assert sum(c["share"] for c in tallies["true_boxes"]["top_classes"]) == pytest.approx(100)
    assert tallies["true_boxes"]["named_ship_class"] == 0  # level 1 has no named ship classes
    assert len(list((tmp_path / "examples").glob("*.png"))) == 2  # the vessels sit in two chips


def test_catalogue_query_and_newest_product(monkeypatch):
    assert "contains(Name,'_T34VEN_')" in cdse.l1c_filter("34VEN", "20220813")
    assert "ContentDate/Start ge 2022-08-13T00:00:00.000Z" in cdse.l1c_filter("34VEN", "20220813")

    older = {"Id": "a", "Name": "S2A_MSIL1C_20220813T095601_N0400_R122_T34VEN_20220813T115958.SAFE"}
    newer = {"Id": "b", "Name": "S2A_MSIL1C_20220813T095601_N0510_R122_T34VEN_20240717T115958.SAFE"}
    monkeypatch.setattr(cdse, "_get_json", lambda url: {"value": [newer, older]})
    assert cdse.find_l1c_product("34VEN", "20220813") == newer
    assert cdse._node_path(newer, ("GRANULE", "g")) == f"Products(b)/Nodes({newer['Name']})/Nodes(GRANULE)/Nodes(g)"

    monkeypatch.setattr(cdse, "_get_json", lambda url: {"value": []})
    with pytest.raises(LookupError):
        cdse.find_l1c_product("34VEN", "19990101")


def test_upper_left_corner_from_tile_metadata():
    xml = ('<Geoposition resolution="10">\n  <ULX>499980</ULX>\n  <ULY>6800040</ULY>\n  <XDIM>10</XDIM>\n</Geoposition>'
           '<Geoposition resolution="20"><ULX>1</ULX><ULY>2</ULY></Geoposition>')
    assert cdse.upper_left(xml) == (499980.0, 6800040.0)
    with pytest.raises(ValueError):
        cdse.upper_left("<nothing/>")


def test_download_needs_a_login(monkeypatch):
    monkeypatch.delenv(cdse.USER_ENV, raising=False)
    monkeypatch.delenv(cdse.PASSWORD_ENV, raising=False)
    with pytest.raises(ValueError, match="CDSE_USERNAME"):
        cdse.access_token()
