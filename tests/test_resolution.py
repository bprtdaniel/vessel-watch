import pytest

from vesselwatch.data.crops import box_sides
from vesselwatch.data.resolution import annotation_dir, image_resolutions, main, name_pattern, source_report, summary


def test_box_sides_of_a_tilted_box():
    # 50 x 20 box turned by 90 degrees and given in scrambled corner order
    assert box_sides((0, 0, 20, 50, 0, 50, 20, 0)) == pytest.approx((50, 20))


def test_resolution_is_read_per_image(data_root):
    resolutions = image_resolutions(data_root)
    assert len(resolutions) == 13
    assert resolutions["000001"] == 0.5 and resolutions["000101"] == 2


def test_summary_in_metres_and_target_pixels(data_root):
    report = summary(data_root, target=10)
    assert (report["vessels"], report["images"], report["vessels_without_resolution"]) == (19, 12, 0)
    assert report["by_resolution"] == [{"resolution": 0.5, "images": 8, "vessels": 13},
                                       {"resolution": 2, "images": 4, "vessels": 6}]
    # The boxes are 50 x 20 px: 25 m long in the train images, 100 m in the val images
    assert report["vessel_length_m"]["p0"] == pytest.approx(25) and report["vessel_length_m"]["p100"] == pytest.approx(100)
    assert report["vessels_at_least"]["20 m"] == 19 and report["vessels_at_least"]["30 m"] == 6
    # A 360 px wide scene becomes 18 px at 0.5 m and 72 px at 2 m
    assert report["scene_long_side_px_at_target"]["p0"] == pytest.approx(18)
    assert report["scene_long_side_px_at_target"]["p100"] == pytest.approx(72)
    assert sum(row["vessels"] for row in report["by_class"]) == 19


def test_images_without_resolution_are_counted_not_dropped_silently(data_root, capsys):
    (annotation_dir(data_root) / "000002.xml").write_text("<annotation><Img_Resolution></Img_Resolution></annotation>")
    report = summary(data_root)
    assert report["vessels_without_resolution"] == 3   # image 2 holds three vessels
    assert report["images"] == 11

    main(["--data-root", str(data_root)])
    out = capsys.readouterr().out
    assert "3 in images without a recorded resolution" in out and "at least   30 m" in out


def test_measured_resolution_from_known_ship_lengths(data_root, monkeypatch, capsys):
    import vesselwatch.detect.sizes as sizes

    # Pretend a Warship is always 100 m long; its boxes are 50 px, so the images must be 2 m per pixel
    monkeypatch.setattr(sizes, "CLASS_LENGTH_M", {"Warship": 100})
    for xml in annotation_dir(data_root).glob("0000*.xml"):
        xml.write_text("<annotation><source><database>WorldView 3</database><dataset_source>xView</dataset_source>"
                       "</source><Img_Resolution>0.5</Img_Resolution></annotation>")

    rows = source_report(data_root, level=1)
    train = next(r for r in rows if r["source"] == "xView")
    assert (train["sensor"], train["recorded"], train["file_names"]) == ("WorldView 3", "0.5", "9")
    assert (train["images"], train["vessels"], train["named_class_vessels"]) == (8, 13, 3)
    assert train["measured_median"] == pytest.approx(2.0)     # the recorded 0.5 would be wrong
    assert train["median_box_length_px"] == pytest.approx(50)

    val = next(r for r in rows if r["source"] == "-")
    assert val["recorded"] == "2" and val["vessels"] == 6

    main(["--data-root", str(data_root), "--sources", "--level", "1"])
    assert "xView" in capsys.readouterr().out


def test_name_pattern():
    assert name_pattern("1472__1840_0") == "9__9_9"
    assert name_pattern("003313") == "9"
