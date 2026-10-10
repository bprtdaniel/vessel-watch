import pytest

from vesselwatch.data.crops import box_sides
from vesselwatch.data.resolution import annotation_dir, image_resolutions, main, summary


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
