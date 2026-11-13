"""ImageJ 1.54p ports behind Edit ▸ Selection, Edit ▸ Draw, getProfile(), Z Project and Stack Statistics
(lumen/core/ops/_imagej_roi.py, edit.py, stacks.py, noise.py; the reference numbers are ImageJ's, see tests/crossval)."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core import roi as roimod
from lumen.core.ij_compat import IJ
from lumen.core.image import LumenImage
from lumen.core.ops import _imagej_roi as R
from lumen.core.ops._imagej import rank_filter
from lumen.core.ops._imagej_measure import macro_histogram
from lumen.core.session import Session


def _blobs8():
    rng = np.random.default_rng(0)
    return (rng.random((120, 160)) * 255).astype(np.uint8)


@pytest.fixture
def ij():
    s = Session()
    s.add_image(LumenImage(_blobs8(), name="input.tif"))
    return IJ(s)


# ---------------------------------------------------------------- RoiEnlarger / Make Band
def test_enlarge_shrink_and_band_match_imagej(ij):
    ij.makePolygon(10, 10, 120, 30, 100, 100, 40, 90, 20, 60)
    ij.run("Enlarge...", "enlarge=-5")                      # shrink255: rounded EDM ≥ 6
    assert ij.getSelectionBounds() == (17, 17, 96, 77) and ij.getValue("Area") == 5001
    ij.makePolygon(10, 10, 120, 30, 100, 100, 40, 90, 20, 60)
    ij.run("Enlarge...", "enlarge=5")                       # enlarge255: rounded EDM of the outside ≤ 5
    assert ij.getSelectionBounds() == (5, 5, 120, 100) and ij.roi["type"] == "mask"
    ij.makeOval(40, 30, 50, 40)
    ij.run("Enlarge...", "enlarge=5")                       # ovals stay ovals (enlargeRectOrOval)
    assert ij.roi["type"] == "oval" and ij.getSelectionBounds() == (35, 25, 60, 50)
    ij.makeRectangle(20, 15, 60, 45)
    ij.run("Enlarge...", "enlarge=-30")                     # would vanish: unchanged
    assert ij.getSelectionBounds() == (20, 15, 60, 45)
    ij.makeOval(40, 30, 50, 40)
    ij.run("Make Band...", "band=5")
    assert ij.getSelectionBounds() == (35, 25, 60, 50) and ij.getValue("Area") == 832
    with pytest.raises(ValueError, match="255"):
        R.make_band({"type": "oval", "x": 40, "y": 30, "w": 50, "h": 40}, 300.0, (120, 160))


# ---------------------------------------------------------------- Interpolate / Scale / Rotate / Line to Area
def test_interpolate_scale_rotate_line_to_area_match_imagej(ij):
    ij.makePolygon(20, 20, 80, 25, 50, 50, 90, 90, 30, 80, 45, 55)
    ij.run("Interpolate", "interval=5 smooth")               # polygons are not smoothed (only freehand / traced)
    xs, ys = ij.getSelectionCoordinates()
    assert len(xs) == 56 and ij.getValue("Area") == 2015 and ij.roi["type"] == "polygon"
    ij.makePolygon(20, 20, 80, 25, 50, 50, 90, 90, 30, 80, 45, 55)
    ij.run("Interpolate", "interval=3")
    xs, ys = ij.getSelectionCoordinates()
    assert len(xs) == 95 and ij.getValue("Area") == 2016
    ij.makeOval(40, 30, 50, 40)
    ij.run("Scale... ", "x=1.5 y=1.5 centered")              # RoiScaler: rounded vertices about the mask centroid
    assert ij.roi["type"] == "freehand" and ij.getValue("Area") == 3548
    assert all(float(v).is_integer() for p in ij.roi["points"] for v in p)
    ij.makeRectangle(40, 30, 60, 40)
    ij.run("Rotate...", "angle=30")                          # RoiRotator → sub-pixel PolygonRoi: setIntBounds (ceil)
    assert ij.roi["type"] == "polygon" and ij.getSelectionBounds() == (34, 17, 72, 66)
    ij.makeLine(10, 20, 140, 100, 5)
    ij.run("Line to Area")                                   # RotatedRectRoi: FloatPolygon.getBounds
    assert ij.roi["type"] == "polygon" and ij.roi.get("subtype") == "rotated_rect"
    assert ij.getSelectionBounds() == (8, 17, 133, 85) and ij.getValue("Area") == 760
    ij.makeRotatedRectangle(10, 20, 140, 100, 5)
    assert ij.getSelectionBounds() == (8, 17, 133, 85)
    ij.makePolygon(20, 20, 80, 25, 50, 50, 90, 90, 30, 80, 45, 55)
    ij.run("Fit Ellipse")
    assert ij.getSelectionBounds() == (31, 20, 42, 68) and ij.getValue("Area") == 2011


def test_subpixel_polygon_bounds_follow_setintbounds():
    poly = {"type": "polygon", "points": [[34.02, 17.68], [105.98, 52.32], [71.96, 82.32], [40.1, 47.7]]}
    assert roimod.int_bounds(poly) == (34, 17, 72, 66)                       # ceil(max) − floor(min)
    assert roimod.int_bounds({**poly, "subtype": "rotated_rect"}) == (34, 17, 72, 65)   # (int)(max − floor(min) + 0.5)
    (x, y, w, h), m = roimod.roi_rect_mask(poly, (120, 160))
    assert (x, y, w, h) == (34, 17, 72, 66) and m.shape == (66, 72) and not m[-1].any()   # the extra row is empty
    assert np.array_equal(roimod.to_mask(poly, (120, 160)), roimod.to_mask({**poly, "subtype": "rotated_rect"}, (120, 160)))


def test_interpolated_polygon_adjust_and_smoothing():
    xs, ys = R.interpolated_polygon([0, 10, 10, 0], [0, 0, 10, 10], 3.0, False)
    assert len(xs) == 13 and xs.dtype == np.float32
    xa, ya = R.interpolated_polygon([0, 10, 10, 0], [0, 0, 10, 10], -3.0, False)   # 'adjust': the spacing shrinks a little
    seg = np.hypot(np.diff(np.append(xa, xa[0])), np.diff(np.append(ya, ya[0])))
    assert np.all(seg <= 3.0 + 1e-6) and np.ptp(seg) < 0.1
    sx, sy = R.smooth_float_polygon([0, 3, 3, 0], [0, 0, 3, 3], closed=True)
    assert sx[1] == pytest.approx((0 + 3 + 3) / 3) and sx[0] == pytest.approx((0 + 0 + 2) / 3, abs=1e-6)
    s = Session()
    s.add_image(LumenImage(np.zeros((30, 30), np.uint8)))
    line = s.run_op("selection.interpolate", {"interval": 2.0}, roi={"type": "line", "x1": 2, "y1": 2, "x2": 12, "y2": 2})["roi"]
    assert line["type"] == "polyline" and len(line["points"]) == 6 and line["points"][-1] == [12.0, 2.0]


# ---------------------------------------------------------------- Draw
def test_draw_uses_imagej_drawpixels():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((20, 20), np.uint8)))
    s.run_op("edit.set_colors", {"foreground": 255, "background": 0})
    s.run_op("edit.draw", {}, roi={"type": "rect", "x": 2, "y": 2, "w": 5, "h": 5})
    p = img.data[0, 0, 0]
    assert p[2, 2] == 255 and p[7, 7] == 255 and p[2, 7] == 255 and p[3, 3] == 0 and p[8, 8] == 0   # drawRect: x … x+w, y … y+h
    assert int((p == 255).sum()) == 20
    img.data[:] = 0
    s.run_op("edit.draw", {}, roi={"type": "line", "x1": 5, "y1": 10, "x2": 14, "y2": 10, "width": 3})
    assert img.data[0, 0, 0, 9:12, 6:14].all() and not img.data[0, 0, 0, 8, 6:14].any()   # Line.drawPixels 'fillLine': 3-px dots along the line
    img.data[:] = 0
    s.run_op("edit.draw", {"line_width": 5}, roi={"type": "line", "x1": 5, "y1": 10, "x2": 14, "y2": 10})
    assert int((img.data == 255).sum()) == 10                                           # a 1-px line stays 1 px wide (ImageJ ignores the global width)
    img.data[:] = 0
    s.run_op("edit.draw", {}, roi={"type": "oval", "x": 4, "y": 4, "w": 10, "h": 8})
    xs, ys = R.oval_outline_polygon({"type": "oval", "x": 4, "y": 4, "w": 10, "h": 8})
    assert min(xs) == 4 and max(xs) == 14 and min(ys) == 4 and max(ys) == 12           # the Wand outline's corner vertices
    assert img.data[0, 0, 0, 4:13, 4:15].any() and img.data[0, 0, 0, 8, 14] == 255     # drawn one pixel beyond the mask on the right


# ---------------------------------------------------------------- getProfile()
def test_get_profile_matches_profileplot(ij):
    p = ij.img.data[0, 0, 0]
    ij.makeLine(10, 20, 140, 100)
    prof = ij.getProfile()
    assert len(prof) == 154 and prof[0] == p[20, 10] and prof[-1] == pytest.approx(p[100, 140], abs=1e-6)   # getLine: round(length) + 1 samples at the coordinates
    ij.makeLine(10, 20, 140, 100, 5)
    wide = ij.getProfile()                                                             # Straightener: 5 parallel bilinear lines
    assert len(wide) == 154 and all(0 <= v <= 255 for v in wide)
    ij.makeRectangle(20, 15, 60, 45)
    rect = ij.getProfile()
    assert len(rect) == 60 and rect == pytest.approx(p[15:60, 20:80].mean(axis=0).tolist())
    ij.makeOval(20, 15, 60, 45)
    with pytest.raises(ValueError, match="rectangular"):
        ij.getProfile()
    ij.makeLine(0, 0, 10, 0, 20, 5, 30, 5)                                             # a segmented line: getIrregularProfile
    assert len(ij.getProfile()) > 25
    assert ij.macro("makeLine(10, 20, 140, 100); p = getProfile(); print(p.length);").strip() == "154"


# ---------------------------------------------------------------- Z Project
def test_z_project_titles_rounding_and_types():
    s = Session()
    vol = np.zeros((1, 4, 1, 4, 4), np.uint8)
    vol[0, :, 0, 0, 0] = [1, 2, 2, 2]      # mean 1.75 → 2; median (2+2)/2 = 2
    vol[0, :, 0, 0, 1] = [1, 1, 2, 2]      # mean 1.5 → 2 (half up); median 1.5 → 2
    vol[0, :, 0, 0, 2] = [0, 1, 1, 3]      # mean 1.25 → 1; median 1
    vol[0, :, 0, 0, 3] = [2, 3, 3, 2]      # mean 2.5 → 3; median 2.5 → 3
    img = s.add_image(LumenImage(vol, name="stack.tif"))
    out = s.images[s.run_op("stacks.z_project", {"method": "Average Intensity"}, image_id=img.id)["new_images"][0]]
    assert out.name == "AVG_stack.tif" and out.data.dtype == np.uint8 and out.data[0, 0, 0, 0].tolist() == [2, 2, 1, 3]
    out = s.images[s.run_op("stacks.z_project", {"method": "Median"}, image_id=img.id)["new_images"][0]]
    assert out.name == "MED_stack.tif" and out.data.dtype == np.uint8 and out.data[0, 0, 0, 0].tolist() == [2, 2, 1, 3]
    out = s.images[s.run_op("stacks.z_project", {"method": "Standard Deviation"}, image_id=img.id)["new_images"][0]]
    n, v = 4.0, np.array([1, 2, 2, 2], np.float64)
    assert out.name == "STD_stack.tif" and out.data.dtype == np.float32
    assert out.data[0, 0, 0, 0, 0] == np.float32(np.sqrt((n * (v * v).sum() - v.sum() ** 2) / n / (n - 1)))
    for method, prefix in (("Max Intensity", "MAX"), ("Min Intensity", "MIN"), ("Sum Slices", "SUM")):
        assert s.images[s.run_op("stacks.z_project", {"method": method}, image_id=img.id)["new_images"][0]].name == f"{prefix}_stack.tif"
    img16 = s.add_image(LumenImage(vol.astype(np.uint16) * 300, name="s16"))
    out = s.images[s.run_op("stacks.z_project", {"method": "Median"}, image_id=img16.id)["new_images"][0]]
    assert out.data.dtype == np.float32 and out.data[0, 0, 0, 0, 1] == 450.0         # ImageJ: MED_ of a 16-bit stack is 32-bit
    out = s.images[s.run_op("stacks.z_project", {"method": "Average Intensity"}, image_id=img16.id)["new_images"][0]]
    assert out.data.dtype == np.uint16 and out.data[0, 0, 0, 0, 1] == 450


# ---------------------------------------------------------------- Stack Statistics
def test_stack_statistics_percent_volume_and_histogram_median():
    s = Session()
    vol = np.zeros((1, 2, 1, 10, 10), np.uint8)
    vol[0, 0] = 40
    vol[0, 1] = 60
    img = s.add_image(LumenImage(vol, name="z"))
    out = s.run_op("stacks.statistics", {}, image_id=img.id)
    row = s.results.rows[-1]
    assert row["Voxels"] == 200 and row["%Volume"] == 100.0 and row["Median"] == 60      # ImageJ's histogram median: the upper middle value
    assert out["table"]["columns"][-1] == "%Volume" and out["table"]["rows"][0][-1] == 100.0
    s.run_op("stacks.statistics", {}, image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 10})
    assert s.results.rows[-1]["Voxels"] == 100 and s.results.rows[-1]["%Volume"] == 100.0  # % of the selection's volume
    thr = s.add_image(LumenImage(np.arange(16, dtype=np.uint8).reshape(1, 1, 1, 4, 4), name="m"))
    s.run_op("stacks.statistics", {}, image_id=thr.id)
    assert s.results.rows[-1]["Median"] == 8 and s.results.rows[-1]["Mode"] == 0


# ---------------------------------------------------------------- noise: ImageJ kernels
def test_despeckle_and_outliers_use_imagej_rank_filters():
    s = Session()
    p = _blobs8()
    img = s.add_image(LumenImage(p.copy(), name="n"))
    s.run_op("noise.despeckle", {}, image_id=img.id)
    assert np.array_equal(img.data[0, 0, 0], rank_filter(p, 1.0, "median"))
    img = s.add_image(LumenImage(p.copy(), name="o"))
    s.run_op("noise.remove_outliers", {"radius": 2.0, "threshold": 50.0, "which": "Bright"}, image_id=img.id)
    med = rank_filter(p, 2.0, "median").astype(np.int32)
    bright = p.astype(np.int32) > med + 50
    out = img.data[0, 0, 0]
    assert np.array_equal(out[bright], med[bright].astype(np.uint8)) and np.array_equal(out[~bright], p[~bright])
    f = s.add_image(LumenImage(np.where(p > 250, np.nan, p.astype(np.float32)), name="f"))
    s.run_op("noise.remove_nans", {"radius": 2.0}, image_id=f.id)
    assert not np.isnan(f.data).any()


# ---------------------------------------------------------------- macro-layer wiring
def test_macro_layer_wiring(ij):
    calls = []
    real_run = ij._run
    ij._run = lambda op_id, params=None, **kw: calls.append((op_id, dict(params or {}))) or {}
    ij.run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=7 autoscale saturate")
    ij.run("Top Hat...", "radius=10 don't")
    ij.run("Enhance Contrast...", "saturated=0.35 process_all")
    ij.run("Enhance Contrast...", "saturated=0.35 process_all equalize use")
    ij.run("Find Maxima...", "prominence=10 above output=Count")
    ij.run("Analyze Particles...", "size=0-Infinity show=[Bare Outlines] display record composite in_situ")
    ij.run("Interpolate", "interval=2 smooth adjust")
    ij.setLineWidth(4)
    ij.run("Draw", "slice")
    ij._run = real_run
    by_op = {op: p for op, p in calls}
    assert by_op["process.fft_bandpass"]["tolerance"] == 7 and by_op["process.fft_bandpass"]["saturate"] is True
    assert by_op["filters.tophat"]["dont_subtract"] is True
    assert calls[2][1]["normalize"] is True and calls[3][1]["normalize"] is False and calls[3][1]["use_stack_histogram"] is True
    assert by_op["analyze.find_maxima"]["above_lower_threshold"] is True
    ap = by_op["analyze.particles"]
    assert ap["show"] == "Bare Outlines" and ap["record_starts"] and ap["composite_rois"] and ap["in_situ_show"]
    assert by_op["selection.interpolate"] == {"interval": 2.0, "smooth": True, "adjust": True}
    assert by_op["edit.draw"] == {"line_width": 4}
    # setOption, getSelectionBounds, getHistogram
    ij.setOption("BlackBackground", False)
    assert ij.black_background is False
    ij.setOption("BlackBackground", True)
    assert ij.black_background is True
    ij.makePolygon(20.5, 20.5, 80.2, 25.1, 50.7, 50.3)
    assert all(isinstance(v, int) for v in ij.getSelectionBounds())
    img16 = ij.session.add_image(LumenImage((_blobs8().astype(np.uint16) * 37 + 100), name="b16"))
    ij.session.select(img16.id)
    ij.makeRectangle(10, 10, 50, 40)
    values, counts = ij.getHistogram(256)
    rect, mask = roimod.roi_rect_mask(ij.roi, img16.data.shape[-2:])
    v2, c2 = macro_histogram(img16.data[0, 0, 0], rect, mask, 256, None, None)
    assert counts == c2.tolist() and values == v2.tolist() and sum(counts) == 2000
