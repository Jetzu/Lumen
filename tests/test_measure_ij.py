"""Regression tests for the ImageJ 1.54p measurement ports (lumen/core/ops/_imagej_measure.py and their wiring in
roi.py / measure.py / ops/analyze.py).  The reference numbers were produced by ImageJ 1.54p in batch mode on the
synthetic images of tests/crossval (see docs/CROSSVALIDATION.md) and verified exact by the cross-validation harness."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumen.core import roi as roimod
from lumen.core.image import Calibration, LumenImage
from lumen.core.measure import measure_roi, measure_labels
from lumen.core.ops import _imagej_measure as ijm
from lumen.core.session import Session
from tests.crossval import harness as H

ALL = ["area", "mean", "std", "mode", "min", "max", "centroid", "center_of_mass", "perimeter", "bounding_rect", "fit_ellipse",
       "shape_descriptors", "feret", "integrated_density", "median", "skewness", "kurtosis", "area_fraction", "stack_position"]


@pytest.fixture(scope="module")
def blobs():
    return {k: fn()[0] for k, fn in (("b8", H._blobs8), ("b16", H._blobs16), ("b32", H._blobs32))}


# ---------------------------------------------------------------- geometry ports
def test_polygon_filler_matches_imagej_pixel_centre_rule():
    xs, ys = [10, 120, 100, 40, 20], [10, 30, 100, 90, 60]
    m, bounds = ijm.int_polygon_mask(xs, ys)
    assert bounds == (10, 10, 110, 90) and int(m.sum()) == 6645            # ImageJ PolygonFiller (skimage gave 6655)
    full = roimod.to_mask({"type": "polygon", "points": [[x, y] for x, y in zip(xs, ys)]}, (120, 160))
    assert int(full.sum()) == 6645 and full[10:100, 10:120].sum() == 6645
    # a 1-px square outline [0,1]×[0,1] covers exactly pixel (0,0); points on the left/upper edge are outside
    assert roimod.to_mask({"type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}, (4, 4)).sum() == 1
    assert roimod.to_mask({"type": "polygon", "points": [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]]}, (4, 4)).sum() == 4


def test_oval_mask_polygon_and_feret():
    m = ijm.oval_mask(70, 50)
    assert int(m.sum()) == 2748
    assert np.array_equal(roimod.to_mask({"type": "oval", "x": 30, "y": 20, "w": 70, "h": 50}, (120, 160))[20:70, 30:100], m)
    xs, ys = ijm.oval_float_polygon(30, 20, 70, 50)                          # ShapeRoi(Ellipse2D) flattened at 0.1 px
    assert xs.dtype == np.float32 and len(xs) == 64 and xs[0] == pytest.approx(100.0004, abs=1e-4) and ys[0] == 45.0
    assert len(roimod.polygon_points({"type": "oval", "x": 30, "y": 20, "w": 70, "h": 50})) == 64
    assert ijm.oval_feret_values(30, 20, 70, 50) == (70.0, 0.0, 50.0, 30.0, 45.0)
    assert ijm.oval_feret_values(30, 20, 40, 50) == (50.0, 90.0, 40.0, 50.0, 70.0)
    assert ijm.oval_length(70, 50) == pytest.approx(189.80678, abs=1e-4)      # Ramanujan
    assert ijm.hull_int_area(xs, ys) == 2746.0                               # Solidity 2748/2746 > 1, as ImageJ reports


def test_wand_and_traced_perimeter():
    sq = np.zeros((10, 10), bool)
    sq[2:6, 3:8] = True
    out = ijm.wand_outline(sq, 3, 2)
    assert out is not None and sorted(zip(*out)) == [(3, 2), (3, 6), (8, 2), (8, 6)]
    assert ijm.traced_perimeter(*out) == pytest.approx(2 * (5 + 4) - 4 * (2 - math.sqrt(2)))
    assert ijm.is_traced([3, 8, 8, 3], [2, 2, 6, 6]) and not ijm.is_traced([0, 5, 3], [0, 0, 4])
    assert roimod.perimeter({"type": "polygon", "points": [[3, 2], [8, 2], [8, 6], [3, 6]]}) == pytest.approx(18 - 4 * (2 - math.sqrt(2)))


def test_convex_hull_feret_and_ellipse_fitter():
    hx, hy = ijm.convex_hull([20, 80, 80, 20], [15, 15, 60, 60])
    assert len(hx) == 4 and (hx[0], hy[0]) == (20, 15)
    feret, angle, min_feret, fx, fy = ijm.feret_values(hx, hy)
    assert feret == 75.0 and fx == 20.0 and fy == 15.0 and angle == pytest.approx(143.1301, abs=1e-4) and min_feret == 45.0
    ef = ijm.ellipse_fitter(None, 20, 15, 60, 45)                           # rectangle without mask: 2w/√π
    assert ef.major == pytest.approx(67.7028, abs=1e-4) and ef.minor == pytest.approx(50.7771, abs=1e-4) and ef.angle == 0.0
    assert ijm.ellipse_fitter(None, 0, 0, 160, 120).major == pytest.approx(180.5407, abs=1e-4)
    disc = ijm.oval_mask(21, 21)
    ef = ijm.ellipse_fitter(disc, 0, 0, 21, 21)
    assert ef.major == pytest.approx(ef.minor, rel=1e-6) and math.pi * (ef.major / 2) ** 2 == pytest.approx(disc.sum(), rel=1e-9)


def test_fit_ellipse_polygon_is_imagej_ellipse_roi():
    poly = {"type": "polygon", "points": [[20, 20], [80, 25], [50, 50], [90, 90], [30, 80], [45, 55]]}
    (x, y, w, h), m = roimod.roi_rect_mask(poly, (120, 160))
    xs, ys = ijm.fit_ellipse_polygon(m, x, y, w, h)
    roi = {"type": "polygon", "points": np.stack([xs.astype(float), ys.astype(float)], 1).tolist(), "subtype": "ellipse"}  # EllipseRoi bounds rule
    assert len(xs) == 72 and int(roimod.to_mask(roi, (120, 160)).sum()) == 2011 and roimod.int_bounds(roi) == (31, 20, 42, 68)


# ---------------------------------------------------------------- statistics
def test_image_statistics_16bit_histogram_median_and_moments(blobs):
    st = ijm.image_statistics(blobs["b16"], (20, 15, 60, 45), None, median=True, moments=True, ellipse=True)
    assert st.pixel_count == 2700 and st.median == 762.0 and st.min == 162.0 and st.max == 4823.0 and st.dmode == 499.0
    assert st.mean == pytest.approx(1207.5448148, abs=1e-6) and st.std_dev == pytest.approx(980.97680666, abs=1e-6)
    assert st.skewness == pytest.approx(1.53136720, abs=1e-6) and st.kurtosis == pytest.approx(1.59682212, abs=1e-6)
    values, counts = ijm.macro_histogram(blobs["b16"], None, None, 256)
    assert counts[:8].tolist() == [1, 0, 5, 8, 13, 21, 22, 68] and values[0] == 121.0 and int(counts.sum()) == 160 * 120
    values, counts = ijm.macro_histogram(blobs["b32"], None, None, 64)
    assert counts[:4].tolist() == [37, 236, 1143, 2786] and int(counts.sum()) == 160 * 120
    v8, c8 = ijm.macro_histogram(blobs["b8"], None, None, 256)
    assert np.array_equal(c8, np.bincount(blobs["b8"].ravel(), minlength=256)) and v8[255] == 255.0


def test_float_statistics_median_mode_and_constant_image():
    p = np.array([[1.0, 2.0, 4.0, 10.0]], np.float32)
    st = ijm.image_statistics(p, None, None, median=True)
    assert st.median == 3.0 and st.min == 1.0 and st.max == 10.0 and st.bin_size == pytest.approx(9 / 256)
    flat = ijm.image_statistics(np.full((3, 3), 5.0, np.float32), None, None, median=True)
    assert flat.dmode == 5.0 and flat.histogram[0] == 9 and flat.median == 5.0


def test_get_line_samples_pixel_centres_with_bilinear_interpolation(blobs):
    p = blobs["b8"]
    vals = ijm.get_line(p, 10, 20, 140, 100)
    assert vals.size == 154 and vals[0] == p[20, 10] and vals[-1] == pytest.approx(float(p[100, 140]), abs=1e-9)
    assert ijm.interpolated_value(p.astype(np.float32), 10.5, 20.0) == pytest.approx((float(p[20, 10]) + float(p[20, 11])) / 2)


# ---------------------------------------------------------------- Measure rows
def _img(arr, **cal):
    return LumenImage(arr, name="t.tif", calibration=Calibration(**cal) if cal else Calibration())


def test_measure_rect_oval_polygon_rows(blobs):
    img = _img(blobs["b8"])
    r = measure_roi(img, {"type": "rect", "x": 20, "y": 15, "w": 60, "h": 45}, ALL)[0]
    assert r["Area"] == 2700 and r["Major"] == pytest.approx(67.7028, abs=1e-4) and r["Angle"] == 0.0 and r["Perim."] == 210
    assert (r["FeretX"], r["FeretY"], r["MinFeret"], r["Solidity"]) == (20.0, 15.0, 45.0, 1.0) and r["Median"] == 39.0
    assert r["X"] == 50.0 and r["Y"] == 37.5 and r["BX"] == 20 and r["Width"] == 60 and r["%Area"] == 100.0
    o = measure_roi(img, {"type": "oval", "x": 30, "y": 20, "w": 70, "h": 50}, ALL)[0]
    assert o["Area"] == 2748 and o["FeretAngle"] == 0.0 and o["Feret"] == 70.0 and o["MinFeret"] == 50.0 and o["FeretY"] == 45.0
    assert o["Solidity"] == pytest.approx(2748 / 2746) and o["Perim."] == pytest.approx(189.8068, abs=1e-4)
    p = measure_roi(img, {"type": "polygon", "points": [[10, 10], [120, 30], [100, 100], [40, 90], [20, 60]]}, ALL)[0]
    assert p["Area"] == 6645 and p["IntDen"] == 404650 and p["Angle"] == pytest.approx(155.5778, abs=1e-4) and p["FeretAngle"] == 135.0
    w = measure_roi(img, None, ALL)[0]
    assert w["Area"] == 160 * 120 and w["Perim."] == 560 and w["Major"] == pytest.approx(180.5407, abs=1e-4) and w["Solidity"] == 1.0
    assert list(r)[:8] == ["Label", "Area", "Mean", "StdDev", "Mode", "Min", "Max", "X"]   # ImageJ's column order


def test_measure_line_point_limit_and_invert_y(blobs):
    img = _img(blobs["b8"])
    ln = measure_roi(img, {"type": "line", "x1": 10, "y1": 20, "x2": 140, "y2": 100, "width": 1}, ALL)[0]
    assert ln["Area"] == 154 and ln["Mean"] == pytest.approx(64.94088, abs=1e-4) and ln["X"] == 75.0 and ln["Y"] == 60.0
    assert ln["XM"] == 0.0 and ln["Major"] == 0.0 and ln["Angle"] == pytest.approx(-31.6075, abs=1e-4) and ln["Length"] == ln["Perim."]
    assert ln["MinFeret"] == 0.0 and ln["AR"] == 0.0 and math.isnan(ln["Solidity"]) and ln["%Area"] == 0.0 and ln["Skew"] == 0.0
    assert (ln["BX"], ln["BY"], ln["Width"], ln["Height"]) == (10, 20, 131, 81)
    wide = measure_roi(img, {"type": "line", "x1": 10, "y1": 20, "x2": 140, "y2": 100, "width": 5}, ALL)[0]
    assert wide["Area"] == 760 and wide["IntDen"] == 49582 and wide["MinFeret"] == pytest.approx(5.0, abs=1e-5)   # float32 RotatedRectRoi vertices
    pt = measure_roi(img, {"type": "point", "points": [[37, 52]]}, ["area", "mean", "min", "max", "centroid"])[0]
    assert pt["Area"] == 0 and pt["Mean"] == pt["Min"] == pt["Max"] == 42 and pt["X"] == 37 and pt["Y"] == 52 and pt["Value"] == 42
    lim = measure_roi(img, None, ["area", "mean", "std", "min", "max", "centroid", "area_fraction"], limit_to_threshold=(91, 255))[0]
    assert lim["Area"] == 3082 and lim["Y"] == pytest.approx(70.2352, abs=1e-4) and lim["MinThr"] == 91 and lim["MaxThr"] == 255
    assert lim["%Area"] == pytest.approx(16.0521, abs=1e-4)
    inv = measure_roi(img, {"type": "oval", "x": 30, "y": 20, "w": 70, "h": 50}, ["area", "centroid", "center_of_mass", "bounding_rect", "feret"],
                      options={"invert_y": True})[0]
    assert inv["BY"] == 99 and inv["Y"] == 74.0 and inv["FeretY"] == 45.0       # (height − y − 1); FeretY stays a pixel coordinate


def test_calibrated_feret_x_y_stay_pixels(blobs):
    img = _img(blobs["b8"], pixel_width=0.25, pixel_height=0.25, unit="um")
    r = measure_roi(img, {"type": "rect", "x": 20, "y": 15, "w": 60, "h": 45}, ALL)[0]
    assert r["Area"] == pytest.approx(2700 * 0.0625) and r["Feret"] == 75 * 0.25 and (r["FeretX"], r["FeretY"]) == (20.0, 15.0)
    assert r["Major"] == pytest.approx(67.7028 * 0.25, abs=1e-4) and r["Perim."] == 210 * 0.25


# ---------------------------------------------------------------- Analyze Particles / Find Maxima through the session
def _mask_session(b8):
    s = Session()
    img = s.add_image(LumenImage(b8.copy(), name="input.tif"))
    s.run_op("threshold.compute", {"method": "Default"})
    s.run_op("threshold.apply", {})
    return s, img


def test_analyze_particles_outputs_like_imagej(blobs):
    s, img = _mask_session(blobs["b8"])
    s.measurements = ALL
    opts = {"clear_results": True, "add_to_manager": False, "include_holes": False}
    out = s.run_op("analyze.particles", {"show": "Masks", "display_results": False, **opts})
    masks = s.images[out["new_images"][0]]
    assert masks.name == "Mask of input.tif" and masks.data.dtype == np.uint8 and s.active is masks and len(s.results.rows) == 11
    assert np.array_equal(masks.current_plane() == 255, img.current_plane() == 255)   # rows recorded even without 'display' (batch mode)
    ring = max(s.results.rows, key=lambda r: r["Area"])
    assert all(k in ring for k in ("Perim.", "Major", "Feret", "Solidity")) and ring["%Area"] == 100.0
    out = s.run_op("analyze.particles", {"show": "Count Masks", **opts}, image_id=img.id)   # the mask image became the active one
    cm = s.images[out["new_images"][0]]
    assert cm.name == "Count Masks of input.tif" and cm.data.dtype == np.uint16 and int(cm.current_plane().max()) == 11
    out = s.run_op("analyze.particles", {"show": "Bare Outlines", "record_starts": True, **opts}, image_id=img.id)
    dr = s.images[out["new_images"][0]]
    assert dr.name == "Bare Outlines of input.tif" and set(np.unique(dr.current_plane()).tolist()) == {0, 255} and dr.current_plane()[0, 0] == 255
    assert "XStart" in s.results.rows[0] and s.results.rows[0]["YStart"] <= s.results.rows[-1]["YStart"]   # raster order


def test_particles_from_labels_measure_like_traced_rois():
    a = np.zeros((30, 30), np.uint8)
    a[5:25, 5:25] = 255
    a[12:18, 12:18] = 0
    labels = (a > 0).astype(np.int32)
    row = measure_labels(_img(a), labels, ["area", "perimeter", "shape_descriptors", "feret"], 0, 0, 0)[0]
    assert row["Area"] == 364 and row["Perim."] == pytest.approx(80 - 4 * (2 - math.sqrt(2)))   # outer outline only
    assert row["Feret"] == pytest.approx(20 * math.sqrt(2)) and row["Solidity"] == pytest.approx(364 / 400)


def test_find_maxima_counts_positions_and_images(blobs):
    s = Session()
    for name, expected in (("b8", 13), ("b16", 1164), ("b32", 1)):
        img = s.add_image(LumenImage(blobs[name].copy(), name=name))
        s.run_op("analyze.find_maxima", {"prominence": 30, "output": "List"}, image_id=img.id)
        assert len(s.results.rows) == expected and s.results.columns == ["X", "Y"], name
    img = s.add_image(LumenImage(blobs["b8"].copy(), name="b8"))
    out = s.run_op("analyze.find_maxima", {"prominence": 30, "output": "Point Selection"}, image_id=img.id)
    pts = out["roi"]["points"]
    assert len(pts) == 13 and sum(p[0] for p in pts) == 1050 and sum(p[1] for p in pts) == 867 and all(p[0] == int(p[0]) for p in pts)
    s.run_op("analyze.find_maxima", {"prominence": 30, "output": "List", "strict": True, "exclude_edges": True}, image_id=img.id)
    assert len(s.results.rows) == 13
    out = s.run_op("analyze.find_maxima", {"prominence": 30, "output": "Single Points"}, image_id=img.id)
    assert s.images[out["new_images"][0]].name == f"{img.name} Maxima" and int((s.images[out["new_images"][0]].current_plane() == 255).sum()) == 13
    out = s.run_op("analyze.find_maxima", {"prominence": 30, "output": "Segmented Particles"}, image_id=img.id)
    seg = s.images[out["new_images"][0]].current_plane()
    assert s.images[out["new_images"][0]].name == f"{img.name} Segmented" and int((seg == 255).sum()) == 17918
    img32 = s.add_image(LumenImage(blobs["b32"].copy(), name="b32"))
    out = s.run_op("analyze.find_maxima", {"prominence": 30, "output": "Point Selection"}, image_id=img32.id)
    assert len(out["roi"]["points"]) == 1
    s.run_op("analyze.find_maxima", {"prominence": 30, "output": "List", "light_background": True}, image_id=img.id)
    assert len(s.results.rows) > 0
