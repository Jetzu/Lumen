"""Plugins ▸ MorphoLibJ, part 2 (lumen/core/ops/morpho_extra.py): Labels To RGB, Assign Measure to Label, Microstructure
Analysis, Analyze Regions 3D, Label Edition operations, morphological filters, minima / maxima, chamfer and geodesic
distance maps, marker-controlled watershed, reconstruction, kill borders, fill holes, area opening, directional
filtering and overlays."""
import math

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import measure as skmeas
from skimage.draw import disk

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops import morpho_extra as M
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

NEW_OPS = ["morpho.labels_to_rgb", "morpho.assign_measure_to_label", "morpho.microstructure_analysis", "morpho.analyze_regions_3d",
           "morpho.select_label", "morpho.remove_label_at", "morpho.merge_labels", "morpho.fill_label_holes", "morpho.label_size_opening",
           "morpho.filters", "morpho.filters_3d", "morpho.regional_min_max", "morpho.extended_min_max", "morpho.impose_min_max",
           "morpho.chamfer_distance_map", "morpho.geodesic_distance_map", "morpho.marker_watershed", "morpho.reconstruct",
           "morpho.kill_borders", "morpho.fill_holes", "morpho.area_opening", "morpho.directional_filter", "morpho.binary_overlay",
           "morpho.label_overlay"]


def ball(shape=(40, 50, 50), centre=(20, 25, 25), r=10, spacing=(1.0, 1.0, 1.0)):
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - centre[0]) * spacing[0]) ** 2 + ((yy - centre[1]) * spacing[1]) ** 2 + ((xx - centre[2]) * spacing[2]) ** 2 <= r * r


def three_labels():
    """1 = disk r 12, 2 = 10 × 20 rectangle, 3 = ring (outer 8, inner 3)."""
    lab = np.zeros((80, 100), np.uint16)
    rr, cc = disk((25, 25), 12)
    lab[rr, cc] = 1
    lab[50:60, 40:60] = 2
    rr, cc = disk((30, 75), 8)
    lab[rr, cc] = 3
    rr, cc = disk((30, 75), 3)
    lab[rr, cc] = 0
    return lab


def session_with(arr, name="lab", **kw):
    s = Session()
    im = s.add_image(LumenImage(arr, name=name, **kw))
    return s, im


# ------------------------------------------------------------------ registration / estimators
def test_new_ops_registered_in_the_morpholibj_menu():
    for op_id in NEW_OPS:
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Plugins/MorphoLibJ" and spec.description and spec.keywords, op_id


def test_configuration_estimators_match_scikit_image_and_known_shapes():
    rng = np.random.default_rng(0)
    b = rng.random((40, 50)) > 0.6
    assert M._euler_2d(b, 4, True) == skmeas.euler_number(b, connectivity=1)
    assert M._euler_2d(b, 8, True) == skmeas.euler_number(b, connectivity=2)
    assert M._perimeter_2d(b, 4, 1, 1, True) == pytest.approx(skmeas.perimeter_crofton(b, directions=4))
    assert M._perimeter_2d(b, 2, 1, 1, True) == pytest.approx(skmeas.perimeter_crofton(b, directions=2))
    v = rng.random((12, 14, 16)) > 0.7
    assert M._euler_3d(v, 26, True) == skmeas.euler_number(v, connectivity=3)
    assert M._euler_3d(v, 6, True) == skmeas.euler_number(v, connectivity=1)
    w = M._direction_weights((1.0, 1.0, 1.0), 13)
    assert sum(w) == pytest.approx(1.0) and w[0] == pytest.approx(0.0915558, abs=1e-6) and w[3] == pytest.approx(0.0739613, abs=1e-6) and w[9] == pytest.approx(0.0703913, abs=1e-6)
    sph = ball((61, 61, 61), (30, 30, 30), 20)
    for nd in (3, 13):
        assert M._crofton_surface(sph, (1, 1, 1), nd, True) == pytest.approx(4 * math.pi * 400, rel=0.01)
        assert M._mean_breadth(sph, (1, 1, 1), nd, 8, True) == pytest.approx(40, rel=0.03)
    # anisotropic voxels: the same ball sampled with z spacing 2
    sph_z = ball((31, 61, 61), (15, 30, 30), 20, spacing=(2, 1, 1))
    assert M._crofton_surface(sph_z, (2, 1, 1), 13, True) == pytest.approx(4 * math.pi * 400, rel=0.01)
    assert M._mean_breadth(sph_z, (2, 1, 1), 13, 8, True) == pytest.approx(40, rel=0.03)
    # edge correction: interior configurations only – a ball cut by the border loses the cut faces
    half = sph.copy()
    half[:30] = False
    assert M._crofton_surface(half[30:], (1, 1, 1), 13, False) == pytest.approx(2 * math.pi * 400, rel=0.02)   # hemisphere only
    assert M._crofton_surface(half, (1, 1, 1), 13, True) == pytest.approx(3 * math.pi * 400, rel=0.03)          # + the cut disk


# ------------------------------------------------------------------ Labels To RGB
def test_labels_to_rgb_colours_background_shuffle_and_stack():
    lab = three_labels()
    s, im = session_with(lab)
    out = s.run_op("morpho.labels_to_rgb", {"lut": "Glasbey", "background": "white"})
    rgb = s.images[out["new_images"][0]]
    assert rgb.rgb and rgb.shape == (1, 1, 3, 80, 100) and "3 labels" in out["message"]
    px = np.moveaxis(rgb.data[0, 0], 0, 2)
    assert np.array_equal(px[0, 0], [255, 255, 255])  # background colour
    cols = {k: tuple(px[lab == k][0]) for k in (1, 2, 3)}
    assert len(set(cols.values())) == 3 and all((px[lab == k] == cols[k]).all() for k in cols)
    s.select(im.id)
    out2 = s.run_op("morpho.labels_to_rgb", {"lut": "Glasbey", "shuffle": True, "seed": 3})
    px2 = np.moveaxis(s.images[out2["new_images"][0]].data[0, 0], 0, 2)
    cols2 = {k: tuple(px2[lab == k][0]) for k in (1, 2, 3)}
    assert set(cols2.values()) == set(cols.values()) and cols2 != cols
    for lut in ("Golden angle", "Fire", "Glasbey on dark", "Random"):
        s.select(im.id)
        o = s.run_op("morpho.labels_to_rgb", {"lut": lut, "background": "#000000"})
        p = np.moveaxis(s.images[o["new_images"][0]].data[0, 0], 0, 2)
        assert len({tuple(p[lab == k][0]) for k in (1, 2, 3)}) == 3 and np.array_equal(p[0, 0], [0, 0, 0]), lut
    with pytest.raises(ValueError, match="colour"):
        s.select(im.id)
        s.run_op("morpho.labels_to_rgb", {"background": "nope"})
    # 3-D labels: one colour per label across slices, labels missing on a slice do not shift the colours
    stack = np.stack([lab, np.where(lab == 1, 0, lab)])
    s2, im2 = session_with(stack, axes="ZYX")
    o = s2.run_op("morpho.labels_to_rgb", {"all_planes": True})
    r = s2.images[o["new_images"][0]]
    assert r.shape == (1, 2, 3, 80, 100)
    p0, p1 = np.moveaxis(r.data[0, 0], 0, 2), np.moveaxis(r.data[0, 1], 0, 2)
    assert np.array_equal(p0[lab == 2][0], p1[lab == 2][0]) and np.array_equal(p0[lab == 3][0], p1[lab == 3][0])
    assert np.array_equal(p1[lab == 1][0], [0, 0, 0])


# ------------------------------------------------------------------ Assign Measure to Label
def test_assign_measure_to_label_from_results_and_tables():
    lab = three_labels()
    s, im = session_with(lab)
    s.run_op("morpho.analyze_regions", {"features": ["Area", "Euler number"]})
    out = s.run_op("morpho.assign_measure_to_label", {"column": "Area"})
    par = s.images[out["new_images"][0]]
    assert par.dtype == np.float32 and "3 labels" in out["message"] and par.display.channels[0].lut == "Fire"
    p = par.data[0, 0, 0]
    assert p[lab == 2][0] == 200 and p[lab == 1][0] == pytest.approx((lab == 1).sum()) and (p[lab == 0] == 0).all()
    s.select(im.id)
    out = s.run_op("morpho.assign_measure_to_label", {"column": "eulernumber", "missing": "NaN"})
    p = s.images[out["new_images"][0]].data[0, 0, 0]
    assert p[lab == 3][0] == 0 and p[lab == 1][0] == 1 and np.isnan(p[lab == 0]).all()
    # a named table with an explicit label column; labels without a row get the missing value
    s.tables["Scores"] = {"name": "Scores", "columns": ["Object", "Score"], "rows": [[1, 0.25], [3, 0.75]]}
    s.select(im.id)
    out = s.run_op("morpho.assign_measure_to_label", {"table": "Scores", "column": "Score", "label_column": "Object"})
    p = s.images[out["new_images"][0]].data[0, 0, 0]
    assert p[lab == 1][0] == 0.25 and p[lab == 3][0] == 0.75 and p[lab == 2][0] == 0 and "2 labels" in out["message"]
    with pytest.raises(ValueError, match="No column"):
        s.select(im.id)
        s.run_op("morpho.assign_measure_to_label", {"column": "Nope"})
    with pytest.raises(ValueError, match="No table"):
        s.select(im.id)
        s.run_op("morpho.assign_measure_to_label", {"table": "Nope", "column": "Area"})


def test_assign_measure_to_label_uses_the_id_column_for_frame_suffixed_and_digit_ending_names():
    """Measure Labels 3D writes 'name:t2' Labels (time series) plus a numeric ID; an image called 'cells2' writes
    'cells2'. Neither is a label value: the rows must map through ID, not through the trailing digits of the name."""
    lab = np.zeros((2, 4, 1, 20, 20), np.uint16)
    lab[:, 0:3, 0, 2:6, 2:6] = 1      # 3 × 4 × 4 = 48 voxels per frame
    lab[:, 0:3, 0, 8:14, 8:14] = 2    # 3 × 6 × 6 = 108
    lab[:, 1:3, 0, 15:18, 2:6] = 3    # 2 × 3 × 4 = 24
    expected = {1: 48.0, 2: 108.0, 3: 24.0}
    s, im = session_with(lab, name="labs", axes="TZCYX")
    s.run_op("segment3d.measure_labels_3d", {"all_frames": True, "surface": "None"})
    rows = [dict(zip(s.tables["Labels 3D"]["columns"], r)) for r in s.tables["Labels 3D"]["rows"]]
    assert [r["Label"] for r in rows] == ["labs:t1"] * 3 + ["labs:t2"] * 3 and [r["ID"] for r in rows] == [1, 2, 3, 1, 2, 3]
    im.display.z = 1  # the slice where all three labels are present
    for table in ("Labels 3D", "Results"):
        s.select(im.id)
        out = s.run_op("morpho.assign_measure_to_label", {"column": "Voxels", "table": table})
        p = s.images[out["new_images"][0]].data[0, 0, 0]
        assert "3 labels" in out["message"]
        assert {k: float(p[lab[0, 1, 0] == k][0]) for k in expected} == expected
    # a single-frame image whose name ends in digits
    s2, im2 = session_with(lab[:1], name="cells2", axes="TZCYX")
    s2.run_op("segment3d.measure_labels_3d", {"surface": "None"})
    assert s2.tables["Labels 3D"]["rows"][0][0] == "cells2"
    im2.display.z = 1
    s2.select(im2.id)
    out = s2.run_op("morpho.assign_measure_to_label", {"column": "Voxels", "table": "Labels 3D"})
    p = s2.images[out["new_images"][0]].data[0, 0, 0]
    assert "3 labels" in out["message"] and {k: float(p[lab[0, 1, 0] == k][0]) for k in expected} == expected
    # 'image:12' rows and plain-number labels still match; an explicit column may carry any trailing digits
    assert M._row_label({"Label": "lab:12"}, "") == 12 and M._row_label({"Label": "12"}, "") == 12
    assert M._row_label({"Label": "labs:t2", "ID": 3}, "") == 3 and M._row_label({"Label": "cells2", "ID": 3}, "") == 3
    assert M._row_label({"Label 1": 1, "Label 2": 2}, "") is None and M._row_label({"Spot": "Spot 12"}, "Spot") == 12


# ------------------------------------------------------------------ Microstructure Analysis
def test_microstructure_analysis_2d_densities_with_calibration():
    img = np.zeros((100, 120), np.uint8)
    rr, cc = disk((40, 50), 15)
    img[rr, cc] = 255
    s, im = session_with(img, calibration=Calibration(pixel_width=0.5, pixel_height=0.5, unit="um"))
    out = s.run_op("morpho.microstructure_analysis", {"mode": "2D (current plane)", "directions_2d": "4"})
    row = s.results.rows[-1]
    window = 99 * 119 * 0.25
    assert row["AreaDensity"] == pytest.approx(img.astype(bool).sum() / (100 * 120))
    assert row["PerimeterDensity"] * window == pytest.approx(2 * math.pi * 15 * 0.5, rel=0.03)
    assert row["EulerDensity"] * window == pytest.approx(1.0)
    assert s.tables["Microstructure"]["columns"] == ["Label", "AreaDensity", "PerimeterDensity", "EulerDensity"] and "um²" in out["message"]
    # every label separately: two rows, the ring has Euler number 0
    s2, _ = session_with(three_labels())
    s2.run_op("morpho.microstructure_analysis", {"regions": "Each label", "features": ["Area / volume density", "Euler density"]})
    rows = s2.results.rows[-3:]
    assert [r["Label"] for r in rows] == ["lab:1", "lab:2", "lab:3"]
    assert rows[2]["EulerDensity"] == 0 and rows[1]["AreaDensity"] == pytest.approx(200 / 8000) and "PerimeterDensity" not in rows[0]
    with pytest.raises(ValueError, match="No foreground"):
        session_with(np.zeros((20, 20), np.uint8))[0].run_op("morpho.microstructure_analysis", {})


def test_microstructure_analysis_3d_densities_and_edge_correction():
    vol = ball((40, 50, 50), (20, 25, 25), 10)
    s, im = session_with((vol * 255).astype(np.uint8), axes="ZYX")
    s.run_op("morpho.microstructure_analysis", {"mode": "3D (whole stack)", "directions_3d": "13", "connectivity": "8"})
    row = s.results.rows[-1]
    window = 39 * 49 * 49
    assert row["VolumeDensity"] == pytest.approx(vol.sum() / vol.size)
    assert row["SurfaceDensity"] * window == pytest.approx(4 * math.pi * 100, rel=0.03)
    assert row["MeanBreadthDensity"] * window == pytest.approx(20, rel=0.05)
    assert row["EulerDensity"] * window == pytest.approx(1.0)
    # the ball cut by the image border: its cut face is not part of the sample's surface (edge correction)
    s2, _ = session_with((vol[20:] * 255).astype(np.uint8), axes="ZYX")
    s2.run_op("morpho.microstructure_analysis", {"mode": "3D (whole stack)", "features": ["Perimeter / surface density"]})
    assert s2.results.rows[-1]["SurfaceDensity"] * (19 * 49 * 49) < 0.6 * row["SurfaceDensity"] * window


# ------------------------------------------------------------------ Analyze Regions 3D
def rotated_ellipsoid(shape=(40, 60, 60), centre=(20, 40, 42), a=14, b=8, c=5, angle_deg=30):
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    th = math.radians(angle_deg)
    u = (xx - centre[2]) * math.cos(th) + (yy - centre[1]) * math.sin(th)
    v = -(xx - centre[2]) * math.sin(th) + (yy - centre[1]) * math.cos(th)
    w = zz - centre[0]
    return (u / a) ** 2 + (v / b) ** 2 + (w / c) ** 2 <= 1


def test_analyze_regions_3d_measures_synthetic_objects():
    lab = np.zeros((40, 60, 60), np.uint8)
    lab[ball((40, 60, 60), (20, 15, 15), 10)] = 1
    lab[5:11, 40:48, 30:40] = 2                       # box 10 (x) × 8 (y) × 6 (z)
    lab[rotated_ellipsoid()] = 3
    s, im = session_with(lab, axes="ZYX", calibration=Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=1.0, unit="um"))
    out = s.run_op("morpho.analyze_regions_3d", {"directions": "13", "connectivity": "26"})
    assert "3 regions" in out["message"] and "um" in out["message"]
    rows = {r["Label"]: r for r in s.results.rows}
    sph, box, ell = rows["lab:1"], rows["lab:2"], rows["lab:3"]
    # the ball is sampled on 0.5 × 0.5 × 1 voxels, radius 10 voxels = 5 um in x / y... its true shape is an ellipsoid (5, 5, 10 um)
    assert sph["Volume"] == pytest.approx((lab == 1).sum() * 0.25) and sph["EulerNumber"] == 1
    assert sph["Centroid.X"] == pytest.approx(15.5 * 0.5, abs=0.05) and sph["Centroid.Z"] == pytest.approx(20.5, abs=0.05)
    assert sph["Elli.R1"] == pytest.approx(10, rel=0.03) and sph["Elli.R2"] == pytest.approx(5, rel=0.03) and sph["Elli.R3"] == pytest.approx(5, rel=0.03)
    assert sph["Elli.R1/R2"] == pytest.approx(2.0, rel=0.05) and abs(sph["Elli.Elev"]) == pytest.approx(90, abs=1)  # long axis along z
    assert sph["InscrBall.Radius"] == pytest.approx(5, abs=0.5) and sph["InscrBall.Center.Z"] == pytest.approx(20.5, abs=1.0)
    assert 0.8 < sph["Sphericity"] < 0.95 and sph["SurfaceArea"] > 0  # a 1:2 prolate spheroid (0.91 analytically)
    # box: exact bounding box, volume, inscribed sphere; sphericity well below 1
    assert box["Volume"] == pytest.approx(10 * 8 * 6 * 0.25)
    assert (box["Box.X.Min"], box["Box.X.Max"], box["Box.Y.Min"], box["Box.Y.Max"], box["Box.Z.Min"], box["Box.Z.Max"]) == (15.0, 20.0, 20.0, 24.0, 5.0, 11.0)
    assert box["InscrBall.Radius"] == pytest.approx(2.0, abs=0.3) and box["Sphericity"] < 0.9 and box["EulerNumber"] == 1  # 0.80 analytically; Crofton underestimates box faces
    # rotated ellipsoid (isotropic in x/y): radii 7, 4 (um) in the plane, 5 along z; azimuth = the rotation angle
    assert ell["Elli.R1"] == pytest.approx(7, rel=0.05) and ell["Elli.R3"] == pytest.approx(4, rel=0.06) and ell["Elli.R2"] == pytest.approx(5, rel=0.06)
    assert ell["Elli.Azim"] == pytest.approx(30, abs=3) and abs(ell["Elli.Elev"]) < 3
    assert ell["MeanBreadth"] > 0 and ell["Elli.R1/R3"] == pytest.approx(7 / 4, rel=0.08)
    # a ball with a cavity: Euler number 2 (26-connectivity) and the 3-direction / 6-connectivity variant runs too
    hollow = ball((30, 30, 30), (15, 15, 15), 11) & ~ball((30, 30, 30), (15, 15, 15), 4)
    s2, _ = session_with((hollow * 255).astype(np.uint8), axes="ZYX")
    s2.run_op("morpho.analyze_regions_3d", {"directions": "3", "connectivity": "6", "features": ["Volume", "Surface area", "Sphericity", "Euler number", "Mean breadth"]})
    r = s2.results.rows[-1]
    # mean breadth of a hollow ball = 2 (R_out − R_in): the cavity's surface carries negative mean curvature
    assert r["EulerNumber"] == 2 and r["MeanBreadth"] == pytest.approx(14, rel=0.1) and r["SurfaceArea"] == pytest.approx(4 * math.pi * (121 + 16), rel=0.05)
    s3, _ = session_with((ball((30, 30, 30), (15, 15, 15), 11) * 255).astype(np.uint8), axes="ZYX")
    s3.run_op("morpho.analyze_regions_3d", {"features": ["Volume", "Surface area", "Sphericity", "Inscribed sphere", "Mean breadth"]})
    r = s3.results.rows[-1]
    assert r["Sphericity"] == pytest.approx(1.0, abs=0.03) and r["InscrBall.Radius"] == pytest.approx(11, abs=0.6) and r["MeanBreadth"] == pytest.approx(22, rel=0.03)


# ------------------------------------------------------------------ Label Edition
def test_select_label_uses_picked_value_and_adds_to_manager():
    lab = three_labels()
    s, im = session_with(lab)
    with pytest.raises(ValueError, match="Label 255 is not on the current plane"):  # the default foreground value
        s.run_op("morpho.select_label", {"label": 0})
    s.run_op("edit.set_foreground", {"value": 0})
    with pytest.raises(ValueError, match="Pick a label"):
        s.run_op("morpho.select_label", {"label": 0})
    s.run_op("edit.set_foreground", {"value": 2})  # what the value picker does
    out = s.run_op("morpho.select_label", {"label": 0, "add_to_manager": True})
    roi = out["roi"]
    assert roi["type"] == "polygon" and roi["name"] == "L2" and "200 px" in out["message"]
    assert len(s.roi_manager.rois) == 1 and s.roi_manager.rois[0]["name"] == "L2"
    out = s.run_op("morpho.select_label", {"label": 3})
    assert out["roi"]["type"] == "mask"  # the ring has a hole → exact mask
    with pytest.raises(ValueError, match="not on the current plane"):
        s.run_op("morpho.select_label", {"label": 9})


def test_remove_label_at_point_tool_and_stack():
    lab = three_labels()
    s, im = session_with(lab)
    out = s.run_op("morpho.remove_label_at", {"x": 25, "y": 25})
    assert "Label 1 removed" in out["message"] and (im.data[0, 0, 0] == 1).sum() == 0 and (im.data[0, 0, 0] == 2).sum() == 200
    assert s.histories[im.id].labels()["undo"] == ["Remove Label"]
    out = s.run_op("morpho.remove_label_at", {"x": 0, "y": 0})
    assert "Background" in out["message"] and len(s.histories[im.id].undo) == 1  # no change, no undo entry
    with pytest.raises(ValueError, match="outside"):
        s.run_op("morpho.remove_label_at", {"x": 500, "y": 5})
    # tool interaction parameters are accepted (click removes, drag-end is ignored)
    spec = REGISTRY.get("morpho.remove_label_at")
    p = spec.coerce_params({"x": 45.7, "y": 55.2, "event": "click", "button": 0})
    assert p["x"] == 46 and p["y"] == 55
    s.run_op("morpho.remove_label_at", {"x": 45, "y": 55, "event": "drag-end"})
    assert (im.data[0, 0, 0] == 2).sum() == 200
    s.run_op("morpho.remove_label_at", {"x": 45, "y": 55, "event": "click"})
    assert (im.data[0, 0, 0] == 2).sum() == 0
    # 3-D labels: the whole label across slices
    stack = np.stack([lab, lab, lab])
    s2, im2 = session_with(stack, axes="ZYX")
    im2.display.z = 1
    s2.run_op("morpho.remove_label_at", {"x": 25, "y": 25, "all_planes": False})
    assert (im2.data[0, 1, 0] == 1).sum() == 0 and (im2.data[0, 0, 0] == 1).sum() > 0
    s2.run_op("morpho.remove_label_at", {"x": 45, "y": 55, "all_planes": True})
    assert (im2.data == 2).sum() == 0
    # binary masks: the clicked component goes, the mask stays 0/255
    s3, im3 = session_with((lab > 0).astype(np.uint8) * 255)
    s3.run_op("morpho.remove_label_at", {"x": 45, "y": 55})
    assert im3.data[0, 0, 0, 55, 45] == 0 and im3.data.max() == 255 and set(np.unique(im3.data)) == {0, 255}


def test_merge_labels_by_list_and_by_points_and_replace_label_picked():
    lab = three_labels()
    s, im = session_with(lab)
    s.run_op("morpho.merge_labels", {"labels": "1, 2"})
    p = im.data[0, 0, 0]
    assert (p == 2).sum() == 0 and (p == 1).sum() == (lab == 1).sum() + 200 and (p == 3).sum() == (lab == 3).sum()
    s2, im2 = session_with(lab)
    with pytest.raises(ValueError, match="at least two"):
        s2.run_op("morpho.merge_labels", {"labels": ""})
    pts = {"type": "point", "points": [[50, 55], [75, 25]]}
    s2.run_op("morpho.merge_labels", {"labels": "", "new_value": 7}, roi=pts)
    p = im2.data[0, 0, 0]
    assert (p == 7).sum() == 200 + (lab == 3).sum() and (p == 2).sum() == 0 and (p == 3).sum() == 0
    # point ROIs of the manager work as well
    s3, im3 = session_with(lab)
    s3.roi_manager.add({"type": "point", "points": [[25, 25], [50, 55]]}, image_id=im3.id)
    s3.run_op("morpho.merge_labels", {})
    assert (im3.data[0, 0, 0] == 1).sum() == (lab == 1).sum() + 200
    # Replace / Remove Labels with an empty list removes the picked label
    s4, im4 = session_with(lab)
    s4.run_op("edit.set_foreground", {"value": 3})
    s4.run_op("morpho.replace_label", {"labels": "", "new_value": 0})
    assert (im4.data == 3).sum() == 0 and (im4.data == 1).sum() == (lab == 1).sum()


def test_fill_label_holes_and_size_opening():
    lab = three_labels()
    lab[52:55, 45:48] = 0  # a hole inside the rectangle; the ring's hole is label 3's
    s, im = session_with(lab)
    s.run_op("morpho.fill_label_holes", {})
    p = im.data[0, 0, 0]
    assert (p == 2).sum() == 200 and (p == 3).sum() == (lab == 3).sum() + ndi.binary_fill_holes(lab == 3).sum() - (lab == 3).sum()
    assert (p == 1).sum() == (lab == 1).sum() and (p == 0).sum() == (lab == 0).sum() - 9 - (ndi.binary_fill_holes(lab == 3).sum() - (lab == 3).sum())
    # a hole between two labels is shared
    two = np.zeros((20, 40), np.uint8)
    two[:, :18] = 1
    two[:, 22:] = 2
    two[0, 18:22] = 1
    two[-1, 18:22] = 2
    s2, im2 = session_with(two)
    s2.run_op("morpho.fill_label_holes", {})
    p = im2.data[0, 0, 0]
    assert (p == 0).sum() == 0 and (p[5:15, 18:20] == 1).all() and (p[5:15, 20:22] == 2).all()
    # size opening
    s3, im3 = session_with(three_labels())
    s3.run_op("morpho.label_size_opening", {"min_size": 210})
    p = im3.data[0, 0, 0]
    assert (p == 2).sum() == 0 and (p == 1).sum() > 0 and (p == 3).sum() == 0
    s4, im4 = session_with(three_labels())
    s4.run_op("morpho.label_size_opening", {"min_size": 1, "max_size": 250})
    p = im4.data[0, 0, 0]
    assert (p == 1).sum() == 0 and (p == 2).sum() == 200 and (p == 3).sum() > 0


# ------------------------------------------------------------------ Morphological Filters
def test_morphological_filters_2d_elements_and_operations():
    img = np.zeros((41, 41), np.uint8)
    img[20, 20] = 200
    s, im = session_with(img)
    s.run_op("morpho.filters", {"operation": "Dilation", "element": "Square", "radius": 3})
    assert (im.data[0, 0, 0] == 200).sum() == 49
    # MorphoLibJ's elements (measured with the real plugin): the disk is ImageJ's dx²+dy² <= r²+1 (37 px for r = 3)
    for element, count in (("Disk", 37), ("Diamond", 25), ("Octagon", 37)):
        s2, im2 = session_with(img)
        s2.run_op("morpho.filters", {"operation": "Dilation", "element": element, "radius": 3})
        assert (im2.data[0, 0, 0] == 200).sum() == count, element
    # line elements: horizontal, vertical and diagonal
    s3, im3 = session_with(img)
    s3.run_op("morpho.filters", {"operation": "Dilation", "element": "Line", "radius": 4, "line_angle": 0})
    p = im3.data[0, 0, 0]
    assert (p[20, 16:25] == 200).all() and (p == 200).sum() == 9
    s4, im4 = session_with(img)
    s4.run_op("morpho.filters", {"operation": "Dilation", "element": "Line", "radius": 4, "line_angle": 90})
    assert (im4.data[0, 0, 0][16:25, 20] == 200).all() and (im4.data[0, 0, 0] == 200).sum() == 9
    s5, im5 = session_with(img)
    s5.run_op("morpho.filters", {"operation": "Dilation", "element": "Line", "radius": 4, "line_angle": 45})
    p = im5.data[0, 0, 0]
    assert p[16, 24] == 200 and p[24, 16] == 200 and p[16, 16] == 0
    # erosion of a square shrinks it; opening removes a speck; closing fills a gap
    sq = np.zeros((40, 40), np.uint16)
    sq[10:30, 10:30] = 1000
    sq[5, 5] = 1000
    s6, im6 = session_with(sq)
    s6.run_op("morpho.filters", {"operation": "Erosion", "element": "Square", "radius": 2})
    assert (im6.data[0, 0, 0] == 1000).sum() == 16 * 16
    s7, im7 = session_with(sq)
    s7.run_op("morpho.filters", {"operation": "Opening", "element": "Disk", "radius": 2})
    assert im7.data[0, 0, 0, 5, 5] == 0 and (im7.data[0, 0, 0] == 1000).sum() >= 18 * 18
    s8, im8 = session_with(sq)
    s8.run_op("morpho.filters", {"operation": "White Top Hat", "element": "Disk", "radius": 2})
    assert im8.data[0, 0, 0, 5, 5] == 1000 and im8.data[0, 0, 0, 20, 20] == 0
    # gradients and the Laplacian (mid-gray offset on integer images, signed on 32-bit)
    step = np.zeros((20, 20), np.uint8)
    step[:, 10:] = 100
    s9, im9 = session_with(step)
    s9.run_op("morpho.filters", {"operation": "Gradient", "element": "Square", "radius": 1})
    g = im9.data[0, 0, 0]
    assert g[5, 9] == 100 and g[5, 10] == 100 and g[5, 5] == 0 and g[5, 15] == 0
    s10, im10 = session_with(step)
    s10.run_op("morpho.filters", {"operation": "Laplacian", "element": "Square", "radius": 1})
    lap = im10.data[0, 0, 0]
    assert lap[5, 5] == 128 and lap[5, 9] == 228 and lap[5, 10] == 28  # MorphoLibJ: dilation + erosion - 2·image + 128
    s11, im11 = session_with(step.astype(np.float32))
    s11.run_op("morpho.filters", {"operation": "Laplacian", "element": "Square", "radius": 1})
    assert im11.data[0, 0, 0, 5, 9] == 100 and im11.data[0, 0, 0, 5, 10] == -100  # float: no offset
    s12, im12 = session_with(step)
    s12.run_op("morpho.filters", {"operation": "Internal Gradient", "element": "Square", "radius": 1})
    assert im12.data[0, 0, 0, 5, 10] == 100 and im12.data[0, 0, 0, 5, 9] == 0
    s13, im13 = session_with(step)
    s13.run_op("morpho.filters", {"operation": "External Gradient", "element": "Square", "radius": 1})
    assert im13.data[0, 0, 0, 5, 9] == 100 and im13.data[0, 0, 0, 5, 10] == 0
    # RGB images are processed channel by channel
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[10, 10] = [255, 0, 0]
    s14, im14 = session_with(rgb, rgb=True)
    s14.run_op("morpho.filters", {"operation": "Dilation", "element": "Square", "radius": 1})
    assert im14.rgb and (im14.data[0, 0, 0] == 255).sum() == 9 and im14.data[0, 0, 1].max() == 0


def test_morphological_filters_3d_elements():
    vol = np.zeros((11, 11, 11), np.uint8)
    vol[5, 5, 5] = 255
    for element, count in (("Cube", 5 * 7 * 7), ("Cross", 7 + 7 + 5 - 2), ("Ball", int(M._footprint_3d("Ball", 3, 3, 2).sum()))):
        s, im = session_with(vol, axes="ZYX")
        out = s.run_op("morpho.filters_3d", {"operation": "Dilation", "element": element, "radius_x": 3, "radius_y": 3, "radius_z": 2})
        assert (im.data == 255).sum() == count, element
        assert element in out["message"]
    assert 60 < M._footprint_3d("Ball", 3, 3, 2).sum() < 90  # ≈ 4/3·π·3·3·2 = 75 voxels
    # erosion then dilation of a cube (opening) leaves it intact; a hollow is closed; 16-bit and float stacks work
    cube = np.zeros((12, 12, 12), np.uint16)
    cube[2:10, 2:10, 2:10] = 4000
    s, im = session_with(cube, axes="ZYX")
    s.run_op("morpho.filters_3d", {"operation": "Opening", "element": "Cube", "radius_x": 1, "radius_y": 1, "radius_z": 1})
    assert np.array_equal(im.data[0, :, 0], cube)
    s.run_op("morpho.filters_3d", {"operation": "Erosion", "element": "Cube", "radius_x": 1, "radius_y": 1, "radius_z": 1})
    assert (im.data == 4000).sum() == 6 ** 3
    s2, im2 = session_with(cube.astype(np.float32) / 7, axes="ZYX")
    s2.run_op("morpho.filters_3d", {"operation": "Gradient", "element": "Ball", "radius_x": 1, "radius_y": 1, "radius_z": 1})
    assert im2.data.max() == pytest.approx(4000 / 7) and im2.data[0, 6, 0, 6, 6] == 0


# ------------------------------------------------------------------ minima / maxima
def two_basins():
    yy, xx = np.mgrid[:40, :60]
    img = 200 - 150 * np.exp(-((xx - 15) ** 2 + (yy - 20) ** 2) / 50.0) - 40 * np.exp(-((xx - 45) ** 2 + (yy - 20) ** 2) / 50.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_regional_extended_and_imposed_minima_maxima():
    img = two_basins()
    s, im = session_with(img)
    out = s.run_op("morpho.regional_min_max", {"operation": "Regional Minima", "connectivity": "8"})
    mins = s.images[out["new_images"][0]]
    assert mins.lut_is_binary and "2 regional minima regions" in out["message"]
    m = mins.data[0, 0, 0] > 0
    assert m[20, 15] and m[20, 45]
    s.select(im.id)
    out = s.run_op("morpho.extended_min_max", {"operation": "Extended Minima", "dynamic": 60, "connectivity": "8"})
    assert "1 extended minima region" in out["message"]
    m = s.images[out["new_images"][0]].data[0, 0, 0] > 0
    assert m[20, 15] and not m[20, 45]
    s.select(im.id)
    out = s.run_op("morpho.regional_min_max", {"operation": "Regional Maxima"})
    assert int(out["message"].split()[0]) >= 1 and (s.images[out["new_images"][0]].data[0, 0, 0][20, 15] == 0)
    s.select(im.id)
    out = s.run_op("morpho.extended_min_max", {"operation": "Extended Maxima", "dynamic": 100})
    assert "1 extended maxima region" in out["message"]
    # impose: the only regional minima of the result are the marker pixels (the deep basin disappears)
    marker = np.zeros((40, 60), np.uint8)
    marker[20, 45] = 255
    s.add_image(LumenImage(marker, name="marker"))
    s.select(im.id)
    out = s.run_op("morpho.impose_min_max", {"operation": "Impose Minima", "marker_image": "marker", "connectivity": "8"})
    imp = s.images[out["new_images"][0]]
    assert imp.dtype == np.uint8 and "1 marker region" in out["message"]
    mins = M.skm.local_minima(imp.data[0, 0, 0].astype(np.float32), connectivity=2)
    assert mins[20, 45] and mins.sum() == 1 and imp.data[0, 0, 0, 20, 45] == 0
    assert imp.data[0, 0, 0, 20, 15] > img[20, 15] + 100 and imp.data[0, 0, 0, 0, 0] == img[0, 0] + 1  # the deep basin is flooded, flat areas rise by 1
    s.select(im.id)
    out = s.run_op("morpho.impose_min_max", {"operation": "Impose Maxima", "marker_image": "marker"})
    mx = s.images[out["new_images"][0]].data[0, 0, 0]
    maxs = M.skm.local_maxima(mx.astype(np.float32), connectivity=1)
    assert maxs[20, 45] and maxs.sum() == 1 and mx[20, 45] == 255
    # float images: markers go just below the minimum, the rest one step up
    sf, imf = session_with(img.astype(np.float32) / 10)
    sf.add_image(LumenImage(marker, name="marker"))
    sf.select(imf.id)
    out = sf.run_op("morpho.impose_min_max", {"operation": "Impose Minima", "marker_image": "marker"})
    f = sf.images[out["new_images"][0]].data[0, 0, 0]
    assert f[20, 45] < f.min() + 1e-6 and M.skm.local_minima(f, connectivity=1).sum() == 1
    # 3-D: regional minima of a stack with one dark voxel per slice, extended minima of a 3-D basin
    vol = np.full((5, 20, 20), 100, np.uint8)
    vol[2, 10, 10] = 10
    vol[4, 3, 3] = 95
    s3, im3 = session_with(vol, axes="ZYX")
    out = s3.run_op("morpho.regional_min_max", {"operation": "Regional Minima", "mode": "3D (whole stack)"})
    r = s3.images[out["new_images"][0]]
    assert r.shape == (1, 5, 1, 20, 20) and "2 regional minima" in out["message"] and r.data[0, 2, 0, 10, 10] == 255 and r.data[0, 4, 0, 3, 3] == 255
    s3.select(im3.id)
    out = s3.run_op("morpho.extended_min_max", {"operation": "Extended Minima", "dynamic": 20, "mode": "3D (whole stack)"})
    r = s3.images[out["new_images"][0]]
    assert "1 extended minima" in out["message"] and r.data[0, 2, 0, 10, 10] == 255 and r.data[0, 4, 0, 3, 3] == 0


# ------------------------------------------------------------------ distance maps
def test_chamfer_distance_map_weights_outputs_and_3d():
    img = np.zeros((15, 15), np.uint8)
    img[4:11, 4:11] = 255  # 7 × 7 square: the centre is 4 steps from the background
    s, im = session_with(img)
    for weights, centre in (("Borgefors", 4.0), ("Chessboard", 4.0), ("City-Block", 4.0), ("Quasi-Euclidean", 4.0), ("Chessknight", 4.0)):
        s.select(im.id)
        out = s.run_op("morpho.chamfer_distance_map", {"weights": weights, "normalize": True})
        d = s.images[out["new_images"][0]]
        assert d.dtype == np.float32 and d.data[0, 0, 0, 7, 7] == pytest.approx(centre) and d.data[0, 0, 0, 4, 4] == 1 and d.data[0, 0, 0, 0, 0] == 0, weights
    s.select(im.id)
    out = s.run_op("morpho.chamfer_distance_map", {"weights": "Borgefors", "normalize": False, "output": "16-bit"})
    d = s.images[out["new_images"][0]]
    assert d.dtype == np.uint16 and d.data[0, 0, 0, 7, 7] == 12 and d.data[0, 0, 0, 7, 4] == 3
    # a disk: Borgefors (3,4) and chessknight (5,7,11) stay close to the Euclidean distance; city-block overestimates diagonals
    big = np.zeros((101, 101), np.uint8)
    rr, cc = disk((50, 50), 40)
    big[rr, cc] = 255
    edt = ndi.distance_transform_edt(big > 0)
    s2, im2 = session_with(big)
    for weights, tol in (("Borgefors", 0.06), ("Chessknight", 0.03), ("Quasi-Euclidean", 0.1)):
        s2.select(im2.id)
        out = s2.run_op("morpho.chamfer_distance_map", {"weights": weights})
        d = s2.images[out["new_images"][0]].data[0, 0, 0]
        assert d.max() == pytest.approx(edt.max(), rel=tol) and np.abs(d - edt)[big > 0].max() < tol * 2 * edt.max(), weights
    # 3-D: a 7 × 7 × 7 cube, centre 4 voxels away; chessknight is 2-D only; stacks of 2-D maps with 'whole stack'
    vol = np.zeros((11, 11, 11), np.uint8)
    vol[2:9, 2:9, 2:9] = 255
    s3, im3 = session_with(vol, axes="ZYX")
    out = s3.run_op("morpho.chamfer_distance_map", {"weights": "Borgefors", "mode": "3D (whole stack)"})
    d = s3.images[out["new_images"][0]]
    assert d.shape == (1, 11, 1, 11, 11) and d.data[0, 5, 0, 5, 5] == pytest.approx(4.0) and d.data[0, 2, 0, 5, 5] == 1 and d.data[0, 1, 0, 5, 5] == 0
    with pytest.raises(ValueError, match="2D only"):
        s3.select(im3.id)
        s3.run_op("morpho.chamfer_distance_map", {"weights": "Chessknight", "mode": "3D (whole stack)"})
    s3.select(im3.id)
    out = s3.run_op("morpho.chamfer_distance_map", {"weights": "Borgefors", "all_planes": True})
    d = s3.images[out["new_images"][0]]
    assert d.shape == (1, 11, 1, 11, 11) and d.data[0, 2, 0, 5, 5] == pytest.approx(4.0)  # per slice: the top face is 4 deep in 2-D


def test_geodesic_distance_map_follows_the_mask():
    mask = np.zeros((30, 50), np.uint8)
    mask[5:25, 5:10] = 255   # left arm
    mask[20:25, 5:45] = 255  # bottom bar
    mask[5:25, 40:45] = 255  # right arm
    s, im = session_with(mask)
    out = s.run_op("morpho.geodesic_distance_map", {"weights": "Borgefors"}, roi={"type": "point", "points": [[7, 6]]})
    g = s.images[out["new_images"][0]]
    assert g.dtype == np.float32 and np.isnan(g.data[0, 0, 0, 0, 0]) and g.data[0, 0, 0, 6, 7] == 0
    far = g.data[0, 0, 0, 6, 42]
    assert far > 50 and far > math.hypot(42 - 7, 0) * 1.3  # around the U, not across the gap
    assert g.data[0, 0, 0, 22, 7] == pytest.approx(16, abs=1.5)
    # marker image, chessknight weights, unreachable pixels of a second component
    marker = np.zeros((30, 50), np.uint8)
    marker[6, 7] = 255
    s.add_image(LumenImage(marker, name="marker"))
    mask2 = mask.copy()
    mask2[1:3, 1:3] = 255
    im2 = s.add_image(LumenImage(mask2, name="mask2"))
    out = s.run_op("morpho.geodesic_distance_map", {"marker_image": "marker", "weights": "Chessknight", "normalize": True})
    g = s.images[out["new_images"][0]].data[0, 0, 0]
    assert np.isinf(g[1, 1]) and "4 unreachable px" in out["message"] and g[6, 7] == 0 and g[22, 7] == pytest.approx(16, abs=1.0)
    with pytest.raises(ValueError, match="markers"):
        s.select(im2.id)
        s.run_op("morpho.geodesic_distance_map", {})
    with pytest.raises(ValueError, match="inside the mask"):
        s.select(im2.id)
        s.run_op("morpho.geodesic_distance_map", {}, roi={"type": "point", "points": [[0, 29]]})


# ------------------------------------------------------------------ watershed / reconstruction
def touching_disks_mask():
    mask = np.zeros((80, 120), bool)
    rr, cc = disk((40, 40), 25)
    mask[rr, cc] = True
    rr, cc = disk((40, 78), 25)
    mask[rr, cc] = True
    return mask


def test_marker_watershed_from_points_marker_image_mask_and_3d():
    mask = touching_disks_mask()
    edt = ndi.distance_transform_edt(mask)
    inv = (edt.max() - edt).astype(np.float32)
    s, im = session_with(inv, name="inv")
    s.roi_manager.add({"type": "point", "points": [[40, 40], [78, 40]]}, image_id=im.id)
    out = s.run_op("morpho.marker_watershed", {"markers": "Point ROIs", "connectivity": "8", "calculate_dams": True, "add_to_manager": True})
    ws = s.images[out["new_images"][0]]
    lab = ws.data[0, 0, 0]
    assert "2 catchment basins with dams" in out["message"] and "2 ROIs added" in out["message"] and set(np.unique(lab)) == {0, 1, 2}
    assert lab[40, 30] != lab[40, 88] and (lab[mask] == 0).sum() < 90 and (lab == 0)[:, 59].sum() > 0
    # a binary marker image: its components are the markers; a mask image limits the flooding
    marker = np.zeros((80, 120), np.uint8)
    marker[38:42, 38:42] = 255
    marker[38:42, 76:80] = 255
    s.add_image(LumenImage(marker, name="marker"))
    s.add_image(LumenImage((mask * 255).astype(np.uint8), name="mask"))
    s.select(im.id)
    out = s.run_op("morpho.marker_watershed", {"markers": "Marker image", "marker_image": "marker", "mask_image": "mask", "calculate_dams": False})
    lab = s.images[out["new_images"][0]].data[0, 0, 0]
    assert "2 catchment basins" in out["message"] and (lab[~mask] == 0).all() and (lab[mask] > 0).all()
    assert lab[40, 30] != lab[40, 88] and abs((lab == 1).sum() - (lab == 2).sum()) < 0.15 * mask.sum()
    # area ROIs (the active selection) and a gradient input
    s.roi_manager.clear()
    s.select(im.id)
    out = s.run_op("morpho.marker_watershed", {"markers": "Area ROIs", "input": "Sobel gradient"}, roi={"type": "rect", "x": 30, "y": 30, "w": 10, "h": 10})
    assert "1 catchment basin" in out["message"]
    with pytest.raises(ValueError, match="marker image"):
        s.select(im.id)
        s.run_op("morpho.marker_watershed", {"markers": "Marker image"})
    # 3-D: two touching balls, markers at their centres on the right slices
    vol = np.zeros((30, 40, 70), bool)
    vol |= ball((30, 40, 70), (15, 20, 20), 13)
    vol |= ball((30, 40, 70), (15, 20, 44), 13)
    edt3 = ndi.distance_transform_edt(vol)
    s3, im3 = session_with((edt3.max() - edt3).astype(np.float32), axes="ZYX")
    s3.roi_manager.add({"type": "point", "points": [[20, 20]]}, image_id=im3.id, position={"t": 0, "z": 15, "c": 0})
    s3.roi_manager.add({"type": "point", "points": [[44, 20]]}, image_id=im3.id, position={"t": 0, "z": 15, "c": 0})
    s3.add_image(LumenImage((vol * 255).astype(np.uint8), name="vmask", axes="ZYX"))
    s3.select(im3.id)
    out = s3.run_op("morpho.marker_watershed", {"markers": "Point ROIs", "mask_image": "vmask", "mode": "3D (whole stack)", "calculate_dams": False})
    w3 = s3.images[out["new_images"][0]]
    assert w3.shape == (1, 30, 1, 40, 70) and "2 catchment basins" in out["message"]
    l3 = w3.data[0, :, 0]
    assert l3[15, 20, 10] != l3[15, 20, 54] and (l3[~vol] == 0).all() and (l3[vol] > 0).all()


def test_reconstruction_kill_borders_and_fill_holes():
    img = np.zeros((40, 60), np.uint8)
    rr, cc = disk((20, 15), 8)
    img[rr, cc] = 200
    rr, cc = disk((20, 45), 8)
    img[rr, cc] = 150
    marker = np.zeros((40, 60), np.uint8)
    marker[20, 15] = 255
    s, im = session_with(img, name="mask")
    s.add_image(LumenImage(marker, name="marker"))
    s.select(im.id)
    out = s.run_op("morpho.reconstruct", {"operation": "By Dilation", "marker_image": "marker", "connectivity": "8"})
    rec = s.images[out["new_images"][0]]
    assert rec.dtype == np.uint8 and rec.data[0, 0, 0, 20, 15] == 200 and (rec.data[0, 0, 0] == 200).sum() == (img == 200).sum() and rec.data[0, 0, 0, 20, 45] == 0
    hi = np.full((40, 60), 255, np.uint8)
    hi[0, :], hi[-1, :], hi[:, 0], hi[:, -1] = 0, 0, 0, 0
    s.add_image(LumenImage(hi, name="hi"))
    s.select(im.id)
    out = s.run_op("morpho.reconstruct", {"operation": "By Erosion", "marker_image": "hi"})
    assert np.array_equal(s.images[out["new_images"][0]].data[0, 0, 0], img)  # erosion from the border = fill holes: none here
    s.add_image(LumenImage(np.full((40, 60), 255, np.uint8), name="flat"))
    s.select(im.id)
    out = s.run_op("morpho.reconstruct", {"operation": "By Erosion", "marker_image": "flat"})
    assert (s.images[out["new_images"][0]].data == 255).all()  # a flat marker above the mask stays flat
    # 3-D reconstruction
    vol = np.stack([img, img, img])
    s3, im3 = session_with(vol, axes="ZYX")
    mk3 = np.zeros_like(vol)
    mk3[1, 20, 45] = 255
    s3.add_image(LumenImage(mk3, name="mk3", axes="ZYX"))
    s3.select(im3.id)
    out = s3.run_op("morpho.reconstruct", {"operation": "By Dilation", "marker_image": "mk3", "mode": "3D (whole stack)"})
    r3 = s3.images[out["new_images"][0]]
    assert r3.shape == (1, 3, 1, 40, 60) and r3.data[0, 0, 0, 20, 45] == 150 and r3.data[0, 2, 0, 20, 15] == 0
    # kill borders (binary + gray), fill holes (binary + gray)
    b = np.zeros((40, 40), np.uint8)
    b[0:10, 0:10] = 255
    b[20:30, 20:30] = 255
    s4, im4 = session_with(b)
    s4.run_op("morpho.kill_borders", {})
    assert im4.data[0, 0, 0, 5, 5] == 0 and im4.data[0, 0, 0, 25, 25] == 255
    g = np.full((40, 40), 20, np.uint8)
    g[0:10, 0:10] = 200
    g[20:30, 20:30] = 180
    s5, im5 = session_with(g)
    s5.run_op("morpho.kill_borders", {"connectivity": "8"})
    assert im5.data[0, 0, 0, 5, 5] == 0 and im5.data[0, 0, 0, 25, 25] == 160 and im5.data[0, 0, 0, 15, 15] == 0
    ring = np.zeros((40, 40), np.uint8)
    rr, cc = disk((20, 20), 12)
    ring[rr, cc] = 255
    rr, cc = disk((20, 20), 5)
    ring[rr, cc] = 0
    s6, im6 = session_with(ring)
    s6.run_op("morpho.fill_holes", {})
    assert im6.data[0, 0, 0, 20, 20] == 255 and (im6.data[0, 0, 0] == 255).sum() == ndi.binary_fill_holes(ring > 0).sum()
    basin = np.full((40, 40), 100, np.float32)
    basin[15:25, 15:25] = 30
    basin[0, 0] = 10
    s7, im7 = session_with(basin)
    s7.run_op("morpho.fill_holes", {})
    assert im7.data[0, 0, 0, 20, 20] == 100 and im7.data[0, 0, 0, 0, 0] == 10


def test_area_opening_and_directional_filter():
    img = np.full((50, 50), 20, np.uint8)
    img[10:30, 10:30] = 200   # 400 px square
    img[40, 40] = 255         # a bright speck
    img[5:7, 40:42] = 255     # 4 px speck
    img[35, 5] = 0            # a dark spot
    s, im = session_with(img)
    s.run_op("morpho.area_opening", {"operation": "Area Opening", "min_area": 10})
    p = im.data[0, 0, 0]
    assert p[40, 40] == 20 and p[5, 40] == 20 and p[20, 20] == 200 and p[35, 5] == 0
    s.run_op("morpho.area_opening", {"operation": "Area Closing", "min_area": 10, "connectivity": "8"})
    assert im.data[0, 0, 0, 35, 5] == 20
    # directional opening: a long thin line survives, isolated dots vanish
    img = np.zeros((60, 60), np.uint8)
    rr, cc = M.skm.disk(1).nonzero()
    img[30, 5:55] = 200
    img[10, 10] = 200
    img[45, 50] = 200
    s2, im2 = session_with(img)
    out = s2.run_op("morpho.directional_filter", {"type": "Max", "operation": "Opening", "line_length": 15, "n_directions": 16})
    p = im2.data[0, 0, 0]
    assert (p[30, 10:50] == 200).all() and p[10, 10] == 0 and p[45, 50] == 0
    s3, im3 = session_with(img)
    s3.run_op("morpho.directional_filter", {"type": "Max", "operation": "Mean", "line_length": 9, "n_directions": 8})
    assert im3.data[0, 0, 0, 30, 30] == 200 and 0 < im3.data[0, 0, 0, 10, 10] < 200
    s4, im4 = session_with(img)
    s4.run_op("morpho.directional_filter", {"type": "Min", "operation": "Median", "line_length": 5, "n_directions": 4})
    assert im4.data[0, 0, 0, 30, 30] == 0 or im4.data[0, 0, 0, 10, 10] == 0


# ------------------------------------------------------------------ overlays
def test_binary_and_label_overlays():
    ref = np.zeros((30, 40), np.uint8)
    ref[:, 20:] = 100
    lab = np.zeros((30, 40), np.uint16)
    lab[5:15, 5:15] = 1
    lab[20:28, 22:35] = 2
    s, im = session_with(ref, name="ref")
    s.add_image(LumenImage(lab, name="lab"))
    s.add_image(LumenImage((lab > 0).astype(np.uint8) * 255, name="mask"))
    s.select(im.id)
    out = s.run_op("morpho.binary_overlay", {"mask_image": "mask", "color": "red"})
    o = s.images[out["new_images"][0]]
    assert o.rgb and o.shape == (1, 1, 3, 30, 40)
    px = np.moveaxis(o.data[0, 0], 0, 2)
    assert np.array_equal(px[25, 25], [255, 0, 0]) and np.array_equal(px[0, 0], [0, 0, 0]) and np.array_equal(px[0, 30], [100, 100, 100])
    s.select(im.id)
    out = s.run_op("morpho.binary_overlay", {"mask_image": "mask", "color": "#0000ff", "opacity": 0.5})
    px = np.moveaxis(s.images[out["new_images"][0]].data[0, 0], 0, 2)
    assert np.array_equal(px[25, 25], [50, 50, 177]) or np.array_equal(px[25, 25], [50, 50, 178])
    s.select(im.id)
    out = s.run_op("morpho.label_overlay", {"label_image": "lab", "lut": "Glasbey", "opacity": 1.0})
    px = np.moveaxis(s.images[out["new_images"][0]].data[0, 0], 0, 2)
    assert "2 labels" in out["message"] and np.array_equal(px[0, 30], [100, 100, 100]) and not np.array_equal(px[25, 25], [100, 100, 100])
    assert not np.array_equal(px[lab == 1][0], px[lab == 2][0])
    # RGB reference image, LUT rendering of a 16-bit reference, whole stack
    rgb = np.zeros((30, 40, 3), np.uint8)
    rgb[..., 1] = 80
    s2, im2 = session_with(rgb, name="rgb", rgb=True)
    s2.add_image(LumenImage((lab > 0).astype(np.uint8) * 255, name="mask"))
    s2.select(im2.id)
    out = s2.run_op("morpho.binary_overlay", {"mask_image": "mask", "color": "yellow"})
    px = np.moveaxis(s2.images[out["new_images"][0]].data[0, 0], 0, 2)
    assert np.array_equal(px[25, 25], [255, 255, 0]) and np.array_equal(px[0, 0], [0, 80, 0])
    s3, im3 = session_with(np.stack([ref, ref]).astype(np.uint16) * 100, name="ref16", axes="ZYX")
    im3.display.channels[0].lut = "Red"
    im3.display.channels[0].min, im3.display.channels[0].max = 0.0, 10000.0
    s3.add_image(LumenImage(np.stack([lab, lab]), name="lab", axes="ZYX"))
    s3.select(im3.id)
    out = s3.run_op("morpho.label_overlay", {"label_image": "lab", "lut": "Golden angle", "opacity": 0.5, "all_planes": True})
    o = s3.images[out["new_images"][0]]
    assert o.shape == (1, 2, 3, 30, 40)
    px = np.moveaxis(o.data[0, 1], 0, 2)
    assert np.array_equal(px[0, 30], [255, 0, 0]) and np.array_equal(px[0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="same width"):
        s3.add_image(LumenImage(np.zeros((5, 5), np.uint8), name="small"))
        s3.select(im3.id)
        s3.run_op("morpho.binary_overlay", {"mask_image": "small"})
