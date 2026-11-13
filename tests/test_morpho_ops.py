"""Plugins ▸ MorphoLibJ (lumen/core/ops/morpho_ops.py): region analysis, intensity measurements, label tools,
region adjacency graph, label overlap measures and morphological segmentation."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

MORPHO_OPS = ["morpho.analyze_regions", "morpho.intensity_measurements", "morpho.keep_largest_label", "morpho.remove_largest_label",
              "morpho.replace_label", "morpho.crop_label", "morpho.labels_to_boundaries", "morpho.dilate_labels", "morpho.label_morphology",
              "morpho.region_adjacency_graph", "morpho.label_overlap_measures", "morpho.morphological_segmentation"]


def rotated_ellipse(shape=(200, 200), centre=(100, 100), a=40, b=20, angle_deg=30):
    """Filled ellipse with semi-axes a (major) / b, major axis at ``angle_deg`` counter-clockwise (y up, ImageJ convention)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    th = np.deg2rad(-angle_deg)  # image coordinates (y down)
    u = (xx - centre[1]) * np.cos(th) + (yy - centre[0]) * np.sin(th)
    v = -(xx - centre[1]) * np.sin(th) + (yy - centre[0]) * np.cos(th)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def shapes_label_image():
    """1 = rotated ellipse, 2 = disk r = 15, 3 = L shape (10 px wide arms, 40 tall × 60 wide), 4 = ring."""
    lab = np.zeros((200, 200), np.uint16)
    lab[rotated_ellipse()] = 1
    rr, cc = disk((40, 40), 15)
    lab[rr, cc] = 2
    lab[150:190, 20:30] = 3
    lab[180:190, 20:80] = 3
    rr, cc = disk((160, 150), 20)
    lab[rr, cc] = 4
    rr, cc = disk((160, 150), 10)
    lab[rr, cc] = 0
    return lab


def rows_by_label(s, name="lab"):
    return {r["Label"]: r for r in s.results.rows if r["Label"].startswith(name + ":")}


def test_morpho_ops_registered():
    for op_id in MORPHO_OPS:
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Plugins/MorphoLibJ"
        assert not spec.supports_rgb


# ------------------------------------------------------------------ Analyze Regions
def test_analyze_regions_geometry_of_synthetic_shapes():
    s = Session()
    im = s.add_image(LumenImage(shapes_label_image(), name="lab"))
    out = s.run_op("morpho.analyze_regions", {"overlays": ["Equivalent ellipses", "Oriented boxes", "Inscribed circles", "Geodesic paths"]})
    assert "4 regions" in out["message"] and "16 overlay items" in out["message"]
    rows = rows_by_label(s)
    ell, dsk, ell_l, ring = rows["lab:1"], rows["lab:2"], rows["lab:3"], rows["lab:4"]
    # equivalent ellipse: radii and orientation within 3 %
    assert ell["Ellipse.Radius1"] == pytest.approx(40, rel=0.03) and ell["Ellipse.Radius2"] == pytest.approx(20, rel=0.03)
    assert ell["Ellipse.Orientation"] == pytest.approx(30, rel=0.03) and ell["Ellipse.Elong"] == pytest.approx(2.0, rel=0.03)
    assert ell["Ellipse.Center.X"] == pytest.approx(100.5, abs=0.3) and ell["Ellipse.Center.Y"] == pytest.approx(100.5, abs=0.3)
    assert ell["Area"] == pytest.approx(np.pi * 40 * 20, rel=0.02) and ell["EulerNumber"] == 1 and 0.95 < ell["Convexity"] <= 1.0
    assert ell["MaxFeretDiam"] == pytest.approx(81, abs=1.5) and ell["MaxFeretAngle"] == pytest.approx(30, abs=3)
    assert ell["OrientedBox.Length"] == pytest.approx(81, abs=2) and ell["OrientedBox.Width"] == pytest.approx(41, abs=2)
    assert ell["OrientedBox.Orientation"] == pytest.approx(30, abs=5)
    assert ell["GeodesicDiameter"] > ell["MaxFeretDiam"] and 1.0 < ell["Tortuosity"] < 1.1
    # disk: inscribed circle radius = r, circularity ≈ 1, centroid at the pixel centre, thickness ≈ 2·r/3
    assert dsk["InscrCircle.Radius"] == pytest.approx(15, abs=0.5)
    assert dsk["InscrCircle.Center.X"] == pytest.approx(40.5, abs=0.5) and dsk["InscrCircle.Center.Y"] == pytest.approx(40.5, abs=0.5)
    assert dsk["Circularity"] == pytest.approx(1.0, abs=0.05) and dsk["Perimeter"] == pytest.approx(2 * np.pi * 15, rel=0.03)
    assert dsk["Centroid.X"] == pytest.approx(40.5, abs=0.05) and dsk["Centroid.Y"] == pytest.approx(40.5, abs=0.05)
    assert dsk["Ellipse.Radius1"] == pytest.approx(15, rel=0.03) and dsk["Ellipse.Elong"] == pytest.approx(1.0, abs=0.02)
    assert dsk["MaxFeretDiam"] == pytest.approx(31, abs=1) and dsk["AverageThickness"] == pytest.approx(10, abs=1.5)
    assert dsk["Box.X.Min"] == 26 and dsk["Box.X.Max"] == 55 and dsk["Box.Y.Min"] == 26 and dsk["Box.Y.Max"] == 55
    assert dsk["EulerNumber"] == 1
    # L shape: the geodesic diameter is longer than the Feret diameter (tortuosity > 1), oriented box 60 × 40 at 0°
    assert ell_l["Area"] == 900 and ell_l["GeodesicDiameter"] > ell_l["MaxFeretDiam"] and ell_l["Tortuosity"] > 1.1
    assert ell_l["MaxFeretDiam"] == pytest.approx(np.hypot(60, 40), abs=1)
    assert ell_l["OrientedBox.Length"] == pytest.approx(60, abs=0.5) and ell_l["OrientedBox.Width"] == pytest.approx(40, abs=0.5)
    assert ell_l["OrientedBox.Orientation"] in (pytest.approx(0, abs=0.5), pytest.approx(180, abs=0.5))
    assert ell_l["InscrCircle.Radius"] == pytest.approx(5.5, abs=1) and ell_l["Convexity"] == pytest.approx(900 / (2400 - 0.5 * 50 * 30), rel=0.05)
    assert ell_l["Box.X.Min"] == 20 and ell_l["Box.X.Max"] == 80 and ell_l["Box.Y.Min"] == 150 and ell_l["Box.Y.Max"] == 190
    assert ell_l["Circularity"] < 0.5
    # ring: one hole → Euler number 0; its inscribed circle fits into the ring band (width 10)
    assert ring["EulerNumber"] == 0 and ring["InscrCircle.Radius"] == pytest.approx(5, abs=1) and ring["Convexity"] < 0.8
    # overlays: one ellipse polygon, box polygon, inscribed-circle oval and geodesic polyline per label, pinned to the plane
    kinds = sorted(o["morpho"] for o in im.overlay)
    assert kinds == sorted(["Equivalent ellipses", "Oriented boxes", "Inscribed circles", "Geodesic paths"] * 4)
    assert {o["type"] for o in im.overlay} == {"polygon", "oval", "polyline"}
    assert all(o["position"] == {"t": 0, "z": 0, "c": 0} for o in im.overlay)
    circ = next(o for o in im.overlay if o["name"] == "Circle L2")
    assert circ["w"] == pytest.approx(30, abs=1) and circ["x"] + circ["w"] / 2 == pytest.approx(40.5, abs=0.5)
    ell_poly = np.asarray(next(o for o in im.overlay if o["name"] == "Ellipse L1")["points"])
    assert np.all(np.hypot(ell_poly[:, 0] - 100.5, ell_poly[:, 1] - 100.5) <= 40.5)
    assert np.hypot(ell_poly[:, 0] - 100.5, ell_poly[:, 1] - 100.5).max() == pytest.approx(40, abs=0.5)
    # running it again replaces the previous overlay items of the same kind instead of piling up
    old_ids = {o["id"] for o in im.overlay if o["morpho"] == "Inscribed circles"}
    s.run_op("morpho.analyze_regions", {"overlays": ["Inscribed circles"]})
    circles = [o for o in im.overlay if o["morpho"] == "Inscribed circles"]
    assert len(im.overlay) == 16 and len(circles) == 4 and not old_ids & {o["id"] for o in circles}
    # feature subset → only those columns
    s.run_op("morpho.analyze_regions", {"features": ["Area", "Centroid"]})
    row = s.results.rows[-1]
    assert set(row) == {"Label", "Area", "Centroid.X", "Centroid.Y"}


def test_analyze_regions_calibration_stack_and_crofton():
    lab = np.zeros((2, 100, 100), np.uint8)
    rr, cc = disk((50, 50), 20)
    lab[0][rr, cc] = 1
    lab[1][rr, cc] = 1
    lab[1, 10:20, 10:40] = 2
    s = Session()
    im = s.add_image(LumenImage(lab, name="cal", axes="ZYX"))
    im.calibration = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=2.0, unit="um")
    out = s.run_op("morpho.analyze_regions", {"all_planes": True, "crofton_directions": "8"})
    assert "um" in out["message"]
    rows = s.results.rows
    assert [(r["Label"], r["Slice"]) for r in rows] == [("cal:1", 1), ("cal:1", 2), ("cal:2", 2)]
    r = rows[0]
    assert r["Area"] == pytest.approx(np.pi * 400 * 0.25, rel=0.02) and r["Ellipse.Radius1"] == pytest.approx(10, rel=0.03)
    assert r["InscrCircle.Radius"] == pytest.approx(10, abs=0.3) and r["Centroid.X"] == pytest.approx(25.25, abs=0.05)
    assert r["Perimeter"] == pytest.approx(2 * np.pi * 10, rel=0.03) and r["GeodesicDiameter"] == pytest.approx(20, rel=0.12)
    rect = rows[2]
    assert rect["Area"] == pytest.approx(300 * 0.25) and rect["Box.X.Min"] == 5 and rect["Box.X.Max"] == 20
    assert rect["OrientedBox.Length"] == pytest.approx(15) and rect["OrientedBox.Width"] == pytest.approx(5)
    # a 0/255 mask is labelled into its components first
    mask = np.zeros((60, 60), np.uint8)
    mask[5:15, 5:15] = 255
    mask[30:50, 30:50] = 255
    s.add_image(LumenImage(mask, name="mask"))
    s.run_op("morpho.analyze_regions", {"features": ["Area", "Euler number"]})
    assert [r["Area"] for r in s.results.rows[-2:]] == [100, 400] and all(r["EulerNumber"] == 1 for r in s.results.rows[-2:])


# ------------------------------------------------------------------ Intensity measurements
def test_intensity_measurements_2d_and_3d():
    lab = np.zeros((3, 50, 50), np.uint8)
    lab[:, 5:15, 5:15] = 1   # 100 px per slice
    lab[:, 20:40, 20:45] = 2  # 500 px per slice
    lab[0, 45:48, 45:48] = 3  # only on slice 1
    rng = np.random.default_rng(0)
    inten = (lab.astype(np.float32) * 100 + rng.normal(0, 1, lab.shape)).astype(np.float32)
    s = Session()
    im = s.add_image(LumenImage(lab, name="lab", axes="ZYX"))
    im.calibration = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=2.0, unit="um")
    s.add_image(LumenImage(inten, name="inten", axes="ZYX"))
    s.select(im.id)
    out = s.run_op("morpho.intensity_measurements", {"intensity_image": "inten"})
    assert "3 labels" in out["message"]
    rows = rows_by_label(s)
    assert rows["lab:1"]["NumberOfVoxels"] == 100 and rows["lab:1"]["Volume"] == pytest.approx(25.0)
    assert rows["lab:2"]["Mean"] == pytest.approx(200, abs=0.5) and 0.7 < rows["lab:2"]["StdDev"] < 1.3
    assert rows["lab:2"]["Median"] == pytest.approx(200, abs=0.5) and abs(rows["lab:2"]["Skewness"]) < 0.5 and abs(rows["lab:2"]["Kurtosis"]) < 1
    assert rows["lab:3"]["NumberOfVoxels"] == 9 and rows["lab:1"]["Max"] > rows["lab:1"]["Min"]
    assert set(rows["lab:1"]) == {"Label", *["Mean", "StdDev", "Max", "Min", "Median", "Mode", "Skewness", "Kurtosis", "NumberOfVoxels", "Volume"]}
    out = s.run_op("morpho.intensity_measurements", {"intensity_image": "inten", "mode": "3D (whole stack)", "measurements": ["Mean", "NumberOfVoxels", "Volume"]})
    assert "(3D)" in out["message"]
    rows = rows_by_label(s)
    assert rows["lab:1"]["NumberOfVoxels"] == 300 and rows["lab:1"]["Volume"] == pytest.approx(300 * 0.25 * 2.0)
    assert rows["lab:3"]["NumberOfVoxels"] == 9 and set(rows["lab:2"]) == {"Label", "Mean", "NumberOfVoxels", "Volume"}
    # integer intensities: the mode is the most frequent value
    u8 = np.full((3, 50, 50), 7, np.uint8)
    u8[:, 5:15, 5:15] = 42
    s.add_image(LumenImage(u8, name="u8", axes="ZYX"))
    s.select(im.id)
    s.run_op("morpho.intensity_measurements", {"intensity_image": "u8"})
    assert rows_by_label(s)["lab:1"]["Mode"] == 42 and rows_by_label(s)["lab:2"]["Mode"] == 7
    with pytest.raises(ValueError, match="intensity image"):
        s.run_op("morpho.intensity_measurements", {})
    s.add_image(LumenImage(np.zeros((10, 10), np.uint8), name="small"))
    s.select(im.id)
    with pytest.raises(ValueError, match="same width"):
        s.run_op("morpho.intensity_measurements", {"intensity_image": "small"})


# ------------------------------------------------------------------ label tools
def three_labels():
    lab = np.zeros((2, 80, 80), np.uint16)
    lab[:, 10:30, 10:30] = 1          # 400 px, touches label 2
    lab[:, 10:30, 30:50] = 2          # 400 px
    lab[:, 50:70, 50:75] = 3          # 500 px: largest, isolated
    return lab


def test_keep_remove_replace_and_crop_labels():
    s = Session()
    im = s.add_image(LumenImage(three_labels(), name="lab", axes="ZYX"))
    s.run_op("morpho.keep_largest_label", {}, scope="plane")
    assert np.unique(im.data[0, 0, 0]).tolist() == [0, 3] and np.unique(im.data[0, 1, 0]).tolist() == [0, 1, 2, 3]
    s.undo(im.id)
    s.run_op("morpho.remove_largest_label", {})
    assert np.unique(im.data).tolist() == [0, 1, 2]
    s.undo(im.id)
    s.run_op("morpho.replace_label", {"labels": "2-3", "new_value": 7})
    assert np.unique(im.data).tolist() == [0, 1, 7] and (im.data == 7).sum() == 2 * 900
    s.undo(im.id)
    s.run_op("morpho.replace_label", {"labels": "1, 3", "new_value": 0})
    assert np.unique(im.data).tolist() == [0, 2]
    s.undo(im.id)
    # an empty list means the label picked with the value picker (the session's foreground value)
    s.run_op("edit.set_foreground", {"value": 0})
    with pytest.raises(ValueError, match="Pick a label"):
        s.run_op("morpho.replace_label", {"labels": "", "new_value": 0})
    s.run_op("edit.set_foreground", {"value": 2})
    s.run_op("morpho.replace_label", {"labels": "", "new_value": 0})
    assert np.unique(im.data).tolist() == [0, 1, 3]
    s.undo(im.id)
    out = s.run_op("morpho.crop_label", {"label": 3, "border": 2})
    crop = s.images[out["new_images"][0]]
    assert crop.shape == (1, 2, 1, 24, 29) and crop.lut_is_binary and (crop.data > 0).sum() == 1000
    assert "Label 3: 1000 voxels" in out["message"]
    s.select(im.id)
    with pytest.raises(ValueError, match="does not exist"):
        s.run_op("morpho.crop_label", {"label": 9})


def test_label_boundaries_dilation_and_morphology():
    s = Session()
    im = s.add_image(LumenImage(three_labels(), name="lab", axes="ZYX"))
    lab0 = three_labels()[0]
    out = s.run_op("morpho.labels_to_boundaries", {"mode": "inner"})
    b = s.images[out["new_images"][0]]
    inner = b.data[0, 0, 0] > 0
    assert b.shape == (1, 1, 1, 80, 80) and b.lut_is_binary
    assert inner[lab0 == 0].sum() == 0 and inner[10, 10] and inner[10:30, 29].all() and inner[10:30, 30].all() and not inner[20, 20]
    s.select(im.id)
    out = s.run_op("morpho.labels_to_boundaries", {"mode": "outer", "connectivity": "4", "all_planes": True})
    b = s.images[out["new_images"][0]]
    outer = b.data[0, 1, 0] > 0
    assert b.shape == (1, 2, 1, 80, 80) and outer[49, 50:75].all() and outer[50:70, 49].all() and not outer[55, 60]
    s.select(im.id)
    out = s.run_op("morpho.labels_to_boundaries", {"mode": "thick"})
    thick = s.images[out["new_images"][0]].data[0, 0, 0] > 0
    assert thick.sum() > inner.sum()
    # constrained dilation: labels grow without overlapping, optionally only inside a mask
    s.select(im.id)
    s.run_op("morpho.dilate_labels", {"radius": 3})
    grown = im.data[0, 0, 0]
    assert (grown > 0).sum() > (lab0 > 0).sum() and set(np.unique(grown).tolist()) == {0, 1, 2, 3}
    assert np.array_equal(grown[lab0 > 0], lab0[lab0 > 0]) and (grown[7:10, 10:30] == 1).all() and (grown[30:33, 32:48] == 2).all()
    assert np.array_equal(im.data[0, 1, 0], lab0)  # current plane only
    s.undo(im.id)
    assert np.array_equal(im.data[0, 0, 0], lab0)
    mask = np.zeros((80, 80), np.uint8)
    mask[:, :40] = 255
    s.add_image(LumenImage(mask, name="mask"))
    s.select(im.id)
    s.run_op("morpho.dilate_labels", {"radius": 5, "mask_image": "mask", "all_planes": True})
    grown = im.data[0, 1, 0]
    assert ((grown > 0) & (lab0 == 0) & (mask == 0)).sum() == 0 and (grown[5:10, 10:30] == 1).all() and (grown[70:75, 50:75] == 0).all()
    s.undo(im.id)
    # label morphological filters
    s.run_op("morpho.label_morphology", {"operation": "Erosion", "radius": 2}, scope="plane")
    eroded = im.data[0, 0, 0]
    assert (eroded == 1).sum() == 16 * 16 and (eroded == 3).sum() == 16 * 21 and (eroded[12:28, 12:28] == 1).all()
    s.undo(im.id)
    s.run_op("morpho.label_morphology", {"operation": "Dilation", "radius": 2}, scope="plane")
    dil = im.data[0, 0, 0]
    assert (dil == 3).sum() == 24 * 29 - 12  # 2 px Euclidean ring (the 3 outermost corner pixels are farther than 2)
    assert (dil == 1).sum() == (dil == 2).sum() > 400 and (dil[8:10, 10:30] == 1).all() and (dil[10:30, 8:10] == 1).all()
    assert np.array_equal(dil[lab0 > 0], lab0[lab0 > 0]) and (dil[10:30, 30:32] == 2).all()
    s.undo(im.id)
    for op_name in ("Opening", "Closing"):
        s.run_op("morpho.label_morphology", {"operation": op_name, "radius": 1}, scope="plane")
        assert set(np.unique(im.data[0, 0, 0]).tolist()) == {0, 1, 2, 3}
        s.undo(im.id)


def test_region_adjacency_graph():
    s = Session()
    im = s.add_image(LumenImage(three_labels(), name="lab", axes="ZYX"))
    out = s.run_op("morpho.region_adjacency_graph", {})
    assert "1 adjacent label pair" in out["message"]
    tab = s.tables["Region adjacency graph"]
    assert tab["columns"] == ["Label 1", "Label 2", "Adjacent pairs"] and tab["rows"] == [[1, 2, 20]]
    assert s.results.rows[-1]["Label 1"] == 1 and s.results.rows[-1]["Label 2"] == 2 and s.results.rows[-1]["Adjacent pairs"] == 20
    lines = [o for o in im.overlay if o["morpho"] == "rag"]
    assert len(lines) == 1 and lines[0]["type"] == "line"
    assert lines[0]["x1"] == pytest.approx(20, abs=0.01) and lines[0]["y1"] == pytest.approx(20, abs=0.01) and lines[0]["x2"] == pytest.approx(40, abs=0.01)
    # diagonal contact: only with 8-connectivity; stacking contact: only in 3D
    lab = np.zeros((2, 40, 40), np.uint8)
    lab[0, 5:15, 5:15] = 1
    lab[0, 15:24, 15:24] = 2  # touches label 1 at one corner only
    lab[1, 25:35, 25:35] = 3
    lab[0, 25:35, 25:35] = 4
    s.add_image(LumenImage(lab, name="diag", axes="ZYX"))
    out = s.run_op("morpho.region_adjacency_graph", {"connectivity": "4", "show_overlay": False})
    assert "0 adjacent" in out["message"] and s.tables["Region adjacency graph"]["rows"] == []
    out = s.run_op("morpho.region_adjacency_graph", {"connectivity": "8"})
    assert s.tables["Region adjacency graph"]["rows"] == [[1, 2, 1]]
    out = s.run_op("morpho.region_adjacency_graph", {"mode": "3D (whole stack)", "connectivity": "4"})
    assert s.tables["Region adjacency graph"]["rows"] == [[3, 4, 100]]
    out = s.run_op("morpho.region_adjacency_graph", {"mode": "3D (whole stack)", "connectivity": "8"})
    assert s.tables["Region adjacency graph"]["rows"] == [[1, 2, 1], [3, 4, 28 * 28]]  # 26-connected pairs between two stacked 10×10 squares


def test_label_overlap_measures():
    s = Session()
    lab = three_labels()
    im = s.add_image(LumenImage(lab, name="lab", axes="ZYX"))
    same = s.add_image(LumenImage(lab.copy(), name="same", axes="ZYX"))
    s.select(im.id)
    out = s.run_op("morpho.label_overlap_measures", {"reference_image": "same"})
    rows = s.results.rows
    assert len(rows) == 4 and all(r["Jaccard"] == 1.0 and r["Dice"] == 1.0 and r["Volume similarity"] == 0.0 for r in rows)
    assert rows[-1]["Label"] == "lab vs same (all labels)" and rows[-1]["Overlap"] == 1300
    assert s.tables["Label overlap"]["columns"][:4] == ["Source label", "Target label", "Overlap", "Jaccard"]
    assert "mean Jaccard 1.000" in out["message"]
    # shifted target: known overlap; label 3 missing in the target → false positive error 1
    shifted = np.zeros_like(lab)
    shifted[:, 10:30, 15:35] = 1   # overlap with 1: 20 × 15 = 300 of 400 / 400 → Jaccard 300/500
    shifted[:, 10:30, 35:55] = 2   # overlap with 2: 20 × 15 = 300
    s.add_image(LumenImage(shifted, name="shifted", axes="ZYX"))
    s.select(im.id)
    out = s.run_op("morpho.label_overlap_measures", {"reference_image": "shifted"})
    r1, r2, r3, tot = s.results.rows[-4:]
    assert r1["Overlap"] == 300 and r1["Jaccard"] == pytest.approx(0.6) and r1["Dice"] == pytest.approx(0.75)
    assert r1["Target overlap"] == pytest.approx(0.75) and r1["False negative error"] == pytest.approx(0.25) and r1["False positive error"] == pytest.approx(0.25)
    assert r3["Target label"] == 3 and r3["Overlap"] == 0 and r3["Jaccard"] == 0 and r3["False positive error"] == 1.0
    assert tot["Overlap"] == 600 + 100  # label 2's pixels 30:35 also overlap shifted label 1 at the foreground level
    # best-overlap matching pairs permuted labels
    perm = np.where(lab == 1, 2, np.where(lab == 2, 1, lab))
    s.add_image(LumenImage(perm, name="perm", axes="ZYX"))
    s.select(im.id)
    s.run_op("morpho.label_overlap_measures", {"reference_image": "perm", "matching": "Same label value"})
    assert [r["Jaccard"] for r in s.results.rows[-4:-1]] == [0.0, 0.0, 1.0]
    out = s.run_op("morpho.label_overlap_measures", {"reference_image": "perm", "matching": "Best overlap", "mode": "3D (whole stack)"})
    rows = s.results.rows[-4:]
    assert [(r["Source label"], r["Target label"], r["Jaccard"]) for r in rows[:-1]] == [(1, 2, 1.0), (2, 1, 1.0), (3, 3, 1.0)]
    assert rows[-1]["Overlap"] == 2 * 1300 and "mean Jaccard 1.000" in out["message"]
    with pytest.raises(ValueError, match="target label image"):
        s.run_op("morpho.label_overlap_measures", {})


# ------------------------------------------------------------------ Morphological Segmentation
def two_touching_blobs():
    """Two touching disks of different brightness (a gradient ridge at their interface) on a noisy background."""
    img = np.zeros((120, 160), np.float32)
    rr, cc = disk((60, 60), 30)
    img[rr, cc] = 150
    rr, cc = disk((60, 105), 30)
    img[rr, cc] = 220
    img = ndi.gaussian_filter(img, 1.5) + np.random.default_rng(0).normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def basin_centroids(lab):
    return {int(k): ndi.center_of_mass(lab == k) for k in np.unique(lab) if k > 0}


def test_morphological_segmentation_splits_two_touching_blobs():
    s = Session()
    im = s.add_image(LumenImage(two_touching_blobs(), name="blobs"))
    out = s.run_op("morpho.morphological_segmentation", {"tolerance": 20, "output": "Catchment basins"})
    assert "3 catchment basins" in out["message"]
    basins = s.images[out["new_images"][0]]
    lab = basins.data[0, 0, 0]
    assert basins.display.channels[0].lut == "Glasbey on dark" and (lab == 0).sum() > 0  # dams between the basins
    cents = basin_centroids(lab)
    sizes = {k: int((lab == k).sum()) for k in cents}
    objects = sorted((k for k in cents if sizes[k] < 5000), key=lambda k: cents[k][1])
    assert len(objects) == 2
    assert cents[objects[0]] == pytest.approx((60, 60), abs=4) and cents[objects[1]] == pytest.approx((60, 105), abs=4)
    assert 2000 < sizes[objects[0]] < 3200 and 2000 < sizes[objects[1]] < 3200
    # watershed lines: a binary mask of the dams; overlaid outputs: RGB images of the same size
    s.select(im.id)
    out = s.run_op("morpho.morphological_segmentation", {"tolerance": 20, "output": "Watershed lines"})
    wl = s.images[out["new_images"][0]]
    assert wl.lut_is_binary and np.array_equal(wl.data[0, 0, 0] > 0, lab == 0)
    for output in ("Overlaid basins", "Overlaid dams"):
        s.select(im.id)
        out = s.run_op("morpho.morphological_segmentation", {"tolerance": 20, "output": output, "connectivity": "8", "gradient": "Sobel"})
        rgb = s.images[out["new_images"][0]]
        assert rgb.rgb and rgb.shape == (1, 1, 3, 120, 160) and rgb.dtype == np.uint8
        if output == "Overlaid dams":
            dams = (rgb.data[0, 0, 0] == 255) & (rgb.data[0, 0, 1] == 0) & (rgb.data[0, 0, 2] == 0)
            assert 100 < dams.sum() < 2000
    # no tolerance: every regional minimum of the noisy gradient is a basin → heavy over-segmentation
    s.select(im.id)
    out = s.run_op("morpho.morphological_segmentation", {"tolerance": 0, "output": "Catchment basins", "calculate_dams": False})
    assert int(out["message"].split()[0]) > 100
    assert (s.images[out["new_images"][0]].data == 0).sum() == 0


def test_morphological_segmentation_gradient_image_input_and_stack():
    # the classic split of touching objects: the inverted distance map of the mask as the 'gradient' image
    mask = np.zeros((100, 140), bool)
    rr, cc = disk((50, 50), 28)
    mask[rr, cc] = True
    rr, cc = disk((50, 90), 28)
    mask[rr, cc] = True
    edt = ndi.distance_transform_edt(mask).astype(np.float32)
    inv = (edt.max() - edt) * (mask > 0) + edt.max() * (~mask)
    s = Session()
    im = s.add_image(LumenImage(np.stack([inv, inv]), name="inv", axes="ZYX"))
    out = s.run_op("morpho.morphological_segmentation", {"input": "Gradient image", "tolerance": 2, "output": "Catchment basins", "all_planes": True})
    basins = s.images[out["new_images"][0]]
    assert basins.shape == (1, 2, 1, 100, 140) and "4 catchment basins" in out["message"]  # 2 per plane
    lab = basins.data[0, 1, 0]
    cents = basin_centroids(lab)
    assert len(cents) == 2
    inside = {k: ndi.center_of_mass(mask & (lab == k)) for k in cents}
    assert sorted(v[1] for v in inside.values()) == pytest.approx([50, 90], abs=4)
    assert (lab[mask] == 0).sum() < 60  # only the dam crosses the mask
    assert np.array_equal(basins.data[0, 0, 0], basins.data[0, 1, 0])
