import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import circle_perimeter, line

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.session import Session

SQ2 = float(np.sqrt(2.0))


# ------------------------------------------------------------------ skeleton
def y_shape(size=100):
    """Three 1-px arms meeting at (y, x) = (50, 50): lengths 40 (horizontal) and 30√2 (two diagonals)."""
    img = np.zeros((size, size), np.uint8)
    for (y1, x1) in [(50, 10), (20, 80), (80, 80)]:
        rr, cc = line(50, 50, y1, x1)
        img[rr, cc] = 255
    return img


def ring(size=100, r=30):
    img = np.zeros((size, size), np.uint8)
    rr, cc = circle_perimeter(50, 50, r)
    img[rr, cc] = 255
    return img


def branch_rows(s):
    tab = s.tables["Branches"]
    return [dict(zip(tab["columns"], r)) for r in tab["rows"]]


def test_skeleton_y_shape():
    s = Session()
    s.add_image(LumenImage(y_shape(), name="Y"))
    out = s.run_op("analyze.skeleton", {"skeletonize_first": False})
    assert "3 branches" in out["message"] and "1 junctions" in out["message"] and "3 end points" in out["message"]
    row = s.results.rows[-1]
    assert row["Label"] == "Y#1" and row["Skeleton ID"] == 1
    assert row["# Branches"] == 3 and row["# Junctions"] == 1 and row["# End-point voxels"] == 3
    assert row["# Junction voxels"] == 1 and row["# Triple points"] == 1 and row["# Quadruple points"] == 0
    assert row["# Slab voxels"] == int((y_shape() > 0).sum()) - 4
    expected = sorted([40.0, 30 * SQ2, 30 * SQ2])
    br = branch_rows(s)
    assert s.tables["Branches"]["columns"] == ["Skeleton ID", "Branch", "Length", "Euclidean distance", "V1 x", "V1 y", "V2 x", "V2 y"]
    assert [r["Branch"] for r in br] == [1, 2, 3] and all(r["Skeleton ID"] == 1 for r in br)
    assert np.allclose(sorted(r["Length"] for r in br), expected, rtol=0.05)
    assert row["Maximum Branch Length"] == pytest.approx(30 * SQ2, rel=0.05)
    assert row["Average Branch Length"] == pytest.approx(np.mean(expected), rel=0.05)
    assert row["Longest Shortest Path"] == pytest.approx(60 * SQ2, rel=0.05)  # between the two diagonal end points
    for r in br:  # straight branches: Euclidean distance equals the length; every branch touches the junction
        assert r["Euclidean distance"] == pytest.approx(r["Length"], rel=0.05)
        assert (r["V1 x"], r["V1 y"]) == (50, 50) or (r["V2 x"], r["V2 y"]) == (50, 50)
    # tagged skeleton: end = 1, slab = 2, junction = 3 as a Glasbey label image; the source stays selected
    tag = s.images[out["new_images"][0]]
    assert tag.display.channels[0].lut == "Glasbey on dark" and tag.dtype == np.uint8 and s.active.name == "Y"
    assert (tag.data == 1).sum() == 3 and (tag.data == 3).sum() == 1 and (tag.data == 2).sum() == row["# Slab voxels"]
    assert set(np.unique(tag.data).tolist()) == {0, 1, 2, 3}


def test_skeleton_thick_input_is_skeletonized_and_calibrated():
    thick = ndi.binary_dilation(y_shape() > 0, iterations=2).astype(np.uint8) * 255
    s = Session()
    img = s.add_image(LumenImage(thick, name="thick"))
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "µm"
    out = s.run_op("analyze.skeleton", {"show": "Longest branch mask"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 3 and row["# Junctions"] == 1 and row["# End-point voxels"] == 3
    assert np.allclose(sorted(r["Length"] for r in branch_rows(s)), sorted([20.0, 15 * SQ2, 15 * SQ2]), rtol=0.05)
    assert "µm" in out["message"]
    mask = s.images[out["new_images"][0]]
    assert mask.lut_is_binary and set(np.unique(mask.data).tolist()) <= {0, 255} and mask.calibration.unit == "µm"
    assert 25 <= (mask.data > 0).sum() <= 35  # the longest (diagonal, ~30 px) arm only


def test_skeleton_ring_is_one_circular_branch():
    s = Session()
    s.add_image(LumenImage(ring(), name="ring"))
    out = s.run_op("analyze.skeleton", {})
    row = s.results.rows[-1]
    assert row["# Branches"] == 1 and row["# Junctions"] == 0 and row["# End-point voxels"] == 0 and row["# Junction voxels"] == 0
    tag = s.images[out["new_images"][0]]
    assert set(np.unique(tag.data).tolist()) == {0, 2} and (tag.data == 2).sum() == row["# Slab voxels"]
    br = branch_rows(s)
    assert len(br) == 1 and br[0]["Euclidean distance"] == 0.0
    assert 2 * np.pi * 30 * 0.95 < br[0]["Length"] < 2 * np.pi * 30 * 1.1  # 8-connected circle: slightly longer than 2πr
    assert row["Longest Shortest Path"] == 0.0
    # pruning cycles removes the ring: its only branch is the cycle
    out = s.run_op("analyze.skeleton", {"prune_cycles": "shortest branch", "show": "Nothing"})
    assert "0 skeletons" in out["message"] and "1 cycle branch pruned" in out["message"] and not out["new_images"]


def test_skeleton_prune_cycles_keeps_the_tail_of_a_lollipop():
    lol = ring()
    rr, cc = line(50, 80, 50, 98)
    lol[rr, cc] = 255
    s = Session()
    s.add_image(LumenImage(lol, name="lollipop"))
    s.run_op("analyze.skeleton", {"show": "Nothing"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 2 and row["# Junctions"] == 1 and row["# End-point voxels"] == 1
    lengths = sorted(r["Length"] for r in branch_rows(s))
    assert lengths[0] == pytest.approx(17, abs=2) and lengths[1] > 150  # tail, loop
    out = s.run_op("analyze.skeleton", {"prune_cycles": "shortest branch", "show": "Nothing"})
    row = s.results.rows[-1]
    assert "1 cycle branch pruned" in out["message"]
    assert row["# Branches"] == 1 and row["# Junctions"] == 0 and row["# End-point voxels"] == 2
    assert row["Maximum Branch Length"] == pytest.approx(18, abs=2)


def test_skeleton_prune_short_end_branches():
    spur = y_shape()
    rr, cc = line(50, 30, 44, 30)  # a 6-px spur off the horizontal arm
    spur[rr, cc] = 255
    s = Session()
    s.add_image(LumenImage(spur, name="spur"))
    s.run_op("analyze.skeleton", {"skeletonize_first": False, "show": "Nothing"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 5 and row["# Junctions"] == 2 and row["# End-point voxels"] == 4 and row["# Triple points"] == 2
    out = s.run_op("analyze.skeleton", {"skeletonize_first": False, "prune_ends": 6, "show": "Tagged skeleton"})
    row = s.results.rows[-1]
    assert "1 end branch pruned" in out["message"]
    assert row["# Branches"] == 3 and row["# Junctions"] == 1 and row["# End-point voxels"] == 3 and row["# Junction voxels"] == 1
    assert np.allclose(sorted(r["Length"] for r in branch_rows(s)), sorted([40.0, 30 * SQ2, 30 * SQ2]), rtol=0.05)
    tag = s.images[out["new_images"][0]]
    assert (tag.data == 3).sum() == 1 and (tag.data == 1).sum() == 3
    # a limit below the spur length keeps it
    out = s.run_op("analyze.skeleton", {"skeletonize_first": False, "prune_ends": 3, "show": "Nothing"})
    assert s.results.rows[-1]["# Branches"] == 5 and "pruned" not in out["message"]


def test_skeleton_all_planes_and_selection():
    s = Session()
    s.add_image(LumenImage(np.stack([y_shape(), ring()]), name="stack", axes="ZYX"))
    out = s.run_op("analyze.skeleton", {"skeletonize_first": False, "all_planes": True})
    rows = s.results.rows
    assert "2 skeletons" in out["message"]
    assert [(r["Frame"], r["Slice"], r["Ch"]) for r in rows] == [(1, 1, 1), (1, 2, 1)]
    assert rows[0]["# Branches"] == 3 and rows[1]["# Branches"] == 1
    assert s.tables["Branches"]["columns"][-3:] == ["Ch", "Slice", "Frame"]
    assert [r["Slice"] for r in branch_rows(s)] == [1, 1, 1, 2]
    tag = s.images[out["new_images"][0]]
    assert tag.shape == (1, 2, 1, 100, 100) and (tag.data[0, 1] == 2).sum() == rows[1]["# Slab voxels"]
    # a rectangle around the horizontal arm only: one line, two end points, no junction
    s.select(s.images[[i for i in s.images if i != tag.id][0]].id)
    s.run_op("analyze.skeleton", {"skeletonize_first": False, "show": "Nothing"}, roi={"type": "rect", "x": 0, "y": 40, "w": 45, "h": 20})
    row = s.results.rows[-1]
    assert row["# Branches"] == 1 and row["# Junctions"] == 0 and row["# End-point voxels"] == 2
    assert row["Maximum Branch Length"] == pytest.approx(34, abs=1)  # pixels x = 10 … 44


def test_skeleton_empty_plane():
    s = Session()
    s.add_image(LumenImage(np.zeros((32, 32), np.uint8), name="empty"))
    out = s.run_op("analyze.skeleton", {"show": "Nothing"})
    assert "0 skeletons" in out["message"] and not s.results.rows and s.tables["Branches"]["rows"] == []


# ------------------------------------------------------------------ Costes colocalization
def blobs(seed, n=150, size=256, sigma=3.0):
    r = np.random.default_rng(seed)
    f = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        x, y = r.uniform(0, size, 2)
        f += r.uniform(50, 200) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return f


def test_costes_perfectly_correlated_channels():
    rng = np.random.default_rng(0)
    a = blobs(1) + 20 + rng.normal(0, 3, (256, 256))
    b = 0.5 * a + 10
    s = Session()
    s.add_image(LumenImage(np.stack([a, b]).astype(np.float32), name="pair", axes="CYX"))
    out = s.run_op("analyze.coloc_costes", {"seed": 1, "scramble_iterations": 50})
    row = s.results.rows[-1]
    assert row["Pearson"] == pytest.approx(1.0, abs=1e-5) and row["Spearman"] == pytest.approx(1.0, abs=1e-5)
    assert row["Pearson (above thresholds)"] == pytest.approx(1.0, abs=1e-5)
    assert row["Regression slope"] == pytest.approx(0.5, rel=1e-3) and row["Regression intercept"] == pytest.approx(10.0, abs=0.05)
    # r of the pixels below the thresholds never drops to 0: the thresholds end up at the bottom of the range
    a32 = a.astype(np.float32)
    assert row["Threshold A"] <= a32.min() + (a32.max() - a32.min()) / 256 + 1e-6
    assert row["Threshold B"] == pytest.approx(0.5 * row["Threshold A"] + 10, rel=1e-3)
    assert row["Manders tM1"] > 0.99 and row["Manders tM2"] > 0.99 and row["% pixels above thresholds"] > 99
    assert row["Costes p-value"] == 0.0 and row["Scrambles"] == 50 and row["Block size"] == 5
    assert abs(row["Scrambled r mean"]) < 0.05 and 0 < row["Scrambled r SD"] < 0.05
    assert "p = 0.000" in out["message"] and "tM1 = 1.000" in out["message"]
    # the scatter image: 256 × 256 log counts of A vs B with the Fire LUT; the source image stays selected
    sc = s.images[out["new_images"][0]]
    assert sc.shape == (1, 1, 1, 256, 256) and sc.dtype == np.float32 and sc.display.channels[0].lut == "Fire"
    assert s.active.name == "pair" and sc.metadata["scatter"] and sc.metadata["log_counts"]
    assert sc.metadata["x_range"] == pytest.approx([a32.min(), a32.max()], rel=1e-5)
    assert np.expm1(sc.data.astype(np.float64)).sum() == pytest.approx(256 * 256, rel=1e-4)  # every pixel counted once
    # a linear relation puts all counts on one diagonal of the (B upwards, A rightwards) plot
    yy, xx = np.nonzero(sc.data[0, 0, 0])
    assert np.corrcoef(xx, 255 - yy)[0, 1] > 0.999


def smooth_noise_pair(seed, size=256):
    """Two independent, spatially smooth noise fields."""
    rng = np.random.default_rng(seed)
    a = 50 + 80 * ndi.gaussian_filter(rng.normal(0, 1, (size, size)), 2)
    b = 50 + 80 * ndi.gaussian_filter(rng.normal(0, 1, (size, size)), 2)
    return a.astype(np.float32), b.astype(np.float32)


def test_costes_independent_channels():
    # seed 0: r slightly negative. Costes presupposes a positive correlation: the thresholds go to the
    # channel maxima (nothing colocalized), tM1 = tM2 = 0, and the scrambling test is not significant
    a, b = smooth_noise_pair(0)
    s = Session()
    s.add_image(LumenImage(np.stack([a, b]), name="indep", axes="CYX"))
    out = s.run_op("analyze.coloc_costes", {"seed": 1, "scramble_iterations": 100, "show_scatter": False})
    row = s.results.rows[-1]
    assert -0.05 < row["Pearson"] < 0 and abs(row["Spearman"]) < 0.05 and row["Regression slope"] < 0
    assert row["Manders tM1"] == 0.0 and row["Manders tM2"] == 0.0 and row["% pixels above thresholds"] < 0.01
    assert row["Threshold A"] == pytest.approx(float(a.max())) and row["Threshold B"] == pytest.approx(float(b.max()))
    assert row["Costes p-value"] > 0.5 and row["Scrambles"] == 100 and abs(row["Scrambled r mean"]) < 0.02
    assert "no positive correlation" in out["message"] and not out["new_images"]
    # seed 3: r slightly positive (0.015): the thresholds stay high, little is above them, p is not significant
    a, b = smooth_noise_pair(3)
    s.add_image(LumenImage(np.stack([a, b]), name="indep2", axes="CYX"))
    out = s.run_op("analyze.coloc_costes", {"seed": 1, "scramble_iterations": 100, "show_scatter": False})
    row = s.results.rows[-1]
    assert 0 < row["Pearson"] < 0.05 and row["Regression slope"] > 0
    assert row["Manders tM1"] < 0.3 and row["Manders tM2"] < 0.3 and row["% pixels above thresholds"] < 5
    assert row["Threshold B"] <= float(b.max()) and row["Costes p-value"] > 0.05
    assert "no positive correlation" not in out["message"]


def test_costes_selection_8bit_and_errors():
    rng = np.random.default_rng(2)
    a = blobs(3) + 20 + rng.normal(0, 3, (256, 256))
    u8 = np.clip(np.stack([a, a]), 0, 255).astype(np.uint8)
    s = Session()
    s.add_image(LumenImage(u8, name="u8", axes="CYX"))
    out = s.run_op("analyze.coloc_costes", {"seed": 3, "scramble_iterations": 30, "block_size": 8},
                   roi={"type": "oval", "x": 40, "y": 30, "w": 150, "h": 120})
    row = s.results.rows[-1]
    assert row["Pearson"] == pytest.approx(1.0) and row["Costes p-value"] == 0.0 and abs(row["Scrambled r mean"]) < 0.1
    assert row["Manders tM1"] > 0.99 and row["Block size"] == 8
    assert "Scatter A vs B" in s.images[out["new_images"][0]].name
    # no randomisation
    out = s.run_op("analyze.coloc_costes", {"scramble_iterations": 0, "show_scatter": False})
    assert np.isnan(s.results.rows[-1]["Costes p-value"]) and s.results.rows[-1]["Scrambles"] == 0 and "p =" not in out["message"]
    with pytest.raises(ValueError, match="block size"):
        s.run_op("analyze.coloc_costes", {"block_size": 200, "scramble_iterations": 5}, roi={"type": "rect", "x": 0, "y": 0, "w": 100, "h": 100})
    with pytest.raises(ValueError, match="does not exist"):
        s.run_op("analyze.coloc_costes", {"channel_b": 3})
    s.add_image(LumenImage(u8[0], name="one"))
    with pytest.raises(ValueError, match="two channels"):
        s.run_op("analyze.coloc_costes", {})


def test_costes_is_fast_on_512():
    import time
    rng = np.random.default_rng(0)
    a = blobs(1, n=300, size=512) + 20 + rng.normal(0, 3, (512, 512))
    b = blobs(2, n=300, size=512) + 20 + rng.normal(0, 3, (512, 512))
    s = Session()
    s.add_image(LumenImage(np.stack([a, b]).astype(np.float32), name="big", axes="CYX"))
    s.run_op("analyze.coloc_costes", {"scramble_iterations": 10})  # warm up
    t0 = time.time()
    s.run_op("analyze.coloc_costes", {"scramble_iterations": 100})
    assert time.time() - t0 < 1.0


# ------------------------------------------------------------------ skeleton 3D / longest shortest path
def y_volume(shape=(40, 60, 60)):
    """Three 1-voxel arms meeting at (z, y, x) = (20, 30, 30): one along x (25 voxels), two oblique in z, y and x."""
    from skimage.draw import line_nd
    v = np.zeros(shape, np.uint8)
    for end in [(20, 30, 5), (5, 5, 55), (35, 55, 55)]:
        v[line_nd((20, 30, 30), end, endpoint=True)] = 255
    return v


def test_skeleton_3d_y_shape_counts_lengths_and_tags():
    s = Session()
    im = s.add_image(LumenImage(y_volume(), name="Y3", axes="ZYX"))
    im.calibration = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=2.0, unit="um")
    out = s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False})
    assert out["message"] == "1 skeleton (3D): 3 branches, 1 junctions, 3 end points (lengths in um)"
    row = s.results.rows[-1]
    assert row["Label"] == "Y3#1" and row["# Branches"] == 3 and row["# Junctions"] == 1 and row["# End-point voxels"] == 3
    assert row["# Junction voxels"] == 1 and row["# Triple points"] == 1 and row["# Quadruple points"] == 0
    assert row["# Slab voxels"] == int((y_volume() > 0).sum()) - 4
    assert s.tables["Branches"]["columns"] == ["Skeleton ID", "Branch", "Length", "Euclidean distance", "V1 x", "V1 y", "V1 z", "V2 x", "V2 y", "V2 z"]
    br = branch_rows(s)
    assert [r["Branch"] for r in br] == [1, 2, 3]
    diag = float(np.sqrt((15 * 2.0) ** 2 + (25 * 0.5) ** 2 + (25 * 0.5) ** 2))  # anisotropic z: 2 um per slice
    assert sorted(r["Euclidean distance"] for r in br) == pytest.approx(sorted([12.5, diag, diag]))
    for r in br:
        assert r["Euclidean distance"] <= r["Length"] <= 1.2 * r["Euclidean distance"]
    # every branch runs from an arm end (V1) to the junction (V2) at (x, y, z) = (30, 30, 20) voxels = (15, 15, 40) um
    assert all(r["V2 x"] == pytest.approx(15.0) and r["V2 y"] == pytest.approx(15.0) and r["V2 z"] == pytest.approx(40.0) for r in br)
    assert sorted(r["V1 z"] for r in br) == pytest.approx([10.0, 40.0, 70.0])
    assert row["Maximum Branch Length"] == pytest.approx(max(r["Length"] for r in br))
    assert row["Longest Shortest Path"] == pytest.approx(sum(sorted(r["Length"] for r in br)[1:]))  # the two oblique arms
    tag = s.images[out["new_images"][0]]
    assert tag.shape == (1, 40, 1, 60, 60) and tag.calibration.pixel_depth == 2.0 and tag.display.channels[0].lut == "Glasbey on dark"
    assert (tag.data == 1).sum() == 3 and (tag.data == 3).sum() == 1 and (tag.data == 2).sum() == row["# Slab voxels"]
    assert tag.data[0, 20, 0, 30, 30] == 3 and tag.data[0, 5, 0, 5, 55] == 1 and tag.data[0, 20, 0, 30, 5] == 1
    assert s.active.name == "Y3"


def test_skeleton_3d_thick_input_all_planes_and_longest_shortest_path():
    thick = ndi.binary_dilation(y_volume() > 0, iterations=2).astype(np.uint8) * 255
    s = Session()
    im = s.add_image(LumenImage(np.stack([thick, thick]), name="thick", axes="CZYX"))
    out = s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "all_planes": True, "show": "Longest branch mask"})
    assert "2 skeletons (3D): 6 branches, 2 junctions, 6 end points" in out["message"]
    rows = s.results.rows[-2:]
    assert [(r["Ch"], r["Frame"]) for r in rows] == [(1, 1), (2, 1)] and all(r["# Junctions"] == 1 and r["# End-point voxels"] == 3 for r in rows)
    assert s.tables["Branches"]["columns"][-2:] == ["Ch", "Frame"] and len(branch_rows(s)) == 6
    mask = s.images[out["new_images"][0]]
    assert mask.shape == (1, 40, 2, 60, 60) and mask.lut_is_binary
    assert 25 <= (mask.data[0, :, 0] > 0).sum() <= 45 and np.array_equal(mask.data[0, :, 0], mask.data[0, :, 1])
    assert s.active.name == "thick"
    # the longest shortest path: both oblique arms through the junction, never the short straight arm
    out = s.run_op("analyze.skeleton_longest_path", {"mode": "3D (whole stack)"})
    lsp = s.images[out["new_images"][0]]
    m = lsp.data[0, :, 0] > 0
    assert lsp.shape == (1, 40, 1, 60, 60) and lsp.lut_is_binary and "3 branches" in out["message"]
    assert 45 <= m.sum() <= 60 and m[20, 30, 30] and m[5:8, 5:8, 52:56].any() and m[33:36, 52:56, 52:56].any() and not m[20, 30, :20].any()
    assert s.results.rows[-1]["Longest Shortest Path"] == pytest.approx(sum(sorted(r["Length"] for r in branch_rows(s))[1:]))


def test_skeleton_3d_pruning_and_ring():
    spur = y_volume()
    spur[16:20, 30, 15] = 255  # a 4-voxel spur off the straight arm
    s = Session()
    s.add_image(LumenImage(spur, name="spur", axes="ZYX"))
    s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False, "show": "Nothing"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 5 and row["# Junctions"] == 2 and row["# End-point voxels"] == 4
    out = s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False, "prune_ends": 6, "show": "Tagged skeleton"})
    row = s.results.rows[-1]
    assert "1 end branch pruned" in out["message"] and row["# Branches"] == 3 and row["# Junctions"] == 1 and row["# End-point voxels"] == 3
    tag = s.images[out["new_images"][0]]
    assert (tag.data == 1).sum() == 3 and (tag.data == 3).sum() == 1
    # a ring lying in one slice of a volume is one circular 3-D branch; pruning cycles removes it
    vol = np.zeros((5, 100, 100), np.uint8)
    vol[2] = ring()
    s.add_image(LumenImage(vol, name="ring3", axes="ZYX"))
    s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False, "show": "Nothing"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 1 and row["# Junctions"] == 0 and row["# End-point voxels"] == 0 and row["Longest Shortest Path"] == 0.0
    br = branch_rows(s)
    assert len(br) == 1 and br[0]["Euclidean distance"] == 0.0 and br[0]["V1 z"] == 2.0 and 2 * np.pi * 30 * 0.95 < br[0]["Length"] < 2 * np.pi * 30 * 1.1
    out = s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False, "prune_cycles": "shortest branch", "show": "Nothing"})
    assert "0 skeletons (3D)" in out["message"] and "1 cycle branch pruned" in out["message"]
    # 3D mode on a single-slice image behaves like the plane analysis (with z columns)
    s.add_image(LumenImage(y_shape(), name="Y"))
    s.run_op("analyze.skeleton", {"mode": "3D (whole stack)", "skeletonize_first": False, "show": "Nothing"})
    row = s.results.rows[-1]
    assert row["# Branches"] == 3 and row["# Junctions"] == 1 and all(r["V1 z"] == 0.0 for r in branch_rows(s))


def test_skeleton_longest_shortest_path_mask_2d():
    s = Session()
    im = s.add_image(LumenImage(y_shape(), name="Y"))
    out = s.run_op("analyze.skeleton", {"skeletonize_first": False, "show": "Longest shortest path mask"})
    m = s.images[out["new_images"][0]]
    assert m.lut_is_binary and "Longest shortest paths of Y" in m.name
    mm = m.data[0, 0, 0] > 0
    assert 58 <= mm.sum() <= 64 and mm[20, 80] and mm[80, 80] and mm[50, 50] and not mm[50, 10] and not mm[50, 30]
    assert s.results.rows[-1]["Longest Shortest Path"] == pytest.approx(60 * SQ2, rel=0.05) and s.active is im
    # the dedicated command gives the same mask and keeps the parameters of Analyze Skeleton
    out2 = s.run_op("analyze.skeleton_longest_path", {"skeletonize_first": False})
    assert np.array_equal(s.images[out2["new_images"][0]].data, m.data)
    out3 = s.run_op("analyze.skeleton_longest_path", {"skeletonize_first": False, "prune_ends": 50})
    assert "3 end branches pruned" in out3["message"] or "end branch" in out3["message"]
    # a ring has no vertices → an empty mask
    s.add_image(LumenImage(ring(), name="ring"))
    out = s.run_op("analyze.skeleton_longest_path", {})
    assert (s.images[out["new_images"][0]].data > 0).sum() == 0
