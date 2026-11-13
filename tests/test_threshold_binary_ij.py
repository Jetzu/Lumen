"""ImageJ 1.54p parity of Image ▸ Adjust ▸ Threshold and Process ▸ Binary (crossval families 3 and 4).

The reference numbers and hashes below are ImageJ's own outputs for the cross-validation images
(``tests/crossval/harness.IMAGES``), captured with the harness (`setAutoThreshold(method + " dark")` /
`setAutoThreshold(method)` → `getThreshold`, and the Process ▸ Binary commands after a Default-dark mask with
``Options... iterations=1 count=1 black``).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lumen.core import ij_autothreshold as ijt
from lumen.core.image import LumenImage
from lumen.core.ij_compat import IJ
from lumen.core.ops import _imagej_binary as ijb
from lumen.core.session import Session
from lumen.core.threshold import METHODS, auto_threshold, legacy_isodata_threshold, scale_threshold, to_byte_ij
from tests.crossval.harness import IMAGES

IJ_DARK_8 = {"Default": (91, 255), "Huang": (67, 255), "Intermodes": (113, 255), "IsoData": (95, 255), "Li": (76, 255), "MaxEntropy": (104, 255),
             "Mean": (56, 255), "MinError": (43, 255), "Minimum": (178, 255), "Moments": (104, 255), "Otsu": (95, 255), "Percentile": (37, 255),
             "RenyiEntropy": (84, 255), "Shanbhag": (159, 255), "Triangle": (58, 255), "Yen": (78, 255)}
IJ_LIGHT_8 = {"Default": (0, 90), "Huang": (0, 66), "Intermodes": (0, 112), "IsoData": (0, 94), "Li": (0, 75), "MaxEntropy": (0, 103), "Mean": (0, 55),
              "MinError": (0, 42), "Minimum": (0, 177), "Moments": (0, 103), "Otsu": (0, 94), "Percentile": (0, 36), "RenyiEntropy": (0, 83),
              "Shanbhag": (0, 158), "Triangle": (0, 57), "Yen": (0, 77)}
IJ_DARK_16 = {"Default": (1501, 65535), "Huang": (1072, 65535), "Intermodes": (2358, 65535), "IsoData": (1482, 65535), "Li": (1165, 65535),
              "MaxEntropy": (2023, 65535), "Mean": (923, 65535), "MinError": (699, 65535), "Minimum": (4055, 65535), "Moments": (1706, 65535),
              "Otsu": (1501, 65535), "Percentile": (587, 65535), "RenyiEntropy": (1706, 65535), "Shanbhag": (3048, 65535), "Triangle": (923, 65535),
              "Yen": (1631, 65535)}
IJ_LIGHT_16 = {"Default": (0, 1482), "Huang": (0, 1053), "Intermodes": (0, 2340), "IsoData": (0, 1463), "Li": (0, 1146), "MaxEntropy": (0, 2004),
               "Mean": (0, 904), "MinError": (0, 680), "Minimum": (0, 4036), "Moments": (0, 1687), "Otsu": (0, 1482), "Percentile": (0, 568),
               "RenyiEntropy": (0, 1687), "Shanbhag": (0, 3029), "Triangle": (0, 904), "Yen": (0, 1612)}
IJ_DARK_32 = {"Default": 0.3475, "Huang": 0.04421, "Intermodes": 1.1058, "IsoData": 0.3475, "Li": 0.1327, "MaxEntropy": 0.6508, "Mean": -0.006344,
              "MinError": -0.158, "Minimum": 2.3821, "Moments": 0.4739, "Otsu": 0.3475, "Percentile": -0.2212, "RenyiEntropy": 0.4486,
              "Shanbhag": 1.2448, "Triangle": -0.03162, "Yen": 0.3981}
# sha1 of ImageJ's result pixels (uint8, row major), foreground count, maximum
IJ_BINARY = {
    "erode": ("d60cababe1ca8976828e4333c26f8c04b4cf58e8", 1996, 255),
    "erode_iter3": ("f7c6f3c2192739dd12a8e93340996d69ddaee95c", 631, 255),
    "dilate": ("82bef83ff2aad9cc2f825c5cafee12b5740e496c", 4238, 255),
    "open": ("8db1d2a96224aeec920b3e9d9ffed8887a0bee2a", 2988, 255),
    "close": ("b443ecab95224646d6939bb93bea30e8f255fe76", 3146, 255),
    "outline": ("ad0331ffdd671117242d5b775ccc530724c23c86", 1156, 255),
    "skeletonize": ("d4c2249812a73b1748548ee0fda4fc4960531b1f", 383, 255),
    "distance_map": ("cd6615192d49fb84ffd572f8c8a3b6a4707e7083", 3082, 10),
    "ultimate_points": ("7cfe8a9b81dea770af8cc5dcb297e3fcef8b3a20", 13, 10),
    "voronoi": ("9f0438ebc4a942fe1e459f825b2f7e3d221bfc98", 862, 46),
    "watershed": ("b62d4c1a1bd71dfb619d26deca204d6185beba26", 3045, 255),
}
IJ_MASK16_SHA = "85fd8e52cbd1c575f5171a6a0d6450ce49910b56"   # setAutoThreshold("Otsu dark") + Convert to Mask on blobs16
IJ_MASK32_SHA = "79a6bd3bdb75185b9252a0799e3df960dc6b8985"   # … on blobs32
IJ_STACK_SHA = "5593b28ac441275754bd56927528e42973817430"    # Convert to Mask method=Default background=Dark calculate on zstack8


def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def _blobs(name: str) -> np.ndarray:
    return IMAGES[name]()[0]


def _mask8() -> np.ndarray:
    img = _blobs("blobs8")
    lo, hi = auto_threshold(img, "Default", None, True)
    return (img >= lo) & (img <= hi)


# ---------------------------------------------------------------- thresholds
def test_auto_threshold_8bit_all_methods_dark_and_light():
    img = _blobs("blobs8")
    for m in METHODS:
        assert auto_threshold(img, m, None, True) == IJ_DARK_8[m], m
        assert auto_threshold(img, m, None, False) == IJ_LIGHT_8[m], m   # no histogram flip: Li gives 75, not 108


def test_auto_threshold_16bit_scaling_and_open_ends():
    img = _blobs("blobs16")
    for m in METHODS:
        assert auto_threshold(img, m, None, True) == IJ_DARK_16[m], m
        assert auto_threshold(img, m, None, False) == IJ_LIGHT_16[m], m
    p8, mn, mx = to_byte_ij(img)
    assert (mn, mx) == (121.0, 4875.0) and p8.dtype == np.uint8 and p8.max() == 255
    # ImageJ: (int)((v-min)*256/(max-min+1) + 0.5), clipped
    v = img.astype(np.int64)
    assert np.array_equal(p8, np.clip(np.floor((v - 121) * 256.0 / 4755 + 0.5), 0, 255).astype(np.uint8))


def test_auto_threshold_32bit_values():
    img = _blobs("blobs32")
    for m in METHODS:
        lo, hi = auto_threshold(img, m, None, True)
        assert hi == 1e30 and lo == pytest.approx(IJ_DARK_32[m], abs=5e-5), m
    lo, hi = auto_threshold(img, "Otsu", None, False)
    assert lo == -1e30 and hi == pytest.approx(0.3475 - (2.710714817047119 + 0.5118427276611328) / 255, abs=5e-5)


def test_scale_threshold_rules():
    assert scale_threshold(0.0, 100.0, 121, 4875, np.dtype(np.uint16)) == (0.0, float(np.floor(121 + 100 / 255 * 4754 + 0.5)))
    assert scale_threshold(74.0, 255.0, 121, 4875, np.dtype(np.uint16)) == (1501.0, 65535.0)
    assert scale_threshold(0.0, 255.0, -1.0, 2.0, np.dtype(np.float32)) == (-1e30, 1e30)
    assert scale_threshold(10.0, 20.0, 5.0, 5.0, np.dtype(np.uint16)) == (5.0, 5.0)  # max == min: both limits = min
    assert scale_threshold(3.0, 255.0, 0, 255, np.dtype(np.uint8)) == (3.0, 255.0)


def test_add_one_and_legacy_isodata_on_stack_slices():
    z = _blobs("zstack8")
    # Convert Stack to Binary ('calculate', method=Default background=Dark): the level itself, not level + 1
    assert [auto_threshold(z[k], "Default", None, True, add_one=False)[0] for k in range(6)] == [67, 88, 98, 98, 84, 60]
    assert [auto_threshold(z[k], "Default", None, True)[0] for k in range(6)] == [68, 89, 99, 99, 85, 61]
    assert [legacy_isodata_threshold(z[k]) for k in range(6)] == [(67.0, 255.0), (88.0, 255.0), (98.0, 255.0), (98.0, 255.0), (84.0, 255.0), (60.0, 255.0)]
    # bright objects on a dark background → [level, 255]; the inverted image → [0, level]
    b8 = _blobs("blobs8")
    assert legacy_isodata_threshold(b8) == (90.0, 255.0)
    lo, hi = legacy_isodata_threshold(255 - b8)
    assert lo == 0.0 and 160 <= hi <= 165   # IJIsoData ignores bins 0 and 255, so the level is not exactly mirrored (163)


def test_autothresholder_edge_rules():
    h = np.zeros(256, np.int64)
    h[40] = 10
    assert ijt.bilevel(h, False) == 40 and ijt.bilevel(h, True) == 39
    h[200] = 5
    assert ijt.bilevel(h, False) == 200 and ijt.get_threshold("Otsu", h, subtract_one=True) == 199
    h[100] = 1
    assert ijt.bilevel(h, False) == -1
    assert ijt.mean([0, 0, 3, 0, 4]) == 3   # long division: (2*3 + 4*4) // 7 = 3
    assert ijt.java_round(2.5) == 3 and ijt.java_round(-2.5) == -2
    assert ijt.renyi_entropy(np.bincount(_blobs("blobs8").ravel(), minlength=256)) == 83


# ---------------------------------------------------------------- binary algorithms
def test_binary_ports_match_imagej():
    b = _mask8()
    assert int(b.sum()) == 3082
    results = {
        "erode": ijb.ij_erode(b).astype(np.uint8) * 255,
        "dilate": ijb.ij_dilate(b).astype(np.uint8) * 255,
        "open": ijb.ij_open(b).astype(np.uint8) * 255,
        "close": ijb.ij_close(b).astype(np.uint8) * 255,
        "outline": ijb.ij_outline(b).astype(np.uint8) * 255,
        "skeletonize": ijb.ij_skeletonize(b).astype(np.uint8) * 255,
        "distance_map": ijb.ij_distance_map(b),
        "ultimate_points": ijb.ij_ultimate_points(b)[0],
        "voronoi": ijb.ij_voronoi(b),
        "watershed": ijb.ij_watershed(b).astype(np.uint8) * 255,
    }
    e = b
    for _ in range(3):
        e = ijb.ij_erode(e)
    results["erode_iter3"] = e.astype(np.uint8) * 255
    for name, out in results.items():
        sha, count, mx = IJ_BINARY[name]
        assert (int((out > 0).sum()), int(out.max())) == (count, mx), name
        assert _sha(out) == sha, name


def test_erode_dilate_count_and_pad_edges():
    b = np.zeros((7, 7), bool)
    b[1:6, 1:6] = True
    assert ijb.ij_erode(b, 1).sum() == 9                      # 8-neighbour erosion: the 3×3 core survives
    assert ijb.ij_erode(b, 3).sum() == 9                      # edge pixels have exactly 3 background neighbours → still removed
    assert ijb.ij_erode(b, 4).sum() == 21                     # count 4: only the corners (5 background neighbours) go
    assert ijb.ij_erode(b, 6).sum() == 25                     # … and with count 6 nothing has enough background neighbours
    touching = np.zeros((5, 5), bool)
    touching[0:3, 0:3] = True
    assert ijb.ij_erode(touching, 1).sum() == 1               # the image border counts as background: only (1,1) keeps 8 foreground neighbours …
    assert ijb.ij_erode(touching, 1, pad_edges=True).sum() == 4  # … with 'pad edges' (outside = foreground) the corner 2×2 survives
    # count 1: the whole 7×7; count 2: all but the 4 corners (1 foreground neighbour); count 3: only the 3 middle pixels of each side (3 neighbours)
    assert ijb.ij_dilate(b, 1).sum() == 49 and ijb.ij_dilate(b, 2).sum() == 45 and ijb.ij_dilate(b, 3).sum() == 37


def test_outline_modes_and_skeleton_border():
    b = np.zeros((9, 9), bool)
    b[2:7, 2:7] = True
    ring = ijb.ij_outline(b, black_background=True)
    assert ring.sum() == 24 and not ring[2:7, 2:7].any() and ring[1, 1] and ring[7, 7]   # the ring around the 5×5 square
    inner = ijb.ij_outline(b, black_background=False)
    assert inner.sum() == 16 and not inner[3:6, 3:6].any() and inner[2, 2]
    edge = np.zeros((6, 6), bool)
    edge[0:3, :] = True
    out = ijb.ij_outline(edge)
    # border rows / columns are not processed (they keep their value), the interior ring below the object is set
    assert np.array_equal(out[0], edge[0]) and out[3, 1:-1].all() and not out[3, 0] and not out[3, -1] and not out[1:3, 1:-1].any()
    sk = ijb.ij_skeletonize(edge)
    assert 0 < sk.sum() < edge.sum() and (sk.sum(axis=0) <= 1).all()   # objects on the border: padded, thinned to a 1-px line, cropped


def test_find_maxima_segmented_splits_touching_disks():
    yy, xx = np.mgrid[:80, :120]
    disks = ((xx - 40) ** 2 + (yy - 40) ** 2 < 20 ** 2) | ((xx - 72) ** 2 + (yy - 40) ** 2 < 20 ** 2)
    ws = ijb.ij_watershed(disks)
    from scipy import ndimage as ndi
    assert ndi.label(ws, np.ones((3, 3)))[1] == 2 and ws.sum() > 0.95 * disks.sum()
    uep8, uep = ijb.ij_ultimate_points(disks)
    ys, xs = np.nonzero(uep8)
    assert len(ys) == 2 and sorted(xs.tolist()) == [40, 72] and uep8.max() == 20
    assert ijb.ij_voronoi(disks).max() == 0                      # the touching disks are one object: no Voronoi line
    apart = ((xx - 30) ** 2 + (yy - 40) ** 2 < 15 ** 2) | ((xx - 90) ** 2 + (yy - 40) ** 2 < 15 ** 2)
    vor = ijb.ij_voronoi(apart)
    assert vor.dtype == np.uint8 and vor[40, 60] == 16 and vor.max() >= 30 and not vor[apart].any()   # the line carries the EDM value (16 px to x = 44)
    assert set(np.flatnonzero(vor[40]).tolist()) == {60}
    assert ijb.find_maxima(ijb.edm_float(disks), 0.5, ijb.IN_TOLERANCE, False, True).max() == 255
    empty = np.zeros((10, 10), bool)
    assert ijb.ij_watershed(empty).sum() == 0 and ijb.ij_ultimate_points(empty)[0].sum() == 0 and ijb.ij_voronoi(empty).sum() == 0


# ---------------------------------------------------------------- ops and the macro layer
def _blob_session(name: str = "blobs8") -> tuple[Session, LumenImage]:
    s = Session()
    data = _blobs(name)
    img = s.add_image(LumenImage(data, name="input.tif", axes="ZYX" if data.ndim == 3 else None))
    return s, img


def test_ops_defaults_are_imagej():
    s, img = _blob_session()
    s.run_op("threshold.compute", {"method": "Default", "dark_background": True})
    s.run_op("threshold.apply", {})
    mask = img.current_plane().copy()
    assert int((mask > 0).sum()) == 3082
    for op_id, params, key in [("binary.erode", {}, "erode"), ("binary.erode", {"iterations": 3}, "erode_iter3"), ("binary.dilate", {}, "dilate"),
                               ("binary.open", {}, "open"), ("binary.close", {}, "close"), ("binary.outline", {}, "outline"),
                               ("binary.skeletonize", {}, "skeletonize"), ("binary.watershed", {}, "watershed"), ("binary.voronoi", {}, "voronoi"),
                               ("binary.distance_map", {}, "distance_map"), ("binary.ultimate_points", {}, "ultimate_points")]:
        img.replace_data(mask[None, None, None].copy())
        s.run_op(op_id, params)
        assert _sha(img.current_plane()) == IJ_BINARY[key][0], op_id
    img.replace_data(mask[None, None, None].copy())
    out = s.run_op("binary.distance_map", {"output": "32-bit"})
    edm = s.images[out["new_images"][0]]
    assert edm.dtype == np.float32 and edm.current_plane().max() == pytest.approx(9.848858, abs=1e-5) and img.current_plane().max() == 255
    out = s.run_op("binary.ultimate_points", {"output": "Point selection"}, image_id=img.id)
    assert len(out["roi"]["points"]) == 13 and out["message"] == "13 ultimate points"
    out = s.run_op("binary.ultimate_points", {"output": "Image"}, image_id=img.id)
    uep = s.images[out["new_images"][0]]
    assert uep.dtype == np.float32 and int((uep.current_plane() > 0).sum()) == 13
    s.run_op("binary.outline", {"mode": "Inner boundary"}, image_id=img.id)
    assert 0 < int((img.current_plane() > 0).sum()) < 1156
    img.replace_data(mask[None, None, None].copy())
    s.run_op("binary.voronoi", {"values": "255"}, image_id=img.id)
    assert set(np.unique(img.current_plane()).tolist()) <= {0, 255} and int((img.current_plane() > 0).sum()) == 862


def test_macro_layer_options_and_binary_commands():
    s, img = _blob_session()
    ij = IJ(s)
    ij.macro('setOption("BlackBackground", true); setAutoThreshold("Default dark"); run("Convert to Mask");')
    mask = img.current_plane().copy()
    ij.run("Options...", "iterations=3 count=1 black do=Nothing")
    ij.run("Erode")
    assert _sha(img.current_plane()) == IJ_BINARY["erode_iter3"][0]
    img.replace_data(mask[None, None, None].copy())
    ij.run("Options...", "iterations=1 count=2 black pad do=Nothing")
    ij.run("Erode")
    assert np.array_equal(img.current_plane() > 0, ijb.ij_erode(mask > 0, 2, pad_edges=True))
    img.replace_data(mask[None, None, None].copy())
    ij.run("Options...", "iterations=2 count=1 black do=Close")       # do= executes the operation with the new settings
    assert np.array_equal(img.current_plane() > 0, ijb.ij_close(mask > 0, 2))
    img.replace_data(mask[None, None, None].copy())
    ij.run("Options...", "iterations=1 count=1 do=Nothing")           # no 'black': ImageJ's white-background result on a 0/255 mask
    ij.run("Outline")
    assert np.array_equal(img.current_plane() > 0, ijb.ij_outline_white_bg(mask > 0)) and int((img.current_plane() > 0).sum()) == 17488  # ImageJ: 17488 px
    img.replace_data(mask[None, None, None].copy())
    ij.run("Options...", "iterations=1 count=1 black edm=32-bit do=Nothing")
    ij.run("Distance Map")
    assert ij.getTitle().startswith("EDM of") and ij.img.dtype == np.float32
    ij.selectWindow("input.tif")
    ij.run("Ultimate Points")
    assert ij.getTitle().startswith("UEP of") and ij.img.dtype == np.float32 and int((ij.img.current_plane() > 0).sum()) == 13
    ij.selectWindow("input.tif")
    ij.run("Options...", "iterations=1 count=1 black do=Nothing")
    ij.run("Ultimate Points")
    assert ij.getTitle() == "input.tif" and _sha(img.current_plane()) == IJ_BINARY["ultimate_points"][0]


def test_macro_convert_to_mask_paths():
    # a set threshold is applied (setAutoThreshold adds one: 91 … 255)
    s, img = _blob_session()
    ij = IJ(s)
    ij.macro('setAutoThreshold("Default dark"); run("Convert to Mask");')
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 91)
    # without a threshold: ImageJ's legacy ISODATA (level 90, no +1), method= is ignored for single images
    s, img = _blob_session()
    ij = IJ(s)
    ij.run("Convert to Mask", "method=Otsu background=Dark")
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 90)
    s, img = _blob_session()
    IJ(s).run("Make Binary")
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 90)
    # 16-bit and 32-bit masks after setAutoThreshold("Otsu dark")
    for name, sha in (("blobs16", IJ_MASK16_SHA), ("blobs32", IJ_MASK32_SHA)):
        s, img = _blob_session(name)
        IJ(s).macro('setOption("BlackBackground", true); setAutoThreshold("Otsu dark"); run("Convert to Mask");')
        assert img.dtype == np.uint8 and _sha(img.current_plane()) == sha, name
    # a stack with 'calculate': per-slice Default levels without the +1 (67, 88, 98, 98, 84, 60)
    s, img = _blob_session("zstack8")
    IJ(s).macro('setOption("BlackBackground", true); run("Convert to Mask", "method=Default background=Dark calculate");')
    assert _sha(img.data[0, :, 0]) == IJ_STACK_SHA
    z = _blobs("zstack8")
    s, img = _blob_session("zstack8")
    IJ(s).run("Convert to Mask", "method=Otsu background=Dark calculate")
    for k in range(6):
        lo = auto_threshold(z[k], "Otsu", None, True, add_one=False)[0]
        assert np.array_equal(img.data[0, k, 0] > 0, z[k] >= lo), k
    # method=Default background=Default (or no options) on a stack: the legacy ISODATA per slice
    s, img = _blob_session("zstack8")
    IJ(s).run("Convert to Mask", "calculate")
    for k in range(6):
        assert np.array_equal(img.data[0, k, 0] > 0, z[k] >= legacy_isodata_threshold(z[k])[0]), k
    # a stack without 'calculate': one threshold (set, or the legacy auto threshold of the current slice) for all slices
    s, img = _blob_session("zstack8")
    ij = IJ(s)
    ij.macro('setAutoThreshold("Otsu dark"); run("Convert to Mask", "method=Default background=Dark calculate");')  # threshold set → 'calculate' is ignored
    lo = auto_threshold(z[0], "Otsu", None, True)[0]
    for k in range(6):
        assert np.array_equal(img.data[0, k, 0] > 0, z[k] >= lo), k


def test_threshold_apply_op_rules():
    s, img = _blob_session()
    s.run_op("threshold.apply", {})                                   # no threshold set → legacy ISODATA (90)
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 90)
    s, img = _blob_session()
    s.run_op("threshold.apply", {"method": "Default"})                # explicit method: level + 1 like setAutoThreshold
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 91)
    s, img = _blob_session()
    s.run_op("threshold.apply", {"method": "Default", "add_one": False})
    assert np.array_equal(img.current_plane() > 0, _blobs("blobs8") >= 90)
    s, img = _blob_session()
    s.run_op("threshold.set", {"lower": 100, "upper": 200})
    s.run_op("threshold.apply", {})
    assert np.array_equal(img.current_plane() > 0, (_blobs("blobs8") >= 100) & (_blobs("blobs8") <= 200))
