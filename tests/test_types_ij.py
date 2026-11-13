"""ImageJ-exact type conversions, display rendering, Process ▸ Math and transforms (lumen.core.ops._imagej_types).

The expected values are ImageJ 1.54p's formulas evaluated by hand (TypeConverter, ByteProcessor.setMinAndMax,
ContrastAdjuster.apply, ImageProcessor.process / rotate / resize / translate, DownsizeTable); the pixel-exact
agreement with ImageJ itself is checked by tests/crossval.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401  (registers every op)
from lumen.core.ij_compat import IJ
from lumen.core.image import LumenImage
from lumen.core.ops import _imagej_types as T
from lumen.core.session import Session


def _img16(session, nc=1, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(100, 5000, size=(nc, 24, 32), dtype=np.uint16)
    img = LumenImage(data, name="t16.tif", axes="CYX")
    return session.add_image(img)


# ---------------------------------------------------------------- conversions (TypeConverter)
def test_short_to_byte_formula_and_clamping():
    v = np.array([[0, 500, 1250, 3000, 4000]], np.uint16)
    out = T.short_to_byte(v, 500, 3000)
    sc = 256.0 / (3000 - 500 + 1)
    exp = [min(int(max(int(x) - 500, 0) * sc + 0.5), 255) for x in v[0]]
    assert out.tolist() == [exp]
    assert out.dtype == np.uint8
    assert T.short_to_byte(v, 0, 0, scale=False).tolist() == [[0, 255, 255, 255, 255]]
    # the +0.5 is a half-up rounding of the 256/(max−min+1) scaling, not 255/(max−min)
    assert T.short_to_byte(np.array([[3000]], np.uint16), 500, 3000)[0, 0] == 255
    assert T.short_to_byte(np.array([[2999]], np.uint16), 500, 3000)[0, 0] == min(int(2499 * sc + 0.5), 255)
    assert T.short_to_byte(np.array([[2990]], np.uint16), 500, 3000)[0, 0] == int(2490 * sc + 0.5) == 255
    assert T.short_to_byte(np.array([[2980]], np.uint16), 500, 3000)[0, 0] == int(2480 * sc + 0.5) == 254


def test_float_conversions():
    f = np.array([[-0.5, 0.0, 0.25, 0.5, 1.0, 1.5]], np.float32)
    assert T.float_to_byte(f, 0.0, 1.0).tolist() == [[0, 0, 64, 128, 255, 255]]  # (int)(v·255 + 0.5)
    assert T.float_to_byte(f, 0.0, 1.0, scale=False).tolist() == [[0, 0, 0, 1, 1, 2]]  # v + 0.5f truncated
    s = T.float_to_short(f, 0.0, 1.0)
    assert s.dtype == np.uint16 and s.tolist() == [[0, 0, int(0.25 * 65535 + 0.5), int(0.5 * 65535 + 0.5), 65535, 65535]]
    assert T.float_to_short(f, 0.0, 1.0, scale=False).tolist() == [[0, 0, 0, 1, 1, 2]]
    nan = np.array([[np.nan, 2.0]], np.float32)
    assert T.float_to_byte(nan, 0.0, 1.0).tolist() == [[0, 255]]


def test_rgb_to_gray_weights():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (10, 20, 31)
    rgb[1, 0] = (255, 255, 255)
    assert T.rgb_to_byte(rgb)[0, 0] == 85  # (byte)(255/3 + 0.5)
    assert T.rgb_to_byte(rgb)[0, 1] == int(10 / 3 + 20 / 3 + 31 / 3 + 0.5)
    assert T.rgb_to_byte(rgb)[1, 0] == 255
    assert T.rgb_to_byte(rgb, T.WEIGHTED)[0, 0] == int(0.299 * 255 + 0.5)
    assert T.rgb_to_float(rgb)[0, 0] == pytest.approx(85.0)
    chw = rgb.transpose(2, 0, 1)
    assert np.array_equal(T.rgb_to_byte(chw), T.rgb_to_byte(rgb))


def test_byte_display_index_and_apply_lut_table():
    v = np.array([[0, 49, 50, 51, 115, 179, 180, 181, 255]], np.uint8)
    idx = T.byte_display_index(v, 50, 180)
    assert idx.tolist() == [[0, 0, 0, int(256.0 * 1 / 130), int(256.0 * 65 / 130), int(256.0 * 129 / 130), 255, 255, 255]]
    assert T.byte_display_index(v, 0, 255).tolist() == v.tolist()
    t = T.apply_lut_table(np.uint8, 50, 180)
    assert t[50] == 0 and t[51] == int(1 / 130 * 256) and t[179] == int(129 / 130 * 256) and t[180] == 255 and t[255] == 255
    t16 = T.apply_lut_table(np.uint16, 1000, 3000)
    assert t16.size == 65536 and t16[1000] == 0 and t16[3000] == 65535 and t16[2000] == int(1000 / 2000 * 65536)


def test_composite_rendering_sums_and_clamps():
    data = np.zeros((2, 4, 4), np.uint16)
    data[0] = 1000
    data[1] = 4000
    img = LumenImage(data, name="c", axes="CYX")
    img.display.mode = "composite"
    img.display.channels[0].lut, img.display.channels[1].lut = "Red", "Green"
    for ch in img.display.channels:
        ch.min, ch.max = 0.0, 4095.0
    rgb = T.render_rgb(img, 0, 0)
    assert rgb.shape == (4, 4, 3)
    assert rgb[0, 0].tolist() == [T.short_to_byte(np.array([[1000]], np.uint16), 0, 4095)[0, 0], T.short_to_byte(np.array([[4000]], np.uint16), 0, 4095)[0, 0], 0]
    img.display.channels[1].lut = "Red"  # both channels red: the sum saturates at 255
    assert T.render_rgb(img, 0, 0)[0, 0].tolist() == [255, 0, 0]
    img.display.channels[1].visible = False
    assert T.render_rgb(img, 0, 0)[0, 0, 0] == T.short_to_byte(np.array([[1000]], np.uint16), 0, 4095)[0, 0]


def test_default_display_patch_follows_imagej_open_defaults():
    data = np.stack([np.linspace(100, 5000, 64).reshape(8, 8), np.linspace(50, 2000, 64).reshape(8, 8)]).astype(np.uint16)
    img = LumenImage(data, name="two", axes="CYX")
    img.display.mode = "composite"
    patch = T.imagej_display_patch(img)
    assert patch is not None
    chans = patch["channels"]
    assert (chans[0]["min"], chans[0]["max"]) == (100.0, 5000.0) and (chans[1]["min"], chans[1]["max"]) == (100.0, 5000.0)  # channel 1's range for every channel
    assert chans[0]["lut"] == "Red" and chans[1]["lut"] == "Green"  # CompositeImage.colors instead of Lumen's Cyan / Magenta defaults
    img.display.channels[1].min, img.display.channels[1].max = 300.0, 900.0  # an explicit range is kept
    p2 = T.imagej_display_patch(img)
    assert "min" not in p2["channels"][1] and "min" in p2["channels"][0]
    eight = LumenImage(np.zeros((4, 4), np.uint8), name="b")
    assert T.imagej_display_patch(eight) is None


# ---------------------------------------------------------------- Process ▸ Math (ImageProcessor.process)
def test_byte_math_tables_truncate_and_round_like_imagej():
    g = T.byte_math_table("gamma", 0.5)
    assert g[0] == 0 and g[255] == 255 and g[64] == int(np.exp(np.log(64 / 255.0) * 0.5) * 255.0)
    m = T.byte_math_table("mult", 1.5)
    assert m[1] == 2 and m[3] == 5 and m[200] == 255  # Math.round: half up (numpy rint would give 4 for 4.5)
    assert T.byte_math_table("add", 50.9)[10] == 60  # (int)value
    assert T.byte_math_table("add", -50)[10] == 0
    scale = 255.0 / math.log(255.0)
    assert T.byte_math_table("log", 0)[0] == 0 and T.byte_math_table("log", 0)[100] == int(math.log(100) * scale)
    assert T.byte_math_table("exp", 0)[100] == int(math.exp(100 / scale))
    assert T.byte_math_table("sqrt", 0)[200] == 14 and T.byte_math_table("sqr", 0)[20] == 255
    assert T.byte_math_table("min", 50)[49] == 50 and T.byte_math_table("max", 150)[151] == 150
    assert T.byte_math_table("and", 0b11110000)[0b10101010] == 0b10100000
    assert T.byte_math_table("invert", 0)[3] == 252 and T.byte_math_table("set", 77)[3] == 77


def test_short_math_scales_by_the_display_range():
    v = np.array([[121, 1000, 4875]], np.uint16)
    g = T.short_math(v, "gamma", 0.5, 121, 4875)
    rng = 4875.0 - 121.0
    assert g[0, 0] == 121 and g[0, 2] == 4875 and g[0, 1] == int(math.exp(0.5 * math.log((1000 - 121) / rng)) * rng + 121)
    lg = T.short_math(v, "log", 0, 121, 4875)
    k = 4875 / np.log(4875.0)  # max2/Math.log(max2): the maximum itself truncates to 4874 in IEEE double, exactly as ImageJ computes it
    assert lg[0, 2] == int(np.log(4875.0) * k) == 4874 and lg[0, 1] == int(np.log(1000.0) * k) and lg[0, 0] == int(np.log(121.0) * k)
    ex = T.short_math(v, "exp", 0, 121, 4875)
    assert ex[0, 2] == int(np.exp(4875 * (np.log(4875.0) / 4875))) and ex[0, 0] == int(np.exp(121 * (np.log(4875.0) / 4875)))
    assert T.short_math(np.array([[9]], np.uint16), "sqrt", 0, 0, 0)[0, 0] == 3 and T.short_math(np.array([[8]], np.uint16), "sqrt", 0, 0, 0)[0, 0] == 2
    assert T.short_math(np.array([[300]], np.uint16), "sqr", 0, 0, 0)[0, 0] == 65535
    assert T.short_math(np.array([[3]], np.uint16), "mult", 1.5, 0, 0)[0, 0] == 5
    assert T.short_math(np.array([[50]], np.uint16), "gamma", 0.5, 121, 4875)[0, 0] == 0  # below min: log of a negative → NaN → (int) 0
    assert T.short_math(v, "invert", 0, 121, 4875).tolist() == [[4875, 4875 - (1000 - 121), 121]]


def test_float_math_matches_floatprocessor():
    v = np.array([[-1.0, 0.0, 2.71]], np.float32)
    g = T.float_math(v, "gamma", 0.5, 0, 0)
    assert g.dtype == np.float32 and g[0, 0] == 0 and g[0, 1] == 0 and g[0, 2] == np.float32(np.exp(0.5 * np.log(np.float64(np.float32(2.71)))))
    lg = T.float_math(v, "log", 0, 0, 0)
    assert np.isnan(lg[0, 0]) and lg[0, 1] == -np.inf
    assert T.float_math(v, "sqrt", 0, 0, 0)[0, 0] == 0
    assert T.float_math(v, "mult", 1 / 3, 0, 0)[0, 2] == np.float32(2.71) * np.float32(1 / 3)
    assert T.float_math(v, "invert", 0, -1.0, 2.71).tolist() == [[np.float32(2.71), np.float32(np.float32(2.71) - np.float32(0.0 + 1.0)), np.float32(-1.0)]]


# ---------------------------------------------------------------- transforms
def test_rotate_nearest_90_degrees_is_rot90_and_angle_0_identity():
    a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert np.array_equal(T.rotate(a, 90, T.NONE), np.rot90(a, k=-1))
    assert np.array_equal(T.rotate(a, -90, T.NONE), np.rot90(a, k=1))
    assert np.array_equal(T.rotate(a, 0, T.BILINEAR), a) and np.array_equal(T.rotate(a, 360, T.BICUBIC), a)
    f = a.astype(np.float32) + 0.25
    r = T.rotate(f, 90, T.BILINEAR)  # ImageJ clamps source coordinates ≥ w−1 to w−1.001: the far edge interpolates 0.1 % towards its neighbour
    assert r.dtype == np.float32 and np.allclose(r, np.rot90(f, k=-1), atol=0.005) and np.array_equal(r[:3, 1:], np.rot90(f, k=-1)[:3, 1:])


def test_rotate_fills_uncovered_pixels_with_the_background():
    a = np.full((20, 30), 200, np.uint8)
    r = T.rotate(a, 30, T.BILINEAR, bg=0)
    assert r[0, 0] == 0 and r[10, 15] == 200
    r16 = T.rotate(a.astype(np.uint16) * 100, 30, T.NONE, bg=7)
    assert r16[0, 0] == 7 and r16.dtype == np.uint16
    rb = T.rotate(a, 30, T.BICUBIC, bg=0)
    assert rb[0, 0] == 0 and rb[10, 15] == 200


def test_enlarged_canvas_and_offsets():
    assert T.enlarged_canvas(160, 120, 15) == (186, 157)
    assert T.canvas_offset(160, 120, 186, 157) == (13, 18)
    assert T.enlarged_canvas(100, 100, 90) == (100, 100)
    assert T.canvas_offset(10, 10, 20, 30, "Top-Left") == (0, 0) and T.canvas_offset(10, 10, 20, 30, "Bottom-Right") == (10, 20)
    assert T.canvas_offset(10, 10, 5, 5, "Center") == (-2, -2)
    c = T.expand_canvas(np.ones((1, 1, 1, 3, 3), np.uint8), 5, 5, 1, 1)
    assert c.shape[-2:] == (5, 5) and c[..., 1:4, 1:4].all() and c[..., 0, :].sum() == 0


def test_resize_nearest_doubles_pixels_and_bilinear_uses_the_imagej_grid():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    assert np.array_equal(T.resize(a, 8, 6, T.NONE), np.repeat(np.repeat(a, 2, 0), 2, 1))
    ramp = np.tile(np.arange(10, dtype=np.uint8) * 10, (4, 1))
    r = T.resize(ramp, 15, 4, T.BILINEAR, average=True)
    # x = 3: source x = (3 − (7.5 + 1.5/4))/1.5 + 5 = 1.75 → 10·1.75 = 17.5 → 18
    assert r.shape == (4, 15) and r[0, 3] == 18
    f = T.resize(ramp.astype(np.float32), 15, 4, T.BILINEAR)
    assert f.dtype == np.float32 and f[0, 3] == pytest.approx(17.5)


def test_downsize_table_and_averaging():
    dt = T.DownsizeTable(8, 0, 8, 4, T.BILINEAR)
    assert dt.kernel_size == 2 and dt.src_start == 0 and dt.src_end == 7
    # every destination pixel collects weights summing to 1
    sums = np.zeros(4)
    for i, (d, w) in enumerate(zip(dt.indices, dt.weights)):
        sums[d] += float(w)
    assert np.allclose(sums, 1.0)
    const = np.full((6, 8), 77, np.uint8)
    assert np.array_equal(T.resize(const, 4, 3, T.BILINEAR, average=True), np.full((3, 4), 77, np.uint8))
    ramp = np.tile(np.arange(8, dtype=np.float32), (2, 1))
    d = T.downsize(ramp, 4, 2, T.BILINEAR)
    assert d.shape == (2, 4) and d[0, 1] == pytest.approx(2.5) and d[0, 2] == pytest.approx(4.5)  # symmetric kernel: pixel centres 2j + 0.5
    assert T.resize(ramp.astype(np.uint8) * 10, 4, 2, T.BILINEAR, average=True)[0, 1] == 25


def test_scale_in_place_keeps_the_canvas():
    a = np.zeros((8, 8), np.uint8)
    a[2:6, 2:6] = 200
    s = T.scale_in_place(a, 0.5, 0.5, T.NONE, bg=0, fill=9)
    assert s.shape == a.shape and s[0, 0] == 9 and s[3, 3] == 200 and s[4, 4] == 200 and s[2, 2] == 0
    big = T.scale_in_place(a, 2.0, 2.0, T.NONE, bg=0, fill=0)
    assert big.shape == a.shape and big[0, 0] == 200 and big[7, 7] == 200


def test_translate_rounds_half_up_and_zeroes_outside():
    a = np.tile(np.array([1, 2, 3, 4, 5], np.uint8), (4, 1))
    t = T.translate(a, 0.5, 0.0, T.BILINEAR)
    assert t[1, 1] == 2 and t[1, 2] == 3  # 1.5 → 2, 2.5 → 3 (Math.round), numpy's rint would give 2 and 2
    assert t[:, 0].tolist() == [0, 0, 0, 0]  # source x = −0.5 < 0 → 0 (getPixelInterpolated returns 0 outside)
    n = T.translate(a, 2, 1, T.NONE)
    assert n[0].tolist() == [0, 0, 0, 0, 0] and n[1].tolist() == [0, 0, 1, 2, 3]
    f = T.translate(a.astype(np.float32), -0.5, 0.0, T.BILINEAR)
    assert f[1, 0] == pytest.approx(1.5) and f[1, 4] == pytest.approx(0.0)  # source x = 4.5 ≥ w−1 → 0


# ---------------------------------------------------------------- ops and the macro layer
def test_type_8bit_uses_the_display_range_per_channel():
    s = Session()
    img = _img16(s, nc=2)
    img.display.channels[0].min, img.display.channels[0].max = 500.0, 3000.0
    img.display.channels[1].min, img.display.channels[1].max = 100.0, 5000.0
    src = img.data.copy()
    s.run_op("type.8bit", {})
    assert img.dtype == np.uint8
    assert np.array_equal(img.data[0, 0, 0], T.short_to_byte(src[0, 0, 0], 500, 3000)) and np.array_equal(img.data[0, 0, 1], T.short_to_byte(src[0, 0, 1], 100, 5000))
    assert img.display.channels[0].max == 255


def test_type_conversions_rgb_and_float():
    s = Session()
    rgb = np.zeros((3, 4, 4), np.uint8)
    rgb[0] = 255
    img = s.add_image(LumenImage(rgb, name="rgb", axes="CYX", rgb=True))
    s.run_op("type.8bit", {})
    assert img.data[0, 0, 0, 0, 0] == 85 and img.nc == 1 and not img.rgb
    img2 = s.add_image(LumenImage(rgb.copy(), name="rgb2", axes="CYX", rgb=True))
    s.run_op("type.8bit", {"weighted": True})
    assert img2.data[0, 0, 0, 0, 0] == int(0.299 * 255 + 0.5)
    img3 = s.add_image(LumenImage(rgb.copy(), name="rgb3", axes="CYX", rgb=True))
    s.run_op("type.32bit", {})
    assert img3.dtype == np.float32 and img3.data[0, 0, 0, 0, 0] == pytest.approx(85.0)
    f = s.add_image(LumenImage(np.linspace(-0.6, 2.0, 64, dtype=np.float32).reshape(8, 8), name="f"))
    f.display.channels[0].min, f.display.channels[0].max = 0.0, 1.0
    s.run_op("type.16bit", {"scale": False})
    assert f.dtype == np.uint16 and f.data.max() == 2 and f.data.min() == 0
    f2 = s.add_image(LumenImage(np.linspace(-0.6, 2.0, 64, dtype=np.float32).reshape(8, 8), name="f2"))
    f2.display.channels[0].min, f2.display.channels[0].max = -0.6, 2.0
    s.run_op("type.16bit", {})
    assert f2.data.max() == 65535 and f2.data.min() == 0


def test_rgb_color_and_apply_lut_ops():
    s = Session()
    img = s.add_image(LumenImage(np.arange(256, dtype=np.uint8).reshape(16, 16), name="g"))
    img.display.channels[0].min, img.display.channels[0].max = 50.0, 180.0
    src = img.data[0, 0, 0].copy()
    out = s.run_op("type.rgb", {"create_new": True})
    new = s.images[out["new_images"][0]]
    assert new.rgb and new.name == "g (RGB)" and np.array_equal(new.data[0, 0, 0], T.byte_display_index(src, 50, 180)) and np.array_equal(new.data[0, 0, 1], new.data[0, 0, 2])
    s.select(img.id)
    s.run_op("type.apply_lut", {})
    assert np.array_equal(img.data[0, 0, 0], T.apply_lut_table(np.uint8, 50, 180)[src].astype(np.uint8)) and img.display.channels[0].max == 255
    f = s.add_image(LumenImage(np.zeros((4, 4), np.float32), name="f"))
    with pytest.raises(ValueError, match="32-bit"):
        s.run_op("type.apply_lut", {})
    two = s.add_image(LumenImage(np.stack([np.full((4, 4), 10, np.uint8), np.full((4, 4), 200, np.uint8)]), name="two", axes="ZYX"))
    s.run_op("type.slices_to_rgb", {})
    assert two.rgb and two.data[0, 0, 0, 0, 0] == 10 and two.data[0, 0, 1, 0, 0] == 200 and two.data[0, 0, 2, 0, 0] == 0


def test_math_ops_follow_imagej_per_type(tmp_path):
    s = Session()
    img = _img16(s)
    img.display.channels[0].min, img.display.channels[0].max = 100.0, 5000.0
    src = img.data[0, 0, 0].copy()
    s.run_op("math.gamma", {"value": 0.5})
    assert np.array_equal(img.data[0, 0, 0], T.short_math(src, "gamma", 0.5, 100, 5000))
    s.run_op("math.multiply", {"value": 1.5})
    assert np.array_equal(img.data[0, 0, 0], T.short_math(T.short_math(src, "gamma", 0.5, 100, 5000), "mult", 1.5, 100, 5000))
    b = s.add_image(LumenImage(np.array([[3, 200, 10]], np.uint8), name="b"))
    s.run_op("math.multiply", {"value": 1.5})
    assert b.data[0, 0, 0].tolist() == [[5, 255, 15]]
    s.run_op("math.subtract", {"value": 4.7})
    assert b.data[0, 0, 0].tolist() == [[1, 251, 11]]  # (int)(−4.7) = −4
    s.run_op("math.divide", {"value": 2})
    assert b.data[0, 0, 0].tolist() == [[1, 126, 6]]  # Math.round(v·0.5): 0.5 → 1, 125.5 → 126, 5.5 → 6
    s.run_op("edit.invert", {})
    assert b.data[0, 0, 0].tolist() == [[254, 129, 249]]
    f = s.add_image(LumenImage(np.array([[-1.0, 0.0, 2.71]], np.float32), name="f"))
    s.run_op("math.gamma", {"value": 0.5})
    assert f.data[0, 0, 0].tolist() == [[0.0, 0.0, float(np.float32(np.exp(0.5 * np.log(np.float64(np.float32(2.71))))))]]


def test_image_rotate_scale_translate_ops():
    s = Session()
    img = s.add_image(LumenImage(np.full((2, 20, 30), 100, np.uint8), name="st", axes="ZYX"))
    s.run_op("image.rotate", {"angle": 15, "interpolation": "Bilinear", "enlarge": True})
    assert (img.height, img.width) == T.enlarged_canvas(30, 20, 15)[::-1]
    assert img.data[0, 1, 0, 0, 0] == 0 and img.data[0, 1, 0, img.height // 2, img.width // 2] == 100
    src = np.tile(np.arange(10, dtype=np.uint8) * 10, (4, 1))
    im2 = s.add_image(LumenImage(src, name="r"))
    out = s.run_op("image.scale", {"x_scale": 1.5, "y_scale": 1.0, "interpolation": "Bilinear", "create_new": True, "average_when_downsizing": True})
    new = s.images[out["new_images"][0]]
    assert new.width == 15 and new.height == 4 and new.data[0, 0, 0, 0, 3] == 18 and new.calibration.pixel_width == pytest.approx(10 / 15)
    s.select(im2.id)
    s.run_op("image.scale", {"x_scale": 0.5, "y_scale": 0.5, "interpolation": "None", "create_new": False, "keep_canvas": True})
    assert im2.width == 10 and im2.height == 4  # canvas kept
    im3 = s.add_image(LumenImage(np.tile(np.array([1, 2, 3, 4, 5], np.uint8), (4, 1)), name="t"))
    s.run_op("image.translate", {"dx": 0.5, "dy": 0.0, "interpolation": "Bilinear"})
    assert im3.data[0, 0, 0, 1, 1] == 2 and im3.data[0, 0, 0, 1, 2] == 3
    s.run_op("image.translate", {"dx": 1, "dy": 0, "interpolation": "None"})
    assert im3.data[0, 0, 0, 1, 0] == 0


def test_macro_layer_conversion_options_and_display_defaults():
    s = Session()
    ij = IJ(s)
    rgb = np.zeros((3, 4, 4), np.uint8)
    rgb[0] = 255
    s.add_image(LumenImage(rgb, name="rgb", axes="CYX", rgb=True))
    ij.run("8-bit")
    assert s.active.data[0, 0, 0, 0, 0] == 85
    s.add_image(LumenImage(rgb.copy(), name="rgb2", axes="CYX", rgb=True))
    ij.run("Conversions...", "scale weighted")
    ij.run("8-bit")
    assert s.active.data[0, 0, 0, 0, 0] == int(0.299 * 255 + 0.5)
    ij.run("Conversions...", "scale")
    # setOption("ScaleConversions", false): 16-bit → 8-bit clamps instead of scaling
    data = np.array([[100, 255, 300, 4000]], np.uint16)
    s.add_image(LumenImage(data, name="s16"))
    ij.setOption("ScaleConversions", False)
    ij.run("8-bit")
    assert s.active.data[0, 0, 0].tolist() == [[100, 255, 255, 255]]
    ij.setOption("ScaleConversions", True)
    # a freshly added 2-channel 16-bit image: ImageJ's defaults (channel 1's range for both, red / green) are applied by the conversion
    two = np.stack([np.linspace(100, 5000, 64).reshape(8, 8), np.linspace(50, 2000, 64).reshape(8, 8)]).astype(np.uint16)
    img = s.add_image(LumenImage(two, name="two", axes="CYX"))
    img.display.mode = "composite"
    ij.run("Stack to RGB")
    out = s.active
    assert out.rgb and out.name == "two (RGB)"
    assert np.array_equal(out.data[0, 0, 0], T.short_to_byte(two[0], 100, 5000)) and np.array_equal(out.data[0, 0, 1], T.short_to_byte(two[1], 100, 5000)) and out.data[0, 0, 2].max() == 0
    s.select(img.id)
    ij.run("8-bit")
    assert np.array_equal(img.data[0, 0, 1], T.short_to_byte(two[1], 100, 5000))


def test_macro_layer_math_uses_imagej_default_range_and_apply_lut():
    s = Session()
    ij = IJ(s)
    data = np.random.default_rng(3).integers(121, 4876, size=(24, 32), dtype=np.uint16)
    data[0, 0], data[0, 1] = 121, 4875
    img = s.add_image(LumenImage(data, name="g16"))
    ij.run("Gamma...", "value=0.5")
    assert np.array_equal(img.data[0, 0, 0], T.short_math(data, "gamma", 0.5, 121, 4875))
    assert img.data[0, 0, 0, 0, 1] == 4875
    b = s.add_image(LumenImage(np.arange(256, dtype=np.uint8).reshape(16, 16), name="b"))
    ij.setMinAndMax(50, 180)
    ij.run("Apply LUT")
    assert np.array_equal(b.data[0, 0, 0], T.apply_lut_table(np.uint8, 50, 180)[np.arange(256).reshape(16, 16)].astype(np.uint8))
