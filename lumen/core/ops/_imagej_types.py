"""Bit-exact ports of the ImageJ 1.54p arithmetic behind Image ▸ Type, the display / composite rendering,
Image ▸ Lookup Tables ▸ Apply LUT, Process ▸ Math and the geometric transforms.

Every function reproduces the Java of the named class/method – double where ImageJ uses ``double``, float32 where it
uses ``float``, the same operation order and ImageJ's ``(int)`` truncation / ``Math.round`` / ``+0.5`` conventions when a
result is written back into 8/16-bit pixels – so that Lumen's output is pixel-identical to ImageJ (tests/crossval).

Ported here:

* ``ij.process.TypeConverter`` – 16→8 (``(int)((v−min)·256/(max−min+1)+0.5)``), 32→8 (``create8BitImage``), 32→16,
  the unscaled variants (``setOption("ScaleConversions", false)``), RGB→8/32-bit with ``ColorProcessor``'s weighting
  factors (unweighted 1/3 by default, 0.299/0.587/0.114 with "Weighted RGB Conversions").
* the 8-bit display index of each pixel type (``ByteProcessor.setMinAndMax``, ``ShortProcessor/FloatProcessor
  .create8BitImage``), ``ImageProcessor.updateComposite`` (SUM_PROJECTION with clamping) and the channel colours of
  ``ij.CompositeImage`` – i.e. what ``RGB Color`` / ``Stack to RGB`` bake into pixels.
* ``ij.plugin.frame.ContrastAdjuster.apply`` (Apply LUT) lookup tables for 8- and 16-bit images.
* ``ByteProcessor.process`` (a 256-entry LUT), ``ShortProcessor.process`` (scaled by the display range) and
  ``FloatProcessor.process`` for Add / Subtract / Multiply / Divide / AND / OR / XOR / Min / Max / Gamma / Set / Log /
  Exp / Square / Square Root / Abs / Invert.
* ``ImageProcessor.rotate`` (centre ``(width−1)/2``, background fill, nearest / bilinear / Catmull-Rom bicubic with
  the bilinear edge fallback), ``ij.plugin.filter.Rotator.enlargeCanvas``, ``ImageProcessor.resize`` (destination
  centre offset ``+scale/4``), ``resize(w, h, useAveraging)`` → ``FloatProcessor.downsize`` + ``ij.process.DownsizeTable``,
  ``ImageProcessor.scale`` (in place, canvas kept) and ``ImageProcessor.translate``.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.float32
F64 = np.float64
_HALF32 = np.float32(0.5)

NONE, BILINEAR, BICUBIC = 0, 1, 2
INTERPOLATION = {"None": NONE, "Bilinear": BILINEAR, "Bicubic": BICUBIC}
#: ``CompositeImage.colors`` (red, green, blue, white, cyan, magenta, yellow) as Lumen LUT names
COMPOSITE_LUTS = ("Red", "Green", "Blue", "Grays", "Cyan", "Magenta", "Yellow")
#: ``ColorProcessor`` weighting factors: the default (unweighted) and Edit ▸ Options ▸ Conversions "Weighted RGB Conversions"
UNWEIGHTED = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
WEIGHTED = (0.299, 0.587, 0.114)


# ====================================================================== Java casts
def java_int(x) -> np.ndarray:
    """Java ``(int)`` of a double: truncation toward zero, NaN → 0, ±∞ → ±2³¹−1 (as int64 values)."""
    v = np.nan_to_num(np.asarray(x, dtype=F64), nan=0.0, posinf=2147483647.0, neginf=-2147483648.0)
    return np.clip(np.trunc(v), -2147483648, 2147483647).astype(np.int64)


def java_round(x) -> np.ndarray:
    """``Math.round(double)``: floor(x + 0.5) (NaN → 0)."""
    v = np.nan_to_num(np.asarray(x, dtype=F64), nan=0.0, posinf=9.2e18, neginf=-9.2e18)
    return np.floor(v + 0.5).astype(np.int64)


def _jint(v: float) -> int:
    """Java ``(int)`` of a scalar double."""
    return int(java_int(v))


def _clip_cast(values: np.ndarray, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(values, info.min, info.max).astype(dtype)


def set_pixels_from_float(fp: np.ndarray, dtype) -> np.ndarray:
    """``Byte/ShortProcessor.setPixels(channel, FloatProcessor)``: ``v + 0.5f`` in float32, clamp, ``(byte)``/``(short)``."""
    if np.dtype(dtype) == F32:
        return np.asarray(fp, dtype=F32)
    hi = float(np.iinfo(dtype).max)
    v = np.asarray(fp, dtype=F32) + _HALF32
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=hi, neginf=0.0), 0.0, hi)
    return np.trunc(v).astype(dtype)


# ====================================================================== TypeConverter
def short_to_byte(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True) -> np.ndarray:
    """``TypeConverter.convertShortToByte`` (identical to ``ShortProcessor.create8BitImage``): with scaling
    ``(int)((v − min)·256/(max − min + 1) + 0.5)`` clamped to 0–255 (min/max are the integer display range), without
    scaling the values are clamped to 255."""
    v = np.asarray(plane).astype(np.int64)
    if not scale:
        return np.minimum(v, 255).astype(np.uint8)
    mn, mx = _jint(vmin), _jint(vmax)
    sc = 256.0 / (mx - mn + 1)
    val = np.maximum(v - mn, 0).astype(F64)
    out = java_int(val * sc + 0.5)
    return np.minimum(out, 255).astype(np.uint8)


def float_to_byte(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True) -> np.ndarray:
    """``TypeConverter.convertFloatToByte``: scaled → ``FloatProcessor.create8BitImage``
    (``(int)((v − min)·255/(max − min) + 0.5)``, v < min → 0, clamped to 255); unscaled → ``ByteProcessor.setPixels(0, fp)``
    (float32 ``v + 0.5f``, clamped)."""
    if not scale:
        return set_pixels_from_float(plane, np.uint8)
    mn, mx = float(F32(vmin)), float(F32(vmax))  # FloatProcessor.min/max are floats
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = 255.0 / (mx - mn)
        val = np.maximum(np.asarray(plane).astype(F64) - mn, 0.0)
        out = java_int(val * sc + 0.5)
    return np.minimum(out, 255).astype(np.uint8)


def float_to_short(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True) -> np.ndarray:
    """``TypeConverter.convertFloatToShort``: ``(v − min)·65535/(max − min)`` (scale 1 when max == min) or the raw
    value, clamped to 0–65535, then ``(short)(value + 0.5)`` (truncation → half up)."""
    mn, mx = float(F32(vmin)), float(F32(vmax))
    sc = 1.0 if (mx - mn) == 0.0 else 65535.0 / (mx - mn)
    v = np.asarray(plane).astype(F64)
    value = (v - mn) * sc if scale else v
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=65535.0, neginf=0.0), 0.0, 65535.0)
    return np.trunc(value + 0.5).astype(np.uint16)


def rgb_to_byte(rgb: np.ndarray, weights: tuple[float, float, float] | None = None) -> np.ndarray:
    """``TypeConverter.convertRGBToByte``: ``(byte)(r·rw + g·gw + b·bw + 0.5)`` with ColorProcessor's weighting factors
    (unweighted thirds unless "Weighted RGB Conversions" is set).  ``rgb`` is (3, H, W) or (H, W, 3)."""
    r, g, b = _split_rgb(rgb)
    rw, gw, bw = weights or UNWEIGHTED
    return java_int(r * rw + g * gw + b * bw + 0.5).astype(np.uint8)


def rgb_to_float(rgb: np.ndarray, weights: tuple[float, float, float] | None = None) -> np.ndarray:
    """``TypeConverter.convertRGBToFloat``: ``(float)(r·rw + g·gw + b·bw)``."""
    r, g, b = _split_rgb(rgb)
    rw, gw, bw = weights or UNWEIGHTED
    return (r * rw + g * gw + b * bw).astype(F32)


def _split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        chans = [a[i] for i in range(3)]
    elif a.ndim == 3 and a.shape[0] == 3:
        chans = [a[i] for i in range(3)]
    else:
        chans = [a[..., i] for i in range(3)]
    return tuple(c.astype(F64) for c in chans)  # type: ignore[return-value]


# ====================================================================== display index / LUT rendering
def byte_display_index(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``ByteProcessor.setMinAndMax``: the LUT entry an 8-bit pixel maps to – ``(int)(256·(i − round(min))/(max − min))``
    clamped to 0–255, pixels below min → entry 0, above max → entry 255."""
    i = np.asarray(plane).astype(np.int64)
    if vmax < vmin:
        return i.astype(np.uint8)
    mn = int(java_round(vmin))
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.clip(java_int(256.0 * (i - mn).astype(F64) / (vmax - vmin)), 0, 255)
    idx = np.where(i < vmin, 0, np.where(i > vmax, 255, idx))
    return idx.astype(np.uint8)


def rgb_display_index(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``ColorProcessor.setMinAndMax`` per channel: ``(int)(256·(i − (int)min)/(max − min))`` clamped to 0–255."""
    i = np.asarray(plane).astype(np.int64)
    if vmax < vmin:
        return i.astype(np.uint8)
    mn = _jint(vmin)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.clip(java_int(256.0 * (i - mn).astype(F64) / (vmax - vmin)), 0, 255)
    return idx.astype(np.uint8)


def display_index(plane: np.ndarray, vmin: float, vmax: float, rgb: bool = False) -> np.ndarray:
    """The 8-bit index ImageJ's ``createImage`` / ``create8BitImage`` maps a pixel of any type to (uint8 plane)."""
    plane = np.asarray(plane)
    if plane.dtype == np.uint8:
        return rgb_display_index(plane, vmin, vmax) if rgb else byte_display_index(plane, vmin, vmax)
    if plane.dtype == np.uint16:
        return short_to_byte(plane, vmin, vmax, True)
    return float_to_byte(plane.astype(F32), vmin, vmax, True)


def lut_table(name: str, inverted: bool = False) -> np.ndarray:
    """A (256, 3) uint8 table (ImageJ's IndexColorModel; "Invert LUT" reverses it)."""
    from ..luts import get_lut
    t = np.asarray(get_lut(name), dtype=np.uint8)
    return t[::-1] if inverted else t


def render_rgb(img, t: int | None = None, z: int | None = None, mode: str | None = None, c: int | None = None) -> np.ndarray:
    """What ImageJ bakes into an RGB image for plane (t, z): ``CompositeImage.updateImage`` (every visible channel through
    its LUT at its own display range, summed with clamping at 255 – SUM_PROJECTION) in composite mode, otherwise the
    current channel's ``createImage`` (grayscale mode: the Grays ramp).  Returns (H, W, 3) uint8."""
    d = img.display
    t = d.t if t is None else t
    z = d.z if z is None else z
    mode = mode or d.mode
    H, W = img.height, img.width
    if img.rgb:
        out = np.zeros((H, W, 3), np.uint8)
        for ci in range(3):
            ch = d.channels[ci]
            out[..., ci] = rgb_display_index(img.data[t, z, ci], ch.min, ch.max)
        return out
    if mode == "composite" and img.nc > 1:
        acc = np.zeros((H, W, 3), np.int32)
        for ci, ch in enumerate(d.channels[:img.nc]):
            if not ch.visible:
                continue
            idx = display_index(img.data[t, z, ci], ch.min, ch.max)
            acc += lut_table(ch.lut, ch.inverted)[idx].astype(np.int32)
            np.minimum(acc, 255, out=acc)
        return acc.astype(np.uint8)
    ci = d.c if c is None else c
    ch = d.channels[ci]
    table = lut_table("Grays" if mode == "grayscale" else ch.lut, ch.inverted)
    return table[display_index(img.data[t, z, ci], ch.min, ch.max)]


# ====================================================================== Apply LUT (ContrastAdjuster.apply)
def apply_lut_table(dtype, vmin: float, vmax: float) -> np.ndarray:
    """``ContrastAdjuster.apply``: the lookup table that bakes the display range into 8-bit (range 256) or 16-bit
    (range 65536) pixels – ``i ≤ min → 0``, ``i ≥ max → range − 1``, else ``(int)((i − min)/(max − min)·range)``."""
    size = 65536 if np.dtype(dtype) == np.uint16 else 256
    rng = size
    mn, mx = _jint(vmin), _jint(vmax)
    i = np.arange(size, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        table = java_int((i - mn).astype(F64) / F64(mx - mn) * rng)
    table[i <= mn] = 0
    table[i >= mx] = rng - 1
    return np.clip(table, 0, size - 1).astype(np.int64)


# ====================================================================== ImageJ display-range defaults
def default_ranges(img) -> list[tuple[float, float]]:
    """The display ranges ImageJ gives a freshly opened image (no stored ranges): 8-bit 0–255; 16/32-bit the min–max
    of the first plane (``ip.resetMinAndMax()``); a multi-channel image gets that same range for *every* channel
    (``CompositeImage.setupLuts`` copies ``ip.getMin()/getMax()`` of channel 1 into each LUT)."""
    if img.data.dtype == np.uint8:
        return [(0.0, 255.0) for _ in range(img.nc)]
    first = img.data[0, 0, 0]
    finite = first[np.isfinite(first)] if first.dtype == F32 else first.ravel()
    if finite.size == 0:
        lo, hi = (0.0, 0.0)
    else:
        lo, hi = float(finite.min()), float(finite.max())
    return [(lo, hi) for _ in range(img.nc)]


def lumen_default_range(img, c: int) -> tuple[float, float]:
    """The range ``LumenImage.reset_display`` assigns (the auto-contrast of the current plane; exact min/max for float)."""
    if img.data.dtype == np.uint8:
        return 0.0, 255.0
    if img.data.dtype == np.uint16:
        return img.auto_range(c)
    return img.auto_range(c, saturated=0.0)


def imagej_display_patch(img) -> dict | None:
    """A ``Session.update_display`` patch that replaces Lumen's *untouched* open-time defaults (auto-contrast ranges,
    the Cyan/Magenta/… channel LUTs) by ImageJ's (min–max of channel 1 for every channel, red/green/blue/… composite
    colours), so that conversions, Math and RGB rendering in the macro layer start from ImageJ's state.  Ranges or LUTs
    the macro or the user changed (setMinAndMax, a LUT command, stored file ranges) are left alone.  None when nothing
    needs changing."""
    if img.rgb:
        return None
    from ..image import _default_lut_for
    ij_ranges = default_ranges(img)
    patch = []
    changed = False
    luts_default = img.nc > 1 and all(ch.lut == _default_lut_for(c, img.nc, False) and not ch.inverted
                                      for c, ch in enumerate(img.display.channels[:img.nc]))
    for c, ch in enumerate(img.display.channels[:img.nc]):
        p: dict = {}
        if img.data.dtype != np.uint8:
            lo, hi = lumen_default_range(img, c)
            if (float(ch.min), float(ch.max)) == (float(lo), float(hi)) and (float(ch.min), float(ch.max)) != ij_ranges[c]:
                p["min"], p["max"] = ij_ranges[c]
        if luts_default:
            lut = COMPOSITE_LUTS[c % len(COMPOSITE_LUTS)]
            if lut != ch.lut:
                p["lut"] = lut
        changed = changed or bool(p)
        patch.append(p)
    return {"channels": patch} if changed else None


# ====================================================================== Process ▸ Math (ImageProcessor.process)
MATH_OPS = ("invert", "add", "mult", "and", "or", "xor", "gamma", "log", "exp", "sqr", "sqrt", "abs", "min", "max", "set")


def byte_math_table(op: str, value: float = 0.0) -> np.ndarray:
    """``ByteProcessor.process``: the 256-entry table of ``op`` (``(int)`` truncation for Gamma/Log/Exp/Sqrt,
    ``Math.round`` for Multiply, ``i + (int)value`` for Add), clamped to 0–255."""
    i = np.arange(256, dtype=F64)
    scale = 255.0 / math.log(255.0)
    iv = _jint(value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "invert":
            v = 255 - i.astype(np.int64)
        elif op == "set":
            v = np.full(256, iv, np.int64)
        elif op == "add":
            v = i.astype(np.int64) + iv
        elif op == "mult":
            v = java_round(i * value)
        elif op == "and":
            v = i.astype(np.int64) & iv
        elif op == "or":
            v = i.astype(np.int64) | iv
        elif op == "xor":
            v = i.astype(np.int64) ^ iv
        elif op == "gamma":
            v = java_int(np.exp(np.log(i / 255.0) * value) * 255.0)
        elif op == "log":
            v = np.where(i == 0, 0, java_int(np.log(i) * scale))
        elif op == "exp":
            v = java_int(np.exp(i / scale))
        elif op == "sqr":
            v = i.astype(np.int64) * i.astype(np.int64)
        elif op == "sqrt":
            v = java_int(np.sqrt(i))
        elif op == "min":
            v = np.where(i < value, iv, i.astype(np.int64))
        elif op == "max":
            v = np.where(i > value, iv, i.astype(np.int64))
        else:  # abs and unknown ops leave 8-bit pixels unchanged
            v = i.astype(np.int64)
    return np.clip(v, 0, 255).astype(np.uint8)


def short_math(plane: np.ndarray, op: str, value: float, vmin: float, vmax: float) -> np.ndarray:
    """``ShortProcessor.process`` (unsigned): Gamma / Log / Exp are scaled by the *display* range ``[min, max]``
    (``getMin()/getMax()``), results ``(int)``-truncated and clamped to 0–65535."""
    v1 = np.asarray(plane).astype(np.int64)
    min2, max2 = _jint(vmin), _jint(vmax)
    rng = float(vmax) - float(vmin)
    iv = _jint(value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "invert":
            v2 = max2 - (v1 - min2)
        elif op == "set":
            v2 = np.full(v1.shape, iv, np.int64)
        elif op == "add":
            v2 = v1 + iv
        elif op == "mult":
            v2 = java_round(v1.astype(F64) * value)
        elif op == "and":
            v2 = v1 & iv
        elif op == "or":
            v2 = v1 | iv
        elif op == "xor":
            v2 = v1 ^ iv
        elif op == "gamma":
            if rng <= 0.0:
                v2 = v1
            else:
                g = java_int(np.exp(value * np.log((v1 - min2).astype(F64) / rng)) * rng + min2)
                v2 = np.where(v1 == min2, v1, g)
        elif op == "log":
            k = F64(max2) / np.log(F64(max2))  # max2/Math.log(max2): ±∞ / NaN for max2 ≤ 1 like Java
            v2 = np.where(v1 <= 0, 0, java_int(np.log(v1.astype(F64)) * k))
        elif op == "exp":
            k = np.log(F64(max2)) / F64(max2)
            v2 = java_int(np.exp(v1.astype(F64) * k))
        elif op == "sqr":
            d1 = v1.astype(F64)
            v2 = java_int(d1 * d1)
        elif op == "sqrt":
            v2 = java_int(np.sqrt(v1.astype(F64)))
        elif op == "abs":
            v2 = v1
        elif op == "min":
            v2 = np.where(v1 < value, iv, v1)
        elif op == "max":
            v2 = np.where(v1 > value, iv, v1)
        else:
            v2 = v1
    return np.clip(v2, 0, 65535).astype(np.uint16)


def float_math(plane: np.ndarray, op: str, value: float, vmin: float, vmax: float) -> np.ndarray:
    """``FloatProcessor.process``: float32 arithmetic with ``c = (float)value`` (Gamma ``exp(c·ln v)`` for v > 0 else 0,
    Log/Exp/Sqrt through ``Math`` and cast back to float, Invert ``max − (v − min)`` with the display range)."""
    v1 = np.asarray(plane).astype(F32)
    c = F32(value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "invert":
            min2, max2 = F32(vmin), F32(vmax)
            v2 = max2 - (v1 - min2)
        elif op == "set":
            v2 = np.full(v1.shape, c, F32)
        elif op == "add":
            v2 = v1 + c
        elif op == "mult":
            v2 = v1 * c
        elif op == "gamma":
            v2 = np.where(v1 <= 0, F32(0), np.exp(F64(c) * np.log(v1.astype(F64))).astype(F32))
        elif op == "log":
            v2 = np.log(v1.astype(F64)).astype(F32)
        elif op == "exp":
            v2 = np.exp(v1.astype(F64)).astype(F32)
        elif op == "sqr":
            v2 = v1 * v1
        elif op == "sqrt":
            v2 = np.where(v1 <= 0, F32(0), np.sqrt(v1.astype(F64)).astype(F32))
        elif op == "abs":
            v2 = np.abs(v1)
        elif op == "min":
            v2 = np.where(v1 < value, F32(value), v1)
        elif op == "max":
            v2 = np.where(v1 > value, F32(value), v1)
        else:
            v2 = v1
    return np.asarray(v2, dtype=F32)


def math_plane(plane: np.ndarray, op: str, value: float = 0.0, vmin: float = 0.0, vmax: float = 0.0) -> np.ndarray:
    """Process ▸ Math on one plane like ``ImageProcessor.process`` of its type (``vmin``/``vmax`` = the display range
    ImageJ's ``getMin()/getMax()`` return; only 16/32-bit Gamma/Log/Exp/Invert use it)."""
    plane = np.asarray(plane)
    if plane.dtype == np.uint8:
        return byte_math_table(op, value)[plane]
    if plane.dtype == np.uint16:
        return short_math(plane, op, value, vmin, vmax)
    return float_math(plane, op, value, vmin, vmax)


# ====================================================================== interpolation
def cubic(x: np.ndarray) -> np.ndarray:
    """``ImageProcessor.cubic`` (Catmull-Rom, a = 0.5), evaluated with Java's operation order."""
    a = 0.5
    x = np.abs(np.asarray(x, dtype=F64))
    z1 = x * x * (x * (-a + 2.0) + (a - 3.0)) + 1.0
    z2 = (-a) * x * x * x + 5.0 * a * x * x - 8.0 * a * x + 4.0 * a
    return np.where(x < 1.0, z1, np.where(x < 2.0, z2, 0.0))


def _clamp_coords(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The clamps of ``getInterpolatedPixel(x, y)`` / rotate / scale / resize: ``x < 0 → 0``, ``x ≥ w−1 → w−1.001``."""
    xs = np.where(xs < 0.0, 0.0, np.where(xs >= w - 1.0, w - 1.001, xs))
    ys = np.where(ys < 0.0, 0.0, np.where(ys >= h - 1.0, h - 1.001, ys))
    return xs, ys


def bilinear(px: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``getInterpolatedPixel(x, y, pixels)`` of Byte/Short/FloatProcessor on clamped coordinates (0 ≤ x < w−1):
    ``xbase = (int)x``, the four neighbours, ``lower + yFraction·(upper − lower)``; the float version skips NaN
    neighbours on the far side of the fraction."""
    px = np.asarray(px, dtype=F64)
    xb = np.trunc(xs).astype(np.int64)
    yb = np.trunc(ys).astype(np.int64)
    xf = xs - xb
    yf = ys - yb
    xb1 = np.minimum(xb + 1, px.shape[1] - 1)
    yb1 = np.minimum(yb + 1, px.shape[0] - 1)
    ll = px[yb, xb]
    lr = px[yb, xb1]
    ur = px[yb1, xb1]
    ul = px[yb1, xb]
    if np.isnan(px).any():
        ua = np.where(np.isnan(ul) & (xf >= 0.5), ur, np.where(np.isnan(ur) & (xf < 0.5), ul, ul + xf * (ur - ul)))
        la = np.where(np.isnan(ll) & (xf >= 0.5), lr, np.where(np.isnan(lr) & (xf < 0.5), ll, ll + xf * (lr - ll)))
        return np.where(np.isnan(la) & (yf >= 0.5), ua, np.where(np.isnan(ua) & (yf < 0.5), la, la + yf * (ua - la)))
    ua = ul + xf * (ur - ul)
    la = ll + xf * (lr - ll)
    return la + yf * (ua - la)


def bicubic(px: np.ndarray, x0: np.ndarray, y0: np.ndarray, bg: float) -> np.ndarray:
    """``ImageProcessor.getBicubicInterpolatedPixel``: a 4×4 Catmull-Rom sum (inner sum over x first) for interior
    points; at the image edge (``u0 ≤ 0 || u0 ≥ w−2 || …``) the bilinear value (clamped coordinates) when the point is
    inside ``[−1, w) × [−1, h)``, else the background value."""
    px = np.asarray(px, dtype=F64)
    h, w = px.shape
    u0 = np.floor(x0).astype(np.int64)
    v0 = np.floor(y0).astype(np.int64)
    edge = (u0 <= 0) | (u0 >= w - 2) | (v0 <= 0) | (v0 >= h - 2)
    us = np.clip(u0, 1, max(w - 3, 1))
    vs = np.clip(v0, 1, max(h - 3, 1))
    q = np.zeros(np.shape(x0), F64)
    for j in range(4):
        v = vs - 1 + j
        p = np.zeros(np.shape(x0), F64)
        for i in range(4):
            u = us - 1 + i
            p = p + px[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)] * cubic(x0 - u)
        q = q + p * cubic(y0 - v)
    inside = (x0 >= -1) & (x0 < w) & (y0 >= -1) & (y0 < h)
    cx, cy = _clamp_coords(np.asarray(x0, dtype=F64), np.asarray(y0, dtype=F64), w, h)
    fallback = np.where(inside, bilinear(px, cx, cy), F64(bg))
    return np.where(edge, fallback, q)


def _write_back(values: np.ndarray, dtype, half_up: bool = True) -> np.ndarray:
    """A double result stored into pixels of ``dtype``: ``(int)(v + 0.5)`` with clamping for 8/16-bit, ``(float)v`` for 32-bit."""
    if np.dtype(dtype) == F32:
        return np.asarray(values, dtype=F32)
    v = java_int(np.asarray(values, dtype=F64) + (0.5 if half_up else 0.0))
    return _clip_cast(v, dtype)


# ====================================================================== rotate
def rotate(plane: np.ndarray, angle: float, method: int = BILINEAR, bg: float = 0.0) -> np.ndarray:
    """``Byte/Short/FloatProcessor.rotate(angle)`` on the whole image (clockwise degrees): rotation about
    ``((w−1)/2, (h−1)/2)``; destination pixels whose source lies outside ``[−0.01, w) × [−0.01, h)`` get the background
    (``Rotator`` uses 0); nearest neighbour ``(int)(xs + 0.5)``, bilinear with clamped coordinates and ``+0.5``
    truncation, bicubic with the edge fallback and ``(int)(v + 0.5)`` clamping."""
    plane = np.asarray(plane)
    h, w = plane.shape
    if angle % 360 == 0:
        return plane.copy()
    px = plane.astype(F64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ar = -angle / (180.0 / math.pi)
    ca, sa = math.cos(ar), math.sin(ar)
    tmp1 = cy * sa - cx * ca
    tmp2 = -cx * sa - cy * ca
    y = np.arange(h, dtype=F64)[:, None]
    x = np.arange(w, dtype=F64)[None, :]
    tmp3 = tmp1 - y * sa + cx
    tmp4 = tmp2 + y * ca + cy
    xs = x * ca + tmp3
    ys = x * sa + tmp4
    if method == BICUBIC:
        return _write_back(bicubic(px, xs, ys, bg), plane.dtype)
    inside = (xs >= -0.01) & (xs < w) & (ys >= -0.01) & (ys < h)
    if method == BILINEAR:
        cxs, cys = _clamp_coords(xs, ys, w, h)
        vals = _write_back(bilinear(px, cxs, cys), plane.dtype)
    else:
        ixs = np.minimum(np.trunc(xs + 0.5).astype(np.int64), w - 1)
        iys = np.minimum(np.trunc(ys + 0.5).astype(np.int64), h - 1)
        vals = plane[np.clip(iys, 0, h - 1), np.clip(ixs, 0, w - 1)]
    bgv = np.array(bg, dtype=plane.dtype) if plane.dtype == F32 else np.array(_clip_cast(np.array([bg]), plane.dtype)[0])
    return np.where(inside, vals, bgv).astype(plane.dtype)


def enlarged_canvas(w: int, h: int, angle: float) -> tuple[int, int]:
    """``Rotator.enlargeCanvas``: the bounds of the Select-All rectangle rotated by ``angle`` about its centre
    (``RoiRotator.rotate`` with float32 vertices), rounded with ``Math.round`` and never smaller than the image."""
    theta = -angle * math.pi / 180.0
    xc, yc = w / 2.0, h / 2.0
    xs, ys = [], []
    for px_, py_ in ((0, 0), (w, 0), (w, h), (0, h)):
        dx = px_ - xc
        dy = yc - py_
        radius = math.sqrt(dx * dx + dy * dy)
        a = math.atan2(dy, dx)
        xs.append(F32(xc + radius * math.cos(a + theta)))
        ys.append(F32(yc - radius * math.sin(a + theta)))
    fw = float(F32(max(xs)) - F32(min(xs)))  # Rectangle2D.Double(minX, minY, maxX − minX, maxY − minY) in float
    fh = float(F32(max(ys)) - F32(min(ys)))
    w2 = int(math.floor(fw + 0.5))
    h2 = int(math.floor(fh + 0.5))
    return max(w2, w), max(h2, h)


def canvas_offset(w_old: int, h_old: int, w_new: int, h_new: int, position: str = "Center") -> tuple[int, int]:
    """``CanvasResizer``: the (x, y) offset of the old image inside the new canvas (``(wNew − wOld)/2`` with Java's
    integer division for centred positions)."""
    def jdiv(a: int) -> int:  # Java int division truncates toward zero
        return int(a / 2) if a >= 0 else -int(-a / 2)
    xc, xr = jdiv(w_new - w_old), w_new - w_old
    yc, yb = jdiv(h_new - h_old), h_new - h_old
    if position == "Center":
        return xc, yc
    v, _, hz = position.partition("-")
    x = {"Left": 0, "Center": xc, "Right": xr}.get(hz, xc)
    y = {"Top": 0, "Center": yc, "Bottom": yb}.get(v, yc)
    return x, y


def expand_canvas(data: np.ndarray, w_new: int, h_new: int, x_off: int, y_off: int, fill=0) -> np.ndarray:
    """``CanvasResizer.expandImage``: a filled canvas with the old planes inserted at (x_off, y_off) – negative offsets crop."""
    H, W = data.shape[-2:]
    out = np.full(data.shape[:-2] + (h_new, w_new), fill, data.dtype)
    sy0, sx0 = max(0, -y_off), max(0, -x_off)
    dy0, dx0 = max(0, y_off), max(0, x_off)
    hh = min(H - sy0, h_new - dy0)
    ww = min(W - sx0, w_new - dx0)
    if hh > 0 and ww > 0:
        out[..., dy0:dy0 + hh, dx0:dx0 + ww] = data[..., sy0:sy0 + hh, sx0:sx0 + ww]
    return out


# ====================================================================== resize / scale / downsize
class DownsizeTable:
    """``ij.process.DownsizeTable``: for every source pixel the destination pixels it contributes to and the
    (normalised, float32) kernel weights – pixel centres aligned so that the left edges of source and destination
    coincide; out-of-range source pixels are replaced by the edge pixel."""

    KERNEL_SIZES = (1, 2, 4)
    UNUSED = -1

    def __init__(self, src_size: int, src_origin: int, src_length: int, dst_size: int, method: int) -> None:
        self.src_origin = src_origin
        self.method = method
        self.scale = src_length / float(dst_size)
        ks = self.KERNEL_SIZES[method]
        self.kernel_size = ks
        start = int(math.ceil(1e-8 + self._src_index(-0.5 * ks)))
        self.src_start = 0 if start < 0 else start
        end = int(math.floor(1e-8 + self._src_index(dst_size - 1 + 0.5 * ks)))
        self.src_end = src_size - 1 if end >= src_size else end
        n = (self.src_end - self.src_start + 1) * ks
        indices = [self.UNUSED] * n
        weights = [F32(0.0)] * n
        for dst in range(dst_size):
            total = 0.0
            lowest = int(math.ceil(1e-8 + self._src_index(dst - 0.5 * ks)))
            highest = int(math.floor(-1e-8 + self._src_index(dst + 0.5 * ks)))
            for src in range(lowest, highest + 1):
                s = 0 if src < 0 else (src_size - 1 if src >= src_size else src)
                p = (s - self.src_start) * ks
                while indices[p] != self.UNUSED and indices[p] != dst:
                    p += 1
                indices[p] = dst
                weight = self._kernel(dst - self._dst_index(src))
                total += float(weight)
                weights[p] = F32(weights[p] + weight)
            i_start = max((lowest - self.src_start) * ks, 0)
            i_stop = min((highest - self.src_start) * ks + (ks - 1), n - 1)
            for i in range(i_start, i_stop + 1):
                if indices[i] == dst:
                    weights[i] = F32(float(weights[i]) / total)
        self.indices = [0 if i == self.UNUSED else i for i in indices]
        self.weights = weights

    def _src_index(self, dst_index: float) -> float:
        return self.src_origin - 0.5 + (dst_index + 0.5) * self.scale

    def _dst_index(self, src_index: int) -> float:
        return (src_index - self.src_origin + 0.5) / self.scale - 0.5

    def _kernel(self, x: float) -> np.float32:
        if self.method == NONE:
            return F32(1.0)
        if self.method == BILINEAR:
            return F32(F32(1.0) - F32(abs(x)))
        return F32(cubic(np.asarray(x))[()])


def downsize_1d(fp: np.ndarray, dst_w: int, dst_h: int, x_direction: bool, method: int) -> np.ndarray:
    """``FloatProcessor.downsize1D``: accumulate ``v·weight`` in float32, source pixels in increasing order."""
    fp = np.asarray(fp, dtype=F32)
    h, w = fp.shape
    out = np.zeros((dst_h, dst_w), F32)
    ks_table = DownsizeTable(w if x_direction else h, 0, w if x_direction else h, dst_w if x_direction else dst_h, method)
    ks = ks_table.kernel_size
    for s in range(ks_table.src_start, ks_table.src_end + 1):
        v = fp[:dst_h, s] if x_direction else fp[s, :dst_w]
        base = (s - ks_table.src_start) * ks
        for k in range(ks):
            wgt = ks_table.weights[base + k]
            d = ks_table.indices[base + k]
            if x_direction:
                out[:, d] += v * wgt
            else:
                out[d, :] += v * wgt
    return out


def downsize(fp: np.ndarray, dst_w: int, dst_h: int, method: int) -> np.ndarray:
    """``FloatProcessor.downsize``: average in x, then in y, then upsize any remaining direction with ``resize``."""
    ip2 = np.asarray(fp, dtype=F32)
    h, w = ip2.shape
    if dst_w < w:
        ip2 = downsize_1d(ip2, dst_w, h, True, method)
    if dst_h < h:
        ip2 = downsize_1d(ip2, ip2.shape[1], dst_h, False, method)
    if ip2.shape != (dst_h, dst_w):
        ip2 = resize(ip2, dst_w, dst_h, method, False)
    return ip2


def resize(plane: np.ndarray, dst_w: int, dst_h: int, method: int = BILINEAR, average: bool = False, bg: float = 0.0) -> np.ndarray:
    """``ImageProcessor.resize(dstWidth, dstHeight, useAveraging)`` of the whole image: downsizing with averaging goes
    through ``FloatProcessor.downsize`` (+ ``setPixels(channel, fp)`` rounding), otherwise ``resize(w, h)`` – source
    coordinate ``(x − dstCenter)/scale + srcCenter`` with ``dstCenter = dst/2 + scale/4`` for interpolating methods."""
    plane = np.asarray(plane)
    h, w = plane.shape
    if (dst_w, dst_h) == (w, h):
        return plane.copy()
    if average and (dst_w < w or dst_h < h):
        return set_pixels_from_float(downsize(plane.astype(F32), dst_w, dst_h, method), plane.dtype)
    if (w == 1 or h == 1) and method != NONE:
        return _resize_linearly(plane, dst_w, dst_h)
    px = plane.astype(F64)
    src_cx, src_cy = w / 2.0, h / 2.0
    dst_cx, dst_cy = dst_w / 2.0, dst_h / 2.0
    x_scale, y_scale = dst_w / float(w), dst_h / float(h)
    if method != NONE:
        if dst_w != w:
            dst_cx += x_scale / 4.0
        if dst_h != h:
            dst_cy += y_scale / 4.0
    x = np.arange(dst_w, dtype=F64)[None, :]
    y = np.arange(dst_h, dtype=F64)[:, None]
    xs = np.broadcast_to((x - dst_cx) / x_scale + src_cx, (dst_h, dst_w))
    ys = np.broadcast_to((y - dst_cy) / y_scale + src_cy, (dst_h, dst_w))
    if method == BICUBIC:
        return _write_back(bicubic(px, xs, ys, bg), plane.dtype)
    if method == BILINEAR:
        cxs, cys = _clamp_coords(xs, ys, w, h)
        return _write_back(bilinear(px, cxs, cys), plane.dtype)
    ix = np.clip(np.trunc(xs).astype(np.int64), 0, w - 1)
    iy = np.clip(np.trunc(ys).astype(np.int64), 0, h - 1)
    return plane[iy, ix]


def _resize_linearly(plane: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """``ImageProcessor.resizeLinearly`` for images one pixel wide or high."""
    plane = np.asarray(plane)
    rotate_ = plane.shape[1] == 1
    row = (plane[:, 0] if rotate_ else plane[0]).astype(F64)
    n1 = row.size
    n2 = dst_h if rotate_ else dst_w
    scale = (n1 - 1) / float(n2 - 1) if n2 > 1 else 0.0
    out = np.zeros(n2, F32)
    for x in range(n2):
        x1 = int(x * scale)
        x2 = min(x1 + 1, n1 - 1)
        frac = x * scale - x1
        out[x] = F32((1.0 - frac) * row[x1] + frac * row[x2])
    arr = np.tile(out[:, None], (1, dst_w)) if rotate_ else np.tile(out[None, :], (dst_h, 1))
    return set_pixels_from_float(arr, plane.dtype) if plane.dtype != F32 else arr.astype(F32)


def scale_in_place(plane: np.ndarray, x_scale: float, y_scale: float, method: int = BILINEAR, bg: float = 0.0, fill: float = 0.0) -> np.ndarray:
    """``ImageProcessor.scale(xScale, yScale)`` on the whole image: the content is scaled about the image centre and the
    canvas is kept; when shrinking, destination pixels whose source falls outside are filled (8-bit: the background
    value, 16/32-bit: the display minimum – ``fill``)."""
    plane = np.asarray(plane)
    h, w = plane.shape
    xc, yc = w / 2.0, h / 2.0
    if x_scale > 1.0 and y_scale > 1.0:
        xmin = max(_jint(xc - xc * x_scale), 0)
        xmax = min(xmin + _jint(w * x_scale) - 1, w - 1)
        ymin = max(_jint(yc - yc * y_scale), 0)
        ymax = min(ymin + _jint(h * y_scale) - 1, h - 1)
    else:
        xmin, xmax, ymin, ymax = 0, w - 1, 0, h - 1
    out = plane.copy()
    px = plane.astype(F64)
    x = np.arange(xmin, xmax + 1, dtype=F64)[None, :]
    y = np.arange(ymin, ymax + 1, dtype=F64)[:, None]
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    xs = np.broadcast_to((x - xc) / x_scale + xc, shape)
    ys = np.broadcast_to((y - yc) / y_scale + yc, shape)
    if method == BICUBIC:
        out[ymin:ymax + 1, xmin:xmax + 1] = _write_back(bicubic(px, xs, ys, bg), plane.dtype)
        return out
    check = x_scale < 1.0 or y_scale < 1.0
    xsi = np.trunc(xs).astype(np.int64)
    ysi = np.trunc(ys).astype(np.int64)
    cxs, cys = _clamp_coords(xs, ys, w, h)
    if method == BILINEAR:
        vals = _write_back(bilinear(px, cxs, cys), plane.dtype)
    else:
        vals = plane[np.trunc(cys).astype(np.int64), np.clip(xsi, 0, w - 1)]
    if check:
        outside = (xsi < xmin) | (xsi > xmax) | (ysi < ymin) | (ysi > ymax)
        fv = np.array(fill, dtype=F32) if plane.dtype == F32 else np.array(_clip_cast(np.array([fill]), plane.dtype)[0])
        vals = np.where(outside, fv, vals).astype(plane.dtype)
    out[ymin:ymax + 1, xmin:xmax + 1] = vals
    return out


def translate(plane: np.ndarray, dx: float, dy: float, method: int = NONE) -> np.ndarray:
    """``ImageProcessor.translate(xOffset, yOffset)``: integer offsets (or no interpolation) copy pixels with 0 outside;
    fractional offsets use ``getPixelInterpolated`` – bilinear ``Math.round`` of the value, 0 when the source is outside
    ``[0, w−1) × [0, h−1)``; bicubic ``(int)(v + 0.5)`` with background 0."""
    plane = np.asarray(plane)
    h, w = plane.shape
    integer = dx == int(dx) and dy == int(dy)
    if integer or method == NONE:
        ix, iy = int(dx), int(dy)
        out = np.zeros_like(plane)
        sx0, sy0 = max(0, -ix), max(0, -iy)
        dx0, dy0 = max(0, ix), max(0, iy)
        ww, hh = w - abs(ix), h - abs(iy)
        if ww > 0 and hh > 0:
            out[dy0:dy0 + hh, dx0:dx0 + ww] = plane[sy0:sy0 + hh, sx0:sx0 + ww]
        return out
    px = plane.astype(F64)
    y = np.arange(h, dtype=F64)[:, None]
    x = np.arange(w, dtype=F64)[None, :]
    xs = np.broadcast_to(x - dx, (h, w))
    ys = np.broadcast_to(y - dy, (h, w))
    if method == BICUBIC:
        return _write_back(bicubic(px, xs, ys, 0.0), plane.dtype)
    inside = (xs >= 0.0) & (ys >= 0.0) & (xs < w - 1) & (ys < h - 1)
    cxs, cys = _clamp_coords(xs, ys, w, h)
    v = bilinear(px, cxs, cys)
    if plane.dtype == F32:
        return np.where(inside, v, 0.0).astype(F32)
    vals = _clip_cast(java_round(v), plane.dtype)
    return np.where(inside, vals, 0).astype(plane.dtype)
