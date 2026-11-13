"""Automatic thresholds the way ImageJ 1.54p computes them.

The 16 methods of Image ▸ Adjust ▸ Threshold are exact ports of ``ij.process.AutoThresholder``
(:mod:`lumen.core.ij_autothreshold`).  This module adds what ``ImageProcessor.setAutoThreshold`` does around
them: the 256-bin histogram (ImageJ's 8-bit conversion for 16-bit and float planes), the dark/light side
selection, the ``addOne`` rule (the macro's ``setAutoThreshold()`` selects level + 1 and above, the Thresholder's
per-slice conversion the level itself) and the scaling of the levels back to 16/32-bit values
(``scaleAndSetThreshold``: 0 / 65535 for 16-bit and ±1e30 for 32-bit at the open end, 16-bit values rounded by
``ShortProcessor.setThreshold``), plus the legacy ``ISODATA`` / ``ISODATA2`` auto threshold the Thresholder falls
back to when no threshold is set.
"""
from __future__ import annotations

import math

import numpy as np

from . import ij_autothreshold as _ij

METHODS = ["Default", "Huang", "Intermodes", "IsoData", "Li", "MaxEntropy", "Mean", "MinError",
           "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy", "Shanbhag", "Triangle", "Yen"]

FLOAT_OPEN_END = 1e30   # ImageJ's "no limit" for 32-bit thresholds (scaleAndSetThreshold)


def ij_default(h: np.ndarray) -> int:
    """``ImageProcessor.getAutoThreshold(int[])`` (= AutoThresholder.IJIsoData): the iterative intermeans on a
    256-bin histogram ignoring bins 0 and 255.  The 'Default' *method* additionally cuts the mode bin
    (:func:`lumen.core.ij_autothreshold.default_isodata`)."""
    return _ij.ij_isodata(h)


huang, intermodes, isodata, li, max_entropy, mean, min_error, minimum, moments, otsu, percentile, renyi_entropy, shanbhag, triangle, yen = (
    _ij.huang, _ij.intermodes, _ij.isodata, _ij.li, _ij.max_entropy, _ij.mean, _ij.min_error, _ij.minimum, _ij.moments, _ij.otsu,
    _ij.percentile, _ij.renyi_entropy, _ij.shanbhag, _ij.triangle, _ij.yen)

_FUNCS = {m: _ij.METHODS[m] for m in METHODS}


def histogram256(plane: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """256-bin histogram over the data range (full range for 8-bit) – the threshold panel's display histogram
    (equal-width bins between the plane's min and max); the thresholds themselves use :func:`ij_histogram256`."""
    vals = plane[mask] if mask is not None else plane.ravel()
    if plane.dtype == np.uint8:
        lo, hi = 0.0, 255.0
        h = np.bincount(vals.astype(np.int64), minlength=256)[:256]
        return h.astype(np.int64), lo, hi
    vals = vals[np.isfinite(vals)] if vals.dtype.kind == "f" else vals
    if vals.size == 0:
        return np.zeros(256, np.int64), 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1
    h, _ = np.histogram(vals, bins=256, range=(lo, hi))
    return h.astype(np.int64), lo, hi


def to_byte_ij(plane: np.ndarray) -> tuple[np.ndarray, float, float]:
    """ImageJ's ``convertToByte(true)`` of a 16-bit / float plane scaled by the plane's own min and max
    (``setAutoThreshold`` resets the display range first): 16-bit ``(int)((v-min)·256/(max-min+1) + 0.5)``,
    float ``(int)((v-min)·255/(max-min) + 0.5)``, both clipped to 0…255.  Returns (8-bit plane, min, max)."""
    if plane.dtype == np.uint8:
        return plane, 0.0, 255.0
    if plane.dtype.kind == "f":
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            return np.zeros(plane.shape, np.uint8), 0.0, 0.0
        mn, mx = float(finite.min()), float(finite.max())
        if mx <= mn:
            return np.zeros(plane.shape, np.uint8), mn, mx
        scale = 255.0 / (mx - mn)
        v = plane.astype(np.float64) - mn
        v[v < 0] = 0.0
        v[~np.isfinite(v)] = 0.0
        iv = np.floor(v * scale + 0.5)
        return np.clip(iv, 0, 255).astype(np.uint8), mn, mx
    mn, mx = int(plane.min()), int(plane.max())
    if mx <= mn:
        return np.zeros(plane.shape, np.uint8), float(mn), float(mx)
    scale = 256.0 / (mx - mn + 1)
    v = plane.astype(np.int64) - mn
    v[v < 0] = 0
    iv = np.floor(v.astype(np.float64) * scale + 0.5)
    return np.clip(iv, 0, 255).astype(np.uint8), float(mn), float(mx)


def ij_histogram256(plane: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """The histogram ImageJ's auto threshold works on: the 8-bit plane (:func:`to_byte_ij`) inside the ROI mask;
    min/max are those of the whole plane (ImageJ scales with the processor's range, not the ROI's)."""
    p8, mn, mx = to_byte_ij(plane)
    vals = p8[mask] if mask is not None else p8.ravel()
    h = np.bincount(vals.astype(np.int64), minlength=256)[:256]
    return h.astype(np.int64), mn, mx


def scale_threshold(lower8: float, upper8: float, mn: float, mx: float, dtype: np.dtype) -> tuple[float, float]:
    """``ImageProcessor.scaleAndSetThreshold`` followed by ``ShortProcessor`` / ``FloatProcessor.setThreshold``:
    map 0…255 levels to image values.  A lower level of 0 or an upper level of 255 is an open end (0 / 65535 for
    16-bit, ±1e30 for float); 16-bit (and other integer) values are rounded half up."""
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return float(lower8), float(upper8)
    is_float = dtype.kind == "f"
    if mx > mn:
        if lower8 == 0.0:
            lower = min(mn, -FLOAT_OPEN_END) if is_float else 0.0
        else:
            lower = mn + (lower8 / 255.0) * (mx - mn)
        if upper8 == 255.0:
            upper = max(mx, FLOAT_OPEN_END) if is_float else (65535.0 if dtype == np.uint16 else float(np.iinfo(dtype).max))
        else:
            upper = mn + (upper8 / 255.0) * (mx - mn)
    else:
        lower = upper = mn
    if is_float:
        return float(lower), float(upper)
    if dtype == np.uint16:
        lower = max(lower, 0.0)
        upper = min(upper, 65535.0)
    return float(math.floor(lower + 0.5)), float(math.floor(upper + 0.5))


def auto_threshold(plane: np.ndarray, method: str = "Default", mask: np.ndarray | None = None,
                   dark_background: bool = True, add_one: bool = True) -> tuple[float, float]:
    """(lower, upper) threshold in image units – ``ImageProcessor.setAutoThreshold(Method, darkBackground, …)``.

    ``add_one`` is ImageJ's ``addOne`` flag: the macro function ``setAutoThreshold("… dark")`` and the threshold
    panel select pixels from level + 1 (``add_one=True``); Process ▸ Binary ▸ Convert to Mask on a stack
    ('calculate threshold for each image') calls the same code without it and selects from the level itself.
    Light background keeps 0 … level in both cases.  The histogram is never flipped (ImageJ computes the level on
    the plain histogram and only chooses the side), so asymmetric methods such as Li give ImageJ's numbers."""
    if method not in _FUNCS:
        raise ValueError(f"Unknown threshold method {method!r}; choose from {METHODS}")
    h, mn, mx = ij_histogram256(plane, mask)
    t = _ij.get_threshold(method, h, subtract_one=add_one)
    one = 1 if add_one else 0
    if dark_background:
        lower8, upper8 = float(t + one), 255.0
    else:
        lower8, upper8 = 0.0, float(t)
    if lower8 > 255:
        lower8 = 255.0
    return scale_threshold(lower8, upper8, mn, mx, plane.dtype)


def _ij_gaussian_kernel(sigma: float, accuracy: float, max_radius: int) -> np.ndarray:
    """``GaussianBlur.makeGaussianKernel`` (the one-sided kernel with ImageJ's edge correction), normalised."""
    k_radius = int(math.ceil(sigma * math.sqrt(-2 * math.log(accuracy)))) + 1
    if max_radius < 50:
        max_radius = 50
    if k_radius > max_radius:
        k_radius = max_radius
    kernel = np.zeros(k_radius, np.float64)
    for i in range(k_radius):
        kernel[i] = math.exp(-0.5 * i * i / sigma / sigma)
    if k_radius < max_radius and k_radius > 3:
        sqrt_slope = 1.7976931348623157e308
        r = k_radius
        while r > k_radius // 2:
            r -= 1
            a = math.sqrt(kernel[r]) / (k_radius - r)
            if a < sqrt_slope:
                sqrt_slope = a
            else:
                break
        for r1 in range(r + 2, k_radius):
            kernel[r1] = (k_radius - r1) * (k_radius - r1) * sqrt_slope * sqrt_slope
    s = kernel[0] + 2 * kernel[1:].sum()
    return kernel / s


def _ij_blur_line(values: np.ndarray, sigma: float = 2.0, accuracy: float = 0.01) -> np.ndarray:
    """``GaussianBlur.blur1Direction`` of a single line: convolution with the ImageJ kernel, out-of-range
    samples replaced by the edge values."""
    k = _ij_gaussian_kernel(sigma, accuracy, len(values))
    r = len(k) - 1
    padded = np.pad(values.astype(np.float64), r, mode="edge")
    full = np.concatenate([k[:0:-1], k])
    return np.convolve(padded, full, mode="valid")


def legacy_isodata_threshold(plane: np.ndarray, mask: np.ndarray | None = None, variant: int = 2) -> tuple[float, float]:
    """``ImageProcessor.setAutoThreshold(ISODATA | ISODATA2, …)`` – what Convert to Mask / Make Binary use when no
    threshold is set (and the stack dialog with method and background 'Default').  ISODATA2 cuts the mode bin to
    1.5× the second highest bin; the level is ``getAutoThreshold`` (IJIsoData) and the *side* is chosen from the
    histogram: with a dominant mode (mode of the σ=2-blurred histogram > 1.5× the mean bin) the objects are on the
    far side of the mode, otherwise the dark side; the lower limit is the level itself (no +1)."""
    h, mn, mx = ij_histogram256(plane, mask)
    hist = h.copy()
    mode = int(np.argmax(hist))
    max_count = int(hist[mode])
    work = hist.copy()
    if variant == 2:
        others = np.delete(hist, mode)
        max_count2 = int(others.max()) if others.size else 0
        if max_count > max_count2 * 2 and max_count2 != 0:
            work[mode] = int(max_count2 * 1.5)
    t = _ij.ij_isodata(work)
    blurred = _ij_blur_line(hist.astype(np.float32).astype(np.float64))
    total = float(blurred.sum())
    mode16 = int(np.argmax(blurred))
    max_blur = float(blurred[mode16])
    mean_bin = total / 256.0
    p8 = to_byte_ij(plane)[0]
    vals = p8[mask] if mask is not None else p8.ravel()
    smin, smax = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 0.0)
    if mean_bin > 0 and max_blur / mean_bin > 1.5:
        bright = (smax - mode16) > (mode16 - smin)
    else:
        bright = False  # ImageJ: isInvertedLut(); Lumen has no inverting LUTs
    lower8, upper8 = (float(t), 255.0) if bright else (0.0, float(t))
    return scale_threshold(lower8, upper8, mn, mx, plane.dtype)
