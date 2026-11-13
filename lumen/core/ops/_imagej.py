"""Bit-exact ports of the ImageJ 1.54p filter algorithms behind the Process menu.

Every function reproduces the Java arithmetic of the named class/method – float32 where ImageJ uses ``float``,
double where it uses ``double``, the same operation order, and ImageJ's rounding or truncation when a float result
is written back into 8/16-bit pixels – so that Lumen's output is pixel-identical to ImageJ (verified by
tests/crossval).  Ported: ``ij.plugin.filter.RankFilters`` (circular kernel, mean / variance / top hat),
``ij.process.{Byte,Short,Float}Processor.filter`` and ``convolve3x3`` (Smooth, Sharpen, Find Edges),
``ij.plugin.filter.Convolver``, ``ij.plugin.filter.GaussianBlur`` (kernel with edge correction, edge handling,
down/upscaling for large sigma), ``ij.plugin.filter.UnsharpMask``, ``ij.plugin.ContrastEnhancer`` (saturation
limits from ImageJ's histograms, normalize LUTs), ``ij.plugin.filter.BackgroundSubtracter`` (rolling ball with
shrink / enlarge, sliding paraboloid with corner correction), ``ij.plugin.FFT.pad`` / ``ij.process.FHT.getPowerSpectrum``
and ``ij.plugin.filter.FFTFilter`` (bandpass).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

F32 = np.float32
_HALF = np.float32(0.5)


# ====================================================================== rounding helpers
def round_half_up(values: np.ndarray, dtype) -> np.ndarray:
    """ImageJ's ``(int)(v + 0.5f)`` with clamping to the type range (``ByteProcessor/ShortProcessor.setPixels(channel, fp)``,
    ``RankFilters.writeLineToPixels``, ``TypeConverter`` without scaling): the 0.5 is added in float32, then truncated."""
    v = np.asarray(values, dtype=np.float32) + _HALF
    v = np.nan_to_num(v, nan=0.0, posinf=np.iinfo(dtype).max, neginf=0.0)
    return np.trunc(np.clip(v, 0, np.iinfo(dtype).max)).astype(dtype)


def to_dtype(values: np.ndarray, dtype) -> np.ndarray:
    """A float32 filter result written back into an image of ``dtype`` like ImageJ (float images keep the float)."""
    if np.dtype(dtype) == np.float32:
        return np.asarray(values, dtype=np.float32)
    return round_half_up(values, dtype)


# ====================================================================== RankFilters
def rank_kernel(radius: float) -> np.ndarray:
    """``RankFilters.makeLineRadii``: boolean mask of the circular kernel – pixel (dx, dy) belongs to it when
    dx² + dy² ≤ (int)(r²) + 1 (21 pixels for r = 2, 37 for r = 3), with ImageJ's remapping of the radii 1.5…1.75 → 1.75
    and 2.5…2.85 → 2.85 (kept for compatibility with older ImageJ versions)."""
    r = float(radius)
    if 1.5 <= r < 1.75:
        r = 1.75
    elif 2.5 <= r < 2.85:
        r = 2.85
    r2 = int(r * r) + 1
    k = int(math.sqrt(r2 + 1e-10))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    dx = np.floor(np.sqrt(np.maximum(r2 - yy * yy, 0) + 1e-10)).astype(np.int64)
    return np.abs(xx) <= dx


def _nan_aware_median(plane: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """``RankFilters.getNaNAwareMedian``: NaNs are ignored; NaN only where the whole neighbourhood is NaN."""
    k = fp.shape[0] // 2
    pad = np.pad(plane.astype(np.float32), k, mode="edge")
    H, W = plane.shape
    offs = np.argwhere(fp) - k
    stack = np.stack([pad[k + dy:k + dy + H, k + dx:k + dx + W] for dy, dx in offs])  # (n, H, W)
    n_nan = np.isnan(stack).sum(axis=0)
    n = stack.shape[0] - n_nan
    s = np.sort(stack, axis=0)                      # NaNs sort to the end
    half = n // 2
    out = np.take_along_axis(s, np.minimum(half, stack.shape[0] - 1)[None], axis=0)[0]
    return np.where(n == 0, np.float32(np.nan), out).astype(np.float32)


def rank_filter(plane: np.ndarray, radius: float, kind: str) -> np.ndarray:
    """``RankFilters.rank`` for MEAN / MIN / MAX / VARIANCE / MEDIAN with ImageJ's circular kernel and edge
    replication.  Mean and variance are summed in double and written as ImageJ does: float32 for float images,
    ``(int)(v + 0.5f)`` for 8/16-bit (variance clipped to the type maximum, never negative)."""
    fp = rank_kernel(radius)
    n = int(fp.sum())
    dt = plane.dtype
    is_float = dt == np.float32
    if kind == "min":
        return ndi.minimum_filter(plane, footprint=fp, mode="nearest")
    if kind == "max":
        return ndi.maximum_filter(plane, footprint=fp, mode="nearest")
    if kind == "median":
        if is_float and np.isnan(plane).any():
            return _nan_aware_median(plane, fp)
        if dt == np.uint8 and n > 400:          # large kernels: histogram-based median (exact for odd kernel sizes)
            from skimage.filters import rank
            return rank.median(plane, fp)
        return ndi.median_filter(plane, footprint=fp, mode="nearest")
    f = plane.astype(np.float64)
    w = fp.astype(np.float64)
    s = ndi.correlate(f, w, mode="nearest")
    if kind == "mean":
        v = (s / n).astype(np.float32)
    elif kind == "variance":
        s2 = ndi.correlate(f * f, w, mode="nearest")
        v = ((s2 - s * s / n) / n).astype(np.float32)
        if not is_float:
            v = np.minimum(v, np.float32(np.iinfo(dt).max))
        v = np.maximum(v, np.float32(0))
    else:
        raise ValueError(f"unknown rank filter {kind!r}")
    return to_dtype(v, dt)


def top_hat(plane: np.ndarray, radius: float, light_background: bool = False, dont_subtract: bool = False) -> np.ndarray:
    """``RankFilters`` TOP_HAT: the image minus its grayscale opening (minimum then maximum with the circular kernel);
    with a light background the closing is subtracted and the type maximum added (ImageJ's ``offset``).
    ``dont_subtract`` returns the opening / closing itself ("Don't subtract (grayscale open)")."""
    fp = rank_kernel(radius)
    if light_background:
        opened = ndi.minimum_filter(ndi.maximum_filter(plane, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        offset = 0.0 if plane.dtype == np.float32 else float(np.iinfo(plane.dtype).max)
    else:
        opened = ndi.maximum_filter(ndi.minimum_filter(plane, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        offset = 0.0
    if dont_subtract:
        return opened
    out = (plane.astype(np.float32) - opened.astype(np.float32)) + np.float32(offset)
    return to_dtype(out, plane.dtype)


# ====================================================================== 3×3 filters (ImageProcessor.filter / convolve3x3)
def _neighbours3x3(plane: np.ndarray) -> list[np.ndarray]:
    """p1 … p9 (row-major 3×3 neighbourhood) with edge pixels replicated (``getEdgePixel`` / the pointer clamping of
    ``ShortProcessor.filter3x3`` and ``FloatProcessor.filter3x3``)."""
    p = np.pad(plane, 1, mode="edge")
    H, W = plane.shape
    return [p[y:y + H, x:x + W] for y in (0, 1, 2) for x in (0, 1, 2)]


def _trunc_div(a: np.ndarray, d: int) -> np.ndarray:
    """Java integer division (truncation toward zero)."""
    return np.where(a >= 0, a // d, -((-a) // d))


def smooth3x3(plane: np.ndarray) -> np.ndarray:
    """``filter(BLUR_MORE)``: ``(p1+…+p9+4)/9`` for 8/16-bit, ``(v1+…+v9)*0.11111111f`` (float32, left to right) for float."""
    p = _neighbours3x3(plane)
    if plane.dtype == np.float32:
        s = p[0]
        for q in p[1:]:
            s = s + q
        return s * np.float32(0.11111111)
    s = sum(q.astype(np.int64) for q in p)
    return ((s + 4) // 9).astype(plane.dtype)


_SHARPEN = (-1, -1, -1, -1, 12, -1, -1, -1, -1)


def sharpen3x3(plane: np.ndarray) -> np.ndarray:
    """``ImageProcessor.sharpen`` = ``convolve3x3({-1,-1,-1,-1,12,-1,-1,-1,-1})``: integer ``(sum + 2)/4`` (truncating)
    clamped to the type range for 8/16-bit; float32 ``(k1*v1 + … + k9*v9) * 0.25f`` for float."""
    p = _neighbours3x3(plane)
    if plane.dtype == np.float32:
        s = np.float32(_SHARPEN[0]) * p[0]
        for k, q in zip(_SHARPEN[1:], p[1:]):
            s = s + np.float32(k) * q
        return s * np.float32(0.25)
    s = sum(k * q.astype(np.int64) for k, q in zip(_SHARPEN, p))
    s = _trunc_div(s + 2, 4)
    return np.clip(s, 0, np.iinfo(plane.dtype).max).astype(plane.dtype)


def find_edges3x3(plane: np.ndarray) -> np.ndarray:
    """``filter(FIND_EDGES)``: Sobel magnitude; truncated ``(int)Math.sqrt`` for 8/16-bit (clamped), float32 for float."""
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = _neighbours3x3(plane)
    if plane.dtype == np.float32:
        two = np.float32(2)
        s1 = ((((p1 + two * p2) + p3) - p7) - two * p8) - p9
        s2 = ((((p1 + two * p4) + p7) - p3) - two * p6) - p9
        return np.sqrt(s1 * s1 + s2 * s2).astype(np.float32)
    i = [q.astype(np.int64) for q in (p1, p2, p3, p4, p5, p6, p7, p8, p9)]
    s1 = i[0] + 2 * i[1] + i[2] - i[6] - 2 * i[7] - i[8]
    s2 = i[0] + 2 * i[3] + i[6] - i[2] - 2 * i[5] - i[8]
    mag = np.floor(np.sqrt((s1 * s1 + s2 * s2).astype(np.float64)))
    return np.clip(mag, 0, np.iinfo(plane.dtype).max).astype(plane.dtype)


# ====================================================================== Convolver
def convolve(plane: np.ndarray, kernel: np.ndarray, normalize: bool = True) -> np.ndarray:
    """``Convolver.convolveFloat``: every product ``pixel * kernel[i]`` is a float32 product, the products are summed in
    double in kernel order, the sum is multiplied by ``scale`` (1 / kernel sum, double) and cast to float32; out-of-image
    pixels are the nearest edge pixels.  8/16-bit results are rounded half up (``convertToByte(false)``: float32
    ``v + 0.5f``; ``convertToShort(false)``: double ``v + 0.5``)."""
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    scale = 1.0
    if normalize:
        total = 0.0
        for v in k.ravel():          # Convolver.getScale: double accumulation of the float coefficients
            total += float(v)
        if total != 0.0:
            scale = 1.0 / total
    f = plane.astype(np.float32)
    vc, uc = kh // 2, kw // 2
    pad = np.pad(f, ((vc, vc), (uc, uc)), mode="edge")
    H, W = plane.shape
    acc = np.zeros((H, W), np.float64)
    for v in range(kh):
        for u in range(kw):
            acc += pad[v:v + H, u:u + W] * k[v, u]          # float32 product, double accumulation
    out = (acc * scale).astype(np.float32)
    if plane.dtype == np.float32:
        return out
    if plane.dtype == np.uint8:
        return round_half_up(out, np.uint8)
    d = np.clip(out.astype(np.float64), 0.0, 65535.0) + 0.5     # TypeConverter.convertFloatToShort (double)
    return np.trunc(d).astype(np.uint16)


# ====================================================================== GaussianBlur
def make_gaussian_kernel(sigma: float, accuracy: float, max_radius: int) -> tuple[np.ndarray, np.ndarray]:
    """``GaussianBlur.makeGaussianKernel``: one-sided float32 kernel (edge-corrected, normalised in double) and the
    running sum beyond each element (``kernel[1]``)."""
    kr = int(math.ceil(sigma * math.sqrt(-2 * math.log(accuracy)))) + 1
    if max_radius < 50:
        max_radius = 50
    if kr > max_radius:
        kr = max_radius
    k = np.zeros(kr, np.float32)
    for i in range(kr):
        k[i] = np.float32(math.exp(-0.5 * i * i / sigma / sigma))
    if kr < max_radius and kr > 3:                 # edge correction: a parabola replaces the tail of the Gaussian
        sqrt_slope = float(np.finfo(np.float64).max)
        r = kr
        while r > kr // 2:
            r -= 1
            a = math.sqrt(float(k[r])) / (kr - r)
            if a < sqrt_slope:
                sqrt_slope = a
            else:
                break
        for r1 in range(r + 2, kr):
            k[r1] = np.float32(((kr - r1) * (kr - r1)) * sqrt_slope * sqrt_slope)
    if kr < max_radius:
        total = float(k[0])
        for i in range(1, kr):
            total += float(np.float32(2) * k[i])
    else:
        total = sigma * math.sqrt(2 * math.pi)
    rsum = 0.5 + 0.5 * float(k[0]) / total
    ksum = np.zeros(kr, np.float32)
    for i in range(kr):
        v = float(k[i]) / total
        k[i] = np.float32(v)
        rsum -= v
        ksum[i] = np.float32(rsum)
    return k, ksum


def _convolve_lines(inp: np.ndarray, kern: np.ndarray, ksum: np.ndarray, write_from: int, write_to: int) -> np.ndarray:
    """``GaussianBlur.convolveLine`` for every row of ``inp`` (float32 (R, L)); out-of-line pixels are the edge pixels,
    applied through the kernel running sums exactly like ImageJ.  Columns outside [write_from, write_to) are 0."""
    R, L = inp.shape
    kr = int(kern.shape[0])
    i = np.arange(L)
    first = inp[:, :1]
    last = inp[:, L - 1:L]
    res = inp * kern[0]
    first_part = min(kr, L)
    start_last = max(first_part, min(L - kr, write_to))
    add_first = i < kr
    add_last = ((i < first_part) & (i + kr > L)) | ((i >= start_last) & (i + kr >= L))
    zero = np.float32(0)
    ks_first = np.where(add_first, ksum[np.minimum(i, kr - 1)], zero).astype(np.float32)
    ks_last = np.where(add_last, ksum[np.clip(L - i - 1, 0, kr - 1)], zero).astype(np.float32)
    res = res + np.where(add_first[None, :], ks_first[None, :] * first, zero)
    res = res + np.where(add_last[None, :], ks_last[None, :] * last, zero)
    for k in range(1, kr):
        v = np.zeros((R, L), np.float32)
        if k < L:
            v[:, k:] = inp[:, :L - k]
            v[:, :L - k] = v[:, :L - k] + inp[:, k:]
        res = res + kern[k] * v
    if write_from > 0 or write_to < L:
        keep = (i >= write_from) & (i < write_to)
        res = np.where(keep[None, :], res, zero)
    return res


def _make_downscale_kernel(u: int) -> np.ndarray:
    mid = u * 3 // 2
    k = np.zeros(3 * u, np.float32)
    for i in range(0, u // 2 + 1):
        x = i / u
        v = np.float32((0.75 - x * x) / u)
        k[mid - i] = v
        k[mid + i] = v
    for i in range(u // 2 + 1, (u * 3 + 1) // 2):
        x = i / u
        v = np.float32((0.125 + 0.5 * (x - 1) * (x - 2)) / u)
        k[mid - i] = v
        k[mid + i] = v
    return k


def _make_upscale_kernel(u: int) -> np.ndarray:
    k = np.zeros(4 * u, np.float32)
    mid = 2 * u
    for i in range(0, u):
        x = i / u
        v = np.float32(2. / 3. - x * x * (1 - 0.5 * x))
        k[mid + i] = v
        k[mid - i] = v
    for i in range(u, 2 * u):
        x = i / u
        v = np.float32((2. - x) * (2. - x) * (2. - x) / 6.)
        k[mid + i] = v
        k[mid - i] = v
    return k


def _downscale_lines(inp: np.ndarray, dk: np.ndarray, r: int, unscaled0: int, new_len: int) -> np.ndarray:
    """``GaussianBlur.downscaleLine``: cache[j] = (sum2 + sum1) + sum0, each a float32 running sum over r pixels."""
    R, L = inp.shape
    p0 = unscaled0 - (r * 3) // 2
    out = np.zeros((R, new_len), np.float32)
    j = np.arange(new_len)
    parts = []
    for g in range(3):                       # g = 0: kernel[x] (sum2), 1: kernel[x+r] (sum1), 2: kernel[x+2r] (sum0)
        s = np.zeros((R, new_len), np.float32)
        for x in range(r):
            idx = np.clip(p0 + j * r + g * r + x, 0, L - 1)
            s = s + inp[:, idx] * dk[x + g * r]
        parts.append(s)
    out[:] = (parts[0] + parts[1]) + parts[2]
    return out


def _upscale_lines(cache: np.ndarray, uk: np.ndarray, r: int, unscaled0: int, L: int) -> np.ndarray:
    """``GaussianBlur.upscaleLine`` (float32, left to right)."""
    xout = np.arange(L)
    t = xout - unscaled0 + r - 1
    xin = t // r
    x = r - 1 - (t % r)
    out = cache[:, xin - 2] * uk[x]
    out = out + cache[:, xin - 1] * uk[x + r]
    out = out + cache[:, xin] * uk[x + 2 * r]
    out = out + cache[:, xin + 1] * uk[x + 3 * r]
    return out


def blur_lines(lines: np.ndarray, sigma: float, accuracy: float) -> np.ndarray:
    """``GaussianBlur.blur1Direction`` on every row of ``lines`` (float32 (R, L)), including the down-/upscaling
    ImageJ uses for sigma > 8.5."""
    lines = np.asarray(lines, dtype=np.float32)
    R, L = lines.shape
    UP = 2
    MIN_DS = 4.0
    do_ds = sigma > 2 * MIN_DS + 0.5
    reduce_by = min(int(math.floor(sigma / MIN_DS)), L) if do_ds else 1
    sigma_g = math.sqrt(sigma * sigma / (reduce_by * reduce_by) - 1. / 3. - 1. / 4.) if do_ds else sigma
    max_len = (L + reduce_by - 1) // reduce_by + 2 * (UP + 1) if do_ds else L
    kern, ksum = make_gaussian_kernel(sigma_g, accuracy, max_len)
    if not do_ds:
        return _convolve_lines(lines, kern, ksum, 0, L)
    new_len = (L + reduce_by - 1) // reduce_by + 2 * (UP + 1)
    unscaled0 = -(UP + 1) * reduce_by
    cache1 = _downscale_lines(lines, _make_downscale_kernel(reduce_by), reduce_by, unscaled0, new_len)
    cache2 = _convolve_lines(cache1, kern, ksum, 1, new_len - 1)
    return _upscale_lines(cache2, _make_upscale_kernel(reduce_by), reduce_by, unscaled0, L)


def gaussian_blur(plane: np.ndarray, sigma_x: float, sigma_y: float, accuracy: float | None = None) -> np.ndarray:
    """``GaussianBlur.blurGaussian``: x then y pass on a float32 copy; accuracy 0.002 for 8-bit (and RGB channels),
    0.0002 for 16/32-bit; 8/16-bit results rounded half up."""
    if accuracy is None:
        accuracy = 0.002 if plane.dtype == np.uint8 else 0.0002
    f = plane.astype(np.float32)
    if sigma_x > 0:
        f = blur_lines(f, sigma_x, accuracy)
    if sigma_y > 0:
        f = blur_lines(np.ascontiguousarray(f.T), sigma_y, accuracy).T
    return to_dtype(f, plane.dtype)


def unsharp_mask(plane: np.ndarray, sigma: float, weight: float) -> np.ndarray:
    """``UnsharpMask.sharpenFloat``: Gaussian blur with accuracy 0.01, then ``(v − w·blurred) / (1 − w)`` in float32."""
    f = plane.astype(np.float32)
    g = f
    if sigma > 0:
        g = blur_lines(f, sigma, 0.01)
        g = blur_lines(np.ascontiguousarray(g.T), sigma, 0.01).T
    w = np.float32(weight)
    out = (f - w * g) / (np.float32(1) - w)
    return to_dtype(out, plane.dtype)


# ====================================================================== ContrastEnhancer
def _limits_from_histogram(hist: np.ndarray, pixel_count: int, saturated: float) -> tuple[int, int]:
    """``ContrastEnhancer.getMinAndMax``: first / last bin whose cumulative count exceeds pixelCount·saturated/200."""
    threshold = int(pixel_count * saturated / 200.0) if saturated > 0.0 else 0
    cum = np.cumsum(hist)
    above = np.flatnonzero(cum > threshold)
    hmin = int(above[0]) if above.size else hist.size - 1
    cum_r = np.cumsum(hist[::-1])
    above_r = np.flatnonzero(cum_r > threshold)
    hmax = hist.size - 1 - int(above_r[0]) if above_r.size else 0
    return hmin, hmax


def contrast_limits(values: np.ndarray, dtype, saturated: float) -> tuple[float, float] | None:
    """The (min, max) that ``ContrastEnhancer.stretchHistogram`` uses for ``values`` (the selection's pixels, or the
    whole stack with 'use stack histogram'): 8-bit from the 256-bin histogram, 16-bit from the full 65536-bin histogram
    (ImageJ's ``histogram16``), 32-bit from the 256-bin histogram over the data range (bin edges, like ImageJ).
    None when ImageJ would leave the image untouched (hmax ≤ hmin)."""
    v = np.asarray(values).ravel()
    if v.size == 0:
        return None
    dt = np.dtype(dtype)
    if dt == np.uint8:
        hist = np.bincount(v.astype(np.int64), minlength=256)
        hmin, hmax = _limits_from_histogram(hist, int(v.size), saturated)
        return (float(hmin), float(hmax)) if hmax > hmin else None
    if dt == np.uint16:
        hist = np.bincount(v.astype(np.int64), minlength=65536)
        hmin, hmax = _limits_from_histogram(hist, int(v.size), saturated)
        return (float(hmin), float(hmax)) if hmax > hmin else None
    v = v[np.isfinite(v)].astype(np.float64)
    if v.size == 0:
        return None
    mn, mx = float(v.min()), float(v.max())
    if mx <= mn:
        return None
    n_bins = 256
    bin_size = (mx - mn) / n_bins
    scale = n_bins / (mx - mn)
    idx = np.minimum((scale * (v - mn)).astype(np.int64), n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins)
    hmin, hmax = _limits_from_histogram(hist, int(v.size), saturated)
    if hmax <= hmin:
        return None
    return mn + hmin * bin_size, mn + hmax * bin_size


def normalize_plane(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``ContrastEnhancer.normalize``: 8/16-bit through ImageJ's LUT (``(int)((i−min)/(max−min)·max2)``, 0 below min,
    max2 above max); 32-bit ``normalizeFloat`` (clipped to 0…1)."""
    if plane.dtype == np.float32:
        scale = 1.0 / (hi - lo) if hi > lo else 1.0
        v = plane.astype(np.float64) - lo
        v = np.maximum(v, 0.0) * scale
        return np.minimum(v, 1.0).astype(np.float32)
    max2 = int(np.iinfo(plane.dtype).max)
    i = np.arange(max2 + 1, dtype=np.float64)
    lut = ((i - lo) / (hi - lo) * max2).astype(np.int64)
    lut[i <= lo] = 0
    lut[i >= hi] = max2
    return lut[plane].astype(plane.dtype)


# ====================================================================== BackgroundSubtracter
def _filter3_lines(a: np.ndarray, maximum: bool) -> tuple[np.ndarray, np.ndarray]:
    """``BackgroundSubtracter.filter3`` along the rows of ``a`` (float32): 3-point maximum or mean with edge replication;
    returns the result and the per-row double sums of (max − v) for the MAXIMUM variant."""
    left = np.concatenate([a[:, :1], a[:, :-1]], axis=1)
    right = np.concatenate([a[:, 1:], a[:, -1:]], axis=1)
    if maximum:
        m = np.maximum(np.maximum(left, right), a)
        shift = np.cumsum((m - a).astype(np.float64), axis=1)[:, -1] if a.shape[1] else np.zeros(a.shape[0])
        return m, shift
    return ((left + a) + right) * np.float32(0.33333333), np.zeros(a.shape[0])


def filter3x3(a: np.ndarray, maximum: bool = False) -> tuple[np.ndarray, float]:
    """``BackgroundSubtracter.filter3x3``: rows then columns; returns (result, shiftBy) – the average rise of the
    3×3 maximum (0 for the mean), accumulated in double in ImageJ's order."""
    a = np.asarray(a, dtype=np.float32)
    H, W = a.shape
    rows, s_rows = _filter3_lines(a, maximum)
    colsT, s_cols = _filter3_lines(np.ascontiguousarray(rows.T), maximum)
    out = np.ascontiguousarray(colsT.T)
    shift = 0.0
    for v in s_rows:
        shift += float(v)
    for v in s_cols:
        shift += float(v)
    return out, shift / W / H


class RollingBall:
    """``RollingBall``: the square patch of the ball (heights above the sphere centre) and the shrink factor."""

    def __init__(self, radius: float):
        if radius <= 10:
            self.shrink, trim = 1, 24
        elif radius <= 30:
            self.shrink, trim = 2, 24
        elif radius <= 100:
            self.shrink, trim = 4, 32
        else:
            self.shrink, trim = 8, 40
        small = radius / self.shrink
        if small < 1:
            small = 1.0
        rsq = small * small
        xtrim = int(trim * small) // 100
        half = int(math.floor(small - xtrim + 0.5))        # Math.round
        self.width = 2 * half + 1
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        temp = rsq - (xx * xx).astype(np.float64) - (yy * yy).astype(np.float64)
        self.data = np.where(temp > 0.0, np.sqrt(np.maximum(temp, 0.0)), 0.0).astype(np.float32)


def shrink_image(a: np.ndarray, s: int) -> np.ndarray:
    """``BackgroundSubtracter.shrinkImage``: block minimum, ceil-sized."""
    H, W = a.shape
    sh, sw = (H + s - 1) // s, (W + s - 1) // s
    pad = np.full((sh * s, sw * s), np.inf, np.float32)
    pad[:H, :W] = a
    return pad.reshape(sh, s, sw, s).min(axis=(1, 3)).astype(np.float32)


def roll_ball(small: np.ndarray, ball: RollingBall) -> np.ndarray:
    """``BackgroundSubtracter.rollBall``: the grayscale opening of the image by the ball patch, with ball centres also
    outside the image (the patch clipped to the image)."""
    r = ball.width // 2
    pad = np.pad(small.astype(np.float32), r, mode="constant", constant_values=np.inf)
    z = ndi.grey_erosion(pad.astype(np.float64), structure=ball.data.astype(np.float64), mode="constant", cval=np.inf)
    z = z.astype(np.float32)                   # ImageJ: float32 differences, min (rounding is monotone)
    bg = ndi.grey_dilation(z.astype(np.float64), structure=ball.data.astype(np.float64), mode="constant", cval=-np.inf)
    bg = bg.astype(np.float32)
    H, W = small.shape
    return bg[r:r + H, r:r + W]


def _interp_arrays(length: int, small_len: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """``BackgroundSubtracter.makeInterpolationArrays`` (float32 weights)."""
    i = np.arange(length)
    idx = i - s // 2
    small_idx = np.where(idx >= 0, idx // s, -((-idx) // s))
    small_idx = np.minimum(small_idx, small_len - 2)
    dist = ((i.astype(np.float32) + _HALF) / np.float32(s)) - (small_idx.astype(np.float32) + _HALF)
    return small_idx, np.float32(1) - dist


def enlarge_image(small: np.ndarray, shape: tuple[int, int], s: int) -> np.ndarray:
    """``BackgroundSubtracter.enlargeImage``: bilinear interpolation / extrapolation back to the full size (float32)."""
    H, W = shape
    sh, sw = small.shape
    if sh < 2 or sw < 2:                        # ImageJ would index out of bounds; replicate the single row / column
        small = np.pad(small, ((0, max(0, 2 - sh)), (0, max(0, 2 - sw))), mode="edge")
        sh, sw = small.shape
    xi, xw = _interp_arrays(W, sw, s)
    yi, yw = _interp_arrays(H, sh, s)
    one = np.float32(1)
    lines = small[:, xi] * xw[None, :] + small[:, xi + 1] * (one - xw)[None, :]      # (sh, W)
    return lines[yi] * yw[:, None] + lines[yi + 1] * (one - yw)[:, None]


def rolling_ball_background(src: np.ndarray, radius: float, invert: bool, presmooth: bool) -> np.ndarray:
    """``BackgroundSubtracter.rollingBallFloatBackground`` on a float32 plane: (optional negation and 3×3 mean), shrink,
    roll, enlarge, negate back."""
    f = np.asarray(src, dtype=np.float32)
    if invert:
        f = -f
    if presmooth:
        f, _ = filter3x3(f, maximum=False)
    ball = RollingBall(radius)
    if ball.shrink > 1:
        small = shrink_image(f, ball.shrink)
        bg = enlarge_image(roll_ball(small, ball), f.shape, ball.shrink)
    else:
        bg = roll_ball(f, ball)
    if invert:
        bg = -bg
    return bg.astype(np.float32)


# ---------------------------------------------------------------- sliding paraboloid
def _hull_over_candidates(xs: np.ndarray, gs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Lower convex hull of the candidate points (xs[r, k], gs[r, k]) of every row r at once (Andrew's monotone chain
    in lock step over the candidate ordinal k; ``valid`` marks the existing candidates, a prefix per row).  Collinear
    points stay on the hull (ImageJ's strict ``<`` keeps the nearest touching point).  Returns a (R, K) mask."""
    R, K = xs.shape
    prev = np.full((R, K), -1, np.int64)
    ia = np.full(R, -1, np.int64)
    ib = np.full(R, -1, np.int64)
    rows = np.arange(R)
    for k in range(K):
        act = np.flatnonzero(valid[:, k])
        if act.size == 0:
            break
        xi, gi = xs[act, k], gs[act, k]
        IA, IB = ia[act], ib[act]
        while True:
            has = IA >= 0
            if not has.any():
                break
            ja, jb = np.maximum(IA, 0), np.maximum(IB, 0)
            xa, ga = xs[act, ja], gs[act, ja]
            xb, gb = xs[act, jb], gs[act, jb]
            cross = (xb - xa) * (gi - ga) - (gb - ga) * (xi - xa)
            pop = has & (cross < 0)
            if not pop.any():
                break
            IB = np.where(pop, IA, IB)
            IA = np.where(pop, prev[act, ja], IA)
        prev[act, k] = IB
        ia[act] = IB
        ib[act] = k
    on_hull = np.zeros((R, K), bool)
    cur = ib.copy()
    live = cur >= 0
    while live.any():
        r = rows[live]
        on_hull[r, cur[live]] = True
        cur[live] = prev[r, cur[live]]
        live = cur >= 0
    return on_hull


def slide_parabola_lines(lines: np.ndarray, lengths: np.ndarray, coeff2: np.float32,
                         corners: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``BackgroundSubtracter.lineSlideParabola`` for every row of ``lines`` (float32 (R, n); row r is a line of
    ``lengths[r]`` samples).  Points that fail ImageJ's curvature test can never be touched; the touching points are
    the lower hull of the remaining points in the parabola-flattened space; the others get ImageJ's float32
    interpolation ``v1 + (j−i1)·(minSlope − (j−i1)·coeff2)``.  Lines shorter than 3 samples are unchanged.
    With ``corners`` the first / last touching points (``firstCorner`` / ``lastCorner``) are returned too."""
    v = np.asarray(lines, dtype=np.float32)
    R, n = v.shape
    lengths = np.asarray(lengths, dtype=np.int64)
    out = v.copy()
    first_corner = np.full(R, 0, np.int64)
    last_corner = np.zeros(R, np.int64)
    if n < 3:
        if corners:
            return out, np.minimum(lengths - 1, first_corner + (lengths - 1)), last_corner
        return out
    cols = np.arange(n)
    inside = cols[None, :] < lengths[:, None]
    # ImageJ's curvature test (float32): point m (1 ≤ m ≤ len−2) can be touched when 2·v[m] − v[m−1] − v[m+1] < 1.999·coeff2
    curvature_test = np.float32(1.999) * coeff2
    elig = np.zeros((R, n), bool)
    mid = (((v[:, 1:-1] + v[:, 1:-1]) - v[:, :-2]) - v[:, 2:]) < curvature_test
    elig[:, 1:-1] = mid
    elig[:, 0] = True
    elig[np.arange(R), np.maximum(lengths - 1, 0)] = True
    elig &= inside
    # candidates per row, compacted to a (R, K) prefix layout
    counts = elig.sum(axis=1)
    K = int(counts.max()) if R else 0
    order = np.argsort(~elig, axis=1, kind="stable")[:, :K]          # candidate column indices, ascending, padded
    valid = np.arange(K)[None, :] < counts[:, None]
    xs = order.astype(np.float64)
    c2 = float(coeff2)
    gs = np.take_along_axis(v, order, axis=1).astype(np.float64) + c2 * xs * xs
    hull = _hull_over_candidates(xs, gs, valid) & valid
    vert = np.zeros((R, n), bool)
    rr = np.repeat(np.arange(R), K)
    vert[rr[hull.ravel()], order.ravel()[hull.ravel()]] = True
    rows_n = np.repeat(np.arange(R), n)
    cand = elig & inside
    cand[:, 0] = False

    def segments(vert):
        """Left (last touching point ≤ j) and right (first touching point ≥ j) of every sample."""
        idx = np.where(vert, cols[None, :], -1)
        left = np.maximum.accumulate(idx, axis=1)
        ridx = np.where(vert, cols[None, :], n)
        right = np.minimum.accumulate(ridx[:, ::-1], axis=1)[:, ::-1]
        return left, right

    # ImageJ walks from touching point to touching point choosing the candidate with the smallest *float32* slope
    # (the first one in a tie).  The exact hull differs from that walk in two tie situations, which are replayed here:
    # (a) a candidate inside a hull segment whose float32 slope equals the segment end's is touched as well;
    # (b) of (nearly) collinear touching points the one with the smaller float32 slope wins, so a touching point
    #     in between is skipped (it gets the parabola value instead of keeping its own).

    def slope32(r, i1, j):
        d = (j - i1).astype(np.float32)
        return ((v[r, j] - v[r, i1]) / d) + coeff2 * d

    for _ in range(32):
        changed = False
        left, right = segments(vert)
        prev_all = np.full((R, n), -1, np.int64)      # the touching point before a touching point
        prev_all[:, 1:] = np.maximum.accumulate(np.where(vert, cols[None, :], -1), axis=1)[:, :-1]
        # (b) triples i1 < i2 < i3 of touching points whose float32 slopes from i1 are out of order
        next_all = np.full((R, n), n, np.int64)
        next_all[:, :-1] = np.minimum.accumulate(np.where(vert, cols[None, :], n)[:, ::-1], axis=1)[:, ::-1][:, 1:]
        mid = vert & (prev_all >= 0) & (next_all < n)
        r_m, j_m = np.nonzero(mid)
        if r_m.size:
            i1, i3 = prev_all[r_m, j_m], next_all[r_m, j_m]
            drop = slope32(r_m, i1, i3) < slope32(r_m, i1, j_m)
            if drop.any():
                drop_mask = np.zeros((R, n), bool)
                drop_mask[r_m[drop], j_m[drop]] = True
                drop &= ~drop_mask[r_m, i1]                 # one skip per chain and round (the walk restarts there)
                vert[r_m[drop], j_m[drop]] = False
                changed = True
                left, right = segments(vert)
                prev_all[:, 1:] = np.maximum.accumulate(np.where(vert, cols[None, :], -1), axis=1)[:, :-1]
        # (a) float32 ties inside a segment
        sel = (cand & ~vert & (left >= 0) & (right < n)) | (vert & (prev_all >= 0))
        r_s, j_s = np.nonzero(sel)
        if r_s.size:
            i1 = np.where(vert[r_s, j_s], prev_all[r_s, j_s], left[r_s, j_s])
            slope = slope32(r_s, i1, j_s)
            seg = r_s * n + i1                                        # non-decreasing in row-major order
            new_seg = np.r_[True, seg[1:] != seg[:-1]]
            starts = np.flatnonzero(new_seg)
            seg_min = np.minimum.reduceat(slope, starts)
            seg_id = np.cumsum(new_seg) - 1
            is_min = slope == seg_min[seg_id]
            first = is_min & (np.cumsum(is_min) - np.r_[0, np.cumsum(is_min)[starts[1:] - 1]][seg_id] == 1)
            new = first & ~vert[r_s, j_s]
            if new.any():
                vert[r_s[new], j_s[new]] = True
                changed = True
        if not changed:
            break
    left, right = segments(vert)
    interp = inside & ~vert & (left >= 0) & (right < n)
    if interp.any():
        r_i, j_i = np.nonzero(interp)
        i1 = left[r_i, j_i]
        i2 = right[r_i, j_i]
        v1 = v[r_i, i1]
        v2 = v[r_i, i2]
        d2 = (i2 - i1).astype(np.float32)
        min_slope = ((v2 - v1) / d2) + coeff2 * d2
        d = (j_i - i1).astype(np.float32)
        out[r_i, j_i] = v1 + d * (min_slope - d * coeff2)
    if corners:
        short = lengths < 3
        vert_len_ok = vert & inside
        # first corner: the first touching point after 0; last corner: the last touching point before len−1
        v2mask = vert_len_ok.copy()
        v2mask[:, 0] = False
        first_corner = np.where(v2mask.any(axis=1), np.argmax(v2mask, axis=1), np.maximum(lengths - 1, 0))
        v1mask = vert_len_ok.copy()
        v1mask[np.arange(R), np.maximum(lengths - 1, 0)] = False
        rev = v1mask[:, ::-1]
        last_corner = np.where(rev.any(axis=1), n - 1 - np.argmax(rev, axis=1), 0)
        first_corner = np.where(short, np.maximum(lengths - 1, 0), first_corner)
        last_corner = np.where(short, 0, last_corner)
        return out, first_corner, last_corner
    return out


def _f32(x) -> np.float32:
    return np.float32(x)


def _corrected_edges(line: np.ndarray, coeff2: np.float32, first_corner: int, last_corner: int) -> tuple[np.float32, np.float32]:
    """The ``correctedEdges`` part of ``lineSlideParabola`` (float32 scalar arithmetic like Java): the estimated edge
    values without an edge particle, allowing for vignetting (6th-order polynomial)."""
    with np.errstate(all="ignore"):          # Java float semantics: ±Inf / NaN instead of warnings
        return _corrected_edges_impl(line, coeff2, first_corner, last_corner)


def _corrected_edges_impl(line: np.ndarray, coeff2: np.float32, first_corner: int, last_corner: int) -> tuple[np.float32, np.float32]:
    length = int(line.shape[0])
    if 4 * first_corner >= length:
        first_corner = 0
    if 4 * (length - 1 - last_corner) >= length:
        last_corner = length - 1
    v1 = _f32(line[first_corner])
    v2 = _f32(line[last_corner])
    denom = last_corner - first_corner
    slope = (v2 - v1) / _f32(denom) if denom != 0 else _f32(np.nan)
    value0 = v1 - slope * _f32(first_corner)
    coeff6 = _f32(0)
    mid = _f32(0.5) * _f32(last_corner + first_corner)
    one = _f32(1)
    for i in range((length + 2) // 3, (2 * length) // 3 + 1):
        dx = (_f32(i) - mid) * _f32(2) / _f32(denom) if denom != 0 else _f32(np.nan)
        poly6 = dx * dx * dx * dx * dx * dx - one
        if _f32(line[i]) < value0 + slope * _f32(i) + coeff6 * poly6:
            coeff6 = -(value0 + slope * _f32(i) - _f32(line[i])) / poly6
    dx = (_f32(first_corner) - mid) * _f32(2) / _f32(denom) if denom != 0 else _f32(np.nan)
    e0 = value0 + coeff6 * (dx * dx * dx * dx * dx * dx - one) + coeff2 * _f32(first_corner) * _f32(first_corner)
    dx = (_f32(last_corner) - mid) * _f32(2) / _f32(denom) if denom != 0 else _f32(np.nan)
    e1 = (value0 + _f32(length - 1) * slope + coeff6 * (dx * dx * dx * dx * dx * dx - one)
          + coeff2 * _f32(length - 1 - last_corner) * _f32(length - 1 - last_corner))
    return e0, e1


def _slide_line_inplace(flat: np.ndarray, start: int, inc: int, length: int, coeff2: np.float32,
                        edges: bool = False) -> tuple[np.float32, np.float32] | None:
    """One ``lineSlideParabola`` call on the flat pixel array (used for the eight corner-correction lines)."""
    idx = start + inc * np.arange(length)
    line = flat[idx]
    out, fc, lc = slide_parabola_lines(line[None, :], np.array([length]), coeff2, corners=True)
    flat[idx] = out[0]
    if edges:
        return _corrected_edges(line, coeff2, int(fc[0]), int(lc[0]))
    return None


def correct_corners(f: np.ndarray, coeff2: np.float32) -> None:
    """``BackgroundSubtracter.correctCorners`` (in place): lowers each corner pixel to the mean of three edge estimates
    (row, column and diagonal through the corner) when a corner particle is suspected."""
    H, W = f.shape
    flat = f.reshape(-1)
    corners = [_f32(0)] * 4
    e = _slide_line_inplace(flat, 0, 1, W, coeff2, True)
    corners[0], corners[1] = e[0], e[1]
    e = _slide_line_inplace(flat, (H - 1) * W, 1, W, coeff2, True)
    corners[2], corners[3] = e[0], e[1]
    e = _slide_line_inplace(flat, 0, W, H, coeff2, True)
    corners[0], corners[2] = corners[0] + e[0], corners[2] + e[1]
    e = _slide_line_inplace(flat, W - 1, W, H, coeff2, True)
    corners[1], corners[3] = corners[1] + e[0], corners[3] + e[1]
    diag = min(W, H)
    c2d = _f32(2) * coeff2
    corners[0] = corners[0] + _slide_line_inplace(flat, 0, 1 + W, diag, c2d, True)[0]
    corners[1] = corners[1] + _slide_line_inplace(flat, W - 1, -1 + W, diag, c2d, True)[0]
    corners[2] = corners[2] + _slide_line_inplace(flat, (H - 1) * W, 1 - W, diag, c2d, True)[0]
    corners[3] = corners[3] + _slide_line_inplace(flat, W * H - 1, -1 - W, diag, c2d, True)[0]
    three = _f32(3)
    for p, c in zip((0, W - 1, (H - 1) * W, W * H - 1), corners):
        if flat[p] > c / three:
            flat[p] = c / three


def _diagonal_lines(W: int, H: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices (R, n) (−1 padded) and lengths of the lines of one ``filter1D`` diagonal direction."""
    if direction == "1A":            # parallel to x = y, starting on the x axis
        starts = np.arange(0, max(W - 2, 0))
        lengths = np.minimum(H, W - starts)
        inc = W + 1
    elif direction == "1B":          # parallel to x = y, starting on the y axis (row 1 …)
        rows = np.arange(1, max(H - 2, 1))
        starts = rows * W
        lengths = np.minimum(W, H - rows)
        inc = W + 1
    elif direction == "2A":          # parallel to x = −y, starting on the x axis (column 2 …)
        starts = np.arange(2, W)
        lengths = np.minimum(H, starts + 1)
        inc = W - 1
    else:                            # "2B": parallel to x = −y, starting at x = W−1 on rows 0 …
        rows = np.arange(0, max(H - 2, 0))
        starts = rows * W + W - 1
        lengths = np.minimum(W, H - rows)
        inc = W - 1
    n = int(lengths.max()) if lengths.size else 0
    k = np.arange(n)
    idx = starts[:, None] + inc * k[None, :]
    idx = np.where(k[None, :] < lengths[:, None], idx, -1)
    return idx, lengths


def _filter_direction(f: np.ndarray, direction: str, coeff2: np.float32) -> np.ndarray:
    """``BackgroundSubtracter.filter1D`` for one direction (X, Y or a diagonal), returning the new float32 plane."""
    H, W = f.shape
    if direction == "X":
        return slide_parabola_lines(f, np.full(H, W), coeff2)
    if direction == "Y":
        return np.ascontiguousarray(slide_parabola_lines(np.ascontiguousarray(f.T), np.full(W, H), coeff2).T)
    idx, lengths = _diagonal_lines(W, H, direction)
    if idx.size == 0:
        return f
    flat = f.reshape(-1).copy()
    order = np.argsort(-lengths, kind="stable")           # longest lines first (prefix layout of the samples)
    idx, lengths = idx[order], lengths[order]
    vals = flat[np.maximum(idx, 0)]
    out = slide_parabola_lines(vals, lengths, coeff2)
    sel = idx >= 0
    flat[idx[sel]] = out[sel]
    return flat.reshape(H, W)


def sliding_paraboloid_background(src: np.ndarray, radius: float, invert: bool = False, presmooth: bool = False,
                                  correct: bool = True) -> np.ndarray:
    """``BackgroundSubtracter.slidingParaboloidFloatBackground``: (negate), 3×3 maximum + mean when presmoothing,
    corner correction, then the nine directional parabola passes X, Y, X, 1A, 1B, 2A, 2B, 1A, 1B; the presmoothing
    shift is subtracted at the end."""
    f = np.array(src, dtype=np.float32)
    if invert:
        f = -f
    coeff2 = np.float32(0.5) / np.float32(radius)
    coeff2diag = np.float32(1.0) / np.float32(radius)
    shift = np.float32(0)
    if presmooth:
        f, s = filter3x3(f, maximum=True)
        shift = np.float32(s)
        f, _ = filter3x3(f, maximum=False)
    if correct and f.shape[0] > 0 and f.shape[1] > 0:
        correct_corners(f, coeff2)
    for d, c in (("X", coeff2), ("Y", coeff2), ("X", coeff2), ("1A", coeff2diag), ("1B", coeff2diag),
                 ("2A", coeff2diag), ("2B", coeff2diag), ("1A", coeff2diag), ("1B", coeff2diag)):
        f = _filter_direction(f, d, c)
    if invert:
        f = -(f - shift)
    elif presmooth:
        f = f - shift
    return f.astype(np.float32)


def subtract_background(plane: np.ndarray, radius: float, light_background: bool = False, create_background: bool = False,
                        paraboloid: bool = False, presmooth: bool = True) -> np.ndarray:
    """``BackgroundSubtracter.rollingBallBackground`` for one plane (8/16/32-bit or one RGB channel): the background
    (``create_background``) or the image minus it – 8/16-bit as ``(int)(v − bg + 0.5f)`` clipped (plus the type maximum
    for a light background), float as float32 ``v − bg``."""
    f = plane.astype(np.float32)
    if paraboloid:
        bg = sliding_paraboloid_background(f, float(radius), invert=light_background, presmooth=presmooth, correct=True)
    else:
        bg = rolling_ball_background(f, float(radius), invert=light_background, presmooth=presmooth)
    if create_background:
        return to_dtype(bg, plane.dtype)
    if plane.dtype == np.float32:
        return f - bg
    mx = float(np.iinfo(plane.dtype).max)
    offset = np.float32(mx + 0.5) if light_background else _HALF
    v = (f - bg) + offset
    v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, mx)
    return np.trunc(v).astype(plane.dtype)


# ====================================================================== FFT (ij.plugin.FFT / ij.process.FHT)
def fft_pad_size(width: int, height: int) -> int:
    """``FFT.pad``: the side of the square power-of-two image ImageJ transforms (no padding only for a square
    power-of-two image)."""
    max_n = max(width, height)
    i = 2
    while i < max_n:
        i *= 2
    return i


def fft_pad(plane: np.ndarray) -> np.ndarray:
    """``FFT.pad``: the plane in the top-left corner of a square power-of-two image filled with its mean
    (``setValue(mean)`` truncates the mean for 8/16-bit)."""
    H, W = plane.shape
    n = fft_pad_size(W, H)
    if n == W and n == H:
        return plane
    mean = float(plane.astype(np.float64).mean())
    fill = mean if plane.dtype == np.float32 else int(mean)
    out = np.full((n, n), fill, dtype=plane.dtype)
    out[:H, :W] = plane
    return out


def power_spectrum_8bit(F: np.ndarray) -> np.ndarray:
    """``FHT.getPowerSpectrum`` of a centred complex transform: |F|² log-scaled to 1…254 with ImageJ's float32
    arithmetic (``scale = 253.999/(log max − log min)``, ``(byte)(r + 1)``)."""
    ps = (np.abs(F) ** 2).astype(np.float32)
    mn_f, mx_f = float(ps.min()), float(ps.max())
    mx = np.float32(math.log(mx_f)) if mx_f > 0 else np.float32(-np.inf)
    mn = np.float32(math.log(mn_f)) if mn_f > 0 else np.float32(-np.inf)
    if np.isnan(mn) or (mx - mn) > np.float32(50):
        mn = mx - np.float32(50)
    d = mx - mn
    scale = np.float32(253.999 / float(d)) if float(d) != 0 else np.float32(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.log(ps.astype(np.float64)).astype(np.float32) - mn) * scale
    r = np.where(np.isnan(r) | (r < 0), np.float32(0), r)
    return np.trunc(r + np.float32(1)).astype(np.uint8)


def _tile_mirror(plane: np.ndarray, n: int, x0: int, y0: int) -> np.ndarray:
    """``FFTFilter.tileMirror``: the image at (x0, y0) of an n×n canvas, mirrored into the surrounding tiles."""
    H, W = plane.shape
    u = np.arange(n) - x0
    tx, sx = np.floor_divide(u, W), np.mod(u, W)
    cx = np.where(tx % 2 == 1, W - 1 - sx, sx)
    v = np.arange(n) - y0
    ty, sy = np.floor_divide(v, H), np.mod(v, H)
    cy = np.where(ty % 2 == 1, H - 1 - sy, sy)
    return plane[cy[:, None], cx[None, :]]


def _bandpass_filter(n: int, filter_large: float, filter_small: float, stripes: int, scale_stripes: float) -> np.ndarray:
    """``FFTFilter.filterLargeSmall``: the (n, n) float32 multiplier in the unshifted frequency layout."""
    scale_large = filter_large * filter_large
    scale_small = filter_small * filter_small
    scale_stripes = scale_stripes * scale_stripes
    half = n // 2
    j = np.arange(n)
    jj = np.minimum(j, n - j)                                 # |frequency| of row / column
    jj2 = (jj * jj).astype(np.float64)
    fl = np.exp(-jj2 * scale_large).astype(np.float32)        # (float) Math.exp(-(j*j) * scaleLarge)
    fs = np.exp(-jj2 * scale_small).astype(np.float32)
    fst = np.exp(-jj2 * scale_stripes).astype(np.float32)
    one = np.float32(1)
    rowL, colL = fl[:, None], fl[None, :]
    rowS, colS = fs[:, None], fs[None, :]
    filt = ((one - rowL * colL) * rowS) * colS                # interior and the column maxN/2 case
    if stripes == 1:
        filt = filt * (one - fst[None, :])                    # horizontal stripes: suppress along columns
    elif stripes == 2:
        filt = filt * (one - fst[:, None])
    rh = ((one - colL * fl[half]) * colS) * fs[half]          # row maxN/2: ImageJ multiplies colFactSmall before rowFactSmall
    if stripes == 1:
        rh = rh * (one - fst[None, :])
    elif stripes == 2:
        rh = rh * (one - fst[half])
    filt[half:half + 1, :] = rh
    # row 0 and column 0 (one factor only); the DC term stays 1
    row0 = (one - colL) * colS
    col0 = (one - rowL) * rowS
    if stripes == 1:
        row0 = row0 * (one - fst[None, :])
        col0 = col0 * np.float32(0)
    elif stripes == 2:
        row0 = row0 * np.float32(0)
        col0 = col0 * (one - fst[:, None])
    filt[0:1, :] = row0
    filt[:, 0:1] = col0
    # meeting points (maxN/2, 0), (0, maxN/2) and (maxN/2, maxN/2)
    f_l, f_s, f_st = fl[half], fs[half], fst[half]
    m_edge = (one - f_l) * f_s
    m_mid = ((one - f_l * f_l) * f_s) * f_s
    if stripes == 1:
        filt[0, half] = m_edge * (one - f_st)
        filt[half, 0] = np.float32(0)
        filt[half, half] = m_mid * (one - f_st)
    elif stripes == 2:
        filt[0, half] = np.float32(0)
        filt[half, 0] = m_edge * (one - f_st)
        filt[half, half] = m_mid * (one - f_st)
    else:
        filt[0, half] = m_edge
        filt[half, 0] = m_edge
        filt[half, half] = m_mid
    filt[0, 0] = one
    return filt


def bandpass_filter(plane: np.ndarray, filter_large: float, filter_small: float, suppress: str = "None",
                    tolerance: float = 5.0, autoscale: bool = True, saturate: bool = True) -> np.ndarray:
    """``FFTFilter.filter``: mirror-tile the plane into a power-of-two square (≥ 1.5× the size), transform, multiply
    by the Gaussian band filter, transform back, crop, optionally autoscale (``ContrastEnhancer.stretchHistogram`` with
    1 % / 0 % saturation) and convert to the original type."""
    H, W = plane.shape
    max_n = max(W, H)
    n = 2
    while n < 1.5 * max_n:
        n *= 2
    f_large = 2.0 * filter_large / n
    f_small = 2.0 * filter_small / n
    x0 = int(math.floor((n - W) / 2.0 + 0.5))
    y0 = int(math.floor((n - H) / 2.0 + 0.5))
    tiled = _tile_mirror(plane.astype(np.float32), n, x0, y0).astype(np.float64)
    stripes = {"None": 0, "Horizontal": 1, "Vertical": 2}.get(str(suppress), 0)
    sharpness = (100.0 - tolerance) / 100.0
    filt = _bandpass_filter(n, f_large, f_small, stripes, sharpness)
    F = np.fft.fft2(tiled) * filt.astype(np.float64)
    res = np.real(np.fft.ifft2(F)).astype(np.float32)[y0:y0 + H, x0:x0 + W]
    if plane.dtype == np.float32:
        return res
    mx_t = float(np.iinfo(plane.dtype).max)
    if not autoscale:
        return round_half_up(res, plane.dtype)
    lim = contrast_limits(res, np.float32, 1.0 if saturate else 0.0)
    lo, hi = lim if lim is not None else (float(res.min()), float(res.max()))
    v = res.astype(np.float64) - lo
    if plane.dtype == np.uint8:
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        iv = np.trunc(np.maximum(v, 0.0) * scale + 0.5)
        return np.minimum(iv, 255).astype(np.uint8)
    scale = 65535.0 / (hi - lo) if hi > lo else 1.0
    val = np.clip(v * scale, 0.0, 65535.0)
    return np.trunc(val + 0.5).astype(np.uint16)


# ====================================================================== FHT (ij.process.FHT, float32 bit-exact)
_FHT_TABLES: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _fht_tables(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``FHT.makeSinCosTables`` / ``makeBitReverseTable``: float32 cos / sin of the accumulated double angle and the
    bit-reversal permutation."""
    t = _FHT_TABLES.get(n)
    if t is None:
        m = n // 4
        C = np.zeros(m, np.float32)
        S = np.zeros(m, np.float32)
        theta = 0.0
        d_theta = 2.0 * math.pi / n
        for i in range(m):
            C[i] = np.float32(math.cos(theta))
            S[i] = np.float32(math.sin(theta))
            theta += d_theta
        nlog2 = int(round(math.log2(n)))
        bitrev = np.zeros(n, np.int64)
        for i in range(n):
            r = 0
            for b in range(nlog2):
                if i & (1 << b):
                    r |= 1 << (nlog2 - b - 1)
            bitrev[i] = r
        t = (C, S, bitrev)
        _FHT_TABLES[n] = t
    return t


def _dfht3_rows(x: np.ndarray, inverse: bool) -> np.ndarray:
    """``FHT.dfht3`` on every row of the float32 (R, n) array ``x`` (n a power of two), in ImageJ's butterfly order."""
    R, n = x.shape
    C, S, bitrev = _fht_tables(n)
    nlog2 = int(round(math.log2(n)))
    x = x[:, bitrev].astype(np.float32)
    if n >= 4:
        a1 = np.arange(0, n, 4)
        a2, a3, a4 = a1 + 1, a1 + 2, a1 + 3
        rt1 = x[:, a1] + x[:, a2]
        rt2 = x[:, a1] - x[:, a2]
        rt3 = x[:, a3] + x[:, a4]
        rt4 = x[:, a3] - x[:, a4]
        x[:, a1] = rt1 + rt3
        x[:, a2] = rt2 + rt4
        x[:, a3] = rt1 - rt3
        x[:, a4] = rt2 - rt4
    elif n == 2:
        a, b = x[:, 0].copy(), x[:, 1].copy()
        x[:, 0] = a + b
        x[:, 1] = a - b
    if nlog2 > 2:
        gp_size, num_bfs, num_gps = 4, 2, n // 8
        for _stage in range(2, nlog2):
            ad0 = np.arange(num_gps) * gp_size * 2
            # first butterfly of every group: no multiplications
            ad1, ad2 = ad0, ad0 + gp_size
            ad3 = ad0 + gp_size // 2
            ad4 = ad3 + gp_size
            v1, v2, v3, v4 = x[:, ad1], x[:, ad2], x[:, ad3], x[:, ad4]
            x[:, ad1] = v1 + v2
            x[:, ad2] = v1 - v2
            x[:, ad3] = v3 + v4
            x[:, ad4] = v3 - v4
            if num_bfs > 1:
                bf = np.arange(1, num_bfs)
                ad1 = (bf[None, :] + ad0[:, None]).ravel()
                ad2 = ad1 + gp_size
                ad3 = (gp_size - bf[None, :] + ad0[:, None]).ravel()
                ad4 = ad3 + gp_size
                cs = np.tile(bf * num_gps, num_gps)
                c, s = C[cs][None, :], S[cs][None, :]
                v1, v2, v3, v4 = x[:, ad1], x[:, ad2], x[:, ad3], x[:, ad4]
                rt1 = v2 * c + v4 * s
                rt2 = v4 * c - v2 * s
                x[:, ad2] = v1 - rt1
                x[:, ad1] = v1 + rt1
                x[:, ad4] = v3 + rt2
                x[:, ad3] = v3 - rt2
            gp_size *= 2
            num_bfs *= 2
            num_gps //= 2
    if inverse:
        x = x / np.float32(n)
    return x


def fht2d(data: np.ndarray, inverse: bool = False) -> np.ndarray:
    """``FHT.rc2DFHT``: the 2-D Hartley transform of a square power-of-two float32 image exactly as ImageJ computes
    it (rows, transpose, rows, transpose, Bracewell's 2-D correction) – forward unnormalised, inverse scaled."""
    x = np.asarray(data, dtype=np.float32)
    n = x.shape[0]
    if x.shape != (n, n) or n & (n - 1) or n < 2:
        raise ValueError(f"FHT needs a square power-of-two image, not {x.shape[1]}×{x.shape[0]}")
    x = _dfht3_rows(x, inverse)
    x = np.ascontiguousarray(x.T)
    x = _dfht3_rows(x, inverse)
    x = np.ascontiguousarray(x.T)
    h = n // 2
    rows = np.arange(h + 1)
    cols = np.arange(h + 1)
    m_rows = (n - rows) % n
    m_cols = (n - cols) % n
    A = x[rows[:, None], cols[None, :]]
    B = x[m_rows[:, None], cols[None, :]]
    Cc = x[rows[:, None], m_cols[None, :]]
    D = x[m_rows[:, None], m_cols[None, :]]
    E = ((A + D) - (B + Cc)) / np.float32(2)
    x[rows[:, None], cols[None, :]] = A - E
    x[m_rows[:, None], cols[None, :]] = B + E
    x[rows[:, None], m_cols[None, :]] = Cc + E
    x[m_rows[:, None], m_cols[None, :]] = D - E
    return x


def fht_power_spectrum(fht: np.ndarray) -> np.ndarray:
    """``FHT.fht2ps``: the float32 power spectrum (uncentred) of a frequency-domain FHT."""
    n = fht.shape[0]
    idx = (n - np.arange(n)) % n
    mirrored = fht[idx[:, None], idx[None, :]]
    return (fht * fht + mirrored * mirrored) / np.float32(2)


def fht_to_complex(fht: np.ndarray) -> np.ndarray:
    """The complex DFT equivalent to a Hartley transform (``FHT.getComplexTransform``), uncentred."""
    n = fht.shape[0]
    idx = (n - np.arange(n)) % n
    m = fht[idx[:, None], idx[None, :]].astype(np.float64)
    f = fht.astype(np.float64)
    return ((f + m) / 2.0) - 1j * ((f - m) / 2.0)


def power_spectrum_8bit_from_fht(fht: np.ndarray) -> np.ndarray:
    """``FHT.getPowerSpectrum``: the centred 8-bit log-scaled power spectrum of a frequency-domain FHT."""
    ps = fht_power_spectrum(fht)
    return np.fft.fftshift(_scale_power_spectrum(ps))


def _scale_power_spectrum(ps: np.ndarray) -> np.ndarray:
    ps = np.asarray(ps, dtype=np.float32)
    mn_f, mx_f = float(ps.min()), float(ps.max())
    mx = np.float32(math.log(mx_f)) if mx_f > 0 else np.float32(-np.inf)
    mn = np.float32(math.log(mn_f)) if mn_f > 0 else np.float32(-np.inf)
    if np.isnan(mn) or (mx - mn) > np.float32(50):
        mn = mx - np.float32(50)
    d = mx - mn
    scale = np.float32(253.999 / float(d)) if float(d) != 0 else np.float32(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.log(ps.astype(np.float64)).astype(np.float32) - mn) * scale
    r = np.where(np.isnan(r) | (r < 0), np.float32(0), r)
    return np.trunc(r + np.float32(1)).astype(np.uint8)


def bandpass_filter_fht(plane: np.ndarray, filter_large: float, filter_small: float, suppress: str = "None",
                        tolerance: float = 5.0, autoscale: bool = True, saturate: bool = True) -> np.ndarray:
    """``FFTFilter.filter`` with ImageJ's float32 Hartley transform (bit-exact variant of ``bandpass_filter``)."""
    H, W = plane.shape
    max_n = max(W, H)
    n = 2
    while n < 1.5 * max_n:
        n *= 2
    f_large = 2.0 * filter_large / n
    f_small = 2.0 * filter_small / n
    x0 = int(math.floor((n - W) / 2.0 + 0.5))
    y0 = int(math.floor((n - H) / 2.0 + 0.5))
    tiled = _tile_mirror(plane.astype(np.float32), n, x0, y0)
    stripes = {"None": 0, "Horizontal": 1, "Vertical": 2}.get(str(suppress), 0)
    sharpness = (100.0 - tolerance) / 100.0
    filt = _bandpass_filter(n, f_large, f_small, stripes, sharpness)
    fht = fht2d(tiled, inverse=False) * filt
    res = fht2d(fht, inverse=True)[y0:y0 + H, x0:x0 + W]
    return _bandpass_to_type(res, plane.dtype, autoscale, saturate)


def _bandpass_to_type(res: np.ndarray, dtype, autoscale: bool, saturate: bool) -> np.ndarray:
    if np.dtype(dtype) == np.float32:
        return np.asarray(res, dtype=np.float32)
    if not autoscale:
        return round_half_up(res, dtype)
    lim = contrast_limits(res, np.float32, 1.0 if saturate else 0.0)
    lo, hi = lim if lim is not None else (float(res.min()), float(res.max()))
    v = res.astype(np.float64) - lo
    if np.dtype(dtype) == np.uint8:
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        iv = np.trunc(np.maximum(v, 0.0) * scale + 0.5)
        return np.minimum(iv, 255).astype(np.uint8)
    scale = 65535.0 / (hi - lo) if hi > lo else 1.0
    val = np.clip(v * scale, 0.0, 65535.0)
    return np.trunc(val + 0.5).astype(np.uint16)
