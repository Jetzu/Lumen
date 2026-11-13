"""Process ▸ Noise."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..registry import Param, op
from ._imagej import _nan_aware_median, rank_filter, rank_kernel
from ._util import as_float, back_to

CAT = "Process/Noise"


@op("noise.add", "Add Noise", CAT, description="Add Gaussian noise with SD 25 (8-bit scale).")
def add_noise(plane, ctx):
    sd = 25.0 if plane.dtype == np.uint8 else (25.0 * 257 if plane.dtype == np.uint16 else 25.0)
    rng = np.random.default_rng()
    return back_to(as_float(plane) + rng.normal(0, sd, plane.shape), plane)


@op("noise.add_specified", "Add Specified Noise", CAT, params=[Param("sd", "float", 25.0, min=0, step=1, label="Standard deviation")], description="Add Gaussian noise with the given SD.")
def add_specified(plane, ctx, sd=25.0):
    rng = np.random.default_rng()
    return back_to(as_float(plane) + rng.normal(0, sd, plane.shape), plane)


@op("noise.salt_pepper", "Salt and Pepper", CAT, params=[Param("percent", "float", 2.5, min=0, max=50, step=0.5)], description="Randomly set pixels to min / max.")
def salt_pepper(plane, ctx, percent=2.5):
    rng = np.random.default_rng()
    out = plane.copy()
    n = int(plane.size * percent / 100)
    ys = rng.integers(0, plane.shape[0], n)
    xs = rng.integers(0, plane.shape[1], n)
    lo, hi = (0, 255) if plane.dtype == np.uint8 else ((0, 65535) if plane.dtype == np.uint16 else (float(np.nanmin(plane)), float(np.nanmax(plane))))
    vals = rng.random(n) < 0.5
    out[ys[vals], xs[vals]] = hi
    out[ys[~vals], xs[~vals]] = lo
    return out


@op("noise.despeckle", "Despeckle", CAT, description="3×3 median filter (ImageJ's RankFilters median of radius 1).")
def despeckle(plane, ctx):
    return rank_filter(plane, 1.0, "median")


@op("noise.remove_outliers", "Remove Outliers", CAT,
    description="Replace pixels that deviate from the local median (ImageJ's circular kernel) by more than a threshold: bright outliers above median + threshold, dark ones below median − threshold.",
    params=[Param("radius", "float", 2.0, min=0.5, max=100, step=0.5), Param("threshold", "float", 50.0, min=0, step=1), Param("which", "choice", "Bright", choices=["Bright", "Dark"])])
def remove_outliers(plane, ctx, radius=2.0, threshold=50.0, which="Bright"):
    # RankFilters OUTLIERS: sign = −1 for bright outliers, +1 for dark; v·sign + threshold < median·sign → v = median
    med = rank_filter(plane, float(radius), "median")
    sign = np.float32(-1 if which == "Bright" else 1)
    v = plane.astype(np.float32)
    m = med.astype(np.float32)
    replace = (v * sign + np.float32(threshold)) < (m * sign)
    return np.where(replace, med, plane).astype(plane.dtype)


@op("noise.remove_nans", "Remove NaNs", CAT, params=[Param("radius", "float", 2.0, min=0.5, max=100, step=0.5)],
    description="Replace NaN pixels with the median of their non-NaN neighbours (ImageJ's circular kernel; 32-bit).")
def remove_nans(plane, ctx, radius=2.0):
    if plane.dtype != np.float32:
        return plane
    nanmask = np.isnan(plane)
    if not nanmask.any():
        return plane
    fp = rank_kernel(float(radius))
    if fp.sum() <= 400:
        med = _nan_aware_median(plane, fp)
    else:   # very large kernels: the NaN-aware stack would not fit – median of the plane with the NaNs filled first
        filled = np.where(nanmask, np.nanmedian(plane), plane)
        med = ndi.median_filter(filled, footprint=fp, mode="nearest")
    return np.where(nanmask, med, plane).astype(np.float32)
