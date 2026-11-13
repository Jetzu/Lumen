"""Process ▸ Filters."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters as skf, restoration

from ..registry import Param, OpResult, op
from ._imagej import (convolve as ij_convolve, find_edges3x3, gaussian_blur as ij_gaussian_blur, rank_filter as ij_rank_filter,
                      rank_kernel, sharpen3x3, smooth3x3, top_hat as ij_top_hat, unsharp_mask as ij_unsharp_mask)
from ._util import as_float, back_to, footprint

CAT = "Process/Filters"


@op("filters.gaussian", "Gaussian Blur", CAT, shortcut="", description="Smooth with a Gaussian kernel of the given sigma (radius) – ImageJ's GaussianBlur (edge-corrected kernel, accuracy 0.002 for 8-bit / 0.0002 otherwise, downscaling for large sigma).",
    params=[Param("sigma", "float", 2.0, min=0.0, step=0.1, help="Standard deviation in pixels, or in calibrated units with 'Scaled units' (any value, like ImageJ)"),
            Param("scaled_units", "bool", False, label="Scaled units", advanced=True,
                  help="Sigma is in the image's calibrated units: ImageJ's 'Scaled units' option divides it by the pixel width (X) and pixel height (Y)")],
    keywords=["blur", "smooth"])
def gaussian(plane, ctx, sigma=2.0, scaled_units=False):
    if sigma <= 0:
        return plane
    sx = sy = float(sigma)
    if scaled_units:  # GaussianBlur.run: sigmaX = sigma / cal.pixelWidth, sigmaY = sigma / cal.pixelHeight
        cal = ctx.img.calibration
        sx = sx / float(cal.pixel_width or 1.0)
        sy = sy / float(cal.pixel_height or 1.0)
    return ij_gaussian_blur(plane, sx, sy)


@op("filters.gaussian3d", "Gaussian Blur 3D", CAT, kind="image", description="Gaussian smoothing across X, Y and Z of every channel/frame.",
    params=[Param("sigma_xy", "float", 2.0, min=0, max=100, step=0.1), Param("sigma_z", "float", 1.0, min=0, max=100, step=0.1)], scope_param=False)
def gaussian3d(ctx, sigma_xy=2.0, sigma_z=1.0):
    img = ctx.img
    out = img.data.astype(np.float32).copy()
    for t in range(img.nt):
        for c in range(img.nc):
            out[t, :, c] = ndi.gaussian_filter(out[t, :, c], (sigma_z, sigma_xy, sigma_xy), mode="nearest")
    return OpResult(data=back_to(out, img.data))


@op("filters.median", "Median", CAT, description="Replace each pixel with the median of its neighbourhood (removes salt-and-pepper noise).",
    params=[Param("radius", "float", 2.0, min=0.5, max=1000, step=0.5), Param("shape", "choice", "disk", choices=["disk", "square"], advanced=True)])
def median(plane, ctx, radius=2.0, shape="disk"):
    if shape == "square":
        return ndi.median_filter(plane, footprint=footprint(radius, shape), mode="nearest")
    return ij_rank_filter(plane, float(radius), "median")   # ImageJ's circular kernel (dx²+dy² ≤ r²+1)


@op("filters.mean", "Mean", CAT, description="Replace each pixel with the neighbourhood average.",
    params=[Param("radius", "float", 2.0, min=0.5, max=1000, step=0.5), Param("shape", "choice", "disk", choices=["disk", "square"], advanced=True)])
def mean(plane, ctx, radius=2.0, shape="disk"):
    if shape == "square":
        fp = footprint(radius, shape).astype(np.float32)
        fp /= fp.sum()
        return back_to(ndi.convolve(as_float(plane), fp, mode="nearest"), plane)
    return ij_rank_filter(plane, float(radius), "mean")


@op("filters.minimum", "Minimum", CAT, description="Grayscale erosion: neighbourhood minimum.",
    params=[Param("radius", "float", 2.0, min=0.5, max=1000, step=0.5)])
def minimum(plane, ctx, radius=2.0):
    return ij_rank_filter(plane, float(radius), "min")


@op("filters.maximum", "Maximum", CAT, description="Grayscale dilation: neighbourhood maximum.",
    params=[Param("radius", "float", 2.0, min=0.5, max=1000, step=0.5)])
def maximum(plane, ctx, radius=2.0):
    return ij_rank_filter(plane, float(radius), "max")


@op("filters.variance", "Variance", CAT, description="Local variance (texture).",
    params=[Param("radius", "float", 2.0, min=0.5, max=1000, step=0.5)])
def variance(plane, ctx, radius=2.0):
    return ij_rank_filter(plane, float(radius), "variance")


@op("filters.unsharp", "Unsharp Mask", CAT, description="Sharpen by subtracting a blurred copy: (I − w·G(I)) / (1 − w).",
    params=[Param("radius", "float", 1.0, min=0.1, max=1000, step=0.1, help="Gaussian sigma"), Param("weight", "float", 0.6, min=0.1, max=0.9, step=0.05, help="Mask weight (0.1–0.9)")])
def unsharp(plane, ctx, radius=1.0, weight=0.6):
    return ij_unsharp_mask(plane, float(radius), float(weight))   # ImageJ's UnsharpMask (Gaussian accuracy 0.01, float32)


@op("filters.convolve", "Convolve", CAT, description="Convolve with a custom kernel (rows separated by newlines, values by spaces).",
    params=[Param("kernel", "text", "0 -1 0\n-1 5 -1\n0 -1 0"), Param("normalize", "bool", True, help="Divide by the kernel sum when it is not zero")])
def convolve(plane, ctx, kernel="", normalize=True):
    rows = [list(map(float, r.replace(",", " ").split())) for r in kernel.strip().splitlines() if r.strip()]
    k = np.asarray(rows, np.float32)
    if k.ndim != 2 or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Kernel must be rectangular")
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError(f"Kernel width or height not odd ({k.shape[1]}x{k.shape[0]})")
    return ij_convolve(plane, k, bool(normalize))   # ImageJ's Convolver (float32 products, double sum, edge pixels replicated)


@op("filters.tophat", "Top Hat", CAT, description="ImageJ's Top Hat: the image minus its grayscale opening with the circular kernel (bright features smaller than the radius; also flattens the background). With a light background the closing is subtracted from the type maximum.",
    params=[Param("radius", "float", 15.0, min=1, max=500, step=1), Param("light_background", "bool", False, help="Dark features on a light background"),
            Param("dont_subtract", "bool", False, advanced=True, label="Don't subtract (grayscale open)", help="Return the opening (closing for a light background) itself")])
def tophat(plane, ctx, radius=15.0, light_background=False, dont_subtract=False):
    return ij_top_hat(plane, float(radius), bool(light_background), bool(dont_subtract))


@op("filters.dog", "Difference of Gaussians", CAT, description="Band-pass blob enhancement: G(σ1) − G(σ2).",
    params=[Param("sigma1", "float", 1.0, min=0, max=100, step=0.1), Param("sigma2", "float", 4.0, min=0, max=100, step=0.1)])
def dog(plane, ctx, sigma1=1.0, sigma2=4.0):
    f = as_float(plane)
    d = ndi.gaussian_filter(f, sigma1, mode="nearest") - ndi.gaussian_filter(f, sigma2, mode="nearest")
    if plane.dtype != np.float32:
        d = d + (np.iinfo(plane.dtype).max // 2 if plane.dtype == np.uint8 else 0)
    return back_to(d, plane)


@op("filters.bilateral", "Bilateral Filter", CAT, description="Edge-preserving smoothing (denoise_bilateral).",
    params=[Param("sigma_spatial", "float", 3.0, min=0.5, max=50, step=0.5), Param("sigma_color", "float", 0.1, min=0.001, max=1.0, step=0.01, help="Relative to the data range")])
def bilateral(plane, ctx, sigma_spatial=3.0, sigma_color=0.1):
    f = as_float(plane)
    lo, hi = float(f.min()), float(f.max())
    span = hi - lo if hi > lo else 1.0
    n = (f - lo) / span
    out = restoration.denoise_bilateral(n, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)
    return back_to(out * span + lo, plane)


@op("filters.nlmeans", "Non-local Means Denoise", CAT, description="Patch-based denoising (slow on large images).",
    params=[Param("h", "float", 0.1, min=0.01, max=1.0, step=0.01, help="Filter strength relative to the data range"), Param("patch_size", "int", 5, min=3, max=15), Param("patch_distance", "int", 6, min=1, max=30)])
def nlmeans(plane, ctx, h=0.1, patch_size=5, patch_distance=6):
    f = as_float(plane)
    lo, hi = float(f.min()), float(f.max())
    span = hi - lo if hi > lo else 1.0
    n = (f - lo) / span
    out = restoration.denoise_nl_means(n, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True, channel_axis=None)
    return back_to(out * span + lo, plane)


@op("process.smooth", "Smooth", "Process", description="3×3 mean filter (ImageJ's Smooth: integer (sum+4)/9 for 8/16-bit).", shortcut="Shift+S")
def smooth(plane, ctx):
    if plane.shape[1] <= 1:   # ImageProcessor.smooth: nothing to do for a 1-pixel-wide image
        return plane
    return smooth3x3(plane)


@op("process.sharpen", "Sharpen", "Process", description="3×3 sharpening kernel (ImageJ's Sharpen: [-1 -1 -1; -1 12 -1; -1 -1 -1]/4 with ImageJ's integer rounding).")
def sharpen(plane, ctx):
    if plane.shape[1] <= 1:
        return plane
    return sharpen3x3(plane)


@op("process.find_edges", "Find Edges", "Process", description="Sobel edge magnitude (ImageJ's Find Edges: truncated for 8/16-bit).")
def find_edges(plane, ctx):
    if plane.shape[1] <= 1:
        return plane
    return find_edges3x3(plane)


@op("process.laplacian", "Laplacian of Gaussian", "Process", description="Blob / edge detector: −σ²·∇²G.",
    params=[Param("sigma", "float", 2.0, min=0.1, max=50, step=0.1)])
def log_filter(plane, ctx, sigma=2.0):
    f = as_float(plane)
    out = -ndi.gaussian_laplace(f, sigma, mode="nearest") * sigma * sigma
    if plane.dtype != np.float32:
        out = out + (128 if plane.dtype == np.uint8 else 0)
    return back_to(out, plane)


@op("filters.frangi", "Ridge / Vessel Filter", CAT, description="Frangi vesselness for tubular structures.",
    params=[Param("sigma_min", "float", 1.0, min=0.5, max=20, step=0.5), Param("sigma_max", "float", 4.0, min=0.5, max=50, step=0.5), Param("bright", "bool", True, help="Ridges are brighter than background")])
def frangi(plane, ctx, sigma_min=1.0, sigma_max=4.0, bright=True):
    f = as_float(plane)
    sig = np.linspace(sigma_min, max(sigma_max, sigma_min), 4)
    out = skf.frangi(f, sigmas=sig, black_ridges=not bright)
    out = out / (out.max() or 1)
    scale = 255 if plane.dtype == np.uint8 else (65535 if plane.dtype == np.uint16 else 1)
    return back_to(out * scale, plane)


_SHADOW = {
    "North": [[1, 2, 1], [0, 1, 0], [-1, -2, -1]], "South": [[-1, -2, -1], [0, 1, 0], [1, 2, 1]],
    "East": [[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]], "West": [[1, 0, -1], [2, 1, -2], [1, 0, -1]],
    "Northwest": [[2, 1, 0], [1, 1, -1], [0, -1, -2]], "Southeast": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    "Northeast": [[0, 1, 2], [-1, 1, 1], [-2, -1, 0]], "Southwest": [[0, -1, -2], [1, 1, -1], [2, 1, 0]],
}


@op("process.shadows", "Shadows", "Process", description="Emboss-like relief with light coming from the chosen direction.",
    params=[Param("direction", "choice", "North", choices=list(_SHADOW.keys()))])
def shadows(plane, ctx, direction="North"):
    k = np.asarray(_SHADOW[direction], np.float32)
    return back_to(ndi.convolve(as_float(plane), k, mode="nearest"), plane)


@op("filters.rank3d", "3D Filters (Median / Mean / Min / Max)", CAT, kind="image", scope_param=False, preview=False,
    description="Rank filters over X, Y and Z of every channel/frame.",
    params=[Param("filter", "choice", "Median", choices=["Median", "Mean", "Minimum", "Maximum", "Variance"]), Param("radius_xy", "float", 2.0, min=0.5, max=20, step=0.5), Param("radius_z", "float", 1.0, min=0, max=20, step=0.5)])
def rank3d(ctx, filter="Median", radius_xy=2.0, radius_z=1.0):
    img = ctx.img
    rz, rxy = int(round(radius_z)), int(round(radius_xy))
    zz, yy, xx = np.mgrid[-rz:rz + 1, -rxy:rxy + 1, -rxy:rxy + 1]
    fp = ((zz / max(rz, 1e-9)) ** 2 + (yy / max(rxy, 1e-9)) ** 2 + (xx / max(rxy, 1e-9)) ** 2) <= 1.0 if rz > 0 else ((yy / max(rxy, 1e-9)) ** 2 + (xx / max(rxy, 1e-9)) ** 2) <= 1.0
    out = np.zeros(img.shape, np.float32)
    for t in range(img.nt):
        for c in range(img.nc):
            vol = img.data[t, :, c].astype(np.float32)
            if filter == "Median":
                out[t, :, c] = ndi.median_filter(vol, footprint=fp, mode="nearest")
            elif filter == "Mean":
                out[t, :, c] = ndi.convolve(vol, fp.astype(np.float32) / fp.sum(), mode="nearest")
            elif filter == "Minimum":
                out[t, :, c] = ndi.minimum_filter(vol, footprint=fp, mode="nearest")
            elif filter == "Maximum":
                out[t, :, c] = ndi.maximum_filter(vol, footprint=fp, mode="nearest")
            else:
                k = fp.astype(np.float32) / fp.sum()
                m = ndi.convolve(vol, k, mode="nearest")
                out[t, :, c] = np.maximum(ndi.convolve(vol * vol, k, mode="nearest") - m * m, 0)
    return OpResult(data=back_to(out, img.data))


@op("filters.normalize_local_contrast", "Normalize Local Contrast", CAT,
    description="Fiji's Normalize Local Contrast: every pixel is centred on the mean and/or stretched by the standard deviation of its local block so that ±n·SD spans the full range.",
    params=[Param("block_radius_x", "int", 40, min=1, max=1000), Param("block_radius_y", "int", 40, min=1, max=1000),
            Param("standard_deviations", "float", 3.0, min=0.1, max=20, step=0.5), Param("center", "bool", True), Param("stretch", "bool", True)],
    keywords=["nlc", "local contrast", "flatten"])
def normalize_local_contrast(plane, ctx, block_radius_x=40, block_radius_y=40, standard_deviations=3.0, center=True, stretch=True):
    f = as_float(plane).astype(np.float64)
    size = (2 * int(block_radius_y) + 1, 2 * int(block_radius_x) + 1)
    mean = ndi.uniform_filter(f, size, mode="reflect")
    var = np.maximum(ndi.uniform_filter(f * f, size, mode="reflect") - mean * mean, 0)
    std = np.sqrt(var)
    if plane.dtype == np.uint8:
        lo, hi = 0.0, 255.0
    elif plane.dtype == np.uint16:
        lo, hi = 0.0, 65535.0
    else:
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        if hi <= lo:
            hi = lo + 1.0
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    v = f
    if center:
        v = v - mean
    if stretch:
        v = v / np.maximum(standard_deviations * std, 1e-9) * half
    if center:
        v = v + mid
    return back_to(np.clip(v, lo, hi), plane)


@op("filters.rof_denoise", "ROF Denoise", CAT, description="Rudin–Osher–Fatemi total-variation denoising (Chambolle's algorithm). Larger theta smooths more.",
    params=[Param("theta", "float", 25.0, min=0.1, max=1000, step=1, help="Fidelity weight in gray levels (ImageJ/Fiji ROF Denoise convention: 25 for 8-bit images)"),
            Param("iterations", "int", 200, min=10, max=2000, advanced=True)], keywords=["total variation", "tv", "chambolle"])
def rof_denoise(plane, ctx, theta=25.0, iterations=200):
    from skimage.restoration import denoise_tv_chambolle
    f = as_float(plane).astype(np.float64)
    if plane.dtype == np.uint8:
        scale = 255.0
    elif plane.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = float(np.nanmax(f) - np.nanmin(f)) or 1.0
    lo = 0.0 if plane.dtype != np.float32 else float(np.nanmin(f))
    n = (f - lo) / scale
    out = denoise_tv_chambolle(n, weight=theta / scale, max_num_iter=int(iterations), channel_axis=None)
    return back_to(out * scale + lo, plane)


@op("filters.shape_index", "Shape Index", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="Koenderink & van Doorn shape index (−1 cup … 0 saddle … +1 cap) from the Hessian eigenvalues at scale sigma; the result is 32-bit (NaN where the surface is flat).",
    params=[Param("sigma", "float", 2.0, min=0.3, max=50, step=0.1), Param("flat_to_zero", "bool", True, help="Write 0 instead of NaN in flat regions")], keywords=["curvature", "hessian"])
def shape_index(ctx, sigma=2.0, flat_to_zero=True):
    from skimage.feature import shape_index as sk_shape_index
    img = ctx.img
    out = np.zeros(img.shape, np.float32)
    mask = ctx.mask
    with np.errstate(all="ignore"):
        for (t, z, c) in img.iter_planes():
            si = sk_shape_index(as_float(img.data[t, z, c]).astype(np.float64), sigma=sigma, mode="nearest").astype(np.float32)
            if flat_to_zero:
                si = np.nan_to_num(si, nan=0.0)
            out[t, z, c] = np.where(mask, si, img.data[t, z, c]) if mask is not None else si
    return OpResult(data=out, rgb=False, display={"channels": [{"min": -1.0, "max": 1.0} for _ in range(img.nc)]})


def circular_kernel(radius: float) -> np.ndarray:
    """ImageJ RankFilters kernel (``makeLineRadii``): pixel (dx, dy) belongs to the mask when dx² + dy² ≤ (int)(r²) + 1."""
    return rank_kernel(radius)


@op("filters.show_circular_masks", "Show Circular Masks", CAT, kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False,
    description="Stack showing the circular kernels used by the rank filters (Mean, Median, Minimum, Maximum, Variance) for radii 0.5 … 10 in steps of 0.5, like ImageJ.",
    params=[Param("max_radius", "float", 10.0, min=0.5, max=50, step=0.5)], keywords=["kernel", "rank filter", "footprint"])
def show_circular_masks(ctx, max_radius=10.0):
    radii = [0.5 * k for k in range(1, int(round(max_radius / 0.5)) + 1)]
    size = 2 * int(np.sqrt(max_radius * max_radius + 1 + 1e-10)) + 1 + 4
    stack = np.zeros((len(radii), size, size), np.uint8)
    cy = cx = size // 2
    for i, r in enumerate(radii):
        k = circular_kernel(r)
        h = k.shape[0] // 2
        stack[i, cy - h:cy + h + 1, cx - h:cx + h + 1] = k.astype(np.uint8) * 255
    from ..image import LumenImage
    res = LumenImage(stack, name="Circular Masks", axes="ZYX")
    res.metadata["slice_labels"] = [f"radius={r:g}" for r in radii]
    res.lut_is_binary = True
    return OpResult(new_images=[res], message=f"{len(radii)} kernels (radius 0.5 … {radii[-1]:g})")
