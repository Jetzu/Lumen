"""Process ▸ Enhance Contrast, Subtract Background, FFT."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure

from ..registry import Param, OpResult, op
from ._imagej import (bandpass_filter_fht, contrast_limits, fft_pad, fht2d, fht_power_spectrum, fht_to_complex, normalize_plane,
                      power_spectrum_8bit_from_fht, shrink_image, slide_parabola_lines,
                      sliding_paraboloid_background as ij_sliding_paraboloid_background, subtract_background as ij_subtract_background)
from ._util import as_float, back_to

CAT = "Process"


def ij_equalize_lut(hist: np.ndarray, classic: bool = False) -> np.ndarray:
    """ImageJ's ``ContrastEnhancer.equalize()`` lookup table for a full-range histogram (256 or 65536 bins).

    Bins are weighted by sqrt(count) (counts below 2 are kept as they are) unless ``classic`` is set – ImageJ's
    Alt-key variant – which uses the plain counts.  The cumulative sum follows ImageJ's "add delta twice" loop
    (``lut[i] = round((w0 + 2·Σ w1..w(i-1) + wi) · scale)``) with Java ``Math.round`` (floor(x + 0.5), not
    banker's rounding); ``lut[0] = 0`` and ``lut[max] = max`` are anchored."""
    h = np.asarray(hist, dtype=np.float64)
    mx = h.size - 1
    w = h if classic else np.where(h < 2, h, np.sqrt(h))
    cs = np.cumsum(w)
    total = 2.0 * cs[mx] - w[0] - w[mx]                 # w0 + 2·Σ w1..w(max-1) + wmax
    scale = mx / total if total > 0 else 0.0
    lut = np.floor((2.0 * cs - w[0] - w) * scale + 0.5).astype(np.int64)
    lut[0] = 0
    lut[mx] = mx
    return np.clip(lut, 0, mx)


@op("process.enhance_contrast", "Enhance Contrast", CAT, kind="image", preview=True, scope_param=False,
    description="Stretch the display range (or the pixel values) so a given fraction of pixels saturate; optionally equalize the histogram. Limits and normalization follow ImageJ's ContrastEnhancer exactly (8-bit: 256-bin histogram; 16-bit: full histogram; 32-bit: 256 bins over the data range, normalized to 0–1).",
    params=[Param("saturated", "float", 0.35, min=0.0, max=100.0, step=0.05, label="Saturated pixels (%)"),
            Param("normalize", "bool", False, help="Rewrite the pixel values instead of only the display range"),
            Param("equalize", "bool", False, help="Histogram equalization (ImageJ's sqrt-weighted variant; 8/16-bit only)"),
            Param("classic_equalization", "bool", False, advanced=True, help="Unweighted (classic) histogram equalization – ImageJ's Alt-key variant"),
            Param("process_all", "bool", True, label="Process all planes", help="Normalize / equalize every plane, each with its own histogram unless 'use stack histogram' is set"),
            Param("use_stack_histogram", "bool", False, label="Use stack histogram", help="Take the histogram from all planes of the channel instead of the current plane")],
    keywords=["brightness", "auto contrast", "normalize", "equalize"])
def enhance_contrast(ctx, saturated=0.35, normalize=False, equalize=False, classic_equalization=False, process_all=True, use_stack_histogram=False):
    img = ctx.img
    d = img.display
    mask = ctx.mask
    dtype = img.data.dtype

    def plane_values(t, z, c):
        p = img.data[t, z, c]
        return p[mask] if mask is not None else p.ravel()

    def channel_values(c):
        return np.concatenate([plane_values(t, z, c) for t in range(img.nt) for z in range(img.nz)])

    if equalize:
        # ImageJ: the LUT comes from the histogram of the selection (whole plane without one) and is applied to
        # the whole plane; one histogram per plane (or the stack histogram); 32-bit images are refused.
        if dtype not in (np.uint8, np.uint16):
            raise ValueError("Equalization of 32-bit images is not supported (convert to 8- or 16-bit first)")
        mx = 255 if dtype == np.uint8 else 65535
        out = img.data.copy()
        planes = img.iter_planes() if process_all else [(d.t, d.z, d.c)]
        stack_luts: dict[int, np.ndarray] = {}
        for (t, z, c) in planes:
            p = img.data[t, z, c]
            if use_stack_histogram:
                lut = stack_luts.get(c)
                if lut is None:
                    lut = stack_luts[c] = ij_equalize_lut(np.bincount(channel_values(c).ravel(), minlength=mx + 1), classic=classic_equalization)
            else:
                lut = ij_equalize_lut(np.bincount(plane_values(t, z, c).ravel(), minlength=mx + 1), classic=classic_equalization)
            out[t, z, c] = lut[p].astype(p.dtype)
        return OpResult(data=out, display={"channels": [{"min": 0, "max": mx} for _ in range(img.nc)]})

    def limits(vals):
        """ImageJ's (min, max) and whether ImageJ would change anything (hmax > hmin); otherwise a usable display range."""
        lim = contrast_limits(vals, dtype, saturated)
        if lim is not None:
            return lim, True
        finite = vals[np.isfinite(vals)] if vals.dtype == np.float32 else vals
        if finite.size == 0:
            return (0.0, 1.0), False
        mn, mx = float(finite.min()), float(finite.max())
        return ((mn, mx) if mx > mn else (mn, mn + 1.0)), False

    if not normalize:
        ranges = []
        for c in range(img.nc):
            vals = channel_values(c) if use_stack_histogram else plane_values(d.t, d.z, c)
            ranges.append(limits(vals)[0])
        return OpResult(display={"channels": [{"min": lo, "max": hi} for lo, hi in ranges]})
    out = img.data.copy()
    full = 255.0 if dtype == np.uint8 else (65535.0 if dtype == np.uint16 else 1.0)
    planes = list(img.iter_planes()) if process_all else [(d.t, d.z, d.c)]
    stack_limits: dict[int, tuple] = {}
    for (t, z, c) in planes:
        if use_stack_histogram:
            if c not in stack_limits:
                stack_limits[c] = limits(channel_values(c))
            (lo, hi), ok = stack_limits[c]
        else:
            (lo, hi), ok = limits(plane_values(t, z, c))
        if not ok:          # ImageJ: hmax ≤ hmin → the plane stays untouched
            continue
        p = img.data[t, z, c]
        res = normalize_plane(p, lo, hi)
        out[t, z, c] = np.where(mask, res, p) if mask is not None else res
    return OpResult(data=out, display={"channels": [{"min": 0, "max": full} for _ in range(img.nc)]})


@op("process.clahe", "CLAHE (Local Contrast)", CAT, description="Contrast-limited adaptive histogram equalization.",
    params=[Param("block_size", "int", 127, min=8, max=2048, help="Tile size in pixels"), Param("clip_limit", "float", 0.01, min=0.001, max=0.2, step=0.005), Param("bins", "int", 256, min=16, max=1024, advanced=True)])
def clahe(plane, ctx, block_size=127, clip_limit=0.01, bins=256):
    f = as_float(plane)
    lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    span = hi - lo if hi > lo else 1.0
    n = (f - lo) / span
    out = exposure.equalize_adapthist(n, kernel_size=block_size, clip_limit=clip_limit, nbins=bins)
    full = 255.0 if plane.dtype == np.uint8 else (65535.0 if plane.dtype == np.uint16 else span)
    off = 0.0 if plane.dtype != np.float32 else lo
    return back_to(out * full + off, plane)


def parabola_envelope_1d(f: np.ndarray, coeff2: float) -> np.ndarray:
    """One pass of ImageJ's ``BackgroundSubtracter.lineSlideParabola`` over a 1-D profile: the parabola
    y = y0 − coeff2·(x − x0)² slides along the line from below; the points it touches keep their value, the others
    get the parabola (ImageJ's float32 arithmetic; lines shorter than 3 samples are returned unchanged)."""
    f = np.asarray(f, dtype=np.float32)
    return slide_parabola_lines(f[None, :], np.array([f.shape[0]]), np.float32(coeff2))[0]


def sliding_paraboloid_background(src: np.ndarray, radius: float) -> np.ndarray:
    """ImageJ's sliding-paraboloid background of a 2-D float image (``slidingParaboloidFloatBackground`` without
    pre-smoothing): corner correction, then the parabola passes along rows, columns, rows again and the four diagonal
    directions (curvature 0.5/radius, doubled on the diagonals where the sample spacing is √2), each pass working on
    the result of the previous one."""
    return ij_sliding_paraboloid_background(np.asarray(src, dtype=np.float32), float(radius), invert=False, presmooth=False, correct=True)


def _shrink_plane(src: np.ndarray, shrink: int) -> np.ndarray:
    """ImageJ's ``BackgroundSubtracter.shrinkImage``: the minimum of every shrink×shrink block (ceil-sized, so a
    1–4 px high/wide image keeps at least one row/column)."""
    return shrink_image(np.asarray(src, dtype=np.float32), int(shrink))


@op("process.subtract_background", "Subtract Background", CAT,
    description="Rolling-ball or sliding-paraboloid background subtraction exactly as ImageJ's BackgroundSubtracter (shrink / roll / interpolate, 3×3 pre-smoothing, corner correction), or a fast top-hat.",
    params=[Param("radius", "float", 50.0, min=1, max=2000, step=1, label="Rolling ball radius"), Param("light_background", "bool", False),
            Param("create_background", "bool", False, help="Return the background instead of subtracting it"),
            Param("method", "choice", "Rolling ball", choices=["Rolling ball", "Sliding paraboloid", "Top-hat (fast)"], advanced=True),
            Param("smooth", "bool", True, help="3×3 smoothing before estimating the background", advanced=True)], keywords=["rolling ball", "flatten", "uneven illumination", "paraboloid"])
def subtract_background(plane, ctx, radius=50.0, light_background=False, create_background=False, method="Rolling ball", smooth=True):
    if method.startswith("Top"):
        from skimage.morphology import opening, disk
        f = as_float(plane)
        if light_background:
            f = -f
        src = ndi.uniform_filter(f, 3, mode="nearest") if smooth else f
        bg = opening(src, disk(int(radius)))
        if light_background:
            bg = -bg
            f = -f
        if create_background:
            return back_to(bg, plane)
        out = f - bg
        if light_background:
            out = out + (255.0 if plane.dtype == np.uint8 else (65535.0 if plane.dtype == np.uint16 else 0.0))
        return back_to(out, plane)
    return ij_subtract_background(plane, float(radius), light_background=bool(light_background), create_background=bool(create_background),
                                  paraboloid=method.startswith("Sliding"), presmooth=bool(smooth))


FFT_OPTION_DEFAULTS = {"raw_power_spectrum": False, "fft_window": False, "complex_output": False, "do_forward": True}


def fft_store(session) -> dict:
    """session.settings['fft']: image id of a power-spectrum image → {F (centred complex transform), source_id, shape, dtype, name}."""
    store = session.settings.setdefault("fft", {})
    for k in [k for k in store if k not in session.images]:  # transforms of closed images
        store.pop(k, None)
    return store


def fft_options(session) -> dict:
    return {**FFT_OPTION_DEFAULTS, **session.settings.get("fft_options", {})}


def power_spectrum_8bit(F: np.ndarray) -> np.ndarray:
    """ImageJ-style 8-bit display (``FHT.getPowerSpectrum`` scaling) of a centred complex transform – used for
    frequency-domain images that only carry a complex transform (FD Math results); images made by Process ▸ FFT carry
    ImageJ's float32 Hartley transform and use ``power_spectrum_8bit_from_fht``."""
    from ._imagej import _scale_power_spectrum
    return _scale_power_spectrum((np.abs(F) ** 2).astype(np.float32))


def _windowed(plane: np.ndarray) -> np.ndarray:
    p = as_float(plane)
    wy = np.hanning(p.shape[0])[:, None] if p.shape[0] > 1 else np.ones((1, 1))
    wx = np.hanning(p.shape[1])[None, :] if p.shape[1] > 1 else np.ones((1, 1))
    return (p * (wy * wx)).astype(np.float32)


def forward_fft(plane: np.ndarray, window: bool = False) -> np.ndarray:
    """Centred complex transform of a plane (no padding; FD Math / Custom Filter)."""
    p = _windowed(plane) if window else as_float(plane)
    return np.fft.fftshift(np.fft.fft2(p.astype(np.float64)))


def make_fft_image(F: np.ndarray, source, session, name: str | None = None, fht: np.ndarray | None = None,
                   orig_shape: tuple[int, int] | None = None):
    """Create the 8-bit power-spectrum image of a centred complex transform (or of ImageJ's Hartley transform
    ``fht`` when given) and remember the transform(s) and the un-padded size."""
    ps = power_spectrum_8bit_from_fht(fht) if fht is not None else power_spectrum_8bit(F)
    res = source.copy(name=name or f"FFT of {source.name}", data=ps[None, None, None])
    res.rgb = False
    res.lut_is_binary = False
    res.display.t = res.display.z = res.display.c = 0
    res.sync_channels()
    res.display.channels[0].lut = "Grays"
    res.display.channels[0].min, res.display.channels[0].max = 0.0, 255.0
    res.metadata["fft_source"] = source.id
    res.metadata["fft"] = True
    fft_store(session)[res.id] = {"F": F.astype(np.complex128), "fht": None if fht is None else fht.astype(np.float32),
                                  "source_id": source.id, "shape": tuple(F.shape), "orig_shape": tuple(orig_shape or F.shape),
                                  "dtype": str(source.data.dtype), "name": source.name}
    return res


def stored_fft(ctx, image=None) -> tuple:
    """(entry, fft image) for the active (or given) frequency-domain image; raises when it has no stored transform."""
    img = ctx.session.get_image(image) if image else ctx.img
    entry = fft_store(ctx.session).get(img.id)
    if entry is None:
        raise ValueError(f"{img.name} is not a frequency-domain image created by Process ▸ FFT ▸ FFT (its transform is unknown)")
    return entry, img


@op("process.fft", "FFT", "Process/FFT", kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="Power spectrum of the current plane (8-bit, log-scaled, origin centred) as a new image, computed like ImageJ: the plane is padded with its mean to a square power of two and transformed with ImageJ's Hartley transform; the transform is kept so that Inverse FFT, FD Math and the custom filter can use it. Honours FFT Options (raw power spectrum, FFT window, complex output).",
    keywords=["fourier", "frequency"])
def fft(ctx):
    img = ctx.img
    opts = fft_options(ctx.session)
    plane = img.current_plane()
    src = _windowed(plane) if opts.get("fft_window") else plane
    padded = fft_pad(src)
    fht = fht2d(padded.astype(np.float32))
    F = np.fft.fftshift(np.fft.fft2(padded.astype(np.float64)))
    out = []
    if opts.get("do_forward", True):
        out.append(make_fft_image(F, img, ctx.session, fht=fht, orig_shape=plane.shape))
    if opts.get("raw_power_spectrum"):
        raw = img.copy(name=f"PS of {img.name}", data=np.fft.fftshift(fht_power_spectrum(fht)).astype(np.float32)[None, None, None])
        raw.rgb = False
        raw.display.t = raw.display.z = raw.display.c = 0
        raw.sync_channels()
        raw.metadata["fft_source"] = img.id
        out.append(raw)
    if opts.get("complex_output"):
        C = np.fft.fftshift(fht_to_complex(fht))
        comp = np.stack([C.real, C.imag]).astype(np.float32)[None, :, None]  # 2 slices: real, imaginary
        ci = img.copy(name=f"Complex of {img.name}", data=comp)
        ci.rgb = False
        ci.display.t = ci.display.z = ci.display.c = 0
        ci.sync_channels()
        ci.metadata["slice_labels"] = ["Real", "Imaginary"]
        ci.metadata["fft_source"] = img.id
        out.append(ci)
    if not out:
        raise ValueError("FFT Options disable every output – enable 'Do forward transform', 'Raw power spectrum' or 'Complex output'")
    return OpResult(new_images=out)


@op("process.fft_bandpass", "Bandpass Filter (FFT)", "Process/FFT",
    description="ImageJ's FFT bandpass filter: keep structures between two sizes (Gaussian band filter applied to the mirror-tiled, power-of-two padded plane); optionally suppress horizontal/vertical stripes and autoscale the result.",
    params=[Param("large", "float", 40.0, min=1, max=10000, step=1, label="Filter large structures down to (px)"), Param("small", "float", 3.0, min=0, max=1000, step=0.5, label="Filter small structures up to (px)"),
            Param("suppress", "choice", "None", choices=["None", "Horizontal", "Vertical"], label="Suppress stripes"),
            Param("tolerance", "float", 5.0, min=0, max=100, step=1, label="Tolerance of direction (%)", help="Width of the stripe filter"),
            Param("autoscale", "bool", True, help="Rescale the result to the full range of the image type (ImageJ: Enhance Contrast with 1 % saturation)"),
            Param("saturate", "bool", True, help="Saturate 1 % of the pixels when autoscaling")], keywords=["fourier", "stripes"])
def fft_bandpass(plane, ctx, large=40.0, small=3.0, suppress="None", tolerance=5.0, autoscale=True, saturate=True):
    return bandpass_filter_fht(plane, float(large), float(small), str(suppress), float(tolerance), bool(autoscale), bool(saturate))
