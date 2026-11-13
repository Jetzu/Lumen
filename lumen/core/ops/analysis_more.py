"""Analyze ▸ Tools ▸ Analyze Line Graph / Grid, Compare Images, Directionality."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..image import LumenImage
from ..registry import Param, OpResult, op

CAT = "Analyze/Tools"


# ====================================================================== Analyze Line Graph
def _border_mode_rgb(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, rgb.shape[-1])
    cols, counts = np.unique(border, axis=0, return_counts=True)
    return cols[int(np.argmax(counts))].astype(np.float64)


def extract_curves(rgb: np.ndarray, tolerance: float = 40.0, max_curves: int = 6, min_pixels: int = 20, axis_tolerance: float = 30.0) -> list[dict[str, Any]]:
    """Curves of a plot image: non-background colours (quantised) → one series per colour, median y per x column."""
    bg = _border_mode_rgb(rgb)
    f = rgb.astype(np.float64)
    diff = np.abs(f - bg).max(axis=-1) > tolerance
    if not diff.any():
        return []
    q = (f[diff] // 32).astype(np.int64)  # 8 levels per channel
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    series = []
    full_keys = np.full(diff.shape, -1, np.int64)
    full_keys[diff] = keys
    gray_bg = bool(np.allclose(bg, bg[0]))
    for idx in order[:max_curves + 2]:
        k, n = int(uniq[idx]), int(counts[idx])
        if n < min_pixels:
            continue
        m = full_keys == k
        colour = f[m].mean(axis=0)
        # a black / gray colour spanning (almost) full rows or columns is an axis / frame – skip it
        rows_hit = np.any(m, axis=1).sum(); cols_hit = np.any(m, axis=0).sum()
        is_gray = np.ptp(colour) < axis_tolerance
        if is_gray and gray_bg and (cols_hit >= 0.9 * rgb.shape[1] or rows_hit >= 0.9 * rgb.shape[0]):
            ys_, xs_ = np.nonzero(m)
            # keep the axis candidate only when it has a non-axis-like shape
            col_counts = np.bincount(xs_, minlength=rgb.shape[1])
            if (col_counts > 0.5 * rgb.shape[0]).any() or (np.bincount(ys_, minlength=rgb.shape[0]) > 0.5 * rgb.shape[1]).any():
                continue
        series.append({"mask": m, "color": "#%02x%02x%02x" % tuple(int(c) for c in colour), "n": n})
        if len(series) >= max_curves:
            break
    return series


@op("analyze.line_graph", "Analyze Line Graph", CAT, kind="analysis", preview=False, roi_aware=True,
    params=[Param("tolerance", "float", 40.0, min=1, max=255, step=1, help="Colour distance from the background that counts as curve"),
            Param("max_curves", "int", 4, min=1, max=20), Param("invert_y", "bool", True, help="Report y measured from the bottom of the plot (image rows run downwards)"),
            Param("x_min", "float", 0.0, step=1, label="X axis start"), Param("x_max", "float", 0.0, step=1, label="X axis end (0 = pixel units)"),
            Param("y_min", "float", 0.0, step=1, label="Y axis start"), Param("y_max", "float", 0.0, step=1, label="Y axis end (0 = pixel units)")],
    description="Digitise the curve(s) of a plot image: pixels that differ from the background are grouped by colour and the median y of every column gives the curve → plot + Results rows (X, Y per curve). Draw a rectangle around the plot area to restrict the search (its edges then map to the axis ranges).",
    keywords=["digitise", "digitize", "plot", "curve", "graph"])
def line_graph(ctx, tolerance=40.0, max_curves=4, invert_y=True, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
    from ..render import render_rgb
    img = ctx.img
    rgb = render_rgb(img)
    x0 = y0 = 0
    H, W = rgb.shape[:2]
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        bx, by, bw, bh = roimod.bounding_box(ctx.roi)
        x0, y0 = max(int(bx), 0), max(int(by), 0)
        rgb = rgb[y0:min(int(by + bh), H), x0:min(int(bx + bw), W)]
        H, W = rgb.shape[:2]
    curves = extract_curves(rgb, tolerance=tolerance, max_curves=max_curves)
    if not curves:
        raise ValueError("No curve found: nothing differs from the background by more than the tolerance")
    sx = (x_max - x_min) / max(W - 1, 1) if x_max != x_min else 1.0
    sy = (y_max - y_min) / max(H - 1, 1) if y_max != y_min else 1.0
    rows: list[dict[str, Any]] = []
    series = []
    xs_union: set[float] = set()
    for k, cv in enumerate(curves, 1):
        m = cv["mask"]
        cols = np.nonzero(m.any(axis=0))[0]
        xs, ys = [], []
        for x in cols:
            yy = np.nonzero(m[:, x])[0]
            y = float(np.median(yy))
            ypix = (H - 1 - y) if invert_y else y
            xv = x_min + x * sx if x_max != x_min else float(x + x0)
            yv = y_min + ypix * sy if y_max != y_min else (ypix if invert_y else float(y + y0))
            xs.append(xv); ys.append(yv)
            rows.append({"Label": f"{img.name}:curve {k}", "Curve": k, "X": xv, "Y": yv})
        series.append({"name": f"curve {k} ({cv['color']})", "values": ys, "x": xs, "color": cv["color"]})
        xs_union.update(xs)
    # Every curve has its own column range, so each series carries its own ``x`` (same length as its ``values``) and
    # must be drawn against it; plot["x"] is the sorted union of all curves' x values (axis range, hover grid, CSV),
    # never the x list of one curve, which would shift the others.
    plot = {"title": f"Line graph of {img.name}", "x": sorted(xs_union), "series": series, "x_label": "X", "y_label": "Y"}
    return OpResult(results=rows, plot=plot, message=f"{len(curves)} curve(s), {len(rows)} points")


# ====================================================================== Grid
GRID_TYPES = ["Lines", "Horizontal Lines", "Crosses", "Points", "Combined"]


@op("overlay.grid", "Grid", CAT, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("type", "choice", "Lines", choices=GRID_TYPES), Param("area_per_point", "float", 0.0, min=0, step=100, help="Calibrated units²; 0 = use the spacing"),
            Param("spacing", "float", 50.0, min=1, step=1, help="Grid spacing in pixels (used when area per point is 0)"), Param("random_offset", "bool", False),
            Param("color", "string", "", help="Empty = session ROI colour"), Param("stroke_width", "float", 0.0, min=0, max=10, step=0.5, help="0 = session ROI stroke width")],
    description="Overlay a stereology grid (lines, horizontal lines, crosses, points or lines + points) with a given area per point or spacing, optionally randomly offset (Analyze ▸ Tools ▸ Grid).",
    keywords=["stereology", "grid", "counting", "points"], undo_label="Grid")
def grid(ctx, type="Lines", area_per_point=0.0, spacing=50.0, random_offset=False, color="", stroke_width=0.0):
    img = ctx.img
    s = ctx.session
    cal = img.calibration
    if area_per_point > 0:
        tile = math.sqrt(area_per_point / (cal.pixel_width * cal.pixel_height))
    else:
        tile = float(spacing)
    if tile < 1:
        raise ValueError("The grid spacing must be at least one pixel")
    W, H = img.width, img.height
    rng = np.random.default_rng()
    xoff = float(rng.random() * tile) if random_offset else tile / 2.0
    yoff = float(rng.random() * tile) if random_offset else tile / 2.0
    col = color or s.settings.get("roi_color", "#00ffff")
    sw = float(stroke_width) if stroke_width > 0 else float(s.settings.get("roi_stroke_width", 1.0))
    xs = np.arange(xoff, W, tile)
    ys = np.arange(yoff, H, tile)
    items: list[dict[str, Any]] = []

    def line(x1, y1, x2, y2):
        items.append({"type": "line", "id": roimod.new_id(), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "width": 1, "color": col, "stroke_width": sw, "grid": True})

    if type in ("Lines", "Combined"):
        for x in xs:
            line(x, 0, x, H)
        for y in ys:
            line(0, y, W, y)
    elif type == "Horizontal Lines":
        for y in ys:
            line(0, y, W, y)
    elif type == "Crosses":
        arm = max(tile / 8.0, 2.0)
        for x in xs:
            for y in ys:
                line(x - arm, y, x + arm, y)
                line(x, y - arm, x, y + arm)
    if type in ("Points", "Combined"):
        pts = [[float(x), float(y)] for y in ys for x in xs]
        items.append({"type": "point", "id": roimod.new_id(), "points": pts, "color": col, "stroke_width": sw, "grid": True, "name": "Grid points"})
    keep = [o for o in img.overlay if not o.get("grid")]
    n_pts = len(xs) * len(ys)
    area = tile * tile * cal.pixel_width * cal.pixel_height
    unit = cal.unit if cal.scaled else "pixel"
    return OpResult(overlay=keep + items, message=f"{type}: spacing {tile:.2f} px, {n_pts} points, {area:.4g} {unit}² per point")


# ====================================================================== Compare Images
@op("analyze.compare_images", "Compare Images", CAT, kind="analysis", preview=False, roi_aware=True,
    params=[Param("image1", "image", None), Param("image2", "image", None), Param("difference_image", "bool", True, help="Create a 32-bit image1 − image2 image")],
    description="Similarity of two same-sized images (current planes): Pearson correlation, RMSE, MAE, PSNR and SSIM → Results row + table 'Compare Images'; optionally the 32-bit difference image.",
    keywords=["similarity", "psnr", "ssim", "rmse", "difference"])
def compare_images(ctx, image1=None, image2=None, difference_image=True):
    s = ctx.session
    a = s.get_image(image1) if image1 else ctx.img
    if image2:
        b = s.get_image(image2)
    else:
        others = [im for im in s.images.values() if im.id != a.id]
        if not others:
            raise ValueError("Open a second image to compare with")
        b = others[-1]
    if (a.height, a.width) != (b.height, b.width):
        raise ValueError(f"{a.name} ({a.width}×{a.height}) and {b.name} ({b.width}×{b.height}) must have the same size")
    pa = a.current_plane().astype(np.float64)
    pb = b.current_plane().astype(np.float64)
    # the selection the client sends belongs to the ACTIVE image (ctx.image) and is rasterised to ITS size: it is only
    # meaningful – and only shape-compatible – when that image is one of the two compared images (a and b are already
    # the same size).  A selection on any other open image is ignored instead of failing with a raw numpy IndexError
    # (or silently masking the wrong image when the sizes happen to match).
    mask = ctx.mask if (ctx.image is not None and ctx.image.id in (a.id, b.id)) else None
    va = pa[mask] if mask is not None else pa.ravel()
    vb = pb[mask] if mask is not None else pb.ravel()
    ok = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[ok], vb[ok]
    if va.size < 2:
        raise ValueError("Not enough finite pixels to compare")
    diff = va - vb
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    sa, sb = va.std(), vb.std()
    corr = float(np.corrcoef(va, vb)[0, 1]) if sa > 0 and sb > 0 else float("nan")
    rng = 255.0 if a.data.dtype == np.uint8 else (65535.0 if a.data.dtype == np.uint16 else float(max(va.max(), vb.max()) - min(va.min(), vb.min())) or 1.0)
    psnr = float(20 * np.log10(rng / rmse)) if rmse > 0 else float("inf")
    from skimage.metrics import structural_similarity
    # SSIM of the selection, like the other metrics: both planes are cropped to the bounding box of the very mask used
    # above (integer, inside the image – unlike roimod.bounding_box) and, for a non-rectangular selection, the SSIM map
    # is averaged over the mask without the window border that skimage's scalar SSIM leaves out too (so rectangles and
    # free-form selections agree). Too small a selection (< 3 px on a side) reports NaN.
    try:
        ca, cb, cm = pa, pb, None
        if mask is not None:
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                raise ValueError("empty selection")
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ca, cb, cm = pa[y0:y1, x0:x1], pb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        win = min(7, min(ca.shape))  # odd window that fits the crop, not the whole plane
        win -= 1 - win % 2
        if win < 3:
            raise ValueError("selection too small for SSIM")
        if cm is None or cm.all():
            ssim = float(structural_similarity(ca, cb, data_range=rng, win_size=win))
        else:
            _, smap = structural_similarity(ca, cb, data_range=rng, win_size=win, full=True)
            pad = (win - 1) // 2
            inner = np.zeros_like(cm)
            inner[pad:cm.shape[0] - pad, pad:cm.shape[1] - pad] = True
            sel = cm & inner
            ssim = float(smap[sel if sel.any() else cm].mean())
    except Exception:
        ssim = float("nan")
    row = {"Label": f"{a.name} vs {b.name}", "Correlation": corr, "RMSE": rmse, "MAE": mae, "PSNR": psnr, "SSIM": ssim, "Pixels": int(va.size)}
    fin = lambda v: v if (isinstance(v, float) and np.isfinite(v)) else None  # noqa: E731 – tables are sent as JSON (no NaN / Infinity)
    table = {"name": "Compare Images", "columns": ["Metric", "Value"],
             "rows": [["Image 1", a.name], ["Image 2", b.name], ["Pearson correlation", fin(corr)], ["RMSE", fin(rmse)], ["MAE", fin(mae)], ["PSNR (dB)", fin(psnr)], ["SSIM", fin(ssim)], ["Pixels", int(va.size)]]}
    out = OpResult(results=[row], table=table, message=f"r = {corr:.4f}, RMSE = {rmse:.4g}, SSIM = {ssim:.4f}")
    if difference_image:
        d = (pa - pb).astype(np.float32)
        res = a.copy(name=f"Difference {a.name} − {b.name}", data=d[None, None, None])
        res.rgb = False
        res.lut_is_binary = False
        res.overlay = []
        res.display.t = res.display.z = res.display.c = 0
        res.sync_channels()
        res.display.channels[0].min, res.display.channels[0].max = float(d.min()), float(d.max() or 1.0)
        out.new_images = [res]
    return out


# ====================================================================== Directionality
def orientation_histogram_tensor(plane: np.ndarray, sigma: float, n_bins: int, bin_start: float) -> np.ndarray:
    """Histogram of local structure orientation (degrees, bin_start … bin_start+180) weighted by the gradient energy."""
    from skimage.feature import structure_tensor
    f = plane.astype(np.float64)
    Arr, Arc, Acc = structure_tensor(f, sigma=sigma, mode="nearest", order="rc")
    # orientation of the dominant *structure* (perpendicular to the gradient), ImageJ convention: 0 = horizontal, counter-clockwise
    theta = 0.5 * np.degrees(np.arctan2(2 * Arc, Acc - Arr))  # gradient direction (y down)
    struct = -theta + 90.0  # structure direction, y up
    energy = Arr + Acc
    coherence = np.where(energy > 0, np.sqrt((Acc - Arr) ** 2 + 4 * Arc ** 2) / (energy + 1e-12), 0.0)
    w = energy * coherence
    ang = (struct - bin_start) % 180.0 + bin_start
    h, _ = np.histogram(ang, bins=n_bins, range=(bin_start, bin_start + 180.0), weights=w)
    return h


def orientation_histogram_fourier(plane: np.ndarray, n_bins: int, bin_start: float) -> np.ndarray:
    f = plane.astype(np.float64)
    f = f - f.mean()
    H, W = f.shape
    wy = np.hanning(H)[:, None] if H > 1 else 1.0
    wx = np.hanning(W)[None, :] if W > 1 else 1.0
    P = np.abs(np.fft.fftshift(np.fft.fft2(f * wy * wx))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    r = np.hypot(fx, fy)
    # the direction of a structure is perpendicular to its spatial frequency; y runs downwards in the image
    ang = (np.degrees(np.arctan2(-fy, fx)) + 90.0)
    ang = (ang - bin_start) % 180.0 + bin_start
    sel = (r > 0.02) & (r < 0.45)
    h, _ = np.histogram(ang[sel], bins=n_bins, range=(bin_start, bin_start + 180.0), weights=P[sel])
    return h


def fit_orientation_gaussian(centres: np.ndarray, hist: np.ndarray) -> dict[str, float]:
    """Fit a (periodic, 180°) Gaussian + offset to the orientation histogram: direction, dispersion, amount, goodness."""
    from scipy.optimize import curve_fit
    total = float(hist.sum())
    if total <= 0:
        return {"direction": float("nan"), "dispersion": float("nan"), "amount": 0.0, "goodness": 0.0}
    h = hist / total
    k = int(np.argmax(h))
    mu0 = float(centres[k])
    span = float(centres[-1] - centres[0]) + float(centres[1] - centres[0])

    def gauss(x, a, mu, sd, c):
        d = (x - mu + 90.0) % 180.0 - 90.0
        return a * np.exp(-d * d / (2 * sd * sd + 1e-12)) + c

    p0 = [float(h.max() - h.min()), mu0, span / 12.0, float(h.min())]
    try:
        params, _ = curve_fit(gauss, centres, h, p0=p0, bounds=([0, mu0 - 90, 0.5, 0], [np.inf, mu0 + 90, 180, 1]), maxfev=5000)
    except Exception:
        params = p0
    a, mu, sd, c = [float(p) for p in params]
    pred = gauss(centres, a, mu, sd, c)
    ss_res = float(np.sum((h - pred) ** 2))
    ss_tot = float(np.sum((h - h.mean()) ** 2))
    goodness = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    lo, hi = centres[0] - (centres[1] - centres[0]) / 2, centres[-1] + (centres[1] - centres[0]) / 2
    direction = (mu - lo) % 180.0 + lo
    amount = float(np.sum((h - c) * (np.abs((centres - direction + 90) % 180 - 90) <= 2 * sd)))
    return {"direction": float(direction), "dispersion": sd, "amount": float(np.clip(amount, 0, 1)), "goodness": float(goodness)}


@op("analyze.directionality", "Directionality", CAT, kind="analysis", preview=False, roi_aware=True, supports_rgb=False,
    params=[Param("method", "choice", "Local gradient orientation", choices=["Local gradient orientation", "Fourier components"]),
            Param("n_bins", "int", 90, min=4, max=720), Param("bin_start", "float", -90.0, min=-180, max=180, step=15, label="Histogram start (°)"),
            Param("sigma", "float", 2.0, min=0.5, max=20, step=0.5, help="Gaussian window of the structure tensor (local gradient method)"),
            Param("all_planes", "bool", False, label="Process every plane")],
    description="Orientation histogram of the structures in the image (structure tensor or Fourier power spectrum) with a Gaussian fit → plot + Results: Direction (°), Dispersion (°), Amount, Goodness (Fiji's Directionality).",
    keywords=["orientation", "fibres", "anisotropy", "angle histogram"])
def directionality(ctx, method="Local gradient orientation", n_bins=90, bin_start=-90.0, sigma=2.0, all_planes=False):
    img = ctx.img
    d = img.display
    planes = list(img.iter_planes()) if all_planes else [(d.t, d.z, d.c)]
    edges = np.linspace(bin_start, bin_start + 180.0, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    rows = []
    series = []
    mask = ctx.mask
    for (t, z, c) in planes:
        p = img.data[t, z, c].astype(np.float64)
        if mask is not None:
            bx, by, bw, bh = roimod.bounding_box(ctx.roi)
            y0, x0 = max(int(by), 0), max(int(bx), 0)
            p = p[y0:int(by + bh), x0:int(bx + bw)]
        if method.startswith("Fourier"):
            h = orientation_histogram_fourier(p, n_bins, bin_start)
        else:
            h = orientation_histogram_tensor(p, sigma, n_bins, bin_start)
        total = float(h.sum()) or 1.0
        hn = h / total
        fitres = fit_orientation_gaussian(centres, h)
        label = img.name if len(planes) == 1 else f"{img.name} c{c + 1} z{z + 1} t{t + 1}"
        rows.append({"Label": label, "Direction (°)": fitres["direction"], "Dispersion (°)": fitres["dispersion"], "Amount": fitres["amount"], "Goodness": fitres["goodness"]})
        series.append({"name": label, "values": hn.tolist()})
        if len(planes) == 1:
            dd = (centres - fitres["direction"] + 90) % 180 - 90
            sd = fitres["dispersion"] if np.isfinite(fitres["dispersion"]) else 1.0
            amp = float(hn.max() - hn.min())
            series.append({"name": "Gaussian fit", "values": (hn.min() + amp * np.exp(-dd * dd / (2 * sd * sd))).tolist()})
    plot = {"title": f"Directionality of {img.name}", "x": centres.tolist(), "series": series, "x_label": "Direction (°)", "y_label": "Amount"}
    table = {"name": "Directionality histograms", "columns": ["Direction (°)"] + [s["name"] for s in series], "rows": [[float(c)] + [s["values"][i] for s in series] for i, c in enumerate(centres)]}
    r0 = rows[0]
    return OpResult(results=rows, plot=plot, table=table, message=f"Direction {r0['Direction (°)']:.1f}°, dispersion {r0['Dispersion (°)']:.1f}°, amount {r0['Amount']:.2f}, goodness {r0['Goodness']:.2f}")
