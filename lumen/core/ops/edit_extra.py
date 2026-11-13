"""Edit menu gaps: Cut, Paste Control / paste modes, extra Selection commands (incl. Fiji's Fit Circle to
Image / Points from Mask), Image ▸ Drawing ▸ Linear / Radial Gradient, the Options (Preferences) dialogs
(incl. Image ▸ Overlay ▸ Overlay Options…), Image ▸ Type ▸ 8-bit Color, Lookup-table viewers/editor and the Help menu.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..image import LumenImage
from ..luts import LUT_NAMES, get_lut, gradient_lut, is_builtin, parse_color, parse_lut_text, register_lut
from ..registry import REGISTRY, OpResult, Param, cast_to, op
from ._util import current_threshold

CAT = "Edit"
SEL = "Edit/Selection"
OPT = "Edit/Options"
HELP = "Help"
LUTS = "Image/Lookup Tables"
DRAW = "Image/Drawing"
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)

# ImageJ's Paste Control transfer modes, in the order of its dialog (the frontend's Preferences list mirrors this).
PASTE_MODES = ["Copy", "Blend", "Average", "Difference", "Transparent-white", "Transparent-zero", "AND", "OR", "XOR", "Add", "Subtract", "Multiply", "Divide", "Min", "Max"]


def current_paste_mode(session) -> str:
    """The Paste Control / Preferences transfer mode, normalised to an entry of ``PASTE_MODES``.

    Lenient on purpose: the value is stored as a plain string (``PUT /api/settings``), so a stale or
    differently spelled value must not make ⌘V fail.  ImageJ aliases are resolved like ``IJ.stringToPasteMode``
    ("transparent" → Transparent-white unless it says "zero"); anything unknown falls back to Copy.
    """
    m = str(session.settings.get("paste_mode", "Copy") or "Copy").strip()
    low = {x.lower(): x for x in PASTE_MODES}
    if m.lower() in low:
        return low[m.lower()]
    if m.lower().startswith("tran"):
        return "Transparent-zero" if "zero" in m.lower() else "Transparent-white"
    return "Copy"


# ============================================================================ clipboard
def _copy_to_clipboard(ctx) -> tuple[int, int, int, int, np.ndarray | None]:
    """Same clipboard record as edit.copy: {"data": (C, H, W), "mask": (H, W) | None, "rgb": bool}."""
    img = ctx.img
    d = img.display
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        x, y, w, h = roimod.bounding_box(ctx.roi)
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + w)), img.width), min(int(round(y + h)), img.height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("The selection lies outside the image")
        mask = roimod.to_mask(ctx.roi, (img.height, img.width))[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, img.width, img.height
        mask = None
    ctx.session.settings["clipboard"] = {"data": img.data[d.t, d.z, :, y0:y1, x0:x1].copy(), "mask": mask, "rgb": img.rgb}
    return x0, y0, x1, y1, mask


@op("edit.cut", "Cut", CAT, kind="image", preview=False, requires_roi="area", roi_aware=False, scope_param=False, shortcut="Cmd+X",
    description="Copy the selection (all channels of the current plane) to the internal clipboard and clear it to the background value.")
def cut(ctx):
    img = ctx.img
    d = img.display
    x0, y0, x1, y1, mask = _copy_to_clipboard(ctx)
    bg = float(ctx.session.settings.get("background", 0))
    if img.data.dtype == np.uint16 and bg == 255:
        bg = 65535.0
    data = img.data.copy()
    for c in range(img.nc):
        target = data[d.t, d.z, c, y0:y1, x0:x1]
        val = cast_to(np.full(target.shape, bg, np.float32), data.dtype)
        if mask is not None:
            target[mask] = val[mask]
        else:
            target[...] = val
    return OpResult(data=data, rgb=img.rgb, message=f"Cut {x1 - x0}×{y1 - y0} px")


@op("edit.paste_control", "Paste Control…", CAT, **_NOIMG, keywords=["transfer mode", "blitter", "paste mode"],
    params=[Param("mode", "choice", "Copy", label="Transfer mode", choices=PASTE_MODES)],
    description="Set how Paste combines the clipboard with the image (ImageJ's Paste Control transfer modes).")
def paste_control(ctx, mode="Copy"):
    ctx.session.settings["paste_mode"] = mode
    return OpResult(message=f"Paste mode: {mode}")


def blend_paste(dst: np.ndarray, src: np.ndarray, mode: str) -> np.ndarray:
    """Combine a clipboard patch with the target pixels (same shape, same dtype) using an ImageJ transfer mode."""
    mode = str(mode or "Copy")
    if mode == "Copy":
        return src.copy()
    dtype = dst.dtype
    is_int = np.issubdtype(dtype, np.integer)
    d = dst.astype(np.float64)
    s = src.astype(np.float64)
    if mode in ("Blend", "Average"):
        out = (d + s) / 2.0
    elif mode == "Difference":
        out = np.abs(d - s)
    elif mode == "Transparent-zero":
        out = np.where(s == 0, d, s)
    elif mode == "Transparent-white":
        # ImageJ ByteBlitter: 255 is the transparent colour of 8-bit images; ShortBlitter / FloatBlitter have no
        # transparent colour and treat COPY_TRANSPARENT as a plain copy.  (RGB images test the whole pixel –
        # see paste_with_mode, which turns both transparent modes into a mask + Copy for them.)
        out = np.where(s == 255, d, s) if dtype == np.uint8 else s
    elif mode in ("AND", "OR", "XOR"):
        a, b = np.rint(d).astype(np.int64), np.rint(s).astype(np.int64)
        out = {"AND": a & b, "OR": a | b, "XOR": a ^ b}[mode].astype(np.float64)
    elif mode == "Add":
        out = d + s
    elif mode == "Subtract":
        out = d - s
    elif mode == "Multiply":
        out = d * s
    elif mode == "Divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(s == 0, d, d / np.where(s == 0, 1, s))
    elif mode == "Min":
        out = np.minimum(d, s)
    elif mode == "Max":
        out = np.maximum(d, s)
    else:
        raise ValueError(f"Unknown paste mode {mode!r}")
    if is_int:
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype)
    return out.astype(dtype)


@op("edit.paste_with_mode", "Paste (with transfer mode)", CAT, kind="image", preview=False, roi_aware=False, scope_param=False, keywords=["paste control", "blend", "transparent"],
    params=[Param("mode", "choice", "Current setting", label="Transfer mode", choices=["Current setting"] + PASTE_MODES)],
    description="Paste the clipboard into the current plane (at the selection, or centred) combining it with the pixels using the Paste Control transfer mode.")
def paste_with_mode(ctx, mode="Current setting"):
    img = ctx.img
    d = img.display
    clip = ctx.session.settings.get("clipboard")
    if clip is None or clip.get("data") is None:
        raise ValueError("Clipboard is empty – use Copy first")
    if mode == "Current setting":
        mode = current_paste_mode(ctx.session)
    if mode not in PASTE_MODES:
        raise ValueError(f"Unknown paste mode {mode!r}")
    src = np.asarray(clip["data"])
    if src.ndim == 2:
        src = src[None]
    nc, h, w = src.shape
    if ctx.roi is not None:
        bx, by, bw, bh = roimod.bounding_box(ctx.roi)
        x0, y0 = int(round(bx + bw / 2 - w / 2)), int(round(by + bh / 2 - h / 2))
    else:
        x0, y0 = (img.width - w) // 2, (img.height - h) // 2
    data = img.data.copy()
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    ww, hh = min(w - sx0, img.width - dx0), min(h - sy0, img.height - dy0)
    if ww <= 0 or hh <= 0:
        raise ValueError("Paste position is outside the image")
    m = clip.get("mask")
    m = None if m is None else np.asarray(m, bool)[sy0:sy0 + hh, sx0:sx0 + ww]
    blend_mode = mode
    if img.rgb and mode in ("Transparent-white", "Transparent-zero"):
        # ImageJ ColorBlitter tests the packed pixel ((src & 0xffffff) == 0xffffff / == 0): an RGB pixel is
        # transparent only when *all* channels are white / zero, not channel by channel.
        key = 255 if mode == "Transparent-white" else 0
        patch_all = cast_to(src[:, sy0:sy0 + hh, sx0:sx0 + ww], img.data.dtype)
        opaque = ~np.all(patch_all == key, axis=0)
        m = opaque if m is None else (m & opaque)
        blend_mode = "Copy"
    for c in range(img.nc):
        sc = min(c, nc - 1)
        patch = cast_to(src[sc, sy0:sy0 + hh, sx0:sx0 + ww], img.data.dtype)
        target = data[d.t, d.z, c, dy0:dy0 + hh, dx0:dx0 + ww]
        mixed = blend_paste(target, patch, blend_mode)
        if m is not None:
            target[m] = mixed[m]
        else:
            target[...] = mixed
    return OpResult(data=data, rgb=img.rgb, roi={"type": "rect", "x": dx0, "y": dy0, "w": ww, "h": hh}, message=f"Pasted ({mode})")


# ============================================================================ selection
def _roi_points(roi: dict[str, Any]) -> np.ndarray:
    t = roi["type"]
    if t == "point":
        return np.asarray(roi["points"], float)
    if roimod.is_line(roi):
        return np.asarray(roimod.line_points(roi), float)
    return np.asarray(roimod.polygon_points(roi), float)


def fit_circle_points(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kåsa) least-squares circle through points → (cx, cy, r)."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        raise ValueError("At least 3 points are required to fit a circle")
    cen = pts - pts.mean(axis=0)
    sv = np.linalg.svd(cen, compute_uv=False)
    if sv[-1] <= 1e-7 * max(sv[0], 1.0):
        raise ValueError("The points are collinear – cannot fit a circle")
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx ** 2 + cy ** 2 - sol[2]
    if not np.isfinite(r2) or r2 <= 0:
        raise ValueError("The points are collinear – cannot fit a circle")
    return float(cx), float(cy), float(math.sqrt(r2))


@op("selection.fit_circle", "Fit Circle", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    description="Replace the selection with the best-fit circle: an area selection becomes the circle with the same area and centroid, points / lines get a least-squares circle through them.")
def fit_circle(ctx):
    img = ctx.img
    roi = ctx.roi
    if roimod.is_area(roi):
        m = roimod.to_mask(roi, (img.height, img.width))
        n = int(m.sum())
        if n == 0:
            raise ValueError("Empty selection")
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
        r = math.sqrt(n / math.pi)
    else:
        pts = _roi_points(roi)
        if roi["type"] == "line" or (len(pts) == 2):
            (xa, ya), (xb, yb) = pts[0], pts[-1]  # the line is a diameter
            cx, cy, r = (xa + xb) / 2, (ya + yb) / 2, math.hypot(xb - xa, yb - ya) / 2
        else:
            cx, cy, r = fit_circle_points(pts)
    return OpResult(roi={"type": "oval", "x": cx - r, "y": cy - r, "w": 2 * r, "h": 2 * r, "name": roi.get("name") or "Circle"},
                    message=f"Circle: centre ({cx:.1f}, {cy:.1f}), radius {r:.2f} px")


def min_area_rect(pts: np.ndarray) -> np.ndarray:
    """Minimum-area enclosing rectangle (rotating calipers over the convex hull) → (4, 2) corners."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        raise ValueError("At least 3 distinct points are required")
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception as e:  # collinear points
        raise ValueError("The points are collinear – cannot fit a rectangle") from e
    # every hull edge at once: rotate the hull so the edge is horizontal and take the axis-aligned box
    q = np.roll(hull, -1, axis=0)
    ang = np.arctan2(q[:, 1] - hull[:, 1], q[:, 0] - hull[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)   # (n, 2, 2)
    rot = hull[None, :, :] @ R.transpose(0, 2, 1)                                     # (n, m, 2)
    lo = rot.min(axis=1)
    hi = rot.max(axis=1)
    areas = ((hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1])).tolist()
    best = 0
    for i in range(1, len(areas)):   # first edge whose box is smaller by more than 1e-9 than the best so far
        if areas[i] < areas[best] - 1e-9:
            best = i
    (x0, y0), (x1, y1) = lo[best], hi[best]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R[best]              # inverse rotation (R is orthogonal)


@op("selection.fit_rectangle", "Fit Rectangle", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    description="Replace the selection with its minimum-area rotated bounding rectangle (rotating calipers) as a 4-point polygon.")
def fit_rectangle(ctx):
    roi = ctx.roi
    pts = _roi_points(roi)
    if roi["type"] == "line":
        raise ValueError("A straight line has no area – use a polygon, polyline or points")
    corners = min_area_rect(pts)
    w = float(np.hypot(*(corners[1] - corners[0])))
    h = float(np.hypot(*(corners[2] - corners[1])))
    ang = math.degrees(math.atan2(-(corners[1][1] - corners[0][1]), corners[1][0] - corners[0][0]))
    return OpResult(roi={"type": "polygon", "points": corners.round(3).tolist(), "name": roi.get("name") or "Rectangle"},
                    message=f"Rectangle {w:.1f} × {h:.1f} px, angle {ang:.1f}°")


def spline_points(pts: np.ndarray, closed: bool, n: int) -> np.ndarray:
    """Cubic (where possible) interpolating B-spline through the vertices, evaluated at n points."""
    from scipy.interpolate import splev, splprep
    pts = np.asarray(pts, float)
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)  # drop repeated consecutive vertices
    pts = pts[keep]
    if closed and len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    m = len(pts)
    if m < 2:
        raise ValueError("Need at least 2 distinct points")
    if closed and m < 3:
        raise ValueError("A closed spline needs at least 3 points")
    k = min(3, m - 1)
    x, y = pts[:, 0], pts[:, 1]
    if closed:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    tck, _ = splprep([x, y], s=0, k=k, per=bool(closed))
    u = np.linspace(0, 1, n, endpoint=not closed)
    xs, ys = splev(u, tck)
    return np.stack([xs, ys], axis=1)


@op("selection.fit_spline", "Fit Spline", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    params=[Param("points", "int", 0, min=0, max=100000, help="Number of output vertices (0 = about one per 2 px of outline, at least 50)")],
    description="Replace a polygon / freehand selection with a closed cubic spline through its vertices (an open spline for segmented lines).")
def fit_spline(ctx, points=0):
    roi = ctx.roi
    t = roi["type"]
    if t in ("rect", "oval", "mask"):
        raise ValueError("Fit Spline needs a polygon, freehand, segmented-line or point selection")
    if t == "line":
        raise ValueError("A straight line is already a spline – use a segmented line")
    closed = t in ("polygon", "freehand")
    pts = _roi_points(roi)
    if points <= 0:
        seg = np.vstack([pts, pts[:1]]) if closed else pts
        length = float(np.sum(np.hypot(*np.diff(seg, axis=0).T))) if len(seg) > 1 else 0.0
        points = int(max(50, min(5000, length / 2)))
    out = spline_points(pts, closed, points)
    if closed:
        new = {"type": "polygon", "points": out.round(3).tolist(), "name": roi.get("name") or "Spline"}
    else:
        new = {"type": "polyline", "points": out.round(3).tolist(), "width": roi.get("width", 1), "name": roi.get("name") or "Spline"}
    return OpResult(roi=new, message=f"Spline with {len(out)} vertices")


@op("selection.image_to_selection", "Image to Selection…", SEL, kind="analysis", preview=False, roi_aware=False, keywords=["mask to selection", "overlay image"],
    params=[Param("image", "image", None, help="An open image (e.g. a mask); its non-zero pixels become the selection"),
            Param("opacity", "int", 75, min=0, max=100, unit="%", help="Display opacity hint for the frontend (the selection itself is a mask)"),
            Param("all_channels", "bool", False, help="Use pixels that are non-zero in any channel (default: the current channel of that image)")],
    description="Make a selection on the current image from the non-zero pixels of another open image (ImageJ's Edit ▸ Selection ▸ Image to Selection).")
def image_to_selection(ctx, image=None, opacity=75, all_channels=False):
    img = ctx.img
    if image in (None, ""):
        raise ValueError("Choose an image")
    src = ctx.get_image(str(image))
    d = src.display
    if all_channels:
        m = np.any(src.data[d.t, d.z] != 0, axis=0)
    else:
        m = src.data[d.t, d.z, d.c] != 0
    H, W = img.height, img.width
    if m.shape != (H, W):
        full = np.zeros((H, W), bool)
        hh, ww = min(H, m.shape[0]), min(W, m.shape[1])
        full[:hh, :ww] = m[:hh, :ww]
        m = full
    r = roimod.from_mask(m, src.name)
    if r is None:
        raise ValueError(f"{src.name} has no non-zero pixels on its current plane")
    r["opacity"] = int(opacity)
    return OpResult(roi=r, message=f"Selection from {src.name}: {int(m.sum())} px (opacity {int(opacity)}%)")


def _border_mode(plane: np.ndarray) -> float:
    border = np.concatenate([plane[0, :], plane[-1, :], plane[:, 0], plane[:, -1]]).ravel()
    if np.issubdtype(border.dtype, np.integer):
        vals, counts = np.unique(border, return_counts=True)
    else:
        vals, counts = np.unique(np.round(border, 3), return_counts=True)
    return float(vals[int(np.argmax(counts))])


@op("selection.select_bounding_box", "Select Bounding Box (guess background)", SEL, kind="analysis", preview=False, roi_aware=False,
    keywords=["autocrop", "trim", "background colour"],
    params=[Param("tolerance", "float", 0.0, min=0, step=1, help="Pixels within ± tolerance of the background value count as background"),
            Param("all_channels", "bool", True, help="A pixel is foreground when any channel differs from the background")],
    description="Select the bounding rectangle of everything that differs from the background, which is guessed as the most frequent value along the image border.")
def select_bounding_box(ctx, tolerance=0.0, all_channels=True):
    img = ctx.img
    d = img.display
    chans = range(img.nc) if (all_channels or img.rgb) else [d.c]
    fg = np.zeros((img.height, img.width), bool)
    bgs = []
    for c in chans:
        p = img.data[d.t, d.z, c]
        bg = _border_mode(p)
        bgs.append(bg)
        fg |= np.abs(p.astype(np.float64) - bg) > float(tolerance)
    ys, xs = np.nonzero(fg)
    if ys.size == 0:
        raise ValueError("The whole image is background")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bgtxt = ", ".join(f"{b:g}" for b in bgs)
    return OpResult(roi={"type": "rect", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "name": "Bounding box"}, message=f"Background {bgtxt}; box {x1 - x0}×{y1 - y0} at ({x0}, {y0})")


def foreground_mask(ctx, use_selection: bool = True) -> tuple[np.ndarray, str]:
    """(H, W) boolean foreground of the current plane plus a short note on where it came from.

    A thresholded image contributes the pixels within its threshold, a binary (or any other) image its non-zero
    pixels (RGB: non-zero in any channel); an area selection restricts the result to the selection.  Raises when
    nothing is foreground.
    """
    img = ctx.img
    d = img.display
    thr = current_threshold(ctx)
    if thr is not None and not img.rgb:
        p = img.current_plane()
        m = (p >= thr[0]) & (p <= thr[1])
        how = f"threshold {thr[0]:g}–{thr[1]:g}"
    elif img.rgb:
        m = np.any(img.data[d.t, d.z] != 0, axis=0)
        how = "non-zero pixels (any channel)"
    else:
        p = img.current_plane()
        m = p != 0
        if p.dtype.kind == "f":
            m &= np.isfinite(p)
        how = "non-zero pixels"
    if use_selection and ctx.mask is not None:
        m = m & ctx.mask
        how += " inside the selection"
    if not m.any():
        raise ValueError("No foreground pixels – threshold the image or use a binary mask")
    return m, how


CIRCLE_FIT_METHODS = ["Outline (least squares)", "All pixels (least squares)", "Area and centroid"]


def fit_circle_to_mask(mask: np.ndarray, method: str = "Outline (least squares)") -> tuple[float, float, float]:
    """Fit a circle to a boolean mask → (cx, cy, r) in pixel units (pixel centres at +0.5)."""
    if method == "Area and centroid":
        n = int(mask.sum())
        ys, xs = np.nonzero(mask)
        return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, math.sqrt(n / math.pi)
    if method.startswith("Outline"):
        edge = mask & ~ndi.binary_erosion(mask, border_value=1)  # image-border pixels are not an outline
        if not edge.any():
            edge = mask
        ys, xs = np.nonzero(edge)
    else:
        ys, xs = np.nonzero(mask)
    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)
    if len(pts) > 200_000:  # the normal equations only need a representative subset
        pts = pts[np.linspace(0, len(pts) - 1, 200_000).astype(int)]
    return fit_circle_points(pts)


@op("selection.fit_circle_to_image", "Fit Circle to Image", SEL, kind="analysis", preview=False, roi_aware=True, keywords=["fit circle", "threshold", "mask", "dish", "well", "fiji"],
    params=[Param("method", "choice", CIRCLE_FIT_METHODS[0], choices=CIRCLE_FIT_METHODS,
                  help="Outline: least-squares circle through the boundary pixels of the foreground; All pixels: least squares over every foreground pixel "
                       "(for ring-like masks); Area and centroid: the circle with the foreground's area and centroid")],
    description="Fit a circle to the thresholded / binary foreground of the current plane (restricted to the selection when there is one) and make it the "
                "selection – Fiji's Edit ▸ Selection ▸ Fit Circle to Image.")
def fit_circle_to_image(ctx, method=CIRCLE_FIT_METHODS[0]):
    img = ctx.img
    mask, how = foreground_mask(ctx)
    cx, cy, r = fit_circle_to_mask(mask, method)
    cal = img.calibration
    unit = f"; radius {r * cal.pixel_width:.3g} {cal.unit}" if cal.scaled else ""
    return OpResult(roi={"type": "oval", "x": cx - r, "y": cy - r, "w": 2 * r, "h": 2 * r, "name": "Fitted circle"},
                    message=f"Circle from {how} ({int(mask.sum())} px): centre ({cx:.1f}, {cy:.1f}), radius {r:.2f} px{unit}")


@op("selection.points_from_mask", "Points from Mask", SEL, kind="analysis", preview=False, roi_aware=True, keywords=["multi-point", "mask to points", "binary", "fiji"],
    params=[Param("step", "int", 1, min=1, max=1000, help="Subsampling: keep one pixel every N rows and columns"),
            Param("max_points", "int", 100000, min=0, max=10_000_000, help="Evenly thin the points down to at most this many (0 = unlimited)"),
            Param("output", "choice", "Selection", choices=["Selection", "ROI Manager"])],
    description="Make a multi-point selection with one point per foreground pixel of the thresholded / binary current plane (inside the selection when there is one), "
                "optionally subsampled – Fiji's Edit ▸ Selection ▸ Points from Mask.")
def points_from_mask(ctx, step=1, max_points=100000, output="Selection"):
    mask, how = foreground_mask(ctx)
    step = max(int(step), 1)
    if step > 1:
        keep = np.zeros_like(mask)
        keep[::step, ::step] = True
        mask = mask & keep
    ys, xs = np.nonzero(mask)
    n_all = int(ys.size)
    if n_all == 0:
        raise ValueError("No foreground pixels left after subsampling")
    thinned = False
    if max_points and n_all > int(max_points):
        idx = np.linspace(0, n_all - 1, int(max_points)).astype(int)
        ys, xs = ys[idx], xs[idx]
        thinned = True
    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)
    roi = {"type": "point", "points": pts.tolist(), "name": "Points from mask"}
    note = f"{len(pts)} points from {how}" + (f" (every {step}. pixel)" if step > 1 else "") + (f", thinned from {n_all}" if thinned else "")
    if output == "ROI Manager":
        return OpResult(rois_add=[roi], message=note + " – added to the ROI Manager")
    return OpResult(roi=roi, message=note)


@op("selection.properties", "Properties…", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False, keywords=["roi name", "roi colour", "stroke", "fill"],
    params=[Param("name", "string", "", help="Leave empty to keep"), Param("color", "string", "", label="Stroke colour", help="#rrggbb, r,g,b or a colour name; empty = keep"),
            Param("stroke_width", "float", 0, min=0, max=100, help="0 = keep"), Param("fill", "string", "", label="Fill colour", help="#rrggbb or 'none'; empty = keep"),
            Param("position", "string", "", help="Slice position as 't,z,c' (1-based), 'none' to clear, empty = keep", advanced=True)],
    description="Set the name, stroke colour, line width, fill colour and position of the active selection.")
def properties(ctx, name="", color="", stroke_width=0.0, fill="", position=""):
    roi = dict(ctx.roi)
    if str(name).strip():
        roi["name"] = str(name).strip()
    if str(color).strip():
        r, g, b = parse_color(color)
        roi["color"] = f"#{r:02x}{g:02x}{b:02x}"
    if stroke_width and stroke_width > 0:
        roi["stroke_width"] = float(stroke_width)
    f = str(fill).strip()
    if f:
        # 'fill' is the key the viewport / overlay renderer read; 'fill_color' is the ImageJ .roi reader/writer's
        # key (lumen.core.io), kept in sync so a fill set here survives Save As .roi and an imported fill is editable.
        if f.lower() in ("none", "no", "off", "-"):
            roi.pop("fill", None)
            roi.pop("fill_color", None)
        else:
            r, g, b = parse_color(f)
            roi["fill"] = roi["fill_color"] = f"#{r:02x}{g:02x}{b:02x}"
    elif roi.get("fill_color") and not roi.get("fill"):
        roi["fill"] = roi["fill_color"]  # an ImageJ .roi fill colour becomes visible to the renderers
    pos = str(position).strip()
    if pos:
        if pos.lower() in ("none", "0", "-"):
            roi.pop("position", None)
        else:
            parts = [int(float(v)) for v in pos.replace(";", ",").split(",") if v.strip()]
            if len(parts) != 3:
                raise ValueError("position must be 't,z,c'")
            roi["position"] = {"t": max(parts[0] - 1, 0), "z": max(parts[1] - 1, 0), "c": max(parts[2] - 1, 0)}
    return OpResult(roi=roi, message=f"{roi.get('name') or roi['type']}: colour {roi.get('color', 'default')}, width {roi.get('stroke_width', 'default')}, fill {roi.get('fill', 'none')}")


def rounded_rect_points(x: float, y: float, w: float, h: float, r: float, arc_points: int = 8) -> np.ndarray:
    r = max(0.0, min(float(r), w / 2, h / 2))
    if r <= 0:
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    centres = np.array([(x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)], float)
    k = np.arange(arc_points + 1)
    a = np.radians(centres[:, 2:3] + 90 * k[None, :] / arc_points)              # (4, arc_points + 1)
    pts = np.stack([centres[:, 0:1] + r * np.cos(a), centres[:, 1:2] + r * np.sin(a)], axis=-1)
    return pts.reshape(-1, 2)


@op("selection.rounded_rect", "Rounded Rectangle…", SEL, kind="analysis", preview=False, roi_aware=False, keywords=["rounded rectangle tool", "corner radius"],
    params=[Param("x", "float", 0), Param("y", "float", 0), Param("w", "float", 100, min=1, label="Width"), Param("h", "float", 100, min=1, label="Height"),
            Param("corner_radius", "float", 20, min=0, help="Default: the Rounded Rectangle Tool setting"), Param("arc_points", "int", 8, min=1, max=64, advanced=True, help="Vertices per corner arc")],
    description="Create a rounded-rectangle selection (polygon approximation).")
def rounded_rect(ctx, x=0.0, y=0.0, w=100.0, h=100.0, corner_radius=20.0, arc_points=8):
    if corner_radius < 0:
        corner_radius = float(ctx.session.settings.get("rounded_rect_radius", 20))
    pts = rounded_rect_points(x, y, w, h, corner_radius, arc_points)
    return OpResult(roi={"type": "polygon", "points": pts.round(3).tolist(), "name": "Rounded rectangle", "corner_radius": float(corner_radius)})


# ============================================================================ Image ▸ Drawing ▸ gradients
GRADIENT_SCOPES = ["Current plane", "Current slice (all channels)", "Whole stack"]


def _gradient_planes(ctx, apply_to: str) -> list[tuple[int, int, int]]:
    img = ctx.img
    d = img.display
    if apply_to == "Whole stack":
        return list(img.iter_planes())
    if apply_to.startswith("Current slice") or img.rgb:
        return [(d.t, d.z, c) for c in range(img.nc)]
    return [(d.t, d.z, d.c)]


def gradient_colours(ctx, fg_value: float = -1.0, bg_value: float = -1.0, fg_color: str = "", bg_color: str = "") -> tuple[list[float], list[float]]:
    """Per-channel (foreground, background) values: one entry for grey images, three for RGB (from the colour strings,
    or the grey values repeated).  -1 / empty = the Edit ▸ Options ▸ Colours setting (255 → 65535 on 16-bit images)."""
    img = ctx.img

    def setting(which: str, override: float) -> float:
        if override is not None and float(override) >= 0:
            return float(override)
        v = float(ctx.session.settings.get(which, 255 if which == "foreground" else 0))
        if img.data.dtype == np.uint16 and v == 255:
            v = 65535.0
        return v

    fg, bg = setting("foreground", fg_value), setting("background", bg_value)
    if img.rgb:
        f = [float(c) for c in parse_color(fg_color)] if str(fg_color or "").strip() else [min(max(fg, 0.0), 255.0)] * 3
        b = [float(c) for c in parse_color(bg_color)] if str(bg_color or "").strip() else [min(max(bg, 0.0), 255.0)] * 3
        return f, b
    return [fg], [bg]


def _gradient_box(ctx) -> tuple[float, float, float, float]:
    """(x, y, w, h) the gradient spans: the area selection's bounding box, else the whole plane."""
    img = ctx.img
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        x, y, w, h = roimod.bounding_box(ctx.roi)
        return float(x), float(y), max(float(w), 1.0), max(float(h), 1.0)
    return 0.0, 0.0, float(img.width), float(img.height)


def linear_gradient_weights(shape: tuple[int, int], box: tuple[float, float, float, float], angle: float) -> np.ndarray:
    """(H, W) float32 in 0 (foreground) … 1 (background) along `angle` (degrees, counter-clockwise; 0 = left → right,
    90 = bottom → top) spanning the box's pixel centres exactly: the first column / row of the box is pure foreground,
    the last pure background (like New Image's Ramp fill)."""
    H, W = shape
    x, y, w, h = box
    th = math.radians(float(angle))
    dx, dy = math.cos(th), -math.sin(th)  # image y points down
    xs = (np.arange(W, dtype=np.float32) + 0.5) * dx
    ys = (np.arange(H, dtype=np.float32) + 0.5) * dy
    proj = ys[:, None] + xs[None, :]
    ix, iy = (0.5 if w > 1 else 0.0), (0.5 if h > 1 else 0.0)  # pixel centres of the outermost columns / rows
    corners = [(x + ix, y + iy), (x + w - ix, y + iy), (x + ix, y + h - iy), (x + w - ix, y + h - iy)]
    pc = [cx * dx + cy * dy for cx, cy in corners]
    lo, hi = min(pc), max(pc)
    return np.clip((proj - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


def line_gradient_weights(shape: tuple[int, int], a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """(H, W) weights 0 at point a, 1 at point b, constant beyond the ends (gradient along a line selection)."""
    H, W = shape
    (xa, ya), (xb, yb) = a, b
    vx, vy = xb - xa, yb - ya
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        raise ValueError("The line selection has no length")
    xs = (np.arange(W, dtype=np.float32) + 0.5 - xa) * (vx / L2)
    ys = (np.arange(H, dtype=np.float32) + 0.5 - ya) * (vy / L2)
    return np.clip(ys[:, None] + xs[None, :], 0.0, 1.0).astype(np.float32)


def radial_gradient_weights(shape: tuple[int, int], cx: float, cy: float, radius: float) -> np.ndarray:
    """(H, W) weights 0 at the centre, 1 at `radius` and beyond."""
    H, W = shape
    xs = (np.arange(W, dtype=np.float32) + 0.5 - cx) ** 2
    ys = (np.arange(H, dtype=np.float32) + 0.5 - cy) ** 2
    r = np.sqrt(ys[:, None] + xs[None, :])
    return np.clip(r / max(float(radius), 1e-9), 0.0, 1.0).astype(np.float32)


def paint_gradient(ctx, weights: np.ndarray, apply_to: str, fg: list[float], bg: list[float], reverse: bool = False, mask: np.ndarray | None = None) -> np.ndarray:
    """Blend foreground → background by `weights` into the selected planes (inside `mask` when given); returns the new 5-D array."""
    img = ctx.img
    if reverse:
        weights = 1.0 - weights
    data = img.data.copy()
    planes = _gradient_planes(ctx, apply_to)
    n = len(planes)
    for i, (t, z, c) in enumerate(planes):
        ctx.check_cancel()
        k = c if len(fg) > 1 else 0
        vals = cast_to(fg[k] + (bg[k] - fg[k]) * weights, data.dtype)
        if mask is not None:
            data[t, z, c][mask] = vals[mask]
        else:
            data[t, z, c] = vals
        if n > 1:
            ctx.progress((i + 1) / n, f"Plane {i + 1}/{n}")
    return data


_GRADIENT_COMMON = [Param("apply_to", "choice", "Current plane", choices=GRADIENT_SCOPES, label="Apply to", help="RGB images are always painted on all three channels"),
                    Param("reverse", "bool", False, help="Background → foreground instead"),
                    Param("fg_value", "float", -1, label="Foreground value", help="-1 = the foreground colour setting", advanced=True),
                    Param("bg_value", "float", -1, label="Background value", help="-1 = the background colour setting", advanced=True),
                    Param("fg_color", "string", "", label="Foreground colour (RGB)", help="#rrggbb or a colour name for RGB images; empty = the foreground value as grey", advanced=True),
                    Param("bg_color", "string", "", label="Background colour (RGB)", help="empty = the background value as grey", advanced=True)]


@op("edit.linear_gradient", "Linear Gradient", DRAW, kind="image", preview=True, scope_param=False, keywords=["gradient", "ramp", "fill", "drawing", "fiji"],
    params=[Param("angle", "float", 0.0, min=-360, max=360, step=15, unit="°", help="0 = foreground at the left, background at the right; 90 = foreground at the bottom (counter-clockwise). "
                                                                                       "Ignored for a line selection, which sets the direction itself")] + _GRADIENT_COMMON,
    description="Fill the selection (or the plane) with a linear gradient from the foreground to the background value. A line selection draws the gradient along the "
                "line across the whole plane (foreground at its first point, background at its last); an area selection spans its bounding box – Fiji's Image ▸ Drawing ▸ Linear Gradient.")
def linear_gradient(ctx, angle=0.0, apply_to="Current plane", reverse=False, fg_value=-1.0, bg_value=-1.0, fg_color="", bg_color=""):
    img = ctx.img
    shape = (img.height, img.width)
    roi = ctx.roi
    if roi is not None and roimod.is_line(roi):
        pts = roimod.line_points(roi)
        weights = line_gradient_weights(shape, pts[0], pts[-1])
        mask = None
        how = f"along the line ({pts[0][0]:.0f}, {pts[0][1]:.0f}) → ({pts[-1][0]:.0f}, {pts[-1][1]:.0f})"
    else:
        weights = linear_gradient_weights(shape, _gradient_box(ctx), angle)
        mask = ctx.mask
        how = f"at {float(angle):g}°" + (" inside the selection" if mask is not None else "")
    fg, bg = gradient_colours(ctx, fg_value, bg_value, fg_color, bg_color)
    data = paint_gradient(ctx, weights, apply_to, fg, bg, reverse, mask)
    return OpResult(data=data, rgb=img.rgb, message=f"Linear gradient {how}")


@op("edit.radial_gradient", "Radial Gradient", DRAW, kind="image", preview=True, scope_param=False, keywords=["gradient", "radial", "vignette", "fill", "drawing", "fiji"],
    params=[Param("center_x", "float", -1, step=1, help="-1 = centre of the selection (or of the plane)"), Param("center_y", "float", -1, step=1, help="-1 = centre of the selection (or of the plane)"),
            Param("radius", "float", 0, min=0, step=1, unit="px", help="Distance at which the background value is reached; 0 = half of the longer side of the selection's bounding box (an oval's radius) or of the plane")] + _GRADIENT_COMMON,
    description="Fill the selection (or the plane) with a radial gradient: the foreground value at the centre fading to the background value at the radius – Fiji's Image ▸ Drawing ▸ Radial Gradient.")
def radial_gradient(ctx, center_x=-1.0, center_y=-1.0, radius=0.0, apply_to="Current plane", reverse=False, fg_value=-1.0, bg_value=-1.0, fg_color="", bg_color=""):
    img = ctx.img
    x, y, w, h = _gradient_box(ctx)
    cx = float(center_x) if center_x is not None and center_x >= 0 else x + w / 2.0
    cy = float(center_y) if center_y is not None and center_y >= 0 else y + h / 2.0
    r = float(radius) if radius and radius > 0 else max(w, h) / 2.0
    weights = radial_gradient_weights((img.height, img.width), cx, cy, r)
    fg, bg = gradient_colours(ctx, fg_value, bg_value, fg_color, bg_color)
    data = paint_gradient(ctx, weights, apply_to, fg, bg, reverse, ctx.mask)
    return OpResult(data=data, rgb=img.rgb, message=f"Radial gradient from ({cx:.1f}, {cy:.1f}), radius {r:.1f} px" + (" inside the selection" if ctx.mask is not None else ""))


# ============================================================================ Options / Preferences
# The keys the frontend's Preferences sheet reads and writes (session.settings).
#
# Every preference Param is declared with ``default=None`` on purpose: ``run_op`` fills omitted parameters with
# the Param default before calling the op, so a real default would make ``edit.options_misc(divide_by_zero="NaN")``
# silently reset paste_mode (and every other sibling) to its default.  With None, an omitted key reaches
# ``apply_prefs`` as None and is skipped – only the keys actually passed are written.  The real defaults live in
# ``PREF_DEFAULTS`` (used by ``edit.options_reset`` and to seed keys that were never set).
PREF_DEFAULTS: dict[str, Any] = {}


def _pref(name: str, type: str, default: Any, **kw: Any) -> Param:
    PREF_DEFAULTS[name] = default
    hint = f"Default {default!r}; leave unset to keep the current value"
    kw["help"] = f"{kw['help']}. {hint}" if kw.get("help") else hint
    return Param(name, type, None, **kw)


PREFS: dict[str, Param] = {
    "roi_color": _pref("roi_color", "color", "#ffd54a", label="Selection colour"),
    "roi_stroke_width": _pref("roi_stroke_width", "float", 1.5, min=0.5, max=20, step=0.5, label="Selection line width"),
    "font_size": _pref("font_size", "int", 14, min=4, max=200, label="Font size"),
    "font_family": _pref("font_family", "choice", "sans-serif", label="Font", choices=["sans-serif", "serif", "monospace", "SansSerif", "Serif", "Monospaced"]),
    "antialiased_text": _pref("antialiased_text", "bool", True, label="Antialiased text"),
    "foreground": _pref("foreground", "float", 255, label="Foreground value"),
    "background": _pref("background", "float", 0, label="Background value"),
    "jpeg_quality": _pref("jpeg_quality", "int", 85, min=1, max=100, label="JPEG quality"),
    "default_save_format": _pref("default_save_format", "choice", "tif", label="Default save format", choices=["tif", "ome.tif", "png", "jpg", "npy", "zip"]),
    "point_size": _pref("point_size", "choice", "4", label="Point size", choices=["1", "2", "3", "4", "6", "8", "12", "16"]),
    "point_color": _pref("point_color", "color", "#ffd54a", label="Point colour"),
    "point_type": _pref("point_type", "choice", "Hybrid", label="Point type", choices=["Hybrid", "Cross", "Dot", "Circle"]),
    "arrow_style": _pref("arrow_style", "choice", "Filled", label="Arrow style", choices=["Filled", "Notched", "Open", "Headless", "Bar"]),
    "arrow_head_size": _pref("arrow_head_size", "float", 12, min=1, max=100, label="Arrow head size"),
    "arrow_width": _pref("arrow_width", "float", 2, min=0.5, max=50, step=0.5, label="Arrow width"),
    "arrow_double": _pref("arrow_double", "bool", False, label="Double-headed arrow"),
    "plot_y_min": _pref("plot_y_min", "float", 0, label="Plot Y min", help="Fixed y range when min < max; equal values = auto"),
    "plot_y_max": _pref("plot_y_max", "float", 0, label="Plot Y max"),
    "plot_interpolate": _pref("plot_interpolate", "bool", True, label="Interpolate line profiles"),
    "plot_width": _pref("plot_width", "int", 600, min=100, max=4000, label="Plot width"),
    "plot_height": _pref("plot_height", "int", 340, min=60, max=4000, label="Plot height"),
    "interpolate_zoom": _pref("interpolate_zoom", "bool", False, label="Interpolate zoomed images"),
    "rounded_rect_radius": _pref("rounded_rect_radius", "float", 20, min=0, max=500, label="Rounded rectangle corner radius"),
    "rounded_rect_stroke_width": _pref("rounded_rect_stroke_width", "float", 2, min=0.5, max=50, label="Rounded rectangle stroke width"),
    "paste_mode": _pref("paste_mode", "choice", "Copy", label="Paste transfer mode", choices=PASTE_MODES),
    "divide_by_zero": _pref("divide_by_zero", "choice", "infinity", label="Divide by zero value", choices=["infinity", "max", "NaN", "0"]),
    "require_control_key": _pref("require_control_key", "bool", False, label="Require Ctrl/Cmd for shortcuts"),
    "beep_on_error": _pref("beep_on_error", "bool", False, label="Beep when a command fails"),
    "auto_contrast_stacks": _pref("auto_contrast_stacks", "bool", True, label="Auto-contrast new stacks"),
    # Image ▸ Overlay ▸ Overlay Options… (ImageJ's OverlayLabels / Overlay Options dialog); read by the viewport and
    # the overlay renderer (default stroke / fill of items without their own, label drawing) and by analyze.label
    "overlay_stroke_width": _pref("overlay_stroke_width", "float", 1.5, min=0.5, max=20, step=0.5, label="Overlay stroke width"),
    "overlay_default_fill": _pref("overlay_default_fill", "string", "", label="Overlay default fill colour", help="#rrggbb / a colour name, or 'none' for no fill"),
    "overlay_show_labels": _pref("overlay_show_labels", "bool", False, label="Show overlay labels"),
    "overlay_use_names": _pref("overlay_use_names", "bool", False, label="Use names as labels"),
    "overlay_label_font_size": _pref("overlay_label_font_size", "int", 12, min=4, max=200, label="Overlay label font size"),
    "overlay_label_color": _pref("overlay_label_color", "color", "#ffff00", label="Overlay label colour"),
    "overlay_label_background": _pref("overlay_label_background", "bool", False, label="Draw overlay label background"),
}
DEFAULT_SETTINGS: dict[str, Any] = dict(PREF_DEFAULTS)
_OVERLAY_KEYS = ["overlay_stroke_width", "overlay_default_fill", "overlay_show_labels", "overlay_use_names", "overlay_label_font_size", "overlay_label_color", "overlay_label_background"]

_IO_KEYS = ["jpeg_quality", "default_save_format"]
_POINT_KEYS = ["point_size", "point_color", "point_type"]
_ARROW_KEYS = ["arrow_style", "arrow_head_size", "arrow_width", "arrow_double"]
_FONT_KEYS = ["font_family", "font_size", "antialiased_text"]
_APPEARANCE_KEYS = ["interpolate_zoom", "antialiased_text", "auto_contrast_stacks"]
_PLOT_KEYS = ["plot_width", "plot_height", "plot_y_min", "plot_y_max", "plot_interpolate"]
_ROI_KEYS = ["roi_color", "roi_stroke_width"]
_ROUNDED_KEYS = ["rounded_rect_radius", "rounded_rect_stroke_width"]
_MISC_KEYS = ["divide_by_zero", "require_control_key", "beep_on_error", "paste_mode"]
_COLOR_KEYS = ["foreground", "background"]


def _coerce_pref(key: str, value: Any) -> Any:
    p = PREFS[key]
    v = p.coerce(value)
    if p.type == "color":
        r, g, b = parse_color(v)
        v = f"#{r:02x}{g:02x}{b:02x}"
    if key == "point_size":
        v = int(v)
    if key == "overlay_default_fill":  # a colour, or 'none' → "" (no default fill)
        v = str(v).strip()
        if v.lower() in ("none", "no", "off", "-"):
            v = ""
        elif v:
            r, g, b = parse_color(v)
            v = f"#{r:02x}{g:02x}{b:02x}"
    return v


def apply_prefs(session, values: dict[str, Any]) -> dict[str, Any]:
    """Write the given preferences into ``session.settings``; None / empty values mean "keep" and are skipped."""
    out = {}
    for k, v in values.items():
        if k in PREFS and v is not None and v != "":
            out[k] = _coerce_pref(k, v)
    session.settings.update(out)
    return out


def _options_op(op_id: str, name: str, keys: list[str], description: str, keywords: list[str] | None = None, menu: str | None = None):
    params = [PREFS[k] for k in keys]

    def run(ctx, **values):
        applied = apply_prefs(ctx.session, values)
        for k in keys:  # keys the dialog owns always exist afterwards, without touching values set earlier
            ctx.session.settings.setdefault(k, DEFAULT_SETTINGS[k])
        title = name.rstrip("…")
        if not applied:
            return OpResult(message=f"{title}: nothing changed")
        return OpResult(message=f"{title}: " + ", ".join(f"{k}={v!r}" for k, v in applied.items()))

    run.__name__ = op_id.replace(".", "_")
    run.__doc__ = description
    return op(op_id, name, OPT, **_NOIMG, params=params, description=description, keywords=keywords or ["preferences", "options", "settings"], menu=menu)(run)


options = _options_op("edit.options", "Preferences…", list(PREFS), "All Lumen preferences in one dialog (selection colour, fonts, I/O, point and arrow tools, plots, appearance, misc).")
options_io = _options_op("edit.options_io", "Input/Output…", _IO_KEYS, "JPEG quality and the default save format (Edit ▸ Options ▸ Input/Output).", ["jpeg", "save format"])
options_point_tool = _options_op("edit.options_point_tool", "Point Tool…", _POINT_KEYS, "Point tool marker size, colour and type (Edit ▸ Options ▸ Point Tool).", ["multi-point", "marker"])
options_arrow_tool = _options_op("edit.options_arrow_tool", "Arrow Tool…", _ARROW_KEYS, "Arrow tool style, head size and width (Edit ▸ Options ▸ Arrow Tool).", ["arrow"])
options_fonts = _options_op("edit.options_fonts", "Fonts…", _FONT_KEYS, "Font family and size used by text tools and labels (Edit ▸ Options ▸ Fonts).", ["text", "font"])
options_appearance = _options_op("edit.options_appearance", "Appearance…", _APPEARANCE_KEYS, "Interpolate zoomed images, antialiasing and stack auto-contrast (Edit ▸ Options ▸ Appearance).", ["zoom", "interpolation"])
options_plots = _options_op("edit.options_plots", "Plots…", _PLOT_KEYS, "Plot window size, fixed Y range and profile interpolation (Edit ▸ Options ▸ Plots).", ["profile", "plot range"])
options_roi_defaults = _options_op("edit.options_roi_defaults", "ROI Defaults…", _ROI_KEYS, "Default selection colour and line width (Edit ▸ Options ▸ ROI Defaults).", ["roi colour", "selection colour"])
options_rounded_rect = _options_op("edit.options_rounded_rect", "Rounded Rect Tool…", _ROUNDED_KEYS, "Corner radius and stroke width of the rounded-rectangle tool (Edit ▸ Options ▸ Rounded Rect Tool).", ["rounded rectangle"])
options_misc = _options_op("edit.options_misc", "Misc…", _MISC_KEYS, "Divide-by-zero value, shortcut modifier, beep and the paste transfer mode (Edit ▸ Options ▸ Misc).", ["misc", "divide by zero"])
options_overlay = _options_op("edit.options_overlay", "Overlay Options…", _OVERLAY_KEYS,
                              "Default stroke width and fill colour of overlay items, and how overlay labels are drawn: show labels, use names, font size, colour, background "
                              "(ImageJ's Image ▸ Overlay ▸ Overlay Options…). Stored as overlay_* settings; only the values passed are changed.",
                              ["overlay", "labels", "fill", "stroke", "overlay options"])  # stays in Edit/Options: the UI seeds + reloads prefs for that menu


@op("edit.options_reset", "Reset Preferences", OPT, **_NOIMG, keywords=["defaults", "reset settings"], description="Restore every preference to its default value.")
def options_reset(ctx):
    ctx.session.settings.update(DEFAULT_SETTINGS)
    return OpResult(message="Preferences reset to defaults")


# ============================================================================ Image ▸ Type ▸ 8-bit Color
@op("type.8bit_color", "8-bit Color", "Image/Type", kind="image", scope_param=False, preview=False, roi_aware=False, keywords=["indexed colour", "palette", "quantize", "median cut"],
    params=[Param("colors", "int", 256, min=2, max=256, label="Number of colours")],
    description="Convert an RGB image to an 8-bit indexed-colour image with a median-cut palette of N colours (stored as a runtime LUT 'Indexed <name>').")
def to_8bit_color(ctx, colors=256):
    from PIL import Image
    img = ctx.img
    if not img.rgb:
        raise ValueError("8-bit Color needs an RGB image")
    H, W = img.height, img.width
    # like ImageJ's StackConverter: one palette for the whole stack, built from all slices stacked vertically
    tall = np.ascontiguousarray(img.data.transpose(0, 1, 3, 4, 2).reshape(img.nt * img.nz * H, W, 3))
    q = Image.fromarray(tall, "RGB").quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.asarray(q.getpalette(), np.uint8)[: 3 * 256].reshape(-1, 3)
    table = np.zeros((256, 3), np.uint8)
    table[: len(pal)] = pal
    out = np.ascontiguousarray(np.asarray(q, np.uint8).reshape(img.nt, img.nz, 1, H, W))
    used = int(len(np.unique(out)))
    lut_name = register_lut(f"Indexed {img.name}", table, replace=True)
    img.display.c = 0
    return OpResult(data=out, rgb=False, display={"mode": "single", "c": 0, "channels": [{"lut": lut_name, "min": 0, "max": 255, "name": "Index"}]},
                    message=f"{used} colour(s) used; LUT {lut_name!r}")


# ============================================================================ Lookup tables
def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=max(int(size), 6))
    except TypeError:  # pragma: no cover – Pillow < 10.1
        return ImageFont.load_default()


def _text_w(draw, text: str, font) -> int:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return int(r - l)
    except Exception:  # pragma: no cover
        return int(len(text) * 7)


def _fmt_tick(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.4g}"


def render_lut_card(name: str, vmin: float = 0.0, vmax: float = 255.0, inverted: bool = False, width: int = 256, bar_height: int = 32, ticks: int = 5,
                    title: str | None = None) -> np.ndarray:
    """ImageJ-style 'Show LUT' card: a colour gradient bar, tick marks and the display range → (H, W, 3) uint8."""
    from PIL import Image, ImageDraw
    table = get_lut(name)
    if inverted:
        table = table[::-1]
    font = _font(11)
    margin, label_h, title_h = 24, 34, 22
    W = int(width) + 2 * margin
    H = title_h + int(bar_height) + label_h + 10 + 3 * 16
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    idx = np.clip((np.arange(width) * 255.0 / max(width - 1, 1)).round().astype(int), 0, 255)
    bar = np.repeat(table[idx][None, :, :], int(bar_height), axis=0)
    canvas.paste(Image.fromarray(bar, "RGB"), (margin, title_h))
    draw.rectangle([margin - 1, title_h - 1, margin + width, title_h + bar_height], outline=(0, 0, 0))
    draw.text((margin, 4), title or f"{name}{' (inverted)' if inverted else ''}", fill=(0, 0, 0), font=font)
    y0 = title_h + bar_height
    for k in range(int(ticks)):
        f = k / max(int(ticks) - 1, 1)
        x = margin + int(round(f * (width - 1)))
        draw.line([(x, y0), (x, y0 + 6)], fill=(0, 0, 0))
        label = _fmt_tick(vmin + f * (vmax - vmin))
        tw = _text_w(draw, label, font)
        draw.text((min(max(x - tw // 2, 2), W - tw - 2), y0 + 8), label, fill=(0, 0, 0), font=font)
    yy = y0 + label_h
    for row, ch_name in enumerate(("red", "green", "blue")):
        vals = table[idx, row].astype(int)
        draw.text((2, yy + row * 16 + 1), ch_name[0].upper(), fill=(0, 0, 0), font=font)
        xs = margin + np.arange(width)
        ys = yy + row * 16 + 14 - (vals * 13 / 255).astype(int)
        draw.line(np.stack([xs, ys], axis=1).tolist(), fill={"red": (200, 0, 0), "green": (0, 150, 0), "blue": (0, 0, 220)}[ch_name], width=1)
    return np.asarray(canvas, np.uint8)


@op("color.show_lut", "Show LUT", LUTS, kind="analysis", preview=False, roi_aware=False, keywords=["lookup table", "colour bar", "show lut"],
    params=[Param("width", "int", 256, min=64, max=2048, advanced=True), Param("ticks", "int", 5, min=2, max=20, advanced=True)],
    description="Show the current channel's lookup table as a colour-gradient card with the display range and the R/G/B curves (ImageJ's Image ▸ Color ▸ Show LUT).")
def show_lut(ctx, width=256, ticks=5):
    img = ctx.img
    ch = img.display.channels[img.display.c]
    card = render_lut_card(ch.lut, ch.min, ch.max, ch.inverted, width=width, ticks=ticks)
    new = LumenImage(card, name=f"LUT of {img.name}", axes="YXS", rgb=True)
    new.metadata["lut"] = ch.lut
    return OpResult(new_images=[new], message=f"{ch.lut}: {ch.min:g} – {ch.max:g}")


@op("color.display_luts", "Display LUTs", LUTS, **dict(_NOIMG, kind="image"), keywords=["all luts", "lut montage", "colormaps"],
    params=[Param("width", "int", 256, min=64, max=2048, advanced=True), Param("row_height", "int", 20, min=8, max=100, advanced=True)],
    description="Montage of every available lookup table (built-in and imported), one labelled bar per LUT.")
def display_luts(ctx, width=256, row_height=20):
    from PIL import Image, ImageDraw
    names = list(LUT_NAMES)
    font = _font(11)
    label_w = 110
    W = label_w + int(width) + 8
    H = len(names) * (int(row_height) + 4) + 4
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    idx = np.clip((np.arange(width) * 255.0 / max(width - 1, 1)).round().astype(int), 0, 255)
    for i, nm in enumerate(names):
        y = 4 + i * (int(row_height) + 4)
        bar = np.repeat(get_lut(nm)[idx][None], int(row_height), axis=0)
        canvas.paste(Image.fromarray(bar, "RGB"), (label_w, y))
        draw.text((4, y + max((row_height - 12) // 2, 0)), nm[:18], fill=(0, 0, 0), font=font)
    new = LumenImage(np.asarray(canvas, np.uint8), name="LUTs", axes="YXS", rgb=True)
    return OpResult(new_images=[new], message=f"{len(names)} lookup tables")


@op("color.edit_lut", "Edit LUT…", LUTS, **_NOIMG, keywords=["custom lut", "gradient", "lut table"],
    params=[Param("name", "string", "Custom", help="Name of the new LUT"),
            Param("table", "text", "", help="256 lines of 'r g b' (or 'index r g b'); fewer lines are interpolated. Leave empty to use the start / end colours"),
            Param("start", "color", "#000000", label="Start colour"), Param("end", "color", "#ffffff", label="End colour"),
            Param("apply", "bool", True, label="Apply to the current channel")],
    description="Create a lookup table from a text table of RGB values or as a gradient between two colours, register it and apply it.")
def edit_lut(ctx, name="Custom", table="", start="#000000", end="#ffffff", apply=True):
    txt = str(table or "").strip()
    tab = parse_lut_text(txt) if txt else gradient_lut(start, end)
    clean = str(name or "").strip() or "Custom"
    # re-running with the same custom name updates that LUT in place; a built-in name ('Grays', 'Fire', …) is
    # never overwritten and becomes e.g. 'Grays 2' (register_lut enforces this too)
    lut_name = register_lut(clean, tab, replace=not is_builtin(clean))
    msg = f"LUT {lut_name!r} created"
    img = ctx.session.active
    if apply and img is not None:
        patch = [{"lut": lut_name} if i == img.display.c else {} for i in range(img.nc)]
        ctx.session.update_display(img.id, {"channels": patch})
        msg += f"; applied to {img.name}"
    return OpResult(message=msg)


# ============================================================================ Help
LINKS = [
    ("Lumen", "Lumen command list (this app)", "lumen://commands"),
    ("ImageJ", "ImageJ User Guide", "https://imagej.net/ij/docs/guide/"),
    ("ImageJ", "ImageJ menu commands reference", "https://imagej.net/ij/docs/menus/"),
    ("ImageJ", "ImageJ macro language", "https://imagej.net/ij/developer/macro/macros.html"),
    ("ImageJ", "Built-in macro functions", "https://imagej.net/ij/developer/macro/functions.html"),
    ("ImageJ", "ImageJ sample images", "https://imagej.net/images/"),
    ("ImageJ", "ImageJ mailing list / forum", "https://forum.image.sc/tag/imagej"),
    ("Fiji", "Fiji website", "https://fiji.sc/"),
    ("Fiji", "Fiji / ImageJ wiki", "https://imagej.net/"),
    ("Fiji", "Image.sc forum", "https://forum.image.sc/"),
    ("Fiji", "Fiji plugins index", "https://imagej.net/list-of-extensions"),
    ("Fiji", "Fiji LUTs", "https://imagej.net/ij/download/luts/"),
    ("Python", "scikit-image documentation", "https://scikit-image.org/docs/stable/"),
    ("Python", "SciPy ndimage", "https://docs.scipy.org/doc/scipy/reference/ndimage.html"),
    ("Python", "NumPy", "https://numpy.org/doc/"),
]


@op("help.links", "Lumen / ImageJ Links", HELP, **_NOIMG, keywords=["documentation", "help", "website", "forum"],
    description="A table of documentation links for Lumen, ImageJ, Fiji and the Python imaging stack.")
def links(ctx):
    rows = [[grp, title, url] for grp, title, url in LINKS]
    return OpResult(table={"name": "Links", "columns": ["Group", "Title", "URL"], "rows": rows}, message=f"{len(rows)} links")


def _module_label(mod: str) -> tuple[str, str]:
    if mod.startswith("lumen.core.ops."):
        return mod.rsplit(".", 1)[1], "built-in"
    if mod.startswith("lumen_plugin_"):
        return mod[len("lumen_plugin_"):], "user plugin"
    return mod, "other"


@op("help.about_plugins", "About Plugins / Command Modules", HELP, **_NOIMG, keywords=["plugins", "modules", "commands", "about"],
    description="List every loaded command module (built-in and ~/.lumen/plugins) with its command count, plus modules that failed to import.")
def about_plugins(ctx):
    counts: dict[str, dict[str, Any]] = {}
    for spec in REGISTRY.ops.values():
        mod = getattr(spec.func, "__module__", "") or ""
        name, kind = _module_label(mod)
        e = counts.setdefault(name, {"kind": kind, "n": 0, "categories": set(), "example": spec.id})
        e["n"] += 1
        e["categories"].add(spec.category.split("/")[0])
    rows = []
    for name in sorted(counts, key=lambda k: (counts[k]["kind"] != "built-in", k.lower())):
        e = counts[name]
        rows.append([name, e["kind"], e["n"], ", ".join(sorted(e["categories"])), e["example"], "loaded"])
    try:
        from . import _FAILED
    except Exception:  # pragma: no cover
        _FAILED = {}
    for name, err in sorted(_FAILED.items()):
        rows.append([name, "built-in", 0, "", "", f"failed: {err}"])
    total = len(REGISTRY.ops)
    n_plugins = sum(1 for e in counts.values() if e["kind"] == "user plugin")
    return OpResult(table={"name": "Plugins", "columns": ["Module", "Kind", "Commands", "Menus", "Example", "Status"], "rows": rows},
                    message=f"{total} commands in {len(counts)} modules ({n_plugins} user plugin(s), {len(_FAILED)} failed)")


@op("help.command", "Command Help…", HELP, **_NOIMG, keywords=["help", "describe", "parameters", "usage"],
    params=[Param("command", "string", "", help="Command id (e.g. filters.gaussian) or menu name (e.g. Gaussian Blur)")],
    description="Describe a command: its id, menu, kind, shortcut and a table of its parameters with types, defaults and choices.")
def command_help(ctx, command=""):
    key = str(command or "").strip()
    if not key:
        raise ValueError("Give a command id or name")
    try:
        spec = REGISTRY.get(key)
    except KeyError:
        low = key.lower()
        hits = [s for s in REGISTRY.ops.values() if low in s.id.lower() or low in s.name.lower() or any(low in k.lower() for k in s.keywords)]
        if not hits:
            raise ValueError(f"No command matches {key!r}")
        if len(hits) > 1:
            rows = [[s.id, s.name, s.menu or s.category, s.description] for s in hits[:50]]
            return OpResult(table={"name": "Command Help", "columns": ["Id", "Name", "Menu", "Description"], "rows": rows},
                            message=f"{len(hits)} commands match {key!r}")
        spec = hits[0]
    rows = []
    for p in spec.params:
        rng = ""
        if p.min is not None or p.max is not None:
            lo = "" if p.min is None else f"{p.min:g}"  # format each bound on its own: a one-sided range must not
            hi = "" if p.max is None else f"{p.max:g}"  # push '' through the 'g' spec (ValueError)
            rng = f"{lo}–{hi}"
        rows.append([p.name, p.label, p.type, p.default, ", ".join(p.choices) if p.choices else rng, p.unit, p.help])
    kind = {"plane": "plane (filter, undoable, ROI-aware)", "image": "image (undoable)", "analysis": "analysis (no pixel change)"}.get(spec.kind, spec.kind)
    msg = f"{spec.name} [{spec.id}] – {spec.description}\nMenu: {spec.menu or spec.category}; kind: {kind}"
    if spec.shortcut:
        msg += f"; shortcut: {spec.shortcut}"
    if spec.requires_roi:
        msg += f"; needs {spec.requires_roi} selection"
    if not spec.requires_image:
        msg += "; works without an image"
    return OpResult(table={"name": "Command Help", "columns": ["Parameter", "Label", "Type", "Default", "Choices / range", "Unit", "Help"], "rows": rows},
                    message=msg, log=msg)
