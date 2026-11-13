"""Ports of the ImageJ 1.54p ROI geometry, drawing and profile algorithms behind Edit ▸ Selection and the macro
functions ``getProfile()`` / ``getSelectionBounds()``.

Each function reproduces the Java arithmetic of the named class / method (float32 where ImageJ uses ``float``, double
where it uses ``double``, ``Math.round`` = floor(v + 0.5)) so that the macro layer gives the same selection masks,
bounds, vertex lists and profile values as ImageJ (verified by tests/crossval).  Ported:

* ``ij.plugin.RoiEnlarger`` – Enlarge… (EDM of the outside ≤ n → dilation, EDM of the inside ≥ n + 1 → shrink,
  rectangles / ovals grow by the integer bounds)
* ``ij.plugin.Selection.makeBand`` – Make Band… (8-bit EDM of the outside, legacy Wand trace of the ≤ n region, minus the ROI)
* ``ij.gui.Roi.getInterpolatedPolygon`` (+ ``PolygonRoi``'s 3-point smoothing of freehand / traced outlines and
  ``Roi.lineCircleIntersection``) – Interpolate
* ``ij.plugin.RoiScaler.scale`` – Edit ▸ Selection ▸ Scale… (centred about the mask centroid with rounded vertices)
* ``ij.gui.Roi.drawPixels`` / ``OvalRoi`` / ``PolygonRoi`` / ``Line`` with ``ImageProcessor.lineTo`` (DDA),
  ``drawDot`` / ``drawDot2`` and ``drawRect`` – Edit ▸ Draw
* ``ij.gui.Roi.convertLineToArea`` for straight lines – Line to Area (the ``RotatedRectRoi`` float polygon, whose
  bounds are the ``FloatPolygon`` bounds)
* ``ij.gui.ProfilePlot`` – ``getProfile()``: ``ImageProcessor.getLine`` for straight lines, column / row averages for
  rectangles, ``getIrregularProfile`` for segmented lines and ``ij.plugin.Straightener.straightenLine`` (with
  ``PolygonRoi.getSpline`` / ``ij.measure.SplineFitter`` / ``getEquidistantPoints``) for wide lines
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import roi as roimod
from ._imagej_binary import edm_float, edm_to_byte
from ._imagej_measure import (get_line, int_polygon_mask, interpolated_value, irregular_profile, jround, line_float_polygon,
                              oval_mask, rotated_rect_polygon, wand_outline)

F32 = np.float32


# ====================================================================== helpers
def _local_mask(roi: dict[str, Any], shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """The ROI as ``ImageProcessor.fill(roi)`` rasterises it: its integer bounds and the rect-sized boolean mask
    (rectangles fill the whole rectangle).  Not clipped to the image."""
    (x, y, w, h), m = roimod.roi_rect_mask(roi, shape)
    if m is None:
        m = np.ones((max(h, 0), max(w, 0)), bool)
    return (int(x), int(y), int(w), int(h)), np.asarray(m, bool)


def _mask_roi_from(sel: np.ndarray, x0: int, y0: int, name: str | None) -> dict[str, Any] | None:
    """A mask ROI of the True pixels of ``sel`` whose pixel (0, 0) sits at image pixel (x0, y0) (may be off-image)."""
    r = roimod.from_mask(sel, name)
    if r is None:
        return None
    r["x"] += int(x0)
    r["y"] += int(y0)
    return r


def _paste(full: np.ndarray, sub: np.ndarray, x0: int, y0: int) -> None:
    """Paste ``sub`` into ``full`` at (x0, y0), clipped (``ImageProcessor.insert``)."""
    H, W = full.shape
    h, w = sub.shape
    ya, yb = max(y0, 0), min(y0 + h, H)
    xa, xb = max(x0, 0), min(x0 + w, W)
    if yb > ya and xb > xa:
        full[ya:yb, xa:xb] = sub[ya - y0:yb - y0, xa - x0:xb - x0]


# ====================================================================== RoiEnlarger
def enlarge_roi(roi: dict[str, Any], n: float, shape: tuple[int, int]) -> dict[str, Any]:
    """``RoiEnlarger.run``: rectangles and ovals grow by round(n) on every side; other area ROIs are rasterised and the
    pixels whose distance to the ROI is ≤ round(n) become the new (mask) selection; a negative n shrinks (see
    :func:`shrink_roi`).  For |n| < 256 ImageJ uses ``enlarge255`` – the 8-bit EDM (``EDM.toEDM``: ``(int)(d + 0.5)``)
    thresholded at 0 … n – and only beyond that the float EDM (``enlarge``: d ≤ n)."""
    if n == 0:
        return roi
    t = roi["type"]
    n2 = jround(float(n))
    if t in ("rect", "oval"):
        return _enlarge_rect_or_oval(roi, n2)
    if n2 < 0:
        return shrink_roi(roi, -n2, shape, byte_edm=abs(n) < 256)
    if n2 == 0:
        return roi
    (bx, by, bw, bh), local = _local_mask(roi, shape)
    if bw <= 0 or bh <= 0 or not local.any():
        return roi
    width, height = bw + 2 * n2 + 2, bh + 2 * n2 + 2
    ip = np.ones((height, width), bool)             # ByteProcessor(width, height).invert(): everything 255 …
    ip[n2 + 1:n2 + 1 + bh, n2 + 1:n2 + 1 + bw] = ~local   # … except the ROI, filled with 0 at (n2+1, n2+1)
    edm = edm_float(ip)                              # distance of every outside pixel to the nearest ROI pixel
    if abs(n) < 256:
        sel = edm_to_byte(edm) <= n2                 # enlarge255: the rounded 8-bit EDM, setThreshold(0, n2)
    else:
        sel = edm <= F32(n2)                         # enlarge: the float EDM, setThreshold(0, n2)
    if not sel.any():
        return roi
    # ThresholdToSelection's ROI is moved so that the rasterised ROI keeps its place: small-image (n2+1, n2+1) ↔ (bx, by)
    out = _mask_roi_from(sel, bx - n2 - 1, by - n2 - 1, roi.get("name"))
    return out if out is not None else roi


def shrink_roi(roi: dict[str, Any], n: int, shape: tuple[int, int], byte_edm: bool = True) -> dict[str, Any]:
    """``RoiEnlarger.shrink255(roi, n)`` (``shrink`` when ``byte_edm`` is off): keep the ROI pixels whose distance to the
    nearest outside pixel – the rounded 8-bit EDM of the ROI inside a 1-pixel background border, or the float EDM – is
    ≥ n + 1."""
    (bx, by, bw, bh), local = _local_mask(roi, shape)
    if bw <= 0 or bh <= 0 or not local.any():
        return roi
    ip = np.zeros((bh + 2, bw + 2), bool)
    ip[1:1 + bh, 1:1 + bw] = local
    edm = edm_float(ip)
    if byte_edm:
        b = edm_to_byte(edm)
        sel = (b >= n + 1) & (b <= 255)
    else:
        sel = edm >= F32(n + 1)
    if not sel.any():
        return roi
    out = _mask_roi_from(sel, bx - 1, by - 1, roi.get("name"))
    return out if out is not None else roi


def _enlarge_rect_or_oval(roi: dict[str, Any], n: int) -> dict[str, Any]:
    """``RoiEnlarger.enlargeRectOrOval``: the integer bounds grow by n on every side (a rectangle stays a rectangle,
    an oval an oval); a selection that would vanish is kept unchanged."""
    x, y, w, h = roimod.int_bounds(roi)
    x, y, w, h = x - n, y - n, w + 2 * n, h + 2 * n
    if w <= 0 or h <= 0:
        return roi
    out = {k: v for k, v in roi.items() if k not in ("x", "y", "w", "h")}
    out.update(type=roi["type"], x=float(x), y=float(y), w=float(w), h=float(h))
    return out


# ====================================================================== Selection.makeBand
def make_band(roi: dict[str, Any], size_px: float, shape: tuple[int, int]) -> dict[str, Any]:
    """``Selection.makeBand``: n = round(size) pixels; the ROI mask is inserted into an image-sized byte image, the
    8-bit EDM of the outside (Distance Map with a white background) is thresholded at 0 … n, the region containing the
    first ROI pixel is traced with the legacy Wand and the original ROI is subtracted (``ShapeRoi.not``)."""
    H, W = shape
    n = jround(float(size_px))
    if n > 255:
        raise ValueError("Cannot make bands wider than 255 pixels")
    if n < 1:
        raise ValueError("Band size must be at least one pixel")
    (rx, ry, rw, rh), local = _local_mask(roi, shape)
    mask = np.zeros((H, W), bool)
    _paste(mask, local, rx, ry)
    if not mask.any() or mask.all():
        raise ValueError("Unable to make band")
    edm8 = edm_to_byte(edm_float(~mask))             # EDM of the background (ROI = 0), BYTE_OVERWRITE rounding
    inside = edm8 <= n
    start = None
    for y in range(max(ry, 0), min(ry + rh, H)):
        for x in range(max(rx, 0), min(rx + rw, W)):
            if edm8[y, x] < n:
                start = (x, y)
                break
        if start is not None:
            break
    if start is None:
        raise ValueError("Unable to make band")
    outline = wand_outline(inside, start[0], start[1], four_connected=False)
    if outline is None or len(outline[0]) == 0:
        raise ValueError("Unable to make band")
    region_sub, (ox, oy, ow, oh) = int_polygon_mask(outline[0], outline[1])
    region = np.zeros((H, W), bool)
    _paste(region, region_sub, ox, oy)
    band = region & ~mask
    out = roimod.from_mask(band, "Band")
    if out is None:
        raise ValueError("Unable to make band")
    return out


# ====================================================================== Roi.getInterpolatedPolygon
def line_circle_intersection(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, rad: float, ignore_outside: bool) -> list[float]:
    """``Roi.lineCircleIntersection``: 0, 2 or 4 coordinates of the intersections of segment A–B with the circle."""
    dx_ac, dy_ac = cx - ax, cy - ay
    len_ac = math.sqrt(dx_ac * dx_ac + dy_ac * dy_ac)
    dx_ab, dy_ab = bx - ax, by - ay
    x_b2 = math.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)
    phi1 = math.atan2(dy_ab, dx_ab)
    phi2 = math.atan2(dy_ac, dx_ac)
    phi3 = phi1 - phi2
    x_c2 = len_ac * math.cos(phi3)
    y_c2 = len_ac * math.sin(phi3)
    if abs(y_c2) > rad:
        return []
    half_chord = math.sqrt(rad * rad - y_c2 * y_c2)
    sect_one = x_c2 - half_chord
    sect_two = x_c2 + half_chord
    out: list[float] = []
    if (0 <= sect_one <= x_b2) or not ignore_outside:
        out += [math.cos(phi1) * sect_one + ax, math.sin(phi1) * sect_one + ay]
    if (0 <= sect_two <= x_b2) or not ignore_outside:
        out += [math.cos(phi1) * sect_two + ax, math.sin(phi1) * sect_two + ay]
    if half_chord == 0 and len(out) > 2:
        out = out[:2]
    return out


def _float_polygon_length(xs: np.ndarray, ys: np.ndarray, is_line: bool) -> float:
    """``FloatPolygon.getLength(isLine)``: float32 coordinate differences summed in double."""
    xs = np.asarray(xs, F32)
    ys = np.asarray(ys, F32)
    length = 0.0
    for i in range(len(xs) - 1):
        dx = float(xs[i + 1] - xs[i])
        dy = float(ys[i + 1] - ys[i])
        length += math.sqrt(dx * dx + dy * dy)
    if not is_line and len(xs):
        dx = float(xs[0] - xs[-1])
        dy = float(ys[0] - ys[-1])
        length += math.sqrt(dx * dx + dy * dy)
    return length


def smooth_float_polygon(xs: np.ndarray, ys: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """``PolygonRoi.getInterpolatedPolygon(interval, smooth=true)`` for TRACED_ROI / FREEROI / FREELINE: the in-place
    3-point running average in float32 (each vertex uses its already smoothed predecessor); closed outlines also wrap the
    first and last vertex."""
    x = np.asarray(xs, F32).copy()
    y = np.asarray(ys, F32).copy()
    n = len(x)
    three = F32(3)
    for i in range(1, n - 2):
        x[i] = F32(F32(x[i - 1] + x[i]) + x[i + 1]) / three
        y[i] = F32(F32(y[i - 1] + y[i]) + y[i + 1]) / three
    if closed and n >= 3:
        x[0] = F32(F32(x[n - 1] + x[0]) + x[1]) / three
        y[0] = F32(F32(y[n - 1] + y[0]) + y[1]) / three
        x[n - 1] = F32(F32(x[n - 2] + x[n - 1]) + x[0]) / three
        y[n - 1] = F32(F32(y[n - 2] + y[n - 1]) + y[0]) / three
    return x, y


def interpolated_polygon(xs: Any, ys: Any, interval: float, is_line: bool) -> tuple[np.ndarray, np.ndarray]:
    """``Roi.getInterpolatedPolygon(FloatPolygon p, double interval, boolean smooth)`` (the smoothing is applied by the
    caller, see :func:`smooth_float_polygon`): vertices spaced |interval| pixels apart along the outline, found as
    circle–segment intersections; a negative interval lets the spacing shrink (≤ 20 %) so that the last segment
    matches too ("adjust interval to match").  Returns float32 arrays."""
    px = np.asarray(xs, F32).astype(np.float64)
    py = np.asarray(ys, F32).astype(np.float64)
    allow_to_adjust = interval < 0
    interval = abs(float(interval))
    length = _float_polygon_length(px, py, is_line)
    npoints = len(px)
    if npoints < 2:
        return px.astype(F32), py.astype(F32)
    if interval < 0.01:
        raise ValueError("Interval must be >= 0.01")
    if not is_line:
        px = np.append(px, px[0])
        py = np.append(py, py[0])
        npoints += 1
    npoints2 = int(10 + (length * 1.5) / interval)
    try_interval = interval
    min_diff = 1e9
    best_interval = 0.0
    dest_x = [0.0] * max(npoints2, 2)
    dest_y = [0.0] * max(npoints2, 2)
    n_trials = 50
    trial = 0
    dest_ptr = 0
    while trial <= n_trials:
        dest_x[0] = px[0]
        dest_y[0] = py[0]
        src_ptr = 0
        dest_ptr = 0
        xa, ya = px[0], py[0]
        while src_ptr < npoints - 1:
            xc, yc = dest_x[dest_ptr], dest_y[dest_ptr]
            xb, yb = px[src_ptr + 1], py[src_ptr + 1]
            inter = line_circle_intersection(xa, ya, xb, yb, xc, yc, try_interval, True)
            if len(inter) >= 2:
                xa, ya = inter[0], inter[1]
                dest_ptr += 1
                if dest_ptr >= len(dest_x):
                    dest_x.append(0.0)
                    dest_y.append(0.0)
                dest_x[dest_ptr] = xa
                dest_y[dest_ptr] = ya
            else:
                src_ptr += 1
                xa, ya = px[src_ptr], py[src_ptr]
        dest_ptr += 1
        if dest_ptr >= len(dest_x):
            dest_x.append(0.0)
            dest_y.append(0.0)
        dest_x[dest_ptr] = px[npoints - 1]
        dest_y[dest_ptr] = py[npoints - 1]
        dest_ptr += 1
        if not allow_to_adjust:
            if is_line:
                dest_ptr -= 1
            break
        n_segments = dest_ptr - 1
        dx = dest_x[dest_ptr - 2] - dest_x[dest_ptr - 1]
        dy = dest_y[dest_ptr - 2] - dest_y[dest_ptr - 1]
        last_seg = math.sqrt(dx * dx + dy * dy)
        diff = last_seg - try_interval
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_interval = try_interval
        feedback = 0.66
        try_interval = try_interval + feedback * diff / n_segments
        if (try_interval < 0.8 * interval or abs(diff) < 0.05 or trial == n_trials - 1) and trial < n_trials:
            trial = n_trials
            try_interval = best_interval
        else:
            trial += 1
    if not is_line:
        dest_ptr -= 1
    return np.asarray(dest_x[:dest_ptr], F32), np.asarray(dest_y[:dest_ptr], F32)


# ====================================================================== RoiScaler
def _roi_float_polygon(roi: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``Roi.getFloatPolygon()`` as float32 (rectangle corners, the flattened ellipse of an oval, the vertices of a
    polygon / polyline / point set, the end points of a straight line)."""
    t = roi["type"]
    if t == "line":
        return np.asarray([roi["x1"], roi["x2"]], F32), np.asarray([roi["y1"], roi["y2"]], F32)
    if t in ("polyline", "point"):
        pts = np.asarray(roi.get("points", []), np.float64).reshape(-1, 2)
    else:
        pts = np.asarray(roimod.polygon_points(roi), np.float64).reshape(-1, 2)
    return pts[:, 0].astype(F32), pts[:, 1].astype(F32)


def _mask_centroid(roi: dict[str, Any], shape: tuple[int, int]) -> tuple[float, float]:
    """``ImageStatistics.getStatistics(ip, CENTROID, null)`` with the ROI set on the image: the uncalibrated centroid
    of the (image-clipped) mask pixels, pixel centres (+0.5)."""
    from ._imagej_measure import clip_rect_mask
    H, W = shape
    rect, m = roimod.roi_rect_mask(roi, shape)
    (x, y, w, h), m = clip_rect_mask(rect, m, W, H)
    if w <= 0 or h <= 0:
        raise ValueError("The selection lies outside the image")
    if m is None:
        m = np.ones((h, w), bool)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        raise ValueError("Empty selection")
    return float(xs.sum()) / xs.size + x + 0.5, float(ys.sum()) / ys.size + y + 0.5


def scale_roi_ij(roi: dict[str, Any], xscale: float, yscale: float, centered: bool, shape: tuple[int, int]) -> dict[str, Any] | None:
    """``RoiScaler.scale(roi, xscale, yscale, centered)`` for non-composite ROIs (None for mask ROIs, which ImageJ
    scales as shapes): the float polygon is scaled about the origin, or – 'centered' – about the mask centroid (lines:
    the rounded bounds centre) with every vertex rounded to an integer.  Rectangles and ovals become polygons
    (ImageJ: POLYGON / FREEROI)."""
    t = roi["type"]
    if t == "mask":
        return None
    xs, ys = _roi_float_polygon(roi)
    if xs.size == 0:
        raise ValueError("Empty selection")
    if centered:
        if t in ("line", "polyline"):
            bx, by, bw, bh = roimod.int_bounds(roi)
            cx = bx + jround(bw / 2.0)
            cy = by + jround(bh / 2.0)
        else:
            cx, cy = _mask_centroid(roi, shape)
        nx = [float(F32(jround((float(x) - cx) * xscale + cx))) for x in xs]
        ny = [float(F32(jround((float(y) - cy) * yscale + cy))) for y in ys]
    else:
        nx = [float(F32(float(x) * xscale)) for x in xs]
        ny = [float(F32(float(y) * yscale)) for y in ys]
    pts = [[x, y] for x, y in zip(nx, ny)]
    keep = {k: v for k, v in roi.items() if k not in ("x", "y", "w", "h", "x1", "y1", "x2", "y2", "points", "rle", "type", "subtype")}
    if t == "line":
        out = {**keep, "type": "line", "x1": pts[0][0], "y1": pts[0][1], "x2": pts[1][0], "y2": pts[1][1]}
    elif t == "point":
        out = {**keep, "type": "point", "points": pts}
    elif t == "polyline":
        out = {**keep, "type": "polyline", "points": pts}
    else:
        out = {**keep, "type": "freehand" if t == "oval" else "polygon", "points": pts}
    if t in ("line", "polyline"):
        w = float(roi.get("width", 1) or 1)
        if w != 0:
            out["width"] = w * xscale
    return out


# ====================================================================== Roi.convertLineToArea
def line_to_area_roi(roi: dict[str, Any]) -> dict[str, Any] | None:
    """``Roi.convertLineToArea`` of a straight line: the ``RotatedRectRoi`` of width max(strokeWidth, 1) as a sub-pixel
    polygon (its mask and integer bounds follow ImageJ's ``PolygonRoi`` float path).  None for segmented lines."""
    if roi.get("type") != "line":
        return None
    width = max(float(roi.get("width", 1) or 1), 1.0)
    xs, ys = rotated_rect_polygon(float(roi["x1"]), float(roi["y1"]), float(roi["x2"]), float(roi["y2"]), width)
    out = {k: v for k, v in roi.items() if k not in ("x1", "y1", "x2", "y2", "width", "arrow", "arrow_style", "type")}
    # 'rotated_rect': its integer bounds follow RotatedRectRoi.makePolygonRelative (FloatPolygon.getBounds), not setIntBounds
    out.update(type="polygon", subtype="rotated_rect", points=[[float(x), float(y)] for x, y in zip(xs, ys)])
    return out


# ====================================================================== RoiRotator
def rotate_roi_ij(roi: dict[str, Any], angle: float) -> dict[str, Any] | None:
    """``RoiRotator.rotate(roi, angle)`` for non-composite ROIs (None for mask ROIs): every vertex of the float polygon
    is rotated by ``angle`` degrees (clockwise on screen) about the centre of the integer bounds
    (``Roi.getRotationCenter``) in double and stored as float32; rectangles become polygons, ovals and traced
    outlines freehand selections, lines stay lines, rotated rectangles stay rotated rectangles."""
    t = roi["type"]
    if t == "mask":
        return None
    if t == "line":   # Line.getRotationCenter: the mid point
        xc, yc = (float(roi["x1"]) + float(roi["x2"])) / 2.0, (float(roi["y1"]) + float(roi["y2"])) / 2.0
    else:             # Roi.getRotationCenter: the centre of getFloatBounds()
        bx, by, bw, bh = roimod.bounding_box(roi)
        xc, yc = bx + bw / 2.0, by + bh / 2.0
    theta = -float(angle) * math.pi / 180.0
    xs, ys = _roi_float_polygon(roi)
    if xs.size == 0:
        raise ValueError("Empty selection")
    nx, ny = [], []
    for x, y in zip(xs, ys):
        dx = float(x) - xc
        dy = yc - float(y)
        radius = math.sqrt(dx * dx + dy * dy)
        a = math.atan2(dy, dx)
        nx.append(float(F32(xc + radius * math.cos(a + theta))))
        ny.append(float(F32(yc - radius * math.sin(a + theta))))
    pts = [[x, y] for x, y in zip(nx, ny)]
    keep = {k: v for k, v in roi.items() if k not in ("x", "y", "w", "h", "x1", "y1", "x2", "y2", "points", "rle", "type", "subtype")}
    if t == "line":
        return {**keep, "type": "line", "x1": pts[0][0], "y1": pts[0][1], "x2": pts[1][0], "y2": pts[1][1]}
    if t == "point":
        return {**keep, "type": "point", "points": pts}
    if t == "polyline":
        return {**keep, "type": "polyline", "points": pts}
    out = {**keep, "type": "freehand" if t == "oval" else "polygon", "points": pts}
    if roi.get("subtype") == "rotated_rect":
        out["subtype"] = "rotated_rect"
    return out


# ====================================================================== drawing (ImageProcessor.lineTo / drawDot, Roi.drawPixels)
class _Pen:
    """``ImageProcessor`` drawing state: current position, line width, the dot mask and the clip (the whole plane)."""

    def __init__(self, target: np.ndarray, value: Any, line_width: int = 1) -> None:
        self.t = target
        self.v = value
        self.H, self.W = target.shape
        self.cx = 0
        self.cy = 0
        self.lw = max(int(line_width), 1)
        self._dot: np.ndarray | None = None

    def set_line_width(self, w: int) -> None:
        self.lw = max(int(w), 1)

    def draw_pixel(self, x: int, y: int) -> None:
        if 0 <= x < self.W and 0 <= y < self.H:
            self.t[y, x] = self.v

    def draw_dot2(self, x: int, y: int) -> None:
        self.draw_pixel(x, y)
        self.draw_pixel(x - 1, y)
        self.draw_pixel(x, y - 1)
        self.draw_pixel(x - 1, y - 1)

    def draw_dot(self, xc: int, yc: int) -> None:
        """``drawDot``: the ``OvalRoi(0, 0, lineWidth, lineWidth)`` mask centred on (xc, yc)."""
        lw = self.lw
        r = lw / 2.0
        xmin, ymin = int(xc - r + 0.5), int(yc - r + 0.5)
        if self._dot is None or self._dot.shape[0] != lw:
            self._dot = oval_mask(lw, lw)
        ys, xs = np.nonzero(self._dot)
        xs = xs + xmin
        ys = ys + ymin
        ok = (xs >= 0) & (xs < self.W) & (ys >= 0) & (ys < self.H)
        self.t[ys[ok], xs[ok]] = self.v

    def move_to(self, x: int, y: int) -> None:
        self.cx, self.cy = int(x), int(y)

    def line_to(self, x2: int, y2: int) -> None:
        """``ImageProcessor.lineTo``: n = max(|dx|, |dy|) + 1 dots at the rounded positions of a DDA walk."""
        x2, y2 = int(x2), int(y2)
        dx, dy = x2 - self.cx, y2 - self.cy
        n = max(abs(dx), abs(dy))
        xinc = dx / n if dx != 0 else 0.0
        yinc = dy / n if dy != 0 else 0.0
        x, y = float(self.cx), float(self.cy)
        self.cx, self.cy = x2, y2
        for _ in range(n + 1):
            px, py = jround(x), jround(y)
            if self.lw == 1:
                self.draw_pixel(px, py)
            elif self.lw == 2:
                self.draw_dot2(px, py)
            else:
                self.draw_dot(px, py)
            x += xinc
            y += yinc

    def draw_polygon(self, xs: list[int], ys: list[int]) -> None:
        """``ImageProcessor.drawPolygon``: moveTo the first vertex, lineTo every vertex, lineTo the first again."""
        if not xs:
            return
        self.move_to(xs[0], ys[0])
        for x, y in zip(xs, ys):
            self.line_to(x, y)
        self.line_to(xs[0], ys[0])

    def draw_polyline(self, xs: list[int], ys: list[int], close: bool) -> None:
        if not xs:
            return
        self.move_to(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            self.line_to(x, y)
        if close:
            self.line_to(xs[0], ys[0])


def oval_outline_polygon(roi: dict[str, Any]) -> tuple[list[int], list[int]]:
    """``OvalRoi.getPolygon()``: the legacy Wand outline (corner vertices) of the oval's mask, in image coordinates."""
    x, y, w, h = roimod.int_bounds(roi)
    if w <= 0 or h <= 0:
        return [], []
    m = oval_mask(w, h)
    out = wand_outline(m, w // 2, h // 2, four_connected=False)
    if out is None:
        return [], []
    return [int(v) + x for v in out[0]], [int(v) + y for v in out[1]]


def _vertex_ints(pts: np.ndarray) -> tuple[list[int], list[int]]:
    """Integer vertices as ``PolygonRoi.drawPixels`` uses them: the stored ints, or round(xbase + xpf[i]) for sub-pixel
    polygons (Java ``Math.round``)."""
    if pts.size and np.all(pts == np.floor(pts)):
        return [int(v) for v in pts[:, 0]], [int(v) for v in pts[:, 1]]
    return [jround(float(F32(v))) for v in pts[:, 0]], [jround(float(F32(v))) for v in pts[:, 1]]


def draw_roi(target: np.ndarray, roi: dict[str, Any], value: Any, line_width: int = 1) -> bool:
    """``Roi.drawPixels(ip)`` of ImageJ's Edit ▸ Draw with ``ip.setLineWidth(Line.getWidth())`` (``line_width``) and
    ``ip.setColor(value)`` – burns the outline into ``target`` (a 2-D plane, modified in place).  Rectangles:
    ``drawRect`` from (x, y) to (x + w, y + h); ovals: the Wand outline of the mask; polygons: the (rounded) vertices,
    closed; segmented lines: open; straight lines: a 1-pixel DDA line, dots of the line's own width ('fillLine'), or the
    4-vertex outline when both widths exceed 1.  Returns False for ROI types ImageJ draws differently (points, composite
    masks) so the caller can fall back."""
    pen = _Pen(target, value, line_width)
    t = roi["type"]
    if t == "rect":
        x, y, w, h = roimod.int_bounds(roi)
        if w < 1 or h < 1:
            return True
        pen.move_to(x, y)
        pen.line_to(x + w, y)
        pen.line_to(x + w, y + h)
        pen.line_to(x, y + h)
        pen.line_to(x, y)
        return True
    if t == "oval":
        xs, ys = oval_outline_polygon(roi)
        pen.draw_polygon(xs, ys)
        return True
    if t in ("polygon", "freehand", "polyline"):
        pts = np.asarray(roi.get("points", []), np.float64).reshape(-1, 2)
        if t == "polyline":
            sw = float(roi.get("width", 1) or 1)
            if sw > 1:
                pen.set_line_width(jround(sw))
        xs, ys = _vertex_ints(pts)
        pen.draw_polyline(xs, ys, close=t != "polyline")
        return True
    if t == "line":
        sw = float(roi.get("width", 1) or 1)
        fill_line = sw > 1 and line_width <= 1
        pen.set_line_width(jround(sw) if fill_line else 1)
        x1, y1, x2, y2 = (float(roi[k]) for k in ("x1", "y1", "x2", "y2"))
        if sw <= 1 or fill_line:
            # ImageJ rounds the end points (Math.round); Lumen's line coordinates address pixel centres, so a line at
            # y = 3.5 is drawn on row 3 (floor) – identical for integer coordinates, the macro case
            pen.move_to(math.floor(x1), math.floor(y1))
            pen.line_to(math.floor(x2), math.floor(y2))
        else:
            px, py = line_float_polygon(x1, y1, x2, y2, sw)
            pen.draw_polygon([jround(float(v)) for v in px], [jround(float(v)) for v in py])
        return True
    return False


# ====================================================================== SplineFitter / PolygonRoi.getSpline / getEquidistantPoints
class SplineFitter:
    """``ij.measure.SplineFitter`` (open curves): the natural cubic spline through (x[i], y[i]) – Numerical Recipes'
    ``spline`` / ``splint`` with ImageJ's mixed float / double arithmetic."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, F32)
        self.y = np.asarray(y, F32)
        n = len(self.x)
        y2 = [0.0] * n
        u = [0.0] * n
        xf, yf = self.x, self.y
        for i in range(1, n - 1):
            sig = float(F32(xf[i] - xf[i - 1]) / F32(xf[i + 1] - xf[i - 1]))
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            ui = float(F32(F32(yf[i + 1] - yf[i]) / F32(xf[i + 1] - xf[i])) - F32(F32(yf[i] - yf[i - 1]) / F32(xf[i] - xf[i - 1])))
            u[i] = (6.0 * ui / float(F32(xf[i + 1] - xf[i - 1])) - sig * u[i - 1]) / p
        qn = un = 0.0
        if n >= 2:
            y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)
            for k in range(n - 2, -1, -1):
                y2[k] = y2[k] * y2[k + 1] + u[k]
        self.y2 = y2

    def eval(self, xp: float) -> float:
        x, y, y2 = self.x, self.y, self.y2
        n = len(x)
        klo, khi = 0, n - 1
        while khi - klo > 1:
            k = (khi + klo) >> 1
            if float(x[k]) > xp:
                khi = k
            else:
                klo = k
        h = float(F32(x[khi] - x[klo]))
        if h == 0.0:
            return 0.0
        a = (float(x[khi]) - xp) / h
        b = (xp - float(x[klo])) / h
        return a * float(y[klo]) + b * float(y[khi]) + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0


def polygon_length(xs: np.ndarray, ys: np.ndarray, close: bool) -> float:
    """``PolygonRoi.getLength(float[] x, float[] y, n, closeShape, null)`` (uncalibrated)."""
    xs = np.asarray(xs, F32)
    ys = np.asarray(ys, F32)
    n = len(xs)
    if n < 2:
        return 0.0
    length = 0.0
    for i in range(n - 1):
        dx = float(xs[i + 1] - xs[i])
        dy = float(ys[i + 1] - ys[i])
        length += math.sqrt(dx * dx + dy * dy)
    if close:
        dx = float(xs[0] - xs[n - 1])
        dy = float(ys[0] - ys[n - 1])
        length += math.sqrt(dx * dx + dy * dy)
    return length


def get_spline(xpf: np.ndarray, ypf: np.ndarray, evaluation_points: int) -> tuple[np.ndarray, np.ndarray]:
    """``PolygonRoi.getSpline(evaluationPoints)`` of an open polyline: node positions grow by the square root of each
    segment length (float32), two natural splines are evaluated at ``evaluationPoints`` equal parameter steps."""
    xpf = np.asarray(xpf, F32)
    ypf = np.asarray(ypf, F32)
    n = len(xpf)
    node = np.zeros(n, F32)
    last = F32(0)
    for i in range(1, n):
        dx = F32(xpf[i] - xpf[i - 1])
        dy = F32(ypf[i] - ypf[i - 1])
        d2 = F32(F32(dx * dx) + F32(dy * dy))
        dl = F32(math.sqrt(math.sqrt(float(d2))))
        if dl < F32(0.001):
            dl = F32(0.001)
        last = F32(last + dl)
        node[i] = last
    sfx = SplineFitter(node, xpf)
    sfy = SplineFitter(node, ypf)
    scale = float(last) / (evaluation_points - 1) if evaluation_points > 1 else 0.0
    xs = np.empty(evaluation_points, F32)
    ys = np.empty(evaluation_points, F32)
    for i in range(evaluation_points):
        xv = i * scale
        xs[i] = F32(sfx.eval(xv))
        ys[i] = F32(sfy.eval(xv))
    return xs, ys


def get_equidistant_points(xpoints: np.ndarray, ypoints: np.ndarray, segment_length: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``PolygonRoi.getEquidistantPoints``: resample an open polyline at round(length / segmentLength) + 1 points."""
    xpoints = np.asarray(xpoints, F32)
    ypoints = np.asarray(ypoints, F32)
    npoints = len(xpoints)
    length = polygon_length(xpoints, ypoints, False)
    np_out = jround(length / segment_length) + 1
    step = length / (np_out - 1) if np_out > 1 else length
    xo = np.empty(np_out, F32)
    yo = np.empty(np_out, F32)
    xo[0], yo[0] = xpoints[0], ypoints[0]
    length_read = 0.0
    written = 1
    x2, y2 = float(xpoints[0]), float(ypoints[0])
    for i in range(1, npoints):
        x1, y1 = x2, y2
        x2, y2 = float(xpoints[i]), float(ypoints[i])
        dx, dy = x2 - x1, y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        length_read += distance
        over = length_read - written * step
        while (over >= 0.0 or i == npoints - 1) and written < np_out:
            frac = over / distance if distance != 0 else 0.0
            xo[written] = F32(x2 - frac * dx)
            yo[written] = F32(y2 - frac * dy)
            over -= step
            written += 1
    return xo[:written], yo[:written]


# ====================================================================== Straightener.straightenLine
def straighten_line(plane: np.ndarray, xs: Any, ys: Any, line_width: int) -> np.ndarray:
    """``Straightener.straightenLine(imp, line, width)``: the polyline is spline-fitted (``fitSplineForStraightening``:
    ``getSpline((int)(length·2) + 1)`` then 1-pixel equidistant points) and ``line_width`` bilinear samples are taken
    perpendicular to the curve at every point → a float32 image of ``line_width`` rows and one column per point."""
    p = np.asarray(plane)
    if p.dtype == np.uint8 or p.dtype == np.uint16:
        p = p.astype(F32)
    xpf = np.asarray(xs, F32)
    ypf = np.asarray(ys, F32)
    lw = int(line_width)
    # PolygonRoi.finishPolygon keeps the vertices relative to the float bounds (float32); the spline and the equidistant
    # points are computed on those, and getFloatPolygon adds the base back: (float)(xSpline[i] + xbase)
    xbase, ybase = float(xpf.min()), float(ypf.min())
    rel_x = (xpf - F32(xbase)).astype(F32)
    rel_y = (ypf - F32(ybase)).astype(F32)
    length = polygon_length(rel_x, rel_y, False)
    sx, sy = get_spline(rel_x, rel_y, int(length * 2) + 1)
    sx, sy = get_equidistant_points(sx, sy, 1.0)
    sx = (sx.astype(np.float64) + xbase).astype(F32)
    sy = (sy.astype(np.float64) + ybase).astype(F32)
    n = len(sx)
    if n < 2:
        return np.zeros((max(lw, 1), 0), F32)
    out = np.zeros((max(lw, 1), n), F32)
    x = float(F32(sx[0] - F32(sx[1] - sx[0])))
    y = float(F32(sy[0] - F32(sy[1] - sy[0])))
    if lw <= 1:
        out[0, 0] = F32(interpolated_value(p, x, y))
    for i in range(n):
        x1, y1 = x, y
        x, y = float(sx[i]), float(sy[i])
        if lw == 1:
            out[0, i] = F32(interpolated_value(p, x, y))
            continue
        dx = x - x1
        dy = y1 - y
        distance = float(F32(math.sqrt(dx * dx + dy * dy)))
        dx /= distance
        dy /= distance
        x2 = x - dy * (lw - 1) / 2.0
        y2 = y - dx * (lw - 1) / 2.0
        for j in range(lw):
            out[j, i] = F32(interpolated_value(p, x2, y2))
            x2 += dy
            y2 += dx
    return out


def column_average_profile(ip2: np.ndarray) -> np.ndarray:
    """``ProfilePlot.getColumnAverageProfile`` of a whole image: the NaN-aware mean of every column, summed row by row
    in double (``ip.getLine`` without interpolation reads the stored values)."""
    a = np.asarray(ip2, np.float64)
    if a.size == 0:
        return np.zeros(a.shape[1], np.float64)
    finite = np.isfinite(a)
    sums = np.add.reduce(np.where(finite, a, 0.0), axis=0)
    counts = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def rgb_gray_plane(data: np.ndarray) -> np.ndarray:
    """``ColorProcessor.getPixelValue`` with the default weights: (r + g + b) / 3 as float32."""
    r, g, b = (data[i].astype(np.float64) for i in range(3))
    return (r * (1 / 3) + g * (1 / 3) + b * (1 / 3)).astype(F32)


def profile_plot(plane: np.ndarray, roi: dict[str, Any], vertical: bool = False, interpolate: bool = True) -> np.ndarray:
    """``new ProfilePlot(imp, vertical).getProfile()`` – the values the macro ``getProfile()`` returns.

    Straight lines: ``ImageProcessor.getLine`` (round(length) + 1 bilinear samples at the line's own coordinates) or, for a
    line wider than one pixel, the column averages of the straightened line; segmented lines: ``getIrregularProfile``
    (width ≤ 1) or the straightened wide line; rectangles: column (or, ``vertical``, row) averages of the integer bounds.
    Other selections raise, as ImageJ's "Line or rectangular selection required"."""
    t = roi.get("type")
    if t == "line":
        x1, y1, x2, y2 = (float(roi[k]) for k in ("x1", "y1", "x2", "y2"))
        width = float(roi.get("width", 1) or 1)
        if width <= 1:
            return get_line(plane, x1, y1, x2, y2, interpolate)
        return column_average_profile(straighten_line(plane, [x1, x2], [y1, y2], int(width)))
    if t == "polyline":
        pts = np.asarray(roi.get("points", []), np.float64).reshape(-1, 2)
        if len(pts) < 2:
            raise ValueError("Selection required")
        lw = jround(float(roi.get("width", 1) or 1))
        if lw <= 1:
            return irregular_profile(plane, pts[:, 0].astype(F32), pts[:, 1].astype(F32), interpolate)
        return column_average_profile(straighten_line(plane, pts[:, 0], pts[:, 1], lw))
    if t == "rect":
        x, y, w, h = roimod.int_bounds(roi)
        p = np.asarray(plane)
        if p.dtype == np.uint8 or p.dtype == np.uint16:
            p = p.astype(F32)
        H, W = p.shape
        if vertical:
            prof = np.zeros(h, np.float64)
            cnt = np.zeros(h, np.int64)
            for xx in range(x, x + w):
                line = get_line(p, xx, y, xx, y + h - 1, False) if 0 <= xx < W else np.full(h, np.nan)
                ok = np.isfinite(line)
                prof[ok] += line[ok]
                cnt[ok] += 1
        else:
            prof = np.zeros(w, np.float64)
            cnt = np.zeros(w, np.int64)
            for yy in range(y, y + h):
                line = get_line(p, x, yy, x + w - 1, yy, False) if 0 <= yy < H else np.full(w, np.nan)
                ok = np.isfinite(line)
                prof[ok] += line[ok]
                cnt[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, prof / np.maximum(cnt, 1), np.nan)
    raise ValueError("Line or rectangular selection required")
