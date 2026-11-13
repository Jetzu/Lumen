"""Bit-exact ports of ImageJ 1.54p measurement code (geometry, statistics, Find Maxima, particle scanning).

Every function names the Java it reproduces; the arithmetic is kept in the same order and precision (double for
``double``, numpy float32 where ImageJ uses ``float``) so that Results values agree with ImageJ to the last printed
digit and masks agree pixel by pixel.  Nothing here depends on the session; ``roi.py``, ``measure.py`` and
``ops/analyze.py`` are the callers.

Ported classes / methods:

* ``ij.process.PolygonFiller``        – polygon rasterisation (even-odd scan line, pixel-centre convention)
* ``ij.gui.OvalRoi``                  – mask, Ramanujan perimeter, Feret override, flattened ellipse polygon
  (``java.awt.geom.FlatteningPathIterator`` / ``EllipseIterator`` / ``CubicCurve2D`` are ported for the latter)
* ``ij.gui.Wand``                     – outline tracing (legacy threshold mode = 8-connected, as the particle analyzer)
* ``ij.process.FloatPolygon``         – gift-wrap convex hull, integer bounds
* ``ij.gui.Roi.getFeretValues``       – Feret, FeretAngle, MinFeret, FeretX/Y from the convex hull
* ``ij.gui.PolygonRoi``               – traced perimeter, smoothed (freehand) perimeter, polygon length, sub-pixel mask
* ``ij.gui.Line`` / ``RotatedRectRoi`` – wide-line polygon, ``ImageProcessor.getLine`` sampling with bilinear
  interpolation (``getInterpolatedValue``)
* ``ij.process.EllipseFitter``        – area-equalising best-fit ellipse (+ ``drawEllipse`` for the Ellipses drawing)
* ``ij.process.Byte/Short/FloatStatistics`` – histogram, mean, SD, mode, min/max, centroid, centre of mass,
  skewness, kurtosis, median, area fraction, limit to threshold
* ``ij.plugin.filter.MaximumFinder`` – prominence maxima, single points, maxima within tolerance, watershed
  segmentation (``make8bit``, ``cleanupMaxima``, ``watershedSegment``, ``cleanupExtraLines``, edge particles)
* ``ij.plugin.filter.ParticleAnalyzer`` – raster scan, wand outline, polygon / flood masks, size and circularity
  filters, edge exclusion, outline / ellipse drawing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

F32 = np.float32
SQRT_PI = 1.772453851          # EllipseFitter.sqrtPi
HALF_PI = 1.5707963267949      # EllipseFitter.HALFPI
NO_THRESHOLD = -808080.0       # ImageProcessor.NO_THRESHOLD
INT_MAX = 2 ** 31 - 1


def jround(v: float) -> int:
    """``java.lang.Math.round(double)``: floor(v + 0.5)."""
    return int(math.floor(v + 0.5))


def _f32(v: Any) -> float:
    """A double holding the float32 rounding of ``v`` (Java ``(float)`` cast)."""
    return float(F32(v))


# ====================================================================== PolygonFiller
def polygon_filler_mask(xs: Any, ys: Any, width: int, height: int, x_offset: float = 0.0, y_offset: float = 0.0) -> np.ndarray:
    """``PolygonFiller.fillByteProcessorMask`` (float-coordinate polygon; NaN x separates sub-polygons).

    Pixel (x, y) is covered by [x, x+1) × [y, y+1); the outline is compared with pixel centres, points exactly on the
    left / upper boundary are outside, on the right / lower boundary inside.  Returns a (height, width) bool mask."""
    xf = [float(v) for v in xs]
    yf = [float(v) for v in ys]
    n = len(xf)
    mask = np.zeros((max(int(height), 0), max(int(width), 0)), dtype=bool)
    if n == 0 or width <= 0 or height <= 0:
        return mask
    ex: list[float] = []
    ey1: list[int] = []
    ey2: list[int] = []
    eslope: list[float] = []
    y_min, y_max = INT_MAX, -INT_MAX - 1
    poly_start = 0
    for i in range(n):
        iplus1 = poly_start if i == n - 1 else i + 1
        if iplus1 >= n or math.isnan(xf[iplus1]):   # after the last point, close the polygon
            iplus1 = poly_start
        if math.isnan(xf[i]):               # a new polygon follows
            poly_start = i + 1
            continue
        if iplus1 >= n:
            continue
        y1f = yf[i] + y_offset
        y2f = yf[iplus1] + y_offset
        x1f = xf[i] + x_offset
        x2f = xf[iplus1] + x_offset
        y1 = jround(y1f)
        y2 = jround(y2f)
        if y1 == y2 or (y1 <= 0 and y2 <= 0):
            continue
        if y1 > y2:
            y1, y2 = y2, y1
            y1f, y2f = y2f, y1f
            x1f, x2f = x2f, x1f
        slope = (x2f - x1f) / (y2f - y1f)
        ex.append(x1f + (y1 - y1f + 0.5) * slope + 1e-8)
        ey1.append(y1)
        ey2.append(y2)
        eslope.append(slope)
        if y1 < y_min:
            y_min = y1
        if y2 > y_max:
            y_max = y2
    edges = len(ex)
    if edges == 0:
        return mask
    y_start = y_min if y_min > 0 else 0
    active: list[int] = []
    if y_min != 0:                          # shiftXValuesAndActivate
        for i in range(edges):
            if ey1[i] < y_start and ey2[i] >= y_start:
                ex[i] += eslope[i] * (y_start - ey1[i])
                active.append(i)
        active.sort(key=lambda k: ex[k])
    for y in range(y_start, min(height, y_max + 1)):
        active = [k for k in active if not (y < ey1[k] or y >= ey2[k])]       # removeInactiveEdges
        for i in range(edges):                                                # activateEdges
            if ey1[i] == y:
                idx = 0
                while idx < len(active) and ex[i] > ex[active[idx]]:
                    idx += 1
                active.insert(idx, i)
        row = mask[y]
        for k in range(0, len(active) - 1, 2):
            x1 = int(ex[active[k]] + 0.5)
            x1 = 0 if x1 < 0 else width if x1 > width else x1
            x2 = int(ex[active[k + 1]] + 0.5)
            x2 = 0 if x2 < 0 else width if x2 > width else x2
            if x2 > x1:
                row[x1:x2] = True
        prev = -math.inf                                                      # updateXCoordinates
        unsorted = False
        for k in active:
            ex[k] += eslope[k]
            if ex[k] < prev:
                unsorted = True
            prev = ex[k]
        if unsorted:
            active.sort(key=lambda k: ex[k])
    return mask


def float_polygon_bounds(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int, int, int]:
    """``FloatPolygon.getBounds`` (float coordinates): x = floor(minX), width = (int)(maxX − x + 0.5)."""
    xs = np.asarray(xs, dtype=F32)
    ys = np.asarray(ys, dtype=F32)
    if xs.size == 0:
        return 0, 0, 0, 0
    min_x, max_x = F32(xs.min()), F32(xs.max())
    min_y, max_y = F32(ys.min()), F32(ys.max())
    ix, iy = int(math.floor(float(min_x))), int(math.floor(float(min_y)))
    w = int(F32(F32(max_x - F32(ix)) + F32(0.5)))
    h = int(F32(F32(max_y - F32(iy)) + F32(0.5)))
    return ix, iy, w, h


def subpixel_polygon_mask(xs: Any, ys: Any) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Mask and integer bounds of a sub-pixel ``PolygonRoi`` (float coordinates): ``makePolygonRelative`` stores
    float32 coordinates relative to the float bounds, ``getMask`` fills them with the fractional offset."""
    xf = np.asarray(xs, dtype=F32)
    yf = np.asarray(ys, dtype=F32)
    x, y, w, h = float_polygon_bounds(xf, yf)
    if w <= 0 or h <= 0 or xf.size == 0:
        return np.zeros((max(h, 0), max(w, 0)), bool), (x, y, w, h)
    xbase = F32(xf.min())
    ybase = F32(yf.min())
    rel_x = (xf - xbase).astype(F32)       # float32 subtraction, as PolygonRoi.makePolygonRelative
    rel_y = (yf - ybase).astype(F32)
    mask = polygon_filler_mask(rel_x.astype(np.float64), rel_y.astype(np.float64), w, h, float(xbase) - x, float(ybase) - y)
    return mask, (x, y, w, h)


def int_polygon_mask(xs: Any, ys: Any) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Mask and bounds of an integer ``PolygonRoi`` (``Polygon.getBounds``: width = maxX − minX)."""
    xi = np.asarray(xs, dtype=np.int64)
    yi = np.asarray(ys, dtype=np.int64)
    if xi.size == 0:
        return np.zeros((0, 0), bool), (0, 0, 0, 0)
    x, y = int(xi.min()), int(yi.min())
    w, h = int(xi.max()) - x, int(yi.max()) - y
    mask = polygon_filler_mask((xi - x).astype(np.float64), (yi - y).astype(np.float64), w, h)
    return mask, (x, y, w, h)


# ====================================================================== OvalRoi
def oval_mask(width: int, height: int) -> np.ndarray:
    """``OvalRoi.getMask``: pixel (x, y) is inside if ((x−a)²/a² + (y−b)²/b²) ≤ 1 with a = w/2 − 0.5, b = h/2 − 0.5."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return np.zeros((max(height, 0), max(width, 0)), bool)
    a, b = width / 2.0, height / 2.0
    a2, b2 = a * a, b * b
    a -= 0.5
    b -= 0.5
    xx = np.arange(width, dtype=np.float64) - a
    yy = np.arange(height, dtype=np.float64) - b
    return (xx * xx / a2)[None, :] + (yy * yy / b2)[:, None] <= 1.0


def oval_length(width: float, height: float, pw: float = 1.0, ph: float = 1.0) -> float:
    """``OvalRoi.getLength``: Ramanujan's approximation of the ellipse circumference."""
    a = 0.5 * width * pw
    b = 0.5 * height * ph
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def oval_feret_values(x: float, y: float, width: float, height: float, pw: float = 1.0, ph: float = 1.0) -> tuple[float, float, float, float, float]:
    """``OvalRoi.getFeretValues``: (Feret, FeretAngle, MinFeret, FeretX, FeretY) – the axes, FeretX/Y calibrated."""
    high_aspect = ph * height > pw * width
    feret = height * ph if high_aspect else width * pw
    angle = 90.0 if high_aspect else 0.0
    min_feret = width * pw if high_aspect else height * ph
    fx = (x + (0.5 * width if high_aspect else 0)) * pw
    fy = (y + (height if high_aspect else 0.5 * height)) * ph
    return feret, angle, min_feret, fx, fy


# ---- java.awt.geom flattening (OvalRoi.getFloatPolygon → ShapeRoi(Ellipse2D).getFloatPolygon, flatness 0.1)
_CTRL_VAL = 0.5522847498307933
_PCV = 0.5 + _CTRL_VAL * 0.5
_NCV = 0.5 - _CTRL_VAL * 0.5
_ELLIPSE_CTRLPTS = (
    (1.0, _PCV, _PCV, 1.0, 0.5, 1.0),
    (_NCV, 1.0, 0.0, _PCV, 0.0, 0.5),
    (0.0, _NCV, _NCV, 0.0, 0.5, 0.0),
    (_PCV, 0.0, 1.0, _NCV, 1.0, 0.5),
)


def _pt_seg_dist_sq(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    """``java.awt.geom.Line2D.ptSegDistSq``."""
    x2 -= x1
    y2 -= y1
    px -= x1
    py -= y1
    dotprod = px * x2 + py * y2
    if dotprod <= 0.0:
        projlen_sq = 0.0
    else:
        px = x2 - px
        py = y2 - py
        dotprod = px * x2 + py * y2
        if dotprod <= 0.0:
            projlen_sq = 0.0
        else:
            projlen_sq = dotprod * dotprod / (x2 * x2 + y2 * y2)
    len_sq = px * px + py * py - projlen_sq
    if len_sq < 0:
        len_sq = 0.0
    return len_sq


def _cubic_flatness_sq(c: list[float]) -> float:
    """``CubicCurve2D.getFlatnessSq``."""
    return max(_pt_seg_dist_sq(c[0], c[1], c[6], c[7], c[2], c[3]), _pt_seg_dist_sq(c[0], c[1], c[6], c[7], c[4], c[5]))


def _cubic_subdivide(c: list[float]) -> tuple[list[float], list[float]]:
    """``CubicCurve2D.subdivide``."""
    x1, y1, ctrlx1, ctrly1, ctrlx2, ctrly2, x2, y2 = c
    left = [x1, y1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x2, y2]
    x1 = (x1 + ctrlx1) / 2.0
    y1 = (y1 + ctrly1) / 2.0
    x2 = (x2 + ctrlx2) / 2.0
    y2 = (y2 + ctrly2) / 2.0
    centerx = (ctrlx1 + ctrlx2) / 2.0
    centery = (ctrly1 + ctrly2) / 2.0
    ctrlx1 = (x1 + centerx) / 2.0
    ctrly1 = (y1 + centery) / 2.0
    ctrlx2 = (x2 + centerx) / 2.0
    ctrly2 = (y2 + centery) / 2.0
    centerx = (ctrlx1 + ctrlx2) / 2.0
    centery = (ctrly1 + ctrly2) / 2.0
    left[2:8] = [x1, y1, ctrlx1, ctrly1, centerx, centery]
    right[0:6] = [centerx, centery, ctrlx2, ctrly2, x2, y2]
    return left, right


def _flatten_cubic(c: list[float], level: int, limit: int, squareflat: float, out: list[tuple[float, float]]) -> None:
    """``FlatteningPathIterator`` for one cubic segment: recursive subdivision until flat or the recursion limit."""
    if level < limit and _cubic_flatness_sq(c) >= squareflat:
        left, right = _cubic_subdivide(c)
        _flatten_cubic(left, level + 1, limit, squareflat, out)
        _flatten_cubic(right, level + 1, limit, squareflat, out)
    else:
        out.append((c[6], c[7]))


def oval_float_polygon(x: float, y: float, width: float, height: float, flatness: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """``OvalRoi.getFloatPolygon``: the ``Ellipse2D`` (enlarged by 0.0004) flattened with ``ShapeRoi.FLATNESS``, in
    float32 image coordinates (the macro ``getSelectionCoordinates`` of an oval, and the hull used for Solidity)."""
    ex, ey = x - 0.0004, y - 0.0004
    ew, eh = width + 0.0008, height + 0.0008
    bx = int(math.floor(ex))                       # Rectangle2D.getBounds → ShapeRoi(Shape) origin
    by = int(math.floor(ey))
    tx, ty = -bx, -by                              # AffineTransform.translate(-x, -y)
    pts: list[tuple[float, float]] = []
    ctrls = _ELLIPSE_CTRLPTS[3]
    cur = ((ex + ctrls[4] * ew) + tx, (ey + ctrls[5] * eh) + ty)   # AffineTransform (translate only): x + m02
    pts.append(cur)                                # SEG_MOVETO
    squareflat = flatness * flatness
    for seg in range(4):
        c = _ELLIPSE_CTRLPTS[seg]
        p1 = ((ex + c[0] * ew) + tx, (ey + c[1] * eh) + ty)
        p2 = ((ex + c[2] * ew) + tx, (ey + c[3] * eh) + ty)
        p3 = ((ex + c[4] * ew) + tx, (ey + c[5] * eh) + ty)
        _flatten_cubic([cur[0], cur[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], 0, 10, squareflat, pts)
        cur = p3
    xf = [F32(p[0]) for p in pts]
    yf = [F32(p[1]) for p in pts]
    if len(xf) > 1 and xf[-1] == xf[0] and yf[-1] == yf[0]:   # SEG_CLOSE: drop the duplicate closing point
        xf.pop()
        yf.pop()
    xa = (np.asarray(xf, dtype=F32) + F32(bx)).astype(F32)     # Tools.addToArray(xpf, (float)getXBase())
    ya = (np.asarray(yf, dtype=F32) + F32(by)).astype(F32)
    return xa, ya


# ====================================================================== Wand
def wand_outline(inside: np.ndarray, start_x: int, start_y: int, four_connected: bool = False) -> tuple[list[int], list[int]] | None:
    """``Wand.autoOutline(x, y, lower, upper)`` in legacy threshold mode on a boolean 'inside' image: the start pixel
    must be inside or left of the object; traces the 8-connected (or 4-connected) outline and returns its corner
    vertices (a vertical boundary at x separates pixels x−1 and x), or None when no border is found."""
    ins = np.asarray(inside, dtype=bool)
    H, W = ins.shape
    x, y = int(start_x), int(start_y)
    if x < 0 or x >= W or y < 0 or y >= H:
        return None

    def inside_at(px: int, py: int) -> bool:
        return 0 <= px < W and 0 <= py < H and bool(ins[py, px])

    if inside_at(x, y):
        x += 1
        while inside_at(x, y):
            x += 1
    else:
        while True:
            x += 1
            if x >= W:
                return None
            if inside_at(x, y):
                break
    return _trace_edge(ins, x, y, four_connected)


def _trace_edge(ins: np.ndarray, start_x: int, start_y: int, four_connected: bool) -> tuple[list[int], list[int]]:
    """``Wand.traceEdge``."""
    H, W = ins.shape

    def inside_at(px: int, py: int) -> bool:
        return 0 <= px < W and 0 <= py < H and bool(ins[py, px])

    def inside_dir(px: int, py: int, direction: int) -> bool:
        d = direction & 3
        if d == 0:
            return inside_at(px, py)
        if d == 1:
            return inside_at(px, py - 1)
        if d == 2:
            return inside_at(px - 1, py - 1)
        return inside_at(px - 1, py)

    xpts: list[int] = []
    ypts: list[int] = []
    if inside_at(start_x, start_y):
        start_direction = 1
    else:
        start_direction = 3
        start_y += 1
    x, y, direction = start_x, start_y, start_direction
    while True:
        if four_connected:
            nd = direction
            while True:
                if not inside_dir(x, y, nd):
                    break
                nd += 1
                if not nd < direction + 2:
                    break
            nd -= 1
        else:
            nd = direction + 1
            while True:
                if inside_dir(x, y, nd):
                    break
                nd -= 1
                if not nd >= direction:
                    break
        if nd != direction:
            xpts.append(x)
            ypts.append(y)
        d = nd & 3
        if d == 0:
            x += 1
        elif d == 1:
            y -= 1
        elif d == 2:
            x -= 1
        else:
            y += 1
        direction = nd
        if not (x != start_x or y != start_y or (direction & 3) != start_direction):
            break
    if not xpts or xpts[0] != x:
        xpts.append(x)
        ypts.append(y)
    return xpts, ypts


# ====================================================================== FloatPolygon / Roi geometry
def convex_hull(xs: Any, ys: Any) -> tuple[np.ndarray, np.ndarray]:
    """``FloatPolygon.getConvexHull`` (gift wrapping, counter-clockwise in image coordinates, float32 points)."""
    xp = np.asarray(xs, dtype=F32)
    yp = np.asarray(ys, dtype=F32)
    n = int(xp.size)
    if n == 0:
        return xp, yp
    xx: list[float] = []
    yy: list[float] = []
    smallest_y = F32(yp.min())
    smallest_x = F32(np.finfo(F32).max)
    p1 = 0
    for i in range(n):
        if yp[i] == smallest_y and xp[i] < smallest_x:
            smallest_x = xp[i]
            p1 = i
    pstart = p1
    while True:
        x1 = float(xp[p1])
        y1 = float(yp[p1])
        p2 = p1
        x2 = y2 = 0.0
        while True:
            p2 += 1
            if p2 == n:
                p2 = 0
            if p2 == p1:
                break
            x2 = float(xp[p2])
            y2 = float(yp[p2])
            if not (x2 == x1 and y2 == y1):
                break
        p3 = p2 + 1
        if p3 == n:
            p3 = 0
        if p2 != p1:
            while True:
                x3 = float(xp[p3])
                y3 = float(yp[p3])
                determinate = x1 * (y2 - y3) - y1 * (x2 - x3) + (y3 * x2 - y2 * x3)
                collinear_and_further = False
                if determinate == 0 and p3 != p2:
                    d2sqr = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    d3sqr = (x3 - x1) ** 2 + (y3 - y1) ** 2
                    collinear_and_further = d3sqr > d2sqr
                if determinate > 0 or collinear_and_further:
                    x2, y2, p2 = x3, y3, p3
                p3 += 1
                if p3 == n:
                    p3 = 0
                if p3 == p1:
                    break
        xx.append(_f32(x1))
        yy.append(_f32(y1))
        if p2 == p1:
            break
        p1 = p2
        if len(xx) > 1 and xp[p1] == F32(xx[0]) and yp[p1] == F32(yy[0]):
            break
        if p1 == pstart:
            break
    return np.asarray(xx, dtype=F32), np.asarray(yy, dtype=F32)


def feret_values(hull_x: Any, hull_y: Any, pw: float = 1.0, ph: float = 1.0) -> tuple[float, float, float, float, float]:
    """``Roi.getFeretValues`` on a convex hull: (Feret, FeretAngle, MinFeret, FeretX, FeretY); FeretX/Y are the
    uncalibrated coordinates of the left end of the maximum caliper diameter."""
    xs = [float(v) for v in np.asarray(hull_x, dtype=F32)]
    ys = [float(v) for v in np.asarray(hull_y, dtype=F32)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    p1 = p2 = 0
    diameter_sqr = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] - xs[j]) * pw
            dy = (ys[i] - ys[j]) * ph
            dsqr = dx * dx + dy * dy
            if dsqr > diameter_sqr:
                diameter_sqr = dsqr
                p1, p2 = i, j
    if xs[p1] > xs[p2]:
        p1, p2 = p2, p1
    xf1, yf1, xf2, yf2 = xs[p1], ys[p1], xs[p2], ys[p2]
    angle = (180.0 / math.pi) * math.atan2((yf1 - yf2) * ph, (xf2 - xf1) * pw)
    if angle < 0.0:
        angle += 180.0
    feret = math.sqrt(diameter_sqr)
    x0, y0 = xs[n - 1], ys[n - 1]
    min_feret = float(np.finfo(np.float64).max)
    for i in range(n):
        xprev, yprev = x0, y0
        x0, y0 = xs[i], ys[i]
        xnorm = (y0 - yprev) * ph
        ynorm = (xprev - x0) * pw
        norm = math.sqrt(xnorm * xnorm + ynorm * ynorm)
        factor = 1 / norm if norm != 0 else math.inf
        xnorm *= factor * pw
        ynorm *= factor * ph
        max_dist = 0.0
        for j in range(n):
            dx = xs[j] - x0
            dy = ys[j] - y0
            dist = dx * xnorm + dy * ynorm
            if dist > max_dist:
                max_dist = dist
        if max_dist < min_feret:
            min_feret = max_dist
    return feret, angle, min_feret, xf1, yf1


def int_polygon_area(xs: Any, ys: Any) -> float:
    """``Analyzer.getArea(Polygon)``: shoelace area of an integer polygon (|Σ (x_i + x_{i−1})(y_i − y_{i−1})| / 2)."""
    xi = [int(v) for v in xs]
    yi = [int(v) for v in ys]
    n = len(xi)
    if n == 0:
        return float("nan")
    carea = 0
    for i in range(n):
        im1 = i - 1 if i > 0 else n - 1
        carea += (xi[i] + xi[im1]) * (yi[i] - yi[im1])
    return abs(carea / 2.0)


def hull_int_area(xs: Any, ys: Any) -> float:
    """Area of ``Roi.getConvexHull()`` – the float hull rounded to integers (``toIntR``: floor(v + 0.5))."""
    hx, hy = convex_hull(xs, ys)
    return int_polygon_area([math.floor(float(v) + 0.5) for v in hx], [math.floor(float(v) + 0.5) for v in hy])


def traced_perimeter(xp: Any, yp: Any, pw: float = 1.0, ph: float = 1.0) -> float:
    """``PolygonRoi.getTracedPerimeter``: straight-edge pixels count 1, corner pixels √2."""
    xs = [int(v) for v in xp]
    ys = [int(v) for v in yp]
    n = len(xs)
    if n < 4:
        return 0.0
    sumdx = sumdy = 0
    n_corners = 0
    dx1 = xs[0] - xs[n - 1]
    dy1 = ys[0] - ys[n - 1]
    side1 = abs(dx1) + abs(dy1)
    corner = False
    for i in range(n):
        nexti = i + 1
        if nexti == n:
            nexti = 0
        dx2 = xs[nexti] - xs[i]
        dy2 = ys[nexti] - ys[i]
        sumdx += abs(dx1)
        sumdy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)
        if side1 > 1 or not corner:
            corner = True
            n_corners += 1
        else:
            corner = False
        dx1, dy1, side1 = dx2, dy2, side2
    return sumdx * pw + sumdy * ph - (n_corners * ((pw + ph) - math.sqrt(pw * pw + ph * ph)))


def is_traced(xs: Any, ys: Any) -> bool:
    """``PolygonRoi.isTraced`` for a POLYGON: integer vertices and only horizontal / vertical edges."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n = len(xs)
    if n == 0:
        return False
    for i in range(n):
        nexti = i + 1 if i + 1 < n else 0
        if not ((xs[nexti] - xs[i]) == 0 or (ys[nexti] - ys[i]) == 0):
            return False
        if xs[i] != int(xs[i]) or ys[i] != int(ys[i]):
            return False
    return True


def polygon_length(xs: Any, ys: Any, close: bool, pw: float = 1.0, ph: float = 1.0) -> float:
    """``PolygonRoi.getLength(points, close, imp)``: sum of the calibrated segment lengths."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n = len(xs)
    if n < 2:
        return 0.0
    length = 0.0
    for i in range(n - 1):
        length += math.sqrt(((xs[i + 1] - xs[i]) * pw) ** 2 + ((ys[i + 1] - ys[i]) * ph) ** 2)
    if close:
        length += math.sqrt(((xs[0] - xs[n - 1]) * pw) ** 2 + ((ys[0] - ys[n - 1]) * ph) ** 2)
    return length


def smoothed_perimeter(xs: Any, ys: Any, pw: float = 1.0, ph: float = 1.0) -> float:
    """``PolygonRoi.getSmoothedPerimeter`` (FREEROI): length after a 3-point running average, plus the closing segment."""
    xp = [float(v) for v in xs]
    yp = [float(v) for v in ys]
    n = len(xp)
    if n < 3:
        return polygon_length(xp, yp, True, pw, ph)
    w2, h2 = pw * pw, ph * ph
    dx = (xp[0] + xp[1] + xp[2]) / 3.0 - xp[0]
    dy = (yp[0] + yp[1] + yp[2]) / 3.0 - yp[0]
    length = math.sqrt(dx * dx * w2 + dy * dy * h2)
    for i in range(1, n - 2):
        dx = (xp[i + 2] - xp[i - 1]) / 3.0
        dy = (yp[i + 2] - yp[i - 1]) / 3.0
        length += math.sqrt(dx * dx * w2 + dy * dy * h2)
    dx = xp[n - 1] - (xp[n - 3] + xp[n - 2] + xp[n - 1]) / 3.0
    dy = yp[n - 1] - (yp[n - 3] + yp[n - 2] + yp[n - 1]) / 3.0
    length += math.sqrt(dx * dx * w2 + dy * dy * h2)
    dx = xp[n - 1] - xp[0]
    dy = yp[n - 1] - yp[0]
    length += math.sqrt(dx * dx * w2 + dy * dy * h2)
    return length


def line_float_polygon(x1: float, y1: float, x2: float, y2: float, stroke_width: float) -> tuple[np.ndarray, np.ndarray]:
    """``Line.getFloatPolygon(strokeWidth)``: the two end points, or the 4-point outline of a wide line."""
    if stroke_width <= 1:
        return np.asarray([x1, x2], F32), np.asarray([y1, y2], F32)
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    dx *= 0.5 / length
    dy *= 0.5 / length
    p1x = x1 + 0.5 - dx + dy * stroke_width
    p1y = y1 + 0.5 - dy - dx * stroke_width
    p2x = x1 + 0.5 - dx - dy * stroke_width
    p2y = y1 + 0.5 - dy + dx * stroke_width
    p3x = x2 + 0.5 + dx - dy * stroke_width
    p3y = y2 + 0.5 + dy + dx * stroke_width
    p4x = x2 + 0.5 + dx + dy * stroke_width
    p4y = y2 + 0.5 + dy - dx * stroke_width
    return np.asarray([p1x, p2x, p3x, p4x], F32), np.asarray([p1y, p2y, p3y, p4y], F32)


def rotated_rect_polygon(x1: float, y1: float, x2: float, y2: float, rect_width: float) -> tuple[np.ndarray, np.ndarray]:
    """``RotatedRectRoi.makeRectangle`` (``Roi.convertLineToArea`` of a straight line): 4 float32 vertices."""
    length = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    wsa = 0.5 * rect_width * (y2 - y1) / length
    wca = -0.5 * rect_width * (x2 - x1) / length
    xs = [x1 + wsa, x2 + wsa, x2 - wsa, x1 - wsa]
    ys = [y1 + wca, y2 + wca, y2 - wca, y1 - wca]
    return np.asarray(xs, F32), np.asarray(ys, F32)


def line_int_bounds(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    """``Roi.setIntBounds`` for a straight line (``Line.updateCoordinates``): the 'draw' area of the line."""
    bx, by = min(x1, x2), min(y1, y2)
    bw, bh = abs(x2 - x1), abs(y2 - y1)
    x = int(math.floor(bx + 0.5))
    y = int(math.floor(by + 0.5))
    return x, y, int(math.floor(bx + bw + 1.5)) - x, int(math.floor(by + bh + 1.5)) - y


def float_angle(x1: float, y1: float, x2: float, y2: float, pw: float = 1.0, ph: float = 1.0) -> float:
    """``Roi.getFloatAngle``: degrees counter-clockwise, y up, calibrated."""
    return (180.0 / math.pi) * math.atan2((y1 - y2) * ph, (x2 - x1) * pw)


# ====================================================================== getLine / interpolation
def _pixel_value(plane: np.ndarray, x: int, y: int) -> float:
    H, W = plane.shape
    if 0 <= x < W and 0 <= y < H:
        return float(plane[y, x])
    return float("nan")


def interpolated_value(plane: np.ndarray, x: float, y: float) -> float:
    """``ImageProcessor.getInterpolatedValue`` (bilinear; ImageJ's edge handling)."""
    H, W = plane.shape
    if x < 0.0 or x >= W - 1.0 or y < 0.0 or y >= H - 1.0:
        if x < -1.0 or x >= W or y < -1.0 or y >= H:
            return 0.0
        xbase, ybase = int(x), int(y)
        xf, yf = x - xbase, y - ybase
        if xf < 0.0:
            xf = 0.0
        if yf < 0.0:
            yf = 0.0

        def edge(ex: int, ey: int) -> float:
            ex = 0 if ex <= 0 else W - 1 if ex >= W else ex
            ey = 0 if ey <= 0 else H - 1 if ey >= H else ey
            return float(plane[ey, ex])
        ll, lr, ur, ul = edge(xbase, ybase), edge(xbase + 1, ybase), edge(xbase + 1, ybase + 1), edge(xbase, ybase + 1)
    else:
        xbase, ybase = int(x), int(y)
        xf, yf = x - xbase, y - ybase
        if xf < 0.0:
            xf = 0.0
        if yf < 0.0:
            yf = 0.0
        ll = float(plane[ybase, xbase])
        lr = float(plane[ybase, xbase + 1])
        ur = float(plane[ybase + 1, xbase + 1])
        ul = float(plane[ybase + 1, xbase])
    upper = ul + xf * (ur - ul)
    lower = ll + xf * (lr - ll)
    return lower + yf * (upper - lower)


def get_line(plane: np.ndarray, x1: float, y1: float, x2: float, y2: float, interpolate: bool = True) -> np.ndarray:
    """``ImageProcessor.getLine``: n = round(length) + 1 samples from (x1, y1) to (x2, y2) at the integer pixel
    coordinates (pixel centres; ImageJ's profile convention), bilinearly interpolated unless ``interpolate`` is off."""
    p = np.asarray(plane)
    if p.dtype == np.uint8 or p.dtype == np.uint16:
        p = p.astype(np.float32)   # getPixelValue returns float
    dx, dy = x2 - x1, y2 - y1
    n = jround(math.sqrt(dx * dx + dy * dy))
    xinc = dx / n if n > 0 else 0.0
    yinc = dy / n if n > 0 else 0.0
    data = np.empty(n + 1, dtype=np.float64)
    rx, ry = x1, y1
    for i in range(n + 1):
        if interpolate:
            data[i] = interpolated_value(p, rx, ry)
        else:
            data[i] = _pixel_value(p, jround(rx), jround(ry))
        rx += xinc
        ry += yinc
    return data


def irregular_profile(plane: np.ndarray, xs: Any, ys: Any, interpolate: bool = True) -> np.ndarray:
    """``ProfilePlot.getIrregularProfile`` (segmented line of width 1): samples every ~1 px along the polyline."""
    p = np.asarray(plane)
    if p.dtype == np.uint8 or p.dtype == np.uint16:
        p = p.astype(np.float32)
    xpoints = [float(v) for v in xs]
    ypoints = [float(v) for v in ys]
    n = len(xpoints)
    values: list[float] = []
    inc = 0.01
    lastx = lasty = 0.0
    x2, y2 = xpoints[0], ypoints[0]
    for i in range(1, n):
        x1, y1 = x2, y2
        x, y = x1, y1
        x2, y2 = xpoints[i], ypoints[i]
        dx, dy = x2 - x1, y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        if distance == 0:
            continue
        xinc = dx * inc / distance
        yinc = dy * inc / distance
        n2 = int(distance / inc)
        if n == 2:
            n2 += 1
        while True:
            ddx, ddy = x - lastx, y - lasty
            distance2 = math.sqrt(ddx * ddx + ddy * ddy)
            if distance2 >= 1.0 - inc / 2.0:
                values.append(interpolated_value(p, x, y) if interpolate else _pixel_value(p, jround(x), jround(y)))
                lastx, lasty = x, y
            x += xinc
            y += yinc
            n2 -= 1
            if n2 <= 0:
                break
    return np.asarray(values, dtype=np.float64)


# ====================================================================== EllipseFitter
@dataclass
class EllipseFit:
    major: float = 0.0
    minor: float = 0.0
    angle: float = 0.0
    theta: float = 0.0
    x_center: float = 0.0
    y_center: float = 0.0
    bit_count: int = 0


def ellipse_fitter(mask: np.ndarray | None, left: int, top: int, width: int, height: int) -> EllipseFit:
    """``EllipseFitter.fit`` / ``getEllipseParam``: the ellipse with the same second-order central moments as the
    mask (a uniform distribution over the pixel areas), scaled to the mask's area; ``mask`` None = the whole rectangle."""
    ef = EllipseFit()
    if mask is None:
        ef.major = (width * 2) / SQRT_PI
        ef.minor = (height * 2) / SQRT_PI
        ef.angle = 0.0
        ef.theta = 0.0
        if ef.major < ef.minor:
            ef.major, ef.minor = ef.minor, ef.major
            ef.angle = 90.0
            ef.theta = math.pi / 2.0
        ef.x_center = left + width / 2.0
        ef.y_center = top + height / 2.0
        ef.bit_count = int(width) * int(height)
        return ef
    m = np.asarray(mask, dtype=bool)
    ys_idx, xs_idx = np.nonzero(m)
    bit_count = int(xs_idx.size)
    ef.bit_count = bit_count
    if bit_count == 0:
        return ef
    xs64 = xs_idx.astype(np.int64)
    ys64 = ys_idx.astype(np.int64)
    # computeSums: every sum is a sum of integers (exact in double while < 2^53)
    xsum = float(int(xs64.sum()))
    ysum = float(int(ys64.sum()))
    x2sum = float(int((xs64 * xs64).sum()))
    y2sum = float(int((ys64 * ys64).sum()))
    xysum = float(int((xs64 * ys64).sum()))
    # getMoments
    x2sum += 0.08333333 * bit_count
    y2sum += 0.08333333 * bit_count
    n = float(bit_count)
    x1 = xsum / n
    y1 = ysum / n
    x2 = x2sum / n
    y2 = y2sum / n
    xy = xysum / n
    xm, ym = x1, y1
    u20 = x2 - (x1 * x1)
    u02 = y2 - (y1 * y1)
    u11 = xy - x1 * y1
    m4 = 4.0 * abs(u02 * u20 - u11 * u11)
    if m4 < 0.000001:
        m4 = 0.000001
    a11 = u02 / m4
    a12 = u11 / m4
    a22 = u20 / m4
    tmp = a11 - a22
    if tmp == 0.0:
        tmp = 0.000001
    theta = 0.5 * math.atan(2.0 * a12 / tmp)
    if theta < 0.0:
        theta += HALF_PI
    if a12 > 0.0:
        theta += HALF_PI
    elif a12 == 0.0:
        if a22 > a11:
            theta = 0.0
            a22, a11 = a11, a22
        elif a11 != a22:
            theta = HALF_PI
    tmp = math.sin(theta)
    if tmp == 0.0:
        tmp = 0.000001
    z = a12 * math.cos(theta) / tmp
    major = math.sqrt(1.0 / abs(a22 + z))
    minor = math.sqrt(1.0 / abs(a11 - z))
    scale = math.sqrt(bit_count / (math.pi * major * minor))
    major = major * scale * 2.0
    minor = minor * scale * 2.0
    angle = 180.0 * theta / math.pi
    if angle == 180.0:
        angle = 0.0
    if major < minor:
        major, minor = minor, major
    ef.major, ef.minor, ef.angle, ef.theta = major, minor, angle, theta
    ef.x_center = left + xm + 0.5
    ef.y_center = top + ym + 0.5
    return ef


def ellipse_outline_points(ef: EllipseFit, image_height: int) -> list[tuple[int, int]]:
    """``EllipseFitter.drawEllipse`` / ``makeRoi``: the integer vertex sequence ImageJ draws for the Ellipses output."""
    if ef.major == 0.0 and ef.minor == 0.0:
        return []
    xc = jround(ef.x_center)
    yc = jround(ef.y_center)
    max_y = int(image_height)
    sint, cost = math.sin(ef.theta), math.cos(ef.theta)
    rmajor2 = 1.0 / (ef.major / 2) ** 2
    rminor2 = 1.0 / (ef.minor / 2) ** 2
    g11 = rmajor2 * cost * cost + rminor2 * sint * sint
    g12 = (rmajor2 - rminor2) * sint * cost
    g22 = rmajor2 * sint * sint + rminor2 * cost * cost
    k1 = -g12 / g11
    k2 = (g12 * g12 - g11 * g22) / (g11 * g11)
    k3 = 1.0 / g11
    ymax = int(math.floor(math.sqrt(abs(k3 / k2))))
    if ymax > max_y:
        ymax = max_y
    if ymax < 1:
        ymax = 1
    ymin = -ymax
    txmin = [0] * (max(max_y, ymax + 1))
    txmax = [0] * (max(max_y, ymax + 1))
    for y in range(0, ymax + 1):
        j2 = math.sqrt(k2 * y * y + k3)
        j1 = k1 * y
        txmin[y] = jround(j1 - j2)
        txmax[y] = jround(j1 + j2)
    pts = [(xc + txmin[ymax - 1], yc + ymin)]
    for y in range(ymin, ymax):
        x = txmax[-y] if y < 0 else -txmin[y]
        pts.append((xc + x, yc + y))
    for y in range(ymax, ymin, -1):
        x = txmin[-y] if y < 0 else -txmax[y]
        pts.append((xc + x, yc + y))
    return pts


def ellipse_roi_polygon(x1: float, y1: float, x2: float, y2: float, aspect_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """``EllipseRoi.makeEllipse``: the 72 float32 vertices of ImageJ's ellipse selection with the major axis from
    (x1, y1) to (x2, y2) and the given minor/major ratio."""
    vertices = 72
    aspect_ratio = min(max(aspect_ratio, 0.0), 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    major = math.sqrt(dx * dx + dy * dy)
    minor = major * aspect_ratio
    phi_b = math.atan2(dy, dx)
    alpha = phi_b * 180.0 / math.pi
    xs, ys = [], []
    for i in range(vertices):
        degrees = i * 360.0 / vertices
        beta1 = degrees / 180.0 * math.pi
        dx = math.cos(beta1) * major / 2.0
        dy = math.sin(beta1) * minor / 2.0
        beta2 = math.atan2(dy, dx)
        rad = math.sqrt(dx * dx + dy * dy)
        beta3 = beta2 + alpha / 180.0 * math.pi
        dx2 = math.cos(beta3) * rad
        dy2 = math.sin(beta3) * rad
        xs.append(F32(center_x + dx2))
        ys.append(F32(center_y + dy2))
    return np.asarray(xs, F32), np.asarray(ys, F32)


def fit_ellipse_polygon(mask: np.ndarray | None, left: int, top: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Edit ▸ Selection ▸ Fit Ellipse (``Selection.fitEllipse``): the ``EllipseRoi`` built from the EllipseFitter's
    centre, axes and angle of the selection's mask (``mask`` None = a rectangle).  Returns float32 vertex arrays."""
    ef = ellipse_fitter(mask, left, top, width, height)
    dx = ef.major * math.cos(ef.angle / 180.0 * math.pi) / 2.0
    dy = -ef.major * math.sin(ef.angle / 180.0 * math.pi) / 2.0
    x1, x2 = ef.x_center - dx, ef.x_center + dx
    y1, y2 = ef.y_center - dy, ef.y_center + dy
    aspect = ef.minor / ef.major if ef.major else 0.0
    return ellipse_roi_polygon(x1, y1, x2, y2, aspect)


def draw_polyline(target: np.ndarray, pts: list[tuple[int, int]], value: int, close: bool = False) -> None:
    """``ImageProcessor.moveTo`` / ``lineTo`` with line width 1: the pixels ImageJ sets for an integer polyline."""
    H, W = target.shape
    if not pts:
        return
    seq = list(pts) + ([pts[0]] if close else [])
    cx, cy = seq[0]
    for (x2, y2) in seq[1:]:
        dx, dy = x2 - cx, y2 - cy
        n = max(abs(dx), abs(dy))
        xinc = dx / n if dx != 0 else 0.0
        yinc = dy / n if dy != 0 else 0.0
        x, y = float(cx), float(cy)
        for _ in range(n + 1):
            px, py = jround(x), jround(y)
            if 0 <= px < W and 0 <= py < H:
                target[py, px] = value
            x += xinc
            y += yinc
        cx, cy = x2, y2


# ====================================================================== ImageStatistics
@dataclass
class IJStats:
    """The ``ij.process.ImageStatistics`` fields a Results row needs (uncalibrated pixel values, calibrated areas)."""
    pixel_count: int = 0
    area: float = 0.0
    mean: float = float("nan")
    umean: float = float("nan")
    std_dev: float = 0.0
    dmode: float = 0.0
    min: float = float("nan")
    max: float = float("nan")
    median: float = float("nan")
    skewness: float = 0.0
    kurtosis: float = 0.0
    x_centroid: float = 0.0
    y_centroid: float = 0.0
    x_center_of_mass: float = 0.0
    y_center_of_mass: float = 0.0
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_width: float = 0.0
    roi_height: float = 0.0
    major: float = 0.0
    minor: float = 0.0
    angle: float = 0.0
    area_fraction: float = 0.0
    lower_threshold: float = float("nan")
    upper_threshold: float = float("nan")
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    hist_min: float = 0.0
    hist_max: float = 0.0
    bin_size: float = 1.0
    n_bins: int = 256
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class Cal:
    """The calibration ``ImageStatistics`` / ``Analyzer`` consult: pixel size and ImageJ's 'Invert Y coordinates'."""
    pw: float = 1.0
    ph: float = 1.0
    invert_y: bool = False
    image_height: int = 0

    def get_x(self, x: float) -> float:
        return x * self.pw

    def get_y(self, y: float) -> float:
        """``Calibration.getY(y, imageHeight)``: (imageHeight − y − 1)·ph with 'Invert Y coordinates'."""
        if self.invert_y:
            return (self.image_height - y - 1) * self.ph
        return y * self.ph


def clip_rect_mask(rect: tuple[int, int, int, int], mask: np.ndarray | None, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray | None]:
    """``ImageProcessor.setRoi(x, y, w, h)``: intersect the ROI rectangle with the image and crop the mask alike."""
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), None
    if mask is not None and (x0 != x or y0 != y or x1 - x0 != w or y1 - y0 != h):
        mask = np.asarray(mask, bool)[y0 - y:y1 - y, x0 - x:x1 - x]
    return (x0, y0, x1 - x0, y1 - y0), mask


def _std_dev(n: float, s: float, s2: float) -> float:
    """``ImageStatistics.calculateStdDev``."""
    if n > 0.0:
        sd = (n * s2 - s * s) / n
        return math.sqrt(sd / (n - 1.0)) if sd > 0.0 else 0.0
    return 0.0


def _hist_median(hist: np.ndarray, first: int, last: int, pixel_count: int) -> float:
    """``ImageStatistics.calculateMedian``: the first bin at which the cumulative count exceeds half the pixels."""
    if pixel_count == 0 or first < 0 or last > hist.size:
        return float("nan")
    s = 0.0
    i = first - 1
    half = pixel_count / 2.0
    while True:
        i += 1
        s += float(hist[i]) if i < hist.size else 0.0
        if not (s <= half and i < last):
            break
    return float(i)


def _moments(xs: np.ndarray, ys: np.ndarray, v: np.ndarray, mean: float, pixel_count: int, cal: Cal) -> tuple[float, float, float, float]:
    """``Byte/Short/FloatStatistics.calculateMoments``: skewness, kurtosis, centre of mass (values ``v`` as doubles)."""
    v = v.astype(np.float64)
    v2 = v * v
    sum1 = float(v.sum())
    sum2 = float(v2.sum())
    sum3 = float((v * v2).sum())
    sum4 = float((v2 * v2).sum())
    xsum = float((xs * v).sum())
    ysum = float((ys * v).sum())
    mean2 = mean * mean
    variance = sum2 / pixel_count - mean2 if pixel_count else float("nan")
    s_dev = math.sqrt(variance) if variance >= 0 else float("nan")
    with np.errstate(all="ignore"):
        skew = ((sum3 - 3.0 * mean * sum2) / pixel_count + 2.0 * mean * mean2) / (variance * s_dev) if pixel_count and variance * s_dev != 0 else float("nan")
        kurt = (((sum4 - 4.0 * mean * sum3 + 6.0 * mean2 * sum2) / pixel_count - 3.0 * mean2 * mean2) / (variance * variance) - 3.0) if pixel_count and variance != 0 else float("nan")
    xcm = cal.get_x(xsum / sum1 + 0.5) if sum1 != 0 else float("nan")
    ycm = cal.get_y(ysum / sum1 + 0.5) if sum1 != 0 else float("nan")
    return float(skew), float(kurt), xcm, ycm


def image_statistics(plane: np.ndarray, rect: tuple[int, int, int, int] | None = None, mask: np.ndarray | None = None, *,
                     cal: Cal | None = None, limit: tuple[float, float] | None = None, threshold: tuple[float, float] | None = None,
                     ellipse: bool = False, centroid: bool = False, moments: bool = False, median: bool = False, area_fraction: bool = False,
                     n_bins: int = 256, hist_min: float = 0.0, hist_max: float = 0.0) -> IJStats:
    """``ImageStatistics.getStatistics(ip, mOptions, cal)`` for a uint8 (``ByteStatistics``), uint16
    (``ShortStatistics``) or float (``FloatStatistics``) plane.

    ``rect`` is the ROI rectangle (image coordinates, clipped here), ``mask`` the rect-sized boolean mask (None =
    rectangular ROI), ``limit`` the threshold when 'Limit to threshold' is on, ``threshold`` the image threshold used by
    the area fraction, ``n_bins`` / ``hist_min`` / ``hist_max`` the histogram options of ``ImagePlus.getStatistics``."""
    cal = cal or Cal(image_height=plane.shape[0])
    H, W = plane.shape
    st = IJStats()
    if rect is None:
        rect = (0, 0, W, H)
    rect, mask = clip_rect_mask(rect, mask, W, H)
    rx, ry, rw, rh = rect
    st.rect = rect
    st.roi_x = cal.get_x(rx)
    st.roi_y = cal.get_y(ry)
    st.roi_width = rw * cal.pw
    st.roi_height = rh * cal.ph
    sub = plane[ry:ry + rh, rx:rx + rw]
    m = np.ones(sub.shape, bool) if mask is None else np.asarray(mask, bool)
    kind = "byte" if plane.dtype == np.uint8 else "short" if plane.dtype == np.uint16 else "float"
    pw_ph = cal.pw * cal.ph
    yy, xx = np.nonzero(m)
    vals = sub[yy, xx]
    xs_abs = xx.astype(np.float64) + rx
    ys_abs = yy.astype(np.float64) + ry
    st.n_bins = n_bins

    if kind in ("byte", "short"):
        top = 255 if kind == "byte" else 65535
        if limit is None:
            min_t, max_t = 0, top
        else:
            min_t, max_t = int(limit[0]), int(limit[1])
            st.lower_threshold, st.upper_threshold = float(min_t), float(max_t)
        counts = np.bincount(vals.astype(np.int64), minlength=top + 1) if vals.size else np.zeros(top + 1, np.int64)
        hist_full = counts.astype(np.int64)
        if kind == "byte":
            st.histogram = hist_full[:256].copy()
            rng_ = np.arange(min_t, max_t + 1)
            c = hist_full[min_t:max_t + 1]
            n_px = int(c.sum())
            s = float(int((rng_ * c).sum()))
            s2 = float(int((rng_ * rng_ * c).sum()))
            st.pixel_count = n_px
            st.area = n_px * pw_ph
            st.mean = s / n_px if n_px else float("nan")
            st.umean = st.mean
            st.dmode = float(min_t + int(np.argmax(c))) if c.size and c.max() > 0 else 0.0
            st.std_dev = _std_dev(n_px, s, s2)
            st.hist_min, st.hist_max = 0.0, 255.0
            # getRawMinAndMax
            mn = min_t
            while hist_full[mn] == 0 and mn < 255:
                mn += 1
            mx = max_t
            while hist_full[mx] == 0 and mx > 0:
                mx -= 1
            st.min, st.max = float(mn), float(mx)
            if n_px == 0:
                st.min = st.max = float("nan")
            lim_sel = (vals >= min_t) & (vals <= max_t)
        else:
            max_t = min(max_t, hist_full.size - 1)
            mn = min_t
            if mn < hist_full.size:
                while hist_full[mn] == 0 and mn < hist_full.size - 1:
                    mn += 1
            mx = max_t
            while hist_full[mx] == 0 and mx > 0:
                mx -= 1
            st.min, st.max = float(mn), float(mx)
            hmin, hmax = float(mn), float(mx)
            if not (hist_min == 0.0 and hist_max == 0.0):
                hmin, hmax = float(hist_min), float(hist_max)
                if mn < hmin:
                    mn = int(hmin)
                if mx > hmax:
                    mx = int(hmax)
            st.hist_min, st.hist_max = hmin, hmax
            st.bin_size = (hmax - hmin + 1) / n_bins
            scale = 1.0 / st.bin_size
            h_min_i = int(hmin)
            rng_ = np.arange(mn, mx + 1) if mx >= mn else np.zeros(0, np.int64)
            c = hist_full[mn:mx + 1] if mx >= mn else np.zeros(0, np.int64)
            n_px = int(c.sum())
            s = float(int((rng_ * c).sum()))
            s2 = float(int((rng_ * rng_ * c).sum()))
            st.pixel_count = n_px
            st.area = n_px * pw_ph
            st.mean = s / n_px if n_px else float("nan")
            st.umean = st.mean
            st.dmode = float(mn + int(np.argmax(c))) if c.size and c.max() > 0 else 0.0
            st.std_dev = _std_dev(n_px, s, s2)
            idx = (scale * (rng_ - h_min_i)).astype(np.int64)
            idx = np.minimum(idx, n_bins - 1)
            st.histogram = np.bincount(idx, weights=c, minlength=n_bins)[:n_bins].astype(np.int64) if c.size else np.zeros(n_bins, np.int64)
            lim_sel = (vals >= min_t) & (vals <= max_t)
        if ellipse:
            emask = m if mask is not None else None
            if limit is not None:
                emask = m & (sub >= min_t) & (sub <= max_t)
            ef = ellipse_fitter(emask, rx, ry, rw, rh)
            psize = cal.pw if (abs(cal.pw - cal.ph) / cal.pw) < .01 else 0.0
            st.major, st.minor, st.angle = ef.major * psize, ef.minor * psize, ef.angle
            st.x_centroid = cal.get_x(ef.x_center)
            st.y_centroid = cal.get_y(ef.y_center)
        elif centroid:
            sel = lim_sel if (min_t > 0 or max_t < top) else np.ones(vals.size, bool)
            cnt = int(sel.sum())
            st.x_centroid = cal.get_x(float(xs_abs[sel].sum()) / cnt + 0.5) if cnt else float("nan")
            st.y_centroid = cal.get_y(float(ys_abs[sel].sum()) / cnt + 0.5) if cnt else float("nan")
        if moments:
            sel = lim_sel
            st.skewness, st.kurtosis, st.x_center_of_mass, st.y_center_of_mass = _moments(xs_abs[sel], ys_abs[sel], vals[sel], st.mean, st.pixel_count, cal)
        if median:
            st.median = _hist_median(hist_full, min_t, max_t, st.pixel_count) if st.pixel_count > 0 else float("nan")
        if area_fraction:
            total = int(hist_full.sum())
            if threshold is None:
                summed = total - int(hist_full[0])
            else:
                t1, t2 = jround(threshold[0]), jround(threshold[1])
                summed = int(hist_full[max(t1, 0):max(min(t2, hist_full.size - 1) + 1, 0)].sum())
            st.area_fraction = summed * 100.0 / total if total else float("nan")
        return st

    # ---- FloatStatistics
    v64 = vals.astype(np.float64)
    if limit is None:
        min_t, max_t = -float(np.finfo(F32).max), float(np.finfo(F32).max)
    else:
        min_t, max_t = float(limit[0]), float(limit[1])
        st.lower_threshold, st.upper_threshold = min_t, max_t
    in_lim = (v64 >= min_t) & (v64 <= max_t)
    vl = v64[in_lim]
    if vl.size:
        roi_min, roi_max = float(vl.min()), float(vl.max())
    else:
        roi_min, roi_max = float(np.finfo(np.float64).max), -float(np.finfo(np.float64).max)
    st.min, st.max = roi_min, roi_max
    if hist_min == 0.0 and hist_max == 0.0:
        hmin, hmax = st.min, st.max
    else:
        hmin, hmax = float(hist_min), float(hist_max)
        if st.min < hmin:
            st.min = hmin
        if st.max > hmax:
            st.max = hmax
    st.hist_min, st.hist_max = hmin, hmax
    st.bin_size = (hmax - hmin) / n_bins
    scale = n_bins / (hmax - hmin) if hmax != hmin else float("inf")
    sel = in_lim & (v64 >= hmin) & (v64 <= hmax)
    vs = v64[sel]
    n_px = int(vs.size)
    st.pixel_count = n_px
    st.area = n_px * pw_ph
    s = float(vs.sum())
    s2 = float((vs * vs).sum())
    st.mean = s / n_px if n_px else float("nan")
    st.umean = st.mean
    st.std_dev = _std_dev(n_px, s, s2)
    if n_px:
        with np.errstate(invalid="ignore", over="ignore"):
            idx = (scale * (vs - hmin))
        idx = np.where(np.isfinite(idx), idx, 0).astype(np.int64)
        idx = np.minimum(idx, n_bins - 1)
        st.histogram = np.bincount(idx, minlength=n_bins)[:n_bins].astype(np.int64)
    else:
        st.histogram = np.zeros(n_bins, np.int64)
    # getMode
    mode = int(np.argmax(st.histogram)) if st.histogram.size and st.histogram.max() > 0 else 0
    st.dmode = hmin + mode * st.bin_size
    if st.bin_size != 1.0:
        st.dmode += st.bin_size / 2.0
    if ellipse:
        emask = m if mask is not None else None
        if limit is not None:
            emask = m & (sub >= min_t) & (sub <= max_t)
        ef = ellipse_fitter(emask, rx, ry, rw, rh)
        psize = cal.pw if (abs(cal.pw - cal.ph) / cal.pw) < .01 else 0.0
        st.major, st.minor, st.angle = ef.major * psize, ef.minor * psize, ef.angle
        st.x_centroid = cal.get_x(ef.x_center)
        st.y_centroid = cal.get_y(ef.y_center)
    elif centroid:
        cnt = int(in_lim.sum())
        st.x_centroid = cal.get_x(float(xs_abs[in_lim].sum()) / cnt + 0.5) if cnt else float("nan")
        st.y_centroid = cal.get_y(float(ys_abs[in_lim].sum()) / cnt + 0.5) if cnt else float("nan")
    if moments:
        st.skewness, st.kurtosis, st.x_center_of_mass, st.y_center_of_mass = _moments(xs_abs[in_lim], ys_abs[in_lim], v64[in_lim], st.mean, st.pixel_count, cal)
    if median:
        if n_px == 0:
            st.median = float("nan")
        else:
            srt = np.sort(vals[in_lim].astype(F32))
            middle = srt.size // 2
            if srt.size % 2 == 0:
                st.median = float((srt[middle - 1] + srt[middle]) / F32(2))   # float arithmetic, as ImageJ
            else:
                st.median = float(srt[middle])
    if area_fraction:
        total = int(v64.size)
        if threshold is None:
            summed = int(np.count_nonzero(v64 != 0.0))
        else:
            t1, t2 = _f32(threshold[0]), _f32(threshold[1])
            summed = int(np.count_nonzero((v64 >= t1) & (v64 <= t2)))
        st.area_fraction = summed * 100.0 / total if total else float("nan")
    return st


def macro_histogram(plane: np.ndarray, rect: tuple[int, int, int, int] | None, mask: np.ndarray | None, n_bins: int,
                    hist_min: float | None = None, hist_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The macro function ``getHistogram(values, counts, nBins[, histMin, histMax])`` (``Functions.getHistogram``):
    8-bit with 256 bins = the raw histogram; otherwise ``ImagePlus.getStatistics(…, nBins, histMin, histMax)``."""
    p = np.asarray(plane)
    H, W = p.shape
    if rect is None:
        rect = (0, 0, W, H)
    if p.dtype == np.uint16 and n_bins == 65536:
        rect2, mask2 = clip_rect_mask(rect, mask, W, H)
        sub = p[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]
        v = sub[np.asarray(mask2, bool)] if mask2 is not None else sub.ravel()
        return np.arange(65536, dtype=np.float64), np.bincount(v.astype(np.int64), minlength=65536)[:65536]
    custom8 = False
    if p.dtype == np.uint8 and n_bins != 256:
        p = p.astype(np.uint16)
        hist_min, hist_max = 0.0, 256.0
        custom8 = True
    if p.dtype == np.uint8:
        st = image_statistics(p, rect, mask)
        return np.arange(256, dtype=np.float64), st.histogram.astype(np.int64)
    hmin = 0.0 if hist_min is None else float(hist_min)
    hmax = 0.0 if hist_max is None else float(hist_max)
    st = image_statistics(p, rect, mask, n_bins=int(n_bins), hist_min=hmin, hist_max=hmax)
    value = st.hist_min
    inc = (st.hist_max - st.hist_min) / st.n_bins if (p.dtype == np.uint16 or p.dtype.kind == "f" or custom8) else 1.0
    values = np.empty(n_bins, np.float64)
    for i in range(n_bins):
        values[i] = value
        value += inc
    return values, st.histogram.astype(np.int64)


# ====================================================================== MaximumFinder
MAXIMUM, LISTED, PROCESSED, MAX_AREA, EQUAL, MAX_POINT, ELIMINATED = 1, 2, 4, 8, 16, 32, 64
_DIR_X = (0, 1, 1, 1, 0, -1, -1, -1)
_DIR_Y = (-1, -1, 0, 1, 1, 1, 0, -1)
OUT_SINGLE_POINTS, OUT_IN_TOLERANCE, OUT_SEGMENTED, OUT_POINT_SELECTION, OUT_LIST, OUT_COUNT = 0, 1, 2, 3, 4, 5
_OUTPUT_TYPE_MASKS = {OUT_SINGLE_POINTS: MAX_POINT, OUT_IN_TOLERANCE: MAX_AREA, OUT_SEGMENTED: MAX_AREA}


def invert_for_minima(plane: np.ndarray, rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """``ip.invert()`` as Find Maxima applies it for 'light background': 255 − v for 8-bit, (min + max) − v for 16-bit
    (image min / max) and float (ROI min / max), inverted inside the ROI rectangle only."""
    p = np.asarray(plane)
    H, W = p.shape
    x, y, w, h = rect if rect is not None else (0, 0, W, H)
    out = p.copy()
    sub = p[y:y + h, x:x + w]
    if p.dtype == np.uint8:
        out[y:y + h, x:x + w] = 255 - sub
    elif p.dtype == np.uint16:
        mn, mx = int(p.min()), int(p.max())
        out[y:y + h, x:x + w] = np.clip(mx - (sub.astype(np.int64) - mn), 0, 65535).astype(np.uint16)
    else:
        f = sub.astype(np.float64)
        mn, mx = float(np.nanmin(f)), float(np.nanmax(f))
        out[y:y + h, x:x + w] = (mx - (f - mn)).astype(p.dtype)
    return out


@dataclass
class MaximaResult:
    points: list[tuple[int, int]]          # (x, y) in processing order (highest maximum first)
    types: np.ndarray                      # the pixel-type notepad (uint8)
    output: np.ndarray | None              # 8-bit output image (255 = marked) for the image outputs


def find_maxima(plane: np.ndarray, tolerance: float, strict: bool = False, threshold: float | None = None, output: int = OUT_LIST,
                exclude_on_edges: bool = False, rect: tuple[int, int, int, int] | None = None, mask: np.ndarray | None = None) -> MaximaResult:
    """``MaximumFinder.findMaxima(ip, tolerance, strict, threshold, outputType, excludeOnEdges, isEDM=false)``.

    ``plane`` is the (possibly inverted) image, ``rect`` / ``mask`` the ROI; ``threshold`` None = no minimum height."""
    p = np.asarray(plane)
    H, W = p.shape
    vals = p.astype(F32)                      # ip.getPixelValue → float
    rx, ry, rw, rh = rect if rect is not None else (0, 0, W, H)
    sub = vals[ry:ry + rh, rx:rx + rw]
    if sub.size == 0:
        return MaximaResult([], np.zeros((H, W), np.uint8), None)
    global_min = F32(np.nanmin(sub))
    global_max = F32(np.nanmax(sub))
    maximum_possible = bool(global_max > global_min)
    if strict and float(F32(global_max - global_min)) <= tolerance:
        maximum_possible = False
    thr = None if threshold is None else float(threshold) - float(F32(global_max - global_min)) * 1e-6
    exclude_edges_now = exclude_on_edges and output != OUT_SEGMENTED
    types = np.zeros(H * W, np.uint8)
    flat = vals.ravel()
    max_points: list[int] = []
    if maximum_possible:
        max_points = _sorted_max_points(vals, types, (rx, ry, rw, rh), exclude_edges_now, global_min, global_max, thr)
    max_sorting_error = F32(0)
    if p.dtype.kind == "f":
        max_sorting_error = F32(F32(1.1) * F32(F32(global_max - global_min) / F32(2e9)))
    points = _analyze_and_mark_maxima(flat, W, H, types, max_points, exclude_edges_now, global_min, tolerance, strict, output,
                                      max_sorting_error, exclude_on_edges, mask, (rx, ry, rw, rh))
    types2 = types.reshape(H, W)
    if output in (OUT_POINT_SELECTION, OUT_LIST, OUT_COUNT):
        return MaximaResult(points, types2, None)
    if output == OUT_SEGMENTED:
        out = _make_8bit(flat, W, H, types, global_min, global_max, thr)
        _cleanup_maxima(out, types, max_points, W, H)
        _watershed_segment(out, W, H)
        _cleanup_extra_lines(out, W, H)
        out[out < 255] = 0                                   # watershedPostProcess
        if exclude_on_edges:
            _delete_edge_particles(out, types, W, H)
        out2 = out.reshape(H, W)
    else:
        out2 = np.where((types2 & _OUTPUT_TYPE_MASKS[output]) != 0, 255, 0).astype(np.uint8)
    if rect is not None:
        keep = np.zeros((H, W), bool)
        keep[ry:ry + rh, rx:rx + rw] = True if mask is None else np.asarray(mask, bool)
        out2 = np.where(keep, out2, 0).astype(np.uint8)
    return MaximaResult(points, types2, out2)


def _sorted_max_points(vals: np.ndarray, types: np.ndarray, rect: tuple[int, int, int, int], exclude_edges_now: bool,
                       global_min: Any, global_max: Any, threshold: float | None) -> list[int]:
    """``MaximumFinder.getSortedMaxPoints``: every local maximum, sorted by value (ties by pixel offset)."""
    H, W = vals.shape
    rx, ry, rw, rh = rect
    padded = np.full((H + 2, W + 2), -np.inf, np.float64)
    padded[1:-1, 1:-1] = vals
    v = padded[1:-1, 1:-1]
    is_max = np.ones((H, W), bool)
    for dx, dy in zip(_DIR_X, _DIR_Y):
        nb = padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        is_max &= ~(nb > v)
    is_max &= vals != global_min
    if exclude_edges_now:
        is_max[0, :] = is_max[-1, :] = False
        is_max[:, 0] = is_max[:, -1] = False
    if threshold is not None:
        is_max &= ~(vals < threshold)
    roi_sel = np.zeros((H, W), bool)
    roi_sel[ry:ry + rh, rx:rx + rw] = True
    is_max &= roi_sel
    is_max &= np.isfinite(vals)
    ys, xs = np.nonzero(is_max)
    offsets = (xs + ys * W).astype(np.int64)
    types[offsets] = MAXIMUM
    v_factor = F32(2e9 / float(F32(global_max - global_min)))
    fvals = vals[ys, xs].astype(F32)
    ivalue = ((fvals - F32(global_min)) * v_factor).astype(np.int64)
    order = np.lexsort((offsets, ivalue))
    return [int(o) for o in offsets[order]]


def _analyze_and_mark_maxima(flat: np.ndarray, W: int, H: int, types: np.ndarray, max_points: list[int], exclude_edges_now: bool,
                             global_min: Any, tolerance: float, strict: bool, output: int, max_sorting_error: Any, exclude_on_edges: bool,
                             roi_mask: np.ndarray | None, rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """``MaximumFinder.analyzeAndMarkMaxima``: flood every maximum within the tolerance, decide whether it is a true
    maximum, mark MAX_AREA / MAX_POINT and collect the single points."""
    v = flat.astype(np.float64).tolist()
    ty = types.tolist()
    tol32 = F32(tolerance)
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    display_or_count = output in (OUT_POINT_SELECTION, OUT_LIST, OUT_COUNT)
    points: list[tuple[int, int]] = []
    rx, ry, rw, rh = rect
    n_max = len(max_points)
    p_list = [0] * (W * H)
    wm1, hm1 = W - 1, H - 1

    def within(x: int, y: int, d: int) -> bool:
        if d == 0:
            return y > 0
        if d == 1:
            return x < wm1 and y > 0
        if d == 2:
            return x < wm1
        if d == 3:
            return x < wm1 and y < hm1
        if d == 4:
            return y < hm1
        if d == 5:
            return x > 0 and y < hm1
        if d == 6:
            return x > 0
        return x > 0 and y > 0

    for i_max in range(n_max - 1, -1, -1):
        offset0 = max_points[i_max]
        if ty[offset0] & PROCESSED:
            continue
        x0 = offset0 % W
        y0 = offset0 // W
        v0 = v[offset0]
        while True:                                         # repeat on sorting errors
            p_list[0] = offset0
            ty[offset0] |= (EQUAL | LISTED)
            list_len = 1
            list_i = 0
            is_edge_maximum = (x0 == 0 or x0 == wm1 or y0 == 0 or y0 == hm1)
            sorting_error = False
            max_possible = True
            x_equal = float(x0)
            y_equal = float(y0)
            n_equal = 1
            v0_hi = float(F32(F32(v0) + max_sorting_error))
            v0_lo = float(F32(F32(v0) - tol32))
            while True:
                offset = p_list[list_i]
                x = offset % W
                y = offset // W
                is_inner = (y != 0 and y != hm1) and (x != 0 and x != wm1)
                for d in range(8):
                    offset2 = offset + dir_off[d]
                    if (is_inner or within(x, y, d)) and not (ty[offset2] & LISTED):
                        if ty[offset2] & PROCESSED:
                            max_possible = False
                            break
                        x2 = x + _DIR_X[d]
                        y2 = y + _DIR_Y[d]
                        v2 = v[offset2]
                        if v2 > v0_hi:
                            max_possible = False
                            break
                        elif v2 >= v0_lo:
                            if v2 > v0:
                                sorting_error = True
                                offset0 = offset2
                                v0 = v2
                                x0, y0 = x2, y2
                                v0_hi = float(F32(F32(v0) + max_sorting_error))
                                v0_lo = float(F32(F32(v0) - tol32))
                            p_list[list_len] = offset2
                            list_len += 1
                            ty[offset2] |= LISTED
                            if (x2 == 0 or x2 == wm1 or y2 == 0 or y2 == hm1) and (strict or v2 >= v0):
                                is_edge_maximum = True
                                if exclude_edges_now:
                                    max_possible = False
                                    break
                            if v2 == v0:
                                ty[offset2] |= EQUAL
                                x_equal += x2
                                y_equal += y2
                                n_equal += 1
                list_i += 1
                if not list_i < list_len:
                    break
            if sorting_error:
                for k in range(list_len):
                    ty[p_list[k]] = 0
            else:
                reset_mask = ~(LISTED if max_possible else (LISTED | EQUAL)) & 0xFF
                x_equal /= n_equal
                y_equal /= n_equal
                min_dist2 = 1e20
                nearest_i = 0
                for k in range(list_len):
                    offset = p_list[k]
                    x = offset % W
                    y = offset // W
                    ty[offset] &= reset_mask
                    ty[offset] |= PROCESSED
                    if max_possible:
                        ty[offset] |= MAX_AREA
                        if ty[offset] & EQUAL:
                            dist2 = (x_equal - x) * (x_equal - x) + (y_equal - y) * (y_equal - y)
                            if dist2 < min_dist2:
                                min_dist2 = dist2
                                nearest_i = k
                if max_possible:
                    offset = p_list[nearest_i]
                    ty[offset] |= MAX_POINT
                    if display_or_count and not (exclude_on_edges and is_edge_maximum):
                        x = offset % W
                        y = offset // W
                        inside = rx <= x < rx + rw and ry <= y < ry + rh
                        if inside and roi_mask is not None:
                            inside = bool(roi_mask[y - ry, x - rx]) if roi_mask.shape == (rh, rw) else bool(roi_mask[y, x])
                        if inside:
                            points.append((x, y))
            if not sorting_error:
                break
    if n_max == 0:
        for k in range(W * H):
            ty[k] = PROCESSED | MAX_AREA
    types[:] = np.asarray(ty, np.uint8)
    return points


def _make_8bit(flat: np.ndarray, W: int, H: int, types: np.ndarray, global_min: Any, global_max: Any, threshold: float | None) -> np.ndarray:
    """``MaximumFinder.make8bit``: scale to 1–254, maximum areas 255, below the threshold 0."""
    gmin, gmax = float(global_min), float(global_max)
    min_value = gmin if threshold is None else threshold
    offset = min_value - (gmax - min_value) * (1. / 253 / 2 - 1e-6)
    factor = 253 / (gmax - min_value)
    raw = flat.astype(np.float64)
    v = 1 + np.floor((raw - offset) * factor + 0.5)
    out = np.where(v < 1, 1, np.where(v <= 254, v, 254)).astype(np.uint8)
    out[(types & MAX_AREA) != 0] = 255
    if threshold is not None:
        out[raw < threshold] = 0
    return out


def _cleanup_maxima(pixels: np.ndarray, types: np.ndarray, max_points: list[int], W: int, H: int) -> None:
    """``MaximumFinder.cleanupMaxima``: flatten every unmarked maximum to the level of the saddle towards a true one."""
    px = pixels.tolist()
    ty = types.tolist()
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    wm1, hm1 = W - 1, H - 1
    p_list = [0] * (W * H)

    def within(x: int, y: int, d: int) -> bool:
        if d == 0:
            return y > 0
        if d == 1:
            return x < wm1 and y > 0
        if d == 2:
            return x < wm1
        if d == 3:
            return x < wm1 and y < hm1
        if d == 4:
            return y < hm1
        if d == 5:
            return x > 0 and y < hm1
        if d == 6:
            return x > 0
        return x > 0 and y > 0

    for i_max in range(len(max_points) - 1, -1, -1):
        offset0 = max_points[i_max]
        if ty[offset0] & (MAX_AREA | ELIMINATED):
            continue
        level = px[offset0]
        lo_level = level + 1
        p_list[0] = offset0
        ty[offset0] |= LISTED
        list_len = 1
        last_len = 1
        saddle_found = False
        while not saddle_found and lo_level > 0:
            lo_level -= 1
            last_len = list_len
            list_i = 0
            while True:
                offset = p_list[list_i]
                x = offset % W
                y = offset // W
                is_inner = (y != 0 and y != hm1) and (x != 0 and x != wm1)
                for d in range(8):
                    offset2 = offset + dir_off[d]
                    if (is_inner or within(x, y, d)) and not (ty[offset2] & LISTED):
                        if (ty[offset2] & MAX_AREA) or ((ty[offset2] & ELIMINATED) and px[offset2] >= lo_level):
                            saddle_found = True
                            break
                        elif px[offset2] >= lo_level and not (ty[offset2] & ELIMINATED):
                            p_list[list_len] = offset2
                            list_len += 1
                            ty[offset2] |= LISTED
                if saddle_found:
                    break
                list_i += 1
                if not list_i < list_len:
                    break
        for k in range(list_len):
            ty[p_list[k]] &= ~LISTED & 0xFF
        for k in range(last_len):
            offset = p_list[k]
            px[offset] = lo_level
            ty[offset] |= ELIMINATED
    pixels[:] = np.asarray(px, np.uint8)
    types[:] = np.asarray(ty, np.uint8)


def _make_fate_table() -> np.ndarray:
    """``MaximumFinder.makeFateTable``."""
    table = np.zeros(256, np.int64)
    for item in range(256):
        is_set = [bool(item & (1 << i)) for i in range(8)]
        for i in range(8):
            if is_set[(i + 4) % 8]:
                table[item] |= 1 << i
        for i in range(0, 8, 2):
            if is_set[i]:
                is_set[(i + 1) % 8] = True
                is_set[(i + 7) % 8] = True
        transitions = sum(1 for i in range(8) if is_set[i] != is_set[(i + 1) % 8])
        if transitions >= 4:
            table[item] = 0
    return table


def _watershed_segment(pixels: np.ndarray, W: int, H: int) -> None:
    """``MaximumFinder.watershedSegment``: dilate the maxima (255) level by level, constrained by the fate table."""
    hist = np.bincount(pixels, minlength=256)[:256].astype(np.int64)
    coords: dict[int, np.ndarray] = {}
    ys_all, xs_all = np.divmod(np.arange(W * H), W)
    for lv in range(1, 255):
        if hist[lv] > 0:
            sel = np.flatnonzero(pixels == lv)           # raster order (y, then x)
            coords[lv] = sel.astype(np.int64)
        else:
            coords[lv] = np.zeros(0, np.int64)
    highest = max((lv for lv in range(1, 255) if hist[lv] > 0), default=0)
    table = _make_fate_table()
    direction_sequence = (7, 3, 1, 5, 0, 4, 2, 6)
    for level in range(highest, 0, -1):
        remaining = int(hist[level])
        idle = 0
        while remaining > 0 and idle < 8:
            d_index = 0
            while True:
                n = _process_level(direction_sequence[d_index % 8], pixels, table, coords, level, W, H)
                remaining -= n
                if n > 0:
                    idle = 0
                d_index += 1
                if not remaining > 0:
                    break
                if not idle < 8:
                    idle += 1
                    break
                idle += 1
        if remaining > 0 and level > 1:
            next_level = level
            while True:
                next_level -= 1
                if not (next_level > 1 and hist[next_level] == 0):
                    break
            if next_level > 0:
                rem = coords[level][:remaining]
                xs = rem % W
                ys = rem // W
                add = (xs == 0) | (ys == 0) | (xs == W - 1) | (ys == H - 1)
                for d in range(8):
                    nx = xs + _DIR_X[d]
                    ny = ys + _DIR_Y[d]
                    ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
                    nb = np.zeros(rem.size, bool)
                    nb[ok] = pixels[nx[ok] + ny[ok] * W] == 0
                    add |= nb
                coords[next_level] = np.concatenate([coords[next_level][:int(hist[next_level])], rem[add]])
                hist[next_level] = coords[next_level].size


def _process_level(pass_: int, pixels: np.ndarray, table: np.ndarray, coords: dict[int, np.ndarray], level: int, W: int, H: int) -> int:
    """``MaximumFinder.processLevel``: one dilation pass in one direction over the remaining points of a level."""
    pts = coords[level]
    if pts.size == 0:
        return 0
    xs = pts % W
    ys = pts // W
    xmax, ymax = W - 1, H - 1
    index = np.zeros(pts.size, np.int64)
    is255 = pixels == 255

    def nb(cond: np.ndarray, off: int, bit: int) -> None:
        sel = cond.copy()
        sel[cond] = is255[pts[cond] + off]
        index[sel] ^= bit
    nb(ys > 0, -W, 1)
    nb((xs < xmax) & (ys > 0), -W + 1, 2)
    nb(xs < xmax, 1, 4)
    nb((xs < xmax) & (ys < ymax), W + 1, 8)
    nb(ys < ymax, W, 16)
    nb((xs > 0) & (ys < ymax), W - 1, 32)
    nb(xs > 0, -1, 64)
    nb((xs > 0) & (ys > 0), -W - 1, 128)
    m = 1 << pass_
    changed = (table[index] & m) == m
    pixels[pts[changed]] = 255
    coords[level] = pts[~changed]
    return int(changed.sum())


def _n_radii(px: list[int], x: int, y: int, W: int, H: int) -> int:
    """``MaximumFinder.nRadii``: number of 4-connected lines emanating from (x, y)."""
    offset = x + y * W
    count = 0
    prev_set = True
    first_set = True
    wm1, hm1 = W - 1, H - 1
    is_inner = (y != 0 and y != hm1) and (x != 0 and x != wm1)
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    for d in range(8):
        pixel_set = prev_set
        if is_inner or _within(x, y, d, W, H):
            is_set = px[offset + dir_off[d]] != 255
            if (d & 1) == 0:
                pixel_set = is_set
            elif not is_set:
                pixel_set = False
        else:
            pixel_set = True
        if pixel_set and not prev_set:
            count += 1
        prev_set = pixel_set
        if d == 0:
            first_set = pixel_set
    if first_set and not prev_set:
        count += 1
    return count


def _within(x: int, y: int, d: int, W: int, H: int) -> bool:
    xmax, ymax = W - 1, H - 1
    if d == 0:
        return y > 0
    if d == 1:
        return x < xmax and y > 0
    if d == 2:
        return x < xmax
    if d == 3:
        return x < xmax and y < ymax
    if d == 4:
        return y < ymax
    if d == 5:
        return x > 0 and y < ymax
    if d == 6:
        return x > 0
    return x > 0 and y > 0


def _cleanup_extra_lines(pixels: np.ndarray, W: int, H: int) -> None:
    """``MaximumFinder.cleanupExtraLines``: remove dots and dangling lines of the watershed of a non-EDM image."""
    px = pixels.tolist()
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    for i in range(W * H):
        v = px[i]
        if v != 255 and v != 0:
            x = i % W
            y = i // W
            n_radii = _n_radii(px, x, y, W, H)
            if n_radii == 0:
                px[i] = 255
            elif n_radii == 1:
                px[x + W * y] = 255                                       # removeLineFrom
                while True:
                    continues = False
                    is_inner = (y != 0 and y != H - 1) and (x != 0 and x != W - 1)
                    for d in range(0, 8, 2):
                        if is_inner or _within(x, y, d, W, H):
                            vv = px[x + W * y + dir_off[d]]
                            if vv != 255 and vv != 0:
                                nr = _n_radii(px, x + _DIR_X[d], y + _DIR_Y[d], W, H)
                                if nr <= 1:
                                    x += _DIR_X[d]
                                    y += _DIR_Y[d]
                                    px[x + W * y] = 255
                                    continues = nr == 1
                                    break
                    if not continues:
                        break
    pixels[:] = np.asarray(px, np.uint8)


def _delete_edge_particles(pixels: np.ndarray, types: np.ndarray, W: int, H: int) -> None:
    """``MaximumFinder.deleteEdgeParticles``: particles whose maximum area touches the image edge are erased."""
    img = pixels.reshape(H, W)
    ty = types.reshape(H, W)

    def delete(x: int, y: int) -> None:
        outline = wand_outline(img == 255, x, y)
        if outline is None:
            return
        m, (bx, by, bw, bh) = int_polygon_mask(outline[0], outline[1])
        if bw <= 0 or bh <= 0:
            return
        sub = img[by:by + bh, bx:bx + bw]
        sub[m[:sub.shape[0], :sub.shape[1]]] = 0

    for x in range(W):
        for y in (0, H - 1):
            if (ty[y, x] & MAX_AREA) != 0 and img[y, x] != 0:
                delete(x, y)
    for y in range(1, H - 1):
        for x in (0, W - 1):
            if (ty[y, x] & MAX_AREA) != 0 and img[y, x] != 0:
                delete(x, y)


# ====================================================================== ParticleAnalyzer
@dataclass
class Particle:
    bounds: tuple[int, int, int, int]      # x, y, w, h (the traced outline's bounds)
    mask: np.ndarray | None               # bounds-sized mask (None = the whole rectangle, as for 1-pixel-wide particles)
    outline_x: list[int]                  # traced outline vertices (image coordinates)
    outline_y: list[int]
    start: tuple[int, int]                # first pixel in raster order (XStart / YStart)
    pixel_count: int = 0


def find_particles(binary: np.ndarray, rect: tuple[int, int, int, int] | None = None, include_holes: bool = False,
                   exclude_on_edges: bool = False, min_size: float = 0.0, max_size: float = float("inf"),
                   min_circ: float = 0.0, max_circ: float = 1.0) -> list[Particle]:
    """``ParticleAnalyzer.analyze`` / ``analyzeParticle``: scan the thresholded pixels in raster order, trace each
    particle with the (8-connected) wand, build its mask from the traced polygon (holes included) or by flood filling
    the thresholded pixels (``particleAnalyzerFill``), erase it, and apply the size (pixels), circularity and edge
    filters.  Returns the particles in ImageJ's order."""
    work = np.asarray(binary, dtype=bool).copy()
    H, W = work.shape
    rx, ry, rw, rh = rect if rect is not None else (0, 0, W, H)
    outside = np.ones((H, W), bool)
    outside[ry:ry + rh, rx:rx + rw] = False
    work[outside] = False
    min_x, max_x, min_y, max_y = rx, rx + rw, ry, ry + rh
    flood_fill = not include_holes
    particles: list[Particle] = []
    flat = work.ravel()
    candidates = np.flatnonzero(flat)
    struct = np.ones((3, 3), bool)
    for idx in candidates:
        if not flat[idx]:
            continue
        x, y = int(idx % W), int(idx // W)
        outline = wand_outline(work, x, y)
        if outline is None:
            flat[idx] = False
            continue
        ox, oy = outline
        bx, by = min(ox), min(oy)
        bw, bh = max(ox) - bx, max(oy) - by
        mask: np.ndarray | None = None
        if bw > 1 and bh > 1:
            mask = polygon_filler_mask([v - bx for v in ox], [v - by for v in oy], bw, bh)
            if flood_fill:
                sub = work[by:by + bh, bx:bx + bw]
                lab, _ = ndi.label(sub, structure=struct)
                mask = lab == lab[y - by, x - bx]
        sub = work[by:by + bh, bx:bx + bw]
        if mask is None:
            pixel_count = int(sub.size)
        else:
            pixel_count = int(mask.sum())
        include = True
        if exclude_on_edges and (bx == min_x or by == min_y or bx + bw == max_x or by + bh == max_y):
            include = False
        if min_circ > 0.0 or max_circ != 1.0:
            per = traced_perimeter(ox, oy)
            circ = 0.0 if per == 0.0 else 4.0 * math.pi * (pixel_count / (per * per))
            if circ > 1.0 and max_circ <= 1.0:
                circ = 1.0
            if circ < min_circ or circ > max_circ:
                include = False
        if pixel_count >= min_size and pixel_count <= max_size and include:
            particles.append(Particle((bx, by, bw, bh), mask, ox, oy, (x, y), pixel_count))
        # ip.fill(mask): erase the particle so that the scan does not find it again (holes included when polygon-filled)
        if mask is None:
            sub[:] = False
        else:
            sub[mask] = False
    return particles
