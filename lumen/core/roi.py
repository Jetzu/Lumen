"""Regions of interest.

A ROI is a plain dict so it round-trips to JSON unchanged between the browser,
the server and scripts:

    {"type": "rect",      "x", "y", "w", "h"}
    {"type": "oval",      "x", "y", "w", "h"}
    {"type": "polygon",   "points": [[x, y], ...]}
    {"type": "freehand",  "points": [[x, y], ...]}
    {"type": "line",      "x1", "y1", "x2", "y2", "width"}
    {"type": "polyline",  "points": [[x, y], ...], "width"}
    {"type": "point",     "points": [[x, y], ...]}
    {"type": "mask",      "x", "y", "w", "h", "rle": [...] }   (run-length encoded mask)

Optional keys: name, position {t, z, c}, color, stroke_width, id.
"""
from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon as sk_polygon, ellipse as sk_ellipse, line as sk_line

AREA_TYPES = {"rect", "oval", "polygon", "freehand", "mask"}
LINE_TYPES = {"line", "polyline"}


def _ijm():
    """The ImageJ geometry ports (imported lazily: the ops package imports every op module, which import this one)."""
    from .ops import _imagej_measure as m
    return m


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _is_integer_points(pts: np.ndarray) -> bool:
    return bool(pts.size) and bool(np.all(pts == np.floor(pts)))


def int_bounds(roi: dict[str, Any]) -> tuple[int, int, int, int]:
    """ImageJ ``Roi.getBounds()``: the integer bounding rectangle – ``(int)x, (int)y, ceil(w), ceil(h)`` for
    rectangles and ovals, ``Polygon.getBounds`` (width = maxX − minX) for integer polygons, ``Roi.setIntBounds``
    (floor(min) … ceil(max), ``PolygonRoi.finishPolygon``) for sub-pixel polygons – except the ``subtype`` 'rotated_rect'
    and 'ellipse' polygons (``RotatedRectRoi`` / ``EllipseRoi.makePolygonRelative``: ``FloatPolygon.getBounds``, width =
    (int)(maxX − floor(minX) + 0.5)) – and the 'draw' area for straight lines."""
    t = roi["type"]
    if t in ("rect", "oval"):
        return int(roi["x"]), int(roi["y"]), int(math.ceil(roi["w"])), int(math.ceil(roi["h"]))
    if t == "mask":
        return int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    if t == "line":
        return _ijm().line_int_bounds(roi["x1"], roi["y1"], roi["x2"], roi["y2"])
    pts = np.asarray(roi.get("points", []), dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return 0, 0, 0, 0
    if _is_integer_points(pts):
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
    return _subpixel_bounds(pts, roi.get("subtype") in ("rotated_rect", "ellipse"))


def _subpixel_bounds(pts: np.ndarray, relative_bounds: bool) -> tuple[int, int, int, int]:
    """Integer bounds of a sub-pixel polygon: ``FloatPolygon.getBounds`` when ``relative_bounds`` (RotatedRectRoi /
    EllipseRoi), otherwise ``Roi.setIntBounds(poly.getFloatBounds())`` of ``PolygonRoi.finishPolygon`` – x = floor(minX),
    width = ceil(minX + (maxX − minX)) − x with float32 min / max."""
    ijm = _ijm()
    x, y, w, h = ijm.float_polygon_bounds(pts[:, 0], pts[:, 1])
    if relative_bounds:
        return x, y, w, h
    xs = pts[:, 0].astype(np.float32)
    ys = pts[:, 1].astype(np.float32)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w2 = int(math.ceil(float(min_x) + float(np.float32(max_x - min_x)))) - x
    h2 = int(math.ceil(float(min_y) + float(np.float32(max_y - min_y)))) - y
    return x, y, w2, h2


def roi_rect_mask(roi: dict[str, Any] | None, shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray | None]:
    """The ROI as ImageJ's ``ImageProcessor.setRoi(roi)`` sees it: the integer bounding rectangle (unclipped) and the
    rect-sized boolean mask (None for rectangles and the whole image).  Ovals use ``OvalRoi.getMask``, polygons the
    ``PolygonFiller`` (integer or sub-pixel path), wide straight lines the ``RotatedRectRoi`` of ``convertLineToArea``."""
    h, w = shape
    if roi is None:
        return (0, 0, w, h), None
    ijm = _ijm()
    t = roi["type"]
    if t == "rect":
        x0, y0 = int(round(roi["x"])), int(round(roi["y"]))
        x1, y1 = int(round(roi["x"] + roi["w"])), int(round(roi["y"] + roi["h"]))
        return (x0, y0, x1 - x0, y1 - y0), None
    if t == "oval":
        x, y, ow, oh = int_bounds(roi)
        return (x, y, ow, oh), ijm.oval_mask(ow, oh)
    if t in ("polygon", "freehand"):
        pts = np.asarray(roi["points"], dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            return (0, 0, 0, 0), np.zeros((0, 0), bool)
        if _is_integer_points(pts):
            m, b = ijm.int_polygon_mask(pts[:, 0], pts[:, 1])
        else:
            m, b = ijm.subpixel_polygon_mask(pts[:, 0], pts[:, 1])
            b2 = _subpixel_bounds(pts, roi.get("subtype") in ("rotated_rect", "ellipse"))
            if b2 != b:   # setIntBounds may enclose one more (empty) row / column than FloatPolygon.getBounds
                m2 = np.zeros((max(b2[3], 0), max(b2[2], 0)), bool)
                hh, ww = min(m.shape[0], m2.shape[0]), min(m.shape[1], m2.shape[1])
                m2[:hh, :ww] = m[:hh, :ww]
                m, b = m2, b2
        return b, m
    if t == "mask":
        return int_bounds(roi), rle_decode(roi["rle"], (int(roi["h"]), int(roi["w"])))
    if t == "line" and float(roi.get("width", 1) or 1) > 1:
        xs, ys = ijm.rotated_rect_polygon(roi["x1"], roi["y1"], roi["x2"], roi["y2"], float(roi["width"]))
        m, b = ijm.subpixel_polygon_mask(xs, ys)
        return b, m
    m = to_mask(roi, shape)
    return (0, 0, w, h), m


def hull_points(roi: dict[str, Any] | None, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The vertices ImageJ's ``Roi.getFloatPolygon("")`` feeds to the convex hull (Feret, Solidity): rectangle corners,
    the flattened ellipse of an oval, polygon vertices, the pixel corners of a mask ROI, the end points (or the 4-point
    outline of a wide line) of a straight line; ``shape`` (H, W) stands in for a missing ROI (the whole image)."""
    ijm = _ijm()
    if roi is None:
        h, w = shape or (0, 0)
        return np.asarray([0, w, w, 0], np.float32), np.asarray([0, 0, h, h], np.float32)
    t = roi["type"]
    if t == "rect":
        x, y, w, h = int_bounds(roi)
        return np.asarray([x, x + w, x + w, x], np.float32), np.asarray([y, y, y + h, y + h], np.float32)
    if t == "oval":
        x, y, w, h = int_bounds(roi)
        return ijm.oval_float_polygon(x, y, w, h)
    if t in ("polygon", "freehand", "polyline", "point"):
        pts = np.asarray(roi.get("points", []), dtype=float).reshape(-1, 2)
        if _is_integer_points(pts) or pts.size == 0:
            return pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
        xb, yb = np.float32(pts[:, 0].min()), np.float32(pts[:, 1].min())   # PolygonRoi.getFloatPolygon: float32 base + relative
        xs = ((pts[:, 0].astype(np.float32) - xb).astype(np.float32) + xb).astype(np.float32)
        ys = ((pts[:, 1].astype(np.float32) - yb).astype(np.float32) + yb).astype(np.float32)
        return xs, ys
    if t == "line":
        return ijm.line_float_polygon(roi["x1"], roi["y1"], roi["x2"], roi["y2"], float(roi.get("width", 1) or 1))
    if t == "mask":
        sub = rle_decode(roi["rle"], (int(roi["h"]), int(roi["w"])))
        ys, xs = np.nonzero(sub)
        if xs.size == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        x0, y0 = int(roi["x"]), int(roi["y"])
        cx = np.concatenate([xs, xs + 1, xs, xs + 1]) + x0
        cy = np.concatenate([ys, ys, ys + 1, ys + 1]) + y0
        corners = np.unique(np.stack([cx, cy], axis=1), axis=0)
        return corners[:, 0].astype(np.float32), corners[:, 1].astype(np.float32)
    return np.zeros(0, np.float32), np.zeros(0, np.float32)


def feret_values(roi: dict[str, Any] | None, pw: float = 1.0, ph: float = 1.0, shape: tuple[int, int] | None = None) -> tuple[float, float, float, float, float]:
    """ImageJ ``Roi.getFeretValues`` → (Feret, FeretAngle, MinFeret, FeretX, FeretY); ovals use ``OvalRoi``'s own
    rule (axes; FeretX/Y calibrated), every other ROI the convex hull of :func:`hull_points` (FeretX/Y in pixels)."""
    ijm = _ijm()
    if roi is not None and roi["type"] == "oval":
        x, y, w, h = int_bounds(roi)
        return ijm.oval_feret_values(x, y, w, h, pw, ph)
    xs, ys = hull_points(roi, shape)
    hx, hy = ijm.convex_hull(xs, ys)
    return ijm.feret_values(hx, hy, pw, ph)


def convex_hull_area(roi: dict[str, Any] | None, shape: tuple[int, int] | None = None) -> float:
    """``Analyzer.getArea(roi.getConvexHull())``: the integer-rounded convex hull's area (Solidity's denominator)."""
    xs, ys = hull_points(roi, shape)
    if xs.size == 0:
        return float("nan")
    return _ijm().hull_int_area(xs, ys)


def normalize(roi: dict[str, Any]) -> dict[str, Any]:
    roi = dict(roi)
    roi.setdefault("id", new_id())
    t = roi.get("type")
    if t in ("rect", "oval"):
        x, y, w, h = float(roi["x"]), float(roi["y"]), float(roi["w"]), float(roi["h"])
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        roi.update(x=x, y=y, w=w, h=h)
    elif t in ("polygon", "freehand", "polyline", "point"):
        roi["points"] = [[float(p[0]), float(p[1])] for p in roi.get("points", [])]
    elif t == "line":
        for k in ("x1", "y1", "x2", "y2"):
            roi[k] = float(roi[k])
        roi.setdefault("width", 1)
    return roi


def is_area(roi: dict[str, Any] | None) -> bool:
    return bool(roi) and roi.get("type") in AREA_TYPES


def is_line(roi: dict[str, Any] | None) -> bool:
    return bool(roi) and roi.get("type") in LINE_TYPES


def bounding_box(roi: dict[str, Any]) -> tuple[float, float, float, float]:
    """(x, y, w, h) in pixel units."""
    t = roi["type"]
    if t in ("rect", "oval", "mask"):
        return roi["x"], roi["y"], roi["w"], roi["h"]
    if t == "line":
        x0, x1 = sorted([roi["x1"], roi["x2"]])
        y0, y1 = sorted([roi["y1"], roi["y2"]])
        return x0, y0, x1 - x0, y1 - y0
    pts = np.asarray(roi["points"], dtype=float)
    if pts.size == 0:
        return 0, 0, 0, 0
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def to_mask(roi: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise an area ROI to a boolean mask of the given (H, W)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    t = roi["type"]
    if t == "rect":
        y0, y1 = max(int(round(roi["y"])), 0), min(int(round(roi["y"] + roi["h"])), h)
        x0, x1 = max(int(round(roi["x"])), 0), min(int(round(roi["x"] + roi["w"])), w)
        if y1 > y0 and x1 > x0:   # a ROI entirely off the image must stay empty (a negative stop would count from the end)
            mask[y0:y1, x0:x1] = True
    elif t in ("oval", "polygon", "freehand"):
        # ImageJ's OvalRoi.getMask / PolygonFiller (even-odd, pixel-centre convention), pasted at the ROI's integer bounds
        (x0, y0, bw, bh), sub = roi_rect_mask(roi, shape)
        if sub is not None and bw > 0 and bh > 0:
            ya, yb = max(y0, 0), min(y0 + bh, h)
            xa, xb = max(x0, 0), min(x0 + bw, w)
            if yb > ya and xb > xa:
                mask[ya:yb, xa:xb] = sub[ya - y0:yb - y0, xa - x0:xb - x0]
    elif t == "mask":
        sub = rle_decode(roi["rle"], (int(roi["h"]), int(roi["w"])))
        x0, y0 = int(roi["x"]), int(roi["y"])
        ya, yb = max(y0, 0), min(y0 + sub.shape[0], h)
        xa, xb = max(x0, 0), min(x0 + sub.shape[1], w)
        if yb > ya and xb > xa:   # empty intersection on any side (above/left: stop < 0; below/right: start > h/w)
            mask[ya:yb, xa:xb] = sub[ya - y0:yb - y0, xa - x0:xb - x0]
    elif t in ("line", "polyline"):
        pts = line_points(roi)
        width = float(roi.get("width", 1) or 1)
        if len(pts) < 2:
            return mask
        if width <= 1:
            # one Bresenham segment per vertex pair (C), scattered into the mask in one go.  Pixel (i, j) covers
            # [i, i+1) and is centred at (j + 0.5, i + 0.5), so a continuous coordinate maps to its pixel with floor
            # (round() put a line through the centre of row 3, y = 3.5, into row 4) – the same mapping as
            # paint.stroke_mask, profile sampling and the viewport.
            ends = [[int(math.floor(v)) for v in (ya, xa, yb, xb)] for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:])]
            segs = [sk_line(*e) for e in ends]
            rr = np.concatenate([s[0] for s in segs])
            cc = np.concatenate([s[1] for s in segs])
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            mask[rr[ok], cc[ok]] = True
        elif t == "line":
            # ImageJ Roi.convertLineToArea: the RotatedRectRoi of the wide line, rasterised by the PolygonFiller
            (x0, y0, bw, bh), sub = roi_rect_mask(roi, shape)
            if sub is not None and bw > 0 and bh > 0:
                ya, yb = max(y0, 0), min(y0 + bh, h)
                xa, xb = max(x0, 0), min(x0 + bw, w)
                if yb > ya and xb > xa:
                    mask[ya:yb, xa:xb] = sub[ya - y0:yb - y0, xa - x0:xb - x0]
        else:
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                poly = thick_segment_polygon(xa, ya, xb, yb, width)
                rr, cc = sk_polygon(poly[:, 1] - 0.5, poly[:, 0] - 0.5, shape=shape)   # same pixel-centre shift as the polygon branch
                mask[rr, cc] = True
    elif t == "point":
        pts = np.asarray(roi["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts):
            xi = pts[:, 0].astype(np.int64)   # truncation toward zero, like int(x)
            yi = pts[:, 1].astype(np.int64)
            ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            mask[yi[ok], xi[ok]] = True
    return mask


def thick_segment_polygon(xa: float, ya: float, xb: float, yb: float, width: float) -> np.ndarray:
    dx, dy = xb - xa, yb - ya
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    return np.array([[xa + nx, ya + ny], [xb + nx, yb + ny], [xb - nx, yb - ny], [xa - nx, ya - ny]])


def line_points(roi: dict[str, Any]) -> list[tuple[float, float]]:
    if roi["type"] == "line":
        return [(roi["x1"], roi["y1"]), (roi["x2"], roi["y2"])]
    return [tuple(p) for p in roi["points"]]


def line_length(roi: dict[str, Any], pw: float = 1.0, ph: float = 1.0) -> float:
    pts = line_points(roi)
    return float(sum(math.hypot((b[0] - a[0]) * pw, (b[1] - a[1]) * ph) for a, b in zip(pts[:-1], pts[1:])))


def line_angle(roi: dict[str, Any], pw: float = 1.0, ph: float = 1.0) -> float:
    """ImageJ convention: degrees, counter-clockwise, y axis pointing up; dx/dy scaled by the pixel size."""
    pts = line_points(roi)
    if len(pts) < 2:
        return 0.0
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    return float(math.degrees(math.atan2((y1 - y2) * ph, (x2 - x1) * pw)))


def polygon_points(roi: dict[str, Any], n_oval: int | None = None) -> np.ndarray:
    """Outline of an area ROI as an (N, 2) array of (x, y).

    An oval gives ImageJ's ``OvalRoi.getFloatPolygon`` (the ellipse flattened to 0.1 px, the vertices the macro
    ``getSelectionCoordinates`` reports) unless ``n_oval`` asks for that many uniformly spaced vertices.
    A mask ROI may have several parts and holes; this returns its LARGEST outline only (callers that need one
    polygon).  Use :func:`outlines` / :func:`mask_contours` to get every part."""
    t = roi["type"]
    if t == "rect":
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    if t == "oval":
        if n_oval:
            cx, cy = roi["x"] + roi["w"] / 2, roi["y"] + roi["h"] / 2
            th = np.linspace(0, 2 * np.pi, n_oval, endpoint=False)
            return np.stack([cx + roi["w"] / 2 * np.cos(th), cy + roi["h"] / 2 * np.sin(th)], axis=1)
        x, y, w, h = int_bounds(roi)
        xs, ys = _ijm().oval_float_polygon(x, y, w, h)
        return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    if t in ("polygon", "freehand"):
        return np.asarray(roi["points"], float)
    if t == "mask":
        cs = mask_contours(roi)
        if not cs:
            return np.zeros((0, 2))
        return max(cs, key=len)
    return np.zeros((0, 2))


def mask_contours(roi: dict[str, Any]) -> list[np.ndarray]:
    """Every closed outline of a mask ROI – outer boundaries and hole boundaries – as (N, 2) arrays of (x, y) in
    image coordinates (vertices on pixel edges, multiples of 0.5), the same tracing the overlay renderer draws."""
    from skimage.measure import find_contours
    sub = rle_decode(roi["rle"], (int(roi["h"]), int(roi["w"])))
    cs = find_contours(np.pad(sub, 1).astype(float), 0.5)
    return [np.stack([c[:, 1] - 1 + roi["x"] + 0.5, c[:, 0] - 1 + roi["y"] + 0.5], axis=1) for c in cs if len(c) > 1]


def outlines(roi: dict[str, Any]) -> list[np.ndarray]:
    """All outlines of an area ROI: every part and hole of a mask ROI, otherwise ``[polygon_points(roi)]``."""
    if roi["type"] == "mask":
        return mask_contours(roi)
    pts = polygon_points(roi)
    return [pts] if len(pts) else []


def perimeter(roi: dict[str, Any], pw: float = 1.0, ph: float = 1.0) -> float:
    """ImageJ ``Roi.getLength``: 2(w + h) for a rectangle, Ramanujan's ellipse circumference for an oval, the closed
    polygon length for a polygon – or ``PolygonRoi.getTracedPerimeter`` (corner pixels count √2) when its edges are all
    horizontal / vertical with integer vertices, as for wand / particle outlines –, the 3-point smoothed perimeter for a
    freehand selection, the line length for lines.  A mask ROI follows ``ShapeRoi.getLength``: the traced perimeter of
    a single hole-free blob, otherwise the length of its pixel-edge outline (every part and hole)."""
    ijm = _ijm()
    t = roi["type"]
    if t == "rect":
        return 2 * (roi["w"] * pw + roi["h"] * ph)
    if t == "oval":
        x, y, w, h = int_bounds(roi)
        return ijm.oval_length(w, h, pw, ph)
    if t in ("line", "polyline"):
        return line_length(roi, pw, ph)
    if t in ("polygon", "freehand"):
        pts = np.asarray(roi.get("points", []), dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            return 0.0
        if t == "polygon" and ijm.is_traced(pts[:, 0], pts[:, 1]):
            return ijm.traced_perimeter(pts[:, 0], pts[:, 1], pw, ph)
        if t == "freehand" and len(pts) > 2:
            return ijm.smoothed_perimeter(pts[:, 0], pts[:, 1], pw, ph)
        xs, ys = (pts[:, 0], pts[:, 1]) if _is_integer_points(pts) else hull_points(roi)
        return ijm.polygon_length(xs, ys, True, pw, ph)
    if t == "mask":
        sub = rle_decode(roi["rle"], (int(roi["h"]), int(roi["w"])))
        if not sub.any():
            return 0.0
        lab, n = ndi.label(sub, structure=np.ones((3, 3), bool))
        if n == 1 and not (ndi.binary_fill_holes(sub) & ~sub).any():
            ys, xs = np.nonzero(sub)
            k = int(np.argmin(ys * sub.shape[1] + xs))
            outline = ijm.wand_outline(sub, int(xs[k]), int(ys[k]))
            if outline is not None:
                return ijm.traced_perimeter(outline[0], outline[1], pw, ph)
        padded = np.pad(sub, 1)
        horiz = int(np.count_nonzero(padded[1:, :] != padded[:-1, :]))   # edges between vertically adjacent pixels
        vert = int(np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
        return horiz * pw + vert * ph
    total = 0.0
    for pts in outlines(roi):
        if len(pts) < 2:
            continue
        closed = np.vstack([pts, pts[:1]])
        d = np.diff(closed, axis=0)
        total += float(np.sum(np.hypot(d[:, 0] * pw, d[:, 1] * ph)))
    return total


def translate(roi: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    r = dict(roi)
    t = r["type"]
    if t in ("rect", "oval", "mask"):
        r["x"] += dx; r["y"] += dy
    elif t == "line":
        r["x1"] += dx; r["x2"] += dx; r["y1"] += dy; r["y2"] += dy
    else:
        r["points"] = [[p[0] + dx, p[1] + dy] for p in r["points"]]
    return r


def scale_roi(roi: dict[str, Any], sx: float, sy: float) -> dict[str, Any]:
    r = dict(roi)
    t = r["type"]
    if t in ("rect", "oval"):
        r["x"] *= sx; r["w"] *= sx; r["y"] *= sy; r["h"] *= sy
    elif t == "line":
        r["x1"] *= sx; r["x2"] *= sx; r["y1"] *= sy; r["y2"] *= sy
    elif t == "mask":
        from skimage.transform import resize
        # rasterise in a non-negative frame (a selection nudged off the top/left edge has negative x/y – the old
        # origin-anchored canvas then had negative dimensions), scale about the origin like the other branches
        # (identical for ROIs inside the image) and shift back
        ox, oy = min(0, int(r["x"])), min(0, int(r["y"]))
        rr = translate(r, -ox, -oy)
        m = to_mask(rr, (int(rr["y"] + rr["h"]) + 1, int(rr["x"] + rr["w"]) + 1))
        m2 = resize(m.astype(float), (max(1, int(round(m.shape[0] * sy))), max(1, int(round(m.shape[1] * sx)))), order=0) > 0.5
        out = from_mask(m2, r.get("name"))
        if out is None:
            raise ValueError("Selection is empty after scaling")
        return translate(out, int(round(ox * sx)), int(round(oy * sy)))
    else:
        r["points"] = [[p[0] * sx, p[1] * sy] for p in r["points"]]
    return r


# ------------------------------------------------------------------ masks ⇄ ROIs
def rle_encode(mask: np.ndarray) -> list[int]:
    """Run lengths of a boolean mask (row-major), alternating background / foreground and starting with background."""
    flat = np.asarray(mask).ravel() != 0
    if flat.size == 0:
        return []
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], idx, [flat.size]])
    out = np.diff(bounds).tolist()            # Python ints
    if flat[0]:
        out.insert(0, 0)                      # RLE alternates background/foreground, starting with background
    return out


def rle_decode(rle: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Inverse of rle_encode → boolean (H, W).  Runs past the end are clipped, a short list is padded with background."""
    n = int(shape[0]) * int(shape[1])
    runs = np.asarray(rle, dtype=np.int64).ravel()
    if runs.size == 0 or n == 0:
        return np.zeros(shape, dtype=bool)
    runs = np.clip(runs, 0, n)
    values = np.zeros(runs.size, dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, runs)
    if flat.size < n:
        flat = np.concatenate([flat, np.zeros(n - flat.size, dtype=bool)])
    return flat[:n].reshape(shape)


def from_mask(mask: np.ndarray, name: str | None = None) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    roi = {"id": new_id(), "type": "mask", "x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
           "rle": rle_encode(sub)}
    if name:
        roi["name"] = name
    return roi


def _crop_mask_roi(sub: np.ndarray, x0: int, y0: int, name: str | None) -> dict[str, Any] | None:
    """from_mask() of a bounding-box crop placed at (x0, y0) – the same ROI as from_mask(full-size mask)."""
    roi = from_mask(sub, name)
    if roi is not None:
        roi["x"] += int(x0)
        roi["y"] += int(y0)
    return roi


def label_to_rois(labels: np.ndarray, simplify: bool = True) -> list[dict[str, Any]]:
    """Convert a label image into one polygon (or mask) ROI per label.

    Every label is processed on its padded bounding-box crop only (contour at level 0.5, run-length mask); a label
    that one polygon cannot hold – several contours (holes, several parts, parts touching only diagonally) or an
    Euler number below 1 – keeps its exact mask; the polygon outline is capped at about 400 vertices."""
    from skimage.measure import euler_number, find_contours
    labels = np.asarray(labels)
    objects = ndi.find_objects(labels)
    present = [i + 1 for i, sl in enumerate(objects) if sl is not None]
    if not present:
        return []
    centroids = ndi.center_of_mass(np.ones(labels.shape, np.float64), labels, present) if simplify else None
    rois = []
    for k, lab in enumerate(present):
        ys, xs = objects[lab - 1]
        y0, x0 = int(ys.start), int(xs.start)
        sub = labels[ys, xs] == lab
        if simplify:
            padded = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), np.float32)
            padded[1:-1, 1:-1] = sub
            contours = find_contours(padded, 0.5)
        else:
            contours = []
        if contours:
            cy, cx = centroids[k]
            name = f"{lab:04d}-{int(cy)}-{int(cx)}"
            # several contours (holes, several parts, parts touching only diagonally) or background enclosed through
            # diagonal gaps (Euler number < 1): one polygon cannot hold the label → keep the exact mask (the largest
            # contour alone silently dropped every other part)
            if len(contours) > 1 or euler_number(sub, connectivity=2) < 1:
                rois.append(_crop_mask_roi(sub, x0, y0, name))
            else:
                c = contours[0]
                pts = np.stack([c[:, 1] - 1 + x0 + 0.5, c[:, 0] - 1 + y0 + 0.5], axis=1)
                if len(pts) > 6:
                    pts = pts[:: max(1, len(pts) // 400)]
                # polygon used for display and manipulation (vertices are multiples of 0.5)
                rois.append({"id": new_id(), "type": "polygon", "points": np.round(pts, 2).tolist(), "name": name})
        else:
            m = _crop_mask_roi(sub, x0, y0, f"{lab:04d}")
            if m:
                rois.append(m)
    return rois


def combine(rois: list[dict[str, Any]], shape: tuple[int, int], mode: str = "or") -> dict[str, Any] | None:
    masks = [to_mask(r, shape) for r in rois if is_area(r) or is_line(r)]
    if not masks:
        return None
    if mode == "or":
        m = np.logical_or.reduce(masks)
    elif mode == "and":
        m = np.logical_and.reduce(masks)
    elif mode == "xor":
        m = np.logical_xor.reduce(masks)
    else:
        raise ValueError(mode)
    return from_mask(m, mode.upper())


def polygons_to_mask_roi(parts: list[np.ndarray], name: str | None = None) -> dict[str, Any] | None:
    """Rasterise several closed outlines ((N, 2) arrays of (x, y), the pixel-edge convention of
    :func:`polygon_points`) into ONE mask ROI with even-odd parity, so hole boundaries stay holes.  The canvas is
    the outlines' own bounding box, so parts beyond the image or in negative coordinates are kept."""
    parts = [np.asarray(p, dtype=float).reshape(-1, 2) for p in parts if len(p) >= 3]
    if not parts:
        return None
    allpts = np.vstack(parts)
    ox, oy = math.floor(allpts[:, 0].min()) - 1, math.floor(allpts[:, 1].min()) - 1
    shape = (int(math.ceil(allpts[:, 1].max())) - oy + 2, int(math.ceil(allpts[:, 0].max())) - ox + 2)
    canvas = np.zeros(shape, dtype=bool)
    for p in parts:
        rr, cc = sk_polygon(p[:, 1] - oy - 0.5, p[:, 0] - ox - 0.5, shape=shape)
        canvas[rr, cc] = ~canvas[rr, cc]
    out = from_mask(canvas, name)
    return translate(out, ox, oy) if out is not None else None


def contains_point(roi: dict[str, Any], x: float, y: float) -> bool:
    bx, by, bw, bh = bounding_box(roi)
    if not (bx <= x <= bx + bw and by <= y <= by + bh):
        return False
    if roi["type"] in AREA_TYPES:
        # a ROI in negative coordinates (nudged off the image, auto-cropped overlay) is rasterised in a frame shifted
        # by its own offset – the origin-anchored canvas had negative dimensions – so the answer stays geometric like
        # ImageJ's Roi.contains; ROIs inside the image use the origin frame as before
        ox, oy = min(0, math.floor(bx)), min(0, math.floor(by))
        m = to_mask(translate(roi, -ox, -oy) if (ox or oy) else roi, (int(by - oy + bh) + 2, int(bx - ox + bw) + 2))
        xi, yi = int(math.floor(x - ox)), int(math.floor(y - oy))
        return 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and bool(m[yi, xi])
    return True


class RoiManager:
    """The session-wide list of saved ROIs (the ImageJ 'ROI Manager')."""

    def __init__(self) -> None:
        self.rois: list[dict[str, Any]] = []

    def add(self, roi: dict[str, Any], image_id: str | None = None, position: dict[str, int] | None = None) -> dict[str, Any]:
        r = normalize({k: v for k, v in roi.items() if k != "id"})  # every manager entry gets its own id
        r.setdefault("name", self._auto_name(r))
        if image_id:
            r["image_id"] = image_id
        if position:
            r["position"] = position
        self.rois.append(r)
        return r

    def _auto_name(self, roi: dict[str, Any]) -> str:
        x, y, w, h = bounding_box(roi)
        return f"{int(y + h / 2):04d}-{int(x + w / 2):04d}"

    def remove(self, ids: list[str]) -> None:
        ids_set = set(ids)
        self.rois = [r for r in self.rois if r["id"] not in ids_set]

    def clear(self) -> None:
        self.rois = []

    def get(self, roi_id: str) -> dict[str, Any] | None:
        return next((r for r in self.rois if r["id"] == roi_id), None)

    def update(self, roi_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        for i, r in enumerate(self.rois):
            if r["id"] == roi_id:
                r = normalize({**r, **patch, "id": roi_id})
                self.rois[i] = r
                return r
        return None

    def to_list(self) -> list[dict[str, Any]]:
        return self.rois
