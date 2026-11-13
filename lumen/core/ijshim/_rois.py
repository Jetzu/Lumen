"""``ij.gui`` ROIs (``Roi``, ``OvalRoi``, ``PolygonRoi``, ``Line``, ``PointRoi``, ``TextRoi``, ``ShapeRoi``),
``Overlay`` and ``ij.process.FloatPolygon`` over Lumen's ROI dicts (:mod:`lumen.core.roi`).

A shim ROI *is* a Lumen ROI dict (``roi._roi``), so it can be handed to ``imp.setRoi``, the ROI Manager, the
overlay and every Lumen command.  The selection is script-wide (like the macro language's): ``imp.setRoi(roi)``
sets the selection the following commands use."""
from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from .. import roi as roimod
from ._context import current
from ._images import ByteProcessor, ImagePlus, ImageStatistics, _img_of
from ._java import Color, Point, Polygon, Rectangle, color_hex


class FloatPolygon(Polygon):
    """``ij.process.FloatPolygon``: float vertices (``xpoints`` / ``ypoints`` / ``npoints``)."""

    _int = False

    def getBounds(self) -> Rectangle:
        return super().getBounds()

    def getFloatBounds(self) -> tuple[float, float, float, float]:
        if not self.xpoints:
            return 0.0, 0.0, 0.0, 0.0
        x0, y0 = min(self.xpoints), min(self.ypoints)
        return x0, y0, max(self.xpoints) - x0, max(self.ypoints) - y0

    def getConvexHull(self) -> "FloatPolygon":
        from scipy.spatial import ConvexHull
        pts = np.array([self.xpoints, self.ypoints]).T
        if len(pts) < 3:
            return self.duplicate()
        try:
            hull = ConvexHull(pts)
        except Exception:
            return self.duplicate()
        return FloatPolygon(pts[hull.vertices, 0].tolist(), pts[hull.vertices, 1].tolist())

    def toPolygon(self) -> Polygon:
        return Polygon([int(round(v)) for v in self.xpoints], [int(round(v)) for v in self.ypoints])


_TYPE_NAMES = {0: "Rectangle", 1: "Oval", 2: "Polygon", 3: "Freehand", 4: "Traced", 5: "Straight Line", 6: "Polyline", 7: "Freeline", 8: "Angle", 9: "Composite", 10: "Point"}
_DICT_TYPE = {"rect": 0, "oval": 1, "polygon": 2, "freehand": 3, "line": 5, "polyline": 6, "point": 10, "mask": 9, "text": 0}


def _active_img():
    ctx = current()
    return ctx.session.active


class Roi:
    """``ij.gui.Roi``: ``Roi(x, y, w, h)`` / ``Roi(Rectangle)`` – the base of every selection type."""

    RECTANGLE, OVAL, POLYGON, FREEROI, TRACED_ROI, LINE, POLYLINE, FREELINE, ANGLE, COMPOSITE, POINT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    NO_ROI = -1
    HANDLE_SIZE = 5
    _js_props = ("name",)
    _default_color = "#ffff00"

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        self._imp: ImagePlus | None = None
        if _roi is not None:
            self._roi = roimod.normalize(_roi)
            return
        if len(args) == 1 and isinstance(args[0], Rectangle):
            r = args[0]
            self._roi = roimod.normalize({"type": "rect", "x": float(r.x), "y": float(r.y), "w": float(r.width), "h": float(r.height)})
        elif len(args) == 1 and isinstance(args[0], Roi):
            self._roi = dict(args[0]._roi)
        elif len(args) >= 4:
            x, y, w, h = (float(v) for v in args[:4])
            self._roi = roimod.normalize({"type": "rect", "x": x, "y": y, "w": w, "h": h})
            if len(args) > 4 and args[4]:
                self._roi["corner_diameter"] = float(args[4])
        elif len(args) == 2 and isinstance(args[0], (int, float)):
            self._roi = roimod.normalize({"type": "rect", "x": float(args[0]), "y": float(args[1]), "w": 0.0, "h": 0.0})
        else:
            raise TypeError("Roi(x, y, width, height) or Roi(Rectangle)")

    # --- plumbing
    @classmethod
    def _wrap(cls, d: dict[str, Any]) -> "Roi":
        t = d.get("type")
        kind: type[Roi] = {"rect": Roi, "oval": OvalRoi, "polygon": PolygonRoi, "freehand": PolygonRoi, "polyline": PolygonRoi, "line": Line,
                           "point": PointRoi, "mask": ShapeRoi, "text": TextRoi}.get(t, Roi)
        return kind(_roi=d)

    @property
    def roi(self) -> dict[str, Any]:
        """The Lumen ROI dict (Lumen extension)."""
        return self._roi

    def __repr__(self) -> str:
        x, y, w, h = self._bounds()
        return f"Roi[{self.getTypeAsString()}, x={int(x)}, y={int(y)}, width={int(w)}, height={int(h)}]"

    def toString(self) -> str:
        return repr(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Roi):
            return False
        a = {k: v for k, v in self._roi.items() if k not in ("id", "name", "image_id", "position", "color", "fill", "stroke_width")}
        b = {k: v for k, v in other._roi.items() if k not in ("id", "name", "image_id", "position", "color", "fill", "stroke_width")}
        return a == b

    def __hash__(self) -> int:
        return hash(self._roi.get("id"))

    def equals(self, other: Any) -> bool:
        return self == other

    def clone(self) -> "Roi":
        d = {k: v for k, v in self._roi.items() if k != "id"}
        r = type(self)(_roi=d)
        r._imp = self._imp
        return r

    copy = clone

    def _bounds(self) -> tuple[float, float, float, float]:
        return roimod.bounding_box(self._roi)

    def _shape(self) -> tuple[int, int]:
        img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
        if img is None:
            x, y, w, h = self._bounds()
            return int(math.ceil(y + h)) + 1, int(math.ceil(x + w)) + 1
        return img.height, img.width

    def _cal(self) -> tuple[float, float]:
        img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
        return (img.calibration.pixel_width, img.calibration.pixel_height) if img is not None else (1.0, 1.0)

    # --- type / geometry
    def getType(self) -> int:
        return _DICT_TYPE.get(self._roi.get("type", "rect"), 0)

    def getTypeAsString(self) -> str:
        return _TYPE_NAMES.get(self.getType(), "Rectangle")

    def isArea(self) -> bool:
        return roimod.is_area(self._roi)

    def isLine(self) -> bool:
        return roimod.is_line(self._roi)

    def isDrawingTool(self) -> bool:
        return self._roi.get("type") == "text"

    def isVisible(self) -> bool:
        return True

    def getBounds(self) -> Rectangle:
        x, y, w, h = self._bounds()
        return Rectangle(int(math.floor(x)), int(math.floor(y)), int(math.ceil(w)), int(math.ceil(h)))

    def getFloatBounds(self) -> Any:
        return self._bounds()

    def getXBase(self) -> float:
        return self._bounds()[0]

    def getYBase(self) -> float:
        return self._bounds()[1]

    def getFloatWidth(self) -> float:
        return self._bounds()[2]

    def getFloatHeight(self) -> float:
        return self._bounds()[3]

    def getLength(self) -> float:
        pw, ph = self._cal()
        if roimod.is_line(self._roi):
            return roimod.line_length(self._roi, pw, ph)
        return roimod.perimeter(self._roi, pw, ph)

    def getRawLength(self) -> float:
        if roimod.is_line(self._roi):
            return roimod.line_length(self._roi)
        return roimod.perimeter(self._roi)

    def getFeretsDiameter(self) -> float:
        return self.getFeretValues()[0]

    def getFeretValues(self) -> list[float]:
        """[Feret, FeretAngle, MinFeret, FeretX, FeretY] (pixel units)."""
        from ..measure import feret_diameters
        pts = self._points()
        vals = feret_diameters(np.asarray(pts, np.float64)) if len(pts) else (0.0, 0.0, 0.0, 0.0, 0.0)
        return [float(v) for v in vals]

    def _points(self) -> list[list[float]]:
        if roimod.is_line(self._roi) or self._roi.get("type") == "point":
            return [[float(x), float(y)] for x, y in roimod.line_points(self._roi)]
        if self._roi.get("type") == "mask":
            ys, xs = np.nonzero(roimod.to_mask(self._roi, self._shape()))
            return [[float(x), float(y)] for x, y in zip(xs, ys)]
        return [[float(p[0]), float(p[1])] for p in roimod.polygon_points(self._roi)]

    def getFloatPolygon(self) -> FloatPolygon:
        pts = self._points()
        return FloatPolygon([p[0] for p in pts], [p[1] for p in pts])

    def getPolygon(self) -> Polygon:
        pts = self._points()
        return Polygon([int(round(p[0])) for p in pts], [int(round(p[1])) for p in pts])

    def getConvexHull(self) -> Polygon:
        return self.getFloatPolygon().getConvexHull().toPolygon()

    def getNCoordinates(self) -> int:
        return len(self._points())

    def size(self) -> int:
        return self.getNCoordinates()

    def getXCoordinates(self) -> list[int]:
        x0 = self.getBounds().x
        return [int(round(p[0])) - x0 for p in self._points()]

    def getYCoordinates(self) -> list[int]:
        y0 = self.getBounds().y
        return [int(round(p[1])) - y0 for p in self._points()]

    def contains(self, x: Any, y: Any = None) -> bool:
        if isinstance(x, Point):
            x, y = x.x, x.y
        return roimod.contains_point(self._roi, float(x), float(y))

    def containsPoint(self, x: Any, y: Any) -> bool:
        return self.contains(x, y)

    def getContainedPoints(self) -> list[Point]:
        m = roimod.to_mask(self._roi, self._shape())
        ys, xs = np.nonzero(m)
        return [Point(int(x), int(y)) for x, y in zip(xs, ys)]

    def getContourCentroid(self) -> list[float]:
        m = roimod.to_mask(self._roi, self._shape()) if self.isArea() else None
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            return [float(xs.mean() + 0.5), float(ys.mean() + 0.5)]
        pts = self._points()
        if not pts:
            return [0.0, 0.0]
        return [float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))]

    getCentroid = getContourCentroid

    def getRotationCenter(self) -> FloatPolygon:
        cx, cy = self.getContourCentroid()
        return FloatPolygon([cx], [cy])

    def getMask(self) -> ByteProcessor | None:
        if self._roi.get("type") == "rect":
            return None
        m = roimod.to_mask(self._roi, self._shape())
        r = self.getBounds()
        sub = m[max(0, r.y): r.y + r.height, max(0, r.x): r.x + r.width]
        return ByteProcessor(_array=(sub.astype(np.uint8) * 255))

    def getStatistics(self) -> ImageStatistics:
        img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
        if img is None:
            raise ValueError("Roi.getStatistics(): no image")
        d = img.display
        return ImageStatistics(img, self._roi, d.t, d.z, d.c)

    # --- placement
    def setLocation(self, x: Any, y: Any) -> None:
        bx, by, _w, _h = self._bounds()
        self._roi = roimod.normalize(roimod.translate(self._roi, float(x) - bx, float(y) - by))

    def translate(self, dx: Any, dy: Any) -> None:
        self._roi = roimod.normalize(roimod.translate(self._roi, float(dx), float(dy)))

    def nudge(self, key: Any) -> None:
        pass

    def getImage(self) -> ImagePlus | None:
        return self._imp

    def setImage(self, imp: Any) -> None:
        self._imp = imp if isinstance(imp, ImagePlus) else (ImagePlus(_img=imp) if imp is not None else None)

    # --- appearance
    def getName(self) -> str | None:
        return self._roi.get("name")

    def setName(self, name: Any) -> None:
        if name is None:
            self._roi.pop("name", None)
        else:
            self._roi["name"] = str(name)

    name = property(getName, setName)

    def setStrokeColor(self, c: Any) -> None:
        if c is None:
            self._roi.pop("color", None)
        else:
            self._roi["color"] = color_hex(c)

    setColor = setStrokeColor

    def getStrokeColor(self) -> Color | None:
        c = self._roi.get("color")
        return Color(c) if c else None

    getColor = getStrokeColor

    def setFillColor(self, c: Any) -> None:
        if c is None:
            self._roi.pop("fill", None)
        else:
            self._roi["fill"] = color_hex(c)

    def getFillColor(self) -> Color | None:
        c = self._roi.get("fill")
        return Color(c) if c else None

    def setStrokeWidth(self, w: Any) -> None:
        self._roi["stroke_width"] = float(w)

    def getStrokeWidth(self) -> float:
        return float(self._roi.get("stroke_width", 1.0))

    def setLineWidth(self, w: Any) -> None:
        self.setStrokeWidth(w)

    def setStroke(self, s: Any) -> None:
        pass

    def setCornerDiameter(self, d: Any) -> None:
        self._roi["corner_diameter"] = float(d)

    def getCornerDiameter(self) -> int:
        return int(self._roi.get("corner_diameter", 0))

    def setPosition(self, *args: Any) -> None:
        """``setPosition(n)`` (stack slice, 1-based; 0 = all) / ``setPosition(c, z, t)``."""
        if len(args) == 1:
            n = int(args[0])
            if n <= 0:
                self._roi.pop("position", None)
                return
            img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
            if img is None:
                self._roi["position"] = {"t": 0, "z": n - 1, "c": 0}
                return
            from ._images import _decode_slice
            t, z, c = _decode_slice(img, n)
            self._roi["position"] = {"t": t, "z": z, "c": c}
            return
        c, z, t = (int(v) for v in args[:3])
        self._roi["position"] = {"t": max(t - 1, 0), "z": max(z - 1, 0), "c": max(c - 1, 0)}

    def getPosition(self) -> int:
        pos = self._roi.get("position")
        if not pos:
            return 0
        img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
        if img is None:
            return int(pos.get("z", 0)) + 1
        from ._images import _encode_slice
        return _encode_slice(img, int(pos.get("t", 0)), int(pos.get("z", 0)), int(pos.get("c", 0)))

    def getCPosition(self) -> int:
        pos = self._roi.get("position")
        return int(pos.get("c", 0)) + 1 if pos else 0

    def getZPosition(self) -> int:
        pos = self._roi.get("position")
        return int(pos.get("z", 0)) + 1 if pos else 0

    def getTPosition(self) -> int:
        pos = self._roi.get("position")
        return int(pos.get("t", 0)) + 1 if pos else 0

    def hasHyperStackPosition(self) -> bool:
        return bool(self._roi.get("position"))

    def setGroup(self, g: Any) -> None:
        self._roi["group"] = int(g)

    def getGroup(self) -> int:
        return int(self._roi.get("group", 0))

    def setProperty(self, k: Any, v: Any) -> None:
        self._roi.setdefault("properties", {})[str(k)] = v

    def getProperty(self, k: Any) -> Any:
        return self._roi.get("properties", {}).get(str(k))

    def getProperties(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self._roi.get("properties", {}).items())

    def setProperties(self, s: Any) -> None:
        props: dict[str, Any] = {}
        for line in str(s).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                props[k.strip()] = v.strip()
        self._roi["properties"] = props

    def copyAttributes(self, other: "Roi") -> None:
        for k in ("name", "color", "fill", "stroke_width", "position", "group", "properties"):
            if k in other._roi:
                self._roi[k] = other._roi[k]

    def setIsCursor(self, b: Any) -> None:
        pass

    def setIgnoreClipRect(self, b: Any) -> None:
        pass

    def getAngle(self, x1: Any, y1: Any, x2: Any, y2: Any) -> float:
        return math.degrees(math.atan2(float(y1) - float(y2), float(x2) - float(x1)))

    @staticmethod
    def setDefaultColor(c: Any) -> None:
        Roi._default_color = color_hex(c)

    @staticmethod
    def getDefaultColor() -> Color:
        return Color(Roi._default_color)

    @staticmethod
    def setDefaultFillColor(c: Any) -> None:
        pass

    @staticmethod
    def getDefaultFillColor() -> None:
        return None

    @staticmethod
    def setDefaultStrokeWidth(w: Any) -> None:
        pass

    @staticmethod
    def getDefaultStrokeWidth() -> float:
        return 1.0


class OvalRoi(Roi):
    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if len(args) == 1 and isinstance(args[0], Rectangle):
            r = args[0]
            args = (r.x, r.y, r.width, r.height)
        if len(args) < 4:
            raise TypeError("OvalRoi(x, y, width, height)")
        x, y, w, h = (float(v) for v in args[:4])
        super().__init__(_roi={"type": "oval", "x": x, "y": y, "w": w, "h": h})


class EllipseRoi(Roi):
    """``EllipseRoi(x1, y1, x2, y2, aspectRatio)`` – stored as a polygon like the macro ``makeEllipse``."""

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        x1, y1, x2, y2, aspect = (float(v) for v in args[:5])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a = math.hypot(x2 - x1, y2 - y1) / 2
        b = a * aspect
        ang = math.atan2(y2 - y1, x2 - x1)
        pts = []
        for i in range(72):
            th = 2 * math.pi * i / 72
            ex, ey = a * math.cos(th), b * math.sin(th)
            pts.append([cx + ex * math.cos(ang) - ey * math.sin(ang), cy + ex * math.sin(ang) + ey * math.cos(ang)])
        super().__init__(_roi={"type": "polygon", "points": pts})


class PolygonRoi(Roi):
    """``PolygonRoi(xs, ys, n, type)`` / ``PolygonRoi(xs, ys, type)`` / ``PolygonRoi(FloatPolygon | Polygon, type)``;
    ``type`` is ``Roi.POLYGON`` / ``FREEROI`` / ``TRACED_ROI`` / ``POLYLINE`` / ``FREELINE`` / ``ANGLE``."""

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if not args:
            raise TypeError("PolygonRoi(xs, ys, n, type) or PolygonRoi(polygon, type)")
        if isinstance(args[0], Polygon):
            xs, ys = list(args[0].xpoints), list(args[0].ypoints)
            kind = int(args[1]) if len(args) > 1 else Roi.POLYGON
        else:
            xs, ys = list(args[0]), list(args[1])
            if len(args) >= 4:
                n, kind = int(args[2]), int(args[3])
                xs, ys = xs[:n], ys[:n]
            else:
                kind = int(args[2]) if len(args) > 2 else Roi.POLYGON
        t = {Roi.POLYGON: "polygon", Roi.FREEROI: "freehand", Roi.TRACED_ROI: "polygon", Roi.POLYLINE: "polyline", Roi.FREELINE: "polyline", Roi.ANGLE: "polyline",
             Roi.POINT: "point"}.get(kind, "polygon")
        pts = [[float(x), float(y)] for x, y in zip(xs, ys)]
        if t == "polyline" and len(pts) == 2:
            super().__init__(_roi={"type": "line", "x1": pts[0][0], "y1": pts[0][1], "x2": pts[1][0], "y2": pts[1][1]})
        else:
            super().__init__(_roi={"type": t, "points": pts})
        self._kind = kind

    def getType(self) -> int:
        k = getattr(self, "_kind", None)
        return k if k is not None else super().getType()

    def fitSpline(self, *a: Any) -> None:
        ctx = current()
        saved = ctx.roi
        try:
            ctx.roi = self._roi
            out = ctx.ij._run("selection.fit_spline", {})
            if out.get("roi") is not None:
                self._roi = roimod.normalize(out["roi"])
        finally:
            ctx.roi = saved

    def removeSplineFit(self) -> None:
        pass

    def isSplineFit(self) -> bool:
        return False

    def getUncalibratedLength(self) -> float:
        return self.getRawLength()

    def getAngle(self, *args: Any) -> float:  # type: ignore[override]
        if args:
            return super().getAngle(*args)
        pts = self._points()
        if len(pts) < 3:
            return 0.0
        (x1, y1), (x2, y2), (x3, y3) = pts[0], pts[1], pts[2]
        a1 = math.degrees(math.atan2(y1 - y2, x1 - x2))
        a2 = math.degrees(math.atan2(y3 - y2, x3 - x2))
        return abs(a1 - a2) if abs(a1 - a2) <= 180 else 360 - abs(a1 - a2)

    def addPoint(self, x: Any, y: Any) -> None:
        if self._roi.get("type") == "line":
            self._roi = roimod.normalize({"type": "polyline", "points": [[self._roi["x1"], self._roi["y1"]], [self._roi["x2"], self._roi["y2"]], [float(x), float(y)]]})
        else:
            self._roi.setdefault("points", []).append([float(x), float(y)])

    def getInterpolatedPolygon(self, interval: Any = 1.0, smooth: Any = True) -> FloatPolygon:
        pts = self._points()
        if len(pts) < 2:
            return self.getFloatPolygon()
        closed = self.isArea()
        seq = pts + ([pts[0]] if closed else [])
        xs, ys = [], []
        step = float(interval) if interval else 1.0
        for (x1, y1), (x2, y2) in zip(seq[:-1], seq[1:]):
            d = math.hypot(x2 - x1, y2 - y1)
            n = max(1, int(d / step))
            for i in range(n):
                f = i / n
                xs.append(x1 + f * (x2 - x1))
                ys.append(y1 + f * (y2 - y1))
        return FloatPolygon(xs, ys)


class Line(Roi):
    """``Line(x1, y1, x2, y2)``; ``line.x1d`` … are the float end points, ``line.x1`` … their ints."""

    _width = 1.0
    _js_props = ("x1d", "y1d", "x2d", "y2d", "x1", "y1", "x2", "y2", "name")

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if len(args) < 4:
            raise TypeError("Line(x1, y1, x2, y2)")
        x1, y1, x2, y2 = (float(v) for v in args[:4])
        super().__init__(_roi={"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": Line._width})

    x1d = property(lambda self: float(self._roi["x1"]))
    y1d = property(lambda self: float(self._roi["y1"]))
    x2d = property(lambda self: float(self._roi["x2"]))
    y2d = property(lambda self: float(self._roi["y2"]))
    x1 = property(lambda self: int(round(self._roi["x1"])))
    y1 = property(lambda self: int(round(self._roi["y1"])))
    x2 = property(lambda self: int(round(self._roi["x2"])))
    y2 = property(lambda self: int(round(self._roi["y2"])))

    def getAngle(self, *args: Any) -> float:  # type: ignore[override]
        if args:
            return super().getAngle(*args)
        pw, ph = self._cal()
        return roimod.line_angle(self._roi, pw, ph)

    def getPixels(self) -> list[float]:
        img = self._imp._img if self._imp is not None and self._imp._img is not None else _active_img()
        if img is None:
            return []
        from ._images import ImageProcessor
        ip = ImageProcessor._for(img, follow=True)
        return ip.getLine(self.x1d, self.y1d, self.x2d, self.y2d)

    def getPoints(self) -> list[Point]:
        return [Point(self.x1, self.y1), Point(self.x2, self.y2)]

    def setStrokeWidth(self, w: Any) -> None:  # noqa: D102
        self._roi["width"] = float(w)
        self._roi["stroke_width"] = float(w)

    def getStrokeWidth(self) -> float:
        return float(self._roi.get("width", 1.0))

    @staticmethod
    def setWidth(w: Any) -> None:
        Line._width = float(w)

    @staticmethod
    def getWidth() -> int:
        return int(Line._width)


class PointRoi(Roi):
    """``PointRoi(x, y)`` / ``PointRoi(xs, ys[, n])`` / ``PointRoi(FloatPolygon | Polygon)``."""

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if len(args) == 1 and isinstance(args[0], Polygon):
            pts = [[float(x), float(y)] for x, y in zip(args[0].xpoints, args[0].ypoints)]
        elif len(args) >= 2 and isinstance(args[0], (list, tuple, np.ndarray)):
            xs, ys = list(args[0]), list(args[1])
            if len(args) > 2:
                xs, ys = xs[: int(args[2])], ys[: int(args[2])]
            pts = [[float(x), float(y)] for x, y in zip(xs, ys)]
        elif len(args) >= 2:
            pts = [[float(args[0]), float(args[1])]]
        else:
            raise TypeError("PointRoi(x, y) or PointRoi(xs, ys, n)")
        super().__init__(_roi={"type": "point", "points": pts})

    def addPoint(self, *args: Any) -> "PointRoi":
        if len(args) == 3:  # addPoint(imp, x, y)
            args = args[1:]
        self._roi.setdefault("points", []).append([float(args[0]), float(args[1])])
        return self

    def getCount(self, counter: Any = 0) -> int:
        return len(self._roi.get("points", []))

    def getNCounters(self) -> int:
        return 1

    def setPointType(self, t: Any) -> None:
        pass

    def setSize(self, s: Any) -> None:
        pass

    def getSize(self) -> int:
        return 2

    def setShowLabels(self, b: Any) -> None:
        pass

    def setCounter(self, c: Any) -> None:
        pass

    def getPointPosition(self, i: Any) -> int:
        return self.getPosition()

    @staticmethod
    def setDefaultSize(s: Any) -> None:
        pass

    @staticmethod
    def setDefaultType(t: Any) -> None:
        pass


class TextRoi(Roi):
    """``TextRoi(x, y, text[, font])``: a text overlay item (``imp.setOverlay`` / ``Overlay.add``)."""

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if len(args) < 3:
            raise TypeError("TextRoi(x, y, text[, font])")
        x, y, text = float(args[0]), float(args[1]), str(args[2])
        size = int(getattr(args[3], "size", 14)) if len(args) > 3 and args[3] is not None else 14
        self._imp = None
        self._roi = {"id": roimod.new_id(), "type": "text", "x": x, "y": y, "text": text, "font_size": size, "color": Roi._default_color, "background": False}

    def _bounds(self) -> tuple[float, float, float, float]:
        fs = float(self._roi.get("font_size", 14))
        return float(self._roi["x"]), float(self._roi["y"]), 0.6 * fs * len(self._roi.get("text", "")), 1.2 * fs

    def getText(self) -> str:
        return str(self._roi.get("text", ""))

    def setText(self, t: Any) -> None:
        self._roi["text"] = str(t)

    def setFont(self, *args: Any) -> None:
        for a in args:
            if hasattr(a, "size"):
                self._roi["font_size"] = int(a.size)
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                self._roi["font_size"] = int(a)

    def setFontSize(self, s: Any) -> None:
        self._roi["font_size"] = int(s)

    def setJustification(self, j: Any) -> None:
        self._roi["justify"] = int(j)

    def setAntialiased(self, b: Any) -> None:
        pass

    def isArea(self) -> bool:
        return False

    def getType(self) -> int:
        return Roi.RECTANGLE

    def getTypeAsString(self) -> str:
        return "Text"


class ShapeRoi(Roi):
    """``ShapeRoi(roi)``: a composite (mask) ROI with ``or`` / ``and`` / ``xor`` / ``not`` set operations."""

    def __init__(self, *args: Any, _roi: dict[str, Any] | None = None) -> None:
        if _roi is not None:
            super().__init__(_roi=_roi)
            return
        if len(args) == 1 and isinstance(args[0], Roi):
            super().__init__(_roi=dict(args[0]._roi))
        elif len(args) == 1 and isinstance(args[0], dict):
            super().__init__(_roi=args[0])
        else:
            raise TypeError("ShapeRoi(roi)")

    def _combine(self, other: Roi, mode: str) -> "ShapeRoi":
        out = roimod.combine([self._roi, other._roi], self._shape(), mode)
        if out is None:
            out = {"type": "rect", "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._roi = roimod.normalize(out)
        return self

    def or_(self, other: Roi) -> "ShapeRoi":
        return self._combine(other, "or")

    def and_(self, other: Roi) -> "ShapeRoi":
        return self._combine(other, "and")

    def xor(self, other: Roi) -> "ShapeRoi":
        return self._combine(other, "xor")

    def not_(self, other: Roi) -> "ShapeRoi":
        m = roimod.to_mask(self._roi, self._shape()) & ~roimod.to_mask(other._roi, self._shape())
        out = roimod.from_mask(m)
        self._roi = roimod.normalize(out) if out is not None else {"type": "rect", "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        return self

    def getRois(self) -> list[Roi]:
        from skimage.measure import label as sk_label
        m = roimod.to_mask(self._roi, self._shape())
        return [Roi._wrap(r) for r in roimod.label_to_rois(sk_label(m, connectivity=2))]

    def shapeToRoi(self) -> Roi:
        rois = self.getRois()
        return rois[0] if len(rois) == 1 else self

    def getType(self) -> int:
        return Roi.COMPOSITE


setattr(ShapeRoi, "or", ShapeRoi.or_)
setattr(ShapeRoi, "and", ShapeRoi.and_)
setattr(ShapeRoi, "not", ShapeRoi.not_)


class Overlay:
    """``ij.gui.Overlay``: a list of ROIs.  ``imp.getOverlay()`` returns a live view of the image's overlay,
    ``Overlay()`` a free-standing one for ``imp.setOverlay(overlay)``."""

    def __init__(self, roi: Roi | None = None, *, _img: Any = None) -> None:
        self._img = _img
        self._list: list[dict[str, Any]] = []
        self._label_color = "#ffffff"
        if roi is not None:
            self.add(roi)

    # --- storage
    def _items(self) -> list[dict[str, Any]]:
        return self._img.overlay if self._img is not None else self._list

    def _touch(self) -> None:
        if self._img is not None:
            self._img.rev += 1
            ctx = current()
            if self._img.id in ctx.session.images:
                ctx.session._changed("images")

    def add(self, roi: Roi, name: Any = None) -> None:
        d = dict(roi._roi)
        if name is not None:
            d["name"] = str(name)
        d.setdefault("color", Roi._default_color)
        self._items().append(d)
        self._touch()

    addElement = add

    def size(self) -> int:
        return len(self._items())

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[Roi]:
        return iter(self.toArray())

    def iterator(self) -> Iterator[Roi]:
        return iter(self.toArray())

    def get(self, key: Any) -> Roi | None:
        items = self._items()
        if isinstance(key, str):
            for d in items:
                if d.get("name") == key:
                    return Roi._wrap(d)
            return None
        i = int(key)
        return Roi._wrap(items[i]) if 0 <= i < len(items) else None

    def getIndex(self, name: Any) -> int:
        for i, d in enumerate(self._items()):
            if d.get("name") == str(name):
                return i
        return -1

    def contains(self, roi: Roi) -> bool:
        return any(d.get("id") == roi._roi.get("id") for d in self._items())

    def indexOf(self, roi: Roi) -> int:
        for i, d in enumerate(self._items()):
            if d.get("id") == roi._roi.get("id"):
                return i
        return -1

    def remove(self, key: Any) -> None:
        items = self._items()
        if isinstance(key, Roi):
            items[:] = [d for d in items if d.get("id") != key._roi.get("id")]
        elif isinstance(key, str):
            items[:] = [d for d in items if d.get("name") != key]
        else:
            i = int(key)
            if 0 <= i < len(items):
                del items[i]
        self._touch()

    def set(self, roi: Roi, i: Any) -> None:  # noqa: A003
        items = self._items()
        items[int(i)] = dict(roi._roi)
        self._touch()

    def clear(self) -> None:
        self._items()[:] = []
        self._touch()

    def toArray(self) -> list[Roi]:
        return [Roi._wrap(d) for d in self._items() if isinstance(d, dict) and "type" in d]

    def duplicate(self) -> "Overlay":
        o = Overlay()
        o._list = [dict(d) for d in self._items()]
        return o

    def crop(self, rect: Any) -> None:
        pass

    def translate(self, dx: Any, dy: Any) -> None:
        items = self._items()
        items[:] = [roimod.normalize(roimod.translate(d, float(dx), float(dy))) if d.get("type") != "text" else {**d, "x": d["x"] + float(dx), "y": d["y"] + float(dy)} for d in items]
        self._touch()

    # --- appearance
    def setStrokeColor(self, c: Any) -> None:
        for d in self._items():
            d["color"] = color_hex(c)
        self._touch()

    def setFillColor(self, c: Any) -> None:
        for d in self._items():
            d["fill"] = color_hex(c)
        self._touch()

    def setStrokeWidth(self, w: Any) -> None:
        for d in self._items():
            d["stroke_width"] = float(w)
        self._touch()

    def drawLabels(self, b: Any) -> None:
        pass

    def drawNames(self, b: Any) -> None:
        pass

    def drawBackgrounds(self, b: Any) -> None:
        pass

    def setLabelColor(self, c: Any) -> None:
        self._label_color = color_hex(c)

    def getLabelColor(self) -> Color:
        return Color(self._label_color)

    def setLabelFont(self, *a: Any) -> None:
        pass

    def setLabelFontSize(self, *a: Any) -> None:
        pass

    def getDrawLabels(self) -> bool:
        return False

    def setIsCalibrationBar(self, b: Any) -> None:
        pass

    def selectable(self, b: Any) -> None:
        pass

    def isSelectable(self) -> bool:
        return True

    def hide(self) -> None:
        pass

    def show(self) -> None:
        pass

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"Overlay[size={self.size()}]"
