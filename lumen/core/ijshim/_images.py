"""``ij.ImagePlus``, ``ij.ImageStack``, ``ij.process.ImageProcessor`` (+ Byte/Short/Float/ColorProcessor),
``ij.measure.Calibration`` and ``ij.process.ImageStatistics`` over a :class:`~lumen.core.image.LumenImage`.

An ``ImagePlus`` wraps a LumenImage.  Images created by a script (``ImagePlus(title, ip)``, ``IJ.createImage``,
``IJ.openImage``, ``imp.duplicate()``) stay private until ``imp.show()`` – or until a command needs them in the
session (``IJ.run(imp, …)``), which adds them as a tab like ImageJ would show a window.  ``imp.close()`` /
``hide()`` remove the tab again.

Processors are *views*: an attached processor reads and writes the plane of its image directly (numpy), so
``ip.putPixel`` / ``ip.getPixels()`` behave like ImageJ's backing arrays; ``imp.updateAndDraw()`` refreshes the
viewport.  Filters on processors (``ip.blurGaussian`` …) run the matching Lumen command on that plane only.
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Iterator

import numpy as np

from .. import roi as roimod
from ..image import Calibration as LumenCalibration, LumenImage
from ..registry import cast_to
from ._context import Context, current
from ._java import Color, Rectangle, color_hex

NO_THRESHOLD = -808080.0


def _img_of(obj: Any) -> LumenImage:
    return obj._img if isinstance(obj, ImagePlus) else obj


def in_session(ctx: Context, img: LumenImage) -> bool:
    return img.id in ctx.session.images


def ensure_in_session(ctx: Context, img: LumenImage, activate: bool = False) -> LumenImage:
    """Add a script-private image to the session (as a tab) if it is not there yet."""
    if img.id not in ctx.session.images:
        ctx.session.add_image(img, activate=activate)
    elif activate:
        ctx.session.select(img.id)
    return img


@contextlib.contextmanager
def as_current(ctx: Context, imp: "ImagePlus | LumenImage | None") -> Iterator[LumenImage | None]:
    """Run a macro-layer command *on* ``imp`` (ImageJ's ``IJ.run(imp, …)``): the image becomes the session's active
    image for the call; the previously active image is restored afterwards unless the command itself switched
    (created or selected) another image."""
    s = ctx.session
    if imp is None:
        yield None
        return
    img = _img_of(imp)
    prev = s.active_id
    ensure_in_session(ctx, img, activate=True)
    try:
        yield img
    finally:
        if s.active_id == img.id and prev and prev != img.id and prev in s.images:
            s.select(prev)


@contextlib.contextmanager
def temporarily_in_session(ctx: Context, img: LumenImage) -> Iterator[LumenImage]:
    """A script-private image is added for the duration of a command and removed again (used for processor
    filters on standalone processors / un-shown images)."""
    if img.id in ctx.session.images:
        yield img
        return
    s = ctx.session
    prev = s.active_id
    s.add_image(img, activate=False)
    try:
        yield img
    finally:
        s.images.pop(img.id, None)
        s.histories.pop(img.id, None)
        s.thresholds.pop(img.id, None)
        if s.active_id == img.id:
            s.active_id = prev if prev in s.images else (next(reversed(s.images)) if s.images else None)
        s._changed("images")


def _bit_depth(img: LumenImage) -> int:
    if img.rgb:
        return 24
    return {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16, np.dtype(np.float32): 32}.get(img.data.dtype, 32)


def _decode_slice(img: LumenImage, n: int) -> tuple[int, int, int]:
    """1-based ImageJ stack index → (t, z, c); channels vary fastest (RGB counts as one channel)."""
    nc = 1 if img.rgb else img.nc
    total = img.nt * img.nz * nc
    if not 1 <= n <= total:
        raise ValueError(f"Stack index {n} out of range (1–{total})")
    i = n - 1
    c = i % nc
    z = (i // nc) % img.nz
    t = i // (nc * img.nz)
    return t, z, (0 if img.rgb else c)


def _encode_slice(img: LumenImage, t: int, z: int, c: int) -> int:
    nc = 1 if img.rgb else img.nc
    return (t * img.nz + z) * nc + (0 if img.rgb else c) + 1


def _pack_rgb(plane: np.ndarray) -> np.ndarray:
    """(3, H, W) uint8 → packed int32 0xRRGGBB."""
    r, g, b = (plane[i].astype(np.int32) for i in range(3))
    return (r << 16) | (g << 8) | b


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed).astype(np.int64)
    return np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255]).astype(np.uint8)


# ====================================================================== Calibration
class Calibration:
    """``ij.measure.Calibration``: ``pixelWidth`` / ``pixelHeight`` / ``pixelDepth`` / ``frameInterval`` and the
    unit accessors.  ``imp.getCalibration()`` returns a *live* view (changes apply to the image at once);
    ``Calibration()`` is a free-standing one for ``imp.setCalibration(cal)``."""

    NONE = 20
    STRAIGHT_LINE = 0
    _js_props = ("pixelWidth", "pixelHeight", "pixelDepth", "frameInterval", "fps", "xOrigin", "yOrigin", "zOrigin", "unit", "units", "info")

    def __init__(self, imp: Any = None) -> None:
        img = _img_of(imp) if imp is not None else None
        self._img = img
        self._cal = img.calibration if img is not None else LumenCalibration()
        self._origin = (img.metadata.get("origin") if img is not None else None) or {}
        self.fps = 0.0
        self.info = ""

    # --- field-style access (Jython: cal.pixelWidth = 0.5)
    @property
    def pixelWidth(self) -> float:
        return float(self._cal.pixel_width)

    @pixelWidth.setter
    def pixelWidth(self, v: Any) -> None:
        self._cal.pixel_width = float(v)
        self._touch()

    @property
    def pixelHeight(self) -> float:
        return float(self._cal.pixel_height)

    @pixelHeight.setter
    def pixelHeight(self, v: Any) -> None:
        self._cal.pixel_height = float(v)
        self._touch()

    @property
    def pixelDepth(self) -> float:
        return float(self._cal.pixel_depth)

    @pixelDepth.setter
    def pixelDepth(self, v: Any) -> None:
        self._cal.pixel_depth = float(v)
        self._touch()

    @property
    def frameInterval(self) -> float:
        return float(self._cal.frame_interval)

    @frameInterval.setter
    def frameInterval(self, v: Any) -> None:
        self._cal.frame_interval = float(v)
        self._touch()

    @property
    def unit(self) -> str:
        return self.getUnit()

    @unit.setter
    def unit(self, v: Any) -> None:
        self.setUnit(v)

    @property
    def units(self) -> str:
        return self.getUnits()

    def _origin_get(self, k: str) -> float:
        return float(self._origin.get(k, 0.0) or 0.0)

    def _origin_set(self, k: str, v: Any) -> None:
        self._origin[k] = float(v)
        if self._img is not None:
            self._img.metadata["origin"] = dict(self._origin)
            self._touch()

    xOrigin = property(lambda self: self._origin_get("x"), lambda self, v: self._origin_set("x", v))
    yOrigin = property(lambda self: self._origin_get("y"), lambda self, v: self._origin_set("y", v))
    zOrigin = property(lambda self: self._origin_get("z"), lambda self, v: self._origin_set("z", v))

    def _touch(self) -> None:
        if self._img is not None:
            self._img.rev += 1

    # --- methods
    def getUnit(self) -> str:
        u = self._cal.unit
        return "pixel" if u in ("", "pixel", "pixels", "px") else u

    def getUnits(self) -> str:
        u = self.getUnit()
        return "pixels" if u == "pixel" else u

    def setUnit(self, u: Any) -> None:
        u = str(u)
        self._cal.unit = "pixel" if u in ("", "pixel", "pixels", "px") else u
        self._touch()

    def getXUnit(self) -> str:
        return self.getUnit()

    def getYUnit(self) -> str:
        return self.getUnit()

    def getZUnit(self) -> str:
        return self.getUnit()

    def setXUnit(self, u: Any) -> None:
        self.setUnit(u)

    def setYUnit(self, u: Any) -> None:
        pass

    def setZUnit(self, u: Any) -> None:
        pass

    def getTimeUnit(self) -> str:
        return self._cal.time_unit or "sec"

    def setTimeUnit(self, u: Any) -> None:
        self._cal.time_unit = str(u)
        self._touch()

    def getValueUnit(self) -> str:
        return self._cal.value_unit or "Gray Value"

    def setValueUnit(self, u: Any) -> None:
        self._cal.value_unit = str(u)
        self._touch()

    def scaled(self) -> bool:
        return self.getUnit() != "pixel" or self.pixelWidth != 1.0 or self.pixelHeight != 1.0

    def calibrated(self) -> bool:
        return False

    def getX(self, x: Any) -> float:
        return (float(x) - self.xOrigin) * self.pixelWidth

    def getY(self, y: Any, height: Any = None) -> float:
        return (float(y) - self.yOrigin) * self.pixelHeight

    def getZ(self, z: Any) -> float:
        return (float(z) - self.zOrigin) * self.pixelDepth

    def getRawX(self, x: Any) -> float:
        return float(x) / self.pixelWidth + self.xOrigin

    def getRawY(self, y: Any, height: Any = None) -> float:
        return float(y) / self.pixelHeight + self.yOrigin

    def getRawZ(self, z: Any) -> float:
        return float(z) / self.pixelDepth + self.zOrigin

    def getCValue(self, v: Any) -> float:
        return float(v)

    def getRawValue(self, v: Any) -> float:
        return float(v)

    def getCTable(self) -> None:
        return None

    def getCoefficients(self) -> None:
        return None

    def getFunction(self) -> int:
        return self.NONE

    def setFunction(self, *a: Any) -> None:
        pass

    def disableDensityCalibration(self) -> None:
        pass

    def setImage(self, imp: Any) -> None:
        pass

    def setInvertY(self, b: Any) -> None:
        pass

    def getInvertY(self) -> bool:
        return False

    def copy(self) -> "Calibration":
        c = Calibration()
        c._cal = LumenCalibration(**self._cal.to_dict())
        c._origin = dict(self._origin)
        c.fps = self.fps
        return c

    clone = copy

    def equals(self, other: Any) -> bool:
        return isinstance(other, Calibration) and other._cal.to_dict() == self._cal.to_dict()

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return (f"w={self.pixelWidth}, h={self.pixelHeight}, d={self.pixelDepth}, unit={self.getUnit()}, "
                f"f={self.frameInterval}, tunit={self.getTimeUnit()}")


# ====================================================================== statistics
_STAT_FIELDS = ("mean", "min", "max", "stdDev", "area", "pixelCount", "histogram", "median", "mode", "dmode", "xCentroid", "yCentroid",
                "xCenterOfMass", "yCenterOfMass", "skewness", "kurtosis", "umean", "major", "minor", "angle", "roiX", "roiY", "roiWidth",
                "roiHeight", "nBins", "histMin", "histMax", "binSize", "areaFraction", "longPixelCount", "longHistogram", "histogram16")


class ImageStatistics:
    """``ij.process.ImageStatistics`` for one plane (optionally inside an area ROI): ImageJ's field names (``mean``,
    ``stdDev``, ``pixelCount``, ``histogram``, ``xCentroid`` …) computed with Lumen's measurement code."""

    _js_props = _STAT_FIELDS

    def __init__(self, img: LumenImage | None = None, roi: dict[str, Any] | None = None, t: int = 0, z: int = 0, c: int = 0,
                 plane: np.ndarray | None = None, n_bins: int = 256, hist_min: float | None = None, hist_max: float | None = None) -> None:
        if plane is None:
            if img is None:
                raise ValueError("ImageStatistics needs an image or a plane")
            plane = img.data[t, z] if img.rgb else img.data[t, z, c]
        area_roi = roi if roimod.is_area(roi) else None
        if plane.ndim == 3:  # RGB: ImageJ measures the (unweighted) grey value
            gray = plane.astype(np.float32).mean(axis=0)
        else:
            gray = plane
        h, w = gray.shape
        mask = roimod.to_mask(area_roi, (h, w)) if area_roi is not None else None
        vals = (gray[mask] if mask is not None else gray.ravel()).astype(np.float64)
        if vals.size == 0:
            vals = np.zeros(1)
        cal = img.calibration if img is not None else LumenCalibration()
        pw, ph = float(cal.pixel_width), float(cal.pixel_height)
        self.pixelCount = int(vals.size)
        self.longPixelCount = self.pixelCount
        self.area = self.pixelCount * pw * ph
        self.mean = float(vals.mean())
        self.umean = self.mean
        self.min = float(vals.min())
        self.max = float(vals.max())
        self.stdDev = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        self.median = float(np.median(vals))
        # histogram like getHistogram(): 256 bins over 0–255 for 8-bit, else over the (given) range
        if gray.dtype == np.uint8:
            counts = np.bincount(vals.astype(np.int64), minlength=256)[:256]
            self.nBins, self.histMin, self.histMax, self.binSize = 256, 0.0, 256.0, 1.0
            values = np.arange(256, dtype=np.float64)
        else:
            lo = float(vals.min()) if hist_min is None else float(hist_min)
            hi = float(vals.max()) if hist_max is None else float(hist_max)
            if hi <= lo:
                hi = lo + 1.0
            n_bins = int(n_bins) if n_bins else 256
            counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
            self.nBins, self.histMin, self.histMax, self.binSize = n_bins, lo, hi, (hi - lo) / n_bins
            values = edges[:-1]
        self.histogram = [int(v) for v in counts]
        self.longHistogram = self.histogram
        self.histogram16 = [int(v) for v in np.bincount(vals.astype(np.int64), minlength=65536)[:65536]] if gray.dtype == np.uint16 else None
        imode = int(np.argmax(counts))
        self.dmode = float(values[imode]) if gray.dtype == np.uint8 else float(values[imode] + self.binSize / 2.0)
        self.mode = self.dmode
        # moments
        d = vals - self.mean
        var = float(d.var()) if vals.size > 1 else 0.0
        self.skewness = float((d ** 3).mean() / var ** 1.5) if var > 0 else 0.0
        self.kurtosis = float((d ** 4).mean() / var ** 2 - 3.0) if var > 0 else 0.0
        # geometry (calibrated, like the Results table)
        ys, xs = np.nonzero(mask) if mask is not None else (None, None)
        if mask is not None and xs is not None and xs.size:
            bx, by, bw, bh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            cx, cy = float(xs.mean() + 0.5), float(ys.mean() + 0.5)
            wsum = float(vals.sum())
            mx, my = (float((xs * vals).sum() / wsum + 0.5), float((ys * vals).sum() / wsum + 0.5)) if wsum else (cx, cy)
        else:
            bx, by, bw, bh = 0, 0, w, h
            cx, cy = w / 2.0, h / 2.0
            yy, xx = np.mgrid[0:h, 0:w]
            wsum = float(vals.sum())
            mx, my = (float((xx.ravel() * vals).sum() / wsum + 0.5), float((yy.ravel() * vals).sum() / wsum + 0.5)) if wsum else (cx, cy)
        self.roiX, self.roiY, self.roiWidth, self.roiHeight = bx, by, bw, bh
        self.xCentroid, self.yCentroid = cx * pw, cy * ph
        self.xCenterOfMass, self.yCenterOfMass = mx * pw, my * ph
        self.major = self.minor = self.angle = 0.0
        self.areaFraction = 100.0 * float((vals != 0).mean())
        if img is not None and area_roi is not None:
            try:
                from ..measure import measure_roi
                row = measure_roi(img, area_roi, ["fit_ellipse"], t=t, z=z, c=c)[0]
                self.major, self.minor, self.angle = float(row.get("Major", 0.0)), float(row.get("Minor", 0.0)), float(row.get("Angle", 0.0))
            except Exception:  # pragma: no cover - ellipse fit is best effort
                pass
        self._stats_roi = area_roi

    def getHistogram(self) -> list[int]:
        return list(self.histogram)

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"stats[count={self.pixelCount}, mean={self.mean:.4f}, min={self.min}, max={self.max}]"

    @staticmethod
    def getStatistics(ip: Any, m_options: Any = None, cal: Any = None) -> "ImageStatistics":
        """``ImageStatistics.getStatistics(ip[, mOptions, cal])`` – the statistics of a processor (and its ROI)."""
        return ip.getStatistics()


class StackStatistics(ImageStatistics):
    """``ij.process.StackStatistics``: statistics over every plane of the current channel (or all planes)."""

    def __init__(self, imp: Any, n_bins: int = 256, hist_min: float | None = None, hist_max: float | None = None) -> None:
        img = _img_of(imp)
        c = img.display.c
        data = img.data[:, :, :, :, :] if img.rgb else img.data[:, :, c]
        if img.rgb:
            plane = data.astype(np.float32).mean(axis=2).reshape(-1, img.width)
        else:
            plane = data.reshape(-1, img.width)
        super().__init__(img=None, plane=plane, n_bins=n_bins, hist_min=hist_min, hist_max=hist_max)
        self.area = self.pixelCount * img.calibration.pixel_width * img.calibration.pixel_height


# ====================================================================== processors
class ImageProcessor:
    """``ij.process.ImageProcessor``: a 2-D plane.  Attached processors (``imp.getProcessor()``,
    ``stack.getProcessor(n)``) are live views of the image; the ``ByteProcessor`` … constructors make standalone
    ones.  Pixel access (``getPixel`` / ``putPixel`` / ``get`` / ``set`` / ``getf`` / ``setf`` / ``getPixels``)
    is direct numpy access; filters run Lumen commands on the plane."""

    # filter types / interpolation / threshold / LUT update modes (ImageJ constants)
    BLUR_MORE, FIND_EDGES, MEDIAN_FILTER, MIN, MAX, CONVOLVE = 0, 1, 2, 3, 4, 5
    NEAREST_NEIGHBOR, NONE, BILINEAR, BICUBIC = 0, 0, 1, 2
    ISODATA, ISODATA2 = 0, 1
    NO_THRESHOLD = NO_THRESHOLD
    NO_LUT_UPDATE, RED_LUT, BLACK_AND_WHITE_LUT, OVER_UNDER_LUT = 0, 1, 2, 3
    LEFT_JUSTIFY, CENTER_JUSTIFY, RIGHT_JUSTIFY = 0, 1, 2
    _js_props = ("width", "height")

    def __init__(self, width: Any = None, height: Any = None, pixels: Any = None, cm: Any = None, *, _img: LumenImage | None = None,
                 _pos: tuple[int, int, int] | None = None, _follow: bool = False, _array: np.ndarray | None = None, _rgb: bool = False) -> None:
        self._img = _img
        self._fixed = _pos
        self._follow = _follow
        self._arr = _array
        self._rgb = _rgb or (_img is not None and _img.rgb)
        self._roi: dict[str, Any] | None = None
        self._value: float = 255.0
        self._color: Color | None = None
        self._line_width = 1
        self._interp = 1
        self._min: float | None = None
        self._max: float | None = None
        self._threshold: tuple[float, float] | None = None
        self._snapshot: np.ndarray | None = None
        self._font_size = 12
        self._justify = 0
        self._bg = 0.0
        self._lut_inverted = False
        self._x, self._y = 0, 0
        if width is not None and _img is None and _array is None:
            self._init_standalone(int(width), int(height), pixels)

    # --- construction helpers
    _dtype: Any = np.uint8

    def _init_standalone(self, w: int, h: int, pixels: Any) -> None:
        if self._rgb:
            if pixels is None:
                self._arr = np.zeros((3, h, w), np.uint8)
            else:
                p = np.asarray(pixels)
                self._arr = _unpack_rgb(p.reshape(h, w)) if p.ndim < 3 or p.shape[0] != 3 else p.astype(np.uint8)
            return
        if pixels is None:
            self._arr = np.zeros((h, w), self._dtype)
        else:
            p = np.asarray(pixels)
            if p.ndim == 2 and p.shape == (w, h) and w != h:  # FloatProcessor(float[][]) is indexed [x][y]
                p = p.T
            elif p.ndim == 2 and p.shape == (w, h) and w == h and self._dtype == np.float32:
                p = p.T
            self._arr = cast_to(np.ascontiguousarray(p.reshape(h, w)), np.dtype(self._dtype))

    @classmethod
    def _for(cls, img: LumenImage, pos: tuple[int, int, int] | None = None, follow: bool = False) -> "ImageProcessor":
        kind = ColorProcessor if img.rgb else {np.dtype(np.uint8): ByteProcessor, np.dtype(np.uint16): ShortProcessor}.get(img.data.dtype, FloatProcessor)
        return kind(_img=img, _pos=pos, _follow=follow)

    @classmethod
    def _standalone(cls, arr: np.ndarray, rgb: bool = False) -> "ImageProcessor":
        if rgb:
            return ColorProcessor(_array=np.ascontiguousarray(arr), _rgb=True)
        kind = {np.dtype(np.uint8): ByteProcessor, np.dtype(np.uint16): ShortProcessor}.get(arr.dtype, FloatProcessor)
        return kind(_array=np.ascontiguousarray(arr))

    # --- plane access
    def _pos(self) -> tuple[int, int, int]:
        if self._follow and self._img is not None:
            d = self._img.display
            return d.t, d.z, (0 if self._img.rgb else d.c)
        return self._fixed or (0, 0, 0)

    def _plane(self) -> np.ndarray:
        if self._img is not None:
            t, z, c = self._pos()
            return self._img.data[t, z] if self._img.rgb else self._img.data[t, z, c]
        assert self._arr is not None
        return self._arr

    def _set_plane(self, arr: np.ndarray) -> None:
        """Replace the plane's pixels (same shape; dtype is cast)."""
        if self._img is not None:
            t, z, c = self._pos()
            if self._img.rgb:
                self._img.data[t, z] = cast_to(np.asarray(arr), np.dtype(np.uint8))
            else:
                self._img.data[t, z, c] = cast_to(np.asarray(arr), self._img.data.dtype)
            self._img.rev += 1
        else:
            assert self._arr is not None
            self._arr = cast_to(np.ascontiguousarray(arr), self._arr.dtype) if arr.dtype != self._arr.dtype else np.ascontiguousarray(arr)

    def _touch(self) -> None:
        if self._img is not None:
            self._img.rev += 1

    def _gray(self) -> np.ndarray:
        p = self._plane()
        return p.astype(np.float32).mean(axis=0) if self._rgb else p

    @property
    def width(self) -> int:
        return int(self._plane().shape[-1])

    @property
    def height(self) -> int:
        return int(self._plane().shape[-2])

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height

    def getPixelCount(self) -> int:
        return self.width * self.height

    def getBitDepth(self) -> int:
        if self._rgb:
            return 24
        return {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}.get(self._plane().dtype, 32)

    def isDefaultLut(self) -> bool:
        return True

    def isColorLut(self) -> bool:
        return False

    def isGrayscale(self) -> bool:
        return not self._rgb

    def isPseudoColorLut(self) -> bool:
        return False

    def isInvertedLut(self) -> bool:
        if self._img is not None and not self._rgb:
            t, z, c = self._pos()
            return bool(self._img.display.channels[c].inverted)
        return self._lut_inverted

    def invertLut(self) -> None:
        if self._img is not None and not self._rgb:
            t, z, c = self._pos()
            ch = self._img.display.channels[c]
            ch.inverted = not ch.inverted
            self._touch()
        else:
            self._lut_inverted = not self._lut_inverted

    def isBinary(self) -> bool:
        p = self._plane()
        return (not self._rgb) and p.dtype == np.uint8 and bool(np.isin(p, (0, 255)).all())

    def isSigned16Bit(self) -> bool:
        return False

    # --- pixels
    def _inside(self, x: Any, y: Any) -> tuple[int, int] | None:
        xi, yi = int(x), int(y)
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return xi, yi
        return None

    def getPixel(self, x: Any, y: Any = None) -> int | float:
        if y is None:  # flat index (ImageJ get(int i))
            return self.get(x)
        xy = self._inside(x, y)
        if xy is None:
            return 0
        xi, yi = xy
        p = self._plane()
        if self._rgb:
            return int((int(p[0, yi, xi]) << 16) | (int(p[1, yi, xi]) << 8) | int(p[2, yi, xi]))
        v = p[yi, xi]
        return float(v) if p.dtype == np.float32 else int(v)

    def get(self, x: Any, y: Any = None) -> int | float:
        if y is None:
            i = int(x)
            p = self._plane()
            if self._rgb:
                return int(_pack_rgb(p).ravel()[i])
            v = p.ravel()[i]
            return float(v) if p.dtype == np.float32 else int(v)
        return self.getPixel(x, y)

    def getf(self, x: Any, y: Any = None) -> float:
        if y is None:
            return float(self._gray().ravel()[int(x)])
        xy = self._inside(x, y)
        if xy is None:
            return 0.0
        return float(self._gray()[xy[1], xy[0]])

    def getPixelValue(self, x: Any, y: Any) -> float:
        xy = self._inside(x, y)
        if xy is None:
            return float("nan") if self._plane().dtype == np.float32 else 0.0
        return float(self._gray()[xy[1], xy[0]])

    def getInterpolatedPixel(self, x: Any, y: Any) -> float:
        from scipy.ndimage import map_coordinates
        return float(map_coordinates(self._gray().astype(np.float64), [[float(y)], [float(x)]], order=1, mode="nearest")[0])

    getInterpolatedValue = getInterpolatedPixel
    getBicubicInterpolatedPixel = getInterpolatedPixel

    def putPixel(self, x: Any, y: Any, value: Any) -> None:
        xy = self._inside(x, y)
        if xy is None:
            return
        xi, yi = xy
        p = self._plane()
        if self._rgb:
            if isinstance(value, (list, tuple)):
                p[:, yi, xi] = [int(v) & 255 for v in value[:3]]
            else:
                v = int(value)
                p[0, yi, xi], p[1, yi, xi], p[2, yi, xi] = (v >> 16) & 255, (v >> 8) & 255, v & 255
        elif p.dtype == np.float32:
            p[yi, xi] = float(value)
        else:
            info = np.iinfo(p.dtype)
            p[yi, xi] = min(max(int(round(float(value))), info.min), info.max)
        self._touch()

    def set(self, x: Any, y: Any, value: Any = None) -> None:  # noqa: A003
        if value is None:  # set(i, v)
            i, v = int(x), y
            p = self._plane()
            if self._rgb:
                yi, xi = divmod(i, self.width)
                self.putPixel(xi, yi, v)
                return
            flat = p.reshape(-1)
            flat[i] = cast_to(np.array(float(v)), p.dtype)
            self._touch()
            return
        self.putPixel(x, y, value)

    def setf(self, x: Any, y: Any, value: Any = None) -> None:
        self.set(x, y, value)

    def putPixelValue(self, x: Any, y: Any, value: Any) -> None:
        if self._rgb:
            v = max(0, min(255, int(round(float(value)))))
            self.putPixel(x, y, (v << 16) | (v << 8) | v)
        else:
            self.putPixel(x, y, value)

    def getPixels(self) -> Any:
        """The plane's pixel array: a flat numpy view for grey images (writes go through), packed ints for RGB."""
        p = self._plane()
        if self._rgb:
            return _pack_rgb(p).ravel()
        return p.reshape(-1)  # a view of the contiguous plane: pixels[i] = v writes through

    def getPixelsCopy(self) -> Any:
        return np.array(self.getPixels())

    def setPixels(self, pixels: Any, channel: Any = None) -> None:
        if channel is not None:  # ColorProcessor.setPixels(channelNumber, FloatProcessor)
            fp = channel
            self._plane()[int(pixels)] = cast_to(fp._plane(), np.dtype(np.uint8))
            self._touch()
            return
        arr = np.asarray(pixels)
        if self._rgb:
            self._set_plane(_unpack_rgb(arr.reshape(self.height, self.width)) if arr.ndim < 3 else arr)
        else:
            self._set_plane(arr.reshape(self.height, self.width))

    def getRow(self, x: Any, y: Any, data: Any, length: Any) -> None:
        row = self._gray()[int(y), int(x): int(x) + int(length)]
        data[: len(row)] = [float(v) for v in row]

    def getColumn(self, x: Any, y: Any, data: Any, length: Any) -> None:
        col = self._gray()[int(y): int(y) + int(length), int(x)]
        data[: len(col)] = [float(v) for v in col]

    def putRow(self, x: Any, y: Any, data: Any, length: Any) -> None:
        p = self._plane()
        n = int(length)
        if self._rgb:
            p[:, int(y), int(x): int(x) + n] = _unpack_rgb(np.asarray(data[:n]))
        else:
            p[int(y), int(x): int(x) + n] = cast_to(np.asarray(data[:n], np.float64), p.dtype)
        self._touch()

    def putColumn(self, x: Any, y: Any, data: Any, length: Any) -> None:
        p = self._plane()
        n = int(length)
        if self._rgb:
            p[:, int(y): int(y) + n, int(x)] = _unpack_rgb(np.asarray(data[:n]))
        else:
            p[int(y): int(y) + n, int(x)] = cast_to(np.asarray(data[:n], np.float64), p.dtype)
        self._touch()

    def getLine(self, x1: Any, y1: Any, x2: Any, y2: Any) -> list[float]:
        from skimage.measure import profile_line
        prof = profile_line(self._gray().astype(np.float64), (float(y1), float(x1)), (float(y2), float(x2)), linewidth=1, order=1, mode="constant")
        return [float(v) for v in prof]

    def getHistogram(self, n_bins: Any = None) -> list[int]:
        p = self._gray()
        if n_bins is None and p.dtype == np.uint8:
            vals = p[roimod.to_mask(self._roi, p.shape)] if self._roi is not None else p.ravel()
            return [int(v) for v in np.bincount(vals.astype(np.int64), minlength=256)[:256]]
        if n_bins is None and p.dtype == np.uint16:
            vals = p[roimod.to_mask(self._roi, p.shape)] if self._roi is not None else p.ravel()
            return [int(v) for v in np.bincount(vals.astype(np.int64), minlength=65536)[:65536]]
        return self.getStatistics().histogram if n_bins is None else ImageStatistics(plane=p, roi=self._roi, n_bins=int(n_bins)).histogram

    def getStatistics(self) -> ImageStatistics:
        if self._img is not None:
            t, z, c = self._pos()
            return ImageStatistics(self._img, self._roi, t, z, c)
        return ImageStatistics(plane=self._plane(), roi=self._roi)

    getStats = getStatistics

    def getStatsRoi(self) -> ImageStatistics:
        return self.getStatistics()

    def getMin(self) -> float:
        if self._min is not None:
            return self._min
        if self._img is not None and not self._rgb:
            return float(self._img.display.channels[self._pos()[2]].min)
        return float(self._gray().min()) if self._gray().size else 0.0

    def getMax(self) -> float:
        if self._max is not None:
            return self._max
        if self._img is not None and not self._rgb:
            return float(self._img.display.channels[self._pos()[2]].max)
        return float(self._gray().max()) if self._gray().size else 0.0

    def setMinAndMax(self, lo: Any, hi: Any) -> None:
        if self._img is not None and not self._rgb:
            ch = self._img.display.channels[self._pos()[2]]
            ch.min, ch.max = float(lo), float(hi)
            self._touch()
        else:
            self._min, self._max = float(lo), float(hi)

    def resetMinAndMax(self) -> None:
        if self._img is not None and not self._rgb:
            g = self._gray()
            ch = self._img.display.channels[self._pos()[2]]
            ch.min, ch.max = (float(g.min()), float(g.max())) if g.size else (0.0, 255.0)
            self._touch()
        self._min = self._max = None

    def findMinAndMax(self) -> None:
        pass

    def getBackgroundValue(self) -> float:
        return self._bg

    def setBackgroundValue(self, v: Any) -> None:
        self._bg = float(v)

    # --- ROI / mask
    def setRoi(self, *args: Any) -> None:
        if not args or args[0] is None:
            self._roi = None
            return
        if len(args) == 4:
            self._roi = {"type": "rect", "x": float(args[0]), "y": float(args[1]), "w": float(args[2]), "h": float(args[3])}
        elif isinstance(args[0], Rectangle):
            r = args[0]
            self._roi = {"type": "rect", "x": float(r.x), "y": float(r.y), "w": float(r.width), "h": float(r.height)}
        elif hasattr(args[0], "_roi"):
            self._roi = dict(args[0]._roi)
        elif isinstance(args[0], dict):
            self._roi = dict(args[0])
        else:
            raise TypeError("setRoi expects (x, y, w, h), a Rectangle or a Roi")

    def resetRoi(self) -> None:
        self._roi = None

    def getRoi(self) -> Rectangle:
        if self._roi is None:
            return Rectangle(0, 0, self.width, self.height)
        x, y, w, h = roimod.bounding_box(self._roi)
        return Rectangle(int(math.floor(x)), int(math.floor(y)), int(math.ceil(w)), int(math.ceil(h)))

    def getMask(self) -> "ImageProcessor | None":
        if self._roi is None or self._roi.get("type") == "rect":
            return None
        m = roimod.to_mask(self._roi, (self.height, self.width))
        r = self.getRoi()
        sub = m[r.y: r.y + r.height, r.x: r.x + r.width]
        return ByteProcessor(_array=(sub.astype(np.uint8) * 255))

    def setMask(self, mask: Any) -> None:
        pass

    def _mask(self) -> np.ndarray | None:
        return roimod.to_mask(self._roi, (self.height, self.width)) if self._roi is not None else None

    # --- drawing values
    def setColor(self, c: Any) -> None:
        if isinstance(c, Color):
            self._color = c
            self._value = float((c.r << 16) | (c.g << 8) | c.b) if self._rgb else float(round(0.299 * c.r + 0.587 * c.g + 0.114 * c.b)) if self.getBitDepth() == 8 else float((c.r + c.g + c.b) / 3.0)
        else:
            self._color = None
            self._value = float(c)

    def setValue(self, v: Any) -> None:
        self._color = None
        self._value = float(v)

    setForegroundValue = setValue

    def getForegroundValue(self) -> float:
        return self._value

    def setLineWidth(self, w: Any) -> None:
        self._line_width = max(1, int(w))

    def getLineWidth(self) -> int:
        return self._line_width

    def setInterpolationMethod(self, m: Any) -> None:
        self._interp = int(m)

    def getInterpolationMethod(self) -> int:
        return self._interp

    def setInterpolate(self, b: Any) -> None:
        self._interp = 1 if b else 0

    def _fill_value(self) -> Any:
        if self._rgb:
            v = int(self._value)
            return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], np.uint8)
        return self._value

    def fill(self, roi: Any = None) -> None:
        if roi is not None:
            saved = self._roi
            self.setRoi(roi)
            try:
                self.fill()
            finally:
                self._roi = saved
            return
        p = self._plane()
        m = self._mask()
        if self._rgb:
            v = self._fill_value()
            if m is None:
                p[:] = v[:, None, None]
            else:
                for i in range(3):
                    p[i][m] = v[i]
        else:
            v = cast_to(np.array(self._value), p.dtype)
            if m is None:
                p[:] = v
            else:
                p[m] = v
        self._touch()

    def fillOutside(self, roi: Any) -> None:
        saved = self._roi
        self.setRoi(roi)
        m = self._mask()
        self._roi = saved
        if m is None:
            return
        p = self._plane()
        if self._rgb:
            v = self._fill_value()
            for i in range(3):
                p[i][~m] = v[i]
        else:
            p[~m] = cast_to(np.array(self._value), p.dtype)
        self._touch()

    def _paint(self, mask: np.ndarray) -> None:
        p = self._plane()
        if self._rgb:
            v = self._fill_value()
            for i in range(3):
                p[i][mask] = v[i]
        else:
            p[mask] = cast_to(np.array(self._value), p.dtype)
        self._touch()

    def drawLine(self, x1: Any, y1: Any, x2: Any, y2: Any) -> None:
        from skimage.draw import line
        rr, cc = line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
        m = np.zeros((self.height, self.width), bool)
        ok = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
        m[rr[ok], cc[ok]] = True
        if self._line_width > 1:
            from scipy.ndimage import binary_dilation
            m = binary_dilation(m, iterations=self._line_width // 2)
        self._paint(m)
        self._x, self._y = int(x2), int(y2)

    def moveTo(self, x: Any, y: Any) -> None:
        self._x, self._y = int(x), int(y)

    def lineTo(self, x: Any, y: Any) -> None:
        self.drawLine(self._x, self._y, x, y)

    def drawDot(self, x: Any, y: Any) -> None:
        r = max(1, self._line_width)
        self.fillOval(int(x) - r // 2, int(y) - r // 2, r, r)

    def drawPixel(self, x: Any, y: Any) -> None:
        xy = self._inside(x, y)
        if xy is not None:
            m = np.zeros((self.height, self.width), bool)
            m[xy[1], xy[0]] = True
            self._paint(m)

    def drawRect(self, x: Any, y: Any, w: Any, h: Any) -> None:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 1 or h < 1:
            return
        m = np.zeros((self.height, self.width), bool)
        m[max(0, y): y + h, max(0, x): x + w] = True
        inner = np.zeros_like(m)
        lw = self._line_width
        inner[max(0, y + lw): max(0, y + h - lw), max(0, x + lw): max(0, x + w - lw)] = True
        self._paint(m & ~inner)

    def fillRect(self, x: Any, y: Any, w: Any, h: Any) -> None:
        m = np.zeros((self.height, self.width), bool)
        m[max(0, int(y)): int(y) + int(h), max(0, int(x)): int(x) + int(w)] = True
        self._paint(m)

    def fillOval(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._paint(roimod.to_mask({"type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h)}, (self.height, self.width)))

    def drawOval(self, x: Any, y: Any, w: Any, h: Any) -> None:
        outer = roimod.to_mask({"type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h)}, (self.height, self.width))
        lw = self._line_width
        inner = roimod.to_mask({"type": "oval", "x": float(x) + lw, "y": float(y) + lw, "w": float(w) - 2 * lw, "h": float(h) - 2 * lw}, (self.height, self.width)) if w > 2 * lw and h > 2 * lw else np.zeros_like(outer)
        self._paint(outer & ~inner)

    def fillPolygon(self, poly: Any) -> None:
        pts = [[float(px), float(py)] for px, py in zip(poly.xpoints, poly.ypoints)]
        self._paint(roimod.to_mask({"type": "polygon", "points": pts}, (self.height, self.width)))

    def drawPolygon(self, poly: Any) -> None:
        n = len(poly.xpoints)
        for i in range(n):
            self.drawLine(poly.xpoints[i], poly.ypoints[i], poly.xpoints[(i + 1) % n], poly.ypoints[(i + 1) % n])

    def setFont(self, font: Any) -> None:
        self._font_size = int(getattr(font, "size", 12))

    def setFontSize(self, size: Any) -> None:
        self._font_size = int(size)

    def setJustification(self, j: Any) -> None:
        self._justify = int(j)

    def setAntialiasedText(self, b: Any) -> None:
        pass

    def getStringWidth(self, s: Any) -> int:
        return int(len(str(s)) * self._font_size * 0.6)

    def getFontMetrics(self) -> Any:
        size = self._font_size

        class _FM:
            def getHeight(self_inner) -> int:
                return int(size * 1.2)

            def stringWidth(self_inner, s: Any) -> int:
                return int(len(str(s)) * size * 0.6)

            def getAscent(self_inner) -> int:
                return int(size * 0.8)

        return _FM()

    def drawString(self, text: Any, x: Any = None, y: Any = None, background: Any = None) -> None:
        """Rasterise text at (x, y) (baseline, ImageJ convention) through the overlay burn-in path."""
        if x is None:
            x, y = self._x, self._y
        from ..overlay_render import draw_text_into_plane
        s = str(text)
        xi = int(x)
        if self._justify == self.CENTER_JUSTIFY:
            xi -= self.getStringWidth(s) // 2
        elif self._justify == self.RIGHT_JUSTIFY:
            xi -= self.getStringWidth(s)
        p = self._plane()
        if self._rgb:
            v = self._fill_value()
            for i in range(3):
                p[i] = draw_text_into_plane(p[i], s, xi, int(y), float(v[i]), self._font_size)
        else:
            p[:] = draw_text_into_plane(p, s, xi, int(y), float(self._value), self._font_size)
        self._touch()
        self._y = int(y) + int(self._font_size * 1.2)

    # --- arithmetic (in place, like ImageJ)
    def _apply(self, fn: Any, float_ok: bool = True) -> None:
        p = self._plane()
        m = self._mask()
        if self._rgb:
            for i in range(3):
                with np.errstate(all="ignore"):
                    out = cast_to(np.nan_to_num(fn(p[i].astype(np.float64))), np.dtype(np.uint8))
                if m is None:
                    p[i][:] = out
                else:
                    p[i][m] = out[m]
            self._touch()
            return
        src = p.astype(np.float64)
        with np.errstate(all="ignore"):
            out = fn(src)
        out = cast_to(np.nan_to_num(out, nan=np.nan if p.dtype == np.float32 else 0.0, posinf=np.inf if p.dtype == np.float32 else 0, neginf=-np.inf if p.dtype == np.float32 else 0), p.dtype)
        if m is None:
            p[:] = out
        else:
            p[m] = out[m]
        self._touch()

    def add(self, v: Any) -> None:
        self._apply(lambda a: a + float(v))

    def subtract(self, v: Any) -> None:
        self._apply(lambda a: a - float(v))

    def multiply(self, v: Any) -> None:
        self._apply(lambda a: a * float(v))

    def divide(self, v: Any) -> None:
        self._apply(lambda a: a / float(v) if float(v) != 0 else a)

    def min(self, v: Any) -> None:  # noqa: A003
        self._apply(lambda a: np.maximum(a, float(v)))

    def max(self, v: Any) -> None:  # noqa: A003
        self._apply(lambda a: np.minimum(a, float(v)))

    def and_(self, v: Any) -> None:
        self._apply(lambda a: a.astype(np.int64) & int(v))

    def or_(self, v: Any) -> None:
        self._apply(lambda a: a.astype(np.int64) | int(v))

    def xor(self, v: Any) -> None:
        self._apply(lambda a: a.astype(np.int64) ^ int(v))

    def gamma(self, v: Any) -> None:
        rng = 255.0 if self.getBitDepth() == 8 else (65535.0 if self.getBitDepth() == 16 else None)
        if rng is None:
            self._apply(lambda a: np.exp(np.log(np.where(a > 0, a, 1e-30)) * float(v)))
        else:
            self._apply(lambda a: np.exp(np.log(np.clip(a, 1e-30, None) / rng) * float(v)) * rng)

    def log(self) -> None:
        rng = 255.0 if self.getBitDepth() == 8 else (65535.0 if self.getBitDepth() == 16 else None)
        if rng is None:
            self._apply(lambda a: np.log(np.where(a > 0, a, np.nan)))
        else:  # ImageJ scales the log so that the type's maximum maps onto itself
            self._apply(lambda a: np.log(np.clip(a, 1, None)) * (rng / np.log(rng)))

    def exp(self) -> None:
        rng = 255.0 if self.getBitDepth() == 8 else (65535.0 if self.getBitDepth() == 16 else None)
        if rng is None:
            self._apply(np.exp)
        else:
            self._apply(lambda a: np.exp(a * (np.log(rng) / rng)))

    def sqr(self) -> None:
        self._apply(lambda a: a * a)

    def sqrt(self) -> None:
        self._apply(lambda a: np.sqrt(np.clip(a, 0, None)))

    def abs(self) -> None:  # noqa: A003
        self._apply(np.abs)

    def invert(self) -> None:
        p = self._plane()
        if self._rgb or p.dtype == np.uint8:
            self._apply(lambda a: 255.0 - a)
        elif p.dtype == np.uint16:
            lo, hi = float(p.min()), float(p.max())
            self._apply(lambda a: hi - (a - lo))
        else:
            lo, hi = self.getMin(), self.getMax()
            self._apply(lambda a: hi - (a - lo))

    def noise(self, sd: Any) -> None:
        rng = np.random.default_rng()
        self._apply(lambda a: a + rng.normal(0.0, float(sd), a.shape))

    def threshold(self, level: Any) -> None:
        self._apply(lambda a: np.where(a > float(level), 255.0, 0.0))

    def applyTable(self, table: Any) -> None:
        lut = np.asarray(table)
        self._apply(lambda a: lut[np.clip(a.astype(np.int64), 0, len(lut) - 1)])

    def setThreshold(self, lo: Any, hi: Any, lut_update: Any = None) -> None:
        self._threshold = (float(lo), float(hi))
        if self._img is not None:
            ctx = current()
            if in_session(ctx, self._img):
                ctx.session.thresholds[self._img.id] = {"lower": float(lo), "upper": float(hi), "method": "Manual"}
                ctx.session._changed("images")

    def resetThreshold(self) -> None:
        self._threshold = None
        if self._img is not None:
            ctx = current()
            ctx.session.thresholds.pop(self._img.id, None)

    def getMinThreshold(self) -> float:
        thr = self._current_threshold()
        return thr[0] if thr else NO_THRESHOLD

    def getMaxThreshold(self) -> float:
        thr = self._current_threshold()
        return thr[1] if thr else NO_THRESHOLD

    def _current_threshold(self) -> tuple[float, float] | None:
        if self._img is not None:
            ctx = current()
            rec = ctx.session.thresholds.get(self._img.id)
            if rec is not None:
                return float(rec["lower"]), float(rec["upper"])
        return self._threshold

    def isThreshold(self) -> bool:
        return self._current_threshold() is not None

    def setAutoThreshold(self, method: Any = "Default", dark: Any = None, lut_update: Any = None) -> None:
        """``setAutoThreshold("Otsu dark")`` / ``setAutoThreshold(method, darkBackground, lutUpdate)`` /
        ``setAutoThreshold(ImageProcessor.ISODATA, …)``."""
        from ..threshold import METHODS, auto_threshold
        if isinstance(method, (int, float)) and not isinstance(method, bool):
            name = "Default"
        else:
            words = str(method).replace("dark", " dark ").replace("no-reset", "").split()
            name = words[0] if words else "Default"
            if dark is None:
                dark = "dark" in [w.lower() for w in words]
        if name not in METHODS:
            low = {m.lower(): m for m in METHODS}
            name = low.get(name.lower(), "Default")
        lo, hi = auto_threshold(self._gray(), name, self._mask(), bool(dark))
        self.setThreshold(lo, hi)
        if self._img is not None:
            ctx = current()
            if in_session(ctx, self._img):
                ctx.session.thresholds[self._img.id]["method"] = name
                ctx.session.thresholds[self._img.id]["dark_background"] = bool(dark)

    def getAutoThreshold(self, histogram: Any = None) -> int:
        from ..threshold import ij_default
        h = np.asarray(histogram if histogram is not None else self.getHistogram(), np.int64)
        return int(ij_default(h))

    def autoThreshold(self) -> None:
        self.threshold(self.getAutoThreshold())

    def setBinaryThreshold(self) -> None:
        self.setThreshold(255, 255)

    # --- filters (Lumen commands on this plane)
    def _op(self, op_id: str, params: dict[str, Any] | None = None, roi: Any = None) -> dict[str, Any]:
        ctx = current()
        use_roi = self._roi if roi is None else roi
        if self._img is not None:
            with temporarily_in_session(ctx, self._img):
                t, z, c = self._pos()
                d = self._img.display
                saved = (d.t, d.z, d.c)
                d.t, d.z, d.c = t, z, c
                try:
                    return ctx.session.run_op(op_id, params or {}, image_id=self._img.id, roi=use_roi, scope="plane", record=False)
                finally:
                    d.t, d.z, d.c = saved
        assert self._arr is not None
        tmp = LumenImage(self._arr.copy() if not self._rgb else np.moveaxis(self._arr.copy(), 0, -1), name="ijshim processor", rgb=self._rgb)
        with temporarily_in_session(ctx, tmp):
            out = ctx.session.run_op(op_id, params or {}, image_id=tmp.id, roi=use_roi, scope="plane", record=False)
            if tmp.id in ctx.session.images and ctx.session.images[tmp.id].data.shape[-2:] == self._arr.shape[-2:]:
                res = ctx.session.images[tmp.id]
                self._arr = np.ascontiguousarray(res.data[0, 0] if res.rgb else res.data[0, 0, 0])
                self._rgb = res.rgb
        return out

    def blurGaussian(self, sigma: Any) -> None:
        self._op("filters.gaussian", {"sigma": float(sigma)})

    def smooth(self) -> None:
        self._op("process.smooth")

    def sharpen(self) -> None:
        self._op("process.sharpen")

    def findEdges(self) -> None:
        self._op("process.find_edges")

    def medianFilter(self) -> None:
        self._op("filters.median", {"radius": 1.0})

    def erode(self, *a: Any) -> None:
        self._op("binary.erode")

    def dilate(self, *a: Any) -> None:
        self._op("binary.dilate")

    def skeletonize(self) -> None:
        self._op("binary.skeletonize")

    def outline(self) -> None:
        self._op("binary.outline")

    def filter(self, kind: Any) -> None:  # noqa: A003
        k = int(kind)
        if k == self.BLUR_MORE:
            self.smooth()
        elif k == self.FIND_EDGES:
            self.findEdges()
        elif k == self.MEDIAN_FILTER:
            self.medianFilter()
        elif k == self.MIN:
            self._op("filters.minimum", {"radius": 1.0})
        elif k == self.MAX:
            self._op("filters.maximum", {"radius": 1.0})

    def convolve(self, kernel: Any, kw: Any, kh: Any) -> None:
        k = np.asarray(kernel, np.float64).reshape(int(kh), int(kw))
        self._op("filters.convolve", {"kernel": "\n".join(" ".join(f"{v:g}" for v in row) for row in k), "normalize": False})

    def convolve3x3(self, kernel: Any) -> None:
        k = np.asarray(kernel, np.float64).reshape(3, 3)
        s = k.sum()
        self._op("filters.convolve", {"kernel": "\n".join(" ".join(f"{v:g}" for v in row) for row in k), "normalize": bool(s != 0)})

    def translate(self, dx: Any, dy: Any) -> None:
        from scipy.ndimage import shift
        p = self._plane()
        if self._rgb:
            self._set_plane(np.stack([shift(p[i].astype(np.float64), (float(dy), float(dx)), order=1 if self._interp else 0, cval=self._bg) for i in range(3)]))
        else:
            self._set_plane(shift(p.astype(np.float64), (float(dy), float(dx)), order=1 if self._interp else 0, cval=self._bg))

    def rotate(self, angle: Any) -> None:
        from scipy.ndimage import rotate
        p = self._plane()
        order = 0 if self._interp == 0 else (3 if self._interp == 2 else 1)
        if self._rgb:
            self._set_plane(np.stack([rotate(p[i].astype(np.float64), -float(angle), reshape=False, order=order, cval=self._bg) for i in range(3)]))
        else:
            self._set_plane(rotate(p.astype(np.float64), -float(angle), reshape=False, order=order, cval=self._bg))

    def flipHorizontal(self) -> None:
        self._set_plane(self._plane()[..., ::-1].copy())

    def flipVertical(self) -> None:
        p = self._plane()
        self._set_plane(p[:, ::-1, :].copy() if self._rgb else p[::-1, :].copy())

    def scale(self, sx: Any, sy: Any) -> None:
        from scipy.ndimage import zoom
        p = self._plane()
        order = 0 if self._interp == 0 else 1
        if self._rgb:
            out = np.stack([zoom(p[i].astype(np.float64), (float(sy), float(sx)), order=order) for i in range(3)])
            canvas = np.full_like(p, 0)
        else:
            out = zoom(p.astype(np.float64), (float(sy), float(sx)), order=order)
            canvas = np.full(p.shape, self._bg, np.float64)
        h, w = p.shape[-2:]
        oh, ow = out.shape[-2:]
        y0, x0 = (h - oh) // 2, (w - ow) // 2
        sy0, sx0 = max(0, -y0), max(0, -x0)
        dy0, dx0 = max(0, y0), max(0, x0)
        hh, ww = min(oh - sy0, h - dy0), min(ow - sx0, w - dx0)
        canvas[..., dy0: dy0 + hh, dx0: dx0 + ww] = out[..., sy0: sy0 + hh, sx0: sx0 + ww]
        self._set_plane(canvas)

    def resize(self, w: Any, h: Any = None, average: Any = None) -> "ImageProcessor":
        from skimage.transform import resize
        w = int(w)
        h = int(h) if h is not None else int(round(self.height * w / self.width))
        p = self._plane()
        order = 0 if self._interp == 0 else (3 if self._interp == 2 else 1)
        if self._rgb:
            out = np.stack([resize(p[i].astype(np.float64), (h, w), order=order, anti_aliasing=bool(average), preserve_range=True) for i in range(3)])
            return ColorProcessor(_array=cast_to(out, np.dtype(np.uint8)), _rgb=True)
        out = resize(p.astype(np.float64), (h, w), order=order, anti_aliasing=bool(average), preserve_range=True)
        return self._standalone(cast_to(out, p.dtype))

    def crop(self) -> "ImageProcessor":
        r = self.getRoi()
        p = self._plane()
        sub = p[..., r.y: r.y + r.height, r.x: r.x + r.width].copy()
        return self._standalone(sub, rgb=self._rgb)

    def duplicate(self) -> "ImageProcessor":
        out = self._standalone(self._plane().copy(), rgb=self._rgb)
        out._roi = dict(self._roi) if self._roi else None
        out._min, out._max = (self.getMin(), self.getMax()) if not self._rgb else (None, None)
        return out

    def createProcessor(self, w: Any, h: Any) -> "ImageProcessor":
        if self._rgb:
            return ColorProcessor(int(w), int(h))
        return self._standalone(np.zeros((int(h), int(w)), self._plane().dtype))

    def createImage(self) -> None:
        return None

    def getBufferedImage(self) -> None:
        return None

    def snapshot(self) -> None:
        self._snapshot = self._plane().copy()

    def reset(self, mask: Any = None) -> None:
        if self._snapshot is not None and self._snapshot.shape == self._plane().shape:
            if mask is not None and hasattr(mask, "_plane"):
                m = mask._plane() > 0
                r = self.getRoi()
                p = self._plane()
                region = p[..., r.y: r.y + r.height, r.x: r.x + r.width]
                snap = self._snapshot[..., r.y: r.y + r.height, r.x: r.x + r.width]
                region[..., ~m] = snap[..., ~m]
                self._touch()
            else:
                self._set_plane(self._snapshot.copy())

    def getSnapshotPixels(self) -> Any:
        return None if self._snapshot is None else (self._snapshot.ravel() if not self._rgb else _pack_rgb(self._snapshot).ravel())

    def swapPixelArrays(self) -> None:
        if self._snapshot is not None:
            cur = self._plane().copy()
            self._set_plane(self._snapshot)
            self._snapshot = cur

    def setSnapshotCopyMode(self, b: Any) -> None:
        pass

    def insert(self, ip: "ImageProcessor", x: Any, y: Any) -> None:
        self.copyBits(ip, x, y, 0)

    def copyBits(self, ip: "ImageProcessor", x: Any, y: Any, mode: Any = 0) -> None:
        """``Blitter`` modes: COPY, COPY_INVERTED, COPY_TRANSPARENT, ADD, SUBTRACT, MULTIPLY, DIVIDE, AVERAGE, DIFFERENCE,
        AND, OR, XOR, MIN, MAX, COPY_ZERO_TRANSPARENT."""
        src = ip._plane()
        if ip._rgb != self._rgb:
            src = ip.convertToRGB()._plane() if self._rgb else ip.convertToFloat()._plane()
        dst = self._plane()
        x, y = int(x), int(y)
        h, w = src.shape[-2:]
        H, W = dst.shape[-2:]
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        ww, hh = min(w - sx0, W - dx0), min(h - sy0, H - dy0)
        if ww <= 0 or hh <= 0:
            return
        s = src[..., sy0: sy0 + hh, sx0: sx0 + ww].astype(np.float64)
        d = dst[..., dy0: dy0 + hh, dx0: dx0 + ww].astype(np.float64)
        m = int(mode)
        if m == 0:
            out = s
        elif m == 1:
            out = (255.0 if self.getBitDepth() <= 8 else 65535.0) - s
        elif m == 2:  # COPY_TRANSPARENT (white = transparent)
            out = np.where(s == 255, d, s)
        elif m == 3:
            out = d + s
        elif m == 4:
            out = d - s
        elif m == 5:
            out = d * s
        elif m == 6:
            with np.errstate(all="ignore"):
                out = np.where(s != 0, d / s, d)
        elif m == 7:
            out = (d + s) / 2.0
        elif m == 8:
            out = np.abs(d - s)
        elif m == 9:
            out = d.astype(np.int64) & s.astype(np.int64)
        elif m == 10:
            out = d.astype(np.int64) | s.astype(np.int64)
        elif m == 11:
            out = d.astype(np.int64) ^ s.astype(np.int64)
        elif m == 12:
            out = np.minimum(d, s)
        elif m == 13:
            out = np.maximum(d, s)
        elif m == 14:  # COPY_ZERO_TRANSPARENT
            out = np.where(s == 0, d, s)
        else:
            out = s
        dst[..., dy0: dy0 + hh, dx0: dx0 + ww] = cast_to(np.asarray(out, np.float64), dst.dtype)
        self._touch()

    # --- conversions (standalone results)
    def convertToFloat(self) -> "FloatProcessor":
        return FloatProcessor(_array=self._gray().astype(np.float32))

    convertToFloatProcessor = convertToFloat

    def convertToByte(self, do_scaling: Any = True) -> "ByteProcessor":
        g = self._gray()
        if g.dtype == np.uint8:
            return ByteProcessor(_array=g.copy())
        if do_scaling:
            lo, hi = self.getMin(), self.getMax()
            if hi <= lo:
                hi = lo + 1
            out = (g.astype(np.float64) - lo) / (hi - lo) * 256.0
            return ByteProcessor(_array=cast_to(np.clip(out, 0, 255), np.dtype(np.uint8)))
        return ByteProcessor(_array=cast_to(g.astype(np.float64), np.dtype(np.uint8)))

    convertToByteProcessor = convertToByte

    def convertToShort(self, do_scaling: Any = True) -> "ShortProcessor":
        g = self._gray()
        if g.dtype == np.uint16:
            return ShortProcessor(_array=g.copy())
        if do_scaling and g.dtype == np.float32:
            lo, hi = self.getMin(), self.getMax()
            if hi <= lo:
                hi = lo + 1
            out = (g.astype(np.float64) - lo) / (hi - lo) * 65536.0
            return ShortProcessor(_array=cast_to(np.clip(out, 0, 65535), np.dtype(np.uint16)))
        return ShortProcessor(_array=cast_to(g.astype(np.float64), np.dtype(np.uint16)))

    convertToShortProcessor = convertToShort

    def convertToRGB(self) -> "ColorProcessor":
        if self._rgb:
            return ColorProcessor(_array=self._plane().copy(), _rgb=True)
        b = self.convertToByte(True)._plane()
        return ColorProcessor(_array=np.stack([b, b, b]), _rgb=True)

    convertToColorProcessor = convertToRGB

    def toFloat(self, channel: Any, fp: Any = None) -> "FloatProcessor":
        p = self._plane()
        if self._rgb:
            return FloatProcessor(_array=p[int(channel)].astype(np.float32))
        return self.convertToFloat()

    def getNChannels(self) -> int:
        return 3 if self._rgb else 1

    def getCalibrationTable(self) -> None:
        return None

    def setCalibrationTable(self, t: Any) -> None:
        pass

    def getLut(self) -> None:
        return None

    def setLut(self, lut: Any) -> None:
        pass

    def getColorModel(self) -> None:
        return None

    def setColorModel(self, cm: Any) -> None:
        pass

    def getSliceNumber(self) -> int:
        if self._img is not None:
            return _encode_slice(self._img, *self._pos())
        return 1

    def setSliceNumber(self, n: Any) -> None:
        pass

    def setProgressBar(self, *a: Any) -> None:
        pass

    def setClipRect(self, *a: Any) -> None:
        pass

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"ip[width={self.width}, height={self.height}, bits={self.getBitDepth()}, min={self.getMin()}, max={self.getMax()}]"

    def getFloatArray(self) -> list[list[float]]:
        g = self._gray().astype(np.float64)
        return [[float(v) for v in g[:, x]] for x in range(self.width)]  # [x][y] like ImageJ

    def getIntArray(self) -> list[list[int]]:
        g = self._plane() if not self._rgb else _pack_rgb(self._plane())
        return [[int(v) for v in g[:, x]] for x in range(self.width)]

    def setFloatArray(self, arr: Any) -> None:
        a = np.asarray(arr, np.float64).T
        self._set_plane(a)

    def setIntArray(self, arr: Any) -> None:
        a = np.asarray(arr).T
        self._set_plane(_unpack_rgb(a) if self._rgb else a)


# the Java keyword methods
setattr(ImageProcessor, "and", ImageProcessor.and_)
setattr(ImageProcessor, "or", ImageProcessor.or_)


class ByteProcessor(ImageProcessor):
    _dtype = np.uint8


class ShortProcessor(ImageProcessor):
    _dtype = np.uint16


class FloatProcessor(ImageProcessor):
    _dtype = np.float32

    def __init__(self, width: Any = None, height: Any = None, pixels: Any = None, cm: Any = None, **kw: Any) -> None:
        if width is not None and height is None and not isinstance(width, (int, float)):  # FloatProcessor(float[][])
            arr = np.asarray(width, np.float64)
            super().__init__(_array=np.ascontiguousarray(arr.T.astype(np.float32)))
            return
        super().__init__(width, height, pixels, cm, **kw)


class ColorProcessor(ImageProcessor):
    _dtype = np.uint8

    def __init__(self, width: Any = None, height: Any = None, pixels: Any = None, cm: Any = None, **kw: Any) -> None:
        kw["_rgb"] = True
        super().__init__(width, height, pixels, cm, **kw)

    def getChannel(self, channel: Any, bp: Any = None) -> ByteProcessor:
        return ByteProcessor(_array=self._plane()[int(channel) - 1].copy())

    def setChannel(self, channel: Any, bp: Any) -> None:
        self._plane()[int(channel) - 1] = cast_to(bp._plane(), np.dtype(np.uint8))
        self._touch()

    def getRGB(self, r: Any, g: Any, b: Any) -> None:
        p = self._plane()
        r[:] = [int(v) for v in p[0].ravel()]
        g[:] = [int(v) for v in p[1].ravel()]
        b[:] = [int(v) for v in p[2].ravel()]

    def setRGB(self, r: Any, g: Any, b: Any) -> None:
        p = self._plane()
        for i, ch in enumerate((r, g, b)):
            p[i] = np.asarray(ch, np.uint8).reshape(self.height, self.width)
        self._touch()

    def getBrightness(self) -> FloatProcessor:
        return FloatProcessor(_array=(self._plane().max(axis=0).astype(np.float32) / 255.0))

    def getHSB(self, h: Any, s: Any, b: Any) -> None:
        import colorsys
        p = self._plane()
        hs, ss, bs = [], [], []
        for r, g, bl in zip(p[0].ravel(), p[1].ravel(), p[2].ravel()):
            hh, sv, vv = colorsys.rgb_to_hsv(int(r) / 255.0, int(g) / 255.0, int(bl) / 255.0)
            hs.append(int(hh * 255)); ss.append(int(sv * 255)); bs.append(int(vv * 255))
        h[:] = hs; s[:] = ss; b[:] = bs

    def getRGBWeights(self) -> list[float]:
        return [1 / 3, 1 / 3, 1 / 3]

    def setRGBWeights(self, *w: Any) -> None:
        pass

    @staticmethod
    def setWeightingFactors(*w: Any) -> None:
        pass


# ====================================================================== stacks
class ImageStack:
    """``ij.ImageStack``: the planes of an image (1-based ImageJ indices, channels fastest) or a standalone list of
    processors built with ``addSlice`` for ``ImagePlus(title, stack)``."""

    def __init__(self, width: Any = None, height: Any = None, size_or_cm: Any = None, *, _img: LumenImage | None = None) -> None:
        self._img = _img
        self._planes: list[np.ndarray] = []
        self._labels: list[str | None] = []
        self._rgb = False
        self._w = int(width) if width is not None else 0
        self._h = int(height) if height is not None else 0
        if _img is None and isinstance(size_or_cm, int) and size_or_cm > 0:
            for _ in range(int(size_or_cm)):
                self._planes.append(np.zeros((self._h, self._w), np.uint8))
                self._labels.append(None)

    # --- shape
    def getSize(self) -> int:
        if self._img is not None:
            return self._img.nt * self._img.nz * (1 if self._img.rgb else self._img.nc)
        return len(self._planes)

    size = getSize

    def __len__(self) -> int:
        return self.getSize()

    def getWidth(self) -> int:
        return self._img.width if self._img is not None else self._w

    def getHeight(self) -> int:
        return self._img.height if self._img is not None else self._h

    def getBitDepth(self) -> int:
        if self._img is not None:
            return _bit_depth(self._img)
        if not self._planes:
            return 8
        return 24 if self._rgb else {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}.get(self._planes[0].dtype, 32)

    def isVirtual(self) -> bool:
        return False

    def isRGB(self) -> bool:
        return self._img.rgb if self._img is not None else self._rgb

    def isHSB(self) -> bool:
        return False

    # --- planes
    def getProcessor(self, n: Any) -> ImageProcessor:
        n = int(n)
        if self._img is not None:
            return ImageProcessor._for(self._img, _decode_slice(self._img, n))
        if not 1 <= n <= len(self._planes):
            raise ValueError(f"Stack index {n} out of range (1–{len(self._planes)})")
        ip = ImageProcessor._standalone(self._planes[n - 1], rgb=self._rgb)
        ip._arr = self._planes[n - 1]  # share, like ImageJ's getProcessor(n)
        return ip

    def getPixels(self, n: Any) -> Any:
        return self.getProcessor(n).getPixels()

    def setPixels(self, pixels: Any, n: Any) -> None:
        self.getProcessor(n).setPixels(pixels)

    def getVoxel(self, x: Any, y: Any, z: Any) -> float:
        return float(self.getProcessor(int(z) + 1).getPixelValue(x, y))

    def setVoxel(self, x: Any, y: Any, z: Any, v: Any) -> None:
        self.getProcessor(int(z) + 1).putPixelValue(x, y, v)

    def getSliceLabel(self, n: Any) -> str | None:
        n = int(n)
        if self._img is not None:
            from ..ops.stacks import slice_labels
            labels = slice_labels(self._img)
            i = n - 1
            return (labels[i] or None) if 0 <= i < len(labels) else None
        return self._labels[n - 1] if 1 <= n <= len(self._labels) else None

    def getShortSliceLabel(self, n: Any) -> str | None:
        lab = self.getSliceLabel(n)
        if lab is None:
            return None
        short = lab.split("\n")[0]
        return short[:60] if short else None

    def setSliceLabel(self, label: Any, n: Any) -> None:
        n = int(n)
        if self._img is not None:
            from ..ops.stacks import slice_labels
            labels = slice_labels(self._img)
            if 1 <= n <= len(labels):
                labels[n - 1] = "" if label is None else str(label)
                self._img.metadata["slice_labels"] = labels
                self._img.rev += 1
            return
        if 1 <= n <= len(self._labels):
            self._labels[n - 1] = None if label is None else str(label)

    def getSliceLabels(self) -> list[str | None]:
        return [self.getSliceLabel(i) for i in range(1, self.getSize() + 1)]

    def addSlice(self, *args: Any) -> None:
        """``addSlice(ip)``, ``addSlice(label, ip)``, ``addSlice(label, pixels)``, ``addSlice(label, ip, n)``."""
        label: Any = None
        ip: Any = None
        at: int | None = None
        if len(args) == 1:
            ip = args[0]
        elif len(args) >= 2:
            label, ip = args[0], args[1]
            if len(args) >= 3:
                at = int(args[2])
        if isinstance(ip, ImageProcessor):
            arr = ip._plane().copy()
            rgb = ip._rgb
        else:
            p = np.asarray(ip)
            rgb = False
            arr = p.reshape(self.getHeight(), self.getWidth()) if p.ndim < 2 else p
        if self._img is not None:
            img = self._img
            axis = 1 if (img.nz > 1 or img.nt == 1) else 0
            plane = arr if img.rgb else arr[None]  # (C, Y, X)
            if img.rgb and (plane.ndim != 3 or plane.shape[0] != 3):
                raise ValueError("addSlice: an RGB stack needs a ColorProcessor")
            if plane.shape[-2:] != (img.height, img.width):
                raise ValueError(f"addSlice: slice size {plane.shape[-1]}×{plane.shape[-2]} does not match the stack ({img.width}×{img.height})")
            block = cast_to(plane[None, None], img.data.dtype)  # (1, 1, C, Y, X)
            if axis == 1:
                block = np.broadcast_to(block, (img.nt, 1, img.nc, img.height, img.width)).copy()
            else:
                block = np.broadcast_to(block, (1, img.nz, img.nc, img.height, img.width)).copy()
            data = np.concatenate([img.data, block], axis=axis) if at is None else np.concatenate([np.take(img.data, range(at), axis=axis), block, np.take(img.data, range(at, img.data.shape[axis]), axis=axis)], axis=axis)
            img.replace_data(data, rgb=img.rgb)
            if label is not None:
                from ..ops.stacks import slice_labels
                labels = slice_labels(img)
                idx = (at if at is not None else img.data.shape[axis] - 1)
                pos = _encode_slice(img, idx if axis == 0 else 0, idx if axis == 1 else 0, 0) - 1
                if 0 <= pos < len(labels):
                    labels[pos] = str(label)
                    img.metadata["slice_labels"] = labels
            return
        if not self._planes:
            self._rgb = rgb
            self._w, self._h = int(arr.shape[-1]), int(arr.shape[-2])
        elif arr.shape[-2:] != (self._h, self._w):
            raise ValueError(f"addSlice: slice size {arr.shape[-1]}×{arr.shape[-2]} does not match the stack ({self._w}×{self._h})")
        elif arr.dtype != self._planes[0].dtype:
            arr = cast_to(arr, self._planes[0].dtype)
        if at is None:
            self._planes.append(arr)
            self._labels.append(None if label is None else str(label))
        else:
            self._planes.insert(at, arr)
            self._labels.insert(at, None if label is None else str(label))

    def deleteSlice(self, n: Any) -> None:
        n = int(n)
        if self._img is not None:
            img = self._img
            if self.getSize() <= 1:
                raise ValueError("deleteSlice: the stack must have more than one slice")
            t, z, c = _decode_slice(img, n)
            axis = 1 if img.nz > 1 else (0 if img.nt > 1 else 2)
            idx = {1: z, 0: t, 2: c}[axis]
            from ..ops.stacks import plane_index, slice_labels
            labels = slice_labels(img)
            removed = {plane_index(img, tt, zz, cc) for (tt, zz, cc) in img.iter_planes() if (tt, zz, cc)[axis] == idx}
            data = np.delete(img.data, idx, axis=axis)
            if axis == 2 and not img.rgb:
                img.display.channels = [ch for i, ch in enumerate(img.display.channels) if i != idx]
            img.replace_data(data, rgb=img.rgb and axis != 2)
            if any(labels):
                img.metadata["slice_labels"] = [lab for i, lab in enumerate(labels) if i not in removed]
            img.clamp_position()
            return
        if 1 <= n <= len(self._planes):
            del self._planes[n - 1]
            del self._labels[n - 1]

    def deleteLastSlice(self) -> None:
        if self.getSize() > 0:
            self.deleteSlice(self.getSize())

    def duplicate(self) -> "ImageStack":
        st = ImageStack(self.getWidth(), self.getHeight())
        for i in range(1, self.getSize() + 1):
            st.addSlice(self.getSliceLabel(i), self.getProcessor(i))
        return st

    def crop(self, x: Any, y: Any, z: Any, w: Any, h: Any, d: Any) -> "ImageStack":
        st = ImageStack(int(w), int(h))
        for i in range(int(z) + 1, int(z) + int(d) + 1):
            ip = self.getProcessor(i)
            ip.setRoi(int(x), int(y), int(w), int(h))
            st.addSlice(self.getSliceLabel(i), ip.crop())
        return st

    def convertToFloat(self) -> "ImageStack":
        st = ImageStack(self.getWidth(), self.getHeight())
        for i in range(1, self.getSize() + 1):
            st.addSlice(self.getSliceLabel(i), self.getProcessor(i).convertToFloat())
        return st

    def getImageArray(self) -> list[Any]:
        return [self.getPixels(i) for i in range(1, self.getSize() + 1)]

    def update(self, ip: Any) -> None:
        pass

    def trim(self) -> None:
        pass

    def _to_image(self, title: str) -> LumenImage:
        if self._img is not None:
            return self._img
        if not self._planes:
            raise ValueError("ImagePlus(title, stack): the stack is empty")
        if self._rgb:
            data = np.stack([np.moveaxis(p, 0, -1) for p in self._planes])  # (Z, Y, X, 3)
            img = LumenImage(data, name=title, axes="ZYXS", rgb=True)
        else:
            img = LumenImage(np.stack(self._planes), name=title, axes="ZYX")
        if any(lab for lab in self._labels):
            img.metadata["slice_labels"] = [lab or "" for lab in self._labels]
        return img

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"stack[{self.getWidth()}x{self.getHeight()}x{self.getSize()}]"


# ====================================================================== ImagePlus
class _Window:
    """What ``imp.getWindow()`` returns for an image that has a tab: the few window methods scripts call are no-ops."""

    def __init__(self, imp: "ImagePlus") -> None:
        self._imp = imp

    def getImagePlus(self) -> "ImagePlus":
        return self._imp

    def getCanvas(self) -> "_Canvas":
        return _Canvas(self._imp)

    def setLocation(self, *a: Any) -> None:
        pass

    def getLocation(self) -> Any:
        from ._java import Point
        return Point(0, 0)

    def setSize(self, *a: Any) -> None:
        pass

    def getSize(self) -> Any:
        from ._java import Dimension
        return Dimension(self._imp.getWidth(), self._imp.getHeight())

    def getWidth(self) -> int:
        return self._imp.getWidth()

    def getHeight(self) -> int:
        return self._imp.getHeight()

    def toFront(self) -> None:
        self._imp.show()

    def close(self) -> bool:
        self._imp.close()
        return True

    def maximize(self) -> None:
        pass

    def setVisible(self, b: Any) -> None:
        if not b:
            self._imp.hide()

    def isVisible(self) -> bool:
        return self._imp.isVisible()

    def getTitle(self) -> str:
        return self._imp.getTitle()

    def setTitle(self, t: Any) -> None:
        self._imp.setTitle(t)

    def repaint(self) -> None:
        pass


class _Canvas:
    def __init__(self, imp: "ImagePlus") -> None:
        self._imp = imp
        self._mag = 1.0

    def getMagnification(self) -> float:
        return self._mag

    def setMagnification(self, m: Any) -> None:
        self._mag = float(m)

    def zoomIn(self, *a: Any) -> None:
        self._mag *= 1.5

    def zoomOut(self, *a: Any) -> None:
        self._mag /= 1.5

    def zoom100Percent(self) -> None:
        self._mag = 1.0

    def fitToWindow(self) -> None:
        pass

    def getSrcRect(self) -> Rectangle:
        return Rectangle(0, 0, self._imp.getWidth(), self._imp.getHeight())

    def setSourceRect(self, r: Any) -> None:
        pass

    def getImage(self) -> "ImagePlus":
        return self._imp

    def repaint(self) -> None:
        pass

    def getSize(self) -> Any:
        from ._java import Dimension
        return Dimension(self._imp.getWidth(), self._imp.getHeight())


class FileInfo:
    """``imp.getOriginalFileInfo()``: directory / fileName / size of the file an image came from."""

    _js_props = ("directory", "fileName", "width", "height", "nImages", "fileType", "fileFormat", "pixelWidth", "pixelHeight", "unit", "description")
    GRAY8, GRAY16_SIGNED, GRAY16_UNSIGNED, GRAY32_INT, GRAY32_FLOAT, COLOR8, RGB = 0, 1, 2, 3, 4, 5, 6
    UNKNOWN, RAW, TIFF, GIF_OR_JPG, FITS, BMP, DICOM, ZIP_ARCHIVE, PGM, IMAGEIO = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    def __init__(self, img: LumenImage | None = None) -> None:
        path = (img.path if img is not None else None) or ""
        self.directory = (os.path.dirname(path) + os.sep) if path else ""
        self.fileName = os.path.basename(path) if path else (img.name if img is not None else "")
        self.width = img.width if img is not None else 0
        self.height = img.height if img is not None else 0
        self.nImages = (img.nt * img.nz * (1 if img.rgb else img.nc)) if img is not None else 0
        self.fileType = (self.RGB if img.rgb else {np.dtype(np.uint8): self.GRAY8, np.dtype(np.uint16): self.GRAY16_UNSIGNED}.get(img.data.dtype, self.GRAY32_FLOAT)) if img is not None else 0
        self.fileFormat = self.TIFF if path.lower().endswith((".tif", ".tiff")) else (self.UNKNOWN if not path else self.IMAGEIO)
        self.pixelWidth = img.calibration.pixel_width if img is not None else 1.0
        self.pixelHeight = img.calibration.pixel_height if img is not None else 1.0
        self.unit = img.calibration.unit if img is not None else "pixel"
        self.description = ""

    def getFilePath(self) -> str:
        return os.path.join(self.directory, self.fileName) if self.directory else self.fileName

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"name={self.fileName}, dir={self.directory}, width={self.width}, height={self.height}, nImages={self.nImages}"


class ImagePlus:
    """``ij.ImagePlus`` over a LumenImage (see the module docstring for the show/tab semantics)."""

    GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB = 0, 1, 2, 3, 4
    COMPOSITE, COLOR, GRAYSCALE = 1, 2, 3
    _js_props = ("title", "width", "height", "changes")

    def __init__(self, title: Any = None, source: Any = None, *, _img: LumenImage | None = None) -> None:
        self.changes = False
        self._closed = False
        self._win: _Window | None = None
        if _img is not None:
            self._img: LumenImage | None = _img
            return
        if title is None and source is None:
            self._img = None
            return
        if source is None:
            path = str(title)
            if os.path.exists(path):
                from ..io import read_image
                self._img = read_image(path)
                return
            raise ValueError(f"ImagePlus({title!r}): no such file")
        name = str(title)
        if isinstance(source, ImageStack):
            self._img = source._to_image(name)
            if source._img is not None and source._img.name != name:
                self._img = source._img
        elif isinstance(source, ImageProcessor):
            arr = source._plane().copy()
            self._img = LumenImage(np.moveaxis(arr, 0, -1) if source._rgb else arr, name=name, rgb=source._rgb)
            if not source._rgb and source._min is not None and source._max is not None:
                self._img.display.channels[0].min, self._img.display.channels[0].max = source._min, source._max
        elif isinstance(source, LumenImage):
            self._img = source
        elif isinstance(source, np.ndarray):
            self._img = LumenImage(source, name=name)
        else:
            raise TypeError("ImagePlus(title, source): source must be an ImageProcessor, an ImageStack or a numpy array")

    # --- plumbing
    @classmethod
    def _wrap(cls, img: LumenImage | None) -> "ImagePlus | None":
        return cls(_img=img) if img is not None else None

    @property
    def img(self) -> LumenImage:
        """The underlying :class:`LumenImage` (Lumen extension)."""
        if self._img is None:
            raise ValueError("This ImagePlus has no image")
        return self._img

    def _ctx(self) -> Context:
        return current()

    def _live(self) -> bool:
        return self._img is not None and in_session(self._ctx(), self._img)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ImagePlus) and other._img is self._img

    def __hash__(self) -> int:
        return id(self._img)

    def __repr__(self) -> str:
        if self._img is None:
            return "img[null]"
        img = self._img
        return f"img[\"{img.name}\" ({img.id}), {_bit_depth(img)}-bit, {img.width}x{img.height}x{img.nc}x{img.nz}x{img.nt}]"

    def toString(self) -> str:
        return repr(self)

    def equals(self, other: Any) -> bool:
        return self == other

    # --- identity
    def getTitle(self) -> str:
        return self.img.name

    @property
    def title(self) -> str:
        return self.getTitle()

    @title.setter
    def title(self, t: Any) -> None:
        self.setTitle(t)

    def setTitle(self, title: Any) -> None:
        img = self.img
        ctx = self._ctx()
        if in_session(ctx, img):
            img.name = ctx.session.unique_name(str(title)) if str(title) != img.name else img.name
            ctx.session._changed("images")
        else:
            img.name = str(title)

    def getShortTitle(self) -> str:
        t = self.getTitle()
        t = t.split(" ")[0]
        return os.path.splitext(t)[0] if "." in t else t

    def getID(self) -> int:
        ctx = self._ctx()
        ids = ctx.ij._ids
        img = self.img
        if img.id not in ids:
            ids[img.id] = -(len(ids) + 1)
        return ids[img.id]

    def getLumenId(self) -> str:
        return self.img.id

    # --- geometry
    def getWidth(self) -> int:
        return self.img.width

    def getHeight(self) -> int:
        return self.img.height

    width = property(lambda self: self.getWidth())
    height = property(lambda self: self.getHeight())

    def getNChannels(self) -> int:
        return 1 if self.img.rgb else self.img.nc

    def getNSlices(self) -> int:
        return self.img.nz

    def getNFrames(self) -> int:
        return self.img.nt

    def getStackSize(self) -> int:
        return self.getNChannels() * self.img.nz * self.img.nt

    getImageStackSize = getStackSize

    def getNDimensions(self) -> int:
        return 2 + sum(1 for n in (self.getNChannels(), self.img.nz, self.img.nt) if n > 1)

    def getDimensions(self, *a: Any) -> list[int]:
        return [self.getWidth(), self.getHeight(), self.getNChannels(), self.img.nz, self.img.nt]

    def setDimensions(self, c: Any, z: Any, t: Any) -> None:
        img = self.img
        if int(c) * int(z) * int(t) != self.getStackSize():
            raise ValueError("setDimensions: channels×slices×frames must equal the stack size")
        ctx = self._ctx()
        with temporarily_in_session(ctx, img):
            ctx.session.run_op("stacks.to_hyperstack", {"channels": int(c), "slices": int(z), "frames": int(t)}, image_id=img.id, record=False)

    def getBitDepth(self) -> int:
        return _bit_depth(self.img)

    def getBytesPerPixel(self) -> int:
        return {8: 1, 16: 2, 24: 4, 32: 4}[self.getBitDepth()]

    def getType(self) -> int:
        return {8: self.GRAY8, 16: self.GRAY16, 32: self.GRAY32, 24: self.COLOR_RGB}[self.getBitDepth()]

    def isRGB(self) -> bool:
        return self.img.rgb

    def isComposite(self) -> bool:
        return (not self.img.rgb) and self.img.nc > 1

    def isHyperStack(self) -> bool:
        img = self.img
        return (not img.rgb and img.nc > 1) or (img.nz > 1 and img.nt > 1)

    isDisplayedHyperStack = isHyperStack

    def isStack(self) -> bool:
        return self.getStackSize() > 1

    def isInvertedLut(self) -> bool:
        return self.getProcessor().isInvertedLut()

    def isVisible(self) -> bool:
        return self._live()

    def isLocked(self) -> bool:
        return False

    def lock(self) -> bool:
        return True

    def lockSilently(self) -> bool:
        return True

    def unlock(self) -> None:
        pass

    def isProcessor(self) -> bool:
        return self._img is not None

    def getOpenAsHyperStack(self) -> bool:
        return self.isHyperStack()

    def setOpenAsHyperStack(self, b: Any) -> None:
        pass

    # --- show / hide / close
    def show(self, status: Any = None) -> None:
        ctx = self._ctx()
        img = self.img
        self._closed = False
        if img.id not in ctx.session.images:
            ctx.session.add_image(img, activate=True)
        else:
            ctx.session.select(img.id)
            ctx.session._changed("images")
        if status:
            ctx.emit(str(status))

    def hide(self) -> None:
        ctx = self._ctx()
        img = self.img
        if img.id in ctx.session.images:
            ctx.session.close_image(img.id)

    def close(self) -> None:
        self.hide()
        self._closed = True
        self.changes = False

    def flush(self) -> None:
        pass

    def getWindow(self) -> _Window | None:
        return _Window(self) if self._live() else None

    def getCanvas(self) -> _Canvas | None:
        return _Canvas(self) if self._live() else None

    def updateAndDraw(self) -> None:
        img = self.img
        img.rev += 1
        img.sync_channels()
        ctx = self._ctx()
        if in_session(ctx, img):
            ctx.session._changed("images")

    updateAndRepaintWindow = updateAndDraw
    updateImage = updateAndDraw
    repaintWindow = updateAndDraw
    draw = updateAndDraw

    def updateStatusbarValue(self) -> None:
        pass

    def setIgnoreFlush(self, b: Any) -> None:
        pass

    def trimProcessor(self) -> None:
        pass

    def startTiming(self) -> None:
        pass

    def getStartTime(self) -> int:
        import time
        return int(time.time() * 1000)

    # --- position
    def getCurrentSlice(self) -> int:
        img = self.img
        d = img.display
        return _encode_slice(img, d.t, d.z, d.c)

    getSlice = getCurrentSlice

    def setSlice(self, n: Any) -> None:
        img = self.img
        t, z, c = _decode_slice(img, int(n))
        d = img.display
        d.t, d.z = t, z
        if not img.rgb:
            d.c = c
        img.clamp_position()
        self._pos_changed()

    setSliceWithoutUpdate = setSlice

    def _pos_changed(self) -> None:
        ctx = self._ctx()
        if in_session(ctx, self.img):
            ctx.session._changed("images")

    def getC(self) -> int:
        return 1 if self.img.rgb else self.img.display.c + 1

    def getZ(self) -> int:
        return self.img.display.z + 1

    def getT(self) -> int:
        return self.img.display.t + 1

    getChannel = getC
    getFrame = getT

    def setC(self, c: Any) -> None:
        self.setPosition(c, self.getZ(), self.getT())

    def setZ(self, z: Any) -> None:
        self.setPosition(self.getC(), z, self.getT())

    def setT(self, t: Any) -> None:
        self.setPosition(self.getC(), self.getZ(), t)

    def setPosition(self, *args: Any) -> None:
        img = self.img
        if len(args) == 1:
            self.setSlice(args[0])
            return
        c, z, t = (int(v) for v in args[:3])
        d = img.display
        if not img.rgb:
            d.c = min(max(c - 1, 0), img.nc - 1)
        d.z = min(max(z - 1, 0), img.nz - 1)
        d.t = min(max(t - 1, 0), img.nt - 1)
        img.clamp_position()
        self._pos_changed()

    setPositionWithoutUpdate = setPosition

    def getStackIndex(self, c: Any, z: Any, t: Any) -> int:
        return _encode_slice(self.img, int(t) - 1, int(z) - 1, int(c) - 1)

    def convertIndexToPosition(self, n: Any) -> list[int]:
        t, z, c = _decode_slice(self.img, int(n))
        return [c + 1, z + 1, t + 1]

    # --- processors / stacks
    def getProcessor(self) -> ImageProcessor:
        return ImageProcessor._for(self.img, follow=True)

    @property
    def processor(self) -> ImageProcessor:
        return self.getProcessor()

    def getStack(self) -> ImageStack:
        return ImageStack(_img=self.img)

    getImageStack = getStack

    @property
    def stack(self) -> ImageStack:
        return self.getStack()

    def setStack(self, *args: Any) -> None:
        """``setStack(stack)`` / ``setStack(title, stack)`` / ``setStack(stack, c, z, t)``."""
        title: str | None = None
        stack: Any
        dims: tuple[int, int, int] | None = None
        if len(args) == 1:
            stack = args[0]
        elif len(args) == 2:
            title, stack = str(args[0]), args[1]
        else:
            stack = args[0]
            dims = (int(args[1]), int(args[2]), int(args[3]))
        img = self.img
        if isinstance(stack, ImageStack):
            if stack._img is img:
                return
            new = stack._to_image(title or img.name)
            img.replace_data(new.data, rgb=new.rgb)
            if "slice_labels" in new.metadata:
                img.metadata["slice_labels"] = new.metadata["slice_labels"]
        else:
            raise TypeError("setStack expects an ImageStack")
        if dims is not None:
            self.setDimensions(*dims)
        img.clamp_position()
        if title:
            self.setTitle(title)
        self.updateAndDraw()

    def setProcessor(self, *args: Any) -> None:
        """``setProcessor(ip)`` / ``setProcessor(title, ip)``: replace the current plane (same size) or the whole
        single-plane image."""
        title, ip = (None, args[0]) if len(args) == 1 else (str(args[0]), args[1])
        img = self.img
        arr = ip._plane()
        if arr.shape[-2:] == (img.height, img.width) and (ip._rgb == img.rgb) and self.getStackSize() > 1:
            ImageProcessor._for(img, follow=True)._set_plane(arr)
        else:
            data = np.moveaxis(arr, 0, -1) if ip._rgb else arr
            new = LumenImage(data, name=img.name, rgb=ip._rgb)
            img.replace_data(new.data, rgb=ip._rgb)
            img.reset_display()
        if title:
            self.setTitle(title)
        self.updateAndDraw()

    def setImage(self, imp: "ImagePlus") -> None:
        self._img = imp._img

    def getImage(self) -> None:
        return None

    def getBufferedImage(self) -> None:
        return None

    def getPixel(self, x: Any, y: Any) -> list[int]:
        ip = self.getProcessor()
        if self.img.rgb:
            v = int(ip.getPixel(x, y))
            return [(v >> 16) & 255, (v >> 8) & 255, v & 255, 0]
        v = ip.getPixel(x, y)
        if self.img.data.dtype == np.float32:
            import struct
            return [struct.unpack("<i", struct.pack("<f", float(v)))[0], 0, 0, 0]
        return [int(v), 0, 0, 0]

    def createImagePlus(self) -> "ImagePlus":
        return ImagePlus(_img=self.img.copy())

    def createEmptyStack(self) -> ImageStack:
        return ImageStack(self.getWidth(), self.getHeight())

    # --- calibration / properties
    def getCalibration(self) -> Calibration:
        return Calibration(self)

    getLocalCalibration = getCalibration
    calibration = property(lambda self: self.getCalibration())

    def setCalibration(self, cal: Calibration | None) -> None:
        img = self.img
        if cal is None:
            img.calibration = LumenCalibration()
        else:
            img.calibration = LumenCalibration(**cal._cal.to_dict())
            if cal._origin:
                img.metadata["origin"] = dict(cal._origin)
        img.rev += 1
        self._pos_changed()

    def setGlobalCalibration(self, cal: Any) -> None:
        pass

    def getGlobalCalibration(self) -> None:
        return None

    def getProperty(self, key: Any) -> Any:
        return self.img.metadata.get(str(key))

    def setProperty(self, key: Any, value: Any) -> None:
        if value is None:
            self.img.metadata.pop(str(key), None)
        else:
            self.img.metadata[str(key)] = value

    def getProp(self, key: Any) -> str | None:
        v = self.img.metadata.get(str(key))
        return None if v is None else str(v)

    def setProp(self, key: Any, value: Any) -> None:
        self.setProperty(key, None if value is None else str(value))

    def getProperties(self) -> dict[str, Any]:
        return self.img.metadata

    def getInfoProperty(self) -> str | None:
        v = self.img.metadata.get("Info")
        return None if v is None else str(v)

    def getOriginalFileInfo(self) -> FileInfo:
        return FileInfo(self.img)

    getFileInfo = getOriginalFileInfo

    def setFileInfo(self, fi: Any) -> None:
        pass

    def getNumericProperty(self, key: Any) -> float:
        try:
            return float(self.img.metadata.get(str(key)))
        except (TypeError, ValueError):
            return float("nan")

    def getStringProperty(self, key: Any) -> str | None:
        return self.getProp(key)

    # --- ROI / overlay
    def getRoi(self) -> Any:
        from ._rois import Roi
        r = self._ctx().roi
        return Roi._wrap(r) if r is not None else None

    roi = property(lambda self: self.getRoi())

    def setRoi(self, *args: Any) -> None:
        from ._rois import Roi
        ctx = self._ctx()
        if not args or args[0] is None:
            ctx.roi = None
            return
        if len(args) >= 4:
            ctx.roi = {"type": "rect", "x": float(args[0]), "y": float(args[1]), "w": float(args[2]), "h": float(args[3])}
        elif isinstance(args[0], Rectangle):
            r = args[0]
            ctx.roi = {"type": "rect", "x": float(r.x), "y": float(r.y), "w": float(r.width), "h": float(r.height)}
        elif isinstance(args[0], Roi):
            args[0]._imp = self
            ctx.roi = dict(args[0]._roi)
        elif isinstance(args[0], dict):
            ctx.roi = dict(args[0])
        else:
            raise TypeError("setRoi expects a Roi, a Rectangle or x, y, w, h")

    def deleteRoi(self) -> None:
        self._ctx().roi = None

    killRoi = deleteRoi

    def restoreRoi(self) -> None:
        self._ctx().ij.restoreSelection()

    def saveRoi(self) -> None:
        pass

    def getMask(self) -> ImageProcessor | None:
        ip = self.getProcessor()
        ip._roi = self._ctx().roi
        return ip.getMask()

    def getOverlay(self) -> Any:
        from ._rois import Overlay
        items = [o for o in self.img.overlay if isinstance(o, dict)]
        return Overlay(_img=self.img) if items else None

    overlay = property(lambda self: self.getOverlay())

    def setOverlay(self, *args: Any) -> None:
        from ._rois import Overlay, Roi
        img = self.img
        if not args or args[0] is None:
            img.overlay = []
        elif isinstance(args[0], Overlay):
            img.overlay = args[0]._items()
        elif isinstance(args[0], Roi):
            item = dict(args[0]._roi)
            if len(args) > 1 and args[1] is not None:
                item["color"] = color_hex(args[1])
            if len(args) > 2 and args[2] is not None:
                item["stroke_width"] = float(args[2])
            if len(args) > 3 and args[3] is not None:
                item["fill"] = color_hex(args[3])
            img.overlay = [item]
        else:
            raise TypeError("setOverlay expects an Overlay or a Roi")
        img.rev += 1
        self._pos_changed()

    def setHideOverlay(self, b: Any) -> None:
        self.img.metadata["hide_overlay"] = bool(b)

    def getHideOverlay(self) -> bool:
        return bool(self.img.metadata.get("hide_overlay", False))

    def flatten(self) -> "ImagePlus":
        ctx = self._ctx()
        img = self.img
        with temporarily_in_session(ctx, img):
            out = ctx.session.run_op("overlay.flatten", {}, image_id=img.id, record=False)
        new = ctx.session.images.get((out.get("new_images") or [None])[0])
        if new is None:
            return ImagePlus(_img=img.copy(name=f"{img.name}-1"))
        return ImagePlus(_img=new)

    # --- statistics
    def getStatistics(self, m_options: Any = None, n_bins: Any = 256, hist_min: Any = 0.0, hist_max: Any = 0.0) -> ImageStatistics:
        img = self.img
        d = img.display
        roi = self._ctx().roi
        lo = None if not hist_min and not hist_max else float(hist_min)
        hi = None if not hist_min and not hist_max else float(hist_max)
        return ImageStatistics(img, roi, d.t, d.z, d.c, n_bins=int(n_bins or 256), hist_min=lo, hist_max=hi)

    getAllStatistics = getStatistics
    getRawStatistics = getStatistics

    @property
    def statistics(self) -> ImageStatistics:
        return self.getStatistics()

    # --- display
    def getDisplayRangeMin(self) -> float:
        return self.getProcessor().getMin()

    def getDisplayRangeMax(self) -> float:
        return self.getProcessor().getMax()

    def setDisplayRange(self, lo: Any, hi: Any, channels: Any = None) -> None:
        img = self.img
        if img.rgb:
            return
        if channels is None:
            self.getProcessor().setMinAndMax(lo, hi)
        else:
            for i, ch in enumerate(img.display.channels):
                if int(channels) & (1 << i):
                    ch.min, ch.max = float(lo), float(hi)
            img.rev += 1
        self._pos_changed()

    def resetDisplayRange(self) -> None:
        self.getProcessor().resetMinAndMax()
        self._pos_changed()

    def resetDisplayRanges(self) -> None:
        img = self.img
        for c, ch in enumerate(img.display.channels):
            if not img.rgb:
                plane = img.data[:, :, c]
                ch.min, ch.max = float(plane.min()), float(plane.max())
        img.rev += 1
        self._pos_changed()

    def getDisplayMode(self) -> int:
        return {"composite": self.COMPOSITE, "single": self.COLOR, "grayscale": self.GRAYSCALE}.get(self.img.display.mode, self.COMPOSITE)

    getCompositeMode = getDisplayMode
    getMode = getDisplayMode

    def setDisplayMode(self, mode: Any) -> None:
        img = self.img
        img.display.mode = {self.COMPOSITE: "composite", self.COLOR: "single", self.GRAYSCALE: "grayscale"}.get(int(mode), "composite")
        img.rev += 1
        self._pos_changed()

    setMode = setDisplayMode

    def setActiveChannels(self, s: Any) -> None:
        img = self.img
        bits = str(s)
        for i, ch in enumerate(img.display.channels):
            ch.visible = i < len(bits) and bits[i] == "1"
        img.rev += 1
        self._pos_changed()

    def getActiveChannels(self) -> str:
        return "".join("1" if ch.visible else "0" for ch in self.img.display.channels)

    def getLuts(self) -> list[str]:
        return [ch.lut for ch in self.img.display.channels]

    def setLut(self, lut: Any) -> None:
        img = self.img
        name = getattr(lut, "name", lut)
        if isinstance(name, str):
            img.display.channels[img.display.c].lut = name
            img.rev += 1
            self._pos_changed()

    def setChannelLut(self, lut: Any, channel: Any = None) -> None:
        img = self.img
        name = getattr(lut, "name", lut)
        if isinstance(name, str):
            c = int(channel) - 1 if channel is not None else img.display.c
            img.display.channels[c].lut = name
            img.rev += 1
            self._pos_changed()

    def getChannelLut(self, channel: Any = None) -> str:
        img = self.img
        return img.display.channels[int(channel) - 1 if channel is not None else img.display.c].lut

    def setColor(self, c: Any) -> None:
        self._ctx().ij.setColor(color_hex(c))

    def getDefaultDirectory(self) -> str:
        return os.getcwd()

    # --- copies
    def duplicate(self) -> "ImagePlus":
        img = self.img
        return ImagePlus(_img=img.copy(name=f"DUP_{img.name}"))

    def crop(self, options: Any = None) -> "ImagePlus":
        """``crop()`` = the selection's bounding box of the current plane; ``crop("stack")`` of every plane."""
        img = self.img
        roi = self._ctx().roi
        if roi is not None and roimod.is_area(roi):
            x, y, w, h = roimod.bounding_box(roi)
            x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
            x1, y1 = min(img.width, int(math.ceil(x + w))), min(img.height, int(math.ceil(y + h)))
        else:
            x0, y0, x1, y1 = 0, 0, img.width, img.height
        whole = options is not None and "stack" in str(options).lower()
        d = img.display
        data = img.data[:, :, :, y0:y1, x0:x1] if whole else (img.data[d.t:d.t + 1, d.z:d.z + 1, :, y0:y1, x0:x1] if img.rgb else img.data[d.t:d.t + 1, d.z:d.z + 1, d.c:d.c + 1, y0:y1, x0:x1])
        new = img.copy(name=f"{img.name}-1", data=np.ascontiguousarray(data))
        if not whole:
            new.display.t = new.display.z = new.display.c = 0
            if not img.rgb:
                new.display.channels = [type(img.display.channels[d.c])(**img.display.channels[d.c].to_dict())]
        return ImagePlus(_img=new)

    def resize(self, w: Any, h: Any, method: Any = "bilinear") -> "ImagePlus":
        img = self.img
        ctx = self._ctx()
        interp = {"none": "None", "nearest": "None", "bilinear": "Bilinear", "bicubic": "Bicubic"}.get(str(method).lower(), "Bilinear")
        with temporarily_in_session(ctx, img):
            out = ctx.session.run_op("image.scale", {"width": int(w), "height": int(h), "x_scale": 0, "y_scale": 0, "interpolation": interp, "create_new": True}, image_id=img.id, record=False)
        new = ctx.session.images.get((out.get("new_images") or [None])[0])
        return ImagePlus(_img=new if new is not None else img.copy())

    def getNativeImage(self) -> None:
        return None


class CompositeImage(ImagePlus):
    """``ij.CompositeImage(imp[, mode])``: the same image with a display mode."""

    def __init__(self, imp: Any = None, mode: Any = None, **kw: Any) -> None:
        if isinstance(imp, ImagePlus):
            super().__init__(_img=imp._img)
            if mode is not None:
                self.setDisplayMode(mode)
        else:
            super().__init__(imp, mode, **kw)

    def setChannelColor(self, color: Any) -> None:
        pass

    def getChannelColor(self) -> Color:
        return Color.white

    def setChannelLut(self, lut: Any, channel: Any = None) -> None:  # noqa: D102
        super().setChannelLut(lut, channel)

    def updateChannelAndDraw(self) -> None:
        self.updateAndDraw()

    def updateAllChannelsAndDraw(self) -> None:
        self.updateAndDraw()

    def reset(self) -> None:
        self.resetDisplayRanges()
