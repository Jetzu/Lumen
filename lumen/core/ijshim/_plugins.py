"""The ``ij.plugin`` / ``ij.plugin.filter`` classes Fiji scripts instantiate directly, each a thin layer over the
matching ImageJ command of the macro layer (``Duplicator``, ``ZProjector``, ``ChannelSplitter``,
``ImageCalculator``, ``HyperStackConverter``, ``RGBStackMerge``, ``SubstackMaker``, ``Concatenator``,
``ContrastEnhancer``, ``GaussianBlur``, ``RankFilters``, ``BackgroundSubtracter``, ``Convolver``,
``ParticleAnalyzer``, ``ThresholdToSelection``, ``MaximumFinder``, ``EDM``, ``Binary``, ``Filters``,
``ImageConverter`` / ``StackConverter``, ``AutoThresholder``)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import roi as roimod
from ._context import current
from ._images import ByteProcessor, ImagePlus, ImageProcessor, ImageStack, as_current, temporarily_in_session
from ._java import Polygon, color_hex
from ._rois import Roi
from ._tables import Measurements, ResultsTable, RoiManager, measurement_keys


def _new_images(out: Any) -> list[ImagePlus]:
    ctx = current()
    ids = out.get("new_images") if isinstance(out, dict) else None
    return [ImagePlus(_img=ctx.session.images[i]) for i in (ids or []) if i in ctx.session.images]


def _run_new(imp: Any, command: str, options: str = "") -> ImagePlus | None:
    """Run an ImageJ command on ``imp`` and return the image it created (if any)."""
    ctx = current()
    with as_current(ctx, imp):
        out = ctx.ij.run(command, options)
    news = _new_images(out)
    return news[-1] if news else None


def _ip_run(ip: ImageProcessor, op_id: str, params: dict[str, Any]) -> None:
    ip._op(op_id, params)


# ====================================================================== ij.plugin
class Duplicator:
    """``Duplicator().run(imp)`` / ``run(imp, firstC, lastC, firstZ, lastZ, firstT, lastT)`` / ``run(imp, firstZ, lastZ)``
    – a script-private copy (``show()`` adds the tab); the selection crops it like ImageJ."""

    def run(self, imp: ImagePlus, *ranges: Any) -> ImagePlus:
        img = imp.img
        ctx = current()
        roi = ctx.roi
        if roi is not None and roimod.is_area(roi):
            x, y, w, h = roimod.bounding_box(roi)
            x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
            x1, y1 = min(img.width, int(math.ceil(x + w))), min(img.height, int(math.ceil(y + h)))
        else:
            x0, y0, x1, y1 = 0, 0, img.width, img.height
        c0, c1, z0, z1, t0, t1 = 0, img.nc, 0, img.nz, 0, img.nt
        if len(ranges) >= 6:
            c0, c1, z0, z1, t0, t1 = (int(v) for v in ranges[:6])
            c0, z0, t0 = max(c0 - 1, 0), max(z0 - 1, 0), max(t0 - 1, 0)
            c1, z1, t1 = min(c1, img.nc), min(z1, img.nz), min(t1, img.nt)
            if img.rgb:
                c0, c1 = 0, 3
        elif len(ranges) >= 2:  # firstSlice, lastSlice of a plain stack (frames when it is a time series)
            a, b = max(int(ranges[0]) - 1, 0), int(ranges[1])
            if img.nz == 1 and img.nt > 1:
                t0, t1 = a, min(b, img.nt)
            else:
                z0, z1 = a, min(b, img.nz)
        data = img.data[t0:t1, z0:z1, c0:c1, y0:y1, x0:x1]
        new = img.copy(name=f"DUP_{img.name}", data=np.ascontiguousarray(data))
        if not img.rgb:
            new.display.channels = [type(ch)(**ch.to_dict()) for ch in img.display.channels[c0:c1]]
        new.display.t = new.display.z = new.display.c = 0
        new.overlay = []
        return ImagePlus(_img=new)

    def crop(self, imp: ImagePlus) -> ImagePlus:
        return self.run(imp)


class ZProjector:
    """``ZProjector.run(imp, "max")`` / ``ZProjector(imp)`` + ``setMethod`` / ``doProjection`` / ``getProjection``."""

    AVG_METHOD, MAX_METHOD, MIN_METHOD, SUM_METHOD, SD_METHOD, MEDIAN_METHOD = 0, 1, 2, 3, 4, 5
    METHODS = ["avg", "max", "min", "sum", "sd", "median"]
    _NAMES = {"avg": "Average Intensity", "average": "Average Intensity", "mean": "Average Intensity", "max": "Max Intensity", "min": "Min Intensity",
              "sum": "Sum Slices", "sd": "Standard Deviation", "std": "Standard Deviation", "median": "Median"}

    def __init__(self, imp: ImagePlus | None = None) -> None:
        self._imp = imp
        self._method = self.MAX_METHOD
        self._start = 1
        self._stop = 0
        self._result: ImagePlus | None = None

    @staticmethod
    def run(imp: ImagePlus, method: Any, start: Any = None, stop: Any = None) -> ImagePlus | None:
        words = str(method).lower().split()
        name = ZProjector._NAMES.get(words[0] if words else "max", "Max Intensity")
        opts = f"projection=[{name}]"
        if start is not None:
            opts += f" start={int(start)}"
        if stop is not None:
            opts += f" stop={int(stop)}"
        if "all" in words:
            opts += " all"
        return _run_new(imp, "Z Project...", opts)

    def setImage(self, imp: ImagePlus) -> None:
        self._imp = imp

    def setMethod(self, m: Any) -> None:
        self._method = int(m)

    def setStartSlice(self, n: Any) -> None:
        self._start = int(n)

    def setStopSlice(self, n: Any) -> None:
        self._stop = int(n)

    def doProjection(self, *a: Any) -> None:
        name = self.METHODS[self._method] if 0 <= self._method < len(self.METHODS) else "max"
        self._result = ZProjector.run(self._imp, name, self._start, self._stop or None)

    def doHyperStackProjection(self, all_frames: Any = True) -> None:
        name = self.METHODS[self._method] if 0 <= self._method < len(self.METHODS) else "max"
        self._result = ZProjector.run(self._imp, name + (" all" if all_frames else ""), self._start, self._stop or None)

    doRGBProjection = doProjection

    def getProjection(self) -> ImagePlus | None:
        return self._result


class ChannelSplitter:
    """``ChannelSplitter.split(imp)`` → one ``ImagePlus`` per channel (shown, like Image ▸ Color ▸ Split Channels)."""

    @staticmethod
    def split(imp: ImagePlus) -> list[ImagePlus]:
        ctx = current()
        with as_current(ctx, imp):
            out = ctx.ij.run("Split Channels", "")
        return _new_images(out)

    @staticmethod
    def getChannel(imp: ImagePlus, c: Any) -> ImageStack:
        img = imp.img
        ci = int(c) - 1
        st = ImageStack(img.width, img.height)
        for t in range(img.nt):
            for z in range(img.nz):
                st.addSlice(None, ImageProcessor._standalone(img.data[t, z, ci].copy()))
        return st


class ImageCalculator:
    """``ImageCalculator().run("Add create 32-bit", imp1, imp2)`` (returns the new image when ``create``) and the
    static ``ImageCalculator.run(imp1, imp2, "add create")``."""

    @staticmethod
    def _do(op: str, imp1: ImagePlus, imp2: ImagePlus) -> ImagePlus | None:
        ctx = current()
        from ._images import ensure_in_session
        ensure_in_session(ctx, imp1.img)
        ensure_in_session(ctx, imp2.img)
        words = op.lower().replace("-", " ").split()
        create = "create" in words
        float_result = "32" in words or "float" in words
        stack = "stack" in words
        name = next((w for w in words if w in ("add", "subtract", "multiply", "divide", "and", "or", "xor", "min", "max", "average", "difference", "copy", "transparent")), "subtract")
        opts = f"image1=[{imp1.img.name}] operation={name.capitalize() if name not in ('and', 'or', 'xor') else name.upper()} image2=[{imp2.img.name}]"
        if create:
            opts += " create"
        if float_result:
            opts += " 32-bit"
        if stack:
            opts += " stack"
        with as_current(ctx, imp1):
            out = ctx.ij.run("Image Calculator...", opts)
        news = _new_images(out)
        return news[-1] if news else (None if create else imp1)


def _ic_run(*args: Any) -> ImagePlus | None:
    """``ImageCalculator.run`` works both as ``ImageCalculator().run(op, a, b)`` and ``ImageCalculator.run(a, b, op)``."""
    if args and isinstance(args[0], ImageCalculator):
        args = args[1:]
    if args and isinstance(args[0], ImagePlus):
        imp1, imp2, op = args[0], args[1], str(args[2])
    else:
        op, imp1, imp2 = str(args[0]), args[1], args[2]
    return ImageCalculator._do(op, imp1, imp2)


class _InstanceOrStatic:
    """A descriptor making a method callable both on the class and on instances with the same free function."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def __get__(self, obj: Any, cls: Any = None) -> Any:
        fn = self._fn
        if obj is None:
            return fn
        return lambda *a, **k: fn(obj, *a, **k)


ImageCalculator.run = _InstanceOrStatic(_ic_run)  # type: ignore[assignment]


class HyperStackConverter:
    """``HyperStackConverter.toHyperStack(imp, c, z, t[, order[, mode]])`` / ``toStack(imp)``."""

    COMPOSITE, COLOR, GRAYSCALE = 1, 2, 3

    @staticmethod
    def toHyperStack(imp: ImagePlus, c: Any, z: Any, t: Any, order: Any = "xyczt", mode: Any = None) -> ImagePlus:
        ctx = current()
        img = imp.img
        with temporarily_in_session(ctx, img):
            ctx.session.run_op("stacks.to_hyperstack", {"channels": int(c), "slices": int(z), "frames": int(t), "order": str(order).lower()[:5] if order else "xyczt"},
                               image_id=img.id, record=False)
        if mode is not None:
            m = {1: "composite", 2: "single", 3: "grayscale"}.get(int(mode) if not isinstance(mode, str) else {"composite": 1, "color": 2, "grayscale": 3}.get(mode.lower(), 1), "composite")
            img.display.mode = m
        return ImagePlus(_img=img)

    @staticmethod
    def toStack(imp: ImagePlus) -> ImagePlus:
        ctx = current()
        img = imp.img
        with temporarily_in_session(ctx, img):
            ctx.session.run_op("stacks.hyperstack_to_stack", {}, image_id=img.id, record=False)
        return ImagePlus(_img=img)

    @staticmethod
    def shuffle(imp: ImagePlus, order: Any) -> ImagePlus:
        return imp


class RGBStackMerge:
    """``RGBStackMerge.mergeChannels([imp1, imp2, …], keep)`` → the composite (``None`` entries skip a colour)."""

    @staticmethod
    def mergeChannels(imps: Any, keep: Any = False) -> ImagePlus | None:
        ctx = current()
        from ._images import ensure_in_session
        parts = []
        for i, imp in enumerate(list(imps)[:7]):
            if imp is None:
                continue
            ensure_in_session(ctx, imp.img)
            parts.append(f"c{i + 1}=[{imp.img.name}]")
        if not parts:
            return None
        opts = " ".join(parts) + " create" + (" keep" if keep else "")
        first = next(imp for imp in imps if imp is not None)
        return _run_new(first, "Merge Channels...", opts)

    @staticmethod
    def mergeStacks(*a: Any) -> ImagePlus | None:
        imps = [x for x in a if isinstance(x, ImagePlus)]
        return RGBStackMerge.mergeChannels(imps, True)

    def mergeHyperstacks(self, imps: Any, keep: Any = False) -> ImagePlus | None:
        return RGBStackMerge.mergeChannels(imps, keep)


class SubstackMaker:
    """``SubstackMaker().makeSubstack(imp, "1-10")`` – a new image of the given slices (shown)."""

    def makeSubstack(self, imp: ImagePlus, spec: Any) -> ImagePlus | None:
        ctx = current()
        img = imp.img
        axis = "T" if (img.nz == 1 and img.nt > 1) else "Z"
        with as_current(ctx, imp):
            out = ctx.session.run_op("stacks.substack", {"slices": str(spec), "axis": axis}, image_id=img.id, record=False)
        news = _new_images(out)
        return news[-1] if news else None


class Concatenator:
    """``Concatenator().concatenate(imp1, imp2, keep)`` / ``Concatenator.run(imp1, imp2)`` along time."""

    def concatenate(self, *args: Any) -> ImagePlus | None:
        imps = [a for a in args if isinstance(a, ImagePlus)]
        if len(imps) < 2 and args and isinstance(args[0], (list, tuple)):
            imps = list(args[0])
        keep = bool(args[-1]) if args and isinstance(args[-1], bool) else False
        ctx = current()
        from ._images import ensure_in_session
        for imp in imps:
            ensure_in_session(ctx, imp.img)
        result = imps[0]
        for nxt in imps[1:]:
            with as_current(ctx, result):
                out = ctx.session.run_op("stacks.concatenate", {"image1": result.img.id, "image2": nxt.img.id, "axis": "T", "title": "Concatenated"}, record=False)
            news = _new_images(out)
            if not news:
                return None
            result = news[-1]
        if not keep:
            for imp in imps:
                imp.close()
        return result

    concatenateHyperstacks = concatenate

    @staticmethod
    def run(*imps: Any) -> ImagePlus | None:
        return Concatenator().concatenate(*imps, True)


class ContrastEnhancer:
    """``ContrastEnhancer().stretchHistogram(imp | ip, saturated)`` / ``equalize(imp | ip)``."""

    def __init__(self) -> None:
        self._normalize = False
        self._process_stack = False
        self._use_stack_histogram = False

    def setNormalize(self, b: Any) -> None:
        self._normalize = bool(b)

    def setProcessStack(self, b: Any) -> None:
        self._process_stack = bool(b)

    def setUseStackHistogram(self, b: Any) -> None:
        self._use_stack_histogram = bool(b)

    def stretchHistogram(self, target: Any, saturated: Any = 0.35, *rest: Any) -> None:
        params = {"saturated": float(saturated), "normalize": self._normalize, "equalize": False, "process_all": self._process_stack}
        if isinstance(target, ImagePlus):
            ctx = current()
            with temporarily_in_session(ctx, target.img):
                ctx.session.run_op("process.enhance_contrast", params, image_id=target.img.id, roi=ctx.roi, scope="stack" if self._process_stack else "plane", record=False)
        else:
            _ip_run(target, "process.enhance_contrast", params)

    def equalize(self, target: Any, *rest: Any) -> None:
        params = {"saturated": 0.35, "normalize": False, "equalize": True, "process_all": self._process_stack}
        if isinstance(target, ImagePlus):
            ctx = current()
            with temporarily_in_session(ctx, target.img):
                ctx.session.run_op("process.enhance_contrast", params, image_id=target.img.id, roi=ctx.roi, scope="stack" if self._process_stack else "plane", record=False)
        else:
            _ip_run(target, "process.enhance_contrast", params)


# ====================================================================== ij.plugin.filter
class PlugInFilter:
    DOES_8G, DOES_8C, DOES_16, DOES_32, DOES_RGB = 1, 2, 4, 8, 16
    DOES_ALL = 31
    DOES_STACKS, SUPPORTS_MASKING, NO_CHANGES, NO_UNDO, NO_IMAGE_REQUIRED, ROI_REQUIRED, STACK_REQUIRED = 32, 64, 128, 256, 512, 1024, 2048
    CONVERT_TO_FLOAT, SNAPSHOT, FINAL_PROCESSING, KEEP_PREVIEW, PARALLELIZE_STACKS, PARALLELIZE_IMAGES, KEEP_THRESHOLD = 4096, 8192, 16384, 32768, 65536, 131072, 262144
    DONE = 4096


class GaussianBlur:
    def blurGaussian(self, ip: ImageProcessor, sigma_x: Any, sigma_y: Any = None, accuracy: Any = None) -> None:
        _ip_run(ip, "filters.gaussian", {"sigma": float(sigma_x)})

    def blur(self, ip: ImageProcessor, sigma: Any) -> bool:
        _ip_run(ip, "filters.gaussian", {"sigma": float(sigma)})
        return True

    def blur1Direction(self, *a: Any) -> None:
        raise ValueError("GaussianBlur.blur1Direction is not supported; use blurGaussian")

    def showProgress(self, *a: Any) -> None:
        pass


class RankFilters:
    MEAN, MIN, MAX, VARIANCE, MEDIAN, OUTLIERS, DESPECKLE, REMOVE_NAN, OPEN, CLOSE, TOP_HAT = range(11)
    BRIGHT_OUTLIERS, DARK_OUTLIERS = 0, 1

    def rank(self, ip: ImageProcessor, radius: Any, kind: Any, which: Any = 0, threshold: Any = 50.0, *rest: Any) -> None:
        k = int(kind)
        r = float(radius)
        if k == self.MEAN:
            _ip_run(ip, "filters.mean", {"radius": r})
        elif k == self.MIN:
            _ip_run(ip, "filters.minimum", {"radius": r})
        elif k == self.MAX:
            _ip_run(ip, "filters.maximum", {"radius": r})
        elif k == self.VARIANCE:
            _ip_run(ip, "filters.variance", {"radius": r})
        elif k == self.MEDIAN:
            _ip_run(ip, "filters.median", {"radius": r})
        elif k == self.OUTLIERS:
            _ip_run(ip, "noise.remove_outliers", {"radius": r, "threshold": float(threshold), "which": "Dark" if int(which) else "Bright"})
        elif k == self.DESPECKLE:
            _ip_run(ip, "noise.despeckle", {})
        elif k == self.REMOVE_NAN:
            _ip_run(ip, "noise.remove_nans", {})
        elif k == self.OPEN:
            _ip_run(ip, "filters.minimum", {"radius": r})
            _ip_run(ip, "filters.maximum", {"radius": r})
        elif k == self.CLOSE:
            _ip_run(ip, "filters.maximum", {"radius": r})
            _ip_run(ip, "filters.minimum", {"radius": r})
        elif k == self.TOP_HAT:
            _ip_run(ip, "filters.tophat", {"radius": r, "light_background": bool(which)})
        else:
            raise ValueError(f"RankFilters.rank: unknown filter type {kind}")


class BackgroundSubtracter:
    def rollingBallBackground(self, ip: ImageProcessor, radius: Any, create_background: Any = False, light_background: Any = False,
                              use_paraboloid: Any = False, do_presmooth: Any = True, correct_corners: Any = True) -> None:
        _ip_run(ip, "process.subtract_background", {"radius": float(radius), "light_background": bool(light_background), "create_background": bool(create_background),
                                                    "method": "Sliding paraboloid" if use_paraboloid else "Rolling ball", "smooth": bool(do_presmooth)})

    def rollingBallBrightBackground(self, ip: ImageProcessor, radius: Any, create_background: Any = False, *rest: Any) -> None:
        self.rollingBallBackground(ip, radius, create_background, True)

    def subtractRGBBackround(self, *a: Any) -> None:
        self.rollingBallBackground(*a)


class Convolver:
    def __init__(self) -> None:
        self._normalize = True

    def setNormalize(self, b: Any) -> None:
        self._normalize = bool(b)

    def convolve(self, ip: ImageProcessor, kernel: Any, kw: Any, kh: Any) -> bool:
        k = np.asarray(kernel, np.float64).reshape(int(kh), int(kw))
        _ip_run(ip, "filters.convolve", {"kernel": "\n".join(" ".join(f"{v:g}" for v in row) for row in k), "normalize": self._normalize})
        return True

    convolveFloat = convolve


class Filters:
    @staticmethod
    def invert(ip: ImageProcessor) -> None:
        ip.invert()


class EDM:
    """``EDM().toEDM(ip)`` (Euclidean distance map in place), ``makeFloatEDM``, ``toWatershed(ip)``, ``toUEP(ip)``."""

    def toEDM(self, ip: ImageProcessor) -> None:
        _ip_run(ip, "binary.distance_map", {"output": "Overwrite"})   # ImageJ: the 8-bit EDM replaces the image

    def makeFloatEDM(self, ip: ImageProcessor, background: Any = 0, edges: Any = False) -> ImageProcessor:
        """``EDM.makeFloatEDM(ip, backgroundValue, edgesAreBackground)``: the exact float32 Euclidean distance of every
        pixel ≠ background to the nearest background pixel (pixels outside the image count as background when ``edges``)."""
        from ..ops._imagej_binary import edm_float
        from ._images import FloatProcessor
        p = ip._plane()
        fg = p != (float(background) if p.dtype == np.float32 else int(background))
        if edges:
            edm = edm_float(np.pad(fg, 1))[1:-1, 1:-1]
        else:
            edm = edm_float(fg)
        return FloatProcessor(_array=np.ascontiguousarray(edm, dtype=np.float32))

    def toWatershed(self, ip: ImageProcessor) -> None:
        _ip_run(ip, "binary.watershed", {"method": "ImageJ EDM/UEP"})

    def toUEP(self, ip: ImageProcessor) -> None:
        _ip_run(ip, "binary.ultimate_points", {"output": "Overwrite"})   # ImageJ: the UEP values overwrite the image

    def run(self, ip: ImageProcessor) -> None:
        self.toEDM(ip)

    def setup(self, arg: Any, imp: Any) -> int:
        self._arg = str(arg)
        return 1


class Binary:
    """``Binary().setup("fill" | "open" | "close" | "outline" | "skel" | "erode" | "dilate", imp); run(ip)``."""

    def __init__(self) -> None:
        self._arg = ""

    def setup(self, arg: Any, imp: Any) -> int:
        self._arg = str(arg).lower()
        return 1

    def run(self, ip: ImageProcessor) -> None:
        a = self._arg
        op = {"erode": "binary.erode", "dilate": "binary.dilate", "open": "binary.open", "close": "binary.close", "outline": "binary.outline",
              "skel": "binary.skeletonize", "skeletonize": "binary.skeletonize", "fill": "binary.fill_holes"}.get(a)
        if op is None:
            raise ValueError(f"Binary: unknown operation {a!r}")
        _ip_run(ip, op, {})


class ThresholdToSelection:
    """``ThresholdToSelection.run(imp)`` / ``ThresholdToSelection().convert(ip)`` → the thresholded area as a Roi."""

    @staticmethod
    def run(imp: ImagePlus) -> Roi | None:
        ctx = current()
        with as_current(ctx, imp):
            out = ctx.ij._run("selection.create", {})
        r = out.get("roi") if isinstance(out, dict) else None
        return Roi._wrap(r) if r is not None else None

    def convert(self, ip: ImageProcessor) -> Roi | None:
        thr = ip._current_threshold()
        if thr is None:
            raise ValueError("ThresholdToSelection.convert: the processor has no threshold (ip.setThreshold)")
        g = ip._gray()
        m = (g >= thr[0]) & (g <= thr[1])
        r = roimod.from_mask(m)
        return Roi._wrap(r) if r is not None else None


class MaximumFinder:
    """``MaximumFinder().findMaxima(ip, tolerance, [threshold,] outputType, excludeOnEdges, isEDM)`` and
    ``getMaxima(ip, tolerance, excludeOnEdges)`` (Process ▸ Find Maxima)."""

    SINGLE_POINTS, IN_TOLERANCE, SEGMENTED, POINT_SELECTION, LIST, COUNT = 0, 1, 2, 3, 4, 5
    NO_THRESHOLD = -808080.0
    _OUTPUT = {0: "Single Points", 1: "Maxima Within Tolerance", 2: "Segmented Particles", 3: "Point Selection", 4: "List", 5: "Count"}

    def findMaxima(self, ip: ImageProcessor, tolerance: Any, *args: Any) -> Any:
        args = list(args)
        if len(args) >= 4:  # (threshold, outputType, excludeOnEdges, isEDM)
            args = args[1:]
        output = int(args[0]) if args else self.POINT_SELECTION
        exclude = bool(args[1]) if len(args) > 1 else False
        light = bool(getattr(self, "_light", False))
        out = ip._op("analyze.find_maxima", {"prominence": float(tolerance), "output": self._OUTPUT.get(output, "Point Selection"), "exclude_edges": exclude,
                                              "light_background": light, "strict": False})
        ctx = current()
        if output in (self.SINGLE_POINTS, self.IN_TOLERANCE, self.SEGMENTED):
            news = _new_images(out)
            if not news:
                return None
            imp = news[-1]
            bp = ByteProcessor(_array=imp.img.data[0, 0, 0].copy())
            imp.close()
            return bp
        if output == self.POINT_SELECTION:
            r = out.get("roi")
            if r is not None:
                ctx.roi = r
            return None
        return None  # LIST / COUNT went to the Results table

    def getMaxima(self, ip: ImageProcessor, tolerance: Any, exclude_on_edges: Any = False, *rest: Any) -> Polygon:
        out = ip._op("analyze.find_maxima", {"prominence": float(tolerance), "output": "Point Selection", "exclude_edges": bool(exclude_on_edges), "light_background": False, "strict": False})
        r = out.get("roi") or {}
        pts = r.get("points", [])
        return Polygon([int(p[0]) for p in pts], [int(p[1]) for p in pts])

    def setup(self, arg: Any, imp: Any) -> int:
        self._light = "light" in str(arg).lower()
        return 1


class ParticleAnalyzer:
    """``ParticleAnalyzer(options, measurements, rt, minSize, maxSize[, minCirc, maxCirc]).analyze(imp)``.

    The results land in ``rt`` (the Results table when it is the live one, else the private table) and the ROIs in
    the ROI Manager with ``ADD_TO_MANAGER``; ``getOutputImage()`` is the outlines / masks image."""

    SHOW_RESULTS, SHOW_SUMMARY, SHOW_OUTLINES, EXCLUDE_EDGE_PARTICLES, SHOW_ROI_MASKS, SHOW_PROGRESS, CLEAR_WORKSHEET, RECORD_STARTS, DISPLAY_SUMMARY = (
        1, 2, 4, 8, 16, 32, 64, 128, 256)
    SHOW_NONE, INCLUDE_HOLES, ADD_TO_MANAGER, SHOW_MASKS, FOUR_CONNECTED, IN_SITU_SHOW, SHOW_OVERLAY_OUTLINES, SHOW_OVERLAY_MASKS, COMPOSITE_ROIS = (
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    _rm: RoiManager | None = None

    def __init__(self, options: Any = 0, measurements: Any = 0, rt: ResultsTable | None = None, min_size: Any = 0.0, max_size: Any = float("inf"),
                 min_circ: Any = 0.0, max_circ: Any = 1.0) -> None:
        self._options = int(options)
        self._measurements = int(measurements)
        self._rt = rt
        self._min = float(min_size)
        self._max = float(max_size)
        self._minc = float(min_circ)
        self._maxc = float(max_circ)
        self._hide = False
        self._output: ImagePlus | None = None
        self._manager: RoiManager | None = None

    def setHideOutputImage(self, b: Any) -> None:
        self._hide = bool(b)

    def setRoiManager(self, rm: Any) -> None:
        self._manager = rm

    @staticmethod
    def setRoiManager_static(rm: Any) -> None:
        ParticleAnalyzer._rm = rm

    def getOutputImage(self) -> ImagePlus | None:
        return self._output

    def analyze(self, imp: ImagePlus, ip: Any = None) -> bool:
        ctx = current()
        o = self._options
        show = "Outlines" if o & self.SHOW_OUTLINES else "Masks" if o & self.SHOW_MASKS else "Count Masks" if o & self.SHOW_ROI_MASKS else "Nothing"
        params = {"min_size": self._min, "max_size": 0.0 if not math.isfinite(self._max) else self._max, "min_circularity": self._minc, "max_circularity": min(self._maxc, 1.0),
                  "show": show, "display_results": True, "clear_results": bool(o & self.CLEAR_WORKSHEET), "summarize": bool(o & (self.SHOW_SUMMARY | self.DISPLAY_SUMMARY)),
                  "add_to_manager": bool(o & self.ADD_TO_MANAGER), "exclude_on_edges": bool(o & self.EXCLUDE_EDGE_PARTICLES), "include_holes": bool(o & self.INCLUDE_HOLES),
                  "all_planes": False, "composite_rois": bool(o & self.COMPOSITE_ROIS), "in_situ_show": bool(o & self.IN_SITU_SHOW), "record_starts": bool(o & self.RECORD_STARTS)}
        saved_keys = list(ctx.session.measurements)
        if self._measurements:
            ctx.session.measurements = measurement_keys(self._measurements) or saved_keys
        before = len(ctx.session.results.rows)
        try:
            with as_current(ctx, imp):
                out = ctx.ij._run("analyze.particles", params)
        finally:
            ctx.session.measurements = saved_keys
        news = _new_images(out)
        if news:
            self._output = news[-1]
            if self._hide:
                self._output.hide()
        rows = ctx.session.results.rows[before:]
        if self._rt is not None and not self._rt._live:
            for r in rows:
                self._rt.incrementCounter()
                for k, v in r.items():
                    self._rt.addValue(k, v)
            if not o & self.SHOW_RESULTS:
                del ctx.session.results.rows[before:]
                ctx.session._changed("results")
        elif not o & self.SHOW_RESULTS and self._rt is None:
            del ctx.session.results.rows[before:]
            ctx.session._changed("results")
        return True

    def setup(self, arg: Any, imp: Any) -> int:
        return 1

    def run(self, ip: Any) -> None:
        self.analyze(current().session.active and ImagePlus(_img=current().session.active), ip)

    @staticmethod
    def setLineWidth(w: Any) -> None:
        pass

    @staticmethod
    def setFontSize(s: Any) -> None:
        pass

    @staticmethod
    def setSummaryTable(rt: Any) -> None:
        pass

    @staticmethod
    def setResultsTable(rt: Any) -> None:
        pass

    @staticmethod
    def getSummaryTable() -> ResultsTable:
        return ResultsTable.getResultsTable("Summary")


# ====================================================================== ij.process converters
class ImageConverter:
    """``ImageConverter(imp).convertToGray8()`` … (Image ▸ Type); ``ImageConverter.setDoScaling(b)``."""

    _scaling = True

    def __init__(self, imp: ImagePlus) -> None:
        self._imp = imp

    def _run(self, op_id: str, params: dict[str, Any] | None = None) -> None:
        ctx = current()
        img = self._imp.img
        with temporarily_in_session(ctx, img):
            ctx.session.run_op(op_id, params or {}, image_id=img.id, record=False)

    def convertToGray8(self) -> None:
        self._run("type.8bit", {"scale": ImageConverter._scaling})

    def convertToGray16(self) -> None:
        self._run("type.16bit")

    def convertToGray32(self) -> None:
        self._run("type.32bit")

    def convertToRGB(self) -> None:
        self._run("type.rgb")

    def convertRGBtoIndexedColor(self, n: Any = 256) -> None:
        self._run("type.8bit_color")

    def convertToRGBStack(self) -> None:
        self._run("type.rgb_stack")

    def convertToHSB(self) -> None:
        self._run("type.hsb_stack")

    def convertToLab(self) -> None:
        self._run("type.lab_stack")

    def convertRGBStackToRGB(self) -> None:
        self._run("type.rgb")

    convertHSBToRGB = convertRGBStackToRGB
    convertLabToRGB = convertRGBStackToRGB

    @staticmethod
    def setDoScaling(b: Any) -> None:
        ImageConverter._scaling = bool(b)

    @staticmethod
    def getDoScaling() -> bool:
        return ImageConverter._scaling


class StackConverter(ImageConverter):
    pass


class AutoThresholder:
    """``AutoThresholder().getThreshold(method, histogram)`` – the 16 ImageJ methods on a 256-bin histogram;
    ``method`` is a name or ``AutoThresholder.Method.Otsu``."""

    class Method:
        pass

    def getThreshold(self, method: Any, histogram: Any) -> int:
        from ..threshold import METHODS, _FUNCS
        name = getattr(method, "name", method)
        name = str(name)
        if name not in METHODS:
            low = {m.lower(): m for m in METHODS}
            name = low.get(name.lower(), "Default")
        h = np.asarray(histogram, np.int64)
        return int(_FUNCS[name](h))

    @staticmethod
    def getMethods() -> list[str]:
        from ..threshold import METHODS
        return list(METHODS)


class _MethodEnum:
    def __init__(self, name: str) -> None:
        self.name = name

    def toString(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


def _install_methods() -> None:
    from ..threshold import METHODS
    for m in METHODS:
        setattr(AutoThresholder.Method, m, _MethodEnum(m))
        setattr(AutoThresholder.Method, m.upper(), _MethodEnum(m))
    setattr(AutoThresholder.Method, "IJ_IsoData", _MethodEnum("Default"))
    setattr(AutoThresholder.Method, "values", staticmethod(lambda: [getattr(AutoThresholder.Method, m) for m in METHODS]))


_install_methods()


class Thresholder:
    @staticmethod
    def setMethod(m: Any) -> None:
        pass

    @staticmethod
    def setBackground(b: Any) -> None:
        pass


class Colors:
    @staticmethod
    def decode(name: Any, default: Any = None) -> Any:
        from ._java import Color
        try:
            return Color(color_hex(name))
        except ValueError:
            return default

    @staticmethod
    def getColors() -> list[str]:
        return ["red", "green", "blue", "magenta", "cyan", "yellow", "orange", "black", "white", "gray"]

    @staticmethod
    def colorToString(c: Any) -> str:
        return color_hex(c)

    @staticmethod
    def getColor(name: Any, default: Any = None) -> Any:
        return Colors.decode(name, default)


class Macro_Runner:
    def runMacro(self, code: Any, arg: Any = "") -> str:
        return current().ij.macro(str(code), arg)

    def runMacroFile(self, path: Any, arg: Any = "") -> str:
        return current().ij.runMacro(path, arg)


class ImageInfo:
    def getImageInfo(self, imp: ImagePlus, *rest: Any) -> str:
        ctx = current()
        with as_current(ctx, imp):
            return ctx.ij.getImageInfo()


class Scaler:
    @staticmethod
    def resize(imp: ImagePlus, w: Any, h: Any, d: Any = 1, options: Any = "bilinear") -> ImagePlus:
        return imp.resize(w, h, str(options))


class Resizer:
    def zScale(self, imp: ImagePlus, depth: Any, interpolation: Any = 1) -> ImagePlus:
        ctx = current()
        img = imp.img
        with as_current(ctx, imp):
            out = ctx.session.run_op("image.scale", {"z_scale": float(depth) / max(img.nz, 1), "x_scale": 1.0, "y_scale": 1.0, "create_new": True}, image_id=img.id, record=False)
        news = _new_images(out)
        return news[-1] if news else imp


class Straightener:
    def straighten(self, imp: ImagePlus, roi: Roi, width: Any) -> ImageProcessor | None:
        ctx = current()
        saved = ctx.roi
        try:
            ctx.roi = roi._roi
            with as_current(ctx, imp):
                out = ctx.session.run_op("stacks.straighten", {"line_width": int(width)}, image_id=imp.img.id, roi=roi._roi, record=False)
        finally:
            ctx.roi = saved
        news = _new_images(out)
        if not news:
            return None
        ip = news[-1].getProcessor().duplicate()
        news[-1].close()
        return ip


__all__ = ["Duplicator", "ZProjector", "ChannelSplitter", "ImageCalculator", "HyperStackConverter", "RGBStackMerge", "SubstackMaker", "Concatenator", "ContrastEnhancer",
           "PlugInFilter", "GaussianBlur", "RankFilters", "BackgroundSubtracter", "Convolver", "Filters", "EDM", "Binary", "ThresholdToSelection", "MaximumFinder",
           "ParticleAnalyzer", "ImageConverter", "StackConverter", "AutoThresholder", "Thresholder", "Colors", "Macro_Runner", "ImageInfo", "Scaler", "Resizer", "Straightener",
           "Measurements"]
