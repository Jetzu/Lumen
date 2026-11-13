"""The ImageJ 1.x API shim: ``import ij`` without a JVM.

Jython scripts written for Fiji import the ImageJ classes (``from ij import IJ, ImagePlus``,
``from ij.plugin.frame import RoiManager`` …).  This package makes those modules importable in Lumen's Python
and implements them over the macro compatibility layer (:mod:`lumen.core.ij_compat`) and the session, so the
scripts run unchanged in the Script panel (language *Python* or *Jython*), through ``POST /api/script/run`` and
in batch commands.  The same classes serve the browser's JavaScript editor through :mod:`.bridge`
(``importClass(Packages.ij.IJ)`` …).

Importing this package installs a ``sys.meta_path`` finder (appended, so real packages always win) for the
roots ``ij``, ``java``, ``jarray``, ``fiji`` and ``loci``; :func:`run_ij_script` is the script runner with the
``#@`` parameter header (:mod:`.params`).  Headless use::

    from lumen.core import ijshim
    with ijshim.bind(session):
        from ij import IJ
        IJ.run("Blobs (25K)"); imp = IJ.getImage(); print(imp.getStatistics().mean)
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Any, Callable

from ._context import Context, _LAST, _PARAMS, _SEED_ROI, _UNSET, bind, current, set_default
# the optional JVM runner (Groovy / BeanShell / Rhino JavaScript, opt-in JRuby / Clojure) – pure Python until a script runs; two modes: a java
# subprocess over Fiji's jars (.jvm_subprocess, no Python package) or the in-process JPype bridge (LUMEN_JVM_MODE, see .jvm_bridge)
from .jvm_bridge import LANGUAGES as JVM_LANGUAGES, effective_mode as effective_jvm_mode, is_jvm_language, jvm_mode, run_jvm_script

__all__ = ["install", "uninstall", "run_ij_script", "run_jvm_script", "is_jvm_language", "JVM_LANGUAGES", "jvm_mode", "effective_jvm_mode", "bind", "current",
           "set_default", "resolve_class", "class_index", "class_path", "MODULES", "Context"]


# ---------------------------------------------------------------------- the module table (lazy, one importer per module)
def _m_ij() -> dict[str, Any]:
    from ._ij import IJ, ImageJ, Macro, Prefs, WindowManager
    from ._images import CompositeImage, ImagePlus, ImageStack
    return {"IJ": IJ, "ImagePlus": ImagePlus, "ImageStack": ImageStack, "VirtualStack": ImageStack, "CompositeImage": CompositeImage, "WindowManager": WindowManager,
            "Prefs": Prefs, "Macro": Macro, "ImageJ": ImageJ}


def _m_ij_gui() -> dict[str, Any]:
    from ._dialogs import (GenericDialog, HTMLDialog, MessageDialog, NonBlockingGenericDialog, Plot, PlotWindow, Toolbar, WaitForUserDialog, YesNoCancelDialog)
    from ._images import _Canvas, _Window
    from ._rois import EllipseRoi, Line, OvalRoi, Overlay, PointRoi, PolygonRoi, Roi, ShapeRoi, TextRoi
    return {"Roi": Roi, "OvalRoi": OvalRoi, "EllipseRoi": EllipseRoi, "PolygonRoi": PolygonRoi, "Line": Line, "PointRoi": PointRoi, "TextRoi": TextRoi, "ShapeRoi": ShapeRoi,
            "Overlay": Overlay, "GenericDialog": GenericDialog, "NonBlockingGenericDialog": NonBlockingGenericDialog, "WaitForUserDialog": WaitForUserDialog,
            "YesNoCancelDialog": YesNoCancelDialog, "MessageDialog": MessageDialog, "HTMLDialog": HTMLDialog, "Plot": Plot, "PlotWindow": PlotWindow, "Toolbar": Toolbar,
            "ImageWindow": _Window, "StackWindow": _Window, "ImageCanvas": _Canvas}


def _m_ij_process() -> dict[str, Any]:
    from ._images import ByteProcessor, ColorProcessor, FloatProcessor, ImageProcessor, ImageStatistics, ShortProcessor, StackStatistics
    from ._plugins import AutoThresholder, ImageConverter, StackConverter
    from ._rois import FloatPolygon

    class Blitter:
        COPY, COPY_INVERTED, COPY_TRANSPARENT, ADD, SUBTRACT, MULTIPLY, DIVIDE, AVERAGE, DIFFERENCE, AND, OR, XOR, MIN, MAX, COPY_ZERO_TRANSPARENT = range(15)

    class LUT:
        def __init__(self, *args: Any) -> None:
            self.name = args[0] if args and isinstance(args[0], str) else ""

    return {"ImageProcessor": ImageProcessor, "ByteProcessor": ByteProcessor, "ShortProcessor": ShortProcessor, "FloatProcessor": FloatProcessor, "ColorProcessor": ColorProcessor,
            "ImageStatistics": ImageStatistics, "StackStatistics": StackStatistics, "ByteStatistics": ImageStatistics, "ShortStatistics": ImageStatistics,
            "FloatStatistics": ImageStatistics, "ColorStatistics": ImageStatistics, "ImageConverter": ImageConverter, "StackConverter": StackConverter,
            "AutoThresholder": AutoThresholder, "FloatPolygon": FloatPolygon, "Blitter": Blitter, "LUT": LUT}


def _m_ij_measure() -> dict[str, Any]:
    from ._images import Calibration
    from ._tables import Measurements, ResultsTable
    return {"ResultsTable": ResultsTable, "Measurements": Measurements, "Calibration": Calibration}


def _m_ij_plugin() -> dict[str, Any]:
    from ._plugins import (ChannelSplitter, Colors, Concatenator, ContrastEnhancer, Duplicator, HyperStackConverter, ImageCalculator, ImageInfo, Macro_Runner,
                           RGBStackMerge, Resizer, Scaler, Straightener, SubstackMaker, Thresholder, ZProjector)
    return {"Duplicator": Duplicator, "ZProjector": ZProjector, "ChannelSplitter": ChannelSplitter, "ImageCalculator": ImageCalculator,
            "HyperStackConverter": HyperStackConverter, "RGBStackMerge": RGBStackMerge, "SubstackMaker": SubstackMaker, "Concatenator": Concatenator,
            "ContrastEnhancer": ContrastEnhancer, "Thresholder": Thresholder, "Colors": Colors, "Macro_Runner": Macro_Runner, "ImageInfo": ImageInfo, "Scaler": Scaler,
            "Resizer": Resizer, "Straightener": Straightener}


def _m_ij_plugin_frame() -> dict[str, Any]:
    from ._tables import RoiManager

    class Recorder:
        record = False

        @staticmethod
        def recordString(*a: Any) -> None:
            pass

        @staticmethod
        def getInstance() -> None:
            return None

    return {"RoiManager": RoiManager, "Recorder": Recorder}


def _m_ij_plugin_filter() -> dict[str, Any]:
    from ._plugins import (EDM, BackgroundSubtracter, Binary, Convolver, Filters, GaussianBlur, MaximumFinder, ParticleAnalyzer, PlugInFilter, RankFilters,
                           ThresholdToSelection)
    from ._tables import Analyzer
    return {"ParticleAnalyzer": ParticleAnalyzer, "Analyzer": Analyzer, "GaussianBlur": GaussianBlur, "RankFilters": RankFilters, "BackgroundSubtracter": BackgroundSubtracter,
            "Convolver": Convolver, "ThresholdToSelection": ThresholdToSelection, "MaximumFinder": MaximumFinder, "EDM": EDM, "Binary": Binary, "Filters": Filters,
            "PlugInFilter": PlugInFilter}


def _m_ij_io() -> dict[str, Any]:
    from ._dialogs import DirectoryChooser, OpenDialog, SaveDialog
    from ._images import FileInfo
    from ._io import FileSaver, Opener
    return {"FileSaver": FileSaver, "Opener": Opener, "OpenDialog": OpenDialog, "SaveDialog": SaveDialog, "DirectoryChooser": DirectoryChooser, "FileInfo": FileInfo}


def _m_ij_text() -> dict[str, Any]:
    from ._dialogs import TextWindow, _TextPanel
    return {"TextWindow": TextWindow, "TextPanel": _TextPanel}


def _m_ij_util() -> dict[str, Any]:
    from ._java import _java_format

    class Tools:
        @staticmethod
        def parseDouble(s: Any, default: Any = float("nan")) -> float:
            try:
                return float(str(s).strip())
            except (TypeError, ValueError):
                return float(default)

        @staticmethod
        def split(s: Any, delim: Any = None) -> list[str]:
            return str(s).split() if delim is None else [p for p in str(s).replace(str(delim), " ").split()]

        @staticmethod
        def getMinMax(arr: Any) -> list[float]:
            vals = [float(v) for v in arr]
            return [min(vals), max(vals)] if vals else [0.0, 0.0]

        @staticmethod
        def toString(arr: Any) -> str:
            return ", ".join(str(v) for v in arr)

        @staticmethod
        def format(fmt: Any, *args: Any) -> str:  # noqa: A003
            return _java_format(fmt, args)

        @staticmethod
        def getDecimalPlaces(*a: Any) -> int:
            return 3

        @staticmethod
        def rank(arr: Any) -> list[int]:
            vals = [float(v) for v in arr]
            return sorted(range(len(vals)), key=lambda i: vals[i])

    return {"Tools": Tools}


def _m_java_lang() -> dict[str, Any]:
    from . import _java as j
    return {"System": j.System, "Math": j.Math, "Integer": j.Integer, "Long": j.Long, "Short": j.Short, "Byte": j.Byte, "Double": j.Double, "Float": j.Float,
            "Boolean": j.Boolean, "Character": j.Character, "String": j.String, "StringBuilder": j.StringBuilder, "StringBuffer": j.StringBuffer, "Thread": j.Thread,
            "Runtime": j.Runtime, "Object": j.Object, "Exception": j.JavaException, "RuntimeException": j.RuntimeException,
            "IllegalArgumentException": j.IllegalArgumentException, "NullPointerException": j.NullPointerException, "Throwable": j.JavaException,
            "InterruptedException": j.JavaException, "ClassCastException": j.RuntimeException, "ArithmeticException": j.RuntimeException,
            "IndexOutOfBoundsException": j.RuntimeException, "ArrayIndexOutOfBoundsException": j.RuntimeException, "NumberFormatException": j.IllegalArgumentException,
            "OutOfMemoryError": j.JavaException, "Error": j.JavaException, "Void": type(None), "Class": type}


def _m_java_io() -> dict[str, Any]:
    from . import _java as j
    return {"File": j.File, "FileWriter": j.FileWriter, "BufferedWriter": j.BufferedWriter, "PrintWriter": j.PrintWriter, "FileReader": j.FileReader,
            "BufferedReader": j.BufferedReader, "FileOutputStream": j.FileOutputStream, "FileInputStream": j.FileInputStream, "IOException": j.IOException,
            "FileNotFoundException": j.IOException, "PrintStream": type(j.System.out)}


def _m_java_util() -> dict[str, Any]:
    from . import _java as j
    return {"ArrayList": j.ArrayList, "Vector": j.Vector, "LinkedList": j.LinkedList, "List": j.ArrayList, "HashMap": j.HashMap, "Hashtable": j.Hashtable,
            "LinkedHashMap": j.LinkedHashMap, "TreeMap": j.TreeMap, "Map": j.HashMap, "HashSet": j.HashSet, "Set": j.HashSet, "Arrays": j.Arrays,
            "Collections": j.Collections, "Random": j.Random, "Date": j.Date, "UUID": j.UUID}


def _m_java_awt() -> dict[str, Any]:
    from . import _java as j
    return {"Color": j.Color, "Rectangle": j.Rectangle, "Point": j.Point, "Dimension": j.Dimension, "Polygon": j.Polygon, "Font": j.Font, "Toolkit": j.Toolkit}


def _m_java_awt_geom() -> dict[str, Any]:
    from . import _java as j

    class Point2D:
        class Double(j.Point):
            def __init__(self, x: Any = 0.0, y: Any = 0.0) -> None:
                self.x, self.y = float(x), float(y)  # type: ignore[assignment]

        Float = Double

    class Rectangle2D:
        class Double(j.Rectangle):
            def __init__(self, x: Any = 0.0, y: Any = 0.0, w: Any = 0.0, h: Any = 0.0) -> None:
                self.x, self.y, self.width, self.height = float(x), float(y), float(w), float(h)  # type: ignore[assignment]

        Float = Double

    return {"Point2D": Point2D, "Rectangle2D": Rectangle2D}


def _m_jarray() -> dict[str, Any]:
    from ._java import array, zeros
    return {"array": array, "zeros": zeros}


def _m_fiji_util_gui() -> dict[str, Any]:
    from ._dialogs import GenericDialogPlus
    return {"GenericDialogPlus": GenericDialogPlus}


def _m_loci_plugins() -> dict[str, Any]:
    from ._io import BF
    return {"BF": BF}


def _m_loci_plugins_in() -> dict[str, Any]:
    from ._io import ImporterOptions
    return {"ImporterOptions": ImporterOptions}


def _m_loci_formats() -> dict[str, Any]:
    from ._io import ImageReader
    return {"ImageReader": ImageReader, "ChannelSeparator": ImageReader}


def _empty() -> dict[str, Any]:
    return {}


MODULES: dict[str, Callable[[], dict[str, Any]]] = {
    "ij": _m_ij, "ij.gui": _m_ij_gui, "ij.process": _m_ij_process, "ij.measure": _m_ij_measure, "ij.plugin": _m_ij_plugin, "ij.plugin.frame": _m_ij_plugin_frame,
    "ij.plugin.filter": _m_ij_plugin_filter, "ij.plugin.tool": _empty, "ij.io": _m_ij_io, "ij.text": _m_ij_text, "ij.util": _m_ij_util, "ij.macro": _empty,
    "java": _empty, "java.lang": _m_java_lang, "java.io": _m_java_io, "java.util": _m_java_util, "java.awt": _m_java_awt, "java.awt.geom": _m_java_awt_geom,
    "java.awt.event": _empty, "java.awt.image": _empty, "jarray": _m_jarray,
    "fiji": _empty, "fiji.util": _empty, "fiji.util.gui": _m_fiji_util_gui,
    "loci": _empty, "loci.plugins": _m_loci_plugins, "loci.plugins.in_": _m_loci_plugins_in, "loci.formats": _m_loci_formats,
}
ROOTS = tuple(sorted({name.split(".")[0] for name in MODULES}))
# the packages ImageJ's JavaScript evaluator imports by default, in precedence order (first wins for a class name)
AUTO_IMPORT = ["ij", "ij.gui", "ij.process", "ij.measure", "ij.util", "ij.plugin", "ij.plugin.filter", "ij.plugin.frame", "ij.io", "ij.text",
               "java.lang", "java.awt", "java.awt.image", "java.awt.geom", "java.util", "java.io", "fiji.util.gui", "loci.plugins", "loci.plugins.in_", "loci.formats"]
_CACHE: dict[str, dict[str, Any]] = {}


def module_contents(name: str) -> dict[str, Any]:
    """The names a shim module exports (cached)."""
    if name not in _CACHE:
        _CACHE[name] = MODULES[name]()
    return _CACHE[name]


def _is_package(name: str) -> bool:
    prefix = name + "."
    return any(k.startswith(prefix) for k in MODULES)


class _ShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if fullname.partition(".")[0] not in ROOTS:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=_is_package(fullname), origin="lumen.core.ijshim")

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        name = module.__name__
        if name not in MODULES:
            parent = name.rpartition(".")[0]
            siblings = sorted(k for k in MODULES if k.rpartition(".")[0] == parent and k != parent)
            hint = f"; the shim provides {', '.join(siblings)}" if siblings else ""
            raise ImportError(f"Lumen's ImageJ shim has no module {name!r} (no JVM – only the ij.* / java.* subset listed in docs/SCRIPTING.md){hint}", name=name)
        contents = module_contents(name)
        module.__dict__.update(contents)
        module.__all__ = sorted(contents)
        module.__file__ = None
        if _is_package(name):
            module.__path__ = []


_FINDER = _ShimFinder()


def install() -> None:
    """Make ``ij``, ``java``, ``jarray``, ``fiji`` and ``loci`` importable (idempotent)."""
    if _FINDER not in sys.meta_path:
        sys.meta_path.append(_FINDER)


def uninstall() -> None:
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    for name in list(sys.modules):
        if name.partition(".")[0] in ROOTS and getattr(sys.modules[name], "__spec__", None) is not None and sys.modules[name].__spec__.origin == "lumen.core.ijshim":
            del sys.modules[name]


install()


# ---------------------------------------------------------------------- class lookup (JavaScript bridge, IJ.runPlugIn)
def resolve_class(path: str) -> Any:
    """``"ij.plugin.frame.RoiManager"`` → the class (or ``None``); nested classes (``AutoThresholder.Method``) and
    Python-qualified names (``lumen.core.ijshim._java.Color``) work too."""
    parts = path.split(".")
    for n in range(len(parts) - 1, 0, -1):
        mod = ".".join(parts[:n])
        if mod in MODULES:
            obj: Any = module_contents(mod)
            for attr in parts[n:]:
                obj = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
                if obj is None:
                    return None
            return obj
    if path.startswith("lumen."):
        for n in range(len(parts) - 1, 0, -1):
            try:
                m = importlib.import_module(".".join(parts[:n]))
            except ImportError:
                continue
            obj = m
            for attr in parts[n:]:
                obj = getattr(obj, attr, None)
                if obj is None:
                    return None
            return obj
    return None


_PATHS: dict[type, str] = {}


def class_path(cls: type) -> str:
    """The Java-style path of a shim class (``ij.ImagePlus``), else its Python qualified name."""
    if not _PATHS:
        for mod in AUTO_IMPORT + [m for m in MODULES if m not in AUTO_IMPORT]:
            for name, obj in module_contents(mod).items():
                if isinstance(obj, type) and obj not in _PATHS:
                    _PATHS[obj] = f"{mod}.{name}"
    p = _PATHS.get(cls)
    return p or f"{cls.__module__}.{cls.__qualname__}"


def class_index() -> dict[str, str]:
    """``{ClassName: "package.ClassName"}`` for the auto-imported packages (the JavaScript scope)."""
    out: dict[str, str] = {}
    for mod in AUTO_IMPORT:
        for name, obj in module_contents(mod).items():
            if name not in out and (isinstance(obj, type) or callable(obj)):
                out[name] = f"{mod}.{name}"
    return out


# ---------------------------------------------------------------------- the script runner
def run_ij_script(session: Any, code: str, params: dict[str, Any] | None = None, roi: Any = _UNSET) -> dict[str, Any]:
    """Run a Python / Jython script with the ``ij`` shim available, its ``#@`` parameters filled from ``params``
    and the viewport selection ``roi`` as the initial selection.  Returns ``run_script``'s ``{"stdout", "error"}``
    plus ``"parameters"`` (the header's descriptors) and, once the shim was used, ``"roi"`` (the final selection)."""
    from ..scripting import run_script
    from .params import prepare
    install()
    prepared = prepare(code, params)
    tok_p = _PARAMS.set(dict(params or {}))
    tok_r = _SEED_ROI.set(roi if isinstance(roi, dict) else _UNSET)
    tok_l = _LAST.set(None)
    try:
        out = run_script(session, prepared.code)
        ctx = _LAST.get()
    finally:
        _PARAMS.reset(tok_p)
        _SEED_ROI.reset(tok_r)
        _LAST.reset(tok_l)
    if prepared.line_offset and out.get("error"):
        out["error"] = str(out["error"]) + f"\n(line numbers are shifted by {prepared.line_offset}: the #@ parameter header was not at the top of the script)"
    if prepared.params:
        out["parameters"] = prepared.descriptors
    if ctx is not None:
        out["roi"] = ctx.api.roi
    return out
