"""``ij.IJ`` (static utility class), ``ij.WindowManager``, ``ij.Prefs`` and ``ij.Macro`` on top of the macro
layer's ``IJ`` object (:class:`lumen.core.ij_compat.IJ`): ``IJ.run(imp, "Gaussian Blur...", "sigma=2")`` is the
macro ``run()`` with ``imp`` made the current image for the call."""
from __future__ import annotations

import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..image import LumenImage
from ._context import Context, current
from ._images import ImagePlus, as_current, ensure_in_session, in_session
from ._java import Dimension, color_hex


def _imp_or_none(args: tuple[Any, ...]) -> tuple[ImagePlus | None, tuple[Any, ...]]:
    if args and isinstance(args[0], ImagePlus):
        return args[0], args[1:]
    if args and args[0] is None and len(args) > 1:
        return None, args[1:]
    return None, args


def _new_image(title: str, kind: str, w: int, h: int, c: int, z: int, t: int, bit_depth: int | None = None) -> LumenImage:
    """``IJ.createImage``: a script-private image filled black / white / ramp / random."""
    k = kind.lower()
    rgb = "rgb" in k or bit_depth == 24
    bits = bit_depth or (16 if "16" in k else 32 if "32" in k else 8)
    dtype = np.uint8 if rgb or bits == 8 else (np.uint16 if bits == 16 else np.float32)
    top = 255.0 if dtype == np.uint8 else (65535.0 if dtype == np.uint16 else 1.0)
    shape = (t, z, 3 if rgb else c, h, w)
    if "white" in k:
        data = np.full(shape, top, dtype)
    elif "ramp" in k:
        ramp = (np.arange(w, dtype=np.float64) / max(w, 1)) * (top if dtype != np.float32 else 1.0)
        data = np.broadcast_to(ramp.astype(dtype), shape).copy()
    elif "random" in k or "noise" in k:
        rng = np.random.default_rng()
        if dtype == np.float32:
            data = rng.normal(0.0, 1.0, shape).astype(np.float32)
        else:
            data = np.clip(rng.normal(top / 2, top / 8, shape), 0, top).astype(dtype)
    else:
        data = np.zeros(shape, dtype)
    img = LumenImage(data, name=title, axes="TZCYX", rgb=rgb)
    if not rgb and dtype == np.float32:
        for ch in img.display.channels:
            ch.min, ch.max = float(data.min()), float(max(data.max(), data.min() + 1e-6))
    return img


class IJ:
    """The ``ij.IJ`` statics scripts use most: ``run`` (with or without an ``ImagePlus``), ``open`` / ``openImage``,
    ``getImage``, ``createImage`` / ``newImage``, ``selectWindow``, ``log`` / ``showMessage`` / ``error``,
    ``saveAs`` / ``save``, thresholds, display range, selections, directories and the small utilities."""

    debugMode = False
    _props: dict[str, Any] = {}

    # ------------------------------------------------------------- commands
    @staticmethod
    def run(*args: Any) -> None:
        """``IJ.run(command)``, ``IJ.run(command, options)``, ``IJ.run(imp, command, options)``."""
        ctx = current()
        imp, rest = _imp_or_none(args)
        if not rest:
            raise TypeError("IJ.run(command[, options]) / IJ.run(imp, command, options)")
        command = str(rest[0])
        options = "" if len(rest) < 2 or rest[1] is None else str(rest[1])
        with as_current(ctx, imp):
            ctx.ij.run(command, options)

    @staticmethod
    def doCommand(command: Any, *rest: Any) -> None:
        IJ.run(command, *rest)

    @staticmethod
    def runPlugIn(*args: Any) -> Any:
        """``IJ.runPlugIn("ij.plugin.frame.RoiManager", "")`` returns the ROI Manager; other plugin classes are
        looked up in the shim's class table and instantiated (their ``run`` is not called)."""
        imp, rest = _imp_or_none(args)
        name = str(rest[0]) if rest else ""
        from . import resolve_class
        cls = resolve_class(name)
        if cls is None:
            raise ValueError(f"IJ.runPlugIn: unknown plugin class {name!r} (Lumen cannot load Java plugins)")
        if name.endswith("RoiManager"):
            return cls.getInstance()
        try:
            return cls()
        except TypeError:
            return cls

    @staticmethod
    def runMacro(code: Any, arg: Any = "") -> str:
        """Run ImageJ-1 macro code through the compatibility layer; returns what the macro printed."""
        return current().ij.macro(str(code), "" if arg is None else arg)

    @staticmethod
    def runMacroFile(path: Any, arg: Any = "") -> str:
        return current().ij.runMacro(os.fspath(path), "" if arg is None else arg)

    # ------------------------------------------------------------- images
    @staticmethod
    def getImage() -> ImagePlus:
        ctx = current()
        img = ctx.session.active
        if img is None:
            raise ValueError("There are no images open")
        return ImagePlus(_img=img)

    @staticmethod
    def open(*args: Any) -> None:  # noqa: A003
        """``IJ.open(path)``: open and show a file (every format Lumen reads)."""
        if not args or not args[0]:
            raise ValueError("IJ.open(path): a file path is required")
        current().ij.open(os.fspath(args[0]))

    @staticmethod
    def openImage(*args: Any) -> ImagePlus | None:
        """``IJ.openImage(path[, n])``: open a file without showing it (``imp.show()`` adds the tab)."""
        if not args or not args[0]:
            raise ValueError("IJ.openImage(path): a file path is required (there is no file dialog in a script)")
        from ..io import read_image
        path = os.fspath(args[0])
        try:
            img = read_image(path)
        except FileNotFoundError:
            return None
        if len(args) > 1 and args[1] is not None:  # a single stack slice
            n = int(args[1])
            from ._images import _decode_slice
            t, z, c = _decode_slice(img, n)
            plane = img.data[t:t + 1, z:z + 1, :, :, :] if img.rgb else img.data[t:t + 1, z:z + 1, c:c + 1]
            img = img.copy(data=np.ascontiguousarray(plane))
        return ImagePlus(_img=img)

    @staticmethod
    def openVirtual(path: Any) -> ImagePlus | None:
        return IJ.openImage(path)

    @staticmethod
    def createImage(*args: Any) -> ImagePlus:
        """``createImage(title, type, w, h, slices)``, ``createImage(title, type, w, h, c, z, t)``,
        ``createImage(title, w, h, slices, bitDepth)`` – a script-private image (``show()`` adds the tab)."""
        title = str(args[0])
        if isinstance(args[1], str):
            kind = args[1]
            w, h = int(args[2]), int(args[3])
            if len(args) >= 7:
                c, z, t = int(args[4]), int(args[5]), int(args[6])
            else:
                c, z, t = 1, int(args[4]) if len(args) > 4 else 1, 1
            return ImagePlus(_img=_new_image(title, kind, w, h, c, z, t))
        w, h, n, bits = int(args[1]), int(args[2]), int(args[3]), int(args[4])
        return ImagePlus(_img=_new_image(title, "black", w, h, 1, n, 1, bit_depth=bits))

    @staticmethod
    def createHyperStack(title: Any, w: Any, h: Any, c: Any, z: Any, t: Any, bit_depth: Any) -> ImagePlus:
        return ImagePlus(_img=_new_image(str(title), "black", int(w), int(h), int(c), int(z), int(t), bit_depth=int(bit_depth)))

    @staticmethod
    def newImage(title: Any, kind: Any, w: Any, h: Any, *dims: Any) -> ImagePlus:
        """``IJ.newImage(title, type, w, h, slices)`` creates *and shows* the image."""
        ctx = current()
        img = ctx.ij.newImage(str(title), str(kind), int(w), int(h), *dims)
        return ImagePlus(_img=img)

    @staticmethod
    def selectWindow(key: Any) -> None:
        ctx = current()
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            ctx.ij.selectImage(int(key))
        else:
            ctx.ij.selectWindow(str(key))

    @staticmethod
    def getImageInfo(imp: Any = None) -> str:
        ctx = current()
        with as_current(ctx, imp):
            return ctx.ij.getImageInfo()

    # ------------------------------------------------------------- output
    @staticmethod
    def log(message: Any = "") -> None:
        s = str(message)
        if s in ("\\Clear", "\\Close", "\\Update"):
            return
        current().emit(s)

    @staticmethod
    def write(message: Any = "") -> None:
        IJ.log(message)

    @staticmethod
    def getLog() -> str | None:
        ctx = current()
        lines = [e.get("msg", "") if isinstance(e, dict) else str(e) for e in ctx.session.log]
        return "\n".join(lines) if lines else None

    @staticmethod
    def showMessage(*args: Any) -> None:
        ctx = current()
        if len(args) >= 2:
            ctx.emit(f"{args[0]}: {args[1]}")
        elif args:
            ctx.emit(str(args[0]))

    @staticmethod
    def showMessageWithCancel(*args: Any) -> bool:
        IJ.showMessage(*args)
        return True

    @staticmethod
    def error(*args: Any) -> None:
        ctx = current()
        msg = f"{args[0]}: {args[1]}" if len(args) >= 2 else (str(args[0]) if args else "Error")
        if ctx._emit is not None:
            ctx._emit("Error: " + msg)
        else:
            print("Error: " + msg, file=sys.stderr)
        ctx.session.add_log(msg, level="error")

    @staticmethod
    def handleException(e: Any) -> None:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)) if isinstance(e, BaseException) else str(e)
        current().emit(tb.rstrip())

    @staticmethod
    def showStatus(message: Any = "") -> None:
        pass

    @staticmethod
    def showProgress(*args: Any) -> None:
        pass

    @staticmethod
    def showTime(*args: Any) -> None:
        pass

    @staticmethod
    def beep() -> None:
        pass

    @staticmethod
    def wait(ms: Any) -> None:
        pass

    @staticmethod
    def redirectErrorMessages(*args: Any) -> None:
        pass

    @staticmethod
    def setDebugMode(b: Any) -> None:
        IJ.debugMode = bool(b)

    @staticmethod
    def getTextPanel() -> None:
        return None

    @staticmethod
    def setColumnHeadings(headings: Any) -> None:
        current().emit(str(headings))

    @staticmethod
    def getInstance() -> None:
        return None  # there is no ImageJ frame

    @staticmethod
    def getApplet() -> None:
        return None

    # ------------------------------------------------------------- user input (answered from the script parameters)
    @staticmethod
    def getNumber(prompt: Any, default: Any) -> float:
        v = current().param(prompt, default=default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def getString(prompt: Any, default: Any) -> str:
        v = current().param(prompt, default=default)
        return str(v) if v is not None else str(default)

    @staticmethod
    def getBoolean(prompt: Any, default: Any = True) -> bool:
        v = current().param(prompt, default=default)
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes", "y")

    @staticmethod
    def getDirectory(what: Any = "") -> str | None:
        """Known keys (``home``, ``imagej``, ``startup``, ``plugins``, ``macros``, ``luts``, ``temp``, ``image``,
        ``current``) are answered like the macro function; any other title is a dialog, answered from the script
        parameters (``directory`` / the title) or ``None`` – no dialog can open from a script."""
        ctx = current()
        w = str(what)
        wl = w.lower()
        if wl in ("home",):
            return str(Path.home()) + os.sep
        if wl in ("imagej", "startup", "plugins", "macros", "luts"):
            from ..ops import lumen_home
            base = lumen_home()
            sub = {"plugins": "plugins", "macros": "plugins", "luts": "luts"}.get(wl)
            return str(base / sub) + os.sep if sub else str(base) + os.sep
        if wl == "temp":
            return tempfile.gettempdir() + os.sep
        if wl in ("image", "current", "cwd", "file", ""):
            return ctx.ij.getDirectory(w)
        v = ctx.param(w, "directory", "dir", "folder", default=None)
        if v is None:
            return None
        p = os.fspath(v)
        return p if p.endswith(os.sep) else p + os.sep

    getDir = getDirectory

    @staticmethod
    def getFilePath(title: Any = "") -> str | None:
        v = current().param(title, "file", "path", default=None)
        return os.fspath(v) if v is not None else None

    # ------------------------------------------------------------- saving
    @staticmethod
    def saveAs(*args: Any) -> None:
        """``IJ.saveAs(imp, format, path)`` / ``IJ.saveAs(format, path)``; formats as in the macro ``saveAs``."""
        ctx = current()
        imp, rest = _imp_or_none(args)
        if len(rest) < 2:
            raise TypeError("IJ.saveAs([imp,] format, path)")
        with as_current(ctx, imp):
            ctx.ij.saveAs(str(rest[0]), os.fspath(rest[1]))

    @staticmethod
    def save(*args: Any) -> None:
        ctx = current()
        imp, rest = _imp_or_none(args)
        if not rest:
            raise TypeError("IJ.save([imp,] path)")
        with as_current(ctx, imp):
            ctx.ij.save(os.fspath(rest[0]))

    @staticmethod
    def saveAsTiff(imp: Any, path: Any) -> bool:
        IJ.saveAs(imp, "Tiff", path)
        return True

    @staticmethod
    def saveString(text: Any, path: Any) -> str | None:
        try:
            with open(os.fspath(path), "w", encoding="utf-8") as fh:
                fh.write(str(text))
            return None
        except OSError as e:
            return str(e)

    @staticmethod
    def append(text: Any, path: Any) -> str | None:
        try:
            with open(os.fspath(path), "a", encoding="utf-8") as fh:
                fh.write(str(text) + "\n")
            return None
        except OSError as e:
            return str(e)

    @staticmethod
    def openAsString(path: Any) -> str:
        try:
            with open(os.fspath(path), encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError as e:
            return f"Error: {e}"

    @staticmethod
    def openUrlAsString(url: Any) -> str:
        return "Error: no network access from scripts"

    # ------------------------------------------------------------- thresholds / display range
    @staticmethod
    def setThreshold(*args: Any) -> None:
        ctx = current()
        imp, rest = _imp_or_none(args)
        with as_current(ctx, imp):
            ctx.ij.setThreshold(float(rest[0]), float(rest[1]))

    setRawThreshold = setThreshold

    @staticmethod
    def setAutoThreshold(*args: Any) -> None:
        ctx = current()
        imp, rest = _imp_or_none(args)
        with as_current(ctx, imp):
            ctx.ij.setAutoThreshold(str(rest[0]) if rest else "Default")

    @staticmethod
    def resetThreshold(imp: Any = None) -> None:
        ctx = current()
        with as_current(ctx, imp):
            ctx.ij.resetThreshold()

    @staticmethod
    def setMinAndMax(*args: Any) -> None:
        ctx = current()
        imp, rest = _imp_or_none(args)
        with as_current(ctx, imp):
            ctx.ij.setMinAndMax(float(rest[0]), float(rest[1]), rest[2] if len(rest) > 2 else None)

    @staticmethod
    def resetMinAndMax(imp: Any = None) -> None:
        ctx = current()
        with as_current(ctx, imp):
            ctx.ij.resetMinAndMax()

    # ------------------------------------------------------------- selections
    @staticmethod
    def makeRectangle(x: Any, y: Any, w: Any, h: Any, arc: Any = 0) -> None:
        current().ij.makeRectangle(x, y, w, h, arc)

    @staticmethod
    def makeOval(x: Any, y: Any, w: Any, h: Any) -> None:
        current().ij.makeOval(x, y, w, h)

    @staticmethod
    def makeLine(*coords: Any) -> None:
        current().ij.makeLine(*coords)

    @staticmethod
    def makePoint(x: Any, y: Any, options: Any = "") -> None:
        current().ij.makePoint(x, y, options)

    @staticmethod
    def makePolygon(*coords: Any) -> None:
        current().ij.makePolygon(*coords)

    @staticmethod
    def makeText(text: Any, x: Any, y: Any) -> None:
        current().ij.makeText(text, x, y)

    @staticmethod
    def doWand(*args: Any) -> int:
        """``IJ.doWand(x, y[, tolerance, mode])`` / ``IJ.doWand(imp, x, y, tolerance, mode)``."""
        ctx = current()
        imp, rest = _imp_or_none(args)
        x, y = int(rest[0]), int(rest[1])
        tol = float(rest[2]) if len(rest) > 2 else 0.0
        mode = str(rest[3]) if len(rest) > 3 else ""
        with as_current(ctx, imp):
            out = ctx.ij._run("selection.wand", {"x": x, "y": y, "tolerance": tol, "connectivity": "4" if "4-connected" in mode else "8"})
        return 1 if out.get("roi") is not None else 0

    @staticmethod
    def setForegroundColor(r: Any, g: Any = None, b: Any = None) -> None:
        ctx = current()
        if g is None:
            ctx.ij.setForegroundColor(color_hex(r))
        else:
            ctx.ij.setForegroundColor(int(r), int(g), int(b))

    @staticmethod
    def setBackgroundColor(r: Any, g: Any = None, b: Any = None) -> None:
        ctx = current()
        if g is None:
            ctx.ij.setBackgroundColor(color_hex(r))
        else:
            ctx.ij.setBackgroundColor(int(r), int(g), int(b))

    @staticmethod
    def setPasteMode(mode: Any) -> None:
        current().ij.setPasteMode(mode)

    @staticmethod
    def setTool(name: Any) -> bool:
        current().ij.setTool(name)
        return True

    @staticmethod
    def getToolName() -> str:
        return current().ij.getToolName()

    @staticmethod
    def setKeyDown(key: Any) -> None:
        current().ij.setKeyDown(key)

    @staticmethod
    def setKeyUp(key: Any) -> None:
        pass

    @staticmethod
    def isKeyDown(key: Any) -> bool:
        return bool(current().ij.isKeyDown(key))

    @staticmethod
    def shiftKeyDown() -> bool:
        return False

    @staticmethod
    def altKeyDown() -> bool:
        return False

    @staticmethod
    def controlKeyDown() -> bool:
        return False

    @staticmethod
    def spaceBarDown() -> bool:
        return False

    @staticmethod
    def escapePressed() -> bool:
        return False

    @staticmethod
    def resetEscape() -> None:
        pass

    # ------------------------------------------------------------- results
    @staticmethod
    def isResultsWindow() -> bool:
        return bool(current().session.results.rows)

    @staticmethod
    def renameResults(*args: Any) -> None:
        current().ij.renameResults(*args)

    @staticmethod
    def deleteRows(first: Any, last: Any) -> None:
        current().ij.deleteRows(first, last)

    # ------------------------------------------------------------- environment
    @staticmethod
    def getVersion() -> str:
        return "1.54"

    @staticmethod
    def getFullVersion() -> str:
        return "1.54l"

    @staticmethod
    def freeMemory() -> str:
        return current().ij.freeMemory()

    @staticmethod
    def currentMemory() -> int:
        return int(current().ij.currentMemory())

    @staticmethod
    def maxMemory() -> int:
        return int(current().ij.maxMemory())

    @staticmethod
    def getScreenSize() -> Dimension:
        return Dimension(1920, 1080)

    @staticmethod
    def isMacro() -> bool:
        return False

    @staticmethod
    def isMacOSX() -> bool:
        return platform.system() == "Darwin"

    @staticmethod
    def isMacintosh() -> bool:
        return platform.system() == "Darwin"

    @staticmethod
    def isWindows() -> bool:
        return platform.system() == "Windows"

    @staticmethod
    def isLinux() -> bool:
        return platform.system() == "Linux"

    @staticmethod
    def is64Bit() -> bool:
        return sys.maxsize > 2 ** 32

    @staticmethod
    def isJava18() -> bool:
        return True

    @staticmethod
    def isJava9() -> bool:
        return False

    @staticmethod
    def javaVersion() -> int:
        return 8

    @staticmethod
    def getClassLoader() -> None:
        return None

    @staticmethod
    def getLuts() -> list[str]:
        from ..luts import LUT_NAMES
        return list(LUT_NAMES)

    @staticmethod
    def setProperty(key: Any, value: Any) -> None:
        IJ._props[str(key)] = value

    @staticmethod
    def getProperty(key: Any) -> Any:
        return IJ._props.get(str(key))

    @staticmethod
    def setOption(name: Any, value: Any = True) -> None:
        current().ij.setOption(name, value)

    @staticmethod
    def openUrl(url: Any) -> None:
        current().emit(f"openUrl: {url} (not opened – scripts have no browser)")

    @staticmethod
    def getClipboard() -> None:
        return None

    @staticmethod
    def exit(*a: Any) -> None:  # noqa: A003
        raise SystemExit(0)

    # ------------------------------------------------------------- formatting
    @staticmethod
    def d2s(n: Any, decimals: Any = 2, max_digits: Any = None) -> str:
        from ..ij_compat import ij_d2s, ij_d2s_sig
        if max_digits is not None:
            return ij_d2s_sig(float(n), int(decimals), int(max_digits))
        return ij_d2s(float(n), int(decimals))

    @staticmethod
    def pad(n: Any, length: Any) -> str:
        return current().ij.pad(n, length)


class WindowManager:
    """``ij.WindowManager``: the open images (tabs) by title or id."""

    @staticmethod
    def getImageTitles() -> list[str]:
        return [im.name for im in current().session.images.values()]

    @staticmethod
    def getIDList() -> list[int] | None:
        ctx = current()
        ids = []
        for im in ctx.session.images.values():
            ids.append(ImagePlus(_img=im).getID())
        return ids or None

    @staticmethod
    def getImageCount() -> int:
        return len(current().session.images)

    getWindowCount = getImageCount

    @staticmethod
    def getImage(key: Any) -> ImagePlus | None:
        ctx = current()
        try:
            return ImagePlus(_img=ctx.ij._image(key))
        except ValueError:
            return None

    @staticmethod
    def getCurrentImage() -> ImagePlus | None:
        img = current().session.active
        return ImagePlus(_img=img) if img is not None else None

    getTempCurrentImage = getCurrentImage

    @staticmethod
    def setTempCurrentImage(imp: Any) -> None:
        if imp is None:
            return
        ctx = current()
        ensure_in_session(ctx, imp._img, activate=True)

    @staticmethod
    def setCurrentWindow(win: Any) -> None:
        imp = win.getImagePlus() if hasattr(win, "getImagePlus") else win
        if isinstance(imp, ImagePlus):
            WindowManager.setTempCurrentImage(imp)

    @staticmethod
    def getCurrentWindow() -> Any:
        imp = WindowManager.getCurrentImage()
        return imp.getWindow() if imp is not None else None

    @staticmethod
    def getActiveWindow() -> Any:
        return WindowManager.getCurrentWindow()

    @staticmethod
    def getFrontWindow() -> Any:
        return WindowManager.getCurrentWindow()

    @staticmethod
    def getWindow(title: Any) -> Any:
        imp = WindowManager.getImage(title)
        return imp.getWindow() if imp is not None else None

    getFrame = getWindow

    @staticmethod
    def getNthImageID(n: Any) -> int:
        ids = WindowManager.getIDList() or []
        i = int(n) - 1
        return ids[i] if 0 <= i < len(ids) else 0

    @staticmethod
    def getNonImageTitles() -> list[str]:
        ctx = current()
        titles = []
        if ctx.session.results.rows:
            titles.append("Results")
        if ctx.session.log:
            titles.append("Log")
        titles.extend(ctx.session.tables)
        return titles

    @staticmethod
    def getNonImageWindows() -> list[Any]:
        return []

    @staticmethod
    def getUniqueName(name: Any) -> str:
        return current().session.unique_name(str(name))

    @staticmethod
    def makeUniqueName(name: Any) -> str:
        return WindowManager.getUniqueName(name)

    @staticmethod
    def closeAllWindows() -> bool:
        ctx = current()
        for im in list(ctx.session.images.values()):
            ctx.session.close_image(im.id)
        return True

    @staticmethod
    def putBehind() -> None:
        pass

    @staticmethod
    def repaintImageWindows() -> None:
        current().session._changed("images")

    @staticmethod
    def toFront(win: Any) -> None:
        WindowManager.setCurrentWindow(win)

    @staticmethod
    def setWindow(win: Any) -> None:
        pass

    @staticmethod
    def addWindow(win: Any) -> None:
        pass

    @staticmethod
    def removeWindow(win: Any) -> None:
        pass


class Prefs:
    """``ij.Prefs``: get / set preferences (kept with the session; ``blackBackground`` is honoured by the binary
    commands through ``Process ▸ Binary ▸ Options``)."""

    blackBackground = False
    useJFileChooser = False
    padEdges = False
    weightedColor = False
    interpolateScaledImages = True
    pointAutoMeasure = False
    noPointLabels = False
    disablePopupMenu = False
    requireControlKey = False
    separator = os.sep

    @staticmethod
    def _store() -> dict[str, Any]:
        return current().prefs

    @staticmethod
    def get(key: Any, default: Any = None) -> Any:
        return Prefs._store().get(str(key), default)

    @staticmethod
    def getString(key: Any, default: Any = None) -> Any:
        v = Prefs._store().get(str(key), default)
        return None if v is None else str(v)

    @staticmethod
    def getInt(key: Any, default: Any = 0) -> int:
        try:
            return int(float(Prefs._store().get(str(key), default)))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def getDouble(key: Any, default: Any = 0.0) -> float:
        try:
            return float(Prefs._store().get(str(key), default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def getBoolean(key: Any, default: Any = False) -> bool:
        v = Prefs._store().get(str(key), default)
        return v if isinstance(v, bool) else str(v).lower() == "true"

    @staticmethod
    def set(key: Any, value: Any) -> None:  # noqa: A003
        Prefs._store()[str(key)] = value

    @staticmethod
    def savePreferences() -> None:
        pass

    @staticmethod
    def getPrefsDir() -> str:
        from ..ops import lumen_home
        return str(lumen_home())

    @staticmethod
    def getImageJDir() -> str:
        from ..ops import lumen_home
        return str(lumen_home()) + os.sep

    @staticmethod
    def getHomeDir() -> str:
        return str(Path.home())

    @staticmethod
    def getFileSeparator() -> str:
        return os.sep

    @staticmethod
    def getDefaultDirectory() -> str:
        return os.getcwd() + os.sep

    @staticmethod
    def setDefaultDirectory(d: Any) -> None:
        pass

    @staticmethod
    def getThreads() -> int:
        return os.cpu_count() or 1

    @staticmethod
    def setThreads(n: Any) -> None:
        pass


class Macro:
    """``ij.Macro``: ``getOptions()`` is the options string a plugin was called with – ``None`` in a script."""

    MACRO_CANCELED = "Macro canceled"
    _options: str | None = None

    @staticmethod
    def getOptions() -> str | None:
        return Macro._options

    @staticmethod
    def setOptions(s: Any) -> None:
        Macro._options = None if s is None else str(s)

    @staticmethod
    def getValue(options: Any, key: Any, default: Any = None) -> Any:
        from ..ij_compat import parse_options
        o = parse_options(str(options or ""))
        v = o.get(str(key))
        return default if v is None else (v if v is not True else default)

    @staticmethod
    def abort() -> None:
        raise SystemExit("Macro canceled")

    @staticmethod
    def trimKey(key: Any) -> str:
        return str(key).split(" ")[0].lower()

    @staticmethod
    def eval(code: Any) -> str:  # noqa: A003
        return IJ.runMacro(code)


class ImageJ:
    """``ij.ImageJ``: the application frame – only the version constants exist here."""

    VERSION = "1.54"
    BUILD = "l"
    NO_SHOW = 1
    EMBEDDED = 2

    @staticmethod
    def main(*a: Any) -> None:
        pass


__all__ = ["IJ", "WindowManager", "Prefs", "Macro", "ImageJ", "Context", "in_session"]
