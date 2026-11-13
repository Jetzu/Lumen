"""Optional JVM script runner: Fiji's JVM-only script languages – Groovy, BeanShell, Rhino JavaScript and, opt-in,
JRuby and Clojure – through the Java bridge (scyjava + JPype) with the *real* ImageJ 1.x (``ij.jar``).

Two modes, selected by ``LUMEN_JVM_MODE=auto|subprocess|inprocess`` (default ``auto``; :func:`effective_mode`):

* **subprocess** – :mod:`.jvm_subprocess`: one ``java`` process per script over a Fiji installation's jars (or
  ``ij.jar`` + the language jars on ``LUMEN_JVM_CLASSPATH``) and a JDK found by :mod:`lumen.core.readers.fiji_jars`
  (``JAVA_HOME``, ``PATH``, Homebrew, Temurin paths); **no Python package needed**, ImageJ's own batch mode, the
  ROI Manager included.  ``auto`` picks it on macOS – an in-process JVM cannot run ImageJ commands there, AWT wants
  the process's main thread – and wherever ``jpype`` is not installed.
* **inprocess** – this module's JPype / scyjava JVM, described below (Linux / Windows with the ``[jvm]`` extra).

Install the extra and a JDK::

    pip install "lumen-imaging[jvm]"          # jpype1 + scyjava
    # plus a JDK 11+ (Eclipse Temurin / OpenJDK) found through JAVA_HOME, ``java`` on PATH or a Homebrew
    # ``openjdk`` keg (/opt/homebrew/opt/openjdk*/libexec/openjdk.jdk/Contents/Home)

The first run resolves the jars through Maven into ``~/.m2`` / ``~/.jgo`` (one-time, internet needed):
``net.imagej:ij:1.54p``, ``org.apache.groovy:groovy:4.0.x``, ``org.beanshell:bsh:2.1.1``, ``org.mozilla:rhino:1.7.x``.
An existing ``ij.jar`` (Fiji's ``jars/ij-1.54p.jar``) is used instead when ``LUMEN_IJ_JAR`` points to it;
``LUMEN_JVM_CLASSPATH`` adds more jars / folders (``os.pathsep``-separated, ``*`` globs allowed, a folder means
its ``*.jar``) – a language whose jar is already on that classpath is not fetched from Maven.
``LUMEN_JVM_LANGUAGES=jruby,clojure`` enables the optional languages (``org.jruby:jruby-complete``,
``org.clojure:clojure``), ``LUMEN_JVM_EXTRA_ENDPOINTS`` adds Maven coordinates, ``LUMEN_JVM_HEAP=4g`` sets the
heap and ``LUMEN_JVM_HEADLESS=0`` starts a non-headless AWT (Linux / Windows; macOS AWT needs the main thread).

One JVM per process: whichever bridge starts it first (this one or the Bio-Formats reader bridge) fixes the
classpath – the other one then finds its jars missing and says so.

The JVM runs headless and ImageJ in macro *batch mode* (``ij.macro.Interpreter.batchMode``), so ``imp.show()``
registers images with the ``WindowManager`` without opening windows.  Exchanged with the session, per run:

* the active image, bound as ``imp`` (converted back *in place*, with an undo entry, when the script changed its
  pixels, calibration or title), and every image the script created and ``show()``-ed – as new session images;
* the viewport selection (``imp.getRoi()`` in, the script's final selection out), the ``#@`` parameters;
* the script's output and ``IJ.log`` lines (returned as ``stdout`` and copied into the Log section);
* the Results table (``ResultsTable.getResultsTable()``, seeded from the session, copied back when changed);
* the ROI Manager – best effort: on a headless JVM ``RoiManager`` (a ``java.awt.Frame``) cannot be constructed,
  so that exchange only works with ``LUMEN_JVM_HEADLESS=0``.

Everything degrades gracefully: :func:`available` is ``False`` with the reason in :func:`status`, nothing
Java-related is imported until a script runs, and the Script panel / ``POST /api/script/run`` report the install
steps instead of failing.
"""
from __future__ import annotations

import glob
import importlib.util
import math
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIP_EXTRA = 'pip install "lumen-imaging[jvm]"'
JDK_HINT = "install a JDK 11+ (Eclipse Temurin / OpenJDK: set JAVA_HOME, put java on PATH or `brew install openjdk`)"
INSTALL_HINT = (f"{PIP_EXTRA} and {JDK_HINT}; the first run downloads ij.jar and the script-language jars through Maven "
                f"(or point LUMEN_IJ_JAR / LUMEN_JVM_CLASSPATH at local jars, e.g. Fiji's jars folder)")
PACKAGES = ("jpype", "scyjava")
IJ_ENDPOINT = "net.imagej:ij:1.54p"
IJ_JAR_RE = r"ij-\d"
HOMEBREW_GLOBS = ("/opt/homebrew/opt/openjdk*/libexec/openjdk.jdk/Contents/Home", "/usr/local/opt/openjdk*/libexec/openjdk.jdk/Contents/Home",
                  "/home/linuxbrew/.linuxbrew/opt/openjdk*/libexec")

#: The script languages of the bridge: Maven endpoint, the jar-name pattern that marks it present on a local
#: classpath, the Fiji file extension and whether it is opt-in (``LUMEN_JVM_LANGUAGES``).
LANGUAGES: dict[str, dict[str, Any]] = {
    "groovy": {"label": "Groovy", "endpoint": "org.apache.groovy:groovy:4.0.23", "jar": r"groovy-\d", "ext": ".groovy", "optional": False},
    "beanshell": {"label": "BeanShell", "endpoint": "org.beanshell:bsh:2.1.1", "jar": r"bsh-\d", "ext": ".bsh", "optional": False},
    "rhino": {"label": "JavaScript (Rhino)", "endpoint": "org.mozilla:rhino:1.7.14", "jar": r"rhino-\d", "ext": ".js", "optional": False},
    "jruby": {"label": "JRuby", "endpoint": "org.jruby:jruby-complete:9.4.12.0", "jar": r"jruby-(complete|core)-\d", "ext": ".rb", "optional": True},
    "clojure": {"label": "Clojure", "endpoint": "org.clojure:clojure:1.12.0", "jar": r"clojure-\d", "ext": ".clj", "optional": True},
}
ALIASES = {"bsh": "beanshell", "javascript-jvm": "rhino", "js-jvm": "rhino", "rhino-js": "rhino", "ruby": "jruby", "clj": "clojure", "gvy": "groovy"}
# the packages ImageJ's own script evaluators import for BeanShell / JavaScript scripts
_IJ_PACKAGES = ("ij", "ij.gui", "ij.process", "ij.measure", "ij.util", "ij.plugin", "ij.plugin.filter", "ij.plugin.frame", "ij.io", "ij.text",
                "java.lang", "java.awt", "java.awt.image", "java.awt.geom", "java.util", "java.io")
BSH_PRELUDE = "".join(f"import {p}.*;" for p in _IJ_PACKAGES if p != "java.lang")
RHINO_PRELUDE = "".join(f"importPackage(Packages.{p});" for p in _IJ_PACKAGES)

_status_cache: dict[str, Any] | None = None


class _State:
    """The per-process JVM: started once, its classes cached, one script at a time (ImageJ's statics)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.started = False
        self.error: str | None = None
        self.languages: tuple[str, ...] = ()
        self.headless = True
        self.classes: dict[str, Any] = {}
        self.rm_error: str | None = None


_STATE = _State()


# ---------------------------------------------------------------------------- availability
def _importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _java_exe(home: str | os.PathLike) -> Path:
    return Path(home) / "bin" / ("java.exe" if os.name == "nt" else "java")


def _java_available() -> tuple[bool, str, str | None]:
    """``(found, description, java_home)`` – JAVA_HOME, ``java`` on PATH, or a Homebrew ``openjdk`` keg."""
    home = os.environ.get("JAVA_HOME", "").strip()
    if home and _java_exe(home).exists():
        return True, f"JAVA_HOME={home}", home
    exe = shutil.which("java")
    if exe:
        return True, f"java at {exe}", None
    for pattern in HOMEBREW_GLOBS:
        for cand in sorted(glob.glob(pattern), reverse=True):
            if _java_exe(cand).exists():
                return True, f"Homebrew OpenJDK at {cand}", cand
    return False, "no JDK found (JAVA_HOME unset, no java on PATH, no Homebrew openjdk)", None


def classpath_entries() -> list[str]:
    """Local jars from ``LUMEN_IJ_JAR`` and ``LUMEN_JVM_CLASSPATH`` (globs expanded, a folder → its ``*.jar``)."""
    out: list[str] = []
    ij = os.environ.get("LUMEN_IJ_JAR", "").strip()
    if ij:
        out.append(os.path.expanduser(ij))
    raw = os.environ.get("LUMEN_JVM_CLASSPATH", "").strip()
    for part in (raw.split(os.pathsep) if raw else []):
        part = os.path.expanduser(part.strip())
        if not part:
            continue
        if any(ch in part for ch in "*?["):
            out.extend(sorted(glob.glob(part)))
        elif os.path.isdir(part):
            out.extend(sorted(glob.glob(os.path.join(part, "*.jar"))))
        else:
            out.append(part)
    return out


def _jar_present(jar_names: list[str], pattern: str) -> bool:
    rx = re.compile(pattern)
    return any(rx.match(n) for n in jar_names)


def enabled_languages() -> list[str]:
    """The core languages plus the optional ones named in ``LUMEN_JVM_LANGUAGES`` (or whose jar is on the local classpath)."""
    extra = {s.strip().lower() for s in os.environ.get("LUMEN_JVM_LANGUAGES", "").replace(";", ",").split(",") if s.strip()}
    jars = [Path(p).name.lower() for p in classpath_entries()]
    return [lang for lang, info in LANGUAGES.items() if not info["optional"] or lang in extra or "all" in extra or _jar_present(jars, info["jar"])]


def headless_requested() -> bool:
    """``LUMEN_JVM_HEADLESS`` (default on): a headless AWT – no windows, but no ROI Manager either."""
    return os.environ.get("LUMEN_JVM_HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off")


def normalize_language(language: Any) -> str | None:
    """``"Groovy"`` / ``"bsh"`` / ``"js-jvm"`` → the language id, ``None`` for anything that is not a JVM language."""
    s = str(language or "").strip().lower().replace(" ", "")
    s = ALIASES.get(s, s)
    return s if s in LANGUAGES else None


def is_jvm_language(language: Any) -> bool:
    return normalize_language(language) is not None


MODES = ("auto", "subprocess", "inprocess")
_MODE_ALIASES = {"sub": "subprocess", "process": "subprocess", "external": "subprocess", "fiji": "subprocess", "fiji-jars": "subprocess", "jars": "subprocess",
                 "in": "inprocess", "in-process": "inprocess", "jpype": "inprocess", "scyjava": "inprocess", "": "auto", "default": "auto"}


def jvm_mode() -> str:
    """``LUMEN_JVM_MODE`` normalised: ``auto`` (default), ``subprocess`` or ``inprocess``."""
    s = os.environ.get("LUMEN_JVM_MODE", "").strip().lower()
    s = _MODE_ALIASES.get(s, s)
    return s if s in MODES else "auto"


def inprocess_status() -> dict[str, Any]:
    """The in-process (JPype / scyjava) mode on its own: ``{"available", "package", "java", "detail"}``."""
    package = all(_importable(n) for n in PACKAGES)
    java, java_detail, _home = _java_available() if package else (False, "not checked", None)
    if package and java:
        detail = f"jpype1 + scyjava installed, {java_detail}"
        if _STATE.started:
            detail += f"; JVM running ({', '.join(_STATE.languages)}{', headless' if _STATE.headless else ''})"
        elif _STATE.error:
            detail += f"; the JVM failed to start: {_STATE.error}"
    elif not package:
        missing = [n for n in PACKAGES if not _importable(n)]
        detail = f"{', '.join('jpype1' if n == 'jpype' else n for n in missing)} not installed – {PIP_EXTRA}"
    else:
        detail = f"{java_detail} – {JDK_HINT}"
    return {"available": bool(package and java), "package": package, "java": java, "detail": detail}


def effective_mode() -> str:
    """The mode a script runs in: the configured one, or for ``auto`` – ``subprocess`` on macOS and whenever ``jpype``
    is not installed (and when the in-process mode lacks a JDK the subprocess mode has), else ``inprocess``."""
    mode = jvm_mode()
    if mode != "auto":
        return mode
    if sys.platform == "darwin" or not _importable("jpype"):
        return "subprocess"
    inp = inprocess_status()
    if not inp["available"]:
        from . import jvm_subprocess
        if jvm_subprocess.available():
            return "subprocess"
    return "inprocess"


def status(refresh: bool = False) -> dict[str, Any]:
    """``{"available", "package", "java", "detail", "install", "languages", "started", "ij", "classpath", "mode", "effective_mode",
    "inprocess"[, "subprocess"]}`` – cached; the top-level keys describe the *effective* mode (``subprocess`` details are
    computed when that mode is in effect or ``refresh`` is set – they probe ``java -version`` once)."""
    global _status_cache
    if _status_cache is not None and not refresh:
        return dict(_status_cache)
    inp = inprocess_status()
    mode, eff = jvm_mode(), effective_mode()
    sub: dict[str, Any] | None = None
    if eff == "subprocess" or refresh:
        from . import jvm_subprocess
        sub = jvm_subprocess.status(refresh)
    if eff == "subprocess":
        assert sub is not None
        langs = [lang for lang, ls in sub["languages"].items() if ls["available"]]
        available = bool(sub["available"])
        detail = "subprocess mode (java + Fiji jars): " + sub["detail"]
        install = f"{sub['install']}; or {INSTALL_HINT} with LUMEN_JVM_MODE=inprocess (Linux / Windows)"
        ij = sub["ij"] or "not found"
        java = sub["java"] is not None
    else:
        langs = enabled_languages()
        available = bool(inp["available"])
        detail = "in-process mode (jpype1 + scyjava): " + inp["detail"]
        install = f"{INSTALL_HINT}; or install Fiji / a JDK and set LUMEN_JVM_MODE=subprocess (no Python package needed)"
        ij = os.environ.get("LUMEN_IJ_JAR", "").strip() or f"{IJ_ENDPOINT} (Maven)"
        java = bool(inp["java"])
    _status_cache = {"available": available, "package": inp["package"], "java": java, "detail": detail, "install": install, "languages": langs,
                     "started": _STATE.started, "ij": ij, "classpath": classpath_entries(), "mode": mode, "effective_mode": eff, "inprocess": inp}
    if sub is not None:
        _status_cache["subprocess"] = sub
    return dict(_status_cache)


def available(refresh: bool = False) -> bool:
    return bool(status(refresh)["available"])


def reset() -> None:
    """Forget the cached status (after installing the extra / changing the environment)."""
    global _status_cache
    _status_cache = None


def _reset_state() -> None:
    """Tests only: forget the (fake) JVM as well."""
    global _STATE
    _STATE = _State()
    reset()


# ---------------------------------------------------------------------------- the JVM
def jimport(name: str) -> Any:
    """``scyjava.jimport`` with a per-process cache (``ij.ImagePlus`` → the Java class)."""
    cls = _STATE.classes.get(name)
    if cls is None:
        import scyjava  # type: ignore
        cls = scyjava.jimport(name)
        _STATE.classes[name] = cls
    return cls


def _cfg(cfg: Any, names: tuple[str, ...], *args: Any) -> None:
    for n in names:
        fn = getattr(cfg, n, None)
        if callable(fn):
            fn(*args)
            return
    raise RuntimeError(f"scyjava.config has none of {names} – scyjava>=1.9 is needed")


def start() -> None:
    """Configure and start the JVM once per process: local jars / Maven endpoints, headless flags, ImageJ batch mode."""
    with _STATE.lock:
        if _STATE.started:
            return
        if not available():
            raise RuntimeError(f"JVM script bridge not available: {status()['detail']} ({INSTALL_HINT})")
        import scyjava  # type: ignore
        cfg = scyjava.config
        home = _java_available()[2]
        if home and not os.environ.get("JAVA_HOME"):
            os.environ["JAVA_HOME"] = home  # JPype locates libjvm through JAVA_HOME
        langs = enabled_languages()
        headless = headless_requested()
        already = False
        try:
            already = bool(scyjava.jvm_started())
        except Exception:  # pragma: no cover - very old scyjava
            already = False
        if not already:
            cp = classpath_entries()
            jars = [Path(p).name.lower() for p in cp]
            endpoints: list[str] = []
            if not _jar_present(jars, IJ_JAR_RE):
                endpoints.append(IJ_ENDPOINT)
            for lang in langs:
                if not _jar_present(jars, LANGUAGES[lang]["jar"]):
                    endpoints.append(LANGUAGES[lang]["endpoint"])
            endpoints += [e.strip() for e in os.environ.get("LUMEN_JVM_EXTRA_ENDPOINTS", "").split(",") if e.strip()]
            if cp:
                _cfg(cfg, ("add_classpath",), *cp)
            if endpoints:
                _cfg(cfg, ("add_endpoints",), *endpoints)
            options = (["-Djava.awt.headless=true"] if headless else []) + ["-Dapple.awt.UIElement=true"]
            heap = os.environ.get("LUMEN_JVM_HEAP", "").strip()
            if heap:
                options.append(f"-Xmx{heap}")
            for opt in options:
                _cfg(cfg, ("add_option", "add_options"), opt)
        try:
            scyjava.start_jvm()
            jimport("ij.IJ")  # the classpath must carry ij.jar (not the case when another bridge started the JVM first)
        except Exception as e:
            _STATE.error = f"{type(e).__name__}: {e}"
            _STATE.classes.clear()
            reset()
            why = "the JVM was already started by another bridge without ij.jar on its classpath (run a JVM script before opening a Bio-Formats file)" \
                if already else INSTALL_HINT
            raise RuntimeError(f"The JVM could not be started for the script bridge: {_STATE.error} – {why}") from e
        _STATE.started = True
        _STATE.error = None
        _STATE.languages = tuple(langs)
        _STATE.headless = headless
        reset()
        _batch_mode(True)


def _batch_mode(on: bool) -> None:
    """ImageJ macro batch mode: ``imp.show()`` registers images with the WindowManager instead of opening windows."""
    try:
        jimport("ij.macro.Interpreter").batchMode = bool(on)
    except Exception:  # pragma: no cover - an ij.jar without the field
        pass


# ---------------------------------------------------------------------------- numpy <-> Java arrays
_SIGNED = {np.dtype(np.uint8): np.int8, np.dtype(np.uint16): np.int16, np.dtype(np.uint32): np.int32}
_PROCESSORS = {np.dtype(np.uint8): "ij.process.ByteProcessor", np.dtype(np.uint16): "ij.process.ShortProcessor", np.dtype(np.float32): "ij.process.FloatProcessor"}


def _jarray(arr: np.ndarray) -> Any:
    """A 1-D int8 / int16 / int32 / float32 numpy array as the matching Java primitive array (buffer protocol)."""
    import jpype  # type: ignore
    kinds = {np.dtype(np.int8): jpype.JByte, np.dtype(np.int16): jpype.JShort, np.dtype(np.int32): jpype.JInt, np.dtype(np.float32): jpype.JFloat}
    arr = np.ascontiguousarray(arr)
    return jpype.JArray(kinds[arr.dtype])(arr)


def _nparray(pixels: Any, dtype: Any) -> np.ndarray:
    """A Java ``byte[]`` / ``short[]`` / ``int[]`` / ``float[]`` as a 1-D numpy array of ``dtype`` (unsigned view of the signed Java values)."""
    want = np.dtype(dtype)
    signed = _SIGNED.get(want, want.type)
    arr: np.ndarray | None = None
    try:
        arr = np.asarray(memoryview(pixels))  # JPype primitive arrays expose the buffer protocol
    except (TypeError, ValueError, BufferError):
        try:
            arr = np.asarray(pixels)
        except Exception:
            arr = None
    if arr is None or arr.dtype == object or arr.dtype.kind not in "iuf":
        arr = np.array([v for v in pixels], dtype=signed)
    arr = np.ascontiguousarray(arr).ravel()
    if arr.dtype.itemsize == want.itemsize and arr.dtype.kind in "iu" and want.kind in "iu":
        return arr.view(want)
    return arr.astype(want, copy=False)


# ---------------------------------------------------------------------------- LumenImage <-> ij.ImagePlus
def _set_calibration(imp: Any, cal: Any) -> None:
    jc = imp.getCalibration()
    jc.pixelWidth = float(cal.pixel_width)
    jc.pixelHeight = float(cal.pixel_height)
    jc.pixelDepth = float(cal.pixel_depth)
    jc.setUnit(str(cal.unit or "pixel"))
    jc.frameInterval = float(cal.frame_interval or 0.0)
    jc.setTimeUnit(str(cal.time_unit or "sec"))
    if cal.value_unit:
        jc.setValueUnit(str(cal.value_unit))
    imp.setCalibration(jc)


def _get_calibration(imp: Any) -> Any:
    from ..image import Calibration
    jc = imp.getCalibration()
    unit = str(jc.getUnit() or "pixel")
    value_unit = str(jc.getValueUnit() or "")
    if value_unit == "Gray Value":  # ImageJ's default
        value_unit = ""
    return Calibration(pixel_width=float(jc.pixelWidth), pixel_height=float(jc.pixelHeight), pixel_depth=float(jc.pixelDepth), unit=unit,
                       frame_interval=float(jc.frameInterval), time_unit=str(jc.getTimeUnit() or "sec"), value_unit=value_unit)


def to_imageplus(img: Any, name: str | None = None) -> Any:
    """A :class:`LumenImage` (T, Z, C, Y, X; 8 / 16 / 32-bit or RGB) as an ``ij.ImagePlus`` hyperstack with its
    calibration, position and display range; channels vary fastest in the stack, like ImageJ."""
    data = img.data
    nt, nz, nc, h, w = (int(v) for v in data.shape)
    rgb = bool(getattr(img, "rgb", False) and nc == 3 and data.dtype == np.uint8)
    stack = jimport("ij.ImageStack")(w, h)
    if rgb:
        ColorProcessor = jimport("ij.process.ColorProcessor")
        for t in range(nt):
            for z in range(nz):
                plane = np.asarray(data[t, z]).astype(np.uint32)
                packed = (plane[0] << 16) | (plane[1] << 8) | plane[2]
                stack.addSlice(ColorProcessor(w, h, _jarray(packed.ravel().view(np.int32))))
    else:
        dtype = data.dtype if data.dtype in _PROCESSORS else np.dtype(np.float32)
        Proc = jimport(_PROCESSORS[dtype])
        signed = _SIGNED.get(dtype, dtype.type)
        for t in range(nt):
            for z in range(nz):
                for c in range(nc):
                    plane = np.ascontiguousarray(np.asarray(data[t, z, c]), dtype=dtype)
                    stack.addSlice(Proc(w, h, _jarray(plane.ravel().view(signed)), None))
    imp = jimport("ij.ImagePlus")(str(name or img.name), stack)
    cc = 1 if rgb else nc
    imp.setDimensions(cc, nz, nt)
    if cc > 1 or (nz > 1 and nt > 1):
        imp.setOpenAsHyperStack(True)
    _set_calibration(imp, img.calibration)
    d = img.display
    try:
        imp.setPosition(1 if rgb else min(int(d.c) + 1, cc), min(int(d.z) + 1, nz), min(int(d.t) + 1, nt))
        if not rgb and d.channels and int(d.c) < len(d.channels):
            ch = d.channels[int(d.c)]
            imp.setDisplayRange(float(ch.min), float(ch.max))
    except Exception:  # pragma: no cover - a display the Java side rejects
        pass
    return imp


def from_imageplus(imp: Any, name: str | None = None) -> Any:
    """An ``ij.ImagePlus`` (any stack / hyperstack of 8, 16, 32-bit or RGB processors) as a new :class:`LumenImage`."""
    from ..image import LumenImage
    stack = imp.getStack()
    n = int(stack.getSize())
    nc, nz, nt = int(imp.getNChannels()), int(imp.getNSlices()), int(imp.getNFrames())
    if nc * nz * nt != n:
        nc, nz, nt = 1, n, 1
    w, h = int(imp.getWidth()), int(imp.getHeight())
    depth = int(imp.getBitDepth())
    rgb = depth == 24
    dtype = {8: np.uint8, 16: np.uint16, 32: np.float32, 24: np.uint8}.get(depth, np.float32)
    out = np.zeros((nt, nz, 3 if rgb else nc, h, w), dtype)
    for i in range(n):
        ip = stack.getProcessor(i + 1)
        c, z, t = i % nc, (i // nc) % nz, i // (nc * nz)
        px = ip.getPixels()
        if rgb:
            packed = _nparray(px, np.uint32).reshape(h, w)
            out[t, z, 0] = (packed >> 16) & 0xFF
            out[t, z, 1] = (packed >> 8) & 0xFF
            out[t, z, 2] = packed & 0xFF
        else:
            out[t, z, c] = _nparray(px, dtype).reshape(h, w)
    title = name or imp.getTitle()
    img = LumenImage(out, name=str(title) if title else "Untitled", axes="TZCYX", rgb=rgb, calibration=_get_calibration(imp))
    try:
        cur = int(imp.getCurrentSlice()) - 1
        if 0 <= cur < n:
            img.display.c, img.display.z, img.display.t = (0 if rgb else cur % nc), (cur // nc) % nz, cur // (nc * nz)
    except Exception:  # pragma: no cover
        pass
    return img


# ---------------------------------------------------------------------------- ROIs
_ROI_RECT, _ROI_OVAL, _ROI_POLYGON, _ROI_FREEROI, _ROI_TRACED, _ROI_LINE, _ROI_POLYLINE, _ROI_FREELINE, _ROI_ANGLE, _ROI_COMPOSITE, _ROI_POINT = range(11)


def _roi_to_java(roi: dict[str, Any] | None) -> Any:
    """A Lumen ROI dict as an ``ij.gui.Roi`` (rect / oval / polygon / freehand / polyline / line / point; masks and
    composites are not transferred)."""
    if not isinstance(roi, dict):
        return None
    t = roi.get("type")
    j = None
    if t == "rect":
        j = jimport("ij.gui.Roi")(float(roi["x"]), float(roi["y"]), float(roi["w"]), float(roi["h"]))
    elif t == "oval":
        j = jimport("ij.gui.OvalRoi")(float(roi["x"]), float(roi["y"]), float(roi["w"]), float(roi["h"]))
    elif t in ("polygon", "freehand", "polyline", "point"):
        pts = [(float(p[0]), float(p[1])) for p in roi.get("points", [])]
        if not pts:
            return None
        xs = _jarray(np.array([p[0] for p in pts], np.float32))
        ys = _jarray(np.array([p[1] for p in pts], np.float32))
        if t == "point":
            j = jimport("ij.gui.PointRoi")(xs, ys, len(pts))
        else:
            kind = {"polygon": _ROI_POLYGON, "freehand": _ROI_FREEROI, "polyline": _ROI_POLYLINE}[t]
            j = jimport("ij.gui.PolygonRoi")(xs, ys, len(pts), kind)
    elif t == "line":
        j = jimport("ij.gui.Line")(float(roi["x1"]), float(roi["y1"]), float(roi["x2"]), float(roi["y2"]))
        if float(roi.get("width", 1) or 1) > 1:
            j.setStrokeWidth(float(roi["width"]))
    if j is not None and roi.get("name"):
        j.setName(str(roi["name"]))
    return j


def _roi_from_java(jroi: Any) -> dict[str, Any] | None:
    """An ``ij.gui.Roi`` as a Lumen ROI dict (``None`` for ``null``)."""
    if jroi is None:
        return None
    t = int(jroi.getType())
    d: dict[str, Any]
    if t in (_ROI_RECT, _ROI_OVAL):
        b = jroi.getBounds()
        d = {"type": "rect" if t == _ROI_RECT else "oval", "x": float(b.x), "y": float(b.y), "w": float(b.width), "h": float(b.height)}
    elif t == _ROI_LINE:
        d = {"type": "line", "x1": float(jroi.x1d), "y1": float(jroi.y1d), "x2": float(jroi.x2d), "y2": float(jroi.y2d)}
        try:
            d["width"] = max(1, int(round(float(jroi.getStrokeWidth()))))
        except Exception:
            d["width"] = 1
    else:
        fp = jroi.getFloatPolygon()
        npts = int(fp.npoints)
        pts = [[float(fp.xpoints[i]), float(fp.ypoints[i])] for i in range(npts)]
        kind = {_ROI_POINT: "point", _ROI_POLYLINE: "polyline", _ROI_FREELINE: "polyline", _ROI_ANGLE: "polyline", _ROI_FREEROI: "freehand", _ROI_TRACED: "freehand"}.get(t, "polygon")
        d = {"type": kind, "points": pts}
    name = jroi.getName()
    if name:
        d["name"] = str(name)
    return d


def _roi_geometry(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in ("id", "image_id", "position")}


# ---------------------------------------------------------------------------- Results table / ROI Manager exchange
def _seed_results(rt: Any, session: Any) -> list[dict[str, Any]]:
    """Fill ImageJ's Results table from the session's and return the rows as the JVM will hand them back."""
    rt.reset()
    for row in session.results.rows:
        rt.incrementCounter()
        for k, v in row.items():
            if k == "Label":
                rt.addLabel(str(v))
            elif isinstance(v, (bool, str)) or v is None:
                rt.addValue(str(k), "" if v is None else str(v))
            else:
                try:
                    rt.addValue(str(k), float(v))
                except (TypeError, ValueError):
                    rt.addValue(str(k), str(v))
    return _collect_results(rt)


def _collect_results(rt: Any) -> list[dict[str, Any]]:
    """ImageJ's Results table as Lumen rows (numbers as floats, string cells and labels as strings)."""
    n = int(rt.size())
    if n == 0:
        return []
    headings = [str(h) for h in rt.getHeadings()]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {}
        for h in headings:
            if h == "Label":
                lab = rt.getLabel(i)
                if lab is not None:
                    row["Label"] = str(lab)
                continue
            col = int(rt.getColumnIndex(h))
            if col < 0:
                continue
            v = float(rt.getValueAsDouble(col, i))
            if math.isnan(v):  # a string cell, or an empty one (left out of the row)
                s = rt.getStringValue(col, i)
                s = "" if s is None else str(s)
                if s and s != "NaN":
                    row[h] = s
            else:
                row[h] = v
        rows.append(row)
    return rows


def _roi_manager() -> Any:
    """ImageJ's ROI Manager, or ``None`` when the JVM cannot build one (headless: it is a ``java.awt.Frame``)."""
    try:
        RoiManager = jimport("ij.plugin.frame.RoiManager")
        rm = RoiManager.getInstance()
        if rm is None:
            rm = RoiManager.getRoiManager()
        _STATE.rm_error = None
        return rm
    except Exception as e:
        _STATE.rm_error = f"{type(e).__name__}: {e}".strip(": ")
        return None


def _seed_rois(rm: Any, session: Any) -> list[dict[str, Any]]:
    rm.reset()
    seeded: list[dict[str, Any]] = []
    for r in session.roi_manager.rois:
        j = _roi_to_java(r)
        if j is None:
            continue
        pos = r.get("position") or {}
        try:
            if pos:
                j.setPosition(int(pos.get("c", 0)) + 1, int(pos.get("z", 0)) + 1, int(pos.get("t", 0)) + 1)
        except Exception:
            pass
        rm.addRoi(j)
        seeded.append(r)
    return seeded


def _copy_rois_back(rm: Any, session: Any, seeded: list[dict[str, Any]], image_id: str | None) -> int:
    """Replace the session's ROI Manager with the JVM's list when the script changed it; unchanged leading entries keep their ids."""
    rois = [_roi_from_java(j) for j in (rm.getRoisAsArray() or [])]
    rois = [r for r in rois if r is not None]
    if [_roi_geometry(r) for r in rois] == [_roi_geometry(r) for r in seeded]:
        return 0
    session.roi_manager.clear()
    jrois = list(rm.getRoisAsArray() or [])
    for i, r in enumerate(rois):
        if i < len(seeded) and _roi_geometry(r) == _roi_geometry(seeded[i]):
            session.roi_manager.rois.append(dict(seeded[i]))
            continue
        pos = None
        try:
            j = jrois[i]
            c, z, t = int(j.getCPosition()), int(j.getZPosition()), int(j.getTPosition())
            if c or z or t:
                pos = {"c": max(c - 1, 0), "z": max(z - 1, 0), "t": max(t - 1, 0)}
        except Exception:
            pos = None
        session.roi_manager.add(r, image_id=image_id, position=pos)
    session._changed("rois")
    return len(rois)


# ---------------------------------------------------------------------------- output capture
class _Capture:
    """Redirect the JVM's ``System.out`` / ``System.err`` into a buffer for the duration of a script (``println``,
    BeanShell ``print``, JRuby ``puts`` and ``IJ.log`` – which prints to stdout when ImageJ has no Log window)."""

    def __init__(self) -> None:
        self.text = ""

    def __enter__(self) -> "_Capture":
        self.System = jimport("java.lang.System")
        self.out0, self.err0 = self.System.out, self.System.err
        self.baos = jimport("java.io.ByteArrayOutputStream")()
        ps = jimport("java.io.PrintStream")(self.baos, True, "UTF-8")
        self.System.setOut(ps)
        self.System.setErr(ps)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.System.out.flush()
        finally:
            self.System.setOut(self.out0)
            self.System.setErr(self.err0)
        self.text = str(self.baos.toString("UTF-8"))


# ---------------------------------------------------------------------------- the engines
def _run_groovy(code: str, bindings: dict[str, Any], name: str) -> None:
    b = jimport("groovy.lang.Binding")()
    for k, v in bindings.items():
        b.setVariable(k, v)
    jimport("groovy.lang.GroovyShell")(b).evaluate(code, name)


def _run_beanshell(code: str, bindings: dict[str, Any], name: str) -> None:
    interp = jimport("bsh.Interpreter")()
    interp.eval(BSH_PRELUDE)
    for k, v in bindings.items():
        if v is not None:
            interp.set(k, v)
    interp.eval(code)


def _run_rhino(code: str, bindings: dict[str, Any], name: str) -> None:
    Context = jimport("org.mozilla.javascript.Context")
    cx = Context.enter()
    try:
        try:
            cx.setLanguageVersion(200)  # ES6
        except Exception:
            pass
        scope = jimport("org.mozilla.javascript.ImporterTopLevel")(cx)
        cx.evaluateString(scope, RHINO_PRELUDE, "<lumen-prelude>", 1, None)
        put = jimport("org.mozilla.javascript.ScriptableObject").putProperty
        for k, v in bindings.items():
            put(scope, k, None if v is None else Context.javaToJS(v, scope))
        cx.evaluateString(scope, code, name, 1, None)
    finally:
        Context.exit()


def _run_jruby(code: str, bindings: dict[str, Any], name: str) -> None:
    container = jimport("org.jruby.embed.ScriptingContainer")()
    for k, v in bindings.items():
        container.put(k, v)
    container.runScriptlet(code)


def _run_clojure(code: str, bindings: dict[str, Any], name: str) -> None:
    Clojure = jimport("clojure.java.api.Clojure")
    ns = Clojure.read("user")
    Clojure.var("clojure.core", "create-ns").invoke(ns)
    intern = Clojure.var("clojure.core", "intern")
    for k, v in bindings.items():
        intern.invoke(ns, Clojure.read(k), v)
    Clojure.var("clojure.core", "load-string").invoke("(in-ns 'user)" + code)


ENGINES: dict[str, Callable[[str, dict[str, Any], str], None]] = {"groovy": _run_groovy, "beanshell": _run_beanshell, "rhino": _run_rhino,
                                                                   "jruby": _run_jruby, "clojure": _run_clojure}
_CLASS_BINDINGS = {"groovy", "rhino"}  # engines where a bound java.lang.Class works as a static receiver (IJ.log(...))


# ---------------------------------------------------------------------------- parameters
def _strip_header(code: str, header_lines: list[int]) -> str:
    """The script without its ``#@`` lines (blanked, so line numbers in errors stay right)."""
    lines = code.splitlines(keepends=True)
    for i in header_lines:
        lines[i] = "\n" if lines[i].endswith("\n") else ""
    return "".join(lines)


def _java_value(p: Any, v: Any) -> Any:
    """A resolved ``#@`` value in the declared Java type (``Integer`` → ``java.lang.Integer`` …)."""
    if v is None:
        return None
    try:
        import jpype  # type: ignore
        jt = str(p.java_type).rsplit(".", 1)[-1].lower()
        if p.kind == "int":
            return jpype.JLong(int(v)) if jt == "long" else jpype.JInt(int(v))
        if p.kind == "float":
            return jpype.JFloat(float(v)) if jt == "float" else jpype.JDouble(float(v))
        if p.kind == "bool":
            return jpype.JBoolean(bool(v))
    except Exception:
        return v
    return v


def _param_bindings(params: list[Any], values: dict[str, Any], imp: Any, rt: Any, rm: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in params:
        if p.io == "output":
            continue
        kind = p.kind
        if kind in ("file", "directory"):
            v = values.get(p.name)
            out[p.name] = jimport("java.io.File")(str(v)) if v else None
        elif kind == "image":
            out[p.name] = imp
        elif kind == "table":
            out[p.name] = rt
        elif kind == "roimanager":
            out[p.name] = rm
        elif kind == "service":
            out[p.name] = None  # SciJava services do not exist on a plain ij.jar
        else:
            out[p.name] = _java_value(p, values.get(p.name))
    return out


def _format_error(e: BaseException) -> str:
    msg = str(e).strip()
    try:
        jname = str(e.getClass().getName())  # type: ignore[attr-defined]
        if jname and jname not in msg:
            msg = f"{jname}: {msg}" if msg else jname
    except Exception:
        if not msg:
            msg = type(e).__name__
        elif type(e).__name__ not in msg and not msg.startswith(("java.", "groovy.", "bsh.", "org.")):
            msg = f"{type(e).__name__}: {msg}"
    return msg


# ---------------------------------------------------------------------------- the runner
def run_jvm_script(session: Any, code: str, language: str, params: dict[str, Any] | None = None, roi: Any = None) -> dict[str, Any]:
    """Run ``code`` in a JVM script language with ImageJ 1.x: ``{"stdout", "error", "language", "roi", "new_images",
    "changed", "results_rows", "roi_manager"[, "parameters", "note"]}`` – like :func:`run_ij_script`, never raises
    for a missing bridge or a failing script (the reason is in ``error``)."""
    lang = normalize_language(language)
    base: dict[str, Any] = {"stdout": "", "error": None, "language": lang or str(language)}
    if lang is None:
        return {**base, "error": f"Unknown JVM script language {language!r} (one of {', '.join(LANGUAGES)})"}
    label = LANGUAGES[lang]["label"]
    if effective_mode() == "subprocess":
        from . import jvm_subprocess
        return jvm_subprocess.run_jvm_script_subprocess(session, code, lang, params, roi)
    st = status()
    if not st["available"]:
        return {**base, "error": f"{label} scripts need the Java bridge, which is not available: {st['detail']}. Install: {st['install']}"}
    if lang not in st["languages"]:
        return {**base, "error": f"{label} is an optional bridge language: start Lumen with LUMEN_JVM_LANGUAGES={lang} (or put its jar on LUMEN_JVM_CLASSPATH)"}
    try:
        start()
    except RuntimeError as e:
        return {**base, "error": str(e)}
    if lang not in _STATE.languages:
        return {**base, "error": f"The JVM was started without {label}: restart Lumen with LUMEN_JVM_LANGUAGES={lang}"}
    from .params import parse, resolve_values
    plist, header_lines = parse(code)
    values = resolve_values(plist, params)
    body = _strip_header(code, header_lines)
    with _STATE.lock:
        out = _run_locked(session, body, lang, label, plist, values, roi if isinstance(roi, dict) else None)
    if plist:
        out["parameters"] = [p.descriptor() for p in plist]
    return out


def _run_locked(session: Any, body: str, lang: str, label: str, plist: list[Any], values: dict[str, Any], roi: dict[str, Any] | None) -> dict[str, Any]:
    from ..undo import History
    IJ = jimport("ij.IJ")
    WM = jimport("ij.WindowManager")
    Interp = jimport("ij.macro.Interpreter")
    _batch_mode(True)
    img = session.active
    imp = snap = None
    if img is not None:
        imp = to_imageplus(img)
        jroi = _roi_to_java(roi)
        if jroi is not None:
            imp.setRoi(jroi)
        Interp.addBatchModeImage(imp)
        WM.setTempCurrentImage(imp)
        snap = History.snapshot(img, f"{label} script")
    rt = jimport("ij.measure.ResultsTable").getResultsTable()
    seed_rows = _seed_results(rt, session)
    rm = _roi_manager()
    seed_rois = _seed_rois(rm, session) if rm is not None else []
    bindings: dict[str, Any] = {"imp": imp}
    if lang in _CLASS_BINDINGS:
        bindings.update({"IJ": IJ, "WindowManager": WM})
    bindings.update(_param_bindings(plist, values, imp, rt, rm))
    error: str | None = None
    with _Capture() as cap:
        try:
            ENGINES[lang](body, bindings, f"script{LANGUAGES[lang]['ext']}")
        except Exception as e:  # noqa: BLE001 - every Java / engine error is the script's error
            error = _format_error(e)
    stdout = cap.text
    try:
        extra = IJ.getLog()
        if extra:
            have = set(stdout.splitlines())
            missing = [ln for ln in str(extra).splitlines() if ln not in have]
            if missing:
                stdout += ("" if not stdout or stdout.endswith("\n") else "\n") + "\n".join(missing) + "\n"
    except Exception:
        pass
    for line in stdout.splitlines():
        if line.strip():
            session.add_log(line)
    # images the script created (and show()-ed in batch mode)
    bound_id = int(imp.getID()) if imp is not None else None
    ids = [int(i) for i in (WM.getIDList() or [])]
    new_ids: list[str] = []
    for jid in ids:
        if jid == bound_id:
            continue
        jimp = WM.getImage(jid)
        if jimp is None:
            continue
        try:
            new_ids.append(session.add_image(from_imageplus(jimp)).id)
        except Exception as e:  # noqa: BLE001 - an image type the converter cannot read is reported, not fatal
            error = (error + "\n" if error else "") + f"could not convert image {jimp.getTitle()!s}: {e}"
        try:
            Interp.removeBatchModeImage(jimp)
        except Exception:
            pass
    # the bound image: back in place when the script changed it
    changed = False
    roi_out = None
    if imp is not None and img is not None:
        try:
            conv = from_imageplus(imp, name=img.name)
            pixels = conv.data.shape != img.data.shape or conv.data.dtype != img.data.dtype or conv.rgb != img.rgb or not np.array_equal(conv.data, img.data)
            cal = conv.calibration
            cal_changed = cal.to_dict() != img.calibration.to_dict()
            title = str(imp.getTitle() or "")
            renamed = bool(title) and title != img.name
            if pixels or cal_changed or renamed:
                hist = session.histories.get(img.id)
                if hist is not None and snap is not None:
                    hist.undo.append(snap)
                    hist.redo.clear()
                    hist._trim()
                if pixels:
                    img.replace_data(conv.data, rgb=conv.rgb)
                    changed = True
                if cal_changed:
                    img.calibration = cal
                if renamed:
                    img.name = session.unique_name(title)
            roi_out = _roi_from_java(imp.getRoi())
        except Exception as e:  # noqa: BLE001
            error = (error + "\n" if error else "") + f"could not convert the image back: {e}"
        try:
            Interp.removeBatchModeImage(imp)
        except Exception:
            pass
    try:
        WM.setTempCurrentImage(None)
    except Exception:
        pass
    rows = _collect_results(rt)
    if rows != seed_rows:
        session.results.clear()
        session.results.append(rows)
        session._changed("results")
    rois_copied = _copy_rois_back(rm, session, seed_rois, img.id if img is not None else None) if rm is not None else 0
    session._changed("images")
    out: dict[str, Any] = {"stdout": stdout, "error": error, "language": lang, "mode": "inprocess", "changed": changed, "new_images": new_ids,
                           "results_rows": len(rows), "roi_manager": rm is not None, "rois_copied": rois_copied}
    if imp is not None:
        out["roi"] = roi_out
    if rm is None:
        out["note"] = f"ROI Manager not available on this JVM ({_STATE.rm_error or 'headless'}): start Lumen with LUMEN_JVM_HEADLESS=0 to exchange ROI Manager entries"
    return out
