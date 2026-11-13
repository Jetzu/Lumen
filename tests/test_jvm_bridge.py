"""Tests for the optional JVM script bridge (lumen.core.ijshim.jvm_bridge) – fully mocked: fake ``jpype`` / ``scyjava``
modules in ``sys.modules`` and a fake Java side (``ij.*``, ``java.*``, the Groovy / BeanShell / Rhino / JRuby engines)
whose "scripts" are executed as Python over the bound objects.  No JVM is ever started, nothing is installed."""
from __future__ import annotations

import io
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.ijshim import jvm_bridge as jb
from lumen.core.image import Calibration, LumenImage
from lumen.core.session import Session
from lumen.server.app import create_app


# ====================================================================== a fake Java side
class HeadlessException(Exception):
    pass


class _Rect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class _FloatPolygon:
    def __init__(self, xs, ys):
        self.xpoints, self.ypoints, self.npoints = list(xs), list(ys), len(xs)


class Roi:
    RECTANGLE, OVAL, POLYGON, FREEROI, TRACED_ROI, LINE, POLYLINE, FREELINE, ANGLE, COMPOSITE, POINT = range(11)
    _type = 0

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)
        self._name = None
        self._c = self._z = self._t = 0
        self._stroke = 1.0

    def getType(self):
        return self._type

    def getBounds(self):
        return _Rect(int(self.x), int(self.y), int(self.w), int(self.h))

    def getName(self):
        return self._name

    def setName(self, n):
        self._name = n

    def setPosition(self, c, z, t):
        self._c, self._z, self._t = c, z, t

    def getCPosition(self):
        return self._c

    def getZPosition(self):
        return self._z

    def getTPosition(self):
        return self._t

    def getFloatPolygon(self):
        return _FloatPolygon([self.x, self.x + self.w, self.x + self.w, self.x], [self.y, self.y, self.y + self.h, self.y + self.h])


class OvalRoi(Roi):
    _type = 1


class PolygonRoi(Roi):
    def __init__(self, xs, ys, n, kind):
        xs, ys = [float(v) for v in xs][:n], [float(v) for v in ys][:n]
        super().__init__(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self._type = kind
        self._xs, self._ys = xs, ys

    def getFloatPolygon(self):
        return _FloatPolygon(self._xs, self._ys)


class PointRoi(PolygonRoi):
    def __init__(self, xs, ys, n):
        super().__init__(xs, ys, n, Roi.POINT)


class Line(Roi):
    _type = 5

    def __init__(self, x1, y1, x2, y2):
        super().__init__(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.x1d, self.y1d, self.x2d, self.y2d = float(x1), float(y1), float(x2), float(y2)

    def setStrokeWidth(self, w):
        self._stroke = float(w)

    def getStrokeWidth(self):
        return self._stroke


class _Processor:
    _dtype = np.int8
    depth = 8

    def __init__(self, w, h, pixels=None, cm=None):
        self.w, self.h = int(w), int(h)
        self._px = np.zeros(self.w * self.h, self._dtype) if pixels is None else np.array(pixels, dtype=self._dtype)

    def getPixels(self):
        return self._px

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

    def invert(self):
        if self._dtype is np.int8:
            self._px = (255 - self._px.view(np.uint8)).astype(np.uint8).view(np.int8)
        elif self._dtype is np.int16:
            self._px = (65535 - self._px.view(np.uint16)).astype(np.uint16).view(np.int16)
        elif self._dtype is np.int32:
            self._px = (0xFFFFFF - (self._px.view(np.uint32) & 0xFFFFFF)).astype(np.uint32).view(np.int32)
        else:
            self._px = -self._px

    def set(self, v):
        self._px[:] = v


class ByteProcessor(_Processor):
    pass


class ShortProcessor(_Processor):
    _dtype = np.int16
    depth = 16


class FloatProcessor(_Processor):
    _dtype = np.float32
    depth = 32


class ColorProcessor(_Processor):
    _dtype = np.int32
    depth = 24


class ImageStack:
    def __init__(self, w, h):
        self.w, self.h = int(w), int(h)
        self._slices: list = []
        self._labels: list = []

    def addSlice(self, *args):  # addSlice(ip) | addSlice(label, ip)
        self._slices.append(args[-1])
        self._labels.append(args[0] if len(args) == 2 else None)

    def getSize(self):
        return len(self._slices)

    def getProcessor(self, n):
        return self._slices[n - 1]

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h


class JCalibration:
    def __init__(self):
        self.pixelWidth = self.pixelHeight = self.pixelDepth = 1.0
        self.frameInterval = 0.0
        self._unit, self._tu, self._vu = "pixel", "sec", "Gray Value"

    def setUnit(self, u):
        self._unit = u or "pixel"

    def getUnit(self):
        return self._unit

    def setTimeUnit(self, u):
        self._tu = u

    def getTimeUnit(self):
        return self._tu

    def setValueUnit(self, u):
        self._vu = u

    def getValueUnit(self):
        return self._vu


class Interpreter:
    batchMode = False
    _images: list = []

    @classmethod
    def addBatchModeImage(cls, imp):
        if imp not in cls._images:
            cls._images.append(imp)

    @classmethod
    def removeBatchModeImage(cls, imp):
        if imp in cls._images:
            cls._images.remove(imp)


class WindowManager:
    _current = None

    @classmethod
    def setTempCurrentImage(cls, imp):
        cls._current = imp

    @classmethod
    def getCurrentImage(cls):
        return cls._current

    @classmethod
    def getIDList(cls):
        ids = [imp.getID() for imp in Interpreter._images]
        return ids or None

    @classmethod
    def getImage(cls, jid):
        return next((i for i in Interpreter._images if i.getID() == jid), None)

    @classmethod
    def getImageCount(cls):
        return len(Interpreter._images)


class ImagePlus:
    _next = [0]

    def __init__(self, title, stack_or_ip=None):
        ImagePlus._next[0] -= 1
        self._id = ImagePlus._next[0]
        self._title = title
        if isinstance(stack_or_ip, ImageStack):
            self._stack = stack_or_ip
        else:
            self._stack = ImageStack(stack_or_ip.w, stack_or_ip.h)
            self._stack.addSlice(stack_or_ip)
        self._nc, self._nz, self._nt = 1, self._stack.getSize(), 1
        self._cal = JCalibration()
        self._roi = None
        self._slice = 1
        self._hyper = False
        self._range = (0.0, 255.0)

    def getID(self):
        return self._id

    def getTitle(self):
        return self._title

    def setTitle(self, t):
        self._title = t

    def getStack(self):
        return self._stack

    def getProcessor(self):
        return self._stack.getProcessor(self._slice)

    def getWidth(self):
        return self._stack.w

    def getHeight(self):
        return self._stack.h

    def getStackSize(self):
        return self._stack.getSize()

    def getBitDepth(self):
        return self._stack.getProcessor(1).depth

    def setDimensions(self, c, z, t):
        if c * z * t == self._stack.getSize():
            self._nc, self._nz, self._nt = c, z, t

    def getNChannels(self):
        return self._nc

    def getNSlices(self):
        return self._nz

    def getNFrames(self):
        return self._nt

    def setOpenAsHyperStack(self, b):
        self._hyper = bool(b)

    def getCalibration(self):
        return self._cal

    def setCalibration(self, cal):
        self._cal = cal

    def setPosition(self, c, z, t):
        self._slice = (t - 1) * self._nz * self._nc + (z - 1) * self._nc + c

    def getCurrentSlice(self):
        return self._slice

    def setSlice(self, n):
        self._slice = n

    def setDisplayRange(self, lo, hi):
        self._range = (lo, hi)

    def getDisplayRangeMin(self):
        return self._range[0]

    def getDisplayRangeMax(self):
        return self._range[1]

    def setRoi(self, r):
        self._roi = r

    def getRoi(self):
        return self._roi

    def deleteRoi(self):
        self._roi = None

    def show(self):  # ImageJ batch mode: registered with the WindowManager, no window
        Interpreter.addBatchModeImage(self)
        WindowManager.setTempCurrentImage(self)

    def close(self):
        Interpreter.removeBatchModeImage(self)


class ByteArrayOutputStream:
    def __init__(self):
        self.buf = io.StringIO()

    def toString(self, enc="UTF-8"):
        return self.buf.getvalue()


class PrintStream:
    def __init__(self, target=None, autoflush=True, enc="UTF-8"):
        self.target = target

    def _write(self, s):
        if self.target is not None:
            self.target.buf.write(s)

    def println(self, s=""):
        self._write(str(s) + "\n")

    def print(self, s):  # noqa: A003
        self._write(str(s))

    def flush(self):
        pass


class System:
    out = PrintStream()
    err = PrintStream()

    @classmethod
    def setOut(cls, ps):
        cls.out = ps

    @classmethod
    def setErr(cls, ps):
        cls.err = ps


class File:
    def __init__(self, p):
        self._p = str(p)

    def getPath(self):
        return self._p


class IJ:
    runs: list = []

    @staticmethod
    def log(s=""):
        System.out.println(str(s))

    @staticmethod
    def getImage():
        return WindowManager.getCurrentImage()

    @staticmethod
    def getLog():
        return None

    @staticmethod
    def run(imp, cmd="", args=""):
        IJ.runs.append((imp, cmd, args))
        if cmd == "Invert":
            for i in range(imp.getStackSize()):
                imp.getStack().getProcessor(i + 1).invert()


class ResultsTable:
    _instance = None
    COLUMN_NOT_FOUND = -1

    def __init__(self):
        self._rows: list = []
        self._cols: list = []
        self._labels: dict = {}

    @classmethod
    def getResultsTable(cls):
        if cls._instance is None:
            cls._instance = ResultsTable()
        return cls._instance

    def reset(self):
        self._rows.clear()
        self._cols.clear()
        self._labels.clear()

    def incrementCounter(self):
        self._rows.append({})

    def addValue(self, col, v):
        if col not in self._cols:
            self._cols.append(col)
        self._rows[-1][col] = v

    def addLabel(self, s):
        self._labels[len(self._rows) - 1] = s

    def getLabel(self, i):
        return self._labels.get(i)

    def size(self):
        return len(self._rows)

    def getCounter(self):
        return self.size()

    def getHeadings(self):
        return (["Label"] if self._labels else []) + list(self._cols)

    def getColumnIndex(self, h):
        return self._cols.index(h) if h in self._cols else -1

    def getValueAsDouble(self, col, row):
        v = self._rows[row].get(self._cols[col])
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")

    def getStringValue(self, col, row):
        v = self._rows[row].get(self._cols[col])
        return "" if v is None else str(v)

    def getValue(self, h, row):
        return self.getValueAsDouble(self.getColumnIndex(h), row)


class RoiManager:
    _instance = None
    headless = True  # a java.awt.Frame cannot be built on a headless JVM

    def __init__(self):
        if RoiManager.headless:
            raise HeadlessException("No X11 DISPLAY variable was set")
        self._rois: list = []

    @classmethod
    def getInstance(cls):
        return cls._instance

    @classmethod
    def getRoiManager(cls):
        if cls._instance is None:
            cls._instance = RoiManager()
        return cls._instance

    def reset(self):
        self._rois.clear()

    def addRoi(self, r):
        self._rois.append(r)

    def getCount(self):
        return len(self._rois)

    def getRoi(self, i):
        return self._rois[i]

    def getRoisAsArray(self):
        return list(self._rois)

    def getName(self, i):
        return self._rois[i].getName()


# ---------------------------------------------------------------- the engines: a "script" is Python run over the bound objects
_SHORT_NAMES = {"ImagePlus": ImagePlus, "ImageStack": ImageStack, "ByteProcessor": ByteProcessor, "ShortProcessor": ShortProcessor, "FloatProcessor": FloatProcessor,
                "ColorProcessor": ColorProcessor, "Calibration": JCalibration, "IJ": IJ, "WindowManager": WindowManager, "ResultsTable": ResultsTable, "RoiManager": RoiManager,
                "Roi": Roi, "OvalRoi": OvalRoi, "PolygonRoi": PolygonRoi, "PointRoi": PointRoi, "Line": Line, "File": File, "System": System}


def _exec(code: str, variables: dict) -> None:
    g = dict(_SHORT_NAMES)
    g.update(variables)
    g["println"] = lambda *a: System.out.println(" ".join(str(x) for x in a))
    g["print"] = g["println"]
    exec(compile(code, "<fake-jvm-script>", "exec"), g)


class Binding:
    def __init__(self):
        self._vars: dict = {}

    def setVariable(self, k, v):
        self._vars[k] = v

    def getVariable(self, k):
        return self._vars[k]


class GroovyShell:
    used: list = []

    def __init__(self, binding):
        self.b = binding

    def evaluate(self, code, name="script.groovy"):
        GroovyShell.used.append(name)
        _exec(code, self.b._vars)


class BshInterpreter:
    last = None

    def __init__(self):
        self._vars: dict = {}
        self.preludes: list = []
        BshInterpreter.last = self

    def set(self, k, v):  # noqa: A003
        self._vars[k] = v

    def eval(self, code):  # noqa: A003
        if code.startswith("import "):
            self.preludes.append(code)
            return
        _exec(code, self._vars)


class ImporterTopLevel:
    last = None

    def __init__(self, cx):
        self.vars: dict = {}
        self.prelude = None
        ImporterTopLevel.last = self


class RhinoContext:
    depth = 0

    @classmethod
    def enter(cls):
        cls.depth += 1
        return cls()

    @classmethod
    def exit(cls):  # noqa: A003
        cls.depth -= 1

    def setLanguageVersion(self, v):
        self.version = v

    def evaluateString(self, scope, code, name, line, sec):
        if name == "<lumen-prelude>":
            scope.prelude = code
            return
        _exec(code, scope.vars)

    @staticmethod
    def javaToJS(v, scope):
        return v


class ScriptableObject:
    @staticmethod
    def putProperty(scope, k, v):
        scope.vars[k] = v


class ScriptingContainer:
    used = 0

    def __init__(self):
        self._vars: dict = {}

    def put(self, k, v):
        self._vars[k] = v

    def runScriptlet(self, code):
        ScriptingContainer.used += 1
        _exec(code, self._vars)


FAKE_CLASSES = {"ij.ImagePlus": ImagePlus, "ij.ImageStack": ImageStack, "ij.process.ByteProcessor": ByteProcessor, "ij.process.ShortProcessor": ShortProcessor,
                "ij.process.FloatProcessor": FloatProcessor, "ij.process.ColorProcessor": ColorProcessor, "ij.measure.Calibration": JCalibration, "ij.IJ": IJ,
                "ij.WindowManager": WindowManager, "ij.macro.Interpreter": Interpreter, "ij.measure.ResultsTable": ResultsTable, "ij.plugin.frame.RoiManager": RoiManager,
                "ij.gui.Roi": Roi, "ij.gui.OvalRoi": OvalRoi, "ij.gui.PolygonRoi": PolygonRoi, "ij.gui.PointRoi": PointRoi, "ij.gui.Line": Line,
                "java.lang.System": System, "java.io.ByteArrayOutputStream": ByteArrayOutputStream, "java.io.PrintStream": PrintStream, "java.io.File": File,
                "groovy.lang.Binding": Binding, "groovy.lang.GroovyShell": GroovyShell, "bsh.Interpreter": BshInterpreter,
                "org.mozilla.javascript.Context": RhinoContext, "org.mozilla.javascript.ImporterTopLevel": ImporterTopLevel,
                "org.mozilla.javascript.ScriptableObject": ScriptableObject, "org.jruby.embed.ScriptingContainer": ScriptingContainer}


class FakeConfig:
    def __init__(self):
        self.endpoints: list = []
        self.classpath: list = []
        self.options: list = []

    def add_endpoints(self, *e):
        self.endpoints.extend(e)

    def add_classpath(self, *p):
        self.classpath.extend(p)

    def add_option(self, o):
        self.options.append(o)


class _JByte:
    pass


class _JShort:
    pass


class JInt(int):
    pass


class JLong(int):
    pass


class JFloat(float):
    pass


class JDouble(float):
    pass


def make_fake_modules(fail_start: bool = False, no_ij: bool = False) -> tuple[types.ModuleType, types.ModuleType]:
    scy = types.ModuleType("scyjava")
    scy.config = FakeConfig()
    scy._started = False

    def start_jvm():
        if fail_start:
            raise RuntimeError("no libjvm found")
        scy._started = True

    def jimport(name):
        if not scy._started:
            raise RuntimeError("JVM not started")
        if no_ij and name.startswith("ij."):
            raise ImportError(f"class {name} not found")
        try:
            return FAKE_CLASSES[name]
        except KeyError:
            raise ImportError(f"class {name} not found") from None

    scy.start_jvm = start_jvm
    scy.jvm_started = lambda: scy._started
    scy.jimport = jimport
    jp = types.ModuleType("jpype")
    jp.JByte, jp.JShort, jp.JInt, jp.JLong, jp.JFloat, jp.JDouble, jp.JBoolean = _JByte, _JShort, JInt, JLong, JFloat, JDouble, bool
    dtypes = {_JByte: np.int8, _JShort: np.int16, JInt: np.int32, JFloat: np.float32}
    jp.JArray = lambda t: (lambda seq: np.array(seq, dtype=dtypes[t]))
    jp.JException = Exception
    return scy, jp


def _reset_java_side() -> None:
    Interpreter._images.clear()
    Interpreter.batchMode = False
    WindowManager._current = None
    ResultsTable._instance = None
    RoiManager._instance = None
    RoiManager.headless = True
    GroovyShell.used.clear()
    BshInterpreter.last = None
    ImporterTopLevel.last = None
    RhinoContext.depth = 0
    ScriptingContainer.used = 0
    IJ.runs.clear()
    System.out = PrintStream()
    System.err = PrintStream()


ENV = ("LUMEN_IJ_JAR", "LUMEN_JVM_CLASSPATH", "LUMEN_JVM_LANGUAGES", "LUMEN_JVM_HEADLESS", "LUMEN_JVM_HEAP", "LUMEN_JVM_EXTRA_ENDPOINTS")


@pytest.fixture(autouse=True)
def _inprocess_mode(monkeypatch):
    """Every test here drives the *in-process* (JPype) mode with a fake Java side: pin LUMEN_JVM_MODE so that a machine
    with a JDK and Fiji's jars (where ``auto`` means the subprocess runner – tests/test_jvm_subprocess.py) behaves the same."""
    monkeypatch.setenv("LUMEN_JVM_MODE", "inprocess")
    jb.reset()
    yield
    jb.reset()


@pytest.fixture
def fake_jvm(monkeypatch):
    """Fake ``scyjava`` / ``jpype`` in sys.modules, a pretend JDK, a clean fake Java side."""
    scy, jp = make_fake_modules()
    monkeypatch.setitem(sys.modules, "scyjava", scy)
    monkeypatch.setitem(sys.modules, "jpype", jp)
    monkeypatch.setattr(jb, "_java_available", lambda: (True, "fake java 21", None))
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    jb._reset_state()
    _reset_java_side()
    yield scy
    jb._reset_state()
    _reset_java_side()


@pytest.fixture
def no_bridge(monkeypatch):
    for n in ("jpype", "scyjava"):
        monkeypatch.delitem(sys.modules, n, raising=False)
    monkeypatch.setattr(jb, "_importable", lambda name: False)
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    jb._reset_state()
    yield jb
    jb._reset_state()


def blobs_session(shape=(1, 1, 1, 6, 8), dtype=np.uint8, name="blobs") -> tuple[Session, LumenImage]:
    s = Session()
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    img = s.add_image(LumenImage(data, name=name, axes="TZCYX"))
    return s, img


# ====================================================================== availability
def test_status_without_packages(no_bridge):
    st = no_bridge.status()
    assert not st["available"] and not st["package"] and not st["java"]
    assert "lumen-imaging[jvm]" in st["detail"] and "JDK" in st["install"] and "Maven" in st["install"]
    assert st["languages"] == ["groovy", "beanshell", "rhino"] and not st["started"]
    assert st["mode"] == "inprocess" and st["effective_mode"] == "inprocess" and st["inprocess"]["available"] is False and "subprocess" not in st
    assert "LUMEN_JVM_MODE=subprocess" in st["install"]  # the install hint names the other way too
    assert no_bridge.available() is False
    out = no_bridge.run_jvm_script(Session(), 'println "hi"', "groovy")
    assert out["stdout"] == "" and "lumen-imaging[jvm]" in out["error"] and out["language"] == "groovy"
    with pytest.raises(RuntimeError, match="not available"):
        no_bridge.start()


def test_status_without_java(monkeypatch):
    monkeypatch.setattr(jb, "_importable", lambda name: name in jb.PACKAGES)
    monkeypatch.setattr(jb, "_java_available", lambda: (False, "no JDK found (JAVA_HOME unset, no java on PATH, no Homebrew openjdk)", None))
    jb._reset_state()
    try:
        st = jb.status()
        assert st["package"] and not st["java"] and not st["available"]
        assert "no JDK found" in st["detail"] and "Temurin" in st["detail"]
        assert jb.status() == st  # cached
        out = jb.run_jvm_script(Session(), "1", "beanshell")
        assert "BeanShell" in out["error"] and "JDK" in out["error"]
    finally:
        jb._reset_state()


def test_java_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(jb.shutil, "which", lambda name: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(jb, "HOMEBREW_GLOBS", (str(tmp_path / "opt" / "openjdk*" / "libexec" / "openjdk.jdk" / "Contents" / "Home"),))
    assert jb._java_available()[0] is False
    exe = "java.exe" if sys.platform == "win32" else "java"
    brew = tmp_path / "opt" / "openjdk@21" / "libexec" / "openjdk.jdk" / "Contents" / "Home"
    (brew / "bin").mkdir(parents=True)
    (brew / "bin" / exe).write_bytes(b"")
    ok, detail, home = jb._java_available()
    assert ok and "Homebrew" in detail and home == str(brew)
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    (jdk / "bin" / exe).write_bytes(b"")
    monkeypatch.setenv("JAVA_HOME", str(jdk))
    ok, detail, home = jb._java_available()
    assert ok and detail.startswith("JAVA_HOME=") and home == str(jdk)
    monkeypatch.delenv("JAVA_HOME")
    monkeypatch.setattr(jb.shutil, "which", lambda name: "/usr/bin/java")
    assert jb._java_available() == (True, "java at /usr/bin/java", None)


def test_language_names():
    assert jb.normalize_language("Groovy") == "groovy" and jb.normalize_language("bsh") == "beanshell" and jb.normalize_language("js-jvm") == "rhino"
    assert jb.normalize_language("JRuby") == "jruby" and jb.normalize_language("clj") == "clojure"
    # the browser's JavaScript, Python and Jython are not JVM languages
    assert jb.normalize_language("javascript") is None and jb.normalize_language("js") is None and jb.normalize_language("python") is None
    assert jb.normalize_language("jython") is None and jb.normalize_language(None) is None
    assert jb.is_jvm_language("rhino") and not jb.is_jvm_language("ijm")
    from lumen.core.ijshim import JVM_LANGUAGES, is_jvm_language
    assert list(JVM_LANGUAGES) == ["groovy", "beanshell", "rhino", "jruby", "clojure"] and is_jvm_language("BeanShell")


def test_enabled_languages_and_local_jars(monkeypatch, tmp_path):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    jb.reset()
    assert jb.enabled_languages() == ["groovy", "beanshell", "rhino"]
    monkeypatch.setenv("LUMEN_JVM_LANGUAGES", "jruby")
    assert jb.enabled_languages() == ["groovy", "beanshell", "rhino", "jruby"]
    monkeypatch.setenv("LUMEN_JVM_LANGUAGES", "all")
    assert jb.enabled_languages() == list(jb.LANGUAGES)
    monkeypatch.delenv("LUMEN_JVM_LANGUAGES")
    jars = tmp_path / "jars"
    jars.mkdir()
    for n in ("clojure-1.12.0.jar", "groovy-4.0.23.jar", "groovy-json-4.0.23.jar", "ij-1.54p.jar"):
        (jars / n).write_bytes(b"")
    monkeypatch.setenv("LUMEN_JVM_CLASSPATH", str(jars))  # a folder → its *.jar
    assert "clojure" in jb.enabled_languages() and "jruby" not in jb.enabled_languages()
    cp = jb.classpath_entries()
    assert [p.split("/")[-1].split("\\")[-1] for p in cp] == ["clojure-1.12.0.jar", "groovy-4.0.23.jar", "groovy-json-4.0.23.jar", "ij-1.54p.jar"]
    monkeypatch.setenv("LUMEN_JVM_CLASSPATH", str(jars / "groovy-*.jar"))  # globs
    assert len(jb.classpath_entries()) == 2
    monkeypatch.setenv("LUMEN_IJ_JAR", str(jars / "ij-1.54p.jar"))
    st = jb.status(refresh=True)
    assert st["ij"].endswith("ij-1.54p.jar") and st["classpath"][0].endswith("ij-1.54p.jar")
    jb.reset()


# ====================================================================== starting the (fake) JVM
def test_start_configures_scyjava_once(fake_jvm):
    jb.start()
    cfg = fake_jvm.config
    assert cfg.endpoints == [jb.IJ_ENDPOINT, "org.apache.groovy:groovy:4.0.23", "org.beanshell:bsh:2.1.1", "org.mozilla:rhino:1.7.14"]
    assert "-Djava.awt.headless=true" in cfg.options and "-Dapple.awt.UIElement=true" in cfg.options and cfg.classpath == []
    assert Interpreter.batchMode is True
    st = jb.status()
    assert st["started"] and "JVM running (groovy, beanshell, rhino, headless)" in st["detail"]
    jb.start()  # idempotent
    assert len(cfg.endpoints) == 4


def test_start_uses_local_jars_heap_and_extra_endpoints(fake_jvm, monkeypatch, tmp_path):
    jars = tmp_path / "jars"
    jars.mkdir()
    for n in ("ij-1.54p.jar", "bsh-2.1.1.jar"):
        (jars / n).write_bytes(b"")
    monkeypatch.setenv("LUMEN_IJ_JAR", str(jars / "ij-1.54p.jar"))
    monkeypatch.setenv("LUMEN_JVM_CLASSPATH", str(jars / "bsh-2.1.1.jar"))
    monkeypatch.setenv("LUMEN_JVM_LANGUAGES", "jruby")
    monkeypatch.setenv("LUMEN_JVM_HEAP", "2g")
    monkeypatch.setenv("LUMEN_JVM_HEADLESS", "0")
    monkeypatch.setenv("LUMEN_JVM_EXTRA_ENDPOINTS", "ome:formats-gpl:8.0.1")
    jb._reset_state()
    jb.start()
    cfg = fake_jvm.config
    assert cfg.classpath == [str(jars / "ij-1.54p.jar"), str(jars / "bsh-2.1.1.jar")]
    assert cfg.endpoints == ["org.apache.groovy:groovy:4.0.23", "org.mozilla:rhino:1.7.14", "org.jruby:jruby-complete:9.4.12.0", "ome:formats-gpl:8.0.1"]
    assert "-Xmx2g" in cfg.options and "-Djava.awt.headless=true" not in cfg.options
    assert jb._STATE.languages == ("groovy", "beanshell", "rhino", "jruby") and not jb._STATE.headless


def test_start_failure_is_reported_not_raised_to_scripts(monkeypatch):
    scy, jp = make_fake_modules(fail_start=True)
    monkeypatch.setitem(sys.modules, "scyjava", scy)
    monkeypatch.setitem(sys.modules, "jpype", jp)
    monkeypatch.setattr(jb, "_java_available", lambda: (True, "fake java", None))
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    jb._reset_state()
    try:
        out = jb.run_jvm_script(Session(), "1", "groovy")
        assert "could not be started" in out["error"] and "no libjvm" in out["error"] and "lumen-imaging[jvm]" in out["error"]
        assert "failed to start" in jb.status()["detail"] and not jb._STATE.started
    finally:
        jb._reset_state()


def test_jvm_started_by_another_bridge_without_ij(monkeypatch):
    scy, jp = make_fake_modules(no_ij=True)
    scy._started = True  # e.g. the Bio-Formats bridge started it first
    monkeypatch.setitem(sys.modules, "scyjava", scy)
    monkeypatch.setitem(sys.modules, "jpype", jp)
    monkeypatch.setattr(jb, "_java_available", lambda: (True, "fake java", None))
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    jb._reset_state()
    try:
        out = jb.run_jvm_script(Session(), "1", "groovy")
        assert "already started by another bridge" in out["error"] and scy.config.endpoints == []
    finally:
        jb._reset_state()


# ====================================================================== conversions
def _cal() -> Calibration:
    return Calibration(pixel_width=0.325, pixel_height=0.325, pixel_depth=1.5, unit="µm", frame_interval=2.5, time_unit="min", value_unit="counts")


@pytest.mark.parametrize("dtype,rgb", [(np.uint8, False), (np.uint16, False), (np.float32, False), (np.uint8, True)])
def test_roundtrip_every_pixel_type(fake_jvm, dtype, rgb):
    jb.start()
    rng = np.random.default_rng(1)
    if rgb:
        data = rng.integers(0, 256, (1, 2, 3, 5, 7), dtype=np.uint8)
    elif dtype == np.float32:
        data = rng.standard_normal((1, 2, 1, 5, 7)).astype(np.float32) * 1e3
    else:
        data = rng.integers(0, np.iinfo(dtype).max + 1, (1, 2, 1, 5, 7), dtype=dtype)
    img = LumenImage(data, name="pic", axes="TZCYX", rgb=rgb, calibration=_cal())
    imp = jb.to_imageplus(img)
    assert imp.getTitle() == "pic" and imp.getWidth() == 7 and imp.getHeight() == 5 and imp.getStackSize() == 2
    assert imp.getBitDepth() == {np.uint8: 8, np.uint16: 16, np.float32: 32}[dtype] if not rgb else imp.getBitDepth() == 24
    assert (imp.getNChannels(), imp.getNSlices(), imp.getNFrames()) == (1, 2, 1)
    back = jb.from_imageplus(imp)
    assert back.data.dtype == data.dtype and back.data.shape == data.shape and np.array_equal(back.data, data)
    assert back.rgb == rgb and back.name == "pic"
    assert back.calibration.to_dict() == _cal().to_dict()


def test_roundtrip_two_channel_z_stack_with_calibration_and_position(fake_jvm):
    jb.start()
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(1, 3, 2, 4, 5) * 7
    img = LumenImage(data, name="cz", axes="TZCYX", calibration=_cal())
    img.display.z, img.display.c = 2, 1
    img.display.channels[1].min, img.display.channels[1].max = 10.0, 500.0
    imp = jb.to_imageplus(img)
    assert (imp.getNChannels(), imp.getNSlices(), imp.getNFrames()) == (2, 3, 1) and imp._hyper
    # channels vary fastest, like ImageJ: slice n = (z * nc + c) + 1
    st = imp.getStack()
    assert np.array_equal(jb._nparray(st.getProcessor(1).getPixels(), np.uint16).reshape(4, 5), data[0, 0, 0])
    assert np.array_equal(jb._nparray(st.getProcessor(2).getPixels(), np.uint16).reshape(4, 5), data[0, 0, 1])
    assert np.array_equal(jb._nparray(st.getProcessor(6).getPixels(), np.uint16).reshape(4, 5), data[0, 2, 1])
    assert imp.getCurrentSlice() == 6 and imp.getDisplayRangeMin() == 10.0 and imp.getDisplayRangeMax() == 500.0
    jc = imp.getCalibration()
    assert (jc.pixelWidth, jc.pixelDepth, jc.getUnit(), jc.frameInterval, jc.getTimeUnit(), jc.getValueUnit()) == (0.325, 1.5, "µm", 2.5, "min", "counts")
    back = jb.from_imageplus(imp)
    assert np.array_equal(back.data, data) and back.data.dtype == np.uint16 and not back.rgb
    assert back.calibration.to_dict() == _cal().to_dict()
    assert (back.display.t, back.display.z, back.display.c) == (0, 2, 1)
    # a time series as well (T=2, Z=2, C=2): frame-major order survives
    data = np.arange(2 * 2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 2, 3, 3)
    imp = jb.to_imageplus(LumenImage(data, name="tzc", axes="TZCYX"))
    assert (imp.getNChannels(), imp.getNSlices(), imp.getNFrames()) == (2, 2, 2)
    assert np.array_equal(jb.from_imageplus(imp).data, data)
    # ImageJ's default value unit is not carried into Lumen
    plain = jb.to_imageplus(LumenImage(np.zeros((3, 3), np.uint8), name="p"))
    assert jb.from_imageplus(plain).calibration.to_dict() == Calibration().to_dict()


def test_roi_conversion_both_ways(fake_jvm):
    jb.start()
    rect = jb._roi_to_java({"type": "rect", "x": 1, "y": 2, "w": 5, "h": 6, "name": "box"})
    assert isinstance(rect, Roi) and rect.getType() == 0 and jb._roi_from_java(rect) == {"type": "rect", "x": 1.0, "y": 2.0, "w": 5.0, "h": 6.0, "name": "box"}
    oval = jb._roi_to_java({"type": "oval", "x": 0, "y": 0, "w": 4, "h": 4})
    assert isinstance(oval, OvalRoi) and jb._roi_from_java(oval)["type"] == "oval"
    poly = jb._roi_to_java({"type": "polygon", "points": [[0, 0], [4, 0], [4, 3]]})
    assert isinstance(poly, PolygonRoi) and poly.getType() == 2 and jb._roi_from_java(poly) == {"type": "polygon", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]]}
    assert jb._roi_from_java(jb._roi_to_java({"type": "polyline", "points": [[0, 0], [4, 1]]}))["type"] == "polyline"
    assert jb._roi_from_java(jb._roi_to_java({"type": "point", "points": [[1, 1], [2, 2]]})) == {"type": "point", "points": [[1.0, 1.0], [2.0, 2.0]]}
    line = jb._roi_to_java({"type": "line", "x1": 0, "y1": 0, "x2": 6, "y2": 8, "width": 3})
    assert isinstance(line, Line) and jb._roi_from_java(line) == {"type": "line", "x1": 0.0, "y1": 0.0, "x2": 6.0, "y2": 8.0, "width": 3}
    assert jb._roi_to_java({"type": "mask", "rle": [1], "x": 0, "y": 0, "w": 1, "h": 1}) is None and jb._roi_to_java(None) is None
    assert jb._roi_from_java(None) is None


# ====================================================================== running scripts (the fake engines execute Python)
GROOVY = '''#@ Integer (label="Count", value=3) n
#@ String (value="hi") greeting
#@ Boolean flag
IJ.log("n=" + str(n) + " " + greeting + " " + str(flag) + " " + imp.getTitle())
println("roi", imp.getRoi().getBounds().width)
imp.getProcessor().invert()
rt = ResultsTable.getResultsTable()
rt.incrementCounter(); rt.addValue("Area", 12.5); rt.addLabel("blob")
rt.incrementCounter(); rt.addValue("Area", 3); rt.addValue("Name", "second")
imp.setRoi(OvalRoi(1, 2, 3, 4))
'''


def test_groovy_run_binds_parameters_image_and_copies_everything_back(fake_jvm):
    s, img = blobs_session()
    before = img.data.copy()
    out = jb.run_jvm_script(s, GROOVY, "groovy", params={"n": 7}, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 4})
    assert out["error"] is None, out["error"]
    assert out["stdout"] == "n=7 hi False blobs\nroi 5\n"
    assert out["language"] == "groovy" and GroovyShell.used == ["script.groovy"]
    assert [p["name"] for p in out["parameters"]] == ["n", "greeting", "flag"] and out["parameters"][0]["type"] == "int"
    # the output lines are in the Log section
    assert [e["msg"] for e in s.log if e["msg"].startswith(("n=", "roi"))] == ["n=7 hi False blobs", "roi 5"]
    # the image changed in place, with an undo entry
    assert out["changed"] and np.array_equal(img.data, 255 - before) and img.data.dtype == np.uint8
    assert [snap.label for snap in s.histories[img.id].undo] == ["Groovy script"]
    assert s.undo(img.id) == "Groovy script" and np.array_equal(img.data, before)
    # the script's final selection comes back; the Results table was copied
    assert out["roi"] == {"type": "oval", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    assert s.results.rows == [{"Label": "blob", "Area": 12.5}, {"Area": 3.0, "Name": "second"}] and s.results.columns == ["Label", "Area", "Name"]
    assert out["results_rows"] == 2 and out["new_images"] == []
    # headless JVM: no ROI Manager, said so
    assert out["roi_manager"] is False and "ROI Manager" in out["note"] and "LUMEN_JVM_HEADLESS=0" in out["note"]
    # the batch-mode image list and the temp current image are cleaned up
    assert Interpreter._images == [] and WindowManager.getCurrentImage() is None
    # the #@ header lines were blanked, not removed: line numbers are preserved
    assert jb._strip_header("#@ int n\nx = 1\n", [0]) == "\nx = 1\n"


def test_script_without_changes_leaves_undo_results_and_rois_alone(fake_jvm):
    s, img = blobs_session()
    s.results.append([{"Area": 1.0, "Label": "a", "Name": "x"}])
    s.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 2, "h": 2})
    out = jb.run_jvm_script(s, 'println("v", ResultsTable.getResultsTable().getValue("Area", 0), ResultsTable.getResultsTable().getLabel(0))', "groovy")
    assert out["error"] is None and out["stdout"] == "v 1.0 a\n"
    assert not out["changed"] and s.histories[img.id].undo == [] and s.results.rows == [{"Area": 1.0, "Label": "a", "Name": "x"}]
    assert len(s.roi_manager.rois) == 1 and out["roi"] is None
    # a reset of the Results table in the script clears the session's
    out = jb.run_jvm_script(s, "ResultsTable.getResultsTable().reset()", "groovy")
    assert out["error"] is None and s.results.rows == [] and s.results.columns == []


def test_script_creates_images_renames_and_recalibrates(fake_jvm):
    s, img = blobs_session()
    code = '''new = ImagePlus("made", ShortProcessor(4, 3))
new.getProcessor().set(300)
new.show()
rgb = ImagePlus("colour", ColorProcessor(2, 2)); rgb.getProcessor().set(0x10FF20); rgb.show()
imp.setTitle("renamed")
imp.getCalibration().pixelWidth = 0.5
imp.getCalibration().setUnit("mm")
'''
    out = jb.run_jvm_script(s, code, "groovy")
    assert out["error"] is None, out["error"]
    assert len(out["new_images"]) == 2 and [s.images[i].name for i in out["new_images"]] == ["made", "colour"]
    made = s.images[out["new_images"][0]]
    assert made.data.shape == (1, 1, 1, 3, 4) and made.data.dtype == np.uint16 and int(made.data.max()) == 300
    colour = s.images[out["new_images"][1]]
    assert colour.rgb and colour.data.shape == (1, 1, 3, 2, 2) and colour.data[0, 0, :, 0, 0].tolist() == [0x10, 0xFF, 0x20]
    assert img.name == "renamed" and img.calibration.unit == "mm" and img.calibration.pixel_width == 0.5
    assert not out["changed"] and [sn.label for sn in s.histories[img.id].undo] == ["Groovy script"]
    assert s.undo(img.id) == "Groovy script" and img.name == "blobs" and img.calibration.unit == "pixel"
    assert Interpreter._images == []


def test_roi_manager_exchange_on_a_non_headless_jvm(fake_jvm):
    RoiManager.headless = False
    s, img = blobs_session()
    first = s.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 2, "h": 2}, image_id=img.id, position={"t": 0, "z": 0, "c": 0})
    code = '''rm = RoiManager.getRoiManager()
println(rm.getCount(), rm.getName(0))
o = OvalRoi(0, 0, 4, 4); o.setName("circle"); o.setPosition(1, 2, 1)
rm.addRoi(o)
'''
    out = jb.run_jvm_script(s, code, "groovy")
    assert out["error"] is None, out["error"]
    assert out["stdout"] == f"1 {first['name']}\n" and out["roi_manager"] is True and out["rois_copied"] == 2 and "note" not in out
    rois = s.roi_manager.rois
    assert len(rois) == 2 and rois[0]["id"] == first["id"] and rois[0]["image_id"] == img.id
    assert rois[1]["type"] == "oval" and rois[1]["name"] == "circle" and rois[1]["image_id"] == img.id and rois[1]["position"] == {"c": 0, "z": 1, "t": 0}
    # unchanged list: nothing is replaced
    out = jb.run_jvm_script(s, "println(RoiManager.getRoiManager().getCount())", "groovy")
    assert out["stdout"] == "2\n" and out["rois_copied"] == 0 and s.roi_manager.rois[1]["id"] == rois[1]["id"]
    # a #@ RoiManager / ResultsTable / ImagePlus / File parameter is supplied automatically
    code = '#@ RoiManager rm\n#@ ResultsTable table\n#@ ImagePlus image\n#@ File (style="directory") folder\nprintln(rm.getCount(), table.size(), image.getTitle(), folder.getPath())'
    out = jb.run_jvm_script(s, code, "groovy", params={"folder": "/data/out"})
    assert out["error"] is None and out["stdout"] == "2 0 blobs /data/out\n"


def test_beanshell_rhino_and_jruby_dispatch(fake_jvm, monkeypatch):
    monkeypatch.setenv("LUMEN_JVM_LANGUAGES", "jruby")
    jb._reset_state()
    s, img = blobs_session()
    out = jb.run_jvm_script(s, 'print("bsh " + imp.getTitle())', "beanshell")
    assert out["error"] is None and out["stdout"] == "bsh blobs\n"
    assert BshInterpreter.last.preludes == [jb.BSH_PRELUDE] and "import ij.*;" in jb.BSH_PRELUDE and "IJ" not in BshInterpreter.last._vars
    out = jb.run_jvm_script(s, 'IJ.log("rhino " + str(WindowManager.getCurrentImage().getTitle()))', "rhino")
    assert out["error"] is None and out["stdout"] == "rhino blobs\n"
    assert ImporterTopLevel.last.prelude == jb.RHINO_PRELUDE and "importPackage(Packages.ij.plugin.frame);" in jb.RHINO_PRELUDE
    assert ImporterTopLevel.last.vars["IJ"] is IJ and RhinoContext.depth == 0
    out = jb.run_jvm_script(s, 'println("ruby " + imp.getTitle())', "jruby")
    assert out["error"] is None and out["stdout"] == "ruby blobs\n" and ScriptingContainer.used == 1
    # Clojure is not enabled: a clear message, no attempt
    out = jb.run_jvm_script(s, "(println 1)", "clojure")
    assert "LUMEN_JVM_LANGUAGES=clojure" in out["error"]
    # aliases and odd capitalisation
    assert jb.run_jvm_script(s, 'println("x")', "BSH")["stdout"] == "x\n"


def test_optional_language_needs_the_env_var_before_the_jvm_starts(fake_jvm, monkeypatch):
    s, _ = blobs_session()
    out = jb.run_jvm_script(s, "1", "jruby")
    assert "optional" in out["error"] and "LUMEN_JVM_LANGUAGES=jruby" in out["error"] and not jb._STATE.started
    jb.run_jvm_script(s, 'println("go")', "groovy")  # starts the JVM with the core languages only
    monkeypatch.setenv("LUMEN_JVM_LANGUAGES", "jruby")
    jb.reset()
    out = jb.run_jvm_script(s, "1", "jruby")
    assert "started without JRuby" in out["error"] and "restart" in out["error"]


def test_script_errors_are_reported_with_the_output_so_far(fake_jvm):
    s, img = blobs_session()
    before = img.data.copy()
    out = jb.run_jvm_script(s, 'println("before")\nraise ValueError("boom at line 2")\nimp.getProcessor().invert()', "groovy")
    assert out["stdout"] == "before\n" and "ValueError: boom at line 2" in out["error"]
    assert np.array_equal(img.data, before) and s.histories[img.id].undo == [] and not out["changed"]
    assert Interpreter._images == [] and System.out.target is None  # streams restored
    assert jb.run_jvm_script(s, "1", "cobol")["error"].startswith("Unknown JVM script language 'cobol'")
    assert jb.run_jvm_script(Session(), 'println("no image")', "groovy") == {"stdout": "no image\n", "error": None, "language": "groovy", "mode": "inprocess", "changed": False,
                                                                            "new_images": [], "results_rows": 0, "roi_manager": False, "rois_copied": 0,
                                                                            "note": jb.run_jvm_script(Session(), "1", "groovy")["note"]}


# ====================================================================== the op and the endpoint
def test_op_and_endpoint_degrade_gracefully_without_the_bridge(no_bridge):
    s, _ = blobs_session()
    with pytest.raises(RuntimeError, match=r"Groovy scripts need the Java bridge.*lumen-imaging\[jvm\]"):
        s.run_op("script.run_jvm_script", {"code": 'println "hi"', "language": "groovy"})
    res = s.run_op("script.jvm_bridge_status", {})
    table = res["table"]
    assert table["rows"][0][0] == "jpype1 + scyjava" and "missing" in table["rows"][0][1] and "not available" in res["message"]
    assert any(r[0] == "JRuby (jruby)" and "LUMEN_JVM_LANGUAGES=jruby" in r[1] for r in table["rows"])
    c = TestClient(create_app(s))
    r = c.post("/api/script/run", json={"code": 'println "hi"', "language": "rhino"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["stdout"] == "" and "lumen-imaging[jvm]" in j["error"] and "JavaScript (Rhino)" in j["error"] and j["language"] == "rhino"
    # Python / Jython keep their runner
    r = c.post("/api/script/run", json={"code": "print(len(lumen.images))", "language": "python"})
    assert r.json() == {"stdout": "1\n", "error": None}


def test_op_runs_through_the_fake_jvm(fake_jvm):
    s, img = blobs_session()
    res = s.run_op("script.run_jvm_script", {"code": 'IJ.run(imp, "Invert", "")\nIJ.log("done")\nimp.setRoi(Roi(0, 0, 2, 2))', "language": "groovy"},
                   image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 3, "h": 3})
    assert res["message"] == "Groovy script finished – image changed" and res["log"] == "done\n"
    assert res["roi"]["type"] == "rect" and res["roi"]["w"] == 2.0 and IJ.runs[0][1] == "Invert"
    res = s.run_op("script.run_jvm_script", {"code": "imp.deleteRoi()", "language": "beanshell"}, image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 3, "h": 3})
    assert res.get("clear_roi") is True and res["message"] == "BeanShell script finished"
    with pytest.raises(RuntimeError, match="boom"):
        s.run_op("script.run_jvm_script", {"code": 'println("partial")\nraise RuntimeError("boom")', "language": "groovy"})
    st = s.run_op("script.jvm_bridge_status", {})
    assert st["table"]["rows"][3] == ["JVM", "running"] and "Java bridge available" in st["message"]


def test_endpoint_runs_groovy_through_the_fake_jvm(fake_jvm):
    s, img = blobs_session()
    c = TestClient(create_app(s))
    code = '#@ Double (value=1.5) sigma\nIJ.log("sigma " + str(sigma))\nimp.getProcessor().invert()\nImagePlus("extra", ByteProcessor(2, 2)).show()\nimp.setRoi(Roi(1, 1, 2, 2))'
    r = c.post("/api/script/run", json={"code": code, "params": {"sigma": 2.25}, "roi": {"type": "rect", "x": 0, "y": 0, "w": 4, "h": 4}, "language": "Groovy"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["error"] is None and j["stdout"] == "sigma 2.25\n" and j["language"] == "groovy" and j["changed"] is True
    assert j["roi"] == {"type": "rect", "x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0} and j["parameters"][0]["type"] == "float"
    assert len(j["new_images"]) == 1 and s.images[j["new_images"][0]].name == "extra"
    st = c.get("/api/state").json()
    assert st["history"][img.id]["undo"] == ["Groovy script"]
