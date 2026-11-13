"""The slice of ``java.*`` / ``jarray`` that Fiji scripts reach for: ``System``, ``Math``, the boxed numbers,
``String.format``, ``java.io.File`` (and the buffered writer / reader trio), ``java.util`` collections,
``java.awt`` geometry and ``Color``.  Everything is plain Python underneath (a ``File`` is also an ``os.PathLike``,
an ``ArrayList`` *is* a list, a ``HashMap`` *is* a dict)."""
from __future__ import annotations

import math as _math
import os
import platform
import random as _random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


# ====================================================================== java.lang
class _PrintStream:
    def __init__(self, err: bool = False) -> None:
        self._err = err

    def println(self, *args: Any) -> None:
        from ._context import current
        current().emit(" ".join(_jstr(a) for a in args))

    def print(self, *args: Any) -> None:  # noqa: A003
        self.println(*args)

    def printf(self, fmt: Any, *args: Any) -> None:
        self.println(String.format(fmt, *args))

    def flush(self) -> None:
        pass


def _jstr(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return f"{v:.1f}"
    return str(v)


class System:
    out = _PrintStream()
    err = _PrintStream(err=True)

    @staticmethod
    def currentTimeMillis() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def nanoTime() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def getProperty(key: Any, default: Any = None) -> Any:
        k = str(key)
        props = {"user.home": str(Path.home()), "user.dir": os.getcwd(), "user.name": os.environ.get("USER") or os.environ.get("USERNAME", ""),
                 "os.name": {"Darwin": "Mac OS X", "Windows": "Windows", "Linux": "Linux"}.get(platform.system(), platform.system()),
                 "os.arch": platform.machine(), "os.version": platform.release(), "file.separator": os.sep, "path.separator": os.pathsep,
                 "line.separator": os.linesep, "java.version": "1.8.0-lumen", "java.vendor": "Lumen (no JVM)", "java.io.tmpdir": tempfile.gettempdir(),
                 "ij.dir": str(_ij_dir()), "plugins.dir": str(_ij_dir()), "python.version": sys.version.split()[0]}
        return props.get(k, _SYSPROPS.get(k, default))

    @staticmethod
    def setProperty(key: Any, value: Any) -> Any:
        old = _SYSPROPS.get(str(key))
        _SYSPROPS[str(key)] = str(value)
        return old

    @staticmethod
    def getenv(key: Any = None) -> Any:
        return dict(os.environ) if key is None else os.environ.get(str(key))

    @staticmethod
    def lineSeparator() -> str:
        return os.linesep

    @staticmethod
    def exit(code: Any = 0) -> None:  # noqa: A003
        raise SystemExit(code)

    @staticmethod
    def gc() -> None:
        import gc
        gc.collect()

    @staticmethod
    def arraycopy(src: Any, src_pos: Any, dst: Any, dst_pos: Any, length: Any) -> None:
        s, d, n = int(src_pos), int(dst_pos), int(length)
        dst[d:d + n] = list(src[s:s + n])

    @staticmethod
    def identityHashCode(obj: Any) -> int:
        return id(obj) & 0x7FFFFFFF


_SYSPROPS: dict[str, str] = {}


def _ij_dir() -> Path:
    from ..ops import lumen_home
    return lumen_home()


class Math:
    PI = _math.pi
    E = _math.e

    @staticmethod
    def abs(x: Any) -> Any:  # noqa: A003
        return abs(x)

    @staticmethod
    def sqrt(x: Any) -> float:
        return _math.sqrt(x) if x >= 0 else float("nan")

    @staticmethod
    def cbrt(x: Any) -> float:
        return _math.copysign(abs(x) ** (1.0 / 3.0), x)

    @staticmethod
    def pow(a: Any, b: Any) -> float:  # noqa: A003
        try:
            return float(a) ** float(b)
        except (OverflowError, ZeroDivisionError):
            return float("inf")
        except ValueError:
            return float("nan")

    @staticmethod
    def round(x: Any) -> int:  # noqa: A003
        return int(_math.floor(float(x) + 0.5))

    @staticmethod
    def rint(x: Any) -> float:
        return float(round(float(x)))

    @staticmethod
    def floor(x: Any) -> float:
        return float(_math.floor(x))

    @staticmethod
    def ceil(x: Any) -> float:
        return float(_math.ceil(x))

    @staticmethod
    def max(a: Any, b: Any) -> Any:  # noqa: A003
        return a if a >= b else b

    @staticmethod
    def min(a: Any, b: Any) -> Any:  # noqa: A003
        return a if a <= b else b

    @staticmethod
    def random() -> float:
        return _random.random()

    @staticmethod
    def signum(x: Any) -> float:
        return float((x > 0) - (x < 0))

    @staticmethod
    def hypot(a: Any, b: Any) -> float:
        return _math.hypot(a, b)

    @staticmethod
    def toRadians(deg: Any) -> float:
        return _math.radians(deg)

    @staticmethod
    def toDegrees(rad: Any) -> float:
        return _math.degrees(rad)

    @staticmethod
    def log(x: Any) -> float:
        return _math.log(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))

    @staticmethod
    def log10(x: Any) -> float:
        return _math.log10(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))

    @staticmethod
    def log1p(x: Any) -> float:
        return _math.log1p(x)

    @staticmethod
    def exp(x: Any) -> float:
        try:
            return _math.exp(x)
        except OverflowError:
            return float("inf")

    @staticmethod
    def expm1(x: Any) -> float:
        return _math.expm1(x)

    @staticmethod
    def floorDiv(a: Any, b: Any) -> int:
        return int(a) // int(b)

    @staticmethod
    def floorMod(a: Any, b: Any) -> int:
        return int(a) % int(b)


for _name in ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh"):
    setattr(Math, _name, staticmethod(getattr(_math, _name)))


class _Boxed:
    """``Integer(…)`` / ``Double(…)`` construct plain numbers; the statics live on the class."""

    _py: Any = int

    def __new__(cls, value: Any = 0) -> Any:  # type: ignore[misc]
        return cls._py(value)

    @classmethod
    def valueOf(cls, v: Any) -> Any:
        return cls._py(v)

    @classmethod
    def toString(cls, v: Any = None) -> str:
        return _jstr(cls._py(v)) if v is not None else cls.__name__

    @staticmethod
    def compare(a: Any, b: Any) -> int:
        return (a > b) - (a < b)


class Integer(_Boxed):
    _py = int
    MAX_VALUE = 2147483647
    MIN_VALUE = -2147483648
    SIZE = 32

    @staticmethod
    def parseInt(s: Any, radix: Any = 10) -> int:
        return int(str(s).strip(), int(radix))

    @staticmethod
    def toHexString(v: Any) -> str:
        return format(int(v) & 0xFFFFFFFF, "x")

    @staticmethod
    def toBinaryString(v: Any) -> str:
        return format(int(v) & 0xFFFFFFFF, "b")


class Long(Integer):
    MAX_VALUE = 9223372036854775807
    MIN_VALUE = -9223372036854775808
    SIZE = 64

    @staticmethod
    def parseLong(s: Any, radix: Any = 10) -> int:
        return int(str(s).strip(), int(radix))


class Short(Integer):
    MAX_VALUE = 32767
    MIN_VALUE = -32768


class Byte(Integer):
    MAX_VALUE = 127
    MIN_VALUE = -128


class Double(_Boxed):
    _py = float
    MAX_VALUE = sys.float_info.max
    MIN_VALUE = 4.9e-324
    NaN = float("nan")
    POSITIVE_INFINITY = float("inf")
    NEGATIVE_INFINITY = float("-inf")

    @staticmethod
    def parseDouble(s: Any) -> float:
        return float(str(s).strip())

    @staticmethod
    def isNaN(v: Any) -> bool:
        return _math.isnan(float(v))

    @staticmethod
    def isInfinite(v: Any) -> bool:
        return _math.isinf(float(v))

    @staticmethod
    def isFinite(v: Any) -> bool:
        return _math.isfinite(float(v))


class Float(Double):
    MAX_VALUE = 3.4028235e38
    MIN_VALUE = 1.4e-45

    @staticmethod
    def parseFloat(s: Any) -> float:
        return float(str(s).strip())


class Boolean(_Boxed):
    _py = bool
    TRUE = True
    FALSE = False

    @staticmethod
    def parseBoolean(s: Any) -> bool:
        return str(s).strip().lower() == "true"


class Character:
    @staticmethod
    def isDigit(c: Any) -> bool:
        return str(c).isdigit()

    @staticmethod
    def isLetter(c: Any) -> bool:
        return str(c).isalpha()

    @staticmethod
    def isLetterOrDigit(c: Any) -> bool:
        return str(c).isalnum()

    @staticmethod
    def isWhitespace(c: Any) -> bool:
        return str(c).isspace()

    @staticmethod
    def toUpperCase(c: Any) -> str:
        return str(c).upper()

    @staticmethod
    def toLowerCase(c: Any) -> str:
        return str(c).lower()


_JFMT = re.compile(r"%(\d+\$)?([-#+ 0,(]*)(\d+)?(\.\d+)?([a-zA-Z%])")


def _java_format(fmt: Any, args: tuple[Any, ...]) -> str:
    """``String.format`` – the usual conversions (%d %f %s %e %x %c %b %n %%); ``%,d`` grouping is honoured."""
    out: list[str] = []
    pos = 0
    i = 0
    text = str(fmt)
    for m in _JFMT.finditer(text):
        out.append(text[pos:m.start()])
        pos = m.end()
        idx, flags, width, prec, conv = m.groups()
        if conv == "n":
            out.append("\n")
            continue
        if conv == "%":
            out.append("%")
            continue
        if idx:
            arg = args[int(idx[:-1]) - 1]
        else:
            arg = args[i] if i < len(args) else ""
            i += 1
        c = conv.lower()
        if c == "b":
            s = "true" if arg else "false"
            conv = "s"
        elif c == "d":
            try:
                arg = int(arg)
            except (TypeError, ValueError):
                arg = 0
            s = format(arg, ("," if "," in flags else "") + "d")
        elif c in ("f", "e", "g"):
            try:
                arg = float(arg)
            except (TypeError, ValueError):
                arg = float("nan")
            p = int(prec[1:]) if prec else 6
            s = format(arg, ("," if "," in flags else "") + f".{p}{c}")
        elif c == "x":
            s = format(int(arg), "x")
        elif c == "c":
            s = chr(int(arg)) if isinstance(arg, int) else str(arg)[:1]
        else:
            s = _jstr(arg)
            if prec:
                s = s[: int(prec[1:])]
        if conv.isupper():
            s = s.upper()
        if width:
            w = int(width)
            s = s.ljust(w) if "-" in flags else (s.rjust(w, "0") if "0" in flags and c in "dfegx" else s.rjust(w))
        out.append(s)
    out.append(text[pos:])
    return "".join(out)


class String(str):
    """``String.valueOf`` / ``String.format`` / ``String.join``; ``String(x)`` is ``str(x)``."""

    def __new__(cls, value: Any = "") -> str:  # type: ignore[misc]
        return str(value)

    @staticmethod
    def valueOf(v: Any) -> str:
        return _jstr(v)

    @staticmethod
    def format(fmt: Any, *args: Any) -> str:  # noqa: A003
        return _java_format(fmt, args)

    @staticmethod
    def join(sep: Any, *parts: Any) -> str:
        items: Iterable[Any] = parts[0] if len(parts) == 1 and isinstance(parts[0], (list, tuple)) else parts
        return str(sep).join(str(p) for p in items)


class StringBuilder:
    def __init__(self, initial: Any = "") -> None:
        self._parts: list[str] = [] if isinstance(initial, int) else [str(initial)]

    def append(self, v: Any) -> "StringBuilder":
        self._parts.append(_jstr(v))
        return self

    def insert(self, i: Any, v: Any) -> "StringBuilder":
        s = str(self)
        self._parts = [s[: int(i)] + _jstr(v) + s[int(i):]]
        return self

    def length(self) -> int:
        return len(str(self))

    def setLength(self, n: Any) -> None:
        self._parts = [str(self)[: int(n)]]

    def reverse(self) -> "StringBuilder":
        self._parts = [str(self)[::-1]]
        return self

    def toString(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "".join(self._parts)


StringBuffer = StringBuilder


class Thread:
    @staticmethod
    def sleep(ms: Any) -> None:
        time.sleep(min(max(float(ms), 0.0), 2000.0) / 1000.0)  # capped: the server thread must not stall

    @staticmethod
    def currentThread() -> "Thread":
        return Thread()

    def getName(self) -> str:
        return "main"

    def interrupt(self) -> None:
        pass

    def isInterrupted(self) -> bool:
        return False


class Runtime:
    @staticmethod
    def getRuntime() -> "Runtime":
        return Runtime()

    def freeMemory(self) -> int:
        from ._context import current
        return int(current().ij.freeMemory().split()[0]) * 1024 * 1024 if current().ij.freeMemory().split()[0].isdigit() else 0

    def totalMemory(self) -> int:
        from ._context import current
        return int(current().ij.currentMemory())

    def maxMemory(self) -> int:
        from ._context import current
        return int(current().ij.maxMemory())

    def availableProcessors(self) -> int:
        return os.cpu_count() or 1

    def gc(self) -> None:
        System.gc()


class Object:
    def toString(self) -> str:
        return str(self)

    def equals(self, other: Any) -> bool:
        return self == other

    def hashCode(self) -> int:
        return hash(self) & 0x7FFFFFFF

    def getClass(self) -> type:
        return type(self)


class JavaException(Exception):
    def getMessage(self) -> str:
        return str(self)

    def printStackTrace(self) -> None:
        from ._context import current
        current().emit(f"{type(self).__name__}: {self}")

    def toString(self) -> str:
        return f"{type(self).__name__}: {self}"


class RuntimeException(JavaException):
    pass


class IllegalArgumentException(RuntimeException):
    pass


class IOException(JavaException):
    pass


class NullPointerException(RuntimeException):
    pass


# ====================================================================== java.io
class File(os.PathLike):
    """``java.io.File``: a path with Java's accessor names; also a ``PathLike`` (``open(f)``, ``os.path.join(f, …)``)."""

    separator = os.sep
    pathSeparator = os.pathsep
    separatorChar = os.sep

    def __init__(self, path: Any, name: Any = None) -> None:
        p = os.fspath(path) if not isinstance(path, str) else path
        self._p = os.path.join(p, str(name)) if name is not None else str(p)

    def __fspath__(self) -> str:
        return self._p

    def __str__(self) -> str:
        return self._p

    def __repr__(self) -> str:
        return f"File({self._p!r})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, File) and other._p == self._p

    def __hash__(self) -> int:
        return hash(self._p)

    def __lt__(self, other: "File") -> bool:
        return self._p < other._p

    def exists(self) -> bool:
        return os.path.exists(self._p)

    def isDirectory(self) -> bool:
        return os.path.isdir(self._p)

    def isFile(self) -> bool:
        return os.path.isfile(self._p)

    def isHidden(self) -> bool:
        return os.path.basename(self._p).startswith(".")

    def canRead(self) -> bool:
        return os.access(self._p, os.R_OK)

    def canWrite(self) -> bool:
        return os.access(self._p, os.W_OK)

    def getName(self) -> str:
        return os.path.basename(self._p.rstrip("/\\")) or self._p

    def getPath(self) -> str:
        return self._p

    def getAbsolutePath(self) -> str:
        return os.path.abspath(self._p)

    def getCanonicalPath(self) -> str:
        return os.path.realpath(self._p)

    def getAbsoluteFile(self) -> "File":
        return File(self.getAbsolutePath())

    def getParent(self) -> str | None:
        d = os.path.dirname(self._p.rstrip("/\\"))
        return d or None

    def getParentFile(self) -> "File | None":
        d = self.getParent()
        return File(d) if d else None

    def list(self) -> list[str] | None:  # noqa: A003
        return sorted(os.listdir(self._p)) if os.path.isdir(self._p) else None

    def listFiles(self, filter_: Any = None) -> list["File"] | None:
        if not os.path.isdir(self._p):
            return None
        files = [File(self._p, n) for n in sorted(os.listdir(self._p))]
        if filter_ is not None and callable(filter_):
            files = [f for f in files if filter_(f)]
        return files

    def mkdir(self) -> bool:
        try:
            os.mkdir(self._p)
            return True
        except OSError:
            return False

    def mkdirs(self) -> bool:
        try:
            os.makedirs(self._p, exist_ok=True)
            return True
        except OSError:
            return False

    def delete(self) -> bool:
        try:
            if os.path.isdir(self._p):
                os.rmdir(self._p)
            else:
                os.remove(self._p)
            return True
        except OSError:
            return False

    def createNewFile(self) -> bool:
        if os.path.exists(self._p):
            return False
        Path(self._p).touch()
        return True

    def renameTo(self, dest: Any) -> bool:
        try:
            os.rename(self._p, os.fspath(dest))
            return True
        except OSError:
            return False

    def length(self) -> int:
        return os.path.getsize(self._p) if os.path.exists(self._p) else 0

    def lastModified(self) -> int:
        return int(os.path.getmtime(self._p) * 1000) if os.path.exists(self._p) else 0

    def toString(self) -> str:
        return self._p

    def toPath(self) -> Path:
        return Path(self._p)

    def toURI(self) -> str:
        return Path(self._p).absolute().as_uri()

    @staticmethod
    def createTempFile(prefix: Any, suffix: Any = None, directory: Any = None) -> "File":
        fd, p = tempfile.mkstemp(prefix=str(prefix), suffix=str(suffix or ".tmp"), dir=os.fspath(directory) if directory else None)
        os.close(fd)
        return File(p)


class FileWriter:
    def __init__(self, path: Any, append: Any = False) -> None:
        self._fh = open(os.fspath(path), "a" if append else "w", encoding="utf-8", newline="")

    def write(self, s: Any) -> None:
        self._fh.write(_jstr(s) if not isinstance(s, str) else s)

    def append(self, s: Any) -> "FileWriter":
        self.write(s)
        return self

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class BufferedWriter:
    def __init__(self, writer: Any) -> None:
        self._w = writer if hasattr(writer, "write") else FileWriter(writer)

    def write(self, s: Any) -> None:
        self._w.write(s)

    def newLine(self) -> None:
        self._w.write("\n")

    def append(self, s: Any) -> "BufferedWriter":
        self._w.write(s)
        return self

    def flush(self) -> None:
        self._w.flush()

    def close(self) -> None:
        self._w.close()


class PrintWriter(BufferedWriter):
    def __init__(self, target: Any, *rest: Any) -> None:
        super().__init__(target if hasattr(target, "write") else FileWriter(target))

    def println(self, s: Any = "") -> None:
        self._w.write(_jstr(s) + "\n")

    def print(self, s: Any) -> None:  # noqa: A003
        self._w.write(_jstr(s))

    def printf(self, fmt: Any, *args: Any) -> None:
        self._w.write(_java_format(fmt, args))


class FileReader:
    def __init__(self, path: Any) -> None:
        self._fh = open(os.fspath(path), encoding="utf-8")

    def readLine(self) -> str | None:
        line = self._fh.readline()
        return None if line == "" else line.rstrip("\r\n")

    def read(self) -> int:
        c = self._fh.read(1)
        return ord(c) if c else -1

    def ready(self) -> bool:
        return True

    def close(self) -> None:
        self._fh.close()

    def lines(self) -> list[str]:
        return [ln.rstrip("\r\n") for ln in self._fh]


class BufferedReader(FileReader):
    def __init__(self, reader: Any) -> None:  # noqa: D107
        if isinstance(reader, FileReader):
            self._fh = reader._fh
        else:
            self._fh = open(os.fspath(reader), encoding="utf-8")


class FileOutputStream(FileWriter):
    pass


class FileInputStream(FileReader):
    pass


# ====================================================================== java.util
class ArrayList(list):
    def __init__(self, items: Any = None) -> None:
        super().__init__([] if items is None or isinstance(items, int) else list(items))

    def add(self, *args: Any) -> bool:
        if len(args) == 2:
            self.insert(int(args[0]), args[1])
        else:
            self.append(args[0])
        return True

    def addAll(self, items: Any) -> bool:
        self.extend(list(items))
        return True

    def get(self, i: Any) -> Any:
        return self[int(i)]

    def set(self, i: Any, v: Any) -> Any:  # noqa: A003
        old = self[int(i)]
        self[int(i)] = v
        return old

    def size(self) -> int:
        return len(self)

    def isEmpty(self) -> bool:
        return not self

    def contains(self, v: Any) -> bool:
        return v in self

    def indexOf(self, v: Any) -> int:
        return self.index(v) if v in self else -1

    def remove(self, v: Any) -> Any:  # type: ignore[override]
        if isinstance(v, int) and not isinstance(v, bool):
            return self.pop(v)
        if v in self:
            super().remove(v)
            return True
        return False

    def toArray(self, *a: Any) -> list[Any]:
        return list(self)

    def iterator(self) -> Any:
        return iter(self)

    def toString(self) -> str:
        return "[" + ", ".join(_jstr(v) for v in self) + "]"


Vector = ArrayList
LinkedList = ArrayList


class HashMap(dict):
    def put(self, k: Any, v: Any) -> Any:
        old = self.get(k)
        self[k] = v
        return old

    def putAll(self, m: Any) -> None:
        self.update(dict(m))

    def containsKey(self, k: Any) -> bool:
        return k in self

    def containsValue(self, v: Any) -> bool:
        return v in self.values()

    def keySet(self) -> list[Any]:
        return list(self.keys())

    def entrySet(self) -> list[tuple[Any, Any]]:
        return list(self.items())

    def remove(self, k: Any) -> Any:  # type: ignore[override]
        return self.pop(k, None)

    def size(self) -> int:
        return len(self)

    def isEmpty(self) -> bool:
        return not self

    def getOrDefault(self, k: Any, d: Any) -> Any:
        return self.get(k, d)


Hashtable = HashMap
LinkedHashMap = HashMap
TreeMap = HashMap


class HashSet(set):
    def add(self, v: Any) -> bool:  # type: ignore[override]
        new = v not in self
        super().add(v)
        return new

    def contains(self, v: Any) -> bool:
        return v in self

    def size(self) -> int:
        return len(self)

    def isEmpty(self) -> bool:
        return not self

    def toArray(self, *a: Any) -> list[Any]:
        return list(self)


class Arrays:
    @staticmethod
    def asList(*items: Any) -> ArrayList:
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            return ArrayList(items[0])
        return ArrayList(items)

    @staticmethod
    def sort(arr: Any, *rest: Any) -> None:
        arr.sort()

    @staticmethod
    def fill(arr: Any, v: Any) -> None:
        for i in range(len(arr)):
            arr[i] = v

    @staticmethod
    def toString(arr: Any) -> str:
        return "[" + ", ".join(_jstr(v) for v in arr) + "]"

    @staticmethod
    def copyOf(arr: Any, n: Any) -> list[Any]:
        out = list(arr)[: int(n)]
        return out + [0] * (int(n) - len(out))

    @staticmethod
    def copyOfRange(arr: Any, a: Any, b: Any) -> list[Any]:
        return list(arr)[int(a): int(b)]

    @staticmethod
    def equals(a: Any, b: Any) -> bool:
        return list(a) == list(b)


class Collections:
    @staticmethod
    def sort(lst: Any, *rest: Any) -> None:
        lst.sort()

    @staticmethod
    def reverse(lst: Any) -> None:
        lst.reverse()

    @staticmethod
    def max(lst: Any) -> Any:  # noqa: A003
        return max(lst)

    @staticmethod
    def min(lst: Any) -> Any:  # noqa: A003
        return min(lst)

    @staticmethod
    def shuffle(lst: Any, *rest: Any) -> None:
        _random.shuffle(lst)


class Random:
    def __init__(self, seed: Any = None) -> None:
        self._r = _random.Random(seed)

    def setSeed(self, seed: Any) -> None:
        self._r.seed(seed)

    def nextInt(self, n: Any = None) -> int:
        return self._r.randrange(int(n)) if n is not None else self._r.randrange(-2147483648, 2147483648)

    def nextLong(self) -> int:
        return self._r.randrange(-9223372036854775808, 9223372036854775808)

    def nextDouble(self) -> float:
        return self._r.random()

    def nextFloat(self) -> float:
        return self._r.random()

    def nextGaussian(self) -> float:
        return self._r.gauss(0.0, 1.0)

    def nextBoolean(self) -> bool:
        return self._r.random() < 0.5


class Date:
    def __init__(self, ms: Any = None) -> None:
        self._t = time.time() if ms is None else float(ms) / 1000.0

    def getTime(self) -> int:
        return int(self._t * 1000)

    def toString(self) -> str:
        return time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime(self._t))

    def __str__(self) -> str:
        return self.toString()


class UUID:
    @staticmethod
    def randomUUID() -> str:
        import uuid
        return str(uuid.uuid4())


# ====================================================================== java.awt
class Color:
    """``java.awt.Color``: ``Color(r, g, b[, a])`` (0–255 ints or 0–1 floats), ``Color(rgb)``, the named constants
    and ``Color.decode("#ff8800")``.  ``str(color)`` is the ``#rrggbb`` hex Lumen's ROIs and overlays use."""

    _js_props = ("r", "g", "b", "a")

    def __init__(self, r: Any = 0, g: Any = None, b: Any = None, a: Any = None) -> None:
        if g is None and b is None:
            if isinstance(r, str):
                r2, g2, b2 = _parse_hex(r)
                self.r, self.g, self.b, self.a = r2, g2, b2, 255
            else:
                v = int(r)
                self.r, self.g, self.b, self.a = (v >> 16) & 255, (v >> 8) & 255, v & 255, 255
            return
        vals = [r, g, b] + ([a] if a is not None else [])
        if all(isinstance(v, float) for v in vals) and all(0.0 <= v <= 1.0 for v in vals):
            vals = [int(round(v * 255)) for v in vals]
        vals = [max(0, min(255, int(v))) for v in vals]
        self.r, self.g, self.b = vals[:3]
        self.a = vals[3] if len(vals) > 3 else 255

    def getRed(self) -> int:
        return self.r

    def getGreen(self) -> int:
        return self.g

    def getBlue(self) -> int:
        return self.b

    def getAlpha(self) -> int:
        return self.a

    def getRGB(self) -> int:
        v = (self.a << 24) | (self.r << 16) | (self.g << 8) | self.b
        return v - (1 << 32) if v >= (1 << 31) else v

    def brighter(self) -> "Color":
        return Color(*[min(255, int(c / 0.7)) if c else 3 for c in (self.r, self.g, self.b)], self.a)

    def darker(self) -> "Color":
        return Color(*[int(c * 0.7) for c in (self.r, self.g, self.b)], self.a)

    def toHex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def toString(self) -> str:
        return f"java.awt.Color[r={self.r},g={self.g},b={self.b}]"

    def equals(self, other: Any) -> bool:
        return self == other

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Color) and (self.r, self.g, self.b, self.a) == (other.r, other.g, other.b, other.a)

    def __hash__(self) -> int:
        return hash((self.r, self.g, self.b, self.a))

    def __str__(self) -> str:
        return self.toHex()

    def __repr__(self) -> str:
        return self.toString()

    @staticmethod
    def decode(s: Any) -> "Color":
        return Color(str(s))

    @staticmethod
    def getHSBColor(h: Any, s: Any, b: Any) -> "Color":
        import colorsys
        r, g, bl = colorsys.hsv_to_rgb(float(h) % 1.0, float(s), float(b))
        return Color(int(round(r * 255)), int(round(g * 255)), int(round(bl * 255)))

    @staticmethod
    def RGBtoHSB(r: Any, g: Any, b: Any, out: Any = None) -> list[float]:
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(int(r) / 255.0, int(g) / 255.0, int(b) / 255.0)
        return [h, s, v]


def _parse_hex(s: str) -> tuple[int, int, int]:
    t = s.strip().lstrip("#")
    if t.lower().startswith("0x"):
        t = t[2:]
    named = {"red": "ff0000", "green": "00ff00", "blue": "0000ff", "yellow": "ffff00", "cyan": "00ffff", "magenta": "ff00ff", "white": "ffffff",
             "black": "000000", "gray": "808080", "grey": "808080", "orange": "ffc800", "pink": "ffafaf", "lightgray": "c0c0c0", "darkgray": "404040"}
    t = named.get(t.lower(), t)
    if len(t) == 8:
        t = t[2:]
    if len(t) != 6:
        raise ValueError(f"Color.decode: cannot parse {s!r}")
    return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)


for _n, _rgb in (("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255)), ("yellow", (255, 255, 0)), ("cyan", (0, 255, 255)),
                 ("magenta", (255, 0, 255)), ("white", (255, 255, 255)), ("black", (0, 0, 0)), ("gray", (128, 128, 128)), ("orange", (255, 200, 0)),
                 ("pink", (255, 175, 175)), ("lightGray", (192, 192, 192)), ("darkGray", (64, 64, 64))):
    setattr(Color, _n, Color(*_rgb))
    setattr(Color, _n.upper() if _n.islower() else {"lightGray": "LIGHT_GRAY", "darkGray": "DARK_GRAY"}[_n], Color(*_rgb))


def color_hex(c: Any, default: str = "#ffff00") -> str:
    """Any colour a script may hand over (``Color``, ``"#rrggbb"``, a name, a packed int) → ``#rrggbb``."""
    if c is None:
        return default
    if isinstance(c, Color):
        return c.toHex()
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return Color(int(c)).toHex()
    s = str(c).strip()
    try:
        return "#%02x%02x%02x" % _parse_hex(s)
    except ValueError:
        return s if s.startswith("#") else default


class Rectangle:
    _js_props = ("x", "y", "width", "height")

    def __init__(self, x: Any = 0, y: Any = 0, width: Any = 0, height: Any = 0) -> None:
        if isinstance(x, Rectangle):
            x, y, width, height = x.x, x.y, x.width, x.height
        self.x, self.y, self.width, self.height = int(x), int(y), int(width), int(height)

    def getX(self) -> float:
        return float(self.x)

    def getY(self) -> float:
        return float(self.y)

    def getWidth(self) -> float:
        return float(self.width)

    def getHeight(self) -> float:
        return float(self.height)

    def getCenterX(self) -> float:
        return self.x + self.width / 2.0

    def getCenterY(self) -> float:
        return self.y + self.height / 2.0

    def getBounds(self) -> "Rectangle":
        return Rectangle(self)

    def getLocation(self) -> "Point":
        return Point(self.x, self.y)

    def setBounds(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self.x, self.y, self.width, self.height = int(x), int(y), int(w), int(h)

    def setLocation(self, x: Any, y: Any) -> None:
        self.x, self.y = int(x), int(y)

    def setSize(self, w: Any, h: Any) -> None:
        self.width, self.height = int(w), int(h)

    def translate(self, dx: Any, dy: Any) -> None:
        self.x += int(dx)
        self.y += int(dy)

    def contains(self, x: Any, y: Any = None, w: Any = None, h: Any = None) -> bool:
        if isinstance(x, Rectangle):
            r = x
            return self.contains(r.x, r.y) and self.contains(r.x + r.width - 1, r.y + r.height - 1)
        if isinstance(x, Point):
            x, y = x.x, x.y
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def intersects(self, r: "Rectangle") -> bool:
        return not (r.x >= self.x + self.width or r.x + r.width <= self.x or r.y >= self.y + self.height or r.y + r.height <= self.y)

    def intersection(self, r: "Rectangle") -> "Rectangle":
        x1, y1 = max(self.x, r.x), max(self.y, r.y)
        x2, y2 = min(self.x + self.width, r.x + r.width), min(self.y + self.height, r.y + r.height)
        return Rectangle(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def union(self, r: "Rectangle") -> "Rectangle":
        x1, y1 = min(self.x, r.x), min(self.y, r.y)
        x2, y2 = max(self.x + self.width, r.x + r.width), max(self.y + self.height, r.y + r.height)
        return Rectangle(x1, y1, x2 - x1, y2 - y1)

    def isEmpty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def clone(self) -> "Rectangle":
        return Rectangle(self)

    def equals(self, other: Any) -> bool:
        return self == other

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Rectangle) and (self.x, self.y, self.width, self.height) == (other.x, other.y, other.width, other.height)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.width, self.height))

    def __iter__(self) -> Any:
        return iter((self.x, self.y, self.width, self.height))

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"java.awt.Rectangle[x={self.x},y={self.y},width={self.width},height={self.height}]"


class Point:
    _js_props = ("x", "y")

    def __init__(self, x: Any = 0, y: Any = 0) -> None:
        if isinstance(x, Point):
            x, y = x.x, x.y
        self.x, self.y = int(x), int(y)

    def getX(self) -> float:
        return float(self.x)

    def getY(self) -> float:
        return float(self.y)

    def setLocation(self, x: Any, y: Any) -> None:
        self.x, self.y = int(x), int(y)

    def translate(self, dx: Any, dy: Any) -> None:
        self.x += int(dx)
        self.y += int(dy)

    def distance(self, x: Any, y: Any = None) -> float:
        if isinstance(x, Point):
            x, y = x.x, x.y
        return _math.hypot(self.x - x, self.y - y)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __iter__(self) -> Any:
        return iter((self.x, self.y))

    def __repr__(self) -> str:
        return f"java.awt.Point[x={self.x},y={self.y}]"


class Dimension:
    _js_props = ("width", "height")

    def __init__(self, width: Any = 0, height: Any = 0) -> None:
        self.width, self.height = int(width), int(height)

    def getWidth(self) -> float:
        return float(self.width)

    def getHeight(self) -> float:
        return float(self.height)

    def __iter__(self) -> Any:
        return iter((self.width, self.height))

    def __repr__(self) -> str:
        return f"java.awt.Dimension[width={self.width},height={self.height}]"


class Polygon:
    """``java.awt.Polygon`` – integer vertices in ``xpoints`` / ``ypoints`` / ``npoints``."""

    _js_props = ("xpoints", "ypoints", "npoints")
    _int = True

    def __init__(self, xs: Any = None, ys: Any = None, n: Any = None) -> None:
        conv = int if self._int else float
        xs = [] if xs is None else [conv(v) for v in xs]
        ys = [] if ys is None else [conv(v) for v in ys]
        n = min(len(xs), len(ys)) if n is None else int(n)
        self.xpoints = xs[:n]
        self.ypoints = ys[:n]

    @property
    def npoints(self) -> int:
        return len(self.xpoints)

    def addPoint(self, x: Any, y: Any) -> None:
        conv = int if self._int else float
        self.xpoints.append(conv(x))
        self.ypoints.append(conv(y))

    def getBounds(self) -> Rectangle:
        if not self.xpoints:
            return Rectangle()
        x0, y0 = _math.floor(min(self.xpoints)), _math.floor(min(self.ypoints))
        x1, y1 = _math.ceil(max(self.xpoints)), _math.ceil(max(self.ypoints))
        return Rectangle(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def contains(self, x: Any, y: Any = None) -> bool:
        if isinstance(x, Point):
            x, y = x.x, x.y
        from .. import roi as roimod
        return roimod.contains_point({"type": "polygon", "points": [[px, py] for px, py in zip(self.xpoints, self.ypoints)]}, float(x), float(y))

    def translate(self, dx: Any, dy: Any) -> None:
        conv = int if self._int else float
        self.xpoints = [conv(v + dx) for v in self.xpoints]
        self.ypoints = [conv(v + dy) for v in self.ypoints]

    def getLength(self, closed: Any = True) -> float:
        n = self.npoints
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n - 1):
            total += _math.hypot(self.xpoints[i + 1] - self.xpoints[i], self.ypoints[i + 1] - self.ypoints[i])
        if closed and n > 2:
            total += _math.hypot(self.xpoints[0] - self.xpoints[-1], self.ypoints[0] - self.ypoints[-1])
        return total

    def duplicate(self) -> "Polygon":
        return type(self)(list(self.xpoints), list(self.ypoints))

    def reset(self) -> None:
        self.xpoints, self.ypoints = [], []

    def __len__(self) -> int:
        return self.npoints

    def __repr__(self) -> str:
        return f"{type(self).__name__}[n={self.npoints}]"


class Font:
    PLAIN = 0
    BOLD = 1
    ITALIC = 2
    _js_props = ("name", "style", "size")

    def __init__(self, name: Any = "SansSerif", style: Any = 0, size: Any = 12) -> None:
        self.name, self.style, self.size = str(name), int(style), int(size)

    def getName(self) -> str:
        return self.name

    def getSize(self) -> int:
        return self.size

    def getStyle(self) -> int:
        return self.style

    def isBold(self) -> bool:
        return bool(self.style & 1)

    def isItalic(self) -> bool:
        return bool(self.style & 2)

    def deriveFont(self, *args: Any) -> "Font":
        f = Font(self.name, self.style, self.size)
        for a in args:
            if isinstance(a, float):
                f.size = int(a)
            elif isinstance(a, int):
                f.style = a
        return f


class Toolkit:
    @staticmethod
    def getDefaultToolkit() -> "Toolkit":
        return Toolkit()

    def getScreenSize(self) -> Dimension:
        return Dimension(1920, 1080)

    def beep(self) -> None:
        pass


# ====================================================================== jarray
_TYPECODES = {"d": float, "f": float, "i": int, "l": int, "s": int, "h": int, "b": int, "B": int, "c": str, "z": bool, "u": str}


def array(seq: Any, typecode: Any = "d") -> list[Any]:
    """``jarray.array(sequence, 'd' | 'i' | … | SomeClass)`` → a plain list with coerced elements."""
    conv = _TYPECODES.get(typecode) if isinstance(typecode, str) else None
    items = list(seq)
    return [conv(v) for v in items] if conv is not None else items


def zeros(n: Any, typecode: Any = "d") -> list[Any]:
    conv = _TYPECODES.get(typecode) if isinstance(typecode, str) else None
    if conv is None:
        return [None] * int(n)
    zero: Any = "" if conv is str else conv(0)
    return [zero] * int(n)
