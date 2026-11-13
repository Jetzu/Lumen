"""SciJava ``#@`` script parameters.

A Fiji script declares its inputs in header lines::

    #@ String (label="Name", value="cells") name
    #@ File (label="Input folder", style="directory") folder
    #@ Integer (label="Threshold", min=0, max=255, value=128) threshold
    #@ Double sigma
    #@ Boolean (value=true) show
    #@ ImagePlus imp
    #@ LogService log
    #@output String result

:func:`parse` reads them (``#@`` and the legacy ``# @`` form, attributes before or after the name) and
:func:`prepare` turns the header into one line of assignments so the variables exist when the script runs,
taking the values from the ``params`` dict handed to the runner (the Script panel's parameter prompt), else from
the ``value`` attribute, else the type's default.  ``ImagePlus`` / ``Dataset`` parameters are the current image,
``File`` parameters become ``java.io.File`` objects (also ``os.PathLike``), the SciJava services become small
stand-ins (``LogService``, ``UIService``, ``StatusService``, ``DatasetIOService`` …), ``#@output`` lines are
ignored (the script sets them itself).  :func:`descriptors` is what the UI uses to build the prompt."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

_HEADER = re.compile(r"^\s*#\s*@\s*(.*?)\s*$")
_IO_PREFIX = re.compile(r"^(output|input|both)\b\s*", re.IGNORECASE)
_TYPE = re.compile(r"^([A-Za-z_][\w.]*(?:\[\])?)\s*")
_NAME = re.compile(r"^([A-Za-z_]\w*)\s*")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*(e[-+]?\d+)?|\.\d+(e[-+]?\d+)?)$", re.IGNORECASE)

_INT_TYPES = {"int", "integer", "long", "short", "byte", "biginteger"}
_FLOAT_TYPES = {"float", "double", "bigdecimal", "number"}
_BOOL_TYPES = {"boolean"}
_STRING_TYPES = {"string", "char", "character", "text"}
_IMAGE_TYPES = {"imageplus", "dataset", "img", "imgplus", "randomaccessibleinterval", "iterableinterval", "imagedisplay", "datasetview", "image"}
_COLOR_TYPES = {"colorrgb", "color", "colorrgba"}
_TABLE_TYPES = {"resultstable", "table", "genericTable".lower()}
_ROI_TYPES = {"roimanager"}
_SERVICE_TYPES = {"logservice", "uiservice", "statusservice", "opservice", "commandservice", "datasetservice", "datasetioservice", "ioservice", "scriptservice",
                  "convertservice", "displayservice", "threadservice", "pluginservice", "moduleservice", "overlayservice", "lutservice", "prefservice", "imagej",
                  "context", "imagedisplayservice", "formatservice", "appservice", "eventservice", "objectservice", "roiservice", "notebookservice"}


@dataclass
class ScriptParam:
    name: str
    java_type: str
    kind: str  # int | float | bool | string | choice | file | directory | color | image | table | roimanager | service | message | other
    io: str = "input"  # input | output | both
    label: str | None = None
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    style: str | None = None
    description: str | None = None
    required: bool = True
    visibility: str | None = None
    persist: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)
    line: int = 0

    def descriptor(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.kind, "java_type": self.java_type, "io": self.io, "label": self.label or self.name,
                             "default": self.default, "required": self.required}
        for k in ("min", "max", "step", "choices", "style", "description", "visibility"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class Prepared:
    code: str
    params: list[ScriptParam]
    values: dict[str, Any]
    line_offset: int = 0

    @property
    def descriptors(self) -> list[dict[str, Any]]:
        return [p.descriptor() for p in self.params]


# ---------------------------------------------------------------------- attribute parsing
def _split_top(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    cur: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch in "{[(":
            depth += 1
            cur.append(ch)
        elif ch in "}])":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        body = s[1:-1]
        return body.encode("utf-8").decode("unicode_escape") if "\\" in body else body
    return s


def _value(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""
    if s[0] in ("'", '"'):
        return _unquote(s)
    if s[0] == "{" and s[-1] == "}":
        return [_value(p) for p in _split_top(s[1:-1])]
    if s[0] == "[" and s[-1] == "]":
        return [_value(p) for p in _split_top(s[1:-1])]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _NUMBER.match(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    return s


def _parse_attrs(text: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for part in _split_top(text):
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = _value(v)
        elif part:
            attrs[part.strip()] = True
    return attrs


def _take_parens(s: str) -> tuple[str | None, str]:
    """``(attrs) rest`` → (attrs, rest); None when ``s`` does not start with ``(``."""
    if not s.startswith("("):
        return None, s
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(s):
        if quote:
            if ch == "\\":
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[1:i], s[i + 1:].strip()
    return s[1:], ""


def _kind(java_type: str, attrs: dict[str, Any]) -> str:
    t = java_type.split(".")[-1].rstrip("[]").lower()
    vis = str(attrs.get("visibility", "")).upper()
    if vis == "MESSAGE":
        return "message"
    style = str(attrs.get("style", "")).lower()
    if t in _INT_TYPES:
        return "int"
    if t in _FLOAT_TYPES:
        return "float"
    if t in _BOOL_TYPES:
        return "bool"
    if t in _STRING_TYPES:
        return "choice" if attrs.get("choices") else "string"
    if t == "file":
        return "directory" if "director" in style else "file"
    if t in _IMAGE_TYPES:
        return "image"
    if t in _COLOR_TYPES:
        return "color"
    if t in _TABLE_TYPES:
        return "table"
    if t in _ROI_TYPES:
        return "roimanager"
    if t in _SERVICE_TYPES or t.endswith("service"):
        return "service"
    return "other"


def parse(code: str) -> tuple[list[ScriptParam], list[int]]:
    """The ``#@`` parameters of a script and the (0-based) line numbers of the header lines."""
    params: list[ScriptParam] = []
    lines: list[int] = []
    for i, raw in enumerate(code.splitlines()):
        m = _HEADER.match(raw)
        if not m:
            continue
        body = m.group(1)
        if not body:
            continue
        io = "input"
        mio = _IO_PREFIX.match(body)
        if mio:
            io = mio.group(1).lower()
            body = body[mio.end():]
        mt = _TYPE.match(body)
        if not mt:
            continue
        java_type = mt.group(1)
        rest = body[mt.end():]
        attrs: dict[str, Any] = {}
        inner, rest = _take_parens(rest)
        if inner is not None:
            attrs.update(_parse_attrs(inner))
        mn = _NAME.match(rest)
        if not mn:
            continue
        name = mn.group(1)
        rest = rest[mn.end():]
        inner, rest = _take_parens(rest)
        if inner is not None:
            attrs.update(_parse_attrs(inner))
        if rest.startswith("="):  # "#@ String name = default" (not SciJava, but seen in the wild)
            attrs.setdefault("value", _value(rest[1:].strip()))
        kind = _kind(java_type, attrs)
        choices = attrs.get("choices")
        if choices is not None and not isinstance(choices, list):
            choices = [choices]
        default = attrs.get("value", attrs.get("default"))
        if default is None and kind == "choice" and choices:
            default = choices[0]
        if default is None and kind == "message":
            default = attrs.get("label") or attrs.get("value") or ""
        p = ScriptParam(name=name, java_type=java_type, kind=kind, io=io, label=attrs.get("label") or (attrs.get("value") if kind == "message" else None),
                        default=default, min=_num_or_none(attrs.get("min")), max=_num_or_none(attrs.get("max")), step=_num_or_none(attrs.get("stepSize", attrs.get("step"))),
                        choices=[str(c) for c in choices] if choices else None, style=str(attrs["style"]) if "style" in attrs else None,
                        description=str(attrs["description"]) if "description" in attrs else None,
                        required=not (str(attrs.get("required", "true")).lower() == "false"), visibility=str(attrs["visibility"]) if "visibility" in attrs else None,
                        persist=not (str(attrs.get("persist", "true")).lower() == "false"), attrs=attrs, line=i)
        params.append(p)
        lines.append(i)
    return params, lines


def _num_or_none(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def descriptors(code: str) -> list[dict[str, Any]]:
    return [p.descriptor() for p in parse(code)[0]]


# ---------------------------------------------------------------------- values
def coerce(p: ScriptParam, v: Any) -> Any:
    """A caller-supplied value in the parameter's type."""
    if v is None:
        return None
    try:
        if p.kind == "int":
            return int(float(v))
        if p.kind == "float":
            return float(v)
        if p.kind == "bool":
            return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes", "on")
        if p.kind in ("string", "choice", "color", "message"):
            return str(v)
        if p.kind in ("file", "directory"):
            return os.fspath(v) if not isinstance(v, str) else v
    except (TypeError, ValueError):
        return p.default
    return v


def _default(p: ScriptParam) -> Any:
    d = p.default
    if p.kind == "int":
        return int(float(d)) if d is not None and str(d) != "" else (int(p.min) if p.min is not None and p.min > 0 else 0)
    if p.kind == "float":
        return float(d) if d is not None and str(d) != "" else (float(p.min) if p.min is not None and p.min > 0 else 0.0)
    if p.kind == "bool":
        return bool(d) if isinstance(d, bool) else str(d).strip().lower() == "true" if d is not None else False
    if p.kind in ("string", "choice", "color", "message"):
        return "" if d is None else str(d)
    if p.kind in ("file", "directory"):
        return None if d in (None, "") else str(d)
    return d


def resolve_values(params: list[ScriptParam], values: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    given = values or {}
    for p in params:
        if p.io == "output":
            continue
        if p.name in given and given[p.name] is not None and not (given[p.name] == "" and p.kind in ("int", "float", "bool")):
            out[p.name] = coerce(p, given[p.name])
        elif p.label and p.label in given and given[p.label] is not None:
            out[p.name] = coerce(p, given[p.label])
        else:
            out[p.name] = _default(p)
    return out


def _expr(p: ScriptParam, v: Any) -> str:
    if p.kind in ("int", "float", "bool", "string", "choice", "color", "message"):
        return repr(v)
    if p.kind in ("file", "directory"):
        return f"__ijshim.file({v!r})" if v is not None else "None"
    if p.kind == "image":
        return "__ijshim.image()"
    if p.kind == "table":
        return "__ijshim.table()"
    if p.kind == "roimanager":
        return "__ijshim.roi_manager()"
    if p.kind == "service":
        return f"__ijshim.service({p.java_type.split('.')[-1]!r})"
    return repr(v) if isinstance(v, (int, float, str, bool)) or v is None else "None"


_LEGACY_IN = re.compile(r"\bloci\.plugins\.in\b(?=[\s.])")


def prepare(code: str, values: dict[str, Any] | None = None) -> Prepared:
    """The script with its ``#@`` header turned into assignments (one line, numbering preserved whenever the header
    is at the top) and ``loci.plugins.in`` (a Python keyword) rewritten to the shim's ``loci.plugins.in_``."""
    code = _LEGACY_IN.sub("loci.plugins.in_", code)
    params, header_lines = parse(code)
    if not params:
        return Prepared(code=code, params=[], values={})
    resolved = resolve_values(params, values)
    lines = code.splitlines(keepends=True)
    assigns = [f"{p.name} = {_expr(p, resolved.get(p.name))}" for p in params if p.io != "output"]
    stmt = "from lumen.core.ijshim import params as __ijshim; " + "; ".join(assigns) if assigns else ""
    first = header_lines[0]
    top = all(ln.strip() == "" or ln.lstrip().startswith("#") for ln in lines[:first])
    for i in header_lines:
        nl = "\n" if lines[i].endswith("\n") else ""
        lines[i] = "#" + nl
    offset = 0
    if stmt:
        if top:
            nl = "\n" if lines[first].endswith("\n") else ""
            lines[first] = stmt + nl
        else:
            lines.insert(0, stmt + "\n")
            offset = 1
    return Prepared(code="".join(lines), params=params, values=resolved, line_offset=offset)


# ---------------------------------------------------------------------- runtime helpers used by the generated line
def file(path: Any) -> Any:  # noqa: A001
    from ._java import File
    return File(path)


def image() -> Any:
    from ._ij import IJ
    return IJ.getImage()


def table() -> Any:
    from ._tables import ResultsTable
    return ResultsTable.getResultsTable()


def roi_manager() -> Any:
    from ._tables import RoiManager
    return RoiManager.getInstance()


class _LogService:
    def _emit(self, level: str, *args: Any) -> None:
        from ._context import current
        current().emit(f"[{level}] " + " ".join(str(a) for a in args))

    def info(self, *a: Any) -> None:
        self._emit("INFO", *a)

    def warn(self, *a: Any) -> None:
        self._emit("WARN", *a)

    def error(self, *a: Any) -> None:
        self._emit("ERROR", *a)

    def debug(self, *a: Any) -> None:
        self._emit("DEBUG", *a)

    def trace(self, *a: Any) -> None:
        self._emit("TRACE", *a)

    def log(self, level: Any, *a: Any) -> None:
        self._emit(str(level), *a)

    def setLevel(self, *a: Any) -> None:
        pass

    def isDebug(self) -> bool:
        return False


class _UIService:
    def show(self, *args: Any) -> None:
        from ._context import current
        from ._images import ImagePlus
        obj = args[-1] if args else None
        name = str(args[0]) if len(args) > 1 else None
        if isinstance(obj, ImagePlus):
            if name:
                obj.setTitle(name)
            obj.show()
        elif hasattr(obj, "show"):
            obj.show(name) if name else obj.show()
        else:
            current().emit(f"{name + ': ' if name else ''}{obj}")

    def showDialog(self, *args: Any) -> None:
        from ._context import current
        current().emit(": ".join(str(a) for a in args[:2]))

    def isHeadless(self) -> bool:
        return True

    def isVisible(self, *a: Any) -> bool:
        return True

    def chooseFile(self, *a: Any) -> Any:
        from ._dialogs import OpenDialog
        return OpenDialog("Choose a file").getPath()

    def getDefaultUI(self) -> None:
        return None

    def getDisplay(self, *a: Any) -> None:
        return None

    def showUI(self, *a: Any) -> None:
        pass

    def setHeadless(self, *a: Any) -> None:
        pass


class _StatusService:
    def showStatus(self, *a: Any) -> None:
        pass

    def showProgress(self, *a: Any) -> None:
        pass

    def clearStatus(self) -> None:
        pass

    def warn(self, msg: Any) -> None:
        from ._context import current
        current().emit(f"[WARN] {msg}")


class _DatasetIOService:
    def open(self, path: Any, *a: Any) -> Any:  # noqa: A003
        from ._images import ImagePlus
        from ..io import read_image
        return ImagePlus(_img=read_image(os.fspath(path)))

    def save(self, obj: Any, path: Any, *a: Any) -> None:
        from ._io import FileSaver
        FileSaver(obj).saveAsTiff(path) if not str(path).lower().endswith(".png") else FileSaver(obj).saveAsPng(path)

    def canOpen(self, path: Any) -> bool:
        return os.path.exists(os.fspath(path))

    def canSave(self, *a: Any) -> bool:
        return True


class _Unsupported:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        raise AttributeError(f"{self._name}.{item}: the SciJava service {self._name} is not available in Lumen (no JVM) – use the ij.* classes or the lumen API")

    def __repr__(self) -> str:
        return f"<{self._name}: unavailable in Lumen>"


class _ImageJGateway:
    """``#@ ImageJ ij``: ``ij.ui()``, ``ij.log()``, ``ij.status()``, ``ij.io()`` / ``ij.scifio().datasetIO()``."""

    def ui(self) -> _UIService:
        return _UIService()

    def log(self) -> _LogService:
        return _LogService()

    def status(self) -> _StatusService:
        return _StatusService()

    def io(self) -> _DatasetIOService:
        return _DatasetIOService()

    def dataset(self) -> _DatasetIOService:
        return _DatasetIOService()

    def scifio(self) -> "_ImageJGateway":
        return self

    def datasetIO(self) -> _DatasetIOService:
        return _DatasetIOService()

    def op(self) -> _Unsupported:
        return _Unsupported("OpService")

    def command(self) -> _Unsupported:
        return _Unsupported("CommandService")

    def context(self) -> "_ImageJGateway":
        return self

    def getContext(self) -> "_ImageJGateway":
        return self

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        return _Unsupported(item)


def service(name: str) -> Any:
    n = name.lower()
    if n == "logservice":
        return _LogService()
    if n == "uiservice":
        return _UIService()
    if n == "statusservice":
        return _StatusService()
    if n in ("datasetioservice", "ioservice", "datasetservice"):
        return _DatasetIOService()
    if n in ("imagej", "context"):
        return _ImageJGateway()
    return _Unsupported(name)
