"""The JavaScript bridge: the browser's script worker calls the shim classes synchronously (one HTTP round trip per
call, ``POST /api/ijshim/call``) so Fiji JavaScript scripts (``importClass(Packages.ij.IJ); var imp =
IJ.getImage(); IJ.run(imp, "Gaussian Blur...", "sigma=2");``) run with their ordinary synchronous semantics.

One :class:`Bridge` per script run (keyed by the worker's token) owns a scripting façade (so the selection and the
macro-layer state are script-wide, like a Python script's), a handle table for the objects handed to JavaScript
(``ImagePlus``, ``ImageProcessor``, ``ResultsTable`` …) and the ``IJ.log`` lines produced during a call.

Wire format – request ``{"token", "target", "method", "args"}``:

* ``target``: a class path (``"ij.IJ"``), a package path (``"ij.plugin"``), ``""`` (the index) or a handle id;
* ``method``: a method name, or ``__describe__`` (what a class / instance offers), ``__new__`` (construct),
  ``__getattr__`` / ``__setattr__`` (fields), ``__index__`` (class name → package of the auto-imported packages);
* values: JSON scalars, lists, ``{"__nan__": true}``, ``{"__inf__": ±1}``, ``{"__handle__": id, "__class__": path}``
  for shim objects and ``{"__classref__": path}`` for classes.

Response ``{"result", "log": [...]}`` or ``{"error", "log"}``."""
from __future__ import annotations

import inspect
import math
import time
from typing import Any

import numpy as np

from . import class_index, class_path, module_contents, resolve_class, MODULES
from ._context import _CURRENT, Context

_SCALARS = (bool, int, float, str)


def describe(target: Any) -> dict[str, Any]:
    """What JavaScript may do with a class, an instance or a package."""
    if isinstance(target, dict):  # a package
        return {"kind": "package", "classes": sorted(k for k, v in target.items() if isinstance(v, type) or callable(v))}
    cls = target if isinstance(target, type) else type(target)
    consts: dict[str, Any] = {}
    statics: list[str] = []
    nested: list[str] = []
    methods: list[str] = []
    props: list[str] = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        if isinstance(attr, (staticmethod, classmethod)) or inspect.isfunction(attr) or inspect.ismethoddescriptor(attr) or type(attr).__name__ == "_InstanceOrStatic":
            methods.append(name)
        elif isinstance(attr, property):
            props.append(name)
        elif isinstance(attr, type):
            nested.append(name)
        elif attr is None or isinstance(attr, _SCALARS):
            consts[name] = _encode_scalar(attr)
        else:
            statics.append(name)
    props.extend(p for p in getattr(cls, "_js_props", ()) if p not in props)
    if not isinstance(target, type):
        for name, value in list(getattr(target, "__dict__", {}).items())[:64]:
            if not name.startswith("_") and name not in props and name not in methods:
                props.append(name)
    return {"kind": "class", "class": class_path(cls), "methods": sorted(set(methods)), "props": sorted(set(props)), "consts": consts, "statics": sorted(statics),
            "nested": sorted(nested)}


def _encode_scalar(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return {"__nan__": True}
        if math.isinf(v):
            return {"__inf__": 1 if v > 0 else -1}
    return v


class Bridge:
    """One JavaScript script run: the façade, the handle table and the log buffer."""

    def __init__(self, session: Any, roi: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> None:
        from ..scripting import LumenAPI
        self.session = session
        self.api = LumenAPI(session)
        if roi:
            self.api.roi = roi
        self.ctx = Context(session, self.api, emit=self._emit, params=params)
        self.handles: dict[int, Any] = {}
        self._ids: dict[int, int] = {}
        self._next = 1
        self.log: list[str] = []
        self.touched = time.time()

    def _emit(self, text: str) -> None:
        self.log.append(text)

    # --- handles
    def handle_of(self, obj: Any) -> int:
        key = id(obj)
        h = self._ids.get(key)
        if h is not None and self.handles.get(h) is obj:
            return h
        h = self._next
        self._next += 1
        self.handles[h] = obj
        self._ids[key] = h
        return h

    # --- marshalling
    def encode(self, v: Any) -> Any:
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return _encode_scalar(float(v))
        if isinstance(v, np.ndarray):
            return self.encode(v.tolist())
        if isinstance(v, (list, tuple, set, frozenset)):
            return [self.encode(x) for x in v]
        if isinstance(v, dict):
            return {str(k): self.encode(x) for k, x in v.items()}
        if isinstance(v, type):
            return {"__classref__": class_path(v)}
        if isinstance(v, bytes):
            return list(v)
        from ..image import LumenImage
        if isinstance(v, LumenImage):
            from ._images import ImagePlus
            v = ImagePlus(_img=v)
        if inspect.isroutine(v):
            return None
        return {"__handle__": self.handle_of(v), "__class__": class_path(type(v))}

    def decode(self, v: Any) -> Any:
        if isinstance(v, dict):
            if "__handle__" in v:
                h = int(v["__handle__"])
                if h not in self.handles:
                    raise ValueError(f"ij bridge: unknown object handle {h}")
                return self.handles[h]
            if "__classref__" in v:
                cls = resolve_class(str(v["__classref__"]))
                if cls is None:
                    raise ValueError(f"ij bridge: unknown class {v['__classref__']!r}")
                return cls
            if v.get("__nan__"):
                return float("nan")
            if "__inf__" in v:
                return float("inf") if v["__inf__"] > 0 else float("-inf")
            return {k: self.decode(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.decode(x) for x in v]
        return v

    # --- dispatch
    def resolve_target(self, target: Any) -> Any:
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            h = int(target)
            if h not in self.handles:
                raise ValueError(f"ij bridge: unknown object handle {h}")
            return self.handles[h]
        path = str(target)
        if path == "":
            return None
        if path in MODULES:
            return module_contents(path)
        obj = resolve_class(path)
        if obj is None:
            raise ValueError(f"ij bridge: unknown class or package {path!r} (no JVM – see docs/SCRIPTING.md for the supported ij.* subset)")
        return obj

    def call(self, target: Any, method: str, args: list[Any] | None = None) -> dict[str, Any]:
        self.touched = time.time()
        self.log = []
        token = _CURRENT.set(self.ctx)
        try:
            obj = self.resolve_target(target)
            a = [self.decode(x) for x in (args or [])]
            if method == "__index__":
                result: Any = class_index()
            elif method == "__describe__":
                result = describe(obj)
            elif method == "__new__":
                if not callable(obj):
                    raise TypeError(f"{target} is not constructible")
                result = obj(*a)
            elif method == "__getattr__":
                name = str(a[0])
                if isinstance(obj, dict):
                    result = obj.get(name)
                else:
                    result = getattr(obj, name)
            elif method == "__setattr__":
                setattr(obj, str(a[0]), a[1])
                result = None
            elif method == "toString":
                fn = getattr(obj, "toString", None)
                result = fn() if callable(fn) else str(obj)
            else:
                if isinstance(obj, dict):  # a package: only classes live here
                    raise AttributeError(f"{target} is a package, not a class")
                fn = getattr(obj, method)
                result = fn(*a) if callable(fn) else fn
            return {"result": self.encode(result), "log": list(self.log)}
        finally:
            _CURRENT.reset(token)
            self.session._changed("images")

    @property
    def roi(self) -> dict[str, Any] | None:
        return self.api.roi
