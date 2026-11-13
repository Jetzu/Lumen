"""How a shim call finds *its* Lumen session.

Jython-style scripts (``from ij import IJ``) run through :func:`lumen.core.scripting.run_script`, which binds a
``LumenAPI`` as ``lumen`` in the script's globals (``__name__ == "__lumen__"``).  Every static call of the shim
(``IJ.getImage()``, ``RoiManager.getInstance()`` …) resolves the session of the *calling script* by walking the
stack to that frame, so the shim shares the script's selection (``lumen.roi``) and its macro-layer ``IJ``
instance with the injected macro functions (``makeRectangle`` …) – nothing has to be threaded through.

Two other entry points bind a :class:`Context` explicitly (a ``contextvars`` variable, checked first):

* the JavaScript bridge (``lumen.core.ijshim.bridge``) – one context per browser script run;
* :func:`bind` – headless use (``with bind(session): IJ.run(...)``).
"""
from __future__ import annotations

import contextlib
import contextvars
import sys
from typing import Any, Callable, Iterator

_UNSET: Any = object()

_CURRENT: contextvars.ContextVar["Context | None"] = contextvars.ContextVar("ijshim_context", default=None)
_PARAMS: contextvars.ContextVar["dict[str, Any] | None"] = contextvars.ContextVar("ijshim_params", default=None)
_SEED_ROI: contextvars.ContextVar[Any] = contextvars.ContextVar("ijshim_seed_roi", default=_UNSET)
_LAST: contextvars.ContextVar["Context | None"] = contextvars.ContextVar("ijshim_last", default=None)
_DEFAULT: "Context | None" = None


class Context:
    """The session, the scripting façade (``LumenAPI``) and the macro-layer ``IJ`` a shim object works with."""

    def __init__(self, session: Any, api: Any = None, emit: Callable[[str], None] | None = None, params: dict[str, Any] | None = None) -> None:
        if api is None:
            from ..scripting import LumenAPI
            api = LumenAPI(session)
        self.session = session
        self.api = api
        self._emit = emit
        self.params: dict[str, Any] = dict(params or {})
        self.roi_manager: Any = None  # the RoiManager singleton of this context (lazy)
        self.prefs: dict[str, Any] = {}

    # ------------------------------------------------------------------ state shared with the macro layer
    @property
    def ij(self) -> Any:
        """The macro-layer ``IJ`` instance (``lumen.core.ij_compat.IJ``) bound to this session / selection."""
        return self.api.ij

    @property
    def roi(self) -> dict[str, Any] | None:
        return self.api.roi

    @roi.setter
    def roi(self, value: dict[str, Any] | None) -> None:
        self.ij.roi = value  # the macro layer normalises and keeps the previous selection for Restore Selection

    def emit(self, text: Any) -> None:
        """``IJ.log`` / ``System.out.println``: the script's output (stdout for Python scripts, the bridge buffer for
        JavaScript) and Lumen's Log section."""
        s = str(text)
        if self._emit is not None:
            self._emit(s)
        else:
            print(s)
        try:
            self.session.add_log(s)
        except Exception:  # pragma: no cover - a session without a log
            pass

    def param(self, *keys: Any, default: Any = _UNSET) -> Any:
        """A value supplied by the caller (the Script panel's parameter prompt / ``params`` of the runner) under any of
        ``keys`` (exact, or normalised: lower case, no trailing colon, spaces as underscores)."""
        if not self.params:
            return default
        for k in keys:
            if k is None:
                continue
            ks = str(k)
            if ks in self.params:
                return self.params[ks]
        norm = {_norm(k): v for k, v in self.params.items()}
        for k in keys:
            if k is None:
                continue
            n = _norm(k)
            if n in norm:
                return norm[n]
        return default

    @classmethod
    def of(cls, api: Any) -> "Context":
        """The context cached on a script's ``LumenAPI`` (created on first use, seeded with the runner's parameters
        and the viewport selection handed to the runner)."""
        ctx = api.__dict__.get("_ijshim_ctx")
        if ctx is None:
            ctx = cls(api.session, api, params=_PARAMS.get() or {})
            seed = _SEED_ROI.get()
            if seed is not _UNSET and seed is not None and api.roi is None:
                ctx.roi = seed
            api.__dict__["_ijshim_ctx"] = ctx
            _LAST.set(ctx)
        return ctx


def _norm(k: Any) -> str:
    return str(k).strip().rstrip(":").strip().lower().replace(" ", "_")


def current() -> Context:
    """The context of the running shim call (see the module docstring)."""
    ctx = _CURRENT.get()
    if ctx is not None:
        return ctx
    f = sys._getframe(1)
    while f is not None:
        g = f.f_globals
        if g.get("__name__") == "__lumen__":
            api = g.get("lumen")
            if api is not None and hasattr(api, "session") and hasattr(api, "ij"):
                return Context.of(api)
        f = f.f_back
    if _DEFAULT is not None:
        return _DEFAULT
    raise RuntimeError("The ImageJ shim needs a Lumen session: run the script through Lumen's script runner "
                       "(the Script panel, POST /api/script/run, lumen.core.ijshim.run_ij_script) or wrap headless "
                       "code in `with lumen.core.ijshim.bind(session): ...`")


def set_default(session: Any | None) -> None:
    """Fallback session for shim calls made outside any script or bridge (``None`` clears it)."""
    global _DEFAULT
    _DEFAULT = Context(session) if session is not None else None


@contextlib.contextmanager
def bind(session: Any, params: dict[str, Any] | None = None, api: Any = None) -> Iterator[Context]:
    """Headless use: ``with bind(session): IJ.run("Blobs (25K)"); imp = IJ.getImage()``."""
    ctx = Context(session, api=api, params=params)
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
