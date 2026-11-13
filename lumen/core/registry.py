"""The command registry.

Every command in Lumen is declared once with ``@op`` and a list of ``Param``s.  The
same declaration drives: the command palette and menus, the auto-generated parameter
sheet with live preview, the macro recorder, and the Python scripting API.

Kinds of command
----------------
* ``plane``    – ``fn(plane2d, ctx, **params) -> plane2d``. The registry iterates over
                 the planes selected by ``scope`` (plane | slice | stack), applies the ROI
                 mask, keeps dtype, and supports instant preview.
* ``image``    – ``fn(ctx, **params) -> OpResult``. Full control; may replace data,
                 create new images, change dtype/shape.
* ``analysis`` – ``fn(ctx, **params) -> OpResult``. Never modifies pixel data (no undo
                 snapshot); returns results/plots/ROIs.

Plugin tools
------------
Any op (of any kind) declared with ``tool=True`` is also placed on the tool rail of the UI (after the
built-in tools) with ``tool_icon`` (an emoji or 1–2 characters) and ``tool_cursor`` (a CSS cursor name).
While the tool is active, every click or drag on the image runs the op with the interaction added to its
parameters (``TOOL_PARAMS``): ``x``, ``y`` (image px of the press), ``x2``, ``y2`` (release position of a drag,
else None), ``button`` (0 left, 1 middle, 2 right), ``shift`` / ``alt`` / ``ctrl`` and ``event`` (``click``,
``drag-start`` or ``drag-end``).  The op's own ``Param`` list becomes the tool-options bar.  The interaction
parameters are not part of ``params`` (they never show up in a dialog) but ``coerce_params`` accepts them.
``tool_color`` (a CSS colour, optional) tints the icon – ImageJ macro tools derive it from their icon string.

Plugin frames
-------------
An op declared with ``frame=True`` (or ``kind="frame"``, which is an analysis op with the flag set) is a *plugin
frame*: ImageJ's Plugins ▸ New ▸ Plugin Frame, a plugin with its own always-visible window.  Lumen shows the output
of such commands (table, plot, log text, message) in the inspector section "Plugin Frames" – one docked sub-panel
per frame op that has been run, with the op's parameters as a compact form and a Run / Refresh button – instead of
in the shared Table / Plot sections.  The flag is exposed as ``frame`` in ``/api/ops``.
"""
from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .image import LumenImage
    from .session import Session


@dataclass
class Param:
    name: str
    type: str = "float"  # int | float | bool | choice | string | text | image | multichoice | color
    default: Any = None
    label: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    help: str = ""
    advanced: bool = False
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()

    def coerce(self, value: Any) -> Any:
        if value is None:
            return self.default
        t = self.type
        try:
            if t == "int":
                v = int(round(float(value)))
            elif t == "float":
                v = float(value)
            elif t == "bool":
                v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif t == "choice":
                v = str(value)
                if self.choices and v not in self.choices:
                    # allow case-insensitive match
                    low = {c.lower(): c for c in self.choices}
                    if v.lower() in low:
                        v = low[v.lower()]
                    else:
                        raise ValueError(f"{self.name}: {v!r} is not one of {self.choices}")
            elif t == "multichoice":
                v = [str(x) for x in (value if isinstance(value, (list, tuple)) else str(value).split(","))]
                v = [x.strip() for x in v if x.strip()]
            else:
                v = value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {self.name}: {value!r} ({e})") from e
        if t in ("int", "float"):
            # out-of-range values are rejected, never clamped: a silently changed analysis parameter is a correctness
            # risk (docs/USABILITY E7-7) – the message names the parameter, the allowed range and the value given
            lo, hi = self.min, self.max
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                if lo is not None and hi is not None:
                    rng = f"between {_num(lo)} and {_num(hi)}"
                elif lo is not None:
                    rng = f"at least {_num(lo)}"
                else:
                    rng = f"at most {_num(hi)}"
                raise ValueError(f"{self.name} must be {rng} (got {_num(v)})")
        return v

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


def _num(v: Any) -> str:
    """A number for an error message: ``500`` not ``500.0``, ``0.5`` not ``0.500000``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return repr(v)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:g}"


#: interaction parameters handed to a tool op (``tool=True``) by the viewport, see the module docstring
TOOL_EVENTS = ["click", "drag-start", "drag-end"]
TOOL_PARAMS: list["Param"] = [
    Param("x", "float", 0.0, help="Image x of the press (px)"), Param("y", "float", 0.0, help="Image y of the press (px)"),
    Param("x2", "float", None, help="Image x of the release (drags only)"), Param("y2", "float", None, help="Image y of the release (drags only)"),
    Param("button", "int", 0, help="0 left, 1 middle, 2 right"),
    Param("shift", "bool", False), Param("alt", "bool", False), Param("ctrl", "bool", False),
    Param("event", "choice", "click", choices=TOOL_EVENTS),
]


@dataclass
class OpResult:
    data: np.ndarray | None = None
    rgb: bool | None = None
    new_images: list["LumenImage"] = field(default_factory=list)
    results: list[dict[str, Any]] | None = None
    results_replace: bool = False
    plot: dict[str, Any] | None = None
    histogram: dict[str, Any] | None = None
    roi: dict[str, Any] | None = None
    clear_roi: bool = False
    rois_add: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    log: str | None = None
    table: dict[str, Any] | None = None
    overlay: list[dict[str, Any]] | None = None
    close_image: bool = False
    select_image: str | None = None
    display: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    rename: str | None = None
    threshold: dict[str, Any] | None = None


@dataclass
class OpSpec:
    id: str
    name: str
    category: str
    func: Callable[..., Any]
    kind: str = "plane"
    description: str = ""
    params: list[Param] = field(default_factory=list)
    preview: bool = True
    roi_aware: bool = True
    requires_image: bool = True
    requires_roi: str | None = None  # None | "area" | "line" | "line_or_rect" (Plot Profile) | "point" | "any"
    shortcut: str | None = None
    menu: str | None = None  # e.g. "Process/Filters"
    keywords: list[str] = field(default_factory=list)
    scope_param: bool = True  # offer "apply to" for stacks
    supports_rgb: bool = True
    icon: str = ""
    undo_label: str | None = None
    #: plugin tool: shown on the tool rail; clicks / drags on the image run the op with TOOL_PARAMS
    tool: bool = False
    tool_icon: str = ""
    tool_cursor: str = ""
    #: CSS colour tinting the tool icon (macro tools: the first ``C`` colour of the ImageJ icon string); "" = default
    tool_color: str = ""
    #: plugin frame: the op's output is shown in its own docked sub-panel of the inspector section "Plugin Frames"
    frame: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "category": self.category, "kind": self.kind,
            "description": self.description, "params": [p.to_dict() for p in self.params],
            "preview": self.preview and self.kind != "analysis", "roi_aware": self.roi_aware,
            "requires_image": self.requires_image, "requires_roi": self.requires_roi,
            "shortcut": self.shortcut, "menu": self.menu or self.category, "keywords": self.keywords,
            "scope_param": self.scope_param and self.kind == "plane", "supports_rgb": self.supports_rgb,
            "icon": self.icon, "tool": self.tool, "tool_icon": self.tool_icon, "tool_cursor": self.tool_cursor,
            "tool_color": self.tool_color, "frame": self.frame,
            "ij_commands": ij_commands_for(self.id),
        }

    def param_names(self) -> list[str]:
        """The accepted parameter names (the declared params plus the interaction params of a tool op)."""
        names = [p.name for p in self.params]
        if self.tool:
            names += [p.name for p in TOOL_PARAMS if p.name not in names]
        return names

    def coerce_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        known = self.param_names()
        unknown = [k for k in params if k not in known]
        if unknown:
            # a misspelt keyword (lumen.run("filters.gaussian", sigmaa=1)) used to run silently with the default
            k = str(unknown[0])
            close = difflib.get_close_matches(k, known, n=1, cutoff=0.6)
            if close:
                hint = f" – did you mean {close[0]!r}?"
            elif known:
                hint = f" (parameters: {', '.join(known)})"
            else:
                hint = " (this command takes no parameters)"
            raise ValueError(f"unknown parameter {k!r} for {self.id}{hint}")
        out = {}
        for p in self.params:
            out[p.name] = p.coerce(params.get(p.name, p.default))
        if self.tool:
            for p in TOOL_PARAMS:
                if p.name not in out:
                    out[p.name] = p.coerce(params.get(p.name, p.default))
        return out


_IJ_INDEX: tuple[int, dict[str, list[str]]] | None = None  # (len(COMMANDS) it was built from, op id → ImageJ names)


def ij_commands_for(op_id: str) -> list[str]:
    """The ImageJ ``run("…")`` command names registered for an op id in :data:`lumen.core.ij_compat.COMMANDS`
    (``"Gaussian Blur..."`` for ``filters.gaussian``), in registration order; empty when none maps to it (E7-20).

    The inverted table is built once and rebuilt only when commands were added (plugins / macro tools register
    later).  The macro layer is imported lazily: it imports this module itself."""
    global _IJ_INDEX
    try:
        from .ij_compat import COMMANDS
    except Exception:  # noqa: BLE001 – the registry must work without the macro layer
        return []
    if _IJ_INDEX is None or _IJ_INDEX[0] != len(COMMANDS):
        index: dict[str, list[str]] = {}
        for name, (_fn, lumen, _notes) in COMMANDS.items():
            for target in str(lumen).split(" / "):  # a handler serving two ops is listed as "a.op / b.op"
                target = target.strip()
                if target and target not in ("(no-op)", "(unsupported)"):
                    index.setdefault(target, []).append(name)
        _IJ_INDEX = (len(COMMANDS), index)
    return list(_IJ_INDEX[1].get(op_id, []))


class OperationCancelled(ValueError):
    """Raised by :meth:`OpContext.check_cancel` once the running job was cancelled (``POST /api/jobs/{id}/cancel``).

    A ``ValueError`` so that existing error handling treats it like any other failed command: the session
    rolls the image back to its undo snapshot and the server answers ``409 {"error": "Cancelled"}``.
    """

    def __init__(self, message: str = "Cancelled") -> None:
        super().__init__(message)


class Registry:
    def __init__(self) -> None:
        self.ops: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        if spec.id in self.ops:
            raise ValueError(f"duplicate op id {spec.id}")
        self.ops[spec.id] = spec

    def get(self, op_id: str) -> OpSpec:
        if op_id in self.ops:
            return self.ops[op_id]
        # allow lookup by display name
        low = op_id.lower()
        for spec in self.ops.values():
            if spec.name.lower() == low:
                return spec
        close = difflib.get_close_matches(str(op_id), list(self.ops), n=3, cutoff=0.5)
        if not close:  # a display name typo ("Gaussian Blurr") – match on names too
            by_name = {s.name: s.id for s in self.ops.values()}
            close = [by_name[n] for n in difflib.get_close_matches(str(op_id), list(by_name), n=3, cutoff=0.6)]
        hint = f" – did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown command {op_id!r}{hint}")

    def list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.ops.values()]


REGISTRY = Registry()


def op(id: str, name: str, category: str, *, kind: str = "plane", description: str = "",
       params: list[Param] | None = None, preview: bool = True, roi_aware: bool = True,
       requires_image: bool = True, requires_roi: str | None = None, shortcut: str | None = None,
       menu: str | None = None, keywords: list[str] | None = None, scope_param: bool = True,
       supports_rgb: bool = True, icon: str = "", undo_label: str | None = None,
       tool: bool = False, tool_icon: str = "", tool_cursor: str = "", tool_color: str = "", frame: bool = False) -> Callable:
    if kind == "frame":  # a plugin frame is an analysis op (never touches pixels, no preview) shown in its own inspector sub-panel
        kind, frame, preview = "analysis", True, False
    if kind not in ("plane", "image", "analysis"):
        raise ValueError(f"{id}: unknown op kind {kind!r} (plane, image, analysis or frame)")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = OpSpec(id=id, name=name, category=category, func=fn, kind=kind, description=description or (fn.__doc__ or "").strip(),
                      params=params or [], preview=preview, roi_aware=roi_aware, requires_image=requires_image,
                      requires_roi=requires_roi, shortcut=shortcut, menu=menu, keywords=keywords or [],
                      scope_param=scope_param, supports_rgb=supports_rgb, icon=icon, undo_label=undo_label,
                      tool=tool, tool_icon=tool_icon or ("⚙" if tool else ""), tool_cursor=tool_cursor or ("crosshair" if tool else ""),
                      tool_color=tool_color, frame=bool(frame))
        REGISTRY.register(spec)
        fn.spec = spec  # type: ignore[attr-defined]
        return fn
    return deco


@dataclass
class OpContext:
    session: "Session"
    image: "LumenImage | None"
    roi: dict[str, Any] | None = None
    scope: str = "stack"
    preview: bool = False
    _mask: np.ndarray | None = None
    #: the session's job record of the running command (progress + cancel flag); None outside ``Session.run_op``
    job: "dict[str, Any] | None" = None
    #: a command run by a script/batch inside another command: progress belongs to the outer job (cancel is shared)
    nested: bool = False

    def progress(self, fraction: float, message: str = "") -> None:
        """Report the progress (0–1) of the running command, with an optional status message.

        A no-op outside a run (previews, direct calls, nested commands), so ops can always call it.
        Events are throttled by the session to at most 20 per second.
        """
        if self.job is None or self.nested:
            return
        self.session._job_progress(self.job, fraction, message)

    def check_cancel(self) -> None:
        """Raise :class:`OperationCancelled` if the running job was cancelled; call it between planes / files / frames."""
        job = self.job
        if job is not None and job["cancel"].is_set():
            raise OperationCancelled()

    @property
    def img(self) -> "LumenImage":
        if self.image is None:
            raise ValueError("This command needs an open image")
        return self.image

    @property
    def mask(self) -> np.ndarray | None:
        """Boolean (H, W) mask of the active area ROI, or None."""
        from . import roi as roimod
        if self.roi is None or not roimod.is_area(self.roi) or self.image is None:
            return None
        if self._mask is None:
            self._mask = roimod.to_mask(self.roi, (self.image.height, self.image.width))
        return self._mask

    def planes(self) -> list[tuple[int, int, int]]:
        """(t, z, c) indices selected by scope."""
        img = self.img
        d = img.display
        if self.scope == "plane":
            if img.rgb:
                return [(d.t, d.z, c) for c in range(img.nc)]
            return [(d.t, d.z, d.c)]
        if self.scope == "slice":
            return [(d.t, d.z, c) for c in range(img.nc)]
        if self.scope == "channel":
            return [(t, z, d.c) for t in range(img.nt) for z in range(img.nz)]
        return list(img.iter_planes())

    def get_image(self, image_id_or_name: str) -> "LumenImage":
        return self.session.get_image(image_id_or_name)


def apply_plane_op(spec: OpSpec, ctx: OpContext, params: dict[str, Any]) -> np.ndarray:
    """Run a plane-kind op over the selected planes and return the new 5-D array."""
    img = ctx.img
    mask = ctx.mask if spec.roi_aware else None
    planes = ctx.planes()
    out = img.data.copy()
    out_dtype = None
    n = len(planes)
    for i, (t, z, c) in enumerate(planes):
        ctx.check_cancel()
        src = img.data[t, z, c]
        res = spec.func(src, ctx, **params)
        res = np.asarray(res)
        if res.shape != src.shape:
            raise ValueError(f"{spec.name} returned shape {res.shape}, expected {src.shape}")
        if out_dtype is None:
            out_dtype = res.dtype
            if res.dtype != out.dtype:
                if res.dtype == np.bool_:
                    out_dtype = np.uint8
                elif res.dtype not in (np.uint8, np.uint16, np.float32):
                    out_dtype = np.float32
                # keep the image dtype unless the op explicitly returns float32 for the full stack
                if set(planes) == set(img.iter_planes()) and out_dtype in (np.uint8, np.uint16, np.float32):
                    out = out.astype(out_dtype)
                else:
                    out_dtype = out.dtype
        res = cast_to(res, out.dtype)
        if mask is not None:
            out[t, z, c] = np.where(mask, res, out[t, z, c])
        else:
            out[t, z, c] = res
        if n > 1:
            ctx.progress((i + 1) / n, f"Plane {i + 1}/{n}")
    return out


def cast_to(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if arr.dtype == dtype:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * (255 if dtype == np.uint8 else 1)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.nan_to_num(np.rint(arr)), info.min, info.max).astype(dtype)
    return arr.astype(dtype)
