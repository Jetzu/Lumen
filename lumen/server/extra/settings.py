"""Preferences, recent files and bundled documentation (drop-in router).

``GET /api/settings``
    The JSON-serialisable, non-internal entries of ``session.settings`` with defaults filled in
    for the preference keys the Preferences sheet edits (``roi_color``, ``font_size`` …).

``PUT`` / ``PATCH`` / ``POST /api/settings``
    Merge a partial dict into ``session.settings`` and return the merged, visible settings.
    Every value is validated before anything is applied (a rejected request changes nothing):
    values must be JSON-serialisable, known preference keys must have the right type/range and
    internal keys cannot be written.  Special keys that are *not* stored:

    * ``{"recent_clear": true}`` clears the recent-files list (``session.clear_recent()``).
    * ``measurements`` / ``measure_options`` are applied exactly like the core
      ``POST /api/settings`` endpoint does (so this router can replace it).

    Keys no consumer reads lazily are *applied* as well as stored, exactly like the Edit ▸ Options
    commands do (``lumen.core.ops.system_ops``): ``undo_budget_mb`` → ``History.default_budget_bytes``
    and the budget of every open history (``apply_undo_budget``), ``max_threads`` → the BLAS / OpenMP
    pools via threadpoolctl (``apply_thread_limit``; stored only when threadpoolctl is missing) and
    ``proxy`` / ``no_proxy`` → ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` of this server process
    (``apply_proxy``, used by the URL / sample downloads).  ``proxy`` is canonicalised like
    ``edit.options_proxy`` does (``host:port`` → ``http://host:port``; whitespace is rejected) and
    ``no_proxy`` to a comma-separated list without blanks; an empty value removes the variables again.
    ``virtual_cache_mb`` resizes the shared plane cache of virtual stacks (``lumen.core.lazy.set_cache_budget``);
    ``auto_virtual_mb`` (default: half of the physical RAM, 0 = never) is the decoded size above which File ▸ Open
    opens a TIFF / HDF5 / Zarr / npy file as a virtual stack automatically.

    NOTE: ``lumen/server/app.py`` registers its own ``POST /api/settings`` (measurements only)
    *before* the drop-in routers are mounted, so a POST currently reaches that handler first.
    The frontend therefore uses PUT; remove the core route to let this one take over POST too.

``GET /api/docs``                 the bundled markdown documents and whether they are present
``GET /api/docs/{name}``          raw markdown of README | COMMANDS | FIJI_PARITY | RESEARCH | DESIGN

Internal, never exposed: ``clipboard``, ``classifier``, ``fft``, ``tracks``, ``spots``,
``deformation`` (and ``<key>_…`` companions such as ``tracks_image``), plus ``_``-prefixed keys.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ...core.lazy import DEFAULT_CACHE_MB, default_auto_virtual_mb, set_cache_budget
from ..app import get_session

router = APIRouter()

_INTERNAL = ("clipboard", "classifier", "fft", "tracks", "spots", "deformation")

_NUM = (int, float)
_STR = (str,)
_BOOL = (bool,)

#: preference keys edited by the Preferences sheet: default, accepted types, optional (lo, hi) range
PREFERENCES: dict[str, tuple[Any, tuple[type, ...], tuple[float, float] | None]] = {
    "roi_color": ("#ffd54a", _STR, None),
    "roi_stroke_width": (1.5, _NUM, (0.1, 50)),
    "font_size": (14, _NUM, (1, 500)),
    "font_family": ("", _STR, None),
    "foreground": (255, _NUM, None),
    "background": (0, _NUM, None),
    "jpeg_quality": (85, _NUM, (1, 100)),
    "default_save_format": ("tif", _STR, None),
    "point_size": (4, _NUM, (1, 100)),
    "point_color": ("#ffd54a", _STR, None),
    "arrow_style": ("Filled", _STR, None),
    "arrow_head_size": (12, _NUM, (1, 500)),
    "arrow_width": (2, _NUM, (0.1, 200)),
    "plot_y_min": (0, _NUM, None),
    "plot_y_max": (0, _NUM, None),
    "plot_interpolate": (True, _BOOL, None),
    "interpolate_zoom": (False, _BOOL, None),
    "antialiased_text": (True, _BOOL, None),
    "paste_mode": ("Copy", _STR, None),
    "rounded_rect_radius": (20, _NUM, (0, 10000)),
    "undo_budget_mb": (1500, _NUM, (50, 1000000)),
    "max_threads": (0, _NUM, (0, 4096)),
    "proxy": ("", _STR, None),
    "no_proxy": ("", _STR, None),
    "dicom_open_as_32bit": (False, _BOOL, None),
    "dicom_ignore_rescale": (False, _BOOL, None),
    "dicom_calibrate_images": (True, _BOOL, None),
    "dicom_window_level_on_open": (True, _BOOL, None),
    # GUI behaviour (Preferences ▸ Appearance / Saving & editing; docs/USABILITY.md M1 and M11)
    "confirm_close_modified": (True, _BOOL, None),   # ask before closing an image with unsaved changes
    "ui_density": ("comfortable", _STR, None),        # inspector / sheet spacing: "comfortable" | "compact"
    # Image ▸ Overlay ▸ Overlay Options… (edit.options_overlay): defaults for overlay items and their labels
    "overlay_stroke_width": (1.5, _NUM, (0.1, 50)),
    "overlay_default_fill": ("", _STR, None),  # "" / "none" = no default fill, else a hex colour
    "overlay_show_labels": (False, _BOOL, None),
    "overlay_use_names": (False, _BOOL, None),
    "overlay_label_font_size": (12, _NUM, (1, 500)),
    "overlay_label_color": ("#ffff00", _STR, None),
    "overlay_label_background": (False, _BOOL, None),
    # Data larger than RAM (docs/USABILITY.md M10, lumen.core.lazy): the shared plane cache of virtual stacks and the
    # decoded size above which File ▸ Open opens a file as a virtual stack automatically (0 = never)
    "virtual_cache_mb": (DEFAULT_CACHE_MB, _NUM, (16, 1000000)),
    "auto_virtual_mb": (default_auto_virtual_mb(), _NUM, (0, 100000000)),
}

DEFAULTS: dict[str, Any] = {k: v[0] for k, v in PREFERENCES.items()}

#: accepted values of ``ui_density`` (the frontend maps them onto ``<html data-density>``)
UI_DENSITIES = ("comfortable", "compact")

_COLOR_KEYS = {"roi_color", "point_color", "overlay_label_color"}
_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def is_internal(key: str) -> bool:
    """Keys that hold live objects (clipboard pixels, fitted models, track tables …) or are private."""
    return key.startswith("_") or any(key == k or key.startswith(k + "_") for k in _INTERNAL)


def _json_value(value: Any) -> tuple[bool, Any]:
    """(ok, value) with numpy scalars converted; ok is False for anything json cannot encode."""
    if hasattr(value, "item") and not isinstance(value, (list, dict, str, bytes)):
        try:  # numpy scalar
            value = value.item()
        except Exception:
            return False, None
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False, None
    return True, value


def visible_settings(settings: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(DEFAULTS)
    for k, v in settings.items():
        if not isinstance(k, str) or is_internal(k):
            continue
        ok, val = _json_value(v)
        if ok:
            out[k] = val
    return out


def _validate(key: str, value: Any) -> Any:
    """Return the value to store or raise HTTPException(400)."""
    if not isinstance(key, str) or not key:
        raise HTTPException(400, "setting names must be non-empty strings")
    if is_internal(key):
        raise HTTPException(400, f"'{key}' is an internal setting and cannot be changed through the API")
    ok, val = _json_value(value)
    if not ok:
        raise HTTPException(400, f"'{key}' must be a JSON-serialisable value")
    spec = PREFERENCES.get(key)
    if spec is None:
        return val
    _default, types, rng = spec
    if types == _BOOL:
        if not isinstance(val, bool):
            raise HTTPException(400, f"'{key}' must be true or false")
    elif types == _NUM:
        if isinstance(val, bool) or not isinstance(val, _NUM):
            raise HTTPException(400, f"'{key}' must be a number")
        if rng is not None and not (rng[0] <= val <= rng[1]):
            raise HTTPException(400, f"'{key}' must be between {rng[0]:g} and {rng[1]:g}")
    elif types == _STR:
        if not isinstance(val, str):
            raise HTTPException(400, f"'{key}' must be a string")
        if key in _COLOR_KEYS and not _COLOR_RE.match(val.strip()):
            raise HTTPException(400, f"'{key}' must be a hex colour like #ffd54a")
        if key == "overlay_default_fill":
            v = val.strip()
            if v.lower() in ("", "none"):
                return ""
            if not _COLOR_RE.match(v):
                raise HTTPException(400, "'overlay_default_fill' must be a hex colour like #ff000080 or 'none'")
            return v
        if key == "ui_density":
            v = val.strip().lower()
            if v not in UI_DENSITIES:
                raise HTTPException(400, f"'ui_density' must be one of {', '.join(UI_DENSITIES)}")
            return v
        if key == "paste_mode":
            from ...core.ops.edit_extra import PASTE_MODES
            low = {m.lower(): m for m in PASTE_MODES}
            if val.strip().lower() not in low:
                raise HTTPException(400, f"'paste_mode' must be one of {', '.join(PASTE_MODES)}")
            return low[val.strip().lower()]
        if len(val) > 500:
            raise HTTPException(400, f"'{key}' is too long")
        if key == "proxy":  # same canonical form as edit.options_proxy stores
            v = val.strip()
            if v and "://" not in v:
                v = "http://" + v
            if any(ch.isspace() for ch in v):
                raise HTTPException(400, f"'proxy' must be host:port or http://user:pass@host:port, got {v!r}")
            return v
        if key == "no_proxy":
            return ",".join(h.strip() for h in val.split(",") if h.strip())
    return val


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return visible_settings(get_session().settings)


def _apply_side_effects(session: Any, updates: dict[str, Any]) -> None:
    """Keys that nothing reads lazily: mirror them exactly like Edit ▸ Options ▸ Memory & Threads / Proxy Settings do
    (call with ``session.lock`` held – the undo budget walks ``session.histories``).  Uses the *merged* settings for the
    proxy pair, so a request carrying only ``no_proxy`` keeps the stored proxy and ``{"proxy": ""}`` (Reset to defaults)
    drops the environment variables again."""
    from ...core.ops.system_ops import apply_proxy, apply_thread_limit, apply_undo_budget  # lazy: keeps the router import-light
    if "undo_budget_mb" in updates:
        apply_undo_budget(session, updates["undo_budget_mb"])  # History.default_budget_bytes + every open history; stores the int
    if "max_threads" in updates:
        n = int(updates["max_threads"])
        session.settings["max_threads"] = n
        apply_thread_limit(n)
    if "proxy" in updates or "no_proxy" in updates:
        apply_proxy(str(session.settings.get("proxy") or ""), str(session.settings.get("no_proxy") or ""))
    if "virtual_cache_mb" in updates:
        set_cache_budget(updates["virtual_cache_mb"])  # the shared plane cache of every virtual stack (lumen.core.lazy)
    # auto_virtual_mb is read live by read_image(virtual="auto") through lumen.core.lazy.auto_virtual_bytes()


def _update(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(400, "expected a JSON object of settings")
    session = get_session()
    extra: dict[str, Any] = {}
    clear_recent = False
    measurements = None
    measure_options = None
    updates: dict[str, Any] = {}
    for key, value in body.items():
        if key == "recent_clear":
            clear_recent = bool(value)
        elif key == "measurements":
            if value is not None and not (isinstance(value, list) and all(isinstance(m, str) for m in value)):
                raise HTTPException(400, "'measurements' must be a list of measurement ids")
            measurements = value
        elif key == "measure_options":
            if value is not None and not isinstance(value, dict):
                raise HTTPException(400, "'measure_options' must be an object")
            measure_options = value
        else:
            updates[key] = _validate(key, value)
    # everything validated: apply atomically
    with session.lock:
        session.settings.update(updates)
        _apply_side_effects(session, updates)
        if measurements is not None:
            from ...core.measure import ALL_MEASUREMENTS
            session.measurements = [m for m in measurements if m in ALL_MEASUREMENTS]
            extra["measurements"] = session.measurements
        if measure_options:
            session.measure_options.update(measure_options)
            extra["measure_options"] = session.measure_options
        if clear_recent:
            session.clear_recent()
            extra["recent"] = session.recent
            session._changed("recent")
        if updates:
            session.version += 1
    return {**visible_settings(session.settings), **extra}


@router.put("/settings")
def put_settings(body: dict[str, Any]) -> dict[str, Any]:
    return _update(body)


@router.patch("/settings")
def patch_settings(body: dict[str, Any]) -> dict[str, Any]:
    return _update(body)


@router.post("/settings")
def post_settings(body: dict[str, Any]) -> dict[str, Any]:
    # shadowed by the core POST /api/settings while that route exists (see module docstring)
    return _update(body)


# ------------------------------------------------------------------ documentation
_HERE = Path(__file__).resolve()
_DOCS: dict[str, tuple[str, str]] = {  # name -> (title, relative path from the repository root)
    "README": ("Lumen README", "README.md"),
    "COMMANDS": ("Command reference", "docs/COMMANDS.md"),
    "FIJI_PARITY": ("Fiji parity", "docs/FIJI_PARITY.md"),
    "SCRIPTING": ("Scripting", "docs/SCRIPTING.md"),
    "MCP": ("AI assistants (MCP)", "docs/MCP.md"),
    "RESEARCH": ("Research notes", "docs/RESEARCH.md"),
    "DESIGN": ("Design notes", "docs/DESIGN.md"),
}
# the repository root (source checkout) and the package directory (a wheel that bundles the docs)
_DOC_ROOTS = [_HERE.parents[3], _HERE.parents[2], _HERE.parents[2] / "static"]


def _doc_path(name: str) -> Path | None:
    rel = _DOCS[name][1]
    for root in _DOC_ROOTS:
        for candidate in (root / rel, root / Path(rel).name, root / "docs" / Path(rel).name):
            if candidate.is_file():
                return candidate
    return None


def _doc_name(name: str) -> str:
    key = name.strip().upper()
    if key.endswith(".MD"):
        key = key[:-3]
    key = key.replace("-", "_")
    if key not in _DOCS:
        raise HTTPException(404, f"Unknown document {name!r}; one of {', '.join(_DOCS)}")
    return key


@router.get("/docs")
def list_docs() -> dict[str, Any]:
    return {"docs": [{"name": n, "title": t, "available": _doc_path(n) is not None} for n, (t, _p) in _DOCS.items()]}


@router.get("/docs/{name}")
def get_doc(name: str) -> PlainTextResponse:
    key = _doc_name(name)
    path = _doc_path(key)
    if path is None:
        raise HTTPException(404, f"{_DOCS[key][1]} is not bundled with this installation")
    text = path.read_text(encoding="utf-8", errors="replace")
    if key == "README":
        text = _plain_readme(text)
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


def _plain_readme(text: str) -> str:
    """The README for the in-app viewer: GitHub's HTML header (centred logo), the badges and the screenshots are left
    out – the viewer renders plain Markdown and the images are not bundled."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("<div", "</div", "<img", "<p align", "[![", "![")):
            continue
        keep.append(line)
    out = "\n".join(keep).lstrip("\n")
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out + "\n"
