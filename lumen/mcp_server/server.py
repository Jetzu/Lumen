"""lumen-mcp: an MCP server that drives the running Lumen app (the window shows everything it does).

Run it with ``lumen-mcp`` / ``python -m lumen.mcp_server`` (stdio), or from the desktop app as
``Lumen --mcp``.  See docs/MCP.md for the client configuration (Claude Code, Claude Desktop, Codex, …)."""
from __future__ import annotations

import difflib
import io
import json
import math
import re
from typing import Any

import numpy as np

from .client import LumenClient, LumenError
from .mcpcompat import FastMCP, Image

INSTRUCTIONS = """Lumen MCP server. You control a running Lumen (a modern ImageJ/Fiji: 500+ image-processing and analysis
commands, ImageJ macros, Python/Jython/Groovy scripts, ROI Manager, results tables, Fiji plugins).  Everything you do
appears live in the user's Lumen window, and you can see what the user is doing there.

Workflow:
1. `status` – is Lumen running, which images are open, what the user's window shows (active image, selection, tool).
   `launch` starts Lumen (the desktop app when installed).
2. `find_commands` (search by words, menu or ImageJ name) → `describe_command` (parameters, defaults, ranges) →
   `run_command` (any command; uses the user's current selection when the command works on a selection).
   `run_macro` runs ImageJ 1.x macro code (run("Gaussian Blur...", "sigma=2"); …), `run_script` Python/Jython
   (the `lumen` API with numpy access to the pixels, and ImageJ's ij.* classes) or Groovy/BeanShell/JavaScript via
   the Java bridge.  Installed Fiji plugins are commands too (Plugins ▸ Fiji; install with plugins.fiji_install).
3. Look: `view_image` (the rendered image with selection / ROI outlines, as the user sees it), `get_statistics`
   (numbers inside a selection or ROI), `get_pixels`, `image_info`.  Prefer numbers over eyeballing.
4. Images: `list_images`, `open_image`, `open_sample`, `save_image`, `close_image`, `activate_image`, `set_display`.
5. Selections & ROIs: `get_selection` / `set_selection` (ROI dicts: {"type":"rect","x","y","w","h"},
   {"type":"oval",…}, {"type":"polygon","points":[[x,y],…]}, {"type":"line","x1","y1","x2","y2"},
   {"type":"point","points":[…]}), `roi_manager` (list / add / delete / measure / combine / clear).
6. Results: `get_results`, `clear_results`; `history`, `undo`, `redo`; `recorded_macro` (what the user did, as a
   replayable script).  `events` – what changed since you last looked (the user may work in the window meanwhile).
7. The window: `ui` (show a panel, set the tool or plane, notify the user, open a command's dialog pre-filled so the
   user can review and apply it), `jobs` (progress / cancel), `settings`, `help` (Lumen's documentation).

Images are addressed by id, by name or by 1-based position in `list_images`; omitted = the active image.
Pixel coordinates are 0-based image pixels (x right, y down); planes t/z/c are 0-based.
Ask before destructive steps on the user's data (closing modified images, overwriting files).
"""

mcp = FastMCP("lumen", instructions=INSTRUCTIONS)
C = LumenClient()

_META: dict[str, Any] = {}  # url → cached op list
_LAST: dict[str, Any] = {}  # the events baseline


# ============================================================================ helpers
def _ui_state() -> dict[str, Any]:
    try:
        return C.get("/api/ui/state")
    except LumenError:
        return {"connected": False}


def _client() -> str | None:
    """The user's window id: commands the assistant runs land in the window's macro recorder (replayable)."""
    st = _ui_state()
    return st.get("client") if st.get("connected") else None


def _ops() -> list[dict[str, Any]]:
    url = C.ensure()
    if _META.get("url") != url or not _META.get("ops"):
        _META["url"] = url
        _META["ops"] = C.get("/api/ops")
    return _META["ops"]


def _norm(s: str) -> str:
    return re.sub(r"[\s_.…]+", " ", s.lower().replace("...", "")).strip()


def _resolve_op(command: str) -> dict[str, Any]:
    ops = _ops()
    by_id = {o["id"]: o for o in ops}
    if command in by_id:
        return by_id[command]
    _META["ops"] = None  # maybe a plugin was installed since: reload once
    ops = _ops()
    by_id = {o["id"]: o for o in ops}
    if command in by_id:
        return by_id[command]
    n = _norm(command)
    for o in ops:
        if _norm(o["name"]) == n or any(_norm(c) == n for c in (o.get("ij_commands") or [])):
            return o
    names = [o["id"] for o in ops] + [o["name"] for o in ops]
    close = difflib.get_close_matches(command, names, n=6, cutoff=0.5)
    raise LumenError(f"unknown command {command!r}." + (f" Did you mean: {', '.join(close)}?" if close else "") + " Use find_commands.")


def _state() -> dict[str, Any]:
    return C.get("/api/state")


def _resolve_image(image: str | int | None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    st = state or _state()
    imgs = st.get("images") or []
    if not imgs:
        raise LumenError("No image is open. Use open_image or open_sample.")
    if image is None or image == "":
        im = next((i for i in imgs if i["id"] == st.get("active")), imgs[-1])
        return im
    s = str(image)
    for i in imgs:
        if i["id"] == s:
            return i
    for i in imgs:
        if i["name"] == s:
            return i
    low = [i for i in imgs if i["name"].lower() == s.lower()] or [i for i in imgs if s.lower() in i["name"].lower()]
    if len(low) == 1:
        return low[0]
    if s.isdigit() and 1 <= int(s) <= len(imgs):
        return imgs[int(s) - 1]
    raise LumenError(f"no open image {image!r}; open: " + ", ".join(f"{k + 1}. {i['name']} ({i['id']})" for k, i in enumerate(imgs)))


def _shape(im: dict[str, Any]) -> str:
    s = im["shape"]
    dims = [f"{s['x']}×{s['y']}"]
    for k, lab in (("c", "channels"), ("z", "slices"), ("t", "frames")):
        if s[k] > 1:
            dims.append(f"{s[k]} {lab}")
    return ", ".join(dims)


def _img_line(im: dict[str, Any], active: str | None = None) -> str:
    cal = im.get("calibration") or {}
    scale = f", {cal.get('pixel_width'):g} {cal.get('unit')}/px" if cal.get("unit") not in (None, "", "pixel", "pixels") else ""
    bd = im.get("bit_depth")
    t = "RGB" if im.get("rgb") else (bd if isinstance(bd, str) else f"{bd}-bit") if bd else im.get("dtype")
    virt = ", virtual" if im.get("virtual") else ""
    return f"{'* ' if im['id'] == active else ''}{im['name']} [id {im['id']}] {_shape(im)}, {t}{scale}{virt}"


def _trim(obj: Any, max_items: int = 30, depth: int = 0) -> Any:
    """Shrink a result for the assistant: long lists cut (with a note), floats rounded."""
    if isinstance(obj, float):
        return obj if not math.isfinite(obj) else float(f"{obj:.6g}")
    if isinstance(obj, list):
        out = [_trim(v, max_items, depth + 1) for v in obj[:max_items]]
        if len(obj) > max_items:
            out.append(f"… {len(obj) - max_items} more")
        return out
    if isinstance(obj, dict):
        return {k: _trim(v, max_items, depth + 1) for k, v in obj.items()}
    return obj


def _table_text(table: dict[str, Any], max_rows: int = 50) -> str:
    cols = table.get("columns") or []
    rows = table.get("rows") or []

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6g}"
        return "" if v is None else str(v)
    lines = ["\t".join(str(c) for c in cols)]
    lines += ["\t".join(fmt(v) for v in r) for r in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"… {len(rows) - max_rows} more rows (get_results with max_rows / offset)")
    return "\n".join(lines)


def _show(action: str, args: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return C.post("/api/ui/command", {"action": action, "args": args})
    except LumenError:
        return None  # an older Lumen without the bridge: the engine state still updates the window


def _plane(im: dict[str, Any], t: int | None, z: int | None, c: int | None) -> tuple[np.ndarray, tuple[int, int, int]]:
    d = im.get("display") or {}
    t = d.get("t", 0) if t is None else t
    z = d.get("z", 0) if z is None else z
    c = d.get("c", 0) if c is None else c
    r = C.get(f"/api/images/{im['id']}/plane", t=t, z=z, c=c)
    shape = tuple(int(v) for v in r.headers["x-shape"].split(","))
    arr = np.frombuffer(r.content, dtype=np.dtype(r.headers["x-dtype"])).reshape(shape)
    return arr, (t, z, c)


def _mask(roi: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not roi:
        return None
    from lumen.core.roi import to_mask
    return to_mask(roi, shape)


def _selection_for(im_id: str) -> dict[str, Any] | None:
    st = _ui_state()
    return (st.get("selections") or {}).get(im_id) if st.get("connected") else None


def _roi_arg(roi: Any, im_id: str) -> dict[str, Any] | None:
    """`roi`: a ROI dict, "selection" (the user's current selection), a ROI Manager id / name, or None."""
    if roi is None or roi == "":
        return None
    if isinstance(roi, str):
        if roi.lower() in ("selection", "current", "sel"):
            sel = _selection_for(im_id)
            if not sel:
                raise LumenError("The image has no selection in the window.")
            return sel
        if roi.strip().startswith("{"):
            return json.loads(roi)
        for r in C.get("/api/rois"):
            if r.get("id") == roi or r.get("name") == roi:
                return r
        raise LumenError(f"no ROI {roi!r} in the ROI Manager")
    if isinstance(roi, dict):
        return roi
    raise LumenError("roi must be a ROI dict, 'selection' or a ROI Manager id / name")


# ============================================================================ connection
@mcp.tool()
def status() -> str:
    """Is Lumen running; open images; what the user's window shows (active image, selection, tool, panels); running jobs."""
    try:
        C.ensure()
    except LumenError as e:
        return f"Not connected: {e}"
    st = _state()
    ui = _ui_state()
    if _META.get("version_url") != C.url:
        _META["version"], _META["version_url"] = C.get("/api/meta").get("version", "?"), C.url
    lines = [f"Lumen {_META['version']} at {C.url} (found via {C.how})"]
    imgs = st.get("images") or []
    lines.append(f"{len(imgs)} image(s) open" + (":" if imgs else " (open_image / open_sample)"))
    lines += ["  " + _img_line(i, st.get("active")) for i in imgs]
    if ui.get("connected"):
        a = ui.get("active_image") or {}
        sel = (ui.get("selections") or {}).get(a.get("id")) if a else None
        lines.append(f"Window: tool={ui.get('tool')}, plane t={a.get('t')} z={a.get('z')} c={a.get('c')}" if a else f"Window: tool={ui.get('tool')}")
        lines.append("  selection: " + (json.dumps(_trim(sel, 12)) if sel else "none"))
        lines.append("  panels open: " + ", ".join(ui.get("panels_open") or []))
        if ui.get("sheet"):
            lines.append(f"  dialog open: {ui['sheet']['name']} ({ui['sheet']['op']}) params={json.dumps(_trim(ui['sheet']['params']))}")
        if len(ui.get("windows") or []) > 1:
            lines.append(f"  {len(ui['windows'])} windows connected (the most recently active one is described)")
    else:
        lines.append("No window is connected (headless engine, or the window has not reported yet).")
    jobs = C.get("/api/jobs")
    if jobs:
        lines.append("Running: " + "; ".join(f"{j['name']} {int(100 * (j.get('progress') or 0))}% [job {j['id']}]" for j in jobs))
    res = st.get("results") or {}
    lines.append(f"Results table: {len(res.get('rows') or [])} rows; ROI Manager: {len(st.get('rois') or [])} ROIs")
    return "\n".join(lines)


@mcp.tool()
def launch(mode: str = "auto") -> str:
    """Start Lumen if it is not running. mode: auto (the desktop app when installed, else the engine + a browser
    window), desktop, browser, headless (engine only, no window)."""
    r = C.launch(mode)
    _META.clear()
    return json.dumps(r)


# ============================================================================ commands
@mcp.tool()
def find_commands(query: str = "", menu: str = "", limit: int = 25) -> str:
    """Search Lumen's commands by words (name, id, description, keywords, ImageJ command names) and/or menu path
    (e.g. "Process/Filters", "Analyze", "Plugins/MorphoLibJ").  Empty query + menu lists that menu."""
    ops = _ops()
    words = [w for w in _norm(query).split() if w]
    scored = []
    for o in ops:
        if menu and not (o.get("menu") or "").lower().startswith(menu.lower().strip("/")):
            continue
        hay = " ".join([o["id"], o["name"], o.get("description") or "", " ".join(o.get("keywords") or []), " ".join(o.get("ij_commands") or []), o.get("menu") or ""]).lower()
        if words and not all(w in hay for w in words):
            continue
        score = sum(3 if w in o["name"].lower() else 1 for w in words)
        scored.append((-score, o.get("menu") or "", o["name"], o))
    scored.sort(key=lambda s: s[:3])
    if not scored:
        return "No command matches. Try fewer or other words (find_commands with menu='Process' lists a menu)."
    out = [f"{len(scored)} command(s)" + (f", first {limit}" if len(scored) > limit else "") + ":"]
    for *_, o in scored[:limit]:
        params = ", ".join(p["name"] for p in o.get("params") or [])
        need = f" [needs {o['requires_roi']} selection]" if o.get("requires_roi") else ""
        desc = (o.get("description") or "").split("\n")[0][:110]
        out.append(f"- {o['id']} – {o['name']} ({o.get('menu')}){need}: {desc}" + (f" | params: {params}" if params else ""))
    return "\n".join(out)


@mcp.tool()
def describe_command(command: str) -> str:
    """Everything about one command: what it does, its parameters (type, default, range, choices), selection needs,
    the ImageJ command names it answers to."""
    o = _resolve_op(command)
    out = [f"{o['name']}  (id {o['id']}, menu {o.get('menu')}, kind {o.get('kind')})", o.get("description") or ""]
    if o.get("requires_roi"):
        out.append(f"Needs a {o['requires_roi']} selection (set_selection, or the user's selection in the window).")
    if o.get("ij_commands"):
        out.append("ImageJ: " + ", ".join(f'run("{c}")' for c in o["ij_commands"]))
    if o.get("kind") == "plane":
        out.append("scope: 'stack' (every plane, default), 'slice' (the current plane only), 'channel'.")
    out.append("Parameters:" if o.get("params") else "No parameters.")
    for p in o.get("params") or []:
        bits = [p.get("type", "")]
        if p.get("default") is not None:
            bits.append(f"default {json.dumps(p['default'])}")
        if p.get("min") is not None or p.get("max") is not None:
            bits.append(f"range {p.get('min', '')}..{p.get('max', '')}")
        if p.get("unit"):
            bits.append(p["unit"])
        if p.get("choices"):
            bits.append("choices " + json.dumps(p["choices"]))
        out.append(f"- {p['name']}: {', '.join(bits)}" + (f" – {p.get('label')}" if p.get("label") and p.get("label") != p["name"] else "") + (f". {p['help']}" if p.get("help") else ""))
    return "\n".join(out)


def _summarize_result(res: dict[str, Any], st_after: dict[str, Any] | None = None) -> str:
    out = [f"OK {res.get('op')} ({res.get('elapsed_ms', 0)} ms)"]
    if res.get("new_images"):
        imgs = {i["id"]: i for i in (st_after or _state()).get("images", [])}
        out.append("New image(s): " + "; ".join(_img_line(imgs[i]) if i in imgs else i for i in res["new_images"]))
    if res.get("message"):
        out.append(f"Message: {res['message']}")
    if res.get("log"):
        out.append(f"Log:\n{str(res['log'])[:4000]}")
    if res.get("results_added"):
        out.append(f"{res['results_added']} row(s) added to the Results table (get_results).")
    if res.get("table"):
        t = res["table"]
        out.append(f"Table '{t.get('name', 'Table')}' ({len(t.get('rows') or [])} rows):\n" + _table_text(t, 30))
    if res.get("roi") is not None:
        out.append("Selection set: " + json.dumps(_trim(res["roi"], 12)))
    if res.get("clear_roi"):
        out.append("Selection cleared.")
    if res.get("rois_added"):
        out.append(f"{res['rois_added']} ROI(s) added to the ROI Manager.")
    if res.get("threshold"):
        out.append("Threshold: " + json.dumps(_trim(res["threshold"])))
    if res.get("plot"):
        p = res["plot"]
        series = p.get("series") or ([{"x": p.get("x"), "y": p.get("y")}] if p.get("x") is not None else [])
        out.append(f"Plot '{p.get('title', '')}' (shown in the Plot panel): " + json.dumps(_trim({k: v for k, v in p.items() if k not in ("series",)}, 20))[:1500]
                   + (f" + {len(series)} series" if p.get("series") else ""))
    if res.get("histogram"):
        h = res["histogram"]
        out.append("Histogram: " + json.dumps(_trim({k: v for k, v in h.items() if k not in ("counts", "edges")}))[:800])
    extra = {k: v for k, v in res.items() if k not in ("op", "elapsed_ms", "new_images", "message", "log", "results_added", "table", "roi", "clear_roi", "rois_added", "threshold", "plot", "histogram", "preview")}
    if extra:
        out.append("Other: " + json.dumps(_trim(extra, 20))[:2000])
    return "\n".join(out)


@mcp.tool()
def run_command(command: str, params: dict[str, Any] | None = None, image: str | None = None, roi: Any = None,
                scope: str = "stack", use_selection: bool = True) -> str:
    """Run any Lumen command (id like "filters.gaussian", its name "Gaussian Blur…", or an ImageJ command name).
    params: the command's parameters (describe_command); omitted ones take their defaults.
    image: id / name / position (default: active).  roi: a ROI dict, "selection", or a ROI Manager id/name;
    when omitted and use_selection is true, the user's current selection in the window is used (as when the user
    runs it).  scope (plane filters on stacks): "stack" (all planes), "slice" (the current plane), "channel".
    The window shows the result (new images, tables, plots, selection) at once."""
    o = _resolve_op(command)
    st = _state()
    im = None
    if o.get("requires_image", True) or image:
        try:
            im = _resolve_image(image, st)
        except LumenError:
            if o.get("requires_image", True):
                raise
    r = _roi_arg(roi, im["id"]) if im else None
    if r is None and im is not None and use_selection and (o.get("roi_aware") or o.get("requires_roi")):
        r = _selection_for(im["id"])
    if o.get("requires_roi") and r is None:
        raise LumenError(f"{o['name']} needs a {o['requires_roi']} selection: pass roi=… or set_selection first.")
    res = C.post(f"/api/ops/{o['id']}/run", {"params": params or {}, "image_id": im["id"] if im else None, "roi": r, "scope": scope}, client=_client())
    _show("apply_result", {"result": res, "image_id": im["id"] if im else None})
    return _summarize_result(res)


@mcp.tool()
def run_macro(code: str, image: str | None = None) -> str:
    """Run ImageJ 1.x macro code (the macro language of ImageJ/Fiji: run("Gaussian Blur...", "sigma=2");
    setAutoThreshold("Otsu dark"); run("Analyze Particles...", "display"); print(nResults); …) on the active (or
    given) image, with the user's selection.  Returns the macro's log / message."""
    st = _state()
    im = _resolve_image(image, st) if (st.get("images") and (image or st.get("active"))) else None
    roi = _selection_for(im["id"]) if im else None
    res = C.post("/api/ops/script.run_ij_macro/run", {"params": {"macro": code}, "image_id": im["id"] if im else None, "roi": roi, "scope": "stack"}, client=_client())
    _show("apply_result", {"result": res, "image_id": im["id"] if im else None})
    return _summarize_result(res)


@mcp.tool()
def run_script(code: str, language: str = "python", params: dict[str, Any] | None = None) -> str:
    """Run a script in Lumen.  language: python / jython – the `lumen` API: lumen.run(id, **params), lumen.image (the
    active image: .name, .data (T,Z,C,Y,X numpy)), lumen.data / lumen.plane (numpy), lumen.update(array),
    lumen.new_image(array, name), lumen.results(), lumen.rois(), lumen.select(name), lumen.help(id) – plus numpy /
    scipy / skimage and ImageJ's ij.* classes (IJ.run(imp, "Median...", "radius=2")),
    or groovy / beanshell / rhino / jruby / clojure through the Java bridge with the real ImageJ.  params fill the
    script's #@ parameters.  Returns stdout, errors and new images."""
    st = _state()
    roi = None
    if st.get("active"):
        roi = _selection_for(st["active"])
    res = C.post("/api/script/run", {"code": code, "language": language, "params": params, "roi": roi}, client=_client())
    out = []
    if res.get("stdout"):
        out.append(str(res["stdout"])[:8000])
    if res.get("error"):
        out.append("ERROR:\n" + str(res["error"])[:4000])
    if res.get("new_images"):
        out.append(f"New image(s): {res['new_images']}")
    if res.get("roi") is not None and st.get("active"):
        _show("set_selection", {"image_id": st["active"], "roi": res["roi"]})
        out.append("Selection set by the script.")
    rest = {k: v for k, v in res.items() if k not in ("stdout", "error", "new_images", "roi")}
    if rest:
        out.append(json.dumps(_trim(rest))[:2000])
    return "\n".join(out) or "OK (no output)"


# ============================================================================ images
@mcp.tool()
def list_images() -> str:
    """The open images (* = active), with id, size, type and calibration."""
    st = _state()
    imgs = st.get("images") or []
    if not imgs:
        return "No images open."
    return "\n".join(f"{k + 1}. " + _img_line(i, st.get("active")) for k, i in enumerate(imgs))


@mcp.tool()
def image_info(image: str | None = None) -> str:
    """Details of an image: dimensions, type, calibration, display (channels, LUTs, contrast), value range,
    file path, metadata, undo history."""
    st = _state()
    im = _resolve_image(image, st)
    d = im.get("display") or {}
    out = [_img_line(im, st.get("active")), f"path: {im.get('path') or '(not saved)'}", f"value range: {im.get('value_range')}",
           f"display: mode={d.get('mode')}, plane t={d.get('t')} z={d.get('z')} c={d.get('c')}"]
    for k, ch in enumerate(d.get("channels") or []):
        out.append(f"  channel {k + 1}{' ' + ch['name'] if ch.get('name') else ''}: lut={ch.get('lut')} min={ch.get('min')} max={ch.get('max')} visible={ch.get('visible')}")
    out.append("calibration: " + json.dumps(im.get("calibration")))
    md = im.get("metadata") or {}
    if md:
        out.append("metadata: " + json.dumps(_trim(dict(list(md.items())[:40])))[:3000])
    if im.get("overlay"):
        out.append(f"overlay: {len(im['overlay']) if isinstance(im['overlay'], list) else 'yes'} item(s)")
    h = (st.get("history") or {}).get(im["id"]) or {}
    if h:
        out.append(f"undo: {h.get('undo')}  redo: {h.get('redo')}")
    if im["id"] in (st.get("thresholds") or {}):
        out.append("threshold: " + json.dumps(st["thresholds"][im["id"]]))
    return "\n".join(out)


@mcp.tool()
def open_image(path: str, virtual: bool | None = None, series: int | None = None) -> str:
    """Open an image file (TIFF/OME-TIFF, PNG, JPEG, ND2, CZI, LIF, DICOM, NIfTI, HDF5, Zarr, … – every format Lumen
    reads) in the window.  virtual=True opens planes on demand (data larger than RAM); series picks one series of a
    multi-series file."""
    body: dict[str, Any] = {"path": path}
    if virtual is not None:
        body["virtual"] = virtual
    if series is not None:
        body["series"] = series
    r = C.post("/api/images/open", body, client=_client())
    st = _state()
    ids = r.get("ids") or ([r["id"]] if r.get("id") else []) or [st.get("active")]
    imgs = {i["id"]: i for i in st.get("images", [])}
    return "Opened: " + "; ".join(_img_line(imgs[i]) for i in ids if i in imgs) if ids else json.dumps(_trim(r))


@mcp.tool()
def open_sample(name: str = "Blobs") -> str:
    """Open one of Lumen's sample images: Blobs, Cells (3-D, 2 channels), Coins, Astronaut (RGB), Cell nuclei
    (fluorescence), Human mitosis, Moon, T1 head (MRI stack), Microaneurysms (retina), Synthetic nuclei (3 channels),
    Nuclei timelapse (synthetic)."""
    return run_command("file.sample", {"name": name}, use_selection=False)


@mcp.tool()
def save_image(path: str, image: str | None = None, format: str | None = None) -> str:
    """Save an image to a file (the format follows the extension: .tif/.ome.tif keep everything, .png/.jpg export the
    rendered view, .csv …).  Ask the user before overwriting an existing file."""
    im = _resolve_image(image)
    body: dict[str, Any] = {"path": path}
    if format:
        body["format"] = format
    r = C.post(f"/api/images/{im['id']}/save", body, client=_client())
    return f"Saved {im['name']} → {r.get('path', path)}"


@mcp.tool()
def close_image(image: str | None = None) -> str:
    """Close an image (unsaved changes are lost – ask the user first when it was modified)."""
    im = _resolve_image(image)
    C.delete(f"/api/images/{im['id']}")
    return f"Closed {im['name']}"


@mcp.tool()
def activate_image(image: str) -> str:
    """Make an image the active one (its tab comes to the front in the window)."""
    im = _resolve_image(image)
    C.post(f"/api/images/{im['id']}/select")
    _show("select_image", {"image_id": im["id"]})
    return f"Active: {im['name']}"


def _draw_rois(png: bytes, rois: list[tuple[dict[str, Any], tuple[int, int, int]]], scale: float) -> bytes:
    from PIL import Image as PILImage, ImageDraw
    im = PILImage.open(io.BytesIO(png)).convert("RGB")
    if scale != 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), PILImage.Resampling.LANCZOS if scale < 1 else PILImage.Resampling.NEAREST)
    dr = ImageDraw.Draw(im)
    for roi, color in rois:
        t = roi.get("type")
        s = scale
        try:
            if t in ("rect", "oval", "mask"):
                box = [roi["x"] * s, roi["y"] * s, (roi["x"] + roi["w"]) * s, (roi["y"] + roi["h"]) * s]
                (dr.ellipse if t == "oval" else dr.rectangle)(box, outline=color, width=2)
            elif t in ("polygon", "freehand", "traced"):
                pts = [(x * s, y * s) for x, y in roi.get("points") or []]
                if len(pts) > 1:
                    dr.polygon(pts, outline=color, width=2)
            elif t in ("polyline",):
                pts = [(x * s, y * s) for x, y in roi.get("points") or []]
                if len(pts) > 1:
                    dr.line(pts, fill=color, width=2)
            elif t == "line":
                dr.line([(roi["x1"] * s, roi["y1"] * s), (roi["x2"] * s, roi["y2"] * s)], fill=color, width=2)
            elif t == "point":
                for x, y in roi.get("points") or []:
                    dr.ellipse([x * s - 3, y * s - 3, x * s + 3, y * s + 3], outline=color, width=2)
        except Exception:  # noqa: BLE001
            continue
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@mcp.tool()
def view_image(image: str | None = None, t: int | None = None, z: int | None = None, max_size: int = 1024,
               show_selection: bool = True, show_rois: bool = True):
    """The image as the user sees it (LUTs, contrast, composite channels) at plane t/z (default: the shown plane),
    with the window's selection (yellow) and the image's ROI Manager entries (cyan) drawn on top.
    max_size limits the longer side (pixels)."""
    st = _state()
    im = _resolve_image(image, st)
    r = C.get(f"/api/images/{im['id']}/rgb.png", t=t, z=z)
    s = im["shape"]
    scale = min(1.0, max_size / max(s["x"], s["y"])) if max_size else 1.0
    if s["x"] < 256 and s["y"] < 256:
        scale = min(4.0, 512 / max(s["x"], s["y"]))  # tiny images: enlarge so pixels stay visible
    rois: list = []
    if show_rois:
        rois += [(x, (0, 220, 255)) for x in (st.get("rois") or []) if x.get("image_id") in (None, im["id"])]
    if show_selection:
        sel = _selection_for(im["id"])
        if sel:
            rois.append((sel, (255, 230, 0)))
    png = _draw_rois(r.content, rois, scale)
    d = im.get("display") or {}
    text = (f"{_img_line(im, st.get('active'))}; plane t={d.get('t') if t is None else t} z={d.get('z') if z is None else z}; "
            f"shown at {scale:.3g}× ({len(rois)} outline(s)); channels: "
            + ", ".join(f"{ch.get('lut')} [{ch.get('min'):g}, {ch.get('max'):g}]" for ch in d.get("channels") or [] if ch.get("min") is not None))
    return [text, Image(data=png, format="png")]


@mcp.tool()
def get_statistics(image: str | None = None, roi: Any = "selection_or_image", t: int | None = None, z: int | None = None,
                   c: int | None = None, all_channels: bool = False, bins: int = 16) -> str:
    """Pixel statistics (count, area, mean, std, min, max, median, percentiles, histogram) of one plane, inside a
    region: roi = "selection_or_image" (default: the window's selection, else the whole image), "image",
    "selection", a ROI dict, or a ROI Manager id/name.  Does not touch the Results table."""
    st = _state()
    im = _resolve_image(image, st)
    if roi == "selection_or_image":
        region = _selection_for(im["id"])
    elif roi in ("image", None, ""):
        region = None
    else:
        region = _roi_arg(roi, im["id"])
    chans = range(im["shape"]["c"]) if all_channels else [c]
    cal = im.get("calibration") or {}
    px_area = float(cal.get("pixel_width") or 1) * float(cal.get("pixel_height") or 1)
    unit = cal.get("unit") or "pixel"
    out = [f"{im['name']} – region: {json.dumps(_trim(region, 8)) if region else 'whole image'}"]
    for ch in chans:
        arr, (tt, zz, cc) = _plane(im, t, z, ch)
        if arr.ndim == 3:  # RGB plane: luminance like ImageJ's default
            arr = arr[..., :3].astype(np.float64) @ np.array([1 / 3, 1 / 3, 1 / 3])
        m = _mask(region, arr.shape[:2])
        vals = arr[m] if m is not None else arr.ravel()
        vals = vals[np.isfinite(vals)] if vals.dtype.kind == "f" else vals
        if vals.size == 0:
            out.append(f"plane t={tt} z={zz} c={cc}: empty region")
            continue
        v = vals.astype(np.float64)
        p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        hist, edges = np.histogram(v, bins=max(2, min(bins, 256)))
        out.append(f"plane t={tt} z={zz} c={cc}: n={v.size} area={v.size * px_area:.6g} {unit}² mean={v.mean():.6g} std={v.std(ddof=1) if v.size > 1 else 0:.6g} "
                   f"min={v.min():.6g} max={v.max():.6g} sum={v.sum():.6g}\n  percentiles 1/5/25/50/75/95/99: " + " ".join(f"{x:.6g}" for x in p)
                   + "\n  histogram: " + " ".join(f"[{edges[k]:.4g}..{edges[k + 1]:.4g}):{hist[k]}" for k in range(len(hist))))
    return "\n".join(out)


@mcp.tool()
def get_pixels(x: int, y: int, width: int = 1, height: int = 1, image: str | None = None, t: int | None = None,
               z: int | None = None, c: int | None = None) -> str:
    """Raw pixel values of a small region (at most 64×64) of one plane; rows top to bottom."""
    if width * height > 4096:
        raise LumenError("at most 4096 pixels (e.g. 64×64); use get_statistics for larger regions")
    im = _resolve_image(image)
    arr, (tt, zz, cc) = _plane(im, t, z, c)
    sub = arr[max(0, y):y + height, max(0, x):x + width]
    if sub.size == 0:
        raise LumenError(f"the region is outside the image ({im['shape']['x']}×{im['shape']['y']})")
    rows = sub.tolist()
    return f"{im['name']} plane t={tt} z={zz} c={cc}, x={x}..{x + sub.shape[1] - 1}, y={y}..{y + sub.shape[0] - 1}, dtype {arr.dtype}:\n" + "\n".join(
        " ".join(str(v) if not isinstance(v, float) else f"{v:.6g}" for v in (r if isinstance(r, list) else [r])) for r in rows)


@mcp.tool()
def set_display(image: str | None = None, channel: int | None = None, min: float | None = None, max: float | None = None,
                lut: str | None = None, visible: bool | None = None, gamma: float | None = None, mode: str | None = None,
                auto: bool = False, saturated: float = 0.35, t: int | None = None, z: int | None = None, c: int | None = None) -> str:
    """Change how an image is shown (never its pixel data): contrast min/max (or auto=True with `saturated` % clipped,
    like ImageJ's Auto), LUT (Grays, Fire, Green, Magenta, mpl-viridis, …), channel visibility, gamma, composite mode
    (composite | single | grayscale) and the shown plane t/z/c.  channel is 1-based (default: the current channel)."""
    st = _state()
    im = _resolve_image(image, st)
    d = im.get("display") or {}
    patch: dict[str, Any] = {k: v for k, v in (("t", t), ("z", z), ("c", c), ("mode", mode)) if v is not None}
    ci = (channel - 1) if channel else int(d.get("c", 0))
    if not 0 <= ci < im["shape"]["c"]:
        raise LumenError(f"channel {channel} does not exist (1..{im['shape']['c']})")
    chp: dict[str, Any] = {k: v for k, v in (("min", min), ("max", max), ("lut", lut), ("visible", visible), ("gamma", gamma)) if v is not None}
    if auto:
        arr, _ = _plane(im, t, z, ci)
        v = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
        lo, hi = np.percentile(v, [saturated / 2, 100 - saturated / 2])
        chp["min"], chp["max"] = float(lo), float(hi)
    if chp:
        patch["channels"] = [{} for _ in range(ci)] + [chp]
    if not patch:
        return "Nothing to change."
    r = C.post(f"/api/images/{im['id']}/display", patch, client=_client())
    ch = (r.get("channels") or [{}])[ci] if r.get("channels") else {}
    return f"{im['name']}: mode={r.get('mode')}, plane t={r.get('t')} z={r.get('z')} c={r.get('c')}, channel {ci + 1}: lut={ch.get('lut')} min={ch.get('min')} max={ch.get('max')} visible={ch.get('visible')}"


# ============================================================================ selections & ROIs
@mcp.tool()
def get_selection(image: str | None = None) -> str:
    """The selection the user has drawn on an image in the window (a ROI dict), or none."""
    im = _resolve_image(image)
    ui = _ui_state()
    if not ui.get("connected"):
        return "No window is connected, so there is no on-screen selection."
    sel = (ui.get("selections") or {}).get(im["id"])
    return json.dumps(sel) if sel else f"No selection on {im['name']}."


@mcp.tool()
def set_selection(roi: dict[str, Any] | None, image: str | None = None) -> str:
    """Draw a selection on an image in the window (None clears it).  ROI dicts: {"type":"rect","x","y","w","h"},
    {"type":"oval","x","y","w","h"}, {"type":"polygon","points":[[x,y],…]}, {"type":"freehand",…},
    {"type":"line","x1","y1","x2","y2","width"}, {"type":"polyline","points":…}, {"type":"point","points":…}."""
    im = _resolve_image(image)
    r = _show("set_selection", {"image_id": im["id"], "roi": roi})
    if not r or not r.get("windows"):
        return "No window is connected: pass the ROI to run_command / get_statistics as roi=… instead."
    import time
    for _ in range(20):
        ui = _ui_state()
        if (ui.get("seq_applied") or 0) >= r["seq"]:
            if ui.get("last_error"):
                return f"The window refused it: {ui['last_error']}"
            break
        time.sleep(0.1)
    return f"Selection on {im['name']}: {json.dumps(roi) if roi else 'cleared'}"


@mcp.tool()
def roi_manager(action: str = "list", ids: list[str] | None = None, roi: Any = None, name: str | None = None,
                image: str | None = None, mode: str = "or") -> str:
    """The ROI Manager.  action: list | add (roi = a ROI dict or "selection" (default), optional name) |
    delete (ids) | clear | measure (ids, default all → Results table) | combine (ids, mode or|and|xor → the
    selection) | rename (ids=[one id], name)."""
    action = action.lower()
    if action == "list":
        rois = C.get("/api/rois")
        if not rois:
            return "The ROI Manager is empty."
        return "\n".join(f"{k + 1}. {r.get('name') or ''} [id {r.get('id')}] {r.get('type')} image={r.get('image_id')} pos={r.get('position')}" for k, r in enumerate(rois[:200])) + (f"\n… {len(rois) - 200} more" if len(rois) > 200 else "")
    if action == "add":
        im = _resolve_image(image)
        r = _roi_arg(roi if roi is not None else "selection", im["id"])
        if name:
            r = {**r, "name": name}
        d = im.get("display") or {}
        out = C.post("/api/rois", {"roi": r, "image_id": im["id"], "position": {"t": d.get("t", 0), "z": d.get("z", 0), "c": d.get("c", 0)}})
        _show("panel", {"panel": "rois", "open": True})
        return f"Added ROI {out.get('id', '')} ({r.get('type')})"
    if action == "delete":
        if not ids:
            raise LumenError("ids required")
        C.post("/api/rois/delete", {"ids": ids})
        return f"Deleted {len(ids)} ROI(s)"
    if action == "clear":
        C.delete("/api/rois")
        return "ROI Manager cleared (the user can restore it with Undo clear in the window)."
    if action == "measure":
        ids = ids or [r["id"] for r in C.get("/api/rois")]
        out = C.post("/api/rois/measure", {"ids": ids}, client=_client())
        _show("panel", {"panel": "results", "open": True})
        return f"Measured {len(ids)} ROI(s): {out.get('rows', '?')} row(s) added to the Results table (get_results)."
    if action == "combine":
        if not ids:
            raise LumenError("ids required")
        im = _resolve_image(image)
        out = C.post("/api/rois/combine", {"ids": ids, "mode": mode, "image_id": im["id"]})
        roi_out = out if isinstance(out, dict) and out.get("type") else None
        if roi_out:
            _show("set_selection", {"image_id": im["id"], "roi": roi_out})
        return f"Combined ({mode}) → selection: " + json.dumps(_trim(roi_out or out, 10))[:2000]
    if action == "rename":
        if not ids or len(ids) != 1 or not name:
            raise LumenError("rename needs ids=[one id] and name")
        C.patch(f"/api/rois/{ids[0]}", {"patch": {"name": name}})
        return f"Renamed {ids[0]} → {name}"
    raise LumenError("action: list, add, delete, clear, measure, combine or rename")


# ============================================================================ results, history, recorder
@mcp.tool()
def get_results(table: str = "Results", max_rows: int = 100, offset: int = 0) -> str:
    """The Results table (Measure, Analyze Particles, …) or another named table a command produced."""
    if table == "Results":
        t = C.get("/api/results")
    else:
        tables = _state().get("tables") or {}
        if table not in tables:
            return f"No table {table!r}; tables: Results" + "".join(f", {k}" for k in tables)
        t = tables[table]
    rows = t.get("rows") or []
    t = {**t, "rows": rows[offset:]}
    return f"{table}: {len(rows)} rows × {len(t.get('columns') or [])} columns" + (f" (from row {offset})" if offset else "") + "\n" + _table_text(t, max_rows)


@mcp.tool()
def clear_results() -> str:
    """Empty the Results table (the user can restore it from the window)."""
    C.delete("/api/results")
    return "Results cleared."


@mcp.tool()
def history(image: str | None = None) -> str:
    """An image's undo / redo steps (newest last) and the session log's last entries."""
    st = _state()
    im = _resolve_image(image, st)
    h = (st.get("history") or {}).get(im["id"]) or {}
    log = st.get("log") or []
    return (f"{im['name']}\nundo: {json.dumps(h.get('undo'))}\nredo: {json.dumps(h.get('redo'))}\nlog (last 15):\n"
            + "\n".join(f"  {e.get('level')}: {e.get('msg')}" for e in log[-15:]))


@mcp.tool()
def undo(image: str | None = None) -> str:
    """Undo the last change of an image (pixel edits and display changes)."""
    im = _resolve_image(image)
    r = C.post(f"/api/images/{im['id']}/undo")
    return f"Undid: {r.get('label') or 'nothing to undo'}"


@mcp.tool()
def redo(image: str | None = None) -> str:
    """Redo the last undone change of an image."""
    im = _resolve_image(image)
    r = C.post(f"/api/images/{im['id']}/redo")
    return f"Redid: {r.get('label') or 'nothing to redo'}"


@mcp.tool()
def recorded_macro(language: str = "python") -> str:
    """What has been done in the user's window (and by you) since the recorder started, as a replayable script:
    language "python" (lumen.run lines) or "ijm" (an ImageJ macro)."""
    client = _client()
    if language == "ijm":
        r = C.get("/api/recorder/ijm", client=client)
        notes = "".join(f"\n// skipped: {s.get('line')} ({s.get('reason')})" for s in r.get("skipped") or [])
        return (r.get("script") or "// nothing recorded") + notes
    r = C.get("/api/recorder", client=client)
    lines = r.get("lines") or []
    return f"{r.get('marker', '')}\n" + "\n".join(lines) if lines else "The recorder is empty."


@mcp.tool()
def events() -> str:
    """What changed since you last called events (or status): images opened / closed / modified, the active
    image, new Results rows, ROI Manager changes, log entries, the user's selection / tool / dialog, new recorded
    steps.  Call it to notice what the user did in the window."""
    st = _state()
    ui = _ui_state()
    now = {
        "images": {i["id"]: (i["name"], i.get("rev")) for i in st.get("images") or []},
        "active": st.get("active"),
        "results": len((st.get("results") or {}).get("rows") or []),
        "rois": len(st.get("rois") or []),
        "log": [f"{e.get('t')}|{e.get('msg')}" for e in st.get("log") or []],
        "tool": ui.get("tool"),
        "selection": json.dumps((ui.get("selections") or {}).get(st.get("active") or "")),
        "sheet": (ui.get("sheet") or {}).get("op"),
        "recorder": (st.get("recorder") or {}).get("count"),
    }
    prev = dict(_LAST)
    _LAST.clear()
    _LAST.update(now)
    if not prev:
        return "Baseline taken (call events again later to see changes). " + f"{len(now['images'])} image(s) open, {now['results']} result rows, {now['rois']} ROIs."
    out = []
    for i, (n, rev) in now["images"].items():
        if i not in prev["images"]:
            out.append(f"opened: {n} [{i}]")
        elif prev["images"][i][1] != rev:
            out.append(f"modified: {n} [{i}]")
        elif prev["images"][i][0] != n:
            out.append(f"renamed: {prev['images'][i][0]} → {n}")
    out += [f"closed: {n} [{i}]" for i, (n, _) in prev["images"].items() if i not in now["images"]]
    if now["active"] != prev["active"]:
        out.append(f"active image → {now['images'].get(now['active'], ('none',))[0]}")
    if now["results"] != prev["results"]:
        out.append(f"Results table: {prev['results']} → {now['results']} rows")
    if now["rois"] != prev["rois"]:
        out.append(f"ROI Manager: {prev['rois']} → {now['rois']} ROIs")
    seen = set(prev["log"])
    new_log = [e.split("|", 1)[1] for e in now["log"] if e not in seen]
    if new_log:
        out.append("log: " + " | ".join(new_log[-12:]))
    for k, lab in (("tool", "tool"), ("selection", "selection"), ("sheet", "open dialog")):
        if now[k] != prev[k]:
            out.append(f"{lab} → {now[k]}")
    if now["recorder"] != prev["recorder"] and now["recorder"] is not None:
        out.append(f"recorder: {prev['recorder']} → {now['recorder']} lines (recorded_macro)")
    return "\n".join(out) or "No changes."


# ============================================================================ window, jobs, settings, help
@mcp.tool()
def ui(action: str, args: dict[str, Any] | None = None) -> str:
    """Act on the user's window.  action:
    panel {panel: display|threshold|rois|results|table|plot|histogram|history|script|log|info|workflows|tracks|classifier|volume|ortho, open}
    | set_tool {tool: pointer|hand|rect|oval|polygon|freehand|line|polyline|point|wand|angle|track|brush|bucket|text|arrow|zoom|picker}
    | set_position {image_id?, t?, z?, c?} | select_image {image_id}
    | notify {message, kind: info|success|error} – a notice in the window
    | open_sheet {op, params} – open a command's dialog pre-filled; the user reviews and presses Apply
    | palette {open} | theme {theme: dark|light} | view {fit: true} or {zoom: 2}."""
    args = dict(args or {})
    if "image" in args and "image_id" not in args:
        args["image_id"] = _resolve_image(args.pop("image"))["id"]
    if action == "open_sheet" and args.get("op"):
        args["op"] = _resolve_op(str(args["op"]))["id"]
    r = _show(action, args)
    if r is None:
        raise LumenError("This Lumen has no window bridge (update Lumen).")
    if not r.get("windows"):
        return "No window is connected (headless engine): nothing to show."
    import time
    for _ in range(20):
        st = _ui_state()
        if (st.get("seq_applied") or 0) >= r["seq"]:
            return f"Window: {st['last_error']}" if st.get("last_error") else f"Done ({action})."
        time.sleep(0.1)
    return f"Sent ({action}); the window has not confirmed yet."


@mcp.tool()
def jobs(cancel: str | None = None) -> str:
    """Running long commands with their progress; cancel=<job id> stops one (its image is rolled back)."""
    if cancel:
        C.post(f"/api/jobs/{cancel}/cancel")
        return f"Cancel requested for job {cancel}."
    js = C.get("/api/jobs")
    return "\n".join(f"{j['id']}: {j['name']} {int(100 * (j.get('progress') or 0))}% {j.get('status', '')}" for j in js) or "No running jobs."


@mcp.tool()
def settings(patch: dict[str, Any] | None = None) -> str:
    """Lumen's preferences (measurements, ROI colour, undo budget, threads, …); patch={...} changes some."""
    if patch:
        return json.dumps(_trim(C.patch("/api/settings", patch)))
    return json.dumps(_trim(C.get("/api/settings"), 60))


@mcp.tool()
def help(topic: str = "") -> str:
    """Lumen's bundled documentation: no topic lists the documents; a name (e.g. "SCRIPTING", "FIJI_PARITY",
    "COMMANDS", "MCP") returns it (long ones are cut – ask for a later part with "NAME:2")."""
    if not topic:
        docs = C.get("/api/docs").get("docs") or []
        return "Documents (help('<name>')):\n" + "\n".join(f"- {d['name']}: {d.get('title', '')}" for d in docs if d.get("available", True))
    name, _, part = topic.partition(":")
    r = C.get(f"/api/docs/{name}")
    text = r.text if not isinstance(r, dict) else str(r.get("text") or r)
    k = int(part or 1)
    chunk = 12000
    body = text[(k - 1) * chunk:k * chunk]
    more = f"\n… part {k} of {math.ceil(len(text) / chunk)} (help('{name}:{k + 1}'))" if len(text) > k * chunk else ""
    return body + more


@mcp.resource("lumen://commands")
def commands_resource() -> str:
    """Every Lumen command: id, name, menu."""
    return "\n".join(f"{o['id']}\t{o['name']}\t{o.get('menu')}" for o in _ops())


def main(argv: list[str] | None = None) -> int:
    mcp.run(transport="stdio")
    return 0
