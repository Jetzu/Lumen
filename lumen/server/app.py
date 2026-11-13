"""FastAPI application: REST + WebSocket API and the static frontend."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core import ops  # noqa: F401  (registers commands)
from ..core.image import LumenImage
from ..core.io import list_dir, read_image
from ..core.luts import LUT_NAMES, all_luts
from ..core.measure import ALL_MEASUREMENTS, MEASUREMENT_LABELS
from ..core.registry import REGISTRY, OperationCancelled
from ..core.scripting import run_script
from ..core.session import CURRENT_CLIENT, SESSION, Session
from ..core.threshold import METHODS as THRESHOLD_METHODS
from .. import __version__

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
#: request header naming the client (browser tab) – its own recorder; see lumen.core.session.CURRENT_CLIENT
CLIENT_HEADER = "x-lumen-client"
LOG = logging.getLogger("lumen.server")

_CURRENT_SESSION: Session | None = None


def get_session() -> Session:
    """The session served by the running app (for drop-in routers in lumen.server.extra)."""
    return _CURRENT_SESSION or SESSION


class Broadcaster:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[WebSocket] = set()
        self.lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop, session: Session) -> None:
        self.loop = loop
        session.listeners.append(self.on_event)

    def on_event(self, evt: dict[str, Any]) -> None:
        if self.loop is None:
            return
        msg = json.dumps(evt)
        with self.lock:
            clients = list(self.clients)
        for ws in clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), self.loop)
            except Exception:
                pass


BROADCASTER = Broadcaster()


class ClientContext:
    """Pure-ASGI middleware: the ``X-Lumen-Client`` header (or ``?client=``) names the client whose recorder the
    request writes to (lumen.core.session.CURRENT_CLIENT); the contextvar reaches the sync endpoints in the thread
    pool because anyio copies the context.  No BaseHTTPMiddleware: the plane fetches are frequent and must stay cheap."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        client = ""
        for k, v in scope.get("headers") or []:
            if k == b"x-lumen-client":
                client = v.decode("latin-1").strip()
                break
        if not client:
            from urllib.parse import parse_qs
            client = (parse_qs((scope.get("query_string") or b"").decode("latin-1")).get("client") or [""])[0].strip()
        token = CURRENT_CLIENT.set(client[:128] or None)
        try:
            await self.app(scope, receive, send)
        finally:
            CURRENT_CLIENT.reset(token)


class RunBody(BaseModel):
    params: dict[str, Any] | None = None
    image_id: str | None = None
    roi: dict[str, Any] | None = None
    scope: str = "stack"


class DisplayBody(BaseModel):
    t: int | None = None
    z: int | None = None
    c: int | None = None
    mode: str | None = None
    channels: list[dict[str, Any]] | None = None


class PathBody(BaseModel):
    path: str
    format: str | None = None
    series: int | None = None
    level: int = 0
    #: open as a virtual stack (planes on demand); None = automatically when the decoded size exceeds ``auto_virtual_mb``
    virtual: bool | None = None


class ScriptBody(BaseModel):
    code: str
    params: dict[str, Any] | None = None  # values for the script's ``#@`` parameters (lumen.core.ijshim.params)
    roi: dict[str, Any] | None = None  # the viewport selection, the script's initial selection (imp.getRoi() / IJ.run(imp, "Measure"))
    language: str = "python"  # "python" | "jython" (one runner: the ij.* shim is always available) | "groovy" | "beanshell" | "rhino" | "jruby" | "clojure" (the optional JVM bridge)


class RoiBody(BaseModel):
    roi: dict[str, Any]
    image_id: str | None = None
    position: dict[str, int] | None = None


class RoiPatch(BaseModel):
    patch: dict[str, Any]


class RoiIds(BaseModel):
    ids: list[str]


class RoiCombine(BaseModel):
    ids: list[str]
    mode: str = "or"
    image_id: str | None = None


class RestoreBody(BaseModel):
    kind: str | None = None  # "rois" | "results" | None = whichever was cleared last



def _plane_response(arr: np.ndarray, extra: dict[str, str] | None = None) -> Response:
    arr = np.ascontiguousarray(arr)
    headers = {"X-Dtype": str(arr.dtype), "X-Shape": ",".join(str(s) for s in arr.shape), "Cache-Control": "no-store"}
    if extra:
        headers.update(extra)
    return Response(content=arr.tobytes(), media_type="application/octet-stream", headers=headers)


def _roi_plane(img: LumenImage, roi: dict[str, Any]) -> tuple[int, int, int]:
    """The (t, z, c) plane a ROI Manager entry is measured on: its stored position clamped per axis to the active
    image, the current display plane for a missing / non-integer axis.  ROIs are drawn on movies and Z-stacks and
    later measured on projections, other images or the same image after Delete Slice / Make Substack shrank it, so
    the stored plane may not exist any more; ImageJ's RoiManager.restore() → ImagePlus.setPosition() never fails on
    such a position either, and the frontend's jump-to-ROI clamps the same way (so the measured plane is the one it
    shows).  Negative values are clamped too (numpy would silently wrap them to the last plane)."""
    pos = roi.get("position")
    pos = pos if isinstance(pos, dict) else {}
    d = img.display
    out = []
    for key, n, cur in (("t", img.nt, d.t), ("z", img.nz, d.z), ("c", img.nc, d.c)):
        v = pos.get(key)
        try:
            v = cur if v is None else int(v)
        except (TypeError, ValueError):
            v = cur
        out.append(min(max(v, 0), n - 1))
    return out[0], out[1], out[2]


def create_app(session: Session = SESSION) -> FastAPI:
    global _CURRENT_SESSION
    _CURRENT_SESSION = session
    app = FastAPI(title="Lumen", version=__version__)

    @app.on_event("startup")
    async def _startup() -> None:
        BROADCASTER.attach(asyncio.get_running_loop(), session)

    # the recorder is per client (E7-4): the frontend sends a fresh random id per page load in X-Lumen-Client
    # (``?client=`` for plain links); requests without one share the "default" recorder
    app.add_middleware(ClientContext)

    async def _err(request: Request, exc: Exception):
        # error responses carry the message only; the traceback goes to the server log (E7-8)
        if isinstance(exc, HTTPException):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
        if isinstance(exc, OperationCancelled):
            # the op stopped at a check_cancel() after POST /api/jobs/{id}/cancel; the image was rolled back
            return JSONResponse({"error": "Cancelled"}, status_code=409)
        if isinstance(exc, KeyError):
            msg = str(exc.args[0]) if exc.args else str(exc)
        elif isinstance(exc, (ValueError, RuntimeError, FileNotFoundError, PermissionError)):
            msg = str(exc)
        else:
            msg = f"{type(exc).__name__}: {exc}"
        LOG.info("%s %s → 400 %s", request.method, request.url.path, msg)
        LOG.debug("request failed", exc_info=exc)
        return JSONResponse({"error": msg}, status_code=400)

    async def _unexpected(request: Request, exc: Exception):
        LOG.exception("%s %s failed: %s: %s", request.method, request.url.path, type(exc).__name__, exc)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    for et in (ValueError, KeyError, RuntimeError, FileNotFoundError, PermissionError, NotImplementedError, TypeError, IndexError, ZeroDivisionError, OSError, ImportError, OverflowError):
        app.add_exception_handler(et, _err)
    app.add_exception_handler(HTTPException, _err)
    app.add_exception_handler(Exception, _unexpected)

    # ------------------------------------------------------------ meta
    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {"version": __version__, "ops": REGISTRY.list(), "luts": LUT_NAMES, "threshold_methods": THRESHOLD_METHODS,
                "measurements": [{"id": m, "label": MEASUREMENT_LABELS[m]} for m in ALL_MEASUREMENTS], "home": str(Path.home()), "cwd": os.getcwd()}

    @app.get("/api/luts")
    def luts() -> dict[str, Any]:
        return all_luts()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        from ..core.lazy import PLANE_CACHE, auto_virtual_bytes, physical_memory_bytes
        st = session.state()
        # memory guardrails (data larger than RAM): the physical RAM, the auto-virtual limit (None = off), the plane
        # cache budget of virtual stacks and what the cache holds right now
        st["memory_limit_bytes"] = int(physical_memory_bytes())
        limit = auto_virtual_bytes()
        st["auto_virtual_bytes"] = None if limit >= 2 ** 62 else int(limit)  # sys.maxsize = "never" (auto_virtual_mb 0)
        st["virtual_cache_bytes"] = int(PLANE_CACHE.budget)
        st["virtual_resident_bytes"] = int(PLANE_CACHE.total)
        return st

    @app.websocket("/api/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        with BROADCASTER.lock:
            BROADCASTER.clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            with BROADCASTER.lock:
                BROADCASTER.clients.discard(ws)

    # ------------------------------------------------------------ files
    @app.get("/api/fs")
    def fs(path: str = "") -> dict[str, Any]:
        return list_dir(path or str(Path.home()))

    @app.post("/api/images/open")
    def open_path(body: PathBody) -> dict[str, Any]:
        # virtual=None: automatic – a TIFF / HDF5 / Zarr / npy file whose decoded size exceeds the auto_virtual_mb
        # preference opens as a virtual stack (planes on demand) instead of being read into memory
        img = session.open_file(body.path, series=body.series, level=body.level, virtual="auto" if body.virtual is None else body.virtual)
        if body.virtual is None and img.is_virtual:
            from ..core.lazy import auto_virtual_bytes, describe
            mb = 1024 ** 2
            session.add_log(f"{img.name}: {img.full_bytes / mb:,.0f} MB decoded exceeds the auto-virtual limit ({auto_virtual_bytes() / mb:,.0f} MB, "
                            f"Preferences ▸ auto_virtual_mb) – opened as a virtual stack ({describe(img.data)}): the display and the stack commands "
                            "read planes on demand; the first pixel-modifying command reads the whole stack into memory", "warning")
        return {"id": img.id, "image": img.summary()}

    @app.post("/api/images/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        fname = Path(file.filename or "upload.tif").name
        lower = fname.lower()
        # keep compound suffixes so the reader can dispatch on them (.ome.tif, .nii.gz, .ome.zarr …)
        suffix = next((ext for ext in (".ome.tiff", ".ome.tif", ".nii.gz", ".tar.gz") if lower.endswith(ext)), None) or Path(fname).suffix or ".tif"
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            img = read_image(tmp_path, name=file.filename or "upload")
            img.path = None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        session.add_image(img)
        session.add_log(f"Opened {img.name} {img.shape} {img.bit_depth}")
        return {"id": img.id, "image": img.summary()}

    @app.post("/api/images/{image_id}/save")
    def save(image_id: str, body: PathBody) -> dict[str, Any]:
        return {"path": session.save_image(image_id, body.path, body.format)}

    @app.get("/api/images/{image_id}/download")
    def download(image_id: str, format: str = "tif") -> Response:
        from ..core.io import strip_image_ext, write_image
        img = session.get_image(image_id)
        from urllib.parse import quote
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / (strip_image_ext(img.name) or "image")
            out = write_image(img, str(base), format)
            data = Path(out).read_bytes()
            fname = Path(out).name
        # response headers are latin-1: an ASCII fallback in ``filename=`` and, when the name is not pure ASCII
        # (e.g. '細胞.tif'), the real name via RFC 5987 ``filename*`` (the same scheme as the mesh download)
        stem, ext = Path(fname).stem, Path(fname).suffix
        safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_. ")) else "_" for ch in stem).strip("_ ") or "image"
        safe += "".join(ch for ch in ext if ch.isascii() and (ch.isalnum() or ch == "."))
        disposition = f'attachment; filename="{safe}"'
        if safe != fname:
            disposition += f"; filename*=UTF-8''{quote(fname)}"
        return Response(content=data, media_type="application/octet-stream", headers={"Content-Disposition": disposition})

    # ------------------------------------------------------------ images
    @app.get("/api/images")
    def images() -> list[dict[str, Any]]:
        return [im.summary() for im in session.images.values()]

    @app.get("/api/images/{image_id}")
    def image(image_id: str) -> dict[str, Any]:
        return session.get_image(image_id).summary()

    @app.delete("/api/images/{image_id}")
    def close(image_id: str) -> dict[str, Any]:
        session.close_image(image_id)
        return {"ok": True}

    @app.post("/api/images/{image_id}/select")
    def select(image_id: str) -> dict[str, Any]:
        session.select(image_id)
        return {"ok": True}

    @app.get("/api/images/{image_id}/planes")
    def planes(image_id: str, t: int = 0, z: int = 0, stride: int = 1) -> Response:
        img = session.get_image(image_id)
        t = min(max(t, 0), img.nt - 1)
        z = min(max(z, 0), img.nz - 1)
        arr = img.data[t, z]
        if stride > 1:
            arr = arr[:, ::stride, ::stride]
        return _plane_response(arr, {"X-Stride": str(stride)})

    @app.get("/api/images/{image_id}/plane")
    def plane(image_id: str, t: int = 0, z: int = 0, c: int = 0, stride: int = 1) -> Response:
        img = session.get_image(image_id)
        arr = img.data[min(t, img.nt - 1), min(z, img.nz - 1), min(c, img.nc - 1)]
        if stride > 1:
            arr = arr[::stride, ::stride]
        return _plane_response(arr)

    @app.get("/api/images/{image_id}/value")
    def value(image_id: str, x: int, y: int, t: int = 0, z: int = 0) -> dict[str, Any]:
        img = session.get_image(image_id)
        if not (0 <= x < img.width and 0 <= y < img.height):
            return {"values": None}
        vals = img.data[min(t, img.nt - 1), min(z, img.nz - 1), :, y, x]
        return {"values": [None if (isinstance(v, np.floating) and not np.isfinite(v)) else float(v) for v in vals]}

    @app.post("/api/images/{image_id}/display")
    def display(image_id: str, body: DisplayBody) -> dict[str, Any]:
        # inspector edits are recorded (lumen.display lines, collapsed per channel while a slider settles)
        session.update_display(image_id, body.model_dump(exclude_none=True), record=True)
        return session.get_image(image_id).display.to_dict()

    @app.post("/api/images/{image_id}/undo")
    def undo(image_id: str) -> dict[str, Any]:
        return {"label": session.undo(image_id)}

    @app.post("/api/images/{image_id}/redo")
    def redo(image_id: str) -> dict[str, Any]:
        return {"label": session.redo(image_id)}

    @app.post("/api/images/{image_id}/reset_display")
    def reset_display(image_id: str) -> dict[str, Any]:
        img = session.get_image(image_id)
        img.reset_display(keep_luts=True)
        return img.display.to_dict()

    @app.get("/api/images/{image_id}/rgb.png")
    def rgb_png(image_id: str, t: int | None = None, z: int | None = None) -> Response:
        from ..core.render import render_rgb
        import imageio.v3 as iio
        img = session.get_image(image_id)
        rgb = render_rgb(img, t=t, z=z)
        buf = io.BytesIO()
        iio.imwrite(buf, rgb, extension=".png")
        return Response(content=buf.getvalue(), media_type="image/png")

    # ------------------------------------------------------------ ops
    @app.get("/api/ops")
    def list_ops() -> list[dict[str, Any]]:
        return REGISTRY.list()

    @app.post("/api/ops/{op_id}/run")
    def run(op_id: str, body: RunBody) -> dict[str, Any]:
        return session.run_op(op_id, body.params, image_id=body.image_id, roi=body.roi, scope=body.scope)

    @app.post("/api/ops/{op_id}/preview")
    def preview(op_id: str, body: RunBody, stride: int = 1) -> Response:
        out = session.run_op(op_id, body.params, image_id=body.image_id, roi=body.roi, scope=body.scope, preview=True, record=False)
        arr = out["preview"]
        if stride > 1:
            arr = arr[:, ::stride, ::stride]
        return _plane_response(arr, {"X-Stride": str(stride)})

    # ------------------------------------------------------------ jobs (progress + cancellation)
    # Both handlers avoid the session lock on purpose: the op holds it for its whole duration and these
    # requests arrive while it runs (sync endpoints run in the thread pool, so they are not blocked by it).
    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return session.list_jobs()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        if not session.cancel_job(job_id):
            raise HTTPException(404, f"No running job {job_id}")
        return {"ok": True, "id": job_id}

    # ------------------------------------------------------------ rois
    @app.get("/api/rois")
    def rois() -> list[dict[str, Any]]:
        return session.roi_manager.to_list()

    @app.post("/api/rois")
    def add_roi(body: RoiBody) -> dict[str, Any]:
        r = session.roi_manager.add(body.roi, image_id=body.image_id or session.active_id, position=body.position)
        session._changed("rois")
        return r

    @app.patch("/api/rois/{roi_id}")
    def patch_roi(roi_id: str, body: RoiPatch) -> dict[str, Any]:
        r = session.roi_manager.update(roi_id, body.patch)
        if r is None:
            raise HTTPException(404, "ROI not found")
        session._changed("rois")
        return r

    @app.post("/api/rois/delete")
    def delete_rois(body: RoiIds) -> dict[str, Any]:
        # recorded (rois.delete) and restorable: POST /api/session/restore_clear or the rois.restore_cleared command
        n = session.delete_rois(body.ids)
        return {"ok": True, "deleted": n, "remaining": len(session.roi_manager.to_list()), "restorable": n > 0}

    @app.delete("/api/rois")
    def clear_rois() -> dict[str, Any]:
        n = session.clear_rois()
        return {"ok": True, "cleared": n, "restorable": n > 0}

    @app.post("/api/session/restore_clear")
    def restore_clear(body: RestoreBody | None = None) -> dict[str, Any]:
        """One-step undo of the last ROI Manager clear / delete or Results clear (E7-6)."""
        out = session.restore_last_clear(body.kind if body else None)
        if out["restored"] and out["kind"] == "rois":
            session.record('lumen.run("rois.restore_cleared")')
        elif out["restored"]:
            session.record('lumen.run("results.restore_cleared")')
        return out

    @app.post("/api/rois/combine")
    def combine_rois(body: RoiCombine) -> dict[str, Any]:
        from ..core import roi as roimod
        img = session.get_image(body.image_id) if body.image_id else session.get_image()
        sel = [r for r in session.roi_manager.to_list() if r["id"] in set(body.ids)] or session.roi_manager.to_list()
        r = roimod.combine(sel, (img.height, img.width), body.mode)
        if r is None:
            raise HTTPException(400, "Nothing to combine")
        return r

    @app.post("/api/rois/measure")
    def measure_rois(body: RoiIds) -> dict[str, Any]:
        from ..core.measure import measure_roi
        img = session.get_image()
        ids = set(body.ids)
        sel = [r for r in session.roi_manager.to_list() if (not ids or r["id"] in ids)]
        # the Set Measurements options (invert Y, NaN empty cells, display label, limit to threshold) apply exactly as
        # for Measure (analyze.measure); there is no OpContext here, so the threshold is read from the session directly
        opts = session.measure_options
        thr = None
        if opts.get("limit_to_threshold"):
            t = session.thresholds.get(img.id)
            if t is not None:
                thr = (float(t["lower"]), float(t["upper"]))
        rows = []
        for r in sel:
            t, z, c = _roi_plane(img, r)
            rows += measure_roi(img, r, session.measurements, t=t, z=z, c=c,
                                limit_to_threshold=thr, label=r.get("name"), options=opts)
        session.results.append(rows)
        session._changed("results")
        return {"rows": len(rows)}

    @app.get("/api/rois/export")
    def export_rois() -> Response:
        data = json.dumps(session.roi_manager.to_list(), indent=1)
        return Response(content=data, media_type="application/json", headers={"Content-Disposition": 'attachment; filename="rois.json"'})

    @app.post("/api/rois/import")
    async def import_rois(file: UploadFile = File(...)) -> dict[str, Any]:
        data = json.loads(await file.read())
        n = 0
        for r in data if isinstance(data, list) else data.get("rois", []):
            session.roi_manager.add(r)
            n += 1
        session._changed("rois")
        return {"added": n}

    # ------------------------------------------------------------ results
    @app.get("/api/results")
    def results() -> dict[str, Any]:
        return session.results.to_dict()

    @app.delete("/api/results")
    def clear_results() -> dict[str, Any]:
        n = session.clear_results()
        return {"ok": True, "cleared": n, "restorable": n > 0}

    @app.get("/api/results.csv")
    def results_csv() -> Response:
        return Response(content=session.results.to_csv(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="Results.csv"'})

    @app.delete("/api/threshold/{image_id}")
    def clear_threshold(image_id: str) -> dict[str, Any]:
        session.thresholds.pop(image_id, None)
        return {"ok": True}

    # ------------------------------------------------------------ script
    @app.post("/api/script/run")
    def script(body: ScriptBody) -> dict[str, Any]:
        # Python and Jython (ImageJ 1.x API) scripts share one runner: the ij.* shim is importable, the ``#@``
        # parameter header is filled from ``params`` and the viewport selection seeds the script's selection.
        # Groovy / BeanShell / Rhino JavaScript (and opt-in JRuby / Clojure) go to the optional JVM bridge
        # (lumen.core.ijshim.jvm_bridge), which answers with the install steps in "error" when it is unavailable.
        # The plain ``run_script`` stays the fallback should the shim ever fail to import.
        try:
            from ..core.ijshim import is_jvm_language, run_ij_script, run_jvm_script
        except Exception:  # pragma: no cover
            return run_script(session, body.code)
        if is_jvm_language(body.language):
            return run_jvm_script(session, body.code, body.language, body.params, roi=body.roi)
        return run_ij_script(session, body.code, body.params, roi=body.roi)

    @app.get("/api/macro")
    def macro() -> PlainTextResponse:
        rec = session.recorder()
        return PlainTextResponse("\n".join([rec.marker(), *rec.lines]))

    @app.delete("/api/macro")
    def clear_macro() -> dict[str, Any]:
        return {"ok": True, "cleared": session.clear_recorder()}

    # ------------------------------------------------------------ recorder (per client: X-Lumen-Client header)
    @app.get("/api/recorder")
    def recorder() -> dict[str, Any]:
        """This client's recorder: {client, started, started_iso, marker, active_image, lines, count}."""
        return session.recorder().info()

    @app.delete("/api/recorder")
    def clear_recorder() -> dict[str, Any]:
        return {"ok": True, "cleared": session.clear_recorder()}

    @app.get("/api/recorder/preview")
    def recorder_preview() -> dict[str, Any]:
        """Exactly what Apply Recorded Macro to Folder would run: {lines, skipped, script} (open/save/sample/select/
        display/undo lines are skipped – they name files or images of this session)."""
        from ..core.ops.batch import preview_lines
        return preview_lines(session.macro)

    @app.get("/api/recorder/ijm")
    def recorder_ijm() -> dict[str, Any]:
        """This client's recording as an ImageJ-1 macro (best effort, every line accounted for): {script, lines, converted,
        skipped: [{line, reason}], notes, count}. Lines without an ImageJ equivalent stay in the script as comments."""
        from ..core.ij_export import recorder_to_ijm
        rec = session.recorder()
        titles = {img.id: img.name for img in session.images.values()}
        return recorder_to_ijm([rec.marker(), *rec.lines], titles=titles)

    # ------------------------------------------------------------ extra routers (auto-discovered)
    import importlib
    import pkgutil
    from . import extra as _extra
    for _info in sorted(pkgutil.iter_modules(_extra.__path__), key=lambda m: m.name):
        try:
            mod = importlib.import_module(f"{_extra.__name__}.{_info.name}")
            router = getattr(mod, "router", None)
            if router is not None:
                app.include_router(router, prefix="/api")
        except Exception as e:  # pragma: no cover
            print(f"[lumen] router {_info.name} failed to load: {e}")

    # ------------------------------------------------------------ static
    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> Response:
            root = STATIC_DIR.resolve()
            if path:
                try:
                    target = (root / path).resolve()
                except (OSError, ValueError):
                    target = root
                # never serve anything outside the static directory (path traversal)
                if target.is_file() and target != root and root in target.parents:
                    return FileResponse(target)
            return FileResponse(root / "index.html")

    return app


app = create_app()
