"""Agent panel backend (drop-in router, mounted under /api): the ChemDraw MCP's Agent Chat bridge contract, served by
Lumen itself, so the Agent panel (lumen/static_agent/, opened from the top bar) needs no separate bridge process.

    GET  /api/agent/health            {ok, version, app, backends:{claude,codex}, default_backend, settings, runs_active, sessions}
    GET  /api/agent/settings          settings snapshot          POST /api/agent/settings {backend?, model?, effort?, default_backend?}
    POST /api/agent/run               {backend?, prompt, context?, new?, session_id?} -> {run_id, backend, session_id, started}
    GET  /api/agent/events/<run_id>   text/event-stream (session|delta|text|tool|tool_result|stats|done|error, final end)
    GET  /api/agent/runs/<run_id>     POST /api/agent/cancel/<run_id>     GET /api/agent/history?backend=
    POST /api/agent/clear {backend?}  GET /api/agent/sessions

The CLI (the user's Claude Code / Codex, lumen/core/agent/agentcli.py) gets ``LUMEN_URL`` = this server, so the
``lumen`` MCP server it starts drives exactly this Lumen; Claude may use only the ``lumen`` MCP tools.  Each prompt
carries what the window shows (active image, selection, results, ROIs) from /api/ui/state.

Guard: the server listens on 127.0.0.1 only; these endpoints also refuse a non-loopback ``Host`` (DNS rebinding) and
a foreign ``Origin``, and state-changing requests need the ``X-Lumen-Agent: 1`` header – a custom header a page from
another origin cannot send without a CORS preflight, which Lumen never grants.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...core.agent import agentcli as A

router = APIRouter()
MAX_RUNS = 20
KEEPALIVE = 15.0
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _guard(request: Request, write: bool = False) -> None:
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    if host not in {h.strip("[]") for h in _LOOPBACK}:
        raise HTTPException(403, "the Agent panel answers on 127.0.0.1 only")
    origin = request.headers.get("origin")
    if origin and origin != "null":
        o = urlsplit(origin)
        if (o.hostname or "") not in {h.strip("[]") for h in _LOOPBACK}:
            raise HTTPException(403, "foreign origin")
    if write and request.headers.get("x-lumen-agent") != "1":
        raise HTTPException(403, "missing X-Lumen-Agent header")


class Run:
    def __init__(self, run_id: str, backend: str, prompt: str, session_id: Optional[str]) -> None:
        self.run_id, self.backend, self.prompt, self.session_id = run_id, backend, prompt, session_id
        self.started = time.time()
        self.finished: Optional[float] = None
        self.status = "running"
        self.events: List[Tuple[str, Any]] = []
        self.cond = threading.Condition()
        self.final_text = ""
        self.error: Optional[str] = None
        self.cancelled = False
        self.agent: Optional[A.AgentRun] = None

    def to_dict(self) -> dict:
        with self.cond:
            return {"run_id": self.run_id, "backend": self.backend, "status": self.status, "events_count": len(self.events),
                    "final_text": self.final_text, "session_id": self.session_id, "prompt": self.prompt,
                    "started": self.started, "finished": self.finished, "error": self.error}


class Hub:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        self.active: Dict[str, Run] = {}
        self._store: Optional[A.SessionStore] = None
        self._found: Tuple[float, dict] = (0.0, {})

    @property
    def store(self) -> A.SessionStore:
        if self._store is None or self._store.path != A.SessionStore().path:  # LUMEN_HOME may change (tests)
            self._store = A.SessionStore()
        return self._store

    def backends(self, refresh: bool = False) -> dict:
        ts, found = self._found
        if found and not refresh and time.time() - ts < 30:
            return dict(found)
        found = {b: A.find_backend(b) for b in A.BACKENDS}
        self._found = (time.time(), found)
        return dict(found)

    def default_backend(self) -> str:
        stored = A.settings_snapshot().get("default_backend")
        avail = [b for b, p in self.backends().items() if p]
        if stored in avail:
            return stored
        return avail[0] if avail else (stored or "claude")

    def on_event(self, run: Run, kind: str, payload: Any) -> None:
        if kind == "session" and payload:
            run.session_id = str(payload)
            try:
                self.store.set(run.backend, run.session_id)
            except Exception:  # noqa: BLE001
                pass
        finished = False
        with run.cond:
            if kind == "text" and payload:
                run.final_text = str(payload)
            elif kind == "error":
                run.error = str(payload)
            run.events.append((kind, payload))
            if kind == "done":
                if payload:
                    run.final_text = str(payload)
                run.finished = time.time()
                ended_on_error = len(run.events) >= 2 and run.events[-2][0] == "error"
                run.status = "cancelled" if run.cancelled else "error" if ended_on_error else "done"
                finished = True
            run.cond.notify_all()
        if finished:
            with self.lock:
                if self.active.get(run.backend) is run:
                    del self.active[run.backend]
            try:
                self.store.add({"run_id": run.run_id, "backend": run.backend, "prompt": run.prompt, "final_text": run.final_text,
                                "session_id": run.session_id, "started": run.started, "finished": run.finished,
                                "status": run.status, "error": run.error})
            except Exception:  # noqa: BLE001
                pass

    def get(self, run_id: str) -> Run:
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return run


HUB = Hub()


def _window_context() -> str:
    """What the user's window shows, for the agent (the active image, its selection, results, ROIs)."""
    from ..app import get_session
    from . import agent as bridge
    s = get_session()
    lines: List[str] = []
    try:
        st = s.state()
        imgs = st.get("images") or []
        active = next((i for i in imgs if i["id"] == st.get("active")), None)
        if active:
            sh = active["shape"]
            lines.append(f"Active image: {active['name']} (id {active['id']}), {sh['x']}×{sh['y']} px, c={sh['c']} z={sh['z']} t={sh['t']}, {active.get('bit_depth') or active.get('dtype')}")
        others = [i["name"] for i in imgs if not active or i["id"] != active["id"]]
        if others:
            lines.append("Other open images: " + ", ".join(others[:12]) + (" …" if len(others) > 12 else ""))
        if not imgs:
            lines.append("No image is open.")
        lines.append(f"Results table: {len((st.get('results') or {}).get('rows') or [])} rows; ROI Manager: {len(st.get('rois') or [])} ROIs")
        ui = bridge.ui_state()
        if ui.get("connected") and active:
            sel = (ui.get("selections") or {}).get(active["id"])
            if sel:
                lines.append("The user's current selection on the active image: " + json.dumps(sel)[:600])
            a = ui.get("active_image") or {}
            if a:
                lines.append(f"Shown plane: t={a.get('t')} z={a.get('z')} c={a.get('c')}; tool: {ui.get('tool')}")
            if ui.get("sheet"):
                lines.append(f"Open dialog: {ui['sheet'].get('name')} ({ui['sheet'].get('op')})")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


@router.get("/agent/health")
def health(request: Request) -> dict:
    _guard(request)
    from ... import __version__
    with HUB.lock:
        active = len(HUB.active)
    return {"ok": True, "version": __version__, "app": A.APP, "backends": HUB.backends(), "default_backend": HUB.default_backend(),
            "settings": {**A.settings_snapshot(), "default_backend": HUB.default_backend()}, "runs_active": active,
            "sessions": {b: HUB.store.get(b) for b in A.BACKENDS}}


@router.get("/agent/settings")
def get_settings(request: Request) -> dict:
    _guard(request)
    return {**A.settings_snapshot(), "default_backend": HUB.default_backend()}


@router.post("/agent/settings")
async def post_settings(request: Request) -> dict:
    _guard(request, write=True)
    body = await request.json()
    backend = body.get("backend")
    if backend is not None and backend not in A.BACKENDS:
        raise HTTPException(400, f"backend must be one of {A.BACKENDS}")
    effort = body.get("effort")
    if effort is not None and backend and str(effort) not in A.effort_choices(backend):
        raise HTTPException(400, f"effort for {backend} must be one of {A.effort_choices(backend)}")
    snap = A.update_settings(backend, body.get("model"), None if effort is None else str(effort), body.get("default_backend"))
    return {**snap, "default_backend": HUB.default_backend()}


@router.post("/agent/run")
async def start_run(request: Request) -> JSONResponse:
    _guard(request, write=True)
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt (non-empty string) is required")
    backend = body.get("backend") or HUB.default_backend()
    if backend not in A.BACKENDS:
        raise HTTPException(400, f"backend must be one of {A.BACKENDS}")
    exe = HUB.backends().get(backend) or A.find_backend(backend)
    if not exe:
        raise HTTPException(503, f"{A.BACKEND_LABELS[backend]} CLI not found (install it or set "
                                 f"{'LUMEN_CLAUDE_EXE' if backend == 'claude' else 'LUMEN_CODEX_EXE'})")
    sid = body.get("session_id")
    session_id = None if body.get("new") else (sid.strip() if isinstance(sid, str) and sid.strip() else HUB.store.get(backend))
    extra = body.get("context")
    extra = json.dumps(extra) if extra is not None and not isinstance(extra, str) else extra
    context = "\n".join(x for x in (_window_context(), extra or "") if x)
    port = request.url.port or 80
    env = {"LUMEN_URL": f"http://127.0.0.1:{port}"}
    with HUB.lock:
        busy = HUB.active.get(backend)
        if busy is not None:
            return JSONResponse({"error": f"{backend} is busy with run {busy.run_id}", "run_id": busy.run_id}, status_code=409)
        run = Run(secrets.token_hex(8), backend, prompt, session_id)
        try:
            run.agent = A.AgentRun(backend, prompt, callback=lambda k, p: HUB.on_event(run, k, p), session_id=session_id,
                                   extra_env=env, extra_context=context, exe=exe)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"could not prepare the {backend} run: {e}")
        HUB.active[backend] = run
        HUB.runs[run.run_id] = run
        while len(HUB.runs) > MAX_RUNS:
            victim = next((rid for rid, r in HUB.runs.items() if r.status != "running"), None)
            if victim is None:
                break
            del HUB.runs[victim]
        run.agent.start()
    return JSONResponse({"run_id": run.run_id, "backend": run.backend, "session_id": run.session_id, "started": run.started})


@router.get("/agent/events/{run_id}")
def events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
    _guard(request)
    run = HUB.get(run_id)
    try:
        start = max(0, int(request.headers.get("last-event-id") or after))
    except ValueError:
        start = 0

    def stream():
        yield "retry: 2000\n\n"
        idx = start
        while True:
            with run.cond:
                timed_out = False
                if idx >= len(run.events) and run.status == "running":
                    timed_out = not run.cond.wait(KEEPALIVE)
                batch = run.events[idx:]
                finished = run.status != "running"
            for kind, payload in batch:
                idx += 1
                yield f"id: {idx}\nevent: {kind}\ndata: {json.dumps(payload)}\n\n"
            if finished and idx >= len(run.events):
                yield (f"id: {idx}\nevent: end\ndata: " + json.dumps({"run_id": run.run_id, "status": run.status, "final_text": run.final_text,
                                                                    "session_id": run.session_id, "error": run.error}) + "\n\n")
                return
            if timed_out and not batch:
                yield ": keep-alive\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream; charset=utf-8", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})


@router.get("/agent/runs/{run_id}")
def run_info(run_id: str, request: Request) -> dict:
    _guard(request)
    return HUB.get(run_id).to_dict()


@router.get("/agent/runs")
def runs(request: Request) -> list:
    _guard(request)
    with HUB.lock:
        return [r.to_dict() for r in HUB.runs.values()]


@router.post("/agent/cancel/{run_id}")
def cancel(run_id: str, request: Request) -> dict:
    _guard(request, write=True)
    run = HUB.get(run_id)
    with run.cond:
        running = run.status == "running"
        if running:
            run.cancelled = True
    if running and run.agent is not None:
        run.agent.cancel()
    return {"ok": True, "run_id": run_id, "status": run.status, "was_running": running}


@router.get("/agent/history")
def history(request: Request, backend: Optional[str] = None, limit: Optional[int] = None) -> list:
    _guard(request)
    out = HUB.store.history(backend if backend in A.BACKENDS else None)
    return out[-limit:] if limit else out


@router.post("/agent/clear")
async def clear(request: Request) -> dict:
    _guard(request, write=True)
    body = await request.json() if (await request.body()) else {}
    backend = body.get("backend")
    HUB.store.clear(backend if backend in A.BACKENDS else None)
    return {"ok": True, "sessions": {b: HUB.store.get(b) for b in A.BACKENDS}}


@router.get("/agent/sessions")
def sessions(request: Request) -> dict:
    _guard(request)
    return {b: HUB.store.get(b) for b in A.BACKENDS}
