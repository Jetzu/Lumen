"""The bridge between an AI assistant (the Lumen MCP server, ``lumen.mcp_server``) and the open Lumen window (drop-in router).

The engine state (images, results, ROI Manager, history) lives in the server and the window follows it through
``/api/events``.  A few things live only in the window: the selection drawn on each image, the active tool, the open
inspector panels, the table / plot shown after a command.  This router carries those both ways:

``POST /api/ui/command``  ``{"action": ..., "args": {...}}``
    Broadcast ``{"type": "ui", "action", "args", "seq"}`` to every open window (``frontend/src/agentBridge.ts`` applies
    it: set the selection, the tool, the plane, open a panel, show a command's result, open an operation sheet for
    the user to review, show a notice).  Answers ``{"seq", "windows": <how many windows got it>}``.

``POST /api/ui/state``  (sent by the window, debounced)
    What the window shows: its client id, the active image, the tool, every image's selection, the plane, the open
    panels and sheet, the theme.  ``seq_applied`` is the last command it applied.

``GET /api/ui/state``
    The most recently reported window state (plus ``windows``: every window seen in the last minute) – the
    assistant's view of the user's screen.  ``{"connected": false}`` when no window has reported yet.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

#: the UI actions a window applies (everything else is refused, so a typo is an error instead of a silent no-op)
ACTIONS = {
    "apply_result": "show a command's result in the window (selection, table, plot, histogram, panels, message)",
    "set_selection": "args: image_id?, roi (a ROI dict, or null to clear)",
    "set_tool": "args: tool (pointer, hand, rect, oval, polygon, freehand, line, polyline, point, wand, angle, track, brush, bucket, text, arrow, zoom, picker)",
    "set_position": "args: image_id?, t?, z?, c? (0-based plane)",
    "select_image": "args: image_id",
    "panel": "args: panel (display, threshold, rois, results, table, plot, histogram, history, script, log, info, …), open (bool)",
    "notify": "args: message, kind (info | success | error)",
    "open_sheet": "args: op, params? – open the command's operation sheet pre-filled, for the user to review and apply",
    "palette": "args: open (bool) – the command palette",
    "theme": "args: theme (dark | light)",
    "view": "args: fit (bool) | zoom (number) – the viewport zoom",
}

_SEQ = itertools.count(1)
_LOCK = threading.Lock()
_WINDOWS: dict[str, dict[str, Any]] = {}  # client id → last reported state (with "seen")


class UiCommand(BaseModel):
    action: str
    args: dict[str, Any] | None = None


class UiState(BaseModel):
    model_config = {"extra": "allow"}
    client: str = ""


def _broadcaster():
    from ..app import BROADCASTER
    return BROADCASTER


def _live_windows(max_age: float = 60.0) -> list[dict[str, Any]]:
    now = time.time()
    with _LOCK:
        for k in [k for k, v in _WINDOWS.items() if now - v.get("seen", 0) > 600]:
            _WINDOWS.pop(k, None)
        return sorted((dict(v) for v in _WINDOWS.values() if now - v.get("seen", 0) <= max_age), key=lambda v: -v["seen"])


def send_ui(action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Broadcast one UI command; returns its sequence number and the number of connected windows."""
    if action not in ACTIONS:
        raise ValueError(f"unknown UI action {action!r}; one of: {', '.join(sorted(ACTIONS))}")
    seq = next(_SEQ)
    b = _broadcaster()
    with b.lock:
        n = len(b.clients)
    b.on_event({"type": "ui", "action": action, "args": args or {}, "seq": seq})
    return {"seq": seq, "windows": n}


@router.get("/ui/actions")
def ui_actions() -> dict[str, str]:
    return ACTIONS


@router.post("/ui/command")
def ui_command(body: UiCommand) -> dict[str, Any]:
    json.dumps(body.args or {})  # must be JSON (a TypeError → 400)
    return send_ui(body.action, body.args)


@router.post("/ui/state")
def ui_state_report(body: UiState) -> dict[str, Any]:
    st = body.model_dump()
    st["client"] = (st.get("client") or "default")[:128]
    st["seen"] = time.time()
    with _LOCK:
        _WINDOWS[st["client"]] = st
    return {"ok": True}


@router.get("/ui/state")
def ui_state() -> dict[str, Any]:
    wins = _live_windows()
    if not wins:
        b = _broadcaster()
        with b.lock:
            n = len(b.clients)
        return {"connected": False, "event_clients": n}
    cur = dict(wins[0])
    cur["connected"] = True
    cur["age_s"] = round(time.time() - cur.pop("seen"), 2)
    cur["windows"] = [{"client": w["client"], "age_s": round(time.time() - w["seen"], 2)} for w in wins]
    return cur
