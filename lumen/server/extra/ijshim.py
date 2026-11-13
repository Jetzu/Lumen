"""The ImageJ 1.x API shim for scripts (drop-in router; see ``lumen.core.ijshim``).

``POST /api/ijshim/call {token, target, method, args[, roi, params]}``
    one synchronous call of the JavaScript bridge (``lumen.core.ijshim.bridge``): ``{"result", "log"}`` or 400
    ``{"error", "log"}``.  The first call of a token creates its bridge; ``roi`` seeds its selection.
``DELETE /api/ijshim/session/{token}``
    end a script run: ``{"found", "roi"}`` (the script's final selection, for the viewport).
``POST /api/script/params {code}``
    ``{"parameters": [...]}`` – the ``#@`` parameter header of a script as descriptors (name, type, label,
    default, min, max, step, choices, style, description), what the Script panel prompts for before a run.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...core.ijshim import install
from ...core.ijshim.bridge import Bridge
from ...core.ijshim.params import descriptors
from ..app import get_session

router = APIRouter()
install()

MAX_BRIDGES = 16
_BRIDGES: "OrderedDict[str, Bridge]" = OrderedDict()
_LOCK = threading.Lock()


class CallBody(BaseModel):
    token: str
    target: str | int
    method: str
    args: list[Any] = []
    roi: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class CodeBody(BaseModel):
    code: str


def _bridge(body: CallBody) -> Bridge:
    with _LOCK:
        b = _BRIDGES.get(body.token)
        if b is None:
            b = Bridge(get_session(), roi=body.roi, params=body.params)
            _BRIDGES[body.token] = b
            while len(_BRIDGES) > MAX_BRIDGES:
                _BRIDGES.popitem(last=False)
        else:
            _BRIDGES.move_to_end(body.token)
        return b


@router.post("/ijshim/call")
def ijshim_call(body: CallBody) -> Any:
    b = _bridge(body)
    try:
        return b.call(body.target, body.method, body.args)
    except SystemExit as e:  # IJ.exit() / System.exit(): the script ends, not the server
        return JSONResponse(status_code=400, content={"error": f"exit: {e.code}" if e.code not in (None, 0) else "exit", "log": list(b.log)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}", "log": list(b.log)})


@router.delete("/ijshim/session/{token}")
def ijshim_release(token: str) -> dict[str, Any]:
    with _LOCK:
        b = _BRIDGES.pop(token, None)
    if b is None:
        return {"found": False, "roi": None}
    b.session._changed("images")
    return {"found": True, "roi": b.roi}


@router.post("/script/params")
def script_params(body: CodeBody) -> dict[str, Any]:
    return {"parameters": descriptors(body.code)}
