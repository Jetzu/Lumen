"""Plain text files for the script editor and text windows (drop-in router).

``GET /api/textfile?path=…``            ``{"path", "name", "text"}`` – read a UTF-8 text file (≤ 20 MB; a BOM is dropped)
``POST /api/textfile {path, text}``     ``{"path", "bytes"}`` – write it (parent folders are created, UTF-8, ``\\n`` kept as sent)

The script editor's Open… / Save… (server-side paths picked in the file dialog) and the Text Window's Save use these;
browser upload / download never touch the server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..app import get_session

router = APIRouter()

MAX_BYTES = 20 * 1024 * 1024


class TextFileBody(BaseModel):
    path: str
    text: str


@router.get("/textfile")
def read_textfile(path: str) -> dict[str, Any]:
    if not str(path or "").strip():
        raise HTTPException(400, "path is required")
    p = Path(path).expanduser()
    if not p.is_file():
        raise HTTPException(404, f"No such file: {p}")
    if p.stat().st_size > MAX_BYTES:
        raise HTTPException(400, f"{p.name} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    return {"path": str(p), "name": p.name, "text": p.read_text(encoding="utf-8-sig", errors="replace")}


@router.post("/textfile")
def write_textfile(body: TextFileBody) -> dict[str, Any]:
    if not body.path.strip():
        raise HTTPException(400, "path is required")
    p = Path(body.path.strip()).expanduser()
    if p.is_dir():
        raise HTTPException(400, f"{p} is a folder")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = body.text.encode("utf-8")
    p.write_bytes(data)
    get_session().add_log(f"Saved {p}")
    return {"path": str(p), "bytes": len(data)}
