"""The shell's small JSON state file (``<lumen home>/desktop/desktop-state.json``): window geometry, zoom, the
last port and the last save folder."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("lumen.desktop.state")

MIN_WIDTH, MIN_HEIGHT = 1000, 660
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1440, 900
ZOOM_STEPS = (0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)


class DesktopState:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {}
        self.load()

    # ----------------------------------------------------------------- persistence
    def load(self) -> None:
        try:
            if self.path.is_file():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data = raw
        except Exception:  # noqa: BLE001 - a broken state file is simply ignored
            log.warning("could not read %s - starting with defaults", self.path, exc_info=True)
            self.data = {}

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
                tmp.replace(self.path)
            except Exception:  # noqa: BLE001
                log.warning("could not write %s", self.path, exc_info=True)

    # ----------------------------------------------------------------- window geometry
    @property
    def window(self) -> dict[str, Any]:
        return self.data.setdefault("window", {})

    def geometry(self, screens: list[tuple[int, int, int, int]] | None = None) -> dict[str, Any]:
        """``create_window`` kwargs from the saved geometry, sanitised against the current screens
        (``screens``: (x, y, width, height) rectangles; a position off every screen is dropped)."""
        w = self.window
        width = int(w.get("width") or DEFAULT_WIDTH)
        height = int(w.get("height") or DEFAULT_HEIGHT)
        width, height = max(width, MIN_WIDTH), max(height, MIN_HEIGHT)
        kwargs: dict[str, Any] = {"width": width, "height": height, "maximized": bool(w.get("maximized"))}
        x, y = w.get("x"), w.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            x, y = int(x), int(y)
            visible = True
            if screens:
                visible = any(sx - 50 <= x < sx + sw - 100 and sy - 10 <= y < sy + sh - 100 for sx, sy, sw, sh in screens)
            if visible:
                kwargs["x"], kwargs["y"] = x, y
        return kwargs

    def remember_size(self, width: int, height: int) -> None:
        if width >= 200 and height >= 100:
            self.window["width"], self.window["height"] = int(width), int(height)

    def remember_position(self, x: int, y: int) -> None:
        self.window["x"], self.window["y"] = int(x), int(y)

    def remember_maximized(self, maximized: bool) -> None:
        self.window["maximized"] = bool(maximized)

    # ----------------------------------------------------------------- zoom / port / dirs
    @property
    def zoom(self) -> float:
        try:
            z = float(self.data.get("zoom", 1.0))
        except (TypeError, ValueError):
            z = 1.0
        return z if 0.5 <= z <= 3.0 else 1.0

    @zoom.setter
    def zoom(self, value: float) -> None:
        self.data["zoom"] = round(float(value), 3)

    def zoom_step(self, direction: int) -> float:
        """The next zoom level above (+1) / below (-1) the current one."""
        current = self.zoom
        if direction > 0:
            nxt = [z for z in ZOOM_STEPS if z > current + 1e-6]
            self.zoom = nxt[0] if nxt else ZOOM_STEPS[-1]
        else:
            prv = [z for z in ZOOM_STEPS if z < current - 1e-6]
            self.zoom = prv[-1] if prv else ZOOM_STEPS[0]
        return self.zoom

    @property
    def last_port(self) -> int | None:
        p = self.data.get("port")
        return int(p) if isinstance(p, int) and 1024 <= p <= 65535 else None

    @last_port.setter
    def last_port(self, port: int) -> None:
        self.data["port"] = int(port)

    @property
    def last_open_dir(self) -> str:
        d = self.data.get("last_open_dir")
        return d if isinstance(d, str) and Path(d).is_dir() else ""

    @last_open_dir.setter
    def last_open_dir(self, value: str) -> None:
        self.data["last_open_dir"] = value

    @property
    def last_save_dir(self) -> str:
        d = self.data.get("last_save_dir")
        return d if isinstance(d, str) and Path(d).is_dir() else ""

    @last_save_dir.setter
    def last_save_dir(self, value: str) -> None:
        self.data["last_save_dir"] = value
