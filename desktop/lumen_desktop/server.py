"""The Lumen server in-process: uvicorn in a daemon thread on a loopback port, readiness via ``/api/meta``
(the command registry has loaded), orderly shutdown (``should_exit`` + join)."""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("lumen.desktop.server")

HOST = "127.0.0.1"
#: the port the shell prefers on a first launch (the ``lumen`` command uses 8765; a different default keeps a
#: browser session and the desktop app apart when both run)
DEFAULT_PORT = 8770


def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":  # like uvicorn: a port in TIME_WAIT after the last quit is free (same port = same web storage)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(preferred: int | None = None, host: str = HOST) -> int:
    """``preferred`` when it is free (stable URLs across launches, so the web view keeps its storage), else an
    OS-assigned free port."""
    for cand in (preferred, DEFAULT_PORT):
        if cand and 1024 <= cand <= 65535 and port_is_free(cand, host):
            return cand
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def http_get(url: str, timeout: float = 2.0, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback URL
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def http_post_json(url: str, payload: dict, timeout: float = 30.0, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback URL
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}


class BackendServer:
    """uvicorn ``Server`` driven from a thread (no reload, no signal handlers - we are not the main thread)."""

    def __init__(self, port: int, host: str = HOST, access_log: bool = False):
        self.host = host
        self.port = port
        self.access_log = access_log
        self.url = f"http://{host}:{port}"
        self.ready_url = f"{self.url}/api/meta"
        self._server = None
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.started_at = 0.0
        self.ready_at = 0.0
        self.n_commands = 0

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> None:
        import uvicorn

        from lumen.server.app import app as fastapi_app

        config = uvicorn.Config(
            fastapi_app,
            host=self.host,
            port=self.port,
            log_config=None,  # keep our logging setup (uvicorn's default config would reset the root logger)
            log_level="info",
            access_log=self.access_log,
            loop="asyncio",
            lifespan="on",
            workers=1,
        )
        self._server = uvicorn.Server(config)
        self.started_at = time.monotonic()

        def run() -> None:
            try:
                self._server.run()
            except BaseException as exc:  # noqa: BLE001 - surfaced through .error
                self.error = exc
                log.exception("server crashed")

        self._thread = threading.Thread(target=run, name="lumen-uvicorn", daemon=True)
        self._thread.start()
        log.info("server starting on %s", self.url)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Poll ``/api/meta`` until it answers 200 with the command list."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None or (self._thread is not None and not self._thread.is_alive()):
                return False
            try:
                status, body, _ = http_get(self.ready_url, timeout=3.0)
                if status == 200:
                    meta = json.loads(body or b"{}")
                    self.n_commands = len(meta.get("ops") or [])
                    self.ready_at = time.monotonic()
                    log.info("server ready after %.0f ms (%d commands)", (self.ready_at - self.started_at) * 1000, self.n_commands)
                    return True
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(0.1)
        return False

    @property
    def startup_ms(self) -> int:
        return int((self.ready_at - self.started_at) * 1000) if self.ready_at else -1

    def stop(self, timeout: float = 15.0) -> None:
        if self._server is None:
            return
        log.info("stopping server")
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("server thread did not stop within %.0fs - forcing exit", timeout)
                self._server.force_exit = True
                self._thread.join(5)
        self._server = None
