"""The MCP server's connection to a running Lumen: discovery, HTTP calls, launching the app.

Lumen is a FastAPI engine with the window as a client (lumen/server/app.py); the MCP server is just another client
of the same API, so everything it does shows up in the open window at once (the window follows ``/api/events``) and
the window's own state (selections, tool, panels) comes from ``/api/ui/state`` (lumen/server/extra/agent.py).

Discovery order: ``$LUMEN_URL``; the desktop app's lock file (``<lumen home>/desktop/desktop.lock``, written by
the running Lumen.app / Lumen.exe with its server URL); the usual ports on 127.0.0.1 (8770 the desktop app,
8765 ``lumen`` in a browser, and the next few, which both use when theirs is taken)."""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import urllib.error
import urllib.parse
import urllib.request

PORTS = [8770, 8765, 8771, 8772, 8766, 8767, 8768, 8773, 8774, 8769]


class LumenError(RuntimeError):
    """An error reported by Lumen (its message is meant for the user / the assistant)."""


class ConnectError(OSError):
    """Nothing answers at the URL (Lumen is not running there)."""


class Response:
    """The part of an HTTP response the tools use (standard library only: no dependency on httpx, whose package name
    differs between MCP SDK versions)."""

    def __init__(self, status: int, headers: Any, content: bytes) -> None:
        self.status_code = status
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.content or b"null")


class _Http:
    """A tiny HTTP client for 127.0.0.1 (proxies ignored, like the browser does for localhost)."""

    def __init__(self) -> None:
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, url: str, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None,
                json_body: Any = None, timeout: float = 600.0) -> Response:
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        data = None
        hdrs = dict(headers or {})
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return Response(r.status, r.headers, r.read())
        except urllib.error.HTTPError as e:
            return Response(e.code, e.headers or {}, e.read() or b"")
        except urllib.error.URLError as e:
            raise ConnectError(str(e.reason)) from None
        except (ConnectionError, TimeoutError, OSError) as e:
            raise ConnectError(str(e)) from None

    def get(self, url: str, timeout: float = 600.0) -> Response:
        return self.request("GET", url, timeout=timeout)


def lumen_home() -> Path:
    custom = os.environ.get("LUMEN_HOME", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".lumen"


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def desktop_app_command() -> list[str] | None:
    """How to start the Lumen desktop app on this machine (None when it is not installed)."""
    if _frozen():  # running inside Lumen.app / Lumen.exe (``--mcp``): the app is this executable
        return [sys.executable]
    if sys.platform == "darwin":
        for app in (Path("/Applications/Lumen.app"), Path.home() / "Applications" / "Lumen.app"):
            if app.is_dir():
                return ["open", "-a", str(app)]
    elif sys.platform == "win32":
        for base in filter(None, (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles"))):
            exe = Path(base) / "Lumen" / "Lumen.exe"
            if exe.is_file():
                return [str(exe)]
    else:
        for name in ("lumen-desktop", "Lumen"):
            found = shutil.which(name)
            if found:
                return [found]
        for exe in (Path.home() / ".local" / "share" / "Lumen" / "Lumen", Path("/opt/Lumen/Lumen")):
            if exe.is_file():
                return [str(exe)]
    return None


class LumenClient:
    def __init__(self) -> None:
        self.url: str | None = None
        self.how: str = ""  # how the URL was found
        self._ok_at = 0.0  # when the server last answered
        self.proc: subprocess.Popen | None = None  # a server this MCP server started (stopped when it exits)
        self.http = _Http()
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------ discovery
    def _probe(self, url: str, timeout: float = 0.8) -> bool:
        try:
            r = self.http.get(f"{url}/api/ui/actions", timeout=timeout)
            if r.status_code == 200 and "apply_result" in r.json():
                return True
            r = self.http.get(f"{url}/api/jobs", timeout=timeout)  # an older Lumen without the assistant bridge
            if r.status_code == 200 and isinstance(r.json(), list):
                m = self.http.get(f"{url}/api/meta", timeout=5.0).json()
                return isinstance(m, dict) and "ops" in m and "threshold_methods" in m
        except Exception:  # noqa: BLE001
            return False
        return False

    def discover(self) -> str | None:
        env = os.environ.get("LUMEN_URL", "").strip().rstrip("/")
        if env:
            if self._probe(env, 2.0):
                self.url, self.how = env, "LUMEN_URL"
                return env
        lock = lumen_home() / "desktop" / "desktop.lock"
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            u = str(info.get("url") or "").rstrip("/")
            if u and self._probe(u):
                self.url, self.how = u, "desktop app"
                return u
        except Exception:  # noqa: BLE001
            pass
        for port in PORTS:
            u = f"http://127.0.0.1:{port}"
            if self._probe(u, 0.4):
                self.url, self.how = u, f"port {port}"
                return u
        return None

    def alive(self) -> bool:
        return bool(self.url) and self._probe(self.url, 1.5)

    def ensure(self) -> str:
        if self.url and time.time() - self._ok_at < 5.0:
            return self.url  # answered a moment ago: no probe per request
        if self.url and self.alive():
            self._ok_at = time.time()
            return self.url
        self.url = None
        if self.discover():
            self._ok_at = time.time()
            return self.url  # type: ignore[return-value]
        raise LumenError("Lumen is not running. Call `launch` to start it (the desktop app when installed, otherwise "
                         "the engine with a browser window), or start Lumen yourself and call `status` again.")

    # ------------------------------------------------------------------ launching
    def launch(self, mode: str = "auto", timeout: float = 90.0) -> dict[str, Any]:
        if self.discover():
            return {"started": False, "url": self.url, "how": self.how, "note": "Lumen is already running"}
        mode = (mode or "auto").lower()
        desk = desktop_app_command()
        if mode == "auto":
            mode = "desktop" if desk else "browser"
        if mode == "desktop":
            if not desk:
                raise LumenError("The Lumen desktop app is not installed here; use mode='browser' or 'headless'.")
            subprocess.Popen(desk, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=(sys.platform != "win32"))
        elif mode in ("browser", "headless"):
            if _frozen():
                raise LumenError("This Lumen build starts as the desktop app only; use mode='desktop'.")
            cmd = [sys.executable, "-m", "lumen", "--host", "127.0.0.1", "--port", "8765"]
            if mode == "headless":
                cmd.append("--no-browser")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise LumenError(f"unknown mode {mode!r}: auto, desktop, browser or headless")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise LumenError(f"the Lumen server exited with code {self.proc.returncode}")
            if self.discover():
                return {"started": True, "mode": mode, "url": self.url, "startup_s": round(time.time() - t0, 1)}
            time.sleep(0.5)
        raise LumenError(f"Lumen did not answer within {timeout:.0f} s after starting ({mode})")

    def shutdown(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except Exception:  # noqa: BLE001
                p.kill()

    # ------------------------------------------------------------------ HTTP
    def _headers(self, client: str | None) -> dict[str, str]:
        return {"X-Lumen-Client": client} if client else {}

    def _check(self, r: Response) -> Any:
        if r.status_code >= 400:
            try:
                msg = r.json().get("error") or r.text
            except Exception:  # noqa: BLE001
                msg = r.text
            raise LumenError(str(msg))
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            return r.json()
        return r

    def request(self, method: str, path: str, client: str | None = None, **kw: Any) -> Any:
        for attempt in (0, 1):
            url = self.ensure()
            try:
                r = self.http.request(method, url + path, headers=self._headers(client), **kw)
            except ConnectError:
                self.url, self._ok_at = None, 0.0  # Lumen quit or restarted (maybe on another port): find it again
                if attempt:
                    raise LumenError("Lumen stopped answering. Call `launch` or `status`.") from None
                continue
            self._ok_at = time.time()
            return self._check(r)

    def get(self, path: str, client: str | None = None, **params: Any) -> Any:
        return self.request("GET", path, client, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: Any = None, client: str | None = None, **params: Any) -> Any:
        return self.request("POST", path, client, json_body=body if body is not None else {}, params={k: v for k, v in params.items() if v is not None})

    def patch(self, path: str, body: Any, client: str | None = None) -> Any:
        return self.request("PATCH", path, client, json_body=body)

    def delete(self, path: str, client: str | None = None) -> Any:
        return self.request("DELETE", path, client)
