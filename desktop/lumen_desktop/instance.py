"""Single-instance guard.

The running shell writes ``<data dir>/desktop.lock`` (pid, API port, control port, token) and listens
on a loopback *control* socket.  A second launch reads the file, sends ``<token> FOCUS`` to the control
port and exits; the first instance raises its window.  A stale file (no answer, wrong token) is
overwritten.  Pure stdlib, same on every OS; whether the raised window really comes to the front is
up to the window manager (Windows and some Linux WMs only flash the taskbar entry instead)."""
from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("lumen.desktop.instance")

HOST = "127.0.0.1"


def _read_lock(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _send(port: int, line: str, timeout: float = 5.0) -> str:
    with socket.create_connection((HOST, port), timeout=timeout) as s:
        s.sendall((line + "\n").encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = s.recv(256)
            if not data:
                break
            chunks.append(data)
            if len(b"".join(chunks)) > 4096:
                break
        return b"".join(chunks).decode("utf-8", "replace").strip()


def notify_existing(lock_path: Path, files: list[str] | None = None) -> dict | None:
    """If another live instance owns ``lock_path``, ask it to open ``files`` (if any) and come to the front, and
    return its lock info."""
    info = _read_lock(lock_path)
    if not info:
        return None
    try:
        if files:
            reply = _send(int(info["control_port"]), f"{info['token']} OPEN " + json.dumps([str(f) for f in files]))
        else:
            reply = _send(int(info["control_port"]), f"{info['token']} FOCUS")
    except (ConnectionRefusedError, OSError) as exc:
        if isinstance(exc, socket.timeout):  # somebody answered the connection but not in time: assume it is alive
            log.warning("another instance did not answer the focus request in time (%s) - not starting a second one", info.get("url"))
            return info
        return None  # nobody listening: stale lock
    if reply.startswith("OK"):
        log.info("another Lumen instance is running (pid %s, %s) - asked it to come to the front", info.get("pid"), info.get("url"))
        return info
    return None


class SingleInstance:
    def __init__(self, lock_path: Path, on_focus: Callable[[], None], on_eval: Callable[[str], Any] | None = None,
                 on_open: Callable[[list[str]], None] | None = None):
        self.lock_path = lock_path
        self.on_focus = on_focus
        self.on_open = on_open  # ``<token> OPEN ["/path/a.tif", ...]``: a second launch with files hands them over
        self.on_eval = on_eval  # debug mode only: ``<token> EVAL <js>`` runs JavaScript in the window (desktop/dev_eval.py)
        self.token = secrets.token_hex(16)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.control_port: int | None = None
        self._stopping = False

    def acquire(self, url: str) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((HOST, 0))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self.control_port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="lumen-instance", daemon=True)
        self._thread.start()
        payload = {"pid": os.getpid(), "url": url, "control_port": self.control_port, "token": self.token, "started": time.time()}
        try:
            self.lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.warning("could not write %s", self.lock_path, exc_info=True)
        log.info("single-instance guard on 127.0.0.1:%s", self.control_port)

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stopping:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(2.0)
                line = b""
                while b"\n" not in line and len(line) < 1_000_000:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    line += chunk
                text = line.decode("utf-8", "replace").strip()
                if text == f"{self.token} FOCUS":
                    conn.sendall(b"OK\n")
                    conn.close()  # answer first: the client must not wait for UI work
                    threading.Thread(target=self._focus, name="lumen-focus", daemon=True).start()
                elif text.startswith(f"{self.token} OPEN "):
                    try:
                        files = [str(f) for f in json.loads(text[len(self.token) + 6:]) if isinstance(f, str)]
                    except ValueError:
                        files = []
                    conn.sendall(b"OK\n")
                    conn.close()
                    threading.Thread(target=self._open, args=(files,), name="lumen-open", daemon=True).start()
                elif self.on_eval is not None and text.startswith(f"{self.token} EVAL "):
                    try:
                        result = self.on_eval(text[len(self.token) + 6:])
                        conn.sendall(("OK " + json.dumps(result, default=str) + "\n").encode("utf-8"))
                    except Exception as exc:  # noqa: BLE001
                        conn.sendall(("ERROR " + json.dumps(repr(exc)) + "\n").encode("utf-8"))
                else:
                    conn.sendall(b"DENIED\n")
            except Exception:  # noqa: BLE001
                log.debug("control connection failed", exc_info=True)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _open(self, files: list[str]) -> None:
        try:
            if files and self.on_open is not None:
                self.on_open(files)
        except Exception:  # noqa: BLE001
            log.warning("open request failed", exc_info=True)
        self._focus()

    def _focus(self) -> None:
        try:
            self.on_focus()
        except Exception:  # noqa: BLE001
            log.warning("focus request failed", exc_info=True)

    def release(self) -> None:
        self._stopping = True
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(2)
        info = _read_lock(self.lock_path)
        if info and info.get("token") == self.token:
            try:
                self.lock_path.unlink()
            except OSError:
                pass
