#!/usr/bin/env python3
"""Developer helper: talk to a desktop shell started with ``--debug`` through its control socket.

    python desktop/dev_eval.py [--home DIR] "document.title"          run JavaScript in the window, print the JSON result
    python desktop/dev_eval.py [--home DIR] --key cmd+z               deliver a key press through the native event path (macOS)
    python desktop/dev_eval.py [--home DIR] --open a.tif b.tif        open files in the running window
    python desktop/dev_eval.py "__MODAL__ Cancel"                       click a button of the visible alert (macOS)
    python desktop/dev_eval.py "__PANEL__ /path/file.tif"               answer the visible Open / Save panel (macOS)
    python desktop/dev_eval.py "__FOCUS__"                              bring the window to the front (keys need it)
    python desktop/dev_eval.py "__PY__ <expression>"                    evaluate Python on the main thread (macOS)

Reads ``<lumen home>/desktop/desktop.lock`` (control port + token, written by the running instance).  Only a shell
started with ``--debug`` (or ``LUMEN_DESKTOP_DEBUG=1``) answers ``EVAL`` requests."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path


def send(home: Path, payload: str, timeout: float = 30.0) -> str:
    lock = home / "desktop" / "desktop.lock"
    info = json.loads(lock.read_text(encoding="utf-8"))
    with socket.create_connection(("127.0.0.1", int(info["control_port"])), timeout=timeout) as s:
        s.sendall(f"{info['token']} EVAL {payload}\n".encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = s.recv(65536)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode("utf-8", "replace").strip()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("js", nargs="*", help="JavaScript to evaluate (or the files with --open)")
    p.add_argument("--home", help="Lumen folder of the running instance (default: LUMEN_HOME or ~/.lumen)")
    p.add_argument("--key", help="a key combination such as cmd+z (macOS)")
    p.add_argument("--open", action="store_true", help="treat the positional arguments as files to open")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(argv)
    home = Path(args.home or os.environ.get("LUMEN_HOME") or Path.home() / ".lumen").expanduser()
    if not (home / "desktop" / "desktop.lock").is_file():
        print(f"no running instance ({home / 'desktop' / 'desktop.lock'} missing)", file=sys.stderr)
        return 2
    if args.key:
        payload = f"__KEY__ {args.key}"
    elif args.open:
        payload = "__OPEN__ " + json.dumps([str(Path(f).expanduser().resolve()) for f in args.js])
    else:
        payload = " ".join(args.js).replace("\n", " ")
    reply = send(home, payload, args.timeout)
    print(reply)
    return 0 if reply.startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
