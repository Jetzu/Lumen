"""`lumen [files...]` – start the server and open the app in the browser."""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _configure_console() -> None:
    """Windows consoles and pipes may use a legacy code page (cp1252): make sure a non-ASCII
    character in a message (a file name, 'µm', an en dash) is replaced instead of crashing print()."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _run_startup_scripts(session) -> None:
    """Run <lumen home>/startup.py (Python API) and startup.ijm (ImageJ macro) once; $LUMEN_HOME overrides ~/.lumen.
    Failures are written to the session log and stderr – they never stop the server from starting."""
    try:
        from .core.ops.plugins_ops import run_startup_scripts
        for r in run_startup_scripts(session):
            if not r["ok"]:
                print(f"[lumen] startup script {r['path']} failed: {r['error']}", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[lumen] startup scripts skipped: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    _configure_console()
    ap = argparse.ArgumentParser(prog="lumen", description="Lumen – scientific image analysis")
    ap.add_argument("files", nargs="*", help="images to open at startup")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", action="store_true", help="open in a native window (needs pywebview)")
    args = ap.parse_args(argv)

    import uvicorn
    from .server.app import app
    from .core.session import SESSION

    _run_startup_scripts(SESSION)
    for f in args.files:
        try:
            SESSION.open_file(f)
        except Exception as e:  # pragma: no cover
            print(f"Could not open {f}: {e}", file=sys.stderr)

    port = _free_port(args.port)
    url = f"http://{args.host}:{port}"
    print(f"Lumen running at {url}")
    if args.window:
        try:
            import webview  # type: ignore

            def serve() -> None:
                uvicorn.run(app, host=args.host, port=port, log_level="warning")

            threading.Thread(target=serve, daemon=True).start()
            webview.create_window("Lumen", url, width=1500, height=950)
            webview.start()
            return
        except ImportError:
            print("pywebview not installed – falling back to the browser", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
