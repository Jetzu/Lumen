"""The desktop shell: arguments, logging, single instance, the in-process Lumen server, the pywebview window with its
splash page, native menus, files from Finder / Explorer / the command line, and a clean shutdown.

    python -m lumen_desktop [files...] [--smoke] [--port N] [--home DIR] [--debug] [--gui qt|gtk|...]

Environment: LUMEN_HOME (Lumen's per-user folder, shared with the ``lumen`` command), LUMEN_DESKTOP_PORT,
LUMEN_DESKTOP_SMOKE=1 (start the server, check the command registry and a round trip through the image API, print
one line, exit - no window), LUMEN_DESKTOP_DEBUG=1 (web inspector + the test hooks of the control socket),
PYWEBVIEW_GUI (engine override), LUMEN_DESKTOP_SMOKE_OUT (file to append the smoke result line to, for windowed
builds without stdout).  Logs: <lumen home>/desktop/desktop.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from lumen_desktop import paths
from lumen_desktop.server import BackendServer, http_get, http_post_json, pick_port

log = logging.getLogger("lumen.desktop")

EXIT_OK, EXIT_ERROR, EXIT_SERVER, EXIT_SMOKE_FAILED = 0, 1, 6, 7
MIN_COMMANDS = 400  # the smoke check's sanity floor for the registry (499 at the time of writing)


# ----------------------------------------------------------------------------- arguments / logging


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lumen-desktop", description="Lumen desktop app (a native window around the local Lumen server).")
    p.add_argument("files", nargs="*", help="images or folders to open")
    p.add_argument("--smoke", action="store_true", help="start the server, check it, print one line and exit without a window")
    p.add_argument("--port", type=int, help="server port (default: the port used last time when free, else a free one)")
    p.add_argument("--home", help="Lumen's per-user folder (default ~/.lumen; env LUMEN_HOME)")
    p.add_argument("--debug", action="store_true", help="enable the web inspector and the control-socket test hooks")
    p.add_argument("--gui", help="pywebview engine override (cocoa, edgechromium, qt, gtk); env PYWEBVIEW_GUI")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    args, unknown = p.parse_known_args(argv)  # Finder / Explorer may pass launch-service arguments (-psn_…)
    args.files = [f for f in [*args.files, *unknown] if not f.startswith("-psn") and Path(f).expanduser().exists()]
    return args


def setup_logging(log_dir: Path, *, to_console: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "desktop.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if to_console and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pywebview").setLevel(logging.INFO)

    def _hook(exc_type, exc, tb):  # noqa: ANN001
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
    threading.excepthook = lambda a: log.critical("unhandled exception in thread %s", a.thread.name if a.thread else "?", exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    try:  # diagnostics: `kill -USR1 <pid>` appends the Python stack of every thread to <log dir>/threads.log
        import faulthandler
        import signal

        if hasattr(signal, "SIGUSR1"):
            _dump = open(log_dir / "threads.log", "a", encoding="utf-8")  # noqa: SIM115 - kept open for the process lifetime
            faulthandler.register(signal.SIGUSR1, file=_dump, all_threads=True)
    except Exception:  # noqa: BLE001
        pass
    return log_file


def _say(line: str) -> None:
    """stdout when there is one (console runs / CI); the smoke result file when asked for."""
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001 - windowed builds have no stdout
        pass
    out = os.environ.get("LUMEN_DESKTOP_SMOKE_OUT")
    if out:
        try:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            log.warning("could not write %s", out, exc_info=True)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.home:
        os.environ["LUMEN_HOME"] = args.home
    if args.port:
        os.environ["LUMEN_DESKTOP_PORT"] = str(args.port)
    if args.gui:
        os.environ["PYWEBVIEW_GUI"] = args.gui
    smoke = bool(args.smoke) or os.environ.get("LUMEN_DESKTOP_SMOKE") == "1"
    debug = bool(args.debug) or os.environ.get("LUMEN_DESKTOP_DEBUG") == "1"

    paths.prepare_environment()
    try:
        import lumen
        version = getattr(lumen, "__version__", "?")
    except Exception as exc:  # noqa: BLE001 - the bundle is broken; say so where we can
        _say(f"Lumen: the image engine could not be imported: {exc!r}")
        return EXIT_ERROR
    if args.version:
        _say(f"Lumen {version}")
        return EXIT_OK

    ddir = paths.desktop_dir()
    log_file = setup_logging(ddir, to_console=smoke or debug or not paths.is_frozen())
    if debug:
        logging.getLogger("lumen.desktop").setLevel(logging.DEBUG)
    log.info("---- Lumen desktop %s starting: python %s, %s %s (%s), frozen=%s, pid %s", version, platform.python_version(),
             platform.system(), platform.release(), platform.machine(), paths.is_frozen(), os.getpid())
    log.info("lumen home %s; files %s; args %s", paths.lumen_home(), args.files, {k: v for k, v in vars(args).items() if k != "files"})

    from lumen_desktop.state import DesktopState

    state = DesktopState(ddir / "desktop-state.json")
    requested = os.environ.get("LUMEN_DESKTOP_PORT")
    preferred = int(requested) if requested and requested.isdigit() else state.last_port
    port = pick_port(preferred)
    server = BackendServer(port)

    if smoke:
        return run_smoke(server, args.files, log_file)

    from lumen_desktop.instance import notify_existing

    existing = notify_existing(ddir / "desktop.lock", files=[str(Path(f).expanduser().resolve()) for f in args.files])
    if existing:
        _say(f"Lumen is already running ({existing.get('url')}); " + ("handed it the files." if args.files else "brought its window to the front."))
        return EXIT_OK

    shell = Shell(server=server, state=state, version=version, log_file=log_file, debug=debug, files=args.files)
    return shell.run()


# ----------------------------------------------------------------------------- smoke mode (CI, tests)


def run_smoke(server: BackendServer, files: list[str], log_file: Path) -> int:
    """No window: start, wait for the registry, create an image and read it back through the API (and open the
    given files, when any), print one line, stop."""
    server.start()
    if not server.wait_ready(180):
        log.error("server did not become ready (error: %r)", server.error)
        server.stop()
        _say("LUMEN_DESKTOP_SMOKE_FAILED reason=startup")
        return EXIT_SERVER
    problems: list[str] = []
    if server.n_commands < MIN_COMMANDS:
        problems.append(f"only {server.n_commands} commands registered")
    try:
        status, body = http_post_json(f"{server.url}/api/ops/file.new/run", {"params": {"title": "smoke", "type": "8-bit", "width": 64, "height": 48, "fill": "Ramp"}})
        if status != 200 or not body.get("new_images"):
            problems.append(f"file.new answered HTTP {status}")
        for f in files:
            status, body = http_post_json(f"{server.url}/api/images/open", {"path": str(Path(f).expanduser().resolve())}, timeout=120)
            if status != 200:
                problems.append(f"opening {f}: HTTP {status} {str(body)[:120]}")
        status, raw, _ = http_get(f"{server.url}/api/state", timeout=30)
        n_images = len(json.loads(raw or b"{}").get("images") or []) if status == 200 else -1
        if n_images < 1 + len(files):
            problems.append(f"{n_images} images in the session after the round trip")
        status, raw, _ = http_get(f"{server.url}/", timeout=30)
        if status != 200 or b"<script" not in raw:
            problems.append(f"the UI page answered HTTP {status}")
        problems += _smoke_libraries(server.url)
    except Exception as exc:  # noqa: BLE001
        log.exception("smoke round trip failed")
        problems.append(repr(exc))
    server.stop()
    ok = not problems
    line = (f"LUMEN_DESKTOP_SMOKE_{'OK' if ok else 'FAILED'} url={server.url} startup_ms={server.startup_ms} commands={server.n_commands} "
            f"home={paths.lumen_home()} log={log_file}" + ("" if ok else " problems=" + json.dumps(problems)))
    log.info(line)
    _say(line)
    return EXIT_OK if ok else EXIT_SMOKE_FAILED


def _smoke_libraries(url: str) -> list[str]:
    """Touch what a bundle can silently lack because it is imported lazily: scikit-image's sample data, SciPy filters,
    the particle analyser, FFT, a histogram plot (Matplotlib), the ImageJ macro layer, the bundled docs, and the
    optional readers / scikit-learn through the Python console.  Returns the problems found."""
    problems: list[str] = []
    steps: list[tuple[str, dict]] = [
        ("file.sample", {"name": "Blobs"}),
        ("filters.median", {"radius": 2.0}),
        ("filters.gaussian", {"sigma": 1.5}),
        ("threshold.apply", {"method": "Otsu", "dark_background": False}),
        ("analyze.particles", {"min_size": 5.0, "show": "Nothing"}),
        ("analyze.histogram", {}),
        ("process.fft", {}),
        ("script.run_ij_macro", {"macro": 'newImage("m", "8-bit ramp", 64, 64, 1); run("Gaussian Blur...", "sigma=2"); run("Measure");'}),
    ]
    for op, params in steps:
        status, body = http_post_json(f"{url}/api/ops/{op}/run", {"params": params}, timeout=120)
        if status != 200:
            problems.append(f"{op}: HTTP {status} {str(body.get('error') or body)[:160]}")
    status, raw, _ = http_get(f"{url}/api/docs/README", timeout=30)
    if status != 200 or len(raw) < 1000:
        problems.append(f"bundled docs: HTTP {status}, {len(raw)} bytes")
    core = ["sklearn.ensemble", "skimage.segmentation", "scipy.ndimage", "matplotlib.pyplot", "imagecodecs", "tifffile", "joblib", "pooch"]
    # the [formats] readers ship with the packaged app; a source checkout may not have them installed
    optional = ["h5py", "zarr", "nd2", "czifile", "readlif", "pydicom", "nibabel", "mrcfile", "openpyxl", "oiffile", "olefile"]
    mods = core + (optional if paths.is_frozen() else [])
    code = ("import importlib\n"
            f"mods = {mods!r}\n"
            "missing = []\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as e:\n"
            "        missing.append(m + ': ' + repr(e)[:80])\n"
            "print('MISSING ' + '; '.join(missing) if missing else 'ALL_IMPORTS_OK')\n")
    status, body = http_post_json(f"{url}/api/script/run", {"code": code}, timeout=120)
    out = str(body.get("stdout") or "") + str(body.get("error") or "")
    if status != 200 or "ALL_IMPORTS_OK" not in out:
        problems.append(f"python console imports: HTTP {status} {out.strip()[:300]}")
    return problems


# ----------------------------------------------------------------------------- the window


class Shell:
    def __init__(self, *, server: BackendServer, state: Any, version: str, log_file: Path, debug: bool, files: list[str]):
        self.server = server
        self.state = state
        self.version = version
        self.log_file = log_file
        self.debug = debug
        self.pending_files = [str(Path(f).expanduser().resolve()) for f in files]
        self.window: Any = None
        self.instance: Any = None
        self.api: Any = None
        self.is_mac = sys.platform == "darwin"
        self.is_linux = sys.platform.startswith("linux")
        self.native_zoom = False
        self.loaded_app = False
        self._about: Any = None
        self._shutdown_done = False

    # ------------------------------------------------------------------ run / shutdown
    def run(self) -> int:
        if self.is_linux:
            os.environ.setdefault("QT_API", "pyqt6")
            os.environ.setdefault("PYWEBVIEW_GUI", "qt")
            if os.environ.get("LUMEN_QT_SANDBOX") != "1":  # Chromium's sandbox needs user namespaces (blocked on e.g. Ubuntu 24.04 outside snaps)
                os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        try:
            import webview
        except Exception:  # noqa: BLE001
            log.exception("pywebview could not be imported")
            _say("Lumen: pywebview is not installed (see desktop/requirements-*.txt) - run `lumen` to use the browser instead")
            return EXIT_ERROR
        from lumen_desktop.bridge import DesktopApi
        from lumen_desktop.instance import SingleInstance
        from lumen_desktop.menu import build_menus
        from lumen_desktop.splash import SPLASH_HTML
        from lumen_desktop.state import MIN_HEIGHT, MIN_WIDTH

        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False  # --debug enables right-click > Inspect, nothing opens by itself

        actions = self.actions()
        self.api = DesktopApi(self.state, self.server.url, {k: actions[k] for k in ("reload", "zoom_in", "zoom_out", "zoom_reset", "fullscreen")})
        self.api._on_unsaved = self.set_unsaved  # the page reports its unsaved images (frontend/src/native.ts)
        screens: list[tuple[int, int, int, int]] = []
        try:
            screens = [(int(s.x), int(s.y), int(s.width), int(s.height)) for s in webview.screens]
        except Exception:  # noqa: BLE001
            log.debug("screens unavailable", exc_info=True)
        geometry = self.state.geometry(screens)
        log.info("window geometry %s", geometry)
        self.window = webview.create_window("Lumen", html=SPLASH_HTML, js_api=self.api, min_size=(MIN_WIDTH, MIN_HEIGHT),
                                            text_select=True, background_color="#0d0f14", confirm_close=False, **geometry)
        self.api.window = self.window
        w = self.window
        w.events.loaded += self.on_loaded
        w.events.resized += lambda width, height: self.state.remember_size(width, height)
        w.events.moved += lambda x, y: self.state.remember_position(x, y)
        w.events.maximized += lambda: self.state.remember_maximized(True)
        w.events.restored += lambda: self.state.remember_maximized(False)
        w.events.closing += self.on_closing
        w.events.closed += self.on_closed
        w.events.shown += self.on_shown

        ddir = paths.desktop_dir()
        self.instance = SingleInstance(ddir / "desktop.lock", self.focus, on_eval=(self.debug_eval if self.debug else None), on_open=self.open_paths)
        menus = build_menus(actions, include_view=not self.is_mac)  # macOS: the items go into the default View menu
        icon = paths.assets_dir() / "icon.png"
        storage = ddir / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        self.install_signal_handlers()
        code = EXIT_OK
        try:
            webview.start(func=self.bootstrap, menu=menus, private_mode=False, storage_path=str(storage), debug=self.debug,
                          icon=str(icon) if icon.is_file() else None,
                          localization={"global.quitConfirmation": "Some images have unsaved changes. Quit Lumen anyway?"})
        except KeyboardInterrupt:
            log.info("interrupted")
        except Exception:  # noqa: BLE001
            log.exception("GUI loop failed")
            code = EXIT_ERROR
        finally:
            self.shutdown()
        return code

    def install_signal_handlers(self) -> None:
        """SIGTERM / SIGHUP (kill, session end) close the window so the normal shutdown runs.  Python-level
        handlers only run when the main thread executes Python code, which the macOS keepalive tick
        (see bootstrap) guarantees once a second; pywebview itself turns Ctrl+C into KeyboardInterrupt."""
        import signal

        def _handler(signum, _frame):  # noqa: ANN001
            log.info("signal %s - quitting", signum)
            threading.Thread(target=self.act_quit, kwargs={"force": True}, name="lumen-signal", daemon=True).start()

        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError):
                    pass

    def shutdown(self) -> None:
        """Idempotent; runs from the ``closing`` event (macOS terminates the process right after Cmd+Q,
        ``webview.start`` never returns) and again from ``run()``'s finally block."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        log.info("shutting down")
        try:
            self.state.last_port = self.server.port
            self.state.save()
        except Exception:  # noqa: BLE001
            log.debug("state save failed", exc_info=True)
        if self.instance is not None:
            self.instance.release()
        self.server.stop()
        log.info("bye")
        logging.shutdown()

    # ------------------------------------------------------------------ startup sequence (worker thread)
    def bootstrap(self) -> None:
        from lumen_desktop.splash import set_status_js

        w = self.window
        try:
            if self.is_mac:
                from lumen_desktop import macos

                icon = paths.assets_dir() / "icon.png"
                if not paths.is_frozen() and icon.is_file():
                    macos.set_dock_icon(str(icon))
                macos.start_keepalive_tick()
            self.server.start()
            if not self.server.wait_ready(180):
                err = repr(self.server.error) if self.server.error else "no answer from /api/meta within 180 s"
                log.error("server failed to start: %s", err)
                w.evaluate_js(set_status_js("The image engine could not be started.", err, f"Details: {self.log_file}"))
                return
            self.instance.acquire(self.server.url)
            self.run_startup_scripts()
            if self.is_mac:
                from lumen_desktop import macos

                macos.install_open_documents_handler(self.open_paths)
            files, self.pending_files = self.pending_files, []
            if files:
                w.evaluate_js(set_status_js(f"Opening {len(files)} file{'s' if len(files) != 1 else ''}…"))
                self.open_paths(files, focus=False)
            self.load_app()
        except Exception as exc:  # noqa: BLE001
            log.exception("startup failed")
            try:
                w.evaluate_js(set_status_js("Lumen could not start.", repr(exc), f"Details: {self.log_file}"))
            except Exception:  # noqa: BLE001
                pass

    def run_startup_scripts(self) -> None:
        """<lumen home>/startup.py and startup.ijm, like the ``lumen`` command (failures go to the session log)."""
        try:
            from lumen.core.ops.plugins_ops import run_startup_scripts
            from lumen.core.session import SESSION

            for r in run_startup_scripts(SESSION):
                if not r["ok"]:
                    log.warning("startup script %s failed: %s", r["path"], r["error"])
        except Exception:  # noqa: BLE001
            log.warning("startup scripts skipped", exc_info=True)

    def load_app(self) -> None:
        log.info("loading %s", self.server.url)
        self.loaded_app = True
        self.window.load_url(self.server.url + "/")

    # ------------------------------------------------------------------ files
    def open_paths(self, files: list[str], focus: bool = True) -> None:
        """Open images / folders (Finder, a second launch, File ▸ Open Image… in the native menu) through the server's
        API, like the page's own Open: the page sees the new tabs through its event socket."""
        opened, failed = [], []
        for f in files:
            try:
                status, body = http_post_json(f"{self.server.url}/api/images/open", {"path": f}, timeout=600)
            except Exception as exc:  # noqa: BLE001
                status, body = 0, {"error": repr(exc)}
            if status == 200:
                opened.append(Path(f).name)
            else:
                failed.append(f"{Path(f).name}: {body.get('error') or body.get('detail') or f'HTTP {status}'}")
                log.warning("could not open %s: %s", f, body)
        if opened and self.loaded_app:
            self.toast(f"Opened {', '.join(opened[:3])}{' …' if len(opened) > 3 else ''}")
        if failed:
            self.toast("Could not open " + "; ".join(failed[:2]))
        if focus:
            self.focus()

    def set_unsaved(self, count: int) -> bool:
        """Called by the page when its number of images with unsaved changes changes: arms pywebview's quit
        confirmation only while there is something to lose (Fiji asks too)."""
        try:
            n = int(count)
        except (TypeError, ValueError):
            return False
        if self.window is not None:
            if bool(getattr(self.window, "confirm_close", False)) != (n > 0):
                log.info("unsaved images: %d - quit confirmation %s", n, "on" if n > 0 else "off")
            self.window.confirm_close = n > 0
        return True

    # ------------------------------------------------------------------ events
    def on_loaded(self) -> None:
        if not self.loaded_app:
            return
        try:
            url = self.window.get_current_url() or ""
        except Exception:  # noqa: BLE001
            log.debug("get_current_url failed", exc_info=True)
            url = ""
        log.debug("loaded: %s", url)
        if not url.startswith(self.server.url):
            return
        self.apply_zoom()
        self.inject()

    def on_shown(self) -> None:
        log.debug("window shown")
        try:
            self.state.remember_size(self.window.width, self.window.height)
            self.state.remember_position(self.window.x, self.window.y)
        except Exception:  # noqa: BLE001
            log.debug("geometry query failed", exc_info=True)
        if self.is_mac:
            from lumen_desktop import macos

            # pywebview rebuilds the menubar when the window becomes key (shortly after "shown"); apply a few times
            for delay in (0.3, 1.2, 3.0):
                time.sleep(delay)
                macos.polish_menubar(self.mac_view_items())

    def on_closing(self) -> None:
        """Synchronous on the UI thread, *before* pywebview asks "Quit anyway?" (no window queries here - they would wait
        for the UI thread).  The window state is saved now; the server is stopped now only when no question follows.
        With unsaved images the answer may be Cancel, so stopping waits for the ``closed`` event - and a confirmed ⌘Q,
        which ends the process right after the answer, needs no orderly stop (nothing is left to save)."""
        try:
            self.state.last_port = self.server.port
            self.state.save()
        except Exception:  # noqa: BLE001
            log.debug("state save failed", exc_info=True)
        if self.window is not None and getattr(self.window, "confirm_close", False):
            log.info("closing requested with unsaved images - waiting for the answer")
            return
        self.shutdown()

    def on_closed(self) -> None:
        self.shutdown()

    def focus(self) -> None:
        try:
            self.window.restore()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.window.show()
        except Exception:  # noqa: BLE001
            log.debug("show failed", exc_info=True)
        if self.is_mac:
            from lumen_desktop import macos

            macos.activate()

    def debug_eval(self, text: str) -> Any:
        """``--debug`` control-socket hook (desktop/dev_eval.py, the desktop tests): JavaScript in the window, or
        ``__KEY__ cmd+z`` to deliver a synthetic key press through the native event path (macOS)."""
        if text.startswith("__KEY__ "):
            if not self.is_mac:
                return False
            from lumen_desktop import macos

            return macos.send_key(text[len("__KEY__ "):])
        if text.startswith("__PY__ ") and self.is_mac:  # evaluate a Python expression on the main thread (debug only)
            from lumen_desktop import macos

            expr = text[len("__PY__ "):]
            return macos._on_main(lambda: repr(eval(expr, {"AppKit": __import__("AppKit"), "Foundation": __import__("Foundation"), "shell": self})))
        if text.strip() == "__FOCUS__":
            self.focus()
            return True
        if text.startswith("__MODAL__ "):  # click a button of the open alert ("Quit" / "Cancel"), macOS
            if not self.is_mac:
                return False
            from lumen_desktop import macos

            return macos.modal_click(text[len("__MODAL__ "):])
        if text.startswith("__PANEL__ "):  # answer the open Open / Save panel with a path, macOS
            if not self.is_mac:
                return False
            from lumen_desktop import macos

            return macos.panel_choose(text[len("__PANEL__ "):])
        if text.startswith("__OPEN__ "):
            self.open_paths(json.loads(text[len("__OPEN__ "):]), focus=False)
            return True
        return self.window.evaluate_js(text)

    # ------------------------------------------------------------------ page helpers
    def inject(self) -> None:
        from lumen_desktop.bridge import injected_js

        cfg = {"zoom": self.state.zoom, "nativeZoom": self.native_zoom, "nativeDownloads": True, "nativeShortcuts": self.is_mac}
        try:
            self.window.evaluate_js(injected_js(cfg))
        except Exception:  # noqa: BLE001
            log.warning("script injection failed", exc_info=True)

    def apply_zoom(self) -> None:
        z = self.state.zoom
        if self.is_mac:
            from lumen_desktop import macos

            self.native_zoom = macos.set_page_zoom(self.window, z)
        if not self.native_zoom:
            try:
                self.window.evaluate_js(f"window.__lumen ? window.__lumen.configure({{zoom:{z},nativeZoom:false,nativeDownloads:true,nativeShortcuts:{str(self.is_mac).lower()}}}) : "
                                        f"(document.documentElement.style.zoom={z!r})")
            except Exception:  # noqa: BLE001
                log.debug("css zoom failed", exc_info=True)
        self.state.save()

    def toast(self, message: str) -> None:
        if self.api is not None:
            self.api.toast(message)

    def page_call(self, fn: str) -> None:
        """Ask the page to do something it owns (open its docs, its shortcut overlay): ``window.__lumenApp.<fn>()``,
        registered by frontend/src/native.ts."""
        if not self.loaded_app:
            return
        try:
            self.window.evaluate_js(f"window.__lumenApp && window.__lumenApp.{fn} && window.__lumenApp.{fn}()")
        except Exception:  # noqa: BLE001
            log.debug("page call %s failed", fn, exc_info=True)

    # ------------------------------------------------------------------ menu actions
    def actions(self) -> dict[str, Callable[[], Any]]:
        return {
            "open_images": self.act_open_images,
            "open_folder": self.act_open_folder,
            "open_home": self.act_open_home,
            "open_in_browser": self.act_open_in_browser,
            "quit": self.act_quit,
            "reload": self.act_reload,
            "zoom_in": lambda: self.act_zoom(+1),
            "zoom_out": lambda: self.act_zoom(-1),
            "zoom_reset": lambda: self.act_zoom(0),
            "fullscreen": self.act_fullscreen,
            "docs": lambda: self.page_call("openDocs"),
            "shortcuts": lambda: self.page_call("openShortcuts"),
            "show_log": self.act_show_log,
            "about": self.act_about,
        }

    def mac_view_items(self) -> list:
        """(title, fn, key equivalent, extra modifiers) for pywebview's View menu on macOS.  ⌘R is left to the page
        (it runs the script in Lumen's editor) - Reload has no key; ⌘= / ⌘- / ⌘0 are free in Lumen."""
        return [
            ("Reload", self.act_reload, "", 0),
            None,
            ("Zoom In", lambda: self.act_zoom(+1), "=", 0),
            ("Zoom Out", lambda: self.act_zoom(-1), "-", 0),
            ("Actual Size", lambda: self.act_zoom(0), "0", 0),
            None,
        ]

    def act_open_images(self) -> None:
        if self.api is None:
            return
        files = self.api.native_open_images()
        if files:
            self.open_paths(files, focus=False)

    def act_open_folder(self) -> None:
        if self.api is None:
            return
        folder = self.api.native_pick_folder("Open folder as stack")
        if folder:
            self.open_paths([folder], focus=False)

    def act_open_home(self) -> None:
        from lumen_desktop.bridge import open_in_file_manager

        home = paths.lumen_home()
        home.mkdir(parents=True, exist_ok=True)
        open_in_file_manager(home)

    def act_open_in_browser(self) -> None:
        from lumen_desktop.bridge import open_url

        open_url(self.server.url + "/")

    def act_show_log(self) -> None:
        from lumen_desktop.bridge import open_in_file_manager

        open_in_file_manager(self.log_file if self.log_file.is_file() else self.log_file.parent)

    def act_quit(self, force: bool = False) -> None:
        """Close the window (quits the app).  ``force`` (SIGTERM / SIGHUP) skips the unsaved-images question - nobody
        is there to answer it."""
        try:
            if force:
                self.window.confirm_close = False
            self.window.destroy()
        except Exception:  # noqa: BLE001
            log.debug("destroy failed", exc_info=True)

    def act_reload(self) -> None:
        if self.loaded_app:
            try:
                self.window.evaluate_js("location.reload()")
            except Exception:  # noqa: BLE001
                self.window.load_url(self.server.url + "/")

    def act_zoom(self, direction: int) -> None:
        if direction == 0:
            self.state.zoom = 1.0
        else:
            self.state.zoom_step(direction)
        self.apply_zoom()
        self.toast(f"Zoom {int(round(self.state.zoom * 100))}%")

    def act_fullscreen(self) -> None:
        try:
            self.window.toggle_fullscreen()
        except Exception:  # noqa: BLE001
            log.debug("fullscreen failed", exc_info=True)

    def act_about(self) -> None:
        import webview

        from lumen_desktop.splash import about_html

        info = {
            "Version": self.version,
            "Commands": str(self.server.n_commands or "?"),
            "Server": self.server.url,
            "Lumen folder": str(paths.lumen_home()),
            "Log file": str(self.log_file),
            "Python": platform.python_version(),
            "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "Engine": f"pywebview {getattr(webview, '__version__', '?')} / {webview.renderer or '?'}",
        }
        try:
            if self._about is not None:
                self._about.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._about = webview.create_window("About Lumen", html=about_html(info), width=600, height=380, resizable=False, text_select=True, background_color="#13161d")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
