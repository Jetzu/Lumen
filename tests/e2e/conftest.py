"""Fixtures for the Playwright smoke suite (see README.md).

Import-safe on purpose: Playwright is imported lazily inside fixtures, so this directory collects (and
its tests skip) in environments without it, e.g. the plain project venv.

The Lumen server is started once per session from an interpreter that has Lumen installed
(``$LUMEN_PYTHON``, else ``<repo>/.venv``, else the interpreter running pytest) and torn down at the end.
Set ``LUMEN_E2E_URL`` to test an already running server instead.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
START_TIMEOUT = float(os.environ.get("LUMEN_E2E_START_TIMEOUT", "90"))

# Substrings of console messages that are expected noise in headless Chrome and never count as errors.
IGNORED_CONSOLE = (
    "ReadPixels",  # "GL ReadPixels" warnings from the WebGL volume renderer under SwiftShader
    "GL Driver Message",
    "GL_INVALID",
    "SwiftShader",
    "WebGL",  # "WebGL: too many errors …", software-fallback notices
    "GPU stall",
    "Automatic fallback to software",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test in this directory carries the ``e2e`` marker (deselect with ``-m "not e2e"``)."""
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.e2e)


# ------------------------------------------------------------------ server
def server_python() -> str:
    """Interpreter that has Lumen installed: $LUMEN_PYTHON, else the repo's .venv, else this interpreter."""
    env = os.environ.get("LUMEN_PYTHON")
    if env:
        return env
    for cand in (REPO / ".venv" / "bin" / "python", REPO / ".venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    return sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_json(url: str, method: str = "GET", body: Any = None, timeout: float = 30) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def _meta_ok(url: str) -> bool:
    try:
        meta = http_json(url + "/api/meta", timeout=2)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(meta, dict) and "ops" in meta


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)


@pytest.fixture(scope="session")
def server_url() -> Iterator[str]:
    external = os.environ.get("LUMEN_E2E_URL")
    if external:
        url = external.rstrip("/")
        if not _meta_ok(url):
            pytest.skip(f"LUMEN_E2E_URL={url} does not answer /api/meta")
        yield url
        return
    port = int(os.environ.get("LUMEN_E2E_PORT") or _free_port())
    py = server_python()
    cmd = [py, "-m", "lumen", "--no-browser", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: list[str] = []
    announced: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            m = re.search(r"Lumen running at (http://\S+)", line)
            if m:
                announced.append(m.group(1).rstrip("/"))

    threading.Thread(target=pump, daemon=True).start()
    url: str | None = None
    deadline = time.monotonic() + START_TIMEOUT
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            # `lumen` moves to the next free port when the requested one is taken: trust its own announcement
            cand = announced[0] if announced else f"http://127.0.0.1:{port}"
            if _meta_ok(cand):
                url = cand
                break
            time.sleep(0.25)
        if url is None:
            out = "\n".join(lines[-40:])
            _stop(proc)
            if "No module named lumen" in out:
                pytest.skip(f"{py} cannot import lumen - set LUMEN_PYTHON to an interpreter with Lumen installed")
            pytest.fail(f"Lumen server did not start within {START_TIMEOUT:.0f}s ({' '.join(cmd)}):\n{out}")
        yield url
    finally:
        _stop(proc)


# ------------------------------------------------------------------ browser
@pytest.fixture(scope="session")
def playwright_browser() -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed (pip install -e '.[e2e]')")
    pw = sync_api.sync_playwright().start()
    channel = os.environ.get("LUMEN_E2E_CHANNEL", "chrome")
    # Software WebGL so the 3-D view renders the same on machines without a GPU (CI); override with LUMEN_E2E_GPU=1
    args = [] if os.environ.get("LUMEN_E2E_GPU") else ["--ignore-gpu-blocklist", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    try:
        browser = pw.chromium.launch(channel=channel or None, headless=True, args=args)
    except Exception as e:  # playwright.sync_api.Error: executable missing, channel not installed …
        pw.stop()
        pytest.skip(f"browser channel {channel!r} is not available: {str(e).splitlines()[0]}")
    yield browser
    browser.close()
    pw.stop()


class Console:
    """Console errors and uncaught exceptions collected from one page."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def unexpected(self) -> list[str]:
        return [m for m in self.errors if not any(k in m for k in IGNORED_CONSOLE)]


def wait_until(cond: Callable[[], Any], timeout: float = 20, interval: float = 0.2, what: str = "condition") -> Any:
    """Poll ``cond`` until it returns a truthy value; raise AssertionError with ``what`` on timeout."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = cond()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what} (last value: {last!r})")


class LumenApp:
    """High-level driver for the Lumen UI plus REST helpers for assertions and state reset."""

    def __init__(self, page: Any, url: str) -> None:
        self.page = page
        self.url = url

    # --- REST
    def api(self, path: str, method: str = "GET", body: Any = None) -> Any:
        return http_json(self.url + path, method, body)

    def state(self) -> dict[str, Any]:
        return self.api("/api/state")

    def meta(self) -> dict[str, Any]:
        return self.api("/api/meta")

    def active_id(self) -> str:
        st = self.state()
        assert st.get("active"), "no active image"
        return st["active"]

    def reset(self) -> None:
        """Close every image and clear ROIs / results so tests do not depend on each other."""
        st = self.state()
        for img in st.get("images", []):
            self.api(f"/api/images/{img['id']}", "DELETE")
        self.api("/api/rois", "DELETE")
        self.api("/api/results", "DELETE")

    # --- UI
    def palette(self, query: str, name: str) -> None:
        """Open the command palette, type ``query`` and run the entry called exactly ``name``."""
        self.page.keyboard.press("Control+k")
        inp = self.page.locator(".palette input")
        inp.wait_for(timeout=10_000)
        inp.fill(query)
        item = self.page.locator(".palette-item", has=self.page.locator(".pi-name", has_text=re.compile(rf"^{re.escape(name)}$")))
        item.first.click(timeout=10_000)
        self.page.locator(".palette").wait_for(state="hidden", timeout=10_000)

    def sheet(self) -> Any:
        sheet = self.page.locator(".sheet")
        sheet.wait_for(timeout=15_000)
        return sheet

    def apply_sheet(self) -> None:
        """Press the sheet's primary button (Apply / Run) and wait for the sheet to close."""
        self.page.locator(".sheet .sheet-foot button.primary").click()
        self.page.locator(".sheet").wait_for(state="hidden", timeout=60_000)

    def open_sample(self, name: str) -> str:
        """File > Open Sample via the palette; returns the new image id."""
        before = {i["id"] for i in self.state().get("images", [])}
        self.palette("Open Sample", "Open Sample")
        self.sheet().locator("select").first.select_option(label=name)
        self.apply_sheet()
        new = wait_until(lambda: [i["id"] for i in self.state().get("images", []) if i["id"] not in before], 60, what=f"sample {name!r} to open")
        # the UI has caught up once the status bar shows the image dimensions
        self.page.locator(".statusbar").get_by_text(re.compile(r"\d+×\d+ px")).wait_for(timeout=30_000)
        return new[0]

    def section(self, title: str, open: bool = True) -> Any:
        """Inspector section with this title; expand or collapse it and return its body locator."""
        # The head row (div.section-head, class "open" when expanded) holds the disclosure <button
        # class="section-toggle" aria-expanded> whose text is the title immediately followed by an optional
        # badge ("ROI Manager (12)"), so match the title text node itself rather than the concatenated text.
        toggle = self.page.locator(f"xpath=//button[contains(@class,'section-toggle')][text()={title!r}]")
        toggle.first.wait_for(timeout=15_000)
        is_open = (toggle.first.get_attribute("aria-expanded") or "") == "true"
        if is_open != open:
            toggle.first.click()
        head = toggle.first.locator("xpath=ancestor::div[contains(@class,'section-head')][1]")
        body = head.locator("xpath=following-sibling::div[contains(@class,'section-body')]")
        if open:
            body.wait_for(timeout=10_000)
        return body

    def drag(self, locator: Any, fx0: float, fy0: float, fx1: float, fy1: float, steps: int = 12) -> None:
        """Pointer drag across ``locator`` between two fractional positions of its bounding box."""
        box = locator.bounding_box()
        assert box, "element has no bounding box"
        x0, y0 = box["x"] + box["width"] * fx0, box["y"] + box["height"] * fy0
        x1, y1 = box["x"] + box["width"] * fx1, box["y"] + box["height"] * fy1
        self.page.mouse.move(x0, y0)
        self.page.mouse.down()
        self.page.mouse.move(x1, y1, steps=steps)
        self.page.mouse.up()


@pytest.fixture
def console() -> Console:
    return Console()


@pytest.fixture
def page(playwright_browser: Any, server_url: str, console: Console) -> Iterator[Any]:
    """A fresh browser context on a freshly reset Lumen session, loaded and ready."""
    LumenApp(None, server_url).reset()
    ctx = playwright_browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()

    def on_console(msg: Any) -> None:
        if msg.type == "error":
            console.errors.append(f"console.error: {msg.text}")

    pg.on("console", on_console)
    pg.on("pageerror", lambda err: console.errors.append(f"pageerror: {err}"))
    pg.goto(server_url)
    pg.locator(".app").wait_for(timeout=30_000)
    yield pg
    ctx.close()


@pytest.fixture
def app(page: Any, server_url: str) -> LumenApp:
    return LumenApp(page, server_url)
