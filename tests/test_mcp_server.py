"""The Lumen MCP server (lumen/mcp_server) and the window bridge it uses (lumen/server/extra/agent.py).

A real Lumen server runs in a thread on a free port; the tools are called as plain functions (the MCP SDK wraps them
unchanged) and one test goes through the MCP protocol over stdio.  A "window" is simulated by reporting its state to
POST /api/ui/state, as frontend/src/agentBridge.ts does."""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")

import uvicorn  # noqa: E402

from lumen.core.session import Session  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def lumen(tmp_path, monkeypatch):
    """A running Lumen on its own session + the MCP server module pointed at it."""
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    from lumen.server.app import create_app
    from lumen.server.extra import agent
    agent._WINDOWS.clear()  # window states of an earlier test's server
    session = Session()
    app = create_app(session)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(url + "/api/ui/actions", timeout=0.5).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    monkeypatch.setenv("LUMEN_URL", url)
    from lumen.mcp_server import server as S
    from lumen.mcp_server.client import LumenClient
    monkeypatch.setattr(S, "C", LumenClient())
    S._META.clear()
    S._LAST.clear()
    yield S, session, url
    server.should_exit = True
    th.join(10)


def _window(url: str, **state) -> None:
    """Report a window state like agentBridge.ts does."""
    body = {"client": "test-window", "tool": "pointer", "selections": {}, "panels_open": [], "sheet": None, "seq_applied": 0, **state}
    assert httpx.post(url + "/api/ui/state", json=body).status_code == 200


def test_status_open_run_and_results(lumen):
    S, session, url = lumen
    assert "0 image(s) open" in S.status()
    out = S.open_sample("Blobs")
    assert "blobs" in out.lower() and "256×256" in out
    assert session.active is not None
    S.run_command("Gaussian Blur...", {"sigma": 2})  # by ImageJ name
    assert session.histories[session.active.id].labels()["undo"][-1] == "Gaussian Blur"
    S.run_command("analyze.measure", use_selection=False)
    res = S.get_results()
    assert "1 rows" in res and "Mean" in res
    assert S.undo() == "Undid: Gaussian Blur"  # Measure adds no undo step: the blur is the last one
    assert "blobs" in S.list_images()


def test_find_describe_and_errors(lumen):
    S, *_ = lumen
    found = S.find_commands("gaussian blur")
    assert "filters.gaussian" in found
    assert "Process/Filters" in S.find_commands(menu="Process/Filters", limit=200)
    d = S.describe_command("filters.gaussian")
    assert "sigma" in d and "default 2" in d and 'run("Gaussian Blur...")' in d
    with pytest.raises(Exception, match="unknown command"):
        S.describe_command("filters.gausian_blurr")
    with pytest.raises(Exception, match="No image is open"):
        S.image_info()


def test_statistics_pixels_and_view(lumen):
    S, session, _ = lumen
    S.open_sample("Blobs")
    data = session.active.data[0, 0, 0]
    st = S.get_statistics(roi={"type": "rect", "x": 10, "y": 20, "w": 30, "h": 40})
    sub = data[20:60, 10:40].astype(float)
    assert f"n={sub.size}" in st and f"mean={sub.mean():.6g}" in st and f"max={sub.max():.6g}" in st
    px = S.get_pixels(5, 7, 3, 2)
    rows = px.splitlines()[1:]
    assert [list(map(int, r.split())) for r in rows] == data[7:9, 5:8].tolist()
    text, img = S.view_image()
    assert "blobs" in text and img.data[:8] == b"\x89PNG\r\n\x1a\n"
    out = S.set_display(auto=True, lut="Fire")
    assert "lut=Fire" in out
    assert session.active.display.channels[0].lut == "Fire"


def test_window_bridge_selection_ui_and_events(lumen):
    S, session, url = lumen
    S.open_sample("Blobs")
    iid = session.active.id
    # no window yet: UI actions say so instead of failing silently
    assert "No window is connected" in S.ui("notify", {"message": "hi"})
    oval = {"type": "oval", "x": 60, "y": 60, "w": 80, "h": 70}
    _window(url, selections={iid: oval}, active_image={"id": iid, "name": "blobs.tif", "t": 0, "z": 0, "c": 0})
    assert json.loads(S.get_selection()) == oval
    assert "selection: " in S.status() and '"oval"' in S.status()
    # run_command picks up the user's selection (like running it from the window)
    S.clear_results()
    S.run_command("analyze.measure")
    from lumen.core.roi import to_mask
    n = int(to_mask(oval, session.active.data.shape[-2:]).sum())
    assert session.results.rows[-1]["Area"] == n
    # the recorder of the user's window got the line
    assert "analyze.measure" in S.recorded_macro()
    # events: baseline, then what the "user" changed
    S.events()
    _window(url, selections={iid: oval}, tool="rect", sheet={"op": "filters.median", "name": "Median", "params": {}})
    session.run_op("filters.median", {"radius": 2})
    ev = S.events()
    assert "modified: blobs" in ev and "tool → rect" in ev and "open dialog → filters.median" in ev


def test_ui_command_router(lumen):
    _, _, url = lumen
    r = httpx.post(url + "/api/ui/command", json={"action": "set_tool", "args": {"tool": "oval"}})
    assert r.status_code == 200 and r.json()["seq"] >= 1
    r = httpx.post(url + "/api/ui/command", json={"action": "format_disk", "args": {}})
    assert r.status_code == 400 and "unknown UI action" in r.json()["error"]
    assert httpx.get(url + "/api/ui/state").json()["connected"] is False


def test_roi_manager_tool(lumen):
    S, session, _ = lumen
    S.open_sample("Blobs")
    assert "Added ROI" in S.roi_manager("add", roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10}, name="corner")
    assert "corner" in S.roi_manager("list")
    S.clear_results()
    assert "1 row(s)" in S.roi_manager("measure")
    assert session.results.rows[-1]["Area"] == 100
    with pytest.raises(Exception, match="no ROI"):
        S.get_statistics(roi="nonexistent")


def test_discovery_via_desktop_lock(lumen, tmp_path, monkeypatch):
    S, _, url = lumen
    from lumen.mcp_server.client import LumenClient
    monkeypatch.delenv("LUMEN_URL")
    lock = tmp_path / "home" / "desktop" / "desktop.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"url": url, "pid": 1, "control_port": 1, "token": "x"}))
    c = LumenClient()
    assert c.discover() == url and c.how == "desktop app"


def test_config_printout(capsys):
    from lumen.mcp_server import main
    assert main(["--config"]) == 0
    out = capsys.readouterr().out
    assert "claude mcp add --scope user lumen" in out and '"mcpServers"' in out and "[mcp_servers.lumen]" in out


def test_over_the_mcp_protocol(lumen):
    """The real thing: an MCP client starts `python -m lumen.mcp_server` over stdio and calls tools."""
    S, session, url = lumen
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "lumen.mcp_server"], env={**os.environ, "LUMEN_URL": url})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"status", "run_command", "run_macro", "view_image", "ui", "events"} <= names and len(names) >= 30
                res = await s.call_tool("open_sample", {"name": "Blobs"})
                assert "blobs" in res.content[0].text.lower()
                res = await s.call_tool("run_macro", {"code": 'run("Invert");\nprint("w=" + getWidth());'})
                assert "w=256" in res.content[0].text
                res = await s.call_tool("view_image", {})
                assert any(c.type == "image" for c in res.content)
    asyncio.run(go())
    assert session.histories[session.active.id].labels()["undo"]  # the macro's Invert is an undo step
