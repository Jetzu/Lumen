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


def test_installer_registers_claude_codex_and_desktop(tmp_path, monkeypatch):
    """The installer (lumen/mcp_server/install.py, modelled on the ChemDraw MCP's): idempotent, keeps a backup, never
    overwrites a foreign entry silently, removes only its own."""
    from lumen.mcp_server import install as I
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    desk = tmp_path / "desk.json"
    desk.write_text(json.dumps({"mcpServers": {"lumen": {"command": "/other/tool"}, "pymol": {"command": "x"}}}))
    monkeypatch.setattr(I, "claude_desktop_config_path", lambda: str(desk))
    cc = tmp_path / "cc" / ".claude.json"
    cc.parent.mkdir()
    cc.write_text(json.dumps({"mcpServers": {"chemdraw": {"command": "c"}}, "other": 1}))
    assert I.main(["--target", "python"]) == 0
    data = json.loads(cc.read_text())
    assert data["other"] == 1 and "chemdraw" in data["mcpServers"]
    ent = data["mcpServers"]["lumen"]
    assert ent["type"] == "stdio" and I._is_ours(ent)
    assert (tmp_path / "cc" / ".claude.json.lumen-backup").is_file()
    codex = (tmp_path / "codex" / "config.toml").read_text()
    assert "[mcp_servers.lumen]" in codex
    d = json.loads(desk.read_text())["mcpServers"]
    assert d["lumen"] == {"command": "/other/tool"} and I._is_ours(d["lumen-mcp"])  # the foreign entry is kept
    before = cc.read_text()
    I.main(["--target", "python"])
    assert cc.read_text() == before  # idempotent
    I.main(["--uninstall"])
    assert "lumen" not in json.loads(cc.read_text())["mcpServers"] and "[mcp_servers.lumen]" not in (tmp_path / "codex" / "config.toml").read_text()
    d = json.loads(desk.read_text())["mcpServers"]
    assert "lumen-mcp" not in d and d["lumen"] == {"command": "/other/tool"}


def test_installer_entries_are_recognised():
    from lumen.mcp_server.install import _is_ours
    assert _is_ours({"command": "/Applications/Lumen.app/Contents/MacOS/Lumen", "args": ["--mcp"]})
    assert _is_ours({"command": "C:\\Lumen\\Lumen.exe", "args": ["--mcp"]})
    assert _is_ours({"command": "/x/bin/python", "args": ["-m", "lumen.mcp_server"]})
    assert not _is_ours({"command": "/Applications/Lumen.app/Contents/MacOS/Lumen", "args": []})
    assert not _is_ours({"command": "/other/tool"})


def test_more_tools_preview_ortho_mesh_files_rois_workflows(lumen, tmp_path):
    S, session, url = lumen
    S.open_sample("Blobs")
    before = session.active.data.copy()
    text, img = S.preview_command("filters.gaussian", {"sigma": 3})
    assert "not applied" in text and img.data[:4] == b"\x89PNG"
    assert np.array_equal(session.active.data, before)  # a preview never changes the image
    S.open_sample("Cells (3-D, 2 channels)")
    text, img = S.view_orthogonal(channel=2)
    assert "XZ" in text and img.data[:4] == b"\x89PNG"
    out = S.export_mesh(str(tmp_path / "cells.stl"))
    assert (tmp_path / "cells.stl").stat().st_size > 1000 and "STL" in out
    assert "count_nuclei" in S.workflows()
    assert "1." in S.workflows("count_nuclei")
    assert "lumen" in S.workflows("count_nuclei", script=True)
    (tmp_path / "a.tif").write_bytes(b"x")
    assert "a.tif" in S.list_files(str(tmp_path))
    S.activate_image("blobs.tif")
    S.roi_manager("add", roi={"type": "rect", "x": 1, "y": 2, "w": 10, "h": 12}, name="r1")
    S.roi_file("export", str(tmp_path / "set.zip"))
    S.roi_manager("clear")
    assert "restored" in S.restore_cleared("rois").lower() or session.roi_manager.rois
    S.roi_manager("clear")
    S.roi_file("import", str(tmp_path / "set.zip"))
    assert len(session.roi_manager.rois) == 1
    S.roi_manager("measure")
    S.export_results(str(tmp_path / "r.csv"))
    assert "Area" in (tmp_path / "r.csv").read_text()
    assert "Fire" in S.list_luts()
    S.text_file(str(tmp_path / "m.ijm"), 'print("hi");')
    assert 'print("hi")' in S.text_file(str(tmp_path / "m.ijm"))
    assert "Display reset" in S.set_display(reset=True) or "lut=" in S.set_display(reset=True)
    S.scene({"mode": "mip"})
    assert '"mip"' in S.scene()


# every HTTP endpoint of Lumen is reachable through an MCP tool, or is part of the window's own plumbing
COVERAGE = {
    "/api/meta": "status", "/api/luts": "list_luts", "/api/luts/groups": "list_luts", "/api/state": "status", "/api/fs": "list_files",
    "/api/images/open": "open_image", "/api/images/{image_id}/save": "save_image", "/api/images/{image_id}/download": "save_image",
    "/api/images": "list_images", "/api/images/{image_id}": "image_info|close_image", "/api/images/{image_id}/select": "activate_image",
    "/api/images/{image_id}/plane": "get_statistics|get_pixels", "/api/images/{image_id}/value": "get_pixels",
    "/api/images/{image_id}/display": "set_display", "/api/images/{image_id}/reset_display": "set_display",
    "/api/images/{image_id}/undo": "undo", "/api/images/{image_id}/redo": "redo", "/api/images/{image_id}/rgb.png": "view_image",
    "/api/ops": "find_commands|describe_command", "/api/ops/{op_id}/run": "run_command|run_macro", "/api/ops/{op_id}/preview": "preview_command",
    "/api/jobs": "jobs", "/api/jobs/{job_id}/cancel": "jobs", "/api/rois": "roi_manager", "/api/rois/{roi_id}": "roi_manager",
    "/api/rois/delete": "roi_manager", "/api/rois/combine": "roi_manager", "/api/rois/measure": "roi_manager",
    "/api/session/restore_clear": "restore_cleared",
    "/api/results": "get_results|clear_results", "/api/results.csv": "export_results", "/api/threshold/{image_id}": "clear_threshold",
    "/api/script/run": "run_script", "/api/macro": "recorded_macro", "/api/recorder": "recorded_macro", "/api/recorder/ijm": "recorded_macro",
    "/api/settings": "settings", "/api/docs": "help", "/api/docs/{name}": "help", "/api/textfile": "text_file",
    "/api/workflows": "workflows", "/api/workflows/{workflow_id}": "workflows", "/api/workflows/{workflow_id}/script": "workflows",
    "/api/tracking/model": "get_tracks", "/api/images/{image_id}/ortho": "view_orthogonal", "/api/images/{image_id}/mesh.{fmt}": "export_mesh",
    "/api/scene": "scene", "/api/ui/command": "ui|set_selection", "/api/ui/state": "status|get_selection|events", "/api/ui/actions": "ui",
}
WINDOW_ONLY = {  # the viewer's / editor's own plumbing: binary streams and uploads a browser makes, not features
    "/api/events", "/api/images/upload", "/api/images/{image_id}/planes", "/api/recorder/preview", "/api/images/{image_id}/volume",
    "/api/images/{image_id}/mesh", "/api/images/{image_id}/pick", "/api/scene/animation", "/api/images/{image_id}/animation.gif",
    "/api/tracking/histogram", "/api/rois/export", "/api/rois/import",  # (roi_file uses the file.*_rois commands)
    "/api/ijshim/call", "/api/ijshim/session/{token}", "/api/script/params",
}


def test_every_endpoint_is_covered_by_the_mcp():
    import importlib
    import pkgutil
    from lumen.mcp_server import server as S
    from lumen.server import extra
    from lumen.server.app import create_app
    paths = {getattr(r, "path", "") for r in create_app(Session()).routes}
    for info in pkgutil.iter_modules(extra.__path__):
        router = getattr(importlib.import_module(f"lumen.server.extra.{info.name}"), "router", None)
        if router is not None:
            paths |= {"/api" + r.path for r in router.routes}
    paths = {p for p in paths if p.startswith("/api") and not p.startswith("/api/agent")}
    unclassified = sorted(p for p in paths if p not in COVERAGE and p not in WINDOW_ONLY)
    assert not unclassified, f"endpoints without an MCP tool: {unclassified}"
    tools = {name for name in dir(S) if callable(getattr(S, name))}
    missing = sorted({t for v in COVERAGE.values() for t in v.split("|")} - tools)
    assert not missing, missing
