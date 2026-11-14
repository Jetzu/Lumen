"""The Agent panel's backend (lumen/server/extra/agent_chat.py + lumen/core/agent/agentcli.py) with a fake Claude CLI
that prints Claude Code's stream-json (no real CLI or account needed)."""
from __future__ import annotations

import json
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from lumen.core.session import Session
from lumen.server.app import create_app

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake CLI is a shebang script")

FAKE = textwrap.dedent('''\
    #!{py}
    import json, os, sys
    if "--version" in sys.argv:
        print("fake 1.0"); sys.exit(0)
    argv = sys.argv[1:]
    prompt = argv[argv.index("-p") + 1]
    out = lambda o: print(json.dumps(o), flush=True)
    out({{"type": "system", "subtype": "init", "session_id": "sess-123"}})
    out({{"type": "stream_event", "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "Work"}}}}}})
    out({{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": "mcp__lumen__status", "input": {{}}}}]}}}})
    out({{"type": "user", "message": {{"content": [{{"type": "tool_result", "content": [{{"type": "text", "text": "1 image open"}}]}}]}}}})
    info = {{"prompt": prompt, "allowed": argv[argv.index("--allowedTools") + 1], "url": os.environ.get("LUMEN_URL"),
            "resume": argv[argv.index("--resume") + 1] if "--resume" in argv else None,
            "ctx": argv[argv.index("--append-system-prompt") + 1]}}
    out({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": json.dumps(info)}}]}}}})
    out({{"type": "result", "result": json.dumps(info), "duration_ms": 5}})
''')


@pytest.fixture
def client(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(py=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("LUMEN_CLAUDE_EXE", str(exe))
    monkeypatch.setenv("LUMEN_CODEX_EXE", str(tmp_path / "no-codex"))
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    from lumen.server.extra import agent_chat
    agent_chat.HUB._found = (0.0, {})
    agent_chat.HUB.runs.clear()
    agent_chat.HUB.active.clear()
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return TestClient(create_app(s), base_url="http://127.0.0.1:8770")


H = {"X-Lumen-Agent": "1"}


def _events(c, run_id):
    out = []
    with c.stream("GET", f"/api/agent/events/{run_id}") as r:
        kind = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                out.append((kind, json.loads(line[6:])))
                if kind == "end":
                    break
    return out


def test_health_lists_the_cli_and_settings(client):
    h = client.get("/api/agent/health").json()
    assert h["ok"] and h["app"] == "lumen" and h["backends"]["claude"].endswith("claude")
    assert h["default_backend"] == "claude" and "effort_choices" in h["settings"]


def test_guard_refuses_foreign_pages(client):
    assert client.post("/api/agent/run", json={"prompt": "hi"}).status_code == 403  # no X-Lumen-Agent header
    r = client.post("/api/agent/run", json={"prompt": "hi"}, headers={**H, "Origin": "https://evil.example"})
    assert r.status_code == 403
    evil = TestClient(client.app, base_url="http://attacker.example")  # DNS rebinding: a non-loopback Host
    assert evil.get("/api/agent/health").status_code == 403


def test_run_streams_events_and_resumes_the_session(client):
    r = client.post("/api/agent/run", json={"backend": "claude", "prompt": "count the blobs"}, headers=H).json()
    ev = _events(client, r["run_id"])
    kinds = [k for k, _ in ev]
    assert kinds[0] == "session" and "delta" in kinds and "tool" in kinds and "tool_result" in kinds and kinds[-1] == "end"
    assert dict(ev)["tool"].startswith("lumen/status")
    info = json.loads(ev[-1][1]["final_text"])
    assert info["prompt"] == "count the blobs" and info["allowed"] == "mcp__lumen"  # only Lumen's MCP tools
    assert info["url"] == "http://127.0.0.1:8770"  # the MCP server the CLI starts drives exactly this Lumen
    assert "Active image: blobs.tif" in info["ctx"]  # what the window shows goes along
    assert ev[-1][1]["status"] == "done"
    # the next turn resumes the conversation; history and sessions are kept
    r2 = client.post("/api/agent/run", json={"backend": "claude", "prompt": "and now?"}, headers=H).json()
    info2 = json.loads(_events(client, r2["run_id"])[-1][1]["final_text"])
    assert info2["resume"] == "sess-123"
    assert [e["prompt"] for e in client.get("/api/agent/history?backend=claude").json()] == ["count the blobs", "and now?"]
    assert client.get("/api/agent/sessions").json()["claude"] == "sess-123"
    client.post("/api/agent/clear", json={"backend": "claude"}, headers=H)
    assert client.get("/api/agent/sessions").json()["claude"] is None


def test_settings_roundtrip_and_validation(client):
    s = client.post("/api/agent/settings", json={"backend": "claude", "model": "claude-opus-5-5", "effort": "high"}, headers=H).json()
    assert s["claude"] == {"model": "claude-opus-5-5", "effort": "high"}
    assert client.post("/api/agent/settings", json={"backend": "claude", "effort": "ludicrous"}, headers=H).status_code == 400


def test_panel_page_is_served(client):
    r = client.get("/agent/index.html")
    assert r.status_code == 200 and "Agent Chat for Lumen" in r.text
    assert "lumen-agent-close" in client.get("/agent/app.js").text
