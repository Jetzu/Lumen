#!/usr/bin/env python3
"""Smoke test of the MCP mode of a built desktop app: start ``<exe> --mcp`` (or any MCP server command), run the
MCP handshake over stdio (initialize → initialized → tools/list → tools/call status) and check the tool list.

    python desktop/build/mcp_smoke.py dist/Lumen.app/Contents/MacOS/Lumen --mcp
    python desktop/build/mcp_smoke.py python -m lumen.mcp_server

Standard library only (runs with any Python on the CI runner).  Prints ``LUMEN_MCP_SMOKE_OK tools=N`` and exits 0, or
prints the problem and exits 1.  Lumen itself need not be running: ``status`` then answers "Not connected"."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

EXPECTED = {"status", "launch", "find_commands", "describe_command", "run_command", "run_macro", "run_script", "view_image",
            "get_statistics", "set_selection", "roi_manager", "get_results", "events", "ui"}


def main(cmd: list[str]) -> int:
    if not cmd:
        print(__doc__)
        return 2
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines: list[bytes] = []
    err: list[bytes] = []

    def pump(src, dst):
        for line in iter(src.readline, b""):
            dst.append(line)
    threading.Thread(target=pump, args=(proc.stdout, lines), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, err), daemon=True).start()

    def send(msg: dict) -> None:
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        proc.stdin.flush()

    def reply(rid: int, timeout: float = 120.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            for raw in list(lines):
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if msg.get("id") == rid:
                    return msg
            if proc.poll() is not None:
                raise RuntimeError(f"the server exited with code {proc.returncode}")
            time.sleep(0.05)
        raise TimeoutError(f"no answer to request {rid} within {timeout:.0f} s")

    try:
        t0 = time.time()
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                                             "clientInfo": {"name": "lumen-mcp-smoke", "version": "1"}}})
        init = reply(1)
        name = init["result"]["serverInfo"]["name"]
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = {t["name"] for t in reply(2)["result"]["tools"]}
        missing = EXPECTED - tools
        if name != "lumen" or missing:
            print(f"LUMEN_MCP_SMOKE_FAILED server={name} missing={sorted(missing)}")
            return 1
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "status", "arguments": {}}})
        st = reply(3)
        text = st["result"]["content"][0]["text"] if "result" in st else json.dumps(st)
        print(f"status: {text.splitlines()[0][:200]}")
        print(f"LUMEN_MCP_SMOKE_OK tools={len(tools)} handshake_ms={int((time.time() - t0) * 1000)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"LUMEN_MCP_SMOKE_FAILED {type(e).__name__}: {e}")
        print("--- stderr ---\n" + b"".join(err).decode("utf-8", "replace")[-4000:])
        return 1
    finally:
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
