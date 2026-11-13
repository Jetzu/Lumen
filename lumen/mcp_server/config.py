"""``lumen-mcp --config``: how to register the Lumen MCP server with an AI assistant (printed, nothing is written)."""
from __future__ import annotations

import json
import shlex
import sys


def server_command() -> list[str]:
    """The command an MCP client runs: the desktop app's own executable with ``--mcp`` in a frozen build, else this
    Python with ``-m lumen.mcp_server``."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--mcp"]
    return [sys.executable, "-m", "lumen.mcp_server"]


def print_config(argv: list[str] | None = None) -> int:
    cmd = server_command()
    quote = (lambda a: f'"{a}"' if " " in a else a) if sys.platform == "win32" else shlex.quote
    line = " ".join(quote(a) for a in cmd)
    desktop = {"mcpServers": {"lumen": {"command": cmd[0], "args": cmd[1:]}}}
    toml_args = ", ".join(json.dumps(a) for a in cmd[1:])
    print(f"""Lumen MCP server – register it with your assistant:

Claude Code:
  claude mcp add --scope user lumen -- {line}

Claude Desktop (Settings ▸ Developer ▸ Edit Config, claude_desktop_config.json):
{json.dumps(desktop, indent=2)}

Codex (~/.codex/config.toml):
[mcp_servers.lumen]
command = {json.dumps(cmd[0])}
args = [{toml_args}]

Any other MCP client: run `{line}` over stdio.
The server connects to the running Lumen (desktop app or `lumen` in a browser) and can start it (tool `launch`).""")
    return 0
