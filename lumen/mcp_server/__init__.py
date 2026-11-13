"""The Lumen MCP server: lets an AI assistant (Claude Code, Claude Desktop, Codex, …) drive the running Lumen app.

``lumen-mcp`` / ``python -m lumen.mcp_server`` (stdio) or ``Lumen --mcp`` from the desktop app; see docs/MCP.md.
Needs the optional ``mcp`` package (``pip install "lumen-imaging[mcp]"``)."""


def main(argv: list[str] | None = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--config", "config"):
        from .config import print_config
        return print_config(args[1:])
    try:
        from .server import main as run
    except ImportError as e:  # the optional dependency is missing
        print(f"lumen-mcp needs the MCP SDK: pip install \"lumen-imaging[mcp]\" ({e})", file=sys.stderr)
        return 2
    return run(args)
