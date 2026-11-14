"""Lumen's Agent panel: run the user's Claude Code / Codex CLI against this Lumen (through the ``lumen`` MCP server).

``agentcli`` starts and parses the CLI (ported from the ChemDraw MCP's Agent Chat bridge, itself adapted from
mol-mcp); ``lumen/server/extra/agent_chat.py`` serves it to the window's Agent tab (``frontend/src/components/
AgentPanel.tsx``)."""
