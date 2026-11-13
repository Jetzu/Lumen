# Connecting an AI assistant (MCP)

Lumen ships an [MCP](https://modelcontextprotocol.io) server, so an AI assistant such as Claude Code, Claude Desktop or Codex can work in Lumen with you. It can open images, run any of Lumen's 500+ commands, ImageJ macros and scripts, measure, manage ROIs and read results.

The assistant works on the **same running Lumen you have open**. Every change it makes appears in your window straight away: new images, the Results table, plots and selections. It can also see what you are doing: the active image, the selection you drew, the tool you picked and the dialog you have open.

## Setup

There are two ways to run the server.

### From the desktop app (Lumen.app / Lumen.exe / the Linux build)

The app's own executable has an MCP mode, so nothing else needs to be installed:

| OS | Command |
|---|---|
| macOS | `/Applications/Lumen.app/Contents/MacOS/Lumen --mcp` |
| Windows | `C:\path\to\Lumen\Lumen.exe --mcp` |
| Linux | `/path/to/Lumen/Lumen --mcp` |

`Lumen --mcp-config` prints ready-made configuration for the assistants below.

### From a Python install

```bash
pip install "lumen-imaging[mcp]"
lumen-mcp            # or: python -m lumen.mcp_server
lumen-mcp --config   # prints the configuration snippets for this machine
```

### Registering it with an assistant

The examples use the Python command. With the desktop app, use the app's command from the table above instead.

**Claude Code**

```bash
claude mcp add --scope user lumen -- lumen-mcp
```

**Claude Desktop**: Settings ▸ Developer ▸ Edit Config (`claude_desktop_config.json`):

```json
{ "mcpServers": { "lumen": { "command": "lumen-mcp", "args": [] } } }
```

**Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.lumen]
command = "lumen-mcp"
args = []
```

Any other MCP client runs the same command over stdio.

## How it finds Lumen

The server looks for a running Lumen in this order:

1. The `LUMEN_URL` environment variable (for example `http://127.0.0.1:8765`).
2. The desktop app's lock file (`~/.lumen/desktop/desktop.lock`, or `$LUMEN_HOME/desktop/desktop.lock`).
3. The usual local ports: 8770 for the desktop app, 8765 for `lumen` in a browser, and the next few.

If Lumen is not running, the assistant can start it with the `launch` tool:

- `desktop`: the desktop app, when it is installed.
- `browser`: `lumen` with a browser window.
- `headless`: the engine alone, with no window.

The server only ever talks to `127.0.0.1`.

## Tools

| Tool | What it does |
|---|---|
| `status`, `launch` | Connection state, open images, what the window shows, running jobs; start Lumen |
| `find_commands`, `describe_command` | Search the command registry by words, menu or ImageJ name; parameters, defaults and ranges |
| `run_command` | Run any command (by id, name or ImageJ name). Uses your current selection when the command works on one |
| `run_macro` | ImageJ 1.x macro code (`run("Gaussian Blur...", "sigma=2"); …`) |
| `run_script` | Python / Jython (the `lumen` API, numpy, ImageJ's `ij.*`); Groovy / BeanShell / JavaScript / JRuby / Clojure through the Java bridge |
| `list_images`, `image_info`, `open_image`, `open_sample`, `save_image`, `close_image`, `activate_image` | Images |
| `view_image` | The image as you see it (LUTs, contrast, composite), with your selection (yellow) and the image's ROIs (cyan) drawn on top |
| `get_statistics`, `get_pixels` | Numbers inside a selection, a ROI or the whole plane (does not touch the Results table); raw pixel values |
| `set_display` | Contrast (or auto), LUT, channel visibility, gamma, composite mode, shown plane |
| `get_selection`, `set_selection` | The selection drawn in the window, read or set |
| `roi_manager` | List, add, delete, clear, measure, combine, rename |
| `get_results`, `clear_results` | The Results table and named tables |
| `history`, `undo`, `redo` | Undo steps and the session log |
| `recorded_macro` | Everything done so far, in the window and by the assistant, as a replayable Python script or ImageJ macro |
| `events` | What changed since the assistant last looked, including what you did in the window |
| `ui` | Show a panel, set the tool or plane, show a notice, or open a command's dialog pre-filled for you to review and apply; theme, zoom |
| `jobs`, `settings`, `help` | Progress and cancelling, preferences, Lumen's bundled documentation |

Commands the assistant runs go into **your** macro recorder and undo history. Undo (⌘Z / Ctrl+Z) takes back the assistant's steps like your own, and the Script & Recorder panel shows them as lines you can replay.

## How it works

The Lumen window is a client of Lumen's local engine (FastAPI on 127.0.0.1). It follows the engine through the `/api/events` WebSocket.

The MCP server is a second client of the same API. Two endpoints, in `lumen/server/extra/agent.py`, carry what exists only in the window:

- `POST /api/ui/command`: the engine broadcasts a `ui` event, and the window applies it (`frontend/src/agentBridge.ts`).
- `POST /api/ui/state`: the window reports what it shows, and the assistant reads it with `GET /api/ui/state`.

The window applies only the actions it knows. It refuses an unknown one and reports the error back.
