# Connecting an AI assistant (MCP)

Lumen ships an [MCP](https://modelcontextprotocol.io) server, so an AI assistant such as Claude Code, Claude Desktop or Codex can work in Lumen with you. It can open images, run any of Lumen's 500+ commands, ImageJ macros and scripts, measure, manage ROIs and read results.

The assistant works on the **same running Lumen you have open**. Every change it makes appears in your window straight away: new images, the Results table, plots and selections. It can also see what you are doing: the active image, the selection you drew, the tool you picked and the dialog you have open.

## Agent Chat inside Lumen

Click **✻ Agent** in the top bar (or press ⌘J / Ctrl+J; also View ▸ Agent Chat) to open **Agent Chat for Lumen** on the right of the window. It is the Agent Chat panel of the ChemDraw MCP, with the same terminal look, in dark and light mode (it follows Lumen's theme):

- Type a request, for example "count the nuclei in this image" or "measure the mean intensity in my selection".
- Your own Claude Code (or Codex) CLI answers, working in this Lumen through the `lumen` MCP tools. You watch every step happen in the window.
- The active image, your selection, and the Results and ROI counts go along with each prompt.
- Enter sends, Shift+Enter adds a new line, Esc interrupts, and ↑/↓ recall earlier prompts.
- `/model` (or the button at the bottom right) switches the provider, model and effort. The menus are read from the CLIs' own model caches, so new models appear without a Lumen update; Claude models are offered by alias (the newest model of each family), and a saved model that was superseded is switched to its current equivalent on the next run. `/clear` starts a new conversation and `/help` lists the shortcuts.
- Conversations continue across turns and are kept when you reopen the panel.
- An ImageJ macro or Python block in an answer has a **▶ run in Lumen** button.
- Drag the panel's left edge to resize it (double-click resets). It opens narrow (380 px) so the image keeps most of the window. The inspector's edge resizes the same way.

There is no separate bridge to start: Lumen serves the panel's bridge itself (`/api/agent`, in `lumen/server/extra/agent_chat.py`). Only Lumen's own page on this machine can use it. Claude may use only the `lumen` MCP tools from the panel, with no shell and no file edits.

You need Claude Code or Codex installed, and Lumen registered with it once (`Lumen --mcp-install`, see below).

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

### Registering it with an assistant (one command)

The installer registers Lumen with Claude Code, Codex and Claude Desktop in one step. It backs up each file before changing it, never overwrites another server's entry, and does nothing if run twice:

```bash
/Applications/Lumen.app/Contents/MacOS/Lumen --mcp-install   # desktop app (Windows: Lumen.exe --mcp-install)
lumen-mcp --install                                          # Python install
```

Add `--check` to see what it finds, `--dry-run` to see what it would write, or `--uninstall` to remove the entries. Then restart the assistant.

### Registering it by hand

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
| `preview_command` | What a command would do to the shown plane, without applying it |
| `view_orthogonal`, `export_mesh`, `scene` | XZ / YZ views of a stack, a surface mesh as STL / OBJ / PLY, the 3-D viewer's scene |
| `workflows` | The guided workflows (Count nuclei, Track cells …), their steps, or the whole workflow as a script |
| `get_tracks` | The tracking model of a time series |
| `list_files`, `text_file` | Browse folders for images; read or write macros, CSVs and notes |
| `export_results`, `roi_file` | The Results table as CSV; ROI Manager import and export (ImageJ RoiSet `.zip` / `.roi`, or JSON) |
| `list_luts`, `clear_threshold`, `restore_cleared` | LUT names, removing a threshold overlay, bringing back a cleared ROI Manager or Results table |

**Coverage.** `run_command` reaches every one of Lumen's 503 commands, plus installed Fiji plugins and your own plugins. Every other HTTP endpoint is either reachable through a tool or is part of the window's own data streams and uploads (`tests/test_mcp_server.py::test_every_endpoint_is_covered_by_the_mcp` fails when a new endpoint has no tool).

Commands the assistant runs go into **your** macro recorder and undo history. Undo (⌘Z / Ctrl+Z) takes back the assistant's steps like your own, and the Script & Recorder panel shows them as lines you can replay.

## How it works

The Lumen window is a client of Lumen's local engine (FastAPI on 127.0.0.1). It follows the engine through the `/api/events` WebSocket.

The MCP server is a second client of the same API. Two endpoints, in `lumen/server/extra/agent.py`, carry what exists only in the window:

- `POST /api/ui/command`: the engine broadcasts a `ui` event, and the window applies it (`frontend/src/agentBridge.ts`).
- `POST /api/ui/state`: the window reports what it shows, and the assistant reads it with `GET /api/ui/state`.

The window applies only the actions it knows. It refuses an unknown one and reports the error back.
