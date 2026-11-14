"""lumen-mcp installer: register the ``lumen`` MCP server with Claude Code, Codex and Claude Desktop.

    lumen-mcp --install                    (or python -m lumen.mcp_server.install, or the desktop app's Lumen --mcp-install)
    lumen-mcp --install --check            only report what was found
    lumen-mcp --install --dry-run          show what would be written
    lumen-mcp --install --uninstall        remove the registrations

Steps (idempotent; choose with flags, default = everything that applies):
  --claude           Claude Code (user scope, ~/.claude.json, or $CLAUDE_CONFIG_DIR/.claude.json)
  --codex            Codex CLI (~/.codex/config.toml, or $CODEX_HOME/config.toml)
  --claude-desktop   Claude Desktop (when its config file exists)
  --target app|python
                     what the clients start: ``app`` = the installed desktop app with --mcp (no Python needed),
                     ``python`` = this Python with -m lumen.mcp_server; default: the app when it is installed
                     (or when this *is* the app), else this Python.

Every file is backed up once (``<file>.lumen-backup``) before it is first changed, written atomically, and left alone
when it does not parse.  An existing ``lumen`` entry that is not Lumen's is kept: Claude Desktop then gets
``lumen-mcp``; Claude Code / Codex replace it only with the original kept in the backup.  Standard library only;
works on macOS, Linux and Windows.  (Modelled on the ChemDraw MCP's installer.)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SERVER_NAME = "lumen"
ALT_SERVER_NAME = "lumen-mcp"
BACKUP_SUFFIX = ".lumen-backup"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------- what the clients run
def desktop_app_executable() -> Optional[str]:
    """The installed desktop app's executable (None when not installed)."""
    if getattr(sys, "frozen", False):
        return sys.executable
    if sys.platform == "darwin":
        for app in (Path("/Applications/Lumen.app"), Path.home() / "Applications" / "Lumen.app"):
            exe = app / "Contents" / "MacOS" / "Lumen"
            if exe.is_file():
                return str(exe)
    elif sys.platform.startswith("win"):
        for base in filter(None, (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles"))):
            exe = Path(base) / "Lumen" / "Lumen.exe"
            if exe.is_file():
                return str(exe)
    else:
        for exe in (Path.home() / ".local" / "share" / "Lumen" / "Lumen", Path("/opt/Lumen/Lumen")):
            if exe.is_file():
                return str(exe)
    return None


def python_entry() -> Tuple[str, List[str]]:
    script = Path(sys.executable).parent / ("lumen-mcp.exe" if sys.platform.startswith("win") else "lumen-mcp")
    if script.is_file():
        return str(script), []
    return sys.executable, ["-m", "lumen.mcp_server"]


def server_entry(target: str = "auto") -> Tuple[str, List[str]]:
    app = desktop_app_executable()
    if target == "app" or (target == "auto" and app):
        if not app:
            raise SystemExit("the Lumen desktop app is not installed here; use --target python")
        return app, ["--mcp"]
    return python_entry()


def _is_ours(entry: dict) -> bool:
    cmd = ((entry or {}).get("command", "") or "").replace("\\", "/")
    base = cmd.rsplit("/", 1)[-1]
    args = [str(a) for a in ((entry or {}).get("args", []) or [])]
    return (base in ("lumen-mcp", "lumen-mcp.exe")
            or (base in ("Lumen", "Lumen.exe") and "--mcp" in args)
            or any("lumen.mcp_server" in a for a in args))


# ---------------------------------------------------------------------------- file helpers
def _backup(path: str, force: bool = False) -> None:
    """Keep the config as it was before Lumen first touched it (``force``: about to replace a foreign entry)."""
    if os.path.exists(path) and (force or not os.path.exists(path + BACKUP_SUFFIX)):
        shutil.copy(path, path + BACKUP_SUFFIX)


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _save(path: str, original: Optional[str], text: str, force_backup: bool = False) -> bool:
    if original is not None and text == original:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _backup(path, force=force_backup)
    _write_text(path, text)
    return True


def _json_text(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------- Claude Code
def claude_config_path() -> str:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return os.path.join(base, ".claude.json")


def register_claude(entry: Tuple[str, List[str]], dry: bool, remove: bool = False) -> None:
    path = claude_config_path()
    log(f"\n== Claude Code (user scope, {path}) ==")
    original = _read_text(path)
    data: dict = {}
    if original is not None:
        try:
            data = json.loads(original)
        except Exception as e:  # noqa: BLE001
            log(f"cannot parse {path} ({e}); not touching it")
            return
    elif remove:
        log("no config file; nothing to remove")
        return
    servers = data.setdefault("mcpServers", {})
    replacing = False
    if remove:
        if SERVER_NAME in servers and _is_ours(servers[SERVER_NAME]):
            servers.pop(SERVER_NAME)
            log(f"removed {SERVER_NAME}")
        else:
            log(f"{SERVER_NAME} is not registered (or not Lumen's); nothing to remove")
            return
    else:
        if SERVER_NAME in servers and not _is_ours(servers[SERVER_NAME]):
            replacing = True
            log(f"replacing an existing non-Lumen '{SERVER_NAME}' entry (kept in {path}{BACKUP_SUFFIX})")
        cmd, args = entry
        servers[SERVER_NAME] = {"type": "stdio", "command": cmd, "args": args, "env": {}}
        log(f"{SERVER_NAME} -> {cmd} {' '.join(args)}")
    if dry:
        return
    if _save(path, original, _json_text(data), force_backup=replacing):
        log("saved. In Claude Code run /mcp to see the server (restart running sessions).")
    else:
        log("already up to date; nothing written.")


# ---------------------------------------------------------------------------- Codex
def codex_config_path() -> str:
    base = os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    return os.path.join(base, "config.toml")


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007F")


def _toml_key_pattern(name: str) -> str:
    n = re.escape(name)
    return "(?:%s|\"%s\"|'%s')" % (n, n, n)


def _codex_block_pattern(name: str) -> "re.Pattern[str]":
    return re.compile(r"(?ms)^\[mcp_servers\.%s(?:\.[^\]\n]*)?\][^\n]*(?:\n|\Z).*?(?=^\[|\Z)" % _toml_key_pattern(name))


def _set_codex_block(text: str, name: str, block: Optional[str]) -> str:
    pat = _codex_block_pattern(name)
    m = pat.search(text)
    if not m:
        if block is None:
            return text
        if text and not text.endswith("\n"):
            text += "\n"
        if text and not text.endswith("\n\n"):
            text += "\n"
        return text + block
    start = m.start()
    text = pat.sub("", text)
    if block is not None:
        text = text[:start] + block + text[start:]
    return text


def _toml_module():
    for mod in ("tomllib", "tomli"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    return None


def _validate_toml(text: str) -> Optional[str]:
    toml = _toml_module()
    if toml is None:
        return None
    try:
        toml.loads(text)
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def _codex_entry(text: str, name: str):
    toml = _toml_module()
    if toml is not None:
        try:
            servers = toml.loads(text).get("mcp_servers")
        except Exception:  # noqa: BLE001
            return None
        return servers.get(name) if isinstance(servers, dict) else None
    key = _toml_key_pattern(name)
    if re.search(r"(?m)^\s*mcp_servers\s*\.\s*%s\s*[.=]" % key, text):
        return {}
    table = re.search(r"(?ms)^\s*\[\s*mcp_servers\s*\][^\n]*\n(.*?)(?=^\s*\[|\Z)", text)
    if table and re.search(r"(?m)^\s*%s\s*[.=]" % key, table.group(1)):
        return {}
    return {} if _codex_block_pattern(name).search(text) else None


def register_codex(entry: Tuple[str, List[str]], dry: bool, remove: bool = False) -> None:
    path = codex_config_path()
    log(f"\n== Codex CLI ({path}) ==")
    original = _read_text(path)
    if original is None and remove:
        log("no config file; nothing to remove")
        return
    text = original or ""
    err = _validate_toml(text)
    if err:
        log(f"cannot parse {path} ({err}); not touching it")
        return
    has_block = _codex_block_pattern(SERVER_NAME).search(text) is not None
    replacing = False
    if remove:
        if not has_block:
            log(f"{SERVER_NAME} is not registered; nothing to remove")
            return
        text = _set_codex_block(text, SERVER_NAME, None)
        log(f"removed {SERVER_NAME}")
    else:
        existing = _codex_entry(text, SERVER_NAME)
        if existing is not None and not has_block:
            log(f"'{SERVER_NAME}' is already defined in {path} outside a [mcp_servers.{SERVER_NAME}] table; "
                "remove that definition and run the installer again")
            return
        if isinstance(existing, dict) and existing and not _is_ours(existing):
            replacing = True
            log(f"replacing an existing non-Lumen '{SERVER_NAME}' table (kept in {path}{BACKUP_SUFFIX})")
        cmd, args = entry
        block = "[mcp_servers.%s]\ncommand = %s\nargs = [%s]\n\n" % (SERVER_NAME, _toml_str(cmd), ", ".join(_toml_str(a) for a in args))
        text = _set_codex_block(text, SERVER_NAME, block)
        log(f"{SERVER_NAME} -> {cmd} {' '.join(args)}")
    text = text.rstrip("\n") + "\n" if text.strip() else ""
    err = _validate_toml(text)
    if err:
        log(f"ERROR: the edited config would not parse as TOML ({err}); {path} left unchanged")
        return
    if dry:
        return
    if _save(path, original, text, force_backup=replacing):
        log("saved. In Codex run `codex mcp list` to verify.")
    else:
        log("already up to date; nothing written.")


# ---------------------------------------------------------------------------- Claude Desktop
def claude_desktop_config_path() -> Optional[str]:
    if sys.platform == "darwin":
        p = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    elif sys.platform.startswith("win"):
        p = os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude_desktop_config.json")
    else:
        p = os.path.expanduser("~/.config/Claude/claude_desktop_config.json")
    return p if os.path.exists(p) else None


def register_claude_desktop(entry: Tuple[str, List[str]], dry: bool, remove: bool = False) -> None:
    path = claude_desktop_config_path()
    if not path:
        return
    log(f"\n== Claude Desktop ({path}) ==")
    original = _read_text(path) or ""
    try:
        data = json.loads(original or "{}")
    except Exception as e:  # noqa: BLE001
        log(f"cannot parse config ({e}); skipping")
        return
    servers = data.setdefault("mcpServers", {})
    if remove:
        removed = False
        for n in (SERVER_NAME, ALT_SERVER_NAME):
            if n in servers and _is_ours(servers[n]):
                servers.pop(n)
                log(f"removed {n}")
                removed = True
        if not removed:
            log("nothing of Lumen's registered; nothing to remove")
            return
    else:
        name = SERVER_NAME
        if name in servers and not _is_ours(servers[name]):
            name = ALT_SERVER_NAME
            log(f"'{SERVER_NAME}' is already used by another server; registering as '{name}'")
        cmd, args = entry
        servers[name] = {"command": cmd, "args": args}
        log(f"{name} -> {cmd} {' '.join(args)}")
    if dry:
        return
    if _save(path, original, _json_text(data)):
        log("saved (restart Claude Desktop).")
    else:
        log("already up to date; nothing written.")


# ----------------------------------------------------------------------------
def check(target: str) -> None:
    from lumen import __version__
    log(f"== Lumen {__version__} MCP server ==")
    log(f"python:    {sys.executable}")
    log(f"app:       {desktop_app_executable() or 'not installed'}")
    try:
        cmd, args = server_entry(target)
        log(f"server:    {cmd} {' '.join(args)}")
    except SystemExit as e:
        log(f"server:    {e}")
    log(f"configs:   Claude Code    {claude_config_path()}")
    log(f"           Codex CLI      {codex_config_path()}")
    log(f"           Claude Desktop {claude_desktop_config_path() or '(no config file found)'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="lumen-mcp --install", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--codex", action="store_true")
    ap.add_argument("--claude-desktop", action="store_true")
    ap.add_argument("--target", choices=["auto", "app", "python"], default="auto")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="only report what was found")
    a = ap.parse_args(argv)
    check(a.target)
    if a.check:
        return 0
    chosen = any([a.claude, a.codex, a.claude_desktop])
    do = lambda flag: flag or not chosen  # noqa: E731
    dry = a.dry_run
    if dry:
        log("\n(dry run: nothing will be written)")
    entry = server_entry(a.target)
    if do(a.claude):
        register_claude(entry, dry, remove=a.uninstall)
    if do(a.codex):
        register_codex(entry, dry, remove=a.uninstall)
    if do(a.claude_desktop):
        register_claude_desktop(entry, dry, remove=a.uninstall)
    if a.uninstall:
        log(f"\nDone. The lumen server registrations were removed{' (dry run)' if dry else ''}.")
        return 0
    log(f"\nDone{' (dry run)' if dry else ''}. Verify with:")
    if do(a.claude):
        log("  Claude Code:    /mcp  (restart running sessions)")
    if do(a.codex):
        log("  Codex CLI:      codex mcp list")
    if do(a.claude_desktop) and claude_desktop_config_path():
        log("  Claude Desktop: restart the app, then check Settings > Developer")
    log("Then ask the assistant for the lumen `status` tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
