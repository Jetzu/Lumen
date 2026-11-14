"""Drive Claude Code or Codex CLI non-interactively for Lumen's Agent panel (standard library only).

The CLI is the user's own installed Claude Code / Codex.  It loads the ``lumen`` MCP server from the user's MCP
config (registered by ``lumen-mcp --install`` / ``Lumen --mcp-install``) and acts on the running Lumen through its
tools; ``LUMEN_URL`` in the CLI's environment points that MCP server at exactly this Lumen.  Ported from the ChemDraw
MCP's ``bridge/agentcli.py`` + ``bridge/settings.py``.

Events delivered to the callback as (kind, payload):
    ("session", id)        conversation id usable for follow-up turns
    ("delta", str)         streamed partial assistant text (Claude only)
    ("text", str)          complete assistant message
    ("tool", str)          tool call: "<server>/<tool> <json args>"
    ("tool_result", str)   tool result text (truncated)
    ("stats", dict)        cost / usage at the end of a turn
    ("done", str)          final answer
    ("error", str)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

APP = "lumen"
BACKENDS = ("claude", "codex")
BACKEND_LABELS = {"claude": "Claude Code", "codex": "Codex"}
MODEL_CHOICES = {"claude": ["", "fable", "opus", "sonnet", "haiku"], "codex": ["", "gpt-5-codex", "gpt-5", "o3"]}
EFFORT_CHOICES = {"claude": ["", "low", "medium", "high", "xhigh", "max"], "codex": ["", "minimal", "low", "medium", "high", "xhigh"]}
HISTORY_LIMIT = 200

SYSTEM_CONTEXT = (
    "You are embedded in the Agent panel of Lumen, a modern ImageJ/Fiji (scientific image analysis). The user is "
    "looking at the Lumen window right now and reads your answers in a side panel next to the image. Use the 'lumen' "
    "MCP server tools to inspect and change what they see: status, list_images, view_image, get_statistics, "
    "find_commands, describe_command, run_command, run_macro, run_script, set_selection, roi_manager, get_results, "
    "set_display, ui, events, recorded_macro. Everything you do appears live in their window. Act on the active image "
    "and the user's current selection when they say 'this'. Prefer numbers (get_statistics, get_results) over "
    "guessing from pictures. Ask before closing modified images or overwriting files. Keep answers concise: say what "
    "you did and what you found, a few sentences, no long preambles."
)


# ---------------------------------------------------------------------------- paths + settings
def home_dir() -> str:
    custom = os.environ.get("LUMEN_HOME", "").strip()
    d = os.path.join(os.path.expanduser(custom) if custom else os.path.join(os.path.expanduser("~"), ".lumen"), "agent")
    os.makedirs(d, exist_ok=True)
    return d


def workspace_dir() -> str:
    """The CLIs' working directory (Claude Code keys resumable sessions by it, so every turn uses the same one)."""
    d = os.environ.get("LUMEN_AGENT_CWD") or os.path.join(home_dir(), "workspace")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.expanduser("~")
    return d


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _settings_path() -> str:
    return os.path.join(home_dir(), "settings.json")


def load_settings() -> dict:
    d = _read_json(_settings_path())
    return d if isinstance(d, dict) else {}


def backend_settings(backend: str) -> dict:
    d = load_settings().get(backend, {})
    d = d if isinstance(d, dict) else {}
    return {"model": str(d.get("model", "") or ""), "effort": str(d.get("effort", "") or "")}


def update_settings(backend: Optional[str] = None, model: Optional[str] = None, effort: Optional[str] = None,
                    default_backend: Optional[str] = None) -> dict:
    d = load_settings()
    if backend in BACKENDS:
        cur = d.get(backend) if isinstance(d.get(backend), dict) else {}
        if model is not None:
            cur["model"] = str(model)[:80]
        if effort is not None:
            cur["effort"] = str(effort)[:20]
        d[backend] = cur
    if default_backend in BACKENDS:
        d["default_backend"] = default_backend
    _write_json(_settings_path(), d)
    return settings_snapshot()


def settings_snapshot() -> dict:
    d = load_settings()
    return {"claude": backend_settings("claude"), "codex": backend_settings("codex"),
            "default_backend": d.get("default_backend") if d.get("default_backend") in BACKENDS else None,
            "model_choices": MODEL_CHOICES, "effort_choices": EFFORT_CHOICES}


class SessionStore:
    """Conversation id per backend and the turn history (``<lumen home>/agent/sessions.json``), thread-safe."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(home_dir(), "sessions.json")
        self._lock = threading.RLock()
        d = _read_json(self.path)
        d = d if isinstance(d, dict) else {}
        s = d.get("sessions") if isinstance(d.get("sessions"), dict) else {}
        self._sessions: Dict[str, Optional[str]] = {b: (str(s[b]) if s.get(b) else None) for b in BACKENDS}
        self._history: List[dict] = [e for e in d.get("history") or [] if isinstance(e, dict)][-HISTORY_LIMIT:]

    def _save(self) -> None:
        _write_json(self.path, {"sessions": self._sessions, "history": self._history, "saved": time.time()})

    def get(self, backend: str) -> Optional[str]:
        with self._lock:
            return self._sessions.get(backend)

    def set(self, backend: str, sid: Optional[str]) -> None:
        with self._lock:
            self._sessions[backend] = sid or None
            self._save()

    def clear(self, backend: Optional[str] = None) -> None:
        with self._lock:
            for b in ([backend] if backend in BACKENDS else BACKENDS):
                self._sessions[b] = None
                self._history = [e for e in self._history if e.get("backend") != b]
            self._save()

    def add(self, entry: dict) -> None:
        with self._lock:
            self._history.append(dict(entry))
            del self._history[:-HISTORY_LIMIT]
            self._save()

    def history(self, backend: Optional[str] = None) -> List[dict]:
        with self._lock:
            return [dict(e) for e in self._history if not backend or e.get("backend") == backend]


# ---------------------------------------------------------------------------- finding the CLIs
def _win() -> bool:
    return sys.platform.startswith("win")


_STRIP_ENV = {"PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONNOUSERSITE", "PYTHONSTARTUP", "PYTHONSAFEPATH", "__PYVENV_LAUNCHER__",
              "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "QT_PLUGIN_PATH",
              "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "VIRTUAL_ENV",
              "_MEIPASS2", "_PYI_APPLICATION_HOME_DIR", "_PYI_ARCHIVE_FILE", "_PYI_PARENT_PROCESS_LEVEL"}
_STANDARD_PATH = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]


def clean_env() -> dict:
    """The environment without the app's private interpreter settings (a frozen desktop app sets some), and with the
    usual user directories on PATH (apps started from the Dock / Finder / Start menu see a minimal PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
    if not _win():
        home = os.path.expanduser("~")
        parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        for extra in [os.path.join(home, ".local", "bin"), os.path.join(home, ".npm-global", "bin"), os.path.join(home, ".claude", "local")] + _STANDARD_PATH:
            if extra not in parts:
                parts.append(extra)
        env["PATH"] = os.pathsep.join(parts)
    return env


_validated: Dict[str, bool] = {}


def _works(exe: str) -> bool:
    if exe in _validated:
        return _validated[exe]
    ok = False
    try:
        kw: dict = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=20, env=clean_env())
        if _win():
            kw["creationflags"] = 0x08000000
        ok = subprocess.run([exe, "--version"], **kw).returncode == 0
    except Exception:  # noqa: BLE001
        ok = False
    _validated[exe] = ok
    return ok


def _first(patterns: List[str]) -> Optional[str]:
    for p in patterns:
        for hit in sorted(glob.glob(os.path.expanduser(os.path.expandvars(p))), reverse=True):
            if os.path.isfile(hit) and os.access(hit, os.X_OK) and _works(hit):
                return hit
    return None


def _which(names: List[str]) -> Optional[str]:
    path = clean_env().get("PATH")
    for n in names:
        p = shutil.which(n, path=path)
        if p and _works(p):
            return p
    return None


def find_claude() -> Optional[str]:
    v = os.environ.get("LUMEN_CLAUDE_EXE") or os.environ.get("CLAUDE_CLI")
    if v and os.path.exists(v):
        return v
    hit = _which(["claude", "claude.exe", "claude.cmd"])
    if hit:
        return hit
    home = os.path.expanduser("~")
    pats = [os.path.join(home, ".claude", "local", "claude"), os.path.join(home, ".claude", "local", "node_modules", ".bin", "claude"),
            os.path.join(home, ".vscode*", "extensions", "anthropic.claude-code-*", "resources", "native-binary", "claude*"),
            os.path.join(home, ".cursor", "extensions", "anthropic.claude-code-*", "resources", "native-binary", "claude*"),
            os.path.join(home, ".npm-global", "bin", "claude"), "/usr/local/bin/claude", "/opt/homebrew/bin/claude"]
    if _win():
        pats += [os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "claude*", "claude.exe")]
    return _first(pats)


def find_codex() -> Optional[str]:
    v = os.environ.get("LUMEN_CODEX_EXE") or os.environ.get("CODEX_CLI")
    if v and os.path.exists(v):
        return v
    hit = _which(["codex", "codex.exe", "codex.cmd"])
    if hit:
        return hit
    home = os.path.expanduser("~")
    pats = []
    if sys.platform == "darwin":
        pats += ["/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex",
                 os.path.join(home, "Applications", "ChatGPT.app", "Contents", "Resources", "codex-cli", "bin", "codex")]
    pats += [os.path.join(home, ".npm-global", "bin", "codex"), "/usr/local/bin/codex", "/opt/homebrew/bin/codex"]
    if _win():
        pats += [os.path.join(os.environ.get("APPDATA", ""), "npm", "codex.cmd")]
    return _first(pats)


def find_backend(backend: str) -> Optional[str]:
    return find_claude() if backend == "claude" else find_codex() if backend == "codex" else None


def available_backends() -> List[str]:
    return [b for b in BACKENDS if find_backend(b)]


# ---------------------------------------------------------------------------- command line + parsing
def system_context(extra_context: Optional[str] = None) -> str:
    ctx = SYSTEM_CONTEXT
    if extra_context and str(extra_context).strip():
        ctx += "\n\nWhat the user's Lumen window shows right now (supplied by the panel):\n" + str(extra_context).strip()
    return ctx


def build_argv(backend: str, exe: str, prompt: str, session_id: Optional[str] = None, model: str = "", effort: str = "",
               extra_context: Optional[str] = None) -> List[str]:
    context = system_context(extra_context)
    if backend == "claude":
        # only Lumen's MCP tools: no shell, no file edits from the panel
        argv = [exe, "-p", prompt, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--allowedTools", f"mcp__{APP}", "--append-system-prompt", context]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        if session_id:
            argv += ["--resume", session_id]
        return argv
    if backend == "codex":
        if session_id:
            full = prompt if not (extra_context and str(extra_context).strip()) else (
                f"What the user's Lumen window shows right now:\n{str(extra_context).strip()}\n\nUser request: {prompt}")
        else:
            full = f"{context}\n\nUser request: {prompt}"
        opts = ["--json", "--skip-git-repo-check"]
        if model:
            opts += ["-m", model]
        if effort:
            opts += ["-c", f'model_reasoning_effort="{effort}"']
        if session_id:
            return [exe, "exec", "resume"] + opts + [session_id, full]
        return [exe, "exec"] + opts + [full]
    raise ValueError(f"unknown backend {backend!r}")


def _unwrap(obj: Any) -> str:
    for _ in range(3):
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except Exception:  # noqa: BLE001
                return obj
            if isinstance(parsed, (dict, list)):
                obj = parsed
                continue
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("content"), list):
                obj = "\n".join(str(c.get("text", "")) if c.get("type") == "text" else "[image]" for c in obj["content"] if isinstance(c, dict))
                continue
            if set(obj) == {"result"} and isinstance(obj["result"], str):
                obj = obj["result"]
                continue
        break
    return obj if isinstance(obj, str) else json.dumps(obj)


def _tool_result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        text = "\n".join(str(c.get("text", "")) if c.get("type") == "text" else "[image]" for c in content if isinstance(c, dict))
    else:
        text = str(content or "")
    return _unwrap(text)


def parse_claude(ev: dict, cb: Callable[[str, Any], None]) -> None:
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init" and ev.get("session_id"):
        cb("session", ev["session_id"])
    elif t == "stream_event":
        e = ev.get("event", {}) or {}
        if e.get("type") == "content_block_delta":
            d = e.get("delta", {}) or {}
            if d.get("type") == "text_delta" and d.get("text"):
                cb("delta", d["text"])
    elif t == "assistant":
        for c in ev.get("message", {}).get("content", []) or []:
            if c.get("type") == "text" and c.get("text"):
                cb("text", c["text"])
            elif c.get("type") == "tool_use":
                name = str(c.get("name", "")).replace("mcp__", "").replace("__", "/")
                cb("tool", f"{name} {json.dumps(c.get('input', {}))[:300]}")
    elif t == "user":
        for c in ev.get("message", {}).get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                cb("tool_result", _tool_result_text(c)[:4000])
    elif t == "result":
        cb("stats", {k: ev.get(k) for k in ("duration_ms", "total_cost_usd", "num_turns", "usage") if k in ev})
        if ev.get("is_error") or str(ev.get("subtype", "")).startswith("error"):
            cb("error", str(ev.get("result") or ev.get("subtype")))
        cb("done", str(ev.get("result") or ""))


def parse_codex(ev: dict, cb: Callable[[str, Any], None]) -> None:
    t = ev.get("type", "")
    if t == "thread.started" and ev.get("thread_id"):
        cb("session", ev["thread_id"])
    elif t.startswith("item."):
        it = ev.get("item", {}) or {}
        kind = it.get("type")
        if kind == "agent_message" and t == "item.completed" and it.get("text"):
            cb("text", it["text"])
        elif kind == "mcp_tool_call":
            if t == "item.started":
                cb("tool", f"{it.get('server')}/{it.get('tool')} {json.dumps(it.get('arguments', it.get('input', {})))[:300]}")
            elif t == "item.completed":
                res = it.get("result") if it.get("result") is not None else it.get("output")
                if res is not None:
                    cb("tool_result", _unwrap(res)[:4000])
                elif it.get("status") and it.get("status") != "completed":
                    cb("error", f"{it.get('server')}/{it.get('tool')} {it.get('status')}")
    elif t == "turn.completed":
        cb("stats", {"usage": ev.get("usage") or {}})
        cb("done", "")
    elif t in ("turn.failed", "error"):
        cb("error", json.dumps({k: v for k, v in ev.items() if k != "type"})[:500])


class AgentRun:
    """One prompt → one CLI process, read on a background thread."""

    def __init__(self, backend: str, prompt: str, callback: Callable[[str, Any], None], session_id: Optional[str] = None,
                 extra_env: Optional[dict] = None, extra_context: Optional[str] = None, exe: Optional[str] = None) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        self.exe = exe or find_backend(backend)
        if not self.exe:
            raise RuntimeError(f"{BACKEND_LABELS[backend]} CLI not found. Install it, or set "
                               f"{'LUMEN_CLAUDE_EXE' if backend == 'claude' else 'LUMEN_CODEX_EXE'} to its path.")
        st = backend_settings(backend)
        self.argv = build_argv(backend, self.exe, prompt, session_id, st["model"], st["effort"], extra_context)
        self.callback = callback
        self.env = clean_env()
        self.env.update(extra_env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.session_id = session_id
        self.final_text = ""
        self.cancelled = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "AgentRun":
        self.thread = threading.Thread(target=self._run, name=f"lumen-agent-{self.backend}", daemon=True)
        self.thread.start()
        return self

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                if _win():
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True, timeout=15)
                else:
                    self.proc.terminate()
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def _emit(self, kind: str, payload: Any) -> None:
        if kind == "session":
            self.session_id = payload
        elif kind in ("text", "done") and payload:
            self.final_text = payload
        try:
            self.callback(kind, payload)
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        kw: dict = dict(cwd=workspace_dir(), env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1, encoding="utf-8", errors="replace")
        if _win():
            kw["creationflags"] = 0x08000000
        try:
            self.proc = subprocess.Popen(self.argv, **kw)
        except Exception as e:  # noqa: BLE001
            self._emit("error", f"could not start {self.exe}: {e}")
            self._emit("done", "")
            return
        if self.cancelled:
            self.cancel()
        parser = parse_claude if self.backend == "claude" else parse_codex
        err: List[str] = []

        def drain() -> None:
            try:
                for line in self.proc.stderr:  # type: ignore[union-attr]
                    err.append(line.rstrip())
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=drain, daemon=True).start()
        done = False
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001
                    if not line.startswith("Reading additional input"):
                        self._emit("text", line)
                    continue
                if isinstance(ev, dict):
                    parser(ev, self._emit)
                    if ev.get("type") in ("result", "turn.completed"):
                        done = True
        finally:
            code = self.proc.wait()
            if self.cancelled and not done:
                self._emit("error", "stopped")
            elif code not in (0, None) and not done:
                tail = "\n".join(err[-8:])
                self._emit("error", f"{self.backend} exited with code {code}" + (f": {tail}" if tail else ""))
            if not done:
                self._emit("done", self.final_text)
