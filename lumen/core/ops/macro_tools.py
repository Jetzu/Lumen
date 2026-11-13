"""ImageJ macro-language tools: ``macro "Circle Tool - C00cO11cc" { getCursorLoc(x, y, z, flags); … }``.

Fiji installs the tool macros of an ``.ijm`` *macro set* (Plugins ▸ Macros ▸ Install…, the macros / plugins folders,
``StartupMacros.txt``) on the toolbar.  Lumen does the same on its tool rail: every ``… Tool`` block of a macro set in
``<lumen home>/plugins``, ``<lumen home>/macros``, ``$LUMEN_PLUGINS`` or ``<lumen home>/startup.ijm`` becomes a tool op
(``tool=True``, id ``macrotool.<slug>``, menu Plugins ▸ Tools) whose icon and colour are derived from the ImageJ icon
string.  A click or drag on the image runs the macro body with the ImageJ tool functions available:

``getCursorLoc(x, y, z, flags)``   the interaction: x, y (image px), z (slice index) and the flags bits 1 shift, 2 ctrl,
                                   4 meta / right button, 8 alt / middle button, 16 left button (ImageJ's AWT constants).
                                   ImageJ runs a tool macro once per press and lets it loop ``while (flags&16!=0)``; Lumen
                                   hands out the press (button down), for a drag the release position (button still
                                   down), then the release (button bits cleared) – so looping macros see the drag end.
                                   A drag is one run, made at the drag end (the press alone cannot tell whether the macro
                                   will loop); ``drag-start`` does nothing.
``isKeyDown`` / ``setKeyDown``, ``toolID``, ``setTool``, ``IJ.getToolName()``, ``call("ij.gui.Toolbar.getToolId")``,
``call("ij.Prefs.get" / "ij.Prefs.set", …)``, ``getBoolean``, ``showStatus`` (the toast), ``wait`` (no-op) …

``macro "Name Tool Options" { … }`` runs when the tool is run with ``options=True`` (ImageJ: double-click on the tool), and
``macro "Name Tool Selected" { … }`` with ``selected=True``; both parameters only exist when the block does.  The set's
``var`` globals and functions persist between runs (per session, reset when the file changes), so an Options macro can
set a ``var`` the tool reads afterwards.  ``… Action Tool`` blocks (ImageJ: run on a toolbar click) become plain commands,
the other named macros of a tool set commands under Plugins ▸ Macros; ``… Menu Tool`` blocks are not supported.

Bookkeeping: ``MACRO_TOOL_FILES[file] = [op ids]``, ``MACRO_TOOL_ERRORS[file] = message``; ``load_macro_tools()`` rescans
everything (Plugins ▸ Utilities ▸ Compile and Run / Update Menus), ``register_file(path)`` one file (Install…).
"""
from __future__ import annotations

import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..registry import REGISTRY, OpResult, OpSpec, Param

TOOLS = "Plugins/Tools"
MACROS = "Plugins/Macros"
PREFIX = "macrotool"

#: ImageJ's getCursorLoc flag bits (java.awt.event.InputEvent masks, as used by ij.gui.ImageCanvas)
FLAG_SHIFT, FLAG_CTRL, FLAG_META, FLAG_ALT, FLAG_LEFT = 1, 2, 4, 8, 16
FLAG_RIGHT, FLAG_MIDDLE = FLAG_META, FLAG_ALT  # AWT: BUTTON3_MASK == META_MASK, BUTTON2_MASK == ALT_MASK
#: Lumen button numbers (0 left, 1 middle, 2 right) → flag bit
BUTTON_FLAGS = {0: FLAG_LEFT, 1: FLAG_MIDDLE, 2: FLAG_RIGHT}

#: file → op ids registered from it (tool ops, action tools and the named macros of a tool macro set)
MACRO_TOOL_FILES: dict[str, list[str]] = {}
#: file → the last parse / registration error
MACRO_TOOL_ERRORS: dict[str, str] = {}

_MACRO_HEAD = re.compile(r'macro\s*(["\'])((?:[^"\'\\]|\\.)*?)\1\s*\{')


# ============================================================================ parsing
@dataclass
class MacroBlock:
    """One ``macro "Name" { … }`` block of a macro set (``name`` includes the `` - icon`` part, if any)."""
    name: str
    body: str
    line: int = 1

    @property
    def base(self) -> str:
        return self.name.split(" - ", 1)[0].strip()

    @property
    def icon(self) -> str:
        return self.name.split(" - ", 1)[1].strip() if " - " in self.name else ""


@dataclass
class ToolMacro:
    name: str  # "Circle Tool"
    icon: str = ""  # ImageJ icon string, e.g. "C00cO11cc"
    body: str = ""
    options: str | None = None  # "<name> Options" block
    selected: str | None = None  # "<name> Selected" block
    action: bool = False  # "… Action Tool": runs on a toolbar click (a plain command in Lumen)
    line: int = 1


@dataclass
class MacroSet:
    """A parsed ``.ijm`` file: its ``macro`` blocks and the code outside them (``var`` globals, functions, auto-run code)."""
    text: str
    blocks: list[MacroBlock] = field(default_factory=list)
    install_code: str = ""
    path: Path | None = None

    @property
    def tools(self) -> list[ToolMacro]:
        return tool_macros(self.blocks)

    @property
    def macros(self) -> list[MacroBlock]:
        """The named macros that are not tool definitions (ImageJ: menu commands of the installed set)."""
        from ..ij_compat import is_tool_macro_name
        return [b for b in self.blocks if not is_tool_macro_name(b.name)]


def _skip_string_or_comment(text: str, i: int) -> int:
    """If ``text[i]`` starts a string literal or a comment, return the index after it; else ``i``."""
    ch = text[i]
    if ch == "/" and text.startswith("//", i):
        j = text.find("\n", i)
        return len(text) if j < 0 else j
    if ch == "/" and text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return len(text) if j < 0 else j + 2
    if ch in "\"'":
        j = i + 1
        while j < len(text) and text[j] != ch:
            j += 2 if text[j] == "\\" else 1
        return min(j + 1, len(text))
    return i


def _match_brace(text: str, start: int) -> int:
    """Index of the ``}`` closing the block whose body starts at ``start`` (strings and comments are skipped)."""
    depth, i = 1, start
    while i < len(text):
        j = _skip_string_or_comment(text, i)
        if j != i:
            i = j
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"missing '}}' in macro block starting at line {text.count(chr(10), 0, start) + 1}")


def parse_macro_set(text: str, path: "Path | None" = None) -> MacroSet:
    """Split macro-set source into its ``macro "…" { … }`` blocks and the install code around them."""
    blocks: list[MacroBlock] = []
    parts: list[str] = []
    depth, i, seg = 0, 0, 0
    n = len(text)
    while i < n:
        j = _skip_string_or_comment(text, i)
        if j != i:
            i = j
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif depth == 0 and ch == "m" and text.startswith("macro", i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = _MACRO_HEAD.match(text, i)
            if m is not None:
                end = _match_brace(text, m.end())
                blocks.append(MacroBlock(m.group(2).strip(), text[m.end():end], line=text.count("\n", 0, i) + 1))
                parts.append(text[seg:i])
                seg = i = end + 1
                continue
        i += 1
    parts.append(text[seg:])
    return MacroSet(text=text, blocks=blocks, install_code="".join(parts), path=path)


def tool_macros(blocks: list[MacroBlock]) -> list[ToolMacro]:
    """Group the ``… Tool`` / ``… Tool Options`` / ``… Tool Selected`` / ``… Action Tool`` blocks into tools."""
    tools: dict[str, ToolMacro] = {}
    defined: set[str] = set()
    for b in blocks:
        base, icon = b.base, b.icon
        if base.endswith(" Menu Tool"):
            continue  # not supported (a drop-down of sub-commands)
        if base.endswith(" Action Tool"):
            t = tools.setdefault(base, ToolMacro(name=base))
            t.icon, t.body, t.action, t.line = icon, b.body, True, b.line
            defined.add(base)
        elif base.endswith(" Tool Options"):
            tools.setdefault(base[: -len(" Options")], ToolMacro(name=base[: -len(" Options")])).options = b.body
        elif base.endswith(" Tool Selected"):
            tools.setdefault(base[: -len(" Selected")], ToolMacro(name=base[: -len(" Selected")])).selected = b.body
        elif base.endswith(" Tool"):
            t = tools.setdefault(base, ToolMacro(name=base))
            t.icon, t.body, t.line = icon, b.body, b.line
            defined.add(base)
    return [t for name, t in tools.items() if name in defined]


# ============================================================================ icons
_FILL_GLYPHS = {"F": "■", "V": "●", "o": "●"}
_SHAPE_GLYPHS = {"O": "◯", "R": "▭", "L": "╱", "P": "⬠", "D": "⁘", "G": "⁘", "H": "╱"}
#: ImageJ icon commands → number of hex digits that follow (T and P are variable)
_ICON_ARGS = {"B": 2, "D": 2, "G": 2, "R": 4, "F": 4, "O": 4, "V": 4, "o": 4, "L": 4, "H": 4, "C": 3, "N": 2, "A": 0}


def decode_icon(icon: str) -> tuple[str, str]:
    """(glyph, css colour) for an ImageJ tool icon string (``C00cO11cc`` = blue circle, ``Cf00T0b12C`` = red "C").

    The glyph is the text of the first ``T`` commands (up to two characters) or a generic shape for the first drawn shape;
    the colour the first ``C`` command (3 hex digits, each doubled: ``C037`` → #003377), "" when there is none.
    """
    s = str(icon or "").strip()
    text = ""
    color = ""
    shape = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == "T":  # Txyss<char>: x, y (hex), font size (2 digits), one character
            ch = s[i + 5:i + 6]
            if ch:
                text += ch
            i += 6
            continue
        if c == "C":
            hx = s[i + 1:i + 4]
            if len(hx) == 3 and all(h in "0123456789abcdefABCDEF" for h in hx) and not color:
                color = "#" + "".join(h * 2 for h in hx.lower())
            i += 4
            continue
        if c == "P":  # polyline: xy pairs ended by '0'
            j = i + 1
            while j < len(s) and s[j] != "0":
                j += 1
            shape = shape or "P"
            i = j + 1
            continue
        if c in _ICON_ARGS:
            if c in _FILL_GLYPHS or c in _SHAPE_GLYPHS:
                shape = shape or c
            i += 1 + _ICON_ARGS[c]
            continue
        i += 1
    glyph = text[:2] if text else (_FILL_GLYPHS.get(shape) or _SHAPE_GLYPHS.get(shape) or "⚙")
    return glyph, color


# ============================================================================ persistent macro-set state
# session → {"<file>:<text hash>": env}: the ``var`` globals / functions of an installed set live between runs (ImageJ keeps
# them while the set is installed); a changed file starts afresh
_ENVS: "weakref.WeakKeyDictionary[Any, dict[str, dict[str, Any]]]" = weakref.WeakKeyDictionary()
_SETS: dict[str, MacroSet] = {}  # file → parsed set (of the text it had)


def _macro_set(path: Path) -> MacroSet:
    """The parsed macro set of a file, re-read on every run and re-parsed when its text changed (edit the .ijm freely)."""
    key = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise FileNotFoundError(f"{path.name} is gone ({e}) – Plugins ▸ Utilities ▸ Compile and Run / Update Menus removes its tools") from e
    cached = _SETS.get(key)
    if cached is not None and cached.text == text:
        return cached
    ms = parse_macro_set(text, path)
    _SETS[key] = ms
    return ms


def _env(session: Any, key: str, ms: MacroSet, ij: Any) -> dict[str, Any]:
    envs = _ENVS.get(session)
    if envs is None:
        envs = _ENVS[session] = {}
    sig = f"{key}:{hash(ms.text)}"
    env = envs.get(sig)
    if env is None:
        for k in [k for k in envs if k.startswith(key + ":")]:  # the file changed: forget its old state
            del envs[k]
        env = envs[sig] = {}
        if ms.install_code.strip():
            ij.macro(ms.install_code, env=env)  # var initialisers, function definitions, auto-run code
    return env


def _slug(name: str) -> str:
    from .plugins_ops import _slug as slug
    return slug(name)


def _unique_id(prefix: str, slug: str, mine: set[str]) -> str:
    oid, k = f"{prefix}.{slug}", 1
    while oid in REGISTRY.ops and oid not in mine:
        k += 1
        oid = f"{prefix}.{slug}_{k}"
    return oid


def _finish(ij: Any, ctx: Any, output: str, default_message: str) -> OpResult:
    """OpResult of a macro run: the Log text, the status / last printed line as the toast, the macro's selection."""
    last = output.rstrip().splitlines()[-1] if output.strip() else ""
    res = OpResult(log=output or None, message=ij.status or last or default_message)
    if ij.roi is not None:
        res.roi = ij.roi
    elif ctx.roi is not None:
        res.clear_roi = True
    return res


def _ij(ctx: Any, tool_name: str = "") -> Any:
    from ..ij_compat import IJ
    ij = IJ(ctx.session)
    ij.roi = ctx.roi
    if tool_name:
        ij._tool = tool_name
    return ij


def _tool_of(ms: MacroSet, name: str, path: Path) -> ToolMacro:
    for t in ms.tools:
        if t.name == name:
            return t
    raise ValueError(f'"{name}" is no longer defined in {path.name} – Plugins ▸ Utilities ▸ Compile and Run / Update Menus refreshes the tool rail')


def run_tool(ctx: Any, path: Path, name: str, *, options: bool = False, selected: bool = False, x: float = 0.0, y: float = 0.0,
             x2: float | None = None, y2: float | None = None, button: int = 0, shift: bool = False, alt: bool = False, ctrl: bool = False,
             event: str = "click") -> OpResult:
    """Run the macro tool ``name`` of the macro set ``path`` for one interaction (see the module docstring)."""
    ms = _macro_set(path)
    t = _tool_of(ms, name, path)
    ij = _ij(ctx, t.name)
    env = _env(ctx.session, str(path), ms, ij)
    if options or selected:
        which = "Options" if options else "Selected"
        code = t.options if options else t.selected
        if code is None:
            raise ValueError(f'{t.name} has no "{t.name} {which}" macro in {path.name}')
        return _finish(ij, ctx, ij.macro(code, env=env), f"{t.name} {which} macro finished")
    if t.action:
        return _finish(ij, ctx, ij.macro(t.body, env=env), f"{t.name} finished")
    if event == "drag-start":
        return OpResult()  # the run happens at the drag end, with the whole drag (ImageJ: one run per press)
    mods = (FLAG_SHIFT if shift else 0) | (FLAG_CTRL if ctrl else 0) | (FLAG_ALT if alt else 0)
    btn = BUTTON_FLAGS.get(int(button), FLAG_LEFT)
    img = ctx.session.active
    z = int(img.display.z) if img is not None else 0
    x, y = float(x), float(y)
    states = [(x, y, z, mods | btn)]
    if event == "drag-end" and x2 is not None and y2 is not None:
        x, y = float(x2), float(y2)
        states.append((x, y, z, mods | btn))
    states.append((x, y, z, mods & ~btn))
    keys = {k for k, on in (("shift", shift), ("alt", alt), ("ctrl", ctrl)) if on}
    ij.set_cursor(states, keys)
    try:
        out = ij.macro(t.body, env=env)
    finally:
        ij.set_cursor([], set())
    where = f"({states[0][0]:g}, {states[0][1]:g})" + (f" → ({x:g}, {y:g})" if len(states) > 2 else "")
    return _finish(ij, ctx, out, f"{t.name} at {where}")


def run_named_macro(ctx: Any, path: Path, name: str) -> OpResult:
    """Run the named (non-tool) macro ``name`` of a tool macro set, in the set's persistent namespace."""
    ms = _macro_set(path)
    for b in ms.macros:
        if b.base == name:
            ij = _ij(ctx)
            env = _env(ctx.session, str(path), ms, ij)
            return _finish(ij, ctx, ij.macro(b.body, env=env), f"{name} finished")
    raise ValueError(f'"{name}" is no longer defined in {path.name} – Plugins ▸ Utilities ▸ Compile and Run / Update Menus refreshes the commands')


# ============================================================================ registration
def _register_tool(path: Path, t: ToolMacro, mine: set[str]) -> str:
    glyph, color = decode_icon(t.icon)
    name, pname = t.name, path.name
    oid = _unique_id(PREFIX, _slug(name), mine)
    params: list[Param] = []
    if t.options is not None:
        params.append(Param("options", "bool", False, label="Tool Options", help=f'Run the "{name} Options" macro instead (ImageJ: double-click on the tool)'))
    if t.selected is not None:
        params.append(Param("selected", "bool", False, label="Tool Selected", help=f'Run the "{name} Selected" macro instead (ImageJ: when the tool is picked)'))
    extras = " / ".join(f'"{name} {w}"' for w, b in (("Options", t.options), ("Selected", t.selected)) if b is not None)

    if t.action:
        def fn(ctx, _path=path, _name=name):
            return run_tool(ctx, _path, _name)

        description = (f'ImageJ action tool "{name}" from {pname}: runs its macro (ImageJ runs it when its toolbar button is clicked). '
                       f"Edit the .ijm file freely – it is re-read when it changes.")
        spec = OpSpec(id=oid, name=name, category=TOOLS, func=fn, kind="analysis", description=description, params=[], preview=False, roi_aware=False,
                      requires_image=False, menu=TOOLS, keywords=["tool", "macro", "ijm", "action tool", path.stem], scope_param=False, icon=glyph)
    else:
        def fn(ctx, options=False, selected=False, x=0.0, y=0.0, x2=None, y2=None, button=0, shift=False, alt=False, ctrl=False, event="click",
               _path=path, _name=name):
            return run_tool(ctx, _path, _name, options=bool(options), selected=bool(selected), x=x, y=y, x2=x2, y2=y2, button=button, shift=shift,
                            alt=alt, ctrl=ctrl, event=event)

        description = (f'ImageJ macro tool "{name}" from {pname}: a click or drag on the image runs its macro – getCursorLoc(x, y, z, flags) '
                       f"gives the press (and, for a drag, the release); flags: 16 left button, 4 right, 1 shift, 2 ctrl, 8 alt. "
                       + (f"Also defines {extras}. " if extras else "") + "Edit the .ijm file freely – it is re-read when it changes.")
        spec = OpSpec(id=oid, name=name, category=TOOLS, func=fn, kind="analysis", description=description, params=params, preview=False,
                      roi_aware=False, requires_image=False, menu=TOOLS, keywords=["tool", "macro", "ijm", "macro tool", path.stem], scope_param=False,
                      tool=True, tool_icon=glyph, tool_cursor="crosshair", tool_color=color)
    fn.__name__ = _slug(name)
    fn.__qualname__ = f"macro_tools.{fn.__name__}"
    fn.spec = spec  # type: ignore[attr-defined]
    REGISTRY.register(spec)
    mine.add(oid)
    return oid


def _register_macro(path: Path, b: MacroBlock, mine: set[str]) -> str:
    name = re.sub(r"\s*\[[^\]]*\]\s*$", "", b.base).strip() or b.base  # drop ImageJ's "[shortcut key]" suffix
    oid = _unique_id("macros", _slug(name), mine)

    def fn(ctx, _path=path, _name=b.base):
        return run_named_macro(ctx, _path, _name)

    fn.__name__ = _slug(name)
    spec = OpSpec(id=oid, name=name, category=MACROS, func=fn, kind="analysis", preview=False, roi_aware=False, requires_image=False, menu=MACROS,
                  description=f'ImageJ macro "{name}" of the macro set {path.name} (installed with its tools; shares the set\'s var globals and functions).',
                  keywords=["macro", "ijm", "installed", path.stem], scope_param=False)
    fn.spec = spec  # type: ignore[attr-defined]
    REGISTRY.register(spec)
    mine.add(oid)
    return oid


def unregister_file(path: "Path | str") -> list[str]:
    """Drop every command registered from a macro-set file; returns the removed op ids."""
    key = str(path)
    ids = MACRO_TOOL_FILES.pop(key, [])
    MACRO_TOOL_ERRORS.pop(key, None)
    _SETS.pop(key, None)
    for oid in ids:
        REGISTRY.ops.pop(oid, None)
    return ids


def register_file(path: "Path | str", session: Any = None, named_macros: bool = True) -> list[str]:
    """(Re)register the tools of one macro-set file (and, for a tool set, its other named macros as commands).

    Returns the op ids; a file without tool macros registers nothing.  Parse errors are kept in ``MACRO_TOOL_ERRORS``
    and raised.  With a ``session`` the registry change is broadcast (``ops`` event)."""
    p = Path(path)
    key = str(p)
    unregister_file(key)
    try:
        ms = _macro_set(p)
        tools = ms.tools
    except Exception as e:
        MACRO_TOOL_ERRORS[key] = f"{type(e).__name__}: {e}"
        raise
    mine: set[str] = set()
    ids: list[str] = []
    if tools:
        for t in tools:
            ids.append(_register_tool(p, t, mine))
        if named_macros:
            for b in ms.macros:
                ids.append(_register_macro(p, b, mine))
    MACRO_TOOL_FILES[key] = ids
    if session is not None and ids:
        from .plugins_ops import _ops_changed
        _ops_changed(session)
    return ids


def has_tool_macros(text: str) -> bool:
    """True when macro source defines at least one ``… Tool`` block (so Install… registers tools instead of a wrapper command)."""
    try:
        return bool(parse_macro_set(text).tools)
    except ValueError:
        return False


def macro_tool_files() -> list[Path]:
    """The ``.ijm`` files scanned for tool macros: the plugin folders, ``<lumen home>/macros`` and ``<lumen home>/startup.ijm``."""
    from . import lumen_home, plugin_dirs
    home = lumen_home()
    files: list[Path] = []
    seen: set[str] = set()
    for d in plugin_dirs() + [home / "macros"]:
        try:
            if not d.is_dir():
                continue
            cands = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".ijm" and not f.name.startswith("."))
        except OSError:
            continue
        for f in cands:
            if str(f) not in seen:
                seen.add(str(f))
                files.append(f)
    startup = home / "startup.ijm"
    try:
        if startup.is_file() and str(startup) not in seen:
            files.append(startup)
    except OSError:
        pass
    return files


def load_macro_tools(session: Any = None) -> dict[str, Any]:
    """Rescan every macro-set file: drop the commands registered before and register the tools found now.

    Returns ``{"files": [paths with tools], "ops": {file: [ids]}, "errors": {file: message}, "removed": [ids]}``."""
    removed: list[str] = []
    for key in list(MACRO_TOOL_FILES):
        removed += unregister_file(key)
    MACRO_TOOL_ERRORS.clear()
    files: list[str] = []
    for f in macro_tool_files():
        try:
            ids = register_file(f)
        except Exception as e:  # noqa: BLE001 – one broken macro file must not stop the others (nor the app start)
            MACRO_TOOL_ERRORS[str(f)] = f"{type(e).__name__}: {e}"
            continue
        if ids:
            files.append(str(f))
    if session is not None:
        from .plugins_ops import _ops_changed
        _ops_changed(session)
    return {"files": files, "ops": {k: list(v) for k, v in MACRO_TOOL_FILES.items() if v}, "errors": dict(MACRO_TOOL_ERRORS), "removed": removed}


def tool_id_of(name: str) -> int:
    """ImageJ-style tool id of a registered macro tool (CUSTOM1 = 15 onwards, in registration order)."""
    from ..ij_compat import IJ_CUSTOM_TOOL_ID
    k = IJ_CUSTOM_TOOL_ID
    for sp in REGISTRY.ops.values():
        if sp.tool and sp.id.startswith(PREFIX + "."):
            if sp.name == name or sp.id == name:
                return k
            k += 1
    return IJ_CUSTOM_TOOL_ID


try:  # start-up scan of the user's macro files; never let a broken file stop the command modules from loading
    load_macro_tools()
except Exception as _e:  # pragma: no cover
    MACRO_TOOL_ERRORS["<scan>"] = f"{type(_e).__name__}: {_e}"
