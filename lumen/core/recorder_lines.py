"""Shared reading of recorder lines for the consumers of a recording (the folder batch, the ImageJ-macro export).

The recorder (``lumen.core.session.Recorder``) writes Python: ``lumen.run("filters.gaussian", sigma=2.0)``,
``lumen.select("t")``, ``lumen.open(path)  # → "t"``, ``lumen.close("t")``, ``lumen.save(path)``,
``lumen.display("t", …)``, ``lumen.undo("t")`` and ``lumen.redo("t")``.  A replay through the scripting API runs the
undo / redo lines literally, so it ends on the image the user ended up with.  A folder batch or an ImageJ macro cannot
(the batch has no history of the recording session; ImageJ's Undo is single-level, so two ``run("Undo")`` in a row
redo).  :func:`resolve_undo` therefore walks the lines with one undo / redo stack per image title – exactly the
bookkeeping ``Session.run_op`` / ``Session.undo`` / ``Session.redo`` do – and removes every step the user undid
together with the ``lumen.undo`` / ``lumen.redo`` lines themselves (E7-5, E8-3).
"""
from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

_ARROW = re.compile(r"^(.*?)\s+#\s*→\s*(.*)$")
_TITLE = re.compile(r'"((?:[^"\\]|\\.)*)"')
#: op kinds that push an undo entry when they change the image (see ``Session.run_op``)
UNDOABLE_KINDS = ("plane", "image")
#: command families that never leave an undo entry on the image although they are registered as image / plane ops
#: (they act on the session: the ROI Manager, the Results table, folder batches, the recorder)
_NOT_UNDOABLE = ("rois.", "results.", "batch.", "macros.", "file.close")


def parse_line(line: str) -> tuple[str, list[Any], dict[str, Any]] | None:
    """``lumen.<method>(args, kw)`` → (method, args, kwargs) with literal values; None for anything else (comments,
    blank lines, plain Python).  A trailing ``# → "title"`` comment is tolerated (see :func:`created_titles`)."""
    body = line.rstrip()
    m = _ARROW.match(body)
    if m:
        body = m.group(1)
    try:
        tree = ast.parse(body.strip(), mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != "lumen":
        return None
    args: list[Any] = []
    for a in call.args:
        try:
            args.append(ast.literal_eval(a))
        except Exception:  # noqa: BLE001 – a non-literal argument (never written by the recorder) is kept opaque
            args.append(None)
    kw: dict[str, Any] = {}
    for k in call.keywords:
        if k.arg:
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except Exception:  # noqa: BLE001
                kw[k.arg] = None
    return call.func.attr, args, kw


def created_titles(line: str) -> list[str]:
    """The titles in a line's ``# → "a", "b"`` tail (the images the command created / opened), in order."""
    m = _ARROW.match(line.rstrip())
    if not m:
        return []
    out: list[str] = []
    for t in _TITLE.findall(m.group(2)):
        try:
            out.append(json.loads(f'"{t}"'))
        except ValueError:
            out.append(t)
    return out


def creates_undo_entry(op_id: str, registry=None) -> bool | None:
    """Whether ``lumen.run("<op_id>")`` leaves an undo entry on the image it ran on: True for pixel-changing plane /
    image commands, False for analysis commands (measurements, thresholds computed, ROI Manager, Results …), None when
    the command is unknown to the registry (an unloaded plugin: the walk cannot tell)."""
    if any(op_id.startswith(p) for p in _NOT_UNDOABLE):
        return False
    if registry is None:
        from .registry import REGISTRY as registry  # noqa: N813 – lazy: this module is imported by the ops package
    spec = registry.ops.get(op_id)
    if spec is None:
        return None
    return spec.kind in UNDOABLE_KINDS and bool(spec.requires_image)


def _last_clear(lines: list[str], before: int, kinds: tuple[str, ...], dropped: dict[int, Any]) -> int | None:
    """Index of the most recent ``lumen.run("<kind>" …)`` line before ``before`` that is still in the recording."""
    for j in range(before - 1, -1, -1):
        if j in dropped:
            continue
        s = lines[j].strip()
        if any(s.startswith(f'lumen.run("{k}"') for k in kinds):
            return j
    return None


def _rename(stacks: dict[str, list[Any]], old: str, new: str) -> None:
    if old in stacks:
        stacks[new] = stacks.pop(old)


def resolve_undo(lines: list[str], registry=None) -> tuple[list[str], list[dict[str, Any]]]:
    """Remove the undone steps of a recording.

    Returns ``(lines_out, dropped)``: ``lines_out`` is ``lines`` without the steps that were undone and without the
    ``lumen.undo`` / ``lumen.redo`` lines that cancelled / restored them; ``dropped`` lists every removed line as
    ``{"index", "line", "kind", "target"}`` with ``kind`` ``"undone"`` (a command cancelled by an undo), ``"undo"`` or
    ``"redo"`` (the line that did it), in input order.

    The walk keeps one undo and one redo stack per image title.  ``lumen.select("t")``, a ``# → "t"`` tail (the
    created / opened image becomes the recorder's target) and ``lumen.display("t", …)`` set the current image;
    ``lumen.save(path)`` and ``image.rename`` rename it (Save As renames the window, E7-17); ``lumen.close`` /
    ``file.close_all`` forget it.  A ``lumen.run`` whose command creates an undo entry (:func:`creates_undo_entry`) and
    creates no new image is pushed for the current image and empties its redo stack, as ``Session.run_op`` does.
    ``lumen.undo("t")`` pops the most recent such line of ``t`` (it moves to the redo stack; both lines are dropped);
    ``lumen.redo("t")`` moves it back (the command line is kept again, the redo line is dropped).  An undo or redo
    with nothing to cancel / restore – or whose top step is a command unknown to the registry – is left in place for
    the caller to report (the batch skips it, the macro export writes a ``// not converted`` comment).
    """
    undo: dict[str, list[tuple[int, bool]]] = {}  # title → [(line index, known command)] not yet undone, oldest first
    redo: dict[str, list[tuple[int, bool]]] = {}
    dropped: dict[int, dict[str, Any]] = {}
    current: str | None = None
    for i, raw in enumerate(lines):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parsed = parse_line(raw)
        if parsed is None:
            continue
        method, args, kw = parsed
        first = args[0] if args and isinstance(args[0], str) else None
        created = created_titles(raw)
        if method == "select":
            current = first
        elif method == "open":
            current = created[-1] if created else current
        elif method == "close":
            t = first if first is not None else current
            if t is not None:
                undo.pop(t, None)
                redo.pop(t, None)
            if first is None or t == current:
                current = None
        elif method == "save":
            if current is not None and first:
                new = os.path.basename(first)
                if new and new != current:
                    _rename(undo, current, new)
                    _rename(redo, current, new)
                    current = new
        elif method == "display":
            if first is not None:
                current = first
            # a display *edit* (range, LUT, visibility, mode …) is an undo step of its own (E5-8); a plane change is not
            if current is not None and any(k in kw for k in ("min", "max", "lut", "visible", "inverted", "gamma", "mode", "name")):
                undo.setdefault(current, []).append((i, True))
                redo.pop(current, None)
        elif method == "undo":
            t = first if first is not None else current
            stack = undo.get(t) if t is not None else None
            if stack and stack[-1][1]:
                j, _known = stack.pop()
                redo.setdefault(t, []).append((j, True))
                dropped[j] = {"index": j, "line": lines[j], "kind": "undone", "target": t}
                dropped[i] = {"index": i, "line": raw, "kind": "undo", "target": t}
        elif method == "redo":
            t = first if first is not None else current
            stack = redo.get(t) if t is not None else None
            if stack:
                j, _known = stack.pop()
                undo.setdefault(t, []).append((j, True))
                dropped.pop(j, None)
                dropped[i] = {"index": i, "line": raw, "kind": "redo", "target": t}
        elif method == "run":
            op_id = first or ""
            if op_id == "file.close_all":
                undo.clear()
                redo.clear()
                current = None
                continue
            if op_id == "image.rename" and current is not None and isinstance(kw.get("title"), str) and kw["title"]:
                _rename(undo, current, kw["title"])
                _rename(redo, current, kw["title"])
                current = kw["title"]
                continue
            if op_id in ("rois.restore_cleared", "results.restore_cleared"):
                # "Undo clear" took the clear back: neither the clear nor the restore belongs in a replay (E7-5, E10)
                kinds = ("rois.clear", "rois.delete") if op_id.startswith("rois.") else ("analyze.clear_results", "results.clear")
                j = _last_clear(lines, i, kinds, dropped)
                if j is not None:
                    dropped[j] = {"index": j, "line": lines[j], "kind": "undone", "target": current}
                    dropped[i] = {"index": i, "line": raw, "kind": "undo", "target": current}
                continue
            if not created and current is not None:
                undoable = creates_undo_entry(op_id, registry)
                if undoable is not False:  # True, or None (unknown command: pushed as opaque so a later undo is not mis-matched)
                    undo.setdefault(current, []).append((i, undoable is True))
                    redo.pop(current, None)  # a new step empties the redo stack (Session.run_op)
            if created:
                current = created[-1]
    out = [ln for i, ln in enumerate(lines) if i not in dropped]
    return out, [dropped[i] for i in sorted(dropped)]


def undone_lines(dropped: list[dict[str, Any]]) -> list[str]:
    """The command lines an undo cancelled (``kind == "undone"``) from a :func:`resolve_undo` result."""
    return [d["line"] for d in dropped if d["kind"] == "undone"]


__all__ = ["resolve_undo", "undone_lines", "creates_undo_entry", "parse_line", "created_titles", "UNDOABLE_KINDS"]
