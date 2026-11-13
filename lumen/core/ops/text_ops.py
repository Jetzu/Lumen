"""Text windows and empty tables (ImageJ's File ▸ New ▸ Text Window / Table…).

Lumen has no floating windows: a text window is a named table in ``session.tables`` with a single ``Text`` column
and one row per line (the UI shows single-column ``Text`` tables as an editable text area; ``file.import_text`` opens
files the same way).  ``table.new`` creates an empty named table with the given columns.

* ``file.new_text(name, text)``   new text window (the name is made unique, like a new ImageJ window)
* ``table.new(name, columns)``    new empty table
* ``text.append(name, text)``     append lines (``print`` to a text window)
* ``text.set(name, text)``        replace the whole text
* ``file.save_text(name, path)``  save the text (or any table, tab-separated) as UTF-8
* ``text.to_log(name)``           print the text window into the Log
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..registry import OpResult, Param, op

NEW = "File/New"
TEXT = "File/Text"
TEXT_COLUMN = "Text"
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)


# ============================================================================ helpers
def text_table(name: str, text: str = "") -> dict[str, Any]:
    """A text window: a table with a single 'Text' column and one row per line."""
    return {"name": str(name), "columns": [TEXT_COLUMN], "rows": [[line] for line in str(text or "").splitlines()]}


def is_text_table(table: Any) -> bool:
    return isinstance(table, dict) and [str(c) for c in (table.get("columns") or [])] == [TEXT_COLUMN]


def table_text(table: dict[str, Any]) -> str:
    """The lines of a text window – or the header + tab-separated cells of any other table."""
    rows = table.get("rows") or []
    if is_text_table(table):
        return "\n".join("" if not r else str(r[0]) for r in rows)
    cols = [str(c) for c in (table.get("columns") or [])]
    lines = ["\t".join(cols)] if cols else []
    lines += ["\t".join("" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(lines)


def parse_columns(columns: Any) -> list[str]:
    """'A, B, C' / 'A\\tB' / one per line / a list → column names (empty entries dropped, duplicates numbered)."""
    if isinstance(columns, (list, tuple)):
        parts = [str(c) for c in columns]
    else:
        parts = re.split(r"[,\t\n;]+", str(columns or ""))
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        name, k = p, 2
        while name in out:
            name, k = f"{p} {k}", k + 1
        out.append(name)
    return out


def unique_table_name(session, name: str) -> str:
    base = str(name or "").strip() or "Untitled"
    if base not in session.tables:
        return base
    stem, suffix = (base[:-4], base[-4:]) if base.lower().endswith((".txt", ".csv", ".tsv")) else (base, "")
    k = 2
    while f"{stem}-{k}{suffix}" in session.tables:
        k += 1
    return f"{stem}-{k}{suffix}"


def find_table(session, name: str = "", text_only: bool = False) -> tuple[str, dict[str, Any]]:
    """The named table – or, with an empty name, the most recently created text window."""
    key = str(name or "").strip()
    if key:
        tb = session.tables.get(key)
        if tb is None:
            raise ValueError(f"No table named {key!r}; open tables: {', '.join(session.tables) or 'none'}")
        if text_only and not is_text_table(tb):
            raise ValueError(f"{key!r} is a {len(tb.get('columns') or [])}-column table, not a text window")
        return key, tb
    for k in reversed(list(session.tables)):
        if is_text_table(session.tables[k]):
            return k, session.tables[k]
    raise ValueError("No text window is open – use File ▸ New ▸ Text Window first")


# ============================================================================ commands
@op("file.new_text", "Text Window…", NEW, **_NOIMG, keywords=["text window", "notes", "new text", "editor"],
    params=[Param("name", "string", "Untitled.txt", help="Window / table name (made unique when it exists)"),
            Param("text", "text", "", help="Initial contents (one row per line)")],
    description="Open a new text window: a named table with a single 'Text' column, one row per line, shown as an editable text area "
                "(ImageJ's File ▸ New ▸ Text Window). Append with text.append, save with file.save_text.")
def new_text(ctx, name="Untitled.txt", text=""):
    tname = unique_table_name(ctx.session, name)
    tb = text_table(tname, text)
    return OpResult(table=tb, message=f"Text window {tname!r}: {len(tb['rows'])} line(s)")


@op("table.new", "Table…", NEW, **_NOIMG, keywords=["new table", "empty table", "results table", "columns"],
    params=[Param("name", "string", "Table", help="Table name (made unique when it exists)"),
            Param("columns", "string", "", help="Column names, comma / tab separated (empty = no columns yet)")],
    description="Create an empty named table with the given columns (ImageJ's File ▸ New ▸ Table…); fill it from scripts or with Edit ▸ Paste.")
def new_table(ctx, name="Table", columns=""):
    tname = unique_table_name(ctx.session, name)
    cols = parse_columns(columns)
    tb = {"name": tname, "columns": cols, "rows": []}
    return OpResult(table=tb, message=f"Table {tname!r} with {len(cols)} column(s)")


@op("text.append", "Append Text…", TEXT, **_NOIMG, keywords=["print", "append", "text window", "log"],
    params=[Param("name", "string", "", help="Text window; empty = the most recent one"), Param("text", "text", "", help="Lines to append")],
    description="Append lines to a text window (ImageJ's print(\"[window]\", text)).")
def append_text(ctx, name="", text=""):
    key, tb = find_table(ctx.session, name, text_only=True)
    lines = str(text or "").splitlines() or ([""] if text == "" else [])
    tb.setdefault("rows", []).extend([[ln] for ln in lines])
    return OpResult(table=tb, message=f"{len(lines)} line(s) appended to {key!r} ({len(tb['rows'])} lines)")


@op("text.set", "Set Text…", TEXT, **_NOIMG, keywords=["replace text", "text window", "set"],
    params=[Param("name", "string", "", help="Text window; empty = the most recent one"), Param("text", "text", "", help="New contents")],
    description="Replace the whole contents of a text window (what the text area's edits save).")
def set_text(ctx, name="", text=""):
    key, tb = find_table(ctx.session, name, text_only=True)
    tb["rows"] = [[ln] for ln in str(text or "").splitlines()]
    return OpResult(table=tb, message=f"{key!r}: {len(tb['rows'])} line(s)")


@op("file.save_text", "Save Text As…", TEXT, **_NOIMG, keywords=["save text", "export", "txt", "text window"],
    params=[Param("name", "string", "", help="Text window (or any table, saved tab-separated); empty = the most recent text window"),
            Param("path", "string", "", help="Output file (.txt added when there is no extension)")],
    description="Save a text window as a UTF-8 text file (any other named table is written tab-separated with a header).")
def save_text(ctx, name="", path=""):
    key, tb = find_table(ctx.session, name)
    p = Path(str(path or "").strip()).expanduser()
    if not str(path or "").strip():
        raise ValueError("Give an output path")
    if not p.suffix:
        p = p.with_suffix(".txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = table_text(tb)
    p.write_text(text + ("\n" if text else ""), encoding="utf-8")
    n = len(tb.get("rows") or [])
    return OpResult(message=f"Saved {key!r} ({n} line(s)) to {p}", log=f"Text saved: {p}")


@op("text.to_log", "Text to Log", TEXT, **_NOIMG, keywords=["log", "print", "text window"],
    params=[Param("name", "string", "", help="Text window; empty = the most recent one")],
    description="Print the contents of a text window into the Log.")
def text_to_log(ctx, name=""):
    key, tb = find_table(ctx.session, name)
    text = table_text(tb)
    n = len(tb.get("rows") or [])
    return OpResult(log=f"--- {key} ---\n" + (text if len(text) <= 20000 else text[:20000] + "\n… (truncated)"), message=f"{n} line(s) of {key!r} printed to the Log")
