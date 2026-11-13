#!/usr/bin/env python3
"""Regenerate docs/COMMANDS.md from the command registry.

    .venv/bin/python scripts/gen_commands_md.py            write docs/COMMANDS.md
    .venv/bin/python scripts/gen_commands_md.py --check    exit 1 when the file is out of date

Sections follow the menu bar (File, Edit, Image, Process, Analyze, Plugins, Help; sub-menus in the order their first
command registers); commands keep their registration order.  Plugins loaded from the user's Lumen folder and Fiji
plugin jars are left out (they differ per machine)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "COMMANDS.md"
TOP = ["File", "Edit", "Image", "Process", "Analyze", "Plugins", "Help"]
#: the menu order of the sections (a new sub-menu goes after the known ones of its top-level menu)
SECTION_ORDER = ["File", "File/Import", "File/Export", "File/New", "File/Text", "Edit", "Edit/Selection", "Edit/Options", "Image", "Image/Type", "Image/Adjust", "Image/Color", "Image/Lookup Tables", "Image/Stacks", "Image/Stacks/Tools", "Image/Hyperstacks", "Image/Transform", "Image/Overlay", "Image/Drawing", "Process", "Process/Filters", "Process/Binary", "Process/Math", "Process/Noise", "Process/FFT", "Process/Batch", "Analyze", "Analyze/Gels", "Analyze/Tools", "Plugins/Macros", "Plugins/Utilities", "Plugins/Segmentation", "Plugins/MorphoLibJ", "Plugins/Tracking", "Plugins/Registration", "Plugins/3D", "Plugins/Batch", "Plugins/Scripting", "Plugins/Bio-Formats", "Plugins/Feature Extraction", "Plugins/Frames", "Plugins/Time Lapse", "Plugins/Tools", "Help"]

HEADER = """# Lumen command reference

{n} commands, generated from the registry. Every command is reachable from the command palette (⌘K / Ctrl+K), the menu bar, the Python scripting API as `lumen.run("<id>", **params)`, and (for the mapped subset) ImageJ-macro `run("…")` calls via the compatibility layer (see docs/FIJI_PARITY.md).

Notes: *live preview* means the operation sheet renders the result on the current plane while you adjust parameters; *needs … selection* commands are greyed out until a matching ROI exists. Parameters are listed in the order the sheet shows them; `help.command` prints the full table (types, defaults, ranges) for any id.
"""


def _bundled_specs() -> list:
    import os
    import tempfile
    os.environ["LUMEN_HOME"] = tempfile.mkdtemp(prefix="lumen-gen-")  # no user plugins / Fiji jars from this machine
    import lumen.core.ops  # noqa: F401 - registers every bundled command
    from lumen.core.registry import REGISTRY
    return [s for s in REGISTRY.ops.values() if not s.id.startswith(("fiji.", "macrotool.", "macros.user_")) and getattr(s.func, "__module__", "").startswith("lumen.")]


def _notes(s) -> str:
    out = []
    if s.shortcut:
        out.append(f"shortcut **{s.shortcut}**")
    if s.preview and s.kind != "analysis":
        out.append("live preview")
    if s.requires_roi:
        kind = {"area": "area", "line": "line", "any": "any", "line_or_rect": "line or rectangle", "point": "point"}.get(str(s.requires_roi), str(s.requires_roi))
        out.append(f"needs {kind} selection")
    return "; ".join(out)


def render() -> str:
    specs = _bundled_specs()
    order: dict[str, int] = {}
    groups: dict[str, list] = {}
    for i, s in enumerate(specs):
        sec = s.menu or s.category or "Other"
        groups.setdefault(sec, []).append(s)
        order.setdefault(sec, i)

    def key(sec: str):
        if sec in SECTION_ORDER:
            return (SECTION_ORDER.index(sec), 0, 0)
        top = sec.split("/")[0]
        last = max((i for i, s in enumerate(SECTION_ORDER) if s.split("/")[0] == top), default=len(SECTION_ORDER))
        return (last, 1, order[sec])

    lines = [HEADER.format(n=len(specs))]
    for sec in sorted(groups, key=key):
        lines.append(f"## {sec}\n\n| Command | id | Parameters | Notes |\n|---|---|---|---|")
        for s in groups[sec]:
            params = ", ".join(f"`{p.name}`" for p in s.params) or "–"
            lines.append(f"| {s.name.replace('|', '/')} | `{s.id}` | {params} | {_notes(s)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        print("docs/COMMANDS.md is " + ("up to date" if current == text else "out of date"))
        return 0 if current == text else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {text.splitlines()[2].split(' commands')[0]} commands in {text.count(chr(10) + '## ')} sections")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
