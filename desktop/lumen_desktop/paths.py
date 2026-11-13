"""Where things are: the source checkout vs. the PyInstaller bundle, and Lumen's per-user folder.

Lumen keeps one per-user folder on every platform (``$LUMEN_HOME`` or ``~/.lumen``: recent files, plugins, macros,
the sample cache, startup scripts) and the desktop app shares it with the ``lumen`` command, so plugins installed in
one show up in the other.  The shell's own files live in its ``desktop`` sub-folder: ``desktop.log``, the window
state, the single-instance lock and the web view's storage."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Lumen"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path | None:
    """PyInstaller's ``_MEIPASS`` (one-folder: the ``_internal`` directory / ``Contents/Frameworks``)."""
    root = getattr(sys, "_MEIPASS", None)
    return Path(root) if root else None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    root = bundle_root()
    return (root / "desktop_assets") if root else repo_root() / "desktop" / "assets"


def lumen_home() -> Path:
    """The same folder ``lumen.core.io.lumen_home()`` uses (kept here so nothing heavy is imported to find it)."""
    custom = os.environ.get("LUMEN_HOME", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".lumen"


def desktop_dir() -> Path:
    return lumen_home() / "desktop"


def prepare_environment() -> None:
    """Make the ``lumen`` package importable from a source checkout (the bundle carries it already) and keep
    Matplotlib on its file-only backend (Lumen renders plots to images, never to a window)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["LUMEN_DESKTOP"] = "1"
    if not is_frozen():
        root = str(repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
