"""Lumen desktop shell: the FastAPI backend in-process (uvicorn in a thread, serving the bundled
frontend) inside a native pywebview window.  ``python -m lumen_desktop`` from a checkout
(``desktop/run_dev.py`` sets the environment up), or the PyInstaller bundle built by ``desktop/build.py``."""
from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from lumen_desktop.app import main as _main

    return _main(argv)
