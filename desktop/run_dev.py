#!/usr/bin/env python3
"""Run the Lumen desktop shell from source (no PyInstaller): the ``lumen`` package of this checkout, the built UI in
lumen/static and pywebview from ``desktop/.venv-build`` (created on first use, the same environment the packaged
build uses).

    python desktop/run_dev.py [files…]        open the window (Lumen folder ~/.lumen, like the packaged app)
    python desktop/run_dev.py --build-frontend   rebuild lumen/static first
    python desktop/run_dev.py --home DIR      use another Lumen folder (also LUMEN_HOME)
    python desktop/run_dev.py --smoke         start the server, check it, exit (no window)
    python desktop/run_dev.py --debug         web inspector + control-socket test hooks
    python desktop/run_dev.py --recreate-venv rebuild desktop/.venv-build

Every other argument is passed through to ``python -m lumen_desktop`` (``--port``, ``--gui``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _buildenv as be  # noqa: E402


def main(argv: list[str]) -> int:
    build_frontend = "--build-frontend" in argv
    recreate = "--recreate-venv" in argv
    passthrough = [a for a in argv if a not in ("--build-frontend", "--recreate-venv")]
    be.ensure_frontend(build=build_frontend)
    py = be.ensure_build_venv(recreate=recreate)
    env = be.desktop_env()
    be.log(f"starting the desktop shell from source with {py}")
    try:
        return subprocess.call([str(py), "-m", "lumen_desktop", *passthrough], cwd=str(be.DESKTOP), env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
