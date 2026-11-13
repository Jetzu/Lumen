"""PyInstaller entry point of the Lumen desktop app (``desktop/build/*.spec``)."""
from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # harmless on macOS / Linux, required for frozen Windows builds
    from lumen_desktop.app import main

    sys.exit(main(sys.argv[1:]))
