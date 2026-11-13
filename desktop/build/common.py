"""Shared PyInstaller analysis inputs for the three per-OS spec files (``macos.spec``, ``windows.spec``,
``linux.spec``).  The spec files ``exec`` with ``SPECPATH`` set; they import this module from there.

What goes into the bundle:
* ``desktop/build/entry.py`` as the program; ``lumen`` (this checkout) and ``desktop/lumen_desktop`` as modules.
  ``lumen`` is collected as **source files** (``module_collection_mode="py"``), not into the PYZ archive: the command
  modules (``lumen/core/ops/*.py``) and the server's extra routers are discovered at runtime with
  ``pkgutil.iter_modules`` over the package folder, plugin templates read sources, and several modules locate data
  relative to ``__file__`` - all of that behaves exactly as in a checkout this way.  Every module is still named as a
  hidden import so PyInstaller follows their imports.
* ``lumen/static`` (the built UI, a snapshot made by build.py) → ``lumen/static``,
* ``README.md`` and ``docs/*.md`` → ``README.md`` / ``docs`` (Help ▸ docs; settings.py looks two folders above the
  package),
* ``desktop/assets`` → ``desktop_assets`` (window / dock icon),
* scikit-image's bundled sample images (File ▸ Open Sample), scikit-learn, SciPy, imagecodecs and uvicorn through
  their PyInstaller hooks (pyinstaller-hooks-contrib) plus explicit submodule lists where they import lazily."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

DESKTOP = Path(__file__).resolve().parent.parent
ROOT = DESKTOP.parent
PACKAGE = ROOT / "lumen"
STATIC = Path(os.environ.get("LUMEN_BUILD_STATIC") or PACKAGE / "static")  # build.py passes a snapshot
ASSETS = DESKTOP / "assets"
ENTRY = DESKTOP / "build" / "entry.py"
APP_NAME = "Lumen"
BUNDLE_ID = "io.github.jetzu.lumen"

#: the image types the macOS bundle declares (Finder "Open With ▸ Lumen", drop on the Dock icon): (name, UTIs, extensions)
DOCUMENT_TYPES = [
    ("TIFF image", ["public.tiff"], ["tif", "tiff"]),
    ("PNG image", ["public.png"], ["png"]),
    ("JPEG image", ["public.jpeg"], ["jpg", "jpeg"]),
    ("Microscopy image", [], ["czi", "nd2", "lif", "lsm", "oib", "oif", "ims", "zvi", "lei", "stk", "dm3", "dm4", "ics", "ids"]),
    ("Scientific volume", [], ["h5", "hdf5", "nii", "mrc", "mrcs", "rec", "map", "fits", "fit", "mha", "mhd", "nrrd", "npy", "dcm"]),
]


def app_version() -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (PACKAGE / "__init__.py").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "0.0.0"


def lumen_modules() -> list[str]:
    """Every module under lumen/ (dotted names): commands and routers are imported by discovery at runtime."""
    mods: list[str] = []
    for py in PACKAGE.rglob("*.py"):
        rel = py.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if any(p == "__pycache__" for p in parts):
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    return sorted(mods)


def check_inputs() -> None:
    if not (STATIC / "index.html").is_file():
        raise SystemExit(f"UI build missing: {STATIC} (run `npm run build` in frontend/ or desktop/build.py)")
    if not (PACKAGE / "server" / "app.py").is_file():
        raise SystemExit(f"lumen package missing: {PACKAGE}")


def datas() -> list[tuple[str, str]]:
    items = [
        (str(STATIC), os.path.join("lumen", "static")),
        (str(ROOT / "README.md"), "."),
        (str(ASSETS), "desktop_assets"),
    ]
    items += [(str(md), "docs") for md in sorted((ROOT / "docs").glob("*.md"))]
    for pkg in ("skimage", "imageio", "tifffile"):
        try:
            items += collect_data_files(pkg)
        except Exception:  # noqa: BLE001 - optional package / no data
            pass
    return items


def hiddenimports() -> list[str]:
    mods = lumen_modules()
    mods += ["lumen_desktop", "lumen_desktop.app", "lumen_desktop.bridge", "lumen_desktop.instance", "lumen_desktop.menu",
             "lumen_desktop.paths", "lumen_desktop.server", "lumen_desktop.splash", "lumen_desktop.state", "lumen_desktop.macos"]
    mods += collect_submodules("uvicorn")
    try:  # the MCP server (Lumen --mcp; lumen.mcp_server imports the SDK lazily); its CLI extra (typer) is not shipped
        mods += collect_submodules("mcp", filter=lambda name: not name.startswith("mcp.cli"))
        import importlib.util
        mods += [m for m in ("httpx2", "httpx", "sse_starlette", "jsonschema", "pydantic_settings") if importlib.util.find_spec(m)]
    except Exception:  # noqa: BLE001
        pass
    for pkg in ("skimage", "imagecodecs", "tifffile", "imageio.plugins", "sklearn.ensemble", "sklearn.tree", "sklearn.utils", "sklearn.neighbors",
                "scipy.ndimage", "scipy.signal", "scipy.optimize", "scipy.spatial", "scipy.interpolate", "scipy.stats", "scipy.fft"):
        try:
            mods += collect_submodules(pkg, filter=lambda name: ".tests" not in name and "._test" not in name)
        except Exception:  # noqa: BLE001
            pass
    # optional readers ([formats] extra) - imported lazily by lumen/core/readers
    mods += ["h5py", "zarr", "numcodecs", "nd2", "czifile", "readlif", "pydicom", "nibabel", "mrcfile", "openpyxl", "xlrd", "oiffile", "olefile"]
    mods += ["multipart", "python_multipart", "anyio._backends._asyncio", "h11", "websockets", "websockets.legacy", "wsproto",
             "email.mime.multipart", "email.mime.text", "matplotlib.backends.backend_agg", "pooch", "joblib"]
    # the platform modules of pywebview are imported lazily (webview.guilib) - name them explicitly
    if sys.platform == "darwin":
        mods += ["webview.platforms.cocoa", "objc", "AppKit", "Foundation", "WebKit", "Quartz", "UniformTypeIdentifiers", "PyObjCTools.AppHelper"]
    elif os.name == "nt":
        mods += ["webview.platforms.winforms", "webview.platforms.edgechromium", "webview.platforms.win32", "clr", "clr_loader", "pythonnet"]
    else:
        mods += ["webview.platforms.qt", "qtpy", "qtpy.QtCore", "qtpy.QtGui", "qtpy.QtWidgets", "qtpy.QtWebEngineWidgets", "qtpy.QtWebEngineCore",
                 "qtpy.QtWebChannel", "qtpy.QtNetwork", "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebChannel", "PyQt6.QtNetwork",
                 "PyQt6.QtPrintSupport", "PyQt6.sip"]
    return sorted(set(mods))


def excludes() -> list[str]:
    ex = ["tkinter", "_tkinter", "pytest", "_pytest", "IPython", "jupyter", "notebook", "numpy.distutils", "tests", "playwright",
          "torch", "tensorflow", "keras", "cellpose", "stardist", "jpype", "scyjava", "imageio_ffmpeg",
          "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt5agg",
          "matplotlib.backends.backend_gtk3agg", "matplotlib.backends.backend_wxagg", "matplotlib.backends.backend_macosx"]
    if sys.platform == "darwin":
        ex += ["webview.platforms.qt", "webview.platforms.gtk", "webview.platforms.winforms", "webview.platforms.edgechromium", "webview.platforms.cef",
               "webview.platforms.mshtml", "webview.platforms.android", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi", "clr"]
    elif os.name == "nt":
        ex += ["webview.platforms.qt", "webview.platforms.gtk", "webview.platforms.cocoa", "webview.platforms.cef", "webview.platforms.mshtml",
               "webview.platforms.android", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi", "objc", "AppKit", "Foundation", "WebKit"]
    else:
        ex += ["webview.platforms.gtk", "webview.platforms.cocoa", "webview.platforms.winforms", "webview.platforms.edgechromium", "webview.platforms.cef",
               "webview.platforms.mshtml", "webview.platforms.android", "PyQt5", "PySide2", "PySide6", "gi", "clr", "objc", "AppKit", "Foundation", "WebKit",
               "PyQt6.QtBluetooth", "PyQt6.QtDBus", "PyQt6.QtDesigner", "PyQt6.QtHelp", "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets", "PyQt6.QtNfc",
               "PyQt6.QtOpenGL", "PyQt6.QtOpenGLWidgets", "PyQt6.QtPdf", "PyQt6.QtPdfWidgets", "PyQt6.QtPositioning", "PyQt6.QtQml", "PyQt6.QtQuick",
               "PyQt6.QtQuick3D", "PyQt6.QtQuickWidgets", "PyQt6.QtRemoteObjects", "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtSql", "PyQt6.QtSvg",
               "PyQt6.QtSvgWidgets", "PyQt6.QtTest", "PyQt6.QtTextToSpeech", "PyQt6.QtWebSockets", "PyQt6.QtXml", "PyQt6.QtSpatialAudio", "PyQt6.Qt3DCore",
               "PyQt6.Qt3DRender", "PyQt6.QtCharts", "PyQt6.QtDataVisualization", "PyQt6.QtGraphs", "PyQt6.QtWebEngineQuick"]
    return ex


def module_collection_mode() -> dict[str, str]:
    return {"lumen": "py"}


def pathex() -> list[str]:
    return [str(ROOT), str(DESKTOP)]


def macos_document_types() -> list[dict]:
    out = []
    for name, utis, exts in DOCUMENT_TYPES:
        d: dict = {"CFBundleTypeName": name, "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate", "CFBundleTypeExtensions": exts}
        if utis:
            d["LSItemContentTypes"] = utis
        out.append(d)
    return out
