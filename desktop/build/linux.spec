# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Linux: one-folder build dist/Lumen/Lumen with the Qt WebEngine runtime inside
# (desktop/build.py packs it as tar.gz and, when appimagetool is available, as an AppImage).
import sys

sys.path.insert(0, SPECPATH)
import common  # noqa: E402

common.check_inputs()

a = Analysis(
    [str(common.ENTRY)],
    pathex=common.pathex(),
    binaries=[],
    datas=common.datas(),
    hiddenimports=common.hiddenimports(),
    hookspath=[],
    runtime_hooks=[],
    excludes=common.excludes(),
    module_collection_mode=common.module_collection_mode(),
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=common.APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # a terminal launch shows the log lines; a desktop launch has no console anyway
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=common.APP_NAME)
