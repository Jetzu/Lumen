# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for macOS: one-folder build wrapped into Lumen.app (desktop/build.py adds the DMG).
# Ad-hoc signed (PyInstaller does that on arm64), not notarized - see docs/DESKTOP.md.
import sys
from pathlib import Path

sys.path.insert(0, SPECPATH)
import common  # noqa: E402

common.check_inputs()
version = common.app_version()

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
    console=False,
    disable_windowed_traceback=False,
    # off: argv emulation made every `Lumen --mcp` helper (started by AI assistants) check in with LaunchServices as
    # the running app, so the Dock icon activated a windowless helper; the launching 'odoc' event is caught by
    # lumen_desktop/macos.py install_launch_open_handler instead
    argv_emulation=False,
    target_arch=None,  # native (arm64 on Apple silicon runners, x86_64 on Intel runners)
    codesign_identity=None,
    entitlements_file=None,
    icon=str(common.ASSETS / "icon.icns"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=common.APP_NAME)
app = BUNDLE(
    coll,
    name=f"{common.APP_NAME}.app",
    icon=str(common.ASSETS / "icon.icns"),
    bundle_identifier=common.BUNDLE_ID,
    info_plist={
        "CFBundleName": common.APP_NAME,
        "CFBundleDisplayName": common.APP_NAME,
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Lumen – scientific image analysis",
        "LSApplicationCategoryType": "public.app-category.graphics-design",
        "CFBundleDocumentTypes": common.macos_document_types(),
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True, "NSExceptionDomains": {"127.0.0.1": {"NSExceptionAllowsInsecureHTTPLoads": True}, "localhost": {"NSExceptionAllowsInsecureHTTPLoads": True}}},
        "NSSupportsAutomaticGraphicsSwitching": True,
    },
)
