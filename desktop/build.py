#!/usr/bin/env python3
"""Build the Lumen desktop app for the OS this runs on (standard library only).

    python desktop/build.py                 frontend build (npm) -> build venv -> PyInstaller -> artifacts in desktop/dist/
    python desktop/build.py --skip-frontend reuse the existing lumen/static (the built UI)
    python desktop/build.py --no-package    stop after PyInstaller (no dmg / zip / tar.gz / AppImage)
    python desktop/build.py --recreate-venv rebuild desktop/.venv-build from scratch
    python desktop/build.py --python PATH   interpreter for the build venv (default: uv picks Python 3.14)

Artifacts (desktop/dist/):
  macOS    Lumen.app + Lumen-<version>-macos-<arch>.dmg (hdiutil; unsigned)
  Windows  Lumen/ folder + Lumen-<version>-windows-<arch>.zip
  Linux    Lumen/ folder + Lumen-<version>-linux-<arch>.tar.gz (+ Lumen-<version>-linux-<arch>.AppImage when
           appimagetool is on PATH or LUMEN_APPIMAGETOOL points at it)
Environment: LUMEN_BUILD_PYTHON_VERSION (default 3.14), LUMEN_BUILD_EXTRAS (default formats), LUMEN_APPIMAGETOOL.
The macOS app is ad-hoc signed by PyInstaller, not notarized: a copy built on this Mac opens normally; a downloaded
copy needs right-click ▸ Open once (see docs/DESKTOP.md)."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _buildenv as be  # noqa: E402

DIST = be.DESKTOP / "dist"
WORK = DIST / "_work"
SPECS = be.DESKTOP / "build"
APP = "Lumen"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def run(cmd: list[str], what: str, cwd: Path = be.DESKTOP, env: dict | None = None) -> None:
    be.run_step(cmd, cwd, what) if env is None else _run_env(cmd, what, cwd, env)


def _run_env(cmd: list[str], what: str, cwd: Path, env: dict) -> None:
    be.log(f"{what}: {' '.join(cmd)}")
    code = subprocess.call(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL)
    if code != 0:
        raise SystemExit(f"[lumen-desktop] {what} failed (exit code {code})")


# --------------------------------------------------------------------------- packaging per OS


def package_macos(version: str, arch: str) -> list[Path]:
    app = DIST / f"{APP}.app"
    if not app.is_dir():
        raise SystemExit(f"[lumen-desktop] {app} is missing after PyInstaller")
    folder = DIST / APP  # the bare one-folder build is redundant next to the .app
    if folder.is_dir():
        shutil.rmtree(folder)
    dmg = DIST / f"{APP}-{version}-macos-{arch}.dmg"
    staging = DIST / "_dmg"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shutil.copytree(app, staging / app.name, symlinks=True)
    os.symlink("/Applications", staging / "Applications")
    if dmg.exists():
        dmg.unlink()
    run(["hdiutil", "create", "-volname", APP, "-srcfolder", str(staging), "-ov", "-format", "UDZO", "-quiet", str(dmg)], "creating the DMG")
    shutil.rmtree(staging)
    return [app, dmg]


def package_windows(version: str, arch: str) -> list[Path]:
    folder = DIST / APP
    if not (folder / f"{APP}.exe").is_file():
        raise SystemExit(f"[lumen-desktop] {folder / (APP + '.exe')} is missing after PyInstaller")
    out = DIST / f"{APP}-{version}-windows-{arch}.zip"
    if out.exists():
        out.unlink()
    be.log(f"creating {out.name}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                zf.write(f, f"{APP}/{f.relative_to(folder).as_posix()}")
    return [folder, out]


def package_linux(version: str, arch: str) -> list[Path]:
    folder = DIST / APP
    if not (folder / APP).is_file():
        raise SystemExit(f"[lumen-desktop] {folder / APP} is missing after PyInstaller")
    out = DIST / f"{APP}-{version}-linux-{arch}.tar.gz"
    if out.exists():
        out.unlink()
    be.log(f"creating {out.name}")
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        tf.add(folder, arcname=APP)
    artifacts = [folder, out]
    appimage = build_appimage(folder, version, arch)
    if appimage:
        artifacts.append(appimage)
    return artifacts


def find_appimagetool() -> list[str] | None:
    cand = os.environ.get("LUMEN_APPIMAGETOOL") or shutil.which("appimagetool")
    if not cand:
        return None
    p = Path(cand)
    if not p.is_file():
        return None
    cmd = [str(p)]
    if p.suffix.lower() == ".appimage":
        cmd.append("--appimage-extract-and-run")  # no FUSE needed (CI containers)
    return cmd


def build_appimage(folder: Path, version: str, arch: str) -> Path | None:
    tool = find_appimagetool()
    if not tool:
        be.log("appimagetool not found (LUMEN_APPIMAGETOOL / PATH) - skipping the AppImage, the tar.gz is the Linux artifact")
        return None
    appdir = DIST / "AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    (appdir / "usr" / "bin").mkdir(parents=True)
    shutil.copytree(folder, appdir / "usr" / "bin" / APP, symlinks=True)
    apprun = appdir / "AppRun"
    apprun.write_text('#!/bin/sh\nHERE="$(dirname "$(readlink -f "$0")")"\nexec "$HERE/usr/bin/Lumen/Lumen" "$@"\n', encoding="utf-8")
    apprun.chmod(apprun.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (appdir / "lumen.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Lumen\nComment=Scientific image analysis (a Fiji re-imagining)\nExec=Lumen %F\nIcon=lumen\n"
        "Categories=Science;Graphics;Education;\nMimeType=image/tiff;image/png;image/jpeg;\nTerminal=false\n", encoding="utf-8")
    shutil.copy(be.DESKTOP / "assets" / "icon-256.png", appdir / "lumen.png")
    shutil.copy(be.DESKTOP / "assets" / "icon-256.png", appdir / ".DirIcon")
    out = DIST / f"{APP}-{version}-linux-{arch}.AppImage"
    if out.exists():
        out.unlink()
    env = dict(os.environ, ARCH=arch)
    code = subprocess.call([*tool, str(appdir), str(out)], cwd=str(DIST), env=env, stdin=subprocess.DEVNULL)
    shutil.rmtree(appdir, ignore_errors=True)
    if code != 0 or not out.is_file():
        be.log(f"warning: appimagetool failed (exit {code}) - continuing without an AppImage")
        return None
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Build the Lumen desktop app for this OS.")
    p.add_argument("--skip-frontend", action="store_true", help="do not run npm run build (use the existing lumen/static)")
    p.add_argument("--no-package", action="store_true", help="stop after PyInstaller")
    p.add_argument("--recreate-venv", action="store_true", help="rebuild desktop/.venv-build")
    p.add_argument("--python", help="interpreter for the build venv")
    p.add_argument("--keep-work", action="store_true", help="keep desktop/dist/_work (PyInstaller analysis output)")
    args = p.parse_args(argv)

    version = be.app_version()
    os_tag, arch = be.os_tag(), be.arch_tag()
    be.log(f"building {APP} {version} for {os_tag} ({arch}) with Python {platform.python_version()} driving the build")
    t0 = time.monotonic()

    be.ensure_frontend(build=not args.skip_frontend)
    py = be.ensure_build_venv(python=args.python, recreate=args.recreate_venv)
    spec = SPECS / f"{os_tag}.spec"
    if not spec.is_file():
        raise SystemExit(f"[lumen-desktop] no spec for {os_tag}: {spec}")

    DIST.mkdir(parents=True, exist_ok=True)
    for stale in (DIST / APP, DIST / f"{APP}.app", WORK):
        if stale.exists():
            shutil.rmtree(stale)
    WORK.mkdir(parents=True)
    env = be.desktop_env()
    env.pop("LUMEN_HOME", None)  # keep the build free of local state
    # bundle a snapshot of lumen/static: Vite replaces the hashed asset names on every build, and a frontend build
    # running concurrently would otherwise break PyInstaller half-way
    snapshot = WORK / "static"
    shutil.copytree(be.STATIC, snapshot)
    env["LUMEN_BUILD_STATIC"] = str(snapshot)
    # a private PyInstaller cache: the default (~/Library/Application Support/pyinstaller, %APPDATA%\pyinstaller, ~/.cache/pyinstaller)
    # is shared by every PyInstaller project on the machine and a concurrent build can corrupt its index
    cache = be.VENV / "pyinstaller-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["PYINSTALLER_CONFIG_DIR"] = str(cache)
    _run_env([str(py), "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "WARN", "--distpath", str(DIST), "--workpath", str(WORK), str(spec)],
             "running PyInstaller", be.DESKTOP, env)
    if not args.keep_work:
        shutil.rmtree(WORK, ignore_errors=True)

    artifacts: list[Path] = []
    if args.no_package:
        artifacts = [x for x in (DIST / f"{APP}.app", DIST / APP) if x.exists()]
    elif os_tag == "macos":
        artifacts = package_macos(version, arch)
    elif os_tag == "windows":
        artifacts = package_windows(version, arch)
    else:
        artifacts = package_linux(version, arch)

    be.log(f"done in {time.monotonic() - t0:.0f} s - artifacts in {DIST}:")
    for a in artifacts:
        size = dir_size(a) if a.is_dir() else a.stat().st_size
        be.log(f"  {a.name:48s} {human(size):>10s}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
