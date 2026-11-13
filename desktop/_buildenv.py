"""Shared helper for ``desktop/build.py`` and ``desktop/run_dev.py``: the build virtual environment
(``desktop/.venv-build``) with Lumen's dependencies (``pyproject.toml``: the core plus the ``[formats]`` readers) and
the desktop extras for this OS (``desktop/requirements-<os>.txt``: pywebview, PyInstaller).  Standard library only;
uv is preferred, ``python -m venv`` + pip is the fallback.  Nothing here installs a browser or any other executable
outside the venv (pywebview uses the system's own web view: WebKit on macOS, WebView2 on Windows; Qt WebEngine
from the PyQt6 wheels on Linux)."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
FRONTEND = ROOT / "frontend"
STATIC = ROOT / "lumen" / "static"
VENV = DESKTOP / ".venv-build"
IS_WIN = os.name == "nt"
PYTHON_WANTED = os.environ.get("LUMEN_BUILD_PYTHON_VERSION", "3.14")
#: the optional extras bundled with the desktop app: the [formats] readers and the [mcp] SDK for ``Lumen --mcp`` (all wheels; the JVM / Bio-Formats bridges stay opt-in because
#: they need a JDK, and [video] because imageio-ffmpeg ships an unsigned ffmpeg binary)
EXTRAS = os.environ.get("LUMEN_BUILD_EXTRAS", "formats,mcp")


def log(msg: str) -> None:
    print(f"[lumen-desktop] {msg}", flush=True)


def os_tag() -> str:
    return {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system(), platform.system().lower())


def arch_tag() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(m, m)


def app_version() -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (ROOT / "lumen" / "__init__.py").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("__version__ not found in lumen/__init__.py")
    return m.group(1)


def venv_python(venv: Path = VENV) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def requirements_file() -> Path:
    return DESKTOP / f"requirements-{os_tag()}.txt"


def find_uv() -> Path | None:
    p = shutil.which("uv")
    return Path(p) if p else None


def run_step(cmd: list[str], cwd: Path, what: str, fatal: bool = True, env: dict | None = None) -> int:
    log(f"{what}: {' '.join(cmd)}")
    code = subprocess.call(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, env=env)
    if code != 0 and fatal:
        raise SystemExit(f"[lumen-desktop] {what} failed (exit code {code})")
    return code


def lumen_requirements() -> list[str]:
    """Lumen's runtime dependencies from pyproject.toml: ``[project].dependencies`` plus the extras in ``EXTRAS``."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = list(data["project"]["dependencies"])
    optional = data["project"].get("optional-dependencies", {})
    for extra in [e.strip() for e in EXTRAS.split(",") if e.strip()]:
        if extra not in optional:
            raise SystemExit(f"[lumen-desktop] unknown extra {extra!r} (pyproject.toml has {', '.join(sorted(optional))})")
        reqs += optional[extra]
    return reqs


_PROBE = "import numpy, scipy, skimage, sklearn, tifffile, imagecodecs, fastapi, uvicorn, multipart, webview, PyInstaller"


def _venv_ok(py: Path) -> bool:
    if not py.is_file():
        return False
    try:
        r = subprocess.run([str(py), "-c", _PROBE], capture_output=True, timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_build_venv(python: str | None = None, recreate: bool = False) -> Path:
    """Create / refresh ``desktop/.venv-build`` and return its interpreter.  Lumen itself is *not* installed into it:
    the spec files and ``run_dev.py`` import the package from this checkout, so a build always bundles the current
    source."""
    py = venv_python()
    if recreate and VENV.exists():
        shutil.rmtree(VENV)
    req = VENV / "lumen-requirements.txt"
    wanted = "\n".join(lumen_requirements()) + "\n"
    extras = requirements_file()
    stamp = VENV / "requirements.stamp"
    stamp_text = wanted + (extras.read_text(encoding="utf-8") if extras.is_file() else "")
    if _venv_ok(py) and not recreate and stamp.is_file() and stamp.read_text(encoding="utf-8") == stamp_text:
        return py  # up to date (a changed extra, e.g. the new [mcp], reinstalls below)
    uv = find_uv()
    if not VENV.exists():
        if uv:
            run_step([str(uv), "venv", "--python", python or PYTHON_WANTED, str(VENV)], DESKTOP, "creating desktop/.venv-build with uv")
        else:
            base = python or shutil.which(f"python{PYTHON_WANTED}") or sys.executable
            run_step([base, "-m", "venv", str(VENV)], DESKTOP, "creating desktop/.venv-build")
    req.write_text(wanted, encoding="utf-8")
    if not extras.is_file():
        raise SystemExit(f"[lumen-desktop] {extras} is missing")
    if uv:
        run_step([str(uv), "pip", "install", "--python", str(py), "-r", str(req), "-r", str(extras)], DESKTOP, "installing build dependencies with uv")
    else:
        run_step([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"], DESKTOP, "upgrading pip", fatal=False)
        run_step([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req), "-r", str(extras)], DESKTOP, "installing build dependencies with pip")
    if not _venv_ok(py):
        raise SystemExit("[lumen-desktop] desktop/.venv-build is still missing packages after the install step")
    stamp.write_text(stamp_text, encoding="utf-8")
    return py


def ensure_frontend(build: bool) -> Path:
    """``lumen/static`` (the built UI; ``npm run build`` in frontend/ writes it) - rebuilt when ``build`` or missing."""
    index = STATIC / "index.html"
    if index.is_file() and not build:
        return STATIC
    npm = shutil.which("npm.cmd" if IS_WIN else "npm")
    if not npm:
        raise SystemExit("[lumen-desktop] npm not found - install Node.js 20+ or pass --skip-frontend with an existing lumen/static build")
    if not (FRONTEND / "node_modules").is_dir():
        run_step([npm, "ci" if (FRONTEND / "package-lock.json").is_file() else "install"], FRONTEND, "installing frontend dependencies")
    run_step([npm, "run", "build"], FRONTEND, "building the frontend")
    if not index.is_file():
        raise SystemExit(f"[lumen-desktop] the frontend build did not produce {index}")
    return STATIC


def desktop_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(overrides)
    env.setdefault("LUMEN_DESKTOP", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(ROOT), str(DESKTOP), env.get("PYTHONPATH", "")] if p)
    return env
