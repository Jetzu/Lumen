"""Install plugins from a file, a folder, a zip archive or a URL – Fiji's update sites, the manual way.

Lumen has no update-site server. What it has is a plugins folder (``~/.lumen/plugins``, ``LUMEN_PLUGINS``) that is
hot-reloaded, and this command, which gets a plugin *into* that folder from wherever it is published: a ``.py``
module with ``@op`` commands, a plain Python script, an ``.ijm`` / ``.txt`` ImageJ macro or macro-tool set, a
``.lut`` table, a ``.zip`` holding any of those, a folder of them, or an ``http(s)://`` URL of one.  Everything else
in an archive is ignored (compiled Java plugins run through the JVM bridge instead, see docs/SCRIPTING.md).

Installed Python and macro files execute with the server's permissions – install only what you trust.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from ..registry import OpResult, Param, op
from .file_extra import import_lut as _import_lut
from .plugins_ops import MACRO_EXTS, MACROS, UTIL, _NOIMG, install as _install_one, plugins_dir

INSTALL_EXTS = (".py", *MACRO_EXTS, ".lut")
MAX_BYTES = 50 * 1024 * 1024
_UA = "Lumen/0.1 (plugin install)"


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", Path(name.replace("\\", "/")).name).strip() or "download"


def download(url: str, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Fetch ``url`` into ``dest_dir`` (proxies from the environment, 50 MB cap). Returns the file written."""
    parsed = urllib.parse.urlparse(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / _safe_name(urllib.parse.unquote(parsed.path))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(target, "wb") as out:  # noqa: S310 (the user typed the URL)
        total = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError(f"{url}: more than {MAX_BYTES // (1024 * 1024)} MB – not a plugin file")
            out.write(chunk)
    return target


def unzip(zpath: Path, dest: Path) -> tuple[list[Path], list[str]]:
    """Extract the installable members of a zip (flattened, Windows-safe names; no absolute paths, no ``..``, no
    ``__MACOSX``). Returns (files written, members ignored)."""
    files: list[Path] = []
    ignored: list[str] = []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            if name.startswith("/") or ".." in name.split("/") or name.startswith("__MACOSX") or Path(name).name.startswith("."):
                ignored.append(name)
                continue
            if Path(name).suffix.lower() not in INSTALL_EXTS:
                ignored.append(name)
                continue
            if info.file_size > MAX_BYTES:
                ignored.append(name)
                continue
            target = dest / _safe_name(name)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            files.append(target)
    return files, ignored


def _collect(source: str, work: Path) -> tuple[list[Path], list[str], str]:
    """The files to install for ``source`` (downloaded / extracted into ``work``), the members ignored, and what the
    source was ("url", "zip", "folder", "file")."""
    s = source.strip()
    if re.match(r"^https?://", s, re.I) or s.lower().startswith("file://"):
        got = download(s, work)
        kind = "url"
    else:
        got = Path(s).expanduser()
        if not got.exists():
            raise FileNotFoundError(f"No such file or folder: {got}")
        kind = "folder" if got.is_dir() else "file"
    if got.is_dir():
        files = sorted(p for p in got.iterdir() if p.is_file() and p.suffix.lower() in INSTALL_EXTS)
        ignored = sorted(p.name for p in got.iterdir() if p.is_file() and p.suffix.lower() not in INSTALL_EXTS)
        return files, ignored, kind
    if got.suffix.lower() == ".zip" or zipfile.is_zipfile(got):
        files, ignored = unzip(got, work / "unzipped")
        return files, ignored, "zip" if kind != "url" else "url (zip)"
    if got.suffix.lower() not in INSTALL_EXTS:
        raise ValueError(f"{got.name}: only {', '.join(INSTALL_EXTS)} files, folders and .zip archives of them can be installed")
    return [got], [], kind


@op("plugins.install_from", "Install Plugin from File, Folder, Zip or URL…", UTIL, **_NOIMG,
    keywords=["install", "download", "update site", "zip", "url", "plugin", "lut", "macro", "fetch"],
    params=[Param("source", "string", "", help="A .py plugin module or script, an .ijm / .txt macro or tool set, a .lut table, a .zip holding any of these, a folder of them, or an http(s) URL of one"),
            Param("overwrite", "bool", True, help="Replace files of the same name in the plugins folder"),
            Param("menu", "string", MACROS, help="Menu for wrapped macros and scripts", advanced=True)],
    description="Fiji's update sites, the manual way: fetch a plugin from a URL (or pick a file, folder or zip), copy it into the plugins folder and register "
                "its commands right away (hot reload). Only .py modules / scripts, .ijm / .txt macros and tool sets and .lut tables are installed; anything "
                "else in an archive is listed as ignored. Installed Python and macro files execute with the server's permissions – install only what you trust.")
def install_from(ctx, source="", overwrite=True, menu=MACROS):
    if not str(source).strip():
        raise ValueError("Give a file, folder, zip or URL to install")
    rows: list[list[Any]] = []
    logs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lumen-install-") as tmp:
        work = Path(tmp)
        (work / "unzipped").mkdir()
        files, ignored, kind = _collect(str(source), work)
        if not files:
            raise ValueError(f"{source}: nothing installable found ({', '.join(INSTALL_EXTS)})" + (f"; ignored: {', '.join(ignored[:8])}" if ignored else ""))
        for f in files:
            ext = f.suffix.lower()
            if ext == ".lut":
                res = _import_lut(ctx, path=str(f), apply=False)
                dest = plugins_dir() / f.name
                if overwrite or not dest.exists():
                    shutil.copyfile(f, dest)  # keep the table next to the plugins so LUTs from Folder… finds it again
                rows.append([f.name, "LUT", str(dest), res.message or ""])
            else:
                res = _install_one(ctx, path=str(f), menu=menu, overwrite=overwrite)
                cmds = ", ".join(r[1] for r in (res.table or {}).get("rows", []))
                what = "plugin module" if "plugin module" in (res.message or "") else ("ImageJ macro" if ext in MACRO_EXTS else "Python script")
                dest = (res.message or "").split(" → ", 1)[-1].split(";", 1)[0] if " → " in (res.message or "") else str(plugins_dir() / f.name)
                rows.append([f.name, what, dest, cmds])
            if res.log:
                logs.append(res.log)
    for name in ignored:
        rows.append([name, "ignored", "", "not a .py / .ijm / .txt / .lut file"])
    n = sum(1 for r in rows if r[1] != "ignored")
    msg = f"Installed {n} file{'s' if n != 1 else ''} from {kind} {source}" + (f" ({len(ignored)} ignored)" if ignored else "")
    return OpResult(table={"name": "Installed", "columns": ["File", "Kind", "Destination", "Commands"], "rows": rows}, message=msg, log="\n".join(logs + [msg]))
