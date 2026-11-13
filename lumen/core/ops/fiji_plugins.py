"""Fiji's plugin ecosystem in Lumen: update sites, installed plugin jars, and their commands.

Fiji distributes plugins through *update sites* - a folder on a web server with a ``db.xml.gz`` index and the files
(``plugins/Foo.jar-<timestamp>``).  This module reads such an index (``plugins.fiji_update_site``), installs chosen jars
with their dependencies into ``<lumen home>/jars`` (``plugins.fiji_install``), and turns every command a jar declares in
its ``plugins.config`` (``Plugins>Analyze, "My Plugin", my.pkg.My_Plugin("arg")``) into a Lumen command under
*Plugins ▸ Fiji ▸ …*.  Running one goes through the Java bridge (lumen/core/ijshim/jvm_subprocess.py): the real
ImageJ 1.x runs the real plugin class on the active image with ImageJ macro options, and the result comes back with an
undo step.  The installed jars are also on the bridge's classpath for Groovy / BeanShell / Rhino scripts, and the ImageJ
macro layer reaches the commands by name (``run("My Plugin", "radius=3")``).

What this needs: a JDK and Fiji's ``jars`` folder, as for every Java-bridge feature (docs/PLATFORMS.md).  What it cannot
do: plugins that need a window of their own (interactive Swing tools) or ImageJ2 / SciJava commands - the bridge runs
ImageJ 1.x in batch mode.  Installed jars are Java code and run with your permissions: install only what you trust.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..registry import REGISTRY, OpResult, OpSpec, Param, op

UTIL = "Plugins/Utilities"
FIJI_MENU = "Plugins/Fiji"
_UA = "Lumen/0.1 (Fiji update site client)"
MAX_BYTES = 200 * 1024 * 1024
#: well-known Fiji update sites (name → URL); any other URL works too
KNOWN_SITES = {
    "ImageJ": "https://update.imagej.net/",
    "Fiji": "https://update.fiji.sc/",
    "BIG-EPFL": "https://sites.imagej.net/BIG-EPFL/",
    "IJPB-plugins (MorphoLibJ)": "https://sites.imagej.net/IJPB-plugins/",
    "CSBDeep": "https://sites.imagej.net/CSBDeep/",
}

_CONFIG_RE = re.compile(r'^\s*([^,#"]+?)\s*,\s*"([^"]+)"\s*,\s*([\w.$]+)\s*(?:\(\s*"(.*)"\s*\))?\s*$')


def jars_dir() -> Path:
    from ..io import lumen_home
    return lumen_home() / "jars"


# ----------------------------------------------------------------------------- plugins.config


def plugin_commands(jar: Path) -> list[dict[str, str]]:
    """The commands a plugin jar declares: ``[{"menu", "label", "class", "arg"}]`` from its ``plugins.config``."""
    out: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(jar) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith("plugins.config")]
            if not names:
                return out
            text = zf.read(names[0]).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile):
        return out
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _CONFIG_RE.match(line)
        if m:
            out.append({"menu": m.group(1).strip(), "label": m.group(2).strip(), "class": m.group(3).strip(), "arg": (m.group(4) or "").strip()})
    return out


# ----------------------------------------------------------------------------- update sites


def _open(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - the user typed the update-site URL


def _base(url: str) -> str:
    url = url.strip()
    if url in KNOWN_SITES:
        url = KNOWN_SITES[url]
    if not re.match(r"^(https?|file)://", url, re.I):
        p = Path(url).expanduser()
        if p.exists():
            url = p.resolve().as_uri()
    return url if url.endswith("/") else url + "/"


def read_update_site(url: str) -> list[dict[str, Any]]:
    """Parse ``<url>/db.xml.gz``: ``[{"filename", "timestamp", "filesize", "description", "dependencies"}]`` for the
    current version of every file on the site."""
    base = _base(url)
    with _open(base + "db.xml.gz") as resp:
        raw = resp.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{base}db.xml.gz is larger than {MAX_BYTES // 1048576} MB")
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # served uncompressed
    root = ET.fromstring(raw)
    out: list[dict[str, Any]] = []
    for plugin in root.iter("plugin"):
        ver = plugin.find("version")
        if ver is None:
            continue  # only previous versions: the file was removed from the site
        desc = (plugin.findtext("description") or ver.findtext("description") or "").strip()
        out.append({
            "filename": plugin.get("filename", ""),
            "timestamp": ver.get("timestamp", ""),
            "filesize": int(ver.get("filesize") or 0),
            "description": re.sub(r"\s+", " ", desc)[:300],
            "dependencies": [d.get("filename", "") for d in ver.iter("dependency") if d.get("filename")],
        })
    return out


def _installed_or_bundled(filename: str) -> bool:
    """A dependency that Fiji's own ``jars`` folder or Lumen's jars folder already has (by name without version)."""
    from ..readers.fiji_jars import find_fiji_jars
    stem = re.sub(r"-[\d.]+(-SNAPSHOT)?$", "", Path(filename).stem)
    for folder in (jars_dir(), find_fiji_jars()):
        if folder and Path(folder).is_dir() and any(re.sub(r"-[\d.]+(-SNAPSHOT)?$", "", p.stem) == stem for p in Path(folder).rglob("*.jar")):
            return True
    return False


def install_from_site(url: str, files: list[str], with_dependencies: bool = True) -> list[dict[str, str]]:
    """Download ``files`` (filenames as listed by the site, e.g. ``plugins/Foo.jar``) and, optionally, their missing
    dependencies into ``jars_dir()``.  Returns one row per file: ``{"file", "status", "path"}``."""
    base = _base(url)
    records = {r["filename"]: r for r in read_update_site(base)}
    wanted: list[str] = []
    queue = list(files)
    while queue:
        f = queue.pop(0)
        if f in wanted:
            continue
        if f not in records:
            raise ValueError(f"{f}: not on the update site {base} (see the list from Fiji Update Site…)")
        wanted.append(f)
        if with_dependencies:
            queue += [d for d in records[f]["dependencies"] if d in records and d not in wanted and not _installed_or_bundled(d)]
    dest = jars_dir()
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for f in wanted:
        rec = records[f]
        if not f.lower().endswith(".jar"):
            rows.append({"file": f, "status": "skipped (not a jar - macros and scripts install with Install Plugin from File, Folder, Zip or URL…)", "path": ""})
            continue
        src = base + urllib.parse.quote(f) + "-" + rec["timestamp"]
        target = dest / Path(f).name
        with tempfile.NamedTemporaryFile(dir=dest, suffix=".part", delete=False) as tmp:
            with _open(src, timeout=300) as resp:
                total = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError(f"{f}: larger than {MAX_BYTES // 1048576} MB")
                    tmp.write(chunk)
        tmp_path = Path(tmp.name)
        if rec["filesize"] and total != rec["filesize"]:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"{f}: downloaded {total} bytes, the site lists {rec['filesize']}")
        try:
            zipfile.ZipFile(tmp_path).close()
        except zipfile.BadZipFile:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"{f}: the download is not a jar file") from None
        shutil.move(str(tmp_path), target)
        rows.append({"file": f, "status": "installed" + (" (dependency)" if f not in files else ""), "path": str(target)})
    return rows


# ----------------------------------------------------------------------------- commands of installed jars

_REGISTERED: set[str] = set()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "plugin"


def groovy_runner(cls: str, arg: str, options: str) -> str:
    """The Groovy code the bridge runs for one plugin command: ImageJ reads GenericDialog values from the macro options
    only on a thread whose name starts with ``Run$_`` - as when a macro calls run(); a blank options string makes every
    dialog take its defaults instead of waiting for a click."""
    return (
        "import ij.Macro\n"
        f"def __opts = {json.dumps((options or '').strip() + ' ')}\n"
        f"def __t = new Thread({{ Macro.setOptions(__opts); IJ.runPlugIn(imp, {json.dumps(cls)}, {json.dumps(arg)}) }} as Runnable, 'Run$_lumen')\n"
        "__t.start(); __t.join()\n"
    )


def run_plugin_class(ctx, cls: str, arg: str = "", options: str = "") -> OpResult:
    from ..ijshim.jvm_bridge import run_jvm_script
    out = run_jvm_script(ctx.session, groovy_runner(cls, arg, options), "groovy", roi=ctx.roi)
    if out.get("error"):
        raise RuntimeError(f"{cls}: {out['error']}")
    text = (out.get("stdout") or "").strip()
    return OpResult(message=f"Ran the Fiji plugin {cls}" + (f" – {text.splitlines()[-1][:160]}" if text else ""), log=text or None)


def _register(cmd: dict[str, str], jar: Path) -> str | None:
    oid = f"fiji.{_slug(Path(jar).stem.split('-')[0])}.{_slug(cmd['label'])}"
    if oid in REGISTRY.ops:
        return None
    cls, arg = cmd["class"], cmd["arg"]

    def fn(ctx, options="", _cls=cls, _arg=arg):
        return run_plugin_class(ctx, _cls, _arg, options)

    fn.fiji_plugin = cls  # type: ignore[attr-defined]  (the macro layer passes run("Label", "options") through verbatim)
    taken = {s.name.lower() for k, s in REGISTRY.ops.items() if not k.startswith("fiji.")}
    name = cmd["label"] if cmd["label"].lower() not in taken else f"{cmd['label']} (Fiji plugin)"
    menu = FIJI_MENU + "/" + "/".join(p.strip() for p in cmd["menu"].split(">") if p.strip() and p.strip() != "Plugins")
    spec = OpSpec(id=oid, name=name, category=menu.rstrip("/"), func=fn, kind="image", preview=False, roi_aware=True, requires_image=False,
                  scope_param=False, menu=menu.rstrip("/"),
                  params=[Param("options", "string", "", help="ImageJ macro options for the plugin's dialog, as a recorded macro writes them (\"radius=3 method=Otsu\"); empty = the dialog's defaults")],
                  description=f"The Fiji plugin {cls} from {Path(jar).name} (Fiji menu {cmd['menu']}), run with the real ImageJ 1.x through the Java bridge on the active image.",
                  keywords=["fiji", "plugin", "imagej", Path(jar).stem, cls.split(".")[-1]])
    fn.spec = spec  # type: ignore[attr-defined]
    REGISTRY.register(spec)
    _REGISTERED.add(oid)
    return oid


def register_installed() -> list[str]:
    """(Re-)register the commands of every jar in ``jars_dir()``; returns the command ids now registered."""
    for oid in list(_REGISTERED):
        REGISTRY.ops.pop(oid, None)
    _REGISTERED.clear()
    ids: list[str] = []
    d = jars_dir()
    if d.is_dir():
        for jar in sorted(d.glob("*.jar")):
            for cmd in plugin_commands(jar):
                oid = _register(cmd, jar)
                if oid:
                    ids.append(oid)
    return ids


# ----------------------------------------------------------------------------- commands

_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)


@op("plugins.fiji_update_site", "Fiji Update Site…", UTIL, **_NOIMG,
    keywords=["update site", "fiji", "plugins", "install", "updater", "jar", "ecosystem"],
    params=[Param("site", "string", "ImageJ", help="An update-site URL or folder, or one of: " + ", ".join(KNOWN_SITES)),
            Param("filter", "string", "", help="Only files whose name or description contains this text")],
    description="List the files of a Fiji update site (its db.xml.gz): name, version date, size and description. Install the ones you want "
                "with Install from Fiji Update Site…; their commands then appear under Plugins ▸ Fiji.")
def fiji_update_site(ctx, site="ImageJ", filter=""):  # noqa: A002 - the parameter name users see
    recs = read_update_site(site)
    q = (filter or "").strip().lower()
    rows = [[r["filename"], r["timestamp"][:8], f"{r['filesize'] / 1024:.0f} KB", r["description"]] for r in recs
            if not q or q in r["filename"].lower() or q in r["description"].lower()]
    return OpResult(table={"name": "Fiji Update Site", "columns": ["File", "Version", "Size", "Description"], "rows": rows},
                    message=f"{len(rows)} of {len(recs)} files on {_base(site)}")


@op("plugins.fiji_install", "Install from Fiji Update Site…", UTIL, **_NOIMG,
    keywords=["update site", "fiji", "install", "plugin", "jar", "download"],
    params=[Param("site", "string", "ImageJ", help="The update-site URL or folder (or a known site name)"),
            Param("files", "string", "", help="Comma-separated file names as the site lists them, e.g. plugins/Auto_Threshold.jar"),
            Param("with_dependencies", "bool", True, help="Also install the dependencies the site lists that neither Fiji nor Lumen has")],
    description="Download plugin jars (and their dependencies) from a Fiji update site into the Lumen folder's jars/ and register their "
                "commands under Plugins ▸ Fiji. They run with the real ImageJ through the Java bridge (needs a JDK and Fiji's jars). "
                "Plugin jars are Java code and run with your permissions – install only what you trust.")
def fiji_install(ctx, site="ImageJ", files="", with_dependencies=True):
    names = [f.strip() for f in str(files).split(",") if f.strip()]
    if not names:
        raise ValueError("Name at least one file (see Fiji Update Site… for the list)")
    rows = install_from_site(site, names, with_dependencies=bool(with_dependencies))
    ids = register_installed()
    from .plugins_ops import _ops_changed
    _ops_changed(ctx.session)
    n_cmd = sum(1 for oid in ids if any(Path(r["path"]).stem.split("-")[0].lower().replace("-", "_") in oid for r in rows if r["path"]))
    return OpResult(table={"name": "Installed", "columns": ["File", "Status", "Path"], "rows": [[r["file"], r["status"], r["path"]] for r in rows]},
                    message=f"Installed {sum(r['status'].startswith('installed') for r in rows)} file(s); {n_cmd} command(s) under Plugins ▸ Fiji")


@op("plugins.fiji_installed", "Installed Fiji Plugins", UTIL, **_NOIMG, keywords=["fiji", "plugins", "installed", "jars"],
    description="The plugin jars in the Lumen folder's jars/ and the commands each declares (plugins.config).")
def fiji_installed(ctx):
    rows = []
    d = jars_dir()
    for jar in sorted(d.glob("*.jar")) if d.is_dir() else []:
        cmds = plugin_commands(jar)
        rows += [[jar.name, c["label"], c["menu"], c["class"]] for c in cmds] or [[jar.name, "–", "–", "(no plugins.config: a library jar)"]]
    return OpResult(table={"name": "Installed Fiji Plugins", "columns": ["Jar", "Command", "Fiji menu", "Class"], "rows": rows},
                    message=f"{len(rows)} command(s) in {d}")


@op("plugins.run_fiji_plugin", "Run Fiji Plugin Class…", UTIL, kind="image", requires_image=False, preview=False, roi_aware=True, scope_param=False,
    keywords=["fiji", "plugin", "imagej", "class", "run", "jar"],
    params=[Param("plugin_class", "string", "", help="The ImageJ 1.x plugin class, e.g. fiji.threshold.Auto_Threshold"),
            Param("arg", "string", "", help="The argument plugins.config passes (usually empty)"),
            Param("options", "string", "", help="ImageJ macro options for the plugin's dialog (\"method=Otsu white\")")],
    description="Run any ImageJ 1.x plugin class on the classpath of the Java bridge (Fiji's plugins/jars, the Lumen folder's jars/, LUMEN_JVM_CLASSPATH) "
                "on the active image with macro options – the real plugin, through the real ImageJ.")
def run_fiji_plugin(ctx, plugin_class="", arg="", options=""):
    if not str(plugin_class).strip():
        raise ValueError("Name the plugin class (see Installed Fiji Plugins)")
    return run_plugin_class(ctx, str(plugin_class).strip(), str(arg), str(options))

