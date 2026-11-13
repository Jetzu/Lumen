"""Shared discovery of a JDK and the jars of a Fiji installation for Lumen's *out-of-process* Java bridges.

Two bridges run ``java`` as a subprocess with jars taken (read-only) from a Fiji installation – no Python
package, no JPype / in-process JVM, no Maven download:

* the JVM script runner (:mod:`lumen.core.ijshim.jvm_subprocess`): Groovy / BeanShell / Rhino (JRuby / Clojure
  when their jars are complete) with the real ``ij.jar``;
* the Bio-Formats reader (:mod:`lumen.core.readers.bioformats_bridge`, mode ``fiji-jars``): ``loci.formats``
  through a small Groovy / BeanShell runner (Fiji ships no ``bioformats_tools``).

Where things are looked for

* **Jars**: ``LUMEN_FIJI_JARS`` (a Fiji ``jars`` folder, or the Fiji application folder – its ``jars`` is used;
  an invalid value means *no* Fiji, nothing else is searched), else the usual places: ``/Applications/Fiji/jars``,
  ``/Applications/Fiji.app/jars``, ``~/Applications/Fiji(.app)/jars``, ``~/Fiji(.app)/jars``, ``/opt/fiji/jars``,
  ``/opt/Fiji.app/jars``, ``C:\\Fiji.app\\jars``, ``C:\\Fiji\\jars``, ``%USERPROFILE%\\Fiji.app\\jars``,
  ``%USERPROFILE%\\Desktop\\Fiji.app\\jars`` ….  ``LUMEN_JVM_CLASSPATH`` (``os.pathsep``-separated jars, folders –
  their ``*.jar`` – or globs) and ``LUMEN_IJ_JAR`` add jars that take precedence over Fiji's, so an ImageJ-only
  setup (``ij.jar`` + ``groovy-4.x.jar`` on ``LUMEN_JVM_CLASSPATH``) works without Fiji.
* **JDK**: ``JAVA_HOME``, ``java`` on ``PATH``, Homebrew ``openjdk`` kegs, the JDKs under
  ``/Library/Java/JavaVirtualMachines`` (Temurin / Zulu / Oracle .pkg installs), ``/usr/lib/jvm``,
  ``C:\\Program Files\\Eclipse Adoptium`` / ``Java`` / ``Microsoft`` / ``Zulu`` / ``OpenJDK``, and – on Linux /
  Windows – the JDK bundled with Fiji (``<Fiji>/java/…``; on macOS that one is left alone: Gatekeeper).  A
  candidate counts once ``java -version`` succeeds (macOS's ``/usr/bin/java`` stub without a JDK does not).

Everything is cached per process (:func:`reset` forgets); :func:`run_java` is the one place a JVM is started
from: stdin closed, output captured as UTF-8, a timeout, no console window on Windows.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_JARS = "LUMEN_FIJI_JARS"
ENV_CLASSPATH = "LUMEN_JVM_CLASSPATH"
ENV_IJ_JAR = "LUMEN_IJ_JAR"
ENV_HEAP = "LUMEN_JVM_HEAP"

#: component -> regex on the jar file name (the newest version wins when several match)
JAR_PATTERNS: dict[str, str] = {
    "ij": r"^(ij-\d[\w.\-]*|ij)\.jar$",
    "groovy": r"^groovy-\d[\w.\-]*\.jar$",
    "beanshell": r"^bsh-\d[\w.\-]*\.jar$",
    "rhino": r"^rhino-\d[\w.\-]*\.jar$",
    "jruby": r"^jruby-(complete|core)-\d[\w.\-]*\.jar$",
    "jruby-stdlib": r"^jruby-stdlib-\d[\w.\-]*\.jar$",
    "clojure": r"^clojure-\d[\w.\-]*\.jar$",
    "clojure-spec": r"^spec\.alpha-\d[\w.\-]*\.jar$",
    "clojure-core-specs": r"^core\.specs\.alpha-\d[\w.\-]*\.jar$",
    "bioformats": r"^(formats-gpl-\d[\w.\-]*|bioformats_package[\w.\-]*|bio-formats_package[\w.\-]*)\.jar$",
}
#: sub-folders of a Fiji ``jars`` folder that are searched too (Fiji keeps Bio-Formats in ``jars/bio-formats``)
JAR_SUBFOLDERS = ("bio-formats",)

HOMEBREW_GLOBS = ("/opt/homebrew/opt/openjdk*/libexec/openjdk.jdk/Contents/Home", "/usr/local/opt/openjdk*/libexec/openjdk.jdk/Contents/Home",
                  "/home/linuxbrew/.linuxbrew/opt/openjdk*/libexec", "/home/linuxbrew/.linuxbrew/opt/openjdk*")
JDK_GLOBS = ("/Library/Java/JavaVirtualMachines/*/Contents/Home", "~/Library/Java/JavaVirtualMachines/*/Contents/Home", "/usr/lib/jvm/*",
             "C:/Program Files/Eclipse Adoptium/jdk-*", "C:/Program Files/Eclipse Adoptium/jre-*", "C:/Program Files/Java/jdk-*",
             "C:/Program Files/Microsoft/jdk-*", "C:/Program Files/Zulu/zulu-*", "C:/Program Files/OpenJDK/*", "C:/Program Files/Amazon Corretto/jdk*",
             "C:/Program Files/BellSoft/*")
FIJI_CANDIDATES = ("/Applications/Fiji/jars", "/Applications/Fiji.app/jars", "~/Applications/Fiji/jars", "~/Applications/Fiji.app/jars", "~/Fiji.app/jars",
                   "~/Fiji/jars", "~/Desktop/Fiji.app/jars", "/opt/fiji/jars", "/opt/Fiji.app/jars", "/opt/Fiji/jars", "/usr/local/Fiji.app/jars",
                   "C:/Fiji.app/jars", "C:/Fiji/jars", "~/Fiji.app/jars", "~/Desktop/Fiji.app/jars", "~/Downloads/Fiji.app/jars",
                   "C:/Program Files/Fiji.app/jars", "C:/Program Files/Fiji/jars")
JDK_HINT = ("install a JDK (Eclipse Temurin / OpenJDK 11+: set JAVA_HOME, put java on PATH, or `brew install openjdk` / "
            "`winget install EclipseAdoptium.Temurin.21.JDK` / `apt install openjdk-21-jdk`)")
JARS_HINT = ("install Fiji (https://fiji.sc, any folder; set LUMEN_FIJI_JARS when it is not in a standard place) or point "
             "LUMEN_JVM_CLASSPATH at ij.jar plus the language / Bio-Formats jars")

_WINDOWS = os.name == "nt"
_JAVA_NAME = "java.exe" if _WINDOWS else "java"

_cache: dict[str, Any] = {}
_probe_cache: dict[str, str | None] = {}


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", name))


# ---------------------------------------------------------------------------- jars
def expand_classpath(raw: str) -> list[str]:
    """``os.pathsep``-separated entries: a glob is expanded, a folder means its ``*.jar``, ``~`` is expanded."""
    out: list[str] = []
    for part in (raw.split(os.pathsep) if raw else []):
        part = os.path.expanduser(part.strip())
        if not part:
            continue
        if any(ch in part for ch in "*?["):
            out.extend(sorted(glob.glob(part)))
        elif os.path.isdir(part):
            out.extend(sorted(glob.glob(os.path.join(part, "*.jar"))))
        else:
            out.append(part)
    return out


def extra_classpath() -> list[str]:
    """The user's jars: ``LUMEN_IJ_JAR`` first, then ``LUMEN_JVM_CLASSPATH`` (expanded), then the plugin jars installed
    from Fiji update sites into ``<lumen home>/jars`` (lumen/core/ops/fiji_plugins.py)."""
    out: list[str] = []
    ij = os.environ.get(ENV_IJ_JAR, "").strip()
    if ij:
        out.append(os.path.expanduser(ij))
    out.extend(expand_classpath(os.environ.get(ENV_CLASSPATH, "").strip()))
    try:
        from ..io import lumen_home
        installed = lumen_home() / "jars"
        if installed.is_dir():
            out.extend(sorted(str(j) for j in installed.glob("*.jar")))
    except Exception:  # noqa: BLE001 - never block the bridge on this
        pass
    return out


def fiji_jar_candidates() -> list[Path]:
    return [Path(os.path.expanduser(p)) for p in FIJI_CANDIDATES]


def _as_jars_dir(p: Path) -> Path | None:
    """``p`` itself when it holds jars, its ``jars`` sub-folder when ``p`` is a Fiji folder, else ``None``."""
    if p.is_dir():
        if p.name.lower() == "jars" or any(p.glob("*.jar")):
            return p
        sub = p / "jars"
        if sub.is_dir():
            return sub
    return None


def find_fiji_jars() -> Path | None:
    """The Fiji ``jars`` folder (``LUMEN_FIJI_JARS`` wins – and an invalid value means none), or ``None``."""
    env = os.environ.get(ENV_JARS, "").strip()
    if env:
        return _as_jars_dir(Path(os.path.expanduser(env)))
    for cand in fiji_jar_candidates():
        d = _as_jars_dir(cand)
        if d is not None and any(d.glob("*.jar")):
            return d
    return None


def fiji_home(jars_dir: Path | None) -> Path | None:
    """The Fiji folder of a ``jars`` folder (``/Applications/Fiji/jars`` → ``/Applications/Fiji``)."""
    return jars_dir.parent if jars_dir is not None else None


def _pick(paths: list[str], pattern: str) -> str | None:
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [p for p in paths if rx.match(Path(p).name)]
    if not hits:
        return None
    hits.sort(key=lambda p: _version_key(Path(p).name), reverse=True)
    return hits[0]


def find_jar(component: str, jars_dir: Path | None = None, extras: list[str] | None = None) -> str | None:
    """The jar of a component (``"ij"``, ``"groovy"``, ``"bioformats"`` …): the user's jars first
    (``LUMEN_IJ_JAR`` / ``LUMEN_JVM_CLASSPATH``), then Fiji's ``jars`` folder and its sub-folders."""
    pattern = JAR_PATTERNS[component]
    hit = _pick(list(extra_classpath() if extras is None else extras), pattern)
    if hit:
        return hit
    if jars_dir is None:
        jars_dir = find_fiji_jars()
    if jars_dir is None:
        return None
    folders = [jars_dir] + [jars_dir / s for s in JAR_SUBFOLDERS if (jars_dir / s).is_dir()]
    for folder in folders:
        hit = _pick([str(p) for p in folder.glob("*.jar")], pattern)
        if hit:
            return hit
    return None


def fiji_wildcards(jars_dir: Path | None) -> list[str]:
    """``<jars>/*`` and ``<jars>/bio-formats/*`` – the ``java`` launcher expands these itself (every platform), so
    Bio-Formats' and JRuby's many dependencies need no listing and the command line stays short."""
    if jars_dir is None:
        return []
    out = [str(jars_dir / "*")]
    out += [str(jars_dir / s / "*") for s in JAR_SUBFOLDERS if (jars_dir / s).is_dir()]
    return out


def classpath_for(components: list[str], *, include_fiji: bool = True, jars_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """``(classpath entries, missing components)``: the component jars in order, the user's extra entries, then
    (``include_fiji``) the Fiji wildcards.  Components whose jar is not found are listed in ``missing``."""
    if jars_dir is None:
        jars_dir = find_fiji_jars()
    extras = extra_classpath()
    entries: list[str] = []
    missing: list[str] = []
    for c in components:
        jar = find_jar(c, jars_dir, extras)
        if jar:
            entries.append(jar)
        else:
            missing.append(c)
    entries.extend(extras)
    if include_fiji:
        entries.extend(fiji_wildcards(jars_dir))
    seen: set[str] = set()
    out: list[str] = []
    for e in entries:
        key = os.path.normcase(os.path.abspath(e))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out, missing


# ---------------------------------------------------------------------------- the JDK
def _java_exe(home: str | os.PathLike) -> Path:
    return Path(home) / "bin" / _JAVA_NAME


def java_candidates() -> list[tuple[str, str]]:
    """``(java executable, how it was found)`` in precedence order (existence checked, not yet probed)."""
    out: list[tuple[str, str]] = []
    home = os.environ.get("JAVA_HOME", "").strip()
    if home and _java_exe(home).is_file():
        out.append((str(_java_exe(home)), f"JAVA_HOME={home}"))
    exe = shutil.which("java")
    if exe:
        out.append((exe, f"java on PATH ({exe})"))
    for pattern in HOMEBREW_GLOBS:
        for cand in sorted(glob.glob(os.path.expanduser(pattern)), key=_version_key, reverse=True):
            if _java_exe(cand).is_file():
                out.append((str(_java_exe(cand)), f"Homebrew OpenJDK at {cand}"))
    for pattern in JDK_GLOBS:
        for cand in sorted(glob.glob(os.path.expanduser(pattern)), key=_version_key, reverse=True):
            if _java_exe(cand).is_file():
                out.append((str(_java_exe(cand)), f"JDK at {cand}"))
    if sys.platform != "darwin":
        home_dir = fiji_home(find_fiji_jars())
        if home_dir is not None:
            for cand in sorted(glob.glob(str(home_dir / "java" / "*" / "*")), key=_version_key, reverse=True):
                for sub in ("", "jre"):
                    exe_p = _java_exe(Path(cand) / sub) if sub else _java_exe(cand)
                    if exe_p.is_file():
                        out.append((str(exe_p), f"Fiji's bundled JDK at {cand}"))
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for exe, how in out:
        key = os.path.normcase(os.path.abspath(exe))
        if key not in seen:
            seen.add(key)
            uniq.append((exe, how))
    return uniq


def _probe_java(exe: str) -> str | None:
    """The first line of ``java -version`` (cached), ``None`` when it cannot run (a macOS stub, a broken install)."""
    if exe in _probe_cache:
        return _probe_cache[exe]
    try:
        p = subprocess.run([exe, "-version"], capture_output=True, encoding="utf-8", errors="replace", timeout=60, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        text = (p.stderr or p.stdout or "").strip()
        ver = text.splitlines()[0].strip() if text else ""
        _probe_cache[exe] = ver if p.returncode == 0 and ("version" in ver.lower() or ver) else None
    except (OSError, subprocess.SubprocessError):
        _probe_cache[exe] = None
    return _probe_cache[exe]


def find_java() -> tuple[str | None, str, str]:
    """``(java executable or None, how it was found / why not, version line)`` – cached."""
    if "java" in _cache:
        return _cache["java"]
    tried: list[str] = []
    result: tuple[str | None, str, str] = (None, "", "")
    for exe, how in java_candidates():
        ver = _probe_java(exe)
        if ver:
            result = (exe, how, ver)
            break
        tried.append(exe)
    if result[0] is None:
        why = "no JDK found (JAVA_HOME unset, no java on PATH, no Homebrew / Temurin JDK)"
        if tried:
            why = f"no working JDK ({', '.join(tried)} cannot run)"
        result = (None, why, "")
    _cache["java"] = result
    return result


# ---------------------------------------------------------------------------- status
def status(refresh: bool = False) -> dict[str, Any]:
    """``{"java", "java_detail", "java_version", "jars", "jars_detail", "found": {component: jar}, "extra": [...], "install"}``."""
    if refresh:
        reset()
    if "status" in _cache:
        return dict(_cache["status"])
    exe, how, ver = find_java()
    jars = find_fiji_jars()
    env = os.environ.get(ENV_JARS, "").strip()
    if jars is not None:
        jars_detail = f"Fiji jars at {jars}" + (f" ({ENV_JARS})" if env else "")
    elif env:
        jars_detail = f"{ENV_JARS}={env} is not a Fiji jars folder"
    else:
        jars_detail = "no Fiji installation found (/Applications/Fiji, ~/Fiji.app, /opt/fiji, C:\\Fiji.app …)"
    found = {c: find_jar(c, jars) for c in JAR_PATTERNS}
    st = {"java": exe, "java_detail": how if exe else how, "java_version": ver, "jars": str(jars) if jars else None, "jars_detail": jars_detail,
          "found": found, "extra": extra_classpath(), "install": f"{JARS_HINT}; {JDK_HINT}"}
    _cache["status"] = st
    return dict(st)


def reset() -> None:
    """Forget the cached discovery (after installing a JDK / Fiji or changing the environment)."""
    _cache.clear()
    _probe_cache.clear()


# ---------------------------------------------------------------------------- running java
@dataclass
class JavaResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    error: str | None = None  # the launch failure (missing executable …), if any
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


def heap_option(nbytes: int | None = None) -> str:
    """``-Xmx…``: ``LUMEN_JVM_HEAP`` when set, else 1 GiB or three times the image (+ 512 MiB), at most 8 GiB."""
    heap = os.environ.get(ENV_HEAP, "").strip()
    if heap:
        return f"-Xmx{heap}"
    need = 1024
    if nbytes:
        need = max(need, min(8192, (3 * int(nbytes)) // (1024 * 1024) + 512))
    return f"-Xmx{need}m"


def base_options(*, headless: bool, home: Path | None = None, heap: str | None = None) -> list[str]:
    """The JVM options every bridge run uses: no Dock icon, UTF-8 output, optional headless AWT, an isolated
    ``user.home`` / ``ij.dir`` / ``plugins.dir`` (ImageJ then neither reads the user's IJ_Prefs nor scans Fiji's
    plugins folder) and the heap."""
    opts = ["-Dapple.awt.UIElement=true", "-Dfile.encoding=UTF-8", "-Dsun.stdout.encoding=UTF-8", "-Dsun.stderr.encoding=UTF-8",
            "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8", "-Djava.awt.headless=" + ("true" if headless else "false")]
    if home is not None:
        home = Path(home)
        plugins = home / "plugins"
        plugins.mkdir(parents=True, exist_ok=True)
        opts += [f"-Duser.home={home}", f"-Dij.dir={home}", f"-Dplugins.dir={plugins}"]
    if heap:
        opts.append(heap)
    return opts


def logback_option(folder: Path) -> str:
    """A quiet logback configuration (WARN and up, to stderr) for Bio-Formats' SLF4J logging – Fiji ships logback,
    which otherwise logs at DEBUG to stdout."""
    cfg = Path(folder) / "logback.xml"
    cfg.write_text('<configuration><appender name="STDERR" class="ch.qos.logback.core.ConsoleAppender"><target>System.err</target>'
                   '<encoder><pattern>%level %logger{0}: %msg%n</pattern></encoder></appender>'
                   '<root level="WARN"><appender-ref ref="STDERR"/></root></configuration>\n', encoding="utf-8")
    return f"-Dlogback.configurationFile={cfg}"


def run_java(args: list[str], *, cwd: str | os.PathLike | None = None, timeout: float = 600.0, java: str | None = None,
             env: dict[str, str] | None = None) -> JavaResult:
    """Run ``java <args>`` as a subprocess: stdin closed, stdout / stderr captured as UTF-8, killed after ``timeout``
    seconds, no console window on Windows.  Never raises for a failed run (see :attr:`JavaResult.error`)."""
    exe = java or find_java()[0]
    if not exe:
        return JavaResult(command=[], returncode=-1, stdout="", stderr="", seconds=0.0, error=find_java()[1])
    cmd = [exe] + [str(a) for a in args]
    run_env = dict(os.environ if env is None else env)
    run_env.setdefault("LUMEN_JVM_SUBPROCESS", "1")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=str(cwd) if cwd else None,
                           stdin=subprocess.DEVNULL, env=run_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return JavaResult(command=cmd, returncode=-1, stdout=out, stderr=err, seconds=time.perf_counter() - t0, timed_out=True,
                          error=f"the JVM did not finish within {timeout:.0f} s (LUMEN_JVM_TIMEOUT) and was killed")
    except OSError as e:
        return JavaResult(command=cmd, returncode=-1, stdout="", stderr="", seconds=time.perf_counter() - t0, error=f"could not start {exe}: {e}")
    return JavaResult(command=cmd, returncode=int(p.returncode), stdout=p.stdout or "", stderr=p.stderr or "", seconds=time.perf_counter() - t0)


def filter_stderr(text: str) -> str:
    """Drop the SLF4J / JVM boiler-plate warnings from a run's stderr (what is left is worth showing)."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("SLF4J:") or "picked up JAVA_TOOL_OPTIONS" in s.lower() or s.startswith("WARNING: A terminally deprecated"):
            continue
        keep.append(line.rstrip())
    return "\n".join(keep)
