"""Cross-validation harness: the same ImageJ macro, run by ImageJ 1.54p and by Lumen, compared numerically.

How a case works
----------------
* A synthetic image (deterministic numpy, see ``IMAGES``) is written as an ImageJ-flavoured TIFF (tifffile, ImageJ
  metadata: unit, spacing, axes) into two scratch folders, ``<work>/ij`` and ``<work>/lumen``.
* The macro ``macros/<name>.ijm`` is run **unchanged** on both sides.  Every macro starts with
  ``dir = getArgument(); open(dir + "/input.tif");`` and ends by saving what it produced into ``dir``:
  ``saveAs("Tiff", dir + "/result.tif")`` for pixels, ``saveAs("Results", dir + "/Results.csv")`` for the Results
  table, and ``print("key=value")`` lines for anything else (threshold levels, counts, histograms, bounds …).
* ImageJ side: ``java -cp ij.jar ij.ImageJ -batch macro.ijm <work>/ij`` (ImageJ's batch mode – no ImageJ window, no
  image windows; ``-Dapple.awt.UIElement=true`` keeps it off the Dock; prefs are isolated with ``-Duser.home``).
  ``print()`` goes to stdout, the files land in the folder.
* Lumen side: ``IJ(Session()).macro(text, arg=<work>/lumen)``; the pixels are taken from the session's current image
  (in memory, not from Lumen's TIFF writer), the rows from ``session.results``, the log from the macro's return value.
* Comparison: pixel arrays (shape, dtype, max |Δ|, fraction of differing pixels), Results rows (per column, rows
  matched positionally or by centroid), log values (per key, arrays elementwise).

The macro header line ``// crossval: title=… | images=a,b | compare=pixels,results,log | group=…`` turns one macro into
one case per listed image (case id ``<macro>`` or ``<macro>[<image>]``).  ``EXPECT`` records, per case, the tolerance
that is asserted, why it is needed, and the known mismatches (findings – see docs/CROSSVALIDATION.md).

Re-run / regenerate the table::

    .venv/bin/python tests/crossval/harness.py --markdown report.md        # all cases, markdown table + summary
    .venv/bin/python tests/crossval/harness.py --only gaussian_s2,median_r2  # a few
    .venv/bin/python -m pytest tests/crossval -m crossval -q                 # as tests

Environment: ``LUMEN_IJ_JAR`` (default /Applications/Fiji/jars/ij-1.54p.jar), ``JAVA_HOME`` or the Homebrew OpenJDK 21
path or ``java`` on PATH.  ImageJ 1.x cannot run ``IJ.run`` with ``java.awt.headless=true`` (its menus are AWT objects),
so the JVM needs a display session (macOS login session / X11); nothing is ever shown.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE / "macros"
DEFAULT_JAR = "/Applications/Fiji/jars/ij-1.54p.jar"
HOMEBREW_JAVA = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/bin/java"

#: ImageJ writes the Results CSV with ``decimal=9``: a difference below this is only the printing (counted as exact)
CSV_EPS = 1e-8
#: printed values use d2s(v, 6) (or integers): differences below this are only the printing
LOG_EPS = 2e-6


# ====================================================================== environment
def find_java() -> str | None:
    cands: list[Path] = []
    jh = os.environ.get("JAVA_HOME", "").strip()
    if jh:
        cands.append(Path(jh) / "bin" / ("java.exe" if os.name == "nt" else "java"))
    cands.append(Path(HOMEBREW_JAVA))
    w = shutil.which("java")
    if w:
        cands.append(Path(w))
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def find_jar() -> str | None:
    p = os.environ.get("LUMEN_IJ_JAR", "").strip() or DEFAULT_JAR
    return p if Path(p).is_file() else None


@dataclass
class Environment:
    java: str
    jar: str
    scratch: Path
    timeout: float = float(os.environ.get("LUMEN_CROSSVAL_TIMEOUT", "120"))  # seconds per ImageJ process; a macro stuck in a dialog is reported as a timeout

    @classmethod
    def discover(cls, scratch: Path) -> "Environment | None":
        java, jar = find_java(), find_jar()
        if not java or not jar:
            return None
        return cls(java=java, jar=jar, scratch=Path(scratch))

    @property
    def ij_home(self) -> Path:
        p = self.scratch / "ijhome"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def ij_plugins(self) -> Path:
        p = self.scratch / "ijplugins"  # empty: Fiji's plugins folder next to the jar must not be loaded
        p.mkdir(parents=True, exist_ok=True)
        return p

    def java_version(self) -> str:
        try:
            p = subprocess.run([self.java, "-version"], capture_output=True, text=True, timeout=60)
            return (p.stderr or p.stdout).strip().splitlines()[0]
        except Exception as e:  # pragma: no cover
            return f"? ({e})"

    def run_macro(self, macro: Path, arg: str) -> tuple[str, str, float]:
        """Run ``macro`` in ImageJ batch mode with ``getArgument() == arg``; returns (stdout, stderr, seconds)."""
        cmd = [self.java, "-Dapple.awt.UIElement=true", f"-Duser.home={self.ij_home}", f"-Dij.dir={self.ij_home}",
               f"-Dplugins.dir={self.ij_plugins}", "-Xmx1024m", "-cp", self.jar, "ij.ImageJ", "-batch", str(macro), arg]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=self.timeout, cwd=str(self.ij_home))
        return p.stdout, p.stderr, time.perf_counter() - t0


# ====================================================================== synthetic images
def _field(h: int, w: int, seed: int, n: int = 14, rings: int = 3, noise: float = 6.0, base: float = 30.0,
           dx: float = 0.0, dy: float = 0.0, gain: float = 1.0, noise_seed: int | None = None) -> np.ndarray:
    """Bright Gaussian blobs and a few rings (holes for Fill Holes) on a dark background, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), base, dtype=np.float64)
    for _ in range(n):
        cx, cy = rng.uniform(10, w - 10) + dx, rng.uniform(10, h - 10) + dy
        s, a = rng.uniform(2.5, 8.0), rng.uniform(80, 200)
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    for _ in range(rings):
        cx, cy = rng.uniform(18, w - 18) + dx, rng.uniform(18, h - 18) + dy
        r = rng.uniform(8, 13)
        d = np.hypot(xx - cx, yy - cy)
        img += 170 * np.exp(-((d - r) ** 2) / (2 * 1.8 ** 2))
    nrng = rng if noise_seed is None else np.random.default_rng(noise_seed)
    return img * gain + nrng.normal(0, noise, (h, w))


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _blobs8() -> tuple[np.ndarray, dict[str, Any]]:
    return _u8(_field(120, 160, 1)), {}


def _blobs16() -> tuple[np.ndarray, dict[str, Any]]:
    return np.clip(np.rint(_field(120, 160, 2) * 16), 0, 65535).astype(np.uint16), {}


def _blobs32() -> tuple[np.ndarray, dict[str, Any]]:
    return ((_field(120, 160, 3) - 60.0) / 100.0).astype(np.float32), {}


def _zstack8() -> tuple[np.ndarray, dict[str, Any]]:
    gains = [0.55, 0.8, 1.0, 1.0, 0.75, 0.5]
    planes = [_u8(_field(96, 96, 4, n=10, rings=2, dx=1.5 * i, dy=-1.0 * i, gain=g, noise_seed=100 + i)) for i, g in enumerate(gains)]
    return np.stack(planes), {"metadata": {"axes": "ZYX"}}


def _twochan16() -> tuple[np.ndarray, dict[str, Any]]:
    c1 = np.clip(np.rint(_field(96, 96, 5, n=9, rings=2) * 12), 0, 65535)
    c2 = np.clip(np.rint(_field(96, 96, 6, n=7, rings=1) * 9), 0, 65535)
    return np.stack([c1, c2]).astype(np.uint16), {"metadata": {"axes": "CYX", "mode": "composite"}}


def _cal8() -> tuple[np.ndarray, dict[str, Any]]:
    return _u8(_field(120, 160, 1)), {"resolution": (4.0, 4.0), "metadata": {"unit": "um"}}


IMAGES: dict[str, Callable[[], tuple[np.ndarray, dict[str, Any]]]] = {
    "blobs8": _blobs8, "blobs16": _blobs16, "blobs32": _blobs32, "zstack8": _zstack8, "twochan16": _twochan16, "cal8": _cal8,
}
IMAGE_NOTES = {
    "blobs8": "160×120 8-bit: 14 Gaussian blobs + 3 rings on background 30, noise σ=6",
    "blobs16": "160×120 16-bit: the same kind of field ×16 (0–4095)",
    "blobs32": "160×120 32-bit: (field − 60)/100, values ≈ −0.6 … 2.0 (negative values included)",
    "zstack8": "96×96×6 8-bit Z stack: blobs drifting 1.5 px/slice with per-slice gain 0.55…1.0",
    "twochan16": "96×96 2-channel 16-bit composite (different blob fields per channel)",
    "cal8": "blobs8 with a 0.25 µm pixel size stored in the TIFF (ImageJ metadata unit=um, 4 px/µm)",
}


def write_input(name: str, path: Path) -> None:
    import tifffile
    data, kw = IMAGES[name]()
    tifffile.imwrite(str(path), data, imagej=True, **kw)


# ====================================================================== cases
_HEADER_RE = re.compile(r"^//\s*crossval:\s*(.*)$")


@dataclass
class Case:
    id: str
    macro: str            # file name in macros/
    image: str            # key of IMAGES
    title: str
    compare: tuple[str, ...]
    group: str = ""
    expect: str = "exact"         # exact | tolerance | mismatch
    tol: dict[str, float] = field(default_factory=dict)
    note: str = ""                # why the tolerance / the finding
    match: tuple[str, ...] = ()   # Results rows matched by these columns (centroids) instead of by position
    fallback: str = ""            # name in FALLBACKS: Lumen-side extraction used when the macro layer lacks a function

    @property
    def macro_path(self) -> Path:
        return MACRO_DIR / self.macro

    def macro_text(self) -> str:
        return self.macro_path.read_text(encoding="utf-8")


def _parse_header(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = _HEADER_RE.match(line.strip())
            if m:
                out: dict[str, str] = {}
                for part in m.group(1).split(" | "):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        out[k.strip()] = v.strip()
                return out
            if line.strip() and not line.startswith("//"):
                break
    raise ValueError(f"{path.name}: missing '// crossval:' header")


#: per-case expectations (tolerance keys pix_abs, pix_frac, dtype_free, res_atol, res_rtol, log_atol, log_rtol; expect; note), see expectations.py
try:
    from .expectations import EXPECT  # noqa: E402
except ImportError:  # run as a script: python tests/crossval/harness.py
    sys.path.insert(0, str(HERE))
    from expectations import EXPECT  # type: ignore[no-redef]  # noqa: E402


def load_cases(expect: dict[str, dict[str, Any]] | None = None) -> list[Case]:
    expect = EXPECT if expect is None else expect
    cases: list[Case] = []
    for path in sorted(MACRO_DIR.glob("*.ijm")):
        h = _parse_header(path)
        images = [s.strip() for s in h.get("images", "blobs8").split(",") if s.strip()]
        compare = tuple(s.strip() for s in h.get("compare", "pixels").split(",") if s.strip())
        for img in images:
            cid = path.stem if len(images) == 1 else f"{path.stem}[{img}]"
            spec = dict(expect.get(cid, {}))
            if img not in IMAGES:
                raise ValueError(f"{path.name}: unknown image {img!r}")
            cases.append(Case(id=cid, macro=path.name, image=img, title=h.get("title", path.stem), compare=compare,
                              group=h.get("group", ""), expect=spec.pop("expect", "exact"), tol=spec.pop("tol", {}),
                              note=spec.pop("note", ""), match=tuple(spec.pop("match", ())) or tuple(s for s in h.get("match", "").split(",") if s),
                              fallback=spec.pop("fallback", "") or h.get("fallback", "")))
    return cases


# ====================================================================== running one side
@dataclass
class SideOut:
    pixels: np.ndarray | None = None   # (T, Z, C, Y, X)
    rows: list[dict[str, Any]] = field(default_factory=list)
    log: str = ""
    error: str | None = None
    seconds: float = 0.0
    title: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # fallback log values (Lumen)


_AXIS_MAP = {"S": "C", "I": "Z", "Q": "Z", "R": "T"}


def to_tzcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = "".join(_AXIS_MAP.get(a, a) for a in axes.upper())
    if len(set(axes)) != len(axes):
        raise ValueError(f"cannot canonicalise axes {axes!r}")
    for a in "TZCYX":
        if a not in axes:
            arr = np.expand_dims(arr, 0)
            axes = a + axes
    order = [axes.index(a) for a in "TZCYX"]
    return np.ascontiguousarray(np.transpose(arr, order))


def _num(v: Any) -> Any:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if v is None:
        return float("nan")
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        return s


def read_ij_results(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return []
    header = rows[0]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        d: dict[str, Any] = {}
        for k, v in zip(header, r):
            if k.strip() == "":
                continue  # the row-number column
            d[k] = _num(v)
        out.append(d)
    return out


_ERR_RE = re.compile(r"(Macro Error|Exception|Unrecognized command|not supported|is not a valid)")


def run_imagej(env: Environment, case: Case, work: Path) -> SideOut:
    work.mkdir(parents=True, exist_ok=True)
    write_input(case.image, work / "input.tif")
    macro = work / case.macro
    shutil.copyfile(case.macro_path, macro)
    out = SideOut()
    try:
        stdout, stderr, out.seconds = env.run_macro(macro, str(work))
    except subprocess.TimeoutExpired:
        out.error = f"ImageJ timed out after {env.timeout:.0f} s"
        return out
    out.log = stdout
    for line in (stdout + "\n" + stderr).splitlines():
        if _ERR_RE.search(line):
            out.error = "ImageJ: " + line.strip()[:200]
            break
    rp = work / "result.tif"
    if rp.exists():
        import tifffile
        with tifffile.TiffFile(str(rp)) as tf:
            ser = tf.series[0]
            out.pixels = to_tzcyx(ser.asarray(), ser.axes)
    cp = work / "Results.csv"
    if cp.exists():
        out.rows = read_ij_results(cp)
    return out


def _profile_fallback(session: Any, ij: Any) -> dict[str, Any]:
    """Lumen's macro layer has no getProfile(): Analyze ▸ Plot Profile on the same selection gives the values."""
    if ij.roi is None:
        return {}
    res = session.run_op("analyze.plot_profile", {}, roi=ij.roi)
    plot = res.get("plot") or session.last_plot or {}
    series = plot.get("series") or []
    if not series:
        return {}
    ys = np.asarray(series[0].get("values", []), dtype=np.float64)
    return {"n": np.array([float(ys.size)]), "profile": ys}


FALLBACKS: dict[str, Callable[[Any, Any], dict[str, Any]]] = {"profile": _profile_fallback}


def run_lumen(case: Case, work: Path) -> SideOut:
    work.mkdir(parents=True, exist_ok=True)
    write_input(case.image, work / "input.tif")
    os.environ.setdefault("LUMEN_HOME", str(work.parent / "lumen_home"))  # keep recent-files bookkeeping out of ~/.lumen
    from lumen.core import ops  # noqa: F401  (registers every op)
    from lumen.core.ij_compat import IJ
    from lumen.core.session import Session

    session = Session()
    ij = IJ(session)
    out = SideOut()
    t0 = time.perf_counter()
    try:
        out.log = ij.macro(case.macro_text(), arg=str(work))
    except Exception as e:  # the macro layer raised: an unsupported function / command or a failing op
        out.error = f"Lumen: {type(e).__name__}: {e}"
        buf = getattr(ij, "_out", None)
        out.log = buf.getvalue() if buf is not None else ""
    out.seconds = time.perf_counter() - t0
    img = session.active
    if img is not None:
        out.pixels = np.ascontiguousarray(img.data)
        out.title = img.name
    out.rows = [dict(r) for r in session.results.rows]
    if case.fallback:
        try:
            out.extra = FALLBACKS[case.fallback](session, ij)
        except Exception as e:  # pragma: no cover - reported, not raised
            out.extra = {}
            out.error = (out.error + "; " if out.error else "") + f"fallback failed: {e}"
    return out


# ====================================================================== comparison
def _status_max(*s: str) -> str:
    order = {"exact": 0, "tolerance": 1, "mismatch": 2}
    return max(s, key=lambda x: order.get(x, 2)) if s else "exact"


def compare_pixels(a: np.ndarray | None, b: np.ndarray | None, tol: dict[str, float]) -> dict[str, Any]:
    r: dict[str, Any] = {}
    if a is None or b is None:
        r.update(status="mismatch", text="pixels: " + ("ImageJ wrote no result.tif" if a is None else "Lumen has no image"))
        return r
    a2, b2 = np.squeeze(a), np.squeeze(b)
    r.update(shape_ij=tuple(int(x) for x in a2.shape), shape_lumen=tuple(int(x) for x in b2.shape), dtype_ij=str(a2.dtype), dtype_lumen=str(b2.dtype))
    if a2.shape != b2.shape:
        r.update(status="mismatch", text=f"pixels: shape {r['shape_ij']} (ImageJ) vs {r['shape_lumen']} (Lumen)")
        return r
    da, db = a2.astype(np.float64), b2.astype(np.float64)
    both_nan = np.isnan(da) & np.isnan(db)
    d = np.abs(da - db)
    d[both_nan] = 0.0
    d[np.isnan(d)] = np.inf
    n = d.size
    max_abs = float(d.max()) if n else 0.0
    n_diff = int(np.count_nonzero(d > 0))
    frac = n_diff / n if n else 0.0
    finite = d[np.isfinite(d)]
    mean_abs = float(finite.mean()) if finite.size else float("inf")
    dtype_same = a2.dtype == b2.dtype
    r.update(max_abs=max_abs, frac_diff=frac, n_diff=n_diff, mean_abs=mean_abs, dtype_same=dtype_same)
    shape = "×".join(str(x) for x in reversed(r["shape_ij"]))
    if max_abs == 0 and dtype_same:
        r.update(status="exact", text=f"pixels {r['dtype_ij']} {shape}: identical")
        return r
    within = (max_abs <= tol.get("pix_abs", 0.0)) and (frac <= tol.get("pix_frac", 0.0)) and (dtype_same or bool(tol.get("dtype_free", 0)))
    dt = "" if dtype_same else f"; dtype {r['dtype_ij']} (ImageJ) vs {r['dtype_lumen']} (Lumen)"
    r.update(status="tolerance" if within else "mismatch",
             text=f"pixels {r['dtype_ij']} {shape}: max|Δ|={max_abs:.4g}, {frac:.2%} of pixels differ, mean|Δ|={mean_abs:.3g}{dt}")
    return r


def _ordered_cols(rows: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def _match_rows(ij_rows: list[dict[str, Any]], lu_rows: list[dict[str, Any]], match: tuple[str, ...], radius: float = 3.0) -> list[tuple[int, int]]:
    if not match or not ij_rows or not lu_rows:
        n = min(len(ij_rows), len(lu_rows))
        return [(i, i) for i in range(n)]

    def vec(r: dict[str, Any]) -> np.ndarray:
        return np.array([_num(r.get(c)) if isinstance(_num(r.get(c)), float) else np.nan for c in match], dtype=np.float64)

    A = np.array([vec(r) for r in ij_rows])
    B = np.array([vec(r) for r in lu_rows])
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    D = np.where(np.isnan(D), np.inf, D)
    pairs: list[tuple[int, int]] = []
    used_i: set[int] = set()
    used_j: set[int] = set()
    for idx in np.argsort(D, axis=None):
        i, j = divmod(int(idx), D.shape[1])
        if D[i, j] > radius:
            break
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)
    return pairs


def _diff(a: Any, b: Any) -> tuple[float, float]:
    """(|a-b|, |a|) for two cell / log values; inf when they are not comparable."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.shape != b.shape:
            return float("inf"), 0.0
        both = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both] = 0.0
        d[np.isnan(d)] = np.inf
        return (float(d.max()) if d.size else 0.0), (float(np.nanmax(np.abs(a))) if a.size else 0.0)
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return 0.0, 0.0
        return abs(a - b), abs(a)
    return (0.0 if str(a) == str(b) else float("inf")), 0.0


def _fmt(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(f"{x:.6g}" for x in v[:6]) + (", …" if v.size > 6 else "") + "]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def compare_results(ij_rows: list[dict[str, Any]], lu_rows: list[dict[str, Any]], tol: dict[str, float], match: tuple[str, ...]) -> dict[str, Any]:
    r: dict[str, Any] = {"rows_ij": len(ij_rows), "rows_lumen": len(lu_rows)}
    cols_ij, cols_lu = _ordered_cols(ij_rows), _ordered_cols(lu_rows)
    r["cols_missing"] = [c for c in cols_ij if c not in cols_lu]   # ImageJ writes them, Lumen does not
    r["cols_extra"] = [c for c in cols_lu if c not in cols_ij]     # Lumen-only columns (reported, not a failure)
    common = [c for c in cols_ij if c in cols_lu]
    pairs = _match_rows(ij_rows, lu_rows, match)
    r["unmatched"] = len(ij_rows) - len(pairs)
    atol, rtol = tol.get("res_atol", 0.0), tol.get("res_rtol", 0.0)
    worst: tuple[str, float, Any, Any, float] | None = None  # col, |Δ|, ij value, lumen value, ratio
    per_col: dict[str, float] = {}
    for c in common:
        cmax, cij, clu = 0.0, None, None
        for i, j in pairs:
            a, b = _num(ij_rows[i].get(c)), _num(lu_rows[j].get(c))
            d, _ref = _diff(a, b)
            if d > cmax or cij is None:
                cmax, cij, clu = d, a, b
        per_col[c] = cmax
        ref = abs(cij) if isinstance(cij, float) and np.isfinite(cij) else 0.0
        limit = atol + rtol * ref
        ratio = 0.0 if cmax == 0 else (cmax / limit if limit > 0 else float("inf"))
        if worst is None or ratio > worst[4] or (ratio == worst[4] and cmax > worst[1]):
            worst = (c, cmax, cij, clu, ratio)
    r["per_col"] = per_col
    parts = [f"results: {len(ij_rows)} vs {len(lu_rows)} rows"]
    status = "exact"
    if len(ij_rows) != len(lu_rows) or r["unmatched"]:
        status = "mismatch"
        if r["unmatched"] and len(ij_rows) == len(lu_rows):
            parts.append(f"{r['unmatched']} unmatched by {'/'.join(match)}")
    if r["cols_missing"]:
        status = "mismatch"
        parts.append("Lumen lacks " + ", ".join(r["cols_missing"]))
    if worst is not None:
        c, cmax, cij, clu, ratio = worst
        if cmax == 0:
            parts.append(f"all {len(common)} common columns identical")
        else:
            parts.append(f"worst {c}: |Δ|={cmax:.4g} (ImageJ {_fmt(cij)}, Lumen {_fmt(clu)})")
            ref = abs(cij) if isinstance(cij, float) and np.isfinite(cij) else 0.0
            if cmax <= CSV_EPS * (1 + ref):
                pass  # only the 9-decimal printing
            elif ratio <= 1.0:
                status = _status_max(status, "tolerance")
            else:
                status = "mismatch"
    if r["cols_extra"]:
        parts.append("Lumen adds " + ", ".join(r["cols_extra"]))
    r.update(status=status, text="; ".join(parts), worst=worst)
    return r


_LOG_RE = re.compile(r"^([A-Za-z_][\w.%-]*)=(.*)$")


def parse_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _LOG_RE.match(line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        parts = [p.strip() for p in val.split(",") if p.strip() != ""]
        try:
            out[key] = np.array([float(p) for p in parts], dtype=np.float64) if parts else np.array([], dtype=np.float64)
        except ValueError:
            out[key] = val
    return out


def compare_log(lij: dict[str, Any], llu: dict[str, Any], tol: dict[str, float]) -> dict[str, Any]:
    r: dict[str, Any] = {"keys_ij": list(lij), "missing": [k for k in lij if k not in llu]}
    atol, rtol = tol.get("log_atol", 0.0), tol.get("log_rtol", 0.0)
    worst: tuple[str, float, Any, Any, float] | None = None
    for k in lij:
        if k not in llu:
            continue
        a, b = lij[k], llu[k]
        d, ref = _diff(a, b)
        limit = atol + rtol * ref
        ratio = 0.0 if d == 0 else (d / limit if limit > 0 else float("inf"))
        if worst is None or ratio > worst[4] or (ratio == worst[4] and d > worst[1]):
            worst = (k, d, a, b, ratio)
    status = "exact"
    parts = []
    if not lij:
        parts.append("log: ImageJ printed no key=value lines")
        status = "mismatch"
    if r["missing"]:
        parts.append("log: Lumen did not print " + ", ".join(r["missing"]))
        status = "mismatch"
    if worst is not None:
        k, d, a, b, ratio = worst
        if d == 0:
            parts.append(f"log: all {len(lij) - len(r['missing'])} printed values identical")
        else:
            parts.append(f"log {k}: |Δ|={d:.4g} (ImageJ {_fmt(a)}, Lumen {_fmt(b)})")
            if d <= LOG_EPS * (1 + ref_of(a)):
                pass
            elif ratio <= 1.0:
                status = _status_max(status, "tolerance")
            else:
                status = "mismatch"
    r.update(status=status, text="; ".join(parts), worst=worst)
    return r


def ref_of(a: Any) -> float:
    if isinstance(a, np.ndarray):
        return float(np.nanmax(np.abs(a))) if a.size else 0.0
    return abs(a) if isinstance(a, float) and np.isfinite(a) else 0.0


# ====================================================================== one case
@dataclass
class Outcome:
    case: Case
    status: str
    verdict: str
    pixels: dict[str, Any] | None
    results: dict[str, Any] | None
    log: dict[str, Any] | None
    errors: list[str]
    t_ij: float
    t_lumen: float
    title_lumen: str | None = None
    ij_log: str = ""
    lumen_log: str = ""


def run_case(case: Case, env: Environment, work: Path) -> Outcome:
    work = Path(work)
    errors: list[str] = []
    try:
        ij = run_imagej(env, case, work / "ij")
    except Exception as e:
        ij = SideOut(error=f"harness (ImageJ side): {type(e).__name__}: {e}")
    try:
        lu = run_lumen(case, work / "lumen")
    except Exception as e:
        lu = SideOut(error=f"harness (Lumen side): {type(e).__name__}: {e}")
    for side in (ij, lu):
        if side.error:
            errors.append(side.error)
    px = res = lg = None
    statuses: list[str] = []
    texts: list[str] = []
    if "pixels" in case.compare:
        px = compare_pixels(ij.pixels, lu.pixels, case.tol)
        statuses.append(px["status"])
        texts.append(px["text"])
    if "results" in case.compare:
        res = compare_results(ij.rows, lu.rows, case.tol, case.match)
        statuses.append(res["status"])
        texts.append(res["text"])
    if "log" in case.compare:
        llu = parse_log(lu.log)
        for k, v in lu.extra.items():
            llu.setdefault(k, v)
        lg = compare_log(parse_log(ij.log), llu, case.tol)
        statuses.append(lg["status"])
        texts.append(lg["text"])
    status = _status_max(*statuses) if statuses else "exact"
    if errors:
        status = "mismatch"
        texts = errors + texts
    return Outcome(case=case, status=status, verdict="; ".join(t for t in texts if t), pixels=px, results=res, log=lg,
                   errors=errors, t_ij=ij.seconds, t_lumen=lu.seconds, title_lumen=lu.title, ij_log=ij.log, lumen_log=lu.log)


# ====================================================================== report
STATUS_ICON = {"exact": "✅ exact", "tolerance": "🟢 within tolerance", "mismatch": "❌ mismatch"}


def _tol_text(case: Case) -> str:
    t = case.tol
    bits = []
    if "pix_abs" in t or "pix_frac" in t:
        bits.append(f"pixels: max|Δ| ≤ {t.get('pix_abs', 0):g}, ≤ {t.get('pix_frac', 0):.0%} of pixels")
    if "res_atol" in t or "res_rtol" in t:
        bits.append(f"results: |Δ| ≤ {t.get('res_atol', 0):g} + {t.get('res_rtol', 0):g}·|v|")
    if "log_atol" in t or "log_rtol" in t:
        bits.append(f"printed: |Δ| ≤ {t.get('log_atol', 0):g} + {t.get('log_rtol', 0):g}·|v|")
    return "; ".join(bits) or "none (exact)"


def markdown_table(outcomes: list[Outcome]) -> str:
    lines = ["| # | Case | ImageJ command(s) | Image | Compared | Measured | Tolerance asserted | Status | Note |", "|---|---|---|---|---|---|---|---|---|"]
    group = None
    for n, o in enumerate(outcomes, 1):
        c = o.case
        if c.group != group:
            group = c.group
            lines.append(f"| | **{group}** | | | | | | | |")
        note = c.note.replace("|", "\\|")
        verdict = o.verdict.replace("|", "\\|")
        lines.append(f"| {n} | `{c.id}` | {c.title} | {c.image} | {', '.join(c.compare)} | {verdict} | {_tol_text(c)} | {STATUS_ICON.get(o.status, o.status)} | {note} |")
    return "\n".join(lines)


def summary(outcomes: list[Outcome]) -> dict[str, Any]:
    counts = {k: sum(1 for o in outcomes if o.status == k) for k in ("exact", "tolerance", "mismatch")}
    return {"n": len(outcomes), **counts, "t_ij": sum(o.t_ij for o in outcomes), "t_lumen": sum(o.t_lumen for o in outcomes)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Cross-validate Lumen against ImageJ 1.54p (see docs/CROSSVALIDATION.md)")
    ap.add_argument("--only", default="", help="comma-separated case ids (or macro stems)")
    ap.add_argument("--markdown", default="", help="write the markdown table + summary here")
    ap.add_argument("--json", default="", help="write every outcome as JSON here")
    ap.add_argument("--work", default="", help="scratch folder to keep (default: a temporary folder)")
    ap.add_argument("--ignore-expect", action="store_true", help="compare with zero tolerance (shows the raw numbers)")
    ap.add_argument("--range", default="", help="start:stop slice of the case list (chunked runs)")
    a = ap.parse_args(argv)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="lumen-crossval-"))
    env = Environment.discover(work)
    if env is None:
        print("java or the ImageJ jar not found (JAVA_HOME / LUMEN_IJ_JAR)", file=sys.stderr)
        return 2
    cases = load_cases({} if a.ignore_expect else None)
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        cases = [c for c in cases if c.id in wanted or c.macro[:-4] in wanted]
    if a.range:
        lo, _, hi = a.range.partition(":")
        cases = cases[int(lo or 0):int(hi) if hi else None]
    print(f"java: {env.java_version()}  jar: {env.jar}  cases: {len(cases)}  work: {work}", flush=True)
    outcomes: list[Outcome] = []
    for c in cases:
        o = run_case(c, env, work / c.id.replace("[", "_").replace("]", ""))
        outcomes.append(o)
        print(f"{STATUS_ICON[o.status]:22s} {c.id:40s} ij {o.t_ij:5.2f}s  lumen {o.t_lumen:5.2f}s  {o.verdict}", flush=True)
    s = summary(outcomes)
    text = markdown_table(outcomes) + f"\n\n**{s['n']} cases: {s['exact']} exact, {s['tolerance']} within tolerance, {s['mismatch']} mismatch.** ImageJ {s['t_ij']:.1f} s, Lumen {s['t_lumen']:.1f} s.\n"
    if a.markdown:
        Path(a.markdown).write_text(text, encoding="utf-8")
    if a.json:
        import json

        def clean(x: Any) -> Any:
            if isinstance(x, dict):
                return {str(k): clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            if isinstance(x, float) and not np.isfinite(x):
                return str(x)
            return x
        Path(a.json).write_text(json.dumps([clean({"id": o.case.id, "status": o.status, "verdict": o.verdict, "pixels": o.pixels, "results": o.results, "log": o.log,
                                                    "errors": o.errors, "t_ij": o.t_ij, "t_lumen": o.t_lumen, "title_lumen": o.title_lumen, "ij_log": o.ij_log, "lumen_log": o.lumen_log}) for o in outcomes], indent=1), encoding="utf-8")
    print(text.splitlines()[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
