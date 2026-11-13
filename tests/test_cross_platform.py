"""Windows-sensitive code paths, exercised on every OS.

Separator-correct file listings, uniform CSV line endings, ROI zip entry names, non-ASCII HDF5
channel names, ``LUMEN_HOME`` / ``LUMEN_PLUGINS``, UTF-8 MetaImage headers, explicit encodings
for every text file, and importing the whole package with the Unix-only stdlib modules removed.
"""
from __future__ import annotations

import ast
import csv
import io as _io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import lumen.core.ops  # noqa: F401  registers the commands used through Session.run_op
from lumen.core import io
from lumen.core.image import LumenImage
from lumen.core.session import Session

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "lumen"
UNIX_ONLY = ("resource", "fcntl", "termios", "tty", "pty", "pwd", "grp", "posix", "syslog")


# ============================================================================ file browser
def test_list_dir_paths_use_the_native_separator(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "a.txt").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").write_bytes(b"")
    r = io.list_dir(str(tmp_path))
    assert r["path"] == str(tmp_path) and r["parent"] == str(tmp_path.parent)
    assert [e["name"] for e in r["entries"]] == ["sub", "a.txt", "b.tif"]  # folders first, hidden files skipped
    foreign = "/" if os.sep == "\\" else "\\"
    for e in r["entries"]:
        assert e["path"] == os.path.join(str(tmp_path), e["name"])
        assert foreign not in e["path"]
    assert r["entries"][0]["dir"] and r["entries"][2]["image"] and not r["entries"][1]["image"]


def test_list_dir_at_a_filesystem_root_has_no_parent():
    root = Path.cwd().anchor  # "/" on POSIX, "C:\\" on Windows
    r = io.list_dir(root)
    assert r["parent"] is None and Path(r["path"]) == Path(root)


def test_list_dir_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))          # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Windows
    assert Path(io.list_dir("~")["path"]) == tmp_path


# ============================================================================ text files & CSV
def test_export_results_is_utf8_with_uniform_line_endings(tmp_path):
    s = Session()
    s.results.append([{"Label": "zelle µ", "Area": 1.5}, {"Label": "日本", "Area": 2.0}])
    out = tmp_path / "results.csv"
    s.run_op("file.export_results", {"path": str(out), "format": "csv"})
    raw = out.read_bytes()
    text = raw.decode("utf-8")  # would be mojibake / UnicodeEncodeError under cp1252
    assert b"\r\r\n" not in raw  # newline="" is used: the csv terminator is not translated again on Windows
    assert raw.count(b"\n") == 3 and raw.count(b"\r\n") in (0, 3)  # one terminator, used consistently
    rows = list(csv.reader(_io.StringIO(text)))
    assert rows[0] == [" ", "Label", "Area"] and rows[1][1] == "zelle µ" and rows[2][1] == "日本"


def test_results_csv_download_terminator_is_platform_independent():
    s = Session()
    s.results.append([{"Area": 1.0, "Mean": 2.5}, {"Area": 3.0, "Mean": 4.0}])
    text = s.results.to_csv()
    assert "\r\r\n" not in text
    assert text.count("\n") == 3 and text.count("\r\n") in (0, 3)
    assert text.replace("\r\n", "\n").splitlines() == [" ,Area,Mean", "1,1,2.5", "2,3,4"]


def test_table_text_export_uses_lf():
    from lumen.core.ops.file_extra import _table_text
    assert _table_text(["A", "B"], [[1, "x"], [2.5, "µ"]], "\t", True) == " \tA\tB\n1\t1\tx\n2\t2.5\tµ\n"


def test_recent_files_survive_non_ascii_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Session, "_recent_path", staticmethod(lambda: tmp_path / "recent.json"))
    p = str(tmp_path / "zelle µ 日本.tif")
    Session().add_recent(p)
    assert (tmp_path / "recent.json").read_bytes().isascii()  # json escapes non-ASCII, so any code page can read it back
    assert Session()._load_recent()[0] == p


def test_metaimage_header_is_utf8_with_lf_and_round_trips(tmp_path):
    img = LumenImage(np.arange(12, dtype=np.uint8).reshape(3, 4), name="zellen µ")
    out = io.write_image(img, tmp_path / "zellen µ.mhd", "mhd")
    raw = Path(out).read_bytes()
    assert b"\r\n" not in raw
    assert "ElementDataFile = zellen µ.raw".encode("utf-8") in raw
    back = io.read_image(out)
    assert np.array_equal(back.data[0, 0, 0], img.data[0, 0, 0])


def _text_io_without_encoding() -> list[str]:
    """Calls of builtin open() in text mode and Path.read_text()/write_text() that name no encoding."""
    found = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"), filename=str(f))):
            if not isinstance(node, ast.Call):
                continue
            fn, kws = node.func, {k.arg for k in node.keywords}
            if isinstance(fn, ast.Name) and fn.id == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                for k in node.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        mode = k.value.value
                if isinstance(mode, str) and "b" in mode:
                    continue
                if "encoding" not in kws:
                    found.append(f"{rel}:{node.lineno}: open(..., {mode!r}) without encoding=")
            elif isinstance(fn, ast.Attribute) and fn.attr in ("read_text", "write_text") and "encoding" not in kws:
                found.append(f"{rel}:{node.lineno}: .{fn.attr}() without encoding=")
    return found


# Known call sites that still rely on the locale encoding (harmless today: recent.json is pure ASCII;
# the two CSV writers break only for non-ASCII content on a cp1252 Windows).  Remove an entry once fixed.
_ENCODING_TODO = {
}


def test_text_files_name_an_encoding():
    new = [hit for hit in _text_io_without_encoding() if f"{hit.split(':')[0]}:{hit.split(':', 2)[2].split(' without')[0]}" not in _ENCODING_TODO]
    assert not new, "text I/O without encoding='utf-8' (breaks on a cp1252 Windows):\n" + "\n".join(new)


# ============================================================================ ROI zips
def test_roi_zip_entry_names_are_portable(tmp_path):
    names = ['a/b:c*d?e"f<g>h|i', "Cell", "cell", "CELL", "", "nuc 日本"]
    rois = [{"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4, "name": n} for n in names]
    out = io.write_ij_rois(rois, tmp_path / "rois.zip")
    with zipfile.ZipFile(out) as zf:
        entries = zf.namelist()
    assert len(entries) == len(names) and all(e.endswith(".roi") for e in entries)
    assert not any(ch in e for e in entries for ch in '\\/:*?"<>|')
    assert len({e.lower() for e in entries}) == len(entries)  # unique on NTFS / APFS after extraction
    assert [r["name"] for r in io.read_ij_rois(out)] == [n or "0005" for n in names]  # ROI names themselves are untouched


# ============================================================================ non-ASCII channel names
def test_hdf5_non_ascii_channel_names(tmp_path):
    pytest.importorskip("h5py")
    img = LumenImage(np.zeros((2, 4, 5), np.uint16), name="s", axes="CYX")
    img.display.channels[0].name = "α-tubulin"
    img.display.channels[1].name = "日本語"
    back = io.read_image(io.write_image(img, tmp_path / "ü.h5"))
    assert [c.name for c in back.display.channels] == ["α-tubulin", "日本語"]


# ============================================================================ LUMEN_HOME & plugin paths
def test_lumen_home_default_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv("LUMEN_HOME", raising=False)
    assert io.lumen_home() == Path.home() / ".lumen"
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "custom"))
    assert io.lumen_home() == tmp_path / "custom"
    monkeypatch.setenv("LUMEN_HOME", "~/lumen-data")
    assert io.lumen_home() == Path.home() / "lumen-data"
    monkeypatch.setenv("LUMEN_HOME", "   ")
    assert io.lumen_home() == Path.home() / ".lumen"


_HOME_TODO = "call site still hard-codes Path.home() / '.lumen'; route it through lumen.core.io.lumen_home() and drop this marker"


def test_recent_file_honours_lumen_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    assert Session._recent_path() == tmp_path / "home" / "recent.json"


def test_plugin_dir_honours_lumen_home(tmp_path, monkeypatch):
    from lumen.core.ops.system_ops import _plugin_dirs
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LUMEN_PLUGINS", raising=False)
    assert _plugin_dirs() == [tmp_path / "home" / "plugins"]


def test_samples_dir_honours_lumen_home(tmp_path, monkeypatch):
    from lumen.core.ops.file_extra import _samples_dir
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LUMEN_SAMPLES_DIR", raising=False)
    assert _samples_dir() == tmp_path / "home" / "samples"


def test_plugin_path_list_splits_on_os_pathsep(tmp_path, monkeypatch):
    from lumen.core.ops.system_ops import _plugin_dirs
    a, b = tmp_path / "a", tmp_path / "b"  # on Windows these contain ':' – splitting on ':' would shred them
    monkeypatch.setenv("LUMEN_PLUGINS", os.pathsep.join([str(a), str(b)]))
    assert _plugin_dirs()[1:] == [a, b]


# ============================================================================ Unix-only modules
def test_no_module_level_import_of_unix_only_modules():
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        for node in ast.parse(f.read_text(encoding="utf-8"), filename=str(f)).body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            bad += [f"{f.relative_to(ROOT).as_posix()}:{node.lineno}: import {n}" for n in names if n.split(".")[0] in UNIX_ONLY]
    assert not bad, bad


def test_process_rss_survives_missing_resource_and_psutil(monkeypatch):
    from lumen.core.ops import system_ops
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "resource", None)
    assert system_ops.process_rss_bytes() is None


def test_package_imports_without_unix_only_modules(tmp_path):
    code = (
        "import sys\n"
        f"for m in {UNIX_ONLY[:5]!r}:\n"
        "    sys.modules[m] = None\n"
        "import lumen.core.ops, lumen.server.app, lumen.__main__\n"
        "from lumen.core.ops.system_ops import process_rss_bytes\n"
        "sys.modules['psutil'] = None\n"
        "print('OK', process_rss_bytes())\n"
    )
    env = {**os.environ, "LUMEN_HOME": str(tmp_path), "LUMEN_PLUGINS": "", "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("OK None")


# ============================================================================ launcher & console
def test_console_reconfigure_is_windows_only(monkeypatch):
    from lumen import __main__ as m
    calls: list[dict] = []

    class Stream:
        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", Stream())
    monkeypatch.setattr(sys, "stderr", Stream())
    monkeypatch.setattr(sys, "platform", "darwin")
    m._configure_console()
    assert calls == []
    monkeypatch.setattr(sys, "platform", "win32")
    m._configure_console()
    assert calls == [{"errors": "replace"}, {"errors": "replace"}]
    monkeypatch.setattr(sys, "stdout", object())  # a stream without reconfigure() must not raise
    m._configure_console()


def test_launchers_have_native_line_endings():
    bat = (ROOT / "scripts" / "run_lumen.bat").read_bytes()
    sh = (ROOT / "scripts" / "run_lumen.sh").read_bytes()
    assert bat.count(b"\n") == bat.count(b"\r\n") > 0  # cmd.exe wants CRLF
    assert b"\r" not in sh and sh.startswith(b"#!/usr/bin/env bash")
