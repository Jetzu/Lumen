"""Plugins ▸ Utilities ▸ Install Plugin from File, Folder, Zip or URL… (lumen/core/ops/plugin_install.py)."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import lumen.core.ops  # noqa: F401
from lumen.core import luts
from lumen.core import ops as ops_pkg
from lumen.core.ops import plugin_install as PI
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

PLUGIN = 'from lumen.core.registry import op\n\n@op("testzip.invert", "Zip Invert", "Plugins/Test")\ndef inv(plane, ctx):\n    return 255 - plane\n'
MACRO = 'run("Invert");\n'


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "lumen_home"
    monkeypatch.setenv("LUMEN_HOME", str(h))
    monkeypatch.delenv("LUMEN_PLUGINS", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    saved = dict(REGISTRY.ops)
    saved_ops = {k: list(v) for k, v in ops_pkg.PLUGIN_OPS.items()}
    saved_err = dict(ops_pkg.PLUGIN_ERRORS)
    saved_mods = set(sys.modules)
    saved_luts, saved_names = dict(luts._LUTS), list(luts.LUT_NAMES)
    yield h
    REGISTRY.ops.clear()
    REGISTRY.ops.update(saved)
    ops_pkg.PLUGIN_OPS.clear()
    ops_pkg.PLUGIN_OPS.update(saved_ops)
    ops_pkg.PLUGIN_ERRORS.clear()
    ops_pkg.PLUGIN_ERRORS.update(saved_err)
    for m in [m for m in sys.modules if m.startswith("lumen_plugin_") and m not in saved_mods]:
        sys.modules.pop(m, None)
    luts._LUTS.clear()
    luts._LUTS.update(saved_luts)
    luts.LUT_NAMES[:] = saved_names
    luts._invalidate_caches()


def _zip(path: Path) -> Path:
    lut = np.zeros((3, 256), np.uint8)
    lut[0] = np.arange(256)  # a red ramp
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mysite/zip_plugin.py", PLUGIN)
        zf.writestr("mysite/Zip Macro.ijm", MACRO)
        zf.writestr("mysite/RedRamp.lut", lut.tobytes())
        zf.writestr("mysite/README.md", "docs")
        zf.writestr("mysite/Plugin.class", b"\xca\xfe\xba\xbe")
        zf.writestr("__MACOSX/._zip_plugin.py", b"junk")
        zf.writestr("../escape.py", "print('no')")
    return path


def test_zip_installs_plugin_macro_and_lut_and_ignores_the_rest(home, tmp_path):
    s = Session()
    out = s.run_op("plugins.install_from", {"source": str(_zip(tmp_path / "site.zip"))})
    rows = {r[0]: r for r in out["table"]["rows"]}
    assert rows["zip_plugin.py"][1] == "plugin module" and "Zip Invert" in rows["zip_plugin.py"][3]
    assert rows["Zip Macro.ijm"][1] == "ImageJ macro"
    assert rows["RedRamp.lut"][1] == "LUT" and "RedRamp" in luts.LUT_NAMES
    assert rows["mysite/README.md"][1] == "ignored" and rows["mysite/Plugin.class"][1] == "ignored"
    assert "__MACOSX/._zip_plugin.py" in rows and "../escape.py" in rows  # listed as ignored, never written
    assert not (home / "plugins" / "escape.py").exists() and not (tmp_path / "escape.py").exists()
    assert "testzip.invert" in REGISTRY.ops and (home / "plugins" / "zip_plugin.py").exists()
    # the installed command runs
    s.run_op("file.new", {"title": "t", "type": "8-bit", "width": 8, "height": 8, "fill": "White"})
    s.run_op("testzip.invert", {})
    assert int(s.active.data.max()) == 0
    assert "Installed 3 files from zip" in out["message"] and "(4 ignored)" in out["message"]


def test_file_url_and_folder_sources(home, tmp_path):
    s = Session()
    z = _zip(tmp_path / "remote.zip")
    out = s.run_op("plugins.install_from", {"source": z.as_uri()})  # file:// goes through the same downloader as http(s)://
    assert "from url" in out["message"] and "testzip.invert" in REGISTRY.ops
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "one.py").write_text(PLUGIN.replace("testzip.invert", "testfolder.invert").replace("Zip Invert", "Folder Invert"))
    (folder / "notes.txt").write_text("not a macro really, but .txt is a macro extension in ImageJ")
    (folder / "data.csv").write_text("a,b\n")
    out = s.run_op("plugins.install_from", {"source": str(folder)})
    kinds = {r[0]: r[1] for r in out["table"]["rows"]}
    assert kinds["one.py"] == "plugin module" and kinds["notes.txt"] == "ImageJ macro" and kinds["data.csv"] == "ignored"
    assert "testfolder.invert" in REGISTRY.ops


def test_errors_are_explicit(home, tmp_path):
    s = Session()
    with pytest.raises(ValueError, match="Give a file, folder, zip or URL"):
        s.run_op("plugins.install_from", {"source": ""})
    with pytest.raises(FileNotFoundError):
        s.run_op("plugins.install_from", {"source": str(tmp_path / "missing.zip")})
    (tmp_path / "x.jar").write_bytes(b"PK\x03\x04junk")
    with pytest.raises(ValueError, match="only .py"):
        s.run_op("plugins.install_from", {"source": str(tmp_path / "x.jar")})
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt.bak", "x")
    with pytest.raises(ValueError, match="nothing installable"):
        s.run_op("plugins.install_from", {"source": str(empty)})


def test_download_size_cap(home, tmp_path, monkeypatch):
    big = tmp_path / "big.py"
    big.write_bytes(b"#" * 1024)
    monkeypatch.setattr(PI, "MAX_BYTES", 512)
    with pytest.raises(ValueError, match="more than 0 MB"):
        PI.download(big.as_uri(), tmp_path / "dl")
