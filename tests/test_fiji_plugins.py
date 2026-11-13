"""Fiji update sites and plugin jars (lumen/core/ops/fiji_plugins.py).

The update-site part runs everywhere against a local site built in the test (a db.xml.gz + plugin jars made here).
The part that runs real Fiji plugins needs a JDK and an installed Fiji (its jars folder) and is skipped otherwise;
on the development Mac it runs Fiji's own Auto_Threshold and Kuwahara_Filter jars through the Java bridge."""
from __future__ import annotations

import gzip
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import lumen.core.ops  # noqa: F401
from lumen.core.ops import fiji_plugins as FP
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

FIJI_PLUGINS = Path("/Applications/Fiji/plugins")
AUTO_THRESHOLD = next(iter(sorted(FIJI_PLUGINS.glob("Auto_Threshold-*.jar"))), None) if FIJI_PLUGINS.is_dir() else None
KUWAHARA = next(iter(sorted(FIJI_PLUGINS.glob("Kuwahara_Filter-*.jar"))), None) if FIJI_PLUGINS.is_dir() else None


def _fake_jar(path: Path, config: str, extra: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("plugins.config", config)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def _site(root: Path, files: dict[str, Path], deps: dict[str, list[str]] | None = None) -> Path:
    """A Fiji update site: <root>/<filename>-<timestamp> files and db.xml.gz describing them."""
    root.mkdir(parents=True, exist_ok=True)
    ts = "20240101000000"
    recs = []
    for fname, src in files.items():
        dest = root / f"{fname}-{ts}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        dep_xml = "".join(f'<dependency filename="{d}" timestamp="{ts}"/>' for d in (deps or {}).get(fname, []))
        recs.append(f'<plugin filename="{fname}"><version checksum="x" timestamp="{ts}" filesize="{dest.stat().st_size}">'
                    f"<description>{fname} for tests</description>{dep_xml}</version></plugin>")
    recs.append('<plugin filename="plugins/Removed.jar"><previous-version timestamp="2019" checksum="y"/></plugin>')
    xml = f'<?xml version="1.0"?><pluginRecords><update-site name="Test" url="{root.as_uri()}/"/>{"".join(recs)}</pluginRecords>'
    (root / "db.xml.gz").write_bytes(gzip.compress(xml.encode()))
    return root


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "lumen_home"
    monkeypatch.setenv("LUMEN_HOME", str(h))
    saved = dict(REGISTRY.ops)
    yield h
    FP._REGISTERED.clear()
    REGISTRY.ops.clear()
    REGISTRY.ops.update(saved)


def test_plugins_config_parsing(tmp_path):
    jar = _fake_jar(tmp_path / "Demo_.jar", '# comment\nPlugins>Demo, "Demo One", demo.One_\nProcess>Filters, "Demo Two", demo.Two_("fast")\nnot a line\n')
    assert FP.plugin_commands(jar) == [
        {"menu": "Plugins>Demo", "label": "Demo One", "class": "demo.One_", "arg": ""},
        {"menu": "Process>Filters", "label": "Demo Two", "class": "demo.Two_", "arg": "fast"},
    ]
    assert FP.plugin_commands(tmp_path / "missing.jar") == []


def test_update_site_listing_install_dependencies_and_commands(tmp_path, home):
    plugin = _fake_jar(tmp_path / "Demo_.jar", 'Plugins>Demo, "Demo One", demo.One_\n')
    lib = _fake_jar(tmp_path / "demo-lib-1.0.jar", "")
    site = _site(tmp_path / "site", {"plugins/Demo_.jar": plugin, "jars/demo-lib-1.0.jar": lib, "macros/Thing.ijm": tmp_path / "Demo_.jar"},
                 deps={"plugins/Demo_.jar": ["jars/demo-lib-1.0.jar"]})
    recs = FP.read_update_site(str(site))
    assert {r["filename"] for r in recs} == {"plugins/Demo_.jar", "jars/demo-lib-1.0.jar", "macros/Thing.ijm"}  # the removed file is not listed
    s = Session()
    out = s.run_op("plugins.fiji_update_site", {"site": site.as_uri(), "filter": "demo"})
    assert {r[0] for r in out["table"]["rows"]} == {"plugins/Demo_.jar", "jars/demo-lib-1.0.jar"}
    out = s.run_op("plugins.fiji_install", {"site": str(site), "files": "plugins/Demo_.jar"})
    statuses = {r[0]: r[1] for r in out["table"]["rows"]}
    assert statuses == {"plugins/Demo_.jar": "installed", "jars/demo-lib-1.0.jar": "installed (dependency)"}
    assert (home / "jars" / "Demo_.jar").is_file() and (home / "jars" / "demo-lib-1.0.jar").is_file()
    spec = REGISTRY.ops["fiji.demo.demo_one"]
    assert spec.name == "Demo One" and spec.menu == "Plugins/Fiji/Demo" and spec.func.fiji_plugin == "demo.One_"
    rows = s.run_op("plugins.fiji_installed", {})["table"]["rows"]
    assert ["Demo_.jar", "Demo One", "Plugins>Demo", "demo.One_"] in rows and any(r[0] == "demo-lib-1.0.jar" for r in rows)
    # the installed jars are on the Java bridge's classpath
    from lumen.core.readers.fiji_jars import extra_classpath
    assert str(home / "jars" / "Demo_.jar") in extra_classpath()
    with pytest.raises(ValueError, match="not on the update site"):
        s.run_op("plugins.fiji_install", {"site": str(site), "files": "plugins/Nope.jar"})


def test_size_mismatch_is_refused(tmp_path, home):
    plugin = _fake_jar(tmp_path / "Demo_.jar", 'Plugins>Demo, "Demo One", demo.One_\n')
    site = _site(tmp_path / "site", {"plugins/Demo_.jar": plugin})
    (site / "plugins" / "Demo_.jar-20240101000000").write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(ValueError, match="downloaded .* bytes"):
        FP.install_from_site(str(site), ["plugins/Demo_.jar"])
    assert not (home / "jars" / "Demo_.jar").exists()


def test_groovy_runner_uses_a_macro_thread():
    code = FP.groovy_runner("my.Plug_", "", "radius=3")
    assert "Run$_lumen" in code and 'IJ.runPlugIn(imp, "my.Plug_", "")' in code and '"radius=3 "' in code


def _bridge_ready() -> bool:
    try:
        from lumen.core.ijshim import jvm_subprocess
        return jvm_subprocess.available(refresh=True) and AUTO_THRESHOLD is not None and KUWAHARA is not None
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(sys.platform == "win32" or not _bridge_ready(), reason="needs a JDK and an installed Fiji (jars + plugins)")
def test_real_fiji_plugins_from_a_local_update_site(tmp_path, home, monkeypatch):
    monkeypatch.setenv("LUMEN_JVM_MODE", "subprocess")
    site = _site(tmp_path / "site", {"plugins/Auto_Threshold.jar": AUTO_THRESHOLD, "plugins/Kuwahara_Filter.jar": KUWAHARA})
    s = Session()
    s.run_op("plugins.fiji_install", {"site": str(site), "files": "plugins/Auto_Threshold.jar, plugins/Kuwahara_Filter.jar"})
    assert "fiji.auto_threshold.auto_threshold" in REGISTRY.ops and "fiji.kuwahara_filter.kuwahara_filter" in REGISTRY.ops
    # the real Fiji Auto_Threshold plugin on Blobs equals Lumen's own Otsu mask, pixel for pixel
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("fiji.auto_threshold.auto_threshold", {"options": "method=Otsu white"})
    fiji_mask = s.active.data[0, 0, 0].copy()
    ref = Session()
    ref.run_op("file.sample", {"name": "Blobs"})
    ref.run_op("threshold.apply", {"method": "Otsu", "dark_background": True})
    np.testing.assert_array_equal(fiji_mask, ref.active.data[0, 0, 0])
    assert s.histories[s.active.id].labels()["undo"]  # the plugin's change is an undo step
    # a command Lumen does not implement natively, called from an ImageJ macro, runs the installed Fiji plugin
    from lumen.core.ij_compat import IJ
    s2 = Session()
    s2.run_op("file.sample", {"name": "Blobs"})
    before = s2.active.data[0, 0, 0].copy()
    IJ(s2).macro('run("Kuwahara Filter", "sampling=5");')
    after = s2.active.data[0, 0, 0]
    assert after.shape == before.shape and not np.array_equal(after, before)


MORPHOLIBJ = next(iter(sorted((Path.home() / ".lumen" / "jars").glob("MorphoLibJ_*.jar"))), None)


@pytest.mark.skipif(sys.platform == "win32" or MORPHOLIBJ is None or not _bridge_ready(),
                    reason="needs a JDK, an installed Fiji and MorphoLibJ_*.jar in ~/.lumen/jars (Plugins ▸ Utilities ▸ Install from Fiji Update Site…)")
def test_lumen_morphological_filters_match_the_real_morpholibj(tmp_path, home, monkeypatch):
    """Lumen's MorphoLibJ port against the real plugin: exact for disk / square / horizontal / vertical lines and for every
    operation (the Laplacian included); for MorphoLibJ's off-centre octagon and diamond and the diagonal line, exact away
    from a border band of 2r + 2 pixels (MorphoLibJ applies those elements step by step on the image)."""
    monkeypatch.setenv("LUMEN_JVM_MODE", "subprocess")
    site = _site(tmp_path / "site", {"plugins/MorphoLibJ_.jar": MORPHOLIBJ})
    s0 = Session()
    s0.run_op("plugins.fiji_install", {"site": str(site), "files": "plugins/MorphoLibJ_.jar"})
    cases = [("Dilation", "Disk", "Disk", 0.0, 3, True), ("Opening", "Disk", "Disk", 0.0, 2, True), ("Closing", "Square", "Square", 0.0, 3, True),
             ("Laplacian", "Square", "Square", 0.0, 2, True), ("Black Top Hat", "Disk", "Disk", 0.0, 2, True), ("Erosion", "Vertical Line", "Line", 90.0, 3, True),
             ("Gradient", "Line 45 degrees", "Line", 45.0, 3, True), ("Opening", "Octagon", "Octagon", 0.0, 2, False), ("Dilation", "Diamond", "Diamond", 0.0, 3, False)]
    for op, mel, lel, ang, r, whole in cases:
        a = Session()
        a.run_op("file.sample", {"name": "Blobs"})
        a.run_op("fiji.morpholibj.morphological_filters", {"options": f"operation=[{op}] element=[{mel}] radius={r}"})
        ref = list(a.images.values())[-1].data[0, 0, 0]
        b = Session()
        b.run_op("file.sample", {"name": "Blobs"})
        b.run_op("morpho.filters", {"operation": op, "element": lel, "radius": r, "line_angle": ang})
        mine = b.active.data[0, 0, 0]
        m = 2 * r + 2
        if whole:
            np.testing.assert_array_equal(mine, ref, err_msg=f"{op} {mel} r={r}")
        else:
            np.testing.assert_array_equal(mine[m:-m, m:-m], ref[m:-m, m:-m], err_msg=f"{op} {mel} r={r} (interior)")
