"""The out-of-process JVM script runner (lumen.core.ijshim.jvm_subprocess) and the shared JDK / Fiji-jars discovery
(lumen.core.readers.fiji_jars).

Two kinds of tests: mocked ones – a fake ``java`` (``fiji_jars.run_java`` replaced by a recorder that writes what
ImageJ's teardown macro would write) over a fake Fiji ``jars`` folder, no JVM ever starts – and real ones that run
``java`` with the jars of an installed Fiji (skipped when there is no JDK or no Fiji: CI)."""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.ijshim import jvm_bridge as jb
from lumen.core.ijshim import jvm_subprocess as js
from lumen.core.image import Calibration, LumenImage
from lumen.core.io import read_image, write_ij_rois, write_image
from lumen.core.readers import fiji_jars as fj
from lumen.core.session import Session
from lumen.server.app import create_app

REAL = js.available() and not os.environ.get("LUMEN_SKIP_JVM_TESTS")
needs_jvm = pytest.mark.skipif(not REAL, reason="needs a JDK and a Fiji installation's jars (ij.jar + groovy / bsh / rhino)")


# ====================================================================== fixtures
@pytest.fixture
def fake_fiji(monkeypatch, tmp_path):
    """A fake Fiji ``jars`` folder (empty jar files), a fake JDK whose ``java -version`` is never run, subprocess mode."""
    jars = tmp_path / "Fiji" / "jars"
    (jars / "bio-formats").mkdir(parents=True)
    for n in ("ij-1.54p.jar", "ij-1.53t.jar", "groovy-4.0.23.jar", "groovy-json-4.0.23.jar", "bsh-2.1.1.jar", "rhino-1.7.14.jar", "jruby-core-9.1.17.0.jar",
              "jruby-stdlib-9.4.12.0.jar", "clojure-1.12.0.jar", "logback-classic-1.2.12.jar"):
        (jars / n).write_bytes(b"")
    (jars / "bio-formats" / "formats-gpl-8.1.1.jar").write_bytes(b"")
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    exe = jdk / "bin" / ("java.exe" if sys.platform == "win32" else "java")
    exe.write_bytes(b"")
    monkeypatch.setenv("JAVA_HOME", str(jdk))
    monkeypatch.setenv("LUMEN_FIJI_JARS", str(jars))
    monkeypatch.setenv("LUMEN_JVM_MODE", "subprocess")
    for v in ("LUMEN_JVM_CLASSPATH", "LUMEN_IJ_JAR", "LUMEN_JVM_HEAP", "LUMEN_JVM_TIMEOUT", "LUMEN_JVM_KEEP_TEMP"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(fj, "_probe_java", lambda e: 'openjdk version "21.0.2" 2024-01-16' if e == str(exe) else None)
    fj.reset()
    jb.reset()
    yield {"jars": jars, "java": str(exe)}
    fj.reset()
    jb.reset()


class FakeJava:
    """Stands in for ``fiji_jars.run_java``: records the command, inspects the work folder the runner was written
    to and writes what ImageJ's teardown macro would write (bound image inverted, a second image, Results, ROIs, log)."""

    def __init__(self, *, invert=True, extra_image=True, results=True, rois=True, rc=0, stderr="", timed_out=False, script_error=None, no_done=False, close_bound=False):
        self.calls: list[list[str]] = []
        self.works: list[Path] = []
        self.invert, self.extra_image, self.results, self.rois = invert, extra_image, results, rois
        self.rc, self.stderr, self.timed_out, self.script_error, self.no_done, self.close_bound = rc, stderr, timed_out, script_error, no_done, close_bound

    def __call__(self, args, *, cwd=None, timeout=600.0, java=None, env=None):
        cmd = [java or "java"] + [str(a) for a in args]
        self.calls.append(cmd)
        runner = Path(cmd[-1])
        work = runner.parent
        self.works.append(work)
        out = work / "out"
        res = fj.JavaResult(command=cmd, returncode=self.rc, stdout="hello\n", stderr=self.stderr, seconds=0.01, timed_out=self.timed_out,
                            error="the JVM did not finish within 600 s (LUMEN_JVM_TIMEOUT) and was killed" if self.timed_out else None)
        if self.timed_out or self.rc != 0:
            return res
        manifest = ""
        if (work / "image.tif").is_file() and not self.close_bound:
            img = read_image(work / "image.tif")
            m = re.search(r'setTitle\((".*?")\)', runner.read_text(encoding="utf-8"))
            img.name = json.loads(m.group(1).replace("\\$", "$").replace("\\#", "#")) if m else img.name
            if self.invert:
                img.replace_data(255 - img.data if img.data.dtype == np.uint8 else img.data.max() - img.data, rgb=img.rgb)
            write_image(img, out / "bound.tif")
            manifest += f"-2\t{img.name}\t{out / 'bound.tif'}\n"
            if (work / "selection.roi").is_file():
                (out / "selection.roi").write_bytes((work / "selection.roi").read_bytes())
        if self.extra_image:
            write_image(LumenImage(np.full((1, 1, 1, 3, 4), 7, np.uint16), name="second", axes="TZCYX"), out / "image1.tif")
            manifest += f"-3\tsecond\t{out / 'image1.tif'}\n"
        (out / "images.tsv").write_text(manifest, encoding="utf-8")
        if self.results:
            (out / "results.csv").write_text(" ,Label,Area,Mean\n1,blob,12,3.5\n2,,4,NaN\n", encoding="utf-8")
        if self.rois:
            write_ij_rois([{"type": "oval", "x": 1, "y": 2, "w": 3, "h": 4, "name": "o1"}], out / "rois.zip")
        (out / "log.txt").write_text("hello\nfrom the log window\n", encoding="utf-8")
        if self.script_error:
            (out / "error.txt").write_text(self.script_error, encoding="utf-8")
        if not self.no_done:
            (out / "done.txt").write_text("ok", encoding="utf-8")
        return res


def blobs_session(shape=(1, 1, 1, 6, 8), dtype=np.uint8, name="blobs") -> tuple[Session, LumenImage]:
    s = Session()
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    img = s.add_image(LumenImage(data, name=name, axes="TZCYX"))
    return s, img


# ====================================================================== discovery (lumen.core.readers.fiji_jars)
def test_discovery_finds_jars_and_jdk(fake_fiji):
    st = fj.status()
    assert st["java"] == fake_fiji["java"] and "JAVA_HOME" in st["java_detail"] and st["java_version"].startswith("openjdk version")
    assert st["jars"] == str(fake_fiji["jars"])
    found = st["found"]
    assert Path(found["ij"]).name == "ij-1.54p.jar"  # the newest version wins
    assert Path(found["groovy"]).name == "groovy-4.0.23.jar" and Path(found["beanshell"]).name == "bsh-2.1.1.jar" and Path(found["rhino"]).name == "rhino-1.7.14.jar"
    assert Path(found["jruby"]).name == "jruby-core-9.1.17.0.jar" and Path(found["jruby-stdlib"]).name == "jruby-stdlib-9.4.12.0.jar"
    assert Path(found["clojure"]).name == "clojure-1.12.0.jar" and found["clojure-spec"] is None and found["clojure-core-specs"] is None
    assert Path(found["bioformats"]).name == "formats-gpl-8.1.1.jar"
    cp, missing = fj.classpath_for(["ij", "groovy"])
    assert [Path(c).name for c in cp] == ["ij-1.54p.jar", "groovy-4.0.23.jar", "*", "*"] and cp[2] == str(fake_fiji["jars"] / "*") and missing == []
    assert cp[3] == str(fake_fiji["jars"] / "bio-formats" / "*")
    cp, missing = fj.classpath_for(["ij", "clojure", "clojure-spec"])
    assert missing == ["clojure-spec"]


def test_discovery_env_overrides_and_failures(fake_fiji, monkeypatch, tmp_path):
    # an invalid LUMEN_FIJI_JARS means no Fiji – nothing else is searched
    monkeypatch.setenv("LUMEN_FIJI_JARS", str(tmp_path / "nowhere"))
    fj.reset()
    st = fj.status()
    assert st["jars"] is None and "not a Fiji jars folder" in st["jars_detail"] and st["found"]["ij"] is None
    # the Fiji folder itself (not its jars sub-folder) works too
    monkeypatch.setenv("LUMEN_FIJI_JARS", str(fake_fiji["jars"].parent))
    fj.reset()
    assert fj.status()["jars"] == str(fake_fiji["jars"])
    # the user's jars take precedence: an ImageJ-only setup without Fiji
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "ij-1.54q.jar").write_bytes(b"")
    (mine / "groovy-4.0.24.jar").write_bytes(b"")
    monkeypatch.setenv("LUMEN_FIJI_JARS", str(tmp_path / "nowhere"))
    monkeypatch.setenv("LUMEN_JVM_CLASSPATH", str(mine))
    fj.reset()
    st = fj.status()
    assert Path(st["found"]["ij"]).name == "ij-1.54q.jar" and Path(st["found"]["groovy"]).name == "groovy-4.0.24.jar" and st["found"]["beanshell"] is None
    cp, missing = fj.classpath_for(["ij", "groovy"])
    assert [Path(c).name for c in cp] == ["ij-1.54q.jar", "groovy-4.0.24.jar"] and missing == []
    assert fj.expand_classpath(str(mine)) == sorted(str(p) for p in mine.glob("*.jar"))
    assert fj.expand_classpath(str(mine / "groovy-*.jar")) == [str(mine / "groovy-4.0.24.jar")]
    monkeypatch.setenv("LUMEN_IJ_JAR", str(mine / "ij-1.54q.jar"))
    assert fj.extra_classpath()[0] == str(mine / "ij-1.54q.jar")
    # a JDK whose java cannot run (macOS's stub) is skipped and reported
    monkeypatch.setattr(fj, "_probe_java", lambda e: None)
    fj.reset()
    exe, why, ver = fj.find_java()
    assert exe is None and "cannot run" in why and ver == ""
    assert fj.heap_option() == "-Xmx1024m" and fj.heap_option(600 * 1024 * 1024) == "-Xmx2312m" and fj.heap_option(10 ** 11) == "-Xmx8192m"
    monkeypatch.setenv("LUMEN_JVM_HEAP", "3g")
    assert fj.heap_option(10 ** 11) == "-Xmx3g"
    opts = fj.base_options(headless=True, home=tmp_path / "home", heap="-Xmx1g")
    assert "-Dapple.awt.UIElement=true" in opts and "-Djava.awt.headless=true" in opts and f"-Duser.home={tmp_path / 'home'}" in opts and opts[-1] == "-Xmx1g"
    assert (tmp_path / "home" / "plugins").is_dir() and any(o.startswith("-Dplugins.dir=") for o in opts)
    assert fj.filter_stderr("SLF4J: Failed to load class\nreal problem\n\n") == "real problem"


def test_subprocess_status_per_language(fake_fiji):
    st = js.status()
    assert st["available"] and st["ij"].endswith("ij-1.54p.jar") and st["jars"] == str(fake_fiji["jars"])
    langs = st["languages"]
    assert all(langs[k]["available"] for k in ("groovy", "beanshell", "rhino", "jruby"))
    assert not langs["clojure"]["available"] and "clojure-spec" in langs["clojure"]["missing"] and "spec.alpha" in langs["clojure"]["detail"]
    assert "Groovy, BeanShell, JavaScript (Rhino), JRuby" in st["detail"]


# ====================================================================== runner generation
def test_runner_text_and_parameter_literals(tmp_path):
    from lumen.core.ijshim.params import parse
    plist, _ = parse('#@ Double (value=2.5) sigma\n#@ Integer n\n#@ Boolean flag\n#@ String (value="a$b#c") label\n#@ File f\n#@ ImagePlus image\n'
                     '#@ ResultsTable rt\n#@ RoiManager rm\n#@ UIService ui\n')
    vals = {"sigma": 2.5, "n": 3, "flag": True, "label": 'a$b#c"', "f": "/tmp/x y.txt"}
    lits = {lang: {p.name: js.param_literal(lang, p, vals.get(p.name)) for p in plist} for lang in js.LAUNCHERS}
    for lang in js.LAUNCHERS:
        assert lits[lang]["sigma"] == "2.5" and lits[lang]["n"] == "3" and lits[lang]["flag"] == "true" and lits[lang]["image"] == "imp" and lits[lang]["rm"] == "__lumen_rm"
    assert lits["groovy"]["label"] == '"a\\$b#c\\""' and lits["jruby"]["label"] == '"a$b\\#c\\""' and lits["beanshell"]["label"] == '"a$b#c\\""'
    assert lits["groovy"]["f"] == 'new java.io.File("/tmp/x y.txt")' and lits["jruby"]["f"] == 'java.io.File.new("/tmp/x y.txt")' and lits["clojure"]["f"] == '(java.io.File. "/tmp/x y.txt")'
    assert lits["groovy"]["ui"] == "null" and lits["jruby"]["ui"] == "nil" and lits["clojure"]["ui"] == "nil"
    assert lits["rhino"]["rt"] == "Packages.ij.measure.ResultsTable.getResultsTable()" and lits["clojure"]["rt"] == "(ResultsTable/getResultsTable)"
    assert js._float("groovy", float("nan")) == "Double.NaN" and js._float("rhino", float("inf")) == "Infinity" and js._float("jruby", float("-inf")) == "-Float::INFINITY"
    r = js._Run("groovy", tmp_path)
    r.image, r.name, r.roi, r.seed_rois, r.seed_results = str(tmp_path / "image.tif"), "blobs$1", str(tmp_path / "selection.roi"), str(tmp_path / "RoiSet.zip"), str(tmp_path / "results_in.csv")
    r.params = [("sigma", "2.5")]
    text = js._runner_groovy(r)
    assert "Interpreter.batchMode = true" in text and "new RoiManager(true)" in text and 'getDeclaredField("instance")' in text
    assert 'imp.setTitle("blobs\\$1")' in text and "RoiDecoder.open(" in text and '__lumen_rm.runCommand("Open", ' in text and "Analyzer.setResultsTable(ResultsTable.open(" in text
    assert '__lumen_binding.setVariable("sigma", 2.5)' in text and "GroovyShell(this.class.classLoader, __lumen_binding).evaluate(new File(" in text
    assert "ResultsTable.getResultsTable().setPrecision(9)" in text and "IJ.runMacroFile(" in text and text.rstrip().endswith("System.exit(0)")
    for lang, fn in js.RUNNERS.items():
        r2 = js._Run(lang, tmp_path)
        r2.image, r2.name = str(tmp_path / "image.tif"), "img"
        r2.params = [("n", "3")]
        t = fn(r2)
        assert "batchMode" in t and "RoiManager" in t and "setPrecision" in t and "runMacroFile" in t and "exit" in t.lower()
    assert "saveAs(\"Tiff\", p)" in js.TEARDOWN_MACRO and "saveAs(\"Results\"" in js.TEARDOWN_MACRO and 'roiManager("save"' in js.TEARDOWN_MACRO and "done.txt" in js.TEARDOWN_MACRO


def test_build_command(fake_fiji, tmp_path):
    cmd, missing = js.build_command("rhino", str(tmp_path / "runner.js"), home=tmp_path / "home", heap="-Xmx1g")
    assert missing == [] and cmd[0] == fake_fiji["java"] and "-Dapple.awt.UIElement=true" in cmd and "-Djava.awt.headless=false" in cmd and "-Xmx1g" in cmd
    i = cmd.index("-cp")
    cp = cmd[i + 1].split(os.pathsep)
    assert Path(cp[0]).name == "ij-1.54p.jar" and Path(cp[1]).name == "rhino-1.7.14.jar" and cmd[i + 2:] == ["org.mozilla.javascript.tools.shell.Main", "-version", "200", str(tmp_path / "runner.js")]
    assert any(o.startswith("-Dlogback.configurationFile=") for o in cmd) and (tmp_path / "home" / "logback.xml").is_file()
    cmd, missing = js.build_command("clojure", str(tmp_path / "runner.clj"), home=tmp_path / "home", heap="-Xmx1g")
    assert missing == ["clojure-spec", "clojure-core-specs"]
    cmd, _ = js.build_command("jruby", str(tmp_path / "r.rb"), home=tmp_path / "home", heap="-Xmx1g")
    assert cmd[-3:] == ["org.jruby.Main", "--disable-gems", str(tmp_path / "r.rb")] and "jruby-stdlib-9.4.12.0.jar" in cmd[cmd.index("-cp") + 1]


# ====================================================================== Results / ROI helpers
def test_results_csv_roundtrip_and_comparisons(tmp_path):
    rows = [{"Label": "a", "Area": 12, "Mean": 3.25, "Flag": True}, {"Area": 4.0, "Mean": float("nan"), "Note": "x"}]
    seeded = js.write_results_csv(rows, tmp_path / "r.csv")
    text = (tmp_path / "r.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == " ,Label,Area,Mean,Flag,Note" and text.splitlines()[1] == "1,a,12,3.25,True,"
    back = js.read_results_csv(tmp_path / "r.csv")
    assert back == [{"Label": "a", "Area": 12.0, "Mean": 3.25, "Flag": "True"}, {"Label": "", "Area": 4.0, "Note": "x"}]
    assert js.rows_equal(seeded[:1], back[:1]) and not js.rows_equal(seeded, back) and js.rows_equal([{"A": 1.0000000001}], [{"A": 1.0}])
    a = [{"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4, "id": "1", "name": ""}]
    b = [{"type": "rect", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0, "name": "0001", "id": "9"}]
    assert js.rois_equal(a, b) and not js.rois_equal(a, [{**b[0], "w": 5.0}]) and not js.rois_equal([{**a[0], "name": "p"}], b)
    assert not js._cal_changed(Calibration(unit="pixel"), Calibration(unit="")) and js._cal_changed(Calibration(pixel_width=0.5), Calibration(pixel_width=0.25))


# ====================================================================== mocked runs
def test_run_through_a_fake_java(fake_fiji, monkeypatch):
    fake = FakeJava()
    monkeypatch.setattr(fj, "run_java", fake)
    s, img = blobs_session()
    s.results.append([{"Label": "seed", "Area": 1.0}])
    s.roi_manager.add({"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2, "name": "r0"}, image_id=img.id)
    before = img.data.copy()
    code = '#@ Double (value=1.5) sigma\nimport ij.IJ\nIJ.run(imp, "Invert", "")\n'
    out = jb.run_jvm_script(s, code, "groovy", {"sigma": 2.25}, roi={"type": "rect", "x": 1, "y": 1, "w": 3, "h": 2})
    assert out["error"] is None and out["mode"] == "subprocess" and out["language"] == "groovy" and out["changed"] is True
    assert out["stdout"] == "hello\nfrom the log window\n" and out["parameters"][0]["name"] == "sigma"
    assert np.array_equal(img.data, 255 - before) and [sn.label for sn in s.histories[img.id].undo] == ["Groovy script"]
    assert len(out["new_images"]) == 1 and s.images[out["new_images"][0]].name == "second" and s.images[out["new_images"][0]].path is None
    assert s.results.rows == [{"Label": "blob", "Area": 12.0, "Mean": 3.5}, {"Label": "", "Area": 4.0}] and out["results_rows"] == 2
    assert [(r["type"], r.get("name")) for r in s.roi_manager.rois] == [("oval", "o1")] and out["rois_copied"] == 1
    assert out["roi"]["type"] == "rect" and out["roi"]["w"] == 3.0 and out["roi"]["x"] == 1.0
    # the command line and the files the runner got
    cmd = fake.calls[0]
    assert cmd[0] == fake_fiji["java"] and "groovy.ui.GroovyMain" in cmd and cmd[-1].endswith("runner.groovy") and "-Dapple.awt.UIElement=true" in cmd
    assert any(o.startswith("-Duser.home=") for o in cmd) and any(o.startswith("-Dplugins.dir=") for o in cmd) and "-Xmx1024m" in cmd
    work = fake.works[0]
    assert not work.exists()  # the temp folder is removed after the run
    assert "Log" not in s.logs[-1]["message"] if hasattr(s, "logs") and s.logs else True


def test_run_keeps_temp_and_passes_seeds(fake_fiji, monkeypatch):
    fake = FakeJava(invert=False, extra_image=False, results=False, rois=False)
    monkeypatch.setattr(fj, "run_java", fake)
    monkeypatch.setenv("LUMEN_JVM_KEEP_TEMP", "1")
    s, img = blobs_session()
    s.results.append([{"Label": "seed", "Area": 1.0, "Ok": True}])
    s.roi_manager.add({"type": "oval", "x": 1, "y": 1, "w": 2, "h": 2, "name": "seedroi"}, image_id=img.id)
    out = jb.run_jvm_script(s, 'print("x");', "beanshell", roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2})
    work = fake.works[0]
    try:
        assert out["error"] is None and out["changed"] is False and out["new_images"] == [] and not s.histories[img.id].undo
        assert (work / "results_in.csv").read_text(encoding="utf-8").splitlines() == [" ,Label,Area,Ok", "1,seed,1,True"]
        assert zipfile.is_zipfile(work / "RoiSet.zip") and (work / "selection.roi").is_file() and (work / "image.tif").is_file()
        runner = (work / "runner.bsh").read_text(encoding="utf-8")
        assert 'source("' in runner and "RoiSet.zip" in runner and "results_in.csv" in runner and (work / "script.bsh").read_text(encoding="utf-8") == 'print("x");'
        # no results.csv / rois.zip came back although both were seeded: the script emptied them, so the session's are emptied too
        assert s.results.rows == [] and s.roi_manager.rois == [] and out["results_rows"] == 0 and out["rois_copied"] == 0
        assert "bsh.Interpreter" in fake.calls[0]
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_run_failures_are_reported(fake_fiji, monkeypatch):
    s, img = blobs_session()
    monkeypatch.setattr(fj, "run_java", FakeJava(rc=1, stderr="SLF4J: noise\nError: Could not find or load main class x\n"))
    out = jb.run_jvm_script(s, "1", "rhino")
    assert "exited with code 1" in out["error"] and "Could not find or load main class" in out["error"] and "SLF4J" not in out["error"] and out["changed"] is False
    monkeypatch.setattr(fj, "run_java", FakeJava(timed_out=True))
    out = jb.run_jvm_script(s, "1", "groovy")
    assert "did not finish" in out["error"] and "LUMEN_JVM_TIMEOUT" in out["error"]
    monkeypatch.setattr(fj, "run_java", FakeJava(script_error="groovy.lang.MissingMethodException: No signature of method: script.boom() (line 2)"))
    out = jb.run_jvm_script(s, "boom()", "groovy")
    assert out["error"].startswith("groovy.lang.MissingMethodException") and "(line 2)" in out["error"] and out["changed"] is True  # partial results still come back
    monkeypatch.setattr(fj, "run_java", FakeJava(no_done=True))
    out = jb.run_jvm_script(s, "System.exit(0)", "groovy")
    assert "System.exit" in out["error"] and out["new_images"] == []
    monkeypatch.setattr(fj, "run_java", FakeJava(close_bound=True, extra_image=False))
    out = jb.run_jvm_script(s, "imp.close()", "groovy")
    assert out["error"] is None and out["roi"] is None and "closed" in out["note"]
    out = jb.run_jvm_script(s, "1", "clojure")
    assert "spec.alpha" in out["error"] and "Clojure" in out["error"]
    out = jb.run_jvm_script(s, "1", "cobol")
    assert "Unknown JVM script language" in out["error"]


def test_unavailable_subprocess_mode_explains_both_ways(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_JVM_MODE", "subprocess")
    monkeypatch.setenv("LUMEN_FIJI_JARS", str(tmp_path / "nowhere"))
    monkeypatch.delenv("LUMEN_JVM_CLASSPATH", raising=False)
    monkeypatch.delenv("LUMEN_IJ_JAR", raising=False)
    monkeypatch.setattr(fj, "_probe_java", lambda e: None)
    fj.reset()
    jb.reset()
    try:
        st = jb.status()
        assert st["effective_mode"] == "subprocess" and not st["available"] and "lumen-imaging[jvm]" in st["install"] and "JDK" in st["install"]
        out = jb.run_jvm_script(Session(), 'println "hi"', "groovy")
        assert "Groovy" in out["error"] and "JDK" in out["error"] and "Fiji" in out["error"] and "lumen-imaging[jvm]" in out["error"] and out["stdout"] == ""
    finally:
        fj.reset()
        jb.reset()


def test_mode_selection(monkeypatch):
    for value, want in (("", "auto"), ("AUTO", "auto"), ("subprocess", "subprocess"), ("fiji", "subprocess"), ("inprocess", "inprocess"), ("jpype", "inprocess"), ("bogus", "auto")):
        monkeypatch.setenv("LUMEN_JVM_MODE", value)
        assert jb.jvm_mode() == want
    monkeypatch.setenv("LUMEN_JVM_MODE", "inprocess")
    assert jb.effective_mode() == "inprocess"
    monkeypatch.setenv("LUMEN_JVM_MODE", "subprocess")
    assert jb.effective_mode() == "subprocess"
    monkeypatch.setenv("LUMEN_JVM_MODE", "auto")
    monkeypatch.setattr(jb.sys, "platform", "darwin")
    monkeypatch.setattr(jb, "_importable", lambda name: True)
    assert jb.effective_mode() == "subprocess"  # macOS: always the subprocess
    monkeypatch.setattr(jb.sys, "platform", "linux")
    monkeypatch.setattr(jb, "_importable", lambda name: False)
    assert jb.effective_mode() == "subprocess"  # no jpype
    monkeypatch.setattr(jb, "_importable", lambda name: True)
    monkeypatch.setattr(jb, "_java_available", lambda: (True, "fake java", None))
    assert jb.effective_mode() == "inprocess"  # jpype + a JDK: in-process
    monkeypatch.setattr(jb, "_java_available", lambda: (False, "no JDK", None))
    monkeypatch.setattr(js, "available", lambda refresh=False: True)
    assert jb.effective_mode() == "subprocess"  # in-process lacks a JDK but the subprocess has one
    monkeypatch.setattr(js, "available", lambda refresh=False: False)
    assert jb.effective_mode() == "inprocess"


def test_status_op_and_api_in_subprocess_mode(fake_fiji, monkeypatch):
    monkeypatch.setattr(fj, "run_java", FakeJava(extra_image=False, results=False, rois=False))
    s, img = blobs_session()
    st = jb.status(refresh=True)
    assert st["effective_mode"] == "subprocess" and st["available"] and st["languages"] == ["groovy", "beanshell", "rhino", "jruby"] and st["subprocess"]["available"]
    assert st["inprocess"]["available"] is False or st["inprocess"]["package"]
    res = s.run_op("script.jvm_bridge_status", {})
    rows = {r[0]: r[1] for r in res["table"]["rows"]}
    assert rows["Mode"].startswith("subprocess") and rows["Subprocess mode"].startswith("available") and rows["Fiji jars"].startswith("Fiji jars at")
    assert rows["Groovy (groovy)"] == "ready – groovy-4.0.23.jar" and "spec.alpha" in rows["Clojure (clojure)"] and rows["ij.jar"].endswith("ij-1.54p.jar")
    assert "Java bridge available" in res["message"]
    res = s.run_op("script.run_jvm_script", {"code": 'IJ.run(imp, "Invert", "")', "language": "groovy"}, image_id=img.id)
    assert res["message"] == "Groovy script finished – image changed"
    c = TestClient(create_app(s))
    r = c.post("/api/script/run", json={"code": 'println "hi"', "language": "rhino", "roi": {"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["error"] is None and j["mode"] == "subprocess" and j["language"] == "rhino" and j["stdout"].startswith("hello") and j["roi"]["w"] == 2.0


# ====================================================================== real runs (a JDK + Fiji's jars on this machine; skipped in CI)
def _blobs() -> tuple[Session, LumenImage, np.ndarray]:
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    assert img is not None
    return s, img, img.data.copy()


@needs_jvm
@pytest.mark.parametrize("lang", [lang for lang in ("groovy", "beanshell", "rhino", "jruby") if js.status()["languages"][lang]["available"]])
def test_real_blur_and_measure_match_lumen(lang):
    """Gaussian Blur σ=2 on the Blobs sample through the real ImageJ equals Lumen's bit-exact port, the Measure row comes back."""
    s, img, orig = _blobs()
    ref = Session()
    ref.run_op("file.sample", {"name": "Blobs"})
    ref.run_op("filters.gaussian", {"sigma": 2.0})
    expected = ref.active.data
    code = {"groovy": '#@ Double (value=2.0) sigma\nimport ij.IJ\nIJ.run(imp, "Gaussian Blur...", "sigma=" + sigma)\nprintln "mean=" + imp.getStatistics().mean\nIJ.run(imp, "Measure", "")\nIJ.log("logged")\nimp.setRoi(10, 20, 50, 40)\n',
            "beanshell": 'IJ.run(imp, "Gaussian Blur...", "sigma=2");\nprint("mean=" + imp.getStatistics().mean);\nIJ.run(imp, "Measure", "");\nIJ.log("logged");\nimp.setRoi(new OvalRoi(5, 6, 30, 20));\n',
            "rhino": 'IJ.run(imp, "Gaussian Blur...", "sigma=2");\nprint("mean=" + imp.getStatistics().mean);\nIJ.run(imp, "Measure", "");\nIJ.log("logged");\nvar sq = [1, 2].map(v => v * v); print("es6 " + sq);\n',
            "jruby": 'IJ.run(imp, "Gaussian Blur...", "sigma=2")\nputs "mean=#{imp.getStatistics.mean}"\nIJ.run(imp, "Measure", "")\nIJ.log("logged")\n'}[lang]
    out = jb.run_jvm_script(s, code, lang, {"sigma": 2.0})
    assert out["error"] is None, out["error"]
    assert out["mode"] == "subprocess" and out["changed"] is True and np.array_equal(img.data, expected)
    assert "mean=84.5293121337" in out["stdout"] and "logged" in out["stdout"] and out["seconds"] < 60
    assert len(s.results.rows) == 1 and s.results.rows[0]["Area"] == 65536.0 and s.results.rows[0]["Mean"] == pytest.approx(84.529312134, abs=1e-6)
    assert [sn.label for sn in s.histories[img.id].undo] == [f"{jb.LANGUAGES[lang]['label']} script"]
    if lang == "groovy":
        assert out["roi"] == {"type": "rect", "x": 10.0, "y": 20.0, "w": 50.0, "h": 40.0} and out["parameters"][0]["name"] == "sigma"
    if lang == "beanshell":
        assert out["roi"]["type"] == "oval" and out["roi"]["w"] == 30.0
    if lang == "rhino":
        assert "es6 1,4" in out["stdout"] and out["roi"] is None
    s.undo(img.id)
    assert np.array_equal(s.active.data, orig)


@needs_jvm
def test_real_selection_limits_the_filter_and_measure():
    s, img, orig = _blobs()
    out = jb.run_jvm_script(s, 'import ij.IJ\nIJ.run(imp, "Gaussian Blur...", "sigma=2")\nIJ.run(imp, "Measure", "")\n', "groovy", roi={"type": "rect", "x": 1, "y": 2, "w": 30, "h": 20})
    assert out["error"] is None, out["error"]
    plane, before = img.data[0, 0, 0], orig[0, 0, 0]
    assert s.results.rows[0]["Area"] == 600.0 and out["roi"] == {"type": "rect", "x": 1.0, "y": 2.0, "w": 30.0, "h": 20.0}  # no 'selection' file name leaking in
    assert not np.array_equal(plane[2:22, 1:31], before[2:22, 1:31]) and np.array_equal(plane[30:], before[30:]) and np.array_equal(plane[:, 40:], before[:, 40:])


@needs_jvm
def test_real_seeds_new_image_rename_error_and_roi_manager():
    s, img, _orig = _blobs()
    s.results.append([{"Label": "seed", "Area": 1.5, "Mean": 2.0}])
    s.roi_manager.add({"type": "rect", "x": 3, "y": 4, "w": 20, "h": 10, "name": "r1"}, image_id=img.id)
    code = ('import ij.*\nimport ij.plugin.frame.RoiManager\nrm = RoiManager.getRoiManager()\nprintln "rm count " + rm.getCount() + " visible " + rm.isVisible()\n'
            'rm.addRoi(new ij.gui.OvalRoi(1, 1, 8, 9))\nrt = ij.measure.ResultsTable.getResultsTable()\nprintln "seed label " + rt.getLabel(0)\n'
            'rt.incrementCounter(); rt.addValue("Area", 7)\nIJ.createImage("second", "16-bit ramp", 64, 32, 3).show()\nimp.setTitle("renamed")\nthis_is_an_error()\n')
    out = jb.run_jvm_script(s, code, "groovy")
    assert "MissingMethodException" in out["error"] and "this_is_an_error" in out["error"] and "(line 11)" in out["error"]
    assert "rm count 1 visible false" in out["stdout"] and "seed label seed" in out["stdout"]
    assert [(s.images[i].name, s.images[i].shape, str(s.images[i].dtype)) for i in out["new_images"]] == [("second", (1, 3, 1, 32, 64), "uint16")]
    assert s.results.rows[0] == {"Label": "seed", "Area": 1.5, "Mean": 2.0} and s.results.rows[1]["Area"] == 7.0 and len(s.results.rows) == 2
    assert [(r["type"], r.get("name")) for r in s.roi_manager.rois][0] == ("rect", "r1") and s.roi_manager.rois[1]["type"] == "oval" and out["rois_copied"] == 2
    assert img.name == "renamed" and [sn.label for sn in s.histories[img.id].undo] == ["Groovy script"]
    s2 = Session()
    out = jb.run_jvm_script(s2, 'println "no image: " + (imp == null)\nij.IJ.createImage("made", "8-bit black", 10, 10, 1).show()', "groovy")
    assert out["error"] is None and "no image: true" in out["stdout"] and [s2.images[i].name for i in out["new_images"]] == ["made"] and "roi" not in out


@needs_jvm
def test_real_calibrated_hyperstack_and_rgb_roundtrip():
    s = Session()
    data = np.random.default_rng(0).integers(0, 4000, (2, 3, 2, 16, 20)).astype(np.uint16)
    img = s.add_image(LumenImage(data, name="stack", axes="TZCYX", calibration=Calibration(pixel_width=0.25, pixel_height=0.25, pixel_depth=1.5, unit="µm", frame_interval=2.0)))
    out = jb.run_jvm_script(s, 'import ij.IJ\nIJ.run(imp, "Add...", "value=5 stack")\nprintln imp.getNChannels() + "," + imp.getNSlices() + "," + imp.getNFrames() + " " + imp.getCalibration().getUnit()\n', "groovy")
    assert out["error"] is None, out["error"]
    assert "2,3,2 " in out["stdout"] and ("micron" in out["stdout"] or "µm" in out["stdout"])
    assert np.array_equal(img.data, np.minimum(data.astype(np.int64) + 5, 65535).astype(np.uint16)) and img.shape == (2, 3, 2, 16, 20)
    assert img.calibration.unit == "µm" and img.calibration.pixel_width == pytest.approx(0.25) and img.calibration.pixel_depth == pytest.approx(1.5) and img.calibration.frame_interval == pytest.approx(2.0)
    rgb = np.random.default_rng(1).integers(0, 255, (1, 1, 3, 8, 10)).astype(np.uint8)
    img2 = s.add_image(LumenImage(rgb, name="rgb", axes="TZCYX", rgb=True))
    out = jb.run_jvm_script(s, 'IJ.run(imp, "Invert", "");\nprint(imp.getBitDepth());\n', "beanshell")
    assert out["error"] is None and "24" in out["stdout"] and img2.rgb and np.array_equal(img2.data, 255 - rgb)
    out = jb.run_jvm_script(s, 'IJ.run(imp, "Invert", "");', "beanshell")  # unchanged script → the same pixels again, still a change
    assert out["changed"] is True and np.array_equal(img2.data, rgb)
