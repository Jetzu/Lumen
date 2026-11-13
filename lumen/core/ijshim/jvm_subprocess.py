"""Out-of-process JVM script runner: Groovy / BeanShell / Rhino JavaScript (and JRuby / Clojure when their jars are
complete) with the **real ImageJ 1.x**, run as a ``java`` subprocess over a Fiji installation's jars – no Python
package, no JPype, no in-process JVM, no Maven download.  This is the mode that works on macOS (where AWT wants the
process's main thread, so an in-process JVM cannot run ImageJ commands) and the default wherever ``jpype`` is not
installed; :mod:`.jvm_bridge` selects it (``LUMEN_JVM_MODE=subprocess|inprocess|auto``).

How a run works (one ``java`` process per script, ~1 s of overhead on this hardware):

1. The active image is written to a temp ImageJ TIFF (``lumen.core.io.write_image``: calibration, hyperstack
   dimensions, display range), the viewport selection as a ``.roi``, the session's ROI Manager entries as a
   ``RoiSet.zip`` and its Results rows as a CSV (ImageJ's own formats, through ``lumen.core.io``).
2. A runner is generated for the language.  It sets ``ij.macro.Interpreter.batchMode = true`` (what ``-batch``
   does: ``imp.show()`` registers images with the ``WindowManager`` without opening windows), installs a *hidden*
   ``RoiManager`` as the instance (``new RoiManager(true)`` + reflection – ``RoiManager.getRoiManager()`` would
   otherwise open a window), seeds the Results table / ROI Manager, opens the image (``IJ.openImage`` + ``show()``,
   the selection restored with ``RoiDecoder``), binds ``imp`` and the ``#@`` parameters as variables and runs the
   user's script file (``GroovyShell.evaluate`` / BeanShell ``source()`` / Rhino ``load()`` / Ruby ``eval`` with the
   runner's binding / ``load-file``).
3. A teardown **macro** (language-independent, ``IJ.runMacroFile``) saves every open image as TIFF (the bound one as
   ``bound.tif``, its selection as ``selection.roi``), the Results table as CSV (``setPrecision(9)`` first), the
   ROI Manager as a zip and the Log window (if any) to a file; the runner then ``System.exit(0)``-s.
4. ``java -Dapple.awt.UIElement=true -Duser.home=<tmp> -Dij.dir=<tmp> -Dplugins.dir=<empty> -Xmx<heap> -cp
   <ij.jar>:<language jar>:<extras>:<Fiji jars/*>:<jars/bio-formats/*> <main class> runner`` runs with stdin closed,
   a timeout (``LUMEN_JVM_TIMEOUT``, 600 s) and ImageJ's prefs isolated (no IJ_Prefs.txt of the user, no plugin
   scan of Fiji's plugins folder).  Fiji's whole ``jars`` folder is on the classpath (as wildcards, expanded by the
   launcher) so JRuby's dependencies and the Bio-Formats API (``loci.formats``) are available to scripts.
5. The outputs are read back into the session: the bound image in place (with an undo entry) when its pixels,
   calibration or title changed, other images as new session images, the final selection, Results rows and ROI
   Manager entries when they changed, and the output / ``IJ.log`` lines (ImageJ without a Log window prints
   ``IJ.log`` to stdout) into the Log section.

Launchers: ``groovy.ui.GroovyMain`` (``groovy-4.x.jar``), ``bsh.Interpreter`` (``bsh-2.x.jar``),
``org.mozilla.javascript.tools.shell.Main -version 200`` (``rhino-1.7.x.jar``, ES6), ``org.jruby.Main --disable-gems``
(``jruby-core`` + ``jruby-stdlib``; Fiji's 9.1 core / 9.4 stdlib pair only runs with gems disabled) and
``clojure.main`` (``clojure-1.x.jar`` **plus** ``spec.alpha`` and ``core.specs.alpha``, which Fiji does not ship –
put them on ``LUMEN_JVM_CLASSPATH``).  Jars are found by :mod:`lumen.core.readers.fiji_jars` (``LUMEN_FIJI_JARS``,
``LUMEN_JVM_CLASSPATH``, ``LUMEN_IJ_JAR``, the JDK through ``JAVA_HOME`` / ``PATH`` / Homebrew / Temurin paths).

Limits: ImageJ dialogs / windows do not exist (``GenericDialog`` would block – use ``#@`` parameters), a script
that calls ``System.exit`` / ``IJ.run("Quit")`` ends the run before the results are collected, ``IJ.runMacro``
inside a script resets ImageJ's batch-mode image list (ImageJ behaviour), the Results table is echoed to the
output when ``Measure`` runs (ImageJ without a Results window does that), and one script runs at a time.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..readers import fiji_jars

ENV_MODE = "LUMEN_JVM_MODE"
ENV_TIMEOUT = "LUMEN_JVM_TIMEOUT"
ENV_KEEP = "LUMEN_JVM_KEEP_TEMP"
ALT_HINT = 'or pip install "lumen-imaging[jvm]" for the in-process mode (Linux / Windows; LUMEN_JVM_MODE=inprocess)'

#: per language: launcher main class, its own arguments, the jar components it needs (in classpath order), the
#: optional ones (added when found), the script extension and a note shown in the status table
LAUNCHERS: dict[str, dict[str, Any]] = {
    "groovy": {"label": "Groovy", "main": "groovy.ui.GroovyMain", "args": [], "jars": ["ij", "groovy"], "optional": [], "ext": ".groovy",
               "note": "groovy-4.x.jar (Fiji ships it)"},
    "beanshell": {"label": "BeanShell", "main": "bsh.Interpreter", "args": [], "jars": ["ij", "beanshell"], "optional": [], "ext": ".bsh",
                  "note": "bsh-2.x.jar (Fiji ships it)"},
    "rhino": {"label": "JavaScript (Rhino)", "main": "org.mozilla.javascript.tools.shell.Main", "args": ["-version", "200"], "jars": ["ij", "rhino"],
              "optional": [], "ext": ".js", "note": "rhino-1.7.x.jar (Fiji ships it); ES6 syntax"},
    "jruby": {"label": "JRuby", "main": "org.jruby.Main", "args": ["--disable-gems"], "jars": ["ij", "jruby"], "optional": ["jruby-stdlib"], "ext": ".rb",
              "note": "jruby-core + jruby-stdlib (Fiji ships them; gems disabled – its core and stdlib versions differ)"},
    "clojure": {"label": "Clojure", "main": "clojure.main", "args": [], "jars": ["ij", "clojure", "clojure-spec", "clojure-core-specs"], "optional": [],
                "ext": ".clj", "note": "clojure-1.x.jar plus spec.alpha and core.specs.alpha (not in Fiji: put them on LUMEN_JVM_CLASSPATH); not exercised"},
}
_IJ_PACKAGES = ("ij", "ij.gui", "ij.process", "ij.measure", "ij.util", "ij.plugin", "ij.plugin.filter", "ij.plugin.frame", "ij.io", "ij.text")
_JAVA_PACKAGES = ("java.awt", "java.awt.image", "java.awt.geom", "java.util", "java.io")

#: the language-independent teardown: every image as TIFF (the bound one as bound.tif + selection.roi), the Results
#: table as CSV, the ROI Manager as zip, the Log window as text, then a done marker
TEARDOWN_MACRO = '''args = split(getArgument(), "\\t");
dir = args[0]; bound = parseInt(args[1]);
manifest = "";
n = nImages;
ids = newArray(n);
for (i=1; i<=n; i++) { selectImage(i); ids[i-1] = getImageID(); }
k = 0;
for (i=0; i<n; i++) {
  selectImage(ids[i]);
  t = getTitle();
  if (ids[i]==bound) {
    p = dir + "/bound.tif";
    if (selectionType()>=0) saveAs("Selection", dir + "/selection.roi");
  } else {
    k = k + 1;
    p = dir + "/image" + k + ".tif";
  }
  saveAs("Tiff", p);
  manifest = manifest + ids[i] + "\\t" + t + "\\t" + p + "\\n";
}
File.saveString(manifest, dir + "/images.tsv");
if (nResults>0) saveAs("Results", dir + "/results.csv");
if (roiManager("count")>0) roiManager("save", dir + "/rois.zip");
lg = getInfo("log"); if (lengthOf(lg)>0) File.saveString(lg, dir + "/log.txt");
File.saveString("ok", dir + "/done.txt");
'''

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------- availability
def language_status(lang: str, st: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{"available", "detail", "jar", "missing"}`` for one language on the current discovery."""
    st = fiji_jars.status() if st is None else st
    spec = LAUNCHERS[lang]
    found = st["found"]
    missing = [c for c in spec["jars"] if not found.get(c)]
    jar = found.get(spec["jars"][1]) if len(spec["jars"]) > 1 else None
    if st["java"] is None:
        detail = st["java_detail"]
    elif missing:
        detail = "missing jar(s): " + ", ".join(missing) + f" – {spec['note']}"
    else:
        detail = f"ready ({Path(jar).name if jar else 'jar found'})"
    return {"available": st["java"] is not None and not missing, "detail": detail, "jar": jar, "missing": missing}


def status(refresh: bool = False) -> dict[str, Any]:
    """``{"available", "detail", "install", "java", "java_detail", "java_version", "jars", "jars_detail", "ij", "languages": {lang: …}}``."""
    st = fiji_jars.status(refresh)
    langs = {lang: language_status(lang, st) for lang in LAUNCHERS}
    ij = st["found"].get("ij")
    ready = [lang for lang, s in langs.items() if s["available"]]
    available = st["java"] is not None and bool(ij) and bool(ready)
    if available:
        detail = f"{st['java_version'] or 'java'} ({st['java_detail']}); {Path(ij).name} + {', '.join(LAUNCHERS[l]['label'] for l in ready)} from {st['jars'] or 'LUMEN_JVM_CLASSPATH'}"
    elif st["java"] is None:
        detail = f"{st['java_detail']} – {fiji_jars.JDK_HINT}"
    elif not ij:
        detail = f"no ij.jar found ({st['jars_detail']}) – {fiji_jars.JARS_HINT}"
    else:
        detail = f"no script-language jar next to {ij} – {fiji_jars.JARS_HINT}"
    return {"available": available, "detail": detail, "install": f"{st['install']}; {ALT_HINT}", "java": st["java"], "java_detail": st["java_detail"], "java_version": st["java_version"],
            "jars": st["jars"], "jars_detail": st["jars_detail"], "ij": ij, "languages": langs, "extra": st["extra"]}


def available(refresh: bool = False) -> bool:
    return bool(status(refresh)["available"])


def reset() -> None:
    fiji_jars.reset()


# ---------------------------------------------------------------------------- literals per language
_NULL = {"jruby": "nil", "clojure": "nil"}


def _str(lang: str, s: Any) -> str:
    """A string literal valid in the language (JSON escapes; ``$`` escaped for Groovy's GStrings, ``#`` for Ruby)."""
    lit = json.dumps(str(s))
    if lang == "groovy":
        lit = lit.replace("$", "\\$")
    elif lang == "jruby":
        lit = lit.replace("#", "\\#")
    return lit


def _float(lang: str, f: float) -> str:
    if math.isnan(f):
        return {"rhino": "NaN", "jruby": "Float::NAN", "clojure": "Double/NaN"}.get(lang, "Double.NaN")
    if math.isinf(f):
        pos = f > 0
        return {"rhino": "Infinity" if pos else "-Infinity", "jruby": "Float::INFINITY" if pos else "-Float::INFINITY",
                "clojure": "Double/POSITIVE_INFINITY" if pos else "Double/NEGATIVE_INFINITY"}.get(lang, "Double.POSITIVE_INFINITY" if pos else "Double.NEGATIVE_INFINITY")
    return repr(float(f))


def _file(lang: str, path: str) -> str:
    s = _str(lang, path)
    return {"jruby": f"java.io.File.new({s})", "clojure": f"(java.io.File. {s})"}.get(lang, f"new java.io.File({s})")


def _table(lang: str) -> str:
    return {"rhino": "Packages.ij.measure.ResultsTable.getResultsTable()", "jruby": "ResultsTable.getResultsTable",
            "clojure": "(ResultsTable/getResultsTable)"}.get(lang, "ResultsTable.getResultsTable()")


def param_literal(lang: str, p: Any, v: Any) -> str:
    """The value of a ``#@`` parameter as a literal / expression of the language (``imp`` for images, the Results
    table, the hidden ROI Manager, ``java.io.File`` for files, ``null`` for SciJava services)."""
    null = _NULL.get(lang, "null")
    kind = p.kind
    if kind == "image":
        return "imp"
    if kind == "table":
        return _table(lang)
    if kind == "roimanager":
        return "__lumen_rm"
    if kind == "service" or v is None:
        return null
    if kind in ("file", "directory"):
        return _file(lang, str(v))
    if kind == "int":
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return null
    if kind == "float":
        try:
            return _float(lang, float(v))
        except (TypeError, ValueError):
            return null
    if kind == "bool":
        return "true" if v else "false"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _float(lang, v)
    if isinstance(v, str):
        return _str(lang, v)
    return null


# ---------------------------------------------------------------------------- the runners
class _Run:
    """What a runner needs to know (all paths absolute, inside the temp folder)."""

    def __init__(self, lang: str, work: Path) -> None:
        self.lang = lang
        self.work = work
        self.out = work / "out"
        self.image: str | None = None
        self.name = ""
        self.roi: str | None = None
        self.seed_rois: str | None = None
        self.seed_results: str | None = None
        self.script = str(work / ("script" + LAUNCHERS[lang]["ext"]))
        self.teardown = str(work / "teardown.ijm")
        self.error = str(self.out / "error.txt")
        self.params: list[tuple[str, str]] = []  # (name, literal)
        self.roi_name: str | None = None  # the name of the selection passed in (RoiDecoder names an unnamed .roi after its file)

    def s(self, v: Any) -> str:
        return _str(self.lang, v)


def _runner_groovy(r: _Run) -> str:
    L = ["import ij.IJ", "import ij.ImagePlus", "import ij.WindowManager", "import ij.macro.Interpreter", "import ij.measure.ResultsTable",
         "import ij.plugin.frame.RoiManager", "import ij.plugin.filter.Analyzer", "import ij.io.RoiDecoder", "",
         "Interpreter.batchMode = true",
         "def __lumen_rm = new RoiManager(true)",
         'try { def __lumen_f = Class.forName("ij.plugin.frame.RoiManager").getDeclaredField("instance"); __lumen_f.setAccessible(true); __lumen_f.set(null, __lumen_rm) } catch (Throwable __lumen_t) { }']
    if r.seed_rois:
        L.append(f'__lumen_rm.runCommand("Open", {r.s(r.seed_rois)})')
    if r.seed_results:
        L.append(f"Analyzer.setResultsTable(ResultsTable.open({r.s(r.seed_results)}))")
    L.append("def imp = null")
    if r.image:
        L.append(f"imp = IJ.openImage({r.s(r.image)}); imp.setTitle({r.s(r.name)}); imp.show()")
        if r.roi:
            L.append(f"imp.setRoi(RoiDecoder.open({r.s(r.roi)}))")
    L += ["def __lumen_bound = imp == null ? 0 : imp.getID()", "def __lumen_binding = new Binding()", '__lumen_binding.setVariable("imp", imp)',
          '__lumen_binding.setVariable("IJ", IJ)', '__lumen_binding.setVariable("WindowManager", WindowManager)']
    L += [f"__lumen_binding.setVariable({r.s(n)}, {lit})" for n, lit in r.params]
    L += ["def __lumen_err = null",
          "try {", f"    new GroovyShell(this.class.classLoader, __lumen_binding).evaluate(new File({r.s(r.script)}))",
          "} catch (Throwable __lumen_t) { __lumen_err = __lumen_t }",
          "if (__lumen_err != null) {",
          "    def __lumen_msg = __lumen_err.toString()",
          f"    for (__lumen_e in __lumen_err.getStackTrace()) {{ if (__lumen_e.getFileName() == {r.s(Path(r.script).name)} && __lumen_e.getLineNumber() > 0) {{ __lumen_msg += \" (line \" + __lumen_e.getLineNumber() + \")\"; break }} }}",
          f'    new File({r.s(r.error)}).write(__lumen_msg, "UTF-8")',
          "}",
          "try { ResultsTable.getResultsTable().setPrecision(9) } catch (Throwable __lumen_t) { }",
          f'IJ.runMacroFile({r.s(r.teardown)}, {r.s(str(r.out))} + "\\t" + __lumen_bound)',
          "System.out.flush()", "System.exit(0)", ""]
    return "\n".join(L)


def _runner_beanshell(r: _Run) -> str:
    L = ["".join(f"import {p}.*; " for p in _IJ_PACKAGES), "".join(f"import {p}.*; " for p in _JAVA_PACKAGES),
         "ij.macro.Interpreter.batchMode = true;",
         "__lumen_rm = new RoiManager(true);",
         'try { __lumen_f = Class.forName("ij.plugin.frame.RoiManager").getDeclaredField("instance"); __lumen_f.setAccessible(true); __lumen_f.set(null, __lumen_rm); } catch (Exception __lumen_t) { }']
    if r.seed_rois:
        L.append(f'__lumen_rm.runCommand("Open", {r.s(r.seed_rois)});')
    if r.seed_results:
        L.append(f"Analyzer.setResultsTable(ResultsTable.open({r.s(r.seed_results)}));")
    L.append("imp = null;")
    if r.image:
        L.append(f"imp = IJ.openImage({r.s(r.image)}); imp.setTitle({r.s(r.name)}); imp.show();")
        if r.roi:
            L.append(f"imp.setRoi(RoiDecoder.open({r.s(r.roi)}));")
    L.append("__lumen_bound = (imp == null) ? 0 : imp.getID();")
    L += [f"{n} = {lit};" for n, lit in r.params]
    L += ["__lumen_err = null;",
          f"try {{ source({r.s(r.script)}); }} catch (Exception __lumen_t) {{ __lumen_err = __lumen_t; }}",
          "if (__lumen_err != null) {",
          "  __lumen_msg = __lumen_err.toString();",
          f'  __lumen_w = new OutputStreamWriter(new FileOutputStream({r.s(r.error)}), "UTF-8"); __lumen_w.write(__lumen_msg); __lumen_w.close();',
          "}",
          "try { ResultsTable.getResultsTable().setPrecision(9); } catch (Exception __lumen_t) { }",
          f'IJ.runMacroFile({r.s(r.teardown)}, {r.s(str(r.out))} + "\\t" + __lumen_bound);',
          "System.out.flush();", "System.exit(0);", ""]
    return "\n".join(L)


def _runner_rhino(r: _Run) -> str:
    L = ["".join(f"importPackage(Packages.{p}); " for p in _IJ_PACKAGES), "".join(f"importPackage(Packages.{p}); " for p in _JAVA_PACKAGES),
         "Packages.ij.macro.Interpreter.batchMode = true;",
         "var __lumen_rm = new Packages.ij.plugin.frame.RoiManager(true);",
         'try { var __lumen_f = java.lang.Class.forName("ij.plugin.frame.RoiManager").getDeclaredField("instance"); __lumen_f.setAccessible(true); __lumen_f.set(null, __lumen_rm); } catch (__lumen_t) { }']
    if r.seed_rois:
        L.append(f'__lumen_rm.runCommand("Open", {r.s(r.seed_rois)});')
    if r.seed_results:
        L.append(f"Packages.ij.plugin.filter.Analyzer.setResultsTable(Packages.ij.measure.ResultsTable.open({r.s(r.seed_results)}));")
    L.append("var imp = null;")
    if r.image:
        L.append(f"imp = Packages.ij.IJ.openImage({r.s(r.image)}); imp.setTitle({r.s(r.name)}); imp.show();")
        if r.roi:
            L.append(f"imp.setRoi(Packages.ij.io.RoiDecoder.open({r.s(r.roi)}));")
    L.append("var __lumen_bound = imp == null ? 0 : imp.getID();")
    L += [f"var {n} = {lit};" for n, lit in r.params]
    L += ["var __lumen_err = null;",
          f"try {{ load({r.s(r.script)}); }} catch (__lumen_t) {{ __lumen_err = __lumen_t; }}",
          "if (__lumen_err != null) {",
          "  var __lumen_msg = String(__lumen_err);",
          "  try { if (__lumen_err.javaException) __lumen_msg = String(__lumen_err.javaException); } catch (__lumen_t2) { }",
          '  try { if (__lumen_err.lineNumber && __lumen_msg.indexOf("line") < 0) __lumen_msg += " (line " + __lumen_err.lineNumber + ")"; } catch (__lumen_t3) { }',
          f'  var __lumen_w = new java.io.OutputStreamWriter(new java.io.FileOutputStream({r.s(r.error)}), "UTF-8"); __lumen_w.write(__lumen_msg); __lumen_w.close();',
          "}",
          "try { Packages.ij.measure.ResultsTable.getResultsTable().setPrecision(9); } catch (__lumen_t4) { }",
          f'Packages.ij.IJ.runMacroFile({r.s(r.teardown)}, {r.s(str(r.out))} + "\\t" + __lumen_bound);',
          "java.lang.System.out.flush();", "java.lang.System.exit(0);", ""]
    return "\n".join(L)


def _runner_jruby(r: _Run) -> str:
    L = ["require 'java'"] + [f"java_import '{c}'" for c in ("ij.IJ", "ij.ImagePlus", "ij.WindowManager", "ij.macro.Interpreter", "ij.measure.ResultsTable",
                                                               "ij.plugin.frame.RoiManager", "ij.plugin.filter.Analyzer", "ij.io.RoiDecoder")]
    L += ["Interpreter.batchMode = true",
          "__lumen_rm = RoiManager.new(true)",
          "begin",
          '  __lumen_f = java.lang.Class.forName("ij.plugin.frame.RoiManager").getDeclaredField("instance")',
          "  __lumen_f.setAccessible(true)",
          "  __lumen_f.set(nil, __lumen_rm)",
          "rescue Exception, java.lang.Throwable => __lumen_t",
          "end"]
    if r.seed_rois:
        L.append(f'__lumen_rm.runCommand("Open", {r.s(r.seed_rois)})')
    if r.seed_results:
        L.append(f"Analyzer.setResultsTable(ResultsTable.open({r.s(r.seed_results)}))")
    L.append("imp = nil")
    if r.image:
        L.append(f"imp = IJ.openImage({r.s(r.image)}); imp.setTitle({r.s(r.name)}); imp.show")
        if r.roi:
            L.append(f"imp.setRoi(RoiDecoder.open({r.s(r.roi)}))")
    L.append("__lumen_bound = imp.nil? ? 0 : imp.getID")
    L += [f"{n} = {lit}" for n, lit in r.params]
    L += ["__lumen_err = nil",
          "begin",
          f"  eval(File.read({r.s(r.script)}), binding, {r.s(r.script)}, 1)",
          "rescue Exception, java.lang.Throwable => __lumen_t",
          "  __lumen_err = __lumen_t",
          "end",
          "if __lumen_err",
          '  __lumen_msg = "#{__lumen_err.class}: #{__lumen_err.message}"',
          f"  __lumen_bt = (__lumen_err.backtrace || []).find {{ |l| l.include?({r.s(Path(r.script).name)}) }}",
          '  __lumen_msg += " (" + __lumen_bt + ")" if __lumen_bt',
          f'  File.open({r.s(r.error)}, "w:UTF-8") {{ |f| f.write(__lumen_msg) }}',
          "end",
          "begin",
          "  ResultsTable.getResultsTable.setPrecision(9)",
          "rescue Exception, java.lang.Throwable => __lumen_t",
          "end",
          f'IJ.runMacroFile({r.s(r.teardown)}, {r.s(str(r.out))} + "\\t" + __lumen_bound.to_s)',
          "java.lang.System.out.flush", "java.lang.System.exit(0)", ""]
    return "\n".join(L)


def _runner_clojure(r: _Run) -> str:
    L = ["(import '(ij IJ ImagePlus WindowManager) '(ij.macro Interpreter) '(ij.measure ResultsTable) '(ij.plugin.frame RoiManager) '(ij.plugin.filter Analyzer) '(ij.io RoiDecoder))",
         "(set! Interpreter/batchMode true)",
         "(def __lumen_rm (RoiManager. true))",
         '(try (let [f (.getDeclaredField (Class/forName "ij.plugin.frame.RoiManager") "instance")] (.setAccessible f true) (.set f nil __lumen_rm)) (catch Throwable t nil))']
    if r.seed_rois:
        L.append(f'(.runCommand __lumen_rm "Open" {r.s(r.seed_rois)})')
    if r.seed_results:
        L.append(f"(Analyzer/setResultsTable (ResultsTable/open {r.s(r.seed_results)}))")
    L.append("(def imp nil)")
    if r.image:
        L.append(f"(def imp (IJ/openImage {r.s(r.image)})) (.setTitle imp {r.s(r.name)}) (.show imp)")
        if r.roi:
            L.append(f"(.setRoi imp (RoiDecoder/open {r.s(r.roi)}))")
    L.append("(def __lumen_bound (if (nil? imp) 0 (.getID imp)))")
    L += [f"(def {n} {lit})" for n, lit in r.params]
    L += [f"(def __lumen_err (try (load-file {r.s(r.script)}) nil (catch Throwable t t)))",
          f'(when __lumen_err (spit {r.s(r.error)} (str (.getName (class __lumen_err)) ": " (.getMessage __lumen_err))))',
          "(try (.setPrecision (ResultsTable/getResultsTable) 9) (catch Throwable t nil))",
          f'(IJ/runMacroFile {r.s(r.teardown)} (str {r.s(str(r.out))} "\\t" __lumen_bound))',
          "(.flush System/out)", "(System/exit 0)", ""]
    return "\n".join(L)


RUNNERS = {"groovy": _runner_groovy, "beanshell": _runner_beanshell, "rhino": _runner_rhino, "jruby": _runner_jruby, "clojure": _runner_clojure}


def build_command(lang: str, runner_path: str, *, home: Path, heap: str, java: str | None = None) -> tuple[list[str], list[str]]:
    """``(java command line, missing jar components)`` for a runner file (the executable first)."""
    spec = LAUNCHERS[lang]
    cp, missing = fiji_jars.classpath_for(spec["jars"] + spec["optional"])
    missing = [m for m in missing if m not in spec["optional"]]
    exe = java or fiji_jars.find_java()[0] or "java"
    opts = fiji_jars.base_options(headless=False, home=home, heap=heap) + [fiji_jars.logback_option(home)]
    cmd = [exe] + opts + ["-cp", os.pathsep.join(cp), spec["main"]] + list(spec["args"]) + [runner_path]
    return cmd, missing


# ---------------------------------------------------------------------------- Results / ROI helpers
def _norm_cell(v: Any) -> Any:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return "" if v is None else str(v)


def write_results_csv(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """The session's Results rows as an ImageJ-style CSV (row numbers in the first column); returns the rows as
    ImageJ will hand them back (numbers as floats, everything else as strings)."""
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    seeded: list[dict[str, Any]] = []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([" "] + cols)
        for i, row in enumerate(rows):
            vals = []
            back: dict[str, Any] = {}
            for k in cols:
                v = row.get(k)
                if v is None or v == "":
                    vals.append("")
                    continue
                nv = _norm_cell(v)
                if isinstance(nv, float):
                    if math.isnan(nv):
                        vals.append("NaN")  # ImageJ's spelling; an empty cell on the way back
                        continue
                    vals.append(repr(nv) if not nv.is_integer() else str(int(nv)))
                else:
                    vals.append(nv)
                back[k] = nv
            w.writerow([str(i + 1)] + vals)
            seeded.append(back)
    return seeded


def read_results_csv(path: Path) -> list[dict[str, Any]]:
    """ImageJ's Results CSV as Lumen rows: numbers as floats, ``Label`` / text cells as strings, empty cells left out."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return rows
        for raw in reader:
            if not raw or all(c.strip() == "" for c in raw):
                continue
            d: dict[str, Any] = {}
            for h, cell in zip(header, raw):
                if h.strip() == "":
                    continue  # the row-number column
                cell = cell.strip()
                if h == "Label":
                    d["Label"] = cell
                    continue
                if cell == "" or cell == "NaN":
                    continue
                try:
                    f = float(cell)
                except ValueError:
                    d[h] = cell
                    continue
                if not math.isnan(f):
                    d[h] = f
            rows.append(d)
    return rows


def rows_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if set(ra) != set(rb):
            return False
        for k, va in ra.items():
            vb = rb[k]
            na, nb = _norm_cell(va), _norm_cell(vb)
            if isinstance(na, float) and isinstance(nb, float):
                if not (math.isclose(na, nb, rel_tol=1e-8, abs_tol=1e-9) or (math.isnan(na) and math.isnan(nb))):
                    return False
            elif na != nb:
                return False
    return True


def _roi_key(r: dict[str, Any]) -> Any:
    def num(v: Any) -> Any:
        return round(float(v), 3) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v

    out: dict[str, Any] = {}
    for k, v in r.items():
        if k in ("id", "image_id", "position", "name"):
            continue
        if k == "points":
            out[k] = tuple((num(p[0]), num(p[1])) for p in v)
        elif isinstance(v, list):
            out[k] = tuple(num(x) for x in v)
        else:
            out[k] = num(v)
    return tuple(sorted(out.items()))


def rois_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if _roi_key(ra) != _roi_key(rb):
            return False
        if ra.get("name") and rb.get("name") and ra["name"] != rb["name"]:
            return False
    return True


def _cal_changed(a: Any, b: Any) -> bool:
    da, db = a.to_dict(), b.to_dict()
    for k in set(da) | set(db):
        va, vb = da.get(k), db.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not math.isclose(float(va), float(vb), rel_tol=1e-6, abs_tol=1e-9):
                return True
        else:
            sa, sb = str(va or ""), str(vb or "")
            if k == "unit":
                sa, sb = ("" if sa in ("pixel", "pixels") else sa), ("" if sb in ("pixel", "pixels") else sb)
            if sa != sb:
                return True
    return False


# ---------------------------------------------------------------------------- the run
def _timeout() -> float:
    try:
        return max(5.0, float(os.environ.get(ENV_TIMEOUT, "600") or 600))
    except ValueError:
        return 600.0


def run_jvm_script_subprocess(session: Any, code: str, language: str, params: dict[str, Any] | None = None, roi: Any = None) -> dict[str, Any]:
    """Run ``code`` in a JVM script language in a ``java`` subprocess (see the module docstring) and exchange the
    session state through files.  Returns the same dictionary as the in-process runner – ``{"stdout", "error",
    "language", "roi", "new_images", "changed", "results_rows", "roi_manager", "rois_copied", "mode": "subprocess",
    "seconds"[, "parameters", "note"]}`` – and never raises for a missing JDK / jars or a failing script."""
    from .jvm_bridge import LANGUAGES, _strip_header, normalize_language
    from .params import parse, resolve_values
    lang = normalize_language(language)
    base: dict[str, Any] = {"stdout": "", "error": None, "language": lang or str(language), "mode": "subprocess"}
    if lang is None or lang not in LAUNCHERS:
        return {**base, "error": f"Unknown JVM script language {language!r} (one of {', '.join(LAUNCHERS)})"}
    label = LANGUAGES[lang]["label"]
    st = status()
    ls = st["languages"][lang]
    if not ls["available"]:
        why = st["detail"] if not st["available"] else ls["detail"]
        return {**base, "error": f"{label} scripts need a JDK and ImageJ's jars (subprocess mode): {why}. {st['install']}"}
    plist, header_lines = parse(code)
    values = resolve_values(plist, params)
    body = _strip_header(code, header_lines)
    with _LOCK:
        out = _run(session, body, lang, label, plist, values, roi if isinstance(roi, dict) else None, st)
    if plist:
        out["parameters"] = [p.descriptor() for p in plist]
    return out


def _run(session: Any, body: str, lang: str, label: str, plist: list[Any], values: dict[str, Any], roi: dict[str, Any] | None, st: dict[str, Any]) -> dict[str, Any]:
    from ..io import read_ij_rois, write_ij_rois, write_image
    work = Path(tempfile.mkdtemp(prefix="lumen-jvm-"))
    keep = os.environ.get(ENV_KEEP, "").strip().lower() in ("1", "true", "yes")
    try:
        (work / "out").mkdir()
        home = work / "home"
        home.mkdir()
        r = _Run(lang, work)
        img = session.active
        nbytes = 0
        if img is not None:
            r.image = str(work / "image.tif")
            r.name = str(img.name)
            write_image(img, r.image, fmt="tif")
            nbytes = int(getattr(img.data, "nbytes", 0) or 0)
            if roi is not None:
                try:
                    r.roi = str(write_ij_rois([dict(roi)], work / "selection.roi"))
                    r.roi_name = str(roi.get("name") or "") or None
                except Exception:  # noqa: BLE001 – a selection the .roi writer cannot encode is simply not passed
                    r.roi = None
        seed_rows = list(session.results.rows)
        seeded_rows: list[dict[str, Any]] = []
        if seed_rows:
            r.seed_results = str(work / "results_in.csv")
            seeded_rows = write_results_csv(seed_rows, Path(r.seed_results))
        seed_rois = [dict(x) for x in session.roi_manager.rois]
        if seed_rois:
            try:
                r.seed_rois = str(write_ij_rois(seed_rois, work / "RoiSet.zip"))
            except Exception:  # noqa: BLE001
                r.seed_rois, seed_rois = None, []
        r.params = [(p.name, param_literal(lang, p, values.get(p.name))) for p in plist if p.io != "output" and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p.name)]
        Path(r.script).write_text(body, encoding="utf-8")
        Path(r.teardown).write_text(TEARDOWN_MACRO, encoding="utf-8")
        runner = work / ("runner" + LAUNCHERS[lang]["ext"])
        runner.write_text(RUNNERS[lang](r), encoding="utf-8")
        cmd, missing = build_command(lang, str(runner), home=home, heap=fiji_jars.heap_option(nbytes), java=st["java"])
        if missing:
            return {"stdout": "", "error": f"{label}: missing jar(s) {', '.join(missing)} – {LAUNCHERS[lang]['note']}", "language": lang, "mode": "subprocess"}
        res = fiji_jars.run_java(cmd[1:], cwd=home, timeout=_timeout(), java=cmd[0])
        return _collect(session, img, r, res, lang, label, seeded_rows, seed_rois, read_ij_rois)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def _collect(session: Any, img: Any, r: _Run, res: fiji_jars.JavaResult, lang: str, label: str, seeded_rows: list[dict[str, Any]],
             seed_rois: list[dict[str, Any]], read_ij_rois: Any) -> dict[str, Any]:
    from ..io import decode_ij_roi, read_image
    from ..undo import History
    out_dir = r.out
    stdout = res.stdout or ""
    err_lines = fiji_jars.filter_stderr(res.stderr or "")
    error: str | None = None
    if res.error or res.returncode != 0:
        tail = "\n".join(err_lines.splitlines()[-30:])
        error = (res.error or f"the JVM exited with code {res.returncode}") + (f"\n{tail}" if tail else "")
    elif err_lines:
        stdout += ("" if not stdout or stdout.endswith("\n") else "\n") + err_lines + "\n"
    script_error = out_dir / "error.txt"
    if script_error.is_file():
        msg = script_error.read_text(encoding="utf-8", errors="replace").strip()
        error = msg if error is None else f"{msg}\n{error}"
    done = (out_dir / "done.txt").is_file()
    if not done and error is None:
        error = ("the script ended the JVM before Lumen could collect its results (System.exit / IJ.run(\"Quit\") inside a script are not supported)"
                 if res.returncode == 0 else "the run ended before the results were written")
    log_file = out_dir / "log.txt"
    if log_file.is_file():
        have = set(stdout.splitlines())
        missing = [ln for ln in log_file.read_text(encoding="utf-8", errors="replace").splitlines() if ln not in have]
        if missing:
            stdout += ("" if not stdout or stdout.endswith("\n") else "\n") + "\n".join(missing) + "\n"
    for line in stdout.splitlines():
        if line.strip():
            session.add_log(line)
    out: dict[str, Any] = {"stdout": stdout, "error": error, "language": lang, "mode": "subprocess", "changed": False, "new_images": [], "results_rows": len(seeded_rows),
                           "roi_manager": True, "rois_copied": 0, "seconds": round(res.seconds, 3), "command": res.command}
    if not done:
        return out
    # images: the manifest lists every image open at the end (id, title, saved path)
    manifest = out_dir / "images.tsv"
    entries: list[tuple[str, str, str]] = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                entries.append((parts[0], parts[1], parts[2]))
    bound_entry = next((e for e in entries if e[2].endswith("bound.tif")), None)
    new_ids: list[str] = []
    for _jid, title, path in entries:
        if path.endswith("bound.tif"):
            continue
        try:
            new = read_image(path, name=title or "Untitled")
            new.path = None
            new.metadata.pop("source", None)
            new_ids.append(session.add_image(new).id)
        except Exception as e:  # noqa: BLE001
            error = (error + "\n" if error else "") + f"could not read back image {title!r}: {e}"
    out["new_images"] = new_ids
    # the bound image: in place, with undo, when the script changed it
    if img is not None and bound_entry is not None:
        try:
            conv = read_image(bound_entry[2], name=img.name)
            pixels = conv.data.shape != img.data.shape or conv.data.dtype != img.data.dtype or bool(conv.rgb) != bool(img.rgb) or not np.array_equal(conv.data, img.data)
            cal_changed = _cal_changed(conv.calibration, img.calibration)
            title = bound_entry[1]
            renamed = bool(title) and title != img.name
            if pixels or cal_changed or renamed:
                hist = session.histories.get(img.id)
                if hist is not None:
                    hist.undo.append(History.snapshot(img, f"{label} script"))
                    hist.redo.clear()
                    hist._trim()
                if pixels:
                    img.replace_data(np.ascontiguousarray(conv.data), rgb=conv.rgb)
                    out["changed"] = True
                if cal_changed:
                    img.calibration = conv.calibration
                if renamed:
                    img.name = session.unique_name(title)
        except Exception as e:  # noqa: BLE001
            error = (error + "\n" if error else "") + f"could not read the image back: {e}"
        sel = out_dir / "selection.roi"
        roi_out = None
        if sel.is_file():
            try:
                roi_out = decode_ij_roi(sel.read_bytes())
                roi_out.pop("id", None)
                if roi_out.get("name") == "selection" and r.roi_name != "selection":
                    roi_out.pop("name", None)
            except Exception:  # noqa: BLE001
                roi_out = None
        out["roi"] = roi_out
    elif img is not None:
        out["roi"] = None
        out["note"] = "the bound image was closed by the script"
    # Results
    csv_path = out_dir / "results.csv"
    rows = read_results_csv(csv_path) if csv_path.is_file() else []
    if not rows_equal(rows, seeded_rows):
        session.results.clear()
        session.results.append(rows)
        session._changed("results")
    out["results_rows"] = len(rows)
    # ROI Manager
    zip_path = out_dir / "rois.zip"
    rois = read_ij_rois(zip_path) if zip_path.is_file() else []
    if not rois_equal(rois, seed_rois):
        session.roi_manager.clear()
        for i, rr in enumerate(rois):
            if i < len(seed_rois) and _roi_key(rr) == _roi_key(seed_rois[i]):
                session.roi_manager.rois.append(dict(seed_rois[i]))
                continue
            rr = dict(rr)
            rr.pop("id", None)
            pos = rr.pop("position", None)
            session.roi_manager.add(rr, image_id=img.id if img is not None else None, position=pos)
        session._changed("rois")
        out["rois_copied"] = len(rois)
    session._changed("images")
    out["error"] = error
    return out
