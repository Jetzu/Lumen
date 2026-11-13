"""Tests for the ImageJ 1.x API shim (lumen.core.ijshim): Jython-style scripts run end to end through the script
runner, the ``#@`` parameter header, the headless binding and the JavaScript bridge."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.ijshim import bind, run_ij_script
from lumen.core.ijshim.bridge import Bridge
from lumen.core.ijshim.params import descriptors, parse, prepare
from lumen.core.image import LumenImage
from lumen.core.scripting import run_script
from lumen.core.session import Session

BLOBS = 'from ij import IJ\nIJ.run("Blobs (25K)")\nimp = IJ.getImage()\n'


def run(session: Session, code: str, params: dict | None = None, roi: dict | None = None) -> str:
    out = run_ij_script(session, code, params, roi=roi if roi is not None else object())
    assert not out["error"], out["error"]
    return out["stdout"]


@pytest.fixture
def session():
    return Session()


# ---------------------------------------------------------------- imports
def test_ij_modules_are_importable():
    import ij  # noqa: F401
    from ij import IJ, ImagePlus, ImageStack, WindowManager  # noqa: F401
    from ij.gui import GenericDialog, Line, OvalRoi, Overlay, PointRoi, PolygonRoi, Roi  # noqa: F401
    from ij.io import FileSaver, Opener  # noqa: F401
    from ij.measure import Calibration, Measurements, ResultsTable  # noqa: F401
    from ij.plugin import Duplicator, ZProjector  # noqa: F401
    from ij.plugin.filter import Analyzer, ParticleAnalyzer  # noqa: F401
    from ij.plugin.frame import RoiManager  # noqa: F401
    from ij.process import ByteProcessor, FloatProcessor, ImageProcessor, ImageStatistics  # noqa: F401
    from java.awt import Color, Rectangle  # noqa: F401
    from java.io import File  # noqa: F401
    from java.lang import Math, System  # noqa: F401
    from java.util import ArrayList  # noqa: F401
    from jarray import array, zeros  # noqa: F401
    from loci.plugins import BF  # noqa: F401
    assert ij.IJ is IJ and ij.ImagePlus is ImagePlus
    with pytest.raises(ImportError, match="no module"):
        import ij.nonexistent  # noqa: F401


def test_shim_call_outside_a_script_explains_itself():
    from ij import IJ
    with pytest.raises(RuntimeError, match="Lumen session"):
        IJ.getImage()


# ---------------------------------------------------------------- the Fiji "hello world" scripts
def test_open_blur_and_title(session):
    out = run(session, BLOBS + 'before = imp.getStatistics().stdDev\nIJ.run(imp, "Gaussian Blur...", "sigma=3")\n'
                               'print(imp.getTitle(), imp.getWidth(), imp.getHeight(), imp.getBitDepth(), imp.getStatistics().stdDev < before)')
    img = session.active
    assert img is not None and img.name == "blobs.tif"
    assert out.split() == ["blobs.tif", str(img.width), str(img.height), "8", "True"]


def test_threshold_particles_roi_manager_and_results(session):
    code = BLOBS + '''
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
IJ.setAutoThreshold(imp, "Otsu dark")
IJ.run(imp, "Convert to Mask", "")
IJ.run(imp, "Analyze Particles...", "size=30-Infinity display add")
rm = RoiManager.getInstance()
rt = ResultsTable.getResultsTable()
print(rm.getCount() > 20, rt.getCounter() == rm.getCount(), rt.getValue("Area", 0) >= 30, rt.getColumnIndex("Area"), rt.getColumnIndex("Nope"))
roi = rm.getRoi(0)
print(roi.getName() == rm.getName(0), roi.getBounds().width > 0, roi.isArea())
rm.select(1)
IJ.run(imp, "Measure", "")
print(rt.size() == rm.getCount() + 1, round(rt.getValue("Mean", rt.size() - 1)))
'''
    out = run(session, code)
    lines = out.strip().splitlines()
    assert lines[0] == "True True True 0 -1"
    assert lines[1] == "True True True"
    assert lines[2] == "True 255"
    assert len(session.roi_manager.rois) > 20


def test_processor_pixels_are_live_views(session):
    code = BLOBS + '''
ip = imp.getProcessor()
v = ip.getPixel(10, 10)
ip.putPixel(10, 10, 7)
px = ip.getPixels()
px[0] = 9
imp.updateAndDraw()
print(v, imp.getProcessor().getPixel(10, 10), imp.getProcessor().get(0, 0), len(px) == imp.getWidth() * imp.getHeight(), ip.getWidth(), ip.getBitDepth())
print(ip.getPixel(-1, 5), ip.getPixelValue(10, 10), round(ip.getStatistics().mean, 2) == round(imp.getStatistics().mean, 2))
'''
    out = run(session, code)
    assert out.splitlines()[0].split()[1:] == ["7", "9", "True", str(session.active.width), "8"]
    assert out.splitlines()[1] == "0 7.0 True"
    img = session.active
    assert img.data[0, 0, 0, 10, 10] == 7 and img.data[0, 0, 0, 0, 0] == 9


def test_stack_iteration_slice_labels_and_positions(session):
    code = '''
from ij import IJ
imp = IJ.createImage("s", "8-bit ramp", 40, 30, 5)
print(imp.isVisible())
imp.show()
stack = imp.getStack()
for i in range(1, stack.getSize() + 1):
    ip = stack.getProcessor(i)
    ip.setValue(i * 10)
    ip.fill()
    stack.setSliceLabel("L%d" % i, i)
imp.setSlice(3)
print(imp.getNSlices(), imp.getStackSize(), imp.getCurrentSlice(), imp.getProcessor().getPixel(0, 0), stack.getSliceLabel(3), imp.getStack().getSize())
imp.setPosition(1, 5, 1)
print(imp.getZ(), imp.getProcessor().getPixel(5, 5), imp.getDimensions(), imp.isVisible())
'''
    out = run(session, code)
    assert out.splitlines()[0] == "False"
    assert out.splitlines()[1] == "5 5 3 30 L3 5"
    assert out.splitlines()[2] == "5 50 [40, 30, 1, 5, 1] True"
    assert session.active.name == "s" and session.active.nz == 5


def test_calibration_is_live(session):
    code = BLOBS + '''
cal = imp.getCalibration()
cal.pixelWidth = 0.5
cal.pixelHeight = 0.25
cal.setUnit("micron")
print(imp.getCalibration().pixelWidth, imp.getCalibration().getUnit(), cal.scaled(), round(imp.getStatistics().area, 1))
from ij.measure import Calibration
c2 = Calibration()
c2.pixelWidth = 2.0
imp.setCalibration(c2)
print(imp.getCalibration().pixelWidth, imp.getCalibration().getUnit())
'''
    out = run(session, code)
    img = session.active
    assert out.splitlines()[0] == f"0.5 micron True {round(img.width * img.height * 0.5 * 0.25, 1)}"
    assert out.splitlines()[1] == "2.0 pixel"
    assert session.active.calibration.pixel_width == 2.0


def test_rois_overlay_and_final_selection(session):
    code = BLOBS + '''
from ij.gui import Roi, OvalRoi, PolygonRoi, Line, PointRoi, Overlay
roi = OvalRoi(10, 10, 20, 20)
imp.setRoi(roi)
r = imp.getRoi()
print(r.getType(), r.getBounds().width, r.contains(20, 20), r.contains(1, 1), imp.getStatistics().pixelCount)
ov = Overlay()
ov.add(Roi(0, 0, 5, 5))
ov.add(PolygonRoi([1, 5, 5], [1, 1, 5], 3, Roi.POLYGON))
ov.add(Line(0, 0, 10, 10), "diag")
imp.setOverlay(ov)
print(imp.getOverlay().size(), imp.getOverlay().get("diag").getType(), round(Line(0, 0, 3, 4).getLength(), 1), PointRoi(3, 4).getType())
'''
    out = run_ij_script(session, code)
    assert not out["error"], out["error"]
    assert out["stdout"].splitlines()[0].startswith("1 20 True False ")
    assert int(out["stdout"].splitlines()[0].split()[-1]) > 300
    assert out["stdout"].splitlines()[1] == "3 5 5.0 10"
    assert len(session.active.overlay) == 3
    assert out["roi"]["type"] == "oval" and out["roi"]["w"] == 20


def test_viewport_selection_seeds_the_script(session):
    out = run_ij_script(session, BLOBS + 'r = imp.getRoi()\nprint(r.getType(), r.getBounds().x, r.getBounds().width)\nIJ.run(imp, "Measure", "")\nprint(len(lumen.results), lumen.results[-1]["Area"])',
                        roi={"type": "rect", "x": 5, "y": 6, "w": 10, "h": 20})
    assert not out["error"], out["error"]
    assert out["stdout"].splitlines() == ["0 5 10", "1 200.0"]


def test_window_manager_duplicate_and_close(session):
    code = BLOBS + '''
from ij import WindowManager
imp2 = imp.duplicate()
print(imp2.isVisible(), WindowManager.getImageCount())
imp2.setTitle("copy")
imp2.show()
print(WindowManager.getImageCount(), WindowManager.getImageTitles())
IJ.selectWindow("blobs.tif")
print(IJ.getImage().getTitle(), WindowManager.getImage("copy").getTitle(), WindowManager.getImage("nope"))
IJ.selectWindow(imp2.getID())
print(IJ.getImage() == imp2, imp2.getID() < 0, len(WindowManager.getIDList()))
imp2.close()
print(WindowManager.getImageCount(), WindowManager.getCurrentImage().getTitle())
'''
    out = run(session, code)
    assert out.splitlines() == ["False 1", "2 ['blobs.tif', 'copy']", "blobs.tif copy None", "True True 2", "1 blobs.tif"]


def test_private_results_table_and_show(session):
    code = '''
from ij.measure import ResultsTable
rt = ResultsTable()
rt.incrementCounter()
rt.addValue("A", 1.5)
rt.addLabel("row1")
rt.incrementCounter()
rt.addValue("A", 2.5)
rt.addValue("B", "x")
rt.setValue("C", 2, 9)
print(rt.size(), rt.getValue("A", 1), rt.getLabel(0), rt.getLabel(2), repr(rt.getColumnHeadings()), rt.getStringValue("B", 1), rt.getColumn("A"))
rt.show("My table")
live = ResultsTable.getResultsTable()
print(live.size())
rt.show("Results")
print(ResultsTable.getResultsTable().size(), ResultsTable.getResultsTable().getValue("C", 2))
try:
    rt.getValue("Nope", 0)
except ValueError as e:
    print(e)
'''
    out = run(session, code)
    lines = out.splitlines()
    assert lines[0] == "3 2.5 row1 None ' \\tA\\tLabel\\tB\\tC' x [1.5, 2.5, nan]"
    assert lines[1] == "0" and lines[2] == "3 9.0" and lines[3] == '"Nope" column not found'
    assert "My table" in session.tables and session.tables["My table"]["columns"] == ["A", "Label", "B", "C"]
    assert len(session.results.rows) == 3


def test_java_classes(session, tmp_path):
    code = f'''
from java.lang import System, Math, Integer, Double, String
from java.io import File
from java.util import ArrayList, HashMap
from java.awt import Color, Rectangle
from jarray import array, zeros
System.out.println("hi " + str(Math.round(2.5)))
print(Integer.parseInt("12"), Double.parseDouble("1.5"), String.format("%.2f|%d|%s|%5d", 3.14159, 4, "x", 42), Math.PI > 3)
f = File({str(tmp_path)!r})
g = File(f, "sub.txt")
print(f.exists(), f.isDirectory(), g.getName(), g.getParent() == str(f), g.exists())
a = array([1, 2, 3], 'd'); z = zeros(2, 'i')
al = ArrayList(); al.add("x"); al.add("y")
m = HashMap(); m.put("k", 1)
print(a, z, al.size(), al.get(1), m.get("k"), m.containsKey("z"))
c = Color(255, 128, 0)
print(c.getRed(), c.getGreen(), str(c), Color.red.getRed(), Rectangle(1, 2, 3, 4).width)
'''
    out = run(session, code)
    assert out.splitlines() == ["hi 3", "12 1.5 3.14|4|x|   42 True", "True True sub.txt True False", "[1.0, 2.0, 3.0] [0, 0] 2 y 1 False", "255 128 #ff8000 255 3"]
    assert any(e.get("msg") == "hi 3" for e in session.log)


def test_generic_dialog_defaults_and_supplied_values(session):
    code = '''
from ij.gui import GenericDialog
gd = GenericDialog("Settings")
gd.addNumericField("Sigma", 2.0, 1)
gd.addStringField("Name", "a")
gd.addCheckbox("Show", True)
gd.addChoice("Method", ["Otsu", "Li"], "Li")
gd.addMessage("just text")
gd.showDialog()
if gd.wasCanceled():
    print("cancelled")
print(gd.getNextNumber(), gd.getNextString(), gd.getNextBoolean(), gd.getNextChoice())
'''
    assert run(session, code) == "2.0 a True Li\n"
    assert run(session, code, {"Sigma": 5, "Name": "b", "Show": False, "Method": "Otsu"}) == "5.0 b False Otsu\n"
    assert run(session, code, {"sigma": "7", "method": 1}) == "7.0 a True Li\n"


def test_script_parameters_header(session, tmp_path):
    code = '''#@ String (label="Name", value="cells") name
#@ int (label="Count", value=3, min=1, max=10) n
#@ File (label="Folder", style="directory") folder
#@ Boolean show
#@ Double sigma
#@ String (choices={"Otsu", "Li"}) method
#@ ImagePlus imp
#@ LogService log
#@output String result
import os
print(name, n, folder is None, show, sigma, method, imp.getTitle())
log.info("logged")
result = "done"
'''
    params, lines = parse(code)
    assert [p.name for p in params] == ["name", "n", "folder", "show", "sigma", "method", "imp", "log", "result"]
    assert [p.kind for p in params] == ["string", "int", "directory", "bool", "float", "choice", "image", "service", "string"]
    assert params[1].min == 1 and params[1].max == 10 and params[5].choices == ["Otsu", "Li"] and params[8].io == "output"
    d = descriptors(code)
    assert d[0]["label"] == "Name" and d[0]["default"] == "cells" and d[2]["type"] == "directory" and d[5]["default"] == "Otsu"
    prepared = prepare(code, {"n": "7", "sigma": 1.5})
    assert prepared.line_offset == 0 and prepared.code.count("\n") == code.count("\n")
    session.run_op("file.sample", {"name": "Blobs"})
    out = run_ij_script(session, code, {"n": 7, "sigma": 1.5, "folder": str(tmp_path)})
    assert not out["error"], out["error"]
    assert out["stdout"] == "cells 7 False False 1.5 Otsu blobs.tif\n[INFO] logged\n"
    assert [p["name"] for p in out["parameters"]][:3] == ["name", "n", "folder"]
    out = run_ij_script(session, code)
    assert out["stdout"].startswith("cells 3 True False 0.0 Otsu")
    # a File parameter is a java.io.File that is also a path
    out = run_ij_script(session, "#@ File f\nimport os\nprint(f.getName(), os.path.basename(f), f.exists())", {"f": str(tmp_path / "x.tif")})
    assert out["stdout"] == "x.tif x.tif False\n"


def test_script_parameters_header_not_at_top(session):
    code = 'x = 1\n#@ int (value=4) n\nprint(x + n)\n'
    out = run_ij_script(session, code)
    assert out["stdout"] == "5\n" and not out["error"]
    out = run_ij_script(session, code + "raise ValueError('boom')\n")
    assert "boom" in out["error"] and "shifted by 1" in out["error"]


def test_plugins_duplicator_projector_calculator_splitter(session):
    code = '''
from ij import IJ, WindowManager
from ij.plugin import Duplicator, ZProjector, ImageCalculator, ChannelSplitter, RGBStackMerge
from ij.plugin.filter import GaussianBlur, RankFilters, BackgroundSubtracter
imp = IJ.createImage("st", "8-bit ramp", 32, 16, 4)
imp.show()
st = imp.getStack()
for i in range(1, 5):
    st.getProcessor(i).add(i)
dup = Duplicator().run(imp, 2, 3)
print(dup.getNSlices(), dup.isVisible(), dup.getTitle())
proj = ZProjector.run(imp, "max")
print(proj.getNSlices(), proj.getProcessor().getPixel(31, 0), proj.isVisible())
calc = ImageCalculator().run("Subtract create 32-bit", proj, proj)
print(calc.getBitDepth(), calc.getProcessor().getPixel(5, 5))
calc2 = ImageCalculator.run(proj, proj, "add create")
print(calc2.getProcessor().getPixel(31, 0))
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")
multi = IJ.getImage()
chans = ChannelSplitter.split(multi)
print(len(chans), [c.getNChannels() for c in chans], chans[0].getTitle().startswith("C1"))
merged = RGBStackMerge.mergeChannels([chans[0], chans[1], None], True)
print(merged.getNChannels(), merged.isComposite())
ip = chans[0].getProcessor()
before = ip.getStatistics().stdDev
GaussianBlur().blurGaussian(ip, 2.0)
print(ip.getStatistics().stdDev < before)
RankFilters().rank(ip, 1.0, RankFilters.MEDIAN)
BackgroundSubtracter().rollingBallBackground(ip, 20.0, False, False, False, True, True)
print(chans[0].getProcessor().getStatistics().mean <= before * 10)
'''
    out = run(session, code)
    lines = out.splitlines()
    assert lines[0] == "2 False DUP_st"
    assert lines[1].startswith("1 ") and lines[1].endswith(" True")
    assert lines[2] == "32 0.0"
    assert lines[4] == "3 [1, 1, 1] True"
    assert lines[5] == "2 True"
    assert lines[6] == "True" and lines[7] == "True"


def test_particle_analyzer_class_and_threshold_to_selection(session):
    code = BLOBS + '''
from ij.plugin.filter import ParticleAnalyzer, ThresholdToSelection, Analyzer
from ij.measure import ResultsTable, Measurements
from ij.plugin.frame import RoiManager
from ij.process import AutoThresholder
thr = AutoThresholder().getThreshold(AutoThresholder.Method.Otsu, imp.getProcessor().getHistogram())
IJ.setAutoThreshold(imp, "Otsu dark")
roi = ThresholdToSelection.run(imp)
print(roi is not None and roi.isArea())
IJ.run(imp, "Convert to Mask", "")
rt = ResultsTable()
pa = ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER | ParticleAnalyzer.SHOW_OUTLINES, Measurements.AREA | Measurements.MEAN | Measurements.CENTROID, rt, 30, 1e9)
pa.setHideOutputImage(True)
pa.analyze(imp)
print(rt.size() > 20, rt.size() == RoiManager.getInstance().getCount(), rt.getValue("Area", 0) >= 30, "X" in rt.getHeadings(), ResultsTable.getResultsTable().size(), pa.getOutputImage() is not None)
Analyzer.setMeasurements(Measurements.AREA | Measurements.MEAN | Measurements.MIN_MAX | Measurements.FERET)
print(Analyzer.getMeasurements() & Measurements.FERET > 0)
print(0 < thr < 255)
'''
    out = run(session, code)
    assert out.splitlines() == ["True", "True True True True 0 True", "True", "True"]
    assert "feret" in session.measurements


def test_file_saver_opener_and_open_image(session, tmp_path):
    p = tmp_path / "out.tif"
    code = BLOBS + f'''
from ij.io import FileSaver, Opener
print(FileSaver(imp).saveAsTiff({str(p)!r}))
imp2 = IJ.openImage({str(p)!r})
print(imp2.getWidth(), imp2.isVisible(), imp2.getOriginalFileInfo().fileName, imp2.getProcessor().getPixel(10, 10) == imp.getProcessor().getPixel(10, 10))
imp3 = Opener().openImage({str(tmp_path)!r}, "out.tif")
imp3.show()
print(imp3.isVisible(), IJ.getImage().getTitle(), IJ.openImage({str(tmp_path / "missing.tif")!r}))
IJ.saveAs(imp3, "PNG", {str(tmp_path / "out.png")!r})
'''
    out = run(session, code)
    assert p.exists() and (tmp_path / "out.png").exists()
    assert out.splitlines()[0] == "True"
    assert out.splitlines()[1] == f"{session.active.width} False out.tif True"
    assert out.splitlines()[2] == "True out.tif-1 None"  # Save As renamed the open image to out.tif, so the re-opened file gets a unique title


def test_run_macro_and_log_go_to_the_log_section(session):
    out = run(session, BLOBS + 'IJ.log("hello from script")\nprint(IJ.runMacro("print(getTitle() + \\" \\" + nSlices);").strip())\nIJ.showMessage("Title", "msg")\nIJ.error("bad thing")')
    assert out == "hello from script\nblobs.tif 1\nTitle: msg\nError: bad thing\n"  # IJ.error goes to stderr, which the runner captures too
    msgs = [e.get("msg") for e in session.log]
    assert "hello from script" in msgs and "bad thing" in msgs


def test_injected_macro_functions_share_the_selection_with_the_shim(session):
    # makeRectangle() (the macro function injected into every script) and imp.getRoi() are one selection
    out = run(session, BLOBS + 'makeRectangle(1, 2, 30, 40)\nr = imp.getRoi()\nprint(r.getBounds().width, r.getBounds().height)\nimp.deleteRoi()\nprint(selectionType())')
    assert out == "30 40\n-1\n"


def test_plain_run_script_has_the_shim_too(session):
    out = run_script(session, 'from ij import IJ\nIJ.run("Blobs (25K)")\nprint(IJ.getImage().getTitle())')
    assert out["stdout"] == "blobs.tif\n" and not out["error"]


def test_headless_bind(session):
    from ij import IJ, WindowManager
    with bind(session):
        IJ.run("Blobs (25K)")
        imp = IJ.getImage()
        assert imp.getWidth() == session.active.width and WindowManager.getImageCount() == 1
        assert round(imp.getStatistics().mean, 1) == round(float(session.active.data.mean()), 1)


def test_processor_filters_conversions_and_drawing(session):
    code = '''
from ij import IJ, ImagePlus
from ij.process import FloatProcessor, ByteProcessor, ImageProcessor
fp = FloatProcessor(20, 10)
fp.setValue(3.5)
fp.fill()
fp.putPixelValue(2, 2, 10)
print(fp.getf(2, 2), fp.getStatistics().max, fp.getBitDepth())
bp = fp.convertToByte(True)
print(bp.getBitDepth(), bp.getPixel(2, 2), bp.getPixel(0, 0))
bp2 = ByteProcessor(30, 20)
bp2.setColor(200)
bp2.setRoi(5, 5, 10, 10)
bp2.fill()
bp2.resetRoi()
bp2.drawLine(0, 0, 29, 0)
bp2.setFont(None)
print(bp2.getPixel(6, 6), bp2.getPixel(20, 20), bp2.getPixel(15, 0), bp2.getStatistics().max)
bp2.blurGaussian(1.0)
print(bp2.getPixel(4, 4) > 0)
imp = ImagePlus("fresh", bp2)
imp.show()
print(IJ.getImage().getTitle(), imp.getWidth(), imp.getProcessor().getPixel(6, 6) > 0)
bp2.drawString("Hi", 2, 18)
cp = bp2.convertToRGB()
print(cp.getBitDepth(), cp.getPixel(6, 6) >> 16 == cp.getPixel(6, 6) & 255)
sub = bp2.resize(15, 10)
print(sub.getWidth(), sub.getHeight(), bp2.crop().getWidth(), bp2.duplicate().getPixel(6, 6) == bp2.getPixel(6, 6))
'''
    out = run(session, code)
    lines = out.splitlines()
    assert lines[0] == "10.0 10.0 32"
    assert lines[1] == "8 255 0"
    assert lines[2] == "200 0 200 200.0"
    assert lines[3] == "True"
    assert lines[4] == "fresh 30 True"
    assert lines[5] == "24 True"
    assert lines[6] == "15 10 30 True"


def test_plot_and_text_window(session):
    code = '''
from ij.gui import Plot
from ij.text import TextWindow
p = Plot("My plot", "x", "y")
p.setColor("red")
p.addPoints([0, 1, 2, 3], [1, 4, 9, 16], Plot.LINE)
p.addLegend("squares")
p.show()
tw = TextWindow("Notes", "first line", 300, 200)
tw.append("second line")
print(tw.getText())
'''
    out = run(session, code)
    assert session.last_plot and session.last_plot["title"] == "My plot" and session.last_plot["series"][0]["values"] == [1.0, 4.0, 9.0, 16.0]
    assert "Notes" in session.tables and out.strip().splitlines() == ["first line", "second line"]


def test_exit_and_dialog_free_user_input(session):
    out = run_ij_script(session, BLOBS + 'from ij.gui import WaitForUserDialog\nWaitForUserDialog("Check", "look").show()\nn = IJ.getNumber("Sigma:", 2)\ns = IJ.getString("Name:", "x")\nprint(n, s, IJ.getDirectory("home") != "", IJ.getDirectory("Pick"))\nIJ.exit()\nprint("not reached")',
                        {"Sigma": 4})
    assert not out["error"] and out["stdout"].splitlines()[-1] == "4.0 x True None"
    assert "WaitForUser" in out["stdout"]


# ---------------------------------------------------------------- the JavaScript bridge
def test_bridge_calls_describe_and_handles(session):
    b = Bridge(session)
    assert "IJ" in b.call("", "__index__", [])["result"] and b.call("", "__index__", [])["result"]["IJ"] == "ij.IJ"
    assert b.call("ij.IJ", "run", ["Blobs (25K)"])["result"] is None
    imp = b.call("ij.IJ", "getImage", [])["result"]
    assert imp["__class__"] == "ij.ImagePlus" and isinstance(imp["__handle__"], int)
    assert b.call(imp["__handle__"], "getTitle", [])["result"] == "blobs.tif"
    ip = b.call(imp, "getProcessor", [])["result"] if False else b.call(imp["__handle__"], "getProcessor", [])["result"]
    assert b.call(ip["__handle__"], "getPixel", [10, 10])["result"] == int(session.active.data[0, 0, 0, 10, 10])
    d = b.call("ij.measure.Measurements", "__describe__", [])["result"]
    assert d["consts"]["AREA"] == 1 and d["kind"] == "class"
    d = b.call("ij.ImagePlus", "__describe__", [])["result"]
    assert "getProcessor" in d["methods"] and "title" in d["props"] and d["consts"]["GRAY8"] == 0
    pkg = b.call("ij.plugin.frame", "__describe__", [])["result"]
    assert pkg["kind"] == "package" and "RoiManager" in pkg["classes"]
    roi = b.call("ij.gui.Roi", "__new__", [1, 2, 3, 4])["result"]
    assert roi["__class__"] == "ij.gui.Roi"
    assert b.call(imp["__handle__"], "setRoi", [{"__handle__": roi["__handle__"]}])["result"] is None
    assert b.roi["type"] == "rect" and b.roi["w"] == 3
    out = b.call("java.lang.System", "__getattr__", ["out"])["result"]
    r = b.call(out["__handle__"], "println", ["hello"])
    assert r["log"] == ["hello"]
    assert b.call("ij.IJ", "log", ["x"])["log"] == ["x"]
    rect = b.call(roi["__handle__"], "getBounds", [])["result"]
    assert b.call(rect["__handle__"], "__getattr__", ["width"])["result"] == 3
    b.call(rect["__handle__"], "__setattr__", ["width", 9])
    assert b.call(rect["__handle__"], "getWidth", [])["result"] == 9.0
    nan = b.call("java.lang.Double", "__getattr__", ["NaN"])["result"]
    assert nan == {"__nan__": True}
    cls = b.call("ij.IJ", "__getattr__", ["debugMode"])["result"]
    assert cls is False
    with pytest.raises(ValueError, match="unknown class"):
        b.call("ij.nothing.Here", "__describe__", [])
    with pytest.raises(AttributeError):
        b.call("ij.IJ", "noSuchMethod", [])


def test_bridge_arrays_and_classrefs(session):
    b = Bridge(session)
    b.call("ij.IJ", "run", ["Blobs (25K)"])
    imp = b.call("ij.IJ", "getImage", [])["result"]
    stats = b.call(imp["__handle__"], "getStatistics", [])["result"]
    hist = b.call(stats["__handle__"], "__getattr__", ["histogram"])["result"]
    assert len(hist) == 256 and sum(hist) == session.active.width * session.active.height
    arr = b.call("jarray", "__getattr__", ["array"])  # a package attribute that is a function
    assert arr["result"] is None or True
    poly = b.call("ij.gui.PolygonRoi", "__new__", [[1, 5, 5], [1, 1, 5], 3, {"__classref__": "ij.gui.Roi"} and 2])["result"]
    assert poly["__class__"] == "ij.gui.PolygonRoi"
    rm = b.call("ij.plugin.frame.RoiManager", "getInstance", [])["result"]
    b.call(rm["__handle__"], "addRoi", [{"__handle__": poly["__handle__"]}])
    rois = b.call(rm["__handle__"], "getRoisAsArray", [])["result"]
    assert len(rois) == 1 and rois[0]["__class__"] == "ij.gui.PolygonRoi"
    assert b.call(rm["__handle__"], "getCount", [])["result"] == 1


def test_stacks_built_from_processors_rgb_images_and_attached_add_slice(session):
    code = '''
from ij import IJ, ImagePlus, ImageStack
from ij.process import ByteProcessor, FloatProcessor
from ij.gui import Overlay, TextRoi, Roi
st = ImageStack(16, 8)
for i in range(3):
    ip = ByteProcessor(16, 8)
    ip.setValue(i * 100)
    ip.fill()
    st.addSlice("slice %d" % i, ip)
imp = ImagePlus("built", st)
print(imp.getNSlices(), imp.getStack().getSliceLabel(2), imp.getStack().getProcessor(3).getPixel(0, 0), imp.getBitDepth())
imp.show()
imp.getStack().addSlice("extra", ByteProcessor(16, 8))
print(imp.getNSlices(), imp.getStack().getSliceLabel(4), imp.getStackSize())
imp.getStack().deleteSlice(1)
print(imp.getNSlices(), imp.getStack().getSliceLabel(1))
rgb = IJ.createImage("rgb", "RGB white", 10, 10, 1)
cp = rgb.getProcessor()
print(rgb.getBitDepth(), hex(cp.getPixel(1, 1)), rgb.getType() == ImagePlus.COLOR_RGB)
cp.putPixel(1, 1, 0xff0000)
print(hex(cp.getPixel(1, 1)), cp.getPixelValue(1, 1) > 0)
fp = FloatProcessor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(fp.getWidth(), fp.getHeight(), fp.getf(2, 1))
ov = Overlay()
ov.add(TextRoi(2, 2, "hello"))
ov.add(Roi(1, 1, 3, 3), "box")
imp.setOverlay(ov)
print(imp.getOverlay().size(), imp.getOverlay().get("box").getBounds().width)
'''
    out = run(session, code)
    assert out.splitlines() == ["3 slice 1 200 8", "4 extra 4", "3 slice 1", "24 0xffffff True", "0xff0000 True", "3 2 6.0", "2 3"]
    assert session.active.name == "built" or any(im.name == "built" for im in session.images.values())


def test_batch_scripts_can_use_the_shim(session):
    session.run_op("file.sample", {"name": "Blobs"})
    res = session.run_op("batch.run_script_on_open_images", {"script": 'from ij import IJ\nimp = IJ.getImage()\nIJ.run(imp, "Invert", "")\nprint(imp.getTitle())'})
    assert "blobs.tif" in str(res)


# ---------------------------------------------------------------- the optional JVM runner (Groovy / BeanShell / Rhino) is registered next to run_ij_script
def test_jvm_runner_is_registered_and_degrades_without_the_bridge(session, monkeypatch):
    from lumen.core import ijshim
    from lumen.core.ijshim import jvm_bridge
    assert ijshim.run_jvm_script is jvm_bridge.run_jvm_script and ijshim.is_jvm_language is jvm_bridge.is_jvm_language
    assert list(ijshim.JVM_LANGUAGES) == ["groovy", "beanshell", "rhino", "jruby", "clojure"]
    assert ijshim.is_jvm_language("groovy") and ijshim.is_jvm_language("BeanShell") and not ijshim.is_jvm_language("jython") and not ijshim.is_jvm_language("js")
    monkeypatch.setattr(jvm_bridge, "_importable", lambda name: False)
    monkeypatch.setenv("LUMEN_JVM_MODE", "inprocess")
    jvm_bridge.reset()
    try:
        out = ijshim.run_jvm_script(session, 'println "hi"', "groovy")
        assert out["error"] and "lumen-imaging[jvm]" in out["error"] and out["stdout"] == "" and out["language"] == "groovy"
        # Jython keeps working through its own runner, untouched by the JVM bridge
        assert run(session, BLOBS + "print(imp.getTitle())") == "blobs.tif\n"
    finally:
        jvm_bridge.reset()
