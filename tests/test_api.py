import numpy as np
from fastapi.testclient import TestClient

from lumen.core.image import LumenImage
from lumen.core.session import Session
from lumen.server.app import create_app


def client():
    s = Session()
    return TestClient(create_app(s)), s


def test_meta_and_ops():
    c, _ = client()
    r = c.get("/api/meta")
    assert r.status_code == 200
    assert len(r.json()["ops"]) > 100
    assert "Fire" in r.json()["luts"]


def test_open_sample_and_planes():
    c, s = client()
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Synthetic nuclei (3 channels)"}})
    assert r.status_code == 200, r.text
    iid = r.json()["new_images"][0]
    r = c.get(f"/api/images/{iid}/planes", params={"t": 0, "z": 0})
    assert r.headers["x-dtype"] == "uint16" and r.headers["x-shape"] == "3,512,512"
    arr = np.frombuffer(r.content, np.uint16).reshape(3, 512, 512)
    assert arr.max() > 0
    r = c.get(f"/api/images/{iid}/planes", params={"stride": 4})
    assert r.headers["x-shape"] == "3,128,128"
    r = c.post(f"/api/ops/filters.gaussian/preview", json={"params": {"sigma": 3}, "image_id": iid})
    assert r.headers["x-shape"] == "3,512,512"
    r = c.post(f"/api/ops/filters.gaussian/run", json={"params": {"sigma": 3}, "image_id": iid})
    assert r.status_code == 200
    st = c.get("/api/state").json()
    assert st["history"][iid]["undo"] == ["Gaussian Blur"]
    r = c.post(f"/api/images/{iid}/undo")
    assert r.json()["label"] == "Gaussian Blur"


def test_error_is_json():
    s = Session()
    c = TestClient(create_app(s), raise_server_exceptions=False)
    r = c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": 3}})
    assert r.status_code == 400 and "error" in r.json()


def test_rois_and_results():
    c, s = client()
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    iid = r.json()["new_images"][0]
    r = c.post("/api/rois", json={"roi": {"type": "rect", "x": 5, "y": 5, "w": 20, "h": 20}, "image_id": iid})
    rid = r.json()["id"]
    r = c.post("/api/rois/measure", json={"ids": [rid]})
    assert r.json()["rows"] == 1
    r = c.get("/api/results")
    assert r.json()["rows"][0][1] == 400
    r = c.get("/api/results.csv")
    assert "Area" in r.text
    r = c.post("/api/script/run", json={"code": "print(len(lumen.rois))"})
    assert r.json()["stdout"].strip() == "1"


def test_roi_manager_measure_honours_set_measurements():
    # ROI Manager ▸ Measure (POST /api/rois/measure) reports exactly what Measure (M) reports for the same ROI:
    # the Set Measurements options (invert Y, display label, limit to threshold) apply to both
    c, s = client()
    img = s.add_image(LumenImage(np.zeros((40, 60), np.uint8), name="m"))
    img.data[0, 0, 0, 5:15, 10:30] = 200  # the top half of the ROI is bright
    s.measurements = ["area", "mean", "centroid"]
    s.measure_options.update(invert_y=True, display_label=False)
    rect = {"type": "rect", "x": 10, "y": 5, "w": 20, "h": 20}
    rid = c.post("/api/rois", json={"roi": rect, "image_id": img.id}).json()["id"]
    assert c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": img.id, "roi": rect}).status_code == 200
    assert c.post("/api/rois/measure", json={"ids": [rid]}).json()["rows"] == 1
    m_row, rm_row = s.results.rows[-2:]
    assert rm_row == m_row and "Label" not in rm_row and rm_row["Area"] == 400 and rm_row["Y"] == 40 - 15 - 1  # Y from the bottom edge (ImageJ: height − y − 1)
    assert s.results.columns == ["Area", "Mean", "X", "Y"]  # no stray Label column
    # limit to threshold: only the bright pixels count, as for Measure
    s.thresholds[img.id] = {"lower": 100, "upper": 255}
    s.measure_options["limit_to_threshold"] = True
    c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": img.id, "roi": rect})
    c.post("/api/rois/measure", json={"ids": [rid]})
    m_row, rm_row = s.results.rows[-2:]
    assert rm_row == m_row and rm_row["Area"] == 200 and rm_row["Mean"] == 200
    # with the default options the ROI name is the label again
    s.measure_options.update(display_label=True, limit_to_threshold=False, invert_y=False)
    c.post("/api/rois/measure", json={"ids": [rid]})
    assert s.results.rows[-1]["Label"] == s.roi_manager.to_list()[0]["name"] and s.results.rows[-1]["Y"] == 15


def test_static_catch_all_blocks_path_traversal(tmp_path):
    from lumen.server import app as appmod
    c, _ = client()
    c.raise_server_exceptions = False
    if not appmod.STATIC_DIR.exists():
        return  # frontend not built in this checkout
    for bad in ("../pyproject.toml", "..%2Fpyproject.toml", "%2e%2e/%2e%2e/etc/passwd", "assets/../../lumen/__init__.py"):
        r = c.get(f"/{bad}")
        assert b"[project]" not in r.content and b"__version__" not in r.content, bad
        assert r.status_code in (200, 404)


def test_upload_keeps_compound_suffix(tmp_path):
    import numpy as np
    import tifffile
    c, s = client()
    f = tmp_path / "vol.ome.tif"
    tifffile.imwrite(str(f), np.zeros((3, 8, 8), np.uint8), ome=True, metadata={"axes": "ZYX"})
    r = c.post("/api/images/upload", files={"file": ("vol.ome.tif", f.read_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    assert s.active.shape[1] == 3


def test_download_non_ascii_name_uses_rfc5987_filename():
    c, s = client()
    img = s.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="細胞.tif"))
    r = c.get(f"/api/images/{img.id}/download", params={"format": "tif"})
    assert r.status_code == 200 and len(r.content) > 0
    cd = r.headers["content-disposition"]
    assert cd.startswith('attachment; filename="') and cd.isascii()
    assert "filename*=UTF-8''%E7%B4%B0%E8%83%9E.tif" in cd
    plain = s.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="cells.tif"))
    r = c.get(f"/api/images/{plain.id}/download", params={"format": "tif"})
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="cells.tif"'


def test_open_path_series_level_and_virtual(tmp_path):
    import tifffile
    c, s = client()
    multi = tmp_path / "multi.ome.tif"
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4000, (2, 3, 12, 16)).astype(np.uint16)  # Z C Y X
    b = rng.integers(0, 255, (4, 10, 12)).astype(np.uint8)  # Z Y X
    with tifffile.TiffWriter(str(multi), ome=True) as tw:
        tw.write(a, metadata={"axes": "ZCYX", "Name": "cells"})
        tw.write(b, metadata={"axes": "ZYX", "Name": "overview"})
    r = c.post("/api/images/open", json={"path": str(multi)})
    assert r.status_code == 200, r.text
    assert s.active.shape == (1, 2, 3, 12, 16) and s.active.bit_depth == "16-bit"
    assert any("series 0 of 2" in e["msg"] and "overview" in e["msg"] for e in s.log)  # the other series are announced in the Log
    r = c.post("/api/images/open", json={"path": str(multi), "series": 1})
    assert r.status_code == 200, r.text
    assert s.active.shape == (1, 4, 1, 10, 12) and s.active.bit_depth == "8-bit" and s.active.metadata.get("series") == 1
    r = c.post("/api/images/open", json={"path": str(multi), "series": 1, "virtual": True})
    assert r.status_code == 200, r.text
    assert s.active.shape == (1, 4, 1, 10, 12) and s.active.metadata.get("source", {}).get("virtual") is True
    # level= picks a pyramid resolution level
    pyr = tmp_path / "pyr.tif"
    base = rng.integers(0, 255, (64, 96)).astype(np.uint8)
    with tifffile.TiffWriter(str(pyr)) as tw:
        tw.write(base, subifds=1, tile=(16, 16))
        tw.write(base[::2, ::2], subfiletype=1, tile=(16, 16))
    r = c.post("/api/images/open", json={"path": str(pyr), "level": 1})
    assert r.status_code == 200, r.text
    assert s.active.shape[-2:] == (32, 48)


def test_roi_manager_measure_clamps_stale_positions():
    # ROIs drawn on a movie / Z-stack and measured on a single-plane image (or on the same image after it shrank)
    # failed the whole request with an IndexError and added no rows; ImageJ's RoiManager clamps the position and
    # measures the current plane
    c, s = client()
    mov = s.add_image(LumenImage(np.zeros((5, 8, 8), np.uint8), name="mov", axes="TYX"))
    mov.data[4, 0, 0, 1:5, 1:5] = 200
    rect = {"type": "rect", "x": 1, "y": 1, "w": 4, "h": 4}
    rid = c.post("/api/rois", json={"roi": rect, "image_id": mov.id, "position": {"t": 4, "z": 0, "c": 0}}).json()["id"]
    s.measurements = ["area", "mean", "stack_position"]
    assert c.post("/api/rois/measure", json={"ids": [rid]}).json()["rows"] == 1
    assert s.results.rows[-1]["Frame"] == 5 and s.results.rows[-1]["Mean"] == 200  # in range: the stored plane
    s.add_image(LumenImage(np.full((8, 8), 7, np.uint8), name="single"))  # becomes the active image
    r = c.post("/api/rois/measure", json={"ids": []})
    assert r.status_code == 200 and r.json()["rows"] == 1, r.text
    assert s.results.rows[-1]["Frame"] == 1 and s.results.rows[-1]["Mean"] == 7  # clamped to the only frame
    # the same image after it shrank in place, a negative frame (numpy would wrap it to the last one) and a partial
    # position (the missing axes fall back to the current display plane) never raise either
    r = c.post("/api/ops/stacks.substack/run", json={"params": {"slices": "4-5", "axis": "T", "delete_from_original": True}, "image_id": mov.id})
    assert r.status_code == 200 and mov.nt == 3, r.text
    s.select(mov.id)  # the extracted substack became the active image; measure on the shrunken movie
    c.post("/api/rois", json={"roi": rect, "image_id": mov.id, "position": {"t": -1}})
    c.post("/api/rois", json={"roi": rect, "image_id": mov.id, "position": {"z": 0}})
    r = c.post("/api/rois/measure", json={"ids": []})
    assert r.status_code == 200 and r.json()["rows"] == 3, r.text
    assert [row["Frame"] for row in s.results.rows[-3:]] == [3, 1, mov.display.t + 1]


def test_script_endpoint_runs_jython_with_parameters_and_selection():
    c, s = client()
    code = '#@ int (value=2) n\nfrom ij import IJ\nIJ.run("Blobs (25K)")\nimp = IJ.getImage()\nprint(imp.getTitle(), n, imp.getRoi().getBounds().width)\nIJ.makeRectangle(1, 1, 5, 5)\n'
    r = c.post("/api/script/run", json={"code": code, "params": {"n": 9}, "roi": {"type": "rect", "x": 0, "y": 0, "w": 12, "h": 8}, "language": "jython"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["error"] is None and j["stdout"] == "blobs.tif 9 12\n"
    assert j["roi"]["type"] == "rect" and j["roi"]["w"] == 5
    assert j["parameters"][0]["name"] == "n" and j["parameters"][0]["type"] == "int"
    r = c.post("/api/script/params", json={"code": "#@ String (label='Who', value='me') who\n#@ File (style='directory') d\nprint(who)"})
    assert [p["type"] for p in r.json()["parameters"]] == ["string", "directory"]
    # plain Python scripts keep working, without the extra keys
    r = c.post("/api/script/run", json={"code": "print(len(lumen.images))"})
    assert r.json()["stdout"].strip() == "1" and "roi" not in r.json()


def test_ijshim_bridge_endpoints():
    c, s = client()
    call = lambda target, method, args=None, **extra: c.post("/api/ijshim/call", json={"token": "t1", "target": target, "method": method, "args": args or [], **extra})
    r = call("ij.IJ", "run", ["Blobs (25K)"], roi={"type": "rect", "x": 2, "y": 3, "w": 10, "h": 10})
    assert r.status_code == 200 and r.json()["result"] is None
    imp = call("ij.IJ", "getImage").json()["result"]
    assert imp["__class__"] == "ij.ImagePlus"
    assert call(imp["__handle__"], "getTitle").json()["result"] == "blobs.tif"
    assert call(imp["__handle__"], "getRoi").json()["result"]["__class__"] == "ij.gui.Roi"  # the seeded selection
    r = call("ij.IJ", "log", ["from js"])
    assert r.json()["log"] == ["from js"]
    r = call("ij.IJ", "run", ["No Such Command"])
    assert r.status_code == 400 and "Unsupported ImageJ command" in r.json()["error"]
    assert "IJ" in call("", "__index__").json()["result"]
    r = c.delete("/api/ijshim/session/t1")
    assert r.json()["found"] is True and r.json()["roi"]["w"] == 10
    assert c.delete("/api/ijshim/session/t1").json()["found"] is False


def test_script_endpoint_routes_jvm_languages_to_the_bridge(monkeypatch):
    """Groovy / BeanShell / Rhino go to lumen.core.ijshim.jvm_bridge; without the [jvm] extra the answer is a friendly
    error (HTTP 200, like a script error), and Python / Jython are untouched.  The bridge itself is tested with a
    fake JVM in tests/test_jvm_bridge.py."""
    from lumen.core.ijshim import jvm_bridge
    monkeypatch.setattr(jvm_bridge, "_importable", lambda name: False)
    monkeypatch.setenv("LUMEN_JVM_MODE", "inprocess")  # the test checks the in-process error path (a JDK + Fiji machine would run it)
    jvm_bridge.reset()
    try:
        c, s = client()
        s.add_image(LumenImage(np.zeros((4, 4), np.uint8), name="z"))
        for lang, label in (("groovy", "Groovy"), ("beanshell", "BeanShell"), ("rhino", "JavaScript (Rhino)")):
            r = c.post("/api/script/run", json={"code": 'println "hi"', "language": lang})
            assert r.status_code == 200, r.text
            j = r.json()
            assert j["stdout"] == "" and j["language"] == lang and label in j["error"] and "lumen-imaging[jvm]" in j["error"] and "JDK" in j["error"]
        r = c.post("/api/script/run", json={"code": "print(len(lumen.images))", "language": "jython"})
        assert r.json() == {"stdout": "1\n", "error": None}
    finally:
        jvm_bridge.reset()
