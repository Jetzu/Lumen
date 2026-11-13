"""Saved ImageJ TIFFs carry provenance (E7-16): a real ``ImageJ=`` version in the description, the Software tag, and
the processing log in the Info property (Fiji's Image ▸ Show Info), read back into ``metadata["Info"]`` on open."""
from __future__ import annotations

import numpy as np
import tifffile
from fastapi.testclient import TestClient

from lumen import __version__
from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.io import IMAGEJ_VERSION, PROCESSING_PREFIX, history_steps, processing_info, processing_log, read_image, write_image
from lumen.core.session import Session
from lumen.server.app import create_app


def _session_with_steps(tmp_path):
    s = Session()
    src = tmp_path / "src.tif"
    tifffile.imwrite(str(src), (np.arange(3 * 24 * 32, dtype=np.uint16).reshape(3, 24, 32) % 500), imagej=True, metadata={"axes": "ZYX"})
    s.open_file(str(src))
    s.run_op("stacks.z_project", {"method": "Max Intensity"})
    proj = s.active
    s.run_op("filters.gaussian", {"sigma": 1.0}, scope="slice")
    s.run_op("filters.median", {"radius": 3.0})
    s.run_op("edit.invert", {})
    s.undo(proj.id)  # the undone step is not part of the image any more: not logged
    return s, proj, src


def test_description_version_software_and_info_log(tmp_path):
    s, proj, src = _session_with_steps(tmp_path)
    assert IMAGEJ_VERSION == "1.54p"
    out = write_image(proj, tmp_path / "saved.tif", history=s.histories[proj.id])
    with tifffile.TiffFile(out) as tf:
        desc = tf.pages[0].description
        assert desc.startswith("ImageJ=1.54") and desc.startswith(f"ImageJ={IMAGEJ_VERSION}\n") and desc.isascii()
        assert tf.pages[0].tags["Software"].value == f"Lumen {__version__}"
        info = tf.imagej_metadata["Info"]
    lines = info.splitlines()
    assert lines[0].startswith(f"{PROCESSING_PREFIX} Lumen {__version__}, written 20")
    assert any(l.startswith(f"{PROCESSING_PREFIX} Derived from: src.tif ▸ Z Project (") for l in lines)  # with the parameters (E10-4)
    assert f"{PROCESSING_PREFIX} Source: {src}" in lines
    assert f"{PROCESSING_PREFIX} 1. Gaussian Blur (sigma=1.0, scaled_units=False)" in lines
    assert f"{PROCESSING_PREFIX} 2. Median (radius=3.0, shape='disk')" in lines
    assert not any("Invert" in ln for ln in lines) and all(ln.startswith(PROCESSING_PREFIX) for ln in lines)
    # read back: Image Info lists it (scalar metadata), and the macro layer's getInfo sees the same text
    back = read_image(out)
    assert back.metadata["Info"] == info and back.summary()["metadata"]["Info"] == info
    assert history_steps(proj, s.histories[proj.id]) == [("Gaussian Blur", "sigma=1.0, scaled_units=False"), ("Median", "radius=3.0, shape='disk'")]
    assert history_steps(proj, [("A", "x=1"), ("B", "")]) == [("A", "x=1"), ("B", "")]


def test_save_as_through_the_app_session_finds_the_history(tmp_path):
    """Session.save_image passes no history: the writer finds it in the session the app serves (what Save As does)."""
    s, proj, _src = _session_with_steps(tmp_path)
    c = TestClient(create_app(s))
    r = c.post("/api/ops/file.new/run", json={"params": {"title": "other", "type": "8-bit", "width": 8, "height": 8}})
    assert r.status_code == 200
    out = s.save_image(proj.id, str(tmp_path / "via_session.tif"))
    with tifffile.TiffFile(out) as tf:
        info = tf.imagej_metadata["Info"]
        assert tf.pages[0].description.startswith("ImageJ=1.54")
    assert f"{PROCESSING_PREFIX} 1. Gaussian Blur (sigma=1.0, scaled_units=False)" in info.splitlines()
    assert f"{PROCESSING_PREFIX} 2. Median (radius=3.0, shape='disk')" in info.splitlines()
    # the other image has no steps: version and provenance lines only
    other = s.get_image("other")
    log = processing_log(other)
    assert len(log) == 1 and log[0].startswith(f"{PROCESSING_PREFIX} Lumen {__version__}")
    # re-opening the saved file and saving again keeps the earlier log and appends this session's
    s2 = Session()
    s2.open_file(out)
    img2 = s2.active
    assert img2.metadata["Info"] == info
    s2.run_op("filters.gaussian", {"sigma": 3.0}, image_id=img2.id)  # a non-default value: the History detail lists non-default parameters only
    out2 = write_image(img2, tmp_path / "again.tif", history=s2.histories[img2.id])
    with tifffile.TiffFile(out2) as tf:
        again = tf.imagej_metadata["Info"]
    assert again.startswith(info) and again.count(f"{PROCESSING_PREFIX} Lumen ") == 2
    assert again.splitlines()[-1] == f"{PROCESSING_PREFIX} 1. Gaussian Blur (sigma=3.0, scaled_units=False)"


def test_existing_info_is_kept_and_images_outside_a_session_still_get_the_version(tmp_path):
    img = LumenImage(np.zeros((4, 4), np.uint8), name="x", metadata={"Info": "Acquired on scope X\nObjective 63x"})
    text = processing_info(img)
    assert text.startswith("Acquired on scope X\nObjective 63x\n" + PROCESSING_PREFIX)
    out = write_image(img, tmp_path / "x.tif")
    with tifffile.TiffFile(out) as tf:
        assert tf.imagej_metadata["Info"] == text
        assert tf.pages[0].description.startswith("ImageJ=1.54")
    assert read_image(out).metadata["Info"] == text
    # RGB and hyperstacks take the same path
    rgb = LumenImage(np.zeros((4, 4, 3), np.uint8), name="rgb", rgb=True)
    out = write_image(rgb, tmp_path / "rgb.tif")
    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].description.startswith("ImageJ=1.54") and tf.imagej_metadata["Info"].startswith(PROCESSING_PREFIX)
    back = read_image(out)
    assert back.rgb and back.metadata["Info"].startswith(PROCESSING_PREFIX)
