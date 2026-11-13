"""Findings of the final re-check (E11, docs/USABILITY.md §3.14): provenance of a never-opened image, sample-open
lines in a folder batch."""
import numpy as np
import tifffile

import lumen.core.ops  # noqa: F401
from lumen.core.ops.batch import preview_lines
from lumen.core.session import Session


def test_a_resaved_projection_does_not_name_itself_as_source(tmp_path):
    s = Session()
    s.run_op("file.new", {"title": "st", "type": "8-bit", "width": 32, "height": 24, "slices": 4, "fill": "Ramp"})
    s.run_op("stacks.z_project", {"method": "Max Intensity"})
    proj = s.active
    out = tmp_path / "proj.tif"
    s.save_image(proj.id, str(out))
    s.run_op("filters.gaussian", {"sigma": 1.0})
    s.save_image(proj.id, str(out))  # ⌘S: same file again
    info = tifffile.TiffFile(out).imagej_metadata.get("Info", "")
    assert "Derived from: st ▸ Z Project (" in info
    assert f"Source: {out}" not in info and f"Source: {out.resolve()}" not in info


def test_an_opened_file_stays_the_source_after_save_as(tmp_path):
    src = tmp_path / "orig.tif"
    tifffile.imwrite(src, np.arange(64 * 48, dtype="uint16").reshape(48, 64))
    s = Session()
    img = s.open_file(str(src))
    out = tmp_path / "copy.tif"
    s.save_image(img.id, str(out))
    s.save_image(img.id, str(out))
    info = tifffile.TiffFile(out).imagej_metadata.get("Info", "")
    assert f"Source: {src}" in info or f"Source: {src.resolve()}" in info


def test_batch_list_leaves_out_sample_and_url_opens():
    lines = [
        'lumen.run("file.open_imagej_sample", name=\'Blobs (25K)\', use_cache=True)  # → "blobs.gif"',
        'lumen.run("file.open_url", url=\'https://example.org/a.tif\')',
        'lumen.run("filters.median", radius=2.0, shape=\'disk\')',
    ]
    out = preview_lines(lines)
    assert out["lines"] == [lines[2]] and len(out["skipped"]) == 2
