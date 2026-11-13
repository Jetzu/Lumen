"""Bio-Formats parity: series listing, the importer options, the exporter, TIFF virtual stacks, Stack Slicer,
the plugins configuration table, the ImageJ command mappings and the Ext.* macro extensions."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lumen.core import ops  # noqa: F401  (registers the commands)
from lumen.core import io
from lumen.core.image import Calibration, LumenImage
from lumen.core.ij_compat import COMMANDS, IJ, transpile
from lumen.core.ops import bioformats_ops as bf
from lumen.core.session import Session


@pytest.fixture
def multi_ome(tmp_path):
    """Two OME series: a 16-bit T/Z/C hyperstack with calibration + channel names, and an 8-bit Z stack."""
    p = tmp_path / "multi.ome.tif"
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4000, (3, 2, 4, 20, 24)).astype(np.uint16)  # T Z C Y X
    b = rng.integers(0, 255, (5, 10, 12)).astype(np.uint8)
    with tifffile.TiffWriter(str(p), ome=True) as tw:
        tw.write(a, metadata={"axes": "TZCYX", "Name": "cells", "PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm", "PhysicalSizeY": 0.25,
                              "PhysicalSizeYUnit": "µm", "PhysicalSizeZ": 1.5, "PhysicalSizeZUnit": "µm", "TimeIncrement": 2.0,
                              "TimeIncrementUnit": "s", "Channel": {"Name": ["DAPI", "GFP", "RFP", "Cy5"]}})
        tw.write(b, metadata={"axes": "ZYX", "Name": "overview", "PhysicalSizeX": 4.0, "PhysicalSizeXUnit": "µm"})
    return p, a, b


@pytest.fixture
def pyramid(tmp_path):
    """An RGB pyramidal TIFF with three resolution levels (SubIFDs, tiled) and a 1 µm pixel size."""
    p = tmp_path / "pyr.tif"
    base = np.random.default_rng(1).integers(0, 255, (64, 96, 3)).astype(np.uint8)
    with tifffile.TiffWriter(str(p)) as tw:
        tw.write(base, subifds=2, tile=(16, 16), photometric="rgb", resolution=(10000, 10000), resolutionunit="CENTIMETER")
        tw.write(base[::2, ::2], subfiletype=1, tile=(16, 16), photometric="rgb")
        tw.write(base[::4, ::4], subfiletype=1, tile=(16, 16), photometric="rgb")
    return p, base


@pytest.fixture
def stack():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 4000, (2, 3, 2, 17, 23)).astype(np.uint16)
    return LumenImage(data, name="stack", axes="TZCYX", calibration=Calibration(0.5, 0.5, 2.0, "µm", frame_interval=1.5), channel_names=["DAPI", "GFP"])


def _macro_path(p) -> str:
    return str(p).replace("\\", "/")


# ----------------------------------------------------------------------------- series listing / level reads
def test_list_series_multi_ome(multi_ome):
    p, a, b = multi_ome
    infos = io.list_series(p)
    assert [i.name for i in infos] == ["cells", "overview"] and [i.index for i in infos] == [0, 1]
    c = infos[0]
    assert (c.size_x, c.size_y, c.size_z, c.size_c, c.size_t) == (24, 20, 2, 4, 3) and c.dtype == "uint16" and c.pixel_type == "uint16"
    assert c.dimension_order == "XYCZT" and c.image_count == 24 and c.channel_names == ["DAPI", "GFP", "RFP", "Cy5"] and c.format == "OME-TIFF"
    assert c.pixel_width == pytest.approx(0.25) and c.pixel_depth == pytest.approx(1.5) and c.unit == "µm"
    assert c.time_increment == pytest.approx(2.0) and c.time_unit == "sec" and c.levels == [(20, 24)] and c.level_count == 1
    o = infos[1]
    assert (o.size_x, o.size_y, o.size_z, o.size_c, o.size_t) == (12, 10, 5, 1, 1) and o.pixel_type == "uint8" and o.pixel_width == pytest.approx(4.0)
    # Bio-Formats getIndex / getZCTCoords round-trip in the series' dimension order
    for z in range(2):
        for ch in range(4):
            for t in range(3):
                assert c.zct_coords(c.plane_index(z, ch, t)) == (z, ch, t)
    assert c.plane_index(1, 2, 0) == 2 + 4 * 1
    rows = c.metadata_rows()
    assert ["SizeC", 4] in rows and ["Channel 1 Name", "GFP"] in rows and c.to_row()[:3] == [0, "cells", 24]
    # reading series 1 honours its own OME calibration (not the first Image's)
    img = io.read_image(p, series=1)
    assert img.shape == (1, 5, 1, 10, 12) and np.array_equal(img.data[0, :, 0], b) and img.calibration.pixel_width == pytest.approx(4.0)
    assert img.metadata["series_count"] == 2 and img.metadata["series_name"] == "overview" and img.metadata["series_names"] == ["cells", "overview"]
    first = io.read_image(p)  # default: series 0, the other series are listed in the metadata
    assert first.shape == (3, 2, 4, 20, 24) and [ch.name for ch in first.display.channels] == ["DAPI", "GFP", "RFP", "Cy5"]
    assert first.metadata["series"] == 0 and first.metadata["series_names"] == ["cells", "overview"]


def test_pyramid_levels(pyramid):
    p, base = pyramid
    infos = io.list_series(p)
    assert len(infos) == 1 and infos[0].levels == [(64, 96), (32, 48), (16, 24)] and infos[0].level_count == 3 and infos[0].rgb
    assert infos[0].size_c == 3 and infos[0].effective_size_c == 1 and infos[0].image_count == 1 and infos[0].pixel_width == pytest.approx(1.0)
    full = io.read_image(p)
    assert full.rgb and full.width == 96 and full.metadata["levels"] == 3 and full.metadata["level"] == 0
    lvl2 = io.read_image(p, level=2)
    assert lvl2.rgb and (lvl2.height, lvl2.width) == (16, 24) and np.array_equal(np.transpose(lvl2.data[0, 0], (1, 2, 0)), base[::4, ::4])
    assert lvl2.calibration.pixel_width == pytest.approx(4.0) and lvl2.calibration.unit == "µm" and lvl2.metadata["level"] == 2
    assert lvl2.metadata["source"] == {"kind": "file", "path": str(p), "level": 2}
    assert io.read_image(p, level=99).width == 24  # clamped to the last level
    v = io.read_image(p, level=1, virtual=True)  # tiled pages: a VirtualStack over the level's pages
    assert isinstance(v.data, io.VirtualStack) and v.rgb and v.shape == (1, 1, 3, 32, 48)
    assert np.array_equal(np.transpose(v.data[0, 0], (1, 2, 0)), base[::2, ::2])


def test_ome_xml_text(multi_ome, tmp_path):
    p, _a, _b = multi_ome
    xml = io.ome_xml_of(p)
    assert xml.startswith("<?xml") and 'Name="cells"' in xml and "DAPI" in xml
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(str(plain), np.zeros((3, 8, 9), np.uint16), imagej=True, metadata={"axes": "ZYX"})
    synth = io.ome_xml_of(plain)  # synthesised for files without OME metadata
    assert '<Pixels' in synth and 'SizeX="9"' in synth and 'SizeZ="3"' in synth and 'Type="uint16"' in synth


def test_group_file_pattern(tmp_path):
    folder, regex, axis = io.group_file_pattern(tmp_path / "img_t003.tif")
    assert folder == tmp_path and regex == r"^img_t\d+\.tif$" and axis == "T"
    assert io.group_file_pattern(tmp_path / "stack_z12.tif")[2] == "Z" and io.group_file_pattern(tmp_path / "cells_ch2.tif")[2] == "C"
    assert io.group_file_pattern(tmp_path / "x.tif", "img_t<1-30>.tif")[1] == r"^img_t\d+\.tif$"
    assert io.group_file_pattern(tmp_path / "img007.tif")[2] is None
    with pytest.raises(ValueError, match="no number"):
        io.group_file_pattern(tmp_path / "noindex.tif")


def test_file_pattern_globs_and_regexes(tmp_path):
    """Globs that happen to be valid regexes (img_t*.tif) are still treated as globs; regex idioms stay regexes."""
    import re
    rx = io.group_file_pattern(tmp_path / "x.tif", "img_t*.tif")[1]
    assert re.search(rx, "img_t001.tif") and not re.search(rx, "img_t001.png") and not re.search(rx, "img.tif")
    assert not re.search(rx, "old_img_t001.tif")  # a glob matches the whole name, like pathlib.glob
    assert re.search(io.compile_file_pattern("img_*.tif"), "img_t001.tif") and not re.search(io.compile_file_pattern("img_*.tif"), "imgX.tif")
    assert re.search(io.compile_file_pattern("*.tif"), "anything.tif") and re.search(io.compile_file_pattern("img_t???.tif"), "img_t001.tif")
    for regex in ("img_t.*tif", r"\.tif$", "cell_.*.tif", r"img_t\d+\.tif", "_ch1"):  # documented regex forms are returned unchanged
        assert io.compile_file_pattern(regex) == regex
    assert re.search(io.compile_file_pattern("cell_.*.tif"), "cell_01.tif")
    assert io.compile_file_pattern("") == "" and io.compile_file_pattern(None) == ""
    for bad in ("(", "["):
        with pytest.raises(ValueError, match="Invalid file pattern"):
            io.compile_file_pattern(bad)
    # end to end: the importer's group_pattern accepts the glob
    for i in range(1, 4):
        tifffile.imwrite(str(tmp_path / f"img_t{i:03d}.tif"), np.full((8, 8), i, np.uint8))
    tifffile.imwrite(str(tmp_path / "img_t001.png"), np.zeros((8, 8), np.uint8))
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "img_t001.tif"), "group_files": True, "group_pattern": "img_t*.tif"})
    im = s.images[out["new_images"][0]]
    assert im.shape == (1, 3, 1, 8, 8) and len(im.metadata["used_files"]) == 3 and all(f.endswith(".tif") for f in im.metadata["used_files"])


def test_series_spec_parsing(multi_ome):
    p, _a, _b = multi_ome
    infos = io.list_series(p)
    assert bf.parse_series_spec("", infos) == [0] and bf.parse_series_spec("all", infos) == [0, 1] and bf.parse_series_spec("1", infos) == [1]
    assert bf.parse_series_spec("0-1", infos) == [0, 1] and bf.parse_series_spec("over", infos) == [1] and bf.parse_series_spec("", infos, open_all=True) == [0, 1]
    with pytest.raises(ValueError, match="does not exist"):
        bf.parse_series_spec("5", infos)
    with pytest.raises(ValueError, match="No series named"):
        bf.parse_series_spec("nope", infos)


# ----------------------------------------------------------------------------- the importer command
def test_importer_series_all_and_metadata(multi_ome):
    p, a, b = multi_ome
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(p), "series": "all", "display_metadata": True, "display_ome_xml": True})
    ims = [s.images[i] for i in out["new_images"]]
    assert [im.name for im in ims] == ["multi - cells", "multi - overview"] and ims[0].shape == (3, 2, 4, 20, 24) and ims[1].shape == (1, 5, 1, 10, 12)
    assert np.array_equal(ims[0].data, a) and "2 of 2 series" in out["message"]
    table = out["table"]
    assert table["name"] == "Metadata" and table["columns"] == ["Key", "Value"] and ["SizeC", 4] in table["rows"] and ["Channel 0 Name", "DAPI"] in table["rows"]
    assert "OME-XML" in s.tables and any("cells" in r[0] for r in s.tables["OME-XML"]["rows"])
    assert any("has 2 series" in e["msg"] for e in s.log)
    out = s.run_op("file.bioformats_import", {"path": str(p), "series": "overview"})
    assert s.images[out["new_images"][0]].shape == (1, 5, 1, 10, 12)


def test_importer_range_crop_split_autoscale_color(multi_ome):
    p, a, _b = multi_ome
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(p), "specify_range": True, "c_begin": 2, "c_end": 3, "z_begin": 2, "t_begin": 1, "t_end": 3, "t_step": 2,
                                              "crop": True, "crop_x": 4, "crop_y": 2, "crop_width": 10, "crop_height": 8, "split_channels": True,
                                              "autoscale": False, "color_mode": "Grayscale"})
    ims = [s.images[i] for i in out["new_images"]]
    assert [im.name for im in ims] == ["multi - cells - C=0", "multi - cells - C=1"]
    assert all(im.shape == (2, 1, 1, 8, 10) for im in ims) and [im.display.channels[0].name for im in ims] == ["GFP", "RFP"]
    assert np.array_equal(ims[1].data[:, 0, 0], a[0:3:2, 1, 2, 2:10, 4:14])
    assert all(im.display.channels[0].lut == "Grays" and (im.display.channels[0].min, im.display.channels[0].max) == (0.0, 65535.0) for im in ims)
    # autoscale on: the channel's display range is the data range; Colorized shows one coloured channel at a time
    out = s.run_op("file.bioformats_import", {"path": str(p), "autoscale": True, "color_mode": "Colorized"})
    im = s.images[out["new_images"][0]]
    assert im.display.mode == "single" and im.display.channels[1].min == float(a[:, :, 1].min()) and im.display.channels[1].max == float(a[:, :, 1].max())
    out = s.run_op("file.bioformats_import", {"path": str(p), "color_mode": "Custom", "custom_luts": "Red, Fire", "split_timepoints": True, "split_focal_planes": True})
    ims = [s.images[i] for i in out["new_images"]]
    assert len(ims) == 6 and ims[0].name == "multi - cells - Z=0 - T=0" and ims[-1].name == "multi - cells - Z=1 - T=2" and ims[0].shape == (1, 1, 4, 20, 24)
    assert [ch.lut for ch in ims[0].display.channels][:2] == ["Red", "Fire"] and ims[0].display.mode == "composite"
    with pytest.raises(ValueError, match="Unknown LUT"):
        s.run_op("file.bioformats_import", {"path": str(p), "color_mode": "Custom", "custom_luts": "NoSuchLut"})
    with pytest.raises(ValueError, match="outside"):
        s.run_op("file.bioformats_import", {"path": str(p), "crop": True, "crop_x": 50, "crop_y": 0, "crop_width": 5, "crop_height": 5})


def test_importer_view_swap_level_and_rgb(multi_ome, pyramid):
    p, a, _b = multi_ome
    s = Session()
    # Standard ImageJ view: one flat Z stack in stack order (XYZCT: Z fastest, then C, then T) with plane labels
    out = s.run_op("file.bioformats_import", {"path": str(p), "view": "Standard ImageJ", "stack_order": "XYZCT"})
    im = s.images[out["new_images"][0]]
    from lumen.core.ops.stacks import slice_labels
    assert im.shape == (1, 24, 1, 20, 24) and im.metadata["slice_labels"][:3] == ["c:1/4; z:1/2; t:1/3", "c:1/4; z:2/2; t:1/3", "c:2/4; z:1/2; t:1/3"]
    assert slice_labels(im)[:3] == im.metadata["slice_labels"][:3] and len(im.metadata["slice_labels"]) == 24 and "labels" not in im.metadata
    assert np.array_equal(im.data[0, 3, 0], a[0, 1, 1])  # z fastest: plane 3 = c=1, z=1, t=0
    # swap dimensions: the file's Z is time and its T is depth
    out = s.run_op("file.bioformats_import", {"path": str(p), "swap_dimensions": True, "swap_order": "TCZ"})
    im = s.images[out["new_images"][0]]
    assert im.shape == (2, 3, 4, 20, 24) and np.array_equal(im.data[1, 2, 3], a[2, 1, 3]) and im.metadata["swap_dimensions"] == "TCZ"
    # pyramid level of an RGB whole-slide style file
    pp, base = pyramid
    out = s.run_op("file.bioformats_import", {"path": str(pp), "level": 1})
    im = s.images[out["new_images"][0]]
    assert im.rgb and (im.height, im.width) == (32, 48) and im.calibration.pixel_width == pytest.approx(2.0)
    out = s.run_op("file.bioformats_import", {"path": str(pp), "level": 2, "split_channels": True})  # RGB is never split by channel
    assert len(out["new_images"]) == 1 and s.images[out["new_images"][0]].rgb


def test_importer_group_files(tmp_path):
    for i in range(1, 4):
        tifffile.imwrite(str(tmp_path / f"img_t{i:03d}.tif"), np.full((8, 8), i * 10, np.uint8))
    tifffile.imwrite(str(tmp_path / "other.tif"), np.zeros((8, 8), np.uint8))
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "img_t002.tif"), "group_files": True})
    im = s.images[out["new_images"][0]]
    assert im.shape == (3, 1, 1, 8, 8) and [int(im.data[t, 0, 0, 0, 0]) for t in range(3)] == [10, 20, 30]
    assert im.metadata["group_axis"] == "T" and len(im.metadata["used_files"]) == 3 and im.name == "img_t002"
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "img_t001.tif"), "group_files": True, "group_pattern": "img_t<1-2>.tif"})
    assert s.images[out["new_images"][0]].nt == 3  # the pattern selects the numbered siblings (range bounds are not enforced)


def test_importer_group_files_stk_lsm_siblings_and_revert(tmp_path):
    """Numbered MetaMorph .stk / Zeiss .lsm siblings (the classic group-files case) group too, and Revert replays the grouping."""
    for ext in ("stk", "lsm"):
        folder = tmp_path / ext
        folder.mkdir()
        for i in range(1, 4):
            tifffile.imwrite(str(folder / f"scan_{i}.{ext}"), np.full((8, 8), i, np.uint16))
        s = Session()
        out = s.run_op("file.bioformats_import", {"path": str(folder / f"scan_1.{ext}"), "group_files": True})
        im = s.images[out["new_images"][0]]
        assert im.shape == (1, 3, 1, 8, 8) and [int(im.data[0, z, 0, 0, 0]) for z in range(3)] == [1, 2, 3] and len(im.metadata["used_files"]) == 3
        assert im.metadata["source"]["kind"] == "sequence" and f".{ext}" in im.metadata["source"]["exts"]  # the extension set is recorded for Revert
        s.run_op("image.crop", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2}, image_id=im.id)
        assert im.shape == (1, 3, 1, 2, 2)
        s.run_op("file.revert", {}, image_id=im.id)
        assert im.shape == (1, 3, 1, 8, 8) and [int(im.data[0, z, 0, 0, 0]) for z in range(3)] == [1, 2, 3]
    # File ▸ Import ▸ Image Sequence keeps its narrower default set (the .stk files are not swept into a plain sequence read)
    with pytest.raises(FileNotFoundError):
        io.read_sequence(tmp_path / "stk")
    assert io.read_sequence(tmp_path / "stk", exts=io.IMAGE_EXTS).nz == 3


def test_remote_importer_reuses_the_url_download(multi_ome, monkeypatch, tmp_path):
    p, _a, b = multi_ome
    calls = []

    def fake_download(url, timeout=60.0):
        calls.append(url)
        target = tmp_path / "dl" / "remote.ome.tif"
        target.parent.mkdir()
        shutil.copy(p, target)
        return target, "remote.ome.tif"

    from lumen.core.ops import import_export
    s = Session()
    with pytest.raises(ValueError, match="http"):
        s.run_op("file.bioformats_remote_import", {"url": "ftp://nope"})
    monkeypatch.setattr(import_export, "download_url", fake_download)
    out = s.run_op("file.bioformats_remote_import", {"url": "https://example.org/remote.ome.tif", "series": "1", "autoscale": True})
    im = s.images[out["new_images"][0]]
    assert calls == ["https://example.org/remote.ome.tif"] and im.shape == (1, 5, 1, 10, 12) and np.array_equal(im.data[0, :, 0], b)
    assert im.metadata["url"] == "https://example.org/remote.ome.tif" and im.path is None and im.name.startswith("remote - overview")


def test_series_table_and_config_table(multi_ome):
    p, _a, _b = multi_ome
    s = Session()
    out = s.run_op("file.bioformats_series", {"path": str(p)})
    t = out["table"]
    assert t["name"] == "Bio-Formats Series" and t["columns"][:3] == ["Index", "Name", "SizeX"] and [r[1] for r in t["rows"]] == ["cells", "overview"]
    assert t["rows"][0][12] == "DAPI, GFP, RFP, Cy5" and "2 series" in out["message"]
    out = s.run_op("file.bioformats_config", {})
    rows = out["table"]["rows"]
    assert out["table"]["name"] == "Bio-Formats Plugins Configuration" and rows[0][3] == "enabled"
    assert any(r[0].startswith("Nikon ND2") for r in rows) and any(r[2] == "Java Bio-Formats (bridge)" for r in rows)
    assert any(r[0].startswith("Gatan DigitalMicrograph") and r[3] == "enabled" for r in rows)


# ----------------------------------------------------------------------------- exporter
def test_exporter_compression_and_metadata(stack, tmp_path):
    s = Session()
    s.add_image(stack)
    stack.metadata.update({"experiment": "exp-7", "run": 3, "ok": True})
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "s.ome.tif"), "compression": "LZW", "description": "test export"})
    f = tmp_path / "s.ome.tif"
    assert f.exists() and out["table"]["rows"][0][0] == "s.ome.tif" and "LZW" in out["message"]
    with tifffile.TiffFile(str(f)) as tf:
        assert tf.pages[0].compression.name == "LZW" and "<Description>test export</Description>" in tf.ome_metadata
    back = io.read_image(f)
    assert np.array_equal(back.data, stack.data) and [c.name for c in back.display.channels] == ["DAPI", "GFP"]
    assert back.metadata["experiment"] == "exp-7" and back.metadata["run"] == 3 and back.metadata["ok"] is True and back.metadata["description"] == "test export"
    assert back.calibration.pixel_width == pytest.approx(0.5) and back.calibration.frame_interval == pytest.approx(1.5)
    s.run_op("file.bioformats_export", {"path": str(tmp_path / "plain.ome.tif"), "embed_metadata": False})
    assert "experiment" not in io.read_image(tmp_path / "plain.ome.tif").metadata
    with pytest.raises(ValueError, match="8-bit"):
        s.run_op("file.bioformats_export", {"path": str(tmp_path / "j.ome.tif"), "compression": "JPEG"})
    s.run_op("file.bioformats_export", {"path": str(tmp_path / "ij.tif"), "format": "ImageJ TIFF", "compression": "zlib"})
    with tifffile.TiffFile(str(tmp_path / "ij.tif")) as tf:
        assert tf.is_imagej and tf.pages[0].compression.name == "ADOBE_DEFLATE"
    assert np.array_equal(io.read_image(tmp_path / "ij.tif").data, stack.data)
    pytest.importorskip("zarr")
    s.run_op("file.bioformats_export", {"path": str(tmp_path / "z.ome.zarr"), "format": "OME-Zarr"})
    assert (tmp_path / "z.zarr").is_dir() and np.array_equal(io.read_image(tmp_path / "z.zarr").data, stack.data)


def test_exporter_split_files_and_range(stack, tmp_path):
    s = Session()
    s.add_image(stack)
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "s.ome.tif"), "write_each_z_section": True, "write_each_channel": True})
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == sorted(f"s_Z{z}_C{c}.ome.tif" for z in range(3) for c in range(2)) and len(out["table"]["rows"]) == 6
    piece = io.read_image(tmp_path / "s_Z1_C1.ome.tif")
    assert piece.shape == (2, 1, 1, 17, 23) and np.array_equal(piece.data[:, 0, 0], stack.data[:, 1, 1]) and piece.display.channels[0].name == "GFP"
    # save range + timepoints, zero-padded names
    out_dir = tmp_path / "t"
    out_dir.mkdir()
    big = LumenImage(np.random.default_rng(3).integers(0, 255, (12, 1, 1, 4, 5)).astype(np.uint8), name="v", axes="TZCYX")
    s.add_image(big)
    s.run_op("file.bioformats_export", {"path": str(out_dir / "v.ome.tif"), "write_each_timepoint": True, "padded": True, "t_begin": 3, "t_end": 12}, image_id=big.id)
    names = sorted(f.name for f in out_dir.iterdir())
    assert names[0] == "v_T00.ome.tif" and names[-1] == "v_T09.ome.tif" and len(names) == 10
    assert np.array_equal(io.read_image(out_dir / "v_T00.ome.tif").data[0, 0, 0], big.data[2, 0, 0])
    # a split that does not apply (one channel) writes a single file
    s.run_op("file.bioformats_export", {"path": str(out_dir / "single.ome.tif"), "write_each_channel": True}, image_id=big.id)
    assert (out_dir / "single.ome.tif").exists()


def test_exporter_writes_more_than_500_files(tmp_path):
    """The exporter only writes files: split_image's 500-window cap (importer / Stack Slicer) must not apply, its own 10000 cap does."""
    s = Session()
    tl = LumenImage(np.zeros((600, 1, 1, 16, 16), np.uint8), name="tl", axes="TZCYX")
    s.add_image(tl)
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "tl" / "tl.ome.tif"), "write_each_timepoint": True}, image_id=tl.id)
    assert len(list((tmp_path / "tl").iterdir())) == 600 and len(out["table"]["rows"]) == 600 and out["table"]["rows"][599][0] == "tl_T599.ome.tif"
    huge = LumenImage(np.zeros((10001, 1, 1, 2, 2), np.uint8), name="h", axes="TZCYX")
    s.add_image(huge)
    with pytest.raises(ValueError, match="would write 10001 files"):
        s.run_op("file.bioformats_export", {"path": str(tmp_path / "h" / "h.ome.tif"), "write_each_timepoint": True}, image_id=huge.id)
    with pytest.raises(ValueError, match="limit 500"):  # the UI-oriented default still protects the importer / Stack Slicer
        bf.split_image(tl, by_t=True)
    assert len(bf.split_image(tl, by_t=True, limit=None)) == 600


def test_exporter_split_indices_come_from_this_split_only(tmp_path):
    """A Stack Slicer / importer piece ('cells - C=1') exported by Z gets cells_Z0.. names and an empty C column, not a stale _C1."""
    s = Session()
    img = LumenImage(np.zeros((1, 3, 2, 8, 8), np.uint8), name="cells", axes="TZCYX")
    s.add_image(img)
    out = s.run_op("file.stack_slicer", {"split_channels": True}, image_id=img.id)
    piece = s.images[out["new_images"][1]]
    assert piece.name == "cells - C=1"
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "c" / "cells.ome.tif"), "write_each_z_section": True}, image_id=piece.id)
    assert sorted(f.name for f in (tmp_path / "c").iterdir()) == ["cells_Z0.ome.tif", "cells_Z1.ome.tif", "cells_Z2.ome.tif"]
    assert [r[1:4] for r in out["table"]["rows"]] == [["", 0, ""], ["", 1, ""], ["", 2, ""]]
    out = s.run_op("file.stack_slicer", {"split_channels": False, "split_z": True}, image_id=img.id)  # the mirror case: a Z piece by channel
    zp = s.images[out["new_images"][2]]
    assert zp.name == "cells - Z=2"
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "z" / "cells.ome.tif"), "write_each_channel": True}, image_id=zp.id)
    assert sorted(f.name for f in (tmp_path / "z").iterdir()) == ["cells_C0.ome.tif", "cells_C1.ome.tif"] and [r[2] for r in out["table"]["rows"]] == ["", ""]


# ----------------------------------------------------------------------------- virtual stacks
def test_virtual_memmap_open_modify_undo_save(stack, tmp_path):
    p = io.write_image(stack, tmp_path / "u.tif")  # uncompressed ImageJ TIFF → copy-on-write memmap
    s = Session()
    out = s.run_op("file.import_tiff_virtual", {"path": p})
    img = s.images[out["new_images"][0]]
    assert isinstance(img.data, np.memmap) and io.is_virtual(img.data) and "virtual" in out["message"] and "memory-mapped" in out["message"]
    assert img.shape == stack.shape and img.dtype == np.uint16 and np.array_equal(img.data[1, 2], stack.data[1, 2])  # the display fetch
    assert [c.name for c in img.display.channels] == ["DAPI", "GFP"] and img.calibration.pixel_width == pytest.approx(0.5)
    assert img.metadata["source"]["virtual"] is True
    s.run_op("math.add", {"value": 5}, image_id=img.id)  # the first modifying command materialises the stack
    assert not io.is_virtual(img.data) and np.array_equal(img.data, np.clip(stack.data.astype(np.int64) + 5, 0, 65535))
    assert s.undo(img.id) == "Add" and np.array_equal(img.data, stack.data)
    assert np.array_equal(tifffile.imread(p), stack.data)  # the file itself is never written to
    back = io.read_image(s.save_image(img.id, str(tmp_path / "saved.tif")))
    assert np.array_equal(back.data, stack.data)
    with pytest.raises(ValueError, match="TIFF"):
        s.run_op("file.import_tiff_virtual", {"path": str(tmp_path / "x.png")})


def test_virtual_stack_compressed_planes_on_demand(stack, tmp_path):
    p = io.write_image(stack, tmp_path / "c.tif", compress=True)  # zlib pages → VirtualStack
    img = io.read_image(p, virtual=True)
    vs = img.data
    assert isinstance(vs, io.VirtualStack) and io.is_virtual(vs) and vs.shape == stack.shape and vs.dtype == np.uint16 and len(vs) == 2
    assert vs.nbytes < stack.data.nbytes / 2  # only the planes the display range needed are cached
    assert np.array_equal(vs[1, 2, 1], stack.data[1, 2, 1]) and np.array_equal(vs[0, 1], stack.data[0, 1])
    assert np.array_equal(vs[:, 1, :, ::4, 3:9], stack.data[:, 1, :, ::4, 3:9]) and vs[1, -1, 0, 3, 4] == stack.data[1, 2, 0, 3, 4]
    assert np.array_equal(vs[:, :, :, ::5, ::5], stack.data[:, :, :, ::5, ::5])
    with pytest.raises(IndexError):
        vs[5]
    assert not vs.materialized and np.array_equal(vs.copy(), stack.data) and not vs.materialized
    assert float(vs.max()) == float(stack.data.max()) and vs.materialized  # reductions materialise (copy-on-write)
    s = Session()
    img2 = io.read_image(p, virtual=True)
    s.add_image(img2)
    assert img2.is_binary is False and io.is_virtual(img2.data)
    ref = s.add_image(LumenImage(stack.data.copy(), name="ref", axes="TZCYX"))
    s.run_op("edit.invert", {}, image_id=ref.id)
    s.run_op("edit.invert", {}, image_id=img2.id)  # copy-on-write: the virtual stack becomes an ordinary array
    assert isinstance(img2.data, np.ndarray) and not isinstance(img2.data, io.VirtualStack) and np.array_equal(img2.data, ref.data)
    assert not np.array_equal(img2.data, stack.data) and s.undo(img2.id) == "Invert"
    assert np.array_equal(img2.data, stack.data)
    out = s.run_op("file.import_tiff_virtual", {"path": p})
    assert "planes on demand" in out["message"]
    # Bio-Formats importer: virtual + split copies the selection, other formats fall back to memory with a note
    out = s.run_op("file.bioformats_import", {"path": p, "virtual": True, "split_channels": True})
    assert len(out["new_images"]) == 2
    tifffile.imwrite(str(tmp_path / "rgb.tif"), np.zeros((4, 6, 3), np.uint8), photometric="rgb")
    v = io.read_image(tmp_path / "rgb.tif", virtual=True)
    assert v.rgb and v.shape == (1, 1, 3, 4, 6) and io.is_virtual(v.data)
    np.save(str(tmp_path / "a.npy"), np.zeros((4, 4), np.uint8))
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "a.npy"), "virtual": True})
    assert "(virtual)" in out["message"] and io.is_virtual(s.images[out["new_images"][0]].data)  # npy files are memory-mapped too
    import imageio.v3 as iio
    iio.imwrite(str(tmp_path / "a.png"), np.zeros((4, 4), np.uint8))
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "a.png"), "virtual": True})
    assert "only available for TIFF" in out["message"]


def test_virtual_big_endian_tiff_stays_native(tmp_path):
    """ImageJ/Fiji write big-endian ('MM') TIFFs: a virtual open must not expose a '>u2' / '>f4' array (bit_depth KeyError,
    GET /api/images 500, silent float32 promotion by the first pixel op) – planes are byte-swapped on demand."""
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 4000, (3, 16, 20)).astype(np.uint16)
    p = tmp_path / "be.tif"
    tifffile.imwrite(str(p), arr, byteorder=">", imagej=True, metadata={"axes": "ZYX"})
    s = Session()
    out = s.run_op("file.import_tiff_virtual", {"path": str(p)})
    img = s.images[out["new_images"][0]]
    assert img.dtype == np.uint16 and img.bit_depth == "16-bit" and io.is_virtual(img.data) and "16-bit" in out["message"] and "virtual" in out["message"]
    assert isinstance(img.data, io.VirtualStack) and img.data.memmapped and img.data.dtype == np.uint16
    assert np.array_equal(img.data[0, 1, 0], arr[1]) and img.data[0, 2, 0].dtype.isnative
    assert [im.summary()["bit_depth"] for im in s.images.values()] == ["16-bit"]  # the /api/images listing
    s.run_op("math.add", {"value": 1}, image_id=img.id)
    assert img.dtype == np.uint16 and np.array_equal(img.data[0, :, 0], np.clip(arr.astype(np.int64) + 1, 0, 65535))
    f = rng.random((3, 16, 20)).astype(np.float32)
    tifffile.imwrite(str(tmp_path / "be_f.tif"), f, byteorder=">", imagej=True, metadata={"axes": "ZYX"})
    out = s.run_op("file.import_tiff_virtual", {"path": str(tmp_path / "be_f.tif")})
    fv = s.images[out["new_images"][0]]
    assert fv.dtype == np.float32 and fv.bit_depth == "32-bit" and io.is_virtual(fv.data) and np.array_equal(fv.data[0, 2, 0], f[2])
    # a single-IFD "truncated" ImageJ hyperstack (what Fiji writes beyond 4 GB) only has its first plane as a page:
    # the memmap is the only lazy reader that sees every plane
    hs = rng.integers(0, 4000, (2, 3, 2, 16, 20)).astype(np.uint16)
    tifffile.imwrite(str(tmp_path / "trunc.tif"), hs, byteorder=">", imagej=True, truncate=True, metadata={"axes": "TZCYX"})
    tv = io.read_image(tmp_path / "trunc.tif", virtual=True)
    assert io.is_virtual(tv.data) and tv.shape == hs.shape and np.array_equal(tv.data[1, 2, 1], hs[1, 2, 1]) and np.array_equal(np.asarray(tv.data), hs)
    out = s.run_op("file.bioformats_import", {"path": str(p), "virtual": True})
    assert s.images[out["new_images"][0]].dtype == np.uint16 and "(virtual)" in out["message"]
    # little-endian files are still plain copy-on-write memmaps; in-memory non-native arrays are byte-swapped, not promoted
    tifffile.imwrite(str(tmp_path / "le.tif"), arr, byteorder="<", imagej=True, metadata={"axes": "ZYX"})
    assert isinstance(io.read_image(tmp_path / "le.tif", virtual=True).data, np.memmap)
    coerced = LumenImage.coerce_dtype(arr.astype(">u2"))
    assert coerced.dtype == np.uint16 and coerced.dtype.isnative and np.array_equal(coerced, arr)
    assert LumenImage(arr.astype(">f4"), name="f", axes="ZYX").bit_depth == "32-bit"


def test_revert_keeps_a_virtual_stack_lazy(stack, tmp_path):
    """File ▸ Revert of a virtual stack re-reads lazily and keeps the image virtual (Fiji's revert never loads a virtual stack)."""
    p = io.write_image(stack, tmp_path / "c.tif", compress=True)
    s = Session()
    out = s.run_op("file.import_tiff_virtual", {"path": p})
    img = s.images[out["new_images"][0]]
    rev0 = img.rev
    s.run_op("file.revert", {}, image_id=img.id)
    assert isinstance(img.data, io.VirtualStack) and io.is_virtual(img.data) and img.rev > rev0
    assert np.array_equal(img.data[1, 2], stack.data[1, 2]) and [c.name for c in img.display.channels] == ["DAPI", "GFP"]
    assert s.undo(img.id) == "Revert" and np.array_equal(np.asarray(img.data), stack.data)
    pm = io.write_image(stack, tmp_path / "u.tif")  # uncompressed: the memmap stays a memmap
    out = s.run_op("file.import_tiff_virtual", {"path": pm})
    mm = s.images[out["new_images"][0]]
    s.run_op("file.revert", {}, image_id=mm.id)
    assert isinstance(mm.data, np.memmap) and io.is_virtual(mm.data)
    plain = s.add_image(io.read_image(pm))  # an in-memory image reverts as before
    s.run_op("image.crop", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2}, image_id=plain.id)
    s.run_op("file.revert", {}, image_id=plain.id)
    assert plain.shape == stack.shape and np.array_equal(plain.data, stack.data) and not io.is_virtual(plain.data)


# ----------------------------------------------------------------------------- Stack Slicer
def test_stack_slicer(stack):
    s = Session()
    s.add_image(stack)
    out = s.run_op("file.stack_slicer", {"split_channels": True, "keep_original": True})
    ims = [s.images[i] for i in out["new_images"]]
    assert [im.name for im in ims] == ["stack - C=0", "stack - C=1"] and all(im.shape == (2, 3, 1, 17, 23) for im in ims)
    assert ims[1].display.channels[0].name == "GFP" and stack.id in s.images
    out = s.run_op("file.stack_slicer", {"split_channels": False, "split_z": True, "split_timepoints": True, "keep_original": False}, image_id=stack.id)
    assert len(out["new_images"]) == 6 and stack.id not in s.images
    with pytest.raises(ValueError, match="Nothing to split"):
        s.run_op("file.stack_slicer", {"split_channels": True, "split_z": False, "split_timepoints": False}, image_id=ims[0].id)


def test_standard_imagej_view_labels_are_slice_labels(tmp_path):
    """The flat 'Standard ImageJ' stack carries Bio-Formats' 'c:…; z:…; t:…' labels as slice labels (uncapped), and the
    labels follow their planes through split / range / swap – read by Slice Labels to Results, Make Substack, TIFF save."""
    from lumen.core.ops.stacks import plane_index, slice_labels
    img = LumenImage(np.zeros((2, 3, 2, 8, 8), np.uint8), name="f", axes="TZCYX")
    flat = bf.flatten_to_stack(img, "XYCZT")
    assert flat.shape == (1, 12, 1, 8, 8) and "labels" not in flat.metadata
    assert slice_labels(flat)[:3] == ["c:1/2; z:1/3; t:1/2", "c:2/2; z:1/3; t:1/2", "c:1/2; z:2/3; t:1/2"] and len(flat.metadata["slice_labels"]) == 12
    big = bf.flatten_to_stack(LumenImage(np.zeros((3, 5, 4, 4, 4), np.uint8), name="b", axes="TZCYX"))
    assert len(big.metadata["slice_labels"]) == 60 and big.metadata["slice_labels"][-1] == "c:4/4; z:5/5; t:3/3"
    s = Session()
    s.add_image(flat)
    out = s.run_op("stacks.slice_labels_to_results", {}, image_id=flat.id)
    assert "12 of 12" in out["message"]
    pieces = bf.split_image(flat, by_z=True)  # view=Standard ImageJ + split focal planes: each piece gets its own label
    assert [slice_labels(p) for p in pieces[:2]] == [["c:1/2; z:1/3; t:1/2"], ["c:2/2; z:1/3; t:1/2"]]
    out = io.write_image(flat, tmp_path / "flat.tif")
    with tifffile.TiffFile(out) as tf:
        assert list(tf.imagej_metadata["Labels"]) == flat.metadata["slice_labels"]
    assert slice_labels(io.read_image(out)) == flat.metadata["slice_labels"]
    lab = [f"t{t}z{z}c{c}" for t in range(2) for z in range(3) for c in range(2)]
    img.metadata["slice_labels"] = lab
    assert slice_labels(bf.select_range(img, z=(2, 3, 1))) == [f"t{t}z{z}c{c}" for t in range(2) for z in (1, 2) for c in range(2)]
    assert [slice_labels(p) for p in bf.split_image(img, by_c=True)] == [[f"t{t}z{z}c{c}" for t in range(2) for z in range(3)] for c in range(2)]
    sw = bf.swap_dimensions(img, "TCZ")  # the file's Z is time, its T is depth
    assert sw.shape == (3, 2, 2, 8, 8)
    assert all(slice_labels(sw)[plane_index(sw, t, z, c)] == f"t{z}z{t}c{c}" for t in range(3) for z in range(2) for c in range(2))
    assert "slice_labels" not in bf.split_image(LumenImage(np.zeros((1, 2, 2, 4, 4), np.uint8), name="n", axes="TZCYX"), by_c=True)[0].metadata


# ----------------------------------------------------------------------------- ImageJ macro layer
def test_ij_commands_map_to_the_bioformats_ops(multi_ome, tmp_path):
    p, a, b = multi_ome
    s = Session()
    ij = IJ(s)
    for name in ("Bio-Formats Importer", "Bio-Formats", "Bio-Formats (Windowless)", "Bio-Formats Windowless Importer"):
        assert COMMANDS[name][1] == "file.bioformats_import"
    assert COMMANDS["Bio-Formats Exporter"][1] == "file.bioformats_export" and COMMANDS["TIFF Virtual Stack..."][1] == "file.import_tiff_virtual"
    assert COMMANDS["Stack Slicer"][1] == "file.stack_slicer" and COMMANDS["Bio-Formats Remote Importer"][1] == "file.bioformats_remote_import"
    ij.run("Bio-Formats Importer", f"open=[{p}] color_mode=Composite view=Hyperstack series_2 autoscale")  # series_N is 1-based
    assert ij.img.shape == (1, 5, 1, 10, 12) and ij.img.name == "multi - overview"
    ij.run("Bio-Formats Importer", f"open=[{p}] color_mode=Grayscale view=[Standard ImageJ] stack_order=XYCZT specify_range c_begin_1=1 c_end_1=2 c_step_1=1 "
                                   "z_begin_1=1 z_end_1=1 t_begin_1=1 t_end_1=3 t_step_1=2 crop x_coordinate_1=2 y_coordinate_1=3 width_1=6 height_1=5")
    assert ij.img.shape == (1, 4, 1, 5, 6) and np.array_equal(ij.img.data[0, 1, 0], a[0, 0, 1, 3:8, 2:8])
    ij.run("Bio-Formats Importer", f"open=[{p}] open_all_series split_channels")
    assert ij.nImages() == 2 + 4 + 1
    ij.run("Bio-Formats", f"open=[{p}] swap_dimensions z_1=T c_1=C t_1=Z display_metadata")
    assert ij.img.shape == (2, 3, 4, 20, 24) and "Metadata" in s.tables
    ij.run("Bio-Formats Exporter", f"save=[{tmp_path / 'exp.ome.tif'}] compression=LZW write_each_timepoint")
    assert sorted(f.name for f in tmp_path.glob("exp_*")) == ["exp_T0.ome.tif", "exp_T1.ome.tif"]
    ij.run("TIFF Virtual Stack...", f"open=[{tmp_path / 'exp_T0.ome.tif'}]")
    assert io.is_virtual(ij.img.data) and ij.img.shape == (1, 3, 4, 20, 24)
    ij.run("Stack Slicer", "split_channels keep_original")
    assert ij.img.name.endswith(" - C=3")
    out = ij.run("Bio-Formats Plugins Configuration")
    assert out["table"]["name"] == "Bio-Formats Plugins Configuration"
    with pytest.raises(ValueError, match="no file given"):
        ij.run("Bio-Formats Exporter", "compression=LZW")
    with pytest.raises(ValueError, match="unknown compression"):
        ij.run("Bio-Formats Exporter", "save=[x.ome.tif] compression=Brotli")


def test_macro_extensions_end_to_end(multi_ome):
    p, a, b = multi_ome
    s = Session()
    ij = IJ(s)
    path = _macro_path(p)
    out = ij.macro(f'''
run("Bio-Formats Macro Extensions");
Ext.setId("{path}");
Ext.getSeriesCount(n);
Ext.getSeriesName(first);
Ext.getSizeX(w0);
Ext.setSeries(1);
Ext.getSeries(cur);
Ext.getSeriesName(name);
Ext.getSizeX(w); Ext.getSizeY(h); Ext.getSizeZ(nz); Ext.getSizeC(nc); Ext.getSizeT(nt);
Ext.getImageCount(count); Ext.getPixelType(pt); Ext.getDimensionOrder(order); Ext.isInterleaved(inter);
Ext.getPixelsPhysicalSizeX(px); Ext.getPixelsPhysicalSizeZ(pz); Ext.getPixelsTimeIncrement(dt);
Ext.getMetadataValue("Nope", missing);
Ext.getFormat("{path}", fmt);
Ext.getCurrentFile(file);
Ext.getUsedFileCount(nfiles);
Ext.getUsedFile(0, f0);
Ext.getIndex(2, 0, 0, idx);
Ext.getZCTCoords(3, z, c, t);
Ext.getPlaneTimingDeltaT(delta, 0);
Ext.openImage("plane3", 3);
Ext.setSeries(0);
Ext.getPixelsPhysicalSizeX(px0);
Ext.getImageCount(count0);
Ext.getIndex(1, 2, 0, idx0);
Ext.getZCTCoords(6, z0, c0, t0);
Ext.openThumbImage("thumb", 0);
Ext.setGroupFiles("false");
Ext.isGroupFiles(gf);
Ext.setVirtual("true");
Ext.setOpenAllSeries(false);
Ext.openImagePlus("{path}");
print(n + "|" + first + "|" + w0 + "|" + cur + "|" + name + "|" + w + "x" + h + "|" + nz + "/" + nc + "/" + nt + "|" + count + "|" + pt + "|" + order + "|" + inter + "|" + px + "|" + pz + "|" + missing + "|" + fmt + "|" + nfiles + "|" + idx + "|" + z + "," + c + "," + t);
print(px0 + "|" + count0 + "|" + idx0 + "|" + z0 + "," + c0 + "," + t0 + "|" + gf + "|" + isNaN(delta) + "|" + isNaN(dt) + "|" + (file == f0));
Ext.close();
Ext.getSeriesCount(after);
print(after);
''')
    lines = out.strip().splitlines()
    assert lines[0] == "2|cells|24|1|overview|12x10|5/1/1|5|uint8|XYCZT|0|4|NaN|null|OME-TIFF|1|2|3,0,0"
    # XYCZT: index = c + sizeC * (z + sizeZ * t) → getIndex(1, 2, 0) = 6 and getZCTCoords(6) = (1, 2, 0); series 1 has no TimeIncrement (NaN)
    assert lines[1] == "0.25|24|6|1,2,0|0|1|1|1" and lines[2] == "0"
    names = [im.name for im in s.images.values()]
    assert names == ["plane3", "thumb", "multi - cells"]
    plane = s.get_image("plane3")
    assert plane.shape == (1, 1, 1, 10, 12) and np.array_equal(plane.data[0, 0, 0], b[3]) and plane.calibration.pixel_width == pytest.approx(4.0)
    assert s.get_image("thumb").shape == (1, 1, 1, 20, 24)  # ≤ 128 px: the plane itself
    assert s.get_image("multi - cells").shape == (3, 2, 4, 20, 24) and io.is_virtual(s.get_image("multi - cells").data)
    with pytest.raises(ValueError, match="setId"):
        ij.macro("run(\"Bio-Formats Macro Extensions\"); Ext.getSizeX(w);")
    with pytest.raises(FileNotFoundError):
        ij.macro('run("Bio-Formats Macro Extensions"); Ext.setId("/nowhere/none.tif");')
    # the transpiler: output arguments are assigned, bare namespace members are called
    py = transpile('Ext.getSizeX(w);\nExt.getMetadataValue("k", v);\nExt.getZCTCoords(i, z, c, t);\nExt.close;')
    assert "w, = _outs(Ext.getSizeX(), 1)" in py and "v, = _outs(Ext.getMetadataValue('k'), 1)" in py
    assert "z, c, t = _outs(Ext.getZCTCoords(i), 3)" in py and "_call0(Ext.close)" in py


def test_macro_extensions_rgb_plane_and_sub_image(pyramid):
    p, base = pyramid
    s = Session()
    ij = IJ(s)
    ij.macro(f'''
run("Bio-Formats Macro Extensions");
Ext.setId("{_macro_path(p)}");
Ext.getRGBChannelCount(rgbc); Ext.getEffectiveSizeC(effc); Ext.getImageCount(n); Ext.getResolutionCount(levels);
Ext.openImage("rgb", 0);
Ext.openSubImage("sub", 0, 8, 4, 10, 6);
print(rgbc + " " + effc + " " + n + " " + levels);
''')
    assert ij._out.getvalue().strip() == "3 1 1 3"
    rgb = s.get_image("rgb")
    assert rgb.rgb and rgb.shape == (1, 1, 3, 64, 96) and np.array_equal(np.transpose(rgb.data[0, 0], (1, 2, 0)), base)
    sub = s.get_image("sub")
    assert sub.rgb and (sub.height, sub.width) == (6, 10) and np.array_equal(np.transpose(sub.data[0, 0], (1, 2, 0)), base[4:10, 8:18])


def test_nd2_positions_are_series(tmp_path, monkeypatch):
    import sys
    import types

    class FakeND2:
        def __init__(self, path, sizes):
            self.sizes = sizes

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            arr = np.zeros(tuple(self.sizes.values()), np.uint16)
            pi = list(self.sizes).index("P")
            for k in range(self.sizes["P"]):
                np.moveaxis(arr, pi, 0)[k] = k + 1
            return arr

        def voxel_size(self):
            raise RuntimeError("no calibration")

    mod = types.ModuleType("nd2")
    mod.ND2File = lambda path: FakeND2(path, {"P": 3, "Z": 4, "C": 2, "Y": 5, "X": 6})
    monkeypatch.setitem(sys.modules, "nd2", mod)
    (tmp_path / "x.nd2").write_bytes(b"\x00")
    infos = io.list_series(tmp_path / "x.nd2")
    assert len(infos) == 3 and infos[1].name == "x (position 2)" and (infos[1].size_x, infos[1].size_y, infos[1].size_z, infos[1].size_c) == (6, 5, 4, 2)
    assert infos[0].format == "Nikon ND2" and infos[0].dimension_order == "XYCZT"
    img = io.read_image(tmp_path / "x.nd2", series=2)  # an explicit series picks one position
    assert img.shape == (1, 4, 2, 5, 6) and int(img.data.max()) == 3 and img.metadata["series_count"] == 3 and img.metadata["series"] == 2
    assert io.read_image(tmp_path / "x.nd2").shape == (3, 4, 2, 5, 6)  # default: positions as frames, as before
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(tmp_path / "x.nd2"), "series": "all"})
    assert [s.images[i].name for i in out["new_images"]] == [f"x - x (position {k})" for k in (1, 2, 3)]


def test_api_display_fetch_and_state_from_virtual_stack(stack, tmp_path):
    """The server's plane / value endpoints and the session state work on lazy (VirtualStack / memmap) images."""
    from fastapi.testclient import TestClient
    from lumen.server.app import create_app
    p = io.write_image(stack, tmp_path / "c.tif", compress=True)
    s = Session()
    c = TestClient(create_app(s))
    r = c.post("/api/ops/file.import_tiff_virtual/run", json={"params": {"path": p}})
    assert r.status_code == 200, r.text
    iid = r.json()["new_images"][0]
    assert isinstance(s.images[iid].data, io.VirtualStack)
    r = c.get(f"/api/images/{iid}/planes", params={"t": 1, "z": 2})
    assert r.headers["x-dtype"] == "uint16" and r.headers["x-shape"] == "2,17,23"
    assert np.array_equal(np.frombuffer(r.content, np.uint16).reshape(2, 17, 23), stack.data[1, 2])
    r = c.get(f"/api/images/{iid}/plane", params={"t": 0, "z": 1, "c": 1, "stride": 2})
    assert np.array_equal(np.frombuffer(r.content, np.uint16).reshape(9, 12), stack.data[0, 1, 1, ::2, ::2])
    r = c.get(f"/api/images/{iid}/value", params={"x": 3, "y": 4, "t": 1, "z": 0})
    assert r.json()["values"] == [float(v) for v in stack.data[1, 0, :, 4, 3]]
    r = c.get(f"/api/images/{iid}/rgb.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    st = s.state()
    img = next(im for im in st["images"] if im["id"] == iid)
    assert img["shape"] == {"t": 2, "z": 3, "c": 2, "y": 17, "x": 23} and img["nbytes"] < stack.data.nbytes and st["memory_bytes"] < stack.data.nbytes
    assert io.is_virtual(s.images[iid].data)  # nothing above materialised the stack
    r = c.post(f"/api/images/{iid}/undo")
    assert r.status_code == 200 and r.json()["label"] is None  # nothing to undo yet, and the lazy image survives the call
    assert io.is_virtual(s.images[iid].data)


# ---------------------------------------------------------------------------- the Java bridge's fiji-jars mode in the Plugins Configuration table (mocked)
def test_config_table_names_the_fiji_jars_mode(monkeypatch):
    from lumen.core.readers import bioformats_bridge
    monkeypatch.setattr(bioformats_bridge, "_importable", lambda name: False)
    monkeypatch.delenv("LUMEN_BIOFORMATS_MODE", raising=False)
    monkeypatch.setattr(bioformats_bridge, "fiji_status", lambda refresh=False: {
        "available": True, "detail": "Fiji's Bio-Formats jars (formats-gpl-8.1.1.jar in /fake/jars/bio-formats) with openjdk 21 through a groovy runner",
        "java": "/fake/java", "java_version": "openjdk 21", "jar": "/fake/jars/bio-formats/formats-gpl-8.1.1.jar", "runner": "groovy", "jars": "/fake/jars"})
    bioformats_bridge.reset()
    try:
        out = Session().run_op("file.bioformats_config", {})
        rows = {r[0]: r for r in out["table"]["rows"]}
        bridge = next(r for name, r in rows.items() if "Java bridge" in name)
        assert bridge[3].startswith("enabled (Fiji's Bio-Formats jars") and "no Python package needed" in bridge[3]
        assert rows["Olympus VSI (cellSens)"][3] == "enabled (Java bridge)" and "bridge enabled" in out["message"]
        st = bioformats_bridge.status()
        assert st["mode"] == "fiji-jars" and st["available"] and not st["package"]
    finally:
        bioformats_bridge.reset()
