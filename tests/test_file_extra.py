"""Tests for the File menu gaps (lumen/core/ops/file_extra.py) and the runtime LUT helpers (luts.py)."""
import io as _io
import sys
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lumen.core import luts
from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops import file_extra
from lumen.core.session import Session


@pytest.fixture(autouse=True)
def _restore_luts():
    """Runtime LUT registration is global: undo it after every test so other test modules see the built-ins only."""
    saved_tables, saved_names = dict(luts._LUTS), list(luts.LUT_NAMES)
    yield
    luts._LUTS.clear()
    luts._LUTS.update(saved_tables)
    luts.LUT_NAMES[:] = saved_names
    luts._invalidate_caches()


@pytest.fixture(autouse=True)
def _isolated_recent(tmp_path, monkeypatch):
    """Keep the recent-files list out of ~/.lumen while testing."""
    monkeypatch.setattr(Session, "_recent_path", staticmethod(lambda: tmp_path / "recent.json"))


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _write_tif(path: Path, arr: np.ndarray) -> Path:
    tifffile.imwrite(str(path), arr)
    return path


# ---------------------------------------------------------------- Internal clipboard
def test_new_from_clipboard_with_selection(session):
    img = session.active
    roi = {"type": "oval", "x": 10, "y": 20, "w": 40, "h": 30}
    session.run_op("edit.copy", {}, roi=roi)
    out = session.run_op("file.new_from_clipboard", {"title": "Clip"})
    new = session.images[out["new_images"][0]]
    assert new.name == "Clip" and (new.height, new.width) == (30, 40) and new.dtype == img.dtype
    assert np.array_equal(new.data[0, 0, 0], img.data[0, 0, 0, 20:50, 10:50])
    assert out["roi"]["type"] == "mask"  # the oval becomes the selection of the new image
    out2 = session.run_op("file.new_from_clipboard", {"title": "Clip2", "outside": "Background"})
    new2 = session.images[out2["new_images"][0]]
    assert new2.data[0, 0, 0, 0, 0] == 0 and new2.data[0, 0, 0, 15, 20] == img.data[0, 0, 0, 35, 30]


def test_new_from_clipboard_empty():
    s = Session()
    with pytest.raises(ValueError, match="clipboard is empty"):
        s.run_op("file.new_from_clipboard", {})


# ---------------------------------------------------------------- Open Next / Previous / Recent
def _folder_with_images(tmp_path, n=3):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(n):
        _write_tif(d / f"img_{i + 1:02d}.tif", np.full((8, 10), i * 10, np.uint8))
    (d / "notes.txt").write_text("not an image")
    return d


def test_open_next_and_previous(tmp_path):
    d = _folder_with_images(tmp_path)
    s = Session()
    s.open_file(str(d / "img_01.tif"))
    out = s.run_op("file.open_next", {})
    assert s.active.name == "img_02.tif" and len(s.images) == 1 and "2/3" in out["message"]
    s.run_op("file.open_next", {"close_current": False})
    assert s.active.name == "img_03.tif" and len(s.images) == 2
    s.run_op("file.open_next", {})  # wraps around
    assert s.active.name == "img_01.tif"
    s.run_op("file.open_previous", {})
    assert s.active.name == "img_03.tif" and s.active.data.max() == 20
    assert s.recent[0].endswith("img_03.tif")


def test_open_next_needs_a_file_path(session):
    with pytest.raises(ValueError, match="not opened from a file"):
        session.run_op("file.open_next", {})


def test_open_recent(tmp_path):
    d = _folder_with_images(tmp_path, 2)
    s = Session()
    s.open_file(str(d / "img_01.tif"))
    s.open_file(str(d / "img_02.tif"))
    s.run_op("file.close_all", {})
    s.run_op("file.open_recent", {"index": 1})
    assert s.active.name == "img_01.tif"
    assert s.recent[0].endswith("img_01.tif")  # moved to the top
    s.run_op("file.open_recent", {"path": str(d / "img_02.tif")})
    assert s.active.name == "img_02.tif"
    with pytest.raises(ValueError, match="Only 2 recent"):
        s.run_op("file.open_recent", {"index": 7})


def test_open_recent_without_attribute(tmp_path, monkeypatch):
    s = Session()
    monkeypatch.delattr(Session, "add_recent")
    monkeypatch.delattr(s, "recent", raising=False)
    with pytest.raises(ValueError, match="No recent files"):
        s.run_op("file.open_recent", {})


# ---------------------------------------------------------------- ImageJ samples (network mocked)
def _fake_download_factory(calls):
    def fake(url, target, timeout=60.0):
        calls.append(url)
        name = Path(url).name
        if name.endswith(".gif"):
            from PIL import Image
            Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(str(target))
        elif name.endswith(".jpg"):
            from PIL import Image
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(str(target))
        elif name.endswith(".zip"):
            buf = _io.BytesIO()
            tifffile.imwrite(buf, np.arange(2 * 6 * 7, dtype=np.uint16).reshape(2, 6, 7), imagej=True, metadata={"axes": "ZYX"})
            with zipfile.ZipFile(target, "w") as zf:
                zf.writestr("inner/t1-head.tif", buf.getvalue())
                zf.writestr("__MACOSX/._t1-head.tif", b"junk")
        else:
            raise AssertionError(name)
    return fake


def test_open_imagej_sample_download_cache_and_zip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(file_extra, "_samples_dir", lambda: tmp_path / "samples")
    monkeypatch.setattr(file_extra, "_download", _fake_download_factory(calls))
    s = Session()
    out = s.run_op("file.open_imagej_sample", {"name": "Blobs (25K)"})
    img = s.images[out["new_images"][0]]
    assert img.name == "blobs.gif" and img.width == 8 and calls == ["https://imagej.net/images/blobs.gif"]
    assert img.metadata["imagej_sample"] == "Blobs (25K)" and (tmp_path / "samples" / "blobs.gif").is_file()
    # cached: no second download
    s.run_op("file.open_imagej_sample", {"name": "Blobs (25K)"})
    assert len(calls) == 1
    # zip sample: extracted, first image inside is opened
    out = s.run_op("file.open_imagej_sample", {"name": "T1 Head"})
    t1 = s.images[out["new_images"][0]]
    assert t1.name == "t1-head.tif" and t1.nz == 2 and t1.dtype == np.uint16
    assert (tmp_path / "samples" / "t1-head" / "t1-head.tif").is_file()
    assert not (tmp_path / "samples" / "t1-head" / "._t1-head.tif").exists()
    assert all(n in file_extra.IJ_SAMPLES for n in ("Mitosis", "Clown", "Tree Rings", "Organ of Corti", "HeLa Cells"))
    assert len(file_extra.IJ_SAMPLES) == 29


def test_open_imagej_sample_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(file_extra, "_samples_dir", lambda: tmp_path / "samples")

    def offline(url, target, timeout=60.0):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(file_extra, "_download", offline)
    s = Session()
    with pytest.raises(RuntimeError, match="offline"):
        s.run_op("file.open_imagej_sample", {"name": "Boats"})
    assert not list((tmp_path / "samples").glob("*.part"))
    with pytest.raises(ValueError):
        s.run_op("file.open_imagej_sample", {"name": "Nope"})


# ---------------------------------------------------------------- LUT files
def test_lut_parsers_binary_icol_text():
    fire = luts.get_lut("Fire")
    raw = luts.lut_bytes(fire)
    assert len(raw) == 768 and np.array_equal(luts.parse_lut_bytes(raw), fire)
    # NIH Image 'ICOL' header: 32 bytes then the planar table
    header = b"ICOL" + (1).to_bytes(2, "big") + (256).to_bytes(2, "big") + bytes(24)
    assert np.array_equal(luts.parse_lut_bytes(header + raw), fire)
    # fewer colours are interpolated to 256
    header16 = b"ICOL" + (1).to_bytes(2, "big") + (16).to_bytes(2, "big") + bytes(24)
    small = np.linspace(0, 255, 16).round().astype(np.uint8)
    tab = luts.parse_lut_bytes(header16 + bytes(small) + bytes(small) + bytes(small))
    assert tab.shape == (256, 3) and tab[0, 0] == 0 and tab[255, 0] == 255 and 120 < tab[128, 1] < 136
    # text LUTs with 3 and 4 columns, header line allowed
    txt3 = "\n".join(f"{i} {255 - i} 0" for i in range(256))
    txt4 = "Index Red Green Blue\n" + "\n".join(f"{i}\t{i}\t{i}\t{255 - i}" for i in range(256))
    t3, t4 = luts.parse_lut_text(txt3), luts.parse_lut_text(txt4)
    assert t3[10].tolist() == [10, 245, 0] and t4[10].tolist() == [10, 10, 245]
    with pytest.raises(ValueError):
        luts.parse_lut_text("hello\nworld")


def test_register_lut_keeps_names_in_sync():
    n0 = len(luts.LUT_NAMES)
    name = luts.register_lut("UnitTest LUT", luts.gradient_lut("#000000", "#ff0000"))
    assert name == "UnitTest LUT" and name in luts.LUT_NAMES and name in luts.all_luts()
    assert luts.register_lut("UnitTest LUT", luts.gradient_lut("#000000", "#ff0000")) == name  # same table → same name
    other = luts.register_lut("UnitTest LUT", luts.gradient_lut("#000000", "#00ff00"))
    assert other == "UnitTest LUT 2"
    assert luts.register_lut("Fire", np.zeros((256, 3), np.uint8)) == "Fire 2"  # built-ins are protected
    assert np.array_equal(luts.get_lut("Fire"), luts._LUTS["Fire"]) and luts.get_lut("Fire").max() == 255
    assert luts.unregister_lut(other) and other not in luts.LUT_NAMES
    assert not luts.unregister_lut("Fire")
    assert len(luts.LUT_NAMES) == n0 + 2


def test_import_export_lut_ops(tmp_path, session):
    p = tmp_path / "myfire.lut"
    p.write_bytes(luts.lut_bytes(luts.get_lut("Fire")[::-1]))
    out = session.run_op("file.import_lut", {"path": str(p)})
    assert "myfire" in out["message"] and "myfire" in luts.LUT_NAMES
    assert session.active.display.channels[0].lut == "myfire"
    assert np.array_equal(luts.get_lut("myfire"), luts.get_lut("Fire")[::-1])
    # export the current channel's LUT and re-read it
    out = session.run_op("file.export_lut", {"path": str(tmp_path / "out")})
    exported = tmp_path / "out.lut"
    assert exported.is_file() and exported.stat().st_size == 768
    assert np.array_equal(luts.read_lut_file(exported), luts.get_lut("myfire"))
    # inverted channels export the reversed table
    session.run_op("color.lut", {"lut": "Grays"})
    session.run_op("color.invert_lut", {})
    session.run_op("file.export_lut", {"path": str(tmp_path / "inv.lut")})
    assert luts.read_lut_file(tmp_path / "inv.lut")[0].tolist() == [255, 255, 255]
    # text LUT by name, no image needed
    (tmp_path / "ramp.lut").write_text("\n".join(f"{i},{i},{i}" for i in range(256)))
    s2 = Session()
    s2.run_op("file.import_lut", {"path": str(tmp_path / "ramp.lut"), "name": "Ramp Test"})
    assert luts.get_lut("Ramp Test")[200].tolist() == [200, 200, 200]


def test_import_luts_folder(tmp_path):
    d = tmp_path / "luts"
    d.mkdir()
    (d / "a_red.lut").write_bytes(luts.lut_bytes(luts.gradient_lut("black", "red")))
    (d / "b_text.lut").write_text("\n".join(f"{i} 0 {i}" for i in range(256)))
    (d / "broken.lut").write_bytes(b"nonsense")
    s = Session()
    out = s.run_op("file.import_luts_folder", {"folder": str(d)})
    rows = {r[0]: r for r in out["table"]["rows"]}
    assert rows["a_red.lut"][2] == "ok" and rows["b_text.lut"][2] == "ok" and rows["broken.lut"][2].startswith("error")
    assert "Loaded 2 of 3" in out["message"] and "a_red" in luts.LUT_NAMES


def test_export_lut_into_folder_sanitises_name(tmp_path, session):
    """A LUT name with '/' (built-in 'Red/Green') or Windows-illegal characters must not become a sub-directory."""
    folder = tmp_path / "luts"
    folder.mkdir()
    session.run_op("color.lut", {"lut": "Red/Green"})
    out = session.run_op("file.export_lut", {"path": str(folder) + "/"})
    exported = folder / "Red_Green.lut"
    assert exported.is_file() and str(exported) in out["message"]
    assert not (folder / "Red").exists()
    assert sorted(q.name for q in folder.iterdir()) == ["Red_Green.lut"]
    assert np.array_equal(luts.read_lut_file(exported), luts.get_lut("Red/Green"))
    # runtime-registered names come from user input: * ? " < > | : are invalid on Windows
    luts.register_lut('My:Odd*Name?', luts.gradient_lut("black", "blue"))
    session.run_op("file.export_lut", {"path": str(folder), "lut": "My:Odd*Name?"})
    assert (folder / "My_Odd_Name_.lut").is_file() and len(list(folder.rglob("*"))) == 2
    # an explicit file path is still honoured as given
    session.run_op("file.export_lut", {"path": str(tmp_path / "sub" / "rg.lut"), "lut": "Red/Green"})
    assert (tmp_path / "sub" / "rg.lut").is_file()


# ---------------------------------------------------------------- tables, text, coordinates, lists
def test_import_results_csv_with_imagej_row_numbers(tmp_path):
    p = tmp_path / "Results.csv"
    p.write_text(" ,Area,Mean,Label\n1,10,2.5,a\n2,20,3.5,b\n")
    s = Session()
    s.results.append([{"old": 1}])
    out = s.run_op("file.import_results", {"path": str(p)})
    assert out["results_added"] == 2 and s.results.columns == ["Area", "Mean", "Label"]
    assert s.results.rows[1] == {"Area": 20, "Mean": 3.5, "Label": "b"}
    s.run_op("file.import_results", {"path": str(p), "replace": False})
    assert len(s.results.rows) == 4


def test_import_table_tsv_and_xlsx(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("X\tY\n1\t2\n3\t4.5\n")
    s = Session()
    out = s.run_op("file.import_table", {"path": str(p)})
    assert out["table"]["name"] == "data" and out["table"]["columns"] == ["X", "Y"] and out["table"]["rows"] == [[1, 2], [3, 4.5]]
    assert "data" in s.tables
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Value"])
    ws.append(["a", 1.5])
    ws.append(["b", 2])
    xp = tmp_path / "book.xlsx"
    wb.save(str(xp))
    out = s.run_op("file.import_table", {"path": str(xp), "name": "Book"})
    assert out["table"]["columns"] == ["Name", "Value"] and out["table"]["rows"] == [["a", 1.5], ["b", 2]]
    # headerless numeric matrix gets generated column names
    (tmp_path / "m.csv").write_text("1,2,3\n4,5,6\n")
    out = s.run_op("file.import_table", {"path": str(tmp_path / "m.csv")})
    assert out["table"]["columns"] == ["C1", "C2", "C3"] and out["table"]["rows"][1] == [4, 5, 6]


def test_import_imagej_results_xls_text_and_roundtrip(tmp_path, monkeypatch):
    """ImageJ's saveAs("Results", "Results.xls") writes tab-separated TEXT: it must not be routed to openpyxl."""
    p = tmp_path / "Results.xls"
    p.write_text(" \tArea\tMean\n1\t10\t2.5\n")
    s = Session()
    out = s.run_op("file.import_results", {"path": str(p)})
    assert out["results_added"] == 1 and s.results.columns == ["Area", "Mean"] and s.results.rows == [{"Area": 10, "Mean": 2.5}]
    out = s.run_op("file.import_table", {"path": str(p)})
    assert out["table"]["columns"] == ["Area", "Mean"] and out["table"]["rows"] == [[10, 2.5]]
    # round-trip of the real ImageJ Results layout (leading ' ' row-number column) through our own exporters
    src = Session()
    src.results.append([{"Area": 10, "Mean": 2.5, "Label": "cell 1"}, {"Area": 20, "Mean": 3, "Label": "cell 2"}])
    src.run_op("file.export_results", {"path": str(tmp_path / "rt.tsv"), "format": "tsv", "row_numbers": True})
    assert (tmp_path / "rt.tsv").read_text().splitlines()[0] == " \tArea\tMean\tLabel"
    (tmp_path / "rt_Results.xls").write_bytes((tmp_path / "rt.tsv").read_bytes())  # same bytes under ImageJ's historical name
    src.run_op("file.export_text", {"path": str(tmp_path / "rt.txt"), "source": "Results", "row_numbers": True})
    for name in ("rt.tsv", "rt_Results.xls", "rt.txt"):
        back = Session()
        back.run_op("file.import_results", {"path": str(tmp_path / name)})
        assert back.results.columns == ["Area", "Mean", "Label"] and back.results.rows == src.results.rows, name
    # an OOXML workbook saved under a .xls name is read by content (a file object bypasses openpyxl's extension check)
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append(["Name", "Value"])
    wb.active.append(["a", 1.5])
    wb.save(str(tmp_path / "book.xlsx"))
    (tmp_path / "book.xls").write_bytes((tmp_path / "book.xlsx").read_bytes())
    assert file_extra.read_table_file(tmp_path / "book.xls") == (["Name", "Value"], [["a", 1.5]])
    # a genuine BIFF workbook is neither parsed as text garbage nor handed to openpyxl: clear error naming xlrd
    (tmp_path / "old.xls").write_bytes(file_extra._OLE_MAGIC + b"\0" * 504)
    monkeypatch.setitem(sys.modules, "xlrd", None)  # 'import xlrd' then raises ImportError even if it is installed
    with pytest.raises(RuntimeError, match="xlrd"):
        file_extra.read_table_file(tmp_path / "old.xls")


def test_import_table_space_aligned_txt(tmp_path):
    """Runs of spaces are one separator (ImageJ TextReader); tab-separated empty cells are still kept."""
    read = file_extra.read_table_file
    (tmp_path / "eq.txt").write_text("X   Y\n1   2\n3   4\n")
    assert read(tmp_path / "eq.txt") == (["X", "Y"], [[1, 2], [3, 4]])
    (tmp_path / "aligned.txt").write_text("Area  Mean\n  12   3.5\n 100  44.2\n")  # right-aligned, unequal runs
    assert read(tmp_path / "aligned.txt") == (["Area", "Mean"], [[12, 3.5], [100, 44.2]])
    (tmp_path / "q.txt").write_text('Label   Mean\n"cell 1"   2.5\n')  # quoted label with a space survives
    assert read(tmp_path / "q.txt") == (["Label", "Mean"], [["cell 1", 2.5]])
    (tmp_path / "single.txt").write_text("Value\n1\n2\n")
    assert read(tmp_path / "single.txt") == (["Value"], [[1], [2]])
    (tmp_path / "gap.txt").write_text("a\tb\tc\n1\t\t3\n")  # genuine empty cell in a tab-separated file
    assert read(tmp_path / "gap.txt") == (["a", "b", "c"], [[1, "", 3]])
    (tmp_path / "labels.txt").write_text("Label\tMean\ncell 1\t2.5\ncell 2\t3.5\n")  # tabs win over spaces inside labels
    assert read(tmp_path / "labels.txt") == (["Label", "Mean"], [["cell 1", 2.5], ["cell 2", 3.5]])
    s = Session()
    out = s.run_op("file.import_table", {"path": str(tmp_path / "aligned.txt")})
    assert out["table"]["columns"] == ["Area", "Mean"] and out["table"]["rows"] == [[12, 3.5], [100, 44.2]]


def test_import_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("first line\nsecond line\n")
    s = Session()
    out = s.run_op("file.import_text", {"path": str(p)})
    assert out["table"]["name"] == "notes.txt" and out["table"]["rows"] == [[1, "first line"], [2, "second line"]]
    assert any("second line" in e["msg"] for e in s.log)


def test_import_xy_coordinates(tmp_path, session):
    p = tmp_path / "pts.txt"
    p.write_text("X Y\n10 10\n50, 10\n50\t40\n10 40\n")
    out = session.run_op("file.import_xy_coordinates", {"path": str(p)})
    assert out["roi"]["type"] == "polygon" and out["roi"]["points"] == [[10, 10], [50, 10], [50, 40], [10, 40]]
    out = session.run_op("file.import_xy_coordinates", {"path": str(p), "type": "polyline"})
    assert out["roi"]["type"] == "polyline" and out["roi"]["width"] == 1
    out = session.run_op("file.import_xy_coordinates", {"path": str(p), "type": "point", "invert_y": True})
    assert out["roi"]["type"] == "point" and out["roi"]["points"][0][1] == session.active.height - 10
    (tmp_path / "two.txt").write_text("1 2\n3 4\n")
    with pytest.raises(ValueError, match="at least 3"):
        session.run_op("file.import_xy_coordinates", {"path": str(tmp_path / "two.txt")})


def test_import_stack_from_list(tmp_path):
    d = _folder_with_images(tmp_path, 3)
    lst = tmp_path / "list.txt"
    lst.write_text(f"imgs/img_01.tif\n{d / 'img_02.tif'}\n# comment\nimgs/img_03.tif\nimgs/missing.tif\n")
    s = Session()
    out = s.run_op("file.import_stack_from_list", {"path": str(lst)})
    img = s.images[out["new_images"][0]]
    assert img.name == "list.tif" and img.nz == 3 and img.nt == 1 and img.data[0, 2, 0].max() == 20
    assert "1 missing" in out["message"]
    out = s.run_op("file.import_stack_from_list", {"path": str(lst), "stack_as": "T series", "title": "frames"})
    assert s.images[out["new_images"][0]].nt == 3
    out = s.run_op("file.import_stack_from_list", {"path": str(lst), "stack_as": "Channels"})
    assert s.images[out["new_images"][0]].nc == 3


# ---------------------------------------------------------------- exports
def test_export_zip_roundtrip(tmp_path, session):
    from lumen.core.io import read_image
    out = session.run_op("file.export_zip", {"path": str(tmp_path / "blobs")})
    zp = tmp_path / "blobs.zip"
    assert zp.is_file() and "blobs.tif" in out["message"]
    with zipfile.ZipFile(zp) as zf:
        assert zf.namelist() == ["blobs.tif"]
        zf.extractall(tmp_path / "unz")
    back = read_image(tmp_path / "unz" / "blobs.tif")
    assert np.array_equal(back.data, session.active.data)


def test_export_text_sources(tmp_path):
    s = Session()
    s.results.append([{"Area": 10, "Mean": 2.5}, {"Area": 20, "Mean": 3.0}])
    out = s.run_op("file.export_text", {"path": str(tmp_path / "res"), "source": "Results"})
    text = (tmp_path / "res.csv").read_text()
    assert text.splitlines()[0] == " ,Area,Mean" and text.splitlines()[2] == "2,20,3" and "2 line(s)" in out["message"]
    s.run_op("file.export_text", {"path": str(tmp_path / "res.txt"), "source": "Results", "row_numbers": False})
    assert (tmp_path / "res.txt").read_text().splitlines()[1] == "10\t2.5"
    s.tables["T"] = {"name": "T", "columns": ["a", "b"], "rows": [[1, "x"]]}
    s.run_op("file.export_text", {"path": str(tmp_path / "t.tsv"), "source": "Table", "table": "T"})
    assert (tmp_path / "t.tsv").read_text() == " \ta\tb\n1\t1\tx\n"
    s.add_log("hello log")
    s.run_op("file.export_text", {"path": str(tmp_path / "log"), "source": "Log"})
    assert "hello log" in (tmp_path / "log.txt").read_text()
    s.run_op("file.export_text", {"path": str(tmp_path / "macro.py"), "source": "Macro"})
    assert 'lumen.run("file.export_text"' in (tmp_path / "macro.py").read_text()
    with pytest.raises(ValueError, match="No table named"):
        s.run_op("file.export_text", {"path": str(tmp_path / "x"), "source": "Table", "table": "nope"})
