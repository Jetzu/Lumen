"""Data larger than RAM (docs/USABILITY.md M10): lazy 5-D arrays over HDF5 / Imaris / Zarr / NPY / TIFF
(lumen.core.lazy), automatic virtual opening, streamed stack commands and exports, memory guardrails.

Laziness is proven by instrumentation (planes read, resident bytes against the cache budget) on small stacks –
no large files are written in CI.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import io, lazy, ops  # noqa: F401  (registers the commands)
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops.stacks import PROJ
from lumen.core.registry import OperationCancelled
from lumen.core.session import Session
from lumen.server.app import create_app

h5py = pytest.importorskip("h5py")
zarr = pytest.importorskip("zarr")

FORMATS = ["h5", "zarr", "npy", "tif"]


@pytest.fixture(autouse=True)
def _restore_cache():
    """Tests shrink the shared plane cache: give every test the default budget back."""
    budget = lazy.PLANE_CACHE.budget
    yield
    lazy.PLANE_CACHE.set_budget(budget)
    lazy.PLANE_CACHE.clear()


@pytest.fixture
def stack():
    rng = np.random.default_rng(7)
    data = rng.integers(0, 4000, (2, 3, 2, 17, 23)).astype(np.uint16)  # T Z C Y X: 12 planes of 782 bytes
    return LumenImage(data, name="stack", axes="TZCYX", calibration=Calibration(0.5, 0.5, 2.0, "µm", frame_interval=1.5),
                      channel_names=["DAPI", "GFP"])


def _write(stack, tmp_path, fmt):
    if fmt == "tif":
        return io.write_image(stack, tmp_path / "s.tif", compress=True)  # zlib pages → VirtualStack (not a memmap)
    return io.write_image(stack, tmp_path / f"s.{fmt}")


def _client():
    s = Session()
    return TestClient(create_app(s)), s


# ----------------------------------------------------------------------------- the lazy array
@pytest.mark.parametrize("fmt", FORMATS)
def test_lazy_open_reads_planes_on_demand(stack, tmp_path, fmt):
    p = _write(stack, tmp_path, fmt)
    img = io.read_image(p, virtual=True)
    data = img.data
    assert io.is_virtual(data) and img.is_virtual and img.shape == stack.shape and img.dtype == np.uint16 and img.bit_depth == "16-bit"
    assert img.metadata["source"]["virtual"] is True and img.path == str(p)
    assert [c.name for c in img.display.channels] == (["DAPI", "GFP"] if fmt != "npy" else ["C1", "C2"])
    if fmt == "npy":
        assert isinstance(data, np.memmap) and img.nbytes == 0  # file-backed pages are the OS's, not the process's
    else:
        assert isinstance(data, lazy.LazyArray) and not data.materialized
        assert data.planes_read == 2  # the display range of each channel came from the current plane only
        assert 0 < img.nbytes <= 2 * 782 and img.nbytes <= lazy.PLANE_CACHE.budget
    # the viewer's fetch of one (C, Y, X) slab, region reads and strided slices read only what they cover
    assert np.array_equal(data[1, 2], stack.data[1, 2])
    assert np.array_equal(data[:, 1, :, ::4, 3:9], stack.data[:, 1, :, ::4, 3:9]) and data[1, -1, 0, 3, 4] == stack.data[1, 2, 0, 3, 4]
    assert np.array_equal(data[0, 1, :, 5, 6], stack.data[0, 1, :, 5, 6])  # the /value endpoint's pixel probe
    if fmt != "npy":
        assert not data.materialized and img.nbytes < stack.data.nbytes
        assert np.array_equal(data.copy(), stack.data) and not data.materialized  # copy() reads once, stays lazy
        assert data.full_bytes == stack.data.nbytes
    s = img.summary()
    assert s["virtual"] is True and s["resident_bytes"] == img.nbytes and s["full_bytes"] == stack.data.nbytes and s["virtual_source"]
    assert np.array_equal(np.asarray(data), stack.data)  # the whole array: materialised once (copy-on-write)
    if fmt != "npy":
        assert data.materialized and not io.is_virtual(data) and img.nbytes == stack.data.nbytes and img.summary()["virtual"] is False


def test_plane_cache_budget_bounds_resident_bytes(stack, tmp_path):
    p = _write(stack, tmp_path, "h5")
    lazy.PLANE_CACHE.set_budget(2 * 782)  # room for two planes
    img = io.read_image(p, virtual=True)
    data = img.data
    for t in range(2):
        for z in range(3):
            for c in range(2):
                assert np.array_equal(data[t, z, c], stack.data[t, z, c])
                assert data.nbytes <= 2 * 782 and lazy.PLANE_CACHE.total <= 2 * 782
    assert data.planes_read == 12  # the two planes cached at open were hits; every other plane was read exactly once
    assert np.array_equal(data[0, 0, 0], stack.data[0, 0, 0]) and data.planes_read == 13  # evicted since → read again
    assert lazy.PLANE_CACHE.set_budget(1) is None and data.nbytes == 0  # shrinking the budget evicts
    lazy.set_cache_budget(64)
    assert lazy.PLANE_CACHE.budget == 64 * 1024 ** 2 and lazy.cache_budget_bytes() == 64 * 1024 ** 2
    assert data.fresh().planes_read == 0 and np.array_equal(data.fresh()[1, 1, 1], stack.data[1, 1, 1])


def test_lazy_sources_cover_imaris_levels_float_and_rgb(tmp_path):
    rng = np.random.default_rng(5)
    full = rng.integers(0, 4000, (4, 8, 8)).astype(np.uint16)

    def sattr(g, k, v):
        g.attrs[k] = np.array([c.encode() for c in str(v)], dtype="S1")
    with h5py.File(tmp_path / "v.ims", "w") as hf:
        info = hf.create_group("DataSetInfo/Image")
        for k, v in (("X", 8), ("Y", 8), ("Z", 4), ("ExtMin0", 0), ("ExtMin1", 0), ("ExtMin2", 0), ("ExtMax0", 4.0), ("ExtMax1", 4.0), ("ExtMax2", 8.0), ("Unit", "um")):
            sattr(info, k, v)
        sattr(hf.create_group("DataSetInfo/Channel 0"), "Name", "Nuclei")
        for lvl, arr in ((0, full), (1, full[::2, ::2, ::2])):
            g = hf.create_group(f"DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0")
            padded = np.zeros((arr.shape[0], arr.shape[1] + 2, arr.shape[2] + 3), arr.dtype)  # Imaris pads Data to chunk multiples
            padded[:, :arr.shape[1], :arr.shape[2]] = arr
            g.create_dataset("Data", data=padded)
            for k, v in (("ImageSizeX", arr.shape[2]), ("ImageSizeY", arr.shape[1]), ("ImageSizeZ", arr.shape[0])):
                sattr(g, k, v)
    l0 = io.read_image(tmp_path / "v.ims", virtual=True)
    assert isinstance(l0.data, lazy.LazyArray) and l0.data.description == "Imaris ResolutionLevel 0" and l0.shape == (1, 4, 1, 8, 8)
    assert np.array_equal(l0.data[0, :, 0], full) and [c.name for c in l0.display.channels] == ["Nuclei"] and l0.calibration.pixel_width == pytest.approx(0.5)
    assert np.array_equal(l0.data[0, 2, 0, 1:5, 2:7], full[2, 1:5, 2:7])  # region read on the padded dataset
    l1 = io.read_image(tmp_path / "v.ims", level=1, virtual=True)
    assert l1.shape == (1, 2, 1, 4, 4) and np.array_equal(np.asarray(l1.data), full[::2, ::2, ::2][None, :, None]) and l1.calibration.pixel_depth == pytest.approx(4.0)
    assert l1.metadata["source"] == {"kind": "file", "path": str(tmp_path / "v.ims"), "level": 1, "virtual": True}
    # float32 data: display range, summary and is_binary never scan the whole file
    f = LumenImage(rng.random((2, 3, 1, 9, 11)).astype(np.float32), name="f", axes="TZCYX")
    fv = io.read_image(io.write_image(f, tmp_path / "f.zarr"), virtual=True)
    assert fv.dtype == np.float32 and fv.is_binary is False and fv.summary()["virtual"] is True and not fv.data.materialized
    lo, hi = fv.value_range()
    assert lo == pytest.approx(float(f.data[0, 0, 0].min())) and hi == pytest.approx(float(f.data[0, 0, 0].max()))
    # an OME-Zarr written by Lumen stores T C Z Y X: the lazy mapping transposes the axes without copying
    assert np.array_equal(fv.data[1, 2, 0], f.data[1, 2, 0]) and fv.data.description == "Zarr array 0"
    # float64 / int16 files are coerced per plane (the memmap cannot be handed out as is)
    np.save(tmp_path / "f64.npy", f.data.astype(np.float64))
    v64 = io.read_image(tmp_path / "f64.npy", virtual=True)
    assert isinstance(v64.data, lazy.LazyArray) and v64.dtype == np.float32 and np.array_equal(v64.data[1, 1, 0], f.data[1, 1, 0])
    with h5py.File(tmp_path / "i16.h5", "w") as hf:
        hf.create_dataset("vol", data=(rng.integers(-300, 300, (3, 6, 7))).astype(np.int16))
    v16 = io.read_image(tmp_path / "i16.h5", virtual=True)
    assert v16.dtype == np.float32 and v16.shape == (1, 3, 1, 6, 7) and v16.metadata["source"]["series"] == 0
    # an RGB sample axis (Y, X, S) maps onto C = 3 with rgb=True
    rgb = rng.integers(0, 255, (5, 6, 3)).astype(np.uint8)
    with h5py.File(tmp_path / "rgb.h5", "w") as hf:
        ds = hf.create_dataset("rgb", data=rgb)
        ds.attrs["axes"], ds.attrs["rgb"] = "YXS", True
    vr = io.read_image(tmp_path / "rgb.h5", virtual=True)
    assert vr.rgb and vr.shape == (1, 1, 3, 5, 6) and np.array_equal(vr.data[0, 0, 2], rgb[..., 2]) and np.array_equal(np.asarray(vr.data)[0, 0], np.moveaxis(rgb, -1, 0))


# ----------------------------------------------------------------------------- automatic virtual opening
def test_auto_virtual_from_physical_ram(stack, tmp_path, monkeypatch):
    p = _write(stack, tmp_path, "h5")
    nbytes = stack.data.nbytes
    monkeypatch.setattr(lazy, "_setting", lambda key: None)  # no auto_virtual_mb preference: half of the RAM
    monkeypatch.setattr(lazy, "physical_memory_bytes", lambda: 2 * (nbytes - 1000))
    assert lazy.auto_virtual_bytes() == nbytes - 1000 and lazy.default_auto_virtual_mb() == 1
    assert io.estimate_decoded_bytes(p) == nbytes and io.should_open_virtual(p)
    assert io.read_image(p, virtual="auto").is_virtual and not io.read_image(p, virtual=False).is_virtual
    assert io.read_image(p).is_virtual is False  # the default stays in memory
    monkeypatch.setattr(lazy, "physical_memory_bytes", lambda: 4 * nbytes)
    assert not io.should_open_virtual(p) and not io.read_image(p, virtual="auto").is_virtual
    # the preference: an explicit limit, 0 = never
    monkeypatch.setattr(lazy, "_setting", lambda key: {"auto_virtual_mb": nbytes / 2 / 1024 ** 2}[key])
    assert io.should_open_virtual(p) and io.read_image(p, virtual="auto").is_virtual
    monkeypatch.setattr(lazy, "_setting", lambda key: 0)
    assert not io.should_open_virtual(p) and lazy.auto_virtual_bytes() > 2 ** 62
    # the estimate works from the header for every lazy-capable format, and is None for the others
    for fmt in FORMATS:
        assert io.estimate_decoded_bytes(_write(stack, tmp_path, fmt)) == nbytes
    assert io.estimate_decoded_bytes(io.write_image(stack, tmp_path / "s.ome.tif")) == nbytes
    import imageio.v3 as iio
    iio.imwrite(str(tmp_path / "a.png"), np.zeros((4, 4), np.uint8))
    assert io.estimate_decoded_bytes(tmp_path / "a.png") is None and io.estimate_decoded_bytes(tmp_path / "missing.h5") is None


def test_open_endpoint_auto_virtual_and_state_fields(stack, tmp_path, monkeypatch):
    p = _write(stack, tmp_path, "zarr")
    nbytes = stack.data.nbytes
    monkeypatch.setattr(lazy, "_setting", lambda key: None)
    monkeypatch.setattr(lazy, "physical_memory_bytes", lambda: 2 * (nbytes - 1000))
    c, s = _client()
    r = c.post("/api/images/open", json={"path": str(p)})  # virtual unset = automatic
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"]["virtual"] is True and body["image"]["full_bytes"] == nbytes and body["image"]["resident_bytes"] < nbytes
    img = s.images[body["id"]]
    assert img.is_virtual and any("auto-virtual" in e["msg"] and e["level"] == "warning" for e in s.log)
    st = c.get("/api/state").json()
    assert st["memory_limit_bytes"] == 2 * (nbytes - 1000) and st["auto_virtual_bytes"] == nbytes - 1000
    assert st["virtual_cache_bytes"] == lazy.PLANE_CACHE.budget and 0 <= st["virtual_resident_bytes"] <= st["virtual_cache_bytes"]
    assert st["memory_bytes"] < nbytes and next(im for im in st["images"] if im["id"] == img.id)["virtual"] is True
    r = c.get(f"/api/images/{img.id}/planes", params={"t": 1, "z": 2})
    assert np.array_equal(np.frombuffer(r.content, np.uint16).reshape(2, 17, 23), stack.data[1, 2]) and img.is_virtual
    # explicit choices win: virtual=false reads into memory, virtual=true forces a virtual stack of a small file
    r = c.post("/api/images/open", json={"path": str(p), "virtual": False})
    assert r.json()["image"]["virtual"] is False and not s.images[r.json()["id"]].is_virtual
    monkeypatch.setattr(lazy, "physical_memory_bytes", lambda: 100 * nbytes)
    assert c.post("/api/images/open", json={"path": str(p)}).json()["image"]["virtual"] is False
    assert c.post("/api/images/open", json={"path": str(p), "virtual": True}).json()["image"]["virtual"] is True
    monkeypatch.setattr(lazy, "_setting", lambda key: 0)
    assert c.get("/api/state").json()["auto_virtual_bytes"] is None
    # the preferences: the cache budget is applied, the auto limit validated
    c2, _s2 = _client()
    assert c2.get("/api/settings").json()["virtual_cache_mb"] == lazy.DEFAULT_CACHE_MB
    assert c2.put("/api/settings", json={"virtual_cache_mb": 64}).status_code == 200 and lazy.PLANE_CACHE.budget == 64 * 1024 ** 2
    assert c2.put("/api/settings", json={"virtual_cache_mb": 1}).status_code == 400
    assert c2.put("/api/settings", json={"auto_virtual_mb": 0}).status_code == 200 and c2.put("/api/settings", json={"auto_virtual_mb": "big"}).status_code == 400


# ----------------------------------------------------------------------------- streamed stack commands
@pytest.mark.parametrize("fmt", ["h5", "zarr", "tif"])
def test_streamed_projections_equal_in_memory(stack, tmp_path, fmt):
    p = _write(stack, tmp_path, fmt)
    s = Session()
    lz = s.add_image(io.read_image(p, virtual=True))
    ref = s.add_image(LumenImage(stack.data.copy(), name="ref", axes="TZCYX"))
    for method in PROJ:
        for axis in ("Z", "T"):
            for rng_ in ({}, {"start": 2, "stop": 3}):
                a = s.images[s.run_op("stacks.z_project", {"method": method, "axis": axis, **rng_}, image_id=lz.id)["new_images"][0]]
                b = s.images[s.run_op("stacks.z_project", {"method": method, "axis": axis, **rng_}, image_id=ref.id)["new_images"][0]]
                assert a.shape == b.shape and a.dtype == b.dtype and np.allclose(a.data, b.data, rtol=1e-5, atol=1e-3), (method, axis, rng_)
                assert a.name.split("_")[0] == b.name.split("_")[0]  # ImageJ ZProjector prefixes (AVG_, MED_, STD_ …) on both paths
    assert lz.is_virtual and not lz.data.materialized  # 24 projections later the stack still lives in the file
    assert {k: s.histories[lz.id].labels()[k] for k in ("undo", "redo")} == {"undo": [], "redo": []}
    # the grouped projection and a plane-op scope read only the planes they touch
    g = s.images[s.run_op("stacks.grouped_z_project", {"group_size": 3, "method": "Max Intensity"}, image_id=lz.id)["new_images"][0]]
    assert np.array_equal(g.data[:, 0], stack.data.max(axis=1)) and lz.is_virtual


def test_streamed_statistics_profile_substack_duplicate(stack, tmp_path):
    p = _write(stack, tmp_path, "h5")
    s = Session()
    lz = s.add_image(io.read_image(p, virtual=True))
    ref = s.add_image(LumenImage(stack.data.copy(), name="stack", axes="TZCYX", calibration=Calibration(0.5, 0.5, 2.0, "µm", frame_interval=1.5), channel_names=["DAPI", "GFP"]))
    roi = {"type": "rect", "x": 3, "y": 2, "w": 10, "h": 8}
    for r in (None, roi):
        a = s.run_op("stacks.statistics", {}, image_id=lz.id, roi=r)["table"]
        b = s.run_op("stacks.statistics", {}, image_id=ref.id, roi=r)["table"]
        assert a["columns"] == b["columns"] and len(a["rows"]) == 2
        for ra, rb in zip(a["rows"], b["rows"]):
            assert ra[0].split(":")[1] == rb[0].split(":")[1] and ra[1] == rb[1] and np.allclose(ra[2:], rb[2:], rtol=1e-9, atol=1e-9), (ra, rb)  # exact for 16-bit data
    for axis in ("Z", "T"):
        a = s.run_op("stacks.z_profile", {"axis": axis}, image_id=lz.id, roi=roi)["plot"]
        b = s.run_op("stacks.z_profile", {"axis": axis}, image_id=ref.id, roi=roi)["plot"]
        assert a["x"] == b["x"] and [sr["values"] for sr in a["series"]] == [sr["values"] for sr in b["series"]]
    assert lz.is_virtual and lz.data.nbytes < stack.data.nbytes
    # a substack reads only the planes of the range into a new in-memory image; the source stays lazy
    for axis, spec, idx in (("Z", "2-3", [1, 2]), ("T", "2", [1]), ("C", "2", [1])):
        out = s.run_op("stacks.substack", {"slices": spec, "axis": axis}, image_id=lz.id)
        sub = s.images[out["new_images"][0]]
        assert np.array_equal(sub.data, np.take(stack.data, idx, axis={"Z": 1, "T": 0, "C": 2}[axis])) and not sub.is_virtual and lz.is_virtual
    before = lz.data.planes_read
    out = s.run_op("stacks.substack", {"slices": "1", "axis": "Z"}, image_id=lz.id)
    assert lz.data.planes_read - before <= 4  # 2 × 2 planes of slice 1 (the cache may still hold the display planes)
    # Duplicate of the current slice slices lazily too; the whole-stack duplicate is a deliberate full copy
    lz.display.t, lz.display.z = 1, 2
    d = s.images[s.run_op("image.duplicate", {"what": "Current slice (all channels)"}, image_id=lz.id)["new_images"][0]]
    assert np.array_equal(d.data[0, 0], stack.data[1, 2]) and lz.is_virtual
    # delete_from_original turns the image into an in-memory stack of the kept planes (it was modified)
    out = s.run_op("stacks.substack", {"slices": "1", "axis": "Z", "delete_from_original": True}, image_id=lz.id)
    assert np.array_equal(lz.data, stack.data[:, 1:]) and not lz.is_virtual
    assert s.undo(lz.id) == "Make Substack" and lz.is_virtual and np.array_equal(np.asarray(lz.data.copy()), stack.data)
    # float data: the streamed median comes from a 65536-bin histogram (close, not exact)
    f = LumenImage(np.random.default_rng(3).random((2, 3, 1, 9, 11)).astype(np.float32), name="f", axes="TZCYX")
    fl = s.add_image(io.read_image(io.write_image(f, tmp_path / "f.h5"), virtual=True))
    fr = s.add_image(LumenImage(f.data.copy(), name="f", axes="TZCYX"))
    a = s.run_op("stacks.statistics", {}, image_id=fl.id)["table"]["rows"][0]
    b = s.run_op("stacks.statistics", {}, image_id=fr.id)["table"]["rows"][0]
    assert a[1] == b[1] and np.allclose(a[2:7], b[2:7], rtol=1e-6) and abs(a[8] - b[8]) < 1e-3 and abs(a[7] - b[7]) < 0.01


def test_streamed_projection_progress_and_cancel(stack, tmp_path):
    p = _write(stack, tmp_path, "zarr")
    s = Session()
    lz = s.add_image(io.read_image(p, virtual=True))
    events = []
    s.listeners.append(lambda e: events.append(e) if e.get("type") == "progress" else None)
    s.PROGRESS_INTERVAL = 0.0
    s.run_op("stacks.z_project", {"method": "Standard Deviation"}, image_id=lz.id)
    msgs = [e["job"]["message"] for e in events if e["job"].get("message")]
    assert any(m.startswith("Plane") for m in msgs) and events[-1]["job"]["done"] and events[-1]["job"]["status"] == "ok"
    # cancellation between planes: the job ends "cancelled", no image is created, the stack stays lazy
    calls = {"n": 0}

    def cancel_soon(fraction, message=""):
        calls["n"] += 1
        if calls["n"] == 3:
            s.cancel_job(next(iter(s.jobs)))
    s.listeners.append(lambda e: cancel_soon(0) if e.get("type") == "progress" and not e["job"].get("done") else None)
    n_images = len(s.images)
    with pytest.raises(OperationCancelled):
        s.run_op("stacks.z_project", {"method": "Median"}, image_id=lz.id)
    assert len(s.images) == n_images and lz.is_virtual and events[-1]["job"]["status"] == "cancelled"


# ----------------------------------------------------------------------------- export
@pytest.mark.parametrize("fmt", ["h5", "zarr", "tif", "npy"])
def test_streamed_export_round_trip(stack, tmp_path, fmt):
    p = _write(stack, tmp_path, fmt)
    s = Session()
    lz = s.add_image(io.read_image(p, virtual=True))
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "out" / "e.ome.tif"), "format": "OME-TIFF", "compression": "zlib"}, image_id=lz.id)
    back = io.read_image(tmp_path / "out" / "e.ome.tif")
    assert np.array_equal(back.data, stack.data) and "Saved" in out["message"]
    if fmt != "npy":  # a .npy file carries neither calibration nor channel names
        assert back.calibration.pixel_width == pytest.approx(0.5) and [c.name for c in back.display.channels] == ["DAPI", "GFP"]
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "out" / "e.tif"), "format": "ImageJ TIFF"}, image_id=lz.id)
    assert np.array_equal(io.read_image(tmp_path / "out" / "e.tif").data, stack.data)
    s.run_op("file.bioformats_export", {"path": str(tmp_path / "out" / "e.zarr"), "format": "OME-Zarr"}, image_id=lz.id)
    assert np.array_equal(io.read_image(tmp_path / "out" / "e.zarr").data, stack.data)
    # one file per Z section: six small files, the pieces sliced lazily
    out = s.run_op("file.bioformats_export", {"path": str(tmp_path / "split" / "e.ome.tif"), "format": "OME-TIFF", "write_each_z_section": True}, image_id=lz.id)
    assert len(out["table"]["rows"]) == 3 and np.array_equal(io.read_image(tmp_path / "split" / "e_Z1.ome.tif").data, stack.data[:, 1:2])  # Fiji's 0-based Z=1
    out = s.run_op("file.export_sequence", {"folder": str(tmp_path / "seq"), "format": "tif"}, image_id=lz.id)
    assert len(out["table"]["rows"]) == 6
    assert lz.is_virtual  # nothing above read the stack into memory
    if fmt != "npy":
        assert not lz.data.materialized
    # Save As (write_image) of every streamed format, plus the RGB page layout
    for ext in ("tif", "ome.tif", "npy", "h5", "zarr"):
        assert np.array_equal(io.read_image(io.write_image(lz, tmp_path / f"save.{ext}")).data, stack.data) and lz.is_virtual
    rgb = LumenImage(np.random.default_rng(1).integers(0, 255, (1, 2, 3, 6, 7)).astype(np.uint8), name="rgb", axes="TZCYX", rgb=True)
    rv = io.read_image(io.write_image(rgb, tmp_path / "rgb.h5"), virtual=True)
    rv.rgb = True
    for ext in ("tif", "ome.tif"):
        b = io.read_image(io.write_image(rv, tmp_path / f"rgb_out.{ext}"))
        assert b.rgb and np.array_equal(b.data, rgb.data)


def test_export_progress_and_cancel_leave_no_half_file(stack, tmp_path):
    lz = io.read_image(_write(stack, tmp_path, "h5"), virtual=True)
    progress = []
    io.write_image(lz, tmp_path / "p.ome.tif", progress=lambda f, m: progress.append((f, m)))
    assert progress and progress[-1][0] == 1.0 and progress[-1][1] == "Page 12/12"
    seen = {"n": 0}

    def check_cancel():
        seen["n"] += 1
        if seen["n"] > 3:
            raise OperationCancelled()
    with pytest.raises(OperationCancelled):
        io.write_image(lz, tmp_path / "c.tif", check_cancel=check_cancel)
    assert not (tmp_path / "c.tif").exists() and lz.is_virtual
    # through the session: the job ends cancelled and the stack stays lazy
    s = Session()
    im = s.add_image(lz)
    events = []
    s.listeners.append(lambda e: events.append(e) if e.get("type") == "progress" else None)
    s.PROGRESS_INTERVAL = 0.0
    s.listeners.append(lambda e: s.cancel_job(e["job"]["id"]) if e.get("type") == "progress" and e["job"].get("message", "").startswith("Page 4") else None)
    with pytest.raises(OperationCancelled):
        s.run_op("file.bioformats_export", {"path": str(tmp_path / "x.ome.tif"), "format": "OME-TIFF"}, image_id=im.id)
    assert events[-1]["job"]["status"] == "cancelled" and not (tmp_path / "x.ome.tif").exists() and im.is_virtual


# ----------------------------------------------------------------------------- copy-on-write, undo, revert
@pytest.mark.parametrize("fmt", ["h5", "zarr", "tif"])
def test_plane_op_materialises_once_undo_and_revert_stay_lazy(stack, tmp_path, fmt):
    p = _write(stack, tmp_path, fmt)
    s = Session()
    img = s.add_image(io.read_image(p, virtual=True))
    data = img.data
    n0 = data.planes_read
    s.run_op("math.add", {"value": 5}, image_id=img.id)  # scope "stack": apply_plane_op copies once, then writes per plane
    assert not img.is_virtual and isinstance(img.data, np.ndarray) and not isinstance(img.data, lazy.LazyArray)
    assert np.array_equal(img.data, np.clip(stack.data.astype(np.int64) + 5, 0, 65535))
    assert not data.materialized and data.planes_read - n0 <= 2 * 12  # the lazy array itself was never materialised
    # the undo snapshot is a fresh lazy handle (not a 2 GB copy): undo brings the virtual stack back
    hist = s.histories[img.id]
    assert hist.labels()["undo"] == ["Add"] and hist.undo[0].nbytes < stack.data.nbytes and io.is_virtual(hist.undo[0].data)
    assert s.undo(img.id) == "Add" and img.is_virtual and np.array_equal(img.data[1, 2], stack.data[1, 2])
    assert s.redo(img.id) == "Add" and not img.is_virtual
    # Revert re-reads lazily and keeps the image virtual
    s.run_op("file.revert", {}, image_id=img.id)
    assert img.is_virtual and np.array_equal(np.asarray(img.data.copy()), stack.data) and img.metadata["source"]["virtual"] is True
    # a plane op on one slice only still materialises (copy-on-write happens once, on the first write)
    s.run_op("edit.invert", {}, image_id=img.id, scope="slice")
    assert not img.is_virtual and np.array_equal(img.data[1:], stack.data[1:]) if img.display.t == 1 else True
    # a session-level memory figure counts resident bytes only while the stack is virtual
    s.undo(img.id)
    assert img.is_virtual and s.state()["memory_bytes"] < 3 * stack.data.nbytes


def test_virtual_stack_api_surface_and_fresh_handles(stack, tmp_path):
    vs = io.read_image(_write(stack, tmp_path, "tif"), virtual=True).data
    assert isinstance(vs, io.VirtualStack) and isinstance(vs, lazy.LazyArray) and vs.memmapped is False and vs.description == "planes on demand"
    assert len(vs) == 2 and vs.size == stack.data.size and vs.ndim == 5 and repr(vs).startswith("<VirtualStack")
    assert [np.array_equal(sl, stack.data[i]) for i, sl in enumerate(vs)] == [True, True]
    fresh = vs.fresh()
    assert isinstance(fresh, lazy.LazyArray) and not fresh.materialized and np.array_equal(fresh[0, 1], stack.data[0, 1]) and fresh.planes_read == 2
    with pytest.raises(IndexError):
        vs[5]
    assert float(vs.max()) == float(stack.data.max()) and vs.materialized and not io.is_virtual(vs)
    assert np.array_equal(fresh.copy(), stack.data) and not fresh.materialized  # independent of the materialised original
    assert lazy.describe(np.zeros(3)) == "in memory" and lazy.resident_bytes(np.zeros(3, np.uint8)) == 3 and lazy.full_bytes(vs) == stack.data.nbytes
    assert lazy.physical_memory_bytes() > 0 and lazy.default_auto_virtual_mb() >= 1
    # an unknown axis folds into Z exactly as LumenImage.to_canonical does; a layout that needs merging is refused
    with h5py.File(tmp_path / "odd.h5", "w") as hf:
        ds = hf.create_dataset("odd", data=np.arange(2 * 2 * 4 * 5, dtype=np.uint8).reshape(2, 2, 4, 5))
        ds.attrs["axes"] = "QCYX"
    odd = io.read_image(tmp_path / "odd.h5", virtual=True)
    assert odd.is_virtual and odd.shape == (1, 2, 2, 4, 5) and np.array_equal(np.asarray(odd.data), io.read_image(tmp_path / "odd.h5").data)
    with pytest.raises(ValueError, match="without merging"):
        lazy.map_axes("QITYX", (2, 2, 2, 4, 5), False)
