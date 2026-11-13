"""Grid/Collection stitching extras and elastic montaging (lumen/core/ops/stitching_extra.py)."""
import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi
from skimage.data import camera
from skimage.transform import warp

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops.stitching_extra import FUSIONS, find_offset_nd, optimise_positions, read_tile_configuration, write_tile_configuration
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

CAM = camera().astype(np.float32)
GRID_FILES = [(f"tile_{k:02d}.tif", rc) for k, rc in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]


def write_grid(folder, tile=200, overlap=0.2, jitter=True):
    """2 × 2 uint8 crops of camera() with a known jitter; returns the true (x, y) positions."""
    step = int(tile * (1 - overlap))
    truth = []
    for k, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        y = r * step + (3 if (jitter and r) else 0)
        x = c * step + (-4 if (jitter and c) else 0)
        tifffile.imwrite(str(folder / f"tile_{k:02d}.tif"), CAM[y:y + tile, x:x + tile].astype(np.uint8))
        truth.append((x, y))
    return truth


def blobs(shape=(24, 128, 128), n=200, seed=1):
    r = np.random.default_rng(seed)
    v = np.zeros(shape, np.float32)
    for _ in range(n):
        z, y, x = (r.random(3) * np.array(shape)).astype(int)
        v[z, y, x] = r.random() * 100 + 50
    return ndi.gaussian_filter(v, (1.0, 2.0, 2.0)) * 50


def deform(plane, amp=3.0, period=60.0):
    H, W = plane.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    return warp(plane, np.array([rr + amp * np.sin(2 * np.pi * cc / period), cc + amp * np.cos(2 * np.pi * rr / period)]), order=1, mode="constant", preserve_range=True).astype(np.float32)


def positions(s):
    tab = s.tables["Tile positions"]
    return [tuple(int(round(v)) for v in row[2:4]) for row in tab["rows"]]


def test_ops_registered():
    for op_id in ("registration.stitch_collection", "registration.elastic_montage"):
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Plugins/Registration" and spec.params and spec.description and not spec.requires_image


def test_tile_configuration_round_trip(tmp_path):
    entries = [("a.tif", None, (0.0, 0.0)), ("b c.tif", 2, (160.5, -3.25))]
    p = write_tile_configuration(tmp_path / "TileConfiguration.txt", entries, 2)
    text = (tmp_path / "TileConfiguration.txt").read_text(encoding="utf-8")
    assert text.startswith("# Define the number of dimensions we are working on\ndim = 2\n") and "b c.tif; 2; (160.5, -3.25)" in text
    conf = read_tile_configuration(p)
    assert conf["dim"] == 2 and [e["file"] for e in conf["entries"]] == ["a.tif", "b c.tif"]
    assert conf["entries"][0]["series"] is None and conf["entries"][1]["series"] == 2 and conf["entries"][1]["position"] == [160.5, -3.25]
    # Fiji's own 3-D layout (comments, blank lines, empty series field)
    (tmp_path / "tc3.txt").write_text("# Define the number of dimensions we are working on\ndim = 3\n\n# Define the image coordinates\n"
                                      "img_01.tif; ; (0.0, 0.0, 0.0)\nimg_02.tif; ; (100.0, 5.0, 2.0)\n", encoding="utf-8")
    conf = read_tile_configuration(tmp_path / "tc3.txt")
    assert conf["dim"] == 3 and conf["entries"][1]["position"] == [100.0, 5.0, 2.0]
    p3 = write_tile_configuration(tmp_path / "tc3b.txt", [(e["file"], e["series"], e["position"]) for e in conf["entries"]], 3)
    assert read_tile_configuration(p3) == conf
    with pytest.raises(ValueError, match="not found"):
        read_tile_configuration(tmp_path / "missing.txt")
    (tmp_path / "bad.txt").write_text("dim = 2\nfoo.tif (1, 2)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot parse"):
        read_tile_configuration(tmp_path / "bad.txt")
    (tmp_path / "mismatch.txt").write_text("dim = 3\nfoo.tif; ; (1, 2)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 3"):
        read_tile_configuration(tmp_path / "mismatch.txt")


def test_find_offset_and_global_optimisation_helpers():
    a, b = CAM[:100, :100], CAM[20:120, 30:130]
    off, ncc = find_offset_nd(a, b)
    assert tuple(off) == (20, 30) and ncc > 0.9
    vol = blobs((16, 64, 64))
    off, ncc = find_offset_nd(vol[2:14, 0:48, 0:48], vol[0:12, 8:56, 20:68])
    assert tuple(off) == (-2, 8, 20) and ncc > 0.9
    # an inconsistent link (0 → 2 says 250, the chain says 200) is dropped by the displacement thresholds
    pairs = [{"i": 0, "j": 1, "d": np.array([0, 0, 100.0]), "ncc": 0.9}, {"i": 1, "j": 2, "d": np.array([0, 0, 100.0]), "ncc": 0.9},
             {"i": 0, "j": 2, "d": np.array([0, 0, 250.0]), "ncc": 0.8}]
    pos, used, dropped, errors = optimise_positions(3, pairs, None, 0.3, 2.5, 3.5)
    assert used == {0, 1} and not dropped and np.allclose(pos[:, 2] - pos[0, 2], [0, 100, 200]) and max(errors.values()) < 1e-6
    # with huge thresholds the least squares compromise keeps every link
    pos, used, dropped, _ = optimise_positions(3, pairs, None, 0.3, 1e9, 1e9)
    assert used == {0, 1, 2} and 200 < pos[2, 2] - pos[0, 2] < 250
    # a link below the regression threshold is ignored: the tile is unconnected → dropped (unknown positions) or kept at its nominal position
    pairs[1]["ncc"] = 0.1
    pairs[2]["ncc"] = 0.1
    pos, used, dropped, _ = optimise_positions(3, pairs, None, 0.3, 2.5, 3.5)
    assert used == {0} and dropped == [2]
    nominal = np.array([[0, 0, 0], [0, 0, 90.0], [0, 0, 210.0]])
    pos, used, dropped, _ = optimise_positions(3, pairs, nominal, 0.3, 2.5, 3.5)
    assert not dropped and abs(pos[2, 2] - 210) < 1e-6 and abs((pos[1, 2] - pos[0, 2]) - 100) < 1e-3


def test_unknown_positions_2x2_recovers_the_layout(tmp_path):
    truth = write_grid(tmp_path)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    s = Session()
    out = s.run_op("registration.stitch_collection", {"layout": "Unknown positions (folder)", "folder": str(tmp_path), "pattern": "tile_*.tif", "subpixel": False})
    st = s.images[out["new_images"][0]]
    assert s.tables["Tile positions"]["columns"] == ["Tile", "File", "X", "Y"] and positions(s) == truth
    assert (st.width, st.height) == (356, 363) and st.dtype == np.uint8 and st.name == f"Stitched {tmp_path.name}"
    assert np.abs(st.data[0, 0, 0].astype(float) - CAM[:363, :356]).max() <= 1
    assert len(s.results.rows) == 6 and sum(1 for r in s.results.rows if r["Used"]) >= 4 and "TileConfiguration.registered.txt" in out["message"]
    conf = read_tile_configuration(tmp_path / "TileConfiguration.registered.txt")
    assert conf["dim"] == 2 and [tuple(int(round(v)) for v in e["position"]) for e in conf["entries"]] == truth
    assert st.metadata["stitch_positions"][1][:2] == [156.0, 0.0]
    # sub-pixel + downsampled computation land within half a pixel; output written to a file
    out = s.run_op("registration.stitch_collection", {"layout": "Unknown positions (folder)", "folder": str(tmp_path), "pattern": "tile_*.tif", "downsample": 2, "subpixel": True,
                                                      "output": "Write to file", "output_path": str(tmp_path / "out" / "fused.tif"), "title": "fused"})
    assert not out["new_images"] and (tmp_path / "out" / "fused.tif").is_file() and "fused.tif" in out["message"]
    rows = s.tables["Tile positions"]["rows"]
    assert all(abs(r[2] - t[0]) <= 0.5 and abs(r[3] - t[1]) <= 0.5 for r, t in zip(rows, truth))
    with pytest.raises(ValueError, match="No tiles matching"):
        s.run_op("registration.stitch_collection", {"layout": "Unknown positions (folder)", "folder": str(tmp_path), "pattern": "nothing_*.tif"})


@pytest.mark.parametrize("fusion", [f for f in FUSIONS if f != "Do not fuse"])
def test_positions_from_file_compute_overlap_and_fusion_methods(tmp_path, fusion):
    truth = write_grid(tmp_path)
    write_tile_configuration(tmp_path / "TileConfiguration.txt", [(name, None, (c * 160, r * 160)) for name, (r, c) in GRID_FILES], 2)
    s = Session()
    out = s.run_op("registration.stitch_collection", {"layout": "Positions from file", "layout_file": str(tmp_path / "TileConfiguration.txt"), "fusion": fusion, "subpixel": False})
    st = s.images[out["new_images"][0]]
    assert positions(s) == truth and (st.width, st.height) == (356, 363)
    if fusion == "Overlay into composite":
        assert st.nc == 4 and st.display.mode == "composite" and np.array_equal(st.data[0, 0, 1, :200, 156:356], CAM[:200, 156:356].astype(np.uint8))
    else:
        assert st.nc == 1 and np.abs(st.data[0, 0, 0].astype(float) - CAM[:363, :356]).max() <= 1  # identical content in the overlaps


def test_positions_from_file_without_overlap_computation_and_do_not_fuse(tmp_path):
    write_grid(tmp_path)
    write_tile_configuration(tmp_path / "TileConfiguration.txt", [(name, None, (c * 160, r * 160)) for name, (r, c) in GRID_FILES], 2)
    s = Session()
    out = s.run_op("registration.stitch_collection", {"layout": "Positions from file", "folder": str(tmp_path), "compute_overlap": False})
    st = s.images[out["new_images"][0]]
    assert positions(s) == [(0, 0), (160, 0), (0, 160), (160, 160)] and (st.width, st.height) == (360, 360) and not s.results.rows
    out = s.run_op("registration.stitch_collection", {"layout": "Positions from file", "folder": str(tmp_path), "fusion": "Do not fuse", "subpixel": False,
                                                      "registered_file": str(tmp_path / "reg" / "mine.txt")})
    assert not out["new_images"] and "mine.txt" in out["message"]
    conf = read_tile_configuration(tmp_path / "reg" / "mine.txt")
    assert [tuple(int(round(v)) for v in e["position"]) for e in conf["entries"]] == [(0, 0), (156, 0), (0, 163), (156, 163)]
    # grid layout through the same command
    out = s.run_op("registration.stitch_collection", {"layout": "Grid (folder)", "folder": str(tmp_path), "pattern": "tile_*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "subpixel": False})
    assert positions(s) == [(0, 0), (156, 0), (0, 163), (156, 163)]
    with pytest.raises(ValueError, match="need 9"):
        s.run_op("registration.stitch_collection", {"layout": "Grid (folder)", "folder": str(tmp_path), "pattern": "tile_*.tif", "grid_x": 3, "grid_y": 3})
    with pytest.raises(ValueError, match="not found in"):
        write_tile_configuration(tmp_path / "ghost.txt", [("ghost.tif", None, (0, 0)), ("tile_00.tif", None, (10, 0))], 2)
        s.run_op("registration.stitch_collection", {"layout": "Positions from file", "layout_file": str(tmp_path / "ghost.txt")})


def test_open_images_multichannel_time_series():
    def tile(x0):
        c0 = CAM[:200, x0:x0 + 200]
        c1 = np.roll(CAM[:200, x0:x0 + 200], 7, axis=0)
        f0 = np.stack([c0, c1])
        return np.stack([f0, f0 * 0.5]).astype(np.uint8)[:, None]  # (T, Z, C, Y, X)

    s = Session()
    a = s.add_image(LumenImage(tile(0), name="A", axes="TZCYX"))
    b = s.add_image(LumenImage(tile(150), name="B", axes="TZCYX"))
    out = s.run_op("registration.stitch_collection", {"layout": "Open images", "fusion": "Average", "subpixel": False, "title": "AB"})
    st = s.images[out["new_images"][0]]
    assert st.shape == (2, 1, 2, 200, 350) and st.name == "AB" and positions(s) == [(0, 0), (150, 0)]
    assert np.abs(st.data[0, 0, 0].astype(float) - CAM[:200, :350]).max() <= 1 and np.abs(st.data[1, 0, 1].astype(float) - np.roll(CAM[:200, :350], 7, axis=0) * 0.5).max() <= 1
    assert "TileConfiguration" not in out["message"]
    # every frame registered separately: a Frame column, two sets of positions
    out = s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A, B", "timelapse": "Register every frame separately", "subpixel": False, "channel": 2})
    tab = s.tables["Tile positions"]
    assert tab["columns"] == ["Tile", "File", "X", "Y", "Frame"] and [r[4] for r in tab["rows"]] == [1, 1, 2, 2] and s.images[out["new_images"][0]].shape == (2, 1, 2, 200, 350)
    # approximate positions given as text, refined by the overlap computation
    out = s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A, B", "positions": "0,0; 146,2", "subpixel": False})
    assert positions(s) == [(0, 0), (150, 0)]
    s.add_image(LumenImage(tile(300), name="C", axes="TZCYX"))
    with pytest.raises(ValueError, match="2 positions given for 3"):
        s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A, B, C", "positions": "0,0; 146,2"})
    with pytest.raises(ValueError, match="at least two"):
        s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A"})


def test_3d_tiles_unknown_positions_and_from_file(tmp_path):
    big = blobs((24, 128, 128))
    a = big[2:18, 0:80, 0:80]
    b = big[0:16, 4:84, 56:128]  # 72 px wide; origin relative to a: z −2, y +4, x +56
    tifffile.imwrite(str(tmp_path / "a.tif"), a)
    tifffile.imwrite(str(tmp_path / "b.tif"), b)
    s = Session()
    out = s.run_op("registration.stitch_collection", {"layout": "Unknown positions (folder)", "folder": str(tmp_path), "pattern": "*.tif", "fusion": "Max intensity", "subpixel": False})
    st = s.images[out["new_images"][0]]
    tab = s.tables["Tile positions"]
    assert tab["columns"] == ["Tile", "File", "X", "Y", "Z"]
    assert [tuple(int(round(v)) for v in r[2:5]) for r in tab["rows"]] == [(0, 0, 2), (56, 4, 0)]
    assert st.shape == (1, 18, 1, 84, 128) and st.dtype == np.float32 and "3-D" in out["message"]
    assert np.allclose(st.data[0, 2:18, 0, 0:80, 0:80], a, atol=1e-3) and np.allclose(st.data[0, 0:16, 0, 4:84, 56:128], b, atol=1e-3)
    assert s.results.rows[-1]["dz"] == -2 and read_tile_configuration(tmp_path / "TileConfiguration.registered.txt")["dim"] == 3
    # 3-D positions from a file (slightly off), refined with sub-voxel accuracy
    write_tile_configuration(tmp_path / "TileConfiguration.txt", [("a.tif", None, (0, 0, 0)), ("b.tif", None, (58, 2, -3))], 3)
    out = s.run_op("registration.stitch_collection", {"layout": "Positions from file", "folder": str(tmp_path), "subpixel": True, "fusion": "Linear blending"})
    rows = s.tables["Tile positions"]["rows"]
    assert abs(rows[1][2] - 56) <= 0.5 and abs(rows[1][3] - 4) <= 0.5 and abs(rows[1][4] - 0) <= 0.5 and abs(rows[0][4] - 2) <= 0.5
    assert s.images[out["new_images"][0]].nz in (18, 19)


def test_subpixel_placement_has_no_dark_seam_and_keeps_integer_offsets_exact():
    """Sub-pixel tile placement: an exact integer offset that comes out of the least squares as 119.99999999999999 is
    snapped (no 1-px zero shift, no lost column), and a genuinely fractional tile does not fuse its zero-blended
    leading column (Fiji contributes a tile from ceil(position) on) – no dark seam line in any fusion mode."""
    from lumen.core.ops import stitching_extra as se
    rng = np.random.default_rng(0)
    big = ndi.gaussian_filter(rng.random((200, 300)) * 255, 1.5).astype(np.float32)
    A, B = big[:, :180], big[:, 120:]
    xs, ys = np.arange(180) + 120.4, np.arange(200)
    B4 = ndi.map_coordinates(big, np.meshgrid(ys, xs, indexing="ij"), order=1, mode="nearest").astype(np.float32)
    for fusion in ("Average", "Median", "Linear blending", "Min intensity", "Max intensity"):
        s = Session()
        s.add_image(LumenImage(A.copy(), name="A"))
        s.add_image(LumenImage(B.copy(), name="B"))
        out = s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A, B", "subpixel": True, "fusion": fusion})
        st = s.images[out["new_images"][0]]
        assert s.tables["Tile positions"]["rows"][1][2] == 120.0 and st.width == 300, fusion
        assert np.abs(st.data[0, 0, 0] - big).max() < 1e-3, fusion
        s = Session()
        s.add_image(LumenImage(A.copy(), name="A"))
        s.add_image(LumenImage(B4.copy(), name="B"))
        out = s.run_op("registration.stitch_collection", {"layout": "Open images", "images": "A, B", "subpixel": True, "fusion": fusion})
        st = s.images[out["new_images"][0]]
        posx = s.tables["Tile positions"]["rows"][1][2]
        assert abs(posx - 120.4) < 0.15, (fusion, posx)
        col = int(np.floor(posx))
        colerr = np.abs(st.data[0, 0, 0, :, :300] - big[:, :st.width][:, :300]).mean(axis=0)
        assert colerr[col] < 1.0 and colerr[col - 1] < 1.0 and colerr[col + 1] < 1.0, (fusion, colerr[col - 1:col + 2])
    # the helpers: a fractional shift marks the leading row / column invalid, an integer one returns the tile as is
    tile = np.ones((1, 1, 1, 4, 5), np.float32)
    shifted, valid = se._shift_fraction(tile, np.array([0.0, 0.0, 0.3]))
    # scipy's constant mode does not interpolate across the edge: the leading column is the fill value itself
    assert shifted[0, 0, 0, 0, 0] == 0.0 and shifted[0, 0, 0, 0, 1] == pytest.approx(1.0) and not valid[0, :, 0].any() and valid[0, :, 1:].all()
    same, none = se._shift_fraction(tile, np.array([0.0, 0.0, 0.0]))
    assert same is tile and none is None
    data, masks = se._placed([tile, tile], [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])], True)
    assert masks[0] is None and masks[1] is not None and not masks[1][0, 0].any()
    assert se._placed([tile], [np.array([0.0, 0.5, 0.0])], False) == ([tile], None)


def test_elastic_montage_of_two_overlapping_tiles():
    A = CAM[100:300, 100:300]
    B = CAM[100:300, 250:450]  # 50 px overlap, x offset 150
    s = Session()
    s.add_image(LumenImage(A.astype(np.uint8), name="A"))
    s.add_image(LumenImage(B.astype(np.uint8), name="B"))
    out = s.run_op("registration.elastic_montage", {"layout": "Open images", "title": "M"})
    m = s.images[out["new_images"][0]]
    tab = s.tables["Montage"]
    assert tab["columns"] == ["Tile", "File", "X", "Y", "Mean deformation", "Max deformation"] and len(tab["rows"]) == 2
    x0, y0 = int(tab["rows"][0][2]), int(tab["rows"][0][3])
    assert tab["rows"][1][2] - x0 == 150 and tab["rows"][1][3] == y0 and m.dtype == np.uint8
    inner = m.data[0, 0, 0, y0 + 4:y0 + 196, x0 + 4:x0 + 346].astype(float)
    assert np.abs(inner - CAM[104:296, 104:446]).mean() < 1.5  # identical content: the montage reproduces the scene
    row = s.results.rows[-1]
    assert row["Label"] == "A → B" and row["Matches"] > 20 and row["Residual before"] < 1.0 and row["Residual after"] <= row["Residual before"] + 0.05
    # a deformed second tile: the spring meshes pull the overlap into agreement
    s2 = Session()
    s2.add_image(LumenImage(A.astype(np.uint8), name="A"))
    s2.add_image(LumenImage(deform(B).astype(np.uint8), name="B2"))
    out = s2.run_op("registration.elastic_montage", {"layout": "Open images", "images": "A, B2", "mesh_spacing": 16})
    row = s2.results.rows[-1]
    assert row["Residual before"] > 1.0 and row["Residual after"] < row["Residual before"] / 3
    mont = s2.images[out["new_images"][0]]
    assert mont.width >= 350 and mont.height >= 200 and "mean pair residual" in out["message"]
    assert max(r[5] for r in s2.tables["Montage"]["rows"]) > 0.5  # the tiles were deformed
    with pytest.raises(ValueError, match="at least two"):
        s2.run_op("registration.elastic_montage", {"layout": "Open images", "images": "A"})
    with pytest.raises(ValueError, match="No overlap"):
        s2.run_op("registration.elastic_montage", {"layout": "Open images", "images": "A, B2", "positions": "0,0; 1000,0"})


def test_elastic_montage_from_folder_configuration_and_stacks(tmp_path):
    A = np.stack([CAM[100:300, 100:300], CAM[100:300, 100:300] * 0.5]).astype(np.uint8)  # 2-slice stacks (plain TIFF → ZYX)
    B = np.stack([CAM[100:300, 250:450], CAM[100:300, 250:450] * 0.5]).astype(np.uint8)
    tifffile.imwrite(str(tmp_path / "a.tif"), A)
    tifffile.imwrite(str(tmp_path / "b.tif"), B)
    write_tile_configuration(tmp_path / "TileConfiguration.txt", [("a.tif", None, (0, 0)), ("b.tif", None, (148, 1))], 2)
    s = Session()
    out = s.run_op("registration.elastic_montage", {"layout": "Positions from file", "folder": str(tmp_path), "output": "Write to file and open", "output_path": str(tmp_path / "montage.tif")})
    m = s.images[out["new_images"][0]]
    assert m.nz == 2 and (tmp_path / "montage.tif").is_file() and s.tables["Montage"]["rows"][1][2] - s.tables["Montage"]["rows"][0][2] == 150
    x0, y0 = int(s.tables["Montage"]["rows"][0][2]), int(s.tables["Montage"]["rows"][0][3])
    assert np.abs(m.data[0, 1, 0, y0 + 4:y0 + 196, x0 + 4:x0 + 346].astype(float) - CAM[104:296, 104:446] * 0.5).mean() < 1.5  # the same field warps every plane
    with pytest.raises(ValueError, match="Give the output file"):
        s.run_op("registration.elastic_montage", {"layout": "Positions from file", "folder": str(tmp_path), "output": "Write to file"})
