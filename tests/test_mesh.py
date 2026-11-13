"""Iso-surface meshes: /api/images/{id}/mesh, the mesh.{obj,stl,ply} downloads and the mesh.export / mesh.measure commands."""
import math
import os
import struct

import numpy as np
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops.mesh_ops import block_mean, extract_mesh, level_key, measure_components, mesh_strides, vertex_normals
from lumen.core.session import Session
from lumen.server.app import create_app


def _client():
    s = Session()
    return TestClient(create_app(s)), s


def _open_cells(c):
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Cells (3-D, 2 channels)"}})
    assert r.status_code == 200, r.text
    return r.json()["new_images"][0]


def _sphere_volume(nz=64, n=64, zr=1.0, r=20.0):
    """Smooth field 100 - distance (in xy pixel units) to the centre: the level 100 - r is a sphere of radius r."""
    zz, yy, xx = np.mgrid[:nz, :n, :n]
    d = np.sqrt(((zz - nz // 2) * zr) ** 2 + (yy - n // 2) ** 2 + (xx - n // 2) ** 2).astype(np.float32)
    return 100.0 - d


def _sphere_session(nz=64, n=64, zr=1.0, r=20.0, unit="µm"):
    s = Session()
    im = LumenImage(_sphere_volume(nz, n, zr, r), name="sphere.tif", axes="ZYX")
    im.calibration = Calibration(1.0, 1.0, zr, unit)
    s.add_image(im)
    return s, im


def _parse_body(r):
    n, m = int(r.headers["x-vertices"]), int(r.headers["x-faces"])
    assert len(r.content) == n * 24 + m * 12
    v = np.frombuffer(r.content, np.float32, n * 3, 0).reshape(n, 3)
    nrm = np.frombuffer(r.content, np.float32, n * 3, n * 12).reshape(n, 3)
    f = np.frombuffer(r.content, np.uint32, m * 3, n * 24).reshape(m, 3)
    return v, nrm, f


# ------------------------------------------------------------------ endpoint
def test_mesh_endpoint_cells():
    c, s = _client()
    iid = _open_cells(c)
    r = c.get(f"/api/images/{iid}/mesh", params={"c": 1, "t": 0, "max_dim": 192})
    assert r.status_code == 200, r.text
    n, m = int(r.headers["x-vertices"]), int(r.headers["x-faces"])
    assert n > 100 and m > 100
    v, nrm, f = _parse_body(r)
    assert f.max() < n and f.min() >= 0
    # vertices live inside the image box (pixel units, z scaled by 0.29/0.26)
    W, H, D = map(float, r.headers["x-extent"].split(","))
    assert (W, H) == (256.0, 256.0) and abs(D - 60 * 0.29 / 0.26) < 1e-3
    assert v[:, 0].min() >= 0 and v[:, 0].max() <= W and v[:, 1].max() <= H and v[:, 2].max() <= D
    b = list(map(float, r.headers["x-bounds"].split(",")))
    assert len(b) == 6 and b[3] >= b[0] and abs(b[0] - v[:, 0].min()) < 1e-3
    # unit normals
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-3)
    # the default level is the midpoint of the display range
    ch = s.get_image(iid).display.channels[1]
    assert abs(float(r.headers["x-level"]) - (ch.min + 0.5 * (ch.max - ch.min))) < 1e-3
    assert r.headers["x-channel"] == "1" and r.headers["x-lut"] == "Green"
    assert abs(float(r.headers["x-spacing"]) - 0.29 / 0.26) < 1e-4
    assert int(r.headers["x-components"]) >= 1
    assert r.headers["x-stride"] == "2,2,2"  # 256 px > 192 → reduced by 2 (z too, to stay ~isotropic)
    # face winding is counter-clockwise with respect to the normals (right-hand rule)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    agree = (np.einsum("ij,ij->i", fn, nrm[f].mean(axis=1)) > 0).mean()
    assert agree > 0.95


def test_mesh_endpoint_cache_level_and_otsu():
    c, s = _client()
    iid = _open_cells(c)
    r1 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128})
    r2 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128})
    assert r1.content == r2.content and r1.headers["x-level"] == r2.headers["x-level"]
    r3 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128, "level": 30000})
    assert r3.headers["x-level"] == "30000" and r3.content != r1.content
    r4 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128, "level": "otsu"})
    assert 0 < float(r4.headers["x-level"]) < 65535 and float(r4.headers["x-level"]) != 30000
    # a numeric 0 is an ordinary level, not an Otsu alias (it is the zero crossing of signed data)
    r4b = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128, "level": 0})
    assert r4b.status_code == 200 and r4b.headers["x-level"] == "0"
    assert c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128, "level": "foo"}).status_code == 422
    # the 'auto' default follows the display range (a cache key component)
    ch = s.get_image(iid).display.channels[0]
    c.post(f"/api/images/{iid}/display", json={"channels": [{"min": ch.min, "max": ch.max * 0.5}]})
    r5 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128})
    assert float(r5.headers["x-level"]) < float(r1.headers["x-level"])
    # smoothing changes the vertices but not the topology
    r6 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "max_dim": 128, "level": 30000, "smooth": 5})
    assert r6.headers["x-faces"] == r3.headers["x-faces"] and r6.content != r3.content
    # level above the data range → an empty (but valid) mesh
    r7 = c.get(f"/api/images/{iid}/mesh", params={"c": 0, "level": 1e9})
    assert r7.status_code == 200 and r7.headers["x-vertices"] == "0" and r7.headers["x-faces"] == "0" and len(r7.content) == 0
    # bad channel / non-stack
    assert c.get(f"/api/images/{iid}/mesh", params={"c": 5}).status_code == 400
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    assert c.get(f"/api/images/{r.json()['new_images'][0]}/mesh").status_code == 400


def test_mesh_download_routes():
    c, s = _client()
    iid = _open_cells(c)
    r = c.get(f"/api/images/{iid}/mesh.obj", params={"c": 0, "max_dim": 128})
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"] == 'attachment; filename="cells3d_Membrane.obj"'
    assert r.headers["x-unit"] == "um"
    lines = r.text.splitlines()
    n = int(r.headers["x-vertices"])
    assert sum(1 for ln in lines if ln.startswith("v ")) == n
    assert sum(1 for ln in lines if ln.startswith("vn ")) == n
    assert sum(1 for ln in lines if ln.startswith("f ")) == int(r.headers["x-faces"])
    # calibrated by default (µm): the first vertex is within the 256 * 0.26 µm box
    xs = [float(ln.split()[1]) for ln in lines if ln.startswith("v ")]
    assert max(xs) <= 256 * 0.26 + 1e-3
    r_px = c.get(f"/api/images/{iid}/mesh.obj", params={"c": 0, "max_dim": 128, "calibrated": 0})
    assert r_px.headers["x-unit"] == "pixel"
    xs_px = [float(ln.split()[1]) for ln in r_px.text.splitlines() if ln.startswith("v ")]
    assert max(xs_px) > 100
    # STL: 80-byte header, uint32 count, 50 bytes per triangle
    r = c.get(f"/api/images/{iid}/mesh.stl", params={"c": 0, "max_dim": 128})
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('.stl"')
    m = struct.unpack("<I", r.content[80:84])[0]
    assert m == int(r.headers["x-faces"]) and len(r.content) == 84 + 50 * m
    # PLY: binary little endian with per-vertex normals
    r = c.get(f"/api/images/{iid}/mesh.ply", params={"c": 0, "max_dim": 128})
    assert r.status_code == 200 and r.content.startswith(b"ply\nformat binary_little_endian 1.0\n")
    header, _, body = r.content.partition(b"end_header\n")
    nv = int([ln for ln in header.split(b"\n") if ln.startswith(b"element vertex")][0].split()[-1])
    nf = int([ln for ln in header.split(b"\n") if ln.startswith(b"element face")][0].split()[-1])
    assert nv == int(r.headers["x-vertices"]) and len(body) == nv * 24 + nf * 13
    assert c.get(f"/api/images/{iid}/mesh.xyz").status_code == 404


# ------------------------------------------------------------------ geometry
def test_mesh_measure_sphere_area_and_volume():
    r = 20.0
    for nz, zr in ((64, 1.0), (33, 2.0)):  # isotropic and 2× anisotropic z
        s, im = _sphere_session(nz=nz, zr=zr, r=r)
        out = s.run_op("mesh.measure", {"level": 100 - r})
        assert out["results_added"] == 1, out
        row = s.results.rows[0]
        assert row["Unit"] == "µm" and row["Channel"] == 1 and row["Level"] == 100 - r
        assert abs(row["Area"] / (4 * math.pi * r ** 2) - 1) < 0.10
        assert abs(row["Volume"] / (4 / 3 * math.pi * r ** 3) - 1) < 0.10
        # centroid at the sphere centre (block-centre convention: voxel i ↦ i + 0.5)
        assert abs(row["X"] - 32.5) < 0.5 and abs(row["Y"] - 32.5) < 0.5 and abs(row["Z"] - ((nz // 2) + 0.5) * zr) < 0.5
        assert abs(row["Width"] - 2 * r) < 1.5 and abs(row["Depth"] - 2 * r) < 2.0
        assert row["Vertices"] > 100 and row["Faces"] > 100
        assert "1 surface component" in out["message"]


def test_mesh_smoothing_keeps_volume_and_removes_small_components():
    s, im = _sphere_session(r=20.0)
    s.run_op("mesh.measure", {"level": 80, "smooth": 20})
    row = s.results.rows[0]
    assert abs(row["Volume"] / (4 / 3 * math.pi * 20 ** 3) - 1) < 0.02  # Taubin: no shrinkage
    # a big and a small sphere → two components; min_faces drops the small one
    zz, yy, xx = np.mgrid[:48, :64, :64]
    big = 100 - np.sqrt((zz - 24) ** 2 + (yy - 32) ** 2 + (xx - 20) ** 2)
    small = 100 - np.sqrt((zz - 24) ** 2 + (yy - 32) ** 2 + (xx - 54) ** 2)
    vol = np.maximum(big, small).astype(np.float32)
    s2 = Session()
    s2.add_image(LumenImage(vol, name="two.tif", axes="ZYX"))
    out = s2.run_op("mesh.measure", {"level": 88, "max_dim": 0})  # level 88 → two spheres of radius 12
    assert out["results_added"] == 2
    vol2 = np.maximum(big, small - 6).astype(np.float32)  # the second sphere is 6 px smaller at the same level
    s3 = Session()
    s3.add_image(LumenImage(vol2, name="two.tif", axes="ZYX"))
    out = s3.run_op("mesh.measure", {"level": 88})
    assert out["results_added"] == 2
    rows = s3.results.rows
    assert rows[0]["Volume"] > rows[1]["Volume"]  # largest first
    assert rows[0]["Label"].endswith("#1") and rows[1]["Unit"] == "pixel"
    s3.results.clear()
    out = s3.run_op("mesh.measure", {"level": 88, "min_faces": int(rows[1]["Faces"]) + 1})
    assert out["results_added"] == 1 and "1 surface component" in out["message"]


def test_mesh_helpers():
    assert mesh_strides(60, 256, 256, 0.29 / 0.26, 192) == (2, 2, 2)
    assert mesh_strides(60, 256, 256, 0.29 / 0.26, 0) == (1, 1, 1)
    assert mesh_strides(60, 256, 256, 0.29 / 0.26, 512) == (1, 1, 1)
    assert mesh_strides(200, 64, 64, 5.0, 100) == (2, 10, 10)  # 200 slices × 5 = 1000 px extent → xy ÷ 10, z ÷ 2
    v = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    m = block_mean(v, 2, 2, 2)
    assert m.shape == (2, 2, 2) and m.dtype == np.float32 and abs(m[0, 0, 0] - v[:2, :2, :2].mean()) < 1e-5
    im = LumenImage(_sphere_volume(), name="s.tif", axes="ZYX")
    mesh = extract_mesh(im, 0, 0, level=80, step=2)
    assert mesh.n_faces > 0 and mesh.faces.dtype == np.uint32 and mesh.vertices.dtype == np.float32
    rows = measure_components(mesh, scale=2.0)  # calibrated scale: area × 4, volume × 8
    assert abs(rows[0]["volume"] / (8 * 4 / 3 * math.pi * 20 ** 3) - 1) < 0.1
    assert abs(rows[0]["area"] / (4 * 4 * math.pi * 20 ** 2) - 1) < 0.1
    empty = extract_mesh(im, 0, 0, level=1e6)
    assert empty.n_vertices == 0 and empty.to_bytes() == b"" and measure_components(empty) == []


# ------------------------------------------------------------------ export command
def _count_obj(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for ln in lines if ln.startswith("v ")), sum(1 for ln in lines if ln.startswith("f "))


def _count_stl(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    assert len(data) == 84 + 50 * n
    return n


def _count_ply(path):
    data = path.read_bytes()
    header, _, body = data.partition(b"end_header\n")
    nv = int([ln for ln in header.split(b"\n") if ln.startswith(b"element vertex")][0].split()[-1])
    nf = int([ln for ln in header.split(b"\n") if ln.startswith(b"element face")][0].split()[-1])
    assert len(body) == nv * 24 + nf * 13
    # faces are triangles with in-range indices
    idx = np.frombuffer(body[nv * 24:], np.dtype([("n", "u1"), ("i", "<u4", (3,))]))
    assert (idx["n"] == 3).all() and idx["i"].max() < nv
    return nv, nf


def test_mesh_export_formats(tmp_path):
    s, im = _sphere_session(r=15.0)
    im.display.c = 0
    out = s.run_op("mesh.export", {"path": str(tmp_path / "sphere"), "format": "obj", "level": 85})
    assert "Wrote" in out["message"] and "µm" in out["message"]
    p = tmp_path / "sphere.obj"
    assert p.exists()
    nv, nf = _count_obj(p)
    mesh = extract_mesh(im, 0, 0, level=85)
    assert nv == mesh.n_vertices and nf == mesh.n_faces
    # STL: a directory → <image>_<channel>.stl inside it
    out = s.run_op("mesh.export", {"path": str(tmp_path) + os.sep, "format": "stl", "level": 85})
    p = tmp_path / "sphere_C1.stl"
    assert p.exists() and _count_stl(p) == mesh.n_faces
    # PLY with explicit extension and smoothing
    out = s.run_op("mesh.export", {"path": str(tmp_path / "s.ply"), "format": "ply", "level": 85, "smooth": 3})
    nv, nf = _count_ply(tmp_path / "s.ply")
    assert nv == mesh.n_vertices and nf == mesh.n_faces
    # calibrated coordinates (pixel width 1 here) vs. a scaled image: the OBJ shrinks by the pixel size
    im.calibration = Calibration(0.5, 0.5, 0.5, "µm")
    s.run_op("mesh.export", {"path": str(tmp_path / "cal.obj"), "format": "obj", "level": 85})
    xs = [float(ln.split()[1]) for ln in (tmp_path / "cal.obj").read_text(encoding="utf-8").splitlines() if ln.startswith("v ")]
    assert max(xs) < 32 * 0.5 + 15 * 0.5 + 1  # half box + radius, in µm
    s.run_op("mesh.export", {"path": str(tmp_path / "px.obj"), "format": "obj", "level": 85, "calibrated": False})
    xs_px = [float(ln.split()[1]) for ln in (tmp_path / "px.obj").read_text(encoding="utf-8").splitlines() if ln.startswith("v ")]
    assert max(xs_px) > 40
    # errors
    try:
        s.run_op("mesh.export", {"path": "", "format": "obj"})
        assert False, "expected an error"
    except ValueError as e:
        assert "output path" in str(e)


# ------------------------------------------------------------------ regressions
def _open_edge_count(faces):
    """Edges not shared by exactly two triangles (0 for a closed manifold surface)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def test_mesh_step_padding_keeps_boundary_surfaces_closed():
    # a cylinder along z touches both z faces of the stack: with step > 1 the closure pad must still be reached
    # (marching cubes only visits cells at index multiples of `step`), for even and odd stack sizes alike
    for n in (64, 65):
        zz, yy, xx = np.mgrid[:n, :n, :n]
        vol = np.where((yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= 16 ** 2, 200.0, 0.0).astype(np.float32)
        im = LumenImage(vol, name="cyl.tif", axes="ZYX")
        for step in (1, 2, 3):
            m = extract_mesh(im, 0, 0, level=100, step=step)
            assert m.n_faces > 0
            assert _open_edge_count(m.faces) == 0, (n, step)
            rows = measure_components(m)
            assert len(rows) == 1 and rows[0]["orientation"] == "outward"
            assert abs(rows[0]["volume"] / (math.pi * 16 ** 2 * n) - 1) < 0.10, (n, step, rows[0]["volume"])
            # the un-pad offset is right: the cylinder spans the whole z extent (the caps sit within half a step of
            # the box faces, where the sub-level pad is interpolated) and it is centred in xy
            assert abs(m.vertices[:, 2].min()) <= 0.5 * step + 0.01 and abs(m.vertices[:, 2].max() - n) <= 0.5 * step + 0.01
            assert abs(rows[0]["x"] - n / 2) < 0.6 and abs(rows[0]["y"] - n / 2) < 0.6


def test_mesh_normals_follow_anisotropic_spacing():
    # scikit-image's normals are index-space gradients: without the spacing correction they tilt toward z
    for nz, zr in ((64, 1.0), (33, 2.0), (17, 4.0)):
        s, im = _sphere_session(nz=nz, zr=zr, r=20.0)
        m = extract_mesh(im, 0, 0, level=80)
        centre = np.array([32.5, 32.5, (nz // 2 + 0.5) * zr], np.float64)
        radial = m.vertices.astype(np.float64) - centre
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-3)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", m.normals.astype(np.float64), radial), -1, 1)))
        assert ang.mean() < 2.0 and ang.max() < 8.0, (zr, ang.mean(), ang.max())


def test_mesh_download_non_ascii_names(tmp_path):
    s, im = _sphere_session()
    im.display.channels[0].name = "α-tubulin"
    im2 = LumenImage(_sphere_volume(), name="細胞.tif", axes="ZYX")
    s.add_image(im2)
    c = TestClient(create_app(s))
    for fmt in ("obj", "stl", "ply"):
        r = c.get(f"/api/images/{im.id}/mesh.{fmt}", params={"level": 95})
        assert r.status_code == 200, r.text
        cd = r.headers["content-disposition"]
        assert cd.isascii()
        assert cd.startswith(f'attachment; filename="sphere__-tubulin.{fmt}"')
        assert f"filename*=UTF-8''sphere_%CE%B1-tubulin.{fmt}" in cd
        assert int(r.headers["x-vertices"]) > 0
    r = c.get(f"/api/images/{im.id}/mesh.obj", params={"level": 95})
    assert "o sphere_α-tubulin" in r.text  # the OBJ body keeps the Unicode name
    r = c.get(f"/api/images/{im2.id}/mesh.stl", params={"level": 95})
    assert r.status_code == 200 and r.headers["content-disposition"].isascii()
    assert "filename*=UTF-8''%E7%B4%B0%E8%83%9E_C1.stl" in r.headers["content-disposition"]
    r = c.get(f"/api/images/{im2.id}/mesh.ply", params={"level": 95})
    assert r.status_code == 200 and r.content.startswith(b"ply\n")
    # the PLY header is ASCII by specification; the export command works for every format (α-tubulin image active)
    s.select(im.id)
    im.display.c = 0
    for fmt in ("obj", "stl", "ply"):
        out = s.run_op("mesh.export", {"path": str(tmp_path / f"a.{fmt}"), "format": fmt, "level": 95})
        assert "Wrote" in out["message"] and (tmp_path / f"a.{fmt}").exists()
    nv, nf = _count_ply(tmp_path / "a.ply")
    assert nv > 0 and nf > 0


def test_mesh_level_zero_is_a_real_level():
    n = 48
    zz, yy, xx = np.mgrid[:n, :n, :n]
    vol = (12.0 - np.sqrt((zz - n // 2) ** 2 + (yy - n // 2) ** 2 + (xx - n // 2) ** 2)).astype(np.float32)  # zero crossing: r = 12
    s = Session()
    im = LumenImage(vol, name="signed.tif", axes="ZYX")
    im.display.channels[0].min, im.display.channels[0].max = -100.0, 100.0  # the viewer's slider midpoint is exactly 0
    s.add_image(im)
    c = TestClient(create_app(s))
    assert level_key(im, 0, 0) == (0.0,) and level_key(im, 0, "0.0000000") == (0.0,) and level_key(im, 0, "otsu") == ("otsu",)
    r = c.get(f"/api/images/{im.id}/mesh", params={"c": 0, "level": "0.0000000", "max_dim": 0})
    assert r.status_code == 200 and r.headers["x-level"] == "0"
    v, nrm, f = _parse_body(r)
    from lumen.core.ops.mesh_ops import Mesh
    rows = measure_components(Mesh(v.copy(), nrm.copy(), f.copy(), 0.0, (n, n, n)))
    assert len(rows) == 1 and abs(rows[0]["volume"] / (4 / 3 * math.pi * 12 ** 3) - 1) < 0.05
    r_otsu = c.get(f"/api/images/{im.id}/mesh", params={"c": 0, "level": "otsu", "max_dim": 0})
    assert r_otsu.status_code == 200 and float(r_otsu.headers["x-level"]) != 0.0
    assert c.get(f"/api/images/{im.id}/mesh", params={"c": 0, "level": "foo"}).status_code == 422
    assert c.get(f"/api/images/{im.id}/mesh.obj", params={"c": 0, "level": "foo"}).status_code == 422
    out = s.run_op("mesh.measure", {"level": 0})
    assert out["results_added"] == 1
    row = s.results.rows[0]
    assert row["Level"] == 0.0 and abs(row["Volume"] / (4 / 3 * math.pi * 12 ** 3) - 1) < 0.05
    s.results.clear()
    s.run_op("mesh.measure", {"level": "otsu"})
    assert s.results.rows[0]["Level"] != 0.0


def test_mesh_measure_hollow_sphere_subtracts_cavities():
    zz, yy, xx = np.mgrid[:64, :64, :64]
    rr = np.sqrt((zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 32) ** 2)
    hollow = np.where((rr >= 10) & (rr <= 20), 200.0, 0.0).astype(np.float32)
    s = Session()
    s.add_image(LumenImage(hollow, name="hollow.tif", axes="ZYX"))
    out = s.run_op("mesh.measure", {"level": 100})
    assert out["results_added"] == 2
    rows = s.results.rows
    assert "Orientation" in s.results.columns
    assert rows[0]["Orientation"] == "outward" and rows[1]["Orientation"] == "inward"
    assert rows[0]["Volume"] > rows[1]["Volume"] > 0  # per-row volumes stay unsigned, largest first
    material = 4 / 3 * math.pi * (20 ** 3 - 10 ** 3)
    msg = out["message"]
    total = float(msg.split("volume ")[1].split()[0])
    assert abs(total / material - 1) < 0.02, msg
    assert abs((rows[0]["Volume"] - rows[1]["Volume"]) / material - 1) < 0.02
    assert "1 cavity" in msg and "enclosed" in msg
    # solid objects keep the plain message
    s2, _ = _sphere_session()
    out2 = s2.run_op("mesh.measure", {"level": 80})
    assert "cavit" not in out2["message"] and s2.results.rows[0]["Orientation"] == "outward"


def test_mesh_rectangular_pixels_use_pixel_height():
    s, im = _sphere_session(r=20.0)
    im.calibration = Calibration(0.5, 1.0, 0.5, "um")  # rectangular pixels: the sphere is a 10 x 20 x 10 um ellipsoid
    out = s.run_op("mesh.measure", {"level": 80})
    row = s.results.rows[0]
    assert row["Unit"] == "um"
    assert abs(row["Volume"] / (4 / 3 * math.pi * 10 * 20 * 10) - 1) < 0.02
    assert abs(row["Width"] - 20) < 1 and abs(row["Height"] - 40) < 1 and abs(row["Depth"] - 20) < 1.5
    assert abs(row["X"] - 32.5 * 0.5) < 0.3 and abs(row["Y"] - 32.5) < 0.3 and abs(row["Z"] - 32.5 * 0.5) < 0.3
    # the exported geometry uses the same per-axis scale; normals are transformed with it (unit length, outward)
    c = TestClient(create_app(s))
    r = c.get(f"/api/images/{im.id}/mesh.obj", params={"level": 80, "calibrated": 1})
    assert r.status_code == 200 and r.headers["x-unit"] == "um"
    vs = np.array([[float(t) for t in ln.split()[1:]] for ln in r.text.splitlines() if ln.startswith("v ")])
    ns = np.array([[float(t) for t in ln.split()[1:]] for ln in r.text.splitlines() if ln.startswith("vn ")])
    fs = np.array([[int(t.split("//")[0]) - 1 for t in ln.split()[1:]] for ln in r.text.splitlines() if ln.startswith("f ")])
    assert abs(vs[:, 0].min() - 6.25) < 0.6 and abs(vs[:, 0].max() - 26.25) < 0.6
    assert abs(vs[:, 1].min() - 12.5) < 0.6 and abs(vs[:, 1].max() - 52.5) < 0.6
    assert abs(vs[:, 2].min() - 6.25) < 0.6 and abs(vs[:, 2].max() - 26.25) < 0.6
    assert np.allclose(np.linalg.norm(ns, axis=1), 1.0, atol=2e-3)
    geo = vertex_normals(vs.astype(np.float32), fs.astype(np.uint32))
    assert np.einsum("ij,ij->i", ns, geo).mean() > 0.99
    # the binary viewer endpoint stays in pixel space (the volume box the renderer draws)
    r = c.get(f"/api/images/{im.id}/mesh", params={"level": 80, "max_dim": 0})
    W, H, D = map(float, r.headers["x-extent"].split(","))
    assert (W, H, D) == (64.0, 64.0, 64.0)
    # a scalar scale still works for callers that pass one
    mesh = extract_mesh(im, 0, 0, level=80)
    rows = measure_components(mesh, scale=0.5)
    assert abs(rows[0]["volume"] / (4 / 3 * math.pi * 10 ** 3) - 1) < 0.02


def test_mesh_ops_registered_in_menu():
    from lumen.core.registry import REGISTRY
    exp, mea = REGISTRY.get("mesh.export"), REGISTRY.get("mesh.measure")
    assert exp.name == "Export Surface (OBJ/STL)" and exp.to_dict()["menu"] == "Plugins/3D" and exp.kind == "analysis"
    assert mea.to_dict()["menu"] == "Plugins/3D" and mea.kind == "analysis"
    names = [p.name for p in exp.params]
    for k in ("path", "format", "level", "step", "smooth", "channel"):
        assert k in names
    assert [p for p in exp.params if p.name == "format"][0].choices == ["obj", "stl", "ply"]
