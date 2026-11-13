"""3-D scenes (Plugins ▸ 3D Viewer): scene JSON round trip, landmarks → Results / ROIs, ray picking and animation assembly."""
import io
import json
import math

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops.mesh_ops import extract_mesh
from lumen.core.ops.scene_ops import (SETTINGS_KEY, assemble_animation, decode_frames, default_scene, first_voxel_above, frames_to_stack,
                                      landmark_distances, landmark_rois, landmark_rows, normalize_scene, pick_image, ray_box, ray_triangles,
                                      resolve_scene_images, scene_to_json, write_gif)
from lumen.core.session import Session
from lumen.server.app import create_app


def _sphere_volume(nz=48, n=48, zr=1.0):
    zz, yy, xx = np.mgrid[:nz, :n, :n]
    d = np.sqrt(((zz - nz // 2) * zr) ** 2 + (yy - n // 2) ** 2 + (xx - n // 2) ** 2).astype(np.float32)
    return 100.0 - d


def _sphere_session(nz=48, n=48, zr=1.0, unit="µm", pw=1.0):
    s = Session()
    im = LumenImage(_sphere_volume(nz, n, zr), name="sphere.tif", axes="ZYX")
    im.calibration = Calibration(pw, pw, pw * zr, unit)
    s.add_image(im)
    return s, im


def _png(w, h, color):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ scene document
def test_scene_normalize_and_json_round_trip(tmp_path):
    sc = default_scene()
    assert sc["version"] == 1 and sc["objects"] == [] and sc["camera"]["projection"] == "perspective"
    assert sc["clipping"] == {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]} and sc["background"] is None
    raw = {
        "objects": [{"id": "a", "image": "img1", "name": "cells.tif", "display": "Surface", "mode": "ISO", "color": "#ff8000", "transparency": 0.4,
                     "threshold": 2.0, "level": 0.6, "visible": False, "transform": {"translate": [1, 2, 3], "rotate": [0, 90, 0], "scale": 2},
                     "slices": {"pos": [0.1, 0.5, 0.9], "show": [1, 0, 1]}},
                    {"image": "img2", "display": "nonsense", "color": [255, 0, 0], "transform": "garbage"}],
        "camera": {"theta": 1.0, "phi": 9, "zoom": 2, "target": [1, 1, 1], "projection": "orthographic", "roll": 0.5},
        "lights": {"ambient": 0.5, "diffuse": 1.2, "specular": 0.1, "direction": [1, 0, 0], "headlight": False},
        "background": [0.1, 0.2, 0.3], "clipping": {"min": [0.2, 0, 0], "max": [0.8, 1, 2]},
        "landmarks": [{"name": "tip", "px": [10, 20, 3]}, {"x": 1, "y": 2, "z": 3}],
        "mode": "mesh",
    }
    sc = normalize_scene(raw)
    o = sc["objects"][0]
    assert o["display"] == "surface" and o["mode"] == "iso" and o["color"] == [1.0, 128 / 255, 0.0] and o["transparency"] == 0.4
    assert o["threshold"] == 1.0 and o["visible"] is False and o["transform"] == {"translate": [1.0, 2.0, 3.0], "rotate": [0.0, 90.0, 0.0], "scale": 2.0}
    assert o["slices"] == {"pos": [0.1, 0.5, 0.9], "show": [True, False, True]}
    o2 = sc["objects"][1]
    assert o2["id"] == "obj2" and o2["display"] == "volume" and o2["color"] == [1.0, 0.0, 0.0] and o2["transform"]["scale"] == 1.0
    assert sc["camera"]["phi"] == 1.5698 and sc["camera"]["projection"] == "orthographic" and sc["camera"]["roll"] == 0.5
    assert sc["lights"]["headlight"] is False and sc["background"] == [0.1, 0.2, 0.3]
    assert sc["clipping"]["max"] == [0.8, 1.0, 1.0]
    assert sc["landmarks"][0]["px"] == [10.0, 20.0, 3.0] and sc["landmarks"][1] == {"id": "lm2", "name": "L2", "image": "", "px": [1.0, 2.0, 3.0]}
    assert sc["mode"] == "mesh"
    # idempotent and stable through JSON text
    text = scene_to_json(sc)
    assert normalize_scene(text) == sc and normalize_scene(json.loads(text)) == sc
    for bad in ("{not json", "[1, 2]", 42):
        try:
            normalize_scene(bad)
            assert False, bad
        except ValueError:
            pass


def test_scene_save_load_ops_and_api(tmp_path):
    s, im = _sphere_session()
    im2 = LumenImage(_sphere_volume(), name="other.tif", axes="ZYX")
    s.add_image(im2)
    c = TestClient(create_app(s))
    # nothing published yet
    assert c.get("/api/scene").json() == {"scene": None, "rev": 0}
    try:
        s.run_op("scene.save", {"path": str(tmp_path / "x.json")})
        assert False
    except ValueError as e:
        assert "No 3-D scene" in str(e)
    scene = {"primary": im.id, "objects": [{"id": "o1", "image": im2.id, "display": "surface", "color": "#00ff00", "transform": {"translate": [5, 0, 0]}}],
             "camera": {"theta": 0.3}, "landmarks": [{"name": "A", "px": [1, 2, 3], "image": im.id}]}
    r = c.put("/api/scene", json={"scene": scene})
    assert r.status_code == 200 and r.json()["rev"] == 1 and r.json()["objects"] == 1
    got = c.get("/api/scene").json()
    assert got["rev"] == 1 and got["scene"]["objects"][0]["color"] == [0.0, 1.0, 0.0]
    # save writes UTF-8 JSON with the image names remembered; a directory path gets scene.json
    out = s.run_op("scene.save", {"path": str(tmp_path) + "/"})
    p = tmp_path / "scene.json"
    assert p.exists() and "Saved 3-D scene (1 object, 1 landmark)" in out["message"]
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["primary_name"] == "sphere.tif" and doc["objects"][0]["name"] == "other.tif" and doc["camera"]["theta"] == 0.3
    # load in a fresh session: ids differ, objects resolve by name; an unknown image is flagged
    doc["objects"].append({"id": "o2", "image": "nope", "name": "missing.tif"})
    # landmarks placed on the secondary object, on the missing object and on an image no object refers to
    doc["landmarks"] += [{"name": "B", "image": im2.id, "px": [4, 5, 6]}, {"name": "C", "image": "nope", "px": [1, 1, 1]},
                         {"name": "D", "image": "ghost", "px": [2, 2, 2]}]
    p2 = tmp_path / "edited.json"
    p2.write_text(json.dumps(doc), encoding="utf-8")
    s2 = Session()
    a = LumenImage(_sphere_volume(), name="sphere.tif", axes="ZYX")
    b = LumenImage(_sphere_volume(), name="other.tif", axes="ZYX")
    s2.add_image(a)
    s2.add_image(b)
    out = s2.run_op("scene.load", {"path": str(p2)})
    assert "2 objects" in out["message"] and "not open: missing.tif" in out["message"]
    tab = out["table"]
    assert tab["name"] == "Scene" and tab["columns"][0] == "Object" and [row[-1] for row in tab["rows"]] == ["ok", "missing"]
    sc = s2.settings[SETTINGS_KEY]
    assert sc["primary"] == a.id and sc["objects"][0]["image"] == b.id and sc["objects"][0]["missing"] is False
    assert sc["objects"][1]["missing"] is True and sc["landmarks"][0]["image"] == a.id
    # a landmark of a secondary object follows that object's new id (it used to be re-pointed at the primary); one on the
    # missing object keeps its stale id (the viewer skips it); one whose owner is unknown to the file falls back to the primary
    assert [lm["image"] for lm in sc["landmarks"][1:]] == [b.id, "nope", a.id]
    assert landmark_rows(a, sc["landmarks"], s2.images)[1]["Label"] == "other.tif"
    assert landmark_rois(a, sc["landmarks"], s2.images)[1]["image_id"] == b.id
    assert s2.settings["_scene_rev"] == 1
    c2 = TestClient(create_app(s2))
    assert c2.get("/api/scene").json()["rev"] == 1
    # the settings router never exposes the underscore keys
    assert "_scene" not in c2.get("/api/settings").json()
    # a saved scene passed explicitly as JSON text, without the viewer
    out = s2.run_op("scene.save", {"path": str(tmp_path / "explicit"), "scene": json.dumps({"objects": [], "background": "#000"})})
    assert (tmp_path / "explicit.json").exists() and json.loads((tmp_path / "explicit.json").read_text(encoding="utf-8"))["background"] == [0.0, 0.0, 0.0]
    try:
        s2.run_op("scene.load", {"path": str(tmp_path / "none.json")})
        assert False
    except FileNotFoundError:
        pass
    # resolve_scene_images keeps ids that are open
    sc3 = resolve_scene_images(normalize_scene({"objects": [{"image": b.id}]}), s2.images)
    assert sc3["objects"][0]["name"] == "other.tif" and not sc3["objects"][0]["missing"]


# ------------------------------------------------------------------ landmarks
def test_landmarks_to_results_rois_and_distances():
    s, im = _sphere_session(unit="µm", pw=0.5, zr=2.0)  # pixel 0.5 µm, slice 1 µm
    lms = [{"name": "A", "px": [10, 20, 3]}, {"name": "B", "px": [10, 20, 7]}, {"name": "C", "px": [16, 28, 3]}]
    rows = landmark_rows(im, [dict(lm, id=f"l{i}", image=im.id) for i, lm in enumerate(lms)])
    assert [r["Landmark"] for r in rows] == ["A", "B", "C"] and rows[0]["Unit"] == "µm"
    assert (rows[0]["X"], rows[0]["Y"], rows[0]["Z"]) == (5.0, 10.0, 3.0) and rows[0]["Slice"] == 4 and rows[0]["Xpx"] == 10.0
    names, mat = landmark_distances(im, [dict(lm, id=f"l{i}", image=im.id) for i, lm in enumerate(lms)])
    assert names == ["A", "B", "C"] and mat.shape == (3, 3) and np.allclose(mat, mat.T) and np.allclose(np.diag(mat), 0)
    assert abs(mat[0, 1] - 4.0) < 1e-9  # 4 slices × 1 µm
    assert abs(mat[0, 2] - 5.0) < 1e-9  # (6, 8) px × 0.5 µm = (3, 4) µm
    rois = landmark_rois(im, [dict(lm, id=f"l{i}", image=im.id) for i, lm in enumerate(lms)])
    assert rois[1]["type"] == "point" and rois[1]["points"] == [[10.0, 20.0]] and rois[1]["position"]["z"] == 7 and rois[1]["image_id"] == im.id
    # the command: Results rows, the distance table and ROI Manager entries with slice positions
    out = s.run_op("scene.landmarks_to_results", {"landmarks": json.dumps(lms)})
    assert out["results_added"] == 3 and out["rois_added"] == 3 and "3×3 distance table" in out["message"]
    assert s.results.rows[2]["Landmark"] == "C" and abs(s.results.rows[2]["X"] - 8.0) < 1e-9
    tab = out["table"]
    assert tab["name"] == "Landmark distances" and tab["columns"] == ["Landmark", "A", "B", "C"] and abs(tab["rows"][0][2] - 4.0) < 1e-9
    mgr = s.roi_manager.to_list()
    assert len(mgr) == 3 and mgr[0]["name"] == "A" and mgr[0]["position"] == {"t": 0, "z": 3, "c": 0} and mgr[1]["position"]["z"] == 7
    # without explicit landmarks the command uses the scene the viewer published; none → a clear error
    s.results.clear()
    s.roi_manager.clear()
    try:
        s.run_op("scene.landmarks_to_results", {})
        assert False
    except ValueError as e:
        assert "No landmarks" in str(e)
    c = TestClient(create_app(s))
    c.put("/api/scene", json={"scene": {"primary": im.id, "landmarks": [{"name": "P", "px": [1, 1, 1], "cal": [9, 9, 9]}]}})
    out = s.run_op("scene.landmarks_to_results", {"add_rois": False, "distances": True})
    assert out["results_added"] == 1 and "rois_added" not in out and out.get("table") is None
    row = s.results.rows[0]
    assert (row["X"], row["Y"], row["Z"]) == (9.0, 9.0, 9.0)  # a viewer-supplied calibrated position wins over px × voxel size
    # uncalibrated images report pixels
    s3 = Session()
    s3.add_image(LumenImage(_sphere_volume(), name="px.tif", axes="ZYX"))
    out = s3.run_op("scene.landmarks_to_results", {"landmarks": json.dumps(lms[:1])})
    assert s3.results.rows[0]["Unit"] == "pixel" and s3.results.rows[0]["X"] == 10.0
    try:
        s3.run_op("scene.landmarks_to_results", {"landmarks": "{bad"})
        assert False
    except ValueError:
        pass


def test_landmark_unit_follows_the_frame_of_the_coordinates():
    # the viewer sends `cal` = worldToCalibrated(primary, …), i.e. in the primary's unit, together with the id of the
    # scene object the point was placed on; the Unit column must follow the coordinates, not the object's calibration
    s, prim = _sphere_session(unit="µm", pw=0.5, zr=2.0)
    obj = LumenImage(_sphere_volume(), name="obj.tif", axes="ZYX")  # an uncalibrated mask added as a scene object
    s.add_image(obj, activate=False)
    images = dict(s.images)
    on_obj = {"name": "A", "image": obj.id, "px": [4, 4, 2], "cal": [2.0, 2.0, 2.0]}
    row = landmark_rows(prim, [on_obj], images)[0]
    assert row["Label"] == "obj.tif" and row["Unit"] == "µm" and (row["X"], row["Y"], row["Z"]) == (2.0, 2.0, 2.0)
    nm = LumenImage(_sphere_volume(), name="nm.tif", axes="ZYX")
    nm.calibration = Calibration(10.0, 10.0, 10.0, "nm")
    s.add_image(nm, activate=False)
    images = dict(s.images)
    assert landmark_rows(prim, [dict(on_obj, image=nm.id)], images)[0]["Unit"] == "µm"
    # the px × voxel fallback stays in the landmark image's own unit
    px_obj = {"name": "B", "image": obj.id, "px": [4, 0, 0]}
    assert landmark_rows(prim, [px_obj], images)[0]["Unit"] == "pixel"
    assert landmark_rows(prim, [dict(px_obj, image=nm.id)], images)[0]["Unit"] == "nm"
    # distances: a calibrated point and a px point on another object are in different frames → NaN, not a mixed number
    names, mat = landmark_distances(prim, [dict(on_obj, cal=[0.0, 0.0, 0.0]), px_obj], images)
    assert names == ["A", "B"] and np.isnan(mat[0, 1]) and np.isnan(mat[1, 0]) and mat[0, 0] == 0 and mat[1, 1] == 0
    # a calibrated point and a px point on the primary itself are both in the primary's frame
    names, mat = landmark_distances(prim, [{"name": "P", "image": prim.id, "px": [0, 0, 0], "cal": [0.0, 0.0, 0.0]},
                                           {"name": "Q", "image": prim.id, "px": [4, 0, 0]}], images)
    assert abs(mat[0, 1] - 2.0) < 1e-9  # 4 px × 0.5 µm
    # the command: unit per row, the message reports mixed units, the cross-frame cell is blank (None, JSON-safe)
    out = s.run_op("scene.landmarks_to_results", {"landmarks": json.dumps([on_obj, px_obj]), "add_rois": False}, image_id=prim.id)
    assert [r["Unit"] for r in s.results.rows] == ["µm", "pixel"] and "mixed: pixel, µm" in out["message"]
    assert out["table"]["rows"][0][2] is None and out["table"]["rows"][0][1] == 0.0 and "left blank" in out["message"]
    s.results.clear()
    out = s.run_op("scene.landmarks_to_results", {"landmarks": json.dumps([on_obj, dict(on_obj, name="A2", cal=[2.0, 5.0, 6.0])]), "add_rois": False},
                   image_id=prim.id)
    assert "(µm)" in out["message"] and "left blank" not in out["message"] and abs(out["table"]["rows"][0][2] - 5.0) < 1e-9
    # landmarks taken from the published scene: `cal` is in the scene's primary frame even when another image is active
    c = TestClient(create_app(s))
    c.put("/api/scene", json={"scene": {"primary": prim.id, "landmarks": [{"name": "S", "image": obj.id, "px": [1, 1, 1], "cal": [3, 3, 3]}]}})
    s.results.clear()
    out = s.run_op("scene.landmarks_to_results", {"add_rois": False}, image_id=obj.id)
    assert s.results.rows[0]["Unit"] == "µm" and "(µm)" in out["message"]


# ------------------------------------------------------------------ ray picking
def test_ray_helpers():
    # a single triangle in the z = 5 plane
    verts = np.array([[0, 0, 5], [10, 0, 5], [0, 10, 5]], np.float32)
    faces = np.array([[0, 1, 2]], np.uint32)
    hit = ray_triangles([2, 2, 0], [0, 0, 1], verts, faces)
    assert hit is not None and abs(hit["t"] - 5) < 1e-9 and np.allclose(hit["point"], [2, 2, 5]) and hit["face"] == 0
    assert abs(abs(hit["normal"][2]) - 1) < 1e-9
    assert ray_triangles([8, 8, 0], [0, 0, 1], verts, faces) is None  # outside the triangle
    assert ray_triangles([2, 2, 9], [0, 0, 1], verts, faces) is None  # behind the origin
    assert ray_triangles([2, 2, 0], [0, 0, 1], verts, np.zeros((0, 3), np.uint32)) is None
    assert ray_triangles([2, 2, 0], [0, 0, 0], verts, faces) is None
    # ray/box
    assert ray_box(np.array([-5.0, 5, 5]), np.array([1.0, 0, 0]), np.zeros(3), np.full(3, 10.0)) == (5.0, 15.0)
    assert ray_box(np.array([-5.0, 50, 5]), np.array([1.0, 0, 0]), np.zeros(3), np.full(3, 10.0)) is None
    assert ray_box(np.array([5.0, 5, 5]), np.array([1.0, 0, 0]), np.zeros(3), np.full(3, 10.0)) == (0.0, 5.0)
    # a sphere mesh (radius 20, centre 24.5 in the block-centre convention): a ray from outside hits at distance ≈ 20 from the centre
    im = LumenImage(_sphere_volume(), name="s.tif", axes="ZYX")
    mesh = extract_mesh(im, 0, 0, level=80)
    hit = ray_triangles([-30, 24.5, 24.5], [1, 0, 0], mesh.vertices, mesh.faces)
    assert hit is not None and abs(hit["point"][0] - (24.5 - 20)) < 1.0 and abs(hit["point"][1] - 24.5) < 0.6
    assert ray_triangles([-30, 24.5, 24.5], [-1, 0, 0], mesh.vertices, mesh.faces) is None
    # first voxel above a level along a ray through the raw volume (level 80 ↔ radius 20)
    vol = im.data[0, :, 0]
    h = first_voxel_above(vol, [-30, 24.5, 24.5], [1, 0, 0], 80.0)
    assert h is not None and abs(h["point"][0] - 4.5) < 1.0 and h["value"] >= 80 and h["voxel"][0] in (4, 5) and h["voxel"][1:] == [24, 24]
    assert first_voxel_above(vol, [-30, 24.5, 24.5], [1, 0, 0], 1e6) is None
    assert first_voxel_above(vol, [-30, 200, 24.5], [1, 0, 0], 80.0) is None
    # anisotropic z: the same sphere with 2× thinner slice count, coordinates in xy-pixel units
    im2 = LumenImage(_sphere_volume(nz=25, zr=2.0), name="aniso.tif", axes="ZYX")
    h2 = first_voxel_above(im2.data[0, :, 0], [24.5, 24.5, -10], [0, 0, 1], 80.0, zratio=2.0)
    assert h2 is not None and abs(h2["point"][2] - (12 * 2 + 1 - 20)) < 2.5


def test_pick_route_and_pick_image():
    s, im = _sphere_session(unit="µm", pw=0.5)
    c = TestClient(create_app(s))
    r = c.post(f"/api/images/{im.id}/pick", json={"origin": [-30, 24.5, 24.5], "direction": [1, 0, 0], "mode": "surface", "level": 80, "max_dim": 0})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["hit"] is True and j["mode"] == "surface" and abs(j["point"][0] - 4.5) < 1.0 and j["unit"] == "µm"
    assert abs(j["calibrated"][0] - j["point"][0] * 0.5) < 1e-6 and abs(j["px"][2] - j["point"][2]) < 1e-6 and j["level"] == 80
    r = c.post(f"/api/images/{im.id}/pick", json={"origin": [-30, 24.5, 24.5], "direction": [1, 0, 0], "mode": "volume", "level": 80})
    j2 = r.json()
    assert j2["hit"] is True and j2["mode"] == "volume" and abs(j2["point"][0] - j["point"][0]) < 1.5 and j2["value"] >= 80
    r = c.post(f"/api/images/{im.id}/pick", json={"origin": [-30, 200, 24.5], "direction": [1, 0, 0], "mode": "volume", "level": 80})
    assert r.json() == {"hit": False, "mode": "volume", "level": 80.0}
    assert c.post(f"/api/images/{im.id}/pick", json={"origin": [0, 0], "direction": [1, 0, 0]}).status_code == 400
    assert c.post(f"/api/images/{im.id}/pick", json={"origin": [0, 0, 0], "direction": [1, 0, 0], "c": 7}).status_code == 400
    # the auto level (display-range midpoint) is used when none is given
    j3 = c.post(f"/api/images/{im.id}/pick", json={"origin": [-30, 24.5, 24.5], "direction": [1, 0, 0], "mode": "volume"}).json()
    ch = im.display.channels[0]
    assert abs(j3["level"] - (ch.min + 0.5 * (ch.max - ch.min))) < 1e-6
    out = pick_image(im, [-30, 24.5, 24.5], [1, 0, 0], "surface", level=80, max_dim=0)
    assert out["hit"] and "face" in out and len(out["normal"]) == 3


# ------------------------------------------------------------------ animation assembly
def test_animation_frames_to_stack_and_gif(tmp_path):
    blobs = [_png(32, 24, (255, 0, 0)), _png(32, 24, (0, 255, 0)), _png(40, 20, (0, 0, 255))]  # the third frame has another size
    frames = decode_frames(blobs)
    assert len(frames) == 3 and all(f.shape == (24, 32, 3) for f in frames)
    assert frames[0][0, 0].tolist() == [255, 0, 0] and frames[2][0, 0].tolist() == [0, 0, 255] and frames[2][-1, 0].tolist() == [0, 0, 0]  # padded rows are black
    img = frames_to_stack(frames, "anim", fps=10)
    assert img.shape == (3, 1, 3, 24, 32) and img.rgb and img.dtype == np.uint8 and img.bit_depth == "RGB"
    assert abs(img.calibration.frame_interval - 0.1) < 1e-9 and img.calibration.time_unit == "sec"
    assert img.data[1, 0, 1].max() == 255 and img.data[1, 0, 0].max() == 0
    p = write_gif(frames, tmp_path / "a", fps=20)
    data = (tmp_path / "a.gif").read_bytes()
    assert p.endswith("a.gif") and data[:6] in (b"GIF89a", b"GIF87a")
    with Image.open(tmp_path / "a.gif") as g:
        assert g.n_frames == 3
        assert g.info.get("duration") == 50
    buf = io.BytesIO()
    write_gif(frames, buf, fps=5)
    assert buf.getvalue()[:6] == b"GIF89a"
    try:
        frames_to_stack([], "x")
        assert False
    except ValueError:
        pass


def test_animation_route(tmp_path):
    s, src = _sphere_session()  # the Z stack being recorded stays the active image: the 3-D View keeps showing it
    c = TestClient(create_app(s))
    files = [("frames", (f"f{i}.png", _png(48, 32, (i * 40, 100, 200 - i * 30)), "image/png")) for i in range(5)]
    r = c.post("/api/scene/animation", files=files, data={"name": "cells_turntable", "fps": "12"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["frames"] == 5 and (j["width"], j["height"]) == (48, 32) and j["fps"] == 12.0 and "path" not in j
    img = s.get_image(j["id"])
    assert img.name == "cells_turntable" and img.shape == (5, 1, 3, 32, 48) and img.rgb
    assert s.active_id == src.id and img.id in s.images  # opened beside the stack, not activated
    # with nothing open the recording still becomes the active image
    s_empty = Session()
    assemble_animation(s_empty, [_png(8, 8, (1, 2, 3))] * 2, "solo", 5)
    assert s_empty.active is not None and s_empty.active.name == "solo"
    assert img.data[3, 0, 0, 0, 0] == 120 and img.data[3, 0, 2, 0, 0] == 110
    # the assembled stack as an animated GIF download
    r = c.get(f"/api/images/{img.id}/animation.gif", params={"fps": 10})
    assert r.status_code == 200 and r.headers["content-type"] == "image/gif" and r.content[:6] == b"GIF89a"
    assert r.headers["content-disposition"] == 'attachment; filename="cells_turntable.gif"' and r.headers["x-frames"] == "5"
    with Image.open(io.BytesIO(r.content)) as g:
        assert g.n_frames == 5
    # format + path also writes the file (GIF through PIL, no optional extra needed)
    out = tmp_path / "movie"
    r = c.post("/api/scene/animation", files=files[:2], data={"name": "two", "fps": "4", "format": "gif", "path": str(out)})
    assert r.status_code == 200, r.text
    assert r.json()["path"].endswith("movie.gif") and (tmp_path / "movie.gif").exists() and "saved" in r.json()["message"]
    # a video format without the [video] extra (or an unknown one) is a clean error, and nothing is half-done
    n_before = len(s.images)
    r = c.post("/api/scene/animation", files=files[:2], data={"name": "v", "format": "xyz", "path": str(tmp_path / "v")})
    assert r.status_code == 400 and "Unknown animation format" in r.json()["error"]
    r = c.post("/api/scene/animation", files=files[:2], data={"name": "v", "format": "gif"})
    assert r.status_code == 400 and "output path" in r.json()["error"]
    assert len(s.images) == n_before + 2  # the stacks were added before the file step failed (the frames are not lost)
    # no frames → 400
    r = c.post("/api/scene/animation", files=[("frames", ("empty.png", b"", "image/png"))], data={"name": "x"})
    assert r.status_code == 400
    # direct helper with a video format that needs the extra: an ImportError/RuntimeError hint or success when installed
    try:
        assemble_animation(s, [_png(8, 8, (1, 2, 3))] * 2, "v", 5, "mp4", str(tmp_path / "v.mp4"))
    except RuntimeError as e:
        assert "imageio-ffmpeg" in str(e)


def test_scene_ops_registered():
    from lumen.core.registry import REGISTRY
    for oid in ("scene.save", "scene.load", "scene.landmarks_to_results"):
        spec = REGISTRY.get(oid)
        assert spec.kind == "analysis" and spec.to_dict()["menu"] == "Plugins/3D" and spec.description and spec.keywords
    assert REGISTRY.get("scene.save").requires_image is False and REGISTRY.get("scene.landmarks_to_results").requires_image is True
    assert [p.name for p in REGISTRY.get("scene.landmarks_to_results").params] == ["landmarks", "add_rois", "distances"]
