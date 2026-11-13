"""Plugins ▸ Segmentation (3-D): voxel classifier, 3-D label tools, distance / seeded watershed, random walker."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops import segmentation3d as s3d
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

OPS_3D = ["segment3d.train_classifier", "segment3d.apply_classifier", "segment3d.save_classifier", "segment3d.load_classifier",
          "segment3d.label_3d", "segment3d.measure_labels_3d", "segment3d.filter_labels_3d", "segment3d.watershed_3d",
          "segment3d.expand_labels_3d", "segment3d.labels_to_rois_3d", "segment3d.seeded_watershed_3d", "segment3d.random_walker_3d"]

SPHERES = {"shape": (24, 64, 64), "centers": ((12, 20, 20), (12, 44, 44)), "radii": (6, 8)}


# ------------------------------------------------------------------ helpers
def _spheres(shape=SPHERES["shape"], centers=SPHERES["centers"], radii=SPHERES["radii"], spacing=(1.0, 1.0, 1.0)):
    """Boolean volume with spheres (centres in voxels, radii in calibrated units, spacing (z, y, x))."""
    zz, yy, xx = np.indices(shape)
    m = np.zeros(shape, bool)
    for (cz, cy, cx), r in zip(centers, radii):
        m |= ((zz - cz) * spacing[0]) ** 2 + ((yy - cy) * spacing[1]) ** 2 + ((xx - cx) * spacing[2]) ** 2 <= r * r
    return m


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


def _new_images(session, out):
    return [session.images[i] for i in out["new_images"]]


def _binary_session(mask, name="spheres", calibration=None):
    s = Session()
    img = s.add_image(LumenImage(mask.astype(np.uint8) * 255, name=name, calibration=calibration))
    return s, img


def _bright_sphere(seed=0, shape=(20, 48, 48), radius=10):
    """Noisy uint8 volume with one bright sphere in the middle plus the true mask."""
    rng = np.random.default_rng(seed)
    sph = _spheres(shape, ((shape[0] // 2, shape[1] // 2, shape[2] // 2),), (radius,))
    vol = ndi.gaussian_filter(np.where(sph, 200.0, 50.0) + rng.normal(0, 20, shape), 0.5)
    return np.clip(vol, 0, 255).astype(np.uint8), sph


def _add_class_rects(session, img):
    """A foreground rectangle inside the sphere on the middle slice and a background one in a corner of another slice."""
    session.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 8, "h": 8, "name": "foreground 1"}, image_id=img.id, position={"t": 0, "z": 10, "c": 0})
    session.roi_manager.add({"type": "rect", "x": 2, "y": 2, "w": 10, "h": 10, "name": "background 1"}, image_id=img.id, position={"t": 0, "z": 3, "c": 0})


# ------------------------------------------------------------------ registry
def test_segmentation3d_ops_registered():
    for op_id in OPS_3D:
        spec = REGISTRY.get(op_id)
        assert spec.category == "Plugins/Segmentation", op_id
        assert spec.to_dict()["menu"] == "Plugins/Segmentation"
        assert spec.supports_rgb is (op_id in ("segment3d.train_classifier", "segment3d.apply_classifier") or spec.requires_image is False)
    assert REGISTRY.get("segment3d.expand_labels_3d").kind == "image"
    assert all(REGISTRY.get(o).kind == "analysis" for o in OPS_3D if o != "segment3d.expand_labels_3d")


# --------------------------------------------------------------- label 3-D
def test_label_3d_two_spheres_and_volumes():
    mask = _spheres()
    s, img = _binary_session(mask)
    out = s.run_op("segment3d.label_3d", {})
    lab = _new_images(s, out)[0]
    assert lab.shape == (1, 24, 1, 64, 64) and lab.dtype == np.uint8 and lab.display.channels[0].lut == "Glasbey on dark"
    labels = lab.data[0, :, 0]
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert out["message"].startswith("2 objects") and s.results.rows[-1] == {"Label": "spheres", "Count": 2}
    counts = sorted(np.bincount(labels.ravel())[1:].tolist())
    for n, r in zip(counts, sorted(SPHERES["radii"])):
        assert abs(n - 4 / 3 * np.pi * r ** 3) / (4 / 3 * np.pi * r ** 3) < 0.05
    assert _iou(labels > 0, mask) == 1.0


def test_label_3d_connectivity_and_threshold_input():
    m = np.zeros((4, 8, 8), bool)
    m[1, 2, 2] = True
    m[2, 3, 3] = True  # touches the first voxel only at a corner
    s, img = _binary_session(m, "diag")
    assert int(_new_images(s, s.run_op("segment3d.label_3d", {"connectivity": "26"}))[0].data.max()) == 1
    assert int(_new_images(s, s.run_op("segment3d.label_3d", {"connectivity": "6"}, image_id=img.id))[0].data.max()) == 2
    grey = s.add_image(LumenImage((_spheres() * 180 + 20).astype(np.uint8), name="grey"))
    with pytest.raises(ValueError, match="binary"):
        s.run_op("segment3d.label_3d", {})
    s.run_op("threshold.set", {"lower": 100, "upper": 255}, image_id=grey.id)
    out = s.run_op("segment3d.label_3d", {}, image_id=grey.id)
    assert int(_new_images(s, out)[0].data.max()) == 2
    # an area selection limits the labelling to its x/y footprint on every slice
    out = s.run_op("segment3d.label_3d", {}, image_id=grey.id, roi={"type": "rect", "x": 0, "y": 0, "w": 32, "h": 64})
    assert int(_new_images(s, out)[0].data.max()) == 1


# ----------------------------------------------------------- measure labels
def test_measure_labels_3d_volume_surface_intensity():
    mask = _spheres()
    s, img = _binary_session(mask)
    inten = s.add_image(LumenImage(np.where(mask, 200, 10).astype(np.uint8) + np.indices(mask.shape)[0].astype(np.uint8), name="inten"))
    out = s.run_op("segment3d.measure_labels_3d", {"intensity_image": inten.id}, image_id=img.id)
    rows = s.results.rows
    assert len(rows) == 2 and out["message"] == "2 objects measured"
    assert out["table"]["name"] == "Labels 3D" and "Volume (pixel³)" in out["table"]["columns"] and "Surface (pixel²)" in out["table"]["columns"]
    big = max(rows, key=lambda r: r["Voxels"])
    assert big["ID"] in (1, 2) and big["Voxels"] == int(mask.sum()) - min(r["Voxels"] for r in rows)
    assert abs(big["Volume"] - 4 / 3 * np.pi * 8 ** 3) / (4 / 3 * np.pi * 8 ** 3) < 0.05
    assert abs(big["X"] - 44.5) < 0.1 and abs(big["Y"] - 44.5) < 0.1 and abs(big["Z"] - 12.5) < 0.1
    assert big["BX"] == 36 and big["BZ"] == 4 and big["Width"] == 17 and big["Depth"] == 17
    assert abs(big["Surface"] - 4 * np.pi * 64) / (4 * np.pi * 64) < 0.15 and 0.85 < big["Sphericity"] <= 1.0
    assert 200 <= big["Mean"] <= 225 and big["Min"] >= 200 and big["Max"] <= 225 and big["StdDev"] > 0
    out = s.run_op("segment3d.measure_labels_3d", {"surface": "Crofton (3 directions)", "min_voxels": 1000, "clear_results": True}, image_id=img.id)
    assert len(s.results.rows) == 1 and s.results.rows[0]["Voxels"] == big["Voxels"]
    assert abs(s.results.rows[0]["Surface"] - 4 * np.pi * 64) / (4 * np.pi * 64) < 0.1
    assert "Mean" not in s.results.rows[0]


def test_measure_labels_3d_calibrated_anisotropic():
    cal = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=1.0, unit="um")
    mask = _spheres((20, 40, 40), ((10, 20, 20),), (4.0,), spacing=(1.0, 0.5, 0.5))  # radius 4 um
    s, img = _binary_session(mask, "aniso", calibration=cal)
    out = s.run_op("segment3d.measure_labels_3d", {})
    r = s.results.rows[-1]
    assert out["table"]["columns"][3] == "Volume (um³)"
    assert abs(r["Volume"] - 4 / 3 * np.pi * 64) / (4 / 3 * np.pi * 64) < 0.05
    assert abs(r["Surface"] - 4 * np.pi * 16) / (4 * np.pi * 16) < 0.15 and r["Sphericity"] > 0.85
    assert abs(r["X"] - 10.25) < 0.01 and abs(r["Z"] - 10.5) < 0.01 and r["Depth"] == 9.0 and r["Width"] == 8.5
    s2 = Session()
    s2.add_image(LumenImage(np.zeros((4, 8, 8), np.uint8), name="empty"))
    with pytest.raises(ValueError, match="No objects"):
        s2.run_op("segment3d.measure_labels_3d", {})


# ------------------------------------------------------------ filter labels
def test_filter_labels_3d():
    mask = _spheres((24, 64, 64), ((12, 20, 20), (12, 44, 44), (3, 60, 60)), (6, 8, 5))
    s, img = _binary_session(mask, "three")
    lab = _new_images(s, s.run_op("segment3d.label_3d", {}))[0]
    assert int(lab.data.max()) == 3
    out = s.run_op("segment3d.filter_labels_3d", {"min_volume": 1000}, image_id=lab.id)
    f = _new_images(s, out)[0]
    assert out["message"] == "Kept 1 of 3 objects" and set(np.unique(f.data).tolist()) == {0, 1}
    assert int((f.data > 0).sum()) == max(np.bincount(lab.data.ravel())[1:])
    out = s.run_op("segment3d.filter_labels_3d", {"max_volume": 1000}, image_id=lab.id)
    assert out["message"] == "Kept 2 of 3 objects"  # the r = 6 sphere (925 voxels) and the clipped corner sphere
    out = s.run_op("segment3d.filter_labels_3d", {"exclude_edges": "X/Y edges"}, image_id=lab.id)
    assert out["message"] == "Kept 2 of 3 objects" and int(_new_images(s, out)[0].data.max()) == 2
    flat = np.zeros((24, 64, 64), bool)
    flat[0:3, 30:34, 30:34] = True  # touches z = 0 only
    s2, img2 = _binary_session(mask | flat, "zedge")
    lab2 = _new_images(s2, s2.run_op("segment3d.label_3d", {}))[0]
    assert s2.run_op("segment3d.filter_labels_3d", {"exclude_edges": "X/Y edges"}, image_id=lab2.id)["message"] == "Kept 3 of 4 objects"
    assert s2.run_op("segment3d.filter_labels_3d", {"exclude_edges": "All edges (incl. Z)"}, image_id=lab2.id)["message"] == "Kept 2 of 4 objects"
    inten = s.add_image(LumenImage(np.where(_spheres((24, 64, 64), ((12, 44, 44),), (8,)), 200, 20).astype(np.uint8), name="inten"))
    out = s.run_op("segment3d.filter_labels_3d", {"intensity_image": inten.id, "min_mean": 100}, image_id=lab.id)
    assert out["message"] == "Kept 1 of 3 objects" and s.results.rows[-1]["Count"] == 1


# ------------------------------------------------------- distance watershed
def test_watershed_3d_splits_touching_spheres():
    mask = _spheres((24, 48, 48), ((12, 24, 16), (12, 24, 30)), (8, 8))
    assert ndi.label(mask)[1] == 1  # the spheres overlap: one component
    s, img = _binary_session(mask, "touching")
    out = s.run_op("segment3d.watershed_3d", {"tolerance": 1.0})
    lab = _new_images(s, out)[0]
    labels = lab.data[0, :, 0]
    assert out["message"].startswith("2 objects") and set(np.unique(labels).tolist()) == {0, 1, 2}
    counts = np.bincount(labels.ravel())[1:]
    assert abs(counts[0] - counts[1]) / counts.sum() < 0.1 and _iou(labels > 0, mask) == 1.0
    assert s.results.rows[-1] == {"Label": "touching", "Count": 2}
    out = s.run_op("segment3d.watershed_3d", {"output": "Binary (split)"}, image_id=img.id)
    b = _new_images(s, out)[0]
    assert b.is_binary and 0 < int(mask.sum()) - int((b.data > 0).sum()) < 200
    assert ndi.label(b.data[0, :, 0] > 0)[1] == 2
    # a huge tolerance keeps the component in one piece
    out = s.run_op("segment3d.watershed_3d", {"tolerance": 50}, image_id=img.id)
    assert int(_new_images(s, out)[0].data.max()) == 1


def test_watershed_3d_anisotropic_split_along_z():
    cal = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=1.0, unit="um")
    mask = _spheres((32, 32, 32), ((12, 16, 16), (19, 16, 16)), (4.0, 4.0), spacing=(1.0, 0.5, 0.5))  # 4 um = 4 slices = 8 px
    assert ndi.label(mask)[1] == 1
    s, img = _binary_session(mask, "aniso", calibration=cal)
    out = s.run_op("segment3d.watershed_3d", {"tolerance": 1.0})
    labels = _new_images(s, out)[0].data[0, :, 0]
    counts = np.bincount(labels.ravel())[1:]
    assert len(counts) == 2 and abs(counts[0] - counts[1]) / counts.sum() < 0.1


# ------------------------------------------------------------ expand labels
def test_expand_labels_3d():
    mask = _spheres()
    s, img = _binary_session(mask)
    lab = _new_images(s, s.run_op("segment3d.label_3d", {}))[0]
    before = int((lab.data > 0).sum())
    out = s.run_op("segment3d.expand_labels_3d", {"distance": 2}, image_id=lab.id)
    assert "expanded" in out["message"] and lab.dtype == np.uint8
    assert set(np.unique(lab.data).tolist()) == {0, 1, 2} and before < int((lab.data > 0).sum()) < 2.5 * before
    assert np.all(lab.data[lab.data > 0] == ndi.grey_dilation(lab.data, size=(1, 1, 1, 1, 1))[lab.data > 0])
    assert s.histories[lab.id].labels()["undo"]  # in-place edit with undo
    cal = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=1.0, unit="um")
    s2, img2 = _binary_session(_spheres((20, 40, 40), ((10, 20, 20),), (4.0,), spacing=(1.0, 0.5, 0.5)), "aniso", calibration=cal)
    lab2 = _new_images(s2, s2.run_op("segment3d.label_3d", {}))[0]
    sl0 = ndi.find_objects(lab2.data[0, :, 0].astype(np.int32))[0]
    s2.run_op("segment3d.expand_labels_3d", {"distance": 4}, image_id=lab2.id)
    sl1 = ndi.find_objects(lab2.data[0, :, 0].astype(np.int32))[0]
    assert sl1[0].start == sl0[0].start - 2 and sl1[0].stop == sl0[0].stop + 2  # z grows by 4 px / (z spacing 2)
    assert sl1[1].start == sl0[1].start - 4 and sl1[2].stop == sl0[2].stop + 4


# ---------------------------------------------------------- labels → ROIs
def test_labels_to_rois_3d_counts_positions_and_limit():
    mask = _spheres()
    s, img = _binary_session(mask)
    lab = _new_images(s, s.run_op("segment3d.label_3d", {}))[0]
    out = s.run_op("segment3d.labels_to_rois_3d", {}, image_id=lab.id)
    rois = s.roi_manager.to_list()
    expected = sum(2 * r + 1 for r in SPHERES["radii"])  # one outline per slice an object occupies
    assert len(rois) == expected and out["message"].startswith(f"{expected} ROIs (2 objects)")
    assert s.results.rows[-1]["ROIs"] == expected and s.results.rows[-1]["Objects"] == 2
    assert all(r["image_id"] == lab.id and r["position"]["t"] == 0 and r["position"]["c"] == 0 for r in rois)
    assert all(r["type"] in ("polygon", "mask") for r in rois)
    first = rois[0]
    assert first["name"] == f"L1 z{first['position']['z'] + 1}"
    by_label = {}
    for r in rois:
        by_label.setdefault(r["name"].split(" ")[0], set()).add(r["position"]["z"])
    assert sorted(len(v) for v in by_label.values()) == [13, 17]
    out = s.run_op("segment3d.labels_to_rois_3d", {"max_rois": 5, "clear_manager": True}, image_id=lab.id)
    assert len(s.roi_manager.rois) == 5 and "stopped at the limit" in out["message"]
    out = s.run_op("segment3d.labels_to_rois_3d", {"min_voxels": 1500, "clear_manager": True}, image_id=lab.id)
    assert len(s.roi_manager.rois) == 17 and all(r["name"].startswith("L") for r in s.roi_manager.rois)
    out = s.run_op("segment3d.labels_to_rois_3d", {"slice_step": 4, "clear_manager": True}, image_id=lab.id)
    assert 0 < len(s.roi_manager.rois) < expected / 3


# --------------------------------------------------------- 3-D classifier
def test_train_classifier3d_separates_bright_sphere():
    data, sph = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(data, name="sphere"))
    with pytest.raises(ValueError, match="No training ROIs"):
        s.run_op("segment3d.train_classifier", {"n_trees": 5})
    _add_class_rects(s, img)
    out = s.run_op("segment3d.train_classifier", {"n_trees": 20, "max_sigma": 2})
    row = s.results.rows[-1]
    assert row["Label"].startswith("3D classifier") and row["Classes"] == 2 and row["Training slices"] == 2 and "OOB accuracy" in row
    assert row["Voxels 'foreground'"] == 64 and row["Voxels 'background'"] == 100
    model = s.settings["classifier3d"]
    assert model["classes"] == ["background", "foreground"] and model["kind"] == "classifier3d"
    assert any(n.startswith("Hessian3D") for n in model["feature_names"]) and model["settings"]["sigmas"] == [1.0, 2.0]
    probs, labels = _new_images(s, out)
    assert probs.shape == (1, 20, 2, 48, 48) and probs.dtype == np.float32 and [c.name for c in probs.display.channels] == ["background", "foreground"]
    assert np.allclose(probs.data.sum(axis=2), 1.0, atol=1e-4)
    assert labels.shape == (1, 20, 1, 48, 48) and labels.dtype == np.uint8 and labels.display.channels[0].lut == "Glasbey on dark"
    assert labels.metadata["classes"] == "background, foreground"
    assert _iou(labels.data[0, :, 0] == 1, sph) > 0.8


def test_apply_classifier3d_chunked_mask_and_errors():
    data, sph = _bright_sphere()
    s = Session()
    with pytest.raises(ValueError, match="No trained 3-D pixel classifier"):
        s.add_image(LumenImage(data, name="sphere"))
        s.run_op("segment3d.apply_classifier", {})
    img = s.active
    _add_class_rects(s, img)
    s.run_op("segment3d.train_classifier", {"n_trees": 20, "max_sigma": 2, "apply_now": False})
    assert len(s.images) == 1
    model = s.settings["classifier3d"]
    assert s3d._chunk_slices(img, model["settings"], model["n_features"], 1.0) == 1  # one slice per chunk at a 1 MB budget
    whole = s.run_op("segment3d.apply_classifier", {"output": "Probabilities", "max_memory_mb": 512}, image_id=img.id)
    chunked = s.run_op("segment3d.apply_classifier", {"output": "Both", "mask_class": "foreground", "max_memory_mb": 1}, image_id=img.id)
    p_whole = _new_images(s, whole)[0]
    p_chunk, labels, mask = _new_images(s, chunked)
    assert np.array_equal(p_whole.data, p_chunk.data)  # the halo makes chunked features identical to whole-volume ones
    assert labels.name == "Classification of sphere" and mask.name == "foreground mask of sphere" and mask.is_binary
    assert _iou(mask.data > 0, labels.data == 1) > 0.95 and _iou(labels.data[0, :, 0] == 1, sph) > 0.8
    with pytest.raises(ValueError, match="Unknown class"):
        s.run_op("segment3d.apply_classifier", {"mask_class": "nucleus"}, image_id=img.id)
    # a second image with another number of channels yields a different feature count
    two = s.add_image(LumenImage(np.stack([data, data], axis=1), name="two channels", axes="ZCYX"))
    s.roi_manager.clear()
    s.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 8, "h": 8, "name": "foreground"}, image_id=two.id, position={"t": 0, "z": 10, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 2, "y": 2, "w": 6, "h": 6, "name": "background"}, image_id=two.id, position={"t": 0, "z": 2, "c": 0})
    s.run_op("segment3d.train_classifier", {"n_trees": 5, "max_sigma": 1, "channels": "All channels", "apply_now": False}, image_id=two.id)
    assert s.settings["classifier3d"]["feature_names"][0] == "C1_Original"
    with pytest.raises(ValueError, match="expects"):
        s.run_op("segment3d.apply_classifier", {}, image_id=img.id)


def test_classifier3d_multi_frame_and_smooth():
    data, sph = _bright_sphere()
    stack = np.stack([data, data])[:, :, None]  # (T, Z, C, Y, X)
    s = Session()
    img = s.add_image(LumenImage(stack, name="frames"))
    s.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 8, "h": 8, "name": "foreground"}, position={"t": 1, "z": 10, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 2, "y": 2, "w": 10, "h": 10, "name": "background"}, position={"t": 0, "z": 3, "c": 0})
    s.run_op("segment3d.train_classifier", {"n_trees": 10, "max_sigma": 1, "apply_now": False})
    out = s.run_op("segment3d.apply_classifier", {"all_frames": True, "output": "Labels", "smooth": 1.0}, image_id=img.id)
    labels = _new_images(s, out)[0]
    assert labels.shape == (2, 20, 1, 48, 48)
    assert np.array_equal(labels.data[0], labels.data[1]) and _iou(labels.data[1, :, 0] == 1, sph) > 0.8


def test_classifier3d_overlay_of_one_frame_does_not_paint_the_other_frames(monkeypatch):
    """Overlay output with all_frames=False on a time series: the overlay image holds only the classified frame and is
    pinned to it (z free), so the other frames show no overlay – not the first class colour over every voxel."""
    from lumen.core.ops import segmentation as seg
    data, sph = _bright_sphere()
    stack = np.stack([data, data, data])[:, :, None]  # (T=3, Z, C, Y, X)
    s = Session()
    img = s.add_image(LumenImage(stack, name="frames"))
    img.display.t = 1
    s.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 8, "h": 8, "name": "foreground"}, position={"t": 1, "z": 10, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 2, "y": 2, "w": 10, "h": 10, "name": "background"}, position={"t": 1, "z": 3, "c": 0})
    calls = []

    def attach(session, image, labels, names, colors, all_planes, opacity, suffix="", pin_t=None):
        """What segmentation._attach_overlay(pin_t=…) does once wired: frame pinned, z free (-1), item position t only."""
        calls.append((labels.shape, pin_t))
        ov = seg._attach_overlay(session, image, labels, names, colors, all_planes, opacity, suffix=suffix)
        if pin_t is not None:
            ov.metadata["classifier_overlay_t"], ov.metadata["classifier_overlay_z"] = int(pin_t), -1
            for o in image.overlay:
                if o.get("source") == seg.OVERLAY_SOURCE:
                    o["position"] = {"t": int(pin_t), "z": None, "c": None}
        return ov

    monkeypatch.setattr(s3d, "_attach_overlay", attach)
    assert s3d._overlay_pins_frames()
    s.run_op("segment3d.train_classifier", {"n_trees": 10, "max_sigma": 1, "apply_now": True, "output": "Overlay", "class_colors": "#ff0000,#00ff00"})
    ov = seg._find_overlay_image(s, img)
    item = next(o for o in img.overlay if o.get("source") == "classifier")
    assert calls[-1] == ((1, 20, 1, 48, 48), 1) and ov.shape == (1, 20, 3, 48, 48)
    assert item["position"] == {"t": 1, "z": None, "c": None} and ov.metadata["classifier_overlay_t"] == 1 and ov.metadata["classifier_overlay_z"] == -1
    assert set(map(tuple, ov.data[0, 10, :, 24, 24][None])) == {(0, 255, 0)}  # the sphere centre on the classified frame
    # all frames classified: full series, no pin; a single-frame image: no pin either
    s.select(img.id)
    s.run_op("segment3d.apply_classifier", {"all_frames": True, "output": "Overlay"})
    assert calls[-1] == ((3, 20, 1, 48, 48), None) and seg._find_overlay_image(s, img).shape == (3, 20, 3, 48, 48)
    assert "position" not in next(o for o in img.overlay if o.get("source") == "classifier")
    one = s.add_image(LumenImage(data, name="one"))
    s.select(one.id)
    s.run_op("segment3d.apply_classifier", {"output": "Overlay"})
    assert calls[-1][1] is None and seg._find_overlay_image(s, one).shape == (1, 20, 3, 48, 48)


def test_classifier3d_overlay_frame_pin_matches_the_helper_in_segmentation():
    """With the real helper: once segmentation._attach_overlay accepts pin_t the overlay is a pinned single frame;
    until then the legacy zero-padded series (no position) is kept so the overlay still follows z scrolling."""
    from lumen.core.ops import segmentation as seg
    data, _ = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(np.stack([data, data, data])[:, :, None], name="frames"))
    img.display.t = 1
    s.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 8, "h": 8, "name": "foreground"}, position={"t": 1, "z": 10, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 2, "y": 2, "w": 10, "h": 10, "name": "background"}, position={"t": 1, "z": 3, "c": 0})
    s.run_op("segment3d.train_classifier", {"n_trees": 10, "max_sigma": 1, "apply_now": True, "output": "Overlay"})
    ov = seg._find_overlay_image(s, img)
    item = next(o for o in img.overlay if o.get("source") == "classifier")
    if s3d._overlay_pins_frames():
        assert ov.shape == (1, 20, 3, 48, 48) and item["position"] == {"t": 1, "z": None, "c": None}
        s.select(img.id)
        s.run_op("segment.classifier_overlay", {"action": "Hide"})
        s.run_op("segment.classifier_overlay", {"action": "Show"})
        assert next(o for o in img.overlay if o.get("source") == "classifier")["position"] == {"t": 1, "z": None, "c": None}
    else:
        assert ov.shape == (3, 20, 3, 48, 48) and "position" not in item


def test_save_load_classifier3d_roundtrip(tmp_path):
    data, sph = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(data, name="sphere"))
    _add_class_rects(s, img)
    out = s.run_op("segment3d.train_classifier", {"n_trees": 10, "max_sigma": 1, "output": "Labels"})
    first = _new_images(s, out)[0].data.copy()
    path = tmp_path / "sphere.joblib"
    s.run_op("segment3d.save_classifier", {"path": str(path)})
    assert path.exists()
    s.run_op("segment3d.save_classifier", {"path": str(tmp_path)})
    assert (tmp_path / "classifier3d.joblib").exists()
    s2 = Session()
    s2.add_image(LumenImage(data, name="sphere"))
    out = s2.run_op("segment3d.load_classifier", {"path": str(path)})
    assert "background, foreground" in out["message"] and s2.settings["classifier3d"]["trained_on"] == "sphere"
    out = s2.run_op("segment3d.apply_classifier", {"output": "Labels"})
    assert np.array_equal(_new_images(s2, out)[0].data, first)
    with pytest.raises(ValueError, match="No such file"):
        s2.run_op("segment3d.load_classifier", {"path": str(tmp_path / "missing.joblib")})
    import joblib
    joblib.dump({"model": None, "classes": ["a", "b"], "settings": {}}, tmp_path / "flat.joblib")
    with pytest.raises(ValueError, match="2-D pixel classifier"):
        s2.run_op("segment3d.load_classifier", {"path": str(tmp_path / "flat.joblib")})
    with pytest.raises(ValueError, match="No trained 3-D"):
        Session().run_op("segment3d.save_classifier", {"path": str(tmp_path / "none.joblib")})


# --------------------------------------------------------- seeded watershed
def _two_bright_spheres(seed=1):
    rng = np.random.default_rng(seed)
    mask = _spheres((24, 48, 48), ((12, 24, 16), (12, 24, 30)), (8, 8))
    vol = ndi.gaussian_filter(np.where(mask, 220.0, 30.0), 1.0) + rng.normal(0, 5, mask.shape)
    return np.clip(vol, 0, 255).astype(np.uint8), mask


def test_seeded_watershed_3d_from_point_rois():
    data, mask = _two_bright_spheres()
    s = Session()
    img = s.add_image(LumenImage(data, name="two"))
    s.run_op("threshold.set", {"lower": 120, "upper": 255})
    with pytest.raises(ValueError, match="point selection"):
        s.run_op("segment3d.seeded_watershed_3d", {})
    s.roi_manager.add({"type": "point", "points": [[16.5, 24.5]], "name": "seed 1"}, image_id=img.id, position={"t": 0, "z": 12, "c": 0})
    s.roi_manager.add({"type": "point", "points": [[30.5, 24.5]], "name": "seed 2"}, image_id=img.id, position={"t": 0, "z": 10, "c": 0})
    out = s.run_op("segment3d.seeded_watershed_3d", {"add_to_manager": True, "max_rois": 10})
    lab = _new_images(s, out)[0]
    labels = lab.data[0, :, 0]
    assert lab.shape == (1, 24, 1, 48, 48) and set(np.unique(labels).tolist()) == {0, 1, 2}
    counts = np.bincount(labels.ravel())[1:]
    assert abs(counts[0] - counts[1]) / counts.sum() < 0.1 and _iou(labels > 0, mask) > 0.9
    assert labels[12, 24, 16] == 1 and labels[10, 24, 30] == 2  # seed order = label order
    assert s.results.rows[-1] == {"Label": "two", "Count": 2}
    assert out["message"] == "2 objects, 10 ROIs added (limit reached)" and len(s.roi_manager.rois) == 12
    # the active point selection (current slice) takes precedence over the manager
    img.display.z = 12
    out = s.run_op("segment3d.seeded_watershed_3d", {}, image_id=img.id, roi={"type": "point", "points": [[16.5, 24.5], [30.5, 24.5], [23.5, 24.5]]})
    assert int(_new_images(s, out)[0].data.max()) == 3
    # without the threshold the flooding fills the whole volume
    out = s.run_op("segment3d.seeded_watershed_3d", {"use_threshold": False}, image_id=img.id)
    assert bool((_new_images(s, out)[0].data > 0).all())


def test_seeded_watershed_3d_local_maxima_and_markers_image():
    data, mask = _two_bright_spheres()
    s = Session()
    img = s.add_image(LumenImage(data, name="two"))
    s.run_op("threshold.set", {"lower": 120, "upper": 255})
    out = s.run_op("segment3d.seeded_watershed_3d", {"markers": "Local maxima", "smooth": 2.0, "prominence": 5})
    labels = _new_images(s, out)[0].data[0, :, 0]
    assert int(labels.max()) == 2 and _iou(labels > 0, mask) > 0.9
    with pytest.raises(ValueError, match="lower the prominence"):
        s.run_op("segment3d.seeded_watershed_3d", {"markers": "Local maxima", "smooth": 2.0, "prominence": 500}, image_id=img.id)
    mk = np.zeros(mask.shape, np.uint8)
    mk[12, 24, 16] = 5
    mk[12, 24, 30] = 9
    markers = s.add_image(LumenImage(mk, name="markers"))
    with pytest.raises(ValueError, match="markers image"):
        s.run_op("segment3d.seeded_watershed_3d", {"markers": "Markers image"}, image_id=img.id)
    out = s.run_op("segment3d.seeded_watershed_3d", {"markers": "Markers image", "markers_image": markers.id, "use_gradient": True}, image_id=img.id)
    labels = _new_images(s, out)[0].data[0, :, 0]
    assert int(labels.max()) == 2 and _iou(labels > 0, mask) > 0.85
    binary = s.add_image(LumenImage(mask.astype(np.uint8) * 255, name="mask"))
    out = s.run_op("segment3d.seeded_watershed_3d", {"markers": "Markers image", "markers_image": markers.id, "use_threshold": False, "mask_image": binary.id},
                   image_id=img.id)
    labels = _new_images(s, out)[0].data[0, :, 0]
    assert _iou(labels > 0, mask) == 1.0


# ----------------------------------------------------------- random walker
def test_random_walker_3d():
    data, sph = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(data, name="sphere"))
    with pytest.raises(ValueError, match="at least two classes"):
        s.run_op("segment3d.random_walker_3d", {})
    _add_class_rects(s, img)
    s.roi_manager.add({"type": "rect", "x": 36, "y": 36, "w": 10, "h": 10, "name": "background 2"}, image_id=img.id, position={"t": 0, "z": 10, "c": 0})
    out = s.run_op("segment3d.random_walker_3d", {"beta": 500, "output": "Labels and probabilities"})
    labels, probs = _new_images(s, out)
    assert out["message"] == "Random walker 3D: 2 classes"
    assert labels.shape == (1, 20, 1, 48, 48) and labels.display.channels[0].lut == "Glasbey on dark" and labels.metadata["classes"] == "background, foreground"
    assert set(np.unique(labels.data).tolist()) == {0, 1}
    assert probs.shape == (1, 20, 2, 48, 48) and [c.name for c in probs.display.channels] == ["background", "foreground"]
    assert np.allclose(probs.data.sum(axis=2), 1.0, atol=1e-3)
    assert _iou(labels.data[0, :, 0] == 1, sph) > 0.9
    out = s.run_op("segment3d.random_walker_3d", {"beta": 500, "downsample": 2}, image_id=img.id)
    small = _new_images(s, out)[0]
    assert "downsampled x2" in out["message"] and small.shape == labels.shape and _iou(small.data[0, :, 0] == 1, sph) > 0.8
    # an area selection restricts the result to its x/y footprint
    out = s.run_op("segment3d.random_walker_3d", {"beta": 500}, image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 48, "h": 24})
    half = _new_images(s, out)[0].data[0, :, 0]
    assert half[:, 24:].max() == 0 and half[:, :24].max() == 1


def test_random_walker_3d_markers_image_keeps_label_values():
    data, sph = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(data, name="sphere"))
    mk = np.zeros(sph.shape, np.uint8)
    mk[10, 22:26, 22:26] = 7
    mk[2, 2:8, 2:8] = 3
    mk[10, 40:46, 40:46] = 3
    markers = s.add_image(LumenImage(mk, name="markers"))
    out = s.run_op("segment3d.random_walker_3d", {"markers_image": markers.id, "beta": 500}, image_id=img.id)
    labels = _new_images(s, out)[0]
    assert set(np.unique(labels.data).tolist()) == {3, 7} and _iou(labels.data[0, :, 0] == 7, sph) > 0.9
    full = s.add_image(LumenImage(np.where(sph, 2, 1).astype(np.uint8), name="all seeds"))
    with pytest.raises(ValueError, match="nothing to segment"):
        s.run_op("segment3d.random_walker_3d", {"markers_image": full.id}, image_id=img.id)
