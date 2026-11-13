"""Regression tests for lumen/core/ops/stacks_extra.py (3D Project, Orthogonal Views, Reduce)."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.session import Session


def _session_with(vol, axes="ZYX", cal=None):
    s = Session()
    im = LumenImage(vol, name="v.tif", axes=axes)
    if cal is not None:
        im.calibration = cal
    s.add_image(im)
    return s, im


def _project(s, **params):
    out = s.run_op("stacks.project3d", params)
    return s.images[out["new_images"][0]], out


# --- 3D Project: rotation sense matches ImageJ's Projector (X and Z were mirrored) -----------------------------------

def _front_voxel_at_90(axis):
    vol = np.zeros((8, 32, 32), np.uint8)
    vol[1, 16, 5] = 255   # front slice, left of centre, centre row
    vol[6, 10, 26] = 200  # back slice, upper right (asymmetric volume)
    s, _ = _session_with(vol)
    proj, _ = _project(s, axis=axis, angle_step=90, total_rotation=360, interpolate=False, interior_depth_cueing=0)
    p = proj.data[1, 0, 0]  # the +90 deg frame
    ys, xs = np.nonzero(p == 255)
    H, W = p.shape
    return xs[0] - (W - 1) / 2, ys[0] - (H - 1) / 2  # (dx, dy) from the centre, y down


def test_project3d_rotation_sense_matches_imagej():
    dx, dy = _front_voxel_at_90("Y-Axis")
    assert dx < 0 and abs(dy) <= 1            # xnew = x cos + z sin: the front slice swings to the left
    dx, dy = _front_voxel_at_90("X-Axis")
    assert dy > 0 and dx < 0                  # ynew = y cos - z sin: the front slice swings DOWN (was up)
    dx, dy = _front_voxel_at_90("Z-Axis")
    assert dy < 0 and abs(dx) <= 1            # xnew = x cos - y sin, ynew = x sin + y cos: left -> top, clockwise on screen (was bottom)


# --- 3D Project: angle count mirrors ImageJ (end angle included, one dropped past 360) -------------------------------

@pytest.mark.parametrize("step,total,expected", [(90, 90, 2), (10, 180, 19), (10, 350, 36), (7, 360, 51), (90, 360, 4), (100, 360, 3)])
def test_project3d_frame_count_matches_imagej(step, total, expected):
    vol = np.zeros((8, 24, 24), np.uint8)
    vol[1, 12, 4] = 255
    s, _ = _session_with(vol)
    proj, out = _project(s, angle_step=step, total_rotation=total)
    assert proj.nt == expected
    assert out["message"].startswith(f"{expected} projections")


def test_project3d_partial_rotation_renders_the_end_angle():
    vol = np.zeros((8, 24, 24), np.uint8)
    vol[1, 12, 4] = 255
    s, _ = _session_with(vol)
    proj, _ = _project(s, angle_step=90, total_rotation=90, interpolate=False, interior_depth_cueing=0)
    assert proj.nt == 2
    assert not np.array_equal(proj.data[0], proj.data[1])  # the 90 deg view is a real rotation, not a copy of 0 deg


def test_project3d_frame_count_independent_of_initial_angle():
    vol = np.zeros((8, 24, 24), np.uint8)
    vol[1, 12, 4] = 255
    s, _ = _session_with(vol)
    proj, _ = _project(s, angle_step=90, total_rotation=360, initial_angle=-90)
    assert proj.nt == 4


# --- 3D Project: calibration follows the max_size downsampling ------------------------------------------------------

def test_project3d_downsampled_volume_scales_calibration():
    vol = np.zeros((4, 512, 512), np.uint8)
    vol[:, 250:262, 100:112] = 200
    vol[:, 250:262, 400:412] = 200  # spot centres 300 px = 75 um apart
    s, _ = _session_with(vol, cal=Calibration(pixel_width=0.25, pixel_height=0.25, pixel_depth=0.25, unit="um"))
    proj, out = _project(s, angle_step=90, max_size=256, interior_depth_cueing=0)
    assert proj.width <= 257 and "downsampled" in out["message"]
    assert proj.calibration.pixel_width == pytest.approx(0.5)
    assert proj.calibration.pixel_height == pytest.approx(0.5)
    assert proj.calibration.unit == "um"
    frame = proj.data[0, 0, 0].astype(float)
    ys, xs = np.nonzero(frame > 50)
    left, right = xs[xs < frame.shape[1] / 2], xs[xs >= frame.shape[1] / 2]
    sep_px = right.mean() - left.mean()
    assert sep_px * proj.calibration.pixel_width == pytest.approx(75.0, abs=1.0)


def test_project3d_small_volume_keeps_calibration():
    vol = np.zeros((4, 64, 64), np.uint8)
    vol[:, 30:34, 10:14] = 200
    s, _ = _session_with(vol, cal=Calibration(pixel_width=0.25, pixel_height=0.25, pixel_depth=0.25, unit="um"))
    proj, out = _project(s, angle_step=90)
    assert "downsampled" not in out["message"]
    assert proj.calibration.pixel_width == pytest.approx(0.25)
    assert proj.calibration.pixel_height == pytest.approx(0.25)


# --- 3D Project: interior depth-cueing of brightest-point projections (ImageJ default 50 %) --------------------------

def _front_back_volume():
    vol = np.zeros((16, 16, 16), np.uint8)
    vol[0, 4, 4] = 200    # front voxel
    vol[15, 12, 12] = 200  # back voxel
    return vol


def _front_back_values(**params):
    s, _ = _session_with(_front_back_volume())
    proj, _ = _project(s, method="Brightest Point", axis="Y-Axis", angle_step=90, total_rotation=360, **params)
    p = proj.data[0, 0, 0].astype(float)  # 0 deg frame: the volume is padded along X to 23 px, so x -> x + 3
    return p[4, 7], p[12, 15]


def test_project3d_brightest_point_interior_depth_cueing_default_matches_imagej():
    front, back = _front_back_values()  # default interior_depth_cueing=50
    assert back < front
    # ImageJ Projector port: front=186, back=118 for a 16^3 volume with projwidth 23 (zmin=-3, zmax=19)
    assert front == pytest.approx(186, abs=2)
    assert back == pytest.approx(118, abs=2)


def test_project3d_brightest_point_interior_depth_cueing_off():
    front, back = _front_back_values(interior_depth_cueing=0)
    assert front == 200 and back == 200


def test_project3d_interior_cueing_selects_raw_brightest_voxel():
    # a ray with 150 at the front and 200 at the back must report the cued 200 (ImageJ), not the cued 150
    vol = np.zeros((16, 16, 16), np.uint8)
    vol[0, 8, 8] = 150
    vol[15, 8, 8] = 200
    s, _ = _session_with(vol)
    proj, _ = _project(s, method="Brightest Point", axis="Y-Axis", angle_step=90, total_rotation=360, interior_depth_cueing=50)
    v = float(proj.data[0, 0, 0, 8, 11])
    assert v == pytest.approx(118, abs=2)


def test_project3d_mean_value_is_not_depth_cued():
    s, _ = _session_with(_front_back_volume())
    proj, _ = _project(s, method="Mean Value", axis="Y-Axis", angle_step=90, total_rotation=360, interior_depth_cueing=50)
    p = proj.data[0, 0, 0]
    assert p[4, 7] == 200 and p[12, 15] == 200


# --- Reduce on time series -----------------------------------------------------------------------------------------

def test_reduce_default_axis_reduces_frames_of_a_time_series():
    vol = np.arange(8, dtype=np.uint8)[:, None, None] * np.ones((1, 6, 6), np.uint8)
    s, im = _session_with(vol, axes="TYX")
    im.calibration = Calibration(pixel_width=1.0, pixel_height=1.0, pixel_depth=1.0, unit="um", frame_interval=0.5, time_unit="s")
    assert im.nt == 8 and im.nz == 1
    out = s.run_op("stacks.reduce", {"factor": 2})
    assert im.nt == 4 and im.nz == 1
    assert list(im.data[:, 0, 0, 0, 0]) == [0, 2, 4, 6]
    assert im.calibration.frame_interval == pytest.approx(1.0)
    assert im.calibration.pixel_depth == pytest.approx(1.0)  # untouched: nothing was removed along Z
    assert out["message"] == "8 → 4 frames"


def test_reduce_axis_t_on_a_z_stack_falls_back_to_z():
    vol = np.arange(6, dtype=np.uint8)[:, None, None] * np.ones((1, 6, 6), np.uint8)
    s, im = _session_with(vol, axes="ZYX")
    s.run_op("stacks.reduce", {"factor": 3, "axis": "T"})
    assert im.nz == 2 and im.nt == 1


def test_reduce_rejects_single_plane_image():
    s, im = _session_with(np.zeros((6, 6), np.uint8), axes="YX")
    with pytest.raises(ValueError):
        s.run_op("stacks.reduce", {"factor": 2})
    assert im.calibration.pixel_depth == pytest.approx(1.0)


# --- Orthogonal Views: calibration along Z -------------------------------------------------------------------------

@pytest.mark.parametrize("scale_z", [False, True])
def test_orthogonal_views_z_calibration(scale_z):
    vol = np.random.RandomState(0).randint(0, 255, (6, 32, 48)).astype(np.uint8)
    s, _ = _session_with(vol, cal=Calibration(pixel_width=0.5, pixel_height=0.7, pixel_depth=2.0, unit="um"))
    out = s.run_op("stacks.orthogonal_views", {"x": 20, "y": 10, "scale_z": scale_z})
    xz, yz = (s.images[i] for i in out["new_images"])
    z_px = 0.5 if scale_z else 2.0  # one row / column along Z is pixel_width after the zoom, else one slice
    assert xz.calibration.pixel_height == pytest.approx(z_px)   # XZ rows are Z
    assert xz.calibration.pixel_width == pytest.approx(0.5)     # XZ columns are X
    assert yz.calibration.pixel_width == pytest.approx(z_px)    # YZ columns are Z
    assert yz.calibration.pixel_height == pytest.approx(0.7)    # YZ rows are Y
    # the full Z extent measures 12 um either way
    assert xz.height * xz.calibration.pixel_height == pytest.approx(12.0)
    assert yz.width * yz.calibration.pixel_width == pytest.approx(12.0)


def test_orthogonal_views_zero_depth_falls_back_to_pixel_width():
    vol = np.zeros((4, 16, 16), np.uint8)
    s, _ = _session_with(vol, cal=Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=0.0, unit="um"))
    out = s.run_op("stacks.orthogonal_views", {"x": 8, "y": 8, "scale_z": False})
    xz, yz = (s.images[i] for i in out["new_images"])
    assert xz.calibration.pixel_height == pytest.approx(0.5) and yz.calibration.pixel_width == pytest.approx(0.5)


# --- Label Slices: starting value / interval and the {label} placeholder (ImageJ Image ▸ Stacks ▸ Label…) -----------------

def test_label_slices_start_interval_and_label_placeholder():
    from lumen.core.overlay_render import time_stamp_text
    s, img = _session_with(np.zeros((5, 8, 8), np.uint8))
    s.run_op("stacks.label_slices", {"format": "{label:04d} slice", "start": 10, "interval": 5})
    item = [o for o in img.overlay if o["type"] == "time_stamp"][0]
    assert (item["label_start"], item["label_interval"]) == (10, 5)
    assert [time_stamp_text(item, img, 0, z) for z in range(5)] == ["0010 slice", "0015 slice", "0020 slice", "0025 slice", "0030 slice"]
    s.run_op("stacks.label_slices", {"format": "{label}"})  # defaults: 1-based numbering, like {slice}
    item = [o for o in img.overlay if o["type"] == "time_stamp"][0]
    assert [time_stamp_text(item, img, 0, z) for z in range(5)] == ["1", "2", "3", "4", "5"]
    s, mov = _session_with(np.zeros((4, 2, 8, 8), np.uint8), axes="TZYX")  # several frames: {label} counts frames, not slices (StackLabeler.updateIndex)
    s.run_op("stacks.label_slices", {"format": "{label}", "start": 0, "interval": 2})
    item = [o for o in mov.overlay if o["type"] == "time_stamp"][0]
    assert [time_stamp_text(item, mov, t, 1) for t in range(4)] == ["0", "2", "4", "6"]


# --- Slice labels follow their planes through the stack editors (ImageJ ImageStack.deleteSlice / addSlice, SubstackMaker,
# --- StackReverser keep the label with the slice; ZProjector's result is unlabelled) -----------------------------------------

def _labelled_z_stack(s, n=6):
    img = s.add_image(LumenImage(np.stack([np.full((4, 4), i, np.uint8) for i in range(n)]), name="st", axes="ZYX"))
    img.metadata["slice_labels"] = [f"s{i + 1}" for i in range(n)]
    return img


def _labelled_hyperstack(s):
    """(T=2, Z=3, C=2): every plane's pixel value encodes its (t, z, c) and its label names the same triple."""
    vol = np.zeros((2, 3, 2, 4, 4), np.uint8)
    labels = [""] * 12
    for t in range(2):
        for z in range(3):
            for c in range(2):
                vol[t, z, c] = t * 6 + z * 2 + c
                labels[(t * 3 + z) * 2 + c] = f"t{t}z{z}c{c}"  # ImageJ plane order: channels fastest, then slices, then frames
    img = s.add_image(LumenImage(vol, name="hs", axes="TZCYX"))
    img.metadata["slice_labels"] = labels
    return img


def _label_of_pixels(v):
    v = int(v)
    return f"t{v // 6}z{(v % 6) // 2}c{v % 2}"


def _labels_follow_pixels(img):
    """Every plane's stored label names the (t, z, c) encoded in its pixels (the data the plane came from)."""
    from lumen.core.ops.stacks import plane_index, slice_labels
    labels = slice_labels(img)
    return all(labels[plane_index(img, t, z, c)] == _label_of_pixels(img.data[t, z, c, 0, 0]) for (t, z, c) in img.iter_planes())


def _vals(img, ax=1):
    return [int(np.take(img.data, k, axis=ax).ravel()[0]) for k in range(img.data.shape[ax])]


def test_substack_carries_the_labels_of_the_extracted_slices():
    s = Session()
    img = _labelled_z_stack(s)
    out = s.run_op("stacks.substack", {"slices": "4-6"}, image_id=img.id)
    sub = s.images[out["new_images"][0]]
    assert _vals(sub) == [3, 4, 5] and sub.metadata["slice_labels"] == ["s4", "s5", "s6"]
    assert img.metadata["slice_labels"] == ["s1", "s2", "s3", "s4", "s5", "s6"]  # the source is untouched
    out = s.run_op("stacks.substack", {"slices": "2,5", "delete_from_original": True}, image_id=img.id)
    sub = s.images[out["new_images"][0]]
    assert sub.metadata["slice_labels"] == ["s2", "s5"]
    assert _vals(img) == [0, 2, 3, 5] and img.metadata["slice_labels"] == ["s1", "s3", "s4", "s6"]  # the original keeps the labels of the kept slices


def test_delete_add_reverse_keep_labels_aligned():
    s = Session()
    img = _labelled_z_stack(s)
    img.display.z = 0
    s.run_op("stacks.delete_slice", {}, image_id=img.id)
    assert _vals(img) == [1, 2, 3, 4, 5] and img.metadata["slice_labels"] == ["s2", "s3", "s4", "s5", "s6"]
    s.run_op("stacks.reverse", {}, image_id=img.id)
    assert _vals(img) == [5, 4, 3, 2, 1] and img.metadata["slice_labels"] == ["s6", "s5", "s4", "s3", "s2"]
    img.display.z = 2
    s.run_op("stacks.add_slice", {}, image_id=img.id)  # blank slice after the current one: unlabelled, later labels move with their planes
    assert _vals(img) == [5, 4, 3, 0, 2, 1] and img.metadata["slice_labels"] == ["s6", "s5", "s4", "", "s3", "s2"]


def test_projection_and_stack_to_images_labels():
    s = Session()
    img = _labelled_z_stack(s)
    out = s.run_op("stacks.z_project", {"method": "Max Intensity"}, image_id=img.id)
    assert "slice_labels" not in s.images[out["new_images"][0]].metadata
    out = s.run_op("stacks.grouped_z_project", {"group_size": 2}, image_id=img.id)
    assert "slice_labels" not in s.images[out["new_images"][0]].metadata
    out = s.run_op("stacks.to_images", {}, image_id=img.id)
    assert [s.images[i].metadata["slice_labels"] for i in out["new_images"]] == [["s1"], ["s2"], ["s3"], ["s4"], ["s5"], ["s6"]]
    out = s.run_op("stacks.montage", {"columns": 3, "rows": 2}, image_id=img.id)
    assert "slice_labels" not in s.images[out["new_images"][0]].metadata


def test_hyperstack_labels_follow_planes_on_every_axis():
    s = Session()
    img = _labelled_hyperstack(s)
    assert _labels_follow_pixels(img)
    out = s.run_op("stacks.substack", {"slices": "2", "axis": "C"}, image_id=img.id)
    sub = s.images[out["new_images"][0]]
    assert sub.shape[:3] == (2, 3, 1) and _labels_follow_pixels(sub)
    out = s.run_op("stacks.substack", {"slices": "2", "axis": "T"}, image_id=img.id)
    assert _labels_follow_pixels(s.images[out["new_images"][0]])
    img.display.t, img.display.z, img.display.c = 1, 1, 0
    s.run_op("stacks.delete_slice", {"axis": "T"}, image_id=img.id)
    assert img.shape[:3] == (1, 3, 2) and _labels_follow_pixels(img)
    s.run_op("stacks.delete_slice", {"axis": "C"}, image_id=img.id)
    assert img.shape[:3] == (1, 3, 1) and _labels_follow_pixels(img)
    s.run_op("stacks.reverse", {"axis": "Z"}, image_id=img.id)
    assert _labels_follow_pixels(img)
    img = _labelled_hyperstack(s)
    s.run_op("stacks.reverse", {"axis": "T"}, image_id=img.id)
    assert _labels_follow_pixels(img)
    s.run_op("stacks.reorder", {"order": "Swap Z and T"}, image_id=img.id)
    assert img.shape[:3] == (3, 2, 2) and _labels_follow_pixels(img)
    s.run_op("stacks.reorder", {"order": "Swap C and T"}, image_id=img.id)
    assert img.shape[:3] == (2, 2, 3) and _labels_follow_pixels(img)
    img.display.t, img.display.z, img.display.c = 0, 0, 1
    s.run_op("stacks.add_slice", {"axis": "C"}, image_id=img.id)
    from lumen.core.ops.stacks import plane_index, slice_labels
    assert img.shape[:3] == (2, 2, 4) and slice_labels(img)[plane_index(img, 1, 1, 2)] == ""
    assert slice_labels(img)[plane_index(img, 1, 1, 3)] == _label_of_pixels(img.data[1, 1, 3, 0, 0])
    img = _labelled_hyperstack(s)
    s.run_op("stacks.hyperstack_to_stack", {}, image_id=img.id)
    flat = s.active
    s.run_op("stacks.to_hyperstack", {"channels": 2, "slices": 3, "frames": 2, "order": "xyzct"}, image_id=flat.id)
    labels = slice_labels(flat)
    assert all(labels[plane_index(flat, t, z, c)].startswith(_label_of_pixels(flat.data[t, z, c, 0, 0])) for (t, z, c) in flat.iter_planes())


def test_unlabelled_stacks_do_not_gain_labels():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((4, 4, 4), np.uint8), name="plain", axes="ZYX"))
    s.run_op("stacks.delete_slice", {}, image_id=img.id)
    s.run_op("stacks.add_slice", {}, image_id=img.id)
    s.run_op("stacks.reverse", {}, image_id=img.id)
    out = s.run_op("stacks.substack", {"slices": "1-2"}, image_id=img.id)
    assert "slice_labels" not in img.metadata and "slice_labels" not in s.images[out["new_images"][0]].metadata


def test_images_to_stack_and_concatenate_labels():
    s = Session()
    a = s.add_image(LumenImage(np.full((4, 4), 1, np.uint8), name="a"))
    a.metadata["slice_labels"] = ["A"]
    s.add_image(LumenImage(np.full((4, 4), 2, np.uint8), name="b"))  # unlabelled
    c = s.add_image(LumenImage(np.full((2, 4, 4), 3, np.uint8), name="c", axes="ZYX"))
    c.metadata["slice_labels"] = ["C1", "C2"]
    out = s.run_op("stacks.images_to_stack", {})
    st = s.images[out["new_images"][0]]
    assert st.nz == 4 and st.metadata["slice_labels"] == ["A", "", "C1", "C2"]
    out = s.run_op("stacks.concatenate", {"image1": c.id, "image2": a.id, "axis": "Z"}, image_id=c.id)
    assert s.images[out["new_images"][0]].metadata["slice_labels"] == ["C1", "C2", "A"]


# --- Images to Stack / Concatenate promote mixed types to the widest one (ImageJ ImagesToStack) instead of wrapping ----------

def test_images_to_stack_promotes_mixed_dtypes():
    s = Session()
    s.add_image(LumenImage(np.full((8, 8), 100, np.uint8), name="u8"))
    s.add_image(LumenImage(np.full((8, 8), 0.75, np.float32), name="f32"))
    s.add_image(LumenImage(np.full((8, 8), 300.0, np.float32), name="f32big"))
    s.add_image(LumenImage(np.full((8, 8), 40000, np.uint16), name="u16"))
    st = s.images[s.run_op("stacks.images_to_stack", {})["new_images"][0]]
    assert st.dtype == np.float32 and [float(st.data[0, k, 0, 0, 0]) for k in range(4)] == [100.0, 0.75, 300.0, 40000.0]
    s = Session()
    s.add_image(LumenImage(np.full((8, 8), 100, np.uint8), name="u8"))
    s.add_image(LumenImage(np.full((8, 8), 40000, np.uint16), name="u16"))
    st = s.images[s.run_op("stacks.images_to_stack", {})["new_images"][0]]
    assert st.dtype == np.uint16 and [int(st.data[0, k, 0, 0, 0]) for k in range(2)] == [100, 40000]


def test_concatenate_promotes_mixed_dtypes_independent_of_order():
    s = Session()
    u8 = s.add_image(LumenImage(np.full((8, 8), 100, np.uint8), name="u8"))
    f32 = s.add_image(LumenImage(np.full((8, 8), 300.0, np.float32), name="f32"))
    r = s.images[s.run_op("stacks.concatenate", {"image1": u8.id, "image2": f32.id, "axis": "Z"}, image_id=u8.id)["new_images"][0]]
    assert r.dtype == np.float32 and [float(r.data[0, k, 0, 0, 0]) for k in range(2)] == [100.0, 300.0]
    r = s.images[s.run_op("stacks.concatenate", {"image1": f32.id, "image2": u8.id, "axis": "Z"}, image_id=u8.id)["new_images"][0]]
    assert r.dtype == np.float32 and [float(r.data[0, k, 0, 0, 0]) for k in range(2)] == [300.0, 100.0]
    u8b = s.add_image(LumenImage(np.full((8, 8), 7, np.uint8), name="u8b"))
    r = s.images[s.run_op("stacks.concatenate", {"image1": u8.id, "image2": u8b.id, "axis": "Z"}, image_id=u8.id)["new_images"][0]]
    assert r.dtype == np.uint8 and [int(r.data[0, k, 0, 0, 0]) for k in range(2)] == [100, 7]


# --- Reslice calibration: rows are the Z planes (pixel_depth apart unless Z was resampled), ImageJ Slicer axis mapping -------

def _anisotropic_stack(s):
    vol = np.random.default_rng(0).integers(0, 255, (1, 5, 1, 20, 30)).astype(np.uint8)
    img = s.add_image(LumenImage(vol, name="vol"))
    img.calibration = Calibration(pixel_width=0.1, pixel_height=0.2, pixel_depth=0.5, unit="um")
    return img


def _cal(img):
    c = img.calibration
    return (pytest.approx(c.pixel_width), pytest.approx(c.pixel_height), pytest.approx(c.pixel_depth))


def test_reslice_calibration_without_z_interpolation():
    s = Session()
    img = _anisotropic_stack(s)
    top = s.images[s.run_op("stacks.reslice", {"direction": "Top (XZ)", "interpolate_z": False}, image_id=img.id)["new_images"][0]]
    assert top.shape == (1, 20, 1, 5, 30) and (0.1, 0.5, 0.2) == _cal(top)  # columns X, rows Z (one slice each), slices Y
    left = s.images[s.run_op("stacks.reslice", {"direction": "Left (YZ)", "interpolate_z": False}, image_id=img.id)["new_images"][0]]
    assert left.shape == (1, 30, 1, 5, 20) and (0.2, 0.5, 0.1) == _cal(left)  # columns Y, rows Z, slices X
    line = {"type": "line", "x1": 2, "y1": 3, "x2": 25, "y2": 15}
    ln = s.images[s.run_op("stacks.reslice", {"interpolate_z": False}, image_id=img.id, roi=line)["new_images"][0]]
    assert ln.shape[3] == 5 and (0.1, 0.5, 0.1) == _cal(ln)
    rot = s.images[s.run_op("stacks.reslice", {"interpolate_z": False, "rotate": True}, image_id=img.id, roi=line)["new_images"][0]]
    assert rot.shape[4] == 5 and (0.5, 0.1, 0.1) == _cal(rot)  # depth now runs along X
    rad = s.images[s.run_op("stacks.radial_reslice", {"n_angles": 2, "interpolate_z": False}, image_id=img.id)["new_images"][0]]
    assert rad.shape[3] == 5 and (0.1, 0.5, 0.1) == _cal(rad)
    assert "slice_labels" not in top.metadata and "slice_labels" not in ln.metadata


def test_reslice_calibration_with_z_interpolation():
    s = Session()
    img = _anisotropic_stack(s)
    top = s.images[s.run_op("stacks.reslice", {"direction": "Top (XZ)", "interpolate_z": True}, image_id=img.id)["new_images"][0]]
    assert top.shape == (1, 20, 1, 25, 30) and (0.1, 0.1, 0.2) == _cal(top)  # Z resampled to pixel_width spacing
    left = s.images[s.run_op("stacks.reslice", {"direction": "Left (YZ)", "interpolate_z": True}, image_id=img.id)["new_images"][0]]
    assert left.shape == (1, 30, 1, 25, 20) and (0.2, 0.1, 0.1) == _cal(left)
    ln = s.images[s.run_op("stacks.reslice", {"interpolate_z": True}, image_id=img.id, roi={"type": "line", "x1": 2, "y1": 3, "x2": 25, "y2": 15})["new_images"][0]]
    assert ln.shape[3] == 25 and (0.1, 0.1, 0.1) == _cal(ln)
    iso = s.add_image(LumenImage(np.zeros((1, 3, 1, 8, 8), np.uint8), name="iso"))
    iso.calibration = Calibration(pixel_width=0.3, pixel_height=0.3, pixel_depth=0.3, unit="um")  # nothing to resample: one row = one slice
    top = s.images[s.run_op("stacks.reslice", {"direction": "Top (XZ)", "interpolate_z": True}, image_id=iso.id)["new_images"][0]]
    assert top.shape[3] == 3 and (0.3, 0.3, 0.3) == _cal(top)


# --- Radial Reslice samples the rotating line 1 px per column (ImageJ Slicer.getLine) ----------------------------------------

def test_radial_reslice_samples_one_pixel_per_column():
    vol = np.zeros((1, 2, 1, 40, 40), np.uint8)
    vol[..., 29] = 255  # bright column 9 px right of a centre placed on a pixel centre
    s = Session()
    img = s.add_image(LumenImage(vol, name="v"))
    out = s.run_op("stacks.radial_reslice", {"n_angles": 1, "length": 12, "interpolate_z": False}, roi={"type": "point", "points": [[20.5, 20.5]]})
    r = s.images[out["new_images"][0]]
    assert r.width == 12 and r.calibration.pixel_width == 1.0
    assert int(np.argmax(r.data[0, 0, 0, 0])) == 9 and int(r.data[0, 0, 0, 0, 9]) == 255  # column k is k px from the centre
    out = s.run_op("stacks.radial_reslice", {"n_angles": 2, "length": 10}, image_id=img.id, roi={"type": "point", "points": [[20, 20]]})
    assert s.images[out["new_images"][0]].width == 10  # round(length) columns, like Fiji


# --- Stack Statistics with an empty selection reports 0 voxels / NaN instead of crashing ------------------------------------

def test_stack_statistics_empty_selection_reports_nan():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((1, 3, 1, 10, 10), np.uint8), name="z"))
    out = s.run_op("stacks.statistics", {}, image_id=img.id, roi={"type": "rect", "x": 50, "y": 50, "w": 5, "h": 5})
    row = out["table"]["rows"][0]
    assert out["message"] == "0 voxels, mean NaN" and row[1] == 0 and row[3] is None
    two = np.full((1, 2, 2, 6, 6), 7.0, np.float32)
    two[:, :, 0] = np.nan  # channel 1 has no finite voxel, channel 2 is fine: one empty channel must not abort the table
    img = s.add_image(LumenImage(two, name="mixed"))
    out = s.run_op("stacks.statistics", {}, image_id=img.id)
    rows = out["table"]["rows"]
    assert out["message"] == "0 voxels, mean NaN" and rows[0][3] is None and rows[1][1] == 72 and rows[1][3] == pytest.approx(7.0)


# --- Reduce keeps the labels of the kept planes (and undo brings the whole list back) --------------------------------------

def test_reduce_keeps_labels_of_kept_planes_and_undo_restores_them():
    vol = np.stack([np.full((6, 6), 10 * (k + 1), np.uint8) for k in range(6)])
    s, img = _session_with(vol)
    img.metadata["slice_labels"] = [f"s{k + 1}" for k in range(6)]
    s.run_op("stacks.reduce", {"factor": 2, "axis": "Z"})
    assert img.nz == 3 and [int(img.data[0, z, 0, 0, 0]) for z in range(3)] == [10, 30, 50]
    assert img.metadata["slice_labels"] == ["s1", "s3", "s5"]
    assert s.undo(img.id) == "Reduce"
    assert img.nz == 6 and img.metadata["slice_labels"] == [f"s{k + 1}" for k in range(6)]
    assert s.redo(img.id) == "Reduce" and img.metadata["slice_labels"] == ["s1", "s3", "s5"]
    # frames
    s, img = _session_with(vol, axes="TYX")
    img.metadata["slice_labels"] = [f"f{k + 1}" for k in range(6)]
    s.run_op("stacks.reduce", {"factor": 3, "axis": "T"})
    assert img.nt == 2 and img.metadata["slice_labels"] == ["f1", "f4"]
    # an unlabelled stack stays unlabelled
    s, img = _session_with(vol)
    s.run_op("stacks.reduce", {"factor": 2, "axis": "Z"})
    assert "slice_labels" not in img.metadata


# ---------------------------------------------------------------- regression tests (code review, fix round 4)

def _rgb(name="rgb", nz=1, colour=(255, 0, 0)):
    a = np.zeros((nz, 8, 8, 3), np.uint8)
    a[...] = colour
    return LumenImage(a, name=name, axes="ZYXS", rgb=True)


def test_substack_delete_channel_updates_rgb_flag_and_channel_display():
    from lumen.core.render import render_rgb
    # RGB source: losing a channel makes it a plain 8-bit 2-channel image that still renders
    # (rgb=True with C=2 made render_rgb / rgb.png index a missing third channel)
    s = Session()
    img = s.add_image(_rgb())
    s.run_op("stacks.substack", {"slices": "1", "axis": "C", "delete_from_original": True})
    assert img.rgb is False and img.nc == 2 and img.bit_depth == "8-bit"
    assert [c.lut for c in img.display.channels] == ["Green", "Blue"]  # the surviving samples keep their own LUTs
    assert render_rgb(img).shape[:2] == (8, 8)
    assert s.undo(img.id) == "Make Substack" and img.rgb is True and img.nc == 3
    # non-RGB: the kept channels keep their names / LUTs instead of being re-labelled by index
    s = Session()
    img = s.add_image(LumenImage(np.stack([np.full((8, 8), v, np.uint16) for v in (10, 20, 30)]), name="mc", axes="CYX"))
    for ch, name, lut in zip(img.display.channels, ["DAPI", "GFP", "RFP"], ["Blue", "Green", "Red"]):
        ch.name, ch.lut = name, lut
    out = s.run_op("stacks.substack", {"slices": "1", "axis": "C", "delete_from_original": True})
    sub = s.images[out["new_images"][0]]
    assert [(c.name, c.lut) for c in img.display.channels] == [("GFP", "Green"), ("RFP", "Red")]
    assert [int(img.data[0, 0, c, 0, 0]) for c in range(img.nc)] == [20, 30]
    assert [(c.name, c.lut) for c in sub.display.channels] == [("DAPI", "Blue")]
    s.run_op("stacks.substack", {"slices": "1", "axis": "C", "delete_from_original": True}, image_id=img.id)
    assert img.nc == 1 and img.display.mode == "single" and img.display.channels[0].name == "RFP"
    assert s.undo(img.id) == "Make Substack" and [c.name for c in img.display.channels] == ["GFP", "RFP"]
    # deleting along Z leaves an RGB stack RGB
    s = Session()
    img = s.add_image(_rgb(nz=3))
    s.run_op("stacks.substack", {"slices": "1", "axis": "Z", "delete_from_original": True})
    assert img.rgb is True and img.nz == 2 and [c.lut for c in img.display.channels] == ["Red", "Green", "Blue"]


def test_rgb_stacks_stay_rgb_through_images_to_stack_reorder_and_to_hyperstack():
    # ImageJ keeps an RGB stack RGB through Images to Stack / Re-order / Stack to Hyperstack; forcing rgb=False gave the
    # R, G, B samples the Cyan / Magenta / Yellow composite LUTs (a red slice rendered cyan)
    from lumen.core.render import render_rgb
    s = Session()
    s.add_image(_rgb("a"))
    s.add_image(_rgb("b", colour=(0, 255, 0)))
    out = s.run_op("stacks.images_to_stack", {})
    st = s.images[out["new_images"][0]]
    assert st.rgb is True and st.bit_depth == "RGB" and [c.lut for c in st.display.channels] == ["Red", "Green", "Blue"]
    assert tuple(render_rgb(st, z=0)[0, 0]) == (255, 0, 0) and tuple(render_rgb(st, z=1)[0, 0]) == (0, 255, 0)
    # mixed RGB + composite sources, or stacking along C, is not RGB
    s = Session()
    s.add_image(_rgb("a"))
    s.add_image(LumenImage(np.zeros((3, 8, 8), np.uint8), name="c", axes="CYX"))
    assert s.images[s.run_op("stacks.images_to_stack", {})["new_images"][0]].rgb is False
    s = Session()
    s.add_image(_rgb("a"))
    s.add_image(_rgb("b"))
    assert s.images[s.run_op("stacks.images_to_stack", {"axis": "C"})["new_images"][0]].rgb is False
    # Re-order: Swap Z and T leaves the samples on C; Swap Z and C on a 3-slice RGB stack must drop the flag
    # (replace_data's shape heuristic alone would keep it, since C is still 3 × uint8)
    s = Session()
    img = s.add_image(_rgb(nz=4))
    s.run_op("stacks.reorder", {"order": "Swap Z and T"})
    assert img.rgb is True and img.shape[:3] == (4, 1, 3) and [c.lut for c in img.display.channels] == ["Red", "Green", "Blue"]
    assert tuple(render_rgb(img)[0, 0]) == (255, 0, 0)
    s = Session()
    img = s.add_image(_rgb(nz=3))
    s.run_op("stacks.reorder", {"order": "Swap Z and C"})
    assert img.rgb is False
    # Stack to Hyperstack: RGB only when channels == 3 and c is the fastest-varying dimension (xyczt / xyctz)
    for order in ["xyczt", "xyzct", "xyctz", "xytcz", "xyztc", "xytzc"]:
        s = Session()
        img = s.add_image(_rgb(nz=4))
        s.run_op("stacks.to_hyperstack", {"channels": 3, "slices": 2, "frames": 2, "order": order})
        expect = order[2] == "c"
        assert img.rgb is expect, order
        assert ([c.lut for c in img.display.channels] == ["Red", "Green", "Blue"]) is expect, order
        if expect:
            assert tuple(render_rgb(img)[0, 0]) == (255, 0, 0)


def test_combine_and_insert_promote_to_the_widest_type():
    # ImageJ refuses stacks of different types; like images_to_stack / concatenate Lumen widens instead of clipping
    # a 16-bit stack to 255 (or rounding floats) when stack 1 / the destination is the narrower type
    s = Session()
    a = s.add_image(LumenImage(np.full((2, 10, 12), 100, np.uint8), name="a8.tif", axes="ZYX"))
    b = s.add_image(LumenImage(np.full((2, 10, 8), 4000, np.uint16), name="b16.tif", axes="ZYX"))
    out = s.run_op("stacks.combine", {"image1": a.id, "image2": b.id})
    comb = s.images[out["new_images"][0]]
    assert comb.data.dtype == np.uint16 and comb.width == 20
    assert comb.data[..., :12].max() == 100 and comb.data[..., 12:].min() == 4000
    assert comb.display.channels[0].max > 255 and "(uint16)" in out["message"]  # display range re-derived, not the 8-bit 0–255
    f = s.add_image(LumenImage(np.full((2, 10, 8), 3.7, np.float32), name="f.tif", axes="ZYX"))
    out = s.run_op("stacks.combine", {"image1": a.id, "image2": f.id})
    comb = s.images[out["new_images"][0]]
    assert comb.data.dtype == np.float32 and float(comb.data[..., 12:].min()) == pytest.approx(3.7)
    out = s.run_op("stacks.combine", {"image1": a.id, "image2": a.id})  # same types: unchanged, no promotion note
    assert s.images[out["new_images"][0]].data.dtype == np.uint8 and "uint8" not in out["message"]
    # Insert widens the destination in place (undoable) instead of clipping the pasted 16-bit values to 255
    s.select(a.id)
    out = s.run_op("stacks.insert", {"source": b.id, "x": 2, "y": 0}, image_id=a.id)
    assert a.data.dtype == np.uint16 and int(a.data[0, 0, 0, 0, 2]) == 4000 and int(a.data[0, 0, 0, 0, 0]) == 100
    assert a.display.channels[0].max > 255 and "promoted to uint16" in out["message"]
    assert s.undo(a.id) == "Insert" and a.data.dtype == np.uint8 and a.data.max() == 100


def test_failed_op_rolls_back_in_place_metadata_edits():
    # the stack editors rewrite the slice labels in place before run_op installs the new data; an op that then
    # fails must leave the labels (and the binary-mask flag) as they were – they were not part of the rollback check
    from lumen.core.registry import REGISTRY
    s = Session()
    img = _labelled_z_stack(s, 3)
    img.metadata["fit"] = np.arange(3)  # a value without plain equality must not raise a secondary error in the handler
    before = list(img.metadata["slice_labels"])
    for op_id, params in (("stacks.reverse", {"axis": "Z"}), ("stacks.add_slice", {"axis": "Z"}), ("stacks.delete_slice", {"axis": "Z"})):
        spec = REGISTRY.get(op_id)
        orig = spec.func

        def boom(ctx, _orig=orig, **kw):
            _orig(ctx, **kw)  # the op rewrote the labels in place ...
            ctx.img.lut_is_binary = True
            raise RuntimeError("boom")  # ... and then failed before run_op installed the new data

        spec.func = boom
        try:
            with pytest.raises(RuntimeError, match="boom"):
                s.run_op(op_id, params, image_id=img.id)
        finally:
            spec.func = orig
        assert img.metadata["slice_labels"] == before and _vals(img) == [0, 1, 2], op_id
        assert img.lut_is_binary is False and {k: s.histories[img.id].labels()[k] for k in ("undo", "redo")} == {"undo": [], "redo": []}
    # a plain validation failure that touched nothing still costs no restore (no rev bump → no client reload)
    one = s.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="one"))
    with pytest.raises(ValueError):
        s.run_op("stacks.delete_slice", {"axis": "Z"}, image_id=one.id)
    assert one.rev == 0


def test_remove_slice_labels_is_a_no_op_on_an_unlabelled_stack():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((3, 8, 8), np.uint8), name="u", axes="ZYX"))
    s.run_op("filters.gaussian", {"sigma": 1})
    assert s.undo(img.id) == "Gaussian Blur"
    rev = img.rev
    out = s.run_op("stacks.remove_slice_labels", {})
    assert out["message"] == "The image has no slice labels" and img.rev == rev
    assert {k: s.histories[img.id].labels()[k] for k in ("undo", "redo")} == {"undo": [], "redo": ["Gaussian Blur"]}  # no undo entry, the redo stack survives
    img.metadata["slice_labels"] = ["", "", ""]  # a list of empty labels counts as unlabelled too
    s.run_op("stacks.remove_slice_labels", {})
    assert "slice_labels" not in img.metadata and img.rev == rev and s.histories[img.id].labels()["redo"] == ["Gaussian Blur"]
    img.metadata["slice_labels"] = ["a", "", ""]
    out = s.run_op("stacks.remove_slice_labels", {})
    assert out["message"] == "Slice labels removed" and img.rev == rev + 1 and "slice_labels" not in img.metadata
    assert {k: s.histories[img.id].labels()[k] for k in ("undo", "redo")} == {"undo": ["Remove slice labels"], "redo": []}
    assert s.undo(img.id) == "Remove slice labels" and img.metadata["slice_labels"] == ["a", "", ""]


def test_z_project_rejects_an_empty_slice_range():
    # start beyond the stack or start > stop: Max / Min raised numpy's raw 'zero-size array' error and the other
    # methods silently produced an all-zero projection
    s = Session()
    img = s.add_image(LumenImage(np.stack([np.full((4, 4), k + 1, np.uint8) for k in range(5)]), name="z", axes="ZYX"))
    for params in ({"start": 7}, {"start": 4, "stop": 2}, {"start": 6, "stop": 6}, {"start": 7, "method": "Average Intensity"}):
        with pytest.raises(ValueError, match="start slice"):
            s.run_op("stacks.z_project", params, image_id=img.id)
    assert len(s.images) == 1
    out = s.run_op("stacks.z_project", {"start": 5, "stop": 5}, image_id=img.id)
    assert int(s.images[out["new_images"][0]].data.max()) == 5
    out = s.run_op("stacks.z_project", {"start": 2, "stop": 99, "method": "Min Intensity"}, image_id=img.id)
    assert int(s.images[out["new_images"][0]].data.min()) == 2  # stop beyond the stack is still clamped
    mov = s.add_image(LumenImage(np.zeros((3, 4, 4), np.uint8), name="t", axes="TYX"))
    with pytest.raises(ValueError, match="start frame 4"):
        s.run_op("stacks.z_project", {"start": 4, "axis": "T"}, image_id=mov.id)


def test_reorder_swap_z_t_keeps_the_channel_display():
    # Fiji's Hyperstack_rearranger copies the LUTs (with their ranges) when channels stay channels; Lumen reset them
    # to the Cyan / Magenta defaults and auto-ranged every channel
    s = Session()
    img = s.add_image(LumenImage(np.random.default_rng(1).integers(0, 1000, (3, 4, 2, 8, 8)).astype(np.uint16), name="hs", axes="TZCYX"))
    c0, c1 = img.display.channels
    c0.lut, c0.min, c0.max, c0.gamma = "Blue", 100.0, 900.0, 2.0
    c1.lut, c1.visible = "Green", False
    img.display.t, img.display.z = 2, 1
    before = [c.to_dict() for c in img.display.channels]
    s.run_op("stacks.reorder", {"order": "Swap Z and T"})
    assert img.shape[:3] == (4, 3, 2) and [c.to_dict() for c in img.display.channels] == before
    assert (img.display.t, img.display.z) == (1, 2)  # the plane that was on screen stays on screen
    assert s.undo(img.id) == "Re-order Hyperstack" and (img.display.t, img.display.z) == (2, 1)
    # the channel axis is part of the swap: the channels are new planes → default LUTs, as in Fiji
    s.run_op("stacks.reorder", {"order": "Swap C and T"})
    assert img.shape[:3] == (2, 4, 3) and [c.lut for c in img.display.channels] == ["Cyan", "Magenta", "Yellow"]
