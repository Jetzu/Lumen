import numpy as np
from fastapi.testclient import TestClient

from lumen.core.session import Session
from lumen.server.app import create_app


def _client():
    s = Session()
    return TestClient(create_app(s)), s


def _open_cells(c):
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Cells (3-D, 2 channels)"}})
    assert r.status_code == 200, r.text
    return r.json()["new_images"][0]


def test_volume_headers_and_sizes():
    c, s = _client()
    iid = _open_cells(c)
    img = s.get_image(iid)
    assert img.nz == 60 and img.nc == 2
    r = c.get(f"/api/images/{iid}/volume", params={"t": 0, "max_dim": 128})
    assert r.status_code == 200, r.text
    assert r.headers["x-dtype"] == "uint8"
    assert r.headers["x-channels"] == "2"
    assert r.headers["x-channel-luts"] == "Magenta,Green"
    assert r.headers["x-channel-index"] == "0,1"
    z, y, x = map(int, r.headers["x-shape"].split(","))
    spacing = float(r.headers["x-spacing"])
    assert x == 128 and y == 128
    # shrinking by 2 in xy: z is resampled to isotropic voxels (60 slices * 0.29/0.26 um / 2 = 33.5)
    assert z == 33 and abs(spacing - 1.0) < 0.05
    assert abs(z * spacing - 60 * (0.29 / 0.26) * 0.5) < 0.05
    assert len(r.content) == 2 * z * y * x
    vol = np.frombuffer(r.content, np.uint8).reshape(2, z, y, x)
    # both channels have signal and are not saturated everywhere
    assert vol[0].max() > 50 and vol[1].max() > 50
    assert vol[0].mean() < 200 and vol[1].mean() < 200

    # full resolution: z is never upsampled, the anisotropy goes into X-Spacing
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 256})
    z, y, x = map(int, r.headers["x-shape"].split(","))
    assert (z, y, x) == (60, 256, 256)
    assert abs(float(r.headers["x-spacing"]) - 0.29 / 0.26) < 1e-4
    assert r.headers["x-downsample"] == "1"

    # 'current' channel selection
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64, "channels": "current"})
    assert r.headers["x-channels"] == "1"
    z, y, x = map(int, r.headers["x-shape"].split(","))
    assert len(r.content) == z * y * x and x == 64
    # explicit index list
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64, "channels": "1"})
    assert r.headers["x-channel-luts"] == "Green"


def test_volume_cache_and_display_range():
    c, s = _client()
    iid = _open_cells(c)
    r1 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    r2 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    assert r1.content == r2.content
    # changing the display range changes the returned intensities (cache key includes the range)
    ch = s.get_image(iid).display.channels[1]
    c.post(f"/api/images/{iid}/display", json={"channels": [{}, {"min": ch.min, "max": ch.max * 4}]})
    r3 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    z, y, x = map(int, r3.headers["x-shape"].split(","))
    v2 = np.frombuffer(r2.content, np.uint8).reshape(2, z, y, x)
    v3 = np.frombuffer(r3.content, np.uint8).reshape(2, z, y, x)
    assert np.array_equal(v2[0], v3[0])
    assert v3[1].mean() < v2[1].mean()


def test_volume_downsamples_z_when_needed():
    c, s = _client()
    iid = _open_cells(c)
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 32})
    z, y, x = map(int, r.headers["x-shape"].split(","))
    assert x == 32 and y == 32 and z < 60
    assert abs(z * float(r.headers["x-spacing"]) - 60 * (0.29 / 0.26) * (32 / 256)) < 1.0


def test_ortho_planes():
    c, s = _client()
    iid = _open_cells(c)
    r = c.get(f"/api/images/{iid}/ortho", params={"x": 100, "y": 50, "t": 0})
    assert r.status_code == 200, r.text
    assert r.headers["x-dtype"] == "uint16"
    assert r.headers["x-channels"] == "2"
    assert r.headers["x-shape-xz"] == "60,256"
    assert r.headers["x-shape-yz"] == "60,256"
    assert r.headers["x-point"] == "100,50"
    assert abs(float(r.headers["x-spacing"]) - 0.29 / 0.26) < 1e-4
    n_xz = 2 * 60 * 256
    n_yz = 2 * 60 * 256
    assert len(r.content) == (n_xz + n_yz) * 2
    arr = np.frombuffer(r.content, np.uint16)
    xz = arr[:n_xz].reshape(2, 60, 256)
    yz = arr[n_xz:].reshape(2, 60, 256)
    img = s.get_image(iid)
    assert np.array_equal(xz[0], img.data[0, :, 0, 50, :])
    assert np.array_equal(yz[1], img.data[0, :, 1, :, 100])
    # out-of-range points are clamped, missing points default to the centre
    r = c.get(f"/api/images/{iid}/ortho", params={"x": 9999, "y": -5})
    assert r.headers["x-point"] == "255,0"
    r = c.get(f"/api/images/{iid}/ortho")
    assert r.headers["x-point"] == "128,128"


def test_volume_single_plane_image():
    c, s = _client()
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    iid = r.json()["new_images"][0]
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    assert r.status_code == 200
    z, y, x = map(int, r.headers["x-shape"].split(","))
    assert z == 1 and max(y, x) == 64


def test_volume_cache_tracks_calibration_changes():
    """Properties / Set Scale change the voxel depth without bumping rev; the volume must be rebuilt."""
    c, s = _client()
    iid = _open_cells(c)
    r1 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 128})
    assert r1.headers["x-shape"] == "33,128,128"
    rev = s.get_image(iid).rev
    r = c.post("/api/ops/image.properties/run", json={"params": {"pixel_width": 0.26, "pixel_height": 0.26, "voxel_depth": 1.0, "unit": "µm"}})
    assert r.status_code == 200, r.text
    assert s.get_image(iid).calibration.pixel_depth == 1.0
    assert s.get_image(iid).rev == rev  # calibration-only change: rev is for pixel data
    r2 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 128})
    assert r2.headers["x-shape"] == "60,128,128"
    assert abs(float(r2.headers["x-spacing"]) - 1.0 / 0.26 / 2) < 1e-4
    assert len(r2.content) != len(r1.content)
    # the ortho endpoint reports the same (guarded) ratio
    r = c.get(f"/api/images/{iid}/ortho")
    assert abs(float(r.headers["x-spacing"]) - 1.0 / 0.26) < 1e-4
    # restoring the calibration gives back the original (still cached) volume
    c.post("/api/ops/image.properties/run", json={"params": {"pixel_width": 0.26, "pixel_height": 0.26, "voxel_depth": 0.29, "unit": "µm"}})
    r3 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 128})
    assert r3.headers["x-shape"] == "33,128,128" and r3.content == r1.content
    # degenerate calibration (pixel width 0) falls back to isotropic instead of Infinity/NaN
    c.post("/api/ops/image.properties/run", json={"params": {"pixel_width": 0.0, "pixel_height": 0.0, "voxel_depth": 0.29, "unit": "µm"}})
    r = c.get(f"/api/images/{iid}/ortho")
    assert float(r.headers["x-spacing"]) == 1.0
    r = c.get(f"/api/images/{iid}/volume", params={"max_dim": 128})
    assert r.headers["x-shape"] == "30,128,128" and r.headers["x-spacing"] == "1"


def test_volume_inverted_lut_does_not_change_intensities():
    """'Invert LUT' is a colour-only table reversal (as in the 2-D views): the voxel densities are unchanged."""
    c, s = _client()
    iid = _open_cells(c)
    r1 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64, "channels": "0"})
    r = c.post(f"/api/images/{iid}/display", json={"channels": [{"inverted": True}]})
    assert r.status_code == 200, r.text
    assert s.get_image(iid).display.channels[0].inverted is True
    r2 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64, "channels": "0"})
    assert r2.content == r1.content
    v = np.frombuffer(r2.content, np.uint8)
    assert v.mean() < 100  # background stays dark / transparent, not saturated


def test_volume_lut_header_follows_display_without_rebuilding():
    c, s = _client()
    iid = _open_cells(c)
    r1 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    assert r1.headers["x-channel-luts"] == "Magenta,Green"
    r = c.post(f"/api/images/{iid}/display", json={"channels": [{"lut": "Red"}, {"lut": "Cyan"}]})
    assert r.status_code == 200, r.text
    r2 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64})
    assert r2.headers["x-channel-luts"] == "Red,Cyan"
    assert r2.headers["x-channel-index"] == "0,1"
    assert r2.content == r1.content  # the body does not depend on the LUT: served from the cache
    r3 = c.get(f"/api/images/{iid}/volume", params={"max_dim": 64, "channels": "1"})
    assert r3.headers["x-channel-luts"] == "Cyan"
