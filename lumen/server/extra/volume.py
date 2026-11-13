"""Volume rendering and orthogonal-view endpoints (drop-in router).

``GET /api/images/{id}/volume``
    Binary body: for every included channel one uint8 volume of shape (Z', Y', X') with the
    channel's display range (min/max, gamma) applied, downsampled so that
    ``max(Z'·spacing, Y', X') <= max_dim``.  Z is never upsampled beyond the native slice count;
    the residual anisotropy (z voxel size / xy voxel size of the *returned* volume) is sent in
    ``X-Spacing`` so the client can stretch the box.  ``inverted`` is *not* applied to the
    intensities: as in the 2-D views it is a colour-only LUT reversal done by the client
    (the voxel densities drive MIP / opacity / iso thresholds and must stay as they are).
    Headers::

        X-Dtype: uint8
        X-Shape: Z,Y,X                 (per channel)
        X-Channels: n
        X-Channel-Index: 0,1           (source channel indices)
        X-Channel-Luts: Magenta,Green
        X-Spacing: 1.115               (z voxel / xy voxel of the returned volume)
        X-Downsample: 2.0              (xy pixels of the source per voxel of the volume)

``GET /api/images/{id}/ortho?x=&y=&t=``
    Binary body (raw dtype): the XZ cross-section at row ``y`` for all channels, laid out
    (C, Z, X), followed by the YZ cross-section at column ``x`` laid out (C, Z, Y) — i.e. the
    long image axis is horizontal in both planes.  Headers ``X-Dtype``, ``X-Shape-XZ: Z,X``,
    ``X-Shape-YZ: Z,Y``, ``X-Channels``, ``X-Layout``, ``X-Point: x,y``, ``X-Spacing``.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..app import get_session

router = APIRouter()

# (image id, rev, t, max_dim, channel selection, display signature, z ratio) -> (bytes, headers)
# ``rev`` only tracks pixel data; the z ratio is keyed separately because calibration-only
# operations (Properties, Set Scale) change the resampled geometry without bumping ``rev``.
# Display-only state that does not affect the body (LUT names) is added to the headers per request.
_VOLUME_CACHE: "OrderedDict[tuple[Any, ...], tuple[bytes, dict[str, str]]]" = OrderedDict()
_CACHE_MAX_ENTRIES = 8
_CACHE_MAX_BYTES = 512 * 1024 * 1024
_CACHE_LOCK = threading.Lock()


def _cache_put(key: tuple[Any, ...], value: tuple[bytes, dict[str, str]]) -> None:
    with _CACHE_LOCK:
        _VOLUME_CACHE[key] = value
        _VOLUME_CACHE.move_to_end(key)
        while len(_VOLUME_CACHE) > _CACHE_MAX_ENTRIES or sum(len(v[0]) for v in _VOLUME_CACHE.values()) > _CACHE_MAX_BYTES:
            if len(_VOLUME_CACHE) <= 1:
                break
            _VOLUME_CACHE.popitem(last=False)


def _cache_get(key: tuple[Any, ...]) -> tuple[bytes, dict[str, str]] | None:
    with _CACHE_LOCK:
        v = _VOLUME_CACHE.get(key)
        if v is not None:
            _VOLUME_CACHE.move_to_end(key)
        return v


def _channel_indices(img: Any, channels: str) -> list[int]:
    channels = (channels or "all").strip().lower()
    if channels == "all":
        return list(range(img.nc))
    if channels == "current":
        return [min(max(img.display.c, 0), img.nc - 1)]
    out: list[int] = []
    for tok in channels.split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < img.nc:
            out.append(int(tok))
    if not out:
        raise HTTPException(400, f"Invalid channel selection {channels!r}")
    return out


def _zratio(cal: Any) -> float:
    """z voxel size / xy voxel size (1 when uncalibrated or degenerate)."""
    z = float(cal.pixel_depth / cal.pixel_width) if cal.pixel_width and cal.pixel_depth else 1.0
    return z if np.isfinite(z) and z > 0 else 1.0


def _to_uint8(vol: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """Apply a display range to a raw volume → float32 in [0, 1] → uint8."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = 1e-9
    f = vol.astype(np.float32, copy=False)
    f = (f - np.float32(lo)) * np.float32(1.0 / span)
    np.nan_to_num(f, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(f, 0.0, 1.0, out=f)
    if gamma and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        np.power(f, np.float32(1.0 / gamma), out=f)
    return (f * 255.0 + 0.5).astype(np.uint8)


def _resample(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a (Z, Y, X) uint8 volume to ``shape`` (linear, anti-aliased when shrinking)."""
    if tuple(vol.shape) == tuple(shape):
        return vol
    from skimage.transform import resize

    shrinking = any(s < o for s, o in zip(shape, vol.shape))
    out = resize(vol, shape, order=1, preserve_range=True, anti_aliasing=shrinking, mode="edge")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def build_volume(img: Any, t: int, max_dim: int, channel_idx: list[int]) -> tuple[list[np.ndarray], tuple[int, int, int], float, float]:
    """Return (volumes, (Z', Y', X'), spacing, downsample)."""
    zratio = _zratio(img.calibration)
    nz, ny, nx = img.nz, img.height, img.width
    extent = max(nz * zratio, ny, nx)
    f = min(1.0, max_dim / extent)
    stride = max(1, int(1.0 / f)) if f < 1 else 1
    # XY: stride first (cheap), then resample the remainder
    ys, xs = max(1, round(ny * f)), max(1, round(nx * f))
    zs = max(1, min(nz, round(nz * zratio * f)))
    spacing = (nz * zratio * f) / zs  # z voxel / xy voxel of the returned volume
    downsample = 1.0 / f
    vols: list[np.ndarray] = []
    for c in channel_idx:
        ch = img.display.channels[c]
        raw = img.data[t, :, c, ::stride, ::stride]
        v8 = _to_uint8(raw, float(ch.min), float(ch.max), float(ch.gamma or 1.0))
        v8 = _resample(v8, (zs, ys, xs))
        vols.append(np.ascontiguousarray(v8))
    return vols, (zs, ys, xs), spacing, downsample


@router.get("/images/{image_id}/volume")
def volume(image_id: str, t: int = 0, max_dim: int = 256, channels: str = "all") -> Response:
    session = get_session()
    img = session.get_image(image_id)
    if img.nz < 1:
        raise HTTPException(400, "Image has no Z dimension")
    t = min(max(t, 0), img.nt - 1)
    max_dim = int(min(max(max_dim, 16), 1024))
    idx = _channel_indices(img, channels)
    disp_sig = tuple((float(img.display.channels[c].min), float(img.display.channels[c].max), float(img.display.channels[c].gamma or 1.0)) for c in idx)
    key = (img.id, img.rev, t, max_dim, tuple(idx), disp_sig, _zratio(img.calibration))
    hit = _cache_get(key)
    if hit is None:
        vols, shape, spacing, downsample = build_volume(img, t, max_dim, idx)
        body = b"".join(v.tobytes() for v in vols)
        headers = {
            "X-Dtype": "uint8",
            "X-Shape": ",".join(str(s) for s in shape),
            "X-Channels": str(len(vols)),
            "X-Channel-Index": ",".join(str(c) for c in idx),
            "X-Spacing": f"{spacing:.6g}",
            "X-Downsample": f"{downsample:.6g}",
            "X-Image-Rev": str(img.rev),
        }
        hit = (body, headers)
        _cache_put(key, hit)
    body, headers = hit
    # LUT names do not affect the body and are not part of the key: always report the live ones.
    luts = ",".join(str(img.display.channels[c].lut) for c in idx)
    return Response(content=body, media_type="application/octet-stream", headers={**headers, "X-Channel-Luts": luts, "Cache-Control": "no-store"})


@router.get("/images/{image_id}/ortho")
def ortho(image_id: str, x: int | None = None, y: int | None = None, t: int = 0) -> Response:
    session = get_session()
    img = session.get_image(image_id)
    t = min(max(t, 0), img.nt - 1)
    px = img.width // 2 if x is None else min(max(int(x), 0), img.width - 1)
    py = img.height // 2 if y is None else min(max(int(y), 0), img.height - 1)
    xz = np.ascontiguousarray(np.transpose(img.data[t, :, :, py, :], (1, 0, 2)))  # (C, Z, X)
    yz = np.ascontiguousarray(np.transpose(img.data[t, :, :, :, px], (1, 0, 2)))  # (C, Z, Y)
    spacing = _zratio(img.calibration)
    headers = {
        "X-Dtype": str(img.data.dtype),
        "X-Shape-XZ": f"{xz.shape[1]},{xz.shape[2]}",
        "X-Shape-YZ": f"{yz.shape[1]},{yz.shape[2]}",
        "X-Channels": str(img.nc),
        "X-Layout": "XZ=(C,Z,X) rows=z cols=x; YZ=(C,Z,Y) rows=z cols=y; XZ block first",
        "X-Point": f"{px},{py}",
        "X-Spacing": f"{spacing:.6g}",
        "Cache-Control": "no-store",
    }
    return Response(content=xz.tobytes() + yz.tobytes(), media_type="application/octet-stream", headers=headers)


# ============================================================================ 3-D scene (Plugins ▸ 3D Viewer)
# ``GET /api/scene``                       the session's current scene document {scene, rev} (null before the viewer published one)
# ``PUT /api/scene``                       body {"scene": {...}} — the 3-D View pushes its state so scene.save / landmarks work from the palette
# ``POST /api/scene/animation``            multipart: frames (PNG files, in order), name, fps, format ('' | gif | mp4 …), path
#                                          → the frames become an RGB time stack (new image); format + path also write a GIF (PIL)
#                                          or a video (the opt-in [video] extra)
# ``GET /api/images/{id}/animation.gif``   the stack rendered along T (or Z) as an animated GIF download (?fps=&axis=)
# ``POST /api/images/{id}/pick``           ray pick: {origin, direction (pixel coords, z · z-ratio), mode: surface | volume, c, t, level,
#                                          step, max_dim, smooth} → {hit, point, px, calibrated, distance, …}
from fastapi import File, Form, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class SceneBody(BaseModel):
    scene: dict[str, Any]


class PickBody(BaseModel):
    origin: list[float]
    direction: list[float]
    mode: str = "surface"
    c: int | None = None
    t: int = 0
    level: float | str | None = None
    step: int = 1
    max_dim: int = 192
    smooth: int = 0


@router.get("/scene")
def scene_get() -> dict[str, Any]:
    from ...core.ops.scene_ops import SETTINGS_REV, session_scene
    session = get_session()
    return {"scene": session_scene(session), "rev": int(session.settings.get(SETTINGS_REV, 0) or 0)}


@router.put("/scene")
def scene_put(body: SceneBody) -> dict[str, Any]:
    from ...core.ops.scene_ops import normalize_scene, store_scene
    session = get_session()
    sc = normalize_scene(body.scene)
    rev = store_scene(session, sc)
    return {"ok": True, "rev": rev, "objects": len(sc["objects"]), "landmarks": len(sc["landmarks"])}


@router.post("/scene/animation")
async def scene_animation(frames: list[UploadFile] = File(...), name: str = Form("animation"), fps: float = Form(15.0),
                          format: str = Form(""), path: str = Form("")) -> dict[str, Any]:
    from ...core.ops.scene_ops import assemble_animation
    session = get_session()
    blobs = [await f.read() for f in frames]
    blobs = [b for b in blobs if b]
    if not blobs:
        raise HTTPException(400, "No frames were uploaded")
    with session.lock:
        out = assemble_animation(session, blobs, name=name or "animation", fps=float(fps), fmt=format or "", path=path or "")
    return out


@router.get("/images/{image_id}/animation.gif")
def animation_gif(image_id: str, fps: float = 15.0, axis: str = "auto") -> Response:
    import io as _io
    from ...core.io import video_frames
    from ...core.ops.scene_ops import write_gif
    from ...core.ops.mesh_ops import ascii_name
    session = get_session()
    img = session.get_image(image_id)
    frames = video_frames(img, axis if axis in ("auto", "T", "Z") else "auto")
    buf = _io.BytesIO()
    write_gif(frames, buf, fps=float(fps))
    stem = ascii_name(img.name.rsplit(".", 1)[0] if "." in img.name else img.name)
    return Response(content=buf.getvalue(), media_type="image/gif",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.gif"', "X-Frames": str(len(frames)), "Cache-Control": "no-store"})


@router.post("/images/{image_id}/pick")
def pick(image_id: str, body: PickBody) -> dict[str, Any]:
    from ...core.ops.scene_ops import pick_image
    session = get_session()
    img = session.get_image(image_id)
    if len(body.origin) != 3 or len(body.direction) != 3:
        raise HTTPException(400, "origin and direction must have three components")
    mode = "surface" if body.mode == "surface" else "volume"
    if img.nz < 2 and mode == "surface":
        raise HTTPException(400, "Image has no Z dimension")
    mesh = None
    if mode == "surface":
        from .mesh import get_mesh  # the viewer's cached surface: the pick lands on what is shown
        mesh = get_mesh(img, body.c, body.t, body.level, body.step, body.max_dim, body.smooth, 0)
    try:
        return pick_image(img, body.origin, body.direction, mode, body.c, body.t, body.level, body.step, body.max_dim, body.smooth, mesh=mesh)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
