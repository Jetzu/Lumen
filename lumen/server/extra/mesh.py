"""Iso-surface mesh endpoints (drop-in router).

``GET /api/images/{id}/mesh?c=&t=&level=&step=&max_dim=&smooth=&min_faces=``
    Marching cubes on one channel / frame (see ``lumen.core.ops.mesh_ops.extract_mesh``).  Binary body:
    float32 vertices (N×3, image pixel coordinates x, y, z with z scaled by the z/xy anisotropy),
    float32 unit normals (N×3), uint32 faces (M×3, counter-clockwise).  Headers::

        X-Vertices: N
        X-Faces: M
        X-Level: 1234.5              (raw intensity units; resolved from 'auto' / Otsu)
        X-Bounds: x0,y0,z0,x1,y1,z1  (of the vertices)
        X-Extent: W,H,D              (the whole volume box in the same units; D = nz · z/xy ratio)
        X-Spacing, X-Channel, X-Lut, X-Components, X-Stride, X-Image-Rev

    ``level`` omitted / ``level=auto`` → midpoint of the channel's display range; ``level=otsu`` → Otsu threshold;
    any number (0 included) is used as given; anything else → 422.

``GET /api/images/{id}/mesh.{obj|stl|ply}?…&calibrated=1``
    The same surface as a downloadable file (Content-Disposition: attachment, RFC 5987 ``filename*`` for
    non-ASCII names); coordinates in calibrated units (pixel width / height) when the image is scaled
    (``calibrated=0`` keeps pixels).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...core.ops.mesh_ops import FORMATS, Mesh, ascii_name, extract_mesh, level_key, mesh_file_bytes, mesh_scale, surface_name, zratio
from ..app import get_session

router = APIRouter()

_MESH_CACHE: "OrderedDict[tuple[Any, ...], Mesh]" = OrderedDict()
_CACHE_MAX_ENTRIES = 16
_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CACHE_LOCK = threading.Lock()


def _mesh_bytes(m: Mesh) -> int:
    return int(m.vertices.nbytes + m.normals.nbytes + m.faces.nbytes)


def _cache_put(key: tuple[Any, ...], mesh: Mesh) -> None:
    with _CACHE_LOCK:
        _MESH_CACHE[key] = mesh
        _MESH_CACHE.move_to_end(key)
        while len(_MESH_CACHE) > _CACHE_MAX_ENTRIES or sum(_mesh_bytes(m) for m in _MESH_CACHE.values()) > _CACHE_MAX_BYTES:
            if len(_MESH_CACHE) <= 1:
                break
            _MESH_CACHE.popitem(last=False)


def _cache_get(key: tuple[Any, ...]) -> Mesh | None:
    with _CACHE_LOCK:
        m = _MESH_CACHE.get(key)
        if m is not None:
            _MESH_CACHE.move_to_end(key)
        return m


def _clamp_params(img: Any, c: int | None, t: int, step: int, max_dim: int, smooth: int, min_faces: int) -> tuple[int, int, int, int, int, int]:
    if img.nz < 2:
        raise HTTPException(400, "Image has no Z dimension")
    cc = img.display.c if c is None or c < 0 else int(c)
    if cc >= img.nc:
        raise HTTPException(400, f"Channel {cc} does not exist (the image has {img.nc})")
    return (cc, min(max(int(t), 0), img.nt - 1), min(max(int(step), 1), 16), min(max(int(max_dim), 0), 4096),
            min(max(int(smooth), 0), 100), max(int(min_faces), 0))


def get_mesh(img: Any, c: int | None, t: int, level: float | str | None, step: int, max_dim: int, smooth: int, min_faces: int) -> Mesh:
    cc, t, step, max_dim, smooth, min_faces = _clamp_params(img, c, t, step, max_dim, smooth, min_faces)
    try:
        lk = level_key(img, cc, level)
    except ValueError:
        raise HTTPException(422, f"level must be a number, 'auto' or 'otsu' (got {level!r})") from None
    # ``rev`` only tracks pixel data; the z ratio (Properties / Set Scale) changes the geometry without bumping it.
    key = (img.id, img.rev, cc, t, lk, step, max_dim, smooth, min_faces, zratio(img))
    mesh = _cache_get(key)
    if mesh is None:
        mesh = extract_mesh(img, cc, t, level, step, max_dim, smooth, min_faces)
        _cache_put(key, mesh)
    return mesh


def _headers(img: Any, mesh: Mesh) -> dict[str, str]:
    b = mesh.bounds()
    lut = ["Red", "Green", "Blue"][mesh.channel % 3] if img.rgb else str(img.display.channels[mesh.channel].lut)
    return {
        "X-Vertices": str(mesh.n_vertices),
        "X-Faces": str(mesh.n_faces),
        "X-Level": f"{mesh.level:.8g}",
        "X-Bounds": ",".join(f"{v:.4f}" for v in b),
        "X-Extent": ",".join(f"{v:.6g}" for v in mesh.extent),
        "X-Spacing": f"{zratio(img):.6g}",
        "X-Channel": str(mesh.channel),
        "X-Lut": lut,
        "X-Components": str(mesh.components),
        "X-Stride": ",".join(str(s) for s in mesh.stride),
        "X-Image-Rev": str(img.rev),
        "X-Notes": "; ".join(mesh.notes).encode("ascii", "replace").decode("ascii"),
        "Cache-Control": "no-store",
    }


@router.get("/images/{image_id}/mesh")
def mesh(image_id: str, c: int | None = None, t: int = 0, level: str | None = None, step: int = 1, max_dim: int = 192, smooth: int = 0, min_faces: int = 0) -> Response:
    img = get_session().get_image(image_id)
    m = get_mesh(img, c, t, level, step, max_dim, smooth, min_faces)
    return Response(content=m.to_bytes(), media_type="application/octet-stream", headers=_headers(img, m))


@router.get("/images/{image_id}/mesh.{fmt}")
def mesh_file(image_id: str, fmt: str, c: int | None = None, t: int = 0, level: str | None = None, step: int = 1, max_dim: int = 0, smooth: int = 0,
              min_faces: int = 0, calibrated: bool = True) -> Response:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise HTTPException(404, f"Unknown mesh format {fmt!r}; use one of {', '.join(FORMATS)}")
    img = get_session().get_image(image_id)
    m = get_mesh(img, c, t, level, step, max_dim, smooth, min_faces)
    scale, unit = mesh_scale(img, calibrated)  # per-axis (pixel width, height, width): rectangular pixels export correctly
    name = surface_name(img, m.channel)  # Unicode is fine in the body (OBJ is UTF-8; STL / PLY headers sanitise)
    body = mesh_file_bytes(m, fmt, scale, name)
    media = "text/plain; charset=utf-8" if fmt == "obj" else "application/octet-stream"
    unit = unit.replace("µ", "u").replace("μ", "u").encode("ascii", "replace").decode("ascii")
    # response headers are latin-1: ASCII fallback in ``filename=``, the real name via RFC 5987 ``filename*`` when needed
    safe = ascii_name(name)
    disposition = f'attachment; filename="{safe}.{fmt}"'
    if safe != name:
        disposition += f"; filename*=UTF-8''{quote(name)}.{fmt}"
    headers = {**_headers(img, m), "Content-Disposition": disposition, "X-Unit": unit}
    return Response(content=body, media_type=media, headers=headers)
