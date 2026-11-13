"""Iso-surface meshes (Plugins ▸ 3D): marching cubes, smoothing, measurement and OBJ / STL / PLY export.

The geometry lives here (pure numpy / scikit-image / scipy) so that the HTTP router in
``lumen.server.extra.mesh`` and the two commands below share one implementation.

Coordinate convention
---------------------
Vertices are returned in *image pixel coordinates*: x in [0, width], y in [0, height] (pixel i covers
[i, i + 1)), z in [0, nz · zratio] where ``zratio = pixel_depth / pixel_width`` — the same box the
volume renderer draws.  Multiply by ``(pixel_width, pixel_height, pixel_width)`` to get calibrated units — z already
carries the depth/width ratio — which is what ``mesh_scale`` returns and the exporters / measurements apply.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..image import LumenImage
from ..registry import OpResult, Param, op

CAT = "Plugins/3D"
FORMATS = ["obj", "stl", "ply"]


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float32, x y z in pixel coordinates (z scaled by the anisotropy)
    normals: np.ndarray  # (N, 3) float32, unit length, pointing away from the bright side
    faces: np.ndarray  # (M, 3) uint32, counter-clockwise seen from outside (right-hand rule = normal)
    level: float  # iso level in raw intensity units
    extent: tuple[float, float, float]  # (width, height, nz · zratio) of the whole volume box
    channel: int = 0
    components: int = 0
    stride: tuple[int, int, int] = (1, 1, 1)  # (z, y, x) voxel strides used before marching cubes
    notes: list[str] = field(default_factory=list)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> tuple[float, ...]:
        if self.n_vertices == 0:
            return (0.0,) * 6
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        return tuple(float(v) for v in (*lo, *hi))

    def to_bytes(self) -> bytes:
        return (np.ascontiguousarray(self.vertices, np.float32).tobytes() + np.ascontiguousarray(self.normals, np.float32).tobytes()
                + np.ascontiguousarray(self.faces, np.uint32).tobytes())


# ------------------------------------------------------------------ helpers
def zratio(img: LumenImage) -> float:
    cal = img.calibration
    z = float(cal.pixel_depth / cal.pixel_width) if cal.pixel_width and cal.pixel_depth else 1.0
    return z if math.isfinite(z) and z > 0 else 1.0


def mesh_strides(nz: int, ny: int, nx: int, zr: float, max_dim: int) -> tuple[int, int, int]:
    """Integer (z, y, x) block sizes so that no axis of the reduced grid exceeds ``max_dim`` voxels (0 = native)."""
    if max_dim <= 0:
        return 1, 1, 1
    extent = max(nz * zr, ny, nx)
    sxy = max(1, int(math.ceil(extent / max_dim))) if extent > max_dim else 1
    sz = max(1, int(round(sxy / zr)))
    sz = max(sz, int(math.ceil(nz / max_dim)))
    return min(sz, nz), min(sxy, ny), min(sxy, nx)


def block_mean(vol: np.ndarray, sz: int, sy: int, sx: int) -> np.ndarray:
    """Anti-aliased integer downsampling of a (Z, Y, X) array → float32 (the trailing remainder is dropped)."""
    if sz == 1 and sy == 1 and sx == 1:
        return vol.astype(np.float32, copy=False)
    nz, ny, nx = vol.shape
    z, y, x = max(1, nz // sz), max(1, ny // sy), max(1, nx // sx)
    v = vol[: z * sz, : y * sy, : x * sx].reshape(z, sz, y, sy, x, sx)
    return v.mean(axis=(1, 3, 5), dtype=np.float32)


def parse_level(level: Any) -> float | str | None:
    """``None``/''/'auto'/'none'/'mid' → ``None``, 'otsu' → ``'otsu'``, anything else → ``float(level)`` (ValueError on garbage).

    A numeric 0 is an ordinary iso level (the zero crossing of signed / filtered data), never an alias for Otsu.
    """
    if isinstance(level, str):
        s = level.strip().lower()
        if s in ("", "auto", "none", "mid"):
            return None
        if s == "otsu":
            return "otsu"
        return float(s)
    return None if level is None else float(level)


def resolve_level(img: LumenImage, c: int, vol: np.ndarray, level: Any) -> float:
    """``None``/'auto' → display-range midpoint, 'otsu' → Otsu on the volume, else the number itself (0 included)."""
    ch = img.display.channels[c]
    lv = parse_level(level)
    if lv is None:
        return float(ch.min + 0.5 * (ch.max - ch.min))
    if lv == "otsu":
        from ..threshold import auto_threshold
        return float(auto_threshold(np.asarray(vol, np.float32), "Otsu")[0])
    return float(lv)


def level_key(img: LumenImage, c: int, level: Any) -> tuple[Any, ...]:
    """Cache-key form of a level request (the 'auto' default follows the display range)."""
    ch = img.display.channels[c]
    lv = parse_level(level)
    if lv is None:
        return ("auto", float(ch.min), float(ch.max))
    return ("otsu",) if lv == "otsu" else (float(lv),)


def _adjacency(n: int, faces: np.ndarray):
    from scipy import sparse
    i = faces[:, [0, 1, 2]].ravel()
    j = faces[:, [1, 2, 0]].ravel()
    a = sparse.coo_matrix((np.ones(2 * len(i), np.float32), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def smooth_mesh(verts: np.ndarray, faces: np.ndarray, iterations: int, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin λ|μ smoothing (a Laplacian step followed by an inflating step: no shrinkage)."""
    if iterations <= 0 or len(faces) == 0 or len(verts) == 0:
        return verts
    a = _adjacency(len(verts), faces)
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    v = verts.astype(np.float64)
    for _ in range(int(iterations)):
        for k in (lam, mu):
            v = v + k * ((a @ v) / deg[:, None] - v)
    return v.astype(np.float32)


def face_labels(verts: np.ndarray, faces: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the surface → (count, label per face)."""
    if len(faces) == 0:
        return 0, np.zeros(0, np.int64)
    from scipy.sparse.csgraph import connected_components
    n, labels = connected_components(_adjacency(len(verts), faces), directed=False)
    return int(n), labels[faces[:, 0]]


def remove_small_components(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, min_faces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop components with fewer than ``min_faces`` triangles; returns (verts, normals, faces, n_components_kept)."""
    n, lab = face_labels(verts, faces)
    if n == 0:
        return verts, normals, faces, 0
    if min_faces <= 1:
        return verts, normals, faces, n
    counts = np.bincount(lab, minlength=n)
    keep = counts[lab] >= min_faces
    faces = faces[keep]
    if len(faces) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), 0
    used, inv = np.unique(faces.ravel(), return_inverse=True)
    return verts[used], normals[used], inv.reshape(-1, 3).astype(np.uint32), int((counts >= min_faces).sum())


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from the (counter-clockwise) face winding."""
    out = np.zeros((len(verts), 3), np.float64)
    if len(faces):
        v0, v1, v2 = verts[faces[:, 0]].astype(np.float64), verts[faces[:, 1]].astype(np.float64), verts[faces[:, 2]].astype(np.float64)
        fn = np.cross(v1 - v0, v2 - v0)
        for k in range(3):
            np.add.at(out, faces[:, k], fn)
    ln = np.linalg.norm(out, axis=1)
    ln[ln == 0] = 1.0
    return (out / ln[:, None]).astype(np.float32)


# ------------------------------------------------------------------ extraction
def extract_mesh(img: LumenImage, c: int, t: int = 0, level: Any = None, step: int = 1, max_dim: int = 0, smooth: int = 0, min_faces: int = 0) -> Mesh:
    """Marching cubes on channel ``c`` / frame ``t`` → a closed, consistently oriented triangle mesh."""
    from skimage.measure import marching_cubes

    c = min(max(int(c), 0), img.nc - 1)
    t = min(max(int(t), 0), img.nt - 1)
    step = min(max(int(step), 1), 16)
    zr = zratio(img)
    nz, ny, nx = img.nz, img.height, img.width
    extent = (float(nx), float(ny), float(nz * zr))
    sz, sy, sx = mesh_strides(nz, ny, nx, zr, int(max_dim))
    vol = block_mean(img.data[t, :, c], sz, sy, sx)
    lvl = resolve_level(img, c, vol, level)
    notes: list[str] = []
    if (sz, sy, sx) != (1, 1, 1):
        notes.append(f"volume reduced x{sx} in xy, x{sz} in z ({vol.shape[2]}x{vol.shape[1]}x{vol.shape[0]} voxels)")
    empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), lvl, extent, c, 0, (sz, sy, sx), notes)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0 or not math.isfinite(lvl):
        return empty
    lo, hi = float(finite.min()), float(finite.max())
    if lvl >= hi:
        empty.notes.append("level is above the maximum intensity: no surface")
        return empty
    # pad with `step` voxels of a value below the level so every surface is closed (needed for a meaningful enclosed
    # volume): marching cubes with step_size=k only visits cells whose corners sit at index multiples of k, so a
    # 1-voxel pad is skipped (far face left open) whenever (padded_dim - 1) % k != 0.  With pad = k the far pad slab
    # spans k consecutive indices and therefore always contains a corner.
    pad_val = min(lo, lvl) - max(1.0, abs(lvl) * 1e-3)
    padded = np.pad(np.nan_to_num(vol, nan=pad_val, posinf=hi, neginf=pad_val), step, constant_values=pad_val)
    spacing = (float(sz * zr), float(sy), float(sx))
    if padded.shape[0] < 2 or padded.shape[1] < 2 or padded.shape[2] < 2:
        return empty
    try:
        verts, faces, normals, _ = marching_cubes(padded, level=lvl, spacing=spacing, step_size=step, allow_degenerate=False)
    except (ValueError, RuntimeError) as e:  # level outside the data range, degenerate volume
        empty.notes.append(str(e))
        return empty
    if len(faces) == 0:
        return empty
    # (z, y, x) → (x, y, z); un-pad (`step` voxels) and move to block centres: voxel i ↦ (i + 0.5) · stride
    xyz = np.empty_like(verts, dtype=np.float32)
    xyz[:, 0] = (verts[:, 2] - step * spacing[2]) + 0.5 * spacing[2]
    xyz[:, 1] = (verts[:, 1] - step * spacing[1]) + 0.5 * spacing[1]
    xyz[:, 2] = (verts[:, 0] - step * spacing[0]) + 0.5 * spacing[0]
    nrm = np.empty_like(normals, dtype=np.float32)
    nrm[:, 0], nrm[:, 1], nrm[:, 2] = normals[:, 2], normals[:, 1], normals[:, 0]
    # scikit-image's normals are intensity gradients on the *index* grid (``spacing`` scales the vertices only);
    # a gradient transforms by the inverse of the vertex scaling, so divide by the per-axis (x, y, z) spacing —
    # otherwise anisotropic z (sz · zr ≠ sx) tilts every normal toward the z axis.  Renormalised below.
    nrm /= np.asarray([spacing[2], spacing[1], spacing[0]], np.float32)
    # scikit-image's winding is clockwise about its normals (descent = outward for bright objects) in (z, y, x);
    # the axis reorder above is a reflection, so in (x, y, z) the same index order is counter-clockwise.
    faces = np.ascontiguousarray(faces.astype(np.uint32))
    xyz, nrm, faces, ncomp = remove_small_components(xyz, nrm, faces, int(min_faces))
    if len(faces) == 0:
        empty.notes.append("every component was smaller than the minimum size")
        return empty
    if smooth > 0:
        xyz = smooth_mesh(xyz, faces, int(smooth))
        nrm = vertex_normals(xyz, faces)
    else:
        ln = np.linalg.norm(nrm, axis=1)
        bad = ~np.isfinite(ln) | (ln < 1e-6)
        if bad.any():
            nrm[bad] = vertex_normals(xyz, faces)[bad]
        ln[ln < 1e-6] = 1.0
        nrm = (nrm / ln[:, None]).astype(np.float32)
    return Mesh(np.ascontiguousarray(xyz), np.ascontiguousarray(nrm), faces, lvl, extent, c, ncomp, (sz, sy, sx), notes)


# ------------------------------------------------------------------ measurement
Scale = float | tuple[float, float, float]


def _scale3(scale: Scale, dtype=np.float64) -> np.ndarray:
    """A scalar or per-axis (x, y, z) scale → a (3,) array."""
    s = np.broadcast_to(np.asarray(scale, dtype), (3,)).copy()
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"mesh scale must be positive and finite, got {scale!r}")
    return s


def scaled_normals(mesh: Mesh, scale: Scale) -> np.ndarray:
    """``mesh.normals`` expressed in the frame of ``mesh.vertices * scale`` (inverse-transpose for anisotropic scales)."""
    s = _scale3(scale, np.float32)
    if s[0] == s[1] == s[2] or mesh.n_vertices == 0:
        return mesh.normals
    n = mesh.normals / s
    ln = np.linalg.norm(n, axis=1)
    ln[ln < 1e-12] = 1.0
    return (n / ln[:, None]).astype(np.float32)


def measure_components(mesh: Mesh, scale: Scale = 1.0) -> list[dict[str, Any]]:
    """Per connected component: surface area, enclosed volume (divergence theorem), counts, bounds, centroid, orientation.

    ``scale`` converts pixel units to calibrated units: a scalar or a per-axis ``(x, y, z)`` factor (rectangular pixels).
    Everything is measured on the scaled vertices, so area and volume are exact for anisotropic scales too.
    ``orientation`` is 'outward' for a surface enclosing bright material and 'inward' for a cavity / dark inclusion.
    """
    verts, faces = mesh.vertices.astype(np.float64) * _scale3(scale), mesh.faces
    n, lab = face_labels(mesh.vertices, faces)
    rows: list[dict[str, Any]] = []
    if n == 0:
        return rows
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    tri_svol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    tri_centroid = (v0 + v1 + v2) / 3.0
    for k in range(n):
        sel = lab == k
        if not sel.any():
            continue
        fk = faces[sel]
        area = float(tri_area[sel].sum())
        svol = float(tri_svol[sel].sum())
        vid = np.unique(fk.ravel())
        pts = verts[vid]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        w = tri_area[sel]
        cen = (tri_centroid[sel] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
        rows.append({
            "component": k + 1, "vertices": int(len(vid)), "faces": int(sel.sum()),
            "area": area, "volume": abs(svol), "orientation": "outward" if svol >= 0 else "inward",
            "x": float(cen[0]), "y": float(cen[1]), "z": float(cen[2]),
            "bx": float(lo[0]), "by": float(lo[1]), "bz": float(lo[2]),
            "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]), "depth": float(hi[2] - lo[2]),
        })
    rows.sort(key=lambda r: (-r["volume"], -r["area"]))
    return rows


# ------------------------------------------------------------------ writers
def _ensure_ext(path: str, fmt: str, default_name: str) -> Path:
    p = Path(path).expanduser()
    if p.is_dir() or str(path).endswith(("/", "\\")):
        p = p / f"{default_name}.{fmt}"
    elif p.suffix.lower() != f".{fmt}":
        p = p.with_name(p.name + f".{fmt}")
    return p


def obj_text(mesh: Mesh, scale: Scale = 1.0, name: str = "surface") -> str:
    """Wavefront OBJ (vertices, vertex normals, 1-based faces)."""
    buf = io.StringIO()
    buf.write(f"# Lumen iso-surface  level={mesh.level:.6g}  vertices={mesh.n_vertices}  faces={mesh.n_faces}\n")
    buf.write(f"o {name}\n")
    if mesh.n_vertices:
        np.savetxt(buf, mesh.vertices * _scale3(scale, np.float32), fmt="v %.5f %.5f %.5f")
        np.savetxt(buf, scaled_normals(mesh, scale), fmt="vn %.5f %.5f %.5f")
    if mesh.n_faces:
        f = mesh.faces.astype(np.int64) + 1
        np.savetxt(buf, np.column_stack([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]]), fmt="f %d//%d %d//%d %d//%d")
    return buf.getvalue()


def stl_bytes(mesh: Mesh, scale: Scale = 1.0, name: str = "surface") -> bytes:
    """Binary STL (80-byte header, uint32 triangle count, 50 bytes per triangle)."""
    n = mesh.n_faces
    header = f"Lumen iso-surface {name} level={mesh.level:.6g}".encode("ascii", "replace")[:80].ljust(80, b"\0")
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tri = np.zeros(n, dt)
    if n:
        v = mesh.vertices * _scale3(scale, np.float32)
        v0, v1, v2 = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(fn, axis=1)
        ln[ln == 0] = 1.0
        tri["n"] = (fn / ln[:, None]).astype(np.float32)
        tri["v"][:, 0], tri["v"][:, 1], tri["v"][:, 2] = v0, v1, v2
    return header + struct.pack("<I", n) + tri.tobytes()


def ply_bytes(mesh: Mesh, scale: Scale = 1.0, name: str = "surface") -> bytes:
    """Binary little-endian PLY with per-vertex normals (the header is ASCII by specification: the name is sanitised)."""
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment Lumen iso-surface {ascii_name(name)} level={mesh.level:.6g}\n"
              f"element vertex {mesh.n_vertices}\nproperty float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              f"element face {mesh.n_faces}\nproperty list uchar uint vertex_indices\nend_header\n").encode("ascii", "replace")
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
    vb = np.zeros(mesh.n_vertices, vdt)
    if mesh.n_vertices:
        v = mesh.vertices * _scale3(scale, np.float32)
        nrm = scaled_normals(mesh, scale)
        vb["x"], vb["y"], vb["z"] = v[:, 0], v[:, 1], v[:, 2]
        vb["nx"], vb["ny"], vb["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    fdt = np.dtype([("n", "u1"), ("i", "<u4", (3,))])
    fb = np.zeros(mesh.n_faces, fdt)
    if mesh.n_faces:
        fb["n"] = 3
        fb["i"] = mesh.faces
    return header + vb.tobytes() + fb.tobytes()


def mesh_file_bytes(mesh: Mesh, fmt: str, scale: Scale = 1.0, name: str = "surface") -> bytes:
    fmt = (fmt or "obj").lower().lstrip(".")
    if fmt == "obj":
        return obj_text(mesh, scale, name).encode("utf-8")
    if fmt == "stl":
        return stl_bytes(mesh, scale, name)
    if fmt == "ply":
        return ply_bytes(mesh, scale, name)
    raise ValueError(f"Unknown mesh format {fmt!r}; choose from {FORMATS}")


def write_mesh(mesh: Mesh, path: str, fmt: str = "obj", scale: Scale = 1.0, name: str = "surface") -> str:
    p = _ensure_ext(path, (fmt or "obj").lower(), name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(mesh_file_bytes(mesh, fmt, scale, name))
    return str(p)


def surface_name(img: LumenImage, c: int) -> str:
    base = Path(img.name).stem or "image"
    if img.rgb:
        ch = ["R", "G", "B"][c % 3]
    else:
        ch = (img.display.channels[c].name or f"C{c + 1}") if c < len(img.display.channels) else f"C{c + 1}"
    safe = "".join(ch_ if (ch_.isalnum() or ch_ in "-_") else "_" for ch_ in f"{base}_{ch}")
    return safe.strip("_") or "surface"


def ascii_name(name: str) -> str:
    """ASCII-only form of a surface name for HTTP headers and ASCII file headers (never '?', which Windows forbids)."""
    safe = "".join(ch_ if (ch_.isascii() and (ch_.isalnum() or ch_ in "-_.")) else "_" for ch_ in name)
    return safe.strip("_") or "surface"


def mesh_scale(img: LumenImage, calibrated: bool) -> tuple[tuple[float, float, float], str]:
    """Per-axis (x, y, z) factor from mesh pixel coordinates to calibrated units, and the unit name.

    Mesh z is already in xy-pixel units (``k · pixel_depth / pixel_width``), so the z factor is ``pixel_width``;
    y uses ``pixel_height`` so rectangular pixels measure and export correctly.
    """
    cal = img.calibration
    if calibrated and cal.scaled:
        pw = float(cal.pixel_width)
        ph = float(cal.pixel_height) if cal.pixel_height and math.isfinite(cal.pixel_height) and cal.pixel_height > 0 else pw
        return (pw, ph, pw), cal.unit
    return (1.0, 1.0, 1.0), "pixel"


# ------------------------------------------------------------------ commands
_COMMON = [
    Param("level", "string", "auto", help="Iso level in raw intensity units (0 is a valid level); 'auto' = midpoint of the channel's display range, 'otsu' = Otsu threshold"),
    Param("channel", "int", -1, min=-1, help="Channel index (0-based); -1 = the current channel"),
    Param("step", "int", 1, min=1, max=16, label="Marching-cubes step (voxels)", help="Sample every n-th voxel (1 = finest surface)"),
    Param("smooth", "int", 0, min=0, max=100, label="Smoothing iterations", help="Taubin (shrink-free Laplacian) smoothing passes"),
    Param("max_dim", "int", 0, min=0, max=4096, advanced=True, label="Max. voxels per axis", help="Downsample the volume so that no axis exceeds this many voxels (0 = native resolution)"),
    Param("min_faces", "int", 0, min=0, advanced=True, label="Min. faces per component", help="Drop connected surface pieces with fewer triangles (0 = keep everything)"),
    Param("calibrated", "bool", True, help="Coordinates in calibrated units (pixel width / height / depth) instead of pixels"),
]


def _channel_index(img: LumenImage, channel: int) -> int:
    c = img.display.c if channel is None or int(channel) < 0 else int(channel)
    if c >= img.nc:
        raise ValueError(f"Channel {c} does not exist (the image has {img.nc})")
    return c


@op("mesh.export", "Export Surface (OBJ/STL)", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False,
    params=[Param("path", "string", "", help="Output file (a directory gets <image>_<channel>.<format> inside); the extension follows the format"),
            Param("format", "choice", "obj", choices=FORMATS)] + _COMMON,
    description="Extract the iso-surface of a channel of the current Z stack (marching cubes) and write it as a Wavefront OBJ, binary STL or binary PLY mesh.",
    keywords=["mesh", "surface", "iso", "marching cubes", "3d", "obj", "stl", "ply", "export"])
def mesh_export(ctx, path="", format="obj", level="auto", channel=-1, step=1, smooth=0, max_dim=0, min_faces=0, calibrated=True):
    img = ctx.img
    if img.nz < 2:
        raise ValueError("Export Surface needs a Z stack")
    if not str(path).strip():
        raise ValueError("Give an output path")
    c = _channel_index(img, channel)
    mesh = extract_mesh(img, c, img.display.t, level, step, max_dim, smooth, min_faces)
    scale, unit = mesh_scale(img, calibrated)
    name = surface_name(img, c)
    out = write_mesh(mesh, str(path), format, scale, name)
    msg = f"Wrote {mesh.n_vertices} vertices / {mesh.n_faces} faces (level {mesh.level:.6g}, {unit}) → {out}"
    if mesh.notes:
        msg += "; " + "; ".join(mesh.notes)
    return OpResult(message=msg, log=msg)


@op("mesh.measure", "Measure Surface (3D)", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False,
    params=_COMMON[:6] + [Param("max_rows", "int", 200, min=1, max=100000, advanced=True, help="Report at most this many components (largest first)")],
    description="Iso-surface statistics per connected component: surface area, enclosed volume (divergence theorem), orientation "
                "('inward' marks cavities / dark inclusions), vertex / face counts, centroid and bounding box → Results "
                "(3D Viewer ▸ Measure surface). The summary volume is the material volume: outward surfaces minus the cavities they enclose.",
    keywords=["mesh", "surface area", "volume", "3d", "iso", "measure"])
def mesh_measure(ctx, level="auto", channel=-1, step=1, smooth=0, max_dim=0, min_faces=0, max_rows=200):
    img = ctx.img
    if img.nz < 2:
        raise ValueError("Measure Surface needs a Z stack")
    c = _channel_index(img, channel)
    mesh = extract_mesh(img, c, img.display.t, level, step, max_dim, smooth, min_faces)
    scale, unit = mesh_scale(img, True)
    comps = measure_components(mesh, scale)
    label = f"{img.name}:{surface_name(img, c).split('_')[-1]}"
    rows = []
    for r in comps[: int(max_rows)]:
        rows.append({"Label": f"{label} #{r['component']}", "Channel": c + 1, "Level": mesh.level, "Vertices": r["vertices"], "Faces": r["faces"],
                     "Area": r["area"], "Volume": r["volume"], "Orientation": r["orientation"], "Unit": unit, "X": r["x"], "Y": r["y"], "Z": r["z"],
                     "BX": r["bx"], "BY": r["by"], "BZ": r["bz"], "Width": r["width"], "Height": r["height"], "Depth": r["depth"]})
    total_area = sum(r["area"] for r in comps)
    # material volume = signed divergence-theorem sum over the whole (closed, consistently oriented) mesh:
    # outward shells add, the cavities they enclose subtract (a bright inclusion inside a cavity is outward again)
    outward = sum(r["volume"] for r in comps if r["orientation"] == "outward")
    inward = sum(r["volume"] for r in comps if r["orientation"] == "inward")
    n_cav = sum(1 for r in comps if r["orientation"] == "inward")
    total_vol = outward - inward
    vol_txt = f"volume {total_vol:.6g} {unit}³"
    if n_cav:
        vol_txt += f" ({outward:.6g} enclosed − {inward:.6g} in {n_cav} cavit{'y' if n_cav == 1 else 'ies'})"
    msg = (f"{len(comps)} surface component{'s' if len(comps) != 1 else ''} at level {mesh.level:.6g}: area {total_area:.6g} {unit}², "
           f"{vol_txt} ({mesh.n_vertices} vertices, {mesh.n_faces} faces)")
    if len(comps) > len(rows):
        msg += f"; only the {len(rows)} largest listed"
    if mesh.notes:
        msg += "; " + "; ".join(mesh.notes)
    return OpResult(results=rows if rows else None, message=msg, log=msg)
