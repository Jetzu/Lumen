"""3-D scenes (Plugins ▸ 3D Viewer): scene JSON, landmarks, ray picking and animation assembly.

The interactive part of the 3D Viewer lives in the browser (``frontend/src/render/scene.ts`` and the 3-D View
section); this module holds everything that is file / table / session related so that it is testable without a GPU:

* the **scene document** (``normalize_scene``): objects (open images shown as volume / orthoslices / surface with
  colour, transparency, threshold, visibility and a rigid transform), camera, lights, background, clipping box and
  landmarks — saved with ``scene.save`` and read back with ``scene.load``;
* **landmarks** placed on surfaces in the viewer → Results rows (calibrated coordinates), a pairwise distance table
  and point ROIs with a slice position in the ROI Manager (``scene.landmarks_to_results``);
* **ray picking** helpers (``ray_triangles``: Möller–Trumbore against a mesh, ``first_voxel_above``: the first voxel
  above a level along a ray through a volume) behind ``POST /api/images/{id}/pick``;
* **animation assembly**: PNG frames recorded from the WebGL canvas → an RGB time stack (new image) and optionally an
  animated GIF (PIL) or a video (``[video]`` extra).

Coordinate convention (same as ``mesh_ops``): x, y in image pixels, z in *slices* for landmarks (``px``) and in
xy-pixel units (``z · pixel_depth / pixel_width``) for rays through meshes / volumes.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..image import Calibration, LumenImage
from ..registry import OpResult, Param, op

CAT = "Plugins/3D"
SCENE_VERSION = 1
DISPLAY_MODES = ["volume", "orthoslice", "surface"]
VOLUME_MODES = ["mip", "alpha", "iso"]
PROJECTIONS = ["perspective", "orthographic"]
SETTINGS_KEY = "_scene"  # session.settings entry (underscore: hidden from GET /api/settings)
SETTINGS_REV = "_scene_rev"


# ------------------------------------------------------------------ small coercions
def _num(v: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(f):
        return float(default)
    if lo is not None:
        f = max(lo, f)
    if hi is not None:
        f = min(hi, f)
    return f


def _vec(v: Any, default: tuple[float, ...], lo: float | None = None, hi: float | None = None) -> list[float]:
    n = len(default)
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return [float(x) for x in default]
    return [_num(x, d, lo, hi) for x, d in zip(v, default)]


def _color(v: Any) -> list[float] | None:
    """``[r, g, b]`` floats 0..1, ``'#rrggbb'`` or ``None`` (object colour follows its LUT)."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        s = v.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            return None
        try:
            return [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        except ValueError:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 3:
        c = [_num(x, 1.0) for x in v]
        if max(c) > 1.0:  # 0..255 bytes
            c = [x / 255.0 for x in c]
        return [min(1.0, max(0.0, x)) for x in c]
    return None


def _choice(v: Any, choices: list[str], default: str) -> str:
    s = str(v).strip().lower() if v is not None else ""
    return s if s in choices else default


# ------------------------------------------------------------------ the scene document
def default_transform() -> dict[str, Any]:
    return {"translate": [0.0, 0.0, 0.0], "rotate": [0.0, 0.0, 0.0], "scale": 1.0}


def normalize_transform(t: Any) -> dict[str, Any]:
    """Rigid transform: ``translate`` (calibrated units), ``rotate`` (degrees about x, y, z) and a uniform ``scale``."""
    t = t if isinstance(t, dict) else {}
    return {"translate": _vec(t.get("translate"), (0.0, 0.0, 0.0)), "rotate": _vec(t.get("rotate"), (0.0, 0.0, 0.0)),
            "scale": _num(t.get("scale"), 1.0, 1e-3, 1e3)}


def normalize_object(o: Any, index: int = 0) -> dict[str, Any]:
    o = o if isinstance(o, dict) else {}
    slices = o.get("slices") if isinstance(o.get("slices"), dict) else {}
    show = slices.get("show")
    if not isinstance(show, (list, tuple)) or len(show) != 3:
        show = [False, False, False]
    channel = o.get("channel")
    try:
        channel = None if channel is None or channel == "" or int(channel) < 0 else int(channel)
    except (TypeError, ValueError):
        channel = None
    return {
        "id": str(o.get("id") or f"obj{index + 1}"),
        "image": str(o.get("image") or ""),
        "name": str(o.get("name") or ""),
        "channel": channel,
        "display": _choice(o.get("display"), DISPLAY_MODES, "volume"),
        "mode": _choice(o.get("mode"), VOLUME_MODES, "mip"),
        "color": _color(o.get("color")),
        "transparency": _num(o.get("transparency"), 0.0, 0.0, 1.0),
        "threshold": _num(o.get("threshold"), 0.12, 0.0, 1.0),
        "level": _num(o.get("level"), 0.4, 0.0, 1.0),
        "brightness": _num(o.get("brightness"), 1.0, 0.05, 5.0),
        "visible": bool(o.get("visible", True)),
        "transform": normalize_transform(o.get("transform")),
        "slices": {"pos": _vec(slices.get("pos"), (0.5, 0.5, 0.5), 0.0, 1.0), "show": [bool(s) for s in show]},
        "missing": bool(o.get("missing", False)),
    }


def normalize_landmark(lm: Any, index: int = 0) -> dict[str, Any]:
    lm = lm if isinstance(lm, dict) else {}
    px = lm.get("px")
    if px is None and all(k in lm for k in ("x", "y", "z")):
        px = [lm["x"], lm["y"], lm["z"]]
    out = {"id": str(lm.get("id") or f"lm{index + 1}"), "name": str(lm.get("name") or f"L{index + 1}"),
           "image": str(lm.get("image") or ""), "px": _vec(px, (0.0, 0.0, 0.0))}
    if isinstance(lm.get("cal"), (list, tuple)) and len(lm["cal"]) == 3:
        out["cal"] = _vec(lm["cal"], (0.0, 0.0, 0.0))
    if lm.get("color") is not None:
        out["color"] = _color(lm.get("color"))
    return out


def default_scene() -> dict[str, Any]:
    return normalize_scene({})


def normalize_scene(scene: Any) -> dict[str, Any]:
    """Coerce a scene document (a dict or JSON text) to the canonical form; raises ``ValueError`` on garbage."""
    if isinstance(scene, (str, bytes)):
        try:
            scene = json.loads(scene or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Scene JSON is not valid: {e}") from e
    if scene is None:
        scene = {}
    if not isinstance(scene, dict):
        raise ValueError("A scene must be a JSON object")
    cam = scene.get("camera") if isinstance(scene.get("camera"), dict) else {}
    lights = scene.get("lights") if isinstance(scene.get("lights"), dict) else {}
    clip = scene.get("clipping") if isinstance(scene.get("clipping"), dict) else {}
    objs = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    lms = scene.get("landmarks") if isinstance(scene.get("landmarks"), list) else []
    return {
        "version": SCENE_VERSION,
        "primary": str(scene.get("primary") or ""),
        "primary_name": str(scene.get("primary_name") or ""),
        "objects": [normalize_object(o, i) for i, o in enumerate(objs)],
        "camera": {
            "theta": _num(cam.get("theta"), 0.55), "phi": _num(cam.get("phi"), 0.38, -1.5698, 1.5698),
            "roll": _num(cam.get("roll"), 0.0), "zoom": _num(cam.get("zoom"), 1.0, 0.05, 50.0),
            "target": _vec(cam.get("target"), (0.0, 0.0, 0.0)),
            "projection": _choice(cam.get("projection"), PROJECTIONS, "perspective"),
        },
        "lights": {
            "ambient": _num(lights.get("ambient"), 0.22, 0.0, 1.0), "diffuse": _num(lights.get("diffuse"), 0.78, 0.0, 2.0),
            "specular": _num(lights.get("specular"), 0.3, 0.0, 2.0),
            "direction": _vec(lights.get("direction"), (0.35, 0.5, 0.8)),
            "headlight": bool(lights.get("headlight", True)),
        },
        "background": _color(scene.get("background")),
        "clipping": {"min": _vec(clip.get("min"), (0.0, 0.0, 0.0), 0.0, 1.0), "max": _vec(clip.get("max"), (1.0, 1.0, 1.0), 0.0, 1.0)},
        "landmarks": [normalize_landmark(lm, i) for i, lm in enumerate(lms)],
        "mode": _choice(scene.get("mode"), VOLUME_MODES + ["mesh"], "mip"),
        "show_box": bool(scene.get("show_box", True)),
    }


def resolve_scene_images(scene: dict[str, Any], images: dict[str, LumenImage]) -> dict[str, Any]:
    """Point every object / landmark at an open image: by id, else by name; otherwise mark it ``missing``.

    Landmarks carry the id of the image they were placed on (the primary or a scene object); when that id is stale (a
    later session) they follow the object that was re-resolved by name, so their ``px`` stay in the right frame."""
    by_name = {im.name: im.id for im in images.values()}

    def find(iid: str, name: str) -> str | None:
        if iid in images:
            return iid
        return by_name.get(name) if name else None

    out = dict(scene)
    remap: dict[str, str] = {}  # saved image id → open image id (the primary and every resolved object)
    known: set[str] = set()  # every image id some scene object refers to (open or not)
    if scene.get("primary") or scene.get("primary_name"):
        pid = find(scene.get("primary", ""), scene.get("primary_name", ""))
        if pid:
            out["primary"] = pid
            out["primary_name"] = images[pid].name
            if scene.get("primary"):
                remap[scene["primary"]] = pid
    objs = []
    for o in scene["objects"]:
        o = dict(o)
        if o["image"]:
            known.add(o["image"])
        iid = find(o["image"], o["name"])
        if iid:
            if o["image"]:
                remap.setdefault(o["image"], iid)
            o["image"], o["name"], o["missing"] = iid, images[iid].name, False
        else:
            o["missing"] = True
        objs.append(o)
    out["objects"] = objs
    lms = []
    for lm in scene["landmarks"]:
        lm = dict(lm)
        old = lm["image"]
        if old and old not in images:
            if old in remap:
                lm["image"] = remap[old]  # the object it was placed on, under its new id
            elif old not in known:
                lm["image"] = out.get("primary", "") or old  # no owning object in the file: assume the primary
            # else: its object is known but not open → keep the stale id so the viewer skips it instead of drawing it on the primary
        lms.append(lm)
    out["landmarks"] = lms
    return out


def scene_to_json(scene: dict[str, Any]) -> str:
    return json.dumps(normalize_scene(scene), indent=1, ensure_ascii=False)


def scene_rows(scene: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    cols = ["Object", "Image", "Display", "Mode", "Color", "Transparency", "Threshold", "Visible", "Translate", "Rotate", "Scale", "Status"]
    rows = []
    for o in scene["objects"]:
        t = o["transform"]
        col = "LUT" if o["color"] is None else "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in o["color"])
        rows.append([o["id"], o["name"] or o["image"], o["display"], o["mode"], col, o["transparency"], o["threshold"], o["visible"],
                     ", ".join(f"{v:g}" for v in t["translate"]), ", ".join(f"{v:g}" for v in t["rotate"]), t["scale"],
                     "missing" if o.get("missing") else "ok"])
    return cols, rows


def _scene_path(path: str) -> Path:
    p = Path(str(path or "").strip()).expanduser()
    if not str(p) or str(p) == ".":
        raise ValueError("Give a scene file path")
    if p.is_dir():
        p = p / "scene.json"
    elif p.suffix.lower() != ".json":
        p = p.with_name(p.name + ".json")
    return p


def session_scene(session: Any) -> dict[str, Any] | None:
    sc = session.settings.get(SETTINGS_KEY)
    return sc if isinstance(sc, dict) else None


def store_scene(session: Any, scene: dict[str, Any]) -> int:
    """Keep ``scene`` as the session's current 3-D scene and bump its revision (the viewer polls it after scene.load)."""
    session.settings[SETTINGS_KEY] = scene
    rev = int(session.settings.get(SETTINGS_REV, 0) or 0) + 1
    session.settings[SETTINGS_REV] = rev
    return rev


# ------------------------------------------------------------------ landmarks
def _cal_factors(img: LumenImage) -> tuple[float, float, float, str]:
    cal: Calibration = img.calibration
    if cal.scaled:
        pw = float(cal.pixel_width)
        ph = float(cal.pixel_height) if cal.pixel_height and math.isfinite(cal.pixel_height) and cal.pixel_height > 0 else pw
        pd = float(cal.pixel_depth) if cal.pixel_depth and math.isfinite(cal.pixel_depth) and cal.pixel_depth > 0 else pw
        return pw, ph, pd, cal.unit
    return 1.0, 1.0, 1.0, "pixel"


def landmark_calibrated(img: LumenImage, lm: dict[str, Any]) -> tuple[float, float, float]:
    """Calibrated (x, y, z) of a landmark: its own ``cal`` when the viewer supplied one, else ``px`` × the voxel size."""
    if "cal" in lm:
        return tuple(float(v) for v in lm["cal"])  # type: ignore[return-value]
    pw, ph, pd, _ = _cal_factors(img)
    x, y, z = lm["px"]
    return float(x) * pw, float(y) * ph, float(z) * pd


def _landmark_image(img: LumenImage, lm: dict[str, Any], images: dict[str, LumenImage] | None) -> LumenImage:
    return (images or {}).get(lm.get("image", ""), img) if images else img


def _landmark_frame(img: LumenImage, im: LumenImage, lm: dict[str, Any]) -> str:
    """Which calibrated frame a landmark's coordinates are in: the primary's (``img``) when the viewer supplied ``cal``
    (``worldToCalibrated`` of the primary) or the point lies on the primary itself, else its own image's px × voxel frame."""
    return "primary" if "cal" in lm or im is img else im.id


def landmark_rows(img: LumenImage, landmarks: list[dict[str, Any]], images: dict[str, LumenImage] | None = None) -> list[dict[str, Any]]:
    """One Results row per landmark: name, calibrated coordinates, pixel coordinates and the (1-based) slice.

    ``img`` is the primary image of the viewer: a viewer-supplied ``cal`` is in *its* calibrated frame and unit, the
    px × voxel fallback in the landmark image's own unit."""
    rows = []
    primary_unit = _cal_factors(img)[3]
    for i, lm in enumerate(landmarks):
        im = _landmark_image(img, lm, images)
        cx, cy, cz = landmark_calibrated(im, lm)
        unit = primary_unit if "cal" in lm else _cal_factors(im)[3]
        x, y, z = lm["px"]
        rows.append({"Label": im.name, "Landmark": lm.get("name") or f"L{i + 1}", "X": cx, "Y": cy, "Z": cz, "Unit": unit,
                     "Xpx": float(x), "Ypx": float(y), "Slice": int(round(float(z))) + 1})
    return rows


def landmark_distances(img: LumenImage, landmarks: list[dict[str, Any]], images: dict[str, LumenImage] | None = None) -> tuple[list[str], np.ndarray]:
    """Pairwise Euclidean distances between landmarks in calibrated units (names, symmetric N×N matrix).

    Only points in the same calibrated frame are comparable (see ``_landmark_frame``); a pair from different frames —
    a viewer-calibrated point and a px × voxel point on another scene object — is NaN instead of a mixed-unit number."""
    names = [lm.get("name") or f"L{i + 1}" for i, lm in enumerate(landmarks)]
    ims = [_landmark_image(img, lm, images) for lm in landmarks]
    pts = np.array([landmark_calibrated(im, lm) for im, lm in zip(ims, landmarks)], np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return names, np.zeros((0, 0))
    d = pts[:, None, :] - pts[None, :, :]
    mat = np.sqrt((d * d).sum(axis=2))
    frames = np.array([_landmark_frame(img, im, lm) for im, lm in zip(ims, landmarks)])
    mat[frames[:, None] != frames[None, :]] = np.nan
    return names, mat


def landmark_rois(img: LumenImage, landmarks: list[dict[str, Any]], images: dict[str, LumenImage] | None = None) -> list[dict[str, Any]]:
    """Point ROIs (one per landmark) pinned to the landmark's slice for the ROI Manager."""
    rois = []
    for i, lm in enumerate(landmarks):
        im = (images or {}).get(lm.get("image", ""), img) if images else img
        x, y, z = lm["px"]
        zi = min(max(int(round(float(z))), 0), max(im.nz - 1, 0))
        rois.append({"type": "point", "points": [[float(x), float(y)]], "name": lm.get("name") or f"L{i + 1}",
                     "color": "#ffd54a", "image_id": im.id, "position": {"t": int(im.display.t), "z": zi, "c": int(im.display.c)}})
    return rois


def parse_landmarks(text: Any) -> list[dict[str, Any]]:
    if isinstance(text, (list, tuple)):
        data: Any = list(text)
    else:
        s = str(text or "").strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Landmarks must be a JSON list: {e}") from e
    if isinstance(data, dict) and "landmarks" in data:
        data = data["landmarks"]
    if not isinstance(data, list):
        raise ValueError("Landmarks must be a JSON list of {name, px: [x, y, z]} objects")
    return [normalize_landmark(lm, i) for i, lm in enumerate(data)]


# ------------------------------------------------------------------ ray picking
def ray_triangles(origin: Any, direction: Any, vertices: np.ndarray, faces: np.ndarray, chunk: int = 262144) -> dict[str, Any] | None:
    """Nearest intersection of a ray with a triangle mesh (Möller–Trumbore, vectorised in chunks).

    Returns ``{"t", "face", "point", "normal"}`` (``t`` = distance along the unit direction; the normal is the
    geometric face normal) or ``None``.  Back faces count too (the inside of open surfaces is visible in the viewer).
    """
    o = np.asarray(origin, np.float64).reshape(3)
    d = np.asarray(direction, np.float64).reshape(3)
    ln = float(np.linalg.norm(d))
    if not np.isfinite(ln) or ln == 0.0 or len(faces) == 0:
        return None
    d = d / ln
    verts = np.asarray(vertices, np.float64)
    faces = np.asarray(faces)
    best_t, best_i = math.inf, -1
    eps = 1e-9
    for start in range(0, len(faces), chunk):
        f = faces[start:start + chunk]
        v0 = verts[f[:, 0]]
        e1 = verts[f[:, 1]] - v0
        e2 = verts[f[:, 2]] - v0
        p = np.cross(d[None, :], e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > eps
        if not ok.any():
            continue
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = o[None, :] - v0
        u = np.einsum("ij,ij->i", tvec, p) * inv
        q = np.cross(tvec, e1)
        v = np.einsum("j,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (t > eps)
        if hit.any():
            idx = np.flatnonzero(hit)
            k = idx[np.argmin(t[idx])]
            if t[k] < best_t:
                best_t, best_i = float(t[k]), int(start + k)
    if best_i < 0:
        return None
    f = faces[best_i]
    n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
    nl = float(np.linalg.norm(n))
    n = n / nl if nl > 0 else np.zeros(3)
    return {"t": best_t, "face": best_i, "point": (o + d * best_t).tolist(), "normal": n.tolist()}


def ray_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float] | None:
    """(t_near, t_far) of a ray against an axis-aligned box, or None when it misses."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    tmin = np.where(np.isnan(t0), -np.inf, np.minimum(t0, t1))
    tmax = np.where(np.isnan(t1), np.inf, np.maximum(t0, t1))
    # a ray parallel to an axis and outside that slab never enters the box
    par = direction == 0
    if np.any(par & ((origin < lo) | (origin > hi))):
        return None
    tn, tf = float(np.max(tmin[~par] if (~par).any() else -np.inf)), float(np.min(tmax[~par] if (~par).any() else np.inf))
    if not np.isfinite(tn) and not np.isfinite(tf):
        return None
    tn = max(tn, 0.0)
    return (tn, tf) if tf > tn else None


def first_voxel_above(vol: np.ndarray, origin: Any, direction: Any, level: float, zratio: float = 1.0, step: float = 0.5, max_samples: int = 65536) -> dict[str, Any] | None:
    """March a ray through a (Z, Y, X) volume in pixel coordinates (z scaled by ``zratio``) and return the first sample
    whose voxel value is ``>= level``: ``{"t", "point", "value", "voxel": [x, y, z_slice]}`` or ``None``.

    The sample position is refined by bisection between the last sample below and the first at / above the level, which
    is what the viewer's iso mode shows.
    """
    nz, ny, nx = vol.shape
    o = np.asarray(origin, np.float64).reshape(3)
    d = np.asarray(direction, np.float64).reshape(3)
    ln = float(np.linalg.norm(d))
    if not np.isfinite(ln) or ln == 0.0:
        return None
    d = d / ln
    zr = float(zratio) if zratio and math.isfinite(zratio) and zratio > 0 else 1.0
    lo = np.zeros(3)
    hi = np.array([nx, ny, nz * zr], np.float64)
    span = ray_box(o, d, lo, hi)
    if span is None:
        return None
    tn, tf = span
    step = max(float(step), 1e-3)
    n = int(min(max_samples, max(2, math.ceil((tf - tn) / step)) + 1))
    ts = tn + np.arange(n) * step
    ts = ts[ts <= tf + 1e-9]
    if len(ts) == 0:
        return None
    pts = o[None, :] + d[None, :] * ts[:, None]
    ix = np.clip(np.floor(pts[:, 0]).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(pts[:, 1]).astype(int), 0, ny - 1)
    iz = np.clip(np.floor(pts[:, 2] / zr).astype(int), 0, nz - 1)
    vals = vol[iz, iy, ix].astype(np.float64)
    above = np.flatnonzero(np.nan_to_num(vals, nan=-np.inf) >= float(level))
    if len(above) == 0:
        return None
    k = int(above[0])
    t_hit = float(ts[k])
    if k > 0:
        a, b = float(ts[k - 1]), t_hit

        def sample(t: float) -> float:
            p = o + d * t
            return float(vol[min(max(int(p[2] / zr), 0), nz - 1), min(max(int(p[1]), 0), ny - 1), min(max(int(p[0]), 0), nx - 1)])

        for _ in range(6):
            m = 0.5 * (a + b)
            if sample(m) >= level:
                b = m
            else:
                a = m
        t_hit = b
    p = o + d * t_hit
    return {"t": t_hit, "point": p.tolist(), "value": float(vals[k]), "voxel": [int(ix[k]), int(iy[k]), int(iz[k])]}


def pick_image(img: LumenImage, origin: Any, direction: Any, mode: str = "surface", c: int | None = None, t: int = 0, level: Any = None,
               step: int = 1, max_dim: int = 192, smooth: int = 0, mesh: Any = None) -> dict[str, Any]:
    """Pick on an image along a ray in pixel coordinates (x, y, z · zratio): on its iso-surface mesh (``mode='surface'``,
    ``mesh`` may be a precomputed ``mesh_ops.Mesh``) or the first voxel above ``level`` (``mode='volume'``)."""
    from .mesh_ops import extract_mesh, resolve_level, zratio

    cc = img.display.c if c is None or int(c) < 0 else int(c)
    if cc >= img.nc:
        raise ValueError(f"Channel {cc} does not exist (the image has {img.nc})")
    tt = min(max(int(t), 0), img.nt - 1)
    zr = zratio(img)
    pw, ph, pd, unit = _cal_factors(img)
    if mode == "surface":
        m = mesh if mesh is not None else extract_mesh(img, cc, tt, level, step, max_dim, smooth)
        hit = ray_triangles(origin, direction, m.vertices, m.faces)
        if hit is None:
            return {"hit": False, "mode": mode, "level": float(m.level)}
        x, y, z = hit["point"]
        out = {"hit": True, "mode": mode, "level": float(m.level), "distance": hit["t"], "face": hit["face"], "normal": hit["normal"]}
    else:
        vol = img.data[tt, :, cc]
        lvl = resolve_level(img, cc, vol, level)
        hit = first_voxel_above(vol, origin, direction, lvl, zr)
        if hit is None:
            return {"hit": False, "mode": "volume", "level": float(lvl)}
        x, y, z = hit["point"]
        out = {"hit": True, "mode": "volume", "level": float(lvl), "distance": hit["t"], "value": hit["value"]}
    out.update({"point": [float(x), float(y), float(z)], "px": [float(x), float(y), float(z) / zr],
                "calibrated": [float(x) * pw, float(y) * ph, float(z) / zr * pd], "unit": unit})
    return out


# ------------------------------------------------------------------ animation assembly
def decode_frames(blobs: list[bytes], size: tuple[int, int] | None = None) -> list[np.ndarray]:
    """PNG / JPEG bytes → (H, W, 3) uint8 RGB frames, all cropped / padded (black) to the first frame's size."""
    from PIL import Image

    frames: list[np.ndarray] = []
    for b in blobs:
        with Image.open(io.BytesIO(b)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if size is None:
            size = (arr.shape[0], arr.shape[1])
        H, W = size
        if arr.shape[0] != H or arr.shape[1] != W:
            out = np.zeros((H, W, 3), np.uint8)
            h, w = min(H, arr.shape[0]), min(W, arr.shape[1])
            out[:h, :w] = arr[:h, :w]
            arr = out
        frames.append(np.ascontiguousarray(arr))
    return frames


def frames_to_stack(frames: list[np.ndarray], name: str = "animation", fps: float = 15.0) -> LumenImage:
    """RGB frames → an RGB time stack (T, 1, 3, H, W) with the frame interval set from ``fps``."""
    if not frames:
        raise ValueError("No frames")
    data = np.stack([np.ascontiguousarray(f if f.ndim == 3 else np.stack([f] * 3, -1), dtype=np.uint8) for f in frames])
    img = LumenImage(data, name=name, axes="TYXS", rgb=True)
    fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else 15.0
    img.calibration = Calibration(1.0, 1.0, 1.0, "pixel", frame_interval=1.0 / fps, time_unit="sec")
    img.metadata["animation_fps"] = fps
    return img


def write_gif(frames: list[np.ndarray], target: Any, fps: float = 15.0, loop: int = 0) -> Any:
    """Animated GIF via PIL (each RGB frame is palettised); ``target`` is a path or a binary file object."""
    from PIL import Image

    if not frames:
        raise ValueError("No frames")
    fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else 15.0
    duration = max(10, int(round(1000.0 / fps)))
    ims = [Image.fromarray(np.ascontiguousarray(f if f.ndim == 3 else np.stack([f] * 3, -1), dtype=np.uint8), "RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for f in frames]
    if isinstance(target, (str, Path)):
        p = Path(target).expanduser()
        if p.suffix.lower() != ".gif":
            p = p.with_name(p.name + ".gif")
        p.parent.mkdir(parents=True, exist_ok=True)
        ims[0].save(str(p), format="GIF", save_all=True, append_images=ims[1:], duration=duration, loop=int(loop), disposal=1)
        return str(p)
    ims[0].save(target, format="GIF", save_all=True, append_images=ims[1:], duration=duration, loop=int(loop), disposal=1)
    return target


def write_animation_file(frames: list[np.ndarray], path: str, fmt: str, fps: float) -> str:
    """GIF (PIL, always available) or a video through the opt-in ``[video]`` extra (imageio-ffmpeg)."""
    fmt = (fmt or "gif").lower().lstrip(".")
    p = Path(str(path)).expanduser()
    if p.is_dir() or str(path).endswith(("/", "\\")):
        p = p / f"animation.{fmt}"
    if fmt == "gif":
        return write_gif(frames, p, fps)
    if fmt not in ("mp4", "avi", "mov", "mkv", "webm"):
        raise ValueError(f"Unknown animation format {fmt!r}; use gif, mp4, avi, mov, mkv or webm")
    from ..io import write_video  # raises RuntimeError with the install hint when imageio-ffmpeg is missing
    p.parent.mkdir(parents=True, exist_ok=True)
    return write_video(frames, p, fmt, fps=float(fps))


def assemble_animation(session: Any, blobs: list[bytes], name: str = "animation", fps: float = 15.0, fmt: str = "", path: str = "") -> dict[str, Any]:
    """Recorded canvas frames → a new RGB time stack in the session (+ an optional GIF / video file)."""
    if not blobs:
        raise ValueError("No frames were recorded")
    frames = decode_frames(blobs)
    img = frames_to_stack(frames, name or "animation", fps)
    # opened beside the stack being viewed (not activated): the 3-D View section stays mounted on its Z stack, as the
    # 3D Viewer keeps running after Record 360° / Take snapshot; add_image still activates it when nothing is open
    session.add_image(img, activate=False)
    out: dict[str, Any] = {"id": img.id, "frames": len(frames), "width": int(frames[0].shape[1]), "height": int(frames[0].shape[0]), "fps": float(fps)}
    msg = f"Animation: {len(frames)} frames ({frames[0].shape[1]}×{frames[0].shape[0]}) → {img.name}"
    if fmt and str(fmt).strip():
        if not str(path or "").strip():
            raise ValueError("Give an output path for the animation file")
        out["path"] = write_animation_file(frames, path, fmt, fps)
        msg += f"; saved {out['path']}"
    session.add_log(msg)
    out["message"] = msg
    return out


# ------------------------------------------------------------------ commands
@op("scene.save", "Save 3D Scene...", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False, requires_image=False,
    params=[Param("path", "string", "", help="Scene file (.json); a directory gets scene.json inside"),
            Param("scene", "text", "", advanced=True, help="Scene JSON; empty = the scene currently shown in the 3-D View section")],
    description="Save the 3-D View scene (objects with colour / transparency / threshold / transform, camera, lights, background, "
                "clipping box and landmarks) as a JSON file that Load 3D Scene restores (3D Viewer ▸ File ▸ Save/Load view).",
    keywords=["3d", "scene", "view", "save", "json", "camera", "landmarks"])
def scene_save(ctx, path="", scene=""):
    sc = normalize_scene(scene) if str(scene or "").strip() else session_scene(ctx.session)
    if sc is None:
        raise ValueError("No 3-D scene to save yet: open the 3-D View section of a Z stack first (or pass the scene JSON)")
    sc = normalize_scene(sc)
    # remember the names so a scene saved today can be restored in a later session (ids change) by image name
    images = ctx.session.images
    if sc.get("primary") in images:
        sc["primary_name"] = images[sc["primary"]].name
    for o in sc["objects"]:
        if o["image"] in images:
            o["name"] = images[o["image"]].name
    p = _scene_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(scene_to_json(sc), encoding="utf-8")
    store_scene(ctx.session, sc)
    msg = f"Saved 3-D scene ({len(sc['objects'])} object{'s' if len(sc['objects']) != 1 else ''}, {len(sc['landmarks'])} landmark{'s' if len(sc['landmarks']) != 1 else ''}) → {p}"
    return OpResult(message=msg, log=msg)


@op("scene.load", "Load 3D Scene...", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False, requires_image=False,
    params=[Param("path", "string", "", help="Scene file written by Save 3D Scene")],
    description="Load a 3-D View scene from JSON: objects are matched to the open images by id, then by name (missing ones are listed); "
                "the 3-D View section applies the camera, lights, background, clipping, objects and landmarks.",
    keywords=["3d", "scene", "view", "load", "open", "json"])
def scene_load(ctx, path=""):
    p = _scene_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No scene file {p}")
    sc = normalize_scene(p.read_text(encoding="utf-8"))
    sc = resolve_scene_images(sc, ctx.session.images)
    rev = store_scene(ctx.session, sc)
    cols, rows = scene_rows(sc)
    missing = [o["name"] or o["image"] for o in sc["objects"] if o.get("missing")]
    msg = f"Loaded 3-D scene from {p.name}: {len(sc['objects'])} object{'s' if len(sc['objects']) != 1 else ''}, {len(sc['landmarks'])} landmark{'s' if len(sc['landmarks']) != 1 else ''}"
    if missing:
        msg += f"; not open: {', '.join(missing)}"
    return OpResult(table={"name": "Scene", "columns": cols, "rows": rows, "rev": rev}, message=msg, log=msg)


@op("scene.landmarks_to_results", "Landmarks to Results (3D)", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False,
    params=[Param("landmarks", "text", "", help="JSON list of {name, px: [x, y, z]} (x, y in pixels, z in slices); empty = the landmarks of the current 3-D scene"),
            Param("add_rois", "bool", True, label="Add point ROIs", help="One point ROI per landmark in the ROI Manager, pinned to its slice"),
            Param("distances", "bool", True, label="Pairwise distances", help="Also show the N×N calibrated distance table")],
    description="Tabulate the landmarks placed in the 3-D View: calibrated X / Y / Z, pixel coordinates and slice → Results; "
                "the pairwise distance matrix → a table; point ROIs with a slice position → ROI Manager (3D Viewer ▸ Landmarks).",
    keywords=["3d", "landmarks", "points", "distance", "measure", "results"])
def landmarks_to_results(ctx, landmarks="", add_rois=True, distances=True):
    img = ctx.img
    images = {k: v for k, v in ctx.session.images.items()}
    frame = img  # the image whose calibrated frame a viewer-supplied `cal` is in (the viewer's primary)
    lms = parse_landmarks(landmarks)
    if not lms:
        sc = session_scene(ctx.session)
        lms = [normalize_landmark(lm, i) for i, lm in enumerate((sc or {}).get("landmarks", []))]
        frame = images.get(str((sc or {}).get("primary") or ""), img)
    if not lms:
        raise ValueError("No landmarks: click on the surface in the 3-D View (Landmarks ▸ Place) first, or pass a JSON list")
    rows = landmark_rows(frame, lms, images)
    units = sorted({r["Unit"] for r in rows})
    unit = units[0] if len(units) == 1 else "mixed: " + ", ".join(units)
    table = None
    blank = False
    if distances and len(lms) > 1:
        names, mat = landmark_distances(frame, lms, images)
        cell = lambda v: float(v) if math.isfinite(v) else None  # noqa: E731 - NaN (different frames) is not valid JSON
        cells = [[cell(mat[i, j]) for j in range(len(names))] for i in range(len(names))]
        blank = any(c is None for r in cells for c in r)
        table = {"name": "Landmark distances", "columns": ["Landmark"] + names, "rows": [[names[i]] + cells[i] for i in range(len(names))]}
    rois = landmark_rois(img, lms, images) if add_rois else []
    msg = f"{len(lms)} landmark{'s' if len(lms) != 1 else ''} → Results ({unit})"
    if table is not None:
        msg += f"; {len(lms)}×{len(lms)} distance table ({unit})"
        if blank:
            msg += ", pairs in different calibrated frames left blank"
    if rois:
        msg += f"; {len(rois)} point ROI{'s' if len(rois) != 1 else ''} added"
    return OpResult(results=rows, table=table, rois_add=rois, message=msg, log=msg)
