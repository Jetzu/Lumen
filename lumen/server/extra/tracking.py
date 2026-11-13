"""The track model for the TrackScheme and the Tracks wizard (drop-in router).

``GET /api/tracking/model?image_id=``
    The tracks of an image with their spots (ids, positions, features), lineage edges and statistics, the
    unlinked detections, the display settings (``tracking.set_display``), the undo depth and the feature /
    colormap / mode lists the UI offers.  Read-only (tracks parked for another image are read from the archive).

``GET /api/tracking/histogram?image_id=&feature=Quality&bins=64``
    Histogram of a spot feature over the stored detection of the image (else over its track spots), for the
    wizard's initial-threshold step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from ..app import get_session
from ...core.ops import tracking as tk
from ...core.ops import tracking_extra as tkx

router = APIRouter()

FREE_SPOT_LIMIT = 5000


def _image(session, image_id: str | None):
    try:
        return session.get_image(image_id) if image_id else session.get_image()
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


def _tracks_of(session, image_id: str) -> list[dict[str, Any]]:
    if session.settings.get("tracks_image") == image_id:
        return session.settings.get("tracks") or []
    return (session.settings.get("tracks_archive") or {}).get(image_id) or []


def _num(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    return v


def _spot_json(sp: dict[str, Any]) -> dict[str, Any]:
    out = {"id": int(sp["id"]), "t": int(sp["t"]), "x": float(sp["x"]), "y": float(sp["y"]), "z": int(sp.get("z", 0)), "r": float(sp.get("r", tk.MANUAL_RADIUS))}
    for key in ("quality", *tk.INTENSITY_KEYS, "visibility"):
        out[key] = _num(tk._spot_value(sp, key))
    if sp.get("name"):
        out["name"] = str(sp["name"])
    if sp.get("manual"):
        out["manual"] = True
    return out


@router.get("/tracking/model")
def tracking_model(image_id: str | None = None) -> dict[str, Any]:
    s = get_session()
    with s.lock:
        img = _image(s, image_id)
        tracks = _tracks_of(s, img.id)
        pool = tkx.spot_pool_for(s, img)
        tk._ensure_ids(pool, *(t["spots"] for t in tracks))
        out_tracks = []
        in_tracks: set[int] = set()
        for t in tracks:
            spots = sorted(t["spots"], key=tk._spot_key)
            in_tracks.update(int(sp["id"]) for sp in spots)
            out_tracks.append({"id": str(t["id"]), "name": str(t["name"]), "color": str(t["color"]), "manual": bool(t.get("manual")), "overlay_id": tk._overlay_id(t),
                               "stats": {k: _num(v) for k, v in (t.get("stats") or {}).items()}, "spots": [_spot_json(sp) for sp in spots],
                               "edges": [[int(a), int(b)] for a, b in tk._track_edges(t)]})
        free = [sp for sp in pool if int(sp["id"]) not in in_tracks]
        undo = sum(1 for e in s.settings.get("tracks_undo") or [] if e.get("image") == img.id)
        from ...core.luts import LUT_NAMES
        return {"image": img.id, "version": s.version, "nt": img.nt, "calibration": img.calibration.to_dict(), "tracks": out_tracks,
                "free_spots": [_spot_json(sp) for sp in free[:FREE_SPOT_LIMIT]], "n_free_spots": len(free), "n_spots": len(pool),
                "display": tkx.display_settings(s), "undo_depth": undo, "spot_features": list(tk.SPOT_FEATURES), "track_features": list(tk.TRACK_FEATURES),
                "color_by": list(tkx.COLOR_BY), "display_modes": list(tkx.DISPLAY_MODES), "colormaps": list(LUT_NAMES)}


@router.get("/tracking/histogram")
def tracking_histogram(image_id: str | None = None, feature: str = "Quality", bins: int = 64) -> dict[str, Any]:
    s = get_session()
    with s.lock:
        img = _image(s, image_id)
        try:
            key = tk._spot_feature_key(feature)
        except ValueError as e:
            raise HTTPException(400, str(e))
        pool = tkx.spot_pool_for(s, img)
        source = "spots"
        if not pool:
            pool = [sp for t in _tracks_of(s, img.id) for sp in t["spots"]]
            source = "tracks"
        vals = np.array([tk._spot_value(sp, key) for sp in pool], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"feature": feature, "counts": [], "edges": [], "min": None, "max": None, "median": None, "n": 0, "source": source}
        nb = max(4, min(int(bins), 1024))
        counts, edges = np.histogram(vals, bins=nb)
        return {"feature": feature, "counts": counts.tolist(), "edges": [float(e) for e in edges], "min": float(vals.min()), "max": float(vals.max()),
                "median": float(np.median(vals)), "n": int(vals.size), "source": source}
