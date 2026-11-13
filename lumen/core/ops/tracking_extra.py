"""Plugins ▸ Tracking — TrackMate's spot filters, feature tables, manual lineage edits and display settings.

Extends ``tracking.py`` (same track model, see its module docstring):

* ``tracking.filter_spots``: threshold any spot feature (TrackMate's initial threshold and spot filters).
* ``tracking.feature_tables`` / ``tracking.export_feature_tables``: the 'Spots', 'Edges' and 'Track statistics'
  tables with TrackMate's feature keys, and the matching CSV files (TrackMate's "Export all spots / links /
  tracks statistics").
* Lineage edits (TrackScheme): split a track at a frame, merge two tracks, link / unlink two spots, add / delete
  a spot.  Every edit snapshots the model first (``settings["tracks_undo"]``); ``tracking.undo_edit`` restores
  the previous model.  After an edit the tracks are recomputed as the connected components of the link graph,
  keeping the identity (id, name, colour) of the track that contributed most spots to each component.
* ``tracking.set_display``: colour tracks / spots by a feature through a LUT, track display mode, tail length,
  spot radius and spot names – stored in ``settings["track_display"]`` for the viewer and the TrackScheme; track
  feature colouring also recolours the track overlays (and the Tracks table's Color column).
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..image import LumenImage
from ..registry import Param, OpResult, op
from .tracking import (CAT, MANUAL_RADIUS, PALETTE, SPOT_FEATURES, TRACK_FEATURES, _EDGE_FEATURES, _SPOT_FEATURES, _TRACK_FEATURES, _TM_SPOT_INTENSITY,
                       _all_track_overlay, _ensure_ids, _find_track, _manual_spot, _next_spot_id, _next_track_number, _push_undo, _spot_key, _spot_pool,
                       _spot_value, _tm_edge_features, _tm_spot_features, _tm_track_features, _track_edges, _track_stats, _tracks_for_image, _tracks_table)

DISPLAY_MODES = ["Full tracks", "Local tails (backward)", "Local tails (both ways)", "Up to current frame", "Fade"]
COLOR_BY = ["Track colour"] + [f"Track: {f}" for f in TRACK_FEATURES] + [f"Spot: {f}" for f in SPOT_FEATURES]
DISPLAY_DEFAULTS: dict[str, Any] = {"color_by": "Track colour", "colormap": "Turbo", "display_mode": "Full tracks", "tail_length": 10,
                                    "spot_radius": True, "show_names": False, "feature": None}
_EDIT_KEYWORDS = ["trackmate", "trackscheme", "lineage", "manual editing", "edit tracks"]


# ================================================================================ model access
def spot_pool_for(session, img: LumenImage) -> list[dict[str, Any]]:
    """The stored detection when it belongs to ``img`` (else an empty list – never another image's spots)."""
    return _spot_pool(session, img)


def display_settings(session) -> dict[str, Any]:
    """The track display settings with defaults filled in."""
    out = dict(DISPLAY_DEFAULTS)
    out.update(session.settings.get("track_display") or {})
    return out


def _model(ctx, img: LumenImage) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """(tracks, detection pool, spot by id) of ``img``; every spot gets an id first."""
    tracks = _tracks_for_image(ctx, img, create=True)
    pool = spot_pool_for(ctx.session, img)
    _ensure_ids(pool, *(t["spots"] for t in tracks))
    by_id: dict[int, dict[str, Any]] = {}
    for sp in pool:
        by_id[int(sp["id"])] = sp
    for t in tracks:
        for sp in t["spots"]:
            by_id[int(sp["id"])] = sp
    return tracks, pool, by_id


def _spot_in_track(tracks: list[dict[str, Any]], sid: int) -> tuple[int | None, dict[str, Any] | None]:
    for k, t in enumerate(tracks):
        for sp in t["spots"]:
            if int(sp["id"]) == sid:
                return k, t
    return None, None


def _resolve_spot(by_id: dict[int, dict[str, Any]], spot_id: int, tracks: list[dict[str, Any]] | None = None, track_id: int = -1,
                  frame: int = -1) -> dict[str, Any]:
    """A spot by id, or (spot_id < 0) the spot of track ``track_id`` on ``frame`` (which must be unique there)."""
    if spot_id >= 0:
        if spot_id not in by_id:
            raise ValueError(f"No spot with id {spot_id}")
        return by_id[spot_id]
    if not tracks:
        raise ValueError("Give a spot id (Spots table / TrackScheme) or a track id and frame")
    track = _find_track(tracks, int(track_id))
    cands = [sp for sp in track["spots"] if int(sp["t"]) == frame]
    if not cands:
        raise ValueError(f"{track['name']} has no spot on frame {frame}")
    if len(cands) > 1:
        raise ValueError(f"{track['name']} has {len(cands)} spots on frame {frame} (a split or merge) – give the spot id")
    return cands[0]


def _rebuild(ctx, img: LumenImage, tracks: list[dict[str, Any]], *, extra_spots: list[dict[str, Any]] = (), extra_edges: list[tuple[int, int]] = (),
             prefer: list[int] = ()) -> list[dict[str, Any]]:
    """Recompute the image's tracks as the connected components of (all track spots + ``extra_spots``, all lineage
    edges + ``extra_edges``). A component keeps the id / name / colour of the old track that contributed most of its
    spots (``prefer``: indices of old tracks that get first pick); other components become new tracks 'Track <n>'.
    Empty manual tracks are kept; old tracks whose spots all moved elsewhere disappear."""
    by_id: dict[int, dict[str, Any]] = {}
    for t in tracks:
        for sp in t["spots"]:
            by_id[int(sp["id"])] = sp
    for sp in extra_spots:
        by_id.setdefault(int(sp["id"]), sp)
    edges: set[tuple[int, int]] = set()
    for t in tracks:
        edges.update(_track_edges(t))
    edges.update((int(a), int(b)) for a, b in extra_edges)
    edges = {(a, b) for a, b in edges if a in by_id and b in by_id and a != b}
    parent = {i: i for i in by_id}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[int, list[int]] = defaultdict(list)
    for i in by_id:
        comps[find(i)].append(i)
    comp_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        comp_edges[find(a)].append((a, b))
    claimed: dict[int, int] = {}  # component root → old track index
    order = [k for k in prefer if 0 <= k < len(tracks)] + [k for k in range(len(tracks)) if k not in prefer]
    for k in order:
        counts = Counter(find(int(sp["id"])) for sp in tracks[k]["spots"] if int(sp["id"]) in by_id)
        for root, _ in counts.most_common():
            if root not in claimed:
                claimed[root] = k
                break

    def make(root: int, old: dict[str, Any]) -> dict[str, Any]:
        ids = sorted(comps[root], key=lambda i: _spot_key(by_id[i]))
        sp_list = [by_id[i] for i in ids]
        ed = sorted(comp_edges.get(root, []), key=lambda e: (_spot_key(by_id[e[0]]), _spot_key(by_id[e[1]])))
        index_of = {i: k for k, i in enumerate(ids)}
        tr = dict(old)
        tr.update(spots=sp_list, edges=[[a, b] for a, b in ed], stats=_track_stats(img, sp_list, list(range(len(ids))), [(index_of[a], index_of[b]) for a, b in ed]))
        return tr

    new_tracks: list[dict[str, Any]] = []
    for k, old in enumerate(tracks):
        roots = [r for r, kk in claimed.items() if kk == k]
        if roots:
            new_tracks.append(make(roots[0], old))
        elif not old["spots"] and old.get("manual"):
            new_tracks.append(old)
    n = _next_track_number(tracks)
    for root in sorted((r for r in comps if r not in claimed), key=lambda r: min(_spot_key(by_id[i]) for i in comps[r])):
        new_tracks.append(make(root, {"id": str(n), "name": f"Track {n}", "color": PALETTE[n % len(PALETTE)]}))
        n += 1
    return new_tracks


def _commit(ctx, img: LumenImage, tracks: list[dict[str, Any]], msg: str) -> OpResult:
    s = ctx.session
    s.settings["tracks"] = tracks
    s.settings["tracks_image"] = img.id
    return OpResult(overlay=_all_track_overlay(img, tracks), table=_tracks_table(tracks), message=msg, log=msg)


def _created_names(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[str]:
    old = {str(t["id"]) for t in before}
    return [t["name"] for t in after if str(t["id"]) not in old]


# ================================================================================ spot filters
@op("tracking.filter_spots", "Filter Spots", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("feature", "choice", "Quality", choices=list(SPOT_FEATURES), help="Spot feature to threshold (TrackMate's spot features)"),
            Param("threshold", "float", 0.0, step=1),
            Param("above", "bool", True, help="Keep the spots whose feature is ≥ the threshold (unticked: ≤)")],
    description="TrackMate's spot filter: keep only the detected spots whose feature passes the threshold (the wizard's initial quality threshold and spot filters). Spots of existing tracks that fail are removed from their tracks and the lineages are recomputed; the step can be undone with Undo Track Edit.",
    keywords=["trackmate", "spot filter", "initial threshold", "quality", "snr", "contrast"])
def filter_spots(ctx, feature="Quality", threshold=0.0, above=True):
    img = ctx.img
    s = ctx.session
    key = SPOT_FEATURES[feature]
    try:
        tracks: list[dict[str, Any]] | None = _tracks_for_image(ctx, img)
    except ValueError:
        tracks = None
    pool = spot_pool_for(s, img)
    if not pool and not tracks:
        raise ValueError("Run Detect Spots first")
    _ensure_ids(pool, *(t["spots"] for t in tracks or []))

    def passes(sp: dict[str, Any]) -> bool:
        v = _spot_value(sp, key)
        return bool(np.isfinite(v) and (v >= threshold if above else v <= threshold))

    _push_undo(s, img, tracks, "Filter Spots")
    n_before = len(pool)
    kept_pool = [sp for sp in pool if passes(sp)] if pool else []
    if pool:
        s.settings["spots"] = kept_pool  # a new list: the undo snapshot keeps the old one
    crit = f"{feature} {'≥' if above else '≤'} {threshold:g}"
    if not tracks:
        msg = f"Kept {len(kept_pool)} of {n_before} spots ({crit})"
        return OpResult(message=msg, log=msg)
    n_track_removed = 0
    for t in tracks:
        before = len(t["spots"])
        t["spots"] = [sp for sp in t["spots"] if passes(sp)]
        n_track_removed += before - len(t["spots"])
    new_tracks = _rebuild(ctx, img, tracks) if n_track_removed else tracks
    msg = (f"Kept {len(kept_pool)} of {n_before} spots ({crit})" if pool else f"{crit}") + f"; {n_track_removed} removed from tracks → {len(new_tracks)} tracks"
    return _commit(ctx, img, new_tracks, msg)


# ================================================================================ feature tables
def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


SPOT_COLUMNS = ["LABEL", "ID", "TRACK_ID", "QUALITY", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "FRAME", "RADIUS", "VISIBILITY"] + [k for k, _ in _TM_SPOT_INTENSITY]
EDGE_COLUMNS = ["LABEL", "TRACK_ID", "SPOT_SOURCE_ID", "SPOT_TARGET_ID", "LINK_COST", "DIRECTIONAL_CHANGE_RATE", "SPEED", "DISPLACEMENT", "EDGE_TIME",
                "EDGE_X_LOCATION", "EDGE_Y_LOCATION", "EDGE_Z_LOCATION"]


def build_feature_tables(session, img: LumenImage, tracks: list[dict[str, Any]], include_unlinked: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """TrackMate's three statistics tables ({name, columns, rows}) of the given tracks: Spots, Edges, Track statistics."""
    cal = img.calibration
    pool = spot_pool_for(session, img)
    _ensure_ids(pool, *(t["spots"] for t in tracks))
    spot_rows: list[list[Any]] = []
    edge_rows: list[list[Any]] = []
    track_rows: list[list[Any]] = []
    track_cols: list[str] = []
    in_tracks: set[int] = set()
    for index, t in enumerate(tracks):
        tf = _tm_track_features(t, index, cal)
        if not track_cols:
            track_cols = list(tf)
        track_rows.append([_clean(tf.get(c)) for c in track_cols])
        tid = tf["TRACK_ID"]
        by_id = {int(sp["id"]): sp for sp in t["spots"]}
        for sp in sorted(t["spots"], key=_spot_key):
            f = _tm_spot_features(sp, int(sp["id"]), cal)
            in_tracks.add(int(sp["id"]))
            spot_rows.append([f["name"], f["ID"], tid] + [_clean(f.get(c)) for c in SPOT_COLUMNS[3:]])
        edges = [(a, b) for a, b in _track_edges(t) if a in by_id and b in by_id]
        preds: dict[int, list[int]] = defaultdict(list)
        for a, b in edges:
            preds[b].append(a)
        for a, b in edges:
            sa, sb = by_id[a], by_id[b]
            prev = by_id[preds[a][0]] if len(preds[a]) == 1 else None
            ef = _tm_edge_features(sa, sb, a, b, cal, prev)
            edge_rows.append([f"ID{a} → ID{b}", tid] + [_clean(ef.get(c)) for c in EDGE_COLUMNS[2:]])
    if include_unlinked:
        for sp in pool:
            if int(sp["id"]) not in in_tracks:
                f = _tm_spot_features(sp, int(sp["id"]), cal)
                spot_rows.append([f["name"], f["ID"], -1] + [_clean(f.get(c)) for c in SPOT_COLUMNS[3:]])
    if not track_cols:
        track_cols = list(_tm_track_features({"id": "0", "name": "", "spots": []}, 0, cal))
    return ({"name": "Spots", "columns": SPOT_COLUMNS, "rows": spot_rows}, {"name": "Edges", "columns": EDGE_COLUMNS, "rows": edge_rows},
            {"name": "Track statistics", "columns": track_cols, "rows": track_rows})


def _trim(table: dict[str, Any], max_rows: int) -> dict[str, Any]:
    if len(table["rows"]) <= max_rows:
        return table
    return {**table, "rows": table["rows"][:max_rows], "truncated": len(table["rows"]) - max_rows}


@op("tracking.feature_tables", "Feature Tables (Spots / Edges / Tracks)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("include_unlinked", "bool", False, help="Spots table: also the detected spots that belong to no track (TRACK_ID = −1)"),
            Param("max_rows", "int", 50000, min=100, help="Row limit of the in-app Spots / Edges tables (Export Feature Tables writes everything)")],
    description="TrackMate's statistics tables in the Table panel: 'Spots' (one row per spot: ID, TRACK_ID, QUALITY, POSITION_X/Y/Z/T, FRAME, RADIUS, VISIBILITY, MEAN/MEDIAN/MIN/MAX/TOTAL/STD_INTENSITY, CONTRAST, SNR), 'Edges' (SPOT_SOURCE_ID, SPOT_TARGET_ID, LINK_COST, SPEED, DISPLACEMENT, DIRECTIONAL_CHANGE_RATE, EDGE_TIME/X/Y/Z) and 'Track statistics' (TRACK_ID, NUMBER_SPOTS/GAPS/SPLITS/MERGES/COMPLEX, LONGEST_GAP, TRACK_DURATION/START/STOP/DISPLACEMENT, speeds, TRACK_MEAN_QUALITY, TOTAL/MAX_DISTANCE_TRAVELED, CONFINEMENT_RATIO …), calibrated in TrackMate's conventions.",
    keywords=["trackmate", "statistics", "spots table", "edges table", "links", "track statistics", "features"])
def feature_tables(ctx, include_unlinked=False, max_rows=50000):
    img = ctx.img
    tracks = _tracks_for_image(ctx, img)
    if not tracks:
        raise ValueError("No tracks – run Track / Link Spots or import a TrackMate file first")
    spots_t, edges_t, tracks_t = build_feature_tables(ctx.session, img, tracks, include_unlinked)
    s = ctx.session
    s.tables["Spots"] = _trim(spots_t, max_rows)
    s.tables["Edges"] = _trim(edges_t, max_rows)
    s.tables["Track statistics"] = tracks_t
    msg = f"Feature tables: {len(spots_t['rows'])} spots, {len(edges_t['rows'])} edges, {len(tracks_t['rows'])} tracks (tables 'Spots', 'Edges', 'Track statistics')"
    return OpResult(table=tracks_t, message=msg, log=msg)


_DECL = {f[0]: f for f in _SPOT_FEATURES + _EDGE_FEATURES + _TRACK_FEATURES}
_DECL.update({"LABEL": ("LABEL", "Label", "Label", "NONE", "false"), "ID": ("ID", "Spot ID", "ID", "NONE", "true"), "TRACK_ID": ("TRACK_ID", "Track ID", "Track ID", "NONE", "true")})


def _write_table(path: Path, table: dict[str, Any], trackmate_headers: bool) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = table["columns"]
        w.writerow(cols)
        if trackmate_headers:  # TrackMate's extra header rows: feature name, short name, dimension
            w.writerow([_DECL.get(c, (c, c, c, "", ""))[1] for c in cols])
            w.writerow([_DECL.get(c, (c, c, c, "", ""))[2] for c in cols])
            w.writerow([_DECL.get(c, (c, c, c, "", ""))[3] for c in cols])
        for r in table["rows"]:
            w.writerow(["NaN" if v is None else (round(v, 6) if isinstance(v, float) else v) for v in r])


@op("tracking.export_feature_tables", "Export Feature Tables (CSV)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("path", "string", "", help="Output prefix (or an existing folder): writes <prefix>_spots.csv, <prefix>_edges.csv and <prefix>_tracks.csv"),
            Param("which", "choice", "All", choices=["All", "Spots", "Edges", "Tracks"]),
            Param("include_unlinked", "bool", False, help="Spots: also the spots that belong to no track (TRACK_ID = −1)"),
            Param("trackmate_headers", "bool", False, help="TrackMate's four header rows (key, name, short name, dimension) instead of one row of keys")],
    description="Write TrackMate's Spots / Edges / Tracks statistics tables as CSV files (TrackMate feature keys, calibrated values, NaN for undefined).",
    keywords=["trackmate", "csv", "export statistics", "spots", "links", "tracks"])
def export_feature_tables(ctx, path="", which="All", include_unlinked=False, trackmate_headers=False):
    img = ctx.img
    tracks = _tracks_for_image(ctx, img)
    if not tracks:
        raise ValueError("No tracks – run Track / Link Spots or import a TrackMate file first")
    if not path or not str(path).strip():
        raise ValueError("Give an output prefix or folder")
    p = Path(path).expanduser()
    base = p / (Path(img.name).stem or "tracks") if p.is_dir() else (p.with_suffix("") if p.suffix.lower() == ".csv" else p)
    base.parent.mkdir(parents=True, exist_ok=True)
    spots_t, edges_t, tracks_t = build_feature_tables(ctx.session, img, tracks, include_unlinked)
    written = []
    for name, table in (("Spots", spots_t), ("Edges", edges_t), ("Tracks", tracks_t)):
        if which not in ("All", name):
            continue
        out = base.with_name(f"{base.name}_{name.lower()}.csv")
        _write_table(out, table, trackmate_headers)
        written.append(f"{out.name} ({len(table['rows'])} rows)")
    msg = f"Wrote {', '.join(written)} to {base.parent}"
    return OpResult(message=msg, log=msg)


# ================================================================================ lineage edits (TrackScheme)
@op("tracking.split_track", "Split Track at Frame", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("track_id", "int", -1, min=-1, help="Track id (−1 = the last track)"),
            Param("frame", "int", -1, min=-1, help="0-based frame: every link from an earlier frame into this frame or later is cut, so the spots from this frame on form new track(s). −1 = the current frame")],
    description="Cut a track at a frame (TrackScheme ▸ remove the links crossing the frame); the pieces become separate tracks. Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["split track", "cut"])
def split_track(ctx, track_id=-1, frame=-1):
    img = ctx.img
    tracks, pool, by_id = _model(ctx, img)
    if not tracks:
        raise ValueError("No tracks on this image")
    track = _find_track(tracks, int(track_id))
    f = int(img.display.t) if frame < 0 else int(frame)
    tb = {int(sp["id"]): sp for sp in track["spots"]}
    edges = _track_edges(track)
    cut = [(a, b) for a, b in edges if tb[a]["t"] < f <= tb[b]["t"]]
    if not cut:
        raise ValueError(f"{track['name']} has no link crossing frame {f}")
    _push_undo(ctx.session, img, tracks, "Split Track")
    track["edges"] = [[a, b] for a, b in edges if (a, b) not in cut]
    new_tracks = _rebuild(ctx, img, tracks, prefer=[tracks.index(track)])
    created = _created_names(tracks, new_tracks)
    msg = f"{track['name']} split at frame {f}: {len(cut)} link{'s' if len(cut) != 1 else ''} cut → new {', '.join(created) if created else 'track(s)'} ({len(new_tracks)} tracks)"
    return _commit(ctx, img, new_tracks, msg)


@op("tracking.merge_tracks", "Merge Two Tracks", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("track_id", "int", 0, min=0, help="First track (keeps its id, name and colour)"),
            Param("other_id", "int", 1, min=0, help="Second track: its first spot is linked after the first track's last spot (or the other way round when it ends before the first one starts)")],
    description="Join two tracks that follow each other in time into one (a link from the end of one to the start of the other). Overlapping tracks: link two specific spots with Link Two Spots instead. Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["merge tracks", "join"])
def merge_tracks(ctx, track_id=0, other_id=1):
    img = ctx.img
    tracks, pool, by_id = _model(ctx, img)
    if not tracks:
        raise ValueError("No tracks on this image")
    a = _find_track(tracks, int(track_id))
    b = _find_track(tracks, int(other_id))
    if a is b:
        raise ValueError("Give two different tracks")
    if not a["spots"] or not b["spots"]:
        raise ValueError("Both tracks need spots")
    a_first, a_last = min(a["spots"], key=_spot_key), max(a["spots"], key=_spot_key)
    b_first, b_last = min(b["spots"], key=_spot_key), max(b["spots"], key=_spot_key)
    if b_first["t"] > a_last["t"]:
        src, dst = a_last, b_first
    elif a_first["t"] > b_last["t"]:
        src, dst = b_last, a_first
    else:
        raise ValueError(f"{a['name']} and {b['name']} overlap in time – link two specific spots with 'Link Two Spots' instead")
    _push_undo(ctx.session, img, tracks, "Merge Tracks")
    new_tracks = _rebuild(ctx, img, tracks, extra_edges=[(int(src["id"]), int(dst["id"]))], prefer=[tracks.index(a), tracks.index(b)])
    msg = f"Merged {b['name']} into {a['name']}: spot {src['id']} (frame {src['t']}) → spot {dst['id']} (frame {dst['t']}); {len(new_tracks)} tracks"
    return _commit(ctx, img, new_tracks, msg)


@op("tracking.link_two_spots", "Link Two Spots", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("spot_a", "int", -1, min=-1, help="Spot id (Spots table 'ID', Results 'Spot ID' or TrackScheme)"),
            Param("spot_b", "int", -1, min=-1, help="Spot id on another frame; the link goes from the earlier to the later spot")],
    description="Add a lineage link between two spots (TrackScheme ▸ link): a second successor makes a split, a second predecessor a merge, a link between two tracks merges them, a link to an unlinked detection pulls it into the track. Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["link spots", "create link", "add edge"])
def link_two_spots(ctx, spot_a=-1, spot_b=-1):
    img = ctx.img
    tracks, pool, by_id = _model(ctx, img)
    sa, sb = _resolve_spot(by_id, int(spot_a)), _resolve_spot(by_id, int(spot_b))
    if sa is sb:
        raise ValueError("Give two different spots")
    if sa["t"] == sb["t"]:
        raise ValueError("Both spots are on the same frame")
    # spot_a's track keeps its identity when two tracks are joined, whatever the time order
    prefer = [k for k in (_spot_in_track(tracks, int(sa["id"]))[0], _spot_in_track(tracks, int(sb["id"]))[0]) if k is not None]
    if sa["t"] > sb["t"]:
        sa, sb = sb, sa
    ida, idb = int(sa["id"]), int(sb["id"])
    if any((ida, idb) in set(_track_edges(t)) for t in tracks):
        raise ValueError(f"Spots {ida} and {idb} are already linked")
    _push_undo(ctx.session, img, tracks, "Link Spots")
    new_tracks = _rebuild(ctx, img, tracks, extra_spots=[sa, sb], extra_edges=[(ida, idb)], prefer=prefer)
    msg = f"Linked spot {ida} (frame {sa['t']}) → spot {idb} (frame {sb['t']}); {len(new_tracks)} tracks"
    return _commit(ctx, img, new_tracks, msg)


@op("tracking.unlink_spots", "Unlink Two Spots", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("spot_a", "int", -1, min=-1, help="Spot id"), Param("spot_b", "int", -1, min=-1, help="Spot id of a spot linked with the first")],
    description="Remove the lineage link between two spots (TrackScheme ▸ remove link); the track splits when nothing else connects the parts. Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["unlink", "remove link", "delete edge"])
def unlink_spots(ctx, spot_a=-1, spot_b=-1):
    img = ctx.img
    tracks, pool, by_id = _model(ctx, img)
    sa, sb = _resolve_spot(by_id, int(spot_a)), _resolve_spot(by_id, int(spot_b))
    ida, idb = int(sa["id"]), int(sb["id"])
    for k, t in enumerate(tracks):
        edges = _track_edges(t)
        if (ida, idb) in edges or (idb, ida) in edges:
            _push_undo(ctx.session, img, tracks, "Unlink Spots")
            t["edges"] = [[a, b] for a, b in edges if (a, b) not in ((ida, idb), (idb, ida))]
            new_tracks = _rebuild(ctx, img, tracks, prefer=[k])
            created = _created_names(tracks, new_tracks)
            msg = f"Unlinked spots {ida} and {idb}" + (f" → new {', '.join(created)}" if created else "") + f"; {len(new_tracks)} tracks"
            return _commit(ctx, img, new_tracks, msg)
    raise ValueError(f"Spots {ida} and {idb} are not linked")


@op("tracking.add_spot", "Add Spot", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("x", "float", -1, step=0.5, help="Position in pixels (−1 = the current point selection)"), Param("y", "float", -1, step=0.5),
            Param("t", "int", -1, min=-1, label="Frame", help="0-based frame (−1 = the current frame)"),
            Param("radius", "float", MANUAL_RADIUS, min=0.5, step=0.5, label="Spot radius (px)"),
            Param("link_to", "int", -1, min=-1, help="Spot id to link the new spot with (before or after it, by frame); −1 = none"),
            Param("track_id", "int", -1, min=-1, help="With link_to = −1: the track to add the spot to – it is inserted into the link that spans its frame, else linked after the track's latest earlier spot (a second successor = a split) or before its earliest later spot. −1 = a new single-spot track")],
    description="Add a spot (intensity features measured on the current plane) to the track model and link it into a track (TrackScheme ▸ add spot). Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["add spot", "insert spot", "missed detection"])
def add_spot(ctx, x=-1.0, y=-1.0, t=-1, radius=MANUAL_RADIUS, link_to=-1, track_id=-1):
    img = ctx.img
    s = ctx.session
    tracks, pool, by_id = _model(ctx, img)
    if x < 0 or y < 0:
        roi = ctx.roi
        if roi is None or roi.get("type") != "point" or not roi.get("points"):
            raise ValueError("Give x and y or make a point selection")
        x, y = (float(v) for v in roi["points"][-1])
    frame = int(img.display.t) if t < 0 else int(t)
    if frame >= img.nt:
        raise ValueError(f"Frame {frame} is beyond the last frame ({img.nt - 1})")
    # resolve / validate the link target before snapshotting and touching the pool: a failing call must leave neither
    # an orphan spot in the detection nor an undo entry (like the other lineage edits)
    other: dict[str, Any] | None = None
    track: dict[str, Any] | None = None
    if int(link_to) >= 0:
        other = _resolve_spot(by_id, int(link_to))
        if int(other["t"]) == frame:
            raise ValueError(f"Spot {link_to} is on the same frame")
    elif int(track_id) >= 0:
        track = _find_track(tracks, int(track_id))
    _push_undo(s, img, tracks, "Add Spot")
    sid = _next_spot_id(pool, *(tr["spots"] for tr in tracks))
    spot = _manual_spot(img, x, y, frame, radius, spot_id=sid)
    if pool is s.settings.get("spots"):
        pool.append(spot)
    extra_spots: list[dict[str, Any]] = [spot]
    extra_edges: list[tuple[int, int]] = []
    prefer: list[int] = []
    where = "as a new track"
    if other is not None:
        extra_edges = [(int(other["id"]), sid)] if other["t"] < frame else [(sid, int(other["id"]))]
        # the target may be an unlinked detection (in no track): _rebuild keeps only edges whose ends it knows
        extra_spots.append(other)
        k, owner = _spot_in_track(tracks, int(other["id"]))
        prefer = [k] if k is not None else []
        where = f"linked with spot {other['id']}" + (f" of {owner['name']}" if owner else "")
    elif track is not None:
        prefer = [tracks.index(track)]
        tb = {int(sp["id"]): sp for sp in track["spots"]}
        edges = _track_edges(track)
        spanning = [(a, b) for a, b in edges if tb[a]["t"] < frame < tb[b]["t"]]
        if len(spanning) == 1:
            a, b = spanning[0]
            track["edges"] = [[p, q] for p, q in edges if (p, q) != (a, b)]
            extra_edges = [(a, sid), (sid, b)]
            where = f"inserted into {track['name']} between spots {a} and {b}"
        else:
            prev = max((sp for sp in track["spots"] if sp["t"] < frame), key=_spot_key, default=None)
            nxt = min((sp for sp in track["spots"] if sp["t"] > frame), key=_spot_key, default=None)
            if prev is not None:
                extra_edges = [(int(prev["id"]), sid)]
                where = f"linked after spot {prev['id']} of {track['name']}"
            elif nxt is not None:
                extra_edges = [(sid, int(nxt["id"]))]
                where = f"linked before spot {nxt['id']} of {track['name']}"
            else:
                track["spots"].append(spot)
                where = f"as the first spot of {track['name']}"
    new_tracks = _rebuild(ctx, img, tracks, extra_spots=extra_spots, extra_edges=extra_edges, prefer=prefer)
    msg = f"Added spot {sid} at ({x:.1f}, {y:.1f}) on frame {frame} {where}; {len(new_tracks)} tracks"
    return _commit(ctx, img, new_tracks, msg)


@op("tracking.delete_spot", "Delete Spot", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("spot_id", "int", -1, min=-1, help="Spot id; −1 = the spot of track_id on frame t"),
            Param("track_id", "int", -1, min=-1, help="Track id (with spot_id = −1; −1 = the last track)"),
            Param("t", "int", -1, min=-1, label="Frame", help="0-based frame (with spot_id = −1; −1 = the current frame)"),
            Param("bridge", "bool", False, help="Join the spot's predecessor and successor so the track stays in one piece (TrackMate removes the links, so the track splits)")],
    description="Remove a spot and its links from the track model and the stored detection (TrackScheme ▸ delete spot). Undo with Undo Track Edit.",
    keywords=_EDIT_KEYWORDS + ["delete spot", "remove spot", "false detection"])
def delete_spot(ctx, spot_id=-1, track_id=-1, t=-1, bridge=False):
    img = ctx.img
    s = ctx.session
    tracks, pool, by_id = _model(ctx, img)
    frame = int(img.display.t) if t < 0 else int(t)
    sp = _resolve_spot(by_id, int(spot_id), tracks, int(track_id), frame)
    sid = int(sp["id"])
    _push_undo(s, img, tracks, "Delete Spot")
    extra_edges: list[tuple[int, int]] = []
    k, track = _spot_in_track(tracks, sid)
    if track is not None:
        edges = _track_edges(track)
        preds = [a for a, b in edges if b == sid]
        succs = [b for a, b in edges if a == sid]
        track["edges"] = [[a, b] for a, b in edges if sid not in (a, b)]
        track["spots"] = [q for q in track["spots"] if int(q["id"]) != sid]
        if bridge and len(preds) == 1 and len(succs) == 1:
            extra_edges = [(preds[0], succs[0])]
    for i, q in enumerate(pool):
        if q is sp:
            del pool[i]
            break
    new_tracks = _rebuild(ctx, img, tracks, extra_edges=extra_edges, prefer=[k] if k is not None else [])
    msg = f"Deleted spot {sid} (frame {sp['t']})" + (f" from {track['name']}" if track is not None else "") + f"; {len(new_tracks)} tracks"
    return _commit(ctx, img, new_tracks, msg)


@op("tracking.undo_edit", "Undo Track Edit", CAT, kind="analysis", preview=False, roi_aware=False, params=[],
    description="Restore the track model (tracks, lineage links, stored spots) as it was before the last manual edit or spot filter on this image (Split / Merge / Link / Unlink / Add / Delete Spot, Manual Tracking, Filter Spots). Up to 25 steps.",
    keywords=_EDIT_KEYWORDS + ["undo", "revert"])
def undo_edit(ctx):
    img = ctx.img
    s = ctx.session
    stack = s.settings.get("tracks_undo") or []
    entry = None
    for i in range(len(stack) - 1, -1, -1):
        if stack[i].get("image") == img.id:
            entry = stack.pop(i)
            break
    if entry is None:
        raise ValueError("Nothing to undo for this image")
    _tracks_for_image(ctx, img, create=True)  # park another image's tracks first
    if entry["tracks"] is None:
        s.settings.pop("tracks", None)
    else:
        s.settings["tracks"] = entry["tracks"]
    s.settings["tracks_image"] = img.id
    if entry["spots"] is None:
        s.settings.pop("spots", None)
        s.settings.pop("spots_meta", None)
    else:
        s.settings["spots"] = entry["spots"]
        s.settings["spots_meta"] = entry["spots_meta"]
    tracks = s.settings.get("tracks") or []
    remaining = sum(1 for e in stack if e.get("image") == img.id)
    msg = f"Undid {entry['label']} ({remaining} more undo step{'s' if remaining != 1 else ''})"
    return OpResult(overlay=_all_track_overlay(img, tracks), table=_tracks_table(tracks), message=msg, log=msg)


# ================================================================================ display settings
def _lut_table(name: str) -> np.ndarray:
    from ..luts import LUT_NAMES, get_lut
    try:
        t = np.asarray(get_lut(name))
    except KeyError:
        raise ValueError(f"Unknown LUT {name!r}; one of {', '.join(LUT_NAMES[:12])} …") from None
    if t.ndim == 2 and t.shape[0] == 3 and t.shape[1] != 3:
        t = t.T
    return t


def lut_hex(table: np.ndarray, value: float, lo: float, hi: float) -> str:
    """The LUT colour of ``value`` in [lo, hi] as '#rrggbb' (grey for undefined values)."""
    if not np.isfinite(value):
        return "#808080"
    u = (value - lo) / (hi - lo) if hi > lo else 0.5
    row = table[int(round(min(max(u, 0.0), 1.0) * (len(table) - 1)))]
    return "#%02x%02x%02x" % (int(row[0]), int(row[1]), int(row[2]))


@op("tracking.set_display", "Track Display Settings", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("color_by", "choice", "Track colour", choices=COLOR_BY, label="Colour by", help="Each track's own colour, or a track / spot feature through the colormap (track features recolour the overlays; spot features colour the spots in the TrackScheme and the tracks by their mean)"),
            Param("colormap", "string", "Turbo", help="LUT for feature colouring: Turbo, Viridis, Fire, Jet, Ice … (any registered LUT)"),
            Param("display_mode", "choice", "Full tracks", choices=DISPLAY_MODES, help="How much of each track the viewer draws around the current frame"),
            Param("tail_length", "int", 10, min=1, max=10000, help="Frames drawn in the local-tails / fade modes"),
            Param("spot_radius", "bool", True, label="Draw spot radius", help="Spots as circles of their detection radius (unticked: fixed dots)"),
            Param("show_names", "bool", False, label="Show spot names")],
    description="TrackMate's display settings: colour tracks / spots by a feature with a colormap, track display mode (full / local tails / up to the current frame / fade), tail length, spot radius and spot names. Stored for the viewer and the TrackScheme; feature colouring recolours the track overlays and the Tracks table.",
    keywords=["trackmate", "display settings", "colour by", "colormap", "tail", "lineage"])
def set_display(ctx, color_by="Track colour", colormap="Turbo", display_mode="Full tracks", tail_length=10, spot_radius=True, show_names=False):
    img = ctx.img
    s = ctx.session
    table = _lut_table(colormap)
    tracks = _tracks_for_image(ctx, img, create=True)
    _ensure_ids(spot_pool_for(s, img), *(t["spots"] for t in tracks))
    feature: dict[str, Any] | None = None
    values: list[float] = []
    if color_by.startswith("Track: "):
        name = color_by[len("Track: "):]
        values = [_spot_value(t.get("stats") or {}, name) for t in tracks]
        feature = {"kind": "track", "name": name, "key": name}
    elif color_by.startswith("Spot: "):
        name = color_by[len("Spot: "):]
        key = SPOT_FEATURES[name]
        per_track = []
        for t in tracks:
            v = np.array([_spot_value(sp, key) for sp in t["spots"]], dtype=np.float64)
            per_track.append(float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan"))
        all_v = np.array([_spot_value(sp, key) for t in tracks for sp in t["spots"]], dtype=np.float64)
        feature = {"kind": "spot", "name": name, "key": key}
        values = per_track
        finite_all = all_v[np.isfinite(all_v)]
        feature["lo"], feature["hi"] = (float(finite_all.min()), float(finite_all.max())) if finite_all.size else (0.0, 0.0)
    if feature is not None:
        finite = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
        if feature["kind"] == "track":
            feature["lo"], feature["hi"] = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        lo, hi = feature["lo"], feature["hi"]
        for t, v in zip(tracks, values):
            t.setdefault("own_color", t["color"])
            t["color"] = lut_hex(table, v, lo, hi)
    else:
        for t in tracks:
            if "own_color" in t:
                t["color"] = t.pop("own_color")
    s.settings["track_display"] = {"color_by": color_by, "colormap": colormap, "display_mode": display_mode, "tail_length": int(tail_length),
                                   "spot_radius": bool(spot_radius), "show_names": bool(show_names), "feature": feature}
    msg = f"Track display: {color_by}" + (f" ({colormap}, {feature['lo']:.4g} – {feature['hi']:.4g})" if feature else "") + f", {display_mode}" + \
        (f" (tail {int(tail_length)} frames)" if display_mode != "Full tracks" else "") + (", spot names" if show_names else "")
    return OpResult(overlay=_all_track_overlay(img, tracks), table=_tracks_table(tracks), message=msg, log=msg)
