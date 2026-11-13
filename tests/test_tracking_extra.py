"""TrackMate parity: LAP tracker with splitting / merging / gap closing, lineage edits with undo, spot & track feature
filters, feature tables, TrackMate XML with lineage edges, display settings and the /api/tracking router."""
import csv
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops import tracking as tk
from lumen.core.session import Session
from lumen.server.app import create_app

LAP = {"radius": 2.5, "threshold": 30, "tracker": "LAP tracker", "max_displacement": 6, "gap_closing_frames": 2, "max_gap_distance": 8,
       "allow_splitting": True, "split_max_distance": 8, "allow_merging": True, "merge_max_distance": 8}


def lineage_movie(nt=10, size=160, amp=200.0, sigma=2.0, noise=3.0, seed=1, gap=(4,)):
    """Synthetic cells: A moves right and divides at frame 5 into two daughters moving apart (3, ±4 px/frame);
    B1 and B2 approach each other and merge at frame 6 into B; C moves alone and is missing on the ``gap`` frames."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    data = np.zeros((nt, size, size), np.float32)

    def put(f, x, y):
        f += amp * np.exp(-(((xx + 0.5) - x) ** 2 + ((yy + 0.5) - y) ** 2) / (2 * sigma ** 2))

    for t in range(nt):
        f = np.full((size, size), 10.0, np.float32)
        if t <= 4:
            put(f, 20 + 3 * t, 40)
        else:
            put(f, 20 + 3 * t, 40 - 4 * (t - 4))
            put(f, 20 + 3 * t, 40 + 4 * (t - 4))
        if t <= 5:
            put(f, 100 + 2 * t, 100 - 3 * (6 - t))
            put(f, 100 + 2 * t, 100 + 3 * (6 - t))
        else:
            put(f, 100 + 2 * t, 100)
        if t not in gap:
            put(f, 120, 20 + 3 * t)
        data[t] = f + rng.normal(0, noise, (size, size))
    return data


@pytest.fixture
def session():
    s = Session()
    img = LumenImage(lineage_movie(), name="lineage", axes="TYX")
    img.calibration.frame_interval = 1.0
    img.calibration.time_unit = "s"
    s.add_image(img)
    return s


def by_spots(tracks):
    return {t["stats"]["N spots"]: t for t in tracks}


def track_items(img):
    return [o for o in img.overlay if o["type"] == "track"]


# ---------------------------------------------------------------- LAP tracker: splits, merges, gaps, trackers
def test_lap_tracker_detects_split_merge_and_gap(session):
    img = session.active
    out = session.run_op("tracking.track", LAP)
    tracks = session.settings["tracks"]
    assert len(tracks) == 3 and "3 tracks from 40 of 40 spots" in out["message"] and "1 split, 1 merge" in out["message"] and "1 gap closed" in out["message"]
    t = by_spots(tracks)
    a, b, c = t[15], t[16], t[9]
    assert a["stats"]["N splits"] == 1 and a["stats"]["N merges"] == 0 and len(a["edges"]) == 14
    assert b["stats"]["N merges"] == 1 and b["stats"]["N splits"] == 0 and len(b["edges"]) == 15
    assert c["stats"]["N gaps"] == 1 and c["stats"]["Longest gap"] == 1 and len(c["edges"]) == 8 and [sp["t"] for sp in c["spots"]] == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    # every spot has a unique id, edges reference them and point forward in time
    ids = [sp["id"] for tr in tracks for sp in tr["spots"]]
    assert len(ids) == len(set(ids)) == 40
    for tr in tracks:
        sp = {s["id"]: s for s in tr["spots"]}
        assert all(sp[x]["t"] < sp[y]["t"] for x, y in tr["edges"])
    # lineage-aware features: daughters move 5 px/frame, the mother 3 px/frame
    assert a["stats"]["Max speed"] == pytest.approx(5.0, abs=0.2) and a["stats"]["Min speed"] == pytest.approx(3.0, abs=0.2)
    assert a["stats"]["Start frame"] == 0 and a["stats"]["End frame"] == 9 and a["stats"]["Duration (frames)"] == 9
    assert {"Median speed", "Mean quality", "N gaps", "Longest gap", "N splits", "N merges"} <= set(a["stats"])
    # one overlay polyline per branch: the daughter branch starts at the mother spot (frame 4) so the lines meet
    items = track_items(img)
    assert len(items) == 3 + 1 + 1
    a_items = [o for o in items if o["track_id"] == a["id"]]
    assert [o["id"] for o in a_items] == [f"track-{a['id']}", f"track-{a['id']}-b1"] and len(a_items[0]["points"]) == 10
    assert [p[2] for p in a_items[1]["points"]] == [4, 5, 6, 7, 8, 9]
    b_items = [o for o in items if o["track_id"] == b["id"]]
    assert len(b_items) == 2 and len(b_items[0]["points"]) == 10 and [p[2] for p in b_items[1]["points"]] == [0, 1, 2, 3, 4, 5, 6]  # ends at the merge spot
    # the Tracks table carries the lineage columns
    cols = session.tables["Tracks"]["columns"]
    assert {"N splits", "N merges", "N gaps", "Longest gap", "Min speed", "Median speed", "Mean quality"} <= set(cols) and cols[-1] == "Color"
    # spot features of TrackMate's analyzers
    sp = a["spots"][0]
    assert {"median", "min", "max", "total", "std", "contrast", "snr"} <= set(sp) and sp["max"] >= sp["median"] >= sp["min"] and 0 < sp["contrast"] <= 1 and sp["snr"] > 0


def test_simple_lap_and_nearest_neighbour_have_no_lineage(session):
    session.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 30, "add_to_manager": False, "add_to_results": False})
    assert all("id" in sp for sp in session.settings["spots"])
    out = session.run_op("tracking.link_spots", {"tracker": "Simple LAP tracker", "max_displacement": 6, "gap_closing_frames": 2, "max_gap_distance": 8,
                                                 "allow_splitting": True, "allow_merging": True})
    tracks = session.settings["tracks"]
    assert len(tracks) == 5 and all(t["stats"]["N splits"] == 0 and t["stats"]["N merges"] == 0 for t in tracks) and "split" not in out["message"]
    assert sorted(t["stats"]["N spots"] for t in tracks) == [5, 6, 9, 10, 10]
    session.run_op("tracking.link_spots", {"tracker": "Nearest neighbour", "max_displacement": 6, "gap_closing_frames": 2})
    tracks = session.settings["tracks"]
    assert len(tracks) == 6 and sorted(t["stats"]["N spots"] for t in tracks) == [4, 5, 5, 6, 10, 10]  # no gap closing: C is cut in two
    assert all(sp["t"] + 1 == sq["t"] for t in tracks for sp, sq in zip(t["spots"], t["spots"][1:]))


def test_feature_penalties_and_alternative_cost(session):
    session.run_op("tracking.track", {**LAP, "linking_penalties": "Quality=1, Mean intensity=0.5", "gap_penalties": "quality=1", "split_penalties": "QUALITY=2",
                                      "merge_penalties": "Contrast: 1", "alternative_cost_factor": 1.5})
    assert len(session.settings["tracks"]) == 3
    with pytest.raises(ValueError, match="Unknown spot feature"):
        session.run_op("tracking.link_spots", {"linking_penalties": "Banana=1"})
    with pytest.raises(ValueError, match="expected 'feature=weight'"):
        session.run_op("tracking.link_spots", {"linking_penalties": "Quality"})
    assert tk._parse_penalties("") == {} and tk._parse_penalties("Quality=1; MEAN_INTENSITY_CH1=0.5") == {"quality": 1.0, "intensity": 0.5}


def test_branches_helper_on_split_and_merge():
    sp = lambda i, t: {"id": i, "t": t, "x": float(i), "y": 0.0}  # noqa: E731
    track = {"id": "0", "name": "T", "color": "#fff", "spots": [sp(1, 0), sp(2, 1), sp(3, 2), sp(4, 2), sp(5, 3), sp(6, 3), sp(7, 4)],
             "edges": [[1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7]]}
    br = [[s["id"] for s in b] for b in tk._branches(track)]
    assert br == [[1, 2, 3, 5, 7], [2, 4, 6, 7]]
    items = tk._track_overlay_items(track)
    assert [o["id"] for o in items] == ["track-0", "track-0-b1"] and all(o["track_id"] == "0" for o in items)
    assert tk._item_track_key(items[1]) == "track-0" and tk._item_track_key({"id": "manual-12-b3"}) == "manual-12"
    # a chain without explicit edges is one branch
    assert [[s["id"] for s in b] for b in tk._branches({"id": "1", "name": "", "color": "", "spots": [sp(9, 0), sp(8, 1)]})] == [[9, 8]]


# ---------------------------------------------------------------- manual lineage edits + undo
def test_split_merge_link_unlink_and_undo(session):
    img = session.active
    session.run_op("tracking.track", LAP)
    a = by_spots(session.settings["tracks"])[15]
    aid = int(a["id"])
    out = session.run_op("tracking.split_track", {"track_id": aid, "frame": 5})
    tracks = session.settings["tracks"]
    assert "split at frame 5" in out["message"] and "2 links cut" in out["message"] and len(tracks) == 5
    mother = next(t for t in tracks if t["id"] == str(aid))
    new = [t for t in tracks if t["name"] in ("Track 3", "Track 4")]
    assert mother["stats"]["N spots"] == 5 and mother["stats"]["N splits"] == 0 and len(new) == 2 and all(t["stats"]["N spots"] == 5 for t in new)
    items = track_items(img)  # mother, two daughters and C are single branches, the merging track still has two
    assert len(items) == 6 and {o["track_id"] for o in items} == {t["id"] for t in tracks} and sum(1 for o in items if o["id"].endswith("-b1")) == 1
    assert [r[0] for r in session.tables["Tracks"]["rows"]] == [t["name"] for t in tracks]
    # undo restores the lineage
    out = session.run_op("tracking.undo_edit", {})
    assert "Undid Split Track" in out["message"]
    tracks = session.settings["tracks"]
    assert len(tracks) == 3 and by_spots(tracks)[15]["stats"]["N splits"] == 1 and len(track_items(img)) == 5
    # merge: split, then join the mother with the first daughter → one 10-spot track keeping the mother's id
    session.run_op("tracking.split_track", {"track_id": aid, "frame": 5})
    d1 = next(t for t in session.settings["tracks"] if t["name"] == "Track 3")
    out = session.run_op("tracking.merge_tracks", {"track_id": aid, "other_id": int(d1["id"])})
    tracks = session.settings["tracks"]
    assert "Merged Track 3 into" in out["message"] and len(tracks) == 4
    merged = next(t for t in tracks if t["id"] == str(aid))
    assert merged["stats"]["N spots"] == 10 and merged["stats"]["N splits"] == 0 and len(merged["edges"]) == 9
    with pytest.raises(ValueError, match="overlap in time"):
        session.run_op("tracking.merge_tracks", {"track_id": aid, "other_id": int(next(t for t in tracks if t["name"] == "Track 4")["id"])})
    session.run_op("tracking.undo_edit", {})
    session.run_op("tracking.undo_edit", {})
    tracks = session.settings["tracks"]
    assert len(tracks) == 3 and by_spots(tracks)[15]["stats"]["N splits"] == 1
    # unlink the split link (mother spot at frame 4 → second daughter at frame 5), then link it back
    a = by_spots(tracks)[15]
    spot = {s["id"]: s for s in a["spots"]}
    mother_spot = next(i for i in spot if sum(1 for e in a["edges"] if e[0] == i) == 2)
    daughters = sorted(e[1] for e in a["edges"] if e[0] == mother_spot)
    out = session.run_op("tracking.unlink_spots", {"spot_a": daughters[1], "spot_b": mother_spot})
    assert "Unlinked" in out["message"] and len(session.settings["tracks"]) == 4
    with pytest.raises(ValueError, match="not linked"):
        session.run_op("tracking.unlink_spots", {"spot_a": daughters[1], "spot_b": mother_spot})
    out = session.run_op("tracking.link_two_spots", {"spot_a": daughters[1], "spot_b": mother_spot})
    tracks = session.settings["tracks"]
    assert "Linked spot" in out["message"] and len(tracks) == 3 and by_spots(tracks)[15]["stats"]["N splits"] == 1
    with pytest.raises(ValueError, match="already linked"):
        session.run_op("tracking.link_two_spots", {"spot_a": mother_spot, "spot_b": daughters[1]})
    with pytest.raises(ValueError, match="same frame"):
        session.run_op("tracking.link_two_spots", {"spot_a": daughters[0], "spot_b": daughters[1]})
    with pytest.raises(ValueError, match="No spot with id"):
        session.run_op("tracking.link_two_spots", {"spot_a": 999, "spot_b": mother_spot})
    # linking two tracks by their spots merges them (the first spot's track keeps its identity)
    c = by_spots(tracks)[9]
    b = by_spots(tracks)[16]
    out = session.run_op("tracking.link_two_spots", {"spot_a": c["spots"][-1]["id"], "spot_b": b["spots"][0]["id"]})
    tracks = session.settings["tracks"]
    assert len(tracks) == 2 and next(t for t in tracks if t["id"] == c["id"])["stats"]["N spots"] == 25
    session.run_op("tracking.undo_edit", {})
    assert len(session.settings["tracks"]) == 3


def test_add_delete_spot_and_undo_depth(session):
    img = session.active
    session.run_op("tracking.track", LAP)
    n_pool = len(session.settings["spots"])
    tracks = session.settings["tracks"]
    a = by_spots(tracks)[15]
    mother_spot = next(s for s in a["spots"] if s["t"] == 4)
    # a third daughter linked to the mother spot
    out = session.run_op("tracking.add_spot", {"x": 36.0, "y": 40.0, "t": 5, "link_to": mother_spot["id"]})
    tracks = session.settings["tracks"]
    a2 = next(t for t in tracks if t["id"] == a["id"])
    assert "linked with spot" in out["message"] and a2["stats"]["N spots"] == 16 and len(a2["edges"]) == 15 and a2["stats"]["N splits"] == 1
    assert len(session.settings["spots"]) == n_pool + 1 and len([o for o in track_items(img) if o["track_id"] == a["id"]]) == 3
    added = next(s for s in a2["spots"] if s["t"] == 5 and abs(s["x"] - 36.0) < 1e-9)
    assert added["manual"] and "intensity" in added and "contrast" in added
    out = session.run_op("tracking.delete_spot", {"spot_id": added["id"]})
    assert "Deleted spot" in out["message"] and by_spots(session.settings["tracks"])[15]["stats"]["N spots"] == 15 and len(session.settings["spots"]) == n_pool
    # a spot added into a gap is inserted into the spanning link
    c = by_spots(session.settings["tracks"])[9]
    out = session.run_op("tracking.add_spot", {"x": 120.0, "y": 32.5, "t": 4, "track_id": int(c["id"])})
    c2 = next(t for t in session.settings["tracks"] if t["id"] == c["id"])
    assert "inserted into" in out["message"] and c2["stats"]["N spots"] == 10 and c2["stats"]["N gaps"] == 0 and len(c2["edges"]) == 9
    # a spot with no link starts a new track; the current point selection / frame are the defaults
    img.display.t = 2
    out = session.run_op("tracking.add_spot", {}, roi={"type": "point", "points": [[80, 80]]})
    assert "as a new track" in out["message"] and len(session.settings["tracks"]) == 4
    # deleting a middle spot cuts the track (TrackMate) unless bridged
    c2 = next(t for t in session.settings["tracks"] if t["id"] == c["id"])
    out = session.run_op("tracking.delete_spot", {"track_id": int(c["id"]), "t": 7})
    assert len(session.settings["tracks"]) == 5 and next(t for t in session.settings["tracks"] if t["id"] == c["id"])["stats"]["N spots"] == 7
    session.run_op("tracking.undo_edit", {})
    out = session.run_op("tracking.delete_spot", {"track_id": int(c["id"]), "t": 7, "bridge": True})
    c3 = next(t for t in session.settings["tracks"] if t["id"] == c["id"])
    assert len(session.settings["tracks"]) == 4 and c3["stats"]["N spots"] == 9 and c3["stats"]["N gaps"] == 1 and c3["stats"]["Longest gap"] == 1
    with pytest.raises(ValueError, match="no spot on frame"):
        session.run_op("tracking.delete_spot", {"track_id": int(c["id"]), "t": 7})
    a_now = next(t for t in session.settings["tracks"] if t["id"] == a["id"])
    with pytest.raises(ValueError, match="give the spot id"):
        session.run_op("tracking.delete_spot", {"track_id": int(a_now["id"]), "t": 6})  # two daughters on frame 6
    # undo all the way back, then nothing is left to undo
    n = 0
    while True:
        try:
            session.run_op("tracking.undo_edit", {})
            n += 1
        except ValueError as e:
            assert "Nothing to undo" in str(e)
            break
    assert n == 5 and len(session.settings["tracks"]) == 3 and len(session.settings["spots"]) == n_pool  # add, delete, add (gap), add (new), delete (bridged)
    assert [o["id"] for o in track_items(img)] == [o["id"] for o in tk._all_track_overlay(img, session.settings["tracks"]) if o["type"] == "track"]
    # manual tracking ops are undoable too
    session.run_op("tracking.manual_new_track", {})
    session.run_op("tracking.manual_add_point", {"x": 5, "y": 5, "t": 0})
    assert len(session.settings["tracks"]) == 4
    session.run_op("tracking.undo_edit", {})
    session.run_op("tracking.undo_edit", {})
    assert len(session.settings["tracks"]) == 3


def test_add_spot_links_to_an_unlinked_detection(session):
    """add_spot(link_to=<a detection in no track>) pulls that spot into the new spot's track: the link is kept in both
    directions (target before or after the new spot) and undo restores the free pool."""
    session.run_op("tracking.track", {**LAP, "min_track_length": 20})
    assert session.settings["tracks"] == [] and len(session.settings["spots"]) == 40
    free = session.settings["spots"][0]
    out = session.run_op("tracking.add_spot", {"x": free["x"] + 1, "y": free["y"], "t": free["t"] + 1, "link_to": free["id"]})
    tracks = session.settings["tracks"]
    assert f"linked with spot {free['id']}" in out["message"] and len(tracks) == 1
    new_id = next(sp["id"] for sp in tracks[0]["spots"] if sp.get("manual"))
    assert sorted(sp["id"] for sp in tracks[0]["spots"]) == sorted([free["id"], new_id]) and tracks[0]["edges"] == [[free["id"], new_id]]
    assert len(session.settings["spots"]) == 41
    # a target on a later frame is linked after the new spot
    in_tracks = {sp["id"] for t in tracks for sp in t["spots"]}
    later = next(sp for sp in session.settings["spots"] if sp["t"] >= 2 and sp["id"] not in in_tracks)
    session.run_op("tracking.add_spot", {"x": later["x"], "y": later["y"] + 1, "t": later["t"] - 1, "link_to": later["id"]})
    tracks = session.settings["tracks"]
    tr = next(t for t in tracks if any(sp["id"] == later["id"] for sp in t["spots"]))
    new2 = next(sp["id"] for sp in tr["spots"] if sp.get("manual"))
    assert len(tracks) == 2 and len(tr["spots"]) == 2 and tr["edges"] == [[new2, later["id"]]]
    session.run_op("tracking.undo_edit", {})
    session.run_op("tracking.undo_edit", {})
    assert session.settings["tracks"] == [] and len(session.settings["spots"]) == 40


def test_failed_add_spot_leaves_no_orphan_spot_or_undo_entry(session):
    """A rejected add_spot (same frame, unknown spot / track, frame past the end) changes nothing: no manual spot stays in
    the detection pool and no 'Add Spot' undo step is recorded."""
    session.run_op("tracking.track", LAP)
    n_pool = len(session.settings["spots"])
    first = session.settings["tracks"][0]["spots"][0]
    for params, msg in (({"x": 50, "y": 50, "t": first["t"], "link_to": first["id"]}, "same frame"),
                        ({"x": 50, "y": 50, "t": 1, "link_to": 99999}, "No spot with id"),
                        ({"x": 50, "y": 50, "t": 1, "track_id": 99999}, "No track with id"),
                        ({"x": 50, "y": 50, "t": 99, "link_to": first["id"]}, "beyond the last frame")):
        with pytest.raises(ValueError, match=msg):
            session.run_op("tracking.add_spot", params)
        assert len(session.settings["spots"]) == n_pool and not session.settings.get("tracks_undo"), msg
        assert not any(sp.get("manual") for sp in session.settings["spots"]), msg
    with pytest.raises(ValueError, match="Nothing to undo"):
        session.run_op("tracking.undo_edit", {})


def test_manual_track_edits_keep_explicit_lineage(session):
    """A manual track that gained explicit edges (through an edit) stays consistent when points are added / replaced / deleted."""
    session.run_op("tracking.manual_new_track", {})
    for t in range(4):
        session.run_op("tracking.manual_add_point", {"track_id": 0, "x": 10 + 3 * t, "y": 10, "t": t})
    track = session.settings["tracks"][0]
    assert track.get("edges") is None
    ids = [s["id"] for s in track["spots"]]
    with pytest.raises(ValueError, match="already linked"):  # the implicit chain counts as links
        session.run_op("tracking.link_two_spots", {"spot_a": ids[1], "spot_b": ids[2]})
    session.run_op("tracking.unlink_spots", {"spot_a": ids[1], "spot_b": ids[2]})  # materialises the edges and cuts the track
    assert len(session.settings["tracks"]) == 2
    session.run_op("tracking.link_two_spots", {"spot_a": ids[1], "spot_b": ids[2]})
    tracks = session.settings["tracks"]
    track = next(t for t in tracks if t["id"] == "0")
    assert len(tracks) == 1 and track.get("edges") is not None and len(track["edges"]) == 3 and track["manual"]
    # manual points added / removed afterwards keep the explicit lineage consistent
    session.run_op("tracking.manual_add_point", {"track_id": 0, "x": 22, "y": 10, "t": 4})
    track = session.settings["tracks"][0]
    assert len(track["spots"]) == 5 and len(track["edges"]) == 4 and track["stats"]["N splits"] == 0
    session.run_op("tracking.manual_delete_point", {"track_id": 0, "t": 2})  # bridged: the manual track stays in one piece
    track = session.settings["tracks"][0]
    assert len(track["spots"]) == 4 and len(track["edges"]) == 3 and track["stats"]["N gaps"] == 1 and len(session.settings["tracks"]) == 1
    session.run_op("tracking.undo_edit", {})
    assert len(session.settings["tracks"][0]["spots"]) == 5


# ---------------------------------------------------------------- filters
def test_filter_spots_before_and_after_linking(session):
    session.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 30, "add_to_manager": False, "add_to_results": False})
    n = len(session.settings["spots"])
    assert n == 40
    out = session.run_op("tracking.filter_spots", {"feature": "X", "threshold": 60, "above": False})
    assert "Kept 15 of 40 spots (X ≤ 60)" in out["message"] and len(session.settings["spots"]) == 15
    session.run_op("tracking.undo_edit", {})
    assert len(session.settings["spots"]) == 40 and "tracks" not in session.settings
    out = session.run_op("tracking.filter_spots", {"feature": "Quality", "threshold": 1e9, "above": True})
    assert "Kept 0 of 40" in out["message"]
    session.run_op("tracking.undo_edit", {})
    # after linking: failing spots leave their tracks and the lineages are recomputed
    session.run_op("tracking.link_spots", {k: v for k, v in LAP.items() if k not in ("radius", "threshold")})
    assert len(session.settings["tracks"]) == 3
    out = session.run_op("tracking.filter_spots", {"feature": "Frame", "threshold": 4, "above": False})
    tracks = session.settings["tracks"]
    assert "removed from tracks" in out["message"] and len(session.settings["spots"]) == 19
    # frames 0–4: the mother, the two cells before their merge and C before its gap – no lineage events left
    assert sorted(t["stats"]["N spots"] for t in tracks) == [4, 5, 5, 5] and all(t["stats"]["N splits"] == 0 and t["stats"]["N merges"] == 0 for t in tracks)
    fresh = Session()
    fresh.add_image(LumenImage(np.zeros((2, 8, 8), np.uint8), name="e", axes="TYX"))
    with pytest.raises(ValueError, match="Detect Spots first"):
        fresh.run_op("tracking.filter_spots", {})


def test_filter_tracks_by_feature(session):
    img = session.active
    session.run_op("tracking.track", LAP)
    out = session.run_op("tracking.filter_tracks", {"feature": "N splits", "threshold": 1, "above": True})
    assert "Kept 1 of 3" in out["message"] and "N splits ≥ 1" in out["message"]
    assert session.settings["tracks"][0]["stats"]["N splits"] == 1 and len(track_items(img)) == 2
    session.run_op("tracking.track", LAP)
    out = session.run_op("tracking.filter_tracks", {"feature": "N gaps", "threshold": 0, "above": False})
    assert "Kept 2 of 3" in out["message"]
    session.run_op("tracking.track", LAP)
    assert "Kept 3 of 3" in session.run_op("tracking.filter_tracks", {"feature": "Mean quality", "threshold": 0, "above": True})["message"]
    assert "Kept 0 of 3" in session.run_op("tracking.filter_tracks", {"feature": "Duration (frames)", "threshold": 100, "above": True})["message"]


# ---------------------------------------------------------------- feature tables + CSV
def test_feature_tables_and_csv_export(session, tmp_path):
    img = session.active
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "µm"
    session.run_op("tracking.track", LAP)
    out = session.run_op("tracking.feature_tables", {})
    assert "40 spots, 37 edges, 3 tracks" in out["message"]
    spots, edges, tracks = session.tables["Spots"], session.tables["Edges"], session.tables["Track statistics"]
    assert spots["columns"][:4] == ["LABEL", "ID", "TRACK_ID", "QUALITY"] and {"POSITION_X", "FRAME", "MEAN_INTENSITY_CH1", "CONTRAST_CH1", "SNR_CH1", "VISIBILITY"} <= set(spots["columns"])
    assert len(spots["rows"]) == 40 and len({r[1] for r in spots["rows"]}) == 40
    assert edges["columns"][:4] == ["LABEL", "TRACK_ID", "SPOT_SOURCE_ID", "SPOT_TARGET_ID"] and len(edges["rows"]) == 37
    assert {"NUMBER_SPLITS", "NUMBER_MERGES", "NUMBER_GAPS", "LONGEST_GAP", "TRACK_MEAN_SPEED", "MEAN_DIRECTIONAL_CHANGE_RATE"} <= set(tracks["columns"])
    rows = [dict(zip(tracks["columns"], r)) for r in tracks["rows"]]
    assert sorted((r["NUMBER_SPOTS"], r["NUMBER_SPLITS"], r["NUMBER_MERGES"], r["NUMBER_GAPS"]) for r in rows) == [(9, 0, 0, 1), (15, 1, 0, 0), (16, 0, 1, 0)]
    assert out["table"]["name"] == "Track statistics"
    # positions follow the TrackMate convention (calibrated, pixel centre i at i * pixel size): a spot at x = 20.0 px → (20 − 0.5) · 0.5 = 9.75 µm
    row = dict(zip(spots["columns"], min(spots["rows"], key=lambda r: r[spots["columns"].index("POSITION_X")])))
    assert row["POSITION_X"] == pytest.approx(9.75, abs=0.15) and row["FRAME"] == 0
    # undefined values are None in the tables (JSON-safe) and NaN in the CSV
    assert all(v is None or isinstance(v, (int, float, str)) for r in edges["rows"] for v in r)
    out = session.run_op("tracking.export_feature_tables", {"path": str(tmp_path / "lin")})
    assert "lin_spots.csv (40 rows)" in out["message"] and "lin_edges.csv (37 rows)" in out["message"] and "lin_tracks.csv (3 rows)" in out["message"]
    with open(tmp_path / "lin_spots.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == spots["columns"] and len(rows) == 41
    with open(tmp_path / "lin_edges.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:3] == ["LABEL", "TRACK_ID", "SPOT_SOURCE_ID"] and any("NaN" in r for r in rows[1:])  # the first edge of a track has no directional change
    session.run_op("tracking.export_feature_tables", {"path": str(tmp_path), "which": "Tracks", "trackmate_headers": True})
    with open(tmp_path / "lineage_tracks.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][1] == "TRACK_ID" and rows[1][1] == "Track ID" and rows[3][1] == "NONE" and len(rows) == 7
    # trimmed in-app tables keep the full count in the CSV
    session.run_op("tracking.feature_tables", {"max_rows": 100})
    assert len(session.tables["Spots"]["rows"]) == 40


# ---------------------------------------------------------------- TrackMate XML with lineage
def test_trackmate_xml_round_trip_keeps_lineage(session, tmp_path):
    img = session.active
    session.run_op("tracking.track", LAP)
    before = {t["stats"]["N spots"]: t for t in session.settings["tracks"]}
    p = tmp_path / "lineage.xml"
    session.run_op("tracking.export_trackmate_xml", {"path": str(p)})
    root = ET.parse(str(p)).getroot()
    tracks = {int(t.get("NUMBER_SPOTS")): t for t in root.findall("Model/AllTracks/Track")}
    assert tracks[15].get("NUMBER_SPLITS") == "1" and tracks[15].get("NUMBER_MERGES") == "0" and len(tracks[15].findall("Edge")) == 14
    assert tracks[16].get("NUMBER_MERGES") == "1" and len(tracks[16].findall("Edge")) == 15
    assert tracks[9].get("NUMBER_GAPS") == "1" and tracks[9].get("LONGEST_GAP") == "1" and len(tracks[9].findall("Edge")) == 8
    decl = [f.get("feature") for f in root.find("Model/FeatureDeclarations/SpotFeatures")]
    assert {"MEDIAN_INTENSITY_CH1", "MIN_INTENSITY_CH1", "MAX_INTENSITY_CH1", "TOTAL_INTENSITY_CH1", "STD_INTENSITY_CH1", "CONTRAST_CH1", "SNR_CH1"} <= set(decl)
    spot = root.find("Model/AllSpots/SpotsInFrame/Spot")
    assert all(spot.get(k) is not None for k in ("MEDIAN_INTENSITY_CH1", "CONTRAST_CH1", "SNR_CH1", "VISIBILITY"))
    # spot IDs in the file are the model's spot ids, so edges refer to the same numbers
    xml_ids = {int(sp.get("ID")) for sp in root.iter("Spot")}
    assert xml_ids == {sp["id"] for t in before.values() for sp in t["spots"]}
    split_edges = {(int(e.get("SPOT_SOURCE_ID")), int(e.get("SPOT_TARGET_ID"))) for e in tracks[15].findall("Edge")}
    assert split_edges == {tuple(e) for e in before[15]["edges"]}
    assert any(e.get("DIRECTIONAL_CHANGE_RATE") not in (None, "NaN", "0.0") for e in tracks[15].findall("Edge"))
    # import: same lineage, same features
    session.run_op("tracking.clear_tracks", {"forget_spots": True})
    out = session.run_op("tracking.import_trackmate_xml", {"path": str(p)})
    assert "Imported 3 tracks (40 linked of 40 spots)" in out["message"] and "2 tracks with splits / merges" in out["message"]
    after = {t["stats"]["N spots"]: t for t in session.settings["tracks"]}
    for n, t in before.items():
        assert {tuple(e) for e in after[n]["edges"]} == {tuple(e) for e in t["edges"]}
        assert after[n]["stats"]["N splits"] == t["stats"]["N splits"] and after[n]["stats"]["N merges"] == t["stats"]["N merges"] and after[n]["stats"]["N gaps"] == t["stats"]["N gaps"]
        assert after[n]["stats"]["Mean speed"] == pytest.approx(t["stats"]["Mean speed"], abs=1e-6)
        sa, sb = t["spots"][0], after[n]["spots"][0]
        assert sa["id"] == sb["id"] and sb["median"] == pytest.approx(sa["median"]) and sb["contrast"] == pytest.approx(sa["contrast"]) and sb["snr"] == pytest.approx(sa["snr"])
    assert len(track_items(img)) == 5
    # a merging track's overlay branch ends at the merge spot after the import too
    b_items = [o for o in track_items(img) if o["track_id"] == after[16]["id"]]
    assert len(b_items) == 2 and b_items[1]["points"][-1][2] == 6


def test_import_hand_written_trackmate_split(tmp_path):
    xml = """<TrackMate version="7.11.1"><Model spatialunits="pixel" timeunits="frame"><AllSpots nspots="5">
<SpotsInFrame frame="0"><Spot ID="1" name="mother" POSITION_X="10.0" POSITION_Y="10.0" POSITION_Z="0.0" FRAME="0" RADIUS="2.0" QUALITY="3.0" MEAN_INTENSITY_CH1="50.0" MEDIAN_INTENSITY_CH1="48.0" CONTRAST_CH1="0.4" SNR_CH1="7.5" VISIBILITY="1"/></SpotsInFrame>
<SpotsInFrame frame="1"><Spot ID="2" POSITION_X="11.0" POSITION_Y="10.0" POSITION_Z="0.0" FRAME="1" RADIUS="2.0"/></SpotsInFrame>
<SpotsInFrame frame="2"><Spot ID="3" POSITION_X="12.0" POSITION_Y="8.0" POSITION_Z="0.0" FRAME="2" RADIUS="2.0"/><Spot ID="4" POSITION_X="12.0" POSITION_Y="12.0" POSITION_Z="0.0" FRAME="2" RADIUS="2.0"/></SpotsInFrame>
<SpotsInFrame frame="4"><Spot ID="5" POSITION_X="14.0" POSITION_Y="7.0" POSITION_Z="0.0" FRAME="4" RADIUS="2.0"/></SpotsInFrame></AllSpots>
<AllTracks><Track name="Lineage" TRACK_ID="7"><Edge SPOT_SOURCE_ID="1" SPOT_TARGET_ID="2"/><Edge SPOT_SOURCE_ID="2" SPOT_TARGET_ID="3"/><Edge SPOT_SOURCE_ID="2" SPOT_TARGET_ID="4"/><Edge SPOT_SOURCE_ID="5" SPOT_TARGET_ID="3"/></Track></AllTracks>
<FilteredTracks><TrackID TRACK_ID="7"/></FilteredTracks></Model></TrackMate>"""
    p = tmp_path / "split.xml"
    p.write_text(xml, encoding="utf-8")
    s = Session()
    img = s.add_image(LumenImage(np.zeros((5, 32, 32), np.uint8), name="z", axes="TYX"))
    out = s.run_op("tracking.import_trackmate_xml", {"path": str(p), "units": "Pixels"})
    assert "Imported 1 tracks (5 linked of 5 spots)" in out["message"] and "1 track with splits / merges" in out["message"]
    t = s.settings["tracks"][0]
    assert t["id"] == "7" and t["name"] == "Lineage" and t["stats"]["N spots"] == 5 and t["stats"]["N splits"] == 1 and t["stats"]["N gaps"] == 1
    assert t["edges"] == [[1, 2], [2, 3], [2, 4], [3, 5]]  # the backwards edge 5 → 3 is turned forward in time
    m = t["spots"][0]
    assert m["name"] == "mother" and m["median"] == 48.0 and m["contrast"] == 0.4 and m["snr"] == 7.5 and m["visibility"] == 1 and m["x"] == 10.5
    items = [o for o in img.overlay if o["type"] == "track"]
    assert [o["id"] for o in items] == ["track-7", "track-7-b1"] and [p[2] for p in items[0]["points"]] == [0, 1, 2, 4] and [p[2] for p in items[1]["points"]] == [1, 2]
    # spot filters work on imported features, and the lineage edits on imported ids
    out = s.run_op("tracking.feature_tables", {})
    assert "5 spots, 4 edges, 1 tracks" in out["message"]
    s.run_op("tracking.split_track", {"track_id": 7, "frame": 2})
    assert len(s.settings["tracks"]) == 3
    s.run_op("tracking.undo_edit", {})
    assert len(s.settings["tracks"]) == 1 and s.settings["tracks"][0]["name"] == "Lineage"
    rois = s.run_op("tracking.tracks_to_rois", {})
    assert "2 track ROIs" in rois["message"] and s.roi_manager.rois[1]["name"] == "Lineage (branch 2)"


# ---------------------------------------------------------------- display settings
def test_display_settings_recolour_tracks(session):
    img = session.active
    session.run_op("tracking.track", LAP)
    own = {t["id"]: t["color"] for t in session.settings["tracks"]}
    out = session.run_op("tracking.set_display", {"color_by": "Track: N spots", "colormap": "Turbo", "display_mode": "Local tails (backward)", "tail_length": 5, "show_names": True})
    d = session.settings["track_display"]
    assert d["color_by"] == "Track: N spots" and d["feature"] == {"kind": "track", "name": "N spots", "key": "N spots", "lo": 9.0, "hi": 16.0} and d["tail_length"] == 5 and d["show_names"]
    assert "Track: N spots (Turbo, 9 – 16)" in out["message"] and "tail 5 frames" in out["message"]
    tracks = session.settings["tracks"]
    colors = {t["stats"]["N spots"]: t["color"] for t in tracks}
    assert len(set(colors.values())) == 3 and all(c.startswith("#") and len(c) == 7 for c in colors.values()) and all(t["own_color"] == own[t["id"]] for t in tracks)
    assert {o["color"] for o in track_items(img)} == set(colors.values())
    tab = session.tables["Tracks"]
    assert [r[-1] for r in tab["rows"]] == [t["color"] for t in tracks]
    # colouring by a spot feature: the range is over all spots, tracks get their mean
    session.run_op("tracking.set_display", {"color_by": "Spot: Quality", "colormap": "Viridis"})
    f = session.settings["track_display"]["feature"]
    assert f["kind"] == "spot" and f["key"] == "quality" and f["lo"] < f["hi"]
    # back to the tracks' own colours
    session.run_op("tracking.set_display", {"color_by": "Track colour"})
    assert {t["id"]: t["color"] for t in session.settings["tracks"]} == own and not any("own_color" in t for t in session.settings["tracks"])
    assert session.settings["track_display"]["feature"] is None
    with pytest.raises(ValueError, match="Unknown LUT"):
        session.run_op("tracking.set_display", {"colormap": "NoSuchLut"})
    # the Tracks panel's display settings survive re-linking (they are not part of the model)
    session.run_op("tracking.set_display", {"display_mode": "Fade", "tail_length": 3})
    session.run_op("tracking.link_spots", {k: v for k, v in LAP.items() if k not in ("radius", "threshold")})
    assert session.settings["track_display"]["display_mode"] == "Fade"


# ---------------------------------------------------------------- router
def test_tracking_api_model_and_histogram():
    s = Session()
    img = s.add_image(LumenImage(lineage_movie(), name="lineage", axes="TYX"))
    s.run_op("tracking.track", LAP)
    s.run_op("tracking.split_track", {"track_id": 0, "frame": 5})
    s.run_op("tracking.set_display", {"color_by": "Spot: Quality"})
    c = TestClient(create_app(s))
    r = c.get("/api/tracking/model", params={"image_id": img.id})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["image"] == img.id and m["nt"] == 10 and len(m["tracks"]) == 5 and m["undo_depth"] == 1 and m["n_spots"] == 40 and m["free_spots"] == []
    t = max(m["tracks"], key=lambda t: len(t["spots"]))
    assert len(t["spots"]) == 16 and len(t["edges"]) == 15 and {"id", "t", "x", "y", "quality", "intensity", "median", "contrast", "snr"} <= set(t["spots"][0])
    assert t["overlay_id"] == f"track-{t['id']}" and t["stats"]["N merges"] == 1 and "N splits" in t["stats"]
    assert m["display"]["color_by"] == "Spot: Quality" and m["display"]["feature"]["kind"] == "spot" and m["display"]["tail_length"] == 10
    assert "Quality" in m["spot_features"] and "N splits" in m["track_features"] and "Turbo" in m["colormaps"] and "Track colour" in m["color_by"] and "Fade" in m["display_modes"]
    assert c.get("/api/tracking/model").status_code == 200  # the active image
    assert c.get("/api/tracking/model", params={"image_id": "nope"}).status_code == 404
    r = c.get("/api/tracking/histogram", params={"image_id": img.id, "feature": "Quality", "bins": 16})
    h = r.json()
    assert r.status_code == 200 and len(h["counts"]) == 16 and len(h["edges"]) == 17 and h["n"] == 40 and h["source"] == "spots" and h["min"] <= h["median"] <= h["max"]
    assert c.get("/api/tracking/histogram", params={"image_id": img.id, "feature": "Banana"}).status_code == 400
    # the feature tables (with None for undefined values) serialise in the session state
    s.run_op("tracking.feature_tables", {})
    r = c.get("/api/state")
    assert r.status_code == 200 and set(r.json()["tables"]) >= {"Spots", "Edges", "Track statistics", "Tracks"}
    # the lineage edit ops are registered with their parameters
    ops = {o["id"]: o for o in c.get("/api/meta").json()["ops"]}
    for oid in ("tracking.filter_spots", "tracking.feature_tables", "tracking.export_feature_tables", "tracking.split_track", "tracking.merge_tracks",
                "tracking.link_two_spots", "tracking.unlink_spots", "tracking.add_spot", "tracking.delete_spot", "tracking.undo_edit", "tracking.set_display"):
        assert oid in ops and ops[oid]["menu"] == "Plugins/Tracking", oid
    assert [p["name"] for p in ops["tracking.link_spots"]["params"]][:5] == ["tracker", "max_displacement", "gap_closing_frames", "max_gap_distance", "allow_splitting"]
