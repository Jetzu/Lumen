"""Analyze menu — skeleton graphs (Fiji's AnalyzeSkeleton) and Costes colocalization (Coloc 2).

Skeleton analysis is pure numpy/scipy: pixels of a 1-px-wide skeleton are tagged by their
8-neighbour (2-D) or 26-neighbour (3-D) count (end point < 2, slab = 2, junction ≥ 3), junction
pixels are grouped into junction clusters, and branches are traced from every end point and
junction cluster through the slab pixels. Branch lengths are the sum of the calibrated steps
between consecutive pixels (√2 for a diagonal step on square pixels; the voxel depth for z steps
of a 3-D skeleton). The same tracer serves the per-plane (2-D) and whole-stack (3-D) modes.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from skimage import measure as skmeas, morphology as skm

from ..image import LumenImage
from ..registry import Param, OpResult, op
from ..threshold import auto_threshold
from ._util import binary_image, current_threshold, is_binary_plane, label_image

CAT = "Analyze"

# ============================================================================ skeleton graph
TAG_END, TAG_SLAB, TAG_JUNCTION = 1, 2, 3
SHOW_SKELETON = ["Nothing", "Tagged skeleton", "Longest branch mask", "Longest shortest path mask"]
PRUNE_CYCLES = ["none", "shortest branch"]
SKELETON_MODES = ["2D (per plane)", "3D (whole stack)"]
MAX_LSP_VERTICES = 2000  # all-pairs shortest paths are skipped above this (per skeleton)


def _neighbour_offsets(ndim: int) -> list[tuple[int, ...]]:
    """The 3ⁿ − 1 neighbour offsets (8 in 2-D, 26 in 3-D), shortest steps first: orthogonal, then face-,
    then body-diagonal (the tracer prefers the shorter step when a pixel has several free neighbours)."""
    offs = [o for o in itertools.product((-1, 0, 1), repeat=ndim) if any(o)]
    return sorted(offs, key=lambda o: (sum(abs(v) for v in o), o))


def _thin(mask: np.ndarray) -> np.ndarray:
    """Thin a 2-D plane or a 3-D volume to a 1-voxel-wide skeleton (a one-slice volume is thinned as a plane)."""
    mask = mask.astype(bool)
    if mask.ndim == 3 and mask.shape[0] == 1:
        return skm.skeletonize(mask[0]).astype(bool)[None]
    return skm.skeletonize(mask).astype(bool)


def _trace_skeleton(skel: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    """Build the graph of a 1-px-wide skeleton (boolean 2-D plane or 3-D volume).

    ``spacing`` is the calibrated size of a pixel per array axis: ``(ph, pw)`` for a plane and
    ``(pd, ph, pw)`` for a volume (anisotropic z is honoured in every step length). Pixels are
    8-connected (2-D) or 26-connected (3-D).

    Returns ``tags`` (end = 1, slab = 2, junction = 3), ``comp`` (skeleton id per pixel, 1-based),
    ``n_skeletons``, ``vertices`` {id: {"type": "end" | "junction", "pixels": [(…), …], "skeleton": k}}
    and ``branches`` [{"skeleton", "v1", "v2", "pixels", "length", "euclid", "circular"}, …] where
    ``v1``/``v2`` are vertex ids (0 = none: a closed ring, or an open dead end) and every pixel is an
    index tuple in array order ((y, x) or (z, y, x)).
    """
    on = skel.astype(bool)
    ndim = on.ndim
    shape = on.shape
    sp = [float(s) for s in spacing]
    if len(sp) != ndim:
        raise ValueError(f"spacing has {len(sp)} entries for a {ndim}-D skeleton")
    sk = on.astype(np.uint8)
    kernel = np.ones((3,) * ndim, np.uint8)
    kernel[(1,) * ndim] = 0
    nb = ndi.convolve(sk, kernel, mode="constant", cval=0) * sk
    end = on & (nb < 2)
    junction = nb >= 3
    slab = nb == 2
    tags = np.zeros(shape, np.uint8)
    tags[end] = TAG_END
    tags[slab] = TAG_SLAB
    tags[junction] = TAG_JUNCTION
    comp = skmeas.label(on, connectivity=ndim)
    n_skel = int(comp.max())
    jlab = skmeas.label(junction, connectivity=ndim)
    end_pts = [tuple(p) for p in np.argwhere(end).tolist()]
    n_end = len(end_pts)
    vid = np.zeros(shape, np.int32)
    for k, p in enumerate(end_pts, 1):
        vid[p] = k
    vid[junction] = jlab[junction] + n_end
    vertices: dict[int, dict[str, Any]] = {}
    for k, p in enumerate(end_pts, 1):
        vertices[k] = {"type": "end", "pixels": [p], "skeleton": int(comp[p])}
    junc_pts = [tuple(p) for p in np.argwhere(junction).tolist()]
    for p in junc_pts:
        v = int(vid[p])
        if v not in vertices:
            vertices[v] = {"type": "junction", "pixels": [], "skeleton": int(comp[p])}
        vertices[v]["pixels"].append(p)
    visited = np.zeros(shape, bool)
    branches: list[dict[str, Any]] = []
    offsets = _neighbour_offsets(ndim)
    step_len = {o: math.sqrt(sum((oi * si) ** 2 for oi, si in zip(o, sp))) for o in offsets}

    def dist(p: tuple[int, ...], q: tuple[int, ...]) -> float:
        return math.sqrt(sum(((a - b) * s) ** 2 for a, b, s in zip(p, q, sp)))

    def step(p: tuple[int, ...], q: tuple[int, ...]) -> float:
        return step_len[tuple(b - a for a, b in zip(p, q))]

    def neighbours(p: tuple[int, ...]):
        for o in offsets:
            q = tuple(a + b for a, b in zip(p, o))
            if all(0 <= qi < si for qi, si in zip(q, shape)) and on[q]:
                yield q

    def add_branch(path: list[tuple[int, ...]], v1: int, v2: int, length: float, circular: bool = False) -> None:
        p0, p1 = path[0], path[-1]
        branches.append({"skeleton": int(comp[p0]), "v1": v1, "v2": v2, "pixels": path, "length": length,
                         "euclid": 0.0 if circular else dist(p0, p1), "circular": circular})

    def walk(p0: tuple[int, ...], p1: tuple[int, ...]) -> None:
        """Follow slab pixels from vertex pixel p0 through its slab neighbour p1 until a vertex."""
        v_start = int(vid[p0])
        path = [p0, p1]
        visited[p1] = True
        length = step(p0, p1)
        prev, cur = p0, p1
        while True:
            term = same = nxt = None
            for q in neighbours(cur):
                if q == prev:
                    continue
                v = int(vid[q])
                if v:
                    # do not bounce straight back into the junction cluster we started from
                    if v != v_start or len(path) > 2:
                        term = term or q
                    else:
                        same = same or q
                elif nxt is None and not visited[q]:
                    nxt = q
            if term is None and nxt is None:
                term = same  # tiny loop inside a junction cluster (or an open dead end)
            if term is not None:
                length += step(cur, term)
                path.append(term)
                add_branch(path, v_start, int(vid[term]), length)
                return
            if nxt is None:
                add_branch(path, v_start, 0, length)
                return
            prev, cur = cur, nxt
            visited[cur] = True
            path.append(nxt)
            length += step(prev, cur)

    # 1. branches starting at end points (a vertex directly next to the end point is a zero-slab branch)
    for p in end_pts:
        for q in neighbours(p):
            v = int(vid[q])
            if v == 0:
                if not visited[q]:
                    walk(p, q)
            elif v > int(vid[p]):
                add_branch([p, q], int(vid[p]), v, step(p, q))
    # 2. branches between junctions (and loops on a junction)
    for p in junc_pts:
        for q in neighbours(p):
            if vid[q] == 0 and not visited[q]:
                walk(p, q)
    # 3. closed rings without any vertex: one circular branch from an arbitrary slab pixel
    for start in (tuple(p) for p in np.argwhere(slab & ~visited).tolist()):
        if visited[start]:
            continue
        path = [start]
        visited[start] = True
        length = 0.0
        cur = start
        while True:
            nxt = None
            for q in neighbours(cur):
                if vid[q] == 0 and not visited[q]:
                    nxt = q
                    break
            if nxt is None:
                break
            length += step(cur, nxt)
            path.append(nxt)
            visited[nxt] = True
            cur = nxt
        if len(path) > 2 and max(abs(a - b) for a, b in zip(cur, start)) == 1:
            length += step(cur, start)  # close the ring
        add_branch(path, 0, 0, length, circular=True)
    return {"tags": tags, "comp": comp, "n_skeletons": n_skel, "vertices": vertices, "branches": branches,
            "end": end, "slab": slab, "junction": junction}


def _longest_shortest_path(vertex_ids: list[int], branches: list[dict[str, Any]], with_path: bool = False):
    """Longest of all shortest vertex-to-vertex paths of one skeleton (Fiji's 'longest shortest path').

    Returns the calibrated length, or ``(length, branches)`` with ``with_path`` where ``branches`` are
    the graph edges the path runs along (empty when there is no path)."""
    empty = (0.0, []) if with_path else 0.0
    if len(vertex_ids) < 2:
        return empty
    if len(vertex_ids) > MAX_LSP_VERTICES:
        return (float("nan"), []) if with_path else float("nan")
    local = {v: i for i, v in enumerate(vertex_ids)}
    best: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for br in branches:
        a, b = br["v1"], br["v2"]
        if a in local and b in local and a != b:
            key = (min(local[a], local[b]), max(local[a], local[b]))
            if key not in best or br["length"] < best[key][0]:
                best[key] = (br["length"], br)
    if not best:
        return empty
    ii = np.array([k[0] for k in best])
    jj = np.array([k[1] for k in best])
    vv = np.array([v[0] for v in best.values()])
    n = len(vertex_ids)
    g = coo_matrix((vv, (ii, jj)), shape=(n, n)).tocsr()
    if not with_path:
        d = shortest_path(g, directed=False, method="D")
        d = d[np.isfinite(d)]
        return float(d.max()) if d.size else 0.0
    d, pred = shortest_path(g, directed=False, method="D", return_predecessors=True)
    dm = np.where(np.isfinite(d), d, -1.0)
    i, j = np.unravel_index(int(np.argmax(dm)), dm.shape)
    length = float(dm[i, j])
    path: list[dict[str, Any]] = []
    k = int(j)
    while k != int(i):
        pk = int(pred[i, k])
        if pk < 0:
            break
        path.append(best[(min(pk, k), max(pk, k))][1])
        k = pk
    return length, path


def _longest_shortest_path_mask(g: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """0/255 mask of the longest shortest path of every skeleton (all pixels of the branches on the path)."""
    out = np.zeros(shape, np.uint8)
    per_vertices: dict[int, list[int]] = {}
    per_branches: dict[int, list[dict[str, Any]]] = {}
    for v, info in g["vertices"].items():
        per_vertices.setdefault(info["skeleton"], []).append(v)
    for br in g["branches"]:
        per_branches.setdefault(br["skeleton"], []).append(br)
    for k, verts in per_vertices.items():
        _, path = _longest_shortest_path(verts, per_branches.get(k, []), with_path=True)
        for br in path:
            for p in br["pixels"]:
                out[p] = 255
    return out


def _skeleton_summaries(g: dict[str, Any]) -> list[dict[str, Any]]:
    """One summary per skeleton (Fiji's Results columns), keyed by skeleton id (1-based)."""
    n = g["n_skeletons"]
    comp = g["comp"]
    counts_end = np.bincount(comp[g["end"]], minlength=n + 1)
    counts_junc = np.bincount(comp[g["junction"]], minlength=n + 1)
    counts_slab = np.bincount(comp[g["slab"]], minlength=n + 1)
    per_branches: dict[int, list[dict[str, Any]]] = {k: [] for k in range(1, n + 1)}
    for br in g["branches"]:
        per_branches[br["skeleton"]].append(br)
    per_vertices: dict[int, list[int]] = {k: [] for k in range(1, n + 1)}
    for v, info in g["vertices"].items():
        per_vertices[info["skeleton"]].append(v)
    out = []
    for k in range(1, n + 1):
        brs = per_branches[k]
        deg: dict[int, int] = {}
        for br in brs:
            for v in (br["v1"], br["v2"]):
                if v and g["vertices"][v]["type"] == "junction":
                    deg[v] = deg.get(v, 0) + 1
        lengths = [br["length"] for br in brs]
        n_junctions = sum(1 for v in per_vertices[k] if g["vertices"][v]["type"] == "junction")
        out.append({"Skeleton ID": k, "# Branches": len(brs), "# Junctions": n_junctions, "# End-point voxels": int(counts_end[k]),
                    "# Junction voxels": int(counts_junc[k]), "# Slab voxels": int(counts_slab[k]),
                    "Average Branch Length": float(np.mean(lengths)) if lengths else 0.0,
                    "# Triple points": sum(1 for d in deg.values() if d == 3), "# Quadruple points": sum(1 for d in deg.values() if d == 4),
                    "Maximum Branch Length": float(max(lengths)) if lengths else 0.0,
                    "Longest Shortest Path": _longest_shortest_path(per_vertices[k], brs)})
    return out


def _prune_cycles(skel: np.ndarray, g: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Break every cycle by removing its shortest branch (maximum spanning forest over the vertices;
    loops on one junction and vertex-less rings are always cycles)."""
    parent = {v: v for v in g["vertices"]}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    removed = [br for br in g["branches"] if br["circular"]]
    for br in sorted((b for b in g["branches"] if not b["circular"] and b["v1"] and b["v2"]), key=lambda b: -b["length"]):
        a, b = find(br["v1"]), find(br["v2"])
        if a == b:
            removed.append(br)
        else:
            parent[a] = b
    out = skel.astype(bool).copy()
    tags = g["tags"]
    for br in removed:
        for p in br["pixels"]:
            if tags[p] != TAG_JUNCTION:
                out[p] = False
    return out, len(removed)


def _prune_ends(skel: np.ndarray, g: dict[str, Any], min_len: float) -> tuple[np.ndarray, int]:
    """Remove end branches (end point ↔ junction) shorter than ``min_len``; isolated segments are kept."""
    out = skel.astype(bool).copy()
    tags = g["tags"]
    removed = 0
    for br in g["branches"]:
        if br["circular"] or not br["v1"] or not br["v2"]:
            continue
        t1, t2 = g["vertices"][br["v1"]]["type"], g["vertices"][br["v2"]]["type"]
        if (t1 == "end") == (t2 == "end"):
            continue
        if br["length"] < min_len:
            for p in br["pixels"]:
                if tags[p] != TAG_JUNCTION:
                    out[p] = False
            removed += 1
    return out, removed


def _analyze_skeleton(skel: np.ndarray, spacing: tuple[float, ...], prune_ends: float, prune_cycles: str) -> tuple[np.ndarray, dict[str, Any], int, int]:
    """Trace (and optionally prune) one skeleton plane or volume. Returns (skeleton, graph, pruned end branches, pruned cycles).

    The pixels of a pruned branch are removed and the skeleton is thinned again: the junction
    pixels the branch was attached to are no longer junctions and thinning turns them back into a
    plain slab (otherwise a two-branch 'junction' blob would remain).
    """
    skel = skel.astype(bool)
    g = _trace_skeleton(skel, spacing)
    n_cycles = n_ends = 0
    if prune_cycles == "shortest branch" and g["branches"]:
        skel, n_cycles = _prune_cycles(skel, g)
        if n_cycles:
            skel = _thin(skel)
            g = _trace_skeleton(skel, spacing)
    if prune_ends > 0 and g["branches"]:
        skel, n_ends = _prune_ends(skel, g, prune_ends)
        if n_ends:
            skel = _thin(skel)
            g = _trace_skeleton(skel, spacing)
    return skel, g, n_ends, n_cycles


_analyze_plane = _analyze_skeleton  # former name


def _plane_mask(ctx, plane: np.ndarray, thr: tuple[float, float] | None, roi_mask: np.ndarray | None) -> np.ndarray:
    if thr is not None:
        b = (plane >= thr[0]) & (plane <= thr[1])
    elif is_binary_plane(plane):
        b = plane > 0
    else:
        lo, hi = auto_threshold(plane, "Default", roi_mask, True)
        b = (plane >= lo) & (plane <= hi)
    if roi_mask is not None:
        b &= roi_mask
    return b


def _volume_mask(ctx, vol: np.ndarray, thr: tuple[float, float] | None, roi_mask: np.ndarray | None) -> np.ndarray:
    """Foreground of a (Z, Y, X) volume: the current threshold, a binary volume as is, else one Default auto-threshold of the stack."""
    if thr is not None:
        b = (vol >= thr[0]) & (vol <= thr[1])
    elif is_binary_plane(vol):
        b = vol > 0
    else:
        m3 = np.repeat(roi_mask[None], vol.shape[0], axis=0) if roi_mask is not None else None
        lo, hi = auto_threshold(vol, "Default", m3, True)
        b = (vol >= lo) & (vol <= hi)
    if roi_mask is not None:
        b &= roi_mask[None]
    return b


def _skeleton_output(show: str, g: dict[str, Any], skel: np.ndarray) -> np.ndarray | None:
    if show == "Tagged skeleton":
        return g["tags"]
    if show == "Longest branch mask":
        out = np.zeros(skel.shape, np.uint8)
        longest: dict[int, dict[str, Any]] = {}
        for br in g["branches"]:
            if br["skeleton"] not in longest or br["length"] > longest[br["skeleton"]]["length"]:
                longest[br["skeleton"]] = br
        for br in longest.values():
            for p in br["pixels"]:
                out[p] = 255
        return out
    if show == "Longest shortest path mask":
        return _longest_shortest_path_mask(g, skel.shape)
    return None


_SKELETON_PARAMS = [
    Param("mode", "choice", "2D (per plane)", choices=SKELETON_MODES, help="2D: every plane on its own (8-connected pixels). 3D: skeletonize and trace the whole z-stack of the current channel/frame (26-connected voxels; z steps use the voxel depth)"),
    Param("skeletonize_first", "bool", True, help="Thin the mask to a 1-px skeleton first (off: the image already is a skeleton)"),
    Param("prune_ends", "float", 0.0, min=0, step=0.5, label="Prune end branches shorter than", help="Calibrated length; 0 = no pruning"),
    Param("prune_cycles", "choice", "none", choices=PRUNE_CYCLES, help="Break every cycle by removing its shortest branch"),
]


@op("analyze.skeleton", "Analyze Skeleton", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[*_SKELETON_PARAMS,
            Param("show", "choice", "Tagged skeleton", choices=SHOW_SKELETON, help="Tagged skeleton: end = 1, slab = 2, junction = 3 (label image). Longest branch mask: the longest branch of every skeleton. Longest shortest path mask: the longest of the shortest vertex-to-vertex paths of every skeleton"),
            Param("all_planes", "bool", False, label="Process whole stack", help="2D: every plane of the stack; 3D: every channel and frame")],
    description="Fiji's Analyze Skeleton: tag the pixels of a binary skeleton (end points, slabs, junctions), trace its branches and report branches, junctions, end points and branch lengths per skeleton plus a Branches table; per plane (2D) or on the whole z-stack (3D).",
    keywords=["skeleton", "branches", "junctions", "end points", "network", "topology", "analyzeskeleton", "3d"])
def analyze_skeleton(ctx, mode="2D (per plane)", skeletonize_first=True, prune_ends=0.0, prune_cycles="none", show="Tagged skeleton", all_planes=False):
    img = ctx.img
    d = img.display
    cal = img.calibration
    pw, ph = float(cal.pixel_width), float(cal.pixel_height)
    pd = float(cal.pixel_depth) if cal.pixel_depth and cal.pixel_depth > 0 else pw
    three_d = str(mode).startswith("3D")
    spacing = (pd, ph, pw) if three_d else (ph, pw)
    thr = current_threshold(ctx)
    roi_mask = ctx.mask
    if three_d:
        units: list[tuple[int, ...]] = [(t, c) for t in range(img.nt) for c in range(img.nc)] if all_planes else [(d.t, d.c)]
    else:
        units = list(img.iter_planes()) if all_planes else [(d.t, d.z, d.c)]
    rows: list[dict[str, Any]] = []
    branch_rows: list[list[Any]] = []
    out_data = np.zeros(img.shape, np.uint8) if (show != "Nothing" and all_planes) else None
    out_single = None
    totals = {"skeletons": 0, "branches": 0, "junctions": 0, "ends": 0, "pruned_ends": 0, "pruned_cycles": 0}
    for unit in units:
        if three_d:
            t, c = unit
            b = _volume_mask(ctx, img.data[t, :, c], thr, roi_mask)
            skel = _thin(b) if skeletonize_first else b
            pos = {"Ch": c + 1, "Frame": t + 1} if all_planes else {}
        else:
            t, z, c = unit
            b = _plane_mask(ctx, img.data[t, z, c], thr, roi_mask)
            skel = skm.skeletonize(b) if skeletonize_first else b
            pos = {"Ch": c + 1, "Slice": z + 1, "Frame": t + 1} if all_planes else {}
        skel, g, n_pe, n_pc = _analyze_skeleton(skel, spacing, prune_ends, prune_cycles)
        totals["pruned_ends"] += n_pe
        totals["pruned_cycles"] += n_pc
        for summ in _skeleton_summaries(g):
            rows.append({"Label": f"{img.name}#{summ['Skeleton ID']}", **summ, **pos})
            totals["skeletons"] += 1
            totals["branches"] += summ["# Branches"]
            totals["junctions"] += summ["# Junctions"]
            totals["ends"] += summ["# End-point voxels"]
        counter: dict[int, int] = {}
        for br in g["branches"]:
            k = br["skeleton"]
            counter[k] = counter.get(k, 0) + 1
            p0, p1 = br["pixels"][0], br["pixels"][-1]
            if three_d:
                coords = [p0[2] * pw, p0[1] * ph, p0[0] * pd, p1[2] * pw, p1[1] * ph, p1[0] * pd]
            else:
                coords = [p0[1] * pw, p0[0] * ph, p1[1] * pw, p1[0] * ph]
            branch_rows.append([k, counter[k], br["length"], br["euclid"], *coords, *pos.values()])
        plane_out = _skeleton_output(show, g, skel)
        if plane_out is not None:
            if out_data is not None:
                if three_d:
                    out_data[t, :, c] = plane_out
                else:
                    out_data[t, z, c] = plane_out
            else:
                out_single = plane_out
    unit_name = cal.unit if cal.scaled else "px"
    msg = (f"{totals['skeletons']} skeleton{'s' if totals['skeletons'] != 1 else ''}{' (3D)' if three_d else ''}: {totals['branches']} branches, "
           f"{totals['junctions']} junctions, {totals['ends']} end points (lengths in {unit_name})")
    if totals["pruned_cycles"]:
        msg += f"; {totals['pruned_cycles']} cycle branch{'es' if totals['pruned_cycles'] != 1 else ''} pruned"
    if totals["pruned_ends"]:
        msg += f"; {totals['pruned_ends']} end branch{'es' if totals['pruned_ends'] != 1 else ''} pruned"
    if three_d:
        cols = ["Skeleton ID", "Branch", "Length", "Euclidean distance", "V1 x", "V1 y", "V1 z", "V2 x", "V2 y", "V2 z"] + (["Ch", "Frame"] if all_planes else [])
    else:
        cols = ["Skeleton ID", "Branch", "Length", "Euclidean distance", "V1 x", "V1 y", "V2 x", "V2 y"] + (["Ch", "Slice", "Frame"] if all_planes else [])
    res = OpResult(results=rows if rows else None, table={"name": "Branches", "columns": cols, "rows": branch_rows}, message=msg, log=msg)
    if show != "Nothing":
        if out_data is not None:
            data = out_data
        elif three_d:
            data = out_single[None, :, None]  # (1, Z, 1, Y, X)
        else:
            data = out_single
        if show == "Tagged skeleton":
            res.new_images = [label_image(data, f"Tagged skeleton of {img.name}", like=img)]
        elif show == "Longest branch mask":
            res.new_images = [binary_image(data, f"Longest branches of {img.name}", like=img)]
        else:
            res.new_images = [binary_image(data, f"Longest shortest paths of {img.name}", like=img)]
        res.select_image = img.id
    return res


@op("analyze.skeleton_longest_path", "Skeleton Longest Shortest Path", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[*_SKELETON_PARAMS, Param("all_planes", "bool", False, label="Process whole stack", help="2D: every plane of the stack; 3D: every channel and frame")],
    description="Binary mask of the longest shortest path of every skeleton (Fiji's Analyze Skeleton 'longest shortest path': the longest of all shortest vertex-to-vertex paths); the Results and Branches tables are filled as by Analyze Skeleton.",
    keywords=["skeleton", "longest shortest path", "backbone", "geodesic", "analyzeskeleton"])
def skeleton_longest_path(ctx, mode="2D (per plane)", skeletonize_first=True, prune_ends=0.0, prune_cycles="none", all_planes=False):
    return analyze_skeleton(ctx, mode=mode, skeletonize_first=skeletonize_first, prune_ends=prune_ends, prune_cycles=prune_cycles,
                            show="Longest shortest path mask", all_planes=all_planes)


# ============================================================================ Costes colocalization
def _r_from_sums(n: float, sa: float, sb: float, saa: float, sbb: float, sab: float) -> float:
    if n < 2:
        return float("nan")
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((sab - sa * sb / n) / math.sqrt(va * vb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    den = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum() / den) if den > 0 else float("nan")


def _orthogonal_regression(da: np.ndarray, db: np.ndarray, ma: float, mb: float) -> tuple[float, float]:
    """Slope and intercept of the total-least-squares line b = slope * a + intercept (Costes / Coloc 2)."""
    sxx, syy, sxy = float((da * da).sum()), float((db * db).sum()), float((da * db).sum())
    if abs(sxy) < 1e-12 * max(sxx, syy, 1e-300):
        slope = 0.0 if sxx >= syy else 1e12
    else:
        slope = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)) / (2.0 * sxy)
    return slope, mb - slope * ma


def _costes_threshold(av: np.ndarray, bv: np.ndarray) -> dict[str, Any]:
    """Costes automatic thresholds on a 256-bin grid.

    Pixels are binned in a 256 × 256 (A, B) histogram carrying the sums needed for Pearson's r;
    cumulative sums give r of the pixels below any (T_a, T_b) in O(1). T_a is stepped down from the
    maximum along the orthogonal-regression line until r of the pixels below both thresholds is ≤ 0
    (or undefined); if that never happens the thresholds are the minimum (everything above).
    The method presupposes a positive correlation: with a regression slope ≤ 0 (r ≤ 0) the
    thresholds are the channel maxima (nothing colocalized) and a warning is returned.
    """
    n = av.size
    ma, mb = float(av.mean()), float(bv.mean())
    da, db = av - ma, bv - mb
    slope, intercept = _orthogonal_regression(da, db, ma, mb)
    lo_a, hi_a = float(av.min()), float(av.max())
    lo_b, hi_b = float(bv.min()), float(bv.max())
    if hi_a <= lo_a:
        hi_a = lo_a + 1.0
    if hi_b <= lo_b:
        hi_b = lo_b + 1.0
    bw_a, bw_b = (hi_a - lo_a) / 256.0, (hi_b - lo_b) / 256.0
    ia = np.minimum(((av - lo_a) / bw_a).astype(np.int64), 255)
    ib = np.minimum(((bv - lo_b) / bw_b).astype(np.int64), 255)
    idx = ia * 256 + ib
    counts = np.bincount(idx, minlength=65536)[:65536].reshape(256, 256)
    out = {"slope": slope, "intercept": intercept, "counts": counts, "range_a": (lo_a, hi_a), "range_b": (lo_b, hi_b), "n": n, "warning": ""}
    if slope <= 0:
        out.update(threshold_a=hi_a, threshold_b=hi_b, r_below=_pearson(av, bv),
                   warning="no positive correlation between the channels (regression slope ≤ 0): Costes thresholds set to the channel maxima")
        return out

    def cum(weights: np.ndarray | None) -> np.ndarray:
        h = np.bincount(idx, weights=weights, minlength=65536)[:65536].reshape(256, 256).astype(np.float64)
        c = np.zeros((257, 257))
        c[1:, 1:] = h.cumsum(axis=0).cumsum(axis=1)
        return c

    C_n, C_a, C_b, C_aa, C_bb, C_ab = cum(None), cum(da), cum(db), cum(da * da), cum(db * db), cum(da * db)
    ka_stop, r_below = 0, float("nan")
    for ka in range(256, 0, -1):
        ta = lo_a + ka * bw_a
        tb = slope * ta + intercept
        kb = int(np.floor((tb - lo_b) / bw_b)) if np.isfinite(tb) else (256 if tb > 0 else 0)
        kb = max(0, min(256, kb))
        r = _r_from_sums(C_n[ka, kb], C_a[ka, kb], C_b[ka, kb], C_aa[ka, kb], C_bb[ka, kb], C_ab[ka, kb])
        if not r > 0:  # ≤ 0 or undefined
            ka_stop, r_below = ka, r
            break
    ta = lo_a + ka_stop * bw_a
    tb = min(max(slope * ta + intercept, lo_b), hi_b)  # the line may leave B's range at the ends of A's
    out.update(threshold_a=ta, threshold_b=tb, r_below=r_below)
    return out


def _costes_scramble(a: np.ndarray, b: np.ndarray, mask: np.ndarray, block: int, iters: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Pearson's r of A against ``iters`` block-scrambled copies of B.

    The ``block`` × ``block`` blocks lying fully inside the mask are permuted among themselves (the
    Costes test proper); the masked pixels that no full block covers (edge strips, the rim of a
    round selection) are permuted pixel-wise so that they do not carry the observed correlation
    into the null distribution. Only the A·B cross term changes under a permutation, so each
    iteration is a single dot product. Returns (observed r, scrambled r values).
    """
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    M = mask[y0:y1, x0:x1]
    am = a[y0:y1, x0:x1] - float(a[mask].mean())
    bm = b[y0:y1, x0:x1] - float(b[mask].mean())
    numer_obs = float((am * bm)[M].sum())
    denom = math.sqrt(float((am * am)[M].sum()) * float((bm * bm)[M].sum()))
    r_obs = numer_obs / denom if denom > 0 else float("nan")
    if iters <= 0 or denom <= 0:
        return r_obs, np.zeros(0)
    block = max(1, min(int(block), M.shape[0], M.shape[1]))
    nby, nbx = M.shape[0] // block, M.shape[1] // block

    def blocks(m: np.ndarray) -> np.ndarray:
        return m[:nby * block, :nbx * block].reshape(nby, block, nbx, block).transpose(0, 2, 1, 3).reshape(nby * nbx, block * block)

    inside = blocks(M).all(axis=1)
    if inside.sum() < 2:
        raise ValueError(f"Fewer than two {block}×{block} blocks fit inside the selection; use a smaller block size")
    A_in = np.ascontiguousarray(blocks(am)[inside])
    B_in = np.ascontiguousarray(blocks(bm)[inside])
    covered = np.zeros(M.shape, bool)
    covered[:nby * block, :nbx * block] = inside.reshape(nby, nbx).repeat(block, axis=0).repeat(block, axis=1)
    rest = M & ~covered
    a_rest, b_rest = am[rest], bm[rest]
    rs = np.empty(iters)
    for k in range(iters):
        numer = float(np.einsum("ij,ij->", A_in, B_in[rng.permutation(len(A_in))]))
        if a_rest.size:
            numer += float(a_rest @ b_rest[rng.permutation(a_rest.size)])
        rs[k] = numer / denom
    return r_obs, rs


@op("analyze.coloc_costes", "Colocalization (Costes)", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[Param("channel_a", "int", 1, min=1), Param("channel_b", "int", 2, min=1),
            Param("scramble_iterations", "int", 100, min=0, max=10000, label="Randomisation iterations", help="Costes significance test: number of block-scrambled images (0 = skip)"),
            Param("block_size", "int", 5, min=1, max=512, label="Block size (px)", help="Edge of the scrambled blocks ≈ PSF width in pixels"),
            Param("show_scatter", "bool", True, label="Scatter plot image", help="2-D histogram of A vs B (256 × 256, log counts) as a new 32-bit image"),
            Param("seed", "int", 0, min=0, advanced=True, help="Random seed of the scrambling (0 = random)")],
    description="Coloc 2-style colocalization of two channels within the selection: Costes automatic thresholds (orthogonal regression), Pearson (whole and above thresholds), thresholded Manders tM1/tM2, Spearman and the Costes block-scrambling significance test.",
    keywords=["coloc 2", "costes", "manders", "pearson", "spearman", "colocalization", "scatter plot", "randomisation"])
def coloc_costes(ctx, channel_a=1, channel_b=2, scramble_iterations=100, block_size=5, show_scatter=True, seed=0):
    img = ctx.img
    d = img.display
    if img.nc < 2:
        raise ValueError("Needs at least two channels")
    for ch in (channel_a, channel_b):
        if ch > img.nc:
            raise ValueError(f"Channel {ch} does not exist (image has {img.nc})")
    a = img.data[d.t, d.z, channel_a - 1].astype(np.float64)
    b = img.data[d.t, d.z, channel_b - 1].astype(np.float64)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)
    mask = ctx.mask if ctx.mask is not None else np.ones(a.shape, bool)
    av, bv = a[mask], b[mask]
    if av.size < 2:
        raise ValueError("The selection contains fewer than two pixels")
    from scipy.stats import spearmanr
    spear = float(spearmanr(av, bv).statistic) if av.size > 2 else float("nan")
    thr = _costes_threshold(av, bv)
    ta, tb = thr["threshold_a"], thr["threshold_b"]
    above = (av >= ta) & (bv >= tb)
    r_above = _pearson(av[above], bv[above])
    sum_a, sum_b = float(av.sum()), float(bv.sum())
    tm1 = float(av[bv > tb].sum() / sum_a) if sum_a else float("nan")
    tm2 = float(bv[av > ta].sum() / sum_b) if sum_b else float("nan")
    rng = np.random.default_rng(seed if seed > 0 else None)
    r_obs, rs = _costes_scramble(a, b, mask, block_size, scramble_iterations, rng)
    if rs.size:
        p = float((rs >= r_obs).mean())
        r_mean, r_sd = float(rs.mean()), float(rs.std(ddof=1)) if rs.size > 1 else 0.0
    else:
        p = r_mean = r_sd = float("nan")
    frac_above = float(above.mean())
    row = {"Label": img.name, "Channel A": channel_a, "Channel B": channel_b, "Pearson": r_obs, "Pearson (above thresholds)": r_above,
           "Pearson (below thresholds)": thr["r_below"], "Spearman": spear, "Manders tM1": tm1, "Manders tM2": tm2,
           "Threshold A": ta, "Threshold B": tb, "Regression slope": thr["slope"], "Regression intercept": thr["intercept"],
           "% pixels above thresholds": 100.0 * frac_above, "Costes p-value": p, "Scrambled r mean": r_mean, "Scrambled r SD": r_sd,
           "Scrambles": int(rs.size), "Block size": int(block_size)}
    msg = (f"Pearson r = {r_obs:.3f} (above thresholds {r_above:.3f}); Costes thresholds A ≥ {ta:.4g}, B ≥ {tb:.4g}; "
           f"tM1 = {tm1:.3f}, tM2 = {tm2:.3f}; Spearman ρ = {spear:.3f}")
    if rs.size:
        msg += f"; p = {p:.3f} ({rs.size} scrambles, r = {r_mean:.3f} ± {r_sd:.3f})"
    if thr["warning"]:
        msg += f". Warning: {thr['warning']}"
    res = OpResult(results=[row], message=msg, log=msg)
    if show_scatter:
        counts = thr["counts"]  # [bin_a, bin_b]
        scatter = np.log1p(counts.T[::-1].astype(np.float32))  # B increases upwards, A to the right
        sc = LumenImage(scatter, name=f"Scatter A vs B of {img.name}")
        sc.display.channels[0].lut = "Fire"
        sc.display.channels[0].min, sc.display.channels[0].max = 0.0, float(max(scatter.max(), 1e-6))
        (lo_a, hi_a), (lo_b, hi_b) = thr["range_a"], thr["range_b"]
        sc.metadata.update({"scatter": True, "x_axis": f"C{channel_a} ({img.display.channels[channel_a - 1].name or 'A'})",
                            "y_axis": f"C{channel_b} ({img.display.channels[channel_b - 1].name or 'B'})",
                            "x_range": [lo_a, hi_a], "y_range": [lo_b, hi_b], "log_counts": True,
                            "thresholds": {"a": ta, "b": tb}, "regression": {"slope": thr["slope"], "intercept": thr["intercept"]},
                            "source": img.name})
        res.new_images = [sc]
        res.select_image = img.id
    return res
