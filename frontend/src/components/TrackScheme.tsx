/**
 * TrackScheme: TrackMate's lineage graph of the active image's tracks – frames on the vertical axis, one
 * column per lineage branch (a split opens a column to the right, a merge joins two), spots as circles and
 * links as lines (dashed over a gap).  Selection is shared with the viewer and the Tracks list: clicking a
 * spot selects its track in the viewer and jumps to its frame; tracks selected elsewhere are highlighted
 * here.  Wheel zooms, dragging pans; the toolbar edits the lineage through the tracking ops (split at the
 * selected spot, delete spot, link / unlink two selected spots, undo), applies the display settings
 * (colour by feature + colormap, spot names) and exports the drawing as SVG.  Opened from the Tracks panel.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal, flushSync } from "react-dom";
import { Download, GitBranch, Link2, Maximize2, Minus, Plus, RefreshCw, Scissors, Trash2, Undo2, Unlink, X } from "lucide-react";
import { useStore } from "../store";
import type { ImageSummary } from "../types";
import { trackKeyOfOverlayId, trackSignature, useTrackStore, type ModelSpot, type ModelTrack, type TrackModel } from "../trackStore";

const CW = 30; // column width (px at zoom 1)
const CH = 24; // frame row height
const R = 6; // spot radius
const GUTTER = 46; // frame-number gutter
const HEADER = 28; // track-name header
const BG = "#15171d";
const BG_ALT = "#1b1e26";
const FG = "#d6d9e0";
const FG_DIM = "#8a8f9c";
const LINE = "#2d313c";

interface Placed { spot: ModelSpot; track: ModelTrack; col: number }
interface Edge { a: Placed; b: Placed; gap: boolean }
interface Span { track: ModelTrack; c0: number; c1: number }
interface Layout { nodes: Placed[]; byId: Map<number, Placed>; edges: Edge[]; spans: Span[]; nCols: number; nFrames: number }

const EMPTY_LAYOUT: Layout = { nodes: [], byId: new Map(), edges: [], spans: [], nCols: 0, nFrames: 0 };

/** Columns per branch: the first child continues its parent's column, every other child opens a new one to the right. */
function layoutModel(model: TrackModel | null): Layout {
  if (!model) return EMPTY_LAYOUT;
  const nodes: Placed[] = [];
  const byId = new Map<number, Placed>();
  const edges: Edge[] = [];
  const spans: Span[] = [];
  let col0 = 0;
  let nFrames = model.nt;
  for (const track of model.tracks) {
    if (!track.spots.length) continue;
    const spots = new Map(track.spots.map((s) => [s.id, s] as const));
    const cmp = (p: number, q: number) => spots.get(p)!.t - spots.get(q)!.t || p - q;
    const succ = new Map<number, number[]>();
    const pred = new Map<number, number[]>();
    for (const [a, b] of track.edges) {
      if (!spots.has(a) || !spots.has(b)) continue;
      if (!succ.has(a)) succ.set(a, []);
      succ.get(a)!.push(b);
      if (!pred.has(b)) pred.set(b, []);
      pred.get(b)!.push(a);
    }
    succ.forEach((list) => list.sort(cmp));
    const roots = [...spots.keys()].filter((id) => !pred.has(id)).sort(cmp);
    const colOf = new Map<number, number>();
    let maxCol = -1;
    for (const root of roots) {
      if (colOf.has(root)) continue;
      maxCol += 1;
      const stack: [number, number][] = [[root, maxCol]];
      while (stack.length) {
        let [id, c] = stack.pop()!;
        if (colOf.has(id)) continue; // reached before (a merge)
        if (c < 0) c = ++maxCol; // a later child: its own column, allocated after the first child's subtree
        colOf.set(id, c);
        const ch = succ.get(id) || [];
        for (let i = ch.length - 1; i >= 0; i--) stack.push([ch[i], i === 0 ? c : -1]);
      }
    }
    for (const sp of track.spots) {
      const c = colOf.get(sp.id);
      if (c === undefined) continue;
      const p: Placed = { spot: sp, track, col: col0 + c };
      nodes.push(p);
      byId.set(sp.id, p);
      if (sp.t + 1 > nFrames) nFrames = sp.t + 1;
    }
    for (const [a, b] of track.edges) {
      const pa = byId.get(a);
      const pb = byId.get(b);
      if (pa && pb) edges.push({ a: pa, b: pb, gap: pb.spot.t - pa.spot.t > 1 });
    }
    spans.push({ track, c0: col0, c1: col0 + maxCol });
    col0 += maxCol + 2;
  }
  return { nodes, byId, edges, spans, nCols: Math.max(col0 - 1, 0), nFrames };
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  return Number.isInteger(v) ? String(v) : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(3);
}

/** Overlay item ids ("track-3", "track-3-b1") of a model track. */
function overlayIdsOf(img: ImageSummary, track: ModelTrack): string[] {
  const out: string[] = [];
  for (const o of img.overlay as unknown as { type?: string; id?: string }[]) {
    if (o.type === "track" && o.id && trackKeyOfOverlayId(o.id) === track.overlay_id) out.push(o.id);
  }
  return out.length ? out : [track.overlay_id];
}

export default function TrackScheme({ img }: { img: ImageSummary }) {
  const open = useTrackStore((s) => s.schemeOpen);
  const close = useCallback(() => useTrackStore.getState().setSchemeOpen(false), []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, close]);
  if (!open) return null;
  return createPortal(
    <div className="modal-backdrop ts-backdrop" onMouseDown={close}>
      <div className="dialog ts-dialog" onMouseDown={(e) => e.stopPropagation()}>
        <SchemeBody img={img} onClose={close} />
      </div>
    </div>,
    document.body,
  );
}

function SchemeBody({ img, onClose }: { img: ImageSummary; onClose: () => void }) {
  const model = useTrackStore((s) => s.model);
  const loading = useTrackStore((s) => s.modelLoading);
  const error = useTrackStore((s) => s.modelError);
  const selectedSpotIds = useTrackStore((s) => s.selectedSpotIds);
  const selectedTrackIds = useStore((s) => s.selectedTrackIds);
  const luts = useStore((s) => s.luts);
  const t = img.display.t;
  const key = useMemo(() => trackSignature(img.overlay as unknown[]), [img.overlay]);
  useEffect(() => {
    void useTrackStore.getState().fetchModel(img.id, key);
  }, [img.id, key]);
  const layout = useMemo(() => layoutModel(model && model.image === img.id ? model : null), [model, img.id]);
  const display = model?.display;
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean; onItem: boolean } | null>(null);
  const [busy, setBusy] = useState(false);

  const selectedTrackKeys = useMemo(() => new Set(selectedTrackIds.map(trackKeyOfOverlayId)), [selectedTrackIds]);
  const selectedSpotSet = useMemo(() => new Set(selectedSpotIds), [selectedSpotIds]);
  const anyTrackSel = selectedTrackKeys.size > 0;

  // ---- colours: the display settings' feature colouring (per spot for spot features; the server recoloured the tracks for track features)
  const lutTable = display?.colormap ? luts[display.colormap] : undefined;
  const spotColor = useCallback((p: Placed): string => {
    const f = display?.feature;
    if (f && f.kind === "spot" && lutTable) {
      const v = (p.spot as unknown as Record<string, number | null>)[f.key];
      if (v === null || v === undefined || !Number.isFinite(v)) return "#808080";
      const u = f.hi > f.lo ? (v - f.lo) / (f.hi - f.lo) : 0.5;
      const row = lutTable[Math.max(0, Math.min(lutTable.length - 1, Math.round(u * (lutTable.length - 1))))];
      return `rgb(${row[0]},${row[1]},${row[2]})`;
    }
    return p.track.color || "#f59e0b";
  }, [display, lutTable]);

  // ---- zoom / pan
  const fit = useCallback(() => {
    const el = wrapRef.current;
    if (!el) return;
    const w = el.clientWidth - GUTTER - 12;
    const h = el.clientHeight - HEADER - 12;
    const need = { w: Math.max(layout.nCols, 1) * CW, h: Math.max(layout.nFrames, 1) * CH };
    const k = Math.max(0.15, Math.min(2.5, Math.min(w / need.w, h / need.h)));
    setView({ k, tx: 0, ty: 0 });
  }, [layout]);
  useEffect(() => {
    fit();
  }, [fit]);
  const zoomBy = (f: number, cx?: number, cy?: number) => {
    setView((v) => {
      const k = Math.max(0.1, Math.min(6, v.k * f));
      const el = wrapRef.current;
      const mx = cx ?? (el ? el.clientWidth / 2 : 0) - GUTTER;
      const my = cy ?? (el ? el.clientHeight / 2 : 0) - HEADER;
      return { k, tx: mx - ((mx - v.tx) * k) / v.k, ty: my - ((my - v.ty) * k) / v.k };
    });
  };
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const rect = wrapRef.current!.getBoundingClientRect();
    zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left - GUTTER, e.clientY - rect.top - HEADER);
  };
  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    // hit-test now: pointerdown is the only event whose target is the element under the pointer (once the canvas
    // has captured the pointer, every later pointer event is retargeted to the canvas div)
    const onItem = !!(e.target as Element).closest?.(".ts-spot, .ts-track-head");
    dragRef.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false, onItem };
  };
  const onMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved) {
      if (Math.hypot(dx, dy) < 3) return;
      d.moved = true;
      // capture only once a drag is under way, so a plain click still reaches the spot / track-head onClick
      // handlers (Chrome ≥ 135 dispatches the click to the capturing element)
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    }
    setView((v) => ({ ...v, tx: d.tx + dx, ty: d.ty + dy }));
  };
  const onUp = (e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (e.type !== "pointerup" || !d || d.moved || d.onItem) return; // a cancelled press, a drag, or a spot / header click
    if (e.shiftKey || e.metaKey || e.ctrlKey) return; // an additive modifier keeps the selection, as in the viewer
    // a plain click on the background clears the selections
    const ts = useTrackStore.getState();
    const st = useStore.getState();
    if (ts.selectedSpotIds.length) ts.setSelectedSpotIds([]);
    if (st.selectedTrackIds.length) st.setSelectedTrackIds([]);
  };

  // ---- selection: a spot click selects its track (viewer + list) and jumps to its frame
  const pickSpot = (p: Placed, e: React.MouseEvent) => {
    e.stopPropagation();
    const ts = useTrackStore.getState();
    const st = useStore.getState();
    const add = e.shiftKey || e.metaKey || e.ctrlKey;
    const cur = ts.selectedSpotIds;
    ts.setSelectedSpotIds(add ? (cur.includes(p.spot.id) ? cur.filter((i) => i !== p.spot.id) : [...cur, p.spot.id]) : [p.spot.id]);
    const ids = overlayIdsOf(img, p.track);
    st.setSelectedTrackIds(add ? Array.from(new Set([...st.selectedTrackIds, ...ids])) : ids);
    const im = st.image(img.id);
    if (im && p.spot.t !== im.display.t && p.spot.t < im.shape.t) st.setPosition(img.id, { t: p.spot.t });
  };

  // ---- edits through the tracking ops
  const run = async (opId: string, params: Record<string, unknown>) => {
    const st = useStore.getState();
    if (!st.ops[opId]) return st.toast(`${opId} is not available`, "error");
    setBusy(true);
    try {
      await st.runOp(opId, params, { silent: false, roi: null, imageId: img.id });
    } finally {
      setBusy(false);
    }
  };
  const selSpots = selectedSpotIds.map((id) => layout.byId.get(id)).filter((p): p is Placed => !!p);
  const one = selSpots.length === 1 ? selSpots[0] : null;
  const two = selSpots.length === 2 ? selSpots : null;
  const twoLinked = two ? layout.edges.some((e) => (e.a.spot.id === two[0].spot.id && e.b.spot.id === two[1].spot.id) || (e.a.spot.id === two[1].spot.id && e.b.spot.id === two[0].spot.id)) : false;
  const splitHere = () => one && void run("tracking.split_track", { track_id: Number(one.track.id), frame: one.spot.t });
  const deleteSpots = async () => {
    for (const p of selSpots) await run("tracking.delete_spot", { spot_id: p.spot.id, track_id: -1, t: -1, bridge: false });
    useTrackStore.getState().setSelectedSpotIds([]);
  };
  const linkTwo = () => two && void run("tracking.link_two_spots", { spot_a: two[0].spot.id, spot_b: two[1].spot.id });
  const unlinkTwo = () => two && void run("tracking.unlink_spots", { spot_a: two[0].spot.id, spot_b: two[1].spot.id });
  const undo = () => void run("tracking.undo_edit", {});
  const setDisplay = (patch: Partial<Record<string, unknown>>) => {
    if (!display) return;
    void run("tracking.set_display", { color_by: display.color_by, colormap: display.colormap, display_mode: display.display_mode, tail_length: display.tail_length, spot_radius: display.spot_radius, show_names: display.show_names, ...patch });
  };

  // ---- export as SVG (standalone: explicit colours, no CSS variables)
  const exportSvg = () => {
    const svg = svgRef.current;
    if (!svg) return;
    // The gutter labels, header boxes, font sizes and the 1/k stroke compensations bake the on-screen zoom into their
    // attributes, so the drawing is cloned at zoom 1 / no pan: commit an identity view synchronously, clone, restore.
    // Both commits happen in this click task, so the intermediate state is never painted (nodeEls / edgeEls are
    // memoised and zoom-independent, so the extra render is cheap).
    const saved = view;
    flushSync(() => setView({ k: 1, tx: 0, ty: 0 }));
    const clone = svg.cloneNode(true) as SVGSVGElement;
    setView(saved);
    const W = GUTTER + Math.max(layout.nCols, 1) * CW + 20;
    const H = HEADER + Math.max(layout.nFrames, 1) * CH + 20;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(W));
    clone.setAttribute("height", String(H));
    clone.setAttribute("viewBox", `0 0 ${W} ${H}`);
    clone.querySelectorAll("[data-view]").forEach((g) => g.setAttribute("transform", g.getAttribute("data-view") || ""));
    const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${img.name.replace(/\.[^.]+$/, "")}-trackscheme.svg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  // ---- rendering
  const k = view.k;
  const cx = (col: number) => (col + 0.5) * CW;
  const cy = (frame: number) => (frame + 0.5) * CH;
  const frames = Array.from({ length: layout.nFrames }, (_, i) => i);
  const showNames = !!display?.show_names;
  const bigSpots = display?.spot_radius !== false;
  const nodeEls = useMemo(() => layout.nodes.map((p) => {
    const sel = selectedSpotSet.has(p.spot.id);
    const tsel = selectedTrackKeys.has(p.track.overlay_id);
    const dim = anyTrackSel && !tsel;
    const r = (bigSpots ? R : R * 0.6) * (sel ? 1.35 : 1);
    const label = p.spot.name || `ID${p.spot.id}`;
    return (
      <g key={p.spot.id} opacity={dim ? 0.3 : 1} className="ts-spot" onClick={(e) => pickSpot(p, e)}>
        <title>{`${p.track.name} · ${label} · frame ${p.spot.t + 1}\nx ${fmt(p.spot.x)}  y ${fmt(p.spot.y)}  quality ${fmt(p.spot.quality)}  mean ${fmt(p.spot.intensity)}`}</title>
        <circle cx={cx(p.col)} cy={cy(p.spot.t)} r={r} fill={spotColor(p)} stroke={sel ? "#ffffff" : tsel ? "#ffd54a" : "rgba(0,0,0,0.55)"} strokeWidth={sel ? 2.2 : tsel ? 1.6 : 1} />
        {showNames && <text x={cx(p.col) + r + 2} y={cy(p.spot.t) + 3.5} fontSize={9} fill={FG} fontFamily="ui-monospace, Menlo, monospace">{label}</text>}
      </g>
    );
  }), [layout, selectedSpotSet, selectedTrackKeys, anyTrackSel, bigSpots, showNames, spotColor]); // eslint-disable-line react-hooks/exhaustive-deps
  const edgeEls = useMemo(() => layout.edges.map((e, i) => {
    const tsel = selectedTrackKeys.has(e.a.track.overlay_id);
    const dim = anyTrackSel && !tsel;
    const both = selectedSpotSet.has(e.a.spot.id) && selectedSpotSet.has(e.b.spot.id);
    return <line key={i} x1={cx(e.a.col)} y1={cy(e.a.spot.t)} x2={cx(e.b.col)} y2={cy(e.b.spot.t)} stroke={both ? "#ffffff" : e.a.track.color || "#f59e0b"} strokeWidth={both ? 2.6 : tsel ? 2 : 1.4} opacity={dim ? 0.25 : 1} strokeDasharray={e.gap ? "4 3" : undefined} strokeLinecap="round" />;
  }), [layout, selectedTrackKeys, anyTrackSel, selectedSpotSet]); // eslint-disable-line react-hooks/exhaustive-deps

  const nTracks = model?.tracks.length ?? 0;
  const viewT = `translate(${GUTTER + view.tx},${HEADER + view.ty}) scale(${k})`;
  return (
    <>
      <div className="dialog-head ts-head">
        <GitBranch size={15} />
        <h3>TrackScheme – {img.name}</h3>
        <span className="hint-text" style={{ margin: 0 }}>{nTracks} tracks · {layout.nodes.length} spots · {layout.edges.length} links{model && model.n_free_spots ? ` · ${model.n_free_spots} unlinked` : ""}</span>
        <span style={{ flex: 1 }} />
        <button className="icon-btn" onClick={onClose} title="Close (Esc)"><X size={14} /></button>
      </div>
      <div className="ts-toolbar">
        <button className="btn sm" disabled={!one || busy} onClick={splitHere} title="Cut the selected spot's track right before its frame (the spot starts a new track)"><Scissors size={12} /> Split here</button>
        <button className="btn sm" disabled={!two || twoLinked || busy} onClick={linkTwo} title="Link the two selected spots (⇧-click to select a second spot)"><Link2 size={12} /> Link</button>
        <button className="btn sm" disabled={!two || !twoLinked || busy} onClick={unlinkTwo} title="Remove the link between the two selected spots"><Unlink size={12} /> Unlink</button>
        <button className="btn sm danger" disabled={!selSpots.length || busy} onClick={() => void deleteSpots()} title="Delete the selected spot(s) and their links"><Trash2 size={12} /> Delete</button>
        <button className="btn sm" disabled={!model?.undo_depth || busy} onClick={undo} title={model?.undo_depth ? `Undo the last track edit (${model.undo_depth} steps)` : "Nothing to undo"}><Undo2 size={12} /> Undo</button>
        <span className="ts-sep" />
        <label className="ts-label" title="Colour the spots / tracks by a feature (tracking.set_display)">Colour
          <select value={display?.color_by || "Track colour"} disabled={!display || busy} onChange={(e) => setDisplay({ color_by: e.target.value })}>
            {(model?.color_by || ["Track colour"]).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <select value={display?.colormap || "Turbo"} disabled={!display || busy || display?.color_by === "Track colour"} onChange={(e) => setDisplay({ colormap: e.target.value })} title="Colormap">
          {(model?.colormaps || ["Turbo"]).map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <label className="switch" title="Draw the spot names (TrackMate's 'show spot names')"><input type="checkbox" checked={showNames} disabled={!display || busy} onChange={(e) => setDisplay({ show_names: e.target.checked })} /> names</label>
        <span className="ts-sep" />
        <button className="icon-btn" onClick={() => zoomBy(1.25)} title="Zoom in"><Plus size={13} /></button>
        <button className="icon-btn" onClick={() => zoomBy(0.8)} title="Zoom out"><Minus size={13} /></button>
        <button className="icon-btn" onClick={fit} title="Fit"><Maximize2 size={13} /></button>
        <button className="icon-btn" onClick={() => void useTrackStore.getState().fetchModel(img.id, key, true)} title="Reload the track model"><RefreshCw size={13} className={loading ? "spin" : undefined} /></button>
        <button className="btn sm" onClick={exportSvg} disabled={!layout.nodes.length} title="Save the lineage graph as an SVG file"><Download size={12} /> SVG</button>
      </div>
      <div ref={wrapRef} className="ts-canvas" onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <svg ref={svgRef} width="100%" height="100%" style={{ display: "block" }}>
          <rect x={0} y={0} width="100%" height="100%" fill={BG} />
          <g data-view={`translate(${GUTTER},${HEADER})`} transform={viewT}>
            {frames.map((f) => (
              <rect key={f} x={-GUTTER / k} y={f * CH} width={Math.max(layout.nCols, 1) * CW + 2 * GUTTER / k} height={CH} fill={f === t ? "rgba(255, 213, 74, 0.16)" : f % 2 ? BG_ALT : BG} />
            ))}
            {layout.spans.map((s) => <rect key={s.track.id} x={s.c0 * CW} y={-HEADER / k} width={(s.c1 - s.c0 + 1) * CW} height={HEADER / k + layout.nFrames * CH} fill="none" stroke={selectedTrackKeys.has(s.track.overlay_id) ? "#ffd54a" : LINE} strokeWidth={selectedTrackKeys.has(s.track.overlay_id) ? 1.5 / k : 1 / k} strokeDasharray={`${3 / k} ${3 / k}`} />)}
            {edgeEls}
            {nodeEls}
          </g>
          {/* frame gutter: pans and scales vertically only */}
          <rect x={0} y={0} width={GUTTER} height="100%" fill={BG} />
          <g data-view={`translate(0,${HEADER})`} transform={`translate(0,${HEADER + view.ty})`}>
            {frames.map((f) => <text key={f} x={GUTTER - 6} y={(f + 0.5) * CH * k + 3.5} fontSize={Math.min(11, Math.max(7, 11 * k))} textAnchor="end" fill={f === t ? "#ffd54a" : FG_DIM} fontFamily="ui-monospace, Menlo, monospace">{f + 1}</text>)}
          </g>
          {/* track header: pans and scales horizontally only */}
          <rect x={0} y={0} width="100%" height={HEADER} fill={BG} />
          <g data-view={`translate(${GUTTER},0)`} transform={`translate(${GUTTER + view.tx},0)`}>
            {layout.spans.map((s) => (
              <g key={s.track.id} className="ts-track-head" onClick={(e) => { e.stopPropagation(); const st = useStore.getState(); st.setSelectedTrackIds(overlayIdsOf(img, s.track)); }}>
                <rect x={s.c0 * CW * k} y={4} width={(s.c1 - s.c0 + 1) * CW * k} height={HEADER - 8} rx={4} fill={s.track.color} opacity={selectedTrackKeys.has(s.track.overlay_id) ? 1 : 0.55} />
                <text x={(s.c0 + (s.c1 - s.c0 + 1) / 2) * CW * k} y={HEADER / 2 + 4} fontSize={10.5} textAnchor="middle" fill="#000" fontWeight={600}>{s.track.name}</text>
                <title>{`${s.track.name}: ${s.track.spots.length} spots, ${s.track.stats["N splits"] ?? 0} splits, ${s.track.stats["N merges"] ?? 0} merges, ${s.track.stats["N gaps"] ?? 0} gaps`}</title>
              </g>
            ))}
          </g>
          <rect x={0} y={0} width={GUTTER} height={HEADER} fill={BG} />
          <text x={GUTTER - 6} y={HEADER / 2 + 4} fontSize={9} textAnchor="end" fill={FG_DIM}>frame</text>
        </svg>
        {!layout.nodes.length && !loading && <div className="ts-empty">{error ? `Could not load the track model: ${error}` : "No tracks on this image – run Track or the wizard first."}</div>}
      </div>
      <div className="dialog-foot ts-foot">
        <span className="hint-text" style={{ margin: 0 }}>
          {one ? `${one.track.name} · ${one.spot.name || `ID${one.spot.id}`} · frame ${one.spot.t + 1} · (${fmt(one.spot.x)}, ${fmt(one.spot.y)}) · quality ${fmt(one.spot.quality)} · mean ${fmt(one.spot.intensity)} · SNR ${fmt(one.spot.snr)}`
            : two ? `${two.length} spots selected (${twoLinked ? "linked" : "not linked"})`
              : "Click a spot: select its track in the viewer and jump to its frame. ⇧-click a second spot to link / unlink. Wheel zooms, drag pans."}
        </span>
        <span className="grow" />
        {display?.feature && <span className="hint-text" style={{ margin: 0 }}>{display.feature.name}: {fmt(display.feature.lo)} – {fmt(display.feature.hi)} ({display.colormap})</span>}
      </div>
    </>
  );
}
