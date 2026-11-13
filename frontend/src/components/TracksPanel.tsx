/**
 * Inspector ▸ Tracks: the track overlays of the active image as a sortable list (name, spots,
 * duration, speed, displacement from the Tracks table, else derived from the points), selection
 * shared with the viewport (click = select + jump to the track's first frame, ⇧/⌘ extend), the
 * colour-by / label modes of the overlay, "show only selected", fit-to-track, and the tracking
 * ops that act on the list (Filter Tracks, Export CSV, delete, Undo edit, Feature tables, Display).
 * A lineage track is drawn as several overlay items (one per branch: "track-3", "track-3-b1" …);
 * they are grouped into one row here and selected together.  The panel's second mode is the
 * TrackMate wizard (TrackWizard), and the TrackScheme lineage graph opens from here.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, Download, Filter, GitBranch, Palette, Table2, Trash2, Undo2, Wand2 } from "lucide-react";
import { useStore, type TrackColorBy, type TrackLabelMode } from "../store";
import { viewportRef } from "../actions";
import { kbd } from "../platform";
import type { ImageSummary } from "../types";
import { TRACK_LABEL_LIMIT, trackStatRows, trackUnits, tracksBounds, type TrackItem, type TrackStatRow } from "./OverlayItems";
import { numericTrackId, trackKeyOfOverlayId, trackSignature, useTrackStore } from "../trackStore";
import TrackScheme from "./TrackScheme";
import TrackWizard from "./TrackWizard";

type SortKey = "name" | "n" | "duration" | "speed" | "displacement";

const COLOR_MODES: { value: TrackColorBy; label: string }[] = [
  { value: "track", label: "Track colour" },
  { value: "speed", label: "Mean speed" },
  { value: "duration", label: "Duration" },
  { value: "displacement", label: "Net displacement" },
  { value: "id", label: "Track id" },
];
const LABEL_MODES: { value: TrackLabelMode; label: string }[] = [
  { value: "none", label: "None" },
  { value: "selected", label: "Selected + hovered" },
  { value: "all", label: `All (up to ${TRACK_LABEL_LIMIT})` },
];

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
const NO_TRACKS: TrackItem[] = [];

function fmtVal(v: number): string {
  if (!Number.isFinite(v)) return "–";
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

/**
 * One list entry per track: the branch items of a lineage ("track-3", "track-3-b1" …) are merged into a
 * single item under the track's key with all their points (shared split / merge points once), so the
 * statistics rows match the Tracks table by spot count.
 */
function groupBranches(items: TrackItem[]): TrackItem[] {
  const out: TrackItem[] = [];
  const byKey = new Map<string, TrackItem>();
  for (const it of items) {
    const key = trackKeyOfOverlayId(it.id);
    const g = byKey.get(key);
    if (!g) {
      const t: TrackItem = { ...it, id: key, points: (it.points || []).slice() };
      byKey.set(key, t);
      out.push(t);
    } else {
      const seen = new Set(g.points.map((p) => p.join(",")));
      for (const p of it.points || []) if (!seen.has(p.join(","))) g.points.push(p);
    }
  }
  for (const t of out) if (t.points.length > 1) t.points.sort((a, b) => (a[2] ?? 0) - (b[2] ?? 0));
  return out;
}

export default function TracksPanel({ img }: { img: ImageSummary }) {
  const mode = useTrackStore((s) => s.panelMode);
  const setMode = useTrackStore((s) => s.setPanelMode);
  return (
    <div className="tracks-panel">
      <div className="seg tracks-mode" role="tablist">
        <button className={mode === "list" ? "on" : undefined} role="tab" aria-selected={mode === "list"} onClick={() => setMode("list")} title="The tracks of this image">Tracks</button>
        <button className={mode === "wizard" ? "on" : undefined} role="tab" aria-selected={mode === "wizard"} onClick={() => setMode("wizard")} title="TrackMate wizard: Detector → Initial threshold → Spot filters → Linker → Track filters → Display → Export"><Wand2 size={11} /> Wizard</button>
      </div>
      {mode === "wizard" ? <TrackWizard img={img} /> : <TrackList img={img} />}
      <TrackScheme img={img} />
    </div>
  );
}

function TrackList({ img }: { img: ImageSummary }) {
  const overlay = img.overlay;
  const tracksTable = useStore((s) => s.state?.tables?.Tracks);
  const ops = useStore((s) => s.ops);
  const selectedIds = useStore((s) => s.selectedTrackIds);
  const setSelectedIds = useStore((s) => s.setSelectedTrackIds);
  const colorBy = useStore((s) => s.trackColorBy);
  const setColorBy = useStore((s) => s.setTrackColorBy);
  const labelMode = useStore((s) => s.trackLabelMode);
  const setLabelMode = useStore((s) => s.setTrackLabelMode);
  const showTrackLabels = useStore((s) => s.showTrackLabels);
  const setShowTrackLabels = useStore((s) => s.setShowTrackLabels);
  const onlySelected = useStore((s) => s.trackShowOnlySelected);
  const setOnlySelected = useStore((s) => s.setTrackShowOnlySelected);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "name", dir: 1 });
  const listRef = useRef<HTMLDivElement>(null);

  // a track picked in the viewport scrolls into view here
  useEffect(() => {
    const last = selectedIds[selectedIds.length - 1];
    if (!last || !listRef.current) return;
    const row = listRef.current.querySelector<HTMLElement>(`tr[data-id="${CSS.escape(trackKeyOfOverlayId(last))}"]`);
    row?.scrollIntoView({ block: "nearest" });
  }, [selectedIds]);

  // the overlay array is referentially stable across frame steps (display patches keep it), so the
  // rows are only rebuilt when tracks or the Tracks table change
  const items = useMemo(() => (overlay as unknown as Record<string, unknown>[]).filter((o) => o.type === "track") as unknown as TrackItem[], [overlay]);
  const tracks = useMemo(() => groupBranches(items), [items]);
  const idsByKey = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const it of items) {
      const k = trackKeyOfOverlayId(it.id);
      const l = m.get(k);
      if (l) l.push(it.id);
      else m.set(k, [it.id]);
    }
    return m;
  }, [items]);
  const expand = useCallback((keys: string[]) => Array.from(new Set(keys.flatMap((k) => idsByKey.get(k) || [k]))), [idsByKey]);
  // a branch picked in the viewport selects its whole lineage
  useEffect(() => {
    const st = useStore.getState();
    const cur = st.selectedTrackIds;
    if (!cur.length) return;
    const full = expand(Array.from(new Set(cur.map(trackKeyOfOverlayId))));
    if (full.length !== cur.length) st.setSelectedTrackIds(full);
  }, [selectedIds, expand]);
  // the track model (undo depth, lineage) for the buttons and the TrackScheme
  const sig = useMemo(() => trackSignature(overlay as unknown[]), [overlay]);
  useEffect(() => {
    void useTrackStore.getState().fetchModel(img.id, sig);
  }, [img.id, sig]);
  const model = useTrackStore((s) => s.model);
  const undoDepth = model && model.image === img.id ? model.undo_depth : 0;
  const nLineage = model && model.image === img.id ? model.tracks.filter((t) => (t.stats["N splits"] || 0) + (t.stats["N merges"] || 0) > 0).length : 0;
  const rows = useMemo(() => trackStatRows(tracks, tracksTable), [tracks, tracksTable]);
  const units = useMemo(() => trackUnits(img.calibration, tracksTable), [img.calibration, tracksTable]);
  const sorted = useMemo(() => {
    const k = sort.key;
    const d = sort.dir;
    const out = rows.slice();
    out.sort((a, b) => (k === "name" ? collator.compare(a.name, b.name) : a[k] - b[k]) * d || collator.compare(a.name, b.name));
    return out;
  }, [rows, sort]);
  const selectedSet = useMemo(() => new Set(selectedIds.map(trackKeyOfOverlayId)), [selectedIds]);
  const selectedTracks = useMemo(() => (selectedIds.length ? tracks.filter((tr) => selectedSet.has(tr.id)) : NO_TRACKS), [tracks, selectedIds, selectedSet]);

  const filterOp = ops["tracking.filter_tracks"];
  const exportOp = ops["tracking.export_tracks"];
  const deleteOp = ops["tracking.manual_delete_point"];
  const displayOp = ops["tracking.set_display"];
  const tablesOp = ops["tracking.feature_tables"];
  const undoOp = ops["tracking.undo_edit"];
  const canDelete = !!deleteOp && deleteOp.params.some((p) => p.name === "delete_track");

  const jumpTo = useCallback((row: TrackStatRow) => {
    const st = useStore.getState();
    const im = st.image(img.id);
    if (!im || im.shape.t <= 1 || row.start < 0) return;
    const t = Math.max(0, Math.min(im.shape.t - 1, Math.round(row.start)));
    if (t !== im.display.t) st.setPosition(im.id, { t });
  }, [img.id]);

  const pick = useCallback((row: TrackStatRow, e: React.MouseEvent) => {
    const st = useStore.getState();
    const cur = Array.from(new Set(st.selectedTrackIds.map(trackKeyOfOverlayId)));
    let keys: string[];
    if (e.shiftKey && cur.length) {
      // range in the list order shown
      const a = sorted.findIndex((r) => r.id === cur[cur.length - 1]);
      const b = sorted.findIndex((r) => r.id === row.id);
      keys = a < 0 ? [...cur, row.id] : Array.from(new Set([...cur, ...sorted.slice(Math.min(a, b), Math.max(a, b) + 1).map((r) => r.id)]));
    } else if (e.metaKey || e.ctrlKey) keys = cur.includes(row.id) ? cur.filter((i) => i !== row.id) : [...cur, row.id];
    else keys = [row.id];
    st.setSelectedTrackIds(expand(keys));
    if (keys.length === 1 || !(e.shiftKey || e.metaKey || e.ctrlKey)) jumpTo(row);
  }, [sorted, jumpTo, expand]);

  const toggleSort = (key: SortKey) => setSort((s) => (s.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: key === "name" ? 1 : -1 }));

  const fitToTrack = () => {
    const b = tracksBounds(selectedTracks.length ? selectedTracks : tracks);
    if (!b) return;
    const pad = Math.max(12, Math.max(b.w, b.h) * 0.15);
    viewportRef.current?.fitRect(b.x - pad, b.y - pad, b.w + 2 * pad, b.h + 2 * pad);
  };

  const deleteSelected = async () => {
    const st = useStore.getState();
    if (!canDelete || !selectedTracks.length) return;
    const ids = selectedTracks.map((tr) => ({ id: tr.id, n: numericTrackId(tr.id), name: tr.name || tr.id }));
    const bad = ids.filter((x) => x.n === null);
    if (bad.length) return st.toast(`Cannot delete ${bad.map((x) => x.name).join(", ")}: no track number in the overlay id`, "error");
    let done = 0;
    for (const x of ids) {
      const res = await useStore.getState().runOp("tracking.manual_delete_point", { track_id: x.n, t: -1, delete_track: true }, { silent: true, roi: null, imageId: img.id });
      if (!res) break; // the error is already toasted; keep the remaining tracks selected
      done++;
    }
    const after = useStore.getState();
    after.setSelectedTrackIds(after.selectedTrackIds.filter((id) => !ids.slice(0, done).some((x) => x.id === trackKeyOfOverlayId(id))));
    if (done) after.toast(`Deleted ${done} track${done === 1 ? "" : "s"}`, "success");
  };

  const runQuick = (opId: string) => void useStore.getState().runOp(opId, {}, { roi: null, imageId: img.id });

  const effectiveLabelMode: TrackLabelMode = showTrackLabels ? labelMode : "none";
  const changeLabelMode = (mode: TrackLabelMode) => {
    // View ▸ Show Track Labels is the master switch: "None" turns it off, any other mode turns it on
    if (mode === "none") setShowTrackLabels(false);
    else {
      setLabelMode(mode);
      if (!showTrackLabels) setShowTrackLabels(true);
    }
  };

  const onListKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape" && selectedIds.length) {
      setSelectedIds([]);
      e.stopPropagation();
    }
  };

  const th = (key: SortKey, label: string, title?: string) => (
    <th className={sort.key === key ? "on" : undefined} onClick={() => toggleSort(key)} title={title || `Sort by ${label}`}>
      {label}{sort.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
    </th>
  );

  return (
    <div className="tracks-list-mode">
      <div className="tracks-controls">
        <label htmlFor="tracks-color-by">Colour by</label>
        <select id="tracks-color-by" value={colorBy} onChange={(e) => setColorBy(e.target.value as TrackColorBy)} title="Colour of the track overlays: each track's own colour, a statistic through the Turbo colormap, or a distinct hue per track">
          {COLOR_MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
        <label htmlFor="tracks-labels">Labels</label>
        <select id="tracks-labels" value={effectiveLabelMode} onChange={(e) => changeLabelMode(e.target.value as TrackLabelMode)} title="Which tracks carry their name in the viewport">
          {LABEL_MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </div>
      <div className="btn-row" style={{ marginTop: 0, marginBottom: 6 }}>
        <button className={"btn sm" + (onlySelected ? " primary" : "")} onClick={() => setOnlySelected(!onlySelected)} title="Draw only the selected tracks while a selection exists">Show only selected</button>
        <button className="btn sm" onClick={fitToTrack} disabled={!tracks.length} title={selectedTracks.length ? "Zoom the view to the selected track(s)" : "Zoom the view to all tracks"}><Crosshair size={12} /> Fit to track</button>
      </div>
      <div ref={listRef} className="tracks-list" tabIndex={0} onKeyDown={onListKey}>
        <table className="tracks">
          <thead>
            <tr>
              {th("name", "Track")}
              {th("n", "N", "Sort by number of spots")}
              {th("duration", "Dur", `Duration (${units.duration})`)}
              {th("speed", "Speed", `Mean speed (${units.length}/${units.time})`)}
              {th("displacement", "Disp", `Net displacement (${units.length})`)}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => <TrackRow key={r.id} row={r} selected={selectedSet.has(r.id)} dimmed={selectedSet.size > 0 && !selectedSet.has(r.id)} onPick={pick} />)}
          </tbody>
        </table>
      </div>
      <div className="btn-row">
        {filterOp && <button className="btn sm" onClick={() => useStore.getState().openSheet(filterOp)} title="Keep only the tracks passing length / duration / displacement / speed criteria"><Filter size={12} /> Filter…</button>}
        {exportOp && <button className="btn sm" onClick={() => useStore.getState().openSheet(exportOp)} title="Export the tracks as CSV (TrackMate or Lumen layout)"><Download size={12} /> Export CSV…</button>}
        <span style={{ flex: 1 }} />
        {canDelete && <button className="btn sm danger" onClick={() => void deleteSelected()} disabled={!selectedTracks.length} title="Delete the selected track(s)"><Trash2 size={12} /> Delete selected</button>}
      </div>
      <div className="btn-row" style={{ marginTop: 4 }}>
        <button className="btn sm" onClick={() => useTrackStore.getState().setSchemeOpen(true)} disabled={!tracks.length} title="Lineage graph (frames × tracks with splits and merges): click a spot to select and jump, edit links, colour by feature, export as SVG"><GitBranch size={12} /> TrackScheme{nLineage ? ` (${nLineage} lineage${nLineage === 1 ? "" : "s"})` : ""}</button>
        {undoOp && <button className="btn sm" disabled={!undoDepth} onClick={() => runQuick("tracking.undo_edit")} title={undoDepth ? `Undo the last track edit (${undoDepth} step${undoDepth === 1 ? "" : "s"})` : "No track edit to undo"}><Undo2 size={12} /> Undo edit</button>}
        {tablesOp && <button className="btn sm" disabled={!tracks.length} onClick={() => runQuick("tracking.feature_tables")} title="TrackMate's Spots / Edges / Track statistics tables"><Table2 size={12} /> Feature tables</button>}
        {displayOp && <button className="btn sm" onClick={() => useStore.getState().openSheet(displayOp)} title="Colour by a track / spot feature with a colormap, track display mode, tail length, spot radius and names"><Palette size={12} /> Display…</button>}
      </div>
      <div className="hint-text">
        Click a track here or in the viewport to select it ({kbd("⇧")} / {kbd("⌘")}-click for several, Esc clears); selecting jumps to its first frame. Units: {units.length}, {units.duration === "frames" ? "frames" : units.duration}{rows.length && !rows[0].fromTable ? " (from the overlay points; run Link Spots for calibrated statistics)" : ""}.{nLineage ? " Lineages (splits / merges) are drawn as one polyline per branch." : ""}
      </div>
    </div>
  );
}

/** One list row; memoised so a frame step or a selection change elsewhere does not re-render 500 rows. */
const TrackRow = React.memo(function TrackRow({ row, selected, dimmed, onPick }: { row: TrackStatRow; selected: boolean; dimmed: boolean; onPick: (row: TrackStatRow, e: React.MouseEvent) => void }) {
  return (
    <tr data-id={row.id} className={(selected ? "sel" : "") + (dimmed ? " dim" : "")} onClick={(e) => onPick(row, e)} title={`${row.name}: frames ${row.start + 1}–${row.end + 1}`}>
      <td><span className="nm"><i className="sw" style={{ background: row.color }} />{row.name}</span></td>
      <td>{row.n}</td>
      <td>{fmtVal(row.duration)}</td>
      <td>{fmtVal(row.speed)}</td>
      <td>{fmtVal(row.displacement)}</td>
    </tr>
  );
});
