/**
 * Plugin Frames (inspector section) – ImageJ's Plugins ▸ New ▸ Plugin Frame: a plugin with its own always-visible window.
 *
 * In Lumen a command declared with `frame=True` / `kind="frame"` (`OpSpec.frame`, served by /api/ops) is a frame: every
 * frame op that has been run gets a docked sub-panel here with the op's parameters as a compact form, a Run / Refresh
 * button and its output (table, plot, log text, message).  A frame run from the palette, a menu or a macro is picked up
 * through store.lastRun and its table / plot captured from the store; the sub-panel's own Run button calls the API
 * directly, so the shared Table / Plot sections stay untouched.  "Add frame…" lists the frame ops that are not open yet.
 */
import React, { useMemo, useState } from "react";
import { create } from "zustand";
import { RefreshCw, X } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import type { OpRunResult, OpSpec, ParamSpec, PlotData, TableData } from "../types";
import Plot from "./Plot";

/** `OpSpec.frame` as served by /api/ops (typed here so types.ts needs no change). */
export function isFrameOp(op: OpSpec | undefined | null): boolean {
  return !!op && (op as OpSpec & { frame?: boolean }).frame === true;
}

export interface FrameEntry {
  opId: string;
  params: Record<string, unknown>;
  table: TableData | null;
  plot: PlotData | null;
  log: string | null;
  message: string | null;
  error: string | null;
  running: boolean;
  /** Time of the last completed run (ms), null before the first one. */
  ran: number | null;
}

interface FrameStore {
  frames: Record<string, FrameEntry>;
  /** Sub-panel order: the order the frames were opened in. */
  order: string[];
  open: (opId: string, params?: Record<string, unknown>) => void;
  close: (opId: string) => void;
  setParams: (opId: string, patch: Record<string, unknown>) => void;
  run: (opId: string) => Promise<void>;
  capture: (opId: string, params: Record<string, unknown>, res: Pick<OpRunResult, "table" | "plot" | "log" | "message">) => void;
}

function defaults(spec: OpSpec | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of spec?.params || []) out[p.name] = p.type === "image" ? null : p.default;
  return out;
}

export const useFrameStore = create<FrameStore>((set, get) => ({
  frames: {},
  order: [],
  open: (opId, params) => {
    const cur = get().frames[opId];
    const spec = useStore.getState().ops[opId];
    const entry: FrameEntry = cur
      ? { ...cur, params: params ? { ...cur.params, ...params } : cur.params }
      : { opId, params: { ...defaults(spec), ...(params || {}) }, table: null, plot: null, log: null, message: null, error: null, running: false, ran: null };
    set({ frames: { ...get().frames, [opId]: entry }, order: cur ? get().order : [...get().order, opId] });
  },
  close: (opId) => {
    const frames = { ...get().frames };
    delete frames[opId];
    set({ frames, order: get().order.filter((id) => id !== opId) });
  },
  setParams: (opId, patch) => {
    const f = get().frames[opId];
    if (f) set({ frames: { ...get().frames, [opId]: { ...f, params: { ...f.params, ...patch } } } });
  },
  run: async (opId) => {
    const st = useStore.getState();
    const spec = st.ops[opId];
    const f = get().frames[opId];
    if (!f) return;
    if (!spec) {
      set({ frames: { ...get().frames, [opId]: { ...f, error: "This command is no longer registered", running: false } } });
      return;
    }
    const img = st.activeImage();
    if (spec.requires_image && !img) {
      set({ frames: { ...get().frames, [opId]: { ...f, error: "Open an image first", running: false } } });
      return;
    }
    set({ frames: { ...get().frames, [opId]: { ...f, running: true, error: null } } });
    try {
      const roi = img ? st.selections[img.id] ?? null : null;
      const res = await api.run(opId, f.params, img?.id ?? null, roi, "stack");
      await useStore.getState().refresh();
      const cur = get().frames[opId] || f;
      set({ frames: { ...get().frames, [opId]: { ...cur, table: res.table ?? null, plot: res.plot ?? null, log: res.log ?? null, message: res.message ?? null, error: null, running: false, ran: Date.now() } } });
      if (res.roi && img) useStore.getState().setSelection(img.id, res.roi);
      if (res.results_added) useStore.getState().togglePanel("results", true);
    } catch (e: any) {
      const cur = get().frames[opId] || f;
      set({ frames: { ...get().frames, [opId]: { ...cur, error: e?.message || String(e), running: false } } });
    }
  },
  capture: (opId, params, res) => {
    get().open(opId, params);
    const cur = get().frames[opId];
    set({ frames: { ...get().frames, [opId]: { ...cur, table: res.table !== undefined ? res.table ?? null : cur.table, plot: res.plot !== undefined ? res.plot ?? null : cur.plot, log: res.log ?? cur.log, message: res.message ?? cur.message, error: null, running: false, ran: Date.now() } } });
  },
}));

// A frame op run anywhere else (palette, menu, macro, console) opens its sub-panel: store.runOp records lastRun first and
// then sets table / plot from the same result, so the next table / plot change (within a moment) belongs to that run.
let expecting: { opId: string; params: Record<string, unknown>; seq: number } | null = null;
useStore.subscribe((s, prev) => {
  if (s.lastRun && s.lastRun !== prev.lastRun && isFrameOp(s.ops[s.lastRun.opId])) {
    const { opId, params, seq } = s.lastRun;
    expecting = { opId, params, seq };
    useFrameStore.getState().open(opId, params);
    s.togglePanel("frames", true);
    window.setTimeout(() => {
      if (expecting?.seq === seq) {
        useFrameStore.getState().capture(opId, params, {}); // no table / plot came: still mark the run
        expecting = null;
      }
    }, 2500);
    return;
  }
  if (expecting && (s.table !== prev.table || s.plot !== prev.plot)) {
    const { opId, params } = expecting;
    expecting = null;
    useFrameStore.getState().capture(opId, params, { table: s.table !== prev.table ? s.table ?? undefined : undefined, plot: s.plot !== prev.plot ? s.plot ?? undefined : undefined });
  }
});

/** Number of open frame sub-panels (the section badge). */
export function useFrameCount(): number {
  return useFrameStore((s) => s.order.length);
}

export default function PluginFrames() {
  const order = useFrameStore((s) => s.order);
  const frames = useFrameStore((s) => s.frames);
  const open = useFrameStore((s) => s.open);
  const ops = useStore((s) => s.ops);
  const frameOps = useMemo(() => Object.values(ops).filter((o) => isFrameOp(o)).sort((a, b) => a.name.localeCompare(b.name)), [ops]);
  const closed = frameOps.filter((o) => !frames[o.id]);
  return (
    <div className="frames">
      {order.map((id) => <FramePanel key={id} entry={frames[id]} spec={ops[id]} />)}
      {order.length === 0 && <div className="hint-text">No plugin frame is open. A frame is a command declared with <code>frame=True</code> (Plugins ▸ New ▸ Plugin Frame): its output lives in its own panel here.</div>}
      {closed.length > 0 && (
        <div className="row">
          <label>Add frame</label>
          <select value="" className="grow" aria-label="Open a plugin frame" onChange={(e) => { if (e.target.value) { open(e.target.value); void useFrameStore.getState().run(e.target.value); } }}>
            <option value="">Choose…</option>
            {closed.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
          </select>
        </div>
      )}
    </div>
  );
}

function FramePanel({ entry, spec }: { entry: FrameEntry; spec: OpSpec | undefined }) {
  const run = useFrameStore((s) => s.run);
  const close = useFrameStore((s) => s.close);
  const setParams = useFrameStore((s) => s.setParams);
  const [showParams, setShowParams] = useState(true);
  const title = spec?.name || entry.opId;
  const params = (spec?.params || []).filter((p) => p.type !== "image");
  return (
    <div className={"frame" + (entry.error ? " error" : "")} data-frame={entry.opId}>
      <div className="frame-head">
        <span className="frame-title" title={spec?.description || entry.opId}>{title}</span>
        {entry.ran && <span className="frame-when">{new Date(entry.ran).toLocaleTimeString()}</span>}
        {params.length > 0 && <button className={"btn sm" + (showParams ? " primary" : "")} onClick={() => setShowParams(!showParams)} title="Show / hide the parameters">Params</button>}
        <button className="btn sm" disabled={entry.running} onClick={() => void run(entry.opId)} title={entry.ran ? "Run again with these parameters" : "Run"}>
          <RefreshCw size={12} className={entry.running ? "spin" : undefined} /> {entry.ran ? "Refresh" : "Run"}
        </button>
        <button className="icon-btn" style={{ width: 22, height: 22 }} onClick={() => close(entry.opId)} title="Close this frame"><X size={13} /></button>
      </div>
      {showParams && params.length > 0 && (
        <div className="frame-params">
          {params.map((p) => <FrameParam key={p.name} spec={p} value={entry.params[p.name]} onChange={(v) => setParams(entry.opId, { [p.name]: v })} />)}
        </div>
      )}
      {entry.error && <div className="frame-error" role="alert">{entry.error}</div>}
      {entry.message && !entry.error && <div className="frame-message">{entry.message}</div>}
      {entry.table && entry.table.columns.length > 0 && (
        <div className="table-wrap frame-table">
          <table className="results">
            <thead><tr>{entry.table.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>{entry.table.rows.map((r, i) => <tr key={i}>{r.map((v, k) => <td key={k}>{fmtCell(v)}</td>)}</tr>)}</tbody>
          </table>
          {entry.table.rows.length === 0 && <div className="hint-text" style={{ padding: "4px 8px" }}>No rows.</div>}
        </div>
      )}
      {entry.plot && <Plot data={entry.plot} height={140} />}
      {entry.log && <pre className="frame-log">{entry.log}</pre>}
      {!entry.ran && !entry.running && !entry.error && <div className="hint-text">Not run yet.</div>}
    </div>
  );
}

function fmtCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

/** One parameter of the compact form: number, checkbox, select, colour or text input. */
function FrameParam({ spec, value, onChange }: { spec: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  const label = <span className="frame-param-label" title={spec.help}>{spec.label}{spec.unit ? <span className="unit"> {spec.unit}</span> : null}</span>;
  if (spec.type === "bool") {
    return <label className="frame-param check"><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {label}</label>;
  }
  if (spec.type === "choice") {
    return (
      <label className="frame-param">{label}
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>{(spec.choices || []).map((c) => <option key={c} value={c}>{c}</option>)}</select>
      </label>
    );
  }
  if (spec.type === "int" || spec.type === "float") {
    return (
      <label className="frame-param">{label}
        <input type="number" value={value === null || value === undefined || value === "" ? "" : Number(value)} min={spec.min} max={spec.max} step={spec.step ?? (spec.type === "int" ? 1 : "any")}
               onChange={(e) => onChange(e.target.value === "" ? spec.default : spec.type === "int" ? Math.round(Number(e.target.value)) : Number(e.target.value))} />
      </label>
    );
  }
  if (spec.type === "color") {
    return <label className="frame-param">{label}<input type="color" value={String(value || "#ffffff")} onChange={(e) => onChange(e.target.value)} /></label>;
  }
  if (spec.type === "text") {
    return <label className="frame-param wide">{label}<textarea rows={2} value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} /></label>;
  }
  // string / multichoice: free text (multichoice as a comma-separated list)
  const text = Array.isArray(value) ? value.join(", ") : String(value ?? "");
  return <label className="frame-param wide">{label}<input type="text" value={text} onChange={(e) => onChange(spec.type === "multichoice" ? e.target.value.split(",").map((s) => s.trim()).filter(Boolean) : e.target.value)} /></label>;
}
