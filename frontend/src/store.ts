import { useMemo } from "react";
import { create } from "zustand";
import { api, type JobInfo } from "./api";
import { pluginToolId, type HistogramData, type ImageSummary, type Meta, type OpRunResult, type OpSpec, type ParamSpec, type PlaneData, type PlotData, type Roi, type SessionState, type TableData, type Tool, type ViewTransform } from "./types";
import type { LutTables } from "./render/compositor";
import { DENSITY_KEY, PREF_DEFAULTS, applyDensity, normalizePrefs, type Prefs } from "./prefs";
import { loadCustomShortcuts, saveCustomShortcuts, type CustomShortcut } from "./customShortcuts";
import { DEFAULT_GROUPS, fetchLutGroups, groupLuts, type LutGroup, type LutGrouping } from "./lutGroups";
import { displayName } from "./menus";
import { runNativeFileDialog, startNativeHooks } from "./native";

export type DocName = "README" | "COMMANDS" | "FIJI_PARITY" | "SCRIPTING" | "MCP" | "RESEARCH" | "DESIGN";

/** A button inside a toast ("Undo clear", "Auto-threshold and run"); the toast closes once it ran. */
export interface ToastAction {
  label: string;
  run: () => void;
}

export interface Toast {
  id: number;
  msg: string;
  kind: "info" | "error" | "success";
  actions?: ToastAction[];
}

/**
 * Tool keys named in the "needs a selection" messages (E5-10 / E7-18): what to draw and which key draws it, derived
 * from the command's `requires_roi` kind.  The keys are the tool rail's (ToolRail.TOOLS): 1 rectangle, 2 oval,
 * 3 polygon, 4 freehand, 5 line, 6 segmented line, 7 points.  A line command never recommends an oval.
 */
const SELECTION_HINTS: Record<string, { kind: string; keys: string; draw: string }> = {
  area: { kind: "an area", keys: "1, 2, 3, 4", draw: "draw a rectangle (1), oval (2), polygon (3) or freehand (4) selection first" },
  line: { kind: "a line", keys: "5, 6", draw: "draw a straight line (5) or a segmented line (6) first" },
  point: { kind: "a point", keys: "7", draw: "place points with the point tool (7) first" },
  line_or_rect: { kind: "a line or rectangle", keys: "5, 6, 1", draw: "draw a straight line (5), a segmented line (6) or a rectangle (1) first" },
  any: { kind: "a", keys: "1–7", draw: "draw a selection first – any of rectangle (1), oval (2), polygon (3), freehand (4), line (5), segmented line (6) or points (7)" },
};

function selectionHint(requires: OpSpec["requires_roi"]): { kind: string; keys: string; draw: string } {
  const k = typeof requires === "string" ? requires.toLowerCase() : "any";
  if (SELECTION_HINTS[k]) return SELECTION_HINTS[k];
  // a kind this client does not know ("line_or_rect" …): name it, list every tool
  return { kind: `a ${k.replace(/_/g, " ")}`, keys: SELECTION_HINTS.any.keys, draw: SELECTION_HINTS.any.draw };
}

/** "Crop needs an area selection – draw a rectangle (1), oval (2) … first" */
export function needsSelectionMessage(op: Pick<OpSpec, "id" | "name" | "requires_roi">): string {
  const h = selectionHint(op.requires_roi);
  return `${displayName(op)} needs ${h.kind} selection – ${h.draw}`;
}

/** The short palette / menu form: "needs a line selection (5, 6)" (E7-18). */
export function needsSelectionShort(op: Pick<OpSpec, "requires_roi">): string {
  const h = selectionHint(op.requires_roi);
  return `needs ${h.kind} selection (${h.keys})`;
}

/** Bytes a File ▸ New image would take (for the size warning, E5-18). */
export function newImageBytes(p: Record<string, any>): number {
  const bpp = p.type === "16-bit" ? 2 : p.type === "32-bit" ? 4 : p.type === "RGB" ? 3 : 1;
  const n = (v: any, d = 1) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Number(v) : d);
  return n(p.width, 512) * n(p.height, 512) * n(p.channels) * n(p.slices) * n(p.frames) * bpp;
}

export function fmtMB(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1048576)} MB`;
}

/** A long-running operation reported by the server (progress events on /api/events). */
export interface Job extends JobInfo {
  /** performance.now() when this client first saw the job (elapsed time is measured from here). */
  seenAt: number;
  /** Cancel was requested from this client; the server confirms with a `done` + `cancelled` event. */
  cancelling?: boolean;
}

export type ConfirmAnswer = "confirm" | "alt" | "cancel";

/** A question for the user (components/ConfirmDialog.tsx): closing a modified image, Close All … */
export interface ConfirmRequest {
  title: string;
  message: string;
  /** Label of the confirming button (the destructive one when `danger`). */
  confirmLabel: string;
  /** Optional third button between Cancel and the confirming one (e.g. "Save…"); answers "alt". */
  altLabel?: string;
  /** Preference switched off when "Don't ask again" is ticked and the dialog is not cancelled. */
  dontAskPref?: "confirm_close_modified";
  danger?: boolean;
  resolve: (answer: ConfirmAnswer, dontAsk: boolean) => void;
}

export interface Sheet {
  op: OpSpec;
  params: Record<string, any>;
  preview: boolean;
  scope: string;
  error?: string | null;
  busy?: boolean;
  /** A note under the fields ("12 recorded lines, 3 skipped …" for Apply Recorded Macro). */
  note?: string | null;
}

export type PanelKey = "display" | "threshold" | "info" | "rois" | "results" | "plot" | "histogram" | "history" | "script" | "log" | "table" | "volume" | "ortho" | "tracks" | "workflows" | "classifier" | "frames";

/** Track overlay colouring: each track's own colour, a statistic through a colormap, or a distinct hue per id. */
export type TrackColorBy = "track" | "speed" | "duration" | "displacement" | "id";
/** Which track overlays carry a name label ("all" still hides them beyond 60 tracks). */
export type TrackLabelMode = "none" | "selected" | "all";

export interface ToolOptions {
  erase: boolean;
  allChannels: boolean;
  bucketTolerance: number;
  bucketConnectivity: "4" | "8";
  fontSize: number;
  textColor: string;
  textBackground: boolean;
  arrowWidth: number;
  arrowHead: number;
  arrowColor: string;
  lineWidth: number;
  /** Rectangle tool: corner radius in image px; > 0 commits a rounded-rect polygon. */
  rectRadius: number;
  /** Point tool: show the running index next to each point. */
  pointCounter: boolean;
}

interface Store {
  meta: Meta | null;
  ops: Record<string, OpSpec>;
  luts: LutTables;
  state: SessionState | null;
  ready: boolean;
  fatal: string | null;
  tool: Tool;
  spaceDown: boolean;
  selections: Record<string, Roi | null>;
  views: Record<string, ViewTransform>;
  paletteOpen: boolean;
  /** The Agent panel (top bar ▸ Agent, ⌘J): Agent Chat docked on the right. */
  agentOpen: boolean;
  setAgentOpen: (open: boolean) => void;
  sheet: Sheet | null;
  previewPlanes: PlaneData | null;
  previewFor: string | null;
  toasts: Toast[];
  /** LUT grouping for the menus / selects (server table, else the client's copy) and the registered names arranged by it. */
  lutGrouping: LutGrouping;
  lutGroups: LutGroup[];
  /** Inspector ▸ ROI Manager / Results: list only the entries of the active image (default on, E5-11 / E6-3). */
  roisThisImageOnly: boolean;
  resultsThisImageOnly: boolean;
  setRoisThisImageOnly: (v: boolean) => void;
  setResultsThisImageOnly: (v: boolean) => void;
  /** Clear the ROI Manager / Results after a confirmation that names the count; the toast offers "Undo clear" (E7-6). */
  clearRoiManager: () => Promise<void>;
  clearResultsTable: () => Promise<void>;
  /** Time series whose Tracks section was opened automatically once (E6-12). */
  tracksAutoOpened: Record<string, true>;
  markTracksAutoOpened: (id: string) => void;
  plot: PlotData | null;
  histogram: HistogramData | null;
  table: TableData | null;
  panels: Record<PanelKey, boolean>;
  theme: "dark" | "light";
  /**
   * The file dialog: "open" / "save" work on images; "pick-open" / "pick-save" / "pick-folder" hand the chosen
   * server path to `onPick` (text files for the script editor and text windows, path / folder parameters of
   * the operation sheet's Browse… button).  `initialPath`: folder (or file whose folder) to start in.
   */
  /** `inApp`: show Lumen's own server file browser even inside the desktop app (which otherwise uses the system dialog). */
  fileDialog: { mode: "open" | "save" | "pick-open" | "pick-save" | "pick-folder"; imageId?: string; format?: string; title?: string; filename?: string; extensions?: string[]; initialPath?: string; onPick?: (path: string) => void; inApp?: boolean } | null;
  /** Pending question (ConfirmDialog); null = none.  Use `askConfirm`. */
  confirm: ConfirmRequest | null;
  askConfirm: (req: Omit<ConfirmRequest, "resolve">) => Promise<ConfirmAnswer>;
  /** Revision (`ImageSummary.rev`) of each image when it was last saved from this client (Save / Save As). */
  savedRevs: Record<string, number>;
  markSaved: (id: string) => void;
  /** Unsaved changes: a non-empty undo history and no save at the current revision (the tab's dot, the close confirmation). */
  isModified: (id: string) => boolean;
  busy: number;
  /** Running long operations by job id (ProgressBar in the top bar). */
  jobs: Record<string, Job>;
  cancelJob: (id: string) => Promise<void>;
  cursor: { x: number; y: number; values: (number | null)[] | null } | null;
  recent: string[];
  selectedRoiIds: string[];
  showManagerRois: boolean;
  helpOpen: boolean;
  wandTolerance: number;
  menuOpen: string | null;
  inspectorOpen: boolean;

  // drawing / annotation tools (Viewport, ToolRail, shortcuts)
  brushSize: number;
  toolOptions: ToolOptions;
  syncViews: boolean;
  showOverlay: boolean;
  showTrackLabels: boolean;
  showTrackFuture: boolean;
  setBrushSize: (v: number) => void;
  setToolOptions: (patch: Partial<ToolOptions>) => void;
  setSyncViews: (v: boolean) => void;
  setShowOverlay: (v: boolean) => void;
  setShowTrackLabels: (v: boolean) => void;
  setShowTrackFuture: (v: boolean) => void;
  setViews: (patch: Record<string, ViewTransform>) => void;

  // preferences (Edit ▸ Options ▸ Preferences…), stored in session.settings on the server
  prefs: Prefs;
  /** Every visible server setting as returned by GET /api/settings (keys beyond `Prefs`: overlay options, colour picker …). */
  settings: Record<string, unknown>;
  prefsLoaded: boolean;
  // Image ▸ Color ▸ Color Picker… popover (foreground / background value or colour)
  colorPickerOpen: boolean;
  setColorPickerOpen: (v: boolean) => void;
  /** Tool-options values of plugin tools (the op's own Params), by op id. */
  toolParams: Record<string, Record<string, any>>;
  setToolParams: (opId: string, patch: Record<string, any>) => void;
  prefsOpen: boolean;
  setPrefsOpen: (v: boolean) => void;
  loadPrefs: () => Promise<void>;
  setPrefs: (patch: Partial<Prefs>) => Promise<void>;
  // Edit ▸ Selection ▸ Restore Selection: last non-null selection per image
  lastSelections: Record<string, Roi>;
  restoreSelection: () => void;
  // in-app documentation (Help menu, "?" in the operation sheet); `compact` = opened from a sheet: no Design / Research tabs (E7-20)
  docsOpen: { name: DocName; anchor?: string; compact?: boolean } | null;
  setDocsOpen: (d: { name: DocName; anchor?: string; compact?: boolean } | null) => void;
  // Plugins ▸ Shortcuts
  shortcutsOpen: boolean;
  setShortcutsOpen: (v: boolean) => void;
  customShortcuts: CustomShortcut[];
  setCustomShortcuts: (list: CustomShortcut[]) => void;
  // Plugins ▸ Utilities ▸ Monitor Memory
  memoryMonitor: boolean;
  setMemoryMonitor: (v: boolean) => void;
  memorySamples: { t: number; bytes: number }[];
  pushMemorySample: (bytes: number) => void;
  // manual tracking tool (X)
  manualTrack: { id: number | null; autoAdvance: boolean };
  setManualTrack: (patch: Partial<{ id: number | null; autoAdvance: boolean }>) => void;
  // track overlays (Viewport selection / hover, Inspector ▸ Tracks)
  /** Overlay item ids ("track-3", "manual-1") of the selected tracks; the others are drawn dimmed. */
  selectedTrackIds: string[];
  setSelectedTrackIds: (ids: string[]) => void;
  trackColorBy: TrackColorBy;
  setTrackColorBy: (mode: TrackColorBy) => void;
  trackLabelMode: TrackLabelMode;
  setTrackLabelMode: (mode: TrackLabelMode) => void;
  /** Draw only the selected tracks while a selection exists. */
  trackShowOnlySelected: boolean;
  setTrackShowOnlySelected: (v: boolean) => void;

  // Workflows panel (guided step-by-step workflows): progress of the running one, for the section badge
  workflowProgress: { title: string; done: number; total: number } | null;
  setWorkflowProgress: (p: { title: string; done: number; total: number } | null) => void;

  init: () => Promise<void>;
  refresh: () => Promise<void>;
  activeImage: () => ImageSummary | undefined;
  image: (id: string | null | undefined) => ImageSummary | undefined;
  /** `onError` replaces the error toast (the operation sheet shows the message inline instead). */
  runOp: (opId: string, params?: Record<string, any>, opts?: { scope?: string; roi?: Roi | null; imageId?: string | null; silent?: boolean; onError?: (message: string) => void }) => Promise<OpRunResult | undefined>;
  invoke: (opId: string, quick?: boolean) => void;
  /** The last completed command, for Repeat (⇧R) and the workflow recorder.  `seq` increases with every run so
   *  a consumer can tell a fresh record from one it already used. */
  lastRun: { opId: string; params: Record<string, any>; scope: string; seq: number } | null;
  repeatLast: () => void;
  openSheet: (op: OpSpec, params?: Record<string, any>) => void;
  closeSheet: () => void;
  setSheet: (patch: Partial<Sheet>) => void;
  applySheet: () => Promise<void>;
  setDisplay: (imageId: string, patch: Record<string, any>, immediate?: boolean) => void;
  flushDisplay: () => Promise<void>;
  setPosition: (imageId: string, pos: { t?: number; z?: number; c?: number }) => void;
  setSelection: (imageId: string, roi: Roi | null) => void;
  selectImage: (id: string) => void;
  /** Close one image; asks first when it has unsaved changes (preference `confirm_close_modified`) unless `force`. */
  closeImage: (id: string, opts?: { force?: boolean }) => Promise<void>;
  /** Close several images with one question (Close Others / Close to the Right / Close All). */
  closeImages: (ids: string[], opts?: { force?: boolean }) => Promise<void>;
  closeAllImages: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  toast: (msg: string, kind?: Toast["kind"], opts?: { actions?: ToastAction[]; timeout?: number }) => void;
  dismissToast: (id: number) => void;
  setTool: (tool: Tool) => void;
  setSpace: (down: boolean) => void;
  setView: (imageId: string, v: ViewTransform) => void;
  togglePanel: (key: PanelKey, open?: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  setTheme: (t: "dark" | "light") => void;
  setFileDialog: (d: Store["fileDialog"]) => void;
  setCursor: (c: Store["cursor"]) => void;
  setPlot: (p: PlotData | null) => void;
  setHistogram: (h: HistogramData | null) => void;
  setSelectedRoiIds: (ids: string[]) => void;
  setShowManagerRois: (v: boolean) => void;
  setHelpOpen: (v: boolean) => void;
  setWandTolerance: (v: number) => void;
  setMenuOpen: (m: string | null) => void;
  setInspectorOpen: (v: boolean) => void;
  openFiles: (files: FileList | File[]) => Promise<void>;
  setPreview: (planes: PlaneData | null, imageId: string | null) => void;

  // ---- layout UX (App / ImageTabs / SplitView / ResultsModal) ----
  /** Inspector column width in px (drag splitter; persisted in localStorage "lumen.inspectorWidth"). */
  inspectorWidth: number;
  setInspectorWidth: (w: number) => void;
  /** Which table is shown maximised in the ResultsModal (null = closed). */
  tableModal: "results" | "table" | null;
  setTableModal: (m: "results" | "table" | null) => void;
  /** Client-side tab order (image ids); images missing from it follow in server order. */
  imageOrder: string[];
  setImageOrder: (ids: string[]) => void;
  /** Side-by-side view: the active image in pane `activeSide`, `other` in the opposite pane. */
  split: { on: boolean; other: string | null; activeSide: 0 | 1 };
  setSplit: (patch: Partial<{ on: boolean; other: string | null; activeSide: 0 | 1 }>) => void;
  toggleSplitView: () => void;
}

let toastId = 0;
let pendingDisplay: Record<string, Record<string, any>> = {};
let displayTimer: number | null = null;
let refreshTimer: number | null = null;
// state.lut_version of the tables held in `luts`; undefined = unknown (older server without the field)
let lutVersion: number | undefined;
/** Memory monitor sampling interval (ms); the sparkline draws a longer silence as a gap. */
export const MEMORY_SAMPLE_MS = 2000;
/** Inspector width defaults – used by the store's initial state (see "layout persistence" below). */
export const INSPECTOR_DEFAULT_W = 332;
export const INSPECTOR_MIN_W = 260;
// Display edits are applied locally at once and POSTed later (debounced); the server never
// broadcasts display changes back.  So that a refresh() cannot snap the viewport back to the
// server's stale t/z/c, every edit is logged with a sequence number and replayed onto any
// fetched state whose GET was issued before the edit, and in-flight POSTs are chained so that an
// awaited flushDisplay() means "the server holds everything the viewport shows".
let displayInflight: Promise<void> = Promise.resolve();
let displaySeq = 0; // bumps on every local display edit
let displayLog: { seq: number; id: string; patch: Record<string, any> }[] = []; // edits not yet confirmed by a later GET /state
let refreshSeq = 0;
let refreshApplied = 0;
let runSeq = 0; // sequence number of the last completed runOp (store.lastRun.seq)

/**
 * Starting value of an Edit ▸ Options sheet field: the live setting when present (for a choice
 * param only when it is one of the listed choices), else the param default, else a typed blank.
 * Those ops declare no defaults and the server keeps every key that is not sent, so the field must
 * show what is currently set (a boolean shows its current value, not "unchecked").
 */
function seedPrefParam(p: ParamSpec, v: unknown): any {
  if (p.type === "choice") {
    const choices = p.choices || [];
    if (typeof v === "string" && choices.includes(v)) return v;
    return typeof p.default === "string" && choices.includes(p.default) ? p.default : choices[0] ?? "";
  }
  if (p.type === "bool") return v === undefined || v === null ? !!p.default : !!v;
  if (p.type === "int" || p.type === "float") {
    const n = typeof v === "number" ? v : typeof v === "string" ? parseFloat(v) : NaN;
    return Number.isFinite(n) ? n : typeof p.default === "number" ? p.default : 0;
  }
  if (v !== undefined && v !== null) return typeof v === "string" ? v : String(v);
  return p.default ?? "";
}

function applyDisplayPatch(im: ImageSummary, patch: Record<string, any>): ImageSummary {
  const d = { ...im.display };
  if (patch.t !== undefined) d.t = patch.t;
  if (patch.z !== undefined) d.z = patch.z;
  if (patch.c !== undefined) d.c = patch.c;
  if (patch.mode !== undefined) d.mode = patch.mode;
  if (patch.channels) d.channels = d.channels.map((ch, i) => (patch.channels[i] ? { ...ch, ...patch.channels[i] } : ch));
  return { ...im, display: d };
}

/** Which revision of each image was last saved: kept per tab in sessionStorage so a page reload does not mark saved
 * images as modified again (E12-2); image ids and revisions belong to the server session, so they stay valid. */
const SAVED_REVS_KEY = "lumen-saved-revs";
function loadSavedRevs(): Record<string, number> {
  try {
    const raw = JSON.parse(sessionStorage.getItem(SAVED_REVS_KEY) || "{}");
    return raw && typeof raw === "object" ? (raw as Record<string, number>) : {};
  } catch {
    return {};
  }
}
function persistSavedRevs(revs: Record<string, number>): void {
  try {
    sessionStorage.setItem(SAVED_REVS_KEY, JSON.stringify(revs));
  } catch {
    /* private mode / storage blocked: the marks last until the reload */
  }
}

export const useStore = create<Store>((set, get) => ({
  meta: null,
  ops: {},
  luts: {},
  state: null,
  ready: false,
  fatal: null,
  tool: "pointer",
  spaceDown: false,
  selections: {},
  views: {},
  paletteOpen: false,
  agentOpen: false,
  sheet: null,
  previewPlanes: null,
  previewFor: null,
  toasts: [],
  lutGrouping: DEFAULT_GROUPS,
  lutGroups: [],
  roisThisImageOnly: localStorage.getItem("lumen.roisThisImageOnly") !== "0",
  resultsThisImageOnly: localStorage.getItem("lumen.resultsThisImageOnly") !== "0",
  setRoisThisImageOnly: (v) => {
    localStorage.setItem("lumen.roisThisImageOnly", v ? "1" : "0");
    set({ roisThisImageOnly: v });
  },
  setResultsThisImageOnly: (v) => {
    localStorage.setItem("lumen.resultsThisImageOnly", v ? "1" : "0");
    set({ resultsThisImageOnly: v });
  },
  clearRoiManager: async () => {
    const st = get();
    const n = st.state?.rois.length ?? 0;
    if (!n) return;
    const answer = await st.askConfirm({ title: "Clear the ROI Manager?", message: `This deletes all ${n} ROI${n === 1 ? "" : "s"} of every image from the manager (measurements already in Results are kept). You can put them back with “Undo clear” right afterwards.`, confirmLabel: `Delete ${n} ROI${n === 1 ? "" : "s"}`, danger: true });
    if (answer !== "confirm") return;
    try {
      const r = await api.clearRois();
      await get().refresh();
      get().setSelectedRoiIds([]);
      const restorable = r?.restorable ?? !!get().ops["rois.restore_cleared"];
      get().toast(`ROI Manager cleared (${r?.cleared ?? n} ROI${(r?.cleared ?? n) === 1 ? "" : "s"})`, "info", restorable ? { actions: [{ label: "Undo clear", run: () => void restoreClear("rois") }] } : undefined);
    } catch (e: any) {
      get().toast(`Could not clear the ROI Manager: ${e?.message || e}`, "error");
    }
  },
  clearResultsTable: async () => {
    const st = get();
    const n = st.state?.results.rows.length ?? 0;
    if (!n) return;
    const answer = await st.askConfirm({ title: "Clear the Results table?", message: `This deletes all ${n} row${n === 1 ? "" : "s"} (every image's measurements). Export them first with CSV if you need them; “Undo clear” brings them back right afterwards.`, confirmLabel: `Delete ${n} row${n === 1 ? "" : "s"}`, danger: true });
    if (answer !== "confirm") return;
    try {
      const r = await api.clearResults();
      await get().refresh();
      const restorable = r?.restorable ?? !!get().ops["results.restore_cleared"];
      get().toast(`Results cleared (${r?.cleared ?? n} row${(r?.cleared ?? n) === 1 ? "" : "s"})`, "info", restorable ? { actions: [{ label: "Undo clear", run: () => void restoreClear("results") }] } : undefined);
    } catch (e: any) {
      get().toast(`Could not clear the Results table: ${e?.message || e}`, "error");
    }
  },
  tracksAutoOpened: {},
  markTracksAutoOpened: (id) => set({ tracksAutoOpened: { ...get().tracksAutoOpened, [id]: true } }),
  plot: null,
  histogram: null,
  table: null,
  panels: { display: true, threshold: false, info: false, rois: true, results: true, plot: false, histogram: false, history: false, script: false, log: false, table: false, volume: false, ortho: false, tracks: true, workflows: false, classifier: false, frames: false },
  theme: (localStorage.getItem("lumen.theme") as "dark" | "light") || "dark",
  fileDialog: null,
  confirm: null,
  askConfirm: (req) =>
    new Promise<ConfirmAnswer>((resolve) => {
      get().confirm?.resolve("cancel", false); // one question at a time: a newer one cancels the pending one
      const resolveOnce = (answer: ConfirmAnswer, dontAsk: boolean) => {
        set({ confirm: null });
        if (dontAsk && answer !== "cancel" && req.dontAskPref === "confirm_close_modified") void get().setPrefs({ confirm_close_modified: false });
        resolve(answer);
      };
      set({ confirm: { ...req, resolve: resolveOnce }, menuOpen: null });
    }),
  savedRevs: loadSavedRevs(),
  markSaved: (id) => {
    const im = get().image(id);
    if (im) {
      const revs = { ...get().savedRevs, [id]: im.rev };
      set({ savedRevs: revs });
      persistSavedRevs(revs);
    }
  },
  isModified: (id) => {
    const st = get();
    const im = st.image(id);
    if (!im) return false;
    const undo = st.state?.history[id]?.undo.length ?? 0;
    return undo > 0 && st.savedRevs[id] !== im.rev;
  },
  busy: 0,
  jobs: {},
  cancelJob: async (id) => {
    const job = get().jobs[id];
    if (!job || job.cancelling) return;
    set({ jobs: { ...get().jobs, [id]: { ...job, cancelling: true } } });
    try {
      await api.cancelJob(id);
    } catch (e: any) {
      // the job may already have finished; the `done` event removes it either way
      const cur = get().jobs[id];
      if (cur) set({ jobs: { ...get().jobs, [id]: { ...cur, cancelling: false } } });
      get().toast(`Could not cancel ${job.name}: ${e.message || e}`, "error");
    }
  },
  cursor: null,
  recent: JSON.parse(localStorage.getItem("lumen.recent") || "[]"),
  selectedRoiIds: [],
  showManagerRois: false,
  helpOpen: false,
  wandTolerance: 0,
  menuOpen: null,
  inspectorOpen: true,
  lastRun: null,

  brushSize: 10,
  toolOptions: { erase: false, allChannels: false, bucketTolerance: 0, bucketConnectivity: "4", fontSize: 14, textColor: "#ffffff", textBackground: false, arrowWidth: 2, arrowHead: 12, arrowColor: "#ffd54a", lineWidth: 1, rectRadius: 0, pointCounter: false },
  syncViews: false,
  showOverlay: true,
  showTrackLabels: true,
  showTrackFuture: true,
  setBrushSize: (v) => set({ brushSize: Math.max(1, Math.min(500, Math.round(v))) }),
  setToolOptions: (patch) => set({ toolOptions: { ...get().toolOptions, ...patch } }),
  setSyncViews: (v) => set({ syncViews: v }),
  setShowOverlay: (v) => set({ showOverlay: v }),
  setShowTrackLabels: (v) => set({ showTrackLabels: v }),
  setShowTrackFuture: (v) => set({ showTrackFuture: v }),
  setViews: (patch) => set({ views: { ...get().views, ...patch } }),

  prefs: { ...PREF_DEFAULTS },
  settings: {},
  prefsLoaded: false,
  prefsOpen: false,
  colorPickerOpen: false,
  setColorPickerOpen: (v) => set({ colorPickerOpen: v, menuOpen: null }),
  toolParams: {},
  setToolParams: (opId, patch) => set({ toolParams: { ...get().toolParams, [opId]: { ...(get().toolParams[opId] || {}), ...patch } } }),
  setPrefsOpen: (v) => set({ prefsOpen: v, menuOpen: null, paletteOpen: false }),
  loadPrefs: async () => {
    try {
      const raw = await api.getSettings();
      const prefs = normalizePrefs(raw);
      const o = get().toolOptions;
      // the annotation tools start from the preferences (the tool-options bar can still override per tool)
      set({ prefs, settings: raw || {}, prefsLoaded: true, toolOptions: { ...o, fontSize: prefs.font_size, arrowWidth: prefs.arrow_width, arrowHead: prefs.arrow_head_size } });
      applyDensity(prefs.ui_density);
    } catch (e: any) {
      // an older server without the settings router: keep the defaults, stay usable
      set({ prefsLoaded: true });
      console.warn("preferences unavailable", e?.message || e);
    }
  },
  setPrefs: async (patch) => {
    const before = get().prefs;
    const next = { ...before, ...patch };
    const o = get().toolOptions;
    const to = { ...o };
    if (patch.font_size !== undefined) to.fontSize = patch.font_size;
    if (patch.arrow_width !== undefined) to.arrowWidth = patch.arrow_width;
    if (patch.arrow_head_size !== undefined) to.arrowHead = patch.arrow_head_size;
    set({ prefs: next, toolOptions: to, settings: { ...get().settings, ...patch } }); // optimistic
    if (patch.ui_density !== undefined) applyDensity(patch.ui_density);
    try {
      const raw = await api.putSettings(patch);
      const saved = normalizePrefs(raw);
      set({ prefs: saved, settings: raw || {} });
      applyDensity(saved.ui_density);
    } catch (e: any) {
      set({ prefs: before });
      applyDensity(before.ui_density);
      get().toast(`Could not save preference: ${e.message || e}`, "error");
    }
  },
  lastSelections: {},
  restoreSelection: () => {
    const st = get();
    const img = st.activeImage();
    if (!img) return st.toast("Open an image first", "error");
    const roi = st.lastSelections[img.id];
    if (!roi) return st.toast("No previous selection to restore", "info");
    st.setSelection(img.id, roi);
  },
  docsOpen: null,
  setDocsOpen: (d) => set({ docsOpen: d, menuOpen: null, paletteOpen: false }),
  shortcutsOpen: false,
  setShortcutsOpen: (v) => set({ shortcutsOpen: v, menuOpen: null, paletteOpen: false }),
  customShortcuts: loadCustomShortcuts(),
  setCustomShortcuts: (list) => {
    saveCustomShortcuts(list);
    set({ customShortcuts: list });
  },
  memoryMonitor: false,
  setMemoryMonitor: (v) => {
    set({ memoryMonitor: v, memorySamples: v ? get().memorySamples : [] });
    // sampling follows this flag, not the Log panel's visibility (startMemorySampling below)
    if (v) startMemorySampling();
    else stopMemorySampling();
  },
  memorySamples: [],
  pushMemorySample: (bytes) => set({ memorySamples: [...get().memorySamples, { t: Date.now(), bytes }].slice(-90) }),
  manualTrack: { id: null, autoAdvance: true },
  setManualTrack: (patch) => set({ manualTrack: { ...get().manualTrack, ...patch } }),
  selectedTrackIds: [],
  setSelectedTrackIds: (ids) => set({ selectedTrackIds: ids }),
  trackColorBy: "track",
  setTrackColorBy: (mode) => set({ trackColorBy: mode }),
  trackLabelMode: "selected",
  setTrackLabelMode: (mode) => set({ trackLabelMode: mode }),
  trackShowOnlySelected: false,
  setTrackShowOnlySelected: (v) => set({ trackShowOnlySelected: v }),

  workflowProgress: null,
  setWorkflowProgress: (p) => set({ workflowProgress: p }),

  init: async () => {
    try {
      const [meta, luts, state] = await Promise.all([api.meta(), api.luts(), api.state()]);
      const ops: Record<string, OpSpec> = {};
      for (const o of meta.ops) ops[o.id] = o;
      lutVersion = state.lut_version;
      set({ meta, ops, luts, state, ready: true, plot: state.last_plot, lutGroups: groupLuts(meta.luts, DEFAULT_GROUPS) });
      void get().loadPrefs();
      // the server's LUT grouping (Image ▸ Lookup Tables, the Display select); the client's copy until it answers
      void fetchLutGroups().then((g) => set({ lutGrouping: g, lutGroups: groupLuts(get().meta?.luts || meta.luts, g) }));
      connectEvents(() => scheduleRefresh());
      void import("./agentBridge").then((m) => m.startAgentBridge()); // the AI-assistant bridge reports what the window shows
      // the desktop shell's hooks (native Help menu, quit confirmation while images are unsaved); inert in a browser
      startNativeHooks({
        openDocs: () => get().setDocsOpen({ name: "README" }),
        openShortcuts: () => get().setHelpOpen(true),
        unsavedCount: () => (get().state?.images || []).filter((im) => get().isModified(im.id)).length,
        subscribe: (fn) => useStore.subscribe(fn),
      });
    } catch (e: any) {
      set({ fatal: e.message || String(e) });
    }
  },

  refresh: async () => {
    try {
      await get().flushDisplay(); // the server now holds every display edit made before this point
      const seq0 = displaySeq;
      const my = ++refreshSeq;
      const state = await api.state();
      if (my < refreshApplied) return; // a newer refresh already landed
      // LUTs registered, replaced or removed at runtime (Import LUT, Edit LUT, 8-bit Color) → refetch the
      // tables.  `lut_version` bumps on every change, including a table replaced under an existing name
      // (which leaves the name count unchanged); the count check stays as a fallback for servers that
      // do not send it (lut_version undefined = unknown).
      const names = (state as any).lut_names as string[] | undefined;
      const version = state.lut_version;
      const versionChanged = version !== undefined && version !== lutVersion;
      const countChanged = names !== undefined && names.length !== Object.keys(get().luts).length;
      if (versionChanged || countChanged) {
        try {
          const luts = await api.luts();
          const meta = get().meta;
          const lutNames = names ?? meta?.luts ?? [];
          set({ luts, meta: meta ? { ...meta, luts: lutNames } : meta, lutGroups: groupLuts(lutNames, get().lutGrouping) });
          if (version !== undefined) lutVersion = version; // recorded only after a successful fetch, so a failure retries
        } catch {
          /* keep the old tables */
        }
      }
      refreshApplied = my;
      // edits made while the GET was in flight (pending, in flight, or POSTed but processed after
      // the GET): the client is authoritative for those, so replay them onto the fetched copy
      const later = displayLog.filter((e) => e.seq > seq0);
      if (later.length) state.images = state.images.map((im) => later.reduce((acc, e) => (e.id === im.id ? applyDisplayPatch(acc, e.patch) : acc), im));
      displayLog = later;
      const cur = get().state;
      const patch: Partial<Store> = { state };
      if (state.last_plot && state.last_plot !== cur?.last_plot && JSON.stringify(state.last_plot) !== JSON.stringify(cur?.last_plot)) patch.plot = state.last_plot;
      // keep selections only for open images
      const sel: Record<string, Roi | null> = {};
      const last: Record<string, Roi> = {};
      for (const im of state.images) {
        if (get().selections[im.id] !== undefined) sel[im.id] = get().selections[im.id];
        if (get().lastSelections[im.id]) last[im.id] = get().lastSelections[im.id];
      }
      patch.selections = sel;
      patch.lastSelections = last;
      const revs: Record<string, number> = {};
      for (const im of state.images) if (get().savedRevs[im.id] !== undefined) revs[im.id] = get().savedRevs[im.id];
      patch.savedRevs = revs;
      persistSavedRevs(revs);
      set(patch);
    } catch (e: any) {
      get().toast(`Lost connection: ${e.message}`, "error");
    }
  },

  activeImage: () => {
    const s = get().state;
    if (!s || !s.active) return undefined;
    return s.images.find((im) => im.id === s.active);
  },
  image: (id) => get().state?.images.find((im) => im.id === id),

  runOp: async (opId, params = {}, opts = {}) => {
    const st = get();
    const spec = st.ops[opId];
    const img = opts.imageId !== undefined ? st.image(opts.imageId) : st.activeImage();
    const roi = opts.roi !== undefined ? opts.roi : img ? st.selections[img.id] ?? null : null;
    await st.flushDisplay();
    set({ busy: st.busy + 1 });
    try {
      const res = await api.run(opId, params, img?.id ?? null, roi, opts.scope || "stack");
      // Record the run BEFORE the refresh that renders its macro line: the workflow recorder reads
      // lastRun in an effect of that render, so writing it afterwards handed it the previous run.
      if (spec) set({ lastRun: { opId, params, scope: opts.scope || "stack", seq: ++runSeq } });
      await get().refresh();
      // Edit ▸ Options ops (and Overlay Options…, Colour Settings …) write the settings: reload them so the
      // Preferences sheet, the colour picker and the viewport's overlay options follow
      if (spec?.menu === "Edit/Options" || spec?.category === "Edit/Options" || /options|set_colors|set_foreground|set_background/.test(opId)) await get().loadPrefs();
      const after = get();
      showRunResult(res, img?.id ?? null);
      if (res.message && !opts.silent) after.toast(res.message, "info");
      if (spec) {
        const recent = [opId, ...after.recent.filter((r) => r !== opId)].slice(0, 12);
        localStorage.setItem("lumen.recent", JSON.stringify(recent));
        set({ recent });
      }
      return res;
    } catch (e: any) {
      // a cancelled op answers 409 "Cancelled"; the job's `done` event already toasts "Cancelled <op>"
      const message: string = e?.message || String(e);
      if (message !== "Cancelled") {
        if (opts.onError) opts.onError(message);
        else if (opId === "analyze.particles" && /threshold/i.test(message) && !params.auto_threshold) {
          // the fix is one click away instead of hidden under "Advanced options" (E5-5 / E6-2)
          get().toast(message, "error", { actions: [{ label: "Auto-threshold and run", run: () => void get().runOp(opId, { ...params, auto_threshold: true }, { ...opts, onError: undefined }) }] });
        } else get().toast(message, "error");
      }
      return undefined;
    } finally {
      set({ busy: Math.max(0, get().busy - 1) });
    }
  },

  invoke: (opId, quick) => {
    const st = get();
    const op = st.ops[opId];
    if (!op) return;
    const img = st.activeImage();
    if (op.tool) {
      // a plugin tool is selected on the tool rail, not run: the viewport runs it on every click / drag
      set({ paletteOpen: false, menuOpen: null });
      st.setTool(`op:${op.id}`);
      st.toast(`${op.name}: ${op.requires_image && !img ? "open an image, then " : ""}click or drag on the image`, "info");
      return;
    }
    if (op.id === "file.close" || op.id === "file.close_all") {
      // closing goes through the client so a modified image can ask first (docs/USABILITY.md M1)
      set({ paletteOpen: false, menuOpen: null });
      if (op.id === "file.close_all") void st.closeAllImages();
      else if (img) void st.closeImage(img.id);
      else st.toast("No image to close", "info");
      return;
    }
    if (op.requires_image && !img) {
      st.toast("Open an image first", "error");
      return;
    }
    if (op.requires_roi && img && !st.selections[img.id]) {
      st.toast(needsSelectionMessage(op), "error");
      return;
    }
    if (quick) {
      // shortcut: run immediately with defaults on the current plane (ImageJ behaviour)
      const defaults: Record<string, any> = {};
      for (const p of op.params) defaults[p.name] = p.type === "image" ? null : p.default;
      void st.runOp(opId, defaults, { scope: op.kind === "plane" ? "plane" : "stack" });
      return;
    }
    const needsSheet = op.params.length > 0 || (op.scope_param && img && (img.shape.z > 1 || img.shape.t > 1 || img.shape.c > 1));
    if (needsSheet) st.openSheet(op);
    else void st.runOp(opId, {});
  },

  repeatLast: () => {
    const st = get();
    const last = st.lastRun;
    if (!last) return st.toast("Nothing to repeat yet", "info");
    const op = st.ops[last.opId];
    if (op && (op.id === "file.sample" || op.id === "file.new" || op.id === "file.close" || op.id === "file.close_all")) return st.toast("Repeat skips file commands", "info");
    void st.runOp(last.opId, last.params, { scope: last.scope });
  },

  openSheet: (op, params) => {
    const st = get();
    const defaults: Record<string, any> = {};
    const images = st.state?.images || [];
    const active = st.activeImage();
    for (const p of op.params) {
      defaults[p.name] = p.default;
      if (p.type === "image" && p.default == null) {
        // sensible defaults: image1 = active, image2 = other
        if (/1$|^c1$|^image$|^intensity/.test(p.name)) defaults[p.name] = active?.id ?? images[0]?.id ?? null;
        else if (/2$|^c2$/.test(p.name)) defaults[p.name] = images.find((i) => i.id !== active?.id)?.id ?? null;
        else if (/^c3$/.test(p.name)) defaults[p.name] = images.filter((i) => i.id !== active?.id)[1]?.id ?? null;
        if (p.name === "intensity_image") defaults[p.name] = null;
      }
    }
    if (op.id === "image.canvas_size" && active) {
      defaults.width = active.shape.x;
      defaults.height = active.shape.y;
    }
    if (op.id === "image.properties" && active) {
      const c = active.calibration;
      Object.assign(defaults, { pixel_width: c.pixel_width, pixel_height: c.pixel_height, voxel_depth: c.pixel_depth, unit: c.unit, frame_interval: c.frame_interval, time_unit: c.time_unit });
    }
    if (op.id === "analyze.set_measurements" && st.state) {
      defaults.measurements = st.state.measurements;
      defaults.limit_to_threshold = !!st.state.measure_options.limit_to_threshold;
      defaults.decimals = st.state.measure_options.decimals ?? 3;
    }
    if (op.id === "image.rename" && active) defaults.title = active.name;
    if (op.id === "stacks.substack" && active) defaults.slices = `1-${active.shape.z}`;
    // Z Project: the real last slice instead of "0 = last" (E7-25); the T axis is picked under Advanced options
    if (op.id === "stacks.z_project" && active && "stop" in defaults && !defaults.stop) defaults.stop = active.shape.z > 1 ? active.shape.z : active.shape.t;
    if (op.id === "color.channel_names" && active) defaults.names = active.display.channels.map((c) => c.name).join(", ");
    if (op.id === "threshold.set" && active) {
      const thr = st.state?.thresholds[active.id];
      if (thr) {
        defaults.lower = thr.lower;
        defaults.upper = thr.upper;
      }
    }
    const isStack = active ? active.shape.z > 1 || active.shape.t > 1 : false;
    // Stacks: a plane filter defaults to the current plane like ImageJ (which asks "Process all N images?" and
    // defaults to No, E5-9); image-kind ops keep the whole stack
    const scope = isStack ? (op.kind === "plane" ? "plane" : "stack") : "slice";
    const open = (seeded: Record<string, any>, note: string | null = null) =>
      set({ sheet: { op, params: { ...seeded, ...(params || {}) }, preview: op.preview && op.kind === "plane", scope, error: null, note }, paletteOpen: false, menuOpen: null });
    if (op.id === "batch.apply_recorded_macro" && op.params.some((p) => p.name === "lines")) {
      // the exact lines the batch will run, editable before Apply (E7-5): GET /api/recorder/preview, else the
      // client's own filter of this recorder's lines
      set({ paletteOpen: false, menuOpen: null });
      const macro = st.state?.macro || [];
      const localSkip = /^\s*(#|lumen\.(open|save|close|select|undo|redo|display)\(|lumen\.run\("(file\.(sample|new|open_url|import_|close)|batch\.|rois\.restore_cleared|results\.restore_cleared))/;
      const fallback = () => {
        const kept = macro.filter((l) => l.trim() && !localSkip.test(l));
        return { lines: kept, skipped: macro.filter((l) => l.trim() && localSkip.test(l)) };
      };
      void api.recorderPreview().catch(fallback).then((r) => {
        const lines = Array.isArray(r?.lines) ? r.lines : [];
        const skipped = Array.isArray(r?.skipped) ? r.skipped.length : 0;
        const undone = Array.isArray((r as { undone?: string[] })?.undone) ? (r as { undone?: string[] }).undone! : [];
        defaults.lines = lines.join("\n");
        // the backend's sentence names the undone steps it removed (E7-5 / E8-3); the local fallback builds its own
        const summary = typeof (r as { summary?: string })?.summary === "string" ? (r as { summary?: string }).summary! : "";
        const note = lines.length
          ? (summary || `${lines.length} recorded line${lines.length === 1 ? "" : "s"} from this session's recorder will run on every file${skipped ? ` (${skipped} open / save / select / display line${skipped === 1 ? "" : "s"} left out)` : ""}. Edit the list above before Apply.`) +
            (undone.length ? ` Undone and left out: ${undone.map((l) => l.replace(/^lumen\.run\("([^"]+)".*$/, "$1")).join(", ")}.` : "")
          : undone.length
            ? `Every recorded step was undone (${undone.length} left out) – apply some commands first, or type the lumen.run(...) lines above.`
            : "The recorder has no command lines yet – apply some commands first, or type the lumen.run(...) lines above.";
        open(defaults, note);
      });
      return;
    }
    if (op.menu === "Edit/Options" || op.category === "Edit/Options") {
      // Edit ▸ Options sheets (also those listed elsewhere, e.g. Image ▸ Overlay ▸ Overlay Options…) edit the
      // live preferences: seed every field from GET /api/settings
      // (falling back to the loaded prefs) so the sheet shows what is set, see seedPrefParam()
      set({ paletteOpen: false, menuOpen: null });
      void api
        .getSettings()
        .catch(() => get().prefs as unknown as Record<string, unknown>)
        .then((settings) => {
          for (const p of op.params) defaults[p.name] = seedPrefParam(p, settings[p.name]);
          open(defaults);
        });
      return;
    }
    open(defaults);
  },

  closeSheet: () => set({ sheet: null, previewPlanes: null, previewFor: null }),
  setSheet: (patch) => {
    const s = get().sheet;
    if (s) set({ sheet: { ...s, ...patch } });
  },

  applySheet: async () => {
    const st = get();
    const s = st.sheet;
    if (!s || s.busy) return; // one run per sheet: a repeated / held Enter must not start a second run
    // out-of-range values never reach the server (it rejects them too, with the same wording)
    for (const p of s.op.params) {
      const v = s.params[p.name];
      if ((p.type === "int" || p.type === "float") && typeof v === "number" && ((p.min !== undefined && v < p.min) || (p.max !== undefined && v > p.max))) {
        set({ sheet: { ...s, error: `${p.label || p.name} must be ${p.min !== undefined && p.max !== undefined ? `between ${p.min} and ${p.max}` : p.min !== undefined ? `at least ${p.min}` : `at most ${p.max}`} (got ${v})` } });
        return;
      }
    }
    if (s.op.id === "file.new") {
      // a size warning before a huge image is allocated (E5-18): above 512 MB, or a quarter of the machine's RAM
      const bytes = newImageBytes(s.params);
      const ram = st.state?.memory_limit_bytes;
      const limit = Math.min(512 * 1048576, ram ? ram / 4 : Infinity);
      if (bytes > limit) {
        const answer = await st.askConfirm({ title: `Create a ${fmtMB(bytes)} image?`, message: `${s.params.width} × ${s.params.height} px × ${s.params.channels || 1} channel(s) × ${s.params.slices || 1} slice(s) × ${s.params.frames || 1} frame(s) ${s.params.type} takes ${fmtMB(bytes)} of RAM${ram ? ` (this machine has ${fmtMB(ram)})` : ""}, plus the same again for every undo step.`, confirmLabel: `Create ${fmtMB(bytes)} image`, danger: bytes > (ram ? ram / 2 : 2 * 1024 ** 3) });
        if (answer !== "confirm") return;
      }
    }
    set({ sheet: { ...s, busy: true, error: null } });
    // an error shows inside the sheet (role=alert next to the buttons) instead of a toast that may cover them (E5-5)
    const res = await st.runOp(s.op.id, s.params, { scope: s.scope, onError: (message) => {
      const cur = get().sheet;
      if (cur && cur.op === s.op) set({ sheet: { ...cur, error: message } });
      else get().toast(message, "error");
    } });
    // The sheet may have been closed (Cancel / X / Esc; it is non-modal) or replaced by another op's
    // sheet while the request was pending: only touch it if it is still the sheet this call marked busy.
    // (Spreading a null sheet here used to yield `{ busy: false }` and crash OperationSheet on `op`.)
    const cur = get().sheet;
    if (!cur || !cur.busy || cur.op !== s.op) return;
    if (res) get().closeSheet();
    else set({ sheet: { ...cur, busy: false } });
  },

  setDisplay: (imageId, patch, immediate) => {
    const st = get();
    if (!st.state) return;
    const images = st.state.images.map((im) => (im.id === imageId ? applyDisplayPatch(im, patch) : im));
    set({ state: { ...st.state, images } });
    displayLog.push({ seq: ++displaySeq, id: imageId, patch });
    const prev = pendingDisplay[imageId] || {};
    const merged: Record<string, any> = { ...prev, ...patch };
    if (prev.channels || patch.channels) {
      const n = Math.max(prev.channels?.length || 0, patch.channels?.length || 0);
      merged.channels = Array.from({ length: n }, (_, i) => ({ ...(prev.channels?.[i] || {}), ...(patch.channels?.[i] || {}) }));
    }
    pendingDisplay[imageId] = merged;
    if (displayTimer) window.clearTimeout(displayTimer);
    if (immediate) void get().flushDisplay();
    else displayTimer = window.setTimeout(() => void get().flushDisplay(), 250);
  },

  flushDisplay: async () => {
    if (displayTimer) {
      window.clearTimeout(displayTimer);
      displayTimer = null;
    }
    const pend = pendingDisplay;
    pendingDisplay = {};
    // serialise behind earlier POSTs so patches reach the server in order and so that awaiting
    // flushDisplay() means the server has everything the viewport is showing
    const seqFlushed = displaySeq;
    displayInflight = displayInflight
      .then(() => Promise.all(Object.entries(pend).map(([id, patch]) => api.display(id, patch).catch(() => undefined))))
      .then(() => {
        // acknowledged edits are only still needed by a GET that is already in flight
        if (refreshSeq === refreshApplied) displayLog = displayLog.filter((e) => e.seq > seqFlushed);
      });
    await displayInflight;
  },

  setPosition: (imageId, pos) => get().setDisplay(imageId, pos),

  setSelection: (imageId, roi) => {
    const patch: Partial<Store> = { selections: { ...get().selections, [imageId]: roi } };
    if (roi) patch.lastSelections = { ...get().lastSelections, [imageId]: roi };
    set(patch);
  },

  selectImage: (id) => {
    const st = get();
    if (!st.state || st.state.active === id) return;
    set({ state: { ...st.state, active: id } });
    void api.select(id).catch(() => undefined);
  },

  closeImage: async (id, opts = {}) => {
    const st = get();
    const img = st.image(id);
    if (!img) return;
    if (!opts.force && st.prefs.confirm_close_modified && st.isModified(id)) {
      const n = st.state?.history[id]?.undo.length ?? 0;
      const answer = await st.askConfirm({
        title: `Close “${img.name}”?`,
        message: `${n === 1 ? "One command has" : `${n} commands have`} changed it since it was last saved (${n === 1 ? "the step" : "the steps"} in History). Closing discards ${n === 1 ? "that change" : "those changes"} unless you save first.`,
        confirmLabel: "Close without saving",
        altLabel: "Save…",
        danger: true,
        dontAskPref: "confirm_close_modified",
      });
      if (answer === "cancel") return;
      if (answer === "alt") {
        set({ fileDialog: { mode: "save", imageId: id } });
        return;
      }
    }
    if (get().ops["file.close"]) {
      // through the registry op so the close is recorded like any other command (macro recorder)
      await get().runOp("file.close", {}, { imageId: id, silent: true });
      return;
    }
    try {
      await api.close(id);
      await get().refresh();
    } catch (e: any) {
      get().toast(e.message, "error");
    }
  },
  closeImages: async (ids, opts = {}) => {
    const st = get();
    const open = new Set((st.state?.images || []).map((im) => im.id));
    const todo = ids.filter((id, i) => open.has(id) && ids.indexOf(id) === i);
    if (!todo.length) return;
    if (!opts.force && st.prefs.confirm_close_modified) {
      const modified = todo.filter((id) => st.isModified(id)).map((id) => st.image(id)!.name);
      if (modified.length) {
        const names = modified.slice(0, 5).map((n) => `“${n}”`).join(", ") + (modified.length > 5 ? ` and ${modified.length - 5} more` : "");
        const answer = await st.askConfirm({
          title: todo.length === 1 ? `Close “${st.image(todo[0])!.name}”?` : `Close ${todo.length} images?`,
          message: `${modified.length === 1 ? `${names} has` : `${modified.length} of them have`} unsaved changes (${modified.length === 1 ? "" : names + "; "}the undo history). Closing discards them.`,
          confirmLabel: "Close without saving",
          danger: true,
          dontAskPref: "confirm_close_modified",
        });
        if (answer !== "confirm") return;
      }
    }
    if (todo.length === open.size && get().ops["file.close_all"]) {
      await get().runOp("file.close_all", {}, { silent: true });
      return;
    }
    for (const id of todo) {
      try {
        await api.close(id);
      } catch (e: any) {
        get().toast(e.message || String(e), "error");
      }
    }
    await get().refresh();
  },
  closeAllImages: async () => {
    const ids = (get().state?.images || []).map((im) => im.id);
    if (!ids.length) return get().toast("No images to close", "info");
    await get().closeImages(ids);
  },

  undo: async () => {
    const img = get().activeImage();
    if (!img) return;
    await get().flushDisplay();
    const r = await api.undo(img.id).catch((e) => ({ label: null, err: e }));
    if (r.label) get().toast(`Undo ${r.label}`, "info");
    await get().refresh();
  },
  redo: async () => {
    const img = get().activeImage();
    if (!img) return;
    await get().flushDisplay();
    const r = await api.redo(img.id).catch(() => ({ label: null }));
    if (r.label) get().toast(`Redo ${r.label}`, "info");
    await get().refresh();
  },

  toast: (msg, kind = "info", opts) => {
    const id = ++toastId;
    const actions = opts?.actions;
    set({ toasts: [...get().toasts, { id, msg, kind, actions }].slice(-5) });
    // a toast with a button stays long enough to be used; errors stay longer than notes
    window.setTimeout(() => get().dismissToast(id), opts?.timeout ?? (actions?.length ? 12000 : kind === "error" ? 7000 : 3500));
  },
  dismissToast: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  setTool: (tool) => {
    const prev = get().tool;
    set({ tool, colorPickerOpen: tool === "picker" ? false : get().colorPickerOpen });
    // ImageJ macro tool sets: picking a tool runs its "<name> Selected" macro once (the op's `selected` param)
    const opId = pluginToolId(tool);
    if (opId && tool !== prev) {
      const spec = get().ops[opId];
      if (spec?.tool && spec.params.some((p) => p.name === "selected")) void get().runOp(opId, { selected: true }, { silent: true });
    }
  },
  setSpace: (down) => set({ spaceDown: down }),
  setView: (imageId, v) => set({ views: { ...get().views, [imageId]: v } }),
  togglePanel: (key, open) => set({ panels: { ...get().panels, [key]: open === undefined ? !get().panels[key] : open } }),
  setPaletteOpen: (open) => set({ paletteOpen: open, menuOpen: null }),
  setAgentOpen: (open) => set({ agentOpen: open, menuOpen: null }),
  setTheme: (t) => {
    localStorage.setItem("lumen.theme", t);
    document.documentElement.dataset.theme = t;
    set({ theme: t });
  },
  setFileDialog: (d) => {
    // the desktop app: the system's Open / Save / folder dialog instead of the in-app server browser (native.ts)
    if (d && !d.inApp) {
      void runNativeFileDialog(d, () => get() as never).then((handled) => { if (!handled) set({ fileDialog: d }); });
      return;
    }
    set({ fileDialog: d });
  },
  setCursor: (c) => set({ cursor: c }),
  setPlot: (p) => set({ plot: p }),
  setHistogram: (h) => set({ histogram: h }),
  setSelectedRoiIds: (ids) => set({ selectedRoiIds: ids }),
  setShowManagerRois: (v) => set({ showManagerRois: v }),
  setHelpOpen: (v) => set({ helpOpen: v }),
  setWandTolerance: (v) => set({ wandTolerance: v }),
  setMenuOpen: (m) => set({ menuOpen: m }),
  setInspectorOpen: (v) => set({ inspectorOpen: v }),

  openFiles: async (files) => {
    const list = Array.from(files);
    for (const f of list) {
      try {
        set({ busy: get().busy + 1 });
        await api.upload(f);
        get().toast(`Opened ${f.name}`, "success");
      } catch (e: any) {
        get().toast(`${f.name}: ${e.message}`, "error");
      } finally {
        set({ busy: Math.max(0, get().busy - 1) });
      }
    }
    await get().refresh();
  },

  setPreview: (planes, imageId) => set({ previewPlanes: planes, previewFor: imageId }),

  inspectorWidth: loadInspectorWidth(),
  setInspectorWidth: (w) => {
    const v = clampInspectorWidth(w);
    if (v === INSPECTOR_DEFAULT_W) localStorage.removeItem("lumen.inspectorWidth");
    else localStorage.setItem("lumen.inspectorWidth", String(v));
    set({ inspectorWidth: v });
  },
  tableModal: null,
  setTableModal: (m) => set({ tableModal: m }),
  imageOrder: loadImageOrder(),
  setImageOrder: (ids) => {
    const open = new Set((get().state?.images || []).map((im) => im.id));
    const order = ids.filter((id, i) => open.has(id) && ids.indexOf(id) === i);
    localStorage.setItem("lumen.imageOrder", JSON.stringify(order));
    set({ imageOrder: order });
  },
  split: { on: false, other: null, activeSide: 0 },
  setSplit: (patch) => set({ split: { ...get().split, ...patch } }),
  toggleSplitView: () => {
    const st = get();
    if (st.split.on) return set({ split: { ...st.split, on: false } });
    const images = st.state?.images || [];
    const active = st.state?.active ?? null;
    if (images.length < 2) return st.toast("Open a second image to use the split view", "info");
    const other = st.split.other && st.split.other !== active && images.some((im) => im.id === st.split.other) ? st.split.other : images.find((im) => im.id !== active)?.id ?? null;
    set({ split: { on: true, other, activeSide: 0 } });
  },
}));

// ---------------------------------------------------------------- layout persistence
// (INSPECTOR_DEFAULT_W / INSPECTOR_MIN_W are declared above `useStore`: the initial state calls
// loadInspectorWidth() while this module is still being evaluated, so they must already be initialised.)
/** Clamp an inspector width to [260 px, 60 % of the window] – and never so wide that less than 420 px are left for the rail + image (narrow windows, E5-20). */
export function clampInspectorWidth(w: number): number {
  const win = typeof window !== "undefined" ? window.innerWidth : 1600;
  const max = Math.max(INSPECTOR_MIN_W, Math.min(Math.floor(win * 0.6), win - 420));
  return Math.round(Math.min(max, Math.max(INSPECTOR_MIN_W, Number.isFinite(w) ? w : INSPECTOR_DEFAULT_W)));
}
function loadInspectorWidth(): number {
  try {
    const v = Number(localStorage.getItem("lumen.inspectorWidth"));
    return v > 0 ? clampInspectorWidth(v) : INSPECTOR_DEFAULT_W;
  } catch {
    return INSPECTOR_DEFAULT_W;
  }
}
function loadImageOrder(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem("lumen.imageOrder") || "[]");
    return Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}
/** Open images in tab order: ids listed in `order` first, the rest in server order. */
export function orderImages(images: ImageSummary[], order: string[]): ImageSummary[] {
  if (!order.length) return images;
  const rank = new Map(order.map((id, i) => [id, i]));
  const listed = images.filter((im) => rank.has(im.id)).sort((a, b) => rank.get(a.id)! - rank.get(b.id)!);
  const rest = images.filter((im) => !rank.has(im.id));
  return listed.concat(rest);
}
/** Open images in tab order (memoised on the images array and the order array). */
export function useOrderedImages(): ImageSummary[] {
  const images = useStore((s) => s.state?.images);
  const order = useStore((s) => s.imageOrder);
  return useMemo(() => orderImages(images || [], order), [images, order]);
}

// Inspector density (Preferences ▸ Appearance): the cached value avoids a layout jump before /api/settings answers.
try {
  applyDensity(localStorage.getItem(DENSITY_KEY));
} catch {
  /* ignore */
}

// Inspector sections: restore the open/closed state the user last had (localStorage "lumen.panels"),
// then keep it up to date.  Done here, outside the initialiser, so new PanelKeys only need the default.
(() => {
  try {
    const saved = JSON.parse(localStorage.getItem("lumen.panels") || "null");
    if (saved && typeof saved === "object") {
      const cur = useStore.getState().panels;
      const next = { ...cur };
      for (const k of Object.keys(cur) as PanelKey[]) if (typeof saved[k] === "boolean") next[k] = saved[k];
      useStore.setState({ panels: next });
    }
  } catch {
    /* ignore a corrupt entry */
  }
  // A new session (a fresh window, not a reload of the same one) opens the Workflows panel on "Choose a workflow…":
  // the guided workflow of an earlier session is not carried over (user request, Nov 2026).
  try {
    if (!sessionStorage.getItem("lumen-session-started")) {
      sessionStorage.setItem("lumen-session-started", "1");
      useStore.setState({ panels: { ...useStore.getState().panels, workflows: true } });
      const raw = JSON.parse(localStorage.getItem("lumen.workflow") || "null");
      if (raw && typeof raw === "object" && raw.id) localStorage.setItem("lumen.workflow", JSON.stringify({ id: null, done: [], skipped: [], params: {}, sourceImage: null, sourceName: null }));
    }
  } catch {
    /* storage blocked: the defaults apply */
  }
  useStore.subscribe((s, prev) => {
    if (s.panels !== prev.panels) {
      try {
        localStorage.setItem("lumen.panels", JSON.stringify(s.panels));
      } catch {
        /* quota / private mode */
      }
    }
  });
})();

let memoryTimer: number | null = null;
let memoryRun = 0;

/**
 * Plugins ▸ Utilities ▸ Monitor Memory: sample /api/state every MEMORY_SAMPLE_MS while the monitor
 * is on.  Lives in the store rather than in the MemoryMonitor component so that collapsing the Log
 * section or hiding the inspector (both unmount the sparkline) cannot silently pause the recording
 * while the status pill, the menu check mark and the Log badge still say it is on.
 */
function startMemorySampling() {
  if (memoryTimer !== null) return; // already running (setMemoryMonitor(true) while on)
  const run = ++memoryRun;
  const tick = async () => {
    try {
      const st = await api.state();
      if (run === memoryRun && typeof st.memory_bytes === "number") useStore.getState().pushMemorySample(st.memory_bytes);
    } catch {
      /* server away: no sample; the sparkline shows the gap */
    }
  };
  memoryTimer = window.setInterval(() => void tick(), MEMORY_SAMPLE_MS);
  void tick();
}

function stopMemorySampling() {
  if (memoryTimer === null) return;
  window.clearInterval(memoryTimer);
  memoryTimer = null;
  memoryRun++; // a request still in flight must not add a sample after Stop
}

/** Show a command's result in the window: the selection it made or cleared, its table / plot / histogram and the
 *  panels its output landed in.  Used by runOp and by the AI-assistant bridge (agentBridge.ts) for commands the
 *  assistant ran through the API. */
export function showRunResult(res: OpRunResult, imageId: string | null) {
  const after = useStore.getState();
  const img = imageId ? after.image(imageId) : after.activeImage();
  const target = res.new_images && res.new_images.length ? after.image(res.new_images[res.new_images.length - 1]) : img;
  if (res.roi && target) after.setSelection(target.id, res.roi);
  if (res.clear_roi && target) after.setSelection(target.id, null);
  if (res.plot) {
    useStore.setState({ plot: res.plot });
    after.togglePanel("plot", true);
  }
  if (res.histogram && res.histogram.edges) {
    useStore.setState({ histogram: res.histogram });
    after.togglePanel("histogram", true);
  }
  if (res.table) {
    useStore.setState({ table: res.table });
    after.togglePanel("table", true);
  }
  if (res.results_added) after.togglePanel("results", true);
  if (res.rois_added) after.togglePanel("rois", true);
  if (res.threshold) after.togglePanel("threshold", true);
}

function scheduleRefresh() {
  if (refreshTimer) return;
  refreshTimer = window.setTimeout(() => {
    refreshTimer = null;
    void useStore.getState().refresh();
  }, 60);
}

function connectEvents(onEvent: () => void) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws: WebSocket | null = null;
  const open = () => {
    ws = new WebSocket(`${proto}://${location.host}/api/events`);
    // (re)connected: resync the running jobs (a job may have finished while the socket was down)
    ws.onopen = () => void syncJobs();
    ws.onmessage = (ev) => {
      let evt: any = null;
      try {
        evt = typeof ev.data === "string" ? JSON.parse(ev.data) : null;
      } catch {
        evt = null;
      }
      if (evt && evt.type === "progress" && evt.job) {
        applyProgress(evt.job as JobInfo); // progress never changes the session state: no refresh
        return;
      }
      if (evt && evt.type === "ui") {
        // a command from the AI assistant (lumen.mcp_server via POST /api/ui/command)
        void import("./agentBridge").then((m) => m.applyAgentCommand(evt));
        return;
      }
      if (evt && evt.type === "ops") {
        // plugins were (re)loaded on the server: rebuild the command registry, menus and palette
        api.meta()
          .then((meta) => {
            const ops: Record<string, OpSpec> = {};
            for (const o of meta.ops) ops[o.id] = o;
            const st = useStore.getState();
            // the tool rail follows the registry: a plugin tool that went away falls back to the pointer
            const cur = st.tool;
            const gone = typeof cur === "string" && cur.startsWith("op:") && !ops[cur.slice(3)]?.tool;
            useStore.setState({ meta, ops, ...(gone ? { tool: "pointer" as const } : {}) });
            st.toast(`Commands reloaded: ${meta.ops.length}${gone ? " – the active plugin tool is gone" : ""}`, "info");
          })
          .catch(() => undefined);
      }
      onEvent();
    };
    ws.onclose = () => window.setTimeout(open, 1500);
  };
  open();
}

/** Dispatch a `progress` event into `store.jobs`; a finished job is removed and a cancelled one toasted. */
function applyProgress(j: JobInfo) {
  const st = useStore.getState();
  const prev = st.jobs[j.id];
  if (j.done) {
    if (prev) {
      const jobs = { ...st.jobs };
      delete jobs[j.id];
      useStore.setState({ jobs });
    }
    if (j.cancelled || j.status === "cancelled") st.toast(`Cancelled ${j.name}`, "info");
    return;
  }
  useStore.setState({ jobs: { ...st.jobs, [j.id]: { ...prev, ...j, seenAt: prev?.seenAt ?? performance.now() } } });
}

async function syncJobs() {
  try {
    const list = await api.jobs();
    const old = useStore.getState().jobs;
    const jobs: Record<string, Job> = {};
    for (const j of list) jobs[j.id] = { ...old[j.id], ...j, seenAt: old[j.id]?.seenAt ?? performance.now() };
    useStore.setState({ jobs });
  } catch {
    /* an older server without /api/jobs: nothing to show */
  }
}

// ---------------------------------------------------------------- plane cache
interface CacheEntry { promise: Promise<PlaneData>; bytes: number } // bytes = 0 while in flight
const planeCache = new Map<string, CacheEntry>(); // insertion order = least recently used first
let cacheBytes = 0; // exact sum of .bytes over the entries currently in planeCache
const CACHE_LIMIT = 600 * 1024 * 1024;

export function planeKey(img: ImageSummary, t: number, z: number, stride: number): string {
  return `${img.id}:${img.rev}:${t}:${z}:${stride}`;
}

function dropEntry(key: string, e: CacheEntry) {
  // identity-guarded: never remove a newer entry that re-used the key after `e` was evicted
  if (planeCache.get(key) !== e) return;
  planeCache.delete(key);
  cacheBytes -= e.bytes;
}

/** Evict least-recently-used entries until the counted bytes fit; always keep the newest one. */
function evictUntilFits() {
  for (const [k, e] of planeCache) {
    if (cacheBytes <= CACHE_LIMIT || planeCache.size <= 1) break;
    dropEntry(k, e);
  }
}

/** Planes of an older rev of `img` can never be requested again (the key embeds the rev; undo bumps it too). */
function dropStaleRevs(img: ImageSummary) {
  const prefix = `${img.id}:`;
  for (const [k, e] of planeCache) {
    if (!k.startsWith(prefix)) continue;
    const rev = Number(k.slice(prefix.length).split(":")[0]);
    if (rev < img.rev) dropEntry(k, e);
  }
}

export function getPlanes(img: ImageSummary, t: number, z: number, stride: number): Promise<PlaneData> {
  const key = planeKey(img, t, z, stride);
  const hit = planeCache.get(key);
  if (hit) {
    // LRU touch: re-insert so "oldest" means least recently used
    planeCache.delete(key);
    planeCache.set(key, hit);
    return hit.promise;
  }
  dropStaleRevs(img);
  const entry: CacheEntry = { promise: undefined as unknown as Promise<PlaneData>, bytes: 0 };
  entry.promise = api.planes(img.id, t, z, stride).then((pd) => {
    // evicted while in flight: hand the data to the caller but do not account for it
    if (planeCache.get(key) !== entry) return pd;
    entry.bytes = pd.channels.reduce((a, c) => a + c.byteLength, 0);
    cacheBytes += entry.bytes;
    evictUntilFits();
    return pd;
  });
  entry.promise.catch(() => dropEntry(key, entry));
  planeCache.set(key, entry);
  return entry.promise;
}

export function peekPlanes(img: ImageSummary, t: number, z: number, stride: number): Promise<PlaneData> | undefined {
  return planeCache.get(planeKey(img, t, z, stride))?.promise;
}

export function useOps(): OpSpec[] {
  return useStore((s) => s.meta?.ops || []);
}

/** One-step undo of the last ROI Manager / Results clear (POST /api/session/restore_clear, else the restore ops). */
export async function restoreClear(kind: "rois" | "results"): Promise<void> {
  const st = useStore.getState();
  try {
    let restored = 0;
    try {
      restored = (await api.restoreClear(kind)).restored;
    } catch {
      const opId = kind === "rois" ? "rois.restore_cleared" : "results.restore_cleared";
      if (!st.ops[opId]) throw new Error("this server cannot restore a cleared " + (kind === "rois" ? "ROI Manager" : "Results table"));
      const r = await api.run(opId, {}, null, null, "stack");
      restored = r.rois_added || r.results_added || 1;
    }
    await useStore.getState().refresh();
    useStore.getState().toast(restored ? `${restored} ${kind === "rois" ? "ROI" : "row"}${restored === 1 ? "" : "s"} restored` : "Nothing to restore – something was cleared or added since", restored ? "success" : "info");
  } catch (e: any) {
    useStore.getState().toast(`Undo clear failed: ${e?.message || e}`, "error");
  }
}
