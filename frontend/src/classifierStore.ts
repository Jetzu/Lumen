/**
 * Pixel Classifier (Trainable Weka Segmentation) – the state of the Inspector ▸ Pixel Classifier panel
 * and the helpers it shares with the viewport.
 *
 * Everything that changes the session goes through the registry ops (`segment.*` / `segment3d.*`) and the
 * ROI-manager REST calls; the classifier's status comes back through the session state: the ops publish the
 * `Classifier classes` / `Classifier summary` tables (`Classifier 3D …` for the voxel classifier).
 *
 * Training traces are ROI-manager entries named `class:<name>` (the convention the training ops understand),
 * drawn in the class colour on the plane they were made on.  They come from
 *  - the trace mode of the panel: every selection drawn with the freehand / rectangle / oval / line tools is
 *    captured as a trace of the active class (a freehand stroke becomes a pen line of `traceWidth` px),
 *  - the "Classifier Trace Tool" plugin tool (`op:segment.trace_tool`, a drag = a pen stroke, a click = a dot),
 *  - the brush tool, when the viewport hands its strokes to {@link captureBrushStroke}.
 */
import { create } from "zustand";
import { api } from "./api";
import { useStore } from "./store";
import type { ImageSummary, OpRunResult, Roi, Tool } from "./types";

export interface ClassifierClass {
  name: string;
  color: string;
}
export type ClassifierMode = "2d" | "3d";
export type TraceStyle = "line" | "area";

export interface ClassifierSettings {
  channels: "Current channel" | "All channels";
  features: string[];
  features3d: string[];
  minSigma: number;
  maxSigma: number;
  membraneThickness: number;
  membranePatchSize: number;
  classifier: string;
  nTrees: number;
  maxDepth: number;
  featuresPerNode: number;
  balanceClasses: boolean;
  maxSamplesPerClass: number;
  svmMaxSamples: number;
  kNeighbors: number;
  hiddenUnits: number;
  smooth: number;
  overlayOpacity: number;
}

/** Weka's class colours: red, green, blue, cyan, magenta, yellow, orange, pink, grey, white. */
export const WEKA_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff00ff", "#ffff00", "#ffa500", "#ff69b4", "#9e9e9e", "#ffffff"];
export const DEFAULT_FEATURES = ["Gaussian", "Difference of Gaussians", "Sobel", "Laplacian", "Hessian", "Variance"];
export const DEFAULT_FEATURES_3D = ["Gaussian", "Difference of Gaussians", "Sobel", "Hessian", "Variance"];
export const DEFAULT_SETTINGS: ClassifierSettings = {
  channels: "Current channel",
  features: DEFAULT_FEATURES,
  features3d: DEFAULT_FEATURES_3D,
  minSigma: 1,
  maxSigma: 8,
  membraneThickness: 1,
  membranePatchSize: 19,
  classifier: "Random forest",
  nTrees: 100,
  maxDepth: 0,
  featuresPerNode: 0,
  balanceClasses: true,
  maxSamplesPerClass: 50000,
  svmMaxSamples: 4000,
  kNeighbors: 5,
  hiddenUnits: 64,
  smooth: 0,
  overlayOpacity: 0.33,
};
export const DEFAULT_CLASSES: ClassifierClass[] = [
  { name: "background", color: WEKA_COLORS[0] },
  { name: "foreground", color: WEKA_COLORS[1] },
];

export const TRACE_PREFIX = "class:";
export const TRACE_TOOL_OP = "segment.trace_tool";
export const TRACE_TOOL: Tool = `op:${TRACE_TOOL_OP}`;

/** The op ids behind the panel, per mode. */
export const OPS: Record<ClassifierMode, { train: string; apply: string; save: string; load: string; prefix: string }> = {
  "2d": { train: "segment.train_classifier", apply: "segment.apply_classifier", save: "segment.save_classifier", load: "segment.load_classifier", prefix: "Classifier" },
  "3d": { train: "segment3d.train_classifier", apply: "segment3d.apply_classifier", save: "segment3d.save_classifier", load: "segment3d.load_classifier", prefix: "Classifier 3D" },
};

interface ClassifierStore {
  classes: ClassifierClass[];
  /** Index of the active class (traces go to it). */
  active: number;
  mode: ClassifierMode;
  traceMode: boolean;
  traceStyle: TraceStyle;
  traceWidth: number;
  settings: ClassifierSettings;
  settingsOpen: boolean;
  /** Wall time of the last training run started from the panel (ms), for the status line. */
  lastTrainMs: number | null;
  /** The panel action running right now (disables the buttons). */
  running: string | null;
  setClasses: (classes: ClassifierClass[]) => void;
  addClass: (name?: string) => void;
  renameClass: (index: number, name: string) => void;
  recolorClass: (index: number, color: string) => void;
  removeClass: (index: number) => void;
  setActive: (index: number) => void;
  setMode: (mode: ClassifierMode) => void;
  setTraceMode: (on: boolean) => void;
  setTraceStyle: (style: TraceStyle) => void;
  setTraceWidth: (w: number) => void;
  patchSettings: (patch: Partial<ClassifierSettings>) => void;
  setSettingsOpen: (open: boolean) => void;
  setLastTrainMs: (ms: number | null) => void;
  setRunning: (what: string | null) => void;
}

const STORAGE_KEY = "lumen.classifier";

function loadPersisted(): Partial<Pick<ClassifierStore, "classes" | "settings" | "traceStyle" | "traceWidth">> {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!raw || typeof raw !== "object") return {};
    const out: Partial<Pick<ClassifierStore, "classes" | "settings" | "traceStyle" | "traceWidth">> = {};
    if (Array.isArray(raw.classes) && raw.classes.length >= 2 && raw.classes.every((c: unknown) => c && typeof (c as ClassifierClass).name === "string"))
      out.classes = raw.classes.map((c: ClassifierClass, i: number) => ({ name: String(c.name), color: /^#[0-9a-f]{6}$/i.test(String(c.color)) ? String(c.color).toLowerCase() : WEKA_COLORS[i % WEKA_COLORS.length] }));
    if (raw.settings && typeof raw.settings === "object") out.settings = { ...DEFAULT_SETTINGS, ...raw.settings };
    if (raw.traceStyle === "line" || raw.traceStyle === "area") out.traceStyle = raw.traceStyle;
    if (typeof raw.traceWidth === "number" && raw.traceWidth >= 1) out.traceWidth = raw.traceWidth;
    return out;
  } catch {
    return {};
  }
}

function uniqueName(classes: ClassifierClass[], base: string): string {
  const taken = new Set(classes.map((c) => c.name.toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  for (let i = 2; ; i++) if (!taken.has(`${base} ${i}`)) return `${base} ${i}`;
}

export const useClassifierStore = create<ClassifierStore>((set, get) => ({
  classes: DEFAULT_CLASSES,
  active: 1,
  mode: "2d",
  traceMode: false,
  traceStyle: "line",
  traceWidth: 6,
  settings: DEFAULT_SETTINGS,
  settingsOpen: false,
  lastTrainMs: null,
  running: null,
  ...loadPersisted(),
  setClasses: (classes) => set({ classes, active: Math.min(get().active, classes.length - 1) }),
  addClass: (name) => {
    const classes = get().classes;
    const nm = uniqueName(classes, (name || "").trim() || `class ${classes.length + 1}`);
    set({ classes: [...classes, { name: nm, color: WEKA_COLORS[classes.length % WEKA_COLORS.length] }], active: classes.length });
  },
  renameClass: (index, name) => {
    const classes = get().classes.slice();
    const nm = name.trim().replace(/,/g, " ");
    if (!classes[index] || !nm) return;
    if (classes.some((c, i) => i !== index && c.name.toLowerCase() === nm.toLowerCase())) return;
    classes[index] = { ...classes[index], name: nm };
    set({ classes });
  },
  recolorClass: (index, color) => {
    const classes = get().classes.slice();
    if (!classes[index]) return;
    classes[index] = { ...classes[index], color: color.toLowerCase() };
    set({ classes });
  },
  removeClass: (index) => {
    const classes = get().classes.filter((_, i) => i !== index);
    if (classes.length < 2) return;
    const active = get().active;
    set({ classes, active: Math.max(0, Math.min(active > index ? active - 1 : active, classes.length - 1)) });
  },
  setActive: (index) => set({ active: Math.max(0, Math.min(index, get().classes.length - 1)) }),
  setMode: (mode) => set({ mode }),
  setTraceMode: (on) => set({ traceMode: on }),
  setTraceStyle: (style) => set({ traceStyle: style }),
  setTraceWidth: (w) => set({ traceWidth: Math.max(1, Math.min(200, Math.round(w))) }),
  patchSettings: (patch) => set({ settings: { ...get().settings, ...patch } }),
  setSettingsOpen: (open) => set({ settingsOpen: open }),
  setLastTrainMs: (ms) => set({ lastTrainMs: ms }),
  setRunning: (what) => set({ running: what }),
}));

useClassifierStore.subscribe((s, prev) => {
  if (s.classes === prev.classes && s.settings === prev.settings && s.traceStyle === prev.traceStyle && s.traceWidth === prev.traceWidth) return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ classes: s.classes, settings: s.settings, traceStyle: s.traceStyle, traceWidth: s.traceWidth }));
  } catch {
    /* storage may be unavailable */
  }
});

// ------------------------------------------------------------------ traces
export function traceName(cls: string): string {
  return TRACE_PREFIX + cls;
}

/** The class name of a trace ROI (`class:<name>`), or null for any other ROI. */
export function traceClass(roi: Roi): string | null {
  const n = (roi.name || "").trim();
  return n.toLowerCase().startsWith(TRACE_PREFIX) ? n.slice(TRACE_PREFIX.length).trim() : null;
}

/** The traces of an image (ROIs without image id belong to every image, like the training ops assume). */
export function tracesOf(rois: Roi[], img: ImageSummary): Roi[] {
  return rois.filter((r) => (!r.image_id || r.image_id === img.id) && traceClass(r) !== null);
}

/** Selection types the trace mode captures as soon as they are committed (polygon / polyline are built click by click, so they are added with the button). */
export const CAPTURE_TYPES = new Set<Roi["type"]>(["freehand", "rect", "oval", "line", "mask"]);

/** A selection as a trace ROI of a class: a freehand outline becomes a pen line (`line` style), everything else keeps its shape. */
export function toTrace(roi: Roi, cls: ClassifierClass, style: TraceStyle, width: number): Roi {
  const { id: _id, image_id: _img, position: _pos, ...rest } = roi;
  let out: Roi = { ...rest } as Roi;
  if (style === "line" && (roi.type === "freehand" || roi.type === "polygon") && roi.points && roi.points.length >= 2) {
    out = { type: "polyline", points: roi.points.map((p) => [p[0], p[1]]), width: Math.max(1, width) };
  } else if (roi.type === "line" || roi.type === "polyline") {
    out = { ...out, width: Math.max(roi.width || 1, style === "line" ? width : 1) };
  }
  return { ...out, name: traceName(cls.name), color: cls.color };
}

export async function addTrace(img: ImageSummary, roi: Roi, cls: ClassifierClass, style: TraceStyle, width: number): Promise<boolean> {
  const st = useStore.getState();
  try {
    await api.addRoi(toTrace(roi, cls, style, width), img.id, { t: img.display.t, z: img.display.z, c: img.display.c });
    await st.refresh();
    return true;
  } catch (e: any) {
    st.toast(e.message || String(e), "error");
    return false;
  }
}

/**
 * Brush strokes as traces: the viewport calls this before painting (`commitStroke`); while the trace mode is
 * on the stroke becomes a pen line of the active class and `true` tells the viewport not to paint it.
 */
export function captureBrushStroke(points: number[][], width: number): boolean {
  const cs = useClassifierStore.getState();
  const st = useStore.getState();
  const img = st.activeImage();
  const cls = cs.classes[cs.active];
  if (!cs.traceMode || !img || !cls || points.length < 1) return false;
  const pts = points.length === 1 ? [points[0], [points[0][0] + 0.01, points[0][1]]] : points;
  void addTrace(img, { type: "polyline", points: pts, width }, cls, "line", width);
  return true;
}

// ------------------------------------------------------------------ ops
/**
 * Run an op through the REST API without the store's result side effects (no table panel opening, no
 * recent-command bookkeeping): the panel's status comes from the session state it refreshes afterwards.
 */
export async function runQuiet(opId: string, params: Record<string, unknown>, imageId?: string | null): Promise<OpRunResult | undefined> {
  const st = useStore.getState();
  const img = imageId === undefined ? st.activeImage() : st.image(imageId);
  await st.flushDisplay();
  useStore.setState({ busy: useStore.getState().busy + 1 });
  try {
    const res = await api.run(opId, params, img?.id ?? null, null, "stack");
    await useStore.getState().refresh();
    return res;
  } catch (e: any) {
    if (e?.message !== "Cancelled") useStore.getState().toast(e.message || String(e), "error");
    return undefined;
  } finally {
    useStore.setState({ busy: Math.max(0, useStore.getState().busy - 1) });
  }
}

/** The parameters of the training op for the panel's classes and settings. */
export function trainParams(mode: ClassifierMode, s: ClassifierSettings, classes: ClassifierClass[], output: string): Record<string, unknown> {
  const common: Record<string, unknown> = {
    classes: classes.map((c) => c.name).join(","),
    class_colors: classes.map((c) => c.color).join(","),
    channels: s.channels,
    features: mode === "3d" ? s.features3d : s.features,
    min_sigma: s.minSigma,
    max_sigma: s.maxSigma,
    classifier: s.classifier,
    n_trees: s.nTrees,
    max_depth: s.maxDepth,
    features_per_node: s.featuresPerNode,
    balance_classes: s.balanceClasses,
    svm_max_samples: s.svmMaxSamples,
    k_neighbors: s.kNeighbors,
    hidden_units: s.hiddenUnits,
    apply_now: true,
    output,
    overlay_opacity: s.overlayOpacity,
  };
  if (mode === "3d") return { ...common, max_voxels_per_class: s.maxSamplesPerClass };
  return { ...common, membrane_thickness: s.membraneThickness, membrane_patch_size: s.membranePatchSize, max_pixels_per_class: s.maxSamplesPerClass };
}

/** The parameters of the apply op (`all` = whole stack / every frame). */
export function applyParams(mode: ClassifierMode, s: ClassifierSettings, output: string, all: boolean, imageId?: string): Record<string, unknown> {
  const p: Record<string, unknown> = { output, smooth: s.smooth, overlay_opacity: s.overlayOpacity, image: imageId ?? null };
  if (mode === "3d") p.all_frames = all;
  else p.all_planes = all;
  return p;
}

// ------------------------------------------------------------------ overlay slice sync
let overlaySyncStarted = false;

/**
 * A whole-stack classifier overlay is an `image` overlay item that shows the overlay colour image's own
 * current plane; keep that plane in step with the source image so scrolling the stack scrolls the overlay.
 */
export function startOverlaySync(): void {
  if (overlaySyncStarted) return;
  overlaySyncStarted = true;
  useStore.subscribe((s, prev) => {
    if (!s.state || s.state.images === prev.state?.images) return;
    const images = s.state.images;
    for (const im of images) {
      for (const o of im.overlay as unknown as Record<string, any>[]) {
        if (o.type !== "image" || o.source !== "classifier") continue;
        const pos = o.position as { t?: number | null; z?: number | null } | undefined;
        if (pos && pos.t != null && pos.z != null) continue; // fully pinned single plane (2-D classifier): nothing to sync
        const ov = images.find((x) => x.id === o.image_id);
        if (!ov) continue;
        // sync only the axes the position leaves free (a frame-pinned 3-D overlay follows the z slider only)
        const t = pos && pos.t != null ? 0 : Math.min(im.display.t, ov.shape.t - 1);
        const z = pos && pos.z != null ? 0 : Math.min(im.display.z, ov.shape.z - 1);
        if (ov.display.t !== t || ov.display.z !== z) s.setDisplay(ov.id, { t, z });
      }
    }
  });
}
