export interface ParamSpec {
  name: string;
  type: "int" | "float" | "bool" | "choice" | "string" | "text" | "image" | "multichoice" | "color";
  default: any;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  choices?: string[];
  help?: string;
  advanced?: boolean;
  unit?: string;
}

export interface OpSpec {
  id: string;
  name: string;
  category: string;
  kind: "plane" | "image" | "analysis";
  description: string;
  params: ParamSpec[];
  preview: boolean;
  roi_aware: boolean;
  requires_image: boolean;
  /** Selection the command needs: an area, a line, a point selection, or any ("any"); null = none. Newer servers may send further kinds. */
  requires_roi: "area" | "line" | "point" | "any" | (string & {}) | null;
  shortcut: string | null;
  menu: string;
  keywords: string[];
  scope_param: boolean;
  supports_rgb: boolean;
  icon?: string;
  /** ImageJ-macro equivalents of the command ("Gaussian Blur...", or full `run("…", "…")` calls); newer servers only (E7-20). */
  ij_commands?: string[];
  /** Plugin tool (`@op(..., tool=True)`): listed on the tool rail; clicks / drags on the image run the op with x, y, x2, y2, button, shift, alt, ctrl, event. */
  tool?: boolean;
  /** Emoji or 1–2 characters drawn as the tool's icon. */
  tool_icon?: string;
  /** Plugin frame: own sub-panel in Inspector ▸ Plugin Frames. */
  frame?: boolean;
  /** CSS colour tinting the tool icon (macro tools). */
  tool_color?: string;
  /** CSS cursor while the tool is active (crosshair, cell, pointer, copy …). */
  tool_cursor?: string;
}

export interface ChannelDisplay {
  name: string;
  lut: string;
  min: number;
  max: number;
  visible: boolean;
  inverted: boolean;
  gamma: number;
}

export type DisplayMode = "composite" | "single" | "grayscale";

export interface Display {
  channels: ChannelDisplay[];
  mode: DisplayMode;
  t: number;
  z: number;
  c: number;
}

export interface Calibration {
  pixel_width: number;
  pixel_height: number;
  pixel_depth: number;
  unit: string;
  frame_interval: number;
  time_unit: string;
  value_unit: string;
}

export type RoiType = "rect" | "oval" | "polygon" | "freehand" | "line" | "polyline" | "point" | "mask";

export interface Roi {
  id?: string;
  type: RoiType;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  points?: number[][];
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  width?: number;
  rle?: number[];
  name?: string;
  color?: string;
  stroke_width?: number;
  /** Fill colour (#rrggbb or #rrggbbaa) of closed shapes and mask runs; absent = outline only.  `fill_color` is the legacy key. */
  fill?: string;
  fill_color?: string;
  position?: { t: number; z: number; c: number };
  image_id?: string;
}

export interface ImageSummary {
  id: string;
  rev: number;
  name: string;
  path: string | null;
  shape: { t: number; z: number; c: number; y: number; x: number };
  dtype: string;
  bit_depth: string;
  rgb: boolean;
  calibration: Calibration;
  display: Display;
  value_range: [number, number];
  /** Bytes held in memory – for a virtual stack only its cached planes (see `virtual`). */
  nbytes: number;
  /** Data larger than RAM (lumen.core.lazy): the pixels still live in the file and planes are read on demand. */
  virtual?: boolean;
  /** Bytes of the stack held in memory right now (= nbytes) and the decoded size of the whole stack. */
  resident_bytes?: number;
  full_bytes?: number;
  /** How a virtual stack is read ("memory-mapped", "HDF5 dataset data", "OME-Zarr level 0" …); null when in memory. */
  virtual_source?: string | null;
  metadata: Record<string, string | number | boolean>;
  overlay: Roi[];
  binary_hint: boolean;
}

export interface Threshold {
  lower: number;
  upper: number;
  method?: string;
  dark_background?: boolean;
}

export interface PlotData {
  title: string;
  /** Sample positions; when series carry their own `x`, this is the sorted union of all of them. */
  x: number[];
  /** `x`, when present, has the same length as `values` and may be shorter than `PlotData.x` (analyze.line_graph: one column range per curve). */
  series: { name: string; values: number[]; lut?: string; x?: number[]; color?: string }[];
  x_label: string;
  y_label: string;
}

export interface HistogramData {
  counts: number[];
  edges?: number[];
  min?: number;
  max?: number;
  stats?: Record<string, number>;
  title?: string;
}

export interface TableData {
  name: string;
  columns: string[];
  rows: any[][];
}

export interface SessionState {
  version: number;
  images: ImageSummary[];
  active: string | null;
  results: { columns: string[]; rows: any[][] };
  rois: Roi[];
  /** Per image: the undoable / redoable step labels and (newer servers) the parameters of each step ("sigma=5, scope=slice"). */
  history: Record<string, { undo: string[]; redo: string[]; undo_detail?: string[]; redo_detail?: string[] }>;
  /** This client's recorder lines (per browser tab: the `X-Lumen-Client` header); the first line is a `#` session marker. */
  macro: string[];
  /** The client's recorder without its lines (newer servers): when it started and which image its last line targeted. */
  recorder?: { client: string; started: number; started_iso: string; marker?: string; active_image: string | null; count: number };
  log: { t: number; level: string; msg: string }[];
  measurements: string[];
  measure_options: Record<string, any>;
  thresholds: Record<string, Threshold>;
  last_plot: PlotData | null;
  tables: Record<string, TableData>;
  /** Recently opened file paths, newest first (File ▸ Open Recent). */
  recent?: string[];
  /** Bytes held by open images and their undo/redo history (status-bar chip, memory monitor). */
  memory_bytes?: number;
  /** Physical RAM of the machine running the server (memory guardrails). */
  memory_limit_bytes?: number;
  /** Decoded size above which File ▸ Open opens a file as a virtual stack automatically (null = never, preference auto_virtual_mb). */
  auto_virtual_bytes?: number | null;
  /** Budget (preference virtual_cache_mb) and current fill of the shared plane cache of virtual stacks. */
  virtual_cache_bytes?: number;
  virtual_resident_bytes?: number;
  /** Bumps whenever a LUT is registered, replaced or removed; the client refetches /api/luts when it changes. */
  lut_version?: number;
}

export interface Meta {
  version: string;
  ops: OpSpec[];
  luts: string[];
  threshold_methods: string[];
  measurements: { id: string; label: string }[];
  home: string;
  cwd: string;
}

export type TypedArray = Uint8Array | Uint16Array | Float32Array;

export interface PlaneData {
  dtype: string;
  shape: number[]; // [C, H, W]
  stride: number;
  channels: TypedArray[];
}

export interface OpRunResult {
  op: string;
  new_images?: string[];
  results_added?: number;
  plot?: PlotData;
  histogram?: HistogramData;
  table?: TableData;
  roi?: Roi;
  clear_roi?: boolean;
  rois_added?: number;
  threshold?: Threshold;
  message?: string;
  log?: string;
  elapsed_ms?: number;
}

export type BuiltinTool = "pointer" | "hand" | "rect" | "oval" | "polygon" | "freehand" | "line" | "polyline" | "point" | "wand" | "zoom" | "picker" | "brush" | "bucket" | "text" | "arrow" | "angle" | "track";
/** A registered plugin tool (`OpSpec.tool`): "op:" followed by the op id. */
export type PluginTool = `op:${string}`;
export type Tool = BuiltinTool | PluginTool;

/** The op id of a plugin tool, or null for a built-in tool. */
export function pluginToolId(tool: Tool | string | null | undefined): string | null {
  return typeof tool === "string" && tool.startsWith("op:") ? tool.slice(3) : null;
}

export interface ViewTransform {
  scale: number;
  tx: number;
  ty: number;
}
