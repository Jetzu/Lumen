/**
 * Renderers for the non-ROI overlay items that backend commands append to `img.overlay`:
 * text, arrow, track, calibration_bar and time_stamp.  Every item is rendered inside the
 * viewport's scaled <g>, in IMAGE pixels, with the same geometry that the backend rasteriser
 * (lumen/core/overlay_render.py, used by Burn In / Flatten) produces: font sizes, stroke widths,
 * arrow heads and bar sizes are image pixels and therefore grow and shrink with the zoom, so what
 * is on screen is what gets burnt in.  Only purely cosmetic strokes (text halos, dash patterns)
 * keep a constant screen size via 1/scale.  The calibration bar and time stamp are anchored to
 * the image corners exactly like the backend `anchor()`.
 */
import React, { useEffect, useMemo, useState } from "react";
import type { ImageSummary } from "../types";
import { lutGradientCss, type LutTables } from "../render/compositor";
import { useStore } from "../store";

export interface TextItem { type: "text"; id?: string; x: number; y: number; text: string; font_size?: number; color?: string; background?: boolean; bold?: boolean }
export interface ArrowItem { type: "arrow"; id?: string; x1: number; y1: number; x2: number; y2: number; color?: string; stroke_width?: number; head_size?: number; double_headed?: boolean }
export interface TrackItem { type: "track"; id: string; name?: string; points: number[][]; color?: string }
/** Another open image drawn on top of this one (Image ▸ Overlay ▸ Add Image…). */
export interface ImageOverlayItem { type: "image"; id?: string; image_id: string; x?: number; y?: number; opacity?: number; scale?: number }
/** `inverted` is a snapshot of the channel's Invert-LUT state when the bar was added (like lut/min/max), honoured
 *  by both this renderer and overlay_render._draw_calibration_bar; absent on legacy items (= not inverted). */
export interface CalibrationBarItem { type: "calibration_bar"; id?: string; position?: string; lut?: string; min?: number; max?: number; unit?: string; width?: number; height?: number; font_size?: number; color?: string; decimals?: number; ticks?: number; fill?: boolean; inverted?: boolean }
export interface TimeStampItem {
  type: "time_stamp"; id?: string; position?: string; format?: string; color?: string; font_size?: number;
  start_frame?: number; start_time?: number; interval?: number; label_start?: number; label_interval?: number;
}

const FONT = '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
let measureCtx: CanvasRenderingContext2D | null | undefined;

/** Width of `text` rendered at `fontPx` with the app font (canvas measureText), in the same units as `fontPx`. */
export function measureText(text: string, fontPx: number): number {
  if (measureCtx === undefined) measureCtx = document.createElement("canvas").getContext("2d");
  if (!measureCtx) return text.length * fontPx * 0.6;
  // measure at a fixed size and scale: canvas fonts below ~1 px are clamped by some browsers
  measureCtx.font = `100px ${FONT}`;
  return (measureCtx.measureText(text).width * fontPx) / 100;
}

const TEXT_HALO: React.CSSProperties = { paintOrder: "stroke", stroke: "rgba(0,0,0,0.65)", strokeLinejoin: "round" };

/**
 * The backend `overlay_render.burn_overlay` rule for per-plane items (optic-flow / block-matching
 * arrows, MorphoLibJ ellipses, tracking spots …): an item whose `position` is an object is drawn only
 * on its own plane, i.e. its `t` (when not null/undefined) must equal the displayed t and its `z`
 * likewise; `c` is not compared because channels are composited.  Items without a position, and
 * those whose `position` is a string anchor ("Lower Right" of scale bars / text), are drawn on
 * every plane.
 */
export function onPlane(item: { position?: unknown }, t: number, z: number): boolean {
  const pos = item.position;
  if (!pos || typeof pos !== "object") return true;
  const p = pos as { t?: number | null; z?: number | null };
  if (p.t !== null && p.t !== undefined && p.t !== t) return false;
  if (p.z !== null && p.z !== undefined && p.z !== z) return false;
  return true;
}

/** Backend `parse_color` luminance rule for the opaque box behind light/dark text. */
function boxBehind(color: string): string {
  const s = color.trim().replace(/^#/, "");
  const hex = s.length === 3 ? s.split("").map((c) => c + c).join("") : s;
  const sum = hex.length === 6 ? parseInt(hex.slice(0, 2), 16) + parseInt(hex.slice(2, 4), 16) + parseInt(hex.slice(4, 6), 16) : 765;
  return sum > 380 ? "#000000" : "#ffffff";
}

// ------------------------------------------------------------------ text
export function TextShape({ item, scale }: { item: TextItem; scale: number }) {
  const fs = item.font_size || 14; // image px (overlay_render._draw_text)
  const color = item.color || "#ffffff";
  const lines = String(item.text ?? "").split("\n");
  const bold = !!item.bold; // Overlay ▸ Labels… "bold": a bold face is ~8 % wider than the regular one
  const w = Math.max(0, ...lines.map((l) => measureText(l, fs) * (bold ? 1.08 : 1)));
  const pad = 2; // the backend pads the background box by 2 image px
  const lineH = fs * 1.22;
  return (
    <g>
      {item.background && <rect x={item.x - pad} y={item.y - pad} width={w + 2 * pad} height={lines.length * lineH + 2 * pad} fill={boxBehind(color)} rx={1} />}
      <text x={item.x} y={item.y + fs * 0.92} fontSize={fs} fill={color} fontWeight={bold ? "bold" : undefined} style={item.background ? undefined : { ...TEXT_HALO, strokeWidth: 2.5 / scale }}>
        {lines.map((l, i) => <tspan key={i} x={item.x} dy={i === 0 ? 0 : lineH}>{l}</tspan>)}
      </text>
    </g>
  );
}

// ------------------------------------------------------------------ arrow
export function arrowGeometry(x1: number, y1: number, x2: number, y2: number, head: number) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const L = Math.hypot(dx, dy) || 1e-6;
  const ux = dx / L;
  const uy = dy / L;
  const h = Math.min(head, L);
  const bx = x2 - ux * h;
  const by = y2 - uy * h;
  const hw = h * 0.42;
  return {
    shaft: { x1, y1, x2: x2 - ux * h * 0.75, y2: y2 - uy * h * 0.75 },
    head: `${x2},${y2} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}`,
    L,
  };
}

export function ArrowShape({ item, scale, dashed }: { item: ArrowItem; scale: number; dashed?: boolean }) {
  const color = item.color || "#ffd54a";
  // image px, like overlay_render._draw_arrow
  const sw = item.stroke_width || 2;
  const head = item.head_size || 12;
  const g = arrowGeometry(item.x1, item.y1, item.x2, item.y2, head);
  const g2 = item.double_headed ? arrowGeometry(item.x2, item.y2, item.x1, item.y1, head) : null;
  return (
    <g>
      <line x1={g2 ? g2.shaft.x2 : g.shaft.x1} y1={g2 ? g2.shaft.y2 : g.shaft.y1} x2={g.shaft.x2} y2={g.shaft.y2} stroke={color} strokeWidth={sw} strokeLinecap="round" strokeDasharray={dashed ? `${4 / scale} ${3 / scale}` : undefined} />
      <polygon points={g.head} fill={color} stroke={color} strokeWidth={sw * 0.4} strokeLinejoin="round" />
      {g2 && <polygon points={g2.head} fill={color} stroke={color} strokeWidth={sw * 0.4} strokeLinejoin="round" />}
    </g>
  );
}

// ------------------------------------------------------------------ image
/** Short key of everything render_rgb() (lumen/core/render.py) reads from the display besides the plane. */
function displayKey(src: ImageSummary): string {
  const d = src.display;
  const s = [d.mode, d.c, ...d.channels.map((ch) => `${ch.lut}|${ch.min}|${ch.max}|${ch.visible ? 1 : 0}|${ch.inverted ? 1 : 0}|${ch.gamma}`)].join(";");
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/**
 * `<image>` of the source image's current rendering (its own LUTs / display range / plane via
 * /api/images/{id}/rgb.png), placed at (x, y) in this image's pixels and scaled by `scale`.
 */
export function ImageOverlay({ item, images }: { item: ImageOverlayItem; images: ImageSummary[] }) {
  const src = images.find((im) => im.id === item.image_id);
  const key = src ? displayKey(src) : null;
  // Display edits (LUT, range, mode, channel visibility …) hit the store at once but reach the server
  // only with setDisplay's debounced POST, and rgb.png renders from the server's copy.  So the URL
  // takes a new key only once flushDisplay() has resolved (= the server holds what the store shows);
  // otherwise it would change once, fetch the pre-edit rendering and never change again.
  const [acked, setAcked] = useState<string | null>(null);
  useEffect(() => {
    if (key === null) return;
    let alive = true;
    void useStore.getState().flushDisplay().then(() => {
      if (alive) setAcked(key);
    });
    return () => {
      alive = false;
    };
  }, [key]);
  if (!src || acked === null) return null;
  const s = Number(item.scale) > 0 ? Number(item.scale) : 1;
  const opacity = item.opacity === undefined ? 1 : Math.max(0, Math.min(1, Number(item.opacity)));
  // t / z are explicit server parameters (rgb_png renders that plane whatever the server's own position
  // is); rev covers pixel edits and v the acknowledged LUT / range / mode / channel edits.  The server
  // ignores unknown params: they only make the URL, and so the browser's image cache entry, change.
  const href = `/api/images/${encodeURIComponent(item.image_id)}/rgb.png?t=${src.display.t}&z=${src.display.z}&rev=${src.rev}&v=${acked}`;
  return <image href={href} x={item.x || 0} y={item.y || 0} width={src.shape.x * s} height={src.shape.y * s} opacity={opacity} preserveAspectRatio="none" />;
}

// ------------------------------------------------------------------ tracks
export type TrackColorBy = "track" | "speed" | "duration" | "displacement" | "id";
export type TrackLabelMode = "none" | "selected" | "all";
/** Above this many tracks the "all" label mode draws no labels (they become unreadable). */
export const TRACK_LABEL_LIMIT = 60;
const TRACK_DEFAULT_COLOR = "#f59e0b";

/**
 * A track's points prepared for drawing and hit-testing: sorted by frame, each point also as an
 * "x,y" string, the whole polyline string and the bounding box.  Cached per overlay item object,
 * so a frame step (new `t`) never re-parses the ~500 × 100 points of a dense field.
 */
export interface PreparedTrack { id: string; name: string; color: string; pts: number[][]; frames: number[]; xy: string[]; whole: string; bbox: [number, number, number, number] }
const prepCache = new WeakMap<TrackItem, PreparedTrack>();

export function prepareTrack(tr: TrackItem): PreparedTrack {
  const hit = prepCache.get(tr);
  if (hit) return hit;
  let pts = (Array.isArray(tr.points) ? tr.points : []).filter((p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
  let frames = pts.map((p, i) => (p.length > 2 && Number.isFinite(p[2]) ? p[2] : i));
  let sorted = true;
  for (let i = 1; i < frames.length; i++) if (frames[i] < frames[i - 1]) { sorted = false; break; }
  if (!sorted) {
    const order = frames.map((_, i) => i).sort((a, b) => frames[a] - frames[b] || a - b);
    pts = order.map((i) => pts[i]);
    frames = order.map((i) => frames[i]);
  }
  const xy = pts.map((p) => p[0] + "," + p[1]);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) {
    if (p[0] < x0) x0 = p[0];
    if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1];
    if (p[1] > y1) y1 = p[1];
  }
  const prep: PreparedTrack = { id: tr.id, name: tr.name || tr.id, color: tr.color || TRACK_DEFAULT_COLOR, pts, frames, xy, whole: xy.join(" "), bbox: [x0, y0, x1, y1] };
  prepCache.set(tr, prep);
  return prep;
}

/** Number of points on or before frame `t` (first index whose frame is > t; frames are sorted). */
function pastCount(frames: number[], t: number): number {
  let lo = 0, hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function segDist(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const L2 = dx * dx + dy * dy || 1e-9;
  const u = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / L2));
  return Math.hypot(px - (x1 + u * dx), py - (y1 + u * dy));
}

/**
 * The id of the track drawn nearest to image point (ix, iy) within `tol` image px, or null.  Only
 * what is on screen counts: past segments, the current spot and (when shown) the future segments;
 * `visible` can exclude tracks that are hidden (e.g. "show only selected").
 */
export function hitTestTracks(tracks: TrackItem[], ix: number, iy: number, tol: number, t: number, showFuture: boolean, visible?: (id: string) => boolean): string | null {
  let best: string | null = null;
  let bestD = tol;
  for (const tr of tracks) {
    if (visible && !visible(tr.id)) continue;
    const p = prepareTrack(tr);
    const b = p.bbox;
    if (!p.pts.length || ix < b[0] - tol || ix > b[2] + tol || iy < b[1] - tol || iy > b[3] + tol) continue;
    const k = pastCount(p.frames, t);
    const n = p.pts.length;
    const upto = showFuture ? n : k; // points that are drawn
    if (upto === 0) continue;
    const pts = p.pts;
    if (upto === 1) {
      const d = Math.hypot(pts[0][0] - ix, pts[0][1] - iy);
      if (d < bestD) { bestD = d; best = tr.id; }
      continue;
    }
    for (let i = 0; i + 1 < upto; i++) {
      const d = segDist(ix, iy, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
      if (d < bestD) { bestD = d; best = tr.id; }
    }
  }
  return best;
}

/** Bounding box (image px) of the given tracks' points, or null when they have none. */
export function tracksBounds(tracks: TrackItem[]): { x: number; y: number; w: number; h: number } | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const tr of tracks) {
    const b = prepareTrack(tr).bbox;
    if (b[0] > b[2]) continue;
    if (b[0] < x0) x0 = b[0];
    if (b[1] < y0) y0 = b[1];
    if (b[2] > x1) x1 = b[2];
    if (b[3] > y1) y1 = b[3];
  }
  if (x0 > x1) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// ---- per-track statistics (Tracks table rows, or derived from the points) and colour mapping
export interface TrackStatRow { id: string; name: string; color: string; n: number; start: number; end: number; duration: number; speed: number; displacement: number; fromTable: boolean }
export interface TrackUnits { length: string; time: string; duration: string }
interface TracksTable { columns: string[]; rows: unknown[][] }

/** Units of the Tracks table values: backend `_track_stats` / `_speed_unit` (calibrated when the image is). */
export function trackUnits(cal: { unit?: string; pixel_width?: number; frame_interval?: number; time_unit?: string } | undefined, table?: TracksTable): TrackUnits {
  const scaled = !!cal && !!cal.unit && !["pixel", "pixels", "px", ""].includes(cal.unit) && (cal.pixel_width ?? 1) > 0;
  const timed = !!cal && !!cal.frame_interval && cal.frame_interval > 0;
  const length = scaled ? cal!.unit! : "px";
  const time = timed ? cal!.time_unit || "" : "frame";
  const durCol = table?.columns.find((c) => /^Duration \(/.test(c) && c !== "Duration (frames)");
  const duration = durCol ? durCol.slice(10, -1) : timed ? cal!.time_unit || "" : "frames";
  return { length, time, duration };
}

/**
 * One row per track: the Tracks table's calibrated statistics when a row with the track's name
 * (and the same spot count) exists, else values derived from the overlay points (px, frames).
 */
export function trackStatRows(tracks: TrackItem[], table?: TracksTable): TrackStatRow[] {
  const cols = table?.columns || [];
  const col = (name: string) => cols.indexOf(name);
  const cLabel = col("Label"), cN = col("N spots"), cStart = col("Start frame"), cEnd = col("End frame"), cSpeed = col("Mean speed"), cDisp = col("Net displacement");
  const durName = cols.find((c) => /^Duration \(/.test(c) && c !== "Duration (frames)");
  const cDur = durName ? col(durName) : col("Duration (frames)");
  const byLabel = new Map<string, unknown[]>();
  if (table && cLabel >= 0) for (const r of table.rows) byLabel.set(String(r[cLabel]), r);
  const num = (r: unknown[], c: number, fallback: number) => {
    if (c < 0) return fallback;
    const v = Number(r[c]);
    return Number.isFinite(v) ? v : fallback;
  };
  return tracks.map((tr) => {
    const p = prepareTrack(tr);
    const n = p.pts.length;
    const start = n ? p.frames[0] : -1;
    const end = n ? p.frames[n - 1] : -1;
    let total = 0;
    for (let i = 1; i < n; i++) total += Math.hypot(p.pts[i][0] - p.pts[i - 1][0], p.pts[i][1] - p.pts[i - 1][1]);
    const dispPx = n > 1 ? Math.hypot(p.pts[n - 1][0] - p.pts[0][0], p.pts[n - 1][1] - p.pts[0][1]) : 0;
    const durF = n ? end - start : 0;
    const derived: TrackStatRow = { id: p.id, name: p.name, color: p.color, n, start, end, duration: durF, speed: durF > 0 ? total / durF : 0, displacement: dispPx, fromTable: false };
    const row = byLabel.get(p.name);
    if (!row || (cN >= 0 && num(row, cN, n) !== n)) return derived;
    return { ...derived, n: num(row, cN, n), start: num(row, cStart, start), end: num(row, cEnd, end), duration: num(row, cDur, durF), speed: num(row, cSpeed, derived.speed), displacement: num(row, cDisp, dispPx), fromTable: true };
  });
}

export interface TrackLegend { title: string; lo: number; hi: number; gradient: string; categorical: boolean }
export interface TrackColoring { colors: Map<string, string> | null; legend: TrackLegend | null }

function hueRamp(u: number): string {
  return `hsl(${Math.round(240 * (1 - u))}, 85%, 55%)`;
}

/**
 * Colour per track id for a colour-by mode (null = each track's own colour).  Values come from the
 * Tracks table rows (or the points), mapped through the Turbo LUT when the server registered it.
 */
export function trackColoring(rows: TrackStatRow[], mode: TrackColorBy, luts: LutTables, units: TrackUnits): TrackColoring {
  if (mode === "track" || !rows.length) return { colors: null, legend: null };
  const colors = new Map<string, string>();
  if (mode === "id") {
    rows.forEach((r, i) => colors.set(r.id, `hsl(${Math.round((i * 137.508) % 360)}, 80%, 60%)`));
    const stops = Array.from({ length: 7 }, (_, i) => `hsl(${Math.round((i * 137.508) % 360)}, 80%, 60%) ${Math.round((i / 6) * 100)}%`);
    return { colors, legend: { title: "Track id", lo: 0, hi: rows.length - 1, gradient: `linear-gradient(to right, ${stops.join(", ")})`, categorical: true } };
  }
  const value = mode === "speed" ? (r: TrackStatRow) => r.speed : mode === "duration" ? (r: TrackStatRow) => r.duration : (r: TrackStatRow) => r.displacement;
  const title = mode === "speed" ? `Mean speed (${units.length}/${units.time})` : mode === "duration" ? `Duration (${units.duration})` : `Net displacement (${units.length})`;
  let lo = Infinity, hi = -Infinity;
  for (const r of rows) {
    const v = value(r);
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo > hi) { lo = 0; hi = 0; }
  const lutName = luts.Turbo ? "Turbo" : luts.Viridis ? "Viridis" : null;
  const table = lutName ? luts[lutName] : null;
  const paint = (u: number) => {
    if (!table) return hueRamp(u);
    const row = table[Math.max(0, Math.min(255, Math.round(u * 255)))];
    return `rgb(${row[0]},${row[1]},${row[2]})`;
  };
  for (const r of rows) {
    const v = value(r);
    colors.set(r.id, paint(!Number.isFinite(v) ? 0 : hi > lo ? (v - lo) / (hi - lo) : 0.5));
  }
  const gradient = table ? lutGradientCss(luts, lutName!) : `linear-gradient(to right, ${Array.from({ length: 7 }, (_, i) => `${hueRamp(i / 6)} ${Math.round((i / 6) * 100)}%`).join(", ")})`;
  return { colors, legend: { title, lo, hi, gradient, categorical: false } };
}

// ---- rendering
/** 0 = dimmed (another track is selected), 1 = normal, 2 = highlighted (selected or hovered). */
type Emphasis = 0 | 1 | 2;

interface GlyphProps {
  prep: PreparedTrack;
  /** Points on or before the current frame (the past / future split). */
  k: number;
  cur: number;
  /** Drawn point range [lo, hi): the whole track in "Full tracks", a window around the current frame in the local-tails / fade modes. */
  lo: number;
  hi: number;
  scale: number;
  color: string;
  emphasis: Emphasis;
  label: boolean;
  showFuture: boolean;
  /** "Fade" display mode: the tail fades out with age (drawn as a few chunks of increasing opacity). */
  fade: boolean;
}

/** Number of opacity steps a fading tail is drawn with. */
const FADE_STEPS = 4;

/**
 * One track.  Memoised so that a frame step only re-renders the tracks whose past/future split
 * actually moved, and the future part is the whole (never changing) polyline drawn dim underneath
 * the opaque past part, so one `points` string per changed track is rebuilt per frame.  Display
 * modes other than "Full tracks" narrow the drawn range to [lo, hi) (tracking.set_display).
 */
const TrackGlyph = React.memo(function TrackGlyph({ prep, k, cur, lo, hi, scale, color, emphasis, label, showFuture, fade }: GlyphProps) {
  const n = prep.xy.length;
  const past = useMemo(() => (lo === 0 && k >= n ? prep.whole : k - lo > 1 ? prep.xy.slice(lo, k).join(" ") : ""), [prep, k, lo, n]);
  const future = useMemo(() => (!showFuture || k >= hi || hi - lo < 2 ? "" : lo === 0 && hi >= n ? prep.whole : prep.xy.slice(lo, hi).join(" ")), [prep, k, lo, hi, n, showFuture]);
  // fade mode: the tail's segments split into FADE_STEPS chunks (sharing their boundary points), oldest drawn lightest
  const fadeParts = useMemo(() => {
    const segs = k - lo - 1;
    if (!fade || segs < 2) return null;
    const steps = Math.min(FADE_STEPS, segs);
    const parts: { points: string; opacity: number }[] = [];
    for (let i = 0; i < steps; i++) {
      const a = lo + Math.floor((segs * i) / steps);
      const b = lo + Math.floor((segs * (i + 1)) / steps) + 1;
      parts.push({ points: prep.xy.slice(a, b).join(" "), opacity: (i + 1) / steps });
    }
    return parts;
  }, [prep, k, lo, fade]);
  const strong = emphasis === 2;
  const sw = (strong ? 3 : 1.5) / scale;
  const r = (strong ? 4.5 : 3.5) / scale;
  const fs = 10 / scale;
  const op = emphasis === 0 ? 0.3 : 1;
  const last = k > lo ? prep.pts[k - 1] : null;
  const curPt = cur >= 0 ? prep.pts[cur] : null;
  return (
    <g data-track={prep.id}>
      {strong && past && <polyline points={past} fill="none" stroke="rgba(0,0,0,0.65)" strokeWidth={sw + 2.5 / scale} strokeLinejoin="round" strokeLinecap="round" />}
      {future && <polyline points={future} fill="none" stroke={color} strokeWidth={sw} opacity={0.25 * op} strokeLinejoin="round" strokeLinecap="round" />}
      {fadeParts
        ? fadeParts.map((p, i) => <polyline key={i} points={p.points} fill="none" stroke={color} strokeWidth={sw} opacity={p.opacity * op} strokeLinejoin="round" strokeLinecap="round" />)
        : past && <polyline points={past} fill="none" stroke={color} strokeWidth={sw} opacity={op} strokeLinejoin="round" strokeLinecap="round" />}
      {curPt && <circle cx={curPt[0]} cy={curPt[1]} r={r} fill={color} stroke={strong ? "#ffffff" : "rgba(0,0,0,0.6)"} strokeWidth={(strong ? 1.2 : 0.9) / scale} opacity={op} />}
      {label && last && <text x={last[0] + r * 1.4} y={last[1] - r * 1.2} fontSize={fs} fontWeight={strong ? 600 : 400} fill={color} opacity={op} style={{ ...TEXT_HALO, strokeWidth: 2 / scale }}>{prep.name}</text>}
    </g>
  );
});

export interface TracksLayerProps {
  tracks: TrackItem[];
  t: number;
  scale: number;
  showFuture: boolean;
  labelMode: TrackLabelMode;
  /** Selected track ids: highlighted, every other track is dimmed. */
  selected: string[];
  hovered: string | null;
  /** Colour per track id (colour-by mode), null = the tracks' own colours. */
  colors: Map<string, string> | null;
  /** Draw only the selected tracks (when any is selected). */
  onlySelected: boolean;
  /** tracking.set_display mode: "Full tracks" (default), "Local tails (backward)", "Local tails (both ways)", "Up to current frame" or "Fade". */
  displayMode?: string;
  /** Frames drawn around the current frame in the local-tails / fade modes (default 10). */
  tailLength?: number;
}

const NO_IDS: string[] = [];

/**
 * All track overlays of an image. Memoised on the overlay array + current frame so pointer moves
 * and other viewport re-renders do not touch ~500 × 100 points; each track is itself memoised.
 * Tracks are display-only (the backend does not burn them in), so they keep constant screen sizes.
 */
export const TracksLayer = React.memo(function TracksLayer({ tracks, t, scale, showFuture, labelMode, selected = NO_IDS, hovered, colors, onlySelected, displayMode, tailLength }: TracksLayerProps) {
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  if (!tracks.length) return null;
  const anySel = selectedSet.size > 0;
  const allLabels = labelMode === "all" && tracks.length <= TRACK_LABEL_LIMIT;
  // display mode (tracking.set_display): which point range of each track is drawn around frame t
  const tail = Math.max(1, tailLength ?? 10);
  const mode = displayMode || "Full tracks";
  const local = mode.startsWith("Local tails") || mode === "Fade";
  const future = showFuture && (mode === "Full tracks" || mode === "Local tails (both ways)");
  const fade = mode === "Fade";
  const nodes: React.ReactNode[] = [];
  const selNodes: React.ReactNode[] = [];
  let hoveredNode: React.ReactNode = null;
  for (const tr of tracks) {
    const isSel = selectedSet.has(tr.id);
    if (onlySelected && anySel && !isSel) continue;
    const isHov = hovered === tr.id;
    const prep = prepareTrack(tr);
    if (!prep.pts.length) continue;
    const kAll = pastCount(prep.frames, t);
    const lo = local ? pastCount(prep.frames, t - tail - 1) : 0;
    const hi = mode === "Local tails (both ways)" ? pastCount(prep.frames, t + tail) : mode === "Local tails (backward)" || mode === "Up to current frame" ? kAll : prep.pts.length;
    const k = Math.min(kAll, hi);
    const cur = k > 0 && prep.frames[k - 1] === t ? k - 1 : -1;
    const emphasis: Emphasis = isSel || isHov ? 2 : anySel ? 0 : 1;
    const label = allLabels || (labelMode === "selected" && (isSel || isHov));
    const node = <TrackGlyph key={tr.id} prep={prep} k={k} cur={cur} lo={lo} hi={hi} scale={scale} color={colors?.get(tr.id) || prep.color} emphasis={emphasis} label={label} showFuture={future} fade={fade} />;
    // selected / hovered tracks go last so they are drawn on top of the dimmed field
    if (isHov) hoveredNode = node;
    else if (isSel) selNodes.push(node);
    else nodes.push(node);
  }
  for (const n of selNodes) nodes.push(n);
  if (hoveredNode) nodes.push(hoveredNode);
  return <g className="tracks">{nodes}</g>;
});

// ------------------------------------------------------------------ image-corner anchoring
/** Mirrors overlay_render.anchor(): position of a w×h box in a W×H image (all image px). */
export function anchor(position: string | undefined, W: number, H: number, w: number, h: number, margin?: number): { x: number; y: number } {
  const m = margin === undefined ? Math.max(Math.floor(Math.min(W, H) * 0.03), 6) : margin;
  const pos = position || "Upper Right";
  if (pos === "Center") return { x: Math.floor((W - w) / 2), y: Math.floor((H - h) / 2) };
  const x = pos.includes("Right") ? W - m - w : m;
  const y = pos.includes("Lower") ? H - m - h : m;
  return { x, y };
}

function fmtNum(v: number, decimals: number): string {
  if (!Number.isFinite(v)) return "";
  return decimals >= 0 ? v.toFixed(decimals) : String(v);
}

// ------------------------------------------------------------------ calibration bar
/** Layout mirrors overlay_render._draw_calibration_bar: unit on top, bar below it, tick labels to the right. */
export function CalibrationBar({ item, img, luts, index, scale }: { item: CalibrationBarItem; img: ImageSummary; luts: LutTables; index: number; scale: number }) {
  const W = item.width || 20;
  const H = item.height || 128;
  const fs = item.font_size || 12;
  const color = item.color || "#ffffff";
  const decimals = Math.max(0, item.decimals ?? 0);
  const ch = img.display.channels[Math.min(img.display.c, img.display.channels.length - 1)];
  const lutName = !item.lut || item.lut === "Current" ? ch?.lut || "Grays" : item.lut;
  const lo = item.min ?? ch?.min ?? 0;
  const hi = item.max ?? ch?.max ?? 255;
  const unit = item.unit ?? img.calibration.value_unit ?? "";
  const table = luts[lutName];
  const nTicks = Math.max(2, Math.min(20, item.ticks || 5));
  // k = 0 is the bottom (lo), k = n-1 the top (hi), like the backend
  const ticks = Array.from({ length: nTicks }, (_, k) => ({ k, label: fmtNum(lo + ((hi - lo) * k) / (nTicks - 1), decimals) }));
  const labelW = Math.max(...ticks.map((t) => measureText(t.label, fs))) + 6;
  const unitH = unit ? fs * 0.8 + 4 : 0;
  const totalW = W + labelW + 8;
  const totalH = H + unitH + 8;
  const { x: x0, y: y0 } = anchor(item.position, img.shape.x, img.shape.y, totalW, totalH);
  const bx = x0 + 4;
  const by = y0 + 4 + unitH;
  const gradId = `lumen-calbar-${item.id || index}`;
  const stops: React.ReactNode[] = [];
  const n = 16;
  for (let i = 0; i <= n; i++) {
    const k = Math.round((1 - i / n) * 255);
    // the stored flag, not the live channel: the burnt-in bar cannot follow later channel / Invert LUT changes either
    const idx = item.inverted ? 255 - k : k;
    const row = table ? table[idx] : [idx, idx, idx];
    stops.push(<stop key={i} offset={`${(i / n) * 100}%`} stopColor={`rgb(${row[0]},${row[1]},${row[2]})`} />);
  }
  const thin = 1 / scale; // cosmetic outline / tick width
  return (
    <g className="calibration-bar">
      <defs><linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient></defs>
      {item.fill !== false && <rect x={x0 - 2} y={y0 - 2} width={totalW + 4} height={totalH + 4} fill={boxBehind(color)} />}
      <rect x={bx} y={by} width={W} height={H} fill={`url(#${gradId})`} stroke={color} strokeWidth={thin} />
      {ticks.map((t) => {
        const y = by + H - 1 - Math.round(((H - 1) * t.k) / (nTicks - 1));
        return (
          <g key={t.k}>
            <line x1={bx + W} x2={bx + W + 3} y1={y} y2={y} stroke={color} strokeWidth={thin} />
            <text x={bx + W + 5} y={y + fs * 0.35} fontSize={fs} fill={color}>{t.label}</text>
          </g>
        );
      })}
      {unit && <text x={bx} y={y0 + 2 + fs * 0.8} fontSize={fs} fill={color}>{unit}</text>}
    </g>
  );
}

// ------------------------------------------------------------------ time stamp
const TS_NAMES = new Set(["time", "frame", "unit", "slice", "z", "t", "label"]);

/** Python's zero/space padding of a numeric string to `width` (a "0" fill goes after the sign, like format(-5, "04d")). */
function padNumber(s: string, width: number, zero: boolean): string {
  if (!zero) return s.padStart(width, " ");
  return s.startsWith("-") ? "-" + s.slice(1).padStart(width - 1, "0") : s.padStart(width, "0");
}

/** Python's format(v, ".Nf"): toFixed rounds an exact tie away from zero, Python to the even digit (2.5 → "2", 0.125 → "0.12"). */
function toFixedPy(v: number, digits: number): string {
  // an exact tie means v = odd / 2^(digits+1); scaling by a power of two is exact (v * 10^digits is not: 2.675 * 100 === 267.5)
  const m = Math.abs(v) * 2 ** (digits + 1);
  if (Number.isInteger(m) && m % 2 === 1) {
    const lo = Math.floor(Math.abs(v) * 10 ** digits);
    const s = ((lo % 2 === 0 ? lo : lo + 1) / 10 ** digits).toFixed(digits);
    return v < 0 ? "-" + s : s;
  }
  return v.toFixed(digits);
}

/**
 * Tiny Python-style formatter mirroring str.format as used by overlay_render.format_time_stamp:
 * supports {time}, {time:.Nf}, {label:0W.Nf}, {frame}, {frame:0Nd}, {label}, {unit}, {slice}, {z}, {t}.
 * Like the backend, a format with an unknown placeholder is returned unchanged.
 */
export function formatTimeStamp(fmt: string, vars: { time: number; frame: number; unit: string; z: number; label?: number }): string {
  const values: Record<string, number | string> = { time: vars.time, unit: vars.unit, frame: vars.frame, slice: vars.z, t: vars.frame, z: vars.z, label: vars.label ?? vars.frame };
  const re = /\{([^{}:]*)(?::([^{}]*))?\}/g;
  for (const m of fmt.matchAll(re)) if (!TS_NAMES.has(m[1])) return fmt;
  return fmt.replace(re, (_m, name: string, spec: string | undefined) => {
    const v = values[name];
    if (typeof v === "string") return v;
    if (spec) {
      const f = /^(0?)(\d*)\.(\d+)f$/.exec(spec);
      if (f) {
        const s = toFixedPy(v, Number(f[3]));
        return f[2] ? padNumber(s, Number(f[2]), !!f[1]) : s;
      }
      const d = /^(0?)(\d+)d$/.exec(spec);
      if (d) return padNumber(String(Math.round(v)), Number(d[2]), !!d[1]);
      const e = /^\.(\d+)e$/.exec(spec);
      if (e) return v.toExponential(Number(e[1]));
    }
    // Python str(): the (float) time keeps a trailing ".0" when integral, frame numbers are ints
    if (name === "time") return Number.isInteger(v) ? v.toFixed(1) : String(v);
    return String(v);
  });
}

/** Mirrors lumen/core/overlay_render.format_time_stamp (what Burn In / Flatten write). */
export function timeStampText(item: TimeStampItem, img: ImageSummary): string {
  const cal = img.calibration;
  const t = img.display.t;
  const z = img.display.z;
  // backend: step = interval or cal.frame_interval or 1.0; unit = time_unit if (interval or frame_interval) else "frame"
  const calibrated = !!item.interval || !!cal.frame_interval;
  const step = item.interval || cal.frame_interval || 1;
  const unit = calibrated ? cal.time_unit || "" : "frame";
  // backend: o.get("format", "t = {time:.1f} {unit}") — an empty string renders empty
  const fmt = item.format ?? "t = {time:.1f} {unit}";
  // time = start_time + (t - start_frame) * step: start_frame only shifts the time origin; {frame}/{t}/{slice}/{z} are the
  // 1-based absolute indices; {label} = label_start + n * label_interval with n the frame (time series) or slice index
  const time = (item.start_time || 0) + (t - (item.start_frame || 0)) * step;
  const n = img.shape.t > 1 ? t : z;
  const label = (item.label_start ?? 1) + n * (item.label_interval ?? 1);
  return formatTimeStamp(fmt, { time, frame: t + 1, unit, z: z + 1, label });
}

export function TimeStamp({ item, img, scale }: { item: TimeStampItem; img: ImageSummary; scale: number }) {
  const fs = item.font_size || 14; // image px (burn_overlay default)
  const color = item.color || "#ffffff";
  const text = timeStampText(item, img);
  const w = measureText(text, fs);
  const { x, y } = anchor(item.position || "Upper Left", img.shape.x, img.shape.y, w, fs);
  return (
    <text className="time-stamp" x={x} y={y + fs * 0.9} fontSize={fs} fill={color} fontWeight={600} style={{ ...TEXT_HALO, strokeWidth: 3 / scale }}>{text}</text>
  );
}
