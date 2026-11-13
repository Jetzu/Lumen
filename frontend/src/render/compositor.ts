import type { ChannelDisplay, Display, Threshold, TypedArray } from "../types";

export type LutTables = Record<string, number[][]>;

// Keyed by the tables object first: the store replaces `luts` whenever /api/luts is refetched (a LUT
// registered, replaced under an existing name, or removed), so a stale 256×3 table is never served
// for the new object, and the old object's entries are garbage-collected with it.
const tableCache = new WeakMap<LutTables, Map<string, Uint8Array>>();

/** 256×3 table for a LUT (optionally inverted). */
function lutTable(luts: LutTables, name: string, inverted: boolean): Uint8Array {
  let per = tableCache.get(luts);
  if (!per) {
    per = new Map();
    tableCache.set(luts, per);
  }
  const key = `${name}|${inverted ? 1 : 0}`;
  let t = per.get(key);
  if (t) return t;
  const src = luts[name] || luts["Grays"] || fallbackGray();
  t = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const row = src[inverted ? 255 - i : i];
    t[i * 3] = row[0];
    t[i * 3 + 1] = row[1];
    t[i * 3 + 2] = row[2];
  }
  per.set(key, t);
  return t;
}

function fallbackGray(): number[][] {
  const g: number[][] = [];
  for (let i = 0; i < 256; i++) g.push([i, i, i]);
  return g;
}

/** Map one channel into an index array (0..255) according to its display range. */
function channelIndex(data: TypedArray, ch: ChannelDisplay, out: Uint8Array): void {
  const n = data.length;
  const lo = ch.min;
  const span = ch.max - ch.min || 1e-9;
  const scale = 255 / span;
  const gamma = ch.gamma && ch.gamma !== 1 ? 1 / ch.gamma : 0;
  if (data instanceof Uint8Array || data instanceof Uint16Array) {
    const maxv = data instanceof Uint8Array ? 256 : 65536;
    const table = new Uint8Array(maxv);
    for (let v = 0; v < maxv; v++) {
      let f = (v - lo) * scale;
      if (f < 0) f = 0;
      else if (f > 255) f = 255;
      if (gamma) f = Math.pow(f / 255, gamma) * 255;
      table[v] = f + 0.5;
    }
    for (let i = 0; i < n; i++) out[i] = table[data[i]];
  } else {
    for (let i = 0; i < n; i++) {
      let f = (data[i] - lo) * scale;
      if (!(f >= 0)) f = 0; // NaN → 0
      else if (f > 255) f = 255;
      if (gamma) f = Math.pow(f / 255, gamma) * 255;
      out[i] = f + 0.5;
    }
  }
}

export interface CompositeOptions {
  threshold?: Threshold | null;
  thresholdChannel?: number;
}

/**
 * Composite planes into RGBA.  Returns a Uint8ClampedArray of length w*h*4.
 */
export function composite(channels: TypedArray[], w: number, h: number, display: Display, rgb: boolean, luts: LutTables, opts: CompositeOptions = {}, out?: Uint8ClampedArray<ArrayBuffer>): Uint8ClampedArray<ArrayBuffer> {
  const n = w * h;
  const rgba: Uint8ClampedArray<ArrayBuffer> = out && out.length === n * 4 ? out : new Uint8ClampedArray(new ArrayBuffer(n * 4));
  const idx = new Uint8Array(n);
  const acc = new Uint16Array(n * 3);
  let anyVisible = false;

  const visibleChannels: number[] = [];
  if (rgb) {
    for (let c = 0; c < Math.min(3, channels.length); c++) if (display.channels[c]?.visible !== false) visibleChannels.push(c);
  } else if (display.mode === "composite" && channels.length > 1) {
    for (let c = 0; c < channels.length; c++) if (display.channels[c]?.visible) visibleChannels.push(c);
  } else {
    visibleChannels.push(Math.min(display.c, channels.length - 1));
  }

  for (const c of visibleChannels) {
    const ch = display.channels[c] || { name: "", lut: "Grays", min: 0, max: 255, visible: true, inverted: false, gamma: 1 };
    channelIndex(channels[c], ch, idx);
    anyVisible = true;
    if (rgb) {
      for (let i = 0; i < n; i++) acc[i * 3 + c] += idx[i];
      continue;
    }
    const lutName = display.mode === "grayscale" ? "Grays" : ch.lut;
    const table = lutTable(luts, lutName, !!ch.inverted);
    for (let i = 0; i < n; i++) {
      const k = idx[i] * 3;
      const j = i * 3;
      acc[j] += table[k];
      acc[j + 1] += table[k + 1];
      acc[j + 2] += table[k + 2];
    }
  }

  for (let i = 0; i < n; i++) {
    const j = i * 3;
    const o = i * 4;
    rgba[o] = acc[j] > 255 ? 255 : acc[j];
    rgba[o + 1] = acc[j + 1] > 255 ? 255 : acc[j + 1];
    rgba[o + 2] = acc[j + 2] > 255 ? 255 : acc[j + 2];
    rgba[o + 3] = 255;
  }
  if (!anyVisible) rgba.fill(0);

  const thr = opts.threshold;
  if (thr) {
    const c = Math.min(opts.thresholdChannel ?? display.c, channels.length - 1);
    const data = channels[c];
    const lo = thr.lower;
    const hi = thr.upper;
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v >= lo && v <= hi) {
        const o = i * 4;
        rgba[o] = 255;
        rgba[o + 1] = 40;
        rgba[o + 2] = 40;
      }
    }
  }
  return rgba;
}

/** 256-bin histogram of a channel between min and max (for display sliders). */
export function histogram(data: TypedArray, lo: number, hi: number, bins = 256): { counts: Float32Array; lo: number; hi: number } {
  const counts = new Float32Array(bins);
  const span = hi - lo || 1e-9;
  const n = data.length;
  const step = n > 1_000_000 ? Math.ceil(n / 1_000_000) : 1;
  for (let i = 0; i < n; i += step) {
    const v = data[i];
    if (!(v >= lo)) {
      if (v < lo) counts[0]++;
      continue;
    }
    let b = ((v - lo) / span) * bins;
    if (b >= bins) b = bins - 1;
    counts[b | 0]++;
  }
  return { counts, lo, hi };
}

export function dataRange(data: TypedArray): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const n = data.length;
  const step = n > 2_000_000 ? Math.ceil(n / 2_000_000) : 1;
  for (let i = 0; i < n; i += step) {
    const v = data[i];
    // ±Infinity (and NaN) take no part, as in ImageJ's FloatProcessor.findMinAndMax and the backend's
    // value_range(): a single inf pixel must not collapse the display range to [lo, Infinity] (all black)
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [0, 1]; // no finite sample at all
  return [lo, hi];
}

export function lutGradientCss(luts: LutTables, name: string, inverted = false): string {
  const src = luts[name];
  if (!src) return "linear-gradient(to right, #000, #fff)";
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round((i / 8) * 255);
    const row = src[inverted ? 255 - k : k];
    stops.push(`rgb(${row[0]},${row[1]},${row[2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export function lutColor(luts: LutTables, name: string): string {
  const src = luts[name];
  if (!src) return "#fff";
  const row = src[255];
  return `rgb(${row[0]},${row[1]},${row[2]})`;
}
