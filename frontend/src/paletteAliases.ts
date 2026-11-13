/**
 * Fiji names as searchable aliases of Lumen commands (GUI fix round 2: E7-10 / E5-3).  The in-app parity map
 * (docs/FIJI_PARITY.md, served by `GET /api/docs/FIJI_PARITY`) lists every Fiji command with the Lumen command ids
 * that cover it, and its "ImageJ macro compatibility layer" table lists the `run("…")` names.  Both are parsed into
 * `op id → [Fiji labels]` the first time the palette opens; `STATIC_ALIASES` covers the most-asked names should the
 * document be unavailable (older server, offline build).
 */
import { api } from "./api";

export type AliasMap = Record<string, string[]>;

export const STATIC_ALIASES: AliasMap = {
  "process.clahe": ["Enhance Local Contrast (CLAHE)", "CLAHE"],
  "process.subtract_background": ["Subtract Background"],
  "process.bleach_correction": ["Bleach Correction"],
  "filters.gaussian": ["Gaussian Blur"],
  "filters.median": ["Median"],
  "process.enhance_contrast": ["Enhance Contrast"],
  "threshold.compute": ["Threshold", "Auto Threshold"],
  "analyze.particles": ["Analyze Particles"],
  "analyze.measure": ["Measure"],
  "stacks.z_project": ["Z Project", "Maximum Intensity Projection"],
  "color.lut": ["Lookup Tables", "LUT"],
  "type.apply_lut": ["Apply LUT"],
  "file.sample": ["Open Samples"],
  "file.open_imagej_sample": ["Open Samples (ImageJ)"],
  "image.crop": ["Crop"],
  "image.duplicate": ["Duplicate"],
  "file.import_lut": ["Import LUT", "LUT..."],
  "file.export_lut": ["Export LUT", "Save As LUT"],
};

let cached: AliasMap | null = null;
let pending: Promise<AliasMap> | null = null;
const listeners = new Set<(m: AliasMap) => void>();

/** Strip markdown emphasis, the "*Fiji* " prefix, submenu arrows and ellipses from a parity-row label. */
function cleanLabel(s: string): string[] {
  const t = s.replace(/\*([^*]*)\*/g, "$1").replace(/`/g, "").replace(/\s*\((\d+ [^)]*)\)\s*$/, "").trim();
  if (!t) return [];
  const out = new Set<string>();
  if (t.includes('"')) {
    // the macro table quotes the run("…") names: "Enhance Local Contrast (CLAHE)" / "CLAHE"
    for (const m of t.matchAll(/"([^"]+)"/g)) out.add(m[1].replace(/(…|\.\.\.)\s*$/, "").trim());
  } else {
    const parts = t.split(/\s*▸\s*/).map((p) => p.replace(/(…|\.\.\.)\s*$/, "").trim()).filter(Boolean);
    if (parts.length) out.add(parts[parts.length - 1]); // "Enhance Local Contrast (CLAHE)"
    if (parts.length > 1) out.add(parts.join(" "));     // "Image Adjust Brightness/Contrast"
  }
  return Array.from(out).filter((x) => x.length > 1 && x.length < 80);
}

/** Parse the parity markdown: every table row whose second column names Lumen ids in backticks. */
export function parseParityAliases(md: string): AliasMap {
  const out: AliasMap = {};
  for (const line of md.split("\n")) {
    if (!line.startsWith("|")) continue;
    const cells = line.split("|").slice(1, -1).map((c) => c.trim());
    if (cells.length < 2) continue;
    const [fiji, lumen] = cells;
    if (!fiji || /^-+$/.test(fiji) || /^Fiji command$/i.test(fiji) || /^Fiji macro/i.test(fiji)) continue;
    const ids = Array.from(lumen.matchAll(/`([a-z0-9_]+\.[a-z0-9_.]+)(?:\([^)]*\))?`/g)).map((m) => m[1]);
    if (!ids.length) continue;
    const labels = cleanLabel(fiji);
    if (!labels.length) continue;
    for (const id of ids.slice(0, 4)) {
      const list = out[id] || (out[id] = []);
      for (const l of labels) if (!list.includes(l)) list.push(l);
    }
  }
  return out;
}

function merge(a: AliasMap, b: AliasMap): AliasMap {
  const out: AliasMap = {};
  for (const m of [a, b]) for (const [id, labels] of Object.entries(m)) {
    const list = out[id] || (out[id] = []);
    for (const l of labels) if (!list.includes(l)) list.push(l);
  }
  return out;
}

/** The alias map, loading it in the background on first use; `onReady` is called once when the document has been parsed. */
export function getAliases(onReady?: (m: AliasMap) => void): AliasMap {
  if (cached) return cached;
  if (onReady) listeners.add(onReady);
  if (!pending) {
    pending = api.doc("FIJI_PARITY").then((md) => merge(STATIC_ALIASES, parseParityAliases(md))).catch(() => STATIC_ALIASES).then((m) => {
      cached = m;
      for (const l of listeners) l(m);
      listeners.clear();
      return m;
    });
  }
  return STATIC_ALIASES;
}
