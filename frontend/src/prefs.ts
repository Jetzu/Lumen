/**
 * User preferences (Edit ▸ Options ▸ Preferences…, ⌘,).  They live in `session.settings` on the
 * server (GET /api/settings, PUT /api/settings) so that backend commands (draw, paste, save …)
 * and the viewport read the same values.  The defaults mirror lumen/server/extra/settings.py.
 */
export interface Prefs {
  roi_color: string;
  roi_stroke_width: number;
  font_size: number;
  font_family: string;
  foreground: number;
  background: number;
  jpeg_quality: number;
  default_save_format: string;
  point_size: number;
  point_color: string;
  arrow_style: string;
  arrow_head_size: number;
  arrow_width: number;
  plot_y_min: number;
  plot_y_max: number;
  plot_interpolate: boolean;
  interpolate_zoom: boolean;
  antialiased_text: boolean;
  paste_mode: string;
  /** Rounded Rectangle tool corner radius (image px). */
  rounded_rect_radius: number;
  /** Undo/redo history budget (MB) across all images. */
  undo_budget_mb: number;
  /** Worker threads for the backend (0 = automatic). */
  max_threads: number;
  /** HTTP proxy "host:port" for downloads ("" = none). */
  proxy: string;
  /** Comma-separated hosts that bypass the proxy. */
  no_proxy: string;
  dicom_open_as_32bit: boolean;
  dicom_ignore_rescale: boolean;
  dicom_calibrate_images: boolean;
  dicom_window_level_on_open: boolean;
  /** Ask before closing an image that has unsaved changes (File ▸ Close, ⌘W, tab ×). */
  confirm_close_modified: boolean;
  /** Inspector / sheet spacing: "comfortable" (default) or "compact"; applied as `<html data-density>`. */
  ui_density: string;
  /** Shared plane cache of virtual stacks (MB; lumen.core.lazy, 16–1000000). */
  virtual_cache_mb: number;
  /**
   * Decoded size (MB) above which File ▸ Open opens a file as a virtual stack; 0 = never.  The server's own default
   * is half of the physical RAM, which the client cannot know up front: 0 here is only the placeholder until
   * GET /api/settings has answered (normalizePrefs copies the server's value), and Reset to defaults recomputes
   * it from `state.memory_limit_bytes` (see PreferencesSheet) instead of sending this 0.
   */
  auto_virtual_mb: number;
}

export const PREF_DEFAULTS: Prefs = {
  roi_color: "#ffd54a",
  roi_stroke_width: 1.5,
  font_size: 14,
  font_family: "",
  foreground: 255,
  background: 0,
  jpeg_quality: 85,
  default_save_format: "tif",
  point_size: 4,
  point_color: "#ffd54a",
  arrow_style: "Filled",
  arrow_head_size: 12,
  arrow_width: 2,
  plot_y_min: 0,
  plot_y_max: 0,
  plot_interpolate: true,
  interpolate_zoom: false,
  antialiased_text: true,
  paste_mode: "Copy",
  rounded_rect_radius: 20,
  undo_budget_mb: 1500,
  max_threads: 0,
  proxy: "",
  no_proxy: "",
  dicom_open_as_32bit: false,
  dicom_ignore_rescale: false,
  dicom_calibrate_images: true,
  dicom_window_level_on_open: true,
  confirm_close_modified: true,
  ui_density: "comfortable",
  virtual_cache_mb: 512,
  auto_virtual_mb: 0,
};

/** Server default of `auto_virtual_mb` (lumen.core.lazy.default_auto_virtual_mb: half of the physical RAM, in MB). */
export function defaultAutoVirtualMb(physicalRamBytes: number | undefined | null): number | null {
  if (!physicalRamBytes || !Number.isFinite(physicalRamBytes) || physicalRamBytes <= 0) return null;
  return Math.max(1, Math.floor(physicalRamBytes / 2 / 1048576));
}

export const DENSITIES: { id: string; label: string }[] = [
  { id: "comfortable", label: "Comfortable" },
  { id: "compact", label: "Compact (tighter inspector and sheets)" },
];
/** localStorage mirror of `ui_density` so the first paint already uses it (the server copy is authoritative). */
export const DENSITY_KEY = "lumen.density";

/** Apply a density preference to the document (`<html data-density="compact">`; comfortable = attribute removed). */
export function applyDensity(density: string | undefined | null): void {
  const d = density === "compact" ? "compact" : "comfortable";
  if (typeof document === "undefined") return;
  if (d === "compact") document.documentElement.dataset.density = d;
  else delete document.documentElement.dataset.density;
  try {
    localStorage.setItem(DENSITY_KEY, d);
  } catch {
    /* private mode */
  }
}

export const PASTE_MODES = ["Copy", "Blend", "Average", "Difference", "Transparent-white", "Transparent-zero", "AND", "OR", "XOR", "Add", "Subtract", "Multiply", "Divide", "Min", "Max"];
export const ARROW_STYLES = ["Filled", "Notched", "Open", "Headless", "Bar"];
export const SAVE_FORMATS: { id: string; label: string }[] = [
  { id: "tif", label: ".tif (ImageJ hyperstack)" },
  { id: "ome.tif", label: ".ome.tif (OME-TIFF)" },
  { id: "png", label: ".png (current view, RGB)" },
  { id: "jpg", label: ".jpg (current view)" },
  { id: "gif", label: ".gif (animated)" },
  { id: "npy", label: ".npy (numpy)" },
];
export const FONT_FAMILIES: { id: string; label: string }[] = [
  { id: "", label: "System default" },
  { id: "Helvetica, Arial, sans-serif", label: "Sans-serif (Helvetica/Arial)" },
  { id: "Georgia, 'Times New Roman', serif", label: "Serif (Georgia/Times)" },
  { id: "Menlo, Consolas, monospace", label: "Monospace (Menlo/Consolas)" },
];

/** Coerce whatever the server holds into a complete, well-typed Prefs object. */
export function normalizePrefs(raw: Record<string, unknown> | null | undefined): Prefs {
  const out: Prefs = { ...PREF_DEFAULTS };
  if (!raw) return out;
  const o = out as unknown as Record<string, unknown>;
  for (const key of Object.keys(PREF_DEFAULTS) as (keyof Prefs)[]) {
    const v = raw[key];
    const def = PREF_DEFAULTS[key];
    if (v === undefined || v === null) continue;
    if (typeof def === "boolean") o[key] = !!v;
    else if (typeof def === "number") {
      const n = typeof v === "number" ? v : parseFloat(String(v));
      if (Number.isFinite(n)) o[key] = n;
    } else o[key] = String(v);
  }
  return out;
}

/** `<input type="color">` only accepts #rrggbb: expand #rgb and fall back to the default. */
export function toHex6(color: string, fallback = "#ffd54a"): string {
  const s = (color || "").trim();
  if (/^#[0-9a-fA-F]{6}$/.test(s)) return s.toLowerCase();
  if (/^#[0-9a-fA-F]{8}$/.test(s)) return s.slice(0, 7).toLowerCase();
  if (/^#[0-9a-fA-F]{3}$/.test(s)) return ("#" + s.slice(1).split("").map((c) => c + c).join("")).toLowerCase();
  return fallback;
}

/** Map a saved format name onto the ids of the save dialog's format list. */
export function normalizeFormat(fmt: string | undefined | null): string {
  const f = (fmt || "tif").toLowerCase().replace(/^\./, "");
  const alias: Record<string, string> = { tiff: "tif", "ome.tiff": "ome.tif", jpeg: "jpg" };
  const id = alias[f] || f;
  return SAVE_FORMATS.some((x) => x.id === id) ? id : "tif";
}
