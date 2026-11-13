import { api } from "./api";
import { orderImages, useStore, type DocName } from "./store";
import type { Roi } from "./types";
import { renderImagePng } from "./render/snapshot";
import { kbd } from "./platform";
import { captureDelayed, captureImage, captureScreen } from "./capture";
import { newTable, newTextWindow } from "./components/TextWindow";
import { useScriptStore } from "./scriptStore";

export interface ViewportApi {
  zoomBy: (factor: number) => void;
  fit: () => void;
  setScale: (scale: number) => void;
  /** Fit the view to an image-space rectangle (zoom to selection). */
  fitRect: (x: number, y: number, w: number, h: number) => void;
}

export const viewportRef: { current: ViewportApi | null } = { current: null };

/** Client-side commands (not registry ops) as listed in the palette and the shortcut dialog. */
export interface ClientAction {
  id: string;
  name: string;
  category: string;
  description: string;
  kbd?: string;
  action: string;
  arg?: any;
  keywords: string[];
}

export const CLIENT_ACTIONS: ClientAction[] = [
  { id: "a:open", name: "Open file…", category: "File", description: "Browse the file system or upload", kbd: "⌘O", action: "open", keywords: ["load", "import"] },
  { id: "a:newFromSystemClipboard", name: "New from System Clipboard", category: "File/New", description: "Open the image on the OS clipboard as a new image (⌘V in the viewport when the browser asks)", action: "newFromSystemClipboard", keywords: ["paste", "clipboard", "screenshot", "import", "system"] },
  { id: "a:newTextWindow", name: "New Text Window", category: "File/New", description: "An editable text window in the inspector: save, append, save to file, download, run as a script", kbd: "⇧N", action: "newTextWindow", keywords: ["text", "notes", "editor", "window", "new"] },
  { id: "a:newTable", name: "New Table…", category: "File/New", description: "An empty named table", action: "newTable", keywords: ["table", "columns", "new"] },
  { id: "a:save", name: "Save", category: "File", description: "Save as ImageJ-compatible TIFF", kbd: "⌘S", action: "save", keywords: [] },
  { id: "a:saveAs", name: "Save As…", category: "File", description: "Choose a path and format", kbd: "⇧⌘S", action: "saveAs", keywords: ["export"] },
  { id: "a:openNext", name: "Open Next", category: "File", description: "Open the next image file in the current image's folder", action: "openNext", keywords: ["folder", "sequence"] },
  { id: "a:openPrevious", name: "Open Previous", category: "File", description: "Open the previous image file in the current image's folder", action: "openPrevious", keywords: ["folder", "sequence"] },
  { id: "a:clearRecent", name: "Clear Recent Files", category: "File", description: "Empty the File ▸ Open Recent list", action: "clearRecent", keywords: ["history"] },
  { id: "a:exportPng", name: "Export PNG", category: "File", description: "Download the current view as PNG", action: "exportPng", keywords: ["screenshot", "figure"] },
  { id: "a:exportCsv", name: "Export Results CSV", category: "File", description: "Download the Results table", action: "exportCsv", keywords: ["table"] },
  { id: "a:exportRois", name: "Export ROIs", category: "File", description: "Download the ROI manager as JSON", action: "exportRois", keywords: [] },
  { id: "a:print", name: "Print…", category: "File", description: "Print the current view with its name, size, calibration and a scale bar", action: "print", keywords: ["printer", "page", "pdf", "paper"] },
  { id: "a:pageSetup", name: "Page Setup…", category: "File", description: "Orientation and fit used by Print (remembered by this browser)", action: "pageSetup", keywords: ["print", "landscape", "portrait", "paper"] },
  { id: "a:nextImage", name: "Next Image", category: "View", description: "Activate the next image tab (⌘1–⌘9 jump to a tab; ←/→ on a focused tab)", action: "nextImage", keywords: ["tab", "switch", "window", "cycle"] },
  { id: "a:previousImage", name: "Previous Image", category: "View", description: "Activate the previous image tab", action: "previousImage", keywords: ["tab", "switch", "window", "cycle"] },
  { id: "a:undo", name: "Undo", category: "Edit", description: "", kbd: "⌘Z", action: "undo", keywords: [] },
  { id: "a:redo", name: "Redo", category: "Edit", description: "", kbd: "⇧⌘Z", action: "redo", keywords: [] },
  { id: "a:cut", name: "Cut", category: "Edit", description: "Copy the selection to the internal clipboard and clear it", kbd: "⌘X", action: "cut", keywords: ["clipboard"] },
  { id: "a:copyToSystem", name: "Copy to System", category: "Edit", description: "Copy the current view (or the selection's bounding box) to the system clipboard as PNG", action: "copyToSystem", keywords: ["clipboard", "screenshot", "png"] },
  { id: "a:restoreSelection", name: "Restore Selection", category: "Edit/Selection", description: "Bring back the previous selection of this image", kbd: "⇧E", action: "restoreSelection", keywords: ["roi", "undo selection"] },
  { id: "a:preferences", name: "Preferences…", category: "Edit/Options", description: "Colours, fonts, point/arrow defaults, zoom interpolation, save format…", kbd: "⌘,", action: "preferences", keywords: ["settings", "options", "theme", "colors"] },
  { id: "a:brightness", name: "Brightness / Contrast", category: "Image/Adjust", description: "Open the display panel (per-channel range and LUT)", kbd: "⇧⌘C", action: "brightness", keywords: ["display", "range", "lut", "levels"] },
  { id: "a:threshold", name: "Threshold…", category: "Image/Adjust", description: "Interactive threshold panel", kbd: "⇧⌘T", action: "threshold", keywords: ["segment", "binary", "mask"] },
  { id: "a:colorPicker", name: "Color Picker…", category: "Image/Color", description: "Foreground / background value or colour: HSB square, RGB and hex inputs, presets, pick from the image", kbd: "⇧K", action: "colorPicker", keywords: ["colour", "foreground", "background", "swatch", "hsb", "rgb", "hex", "picker"] },
  { id: "a:scriptEditor", name: "Script Editor", category: "Plugins/Scripting", description: "Python / ImageJ macro / JavaScript editor with line numbers, Open…, Save…, Run (⌘R) and the recorder ([ while no brush tool is active)", kbd: "[", action: "scriptEditor", keywords: ["script", "python", "javascript", "macro", "editor", "console", "interpreter", "recorder"] },
  { id: "a:scriptEditorLarge", name: "Script Editor in a Large View", category: "Plugins/Scripting", description: "The script editor and recorder as a large dialog over the image", action: "scriptEditorLarge", keywords: ["script", "editor", "large", "window", "maximize"] },
  { id: "a:captureScreen", name: "Capture Screen", category: "Plugins/Utilities", description: "Grab one frame of the screen (the browser asks which screen or window) and open it as a new RGB image", kbd: "⇧G", action: "captureScreen", keywords: ["screenshot", "screen", "grab", "display"] },
  { id: "a:captureDelayed", name: "Capture Delayed…", category: "Plugins/Utilities", description: "Capture the screen after a delay in seconds", action: "captureDelayed", keywords: ["screenshot", "timer", "delay", "screen"] },
  { id: "a:captureImage", name: "Capture Image", category: "Plugins/Utilities", description: "Flatten the current view (LUTs, display ranges, threshold, overlay) into a new RGB image", kbd: "⇧H", action: "captureImage", keywords: ["flatten", "screenshot", "rgb", "snapshot", "grab"] },
  { id: "a:zoomFit", name: "Fit to Window", category: "View", description: "", kbd: "0", action: "zoomFit", keywords: ["zoom"] },
  { id: "a:theme", name: "Toggle Theme", category: "View", description: "Dark / light", action: "theme", keywords: ["dark", "light"] },
  { id: "a:inspector", name: "Toggle Inspector", category: "View", description: "", kbd: "⌘\\", action: "inspector", keywords: ["panel", "sidebar"] },
  { id: "a:addShortcut", name: "Add Shortcut…", category: "Plugins/Shortcuts", description: "Bind a key combination to any command", action: "addShortcut", keywords: ["keyboard", "hotkey", "binding"] },
  { id: "a:removeShortcut", name: "Remove Shortcut…", category: "Plugins/Shortcuts", description: "Manage custom key bindings", action: "removeShortcut", keywords: ["keyboard", "hotkey"] },
  { id: "a:monitorMemory", name: "Monitor Memory", category: "Plugins/Utilities", description: "Live chart of the memory held by open images and undo history", action: "toggleMemoryMonitor", keywords: ["ram", "usage"] },
  { id: "a:searchWiki", name: "Search ImageJ wiki for…", category: "Plugins/Utilities", description: "Open an imagej.net search for what you typed in a new tab", action: "searchWiki", keywords: ["search", "help", "documentation", "web", "imagej", "fiji"] },
  { id: "a:searchForum", name: "Search image.sc forum for…", category: "Plugins/Utilities", description: "Open a forum.image.sc search for what you typed in a new tab", action: "searchForum", keywords: ["search", "help", "forum", "web", "question"] },
  { id: "a:help", name: "Keyboard Shortcuts", category: "Help", description: "", kbd: "?", action: "help", keywords: [] },
  { id: "a:docReadme", name: "Lumen README", category: "Help", description: "Open the README in the documentation viewer", action: "docs", arg: "README", keywords: ["docs", "manual"] },
  { id: "a:docCommands", name: "Command Reference", category: "Help", description: "Every command with its id and parameters", action: "docs", arg: "COMMANDS", keywords: ["docs", "list", "api"] },
  { id: "a:docParity", name: "Fiji Parity", category: "Help", description: "What is covered compared with Fiji/ImageJ", action: "docs", arg: "FIJI_PARITY", keywords: ["docs", "imagej", "coverage"] },
  { id: "a:aboutPlugins", name: "About Plugins", category: "Help", description: "Loaded plugin modules", action: "aboutPlugins", keywords: ["modules", "extensions"] },
];

function download(url: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Bounding box of a ROI in image pixels. */
export function roiBounds(roi: Roi): { x: number; y: number; w: number; h: number } {
  if (roi.type === "line") {
    const x = Math.min(roi.x1!, roi.x2!);
    const y = Math.min(roi.y1!, roi.y2!);
    return { x, y, w: Math.abs(roi.x2! - roi.x1!), h: Math.abs(roi.y2! - roi.y1!) };
  }
  if (roi.points && roi.points.length) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of roi.points) {
      if (p[0] < x0) x0 = p[0];
      if (p[0] > x1) x1 = p[0];
      if (p[1] < y0) y0 = p[1];
      if (p[1] > y1) y1 = p[1];
    }
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }
  return { x: roi.x || 0, y: roi.y || 0, w: roi.w || 0, h: roi.h || 0 };
}

const AREA_TYPES = new Set(["rect", "oval", "polygon", "freehand", "mask"]);

/** Copy the composited current plane (cropped to an area selection) to the system clipboard as PNG. */
async function copyToSystem(): Promise<void> {
  const st = useStore.getState();
  const img = st.activeImage();
  if (!img) return st.toast("Open an image first", "error");
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) return st.toast("This browser cannot write images to the clipboard (needs a secure context: localhost or https)", "error");
  const sel = st.selections[img.id];
  const crop = sel && AREA_TYPES.has(sel.type) ? roiBounds(sel) : undefined;
  const threshold = st.state?.thresholds[img.id] ?? null;
  try {
    // Safari only accepts a ClipboardItem created synchronously within the user gesture, so hand it the promise
    const blob = renderImagePng(img, st.luts, threshold, crop);
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    const b = await blob;
    const w = crop ? Math.round(crop.w) : img.shape.x;
    const h = crop ? Math.round(crop.h) : img.shape.y;
    st.toast(`Copied ${w}×${h} px PNG (${(b.size / 1024).toFixed(0)} KB) to the system clipboard`, "success");
  } catch (e: any) {
    st.toast(`Copy to System failed: ${e?.message || e}`, "error");
  }
}

// ---------------------------------------------------------------- File ▸ New ▸ System Clipboard
const CLIPBOARD_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"];

function clipboardFile(blob: Blob, type: string): File {
  const ext = type === "image/jpeg" ? "jpg" : type === "image/tiff" ? "tif" : (type.split("/")[1] || "png").replace(/^x-/, "");
  const stamp = new Date().toISOString().slice(0, 19).replace("T", " ").replace(/:/g, "-");
  return new File([blob], `Clipboard ${stamp}.${ext}`, { type });
}

/** Upload a clipboard image through the normal upload path so that it opens as a new image. */
async function openClipboardImage(blob: Blob, type: string): Promise<void> {
  const file = clipboardFile(blob, type);
  useStore.getState().setMenuOpen(null);
  try {
    await api.upload(file);
    await useStore.getState().refresh();
    useStore.getState().toast(`Opened ${file.name} from the system clipboard`, "success");
  } catch (e: any) {
    useStore.getState().toast(`Clipboard image could not be opened: ${e?.message || e}`, "error");
  }
}

let cancelPasteFallback: (() => void) | null = null;

/**
 * Without clipboard.read() (Firefox, a denied permission, plain http) wait for one ⌘V while the viewport is
 * focused and take the image from the paste event; the global ⌘V shortcut (internal clipboard) is bypassed once.
 */
function armPasteFallback(reason: string): void {
  cancelPasteFallback?.();
  const onPaste = (e: ClipboardEvent) => {
    const items = Array.from(e.clipboardData?.items || []);
    const it = items.find((i) => i.kind === "file" && i.type.startsWith("image/"));
    cleanup();
    if (!it) return useStore.getState().toast("Clipboard has no image", "error");
    e.preventDefault();
    e.stopPropagation();
    const f = it.getAsFile();
    if (f) void openClipboardImage(f, it.type);
  };
  const onKey = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "v") e.stopImmediatePropagation();
  };
  const timer = window.setTimeout(() => {
    cleanup();
    useStore.getState().toast("System clipboard: nothing pasted within 30 s – cancelled", "info");
  }, 30_000);
  const cleanup = () => {
    window.removeEventListener("paste", onPaste, true);
    window.removeEventListener("keydown", onKey, true);
    window.clearTimeout(timer);
    cancelPasteFallback = null;
  };
  window.addEventListener("paste", onPaste, true);
  window.addEventListener("keydown", onKey, true);
  cancelPasteFallback = cleanup;
  useStore.getState().setMenuOpen(null);
  (document.querySelector(".canvas-wrap") as HTMLElement | null)?.focus();
  useStore.getState().toast(`${reason} – press ${kbd("⌘V")} now to paste the image from the system clipboard`, "info");
}

/** File ▸ New ▸ System Clipboard: read an image from the OS clipboard and open it as a new image. */
async function newFromSystemClipboard(): Promise<void> {
  const st = useStore.getState();
  if (!window.isSecureContext || !navigator.clipboard?.read) return armPasteFallback("This browser cannot read the clipboard directly");
  let items: ClipboardItems;
  try {
    items = await navigator.clipboard.read();
  } catch (e: any) {
    const name = e?.name || "";
    if (name === "NotAllowedError" || name === "SecurityError") return armPasteFallback("Clipboard access was denied");
    if (name === "DataError") return st.toast("Clipboard has no image", "error");
    return st.toast(`Clipboard could not be read: ${e?.message || e}`, "error");
  }
  for (const item of items) {
    const type = CLIPBOARD_IMAGE_TYPES.find((t) => item.types.includes(t)) || item.types.find((t) => t.startsWith("image/"));
    if (!type) continue;
    try {
      const blob = await item.getType(type);
      return openClipboardImage(blob, type);
    } catch (e: any) {
      return st.toast(`Clipboard image could not be read: ${e?.message || e}`, "error");
    }
  }
  st.toast("Clipboard has no image", "error");
}

// ---------------------------------------------------------------- File ▸ Page Setup… / Print…
export interface PageSetup {
  orientation: "portrait" | "landscape";
  fit: "page" | "width" | "actual";
}

const PAGE_SETUP_KEY = "lumen.pageSetup";
const FIT_LABELS: Record<PageSetup["fit"], string> = { page: "fit to page", width: "fill page width", actual: "actual size" };

export function getPageSetup(): PageSetup {
  const d: PageSetup = { orientation: "portrait", fit: "page" };
  try {
    const raw = localStorage.getItem(PAGE_SETUP_KEY);
    if (!raw) return d;
    const p = JSON.parse(raw);
    if (p.orientation === "landscape" || p.orientation === "portrait") d.orientation = p.orientation;
    if (typeof p.fit === "string" && p.fit in FIT_LABELS) d.fit = p.fit;
  } catch {
    /* private mode or blocked storage: defaults */
  }
  return d;
}

/** Page setup without a dialog: a menu choice (`arg` = partial setup), "show", or a one-line prompt from the palette. */
function pageSetup(arg?: unknown): void {
  const st = useStore.getState();
  const next = getPageSetup();
  if (arg && typeof arg === "object") {
    const p = arg as Partial<PageSetup>;
    if (p.orientation === "portrait" || p.orientation === "landscape") next.orientation = p.orientation;
    if (p.fit && p.fit in FIT_LABELS) next.fit = p.fit;
  } else if (arg !== "show") {
    const answer = window.prompt("Page setup – orientation (portrait | landscape) and fit (page | width | actual):", `${next.orientation}, ${next.fit}`);
    if (answer === null) return;
    for (const tok of answer.toLowerCase().split(/[\s,;]+/)) {
      if (tok === "portrait" || tok === "landscape") next.orientation = tok;
      else if (tok === "page" || tok === "width" || tok === "actual") next.fit = tok;
      else if (tok === "fit") next.fit = "page";
    }
  }
  try {
    localStorage.setItem(PAGE_SETUP_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
  st.toast(`Page setup: ${next.orientation}, ${FIT_LABELS[next.fit]}`, "info");
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);
}

/** 1, 2 or 5 × 10^n at or above `target` (scale-bar length). */
function niceLength(target: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(target)));
  for (const m of [1, 2, 5, 10]) if (m * p >= target) return m * p;
  return 10 * p;
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error || new Error("could not read the rendered image"));
    r.readAsDataURL(blob);
  });
}

/** Render the current view (as the viewport shows it) into a print window with name, size, calibration and a scale bar, then print. */
async function printView(): Promise<void> {
  const st = useStore.getState();
  const img = st.activeImage();
  if (!img) return st.toast("Open an image first", "error");
  const setup = getPageSetup();
  // the window must be opened inside the user gesture (pop-up blockers); it is filled once the PNG is rendered
  const win = window.open("", "_blank");
  if (!win) return st.toast("The browser blocked the print window – allow pop-ups for Lumen and try again", "error");
  st.setMenuOpen(null);
  try {
    const blob = await renderImagePng(img, st.luts, st.state?.thresholds[img.id] ?? null);
    const src = await blobToDataUrl(blob);
    const w = img.shape.x;
    const h = img.shape.y;
    const cal = img.calibration;
    const scaled = !!cal.unit && !["pixel", "pixels"].includes(cal.unit) && cal.pixel_width > 0;
    const unit = cal.unit === "micron" || cal.unit === "um" ? "µm" : cal.unit;
    const dims = scaled
      ? `${(w * cal.pixel_width).toPrecision(4)} × ${(h * cal.pixel_height).toPrecision(4)} ${unit} (${w} × ${h} px, ${cal.pixel_width.toPrecision(4)} ${unit}/px)`
      : `${w} × ${h} px (uncalibrated)`;
    const d = img.display;
    const plane = [img.shape.c > 1 ? `c ${d.c + 1}/${img.shape.c}` : "", img.shape.z > 1 ? `z ${d.z + 1}/${img.shape.z}` : "", img.shape.t > 1 ? `t ${d.t + 1}/${img.shape.t}` : ""]
      .filter(Boolean)
      .join(" · ");
    let bar = "";
    if (scaled) {
      const len = niceLength((w * cal.pixel_width) / 5);
      const pct = Math.min(100, (len / (w * cal.pixel_width)) * 100);
      bar = `<div class="bar" style="width:${pct.toFixed(2)}%"><span>${len} ${esc(unit)}</span></div>`;
    }
    const imgCss =
      setup.fit === "actual" ? `width:${w}px;height:${h}px;max-width:none` : setup.fit === "width" ? "width:100%;height:auto" : "max-width:100%;max-height:calc(100vh - 34mm);width:auto;height:auto";
    const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(img.name)} – Lumen</title><style>
@page { size: ${setup.orientation}; margin: 12mm; }
html, body { margin: 0; background: #fff; color: #111; font: 12px/1.4 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
body { padding: 10mm; }
.tools { margin-bottom: 8px; color: #555; }
.hdr { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid #999; padding-bottom: 4px; margin-bottom: 8px; }
.hdr b { font-size: 14px; }
img { display: block; ${imgCss}; }
.bar { margin-top: 6px; height: 6px; background: #111; position: relative; }
.bar span { position: absolute; top: 8px; left: 0; white-space: nowrap; }
.foot { margin-top: ${scaled ? 26 : 6}px; color: #444; font-size: 11px; }
@media print { .tools { display: none; } body { padding: 0; } }
</style></head><body>
<div class="tools"><button onclick="window.print()">Print…</button> ${setup.orientation}, ${FIT_LABELS[setup.fit]} – change it with File ▸ Page Setup in Lumen</div>
<div class="hdr"><b>${esc(img.name)}</b><span>${esc(new Date().toLocaleString())}</span></div>
<img id="lumen-print" src="${src}" alt="${esc(img.name)}">
${bar}
<div class="foot">${esc(dims)}${plane ? " · " + esc(plane) : ""} · ${esc(img.bit_depth)}${img.rgb ? " RGB" : ""} · printed from Lumen</div>
</body></html>`;
    win.document.open();
    win.document.write(html);
    win.document.close();
    const el = win.document.getElementById("lumen-print") as HTMLImageElement | null;
    const go = () => {
      try {
        win.focus();
        win.print();
      } catch {
        /* window already closed */
      }
    };
    if (el && !el.complete) el.addEventListener("load", go, { once: true });
    else win.setTimeout(go, 50);
  } catch (e: any) {
    win.close();
    st.toast(`Print failed: ${e?.message || e}`, "error");
  }
}

// ---------------------------------------------------------------- Plugins ▸ Utilities ▸ Search…
const SEARCH_URLS = { wiki: "https://imagej.net/search?q=", forum: "https://forum.image.sc/search?q=" } as const;

/** Open a web search for `arg` (the palette passes what was typed) or for a prompted query in a new tab. */
function searchWeb(site: keyof typeof SEARCH_URLS, arg?: unknown): void {
  let q = typeof arg === "string" ? arg.trim() : "";
  if (!q) {
    const answer = window.prompt(site === "wiki" ? "Search the ImageJ wiki for:" : "Search the image.sc forum for:", "");
    if (answer === null) return;
    q = answer.trim();
  }
  if (!q) return;
  window.open(SEARCH_URLS[site] + encodeURIComponent(q), "_blank", "noopener,noreferrer");
}

function openRecent(path: string): void {
  const st = useStore.getState();
  st.setMenuOpen(null);
  api.openPath(path)
    .then(async () => {
      await useStore.getState().refresh();
      useStore.getState().toast(`Opened ${path.split(/[\\/]/).pop()}`, "success");
    })
    .catch((e) => useStore.getState().toast(`${path}: ${e.message}`, "error"));
}

function runOptionalOp(opId: string, label: string, quick = true): void {
  const st = useStore.getState();
  if (!st.ops[opId]) return st.toast(`${label} is not available on this server (command ${opId} not registered)`, "error");
  st.invoke(opId, quick);
}

export function doAction(action: string, arg?: any): void {
  const st = useStore.getState();
  const img = st.activeImage();
  switch (action) {
    case "open":
      st.setFileDialog({ mode: "open" });
      break;
    case "openRecent":
      if (typeof arg === "string" && arg) openRecent(arg);
      break;
    case "clearRecent":
      api.putSettings({ recent_clear: true })
        .then(() => useStore.getState().refresh())
        .then(() => useStore.getState().toast("Recent files cleared", "info"))
        .catch((e) => useStore.getState().toast(e.message, "error"));
      break;
    case "openNext":
      runOptionalOp("file.open_next", "Open Next");
      break;
    case "openPrevious":
      runOptionalOp("file.open_previous", "Open Previous");
      break;
    case "save":
      if (!img) return st.toast("No image to save", "error");
      if (img.path && /\.tiff?$/i.test(img.path)) {
        api.save(img.id, img.path)
          .then((r) => {
            useStore.getState().markSaved(img.id); // no longer "modified": closing will not ask
            st.toast(`Saved ${r.path}`, "success");
          })
          .catch((e) => st.toast(e.message, "error"));
      } else st.setFileDialog({ mode: "save", imageId: img.id });
      break;
    case "close":
      if (!img) return st.toast("No image to close", "info");
      void st.closeImage(img.id);
      break;
    case "closeAll":
      void st.closeAllImages();
      break;
    case "selectImage":
    case "nextImage":
    case "previousImage": {
      const images = orderImages(st.state?.images || [], st.imageOrder);
      if (!images.length) return;
      const cur = Math.max(0, images.findIndex((im) => im.id === st.state?.active));
      let i: number;
      if (action === "selectImage") {
        const n = typeof arg === "number" ? arg : parseInt(String(arg), 10);
        if (!Number.isFinite(n) || n < 1) return;
        i = n >= 9 || n > images.length ? images.length - 1 : n - 1; // ⌘9 = last tab, like browsers
      } else i = (cur + (action === "nextImage" ? 1 : images.length - 1)) % images.length;
      st.selectImage(images[i].id);
      break;
    }
    case "saveAs":
      if (!img) return st.toast("No image to save", "error");
      st.setFileDialog({ mode: "save", imageId: img.id });
      break;
    case "exportPng":
      if (img) download(`/api/images/${img.id}/download?format=png`);
      break;
    case "exportTif":
      if (img) download(`/api/images/${img.id}/download?format=tif`);
      break;
    case "exportOme":
      if (img) download(`/api/images/${img.id}/download?format=ome.tif`);
      break;
    case "exportCsv":
      download("/api/results.csv");
      break;
    case "exportRois":
      download("/api/rois/export");
      break;
    case "undo":
      void st.undo();
      break;
    case "redo":
      void st.redo();
      break;
    case "repeat":
      st.repeatLast();
      break;
    case "cut":
      runOptionalOp("edit.cut", "Cut");
      break;
    case "copyToSystem":
      void copyToSystem();
      break;
    case "newFromSystemClipboard":
      void newFromSystemClipboard();
      break;
    case "print":
      void printView();
      break;
    case "pageSetup":
      pageSetup(arg);
      break;
    case "searchWiki":
      searchWeb("wiki", arg);
      break;
    case "searchForum":
      searchWeb("forum", arg);
      break;
    case "restoreSelection":
      st.restoreSelection();
      break;
    case "colorPicker":
      st.setColorPickerOpen(!st.colorPickerOpen);
      break;
    case "captureScreen":
      void captureScreen(0);
      break;
    case "captureDelayed":
      captureDelayed(arg);
      break;
    case "captureImage":
      void captureImage();
      break;
    case "newTextWindow":
      newTextWindow();
      break;
    case "newTable":
      newTable();
      break;
    case "scriptEditor":
      st.setMenuOpen(null);
      useScriptStore.getState().focus();
      break;
    case "scriptEditorLarge":
      st.setMenuOpen(null);
      useScriptStore.getState().setLargeOpen(true);
      break;
    case "preferences":
      st.setPrefsOpen(true);
      break;
    case "palette":
      st.setPaletteOpen(true);
      break;
    case "inspector":
      st.setInspectorOpen(!st.inspectorOpen);
      break;
    case "theme":
      st.setTheme(st.theme === "dark" ? "light" : "dark");
      break;
    case "zoomIn":
      viewportRef.current?.zoomBy(1.5);
      break;
    case "zoomOut":
      viewportRef.current?.zoomBy(1 / 1.5);
      break;
    case "zoomFit":
      viewportRef.current?.fit();
      break;
    case "zoom100":
      viewportRef.current?.setScale(1);
      break;
    case "help":
      st.setHelpOpen(true);
      break;
    case "docs":
      st.setDocsOpen({ name: (typeof arg === "string" ? arg : "README") as DocName, anchor: undefined });
      break;
    case "helpOnCommand":
      // from a sheet's "?": the command reference only (no Design / Research tabs, E7-20)
      st.setDocsOpen({ name: "COMMANDS", anchor: typeof arg === "string" ? arg : undefined, compact: true });
      break;
    case "link":
      if (typeof arg === "string") window.open(arg, "_blank", "noopener,noreferrer");
      break;
    case "aboutPlugins":
      runOptionalOp("help.about_plugins", "About Plugins", false);
      break;
    case "about":
      st.toast(`Lumen ${st.meta?.version} — ${st.meta?.ops.length} commands. Built on numpy, scipy and scikit-image.`, "info");
      break;
    case "addShortcut":
    case "removeShortcut":
      st.setShortcutsOpen(true);
      break;
    case "toggleMemoryMonitor": {
      const on = !st.memoryMonitor;
      st.setMemoryMonitor(on);
      if (on) {
        st.setInspectorOpen(true);
        st.togglePanel("log", true);
      }
      break;
    }
    case "toggleManagerOverlay":
      st.setShowManagerRois(!st.showManagerRois);
      break;
    case "zoomSelection": {
      if (!img) return st.toast("Open an image first", "error");
      const roi = st.selections[img.id];
      if (!roi) return st.toast("Make a selection first", "info");
      const b = roiBounds(roi);
      viewportRef.current?.fitRect(b.x, b.y, b.w, b.h);
      break;
    }
    case "toggleSplit":
      st.toggleSplitView();
      break;
    case "toggleSyncViews":
      st.setSyncViews(!st.syncViews);
      st.toast(st.syncViews ? "Views are independent" : "Zoom and pan are synchronized across images of the same size", "info");
      break;
    case "toggleOverlay":
      st.setShowOverlay(!st.showOverlay);
      break;
    case "toggleTrackLabels":
      st.setShowTrackLabels(!st.showTrackLabels);
      break;
    case "toggleTrackFuture":
      st.setShowTrackFuture(!st.showTrackFuture);
      break;
    case "brightness":
      st.setInspectorOpen(true);
      st.togglePanel("display", true);
      break;
    case "threshold":
      st.setInspectorOpen(true);
      st.togglePanel("threshold", true);
      if (img) void st.runOp("threshold.compute", { method: "Default", dark_background: true }, { silent: true });
      break;
    default:
      st.toast(`Unknown action ${action}`, "error");
  }
}
