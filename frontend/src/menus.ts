import type { OpSpec } from "./types";
import { kbd } from "./platform";
import { lutLabel, type LutGroup, type LutGrouping } from "./lutGroups";

/**
 * Display names of registry ops that the client shows differently from the server's `name` (GUI fix round 2,
 * E5-2 / E7-9 / E7-13): the destructive ImageJ "Apply LUT" says what it does, the two File ▸ Import / Export "LUT…"
 * entries say which is which, and "New ▸ Internal Clipboard" reads as one command outside its submenu.  Ids, macros
 * and `lumen.run(...)` lines are untouched; the palette, the menus and the sheet title all use `displayName()`.
 */
export const OP_LABELS: Record<string, string> = {
  "type.apply_lut": "Apply Display Range to Pixels (Apply LUT)",
  "adjust.brightness_contrast_apply": "Apply Display Range to Pixels (B&C Apply)",
  "color.lut": "Set LUT",
  "file.import_lut": "Import LUT",
  "file.export_lut": "Export LUT",
  "file.new_from_clipboard": "New from Internal Clipboard",
};

/** The name an op is shown under (OP_LABELS override, else the server's name). */
export function displayName(op: Pick<OpSpec, "id" | "name">): string {
  return OP_LABELS[op.id] || op.name;
}

/**
 * The one number the UI quotes as "commands" (E8-5): every registry op (`meta.ops`) – the same count the generated
 * command reference (docs/COMMANDS.md) carries.  The palette's client actions and its LUT entries are counted
 * separately ("+ 3 LUTs"), never folded into this number.
 */
export function commandCount(ops: readonly Pick<OpSpec, "id">[] | undefined | null): number {
  return ops ? ops.length : 0;
}

export interface MenuItem {
  label: string;
  op?: string;
  /** Run `op` straight away with these params (no sheet), e.g. a sample picked from a submenu. */
  params?: Record<string, any>;
  action?: string;
  /** Argument handed to `doAction(action, arg)` (a file path, a URL, a document name …). */
  arg?: any;
  kbd?: string;
  children?: MenuItem[];
  sep?: boolean;
  disabled?: boolean;
  /** Name of a boolean store field; the menu shows a check mark when it is true. */
  checked?: string;
  /** The item is dropped when its `op` is not registered on this server (commands from optional modules). */
  optional?: boolean;
  /** Tooltip (menus show the op description by default). */
  title?: string;
}

const ORDER = ["File", "Edit", "Image", "Process", "Analyze", "Plugins", "View", "Help"];

const SUBMENU_ORDER: Record<string, string[]> = {
  File: ["New", "Import", "Export"],
  Image: ["Type", "Adjust", "Color", "Stacks", "Hyperstacks", "Transform", "Lookup Tables", "Overlay"],
  Process: ["Filters", "Binary", "Math", "Noise", "FFT"],
  Edit: ["Selection", "Options"],
  Plugins: ["Batch", "Registration", "Scripting", "Segmentation", "Tracking", "Shortcuts", "Utilities"],
};

/** Registry ops whose root-menu entry is replaced by a client submenu (File ▸ Open Recent, File ▸ New ▸ …). */
const HIDDEN_ROOT_OPS = new Set(["file.sample", "file.open_imagej_sample", "edit.set_colors", "file.open_recent", "file.new", "file.new_from_clipboard"]);
/**
 * Registry ops listed neither in the menu bar nor in the command palette because a client item is
 * their UI (they stay reachable from macros, scripts and the API): edit.options would duplicate
 * Edit ▸ Options ▸ Preferences… (⌘,), which edits the live settings through PreferencesSheet.
 */
export const HIDDEN_MENU_OPS = new Set(["edit.options"]);

export const DOC_LINKS: { label: string; url: string }[] = [
  { label: "ImageJ Website", url: "https://imagej.net/ij/" },
  { label: "ImageJ User Guide", url: "https://imagej.net/ij/docs/guide/" },
  { label: "Fiji Wiki", url: "https://imagej.net/software/fiji/" },
  { label: "image.sc Forum", url: "https://forum.image.sc/" },
];

const CLIENT_ITEMS: Record<string, MenuItem[]> = {
  File: [
    { label: "Open…", action: "open", kbd: "⌘O" },
    { label: "Open Recent", children: [] }, // filled from state.recent in buildMenus
    { label: "Open Next", op: "file.open_next", optional: true, title: "Open the next image file in the folder of the current image" },
    { label: "Open Previous", op: "file.open_previous", optional: true, title: "Open the previous image file in the folder of the current image" },
    { label: "Open Sample", children: [] }, // filled from the file.sample / file.open_imagej_sample choices
    { sep: true, label: "" },
    { label: "Save", action: "save", kbd: "⌘S" },
    { label: "Save As…", action: "saveAs", kbd: "⇧⌘S" },
    { sep: true, label: "" },
    { label: "Page Setup", title: "Orientation and fit used by Print (remembered by this browser)", children: [
      { label: "Portrait", action: "pageSetup", arg: { orientation: "portrait" } },
      { label: "Landscape", action: "pageSetup", arg: { orientation: "landscape" } },
      { sep: true, label: "" },
      { label: "Fit to Page", action: "pageSetup", arg: { fit: "page" } },
      { label: "Fill Page Width", action: "pageSetup", arg: { fit: "width" } },
      { label: "Actual Size", action: "pageSetup", arg: { fit: "actual" } },
      { sep: true, label: "" },
      { label: "Show Current Setup", action: "pageSetup", arg: "show" },
    ] },
    { label: "Print…", action: "print", title: "Print the current view with its name, size, calibration and a scale bar" },
    { sep: true, label: "" },
    { label: "Close", op: "file.close", kbd: "⌘W", title: "Close the current image (asks first when it has unsaved changes)" },
    { label: "Close All", op: "file.close_all", kbd: "⇧⌘W", title: "Close every image (asks first when any has unsaved changes)" },
  ],
  Edit: [
    { label: "Undo", action: "undo", kbd: "⌘Z" },
    { label: "Redo", action: "redo", kbd: "⇧⌘Z" },
    { label: "Repeat Last Command", action: "repeat", kbd: "⇧R" },
    { sep: true, label: "" },
    { label: "Cut", op: "edit.cut", kbd: "⌘X", optional: true },
    { label: "Copy", op: "edit.copy", kbd: "⌘C" },
    { label: "Copy to System", action: "copyToSystem", title: "Copy the current view (or the selection's bounding box) to the system clipboard as PNG" },
    { label: "Paste", op: "edit.paste", kbd: "⌘V" },
    { sep: true, label: "" },
  ],
  View: [
    { label: "Command Palette", action: "palette", kbd: "⌘K" },
    { label: "Toggle Inspector", action: "inspector", kbd: "⌘\\" },
    { label: "Toggle Theme", action: "theme" },
    { sep: true, label: "" },
    { label: "Zoom In", action: "zoomIn", kbd: "+" },
    { label: "Zoom Out", action: "zoomOut", kbd: "−" },
    { label: "Fit to Window", action: "zoomFit", kbd: "0" },
    { label: "Actual Size (100%)", action: "zoom100" },
    { label: "Zoom to Selection", action: "zoomSelection" },
    { label: "Synchronize Views", action: "toggleSyncViews", checked: "syncViews" },
    { label: "Split View (Compare Two Images)", action: "toggleSplit", kbd: "⇧⌘\\" },
    { sep: true, label: "" },
    { label: "Show Overlay", action: "toggleOverlay", kbd: "⇧O", checked: "showOverlay" },
    { label: "Show Track Labels", action: "toggleTrackLabels", checked: "showTrackLabels" },
    { label: "Show Future Track Points", action: "toggleTrackFuture", checked: "showTrackFuture" },
    { label: "Show ROI Manager overlay", action: "toggleManagerOverlay", checked: "showManagerRois" },
  ],
  Help: [
    { label: "Keyboard Shortcuts", action: "help", kbd: "?" },
    { sep: true, label: "" },
    { label: "Lumen README", action: "docs", arg: "README" },
    { label: "Command Reference", action: "docs", arg: "COMMANDS" },
    { label: "Fiji Parity", action: "docs", arg: "FIJI_PARITY" },
    { label: "Design & Research Notes (background)", action: "docs", arg: "DESIGN" },
    { sep: true, label: "" },
    ...DOC_LINKS.map((l) => ({ label: l.label + " ↗", action: "link", arg: l.url, title: l.url })),
    { sep: true, label: "" },
    { label: "About Plugins", op: "help.about_plugins", optional: true },
    { label: "About Lumen", action: "about" },
  ],
};

/** Client items placed at the top of a registry submenu (before a separator): ImageJ lists Brightness/Contrast… first under Image ▸ Adjust. */
const CLIENT_SUB_ITEMS_FIRST: Record<string, Record<string, MenuItem[]>> = {
  File: {
    // ImageJ's File ▸ New ▸ Image… / Hyperstack… / Text Window / Internal Clipboard / System Clipboard / Table (E7-13: one "New")
    New: [{ label: "Image…", op: "file.new", kbd: "N", title: "A blank image or hyperstack (type, size, channels, slices, frames, fill)" }],
  },
  Image: {
    Adjust: [
      { label: "Brightness/Contrast…", action: "brightness", kbd: "⇧⌘C", title: "Display range and LUT per channel (Display section) – never changes pixel values" },
      { label: "Threshold…", action: "threshold", kbd: "⇧⌘T", title: "Interactive threshold (Threshold section)" },
    ],
  },
};

/** Client items appended to a registry submenu (after a separator), or forming it when no op lives there. */
const CLIENT_SUB_ITEMS: Record<string, Record<string, MenuItem[]>> = {
  File: {
    // next to the registry's File ▸ New ▸ Text Window… / Table… (file.new_text / table.new, which open a sheet)
    New: [
      { label: "New Text Window", action: "newTextWindow", kbd: "⇧N", title: "An editable text window in the inspector: save, append, save to file, download, run as macro or script" },
      { label: "New Table", action: "newTable", title: "An empty named table (table.new)" },
      { sep: true, label: "" },
      { label: "Internal Clipboard…", op: "file.new_from_clipboard", optional: true, title: "Open the pixels copied with Copy / Cut as a new image" },
      { label: "System Clipboard", action: "newFromSystemClipboard", title: "Open the image on the system clipboard as a new image (press ⌘V in the viewport when the browser cannot read it directly)" },
    ],
    // the client's downloads join the registry's File ▸ Export ▸ … (one Export submenu, E7-13)
    Export: [
      { label: "PNG (current view)", action: "exportPng" },
      { label: "TIFF (ImageJ hyperstack)", action: "exportTif" },
      { label: "OME-TIFF", action: "exportOme" },
      { label: "Results as CSV", action: "exportCsv" },
      { label: "ROIs as JSON", action: "exportRois" },
    ],
  },
  Edit: {
    Selection: [{ label: "Restore Selection", action: "restoreSelection", kbd: "⇧E" }],
    Options: [
      { label: "Preferences…", action: "preferences", kbd: "⌘," },
      { label: "Colour Settings…", op: "edit.set_colors", optional: true },
    ],
  },
  Image: {
    Color: [{ label: "Color Picker…", action: "colorPicker", kbd: "⇧K", checked: "colorPickerOpen", title: "Foreground / background value or colour: HSB, RGB, hex, presets and pick from the image" }],
    Overlay: [
      { label: "Show Overlay", action: "toggleOverlay", kbd: "⇧O", checked: "showOverlay", title: "Show or hide the overlay (also View ▸ Show Overlay)" },
      { label: "Overlay Options…", op: "edit.options_overlay", optional: true, title: "Default stroke width and fill of overlay items; labels: show, use names, font size, colour, background" },
    ],
  },
  Plugins: {
    Scripting: [{ label: "Script Editor", action: "scriptEditor", kbd: "[", title: "Python, ImageJ macro or JavaScript editor with line numbers, Open…/Save…, Run (⌘R) and the recorder" }],
    Shortcuts: [
      { label: "Add Shortcut…", action: "addShortcut" },
      { label: "Remove Shortcut…", action: "removeShortcut" },
      { label: "List Shortcuts", action: "help" },
    ],
    Utilities: [
      { label: "Monitor Memory", action: "toggleMemoryMonitor", checked: "memoryMonitor" },
      { sep: true, label: "" },
      { label: "Capture Screen", action: "captureScreen", kbd: "⇧G", title: "Grab one frame of the screen (the browser asks which screen or window) and open it as a new RGB image" },
      { label: "Capture Delayed…", action: "captureDelayed", title: "Capture the screen after a delay in seconds" },
      { label: "Capture Image", action: "captureImage", kbd: "⇧H", title: "Flatten the current view (LUTs, display ranges, threshold, overlay) into a new RGB image" },
      { sep: true, label: "" },
      { label: "Search ImageJ Wiki…", action: "searchWiki", title: "Search imagej.net in a new tab (the ⌘K palette searches for whatever you typed there)" },
      { label: "Search image.sc Forum…", action: "searchForum", title: "Search forum.image.sc in a new tab" },
    ],
  },
};

export interface MenuContext {
  /** Recently opened file paths, newest first (state.recent). */
  recent?: string[];
  /** The step Edit ▸ Undo / Redo would take back ("Gaussian Blur"); undefined = nothing to undo / redo (E6-7). */
  undoLabel?: string;
  redoLabel?: string;
  /** Registered LUT names (meta.luts) and their grouping: Image ▸ Lookup Tables lists them (E5-1 / E7-9). */
  lutNames?: string[];
  lutGrouping?: LutGrouping;
  lutGroups?: LutGroup[];
}

/** Image ▸ Lookup Tables ▸ <name>: `color.lut` with the LUT as its parameter (display only). */
function lutItem(name: string, grouping: LutGrouping | undefined): MenuItem {
  return { label: grouping ? lutLabel(name, grouping) : name, op: "color.lut", params: { lut: name }, title: `Set the current channel's lookup table to ${name} (display only – pixel values unchanged)` };
}

/** Insert `items` as the submenu `path` ("New/Internal Clipboard" → New ▸ Internal Clipboard ▸ …) of `container`. */
function insertNested(container: MenuItem[], path: string[], items: MenuItem[]): void {
  const [head, ...rest] = path;
  let node = container.find((it) => it.children && it.label === head);
  if (!node) {
    node = { label: head, children: [] };
    container.push(node);
  }
  if (rest.length) insertNested(node.children!, rest, items);
  else node.children!.push(...items);
}

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** The last two folders of a path ("…/eval_e8/images"), to tell two recent files of the same name apart. */
function folderHint(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  parts.pop();
  if (!parts.length) return path;
  return (parts.length > 2 ? "…/" : "") + parts.slice(-2).join("/");
}

function choices(ops: Record<string, OpSpec>, opId: string, param: string): string[] {
  return ops[opId]?.params.find((p) => p.name === param)?.choices || [];
}

/** Menu label of a registry op: "…" marks ops that open a sheet, unless the name already ends with an ellipsis; a "..." becomes "…" (E7-13). */
function menuLabel(op: OpSpec): string {
  const name = displayName(op).replace(/\s*\.\.\.$/, "…");
  return name + (op.params.length && !/…$/.test(name) ? "…" : "");
}

/**
 * An `ij_commands` entry (OpSpec.ij_commands, E7-20) as it reads in an ImageJ macro: a bare command name
 * ("Gaussian Blur...") becomes `run("Gaussian Blur...")`; an entry that already is a call or a macro function
 * (`run("…", "sigma=2")`, `setAutoThreshold("Otsu dark")`, `Stack.setChannel(2)`) is shown as is.
 */
export function ijMacroCall(c: string): string {
  const s = String(c).trim();
  if (!s) return "";
  if (/^[A-Za-z_][A-Za-z0-9_.]*\s*\(/.test(s)) return s;
  return `run("${s.replace(/"/g, '\\"')}")`;
}

/** Registry labels that mean the same as a client download with another name (File ▸ Export, E7-13). */
const LABEL_TWINS: Record<string, string> = { "results table": "results as csv" };

/** "New Text Window" / "Text Window…" / "ROIs as JSON..." → "text window" / "rois as json": the key that says two entries of one submenu duplicate each other (E7-13). */
function labelKey(label: string): string {
  const k = label.toLowerCase().replace(/(…|\.\.\.)\s*$/, "").replace(/^new\s+/, "").replace(/\s+/g, " ").trim();
  return LABEL_TWINS[k] || k;
}

/**
 * E7-13: a registry op whose label means the same as a client item of the same submenu (File ▸ New ▸ "Text Window…"
 * next to "New Text Window", Export ▸ "ROIs as JSON..." next to "ROIs as JSON") is left out of the menu – the client
 * item is the one with the shortcut and the direct behaviour; the op stays reachable in the palette and from scripts.
 */
function dedupeByLabel(items: MenuItem[]): MenuItem[] {
  const client = new Set<string>();
  for (const it of items) if (!it.sep && it.action && !it.children && !it.disabled) client.add(labelKey(it.label));
  if (!client.size) return items;
  return items.filter((it) => !(it.op && !it.params && !it.children && client.has(labelKey(it.label))));
}

/** Drop optional items whose op is missing; recurse into submenus; collapse doubled separators. */
function prune(items: MenuItem[], ops: Record<string, OpSpec>): MenuItem[] {
  const out: MenuItem[] = [];
  for (const it of dedupeByLabel(items)) {
    if (it.optional && it.op && !ops[it.op]) continue;
    const item = it.children ? { ...it, children: prune(it.children, ops) } : it;
    if (item.children && item.children.length === 0 && !item.action && !item.op) continue;
    if (item.sep && (out.length === 0 || out[out.length - 1].sep)) continue;
    out.push(item);
  }
  while (out.length && out[out.length - 1].sep) out.pop();
  return out;
}

export function buildMenus(opList: OpSpec[], ctx: MenuContext = {}): { label: string; items: MenuItem[] }[] {
  const ops: Record<string, OpSpec> = {};
  for (const o of opList) ops[o.id] = o;
  const roots: Record<string, MenuItem[]> = {};
  for (const r of ORDER) roots[r] = [];
  const sub: Record<string, Record<string, MenuItem[]>> = {};
  for (const op of opList) {
    if (HIDDEN_MENU_OPS.has(op.id)) continue;
    const parts = op.menu.split("/");
    const root = parts[0];
    if (!(root in roots)) roots[root] = [];
    const item: MenuItem = { label: menuLabel(op), op: op.id, kbd: prettyShortcut(op.shortcut) };
    if (parts.length === 1) {
      if (!HIDDEN_ROOT_OPS.has(op.id)) roots[root].push(item);
    } else {
      sub[root] = sub[root] || {};
      const name = parts.slice(1).join("/");
      sub[root][name] = sub[root][name] || [];
      sub[root][name].push(item);
    }
  }

  // dynamic client submenus
  const recent = ctx.recent || [];
  // two recent files with the same name (im_0.tif in two folders) show their folder too (E7-13)
  const shown = recent.slice(0, 15);
  const names = shown.map(baseName);
  const dupNames = new Set(names.filter((n, i) => names.indexOf(n) !== i));
  const recentItems: MenuItem[] = recent.length
    ? [
        ...shown.map((p) => ({ label: dupNames.has(baseName(p)) ? `${baseName(p)}  –  ${folderHint(p)}` : baseName(p), action: "openRecent", arg: p, title: p })),
        { sep: true, label: "" },
        { label: "Clear Recent", action: "clearRecent" },
      ]
    : [{ label: "No recent files", disabled: true }];
  const samples = choices(ops, "file.sample", "name").map((name) => ({ label: name, op: "file.sample", params: { name } }));
  const ijSamples = choices(ops, "file.open_imagej_sample", "name").map((name) => ({ label: name, op: "file.open_imagej_sample", params: { name } }));
  const sampleItems: MenuItem[] = [...samples];
  if (ijSamples.length) sampleItems.push({ sep: true, label: "" }, { label: "ImageJ samples (download)", disabled: true }, ...ijSamples);
  if (!sampleItems.length) sampleItems.push({ label: "Open Sample…", op: "file.sample" });

  // client-only items in File/Edit/View/Help are placed first
  for (const [root, items] of Object.entries(CLIENT_ITEMS)) {
    const resolved = items.map((it) => {
      if (it.label === "Open Recent") return { ...it, children: recentItems };
      if (it.label === "Open Sample") return { ...it, children: sampleItems };
      // Edit ▸ Undo / Redo name the step they take back (E6-7)
      if (it.action === "undo") return { ...it, label: ctx.undoLabel ? `Undo ${ctx.undoLabel}` : "Undo", disabled: !ctx.undoLabel };
      if (it.action === "redo") return { ...it, label: ctx.redoLabel ? `Redo ${ctx.redoLabel}` : "Redo", disabled: !ctx.redoLabel };
      return it;
    });
    roots[root] = [...resolved, ...(roots[root] || [])];
  }
  // Image ▸ Lookup Tables lists every LUT like Fiji (E5-1 / E7-9): ImageJ's core set flat after the LUT commands, the
  // Fiji / perceptual / label / imported sets as submenus; each entry runs color.lut (Set LUT) for the current channel
  if (ops["color.lut"] && ctx.lutGroups && ctx.lutGroups.length) {
    sub.Image = sub.Image || {};
    const lt = sub.Image["Lookup Tables"] || (sub.Image["Lookup Tables"] = []);
    lt.push({ sep: true, label: "" });
    for (const g of ctx.lutGroups) {
      if (g.name === "ImageJ") {
        lt.push({ label: "ImageJ", disabled: true, title: "ImageJ's built-in lookup tables" }, ...g.luts.map((n) => lutItem(n, ctx.lutGrouping)));
      } else lt.push({ label: g.name, children: g.luts.map((n) => lutItem(n, ctx.lutGrouping)) });
    }
  }
  for (const [root, subs] of Object.entries(CLIENT_SUB_ITEMS)) {
    sub[root] = sub[root] || {};
    for (const [name, items] of Object.entries(subs)) {
      const existing = sub[root][name] || [];
      sub[root][name] = existing.length ? [...existing, { sep: true, label: "" }, ...items] : items;
    }
  }
  for (const [root, subs] of Object.entries(CLIENT_SUB_ITEMS_FIRST)) {
    sub[root] = sub[root] || {};
    for (const [name, items] of Object.entries(subs)) {
      const existing = sub[root][name] || [];
      sub[root][name] = existing.length ? [...items, { sep: true, label: "" }, ...existing] : items;
    }
  }

  const out: { label: string; items: MenuItem[] }[] = [];
  for (const root of ORDER.concat(Object.keys(roots).filter((r) => !ORDER.includes(r)))) {
    const items = [...(roots[root] || [])];
    const subs = sub[root] || {};
    const names = Object.keys(subs).sort((a, b) => {
      const o = SUBMENU_ORDER[root] || [];
      const ia = o.indexOf(a);
      const ib = o.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
    });
    if (names.length && items.length) items.push({ sep: true, label: "" });
    // "New/Internal Clipboard" (a three-level op.menu) nests under New instead of a flattened "New/Internal Clipboard" entry (E7-13)
    for (const n of names) if (!n.includes("/")) items.push({ label: n, children: subs[n] });
    for (const n of names) if (n.includes("/")) insertNested(items, n.split("/"), subs[n]);
    // de-duplicate (file.close, edit.copy … appear both as client and registry items; the client one wins)
    const seen = new Set<string>();
    const dedup = items.filter((i) => {
      if (!i.op || i.params) return true;
      if (seen.has(i.op)) return false;
      seen.add(i.op);
      return true;
    });
    const pruned = prune(dedup, ops);
    if (pruned.length) out.push({ label: root, items: pruned });
  }
  return out;
}

export function prettyShortcut(s: string | null | undefined): string | undefined {
  if (!s) return undefined;
  return kbd(s.replace("Shift+", "⇧").replace("Cmd+", "⌘").replace("Ctrl+", "⌃").replace("Alt+", "⌥"));
}
