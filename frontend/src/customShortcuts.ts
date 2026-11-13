/**
 * User-defined keyboard shortcuts (Plugins ▸ Shortcuts ▸ Add Shortcut…).  Stored in this
 * browser under localStorage "lumen.shortcuts" and checked by shortcuts.ts BEFORE the built-in
 * bindings, so a custom shortcut can override one.
 *
 * A combo is a normalised string: optional "mod+" (⌘ on macOS / Ctrl elsewhere), "shift+",
 * "alt+" followed by the key: lower-case letters/digits/punctuation as `event.key` reports them,
 * "space", or named keys such as "F2", "Enter", "ArrowUp".
 */
import { kbd } from "./platform";

export interface CustomShortcut {
  combo: string;
  /** Registry command id … */
  op?: string;
  /** … or a client action (actions.ts) */
  action?: string;
  /** Argument handed to `doAction(action, arg)` (e.g. the document name for "docs"); absent for ops. */
  arg?: unknown;
  label: string;
}

export const SHORTCUTS_KEY = "lumen.shortcuts";

export function loadCustomShortcuts(): CustomShortcut[] {
  try {
    const raw = JSON.parse(localStorage.getItem(SHORTCUTS_KEY) || "[]");
    if (!Array.isArray(raw)) return [];
    return raw.filter((s) => s && typeof s.combo === "string" && (typeof s.op === "string" || typeof s.action === "string")).map((s) => ({ combo: s.combo, op: s.op, action: s.action, ...(s.arg !== undefined ? { arg: s.arg } : {}), label: String(s.label || s.op || s.action) }));
  } catch {
    return [];
  }
}

export function saveCustomShortcuts(list: CustomShortcut[]): void {
  try {
    localStorage.setItem(SHORTCUTS_KEY, JSON.stringify(list));
  } catch {
    /* private mode / quota: shortcuts live for the session only */
  }
}

const MODIFIER_KEYS = new Set(["Shift", "Control", "Alt", "Meta", "AltGraph", "CapsLock", "Fn", "OS"]);

/** Normalised combo of a key event, or null for a bare modifier / dead key. */
export function comboFromEvent(e: { key: string; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean }): string | null {
  const key = e.key;
  if (!key || key === "Dead" || key === "Unidentified" || MODIFIER_KEYS.has(key)) return null;
  const parts: string[] = [];
  if (e.metaKey || e.ctrlKey) parts.push("mod");
  if (e.shiftKey) parts.push("shift");
  if (e.altKey) parts.push("alt");
  let k = key === " " ? "space" : key;
  if (k.length === 1) k = k.toLowerCase();
  parts.push(k);
  return parts.join("+");
}

/**
 * "mod+shift+g" → "⌘⇧G" on a Mac, "Ctrl+Shift+G" elsewhere.  The key is whatever follows the
 * leading "mod+" / "shift+" / "alt+" modifiers and may itself be "+" (combos "shift++" and "+"),
 * so the combo is parsed from the front rather than split on "+".  Only the modifier glyphs go
 * through kbd(): its "++" → "+" collapse would swallow a "+" key on Windows/Linux.
 */
export function prettyCombo(combo: string): string {
  const m = /^((?:mod\+|shift\+|alt\+)*)(.+)$/.exec(combo);
  const key = m ? m[2] : combo;
  const mods = (m ? m[1] : "").split("+").filter(Boolean).map((x) => (x === "mod" ? "⌘" : x === "shift" ? "⇧" : x === "alt" ? "⌥" : x)).join("");
  const names: Record<string, string> = { space: "Space", arrowup: "↑", arrowdown: "↓", arrowleft: "←", arrowright: "→", enter: "↵", escape: "Esc", backspace: "⌫", delete: "⌦", tab: "⇥" };
  const k = names[key.toLowerCase()] || (key.length === 1 ? key.toUpperCase() : key);
  return (kbd(mods) || "") + k;
}

/** Built-in bindings (shortcuts.ts) so the dialog can warn about an override. */
export const BUILTIN_SHORTCUTS: { combo: string; label: string }[] = [
  { combo: "mod+k", label: "Command palette" }, { combo: "mod+o", label: "Open" }, { combo: "mod+s", label: "Save" }, { combo: "mod+shift+s", label: "Save As" }, { combo: "mod+p", label: "Print" },
  // ⌘⇧\ is matched on e.code "Backslash"; comboFromEvent reports the shifted key, which is "|" on US/UK
  // layouts (like "shift+?" / "shift+<" below), so the override warning is best-effort on other layouts
  { combo: "mod+shift+|", label: "Split view" },
  { combo: "mod+\\", label: "Toggle inspector" }, { combo: "mod+,", label: "Preferences" }, { combo: "mod+z", label: "Undo" }, { combo: "mod+shift+z", label: "Redo" },
  { combo: "mod+w", label: "Close image" }, { combo: "mod+shift+w", label: "Close all images" }, { combo: "F10", label: "Menu bar" },
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => ({ combo: `mod+${n}`, label: n === 9 ? "Last image tab" : `Image tab ${n}` })),
  { combo: "mod+y", label: "Redo" }, { combo: "mod+x", label: "Cut" }, { combo: "mod+c", label: "Copy" }, { combo: "mod+v", label: "Paste" }, { combo: "mod+a", label: "Select all" },
  { combo: "mod+shift+a", label: "Select none" }, { combo: "mod+shift+c", label: "Brightness/Contrast" }, { combo: "mod+shift+t", label: "Threshold" },
  { combo: "v", label: "Pointer tool" }, { combo: "g", label: "Hand tool" }, { combo: "1", label: "Rectangle tool" }, { combo: "2", label: "Oval tool" }, { combo: "3", label: "Polygon tool" },
  { combo: "4", label: "Freehand tool" }, { combo: "5", label: "Line tool" }, { combo: "6", label: "Polyline tool" }, { combo: "7", label: "Point tool" }, { combo: "8", label: "Wand tool" },
  { combo: "9", label: "Brush tool" }, { combo: "z", label: "Zoom tool" }, { combo: "i", label: "Picker tool" }, { combo: "y", label: "Text tool" }, { combo: "u", label: "Bucket tool" },
  { combo: "j", label: "Arrow tool" }, { combo: "q", label: "Angle tool" }, { combo: "x", label: "Manual tracking tool" }, { combo: "shift+?", label: "Keyboard shortcuts" },
  // shortcuts.ts zooms on "+" / "=" / "-" / "_" without looking at Shift, so the shifted forms that
  // comboFromEvent produces for Shift+= and Shift+- are built-ins too
  { combo: "+", label: "Zoom in" }, { combo: "shift++", label: "Zoom in" }, { combo: "=", label: "Zoom in" }, { combo: "-", label: "Zoom out" }, { combo: "shift+_", label: "Zoom out" }, { combo: "0", label: "Fit to window" }, { combo: "[", label: "Smaller brush" }, { combo: "]", label: "Larger brush" },
  { combo: ",", label: "Previous slice" }, { combo: ".", label: "Next slice" }, { combo: "shift+<", label: "Previous frame" }, { combo: "shift+>", label: "Next frame" },
  { combo: "m", label: "Measure" }, { combo: "t", label: "Add to ROI manager" }, { combo: "h", label: "Histogram" }, { combo: "k", label: "Plot profile" }, { combo: "a", label: "Select all" },
  { combo: "shift+a", label: "Select none" }, { combo: "b", label: "Add to overlay" }, { combo: "f", label: "Fill" }, { combo: "shift+x", label: "Crop" }, { combo: "shift+d", label: "Duplicate" },
  { combo: "shift+i", label: "Invert" }, { combo: "n", label: "New image" }, { combo: "shift+s", label: "Smooth" }, { combo: "shift+e", label: "Restore selection" }, { combo: "shift+r", label: "Repeat last command" },
  { combo: "shift+o", label: "Show/hide overlay" }, { combo: "shift+k", label: "Color picker" }, { combo: "shift+g", label: "Capture screen" }, { combo: "shift+h", label: "Capture image" },
  { combo: "shift+n", label: "New text window" }, { combo: "space", label: "Pan (hold)" }, { combo: "Escape", label: "Cancel / deselect" },
];
