import { useEffect } from "react";
import { useStore } from "./store";
import { doAction } from "./actions";
import { comboFromEvent } from "./customShortcuts";
import type { Tool } from "./types";

/** The element that had the focus before F10 / Alt moved it to the menu bar (Esc on a closed menu goes back there, E5-17). */
let beforeMenubar: HTMLElement | null = null;

/** F10 / a tap on Alt (Option) moves the keyboard focus to the menu bar (TopBar.tsx handles the arrow keys from there). */
export function focusMenubar(): boolean {
  const el = document.querySelector<HTMLElement>('.menubar .menu-trigger[tabindex="0"]') || document.querySelector<HTMLElement>(".menubar .menu-trigger");
  if (!el) return false;
  const cur = document.activeElement as HTMLElement | null;
  if (cur && cur !== document.body && !cur.closest(".menubar")) beforeMenubar = cur;
  el.focus();
  return true;
}

/** Esc on the menu bar with no menu open: back to the image (the element focused before F10, else the canvas). */
export function leaveMenubar(): void {
  const prev = beforeMenubar;
  beforeMenubar = null;
  const target = prev && prev.isConnected && !prev.closest(".menubar") ? prev : document.querySelector<HTMLElement>(".canvas-wrap");
  if (target) target.focus();
  else (document.activeElement as HTMLElement | null)?.blur();
}

const TOOL_KEYS: Record<string, Tool> = { v: "pointer", g: "hand", "1": "rect", "2": "oval", "3": "polygon", "4": "freehand", "5": "line", "6": "polyline", "7": "point", "8": "wand", "9": "brush", z: "zoom", i: "picker", y: "text", u: "bucket", j: "arrow", q: "angle", x: "track" };

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
}

/** Brush size steps: fine below 10 px, coarser above. */
export function stepBrushSize(size: number, dir: 1 | -1): number {
  const step = size < 10 || (dir < 0 && size <= 10) ? 1 : size < 30 || (dir < 0 && size <= 30) ? 2 : 5;
  return Math.max(1, Math.min(500, size + dir * step));
}

export function useShortcuts() {
  useEffect(() => {
    // a bare Alt tap (press and release with nothing else in between) focuses the menu bar, like Windows / GTK apps
    let altArmed = false;
    const down = (e: KeyboardEvent) => {
      const st = useStore.getState();
      const mod = e.metaKey || e.ctrlKey;
      const key = e.key;
      const modalOpen = st.paletteOpen || !!st.fileDialog || st.helpOpen || st.prefsOpen || !!st.docsOpen || st.shortcutsOpen || !!st.confirm;
      if (key === "Alt") {
        altArmed = !e.repeat && !mod && !e.shiftKey && !modalOpen;
        return;
      }
      altArmed = false;
      if (key === "F10" && !mod && !e.altKey && !e.shiftKey && !modalOpen) { // ⇧F10 stays the context-menu key
        e.preventDefault();
        focusMenubar();
        return;
      }
      // ⌘R inside the script editor runs the script (ScriptPanel handles it); everywhere else the browser keeps its reload
      if (mod && !e.shiftKey && key.toLowerCase() === "r" && (e.target as HTMLElement | null)?.closest?.(".script-editor")) return;
      // user-defined shortcuts (Plugins ▸ Shortcuts) win over every built-in binding
      if (!modalOpen && st.customShortcuts.length && (!isTyping(e) || mod)) {
        const combo = comboFromEvent(e);
        const cs = combo ? st.customShortcuts.find((c) => c.combo === combo) : undefined;
        if (cs) {
          e.preventDefault();
          if (cs.op) st.invoke(cs.op, !st.ops[cs.op]?.params.length);
          else if (cs.action) doAction(cs.action, cs.arg);
          return;
        }
      }
      // global modifier shortcuts (work even while typing)
      if (mod && key.toLowerCase() === "k") { e.preventDefault(); st.setPaletteOpen(!st.paletteOpen); return; }
      if (mod && key.toLowerCase() === "o") { e.preventDefault(); doAction("open"); return; }
      if (mod && key.toLowerCase() === "s") { e.preventDefault(); doAction(e.shiftKey ? "saveAs" : "save"); return; }
      if (mod && !e.shiftKey && key.toLowerCase() === "p") { e.preventDefault(); doAction("print"); return; }
      if (mod && e.shiftKey && e.code === "Backslash") { e.preventDefault(); st.toggleSplitView(); return; }
      if (mod && key === "\\") { e.preventDefault(); doAction("inspector"); return; }
      if (mod && key === ",") { e.preventDefault(); st.setPrefsOpen(!st.prefsOpen); return; }
      // ⌘W closes the active image (asks when it is modified), ⇧⌘W closes all – File ▸ Close / Close All.
      // (Some browsers reserve ⌘W for their own tab; where the event reaches the page it is ours.)
      if (mod && !e.altKey && !modalOpen && key.toLowerCase() === "w") { e.preventDefault(); doAction(e.shiftKey ? "closeAll" : "close"); return; }
      // ⌘1 … ⌘9 select the n-th image tab (⌘9 = the last one); ⌃Tab / ⌃⇧Tab cycle where the browser lets them through
      if (mod && !e.shiftKey && !e.altKey && !modalOpen && /^[1-9]$/.test(key)) { e.preventDefault(); doAction("selectImage", Number(key)); return; }
      if (e.ctrlKey && !e.metaKey && !e.altKey && !modalOpen && key === "Tab") { e.preventDefault(); doAction(e.shiftKey ? "previousImage" : "nextImage"); return; }
      if (isTyping(e)) return;
      if (mod && key.toLowerCase() === "z") { e.preventDefault(); doAction(e.shiftKey ? "redo" : "undo"); return; }
      if (mod && key.toLowerCase() === "y") { e.preventDefault(); doAction("redo"); return; }
      if (mod && !e.shiftKey && key.toLowerCase() === "x" && st.ops["edit.cut"]) { e.preventDefault(); st.invoke("edit.cut", true); return; }
      if (mod && !e.shiftKey && key.toLowerCase() === "c") { e.preventDefault(); st.invoke("edit.copy", true); return; }
      if (mod && !e.shiftKey && key.toLowerCase() === "v") { e.preventDefault(); st.invoke("edit.paste", true); return; }
      if (mod && key.toLowerCase() === "a") { e.preventDefault(); st.invoke(e.shiftKey ? "selection.none" : "selection.all", true); return; }
      if (mod && e.shiftKey && key.toLowerCase() === "c") { e.preventDefault(); doAction("brightness"); return; }
      if (mod && e.shiftKey && key.toLowerCase() === "t") { e.preventDefault(); doAction("threshold"); return; }
      if (key === "Escape") {
        if (st.confirm) st.confirm.resolve("cancel", false);
        else if (st.docsOpen) st.setDocsOpen(null);
        else if (st.shortcutsOpen) st.setShortcutsOpen(false);
        else if (st.prefsOpen) st.setPrefsOpen(false);
        else if (st.colorPickerOpen) st.setColorPickerOpen(false);
        else if (st.helpOpen) st.setHelpOpen(false);
        else if (st.fileDialog) st.setFileDialog(null);
        else if (st.paletteOpen) st.setPaletteOpen(false);
        else if (st.sheet) st.closeSheet();
        else if (st.menuOpen) st.setMenuOpen(null);
        return;
      }
      if (modalOpen) return;
      // Guard on the store, not e.repeat: after a focus-loss reset the user may still be holding
      // Space, and the OS auto-repeat keydowns (repeat=true) must be able to re-arm panning.
      if (key === " ") { if (!st.spaceDown) st.setSpace(true); e.preventDefault(); return; }
      if (mod) return;
      if (st.sheet && !(e.target as HTMLElement)?.classList?.contains("canvas-wrap")) {
        // let the sheet own Enter/Esc; other keys still work
      }
      const low = key.toLowerCase();
      if (!e.shiftKey && TOOL_KEYS[low]) { st.setTool(TOOL_KEYS[low]); return; }
      if (key === "?" ) { st.setHelpOpen(true); return; }
      if (key === "+" || key === "=") { doAction("zoomIn"); return; }
      if (key === "-" || key === "_") { doAction("zoomOut"); return; }
      if (key === "0") { doAction("zoomFit"); return; }
      // brush size ([ ] while the brush is active); otherwise [ opens the script editor as in Fiji
      if (key === "[" && st.tool !== "brush") { doAction("scriptEditor"); return; }
      if (key === "[" || key === "]") {
        const size = stepBrushSize(st.brushSize, key === "]" ? 1 : -1);
        st.setBrushSize(size);
        if (st.tool !== "brush") st.toast(`Brush size ${size} px`, "info");
        return;
      }
      const img = st.activeImage();
      if (img) {
        if (key === "," || key === ".") { st.setPosition(img.id, { z: Math.max(0, Math.min(img.shape.z - 1, img.display.z + (key === "," ? -1 : 1))) }); return; }
        if (key === "<" || key === ">") { st.setPosition(img.id, { t: Math.max(0, Math.min(img.shape.t - 1, img.display.t + (key === "<" ? -1 : 1))) }); return; }
      }
      const cmd: Record<string, string> = { m: "analyze.measure", t: "selection.add_to_manager", h: "analyze.histogram", k: "analyze.plot_profile", a: "selection.all", A: "selection.none", b: "overlay.add", f: "edit.fill", X: "image.crop", D: "image.duplicate", I: "edit.invert", n: "file.new", S: "process.smooth" };
      const k2 = e.shiftKey ? key.toUpperCase() : low;
      if (k2 === "R") { e.preventDefault(); st.repeatLast(); return; }
      if (k2 === "O") { e.preventDefault(); doAction("toggleOverlay"); return; }
      if (k2 === "E") { e.preventDefault(); doAction("restoreSelection"); return; }
      // ImageJ: ⇧K Color Picker, ⇧G Capture Screen, ⇧H Capture Image, ⇧N New Text Window
      if (k2 === "K") { e.preventDefault(); doAction("colorPicker"); return; }
      if (k2 === "G") { e.preventDefault(); doAction("captureScreen"); return; }
      if (k2 === "H") { e.preventDefault(); doAction("captureImage"); return; }
      if (k2 === "N") { e.preventDefault(); doAction("newTextWindow"); return; }
      if (cmd[k2]) {
        e.preventDefault();
        const quick = ["m", "t", "h", "k", "a", "A", "b", "f", "X", "I", "S"].includes(k2);
        st.invoke(cmd[k2], quick);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === " ") useStore.getState().setSpace(false);
      if (e.key === "Alt") {
        const arm = altArmed;
        altArmed = false;
        // only a clean tap: not after ⌥-drag / ⌥-wheel (pointer use disarms it) and not with another key in between
        if (arm && !isTyping(e)) focusMenubar();
      }
    };
    const disarmAlt = () => {
      altArmed = false;
    };
    // The keyup of a held key goes to whichever window has focus when it is released and browsers
    // never synthesize one on focus loss (Cmd+Tab, an OS dialog, a dock click, a tab switch), so
    // release Space whenever this window loses focus.  Bubble phase on purpose: element blurs do
    // not bubble, so only a real window blur reaches this listener.
    const release = () => {
      if (useStore.getState().spaceDown) useStore.getState().setSpace(false);
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") release();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", release);
    window.addEventListener("blur", disarmAlt);
    window.addEventListener("pointerdown", disarmAlt, true);
    window.addEventListener("wheel", disarmAlt, { capture: true, passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", release);
      window.removeEventListener("blur", disarmAlt);
      window.removeEventListener("pointerdown", disarmAlt, true);
      window.removeEventListener("wheel", disarmAlt, { capture: true } as EventListenerOptions);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);
}
