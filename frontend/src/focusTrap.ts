/**
 * Focus management for modal and dialog-like surfaces (docs/USABILITY.md M24–M26, WCAG 2.4.3):
 *
 *   const ref = useRef<HTMLDivElement>(null);
 *   const trap = useFocusTrap(ref, { onEscape: close });
 *   <div ref={ref} role="dialog" aria-modal="true" aria-labelledby="…" tabIndex={-1} onKeyDown={trap.onKeyDown}>
 *
 * On mount the hook remembers the element that opened the surface (the last focused element outside it
 * that is still in the document – a menu trigger, the toolbar "?" button, the canvas; the ⌘K palette input
 * is unmounted by then, so the element focused before the palette counts) and moves focus to the first
 * meaningful control.  Tab / Shift-Tab cycle inside the surface, Escape calls `onEscape`, and on unmount the
 * opener gets the focus back (falling back to the most recently focused element that still exists, then the
 * image canvas) instead of dropping it to `body`.
 *
 * The keydown handler is returned (not attached natively) so inner React handlers run first and can stop
 * propagation: a filter box that clears its text on Escape keeps the dialog open.
 */
import { useCallback, useEffect, useRef, type RefObject } from "react";
import type React from "react";

export interface FocusTrapOptions {
  /** Called on Escape (the event is consumed); omit to leave Escape to the global shortcut chain. */
  onEscape?: () => void;
  /**
   * Where focus goes when the surface opens: "first" (default) = the first tabbable control outside the head bar
   * (`.dialog-head`, `.sheet-head`), "container" = the surface itself (long reading content: no scroll jump),
   * a CSS selector, a function returning the element, or false = leave focus where it is.
   */
  initialFocus?: "first" | "container" | string | (() => HTMLElement | null | undefined) | false;
  /** Hand focus back to the opener when the surface unmounts (default true). */
  returnFocus?: boolean;
  /** Cycle Tab / Shift-Tab inside the surface (default true). */
  trap?: boolean;
  /** Stop every other key from bubbling to the global shortcut handler (single-letter tool keys, Space …). */
  isolate?: boolean;
  /** Re-run the initial focus when this changes (the operation sheet switches commands without unmounting). */
  resetKey?: unknown;
}

const CANDIDATES = 'a[href], button, input, select, textarea, [tabindex], [contenteditable="true"]';

/** Elements inside `root` that Tab reaches, in DOM order (disabled, hidden and tabindex=-1 ones excluded). */
export function tabbableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(CANDIDATES)).filter((el) => {
    if (el.tabIndex < 0) return false;
    if ((el as HTMLButtonElement | HTMLInputElement).disabled) return false;
    if (el instanceof HTMLInputElement && el.type === "hidden") return false;
    if (el.hidden || el.getAttribute("aria-hidden") === "true") return false;
    // display:none (or an ancestor) → no box; position:fixed also has no offsetParent but is rendered
    if (el.offsetParent === null && getComputedStyle(el).position !== "fixed") return false;
    return true;
  });
}

// ------------------------------------------------------------------ document-level bookkeeping
/** Recently focused elements, oldest first (capped); body and the document itself are never recorded. */
const recent: HTMLElement[] = [];
/** True while the last user input was a key press: a surface opened from the keyboard takes focus, one opened with the mouse does not steal it. */
let keyboard = false;
let installed = false;

function install(): void {
  if (installed || typeof document === "undefined") return;
  installed = true;
  document.addEventListener(
    "focusin",
    (e) => {
      const t = e.target;
      if (!(t instanceof HTMLElement) || t === document.body || t === document.documentElement) return;
      const i = recent.indexOf(t);
      if (i >= 0) recent.splice(i, 1);
      recent.push(t);
      if (recent.length > 16) recent.shift();
    },
    true,
  );
  document.addEventListener("keydown", () => { keyboard = true; }, true);
  document.addEventListener("pointerdown", () => { keyboard = false; }, true);
}
install();

/** Whether the most recent user input was a key press (palette ⏎, menu ⏎, a shortcut) rather than a pointer. */
export function wasKeyboard(): boolean {
  return keyboard;
}

function usable(el: HTMLElement | null | undefined): el is HTMLElement {
  return !!el && el.isConnected && typeof el.focus === "function" && !(el as HTMLButtonElement).disabled && (el.offsetParent !== null || getComputedStyle(el).position === "fixed");
}

/** The element that opened `container`: the newest recently focused element outside it that still exists. */
function findOpener(container: HTMLElement): HTMLElement | null {
  for (let i = recent.length - 1; i >= 0; i--) {
    const el = recent[i];
    if (!el.isConnected || container.contains(el)) continue;
    return el;
  }
  const active = document.activeElement;
  return active instanceof HTMLElement && active !== document.body && !container.contains(active) ? active : null;
}

/** Give focus back after a surface closed: the opener, else the newest surviving recent element, else the canvas. */
function restoreFocus(opener: HTMLElement | null, closed: HTMLElement | null): void {
  // a surface that opened on top of us and already took the focus (a nested dialog) keeps it
  const active = document.activeElement;
  if (active instanceof HTMLElement && active !== document.body && active.isConnected && (!closed || !closed.contains(active)) && active.closest('[role="dialog"], [role="alertdialog"]')) return;
  const candidates: (HTMLElement | null)[] = [opener];
  for (let i = recent.length - 1; i >= 0; i--) candidates.push(recent[i]);
  candidates.push(document.querySelector<HTMLElement>(".canvas-wrap"));
  for (const el of candidates) {
    if (!usable(el) || (closed && closed.contains(el))) continue;
    el.focus({ preventScroll: el.classList.contains("canvas-wrap") });
    return;
  }
}

function moveInitialFocus(container: HTMLElement, how: FocusTrapOptions["initialFocus"]): void {
  if (how === false) return;
  let target: HTMLElement | null | undefined;
  if (typeof how === "function") target = how();
  else if (typeof how === "string" && how !== "first" && how !== "container") target = container.querySelector<HTMLElement>(how);
  if (!target && how !== "container") {
    const all = tabbableIn(container);
    target = all.find((el) => !el.closest(".dialog-head, .sheet-head")) ?? all[0];
  }
  if (!usable(target)) target = container;
  if (target === container && container.tabIndex < 0 && !container.hasAttribute("tabindex")) container.tabIndex = -1;
  target.focus(target === container ? { preventScroll: true } : undefined);
}

// ------------------------------------------------------------------ the hook
export function useFocusTrap<T extends HTMLElement>(ref: RefObject<T>, opts: FocusTrapOptions = {}): { onKeyDown: (e: React.KeyboardEvent) => void } {
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const opener = useRef<HTMLElement | null>(null);
  const container = useRef<HTMLElement | null>(null);
  install();

  // open (and every resetKey change): remember who opened us, move focus in.  The cleanup hands the focus back when
  // the surface is gone – on unmount, and also when a host that stays mounted stops rendering it (the 3-D large view
  // and the script editor's large view render null while closed and key the hook on their open flag, E7-19 / E7-14);
  // a resetKey change that keeps the same node (the operation sheet switching commands) restores nothing.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    container.current = el;
    const who = findOpener(el);
    if (who || !opener.current) opener.current = who;
    moveInitialFocus(el, optsRef.current.initialFocus);
    return () => {
      if (optsRef.current.returnFocus === false) return;
      const cur = ref.current;
      if (!cur || cur !== el || !cur.isConnected) restoreFocus(opener.current, el);
    };
  }, [opts.resetKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const o = optsRef.current;
      const el = ref.current;
      if (e.key === "Escape" && o.onEscape) {
        if (!e.defaultPrevented) {
          e.preventDefault();
          e.stopPropagation(); // the global Esc chain must not close a second surface
          o.onEscape();
        }
        return;
      }
      if (e.key === "Tab" && o.trap !== false && el) {
        const nodes = tabbableIn(el);
        e.preventDefault();
        e.stopPropagation();
        if (!nodes.length) {
          el.focus({ preventScroll: true });
          return;
        }
        const i = nodes.indexOf(document.activeElement as HTMLElement);
        const next = e.shiftKey ? (i <= 0 ? nodes.length - 1 : i - 1) : i < 0 || i === nodes.length - 1 ? 0 : i + 1;
        nodes[next].focus();
        return;
      }
      if (o.isolate) e.stopPropagation();
    },
    [ref],
  );
  return { onKeyDown };
}
