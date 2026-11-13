/**
 * The window's half of the AI-assistant bridge (server: lumen/server/extra/agent.py, assistant: lumen.mcp_server).
 *
 * - `applyAgentCommand` applies a `{"type": "ui"}` event from /api/events: set a selection, the tool, the plane, open
 *   a panel, show the result of a command the assistant ran, open an operation sheet for the user to review, …
 * - `startAgentBridge` reports what the window shows (active image, tool, selections, plane, panels, sheet, theme) to
 *   POST /api/ui/state whenever it changes (debounced) and every 20 s, so the assistant sees the user's screen.
 */
import { CLIENT_ID } from "./api";
import { showRunResult, useStore, type PanelKey } from "./store";
import { TOOLS } from "./components/ToolRail";
import { viewportRef } from "./actions";
import type { Tool } from "./types";

interface UiEvent {
  type: "ui";
  action: string;
  args?: Record<string, any>;
  seq?: number;
}

let seqApplied = 0;
let lastError: string | null = null;

function imageIdOf(args: Record<string, any>): string | null {
  const st = useStore.getState();
  return (args.image_id as string) || st.state?.active || null;
}

export async function applyAgentCommand(evt: UiEvent): Promise<void> {
  const st = useStore.getState();
  const a = evt.args || {};
  try {
    switch (evt.action) {
      case "apply_result": {
        await st.refresh(); // the images the command made must exist before its selection / table lands on them
        showRunResult(a.result || {}, (a.image_id as string) || null);
        if (a.result?.message && !a.silent) st.toast(a.result.message, "info");
        break;
      }
      case "set_selection": {
        const id = imageIdOf(a);
        if (!id) throw new Error("no image");
        if (!st.image(id)) await st.refresh();
        useStore.getState().setSelection(id, a.roi ?? null);
        break;
      }
      case "set_tool": {
        const tool = String(a.tool || "");
        const ok = TOOLS.some((t) => t.id === tool) || (tool.startsWith("op:") && !!st.ops[tool.slice(3)]?.tool);
        if (!ok) throw new Error(`unknown tool ${tool}`);
        st.setTool(tool as Tool);
        break;
      }
      case "set_position": {
        const id = imageIdOf(a);
        if (!id) throw new Error("no image");
        const pos: { t?: number; z?: number; c?: number } = {};
        for (const k of ["t", "z", "c"] as const) if (typeof a[k] === "number") pos[k] = a[k];
        st.setPosition(id, pos);
        await st.flushDisplay();
        break;
      }
      case "select_image": {
        if (!st.image(a.image_id)) await st.refresh();
        useStore.getState().selectImage(String(a.image_id));
        break;
      }
      case "panel": {
        const key = String(a.panel) as PanelKey;
        if (!(key in st.panels)) throw new Error(`unknown panel ${key}`);
        st.setInspectorOpen(true);
        st.togglePanel(key, a.open === undefined ? true : !!a.open);
        break;
      }
      case "notify":
        st.toast(String(a.message || ""), a.kind === "error" ? "error" : a.kind === "success" ? "success" : "info", { timeout: 8000 });
        break;
      case "open_sheet": {
        const op = st.ops[String(a.op)];
        if (!op) throw new Error(`unknown command ${a.op}`);
        st.openSheet(op, a.params || {});
        break;
      }
      case "palette":
        st.setPaletteOpen(a.open === undefined ? true : !!a.open);
        break;
      case "theme":
        st.setTheme(a.theme === "light" ? "light" : "dark");
        break;
      case "view":
        if (a.fit) viewportRef.current?.fit();
        else if (typeof a.zoom === "number" && a.zoom > 0) viewportRef.current?.setScale(a.zoom);
        break;
      default:
        throw new Error(`unknown action ${evt.action}`);
    }
    lastError = null;
  } catch (e: any) {
    lastError = `${evt.action}: ${e?.message || e}`;
  }
  if (evt.seq) seqApplied = Math.max(seqApplied, evt.seq);
  report(true);
}

function snapshot() {
  const st = useStore.getState();
  const images = st.state?.images || [];
  const active = images.find((im) => im.id === st.state?.active);
  const selections: Record<string, any> = {};
  for (const im of images) if (st.selections[im.id]) selections[im.id] = st.selections[im.id];
  return {
    client: CLIENT_ID,
    active_image: active ? { id: active.id, name: active.name, t: active.display.t, z: active.display.z, c: active.display.c } : null,
    tool: st.tool,
    selections,
    selected_roi_ids: st.selectedRoiIds,
    panels_open: Object.entries(st.panels).filter(([, v]) => v).map(([k]) => k),
    sheet: st.sheet ? { op: st.sheet.op.id, name: st.sheet.op.name, params: st.sheet.params, error: st.sheet.error ?? null } : null,
    palette_open: st.paletteOpen,
    theme: st.theme,
    zoom: active && st.views[active.id] ? Math.round(st.views[active.id].scale * 1000) / 1000 : null,
    split: st.split.on ? { other: st.split.other } : null,
    cursor: st.cursor,
    seq_applied: seqApplied,
    last_error: lastError,
  };
}

let lastSent = "";
let timer: number | null = null;

function report(now = false) {
  const send = () => {
    timer = null;
    const body = JSON.stringify(snapshot());
    if (body === lastSent && !now) return;
    lastSent = body;
    fetch("/api/ui/state", { method: "POST", headers: { "Content-Type": "application/json" }, body }).catch(() => undefined);
  };
  if (now) {
    if (timer) window.clearTimeout(timer);
    send();
  } else if (!timer) {
    timer = window.setTimeout(send, 300);
  }
}

let started = false;

export function startAgentBridge() {
  if (started) return;
  started = true;
  useStore.subscribe((s, p) => {
    // the cursor readout changes on every mouse move: report it at most with the debounce, never on its own
    if (s.state !== p.state || s.tool !== p.tool || s.selections !== p.selections || s.panels !== p.panels || s.sheet !== p.sheet || s.theme !== p.theme || s.views !== p.views || s.paletteOpen !== p.paletteOpen || s.selectedRoiIds !== p.selectedRoiIds || s.split !== p.split) report();
  });
  window.setInterval(() => {
    lastSent = "";
    report();
  }, 20000);
  report(true);
}
