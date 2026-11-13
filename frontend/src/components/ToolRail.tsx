import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronsLeft, ChevronsRight, MousePointer2, Hand, Square, Circle, Hexagon, PenLine, Minus, Spline, LocateFixed, Wand2, ZoomIn, Pipette, Brush, PaintBucket, Type, MoveUpRight, DraftingCompass, Waypoints } from "lucide-react";
import { useStore } from "../store";
import type { Tool } from "../types";
import { kbd } from "../platform";
import ColorPicker, { valueToHex } from "./ColorPicker";

export const TOOLS: { id: Tool; icon: React.ReactNode; label: string; key: string; short: string }[] = [
  { id: "pointer", icon: <MousePointer2 size={16} />, label: "Select / move (V)", key: "V", short: "Select" },
  { id: "hand", icon: <Hand size={16} />, label: "Pan (Space or G)", key: "G", short: "Pan" },
  { id: "rect", icon: <Square size={16} />, label: "Rectangle (1)", key: "1", short: "Rectangle" },
  { id: "oval", icon: <Circle size={16} />, label: "Oval (2)", key: "2", short: "Oval" },
  { id: "polygon", icon: <Hexagon size={16} />, label: "Polygon (3) – click points, double-click to close", key: "3", short: "Polygon" },
  { id: "freehand", icon: <PenLine size={16} />, label: "Freehand (4)", key: "4", short: "Freehand" },
  { id: "line", icon: <Minus size={16} />, label: "Line (5)", key: "5", short: "Line" },
  { id: "polyline", icon: <Spline size={16} />, label: "Segmented line (6)", key: "6", short: "Segmented line" },
  { id: "point", icon: <LocateFixed size={16} />, label: "Points (7) – shift-click to add", key: "7", short: "Points" },
  { id: "wand", icon: <Wand2 size={16} />, label: "Wand (8) – trace a region", key: "8", short: "Wand" },
  { id: "angle", icon: <DraftingCompass size={16} />, label: "Angle (Q) – click arm end, vertex, arm end", key: "Q", short: "Angle" },
  { id: "track", icon: <Waypoints size={16} />, label: "Manual tracking (X) – click adds this frame's point and advances, ⌥-click deletes it", key: "X", short: "Manual tracking" },
  { id: "brush", icon: <Brush size={16} />, label: "Brush (9) – paint with the foreground value; [ ] size, ⌥-drag erases", key: "9", short: "Brush" },
  { id: "bucket", icon: <PaintBucket size={16} />, label: "Bucket fill (U) – flood-fill with the foreground value", key: "U", short: "Bucket fill" },
  { id: "text", icon: <Type size={16} />, label: "Text (Y) – click to add an overlay label", key: "Y", short: "Text" },
  { id: "arrow", icon: <MoveUpRight size={16} />, label: "Arrow (J) – drag to add an overlay arrow", key: "J", short: "Arrow" },
  { id: "zoom", icon: <ZoomIn size={16} />, label: "Zoom (Z) – click in, alt-click out", key: "Z", short: "Zoom" },
  { id: "picker", icon: <Pipette size={16} />, label: "Value picker (I)", key: "I", short: "Value picker" },
];

/** Group headings of the sidebar, drawn before these tools (MolBench-style labelled sections). */
const GROUP_BEFORE: Partial<Record<Tool, string>> = { pointer: "Navigate", rect: "Selection", brush: "Annotate", zoom: "Inspect" };

/** Collapsed = the compact icon column; remembered per browser (a convenience, so storage may fail silently). */
const RAIL_KEY = "lumen-rail-collapsed";
const RAIL_W = { expanded: "184px", collapsed: "52px" };
function loadCollapsed(): boolean {
  try {
    return localStorage.getItem(RAIL_KEY) === "1";
  } catch {
    return false;
  }
}

/** The current foreground / background as hex swatches (the colour kept by the picker, else the value's grey). */
function useFgBg(): { fg: number; bg: number; fgHex: string; bgHex: string } {
  const settings = useStore((s) => s.settings);
  const prefs = useStore((s) => s.prefs);
  const img = useStore((s) => s.activeImage());
  const fg = Number(settings.foreground ?? prefs.foreground ?? 255);
  const bg = Number(settings.background ?? prefs.background ?? 0);
  const hex = (v: unknown, fallback: number) => (typeof v === "string" && /^#[0-9a-fA-F]{6}$/.test(v) ? v : valueToHex(fallback, img));
  return { fg, bg, fgHex: hex(settings.foreground_color, fg), bgHex: hex(settings.background_color, bg) };
}

export default function ToolRail() {
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const ops = useStore((s) => s.meta?.ops);
  // registered plugin tools (@op(..., tool=True)) follow the registry: Reload Plugins refetches /api/meta
  const pluginTools = useMemo(() => (ops || []).filter((o) => o.tool), [ops]);
  const pickerOpen = useStore((s) => s.colorPickerOpen);
  const setPickerOpen = useStore((s) => s.setColorPickerOpen);
  const { fg, bg, fgHex, bgHex } = useFgBg();
  const [collapsed, setCollapsed] = useState<boolean>(loadCollapsed);
  useEffect(() => {
    document.documentElement.style.setProperty("--rail-w", collapsed ? RAIL_W.collapsed : RAIL_W.expanded);
    try {
      localStorage.setItem(RAIL_KEY, collapsed ? "1" : "0");
    } catch {
      /* storage blocked: the choice lasts for this page */
    }
    window.dispatchEvent(new Event("resize")); // the viewport re-fits to the new width
  }, [collapsed]);
  const swRef = useRef<HTMLButtonElement>(null);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  useEffect(() => {
    if (pickerOpen) setAnchor(swRef.current?.getBoundingClientRect() ?? null);
  }, [pickerOpen]);
  useEffect(() => {
    if (!pickerOpen) return;
    const close = (e: PointerEvent) => {
      const t = e.target as Element | null;
      if (t?.closest?.(".color-picker") || t?.closest?.(".swatches")) return;
      setPickerOpen(false);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [pickerOpen, setPickerOpen]);
  const swatchLabel = `Foreground ${fg} / background ${bg} – Color Picker (${kbd("⇧K")})`;
  // WAI-ARIA toolbar: one Tab stop (the active tool, else the first); ↑ ↓ ← → Home End move between the tools (E5-16)
  const railRef = useRef<HTMLDivElement>(null);
  const toolIds: Tool[] = [...TOOLS.map((t) => t.id), ...pluginTools.map((o) => `op:${o.id}` as Tool)];
  const focusIdx = Math.max(0, toolIds.indexOf(tool));
  const onRailKey = (e: React.KeyboardEvent) => {
    const buttons = Array.from(railRef.current?.querySelectorAll<HTMLButtonElement>("button.tool") || []);
    if (!buttons.length) return;
    const cur = buttons.indexOf(document.activeElement as HTMLButtonElement);
    let next = -1;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") next = (cur + 1) % buttons.length;
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = (cur - 1 + buttons.length) % buttons.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = buttons.length - 1;
    else return;
    e.preventDefault();
    e.stopPropagation();
    buttons[next]?.focus();
  };
  return (
    <div className={"rail" + (collapsed ? " collapsed" : "")} role="toolbar" aria-label="Tools" aria-orientation="vertical" ref={railRef} onKeyDown={onRailKey}>
      {TOOLS.map((t, i) => (
        <span key={t.id} className="rail-item">
          {GROUP_BEFORE[t.id] && (collapsed ? <div className="rail-sep" role="separator" /> : <div className="rail-group" role="presentation">{GROUP_BEFORE[t.id]}</div>)}
          <button className={"tool" + (tool === t.id ? " active" : "")} title={t.label} aria-label={t.label} aria-pressed={tool === t.id} tabIndex={i === focusIdx ? 0 : -1} onClick={() => setTool(t.id)}>
            <span className="tool-icon" aria-hidden="true">{t.icon}</span>
            <span className="tool-label" aria-hidden="true">{t.short}</span>
            <span className="key" aria-hidden="true">{t.key}</span>
          </button>
        </span>
      ))}
      {pluginTools.length > 0 && (collapsed ? <div className="rail-sep" role="separator" /> : <div className="rail-group" role="presentation">Plugin tools</div>)}
      {pluginTools.map((o, k) => {
        const id = `op:${o.id}` as Tool;
        const label = `${o.name} (plugin tool) – ${o.description}`;
        return (
          <button key={o.id} className={"tool plugin" + (tool === id ? " active" : "")} title={label} aria-label={o.name} aria-pressed={tool === id} tabIndex={TOOLS.length + k === focusIdx ? 0 : -1} onClick={() => setTool(id)}>
            <span className="tool-icon plugin-icon" aria-hidden="true" style={o.tool_color ? { color: o.tool_color } : undefined}>{o.tool_icon || "⚙"}</span>
            <span className="tool-label" aria-hidden="true">{o.name}</span>
          </button>
        );
      })}
      <div className="grow" />
      <div className="rail-foot">
        <div className="swatches">
          <button ref={swRef} className={"swatch-btn" + (pickerOpen ? " active" : "")} title={swatchLabel} aria-label={swatchLabel} aria-expanded={pickerOpen} aria-haspopup="dialog" onClick={() => setPickerOpen(!pickerOpen)}>
            <span className="swatch bg" style={{ background: bgHex }} />
            <span className="swatch fg" style={{ background: fgHex }} />
          </button>
          {!collapsed && <span className="swatch-label" aria-hidden="true">Colours <span className="dim">{fg} / {bg}</span></span>}
        </div>
        <button className="rail-collapse" onClick={() => setCollapsed(!collapsed)} title={collapsed ? "Show tool names" : "Collapse to icons (more room for the image)"} aria-label={collapsed ? "Show tool names" : "Collapse the tool bar to icons"} aria-pressed={collapsed}>
          {collapsed ? <ChevronsRight size={15} /> : <><ChevronsLeft size={15} /><span className="tool-label">Collapse</span></>}
        </button>
      </div>
      {pickerOpen && <ColorPicker anchor={anchor} onClose={() => setPickerOpen(false)} />}
    </div>
  );
}
