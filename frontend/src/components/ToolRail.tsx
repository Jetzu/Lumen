import { useEffect, useMemo, useRef, useState } from "react";
import { MousePointer2, Hand, Square, Circle, Hexagon, PenLine, Minus, Spline, LocateFixed, Wand2, ZoomIn, Pipette, Brush, PaintBucket, Type, MoveUpRight, DraftingCompass, Waypoints } from "lucide-react";
import { useStore } from "../store";
import type { Tool } from "../types";
import { kbd } from "../platform";
import ColorPicker, { valueToHex } from "./ColorPicker";

export const TOOLS: { id: Tool; icon: React.ReactNode; label: string; key: string }[] = [
  { id: "pointer", icon: <MousePointer2 size={17} />, label: "Select / move (V)", key: "V" },
  { id: "hand", icon: <Hand size={17} />, label: "Pan (Space or G)", key: "G" },
  { id: "rect", icon: <Square size={16} />, label: "Rectangle (1)", key: "1" },
  { id: "oval", icon: <Circle size={16} />, label: "Oval (2)", key: "2" },
  { id: "polygon", icon: <Hexagon size={16} />, label: "Polygon (3) – click points, double-click to close", key: "3" },
  { id: "freehand", icon: <PenLine size={16} />, label: "Freehand (4)", key: "4" },
  { id: "line", icon: <Minus size={17} />, label: "Line (5)", key: "5" },
  { id: "polyline", icon: <Spline size={16} />, label: "Segmented line (6)", key: "6" },
  { id: "point", icon: <LocateFixed size={16} />, label: "Points (7) – shift-click to add", key: "7" },
  { id: "wand", icon: <Wand2 size={16} />, label: "Wand (8) – trace a region", key: "8" },
  { id: "angle", icon: <DraftingCompass size={16} />, label: "Angle (Q) – click arm end, vertex, arm end", key: "Q" },
  { id: "track", icon: <Waypoints size={16} />, label: "Manual tracking (X) – click adds this frame's point and advances, ⌥-click deletes it", key: "X" },
  { id: "brush", icon: <Brush size={16} />, label: "Brush (9) – paint with the foreground value; [ ] size, ⌥-drag erases", key: "9" },
  { id: "bucket", icon: <PaintBucket size={16} />, label: "Bucket fill (U) – flood-fill with the foreground value", key: "U" },
  { id: "text", icon: <Type size={16} />, label: "Text (Y) – click to add an overlay label", key: "Y" },
  { id: "arrow", icon: <MoveUpRight size={16} />, label: "Arrow (J) – drag to add an overlay arrow", key: "J" },
  { id: "zoom", icon: <ZoomIn size={16} />, label: "Zoom (Z) – click in, alt-click out", key: "Z" },
  { id: "picker", icon: <Pipette size={16} />, label: "Value picker (I)", key: "I" },
];

/** Separators are drawn before these tools. */
const SEP_BEFORE = new Set<Tool>(["rect", "brush", "zoom"]);

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
    <div className="rail" role="toolbar" aria-label="Tools" aria-orientation="vertical" ref={railRef} onKeyDown={onRailKey}>
      {TOOLS.map((t, i) => (
        <span key={t.id}>
          {SEP_BEFORE.has(t.id) && <div className="rail-sep" role="separator" />}
          <button className={"tool" + (tool === t.id ? " active" : "")} title={t.label} aria-label={t.label} aria-pressed={tool === t.id} tabIndex={i === focusIdx ? 0 : -1} onClick={() => setTool(t.id)}>
            {t.icon}
            <span className="key" aria-hidden="true">{t.key}</span>
          </button>
        </span>
      ))}
      {pluginTools.length > 0 && <div className="rail-sep" role="separator" />}
      {pluginTools.map((o, k) => {
        const id = `op:${o.id}` as Tool;
        const label = `${o.name} (plugin tool) – ${o.description}`;
        return (
          <button key={o.id} className={"tool plugin" + (tool === id ? " active" : "")} title={label} aria-label={o.name} aria-pressed={tool === id} tabIndex={TOOLS.length + k === focusIdx ? 0 : -1} onClick={() => setTool(id)}>
            <span className="plugin-icon" aria-hidden="true" style={o.tool_color ? { color: o.tool_color } : undefined}>{o.tool_icon || "⚙"}</span>
          </button>
        );
      })}
      <div className="grow" />
      <div className="swatches">
        <button ref={swRef} className={"swatch-btn" + (pickerOpen ? " active" : "")} title={swatchLabel} aria-label={swatchLabel} aria-expanded={pickerOpen} aria-haspopup="dialog" onClick={() => setPickerOpen(!pickerOpen)}>
          <span className="swatch bg" style={{ background: bgHex }} />
          <span className="swatch fg" style={{ background: fgHex }} />
        </button>
      </div>
      {pickerOpen && <ColorPicker anchor={anchor} onClose={() => setPickerOpen(false)} />}
    </div>
  );
}
