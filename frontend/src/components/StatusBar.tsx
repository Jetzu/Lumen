import { useStore } from "../store";
import { fmt } from "./RangeSlider";
import { kbd } from "../platform";
import { doAction } from "../actions";
import { fmtBytes } from "./MemoryMonitor";
import { pluginToolId } from "../types";

/** The active plugin tool (tool rail), as a chip naming it. */
function ToolChip() {
  const tool = useStore((s) => s.tool);
  const pid = pluginToolId(tool);
  const spec = useStore((s) => (pid ? s.ops[pid] : undefined));
  if (!pid) return null;
  return <span className="pill tool-chip" title={spec ? `${spec.description} – click or drag on the image` : "plugin tool"}>{spec?.tool_icon || "⚙"} {spec?.name || pid}</span>;
}

/** Foreground / background values (Fill, Clear, brush …); click opens the Color Picker. */
function ColorChip() {
  const settings = useStore((s) => s.settings);
  const prefs = useStore((s) => s.prefs);
  const fg = settings.foreground ?? prefs.foreground;
  const bg = settings.background ?? prefs.background;
  const label = `Foreground ${fg} · background ${bg} – open the Color Picker (${kbd("⇧K")})`;
  return (
    <span className="pill color-chip" role="button" tabIndex={0} aria-label={label} title={label} onClick={() => doAction("colorPicker")} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doAction("colorPicker"); } }}>
      fg {fmt(Number(fg), 5)} · bg {fmt(Number(bg), 5)}
    </span>
  );
}

function MemoryChip() {
  const mem = useStore((s) => s.state?.memory_bytes);
  const on = useStore((s) => s.memoryMonitor);
  if (typeof mem !== "number") return null;
  return (
    <span className={"pill mem" + (on ? " on" : "")} role="button" tabIndex={0} aria-pressed={on} aria-label={`Memory ${fmtBytes(mem)} – toggle the memory monitor`} title="Memory held by open images and undo history · click to toggle the memory monitor (Plugins ▸ Utilities)" onClick={() => doAction("toggleMemoryMonitor")} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doAction("toggleMemoryMonitor"); } }}>
      mem {fmtBytes(mem)}
    </span>
  );
}

export default function StatusBar() {
  const img = useStore((s) => s.activeImage());
  const cursor = useStore((s) => s.cursor);
  const view = useStore((s) => (img ? s.views[img.id] : undefined));
  const sel = useStore((s) => (img ? s.selections[img.id] : null));
  const meta = useStore((s) => s.meta);
  // not a live region (E7-19): the cursor readout changes on every mouse move; messages are announced by the toasts
  if (!img) return <div className="statusbar" role="contentinfo" aria-label="Status"><span>Lumen {meta?.version}</span><MemoryChip /><span className="spacer" /><span className="dim">Open an image to begin · {kbd("⌘K")} for commands</span></div>;
  const cal = img.calibration;
  const scaled = cal.unit && cal.unit !== "pixel";
  const mb = (img.nbytes / 1048576).toFixed(1);
  let selInfo = "";
  if (sel) {
    if (sel.type === "rect" || sel.type === "oval" || sel.type === "mask") selInfo = `${sel.type} x=${fmt(sel.x!, 5)} y=${fmt(sel.y!, 5)} w=${fmt(sel.w!, 5)} h=${fmt(sel.h!, 5)}`;
    else if (sel.type === "line") {
      const L = Math.hypot((sel.x2! - sel.x1!) * cal.pixel_width, (sel.y2! - sel.y1!) * cal.pixel_height);
      const ang = (Math.atan2((sel.y1! - sel.y2!) * cal.pixel_height, (sel.x2! - sel.x1!) * cal.pixel_width) * 180) / Math.PI;
      selInfo = `line length=${fmt(L, 5)}${scaled ? cal.unit : "px"} angle=${fmt(ang, 4)}°`;
    } else selInfo = `${sel.type} · ${sel.points?.length ?? 0} pts`;
  }
  return (
    <div className="statusbar" role="contentinfo" aria-label="Status">
      <span>{img.shape.x}×{img.shape.y} px{scaled ? ` (${fmt(img.shape.x * cal.pixel_width, 5)}×${fmt(img.shape.y * cal.pixel_height, 5)} ${cal.unit})` : ""} · {img.bit_depth} · {mb} MB</span>
      {cursor && (
        <span aria-live="off" aria-label="Cursor position and value">
          x={cursor.x} y={cursor.y}{scaled ? ` (${fmt(cursor.x * cal.pixel_width, 4)}, ${fmt(cursor.y * cal.pixel_height, 4)} ${cal.unit})` : ""}
          {cursor.values && <> · value={cursor.values.map((v) => (v === null ? "NaN" : fmt(v, 5))).join(", ")}</>}
        </span>
      )}
      {selInfo && <span className="pill">{selInfo}</span>}
      <ToolChip />
      <span className="spacer" />
      <ColorChip />
      <MemoryChip />
      {view && <span>{Math.round(view.scale * 100)}%</span>}
    </div>
  );
}
