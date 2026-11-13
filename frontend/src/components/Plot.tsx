import { useMemo, useState } from "react";
import type { PlotData } from "../types";
import { useStore } from "../store";
import { lutColor } from "../render/compositor";
import { fmt } from "./RangeSlider";

const PALETTE = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#f87171"];

export default function Plot({ data, height = 180 }: { data: PlotData; height?: number }) {
  const luts = useStore((s) => s.luts);
  const yMinPref = useStore((s) => s.prefs.plot_y_min);
  const yMaxPref = useStore((s) => s.prefs.plot_y_max);
  const [hover, setHover] = useState<number | null>(null);
  const W = 300;
  const H = height;
  const pad = { l: 42, r: 8, t: 8, b: 26 };
  const { xs, ymin, ymax, xmin, xmax, byX } = useMemo(() => {
    const xs = data.x;
    let ymin = Infinity;
    let ymax = -Infinity;
    let xmin = Infinity;
    let xmax = -Infinity;
    for (const x of xs) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
    }
    // A series may carry its own `x` (same length as its values, possibly shorter than data.x, which
    // is then the union): x → value per series, so hover and CSV never pair a value with the wrong x.
    const byX = data.series.map((s) => {
      const sxs = s.x ?? xs;
      const m = new Map<number, number>();
      for (let k = 0; k < s.values.length && k < sxs.length; k++) {
        const x = sxs[k];
        const v = s.values[k];
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
        m.set(x, v);
      }
      return m;
    });
    if (!isFinite(xmin)) {
      xmin = 0;
      xmax = 1;
    }
    if (!isFinite(ymin)) {
      ymin = 0;
      ymax = 1;
    }
    if (ymax === ymin) ymax = ymin + 1;
    // Preferences ▸ Plots: a fixed y range (min < max) overrides the automatic one
    if (Number.isFinite(yMinPref) && Number.isFinite(yMaxPref) && yMaxPref > yMinPref) {
      ymin = yMinPref;
      ymax = yMaxPref;
    }
    return { xs, ymin, ymax, xmin, xmax, byX };
  }, [data, yMinPref, yMaxPref]);
  const sx = (x: number) => pad.l + ((x - xmin) / (xmax - xmin || 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  /** Series colour: its own `color`, else its LUT's colour, else the palette. */
  const colorOf = (i: number, s: { lut?: string; color?: string }) => {
    if (s.color) return s.color;
    if (s.lut && s.lut !== "Grays" && luts[s.lut]) return lutColor(luts, s.lut);
    return PALETTE[i % PALETTE.length];
  };
  const yticks = 4;
  const csv = () => {
    const lines = [["x", ...data.series.map((s) => s.name)].join(",")];
    // one row per x of the union; a series without a sample at that x leaves its cell empty
    for (const x of xs) lines.push([x, ...byX.map((m) => m.get(x) ?? "")].join(","));
    navigator.clipboard.writeText(lines.join("\n"));
    useStore.getState().toast("Plot data copied as CSV", "success");
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!xs.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const fx = xmin + ((x - pad.l) / (W - pad.l - pad.r)) * (xmax - xmin);
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(xs[i] - fx);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    setHover(best);
  };
  return (
    <div className="plot-wrap">
      <div style={{ display: "flex", alignItems: "center" }}>
        <p className="plot-title" style={{ flex: 1 }}>{data.title}</p>
        <button className="btn sm" onClick={csv}>Copy CSV</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {Array.from({ length: yticks + 1 }, (_, i) => {
          const y = ymin + ((ymax - ymin) * i) / yticks;
          return (
            <g key={i}>
              <line x1={pad.l} x2={W - pad.r} y1={sy(y)} y2={sy(y)} stroke="var(--border)" strokeWidth={0.5} />
              <text x={pad.l - 4} y={sy(y) + 3} fontSize={8} textAnchor="end" fill="var(--text-dim)" fontFamily="var(--mono)">{fmt(y, 3)}</text>
            </g>
          );
        })}
        {[0, 0.5, 1].map((f) => {
          const x = xmin + (xmax - xmin) * f;
          return <text key={f} x={sx(x)} y={H - 10} fontSize={8} textAnchor="middle" fill="var(--text-dim)" fontFamily="var(--mono)">{fmt(x, 3)}</text>;
        })}
        <text x={(pad.l + W - pad.r) / 2} y={H - 1} fontSize={8} textAnchor="middle" fill="var(--text-muted)">{data.x_label}</text>
        {data.series.map((s, i) => {
          const sxs = s.x ?? xs;
          const pts: string[] = [];
          for (let k = 0; k < s.values.length && k < sxs.length; k++) pts.push(`${sx(sxs[k])},${sy(s.values[k])}`);
          return <polyline key={i} fill="none" stroke={colorOf(i, s)} strokeWidth={1.3} points={pts.join(" ")} />;
        })}
        {hover !== null && hover < xs.length && (
          <g>
            <line x1={sx(xs[hover])} x2={sx(xs[hover])} y1={pad.t} y2={H - pad.b} stroke="var(--text-dim)" strokeDasharray="2 2" strokeWidth={0.6} />
            <text x={Math.min(sx(xs[hover]) + 4, W - 70)} y={pad.t + 9} fontSize={8} fill="var(--text)" fontFamily="var(--mono)">
              x={fmt(xs[hover], 4)} {byX.map((m) => { const v = m.get(xs[hover]); return v === undefined ? "–" : fmt(v, 4); }).join(" / ")}
            </text>
          </g>
        )}
      </svg>
      {data.series.length > 1 && (
        <div className="legend">
          {data.series.map((s, i) => (
            <span key={i}><i className="sw" style={{ background: colorOf(i, s) }} />{s.name}</span>
          ))}
        </div>
      )}
    </div>
  );
}
