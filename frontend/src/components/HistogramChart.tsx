import { useState } from "react";
import type { HistogramData } from "../types";
import { fmt } from "./RangeSlider";

export default function HistogramChart({ data }: { data: HistogramData }) {
  const [log, setLog] = useState(false);
  const counts = data.counts;
  const n = counts.length;
  const W = 300;
  const H = 120;
  const pad = { l: 6, r: 6, t: 6, b: 16 };
  const maxc = Math.max(1, ...counts);
  const lo = data.edges ? data.edges[0] : data.min ?? 0;
  const hi = data.edges ? data.edges[data.edges.length - 1] : data.max ?? 255;
  const bw = (W - pad.l - pad.r) / n;
  const st = data.stats || {};
  return (
    <div className="plot-wrap">
      <div style={{ display: "flex", alignItems: "center" }}>
        <p className="plot-title" style={{ flex: 1 }}>{data.title || "Histogram"}</p>
        <label className="switch"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log</label>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {counts.map((c, i) => {
          const f = log ? Math.log1p(c) / Math.log1p(maxc) : c / maxc;
          const h = f * (H - pad.t - pad.b);
          return <rect key={i} x={pad.l + i * bw} y={H - pad.b - h} width={Math.max(bw, 0.6)} height={h} fill="var(--accent)" opacity={0.85} />;
        })}
        <text x={pad.l} y={H - 4} fontSize={8} fill="var(--text-dim)" fontFamily="var(--mono)">{fmt(lo, 4)}</text>
        <text x={W - pad.r} y={H - 4} fontSize={8} textAnchor="end" fill="var(--text-dim)" fontFamily="var(--mono)">{fmt(hi, 4)}</text>
      </svg>
      {data.stats && (
        <div className="kv small" style={{ marginTop: 6 }}>
          <span className="k">Count</span><span className="v">{st.count}</span>
          <span className="k">Mean</span><span className="v">{fmt(st.mean, 5)}</span>
          <span className="k">StdDev</span><span className="v">{fmt(st.std, 5)}</span>
          <span className="k">Min / Max</span><span className="v">{fmt(st.min, 5)} / {fmt(st.max, 5)}</span>
          <span className="k">Mode</span><span className="v">{fmt(st.mode, 5)}</span>
          <span className="k">Median</span><span className="v">{fmt(st.median, 5)}</span>
        </div>
      )}
    </div>
  );
}
