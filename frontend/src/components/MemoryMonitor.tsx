import { useEffect, useState } from "react";
import { MEMORY_SAMPLE_MS, useStore } from "../store";

const MB = 1048576;

export function fmtBytes(b: number): string {
  if (b >= 1024 * MB) return (b / (1024 * MB)).toFixed(2) + " GB";
  if (b >= 100 * MB) return (b / MB).toFixed(0) + " MB";
  return (b / MB).toFixed(1) + " MB";
}

/** Commands that read a virtual stack plane by plane (lumen.core.lazy streaming) and never materialise it. */
const STREAMING_OPS = new Set(["stacks.z_project", "stacks.substack", "stacks.z_profile", "stacks.statistics", "image.duplicate", "file.bioformats_export", "file.export_sequence"]);
/** Images the materialise warning has already been shown for (once per image, per page load). */
const warnedImages = new Set<string>();

/**
 * Memory guardrail (data larger than RAM): when a pixel-modifying command starts on a virtual stack whose decoded
 * size exceeds the auto-virtual limit (`auto_virtual_bytes`, or any size when the limit is off), toast once that the
 * command is about to read the whole stack into memory.  Mounted by the Memory Monitor; mount it in App to warn
 * whether or not the monitor is on.
 */
export function useMaterialiseWarning(): void {
  const jobs = useStore((s) => s.jobs);
  useEffect(() => {
    const st = useStore.getState();
    const img = st.activeImage();
    if (!img || !img.virtual || warnedImages.has(img.id)) return;
    const limit = st.state?.auto_virtual_bytes ?? null;
    const bytes = img.full_bytes ?? 0;
    for (const job of Object.values(jobs)) {
      const op = st.ops[job.op];
      if (!op || op.kind === "analysis" || job.op.startsWith("file.") || STREAMING_OPS.has(job.op)) continue;
      if (limit !== null && bytes <= limit) continue;
      warnedImages.add(img.id);
      const ram = st.state?.memory_limit_bytes;
      st.toast(`${img.name} is a virtual stack: ${job.name} reads all ${fmtBytes(bytes)} into memory${ram ? ` (${fmtBytes(ram)} RAM)` : ""}`, "info");
      break;
    }
  }, [jobs]);
}

/** Consecutive samples further apart than this are drawn as a gap (server away, throttled timer). */
const GAP_MS = 3 * MEMORY_SAMPLE_MS;

/**
 * Plugins ▸ Utilities ▸ Monitor Memory: sparkline of the bytes held by open images and their undo
 * history.  The samples are taken by the store (startMemorySampling) every 2 s while the monitor
 * is on, whether or not this panel is visible; x is wall-clock time, so a pause in sampling shows
 * as a gap instead of being squeezed into one 2 s step.
 */
export default function MemoryMonitor() {
  const samples = useStore((s) => s.memorySamples);
  const limit = useStore((s) => s.state?.memory_limit_bytes);
  const cacheBytes = useStore((s) => s.state?.virtual_cache_bytes);
  const cacheFill = useStore((s) => s.state?.virtual_resident_bytes);
  const images = useStore((s) => s.state?.images);
  const virtualImages = (images ?? []).filter((im) => im.virtual);
  const [hover, setHover] = useState<number | null>(null);
  useMaterialiseWarning();

  const W = 300;
  const H = 54;
  const pad = { l: 2, r: 2, t: 6, b: 3 };
  const plotW = W - pad.l - pad.r;
  const base = H - pad.b;
  const vals = samples.map((s) => s.bytes);
  const n = vals.length;
  const peak = Math.max(1, ...vals);
  const t0 = n ? samples[0].t : 0;
  const tn = n ? samples[n - 1].t : 0;
  const sx = (i: number) => (n > 1 && tn > t0 ? pad.l + ((samples[i].t - t0) / (tn - t0)) * plotW : W / 2);
  const sy = (v: number) => pad.t + (1 - v / peak) * (H - pad.t - pad.b);
  const pt = (i: number) => `${sx(i).toFixed(1)},${sy(vals[i]).toFixed(1)}`;
  // runs of samples with no gap between them: each run is its own line + area, the gaps are shaded
  const runs: number[][] = [];
  for (let i = 0; i < n; i++) {
    if (i === 0 || samples[i].t - samples[i - 1].t > GAP_MS) runs.push([i]);
    else runs[runs.length - 1].push(i);
  }
  const gaps = runs.slice(1).map((r) => ({ from: r[0] - 1, to: r[0] }));
  const shown = hover !== null && hover < n ? hover : n - 1;
  const span = n > 1 ? (tn - t0) / 1000 : 0;

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (n < 2) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    // nearest sample by time: x is a time axis, so the index is not linear in x
    const tt = t0 + ((x - pad.l) / plotW) * (tn - t0);
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(samples[i].t - tt) < Math.abs(samples[best].t - tt)) best = i;
    setHover(best);
  };

  return (
    <div className="mem-monitor">
      <div className="mem-head">
        <span className="mem-val">{n ? fmtBytes(vals[shown]) : "—"}</span>
        <span className="dim small">{hover !== null && n ? `${Math.round((samples[shown].t - samples[n - 1].t) / 1000)} s` : "images + undo history"}</span>
        <span className="grow" />
        <span className="dim small">peak {n ? fmtBytes(peak) : "—"}{span > 0 ? ` · ${Math.round(span)} s` : ""}{gaps.length ? ` · ${gaps.length} gap${gaps.length > 1 ? "s" : ""}` : ""}{limit ? ` · ${fmtBytes(limit)} RAM` : ""}</span>
        <button className="btn sm" onClick={() => useStore.setState({ memorySamples: [] })} disabled={!n}>Clear</button>
        <button className="btn sm" onClick={() => useStore.getState().setMemoryMonitor(false)}>Stop</button>
      </div>
      <svg className="mem-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <line x1={pad.l} x2={W - pad.r} y1={base} y2={base} stroke="var(--border-strong)" strokeWidth={1} />
        {gaps.map((g) => (
          <rect key={g.to} x={sx(g.from)} y={pad.t} width={Math.max(0, sx(g.to) - sx(g.from))} height={base - pad.t} fill="var(--text-dim)" opacity={0.12}>
            <title>{`no samples for ${Math.round((samples[g.to].t - samples[g.from].t) / 1000)} s`}</title>
          </rect>
        ))}
        {runs.map((r) =>
          r.length > 1 ? (
            <g key={r[0]}>
              <path d={`M${sx(r[0]).toFixed(1)},${base} L${r.map(pt).join(" L")} L${sx(r[r.length - 1]).toFixed(1)},${base} Z`} fill="var(--accent)" opacity={0.14} />
              <polyline points={r.map(pt).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            </g>
          ) : (
            <circle key={r[0]} cx={sx(r[0])} cy={sy(vals[r[0]])} r={2} fill="var(--accent)" vectorEffect="non-scaling-stroke" />
          ),
        )}
        {n >= 1 && <circle cx={sx(shown)} cy={sy(vals[shown])} r={3} fill="var(--accent)" stroke="var(--bg1)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
        {hover !== null && n > 1 && <line x1={sx(shown)} x2={sx(shown)} y1={pad.t} y2={base} stroke="var(--text-dim)" strokeDasharray="2 2" strokeWidth={1} vectorEffect="non-scaling-stroke" />}
      </svg>
      {n < 2 && <div className="hint-text">Sampling every {MEMORY_SAMPLE_MS / 1000} s…</div>}
      {virtualImages.length > 0 && (
        <ul className="mem-virtual" aria-label="Virtual stacks">
          {virtualImages.map((im) => (
            <li key={im.id} title={im.virtual_source ? `${im.virtual_source} – planes are read on demand; the first pixel-modifying command reads the whole stack into memory` : undefined}>
              <span className="mem-virtual-tag">virtual</span>
              <span className="mem-virtual-name">{im.name}</span>
              <span className="dim">{fmtBytes(im.resident_bytes ?? im.nbytes)} resident of {fmtBytes(im.full_bytes ?? 0)}{im.virtual_source ? ` · ${im.virtual_source}` : ""}</span>
            </li>
          ))}
          {typeof cacheBytes === "number" && (
            <li className="dim small">plane cache {fmtBytes(cacheFill ?? 0)} of {fmtBytes(cacheBytes)} (Preferences ▸ virtual_cache_mb)</li>
          )}
        </ul>
      )}
    </div>
  );
}
