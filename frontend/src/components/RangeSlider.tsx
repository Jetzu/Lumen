import { useEffect, useRef } from "react";

interface Props {
  lo: number;
  hi: number;
  min: number;
  max: number;
  counts?: Float32Array | number[] | null;
  gradient?: string;
  color?: string;
  log?: boolean;
  onChange: (lo: number, hi: number, final: boolean) => void;
  overlay?: { lo: number; hi: number } | null;
  /** Accessible name of the slider group; the handles are announced as "<label> minimum/maximum". */
  label?: string;
}

export default function RangeSlider({ lo, hi, min, max, counts, gradient, color = "var(--accent)", log = true, onChange, overlay, label = "Range" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const span = max - min || 1e-9;
  const pct = (v: number) => Math.min(100, Math.max(0, ((v - min) / span) * 100));

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const w = cv.clientWidth || 280;
    const h = 40;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!counts || counts.length === 0) return;
    const n = counts.length;
    let maxc = 0;
    for (let i = 0; i < n; i++) if (counts[i] > maxc) maxc = counts[i];
    if (maxc <= 0) return;
    const style = getComputedStyle(document.documentElement);
    ctx.fillStyle = style.getPropertyValue("--text-dim").trim() || "#666";
    ctx.globalAlpha = 0.9;
    const bw = w / n;
    for (let i = 0; i < n; i++) {
      const v = counts[i];
      const f = log ? Math.log1p(v) / Math.log1p(maxc) : v / maxc;
      const bh = f * (h - 2);
      ctx.fillRect(i * bw, h - bh, Math.max(bw, 1), bh);
    }
    ctx.globalAlpha = 1;
    // overlay threshold band
    if (overlay) {
      ctx.fillStyle = "rgba(248,80,80,0.35)";
      ctx.fillRect((pct(overlay.lo) / 100) * w, 0, ((pct(overlay.hi) - pct(overlay.lo)) / 100) * w, h);
    }
    // display range line (min→max mapping)
    ctx.strokeStyle = color.startsWith("var") ? style.getPropertyValue("--accent").trim() : color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo((pct(lo) / 100) * w, h);
    ctx.lineTo((pct(hi) / 100) * w, 0);
    ctx.stroke();
  }, [counts, lo, hi, min, max, log, overlay?.lo, overlay?.hi, color]);

  const startDrag = (which: "lo" | "hi" | "both") => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const el = wrapRef.current!;
    const rect = el.getBoundingClientRect();
    const startX = e.clientX;
    const sLo = lo;
    const sHi = hi;
    const cur = { lo, hi };
    const emit = (a: number, b: number, final: boolean) => {
      cur.lo = a;
      cur.hi = b;
      onChange(a, b, final);
    };
    const move = (ev: PointerEvent) => {
      const dx = ((ev.clientX - startX) / rect.width) * span;
      if (which === "lo") emit(Math.min(Math.max(min, sLo + dx), sHi - span / 1000), sHi, false);
      else if (which === "hi") emit(sLo, Math.max(Math.min(max, sHi + dx), sLo + span / 1000), false);
      else {
        let a = sLo + dx;
        let b = sHi + dx;
        if (a < min) {
          b += min - a;
          a = min;
        }
        if (b > max) {
          a -= b - max;
          b = max;
        }
        emit(a, b, false);
      }
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      onChange(cur.lo, cur.hi, true); // commit the LAST dragged values, not the stale render-time ones
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // keyboard: arrows move a handle by 1 % of the range (10 % with Shift / PageUp / PageDown), Home / End jump to the ends
  const keyStep = (which: "lo" | "hi") => (e: React.KeyboardEvent) => {
    let frac: number;
    switch (e.key) {
      case "ArrowLeft":
      case "ArrowDown": frac = e.shiftKey ? -0.1 : -0.01; break;
      case "ArrowRight":
      case "ArrowUp": frac = e.shiftKey ? 0.1 : 0.01; break;
      case "PageDown": frac = -0.1; break;
      case "PageUp": frac = 0.1; break;
      case "Home": frac = -Infinity; break;
      case "End": frac = Infinity; break;
      default: return;
    }
    e.preventDefault();
    e.stopPropagation(); // arrows would otherwise nudge the selection (global shortcuts)
    const gap = span / 1000;
    if (which === "lo") {
      const v = frac === -Infinity ? min : frac === Infinity ? hi - gap : lo + frac * span;
      onChange(Math.min(Math.max(min, v), hi - gap), hi, true);
    } else {
      const v = frac === -Infinity ? lo + gap : frac === Infinity ? max : hi + frac * span;
      onChange(lo, Math.max(Math.min(max, v), lo + gap), true);
    }
  };

  const clickTrack = (e: React.MouseEvent) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const v = min + ((e.clientX - rect.left) / rect.width) * span;
    if (Math.abs(v - lo) < Math.abs(v - hi)) onChange(Math.min(v, hi), hi, true);
    else onChange(lo, Math.max(v, lo), true);
  };

  return (
    <div className="range-slider" ref={wrapRef} role="group" aria-label={label}>
      <canvas ref={canvasRef} onClick={clickTrack} aria-hidden="true" />
      <div className="track" onClick={clickTrack} aria-hidden="true" />
      <div className="fill" style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%`, background: gradient || color, cursor: "grab" }} onPointerDown={startDrag("both")} aria-hidden="true" />
      <div className="handle" role="slider" tabIndex={0} aria-label={`${label} minimum`} aria-orientation="horizontal" aria-valuemin={min} aria-valuemax={max} aria-valuenow={Math.round(lo * 1000) / 1000} aria-valuetext={fmt(lo)} style={{ left: `${pct(lo)}%` }} onPointerDown={startDrag("lo")} onKeyDown={keyStep("lo")} title={`Min ${fmt(lo)}`} />
      <div className="handle" role="slider" tabIndex={0} aria-label={`${label} maximum`} aria-orientation="horizontal" aria-valuemin={min} aria-valuemax={max} aria-valuenow={Math.round(hi * 1000) / 1000} aria-valuetext={fmt(hi)} style={{ left: `${pct(hi)}%` }} onPointerDown={startDrag("hi")} onKeyDown={keyStep("hi")} title={`Max ${fmt(hi)}`} />
    </div>
  );
}

export function fmt(v: number, digits = 4): string {
  if (!isFinite(v)) return "–";
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return parseFloat(v.toPrecision(digits)).toString();
}
