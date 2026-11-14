/**
 * Image ▸ Color ▸ Color Picker… (⇧K): a popover anchored to the foreground / background swatches of the tool
 * rail.  Foreground and background are the values Fill, Clear, Draw and the brush / bucket use
 * (`session.settings.foreground` / `.background`): for grayscale images the picker shows and edits the VALUE
 * (number + slider over the bit depth), for RGB images the COLOUR (HSB square + hue strip, RGB and hex inputs,
 * presets); both views keep the other in sync (colour ↔ luminance).  Values are applied through the registry
 * ops `edit.set_foreground` / `edit.set_background` / `edit.set_colors` (so they are recorded); the colour is
 * kept alongside as `foreground_color` / `background_color` (hex) in the settings.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight, Pipette, X } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import { toHex6 } from "../prefs";
import { kbd } from "../platform";
import type { ImageSummary } from "../types";

type Target = "foreground" | "background";

export const PRESETS = ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff", "#ff8800", "#808080"];

// ---------------------------------------------------------------- colour maths
export function hexToRgb(hex: string): [number, number, number] {
  const h = toHex6(hex, "#000000").slice(1);
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export function rgbToHex(r: number, g: number, b: number): string {
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rr = r / 255, gg = g / 255, bb = b / 255;
  const max = Math.max(rr, gg, bb), min = Math.min(rr, gg, bb);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === rr) h = ((gg - bb) / d) % 6;
    else if (max === gg) h = (bb - rr) / d + 2;
    else h = (rr - gg) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return [h, max > 0 ? d / max : 0, max];
}

export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/** Luminance 0–255 of a colour (ImageJ's grey for a colour on a grayscale image). */
export function luminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/** Value range the picker's slider covers for an image: 0–255 (8-bit, RGB), 0–65535 (16-bit), the data range (float). */
export function valueRange(img: ImageSummary | undefined): [number, number] {
  if (!img || img.rgb || img.dtype === "uint8") return [0, 255];
  if (img.dtype === "uint16") return [0, 65535];
  const [lo, hi] = img.value_range;
  return hi > lo ? [lo, hi] : [0, 1];
}

/** The grey hex that shows a value of an image (luminance mapped over the image's range). */
export function valueToHex(v: number, img: ImageSummary | undefined): string {
  const [lo, hi] = valueRange(img);
  const g = Math.round(((v - lo) / (hi - lo || 1)) * 255);
  return rgbToHex(g, g, g);
}

/** The value a colour stands for on an image: luminance scaled to the image's range. */
export function hexToValue(hex: string, img: ImageSummary | undefined): number {
  const [lo, hi] = valueRange(img);
  const v = lo + (luminance(hex) / 255) * (hi - lo);
  return img && img.dtype === "float32" ? Math.round(v * 1000) / 1000 : Math.round(v);
}

// ---------------------------------------------------------------- component
export default function ColorPicker({ anchor, onClose }: { anchor: DOMRect | null; onClose: () => void }) {
  const img = useStore((s) => s.activeImage());
  const settings = useStore((s) => s.settings);
  const prefs = useStore((s) => s.prefs);
  const runOp = useStore((s) => s.runOp);
  const setTool = useStore((s) => s.setTool);
  const [target, setTarget] = useState<Target>("foreground");
  const isRgb = !!img?.rgb;
  const value = Number(settings[target] ?? prefs[target] ?? (target === "foreground" ? 255 : 0));
  const storedColor = settings[`${target}_color`];
  const serverHex = typeof storedColor === "string" && /^#[0-9a-fA-F]{6}$/.test(storedColor) ? storedColor.toLowerCase() : valueToHex(value, img);
  // local colour while dragging (committed on release); null = follow the server
  const [draft, setDraft] = useState<{ hex: string; value: number } | null>(null);
  const hex = draft ? draft.hex : serverHex;
  const curValue = draft ? draft.value : value;
  const [h, s, v] = useMemo(() => rgbToHsv(...hexToRgb(hex)), [hex]);
  const [hue, setHue] = useState(h);
  useEffect(() => {
    if (s > 0.001 && v > 0.001) setHue(h); // grey keeps the last hue so the square does not jump
  }, [h, s, v]);
  const [r, g, b] = hexToRgb(hex);
  const [lo, hi] = valueRange(img);
  const fmtVal = (x: number) => (img?.dtype === "float32" ? String(Math.round(x * 1000) / 1000) : String(Math.round(x)));
  const [hexText, setHexText] = useState(hex);
  useEffect(() => setHexText(hex), [hex]);
  // The typed text of the Value and R / G / B fields lives locally until blur / Enter, like the Hex field: committing on
  // every keystroke gave React nothing new to render, so its controlled-input restore snapped the DOM back to the stale
  // server value after each digit ("1" → 255 → "2552"), and a trailing "." could never be typed on float images.
  const [valText, setValText] = useState(fmtVal(curValue));
  useEffect(() => setValText(fmtVal(curValue)), [curValue, img?.dtype]); // eslint-disable-line react-hooks/exhaustive-deps
  const [rgbText, setRgbText] = useState<[string, string, string]>([String(r), String(g), String(b)]);
  useEffect(() => setRgbText([String(r), String(g), String(b)]), [r, g, b]);
  const sqRef = useRef<HTMLDivElement>(null);
  const timer = useRef<number | null>(null);

  const commit = (nextHex: string, nextValue?: number) => {
    const val = nextValue ?? hexToValue(nextHex, img);
    setDraft(null);
    void runOp(`edit.set_${target}`, { value: val }, { silent: true });
    void api.putSettings({ [`${target}_color`]: nextHex }).then((raw) => useStore.setState({ settings: raw || {} })).catch(() => undefined);
  };
  const preview = (nextHex: string, nextValue?: number) => setDraft({ hex: nextHex, value: nextValue ?? hexToValue(nextHex, img) });
  const setColor = (nextHex: string, final = true) => (final ? commit(nextHex) : preview(nextHex));
  const setValue = (val: number, final = true) => {
    const clamped = Math.max(lo, Math.min(hi, val));
    const nh = isRgb ? hex : valueToHex(clamped, img);
    if (final) commit(nh, clamped);
    else preview(nh, clamped);
  };
  const debouncedValue = (val: number) => {
    setValue(val, false);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setValue(val, true), 250);
  };
  const swap = () => {
    const fg = Number(settings.foreground ?? prefs.foreground ?? 255);
    const bg = Number(settings.background ?? prefs.background ?? 0);
    const fgc = typeof settings.foreground_color === "string" ? settings.foreground_color : valueToHex(fg, img);
    const bgc = typeof settings.background_color === "string" ? settings.background_color : valueToHex(bg, img);
    void runOp("edit.set_colors", { foreground: bg, background: fg }, { silent: true });
    void api.putSettings({ foreground_color: bgc, background_color: fgc }).then((raw) => useStore.setState({ settings: raw || {} })).catch(() => undefined);
  };

  // HSB square: x = saturation, y = brightness
  const squareAt = (e: { clientX: number; clientY: number }, final: boolean) => {
    const el = sqRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const sx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const sy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setColor(rgbToHex(...hsvToRgb(hue, sx, 1 - sy)), final);
  };
  const onSquareDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    squareAt(e, false);
  };
  const setHueAndColor = (nh: number, final: boolean) => {
    setHue(nh);
    setColor(rgbToHex(...hsvToRgb(nh, Math.max(s, 0.001), Math.max(v, 0.001))), final);
  };
  const parseNum = (t: string): number | null => {
    const s = t.trim();
    if (s === "") return null; // Number("") is 0: an emptied field cancels instead of committing 0
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  };
  const enterBlurs = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
    e.stopPropagation();
  };
  const commitValText = () => {
    const n = parseNum(valText);
    if (n === null) return setValText(fmtVal(curValue));
    const clamped = Math.max(lo, Math.min(hi, n));
    setValText(fmtVal(clamped)); // explicit: a clamped value equal to the server value does not re-run the re-seed effect
    if (clamped !== curValue) setValue(clamped, true);
  };
  const commitRgbText = (i: number) => {
    const c: [number, number, number] = [r, g, b];
    const n = parseNum(rgbText[i]);
    if (n !== null) c[i] = Math.max(0, Math.min(255, Math.round(n)));
    setRgbText([String(c[0]), String(c[1]), String(c[2])]);
    if (c[i] !== [r, g, b][i]) setColor(rgbToHex(...c));
  };
  const setRgbText1 = (i: number, t: string) =>
    setRgbText((p) => {
      const q: [string, string, string] = [p[0], p[1], p[2]];
      q[i] = t;
      return q;
    });
  // placed beside its anchor (the swatches at the foot of the tool bar), clamped to the window with its *measured*
  // size: the box grows with the UI text size, so fixed guesses ran it under the status bar
  const boxRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ w: number; h: number }>({ w: 320, h: 440 });
  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const measure = () => setBox((b) => (b.w === el.offsetWidth && b.h === el.offsetHeight ? b : { w: el.offsetWidth, h: el.offsetHeight }));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const margin = 8;
  const bottomLimit = window.innerHeight - 34; // above the status bar
  const style: React.CSSProperties = anchor
    ? { position: "fixed", left: Math.max(margin, Math.min(Math.max(anchor.right, document.querySelector(".rail")?.getBoundingClientRect().right ?? 0) + 10, window.innerWidth - box.w - margin)),
        top: Math.max(margin, Math.min(anchor.bottom - box.h, bottomLimit - box.h)) }
    : { position: "fixed", left: 60, top: Math.max(margin, bottomLimit - box.h) };
  const fg = Number(settings.foreground ?? prefs.foreground ?? 255);
  const bg = Number(settings.background ?? prefs.background ?? 0);
  const fgHex = typeof settings.foreground_color === "string" ? settings.foreground_color : valueToHex(fg, img);
  const bgHex = typeof settings.background_color === "string" ? settings.background_color : valueToHex(bg, img);

  return (
    <div ref={boxRef} className="color-picker" role="dialog" aria-label="Color picker" style={style} onPointerDown={(e) => e.stopPropagation()}>
      <div className="cp-head">
        <span className="cp-title">Color Picker</span>
        <span className="grow" />
        <button className="icon-btn" title="Close (Esc)" aria-label="Close" onClick={onClose}><X size={14} /></button>
      </div>
      <div className="cp-targets">
        <button className={"cp-target" + (target === "foreground" ? " on" : "")} onClick={() => setTarget("foreground")} title="Foreground: Fill, Draw, brush, bucket">
          <span className="cp-sw" style={{ background: fgHex }} /><span className="cp-tname">Foreground</span> <span className="mono">{fmtVal(fg)}</span>
        </button>
        <button className={"cp-target" + (target === "background" ? " on" : "")} onClick={() => setTarget("background")} title="Background: Clear, Clear Outside, erase">
          <span className="cp-sw" style={{ background: bgHex }} /><span className="cp-tname">Background</span> <span className="mono">{fmtVal(bg)}</span>
        </button>
        <button className="icon-btn" title="Swap foreground and background" aria-label="Swap" onClick={swap}><ArrowLeftRight size={14} /></button>
      </div>
      {!isRgb && (
        <div className="cp-row">
          <label>Value</label>
          <input type="range" min={lo} max={hi} step={img?.dtype === "float32" ? (hi - lo) / 1000 : 1} value={curValue} onChange={(e) => debouncedValue(Number(e.target.value))} onPointerUp={() => setValue(curValue, true)} aria-label="Value" />
          <input type="number" min={lo} max={hi} step={img?.dtype === "float32" ? 0.001 : 1} value={valText} onChange={(e) => setValText(e.target.value)} onBlur={commitValText} onKeyDown={enterBlurs} aria-label="Value" />
          <span className="dim small cp-bits">{img ? img.bit_depth : "8-bit"}</span>
        </div>
      )}
      <div ref={sqRef} className="cp-square" style={{ background: `linear-gradient(to top, #000, rgba(0,0,0,0)), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))` }}
        onPointerDown={onSquareDown} onPointerMove={(e) => { if (e.buttons & 1) squareAt(e, false); }} onPointerUp={(e) => squareAt(e, true)} role="slider" aria-label="Saturation and brightness" aria-valuenow={Math.round(v * 100)}>
        <span className="cp-dot" style={{ left: `${s * 100}%`, top: `${(1 - v) * 100}%`, background: hex }} />
      </div>
      <input className="cp-hue" type="range" min={0} max={360} step={1} value={Math.round(hue)} aria-label="Hue" onChange={(e) => setHueAndColor(Number(e.target.value), false)} onPointerUp={() => setHueAndColor(hue, true)} onKeyUp={() => setHueAndColor(hue, true)} />
      <div className="cp-row cp-rgb">
        <label>R</label><input type="number" min={0} max={255} value={rgbText[0]} onChange={(e) => setRgbText1(0, e.target.value)} onBlur={() => commitRgbText(0)} onKeyDown={enterBlurs} aria-label="Red" />
        <label>G</label><input type="number" min={0} max={255} value={rgbText[1]} onChange={(e) => setRgbText1(1, e.target.value)} onBlur={() => commitRgbText(1)} onKeyDown={enterBlurs} aria-label="Green" />
        <label>B</label><input type="number" min={0} max={255} value={rgbText[2]} onChange={(e) => setRgbText1(2, e.target.value)} onBlur={() => commitRgbText(2)} onKeyDown={enterBlurs} aria-label="Blue" />
      </div>
      <div className="cp-row cp-hex">
        <label>Hex</label>
        <input type="text" className="mono" value={hexText} spellCheck={false} aria-label="Hex colour" onChange={(e) => setHexText(e.target.value)}
          onBlur={() => { const t = hexText.trim().startsWith("#") ? hexText.trim() : "#" + hexText.trim(); if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(t)) setColor(toHex6(t)); else setHexText(hex); }}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); e.stopPropagation(); }} />
        <input type="color" value={toHex6(hex)} onChange={(e) => setColor(e.target.value)} aria-label="System colour dialog" title="System colour dialog" />
        <span className="cp-sw big" style={{ background: hex }} title={`${hex} = value ${fmtVal(curValue)}`} />
      </div>
      <div className="cp-presets" role="group" aria-label="Presets">
        {PRESETS.map((p) => <button key={p} className={"cp-preset" + (p === hex ? " on" : "")} style={{ background: p }} title={p} aria-label={p} onClick={() => setColor(p)} />)}
      </div>
      <div className="cp-row">
        <button className="btn sm" onClick={() => { setTool("picker"); onClose(); }} title={`Value picker tool (${kbd("I")}): click sets the foreground, ⌥-click the background`}><Pipette size={12} /> Pick from image</button>
        <span className="dim small cp-note">{isRgb ? "RGB image: colour" : `grayscale: value ${lo}–${fmtVal(hi)}`}</span>
      </div>
    </div>
  );
}
