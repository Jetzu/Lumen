import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight, Eye, EyeOff, Trash2, Copy, Download, Plus, Layers, Settings2, RotateCcw, Wand2 } from "lucide-react";
import { api } from "../api";
import { useStore, getPlanes, type PanelKey } from "../store";
import { histogram, dataRange, lutGradientCss, lutColor } from "../render/compositor";
import { histStride } from "../render/stride";
import type { ImageSummary, PlaneData, Roi } from "../types";
import RangeSlider, { fmt } from "./RangeSlider";
import Plot from "./Plot";
import HistogramChart from "./HistogramChart";
import { doAction } from "../actions";
import { kbd } from "../platform";
import VolumeSection, { VolumeModalHost } from "./VolumeViewer";
import OrthoViews from "./OrthoViews";
import MemoryMonitor from "./MemoryMonitor";
import WorkflowPanel from "./WorkflowPanel";
import ClassifierPanel from "./ClassifierPanel";
import { Maximize2 } from "lucide-react";
import { selectResultsRow } from "./ResultsModal";
import TracksPanel from "./TracksPanel";
import { trackKeyOfOverlayId } from "../trackStore";
import ScriptPanel, { ScriptModalHost } from "./ScriptPanel";
import TextWindow, { isTextTable } from "./TextWindow";
import PluginFrames, { useFrameCount } from "./PluginFrames";
import { groupLuts, lutLabel } from "../lutGroups";
import { restoreClear } from "../store";
import NumberInput from "./NumberInput";

/**
 * A readable ROI name (E5-11): Analyze Particles names its ROIs ImageJ-style "0001-0025-0315" (index-y-x), shown
 * here as "Particle 1 (x 315, y 25)"; every other name is shown as is.
 */
export function roiDisplayName(r: Roi): string {
  const name = r.name || r.id || "";
  const m = /^(\d{4})-(\d{4})-(\d{4})$/.exec(name);
  if (m) return `Particle ${parseInt(m[1], 10)} (x ${parseInt(m[3], 10)}, y ${parseInt(m[2], 10)})`;
  return name;
}

/** The Results rows that belong to `imageName` (Label "<image>" or "<image>:<n>"); null = the table has no Label column. */
function rowsOfImage(columns: string[], rows: any[][], imageName: string): number[] | null {
  const li = columns.indexOf("Label");
  if (li < 0) return null;
  const out: number[] = [];
  rows.forEach((r, i) => {
    const v = r[li];
    if (typeof v === "string" && (v === imageName || v.startsWith(imageName + ":"))) out.push(i);
  });
  return out;
}

/** Marks a scroll container that overflows horizontally (`scrollable` class → fade + hint, E6-11). */
function useHorizontalOverflow<T extends HTMLElement>(): { ref: React.RefObject<T>; overflow: boolean } {
  const ref = useRef<T>(null);
  const [overflow, setOverflow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const check = () => setOverflow(el.scrollWidth > el.clientWidth + 2);
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => ro.disconnect();
  });
  return { ref, overflow };
}

export default function Inspector() {
  const img = useStore((s) => s.activeImage());
  const panels = useStore((s) => s.panels);
  const state = useStore((s) => s.state);
  const plot = useStore((s) => s.plot);
  const hist = useStore((s) => s.histogram);
  const table = useStore((s) => s.table);
  const memoryMonitor = useStore((s) => s.memoryMonitor);
  const workflowProgress = useStore((s) => s.workflowProgress);
  const selectedTrackIds = useStore((s) => s.selectedTrackIds);
  const nFrames = useFrameCount();
  const thr = img ? state?.thresholds[img.id] : undefined;
  const planes = useCurrentPlanes(img);
  const scroll = useInspectorScroll();
  const roisThisImageOnly = useStore((s) => s.roisThisImageOnly);
  const resultsThisImageOnly = useStore((s) => s.resultsThisImageOnly);
  // tracks, not branches: a lineage tree is one track overlay per branch ("track-3", "track-3-b1", …)
  const nTracks = img ? new Set(img.overlay.filter((o) => (o as { type: string }).type === "track").map((o) => trackKeyOfOverlayId((o as { id: string }).id))).size : 0;
  const isTimeSeries = !!img && img.shape.t > 1;
  // a time series opens its Tracks section once, above the fold (E6-12)
  const tracksAutoOpened = useStore((s) => s.tracksAutoOpened);
  useEffect(() => {
    if (!img || !isTimeSeries || tracksAutoOpened[img.id]) return;
    useStore.getState().markTracksAutoOpened(img.id);
    useStore.getState().togglePanel("tracks", true);
  }, [img?.id, isTimeSeries]); // eslint-disable-line react-hooks/exhaustive-deps
  const roiBadge = (() => {
    const all = state?.rois.length ?? 0;
    if (!all) return undefined;
    if (!roisThisImageOnly || !img) return String(all);
    const mine = state!.rois.filter((r) => !r.image_id || r.image_id === img.id).length;
    return mine === all ? String(all) : `${mine} / ${all}`;
  })();
  const resultsBadge = (() => {
    const all = state?.results.rows.length ?? 0;
    if (!all) return undefined;
    if (!resultsThisImageOnly || !img) return String(all);
    const mine = rowsOfImage(state!.results.columns, state!.results.rows, img.name);
    return mine === null || mine.length === all ? String(all) : `${mine.length} / ${all}`;
  })();
  const tracksSection = img && (nTracks > 0 || isTimeSeries) && (
    <Section id="tracks" title="Tracks" badge={nTracks > 0 ? (selectedTrackIds.length ? `${selectedTrackIds.length} / ${nTracks}` : String(nTracks)) : isTimeSeries ? `${img.shape.t} frames` : undefined}><TracksPanel img={img} /></Section>
  );
  return (
    <div className="inspector" ref={scroll.ref} onScroll={scroll.onScroll}>
      <Section id="display" title="Display" badge={img ? `${img.shape.c}ch` : undefined}>{img ? <DisplayPanel img={img} planes={planes} /> : <Empty />}</Section>
      {!img?.rgb && <Section id="threshold" title="Threshold" badge={thr ? `${fmt(thr.lower, 4)}–${fmt(thr.upper, 4)}` : undefined}>{img ? <ThresholdPanel img={img} planes={planes} /> : <Empty />}</Section>}
      {isTimeSeries && tracksSection}
      <Section id="workflows" title="Workflows" badge={workflowProgress ? `${workflowProgress.title} ${workflowProgress.done}/${workflowProgress.total}` : undefined}><WorkflowPanel /></Section>
      {img && img.shape.z > 1 && <Section id="volume" title="3-D View" badge={`${img.shape.z} slices`}><VolumeSection img={img} /></Section>}
      {img && img.shape.z > 1 && <Section id="ortho" title="Orthogonal Views"><OrthoViews img={img} /></Section>}
      <Section id="rois" title="ROI Manager" badge={roiBadge}><RoiPanel /></Section>
      <Section id="classifier" title="Pixel Classifier" badge={state?.rois.some((r) => (r.name || "").toLowerCase().startsWith("class:")) ? String(state.rois.filter((r) => (r.name || "").toLowerCase().startsWith("class:")).length) : undefined}>{img ? <ClassifierPanel img={img} /> : <Empty />}</Section>
      {!isTimeSeries && tracksSection}
      <Section id="results" title="Results" badge={resultsBadge} actions={<ExpandButton kind="results" disabled={!state?.results.rows.length} />}><ResultsPanel /></Section>
      {plot && <Section id="plot" title="Plot"><Plot data={plot} /></Section>}
      {hist && <Section id="histogram" title="Histogram"><HistogramChart data={hist} /></Section>}
      {table && isTextTable(table) && <Section id="table" title={table.name || "Text"} badge={table.rows.length ? `${table.rows.length} lines` : "text"}><TextWindow table={table} /></Section>}
      {table && !isTextTable(table) && <Section id="table" title={table.name || "Table"} badge={table.rows.length ? String(table.rows.length) : undefined} actions={<ExpandButton kind="table" disabled={!table.rows.length} />}><GenericTable table={table} /></Section>}
      <Section id="frames" title="Plugin Frames" badge={nFrames ? String(nFrames) : undefined}><PluginFrames /></Section>
      <Section id="info" title="Image Info">{img ? <InfoPanel img={img} /> : <Empty />}</Section>
      <Section id="history" title="History" badge={img && state?.history[img.id]?.undo.length ? String(state.history[img.id].undo.length) : undefined}>{img ? <HistoryPanel img={img} /> : <Empty />}</Section>
      <Section id="script" title="Script & Recorder"><ScriptPanel /></Section>
      <Section id="log" title="Log" badge={memoryMonitor ? "memory" : undefined}>{memoryMonitor && <MemoryMonitor />}<LogPanel /></Section>
      <VolumeModalHost img={img} />
      <ScriptModalHost />
    </div>
  );
}

function Empty() {
  return <div className="hint-text">No image selected.</div>;
}

/**
 * Collapsible inspector section.  The head row holds a real <button> (disclosure pattern: aria-expanded /
 * aria-controls, Enter / Space toggle, Tab reaches it) plus the optional action buttons beside it – they
 * cannot live inside the toggle button (nested interactive content).  The body is a labelled region.
 */
function Section({ id, title, badge, actions, children }: { id: PanelKey; title: string; badge?: string; actions?: React.ReactNode; children: React.ReactNode }) {
  const open = useStore((s) => s.panels[id]);
  const toggle = useStore((s) => s.togglePanel);
  const headId = `section-head-${id}`;
  const bodyId = `section-body-${id}`;
  return (
    <div className="section" data-section={id}>
      <div className={"section-head" + (open ? " open" : "")}>
        <button type="button" id={headId} className="section-toggle" aria-expanded={open} aria-controls={open ? bodyId : undefined} onClick={() => toggle(id)}>
          <span className="chev" aria-hidden="true"><ChevronRight size={14} /></span>
          {title}
          {badge && <span className="badge"><span className="sr-only"> (</span>{badge}<span className="sr-only">)</span></span>}
        </button>
        {actions && <span className="actions">{actions}</span>}
      </div>
      {open && <div className="section-body" id={bodyId} role="region" aria-labelledby={headId}>{children}</div>}
    </div>
  );
}

/** Remember the inspector's scroll position across remounts (⌘\ toggling) and reloads. */
function useInspectorScroll(): { ref: React.RefObject<HTMLDivElement>; onScroll: () => void } {
  const ref = useRef<HTMLDivElement>(null);
  const timer = useRef<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const saved = Number(localStorage.getItem("lumen.inspectorScroll")) || 0;
    if (saved > 0) {
      el.scrollTop = saved;
      // sections fill in asynchronously (planes, histograms): re-apply once the content has grown
      const id = window.setTimeout(() => {
        if (Math.abs(el.scrollTop - saved) > 1) el.scrollTop = saved;
      }, 150);
      return () => window.clearTimeout(id);
    }
  }, []);
  const onScroll = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      if (ref.current) localStorage.setItem("lumen.inspectorScroll", String(Math.round(ref.current.scrollTop)));
    }, 150);
  };
  return { ref, onScroll };
}

function useCurrentPlanes(img: ImageSummary | undefined): PlaneData | null {
  const [planes, setPlanes] = useState<PlaneData | null>(null);
  useEffect(() => {
    if (!img) {
      setPlanes(null);
      return;
    }
    let alive = true;
    getPlanes(img, img.display.t, img.display.z, histStride(img)).then((p) => alive && setPlanes(p)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [img?.id, img?.rev, img?.display.t, img?.display.z]); // eslint-disable-line react-hooks/exhaustive-deps
  return planes;
}

// ------------------------------------------------------------------ display
function DisplayPanel({ img, planes }: { img: ImageSummary; planes: PlaneData | null }) {
  const setDisplay = useStore((s) => s.setDisplay);
  const luts = useStore((s) => s.luts);
  const lutNames = useStore((s) => s.meta?.luts || []);
  const lutGrouping = useStore((s) => s.lutGrouping);
  const lutGroups = useMemo(() => groupLuts(lutNames, lutGrouping), [lutNames, lutGrouping]);
  const d = img.display;
  const chans = img.rgb ? d.channels.slice(0, 3) : d.channels;
  const showAll = !img.rgb && d.mode === "composite";
  const visibleIdx = showAll ? chans.map((_, i) => i) : [Math.min(d.c, chans.length - 1)];

  const ranges = useMemo(() => chans.map((ch, i) => {
    if (img.dtype === "uint8") return [0, 255] as [number, number];
    const data = planes?.channels[i];
    let [lo, hi] = data ? dataRange(data) : img.value_range;
    if (img.dtype === "uint16") {
      lo = 0;
      hi = Math.max(hi, 255);
    }
    lo = Math.min(lo, ch.min);
    hi = Math.max(hi, ch.max);
    if (hi <= lo) hi = lo + 1;
    return [lo, hi] as [number, number];
  }), [chans, planes, img.dtype, img.value_range]);

  const hists = useMemo(() => chans.map((_, i) => (planes?.channels[i] ? histogram(planes.channels[i], ranges[i][0], ranges[i][1]).counts : null)), [planes, ranges, chans]);

  const auto = (i: number) => {
    const h = hists[i];
    if (!h) return;
    const total = h.reduce((a, b) => a + b, 0);
    const [lo, hi] = ranges[i];
    const binw = (hi - lo) / h.length;
    const clip = total * 0.0035 / 2;
    let acc = 0;
    let a = 0;
    for (; a < h.length && acc + h[a] < clip; a++) acc += h[a];
    acc = 0;
    let b = h.length - 1;
    for (; b > a && acc + h[b] < clip; b--) acc += h[b];
    const nmin = lo + a * binw;
    const nmax = lo + (b + 1) * binw;
    setChan(i, { min: nmin, max: nmax > nmin ? nmax : nmin + 1 }, true);
  };
  const setChan = (i: number, patch: Record<string, unknown>, immediate = false) => {
    const arr = chans.map((_, k) => (k === i ? patch : {}));
    setDisplay(img.id, { channels: arr }, immediate);
  };
  const applyAll = () => {
    const src = chans[d.c];
    setDisplay(img.id, { channels: chans.map(() => ({ min: src.min, max: src.max })) }, true);
  };

  return (
    <div>
      {!img.rgb && chans.length > 1 && (
        <div className="row">
          <label>Mode</label>
          <span className="seg">
            {(["composite", "single", "grayscale"] as const).map((m) => <button key={m} className={d.mode === m ? "on" : ""} onClick={() => setDisplay(img.id, { mode: m }, true)}>{m === "composite" ? "Composite" : m === "single" ? "Color" : "Gray"}</button>)}
          </span>
        </div>
      )}
      {visibleIdx.map((i) => {
        const ch = chans[i];
        const gradient = img.rgb ? lutGradientCss(luts, ["Red", "Green", "Blue"][i]) : lutGradientCss(luts, d.mode === "grayscale" ? "Grays" : ch.lut, ch.inverted);
        const chName = ch.name || `Channel ${i + 1}`;
        const who = `channel ${i + 1}${ch.name ? ` (${ch.name})` : ""}`;
        return (
          <div className="channel" key={i}>
            <div className="channel-head">
              {!img.rgb && (
                /* a toggle button keeps one name and reports its state with aria-pressed (E7-19): "Show channel 1 (DAPI), pressed" = visible */
                <button className="icon-btn" style={{ width: 24, height: 24 }} title={`${ch.visible ? "Hide" : "Show"} ${who}`} aria-label={`Show ${who}`} aria-pressed={ch.visible} onClick={() => setChan(i, { visible: !ch.visible }, true)}>{ch.visible ? <Eye size={14} /> : <EyeOff size={14} />}</button>
              )}
              <span className="sw" style={{ background: gradient }} />
              <input className="cname" value={ch.name} aria-label={`Channel ${i + 1} name`} title="Channel name" onChange={(e) => setChan(i, { name: e.target.value })} spellCheck={false} />
              {!img.rgb && (
                <select value={ch.lut} onChange={(e) => setChan(i, { lut: e.target.value }, true)} title={`Lookup table of ${who}`} aria-label={`Lookup table of ${who}`}>
                  {lutGroups.map((g) => (
                    <optgroup key={g.name} label={g.name}>{g.luts.map((n) => <option key={n} value={n}>{lutLabel(n, lutGrouping)}</option>)}</optgroup>
                  ))}
                  {!lutGroups.some((g) => g.luts.includes(ch.lut)) && <option value={ch.lut}>{ch.lut}</option>}
                </select>
              )}
            </div>
            <RangeSlider label={`${chName} display range`} lo={ch.min} hi={ch.max} min={ranges[i][0]} max={ranges[i][1]} counts={hists[i]} gradient={gradient} color={lutColor(luts, img.rgb ? ["Red", "Green", "Blue"][i] : ch.lut)}
              onChange={(lo, hi, final) => setChan(i, { min: lo, max: hi }, final)} />
            <div className="range-vals">
              <BoundInput value={round(ch.min, img.dtype !== "float32")} label={`${chName} display minimum`} invalidWhen={(v) => (v >= ch.max ? `must be below the maximum (${round(ch.max, img.dtype !== "float32")})` : null)} onCommit={(v) => setChan(i, { min: v }, true)} />
              <button className="btn sm" onClick={() => auto(i)} title="Auto: saturate 0.35% of pixels" aria-label={`Auto display range of ${who} (saturate 0.35% of pixels)`}>Auto</button>
              <button className="btn sm" onClick={() => setChan(i, { min: ranges[i][0], max: img.dtype === "uint16" ? Math.max(...(planes?.channels[i] ? dataRange(planes.channels[i]) : [255])) : ranges[i][1] }, true)} title="Reset to the data range" aria-label={`Reset display range of ${who} to the data range`}>Reset</button>
              {!img.rgb && <button className={"btn sm" + (ch.inverted ? " primary" : "")} aria-pressed={ch.inverted} onClick={() => setChan(i, { inverted: !ch.inverted }, true)} title="Invert LUT" aria-label={`Invert the lookup table of ${who}`}>Inv</button>}
              <BoundInput value={round(ch.max, img.dtype !== "float32")} label={`${chName} display maximum`} invalidWhen={(v) => (v <= ch.min ? `must be above the minimum (${round(ch.min, img.dtype !== "float32")})` : null)} onCommit={(v) => setChan(i, { max: v }, true)} />
            </div>
          </div>
        );
      })}
      <div className="btn-row">
        {chans.length > 1 && showAll && <button className="btn sm" onClick={applyAll} title="Copy the current channel's range to all channels">Same range for all</button>}
        <button className="btn sm" onClick={() => void api.resetDisplay(img.id).then(() => useStore.getState().refresh())}><RotateCcw size={12} /> Reset all</button>
        <button className="btn sm" onClick={() => useStore.getState().invoke("process.enhance_contrast")} title="Enhance Contrast (apply to pixels or display)">Enhance…</button>
      </div>
      <div className="hint-text">Display changes never modify pixel values. Use <b>Apply Display Range to Pixels</b> (Image ▸ Lookup Tables) or <b>RGB Color</b> to bake them in.</div>
    </div>
  );
}

function round(v: number, integer = false): number {
  return integer ? Math.round(v) : Math.round(v * 1000) / 1000;
}

/**
 * A number input that commits on change but refuses an invalid value (a display maximum below the minimum, E5-6):
 * the text stays editable, the field is marked invalid with the reason, and blur restores the last good value.
 * A NumberInput (text field): decimal point whatever the browser locale, "," accepted (E7-26).
 */
function BoundInput({ value, label, invalidWhen, onCommit }: { value: number; label: string; invalidWhen: (v: number) => string | null; onCommit: (v: number) => void }) {
  const [problem, setProblem] = useState<string | null>(null);
  // a rejected value is never committed, so the field keeps showing it: `shown` follows the text until blur
  const [shown, setShown] = useState<number>(value);
  useEffect(() => {
    setShown(value);
    setProblem(null);
  }, [value]);
  return (
    <span className="bound-input">
      <NumberInput value={shown} aria-label={label} title={problem ? `${label}: ${problem}` : label} invalid={!!problem}
        onChange={(v) => {
          const why = invalidWhen(v);
          setProblem(why);
          setShown(v);
          if (!why) onCommit(v);
        }}
        onText={(s, n) => {
          if (s.trim() !== "" && n === null) setProblem(null);
        }}
        onBlur={() => {
          if (problem || shown !== value) {
            setShown(value);
            setProblem(null);
          }
        }} />
      {problem && <span className="bound-problem" role="alert">{problem}</span>}
    </span>
  );
}

// ------------------------------------------------------------------ threshold
function ThresholdPanel({ img, planes }: { img: ImageSummary; planes: PlaneData | null }) {
  const state = useStore((s) => s.state)!;
  const runOp = useStore((s) => s.runOp);
  const methods = useStore((s) => s.meta?.threshold_methods || []);
  const thr = state.thresholds[img.id];
  const [method, setMethod] = useState(thr?.method && methods.includes(thr.method) ? thr.method : "Default");
  const [dark, setDark] = useState(thr?.dark_background ?? true);
  const [local, setLocal] = useState<{ lower: number; upper: number } | null>(null);
  const timer = useRef<number | null>(null);
  const c = Math.min(img.display.c, (planes?.channels.length ?? 1) - 1);
  const data = planes?.channels[c];
  const range = useMemo<[number, number]>(() => {
    if (img.dtype === "uint8") return [0, 255];
    if (!data) return img.value_range;
    return dataRange(data);
  }, [data, img.dtype, img.value_range]);
  const counts = useMemo(() => (data ? histogram(data, range[0], range[1]).counts : null), [data, range]);
  const cur = local || thr || null;
  const chan = img.display.channels[c];
  const chanHidden = !img.rgb && img.shape.c > 1 && img.display.mode === "composite" && chan && !chan.visible;
  if (img.rgb) return <div className="hint-text">Convert RGB images to 8-bit (Image ▸ Type) before thresholding.</div>;

  const setRange = (lower: number, upper: number, final: boolean) => {
    setLocal({ lower, upper });
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      void runOp("threshold.set", { lower, upper }, { silent: true }).then(() => setLocal(null));
    }, final ? 0 : 200);
  };
  return (
    <div>
      {img.shape.c > 1 && (
        <div className="thr-channel">
          <span>Channel <b>{c + 1}</b>{chan?.name ? <> · <b>{chan.name}</b></> : null} is thresholded <span className="dim">(the current channel – switch with the channel chips below the image)</span></span>
          {chanHidden && (
            <span className="thr-hidden" role="status">This channel is hidden in the composite: the red overlay marks pixels you cannot see. <button className="btn sm" onClick={() => useStore.getState().setDisplay(img.id, { channels: img.display.channels.map((_, k) => (k === c ? { visible: true } : {})) }, true)}>Show channel {c + 1}</button></span>
          )}
        </div>
      )}
      <div className="row">
        <select value={method} onChange={(e) => setMethod(e.target.value)} style={{ flex: 1 }} aria-label="Threshold method" title="Threshold method">{methods.map((m) => <option key={m} value={m}>{m}</option>)}</select>
        <label className="switch"><input type="checkbox" checked={dark} onChange={(e) => setDark(e.target.checked)} /> dark bg</label>
        <button className="btn sm primary" onClick={() => void runOp("threshold.compute", { method, dark_background: dark }, { silent: true })} title="Compute the threshold for the current plane"><Wand2 size={12} /> Auto</button>
      </div>
      <RangeSlider label={`Threshold of channel ${c + 1}`} lo={cur?.lower ?? range[0]} hi={cur?.upper ?? range[1]} min={range[0]} max={range[1]} counts={counts} color="#f87171" gradient="rgba(248,113,113,0.5)" onChange={setRange} />
      <div className="range-vals">
        <NumberInput aria-label="Threshold lower limit" min={range[0]} max={range[1]} step={img.dtype === "float32" ? (range[1] - range[0]) / 256 || 0.01 : 1} value={cur ? round(cur.lower, img.dtype !== "float32") : null} placeholder="lower" onChange={(v) => setRange(v, cur?.upper ?? range[1], true)} />
        <span className="dim small">{cur ? `${fmtPct(data, cur)} selected` : "no threshold"}</span>
        <NumberInput aria-label="Threshold upper limit" min={range[0]} max={range[1]} step={img.dtype === "float32" ? (range[1] - range[0]) / 256 || 0.01 : 1} value={cur ? round(cur.upper, img.dtype !== "float32") : null} placeholder="upper" onChange={(v) => setRange(cur?.lower ?? range[0], v, true)} />
      </div>
      <div className="btn-row">
        <button className="btn sm primary" disabled={!cur} onClick={() => useStore.getState().openSheet(useStore.getState().ops["threshold.apply"], { method: "Current" })}>Apply → mask</button>
        <button className="btn sm" disabled={!cur} onClick={() => void runOp("selection.create", {})}>Create selection</button>
        <button className="btn sm" disabled={!cur} onClick={() => useStore.getState().openSheet(useStore.getState().ops["analyze.particles"], { show: "Nothing" })} title="Count & measure particles with the current threshold">Particles…</button>
        <button className="btn sm" disabled={!thr} onClick={() => void api.clearThreshold(img.id).then(() => useStore.getState().refresh())}>Clear</button>
      </div>
      <div className="hint-text">Red pixels are inside the range. Drag the handles, or pick a method and press Auto. Local (adaptive) methods: {kbd("⌘K")} → “Auto Local Threshold”.</div>
    </div>
  );
}

function fmtPct(data: PlaneData["channels"][number] | undefined, thr: { lower: number; upper: number }): string {
  if (!data) return "";
  let n = 0;
  const step = data.length > 1_000_000 ? 4 : 1;
  let tot = 0;
  for (let i = 0; i < data.length; i += step) {
    tot++;
    const v = data[i];
    if (v >= thr.lower && v <= thr.upper) n++;
  }
  return `${((100 * n) / Math.max(tot, 1)).toFixed(1)}%`;
}

// ------------------------------------------------------------------ info
function InfoPanel({ img }: { img: ImageSummary }) {
  const cal = img.calibration;
  const invoke = useStore((s) => s.invoke);
  const hist = useStore((s) => s.state?.history[img.id]);
  return (
    <div>
      <div className="kv">
        <span className="k">Name</span><span className="v">{img.name}</span>
        <span className="k">Size</span><span className="v">{img.shape.x} × {img.shape.y} px</span>
        <span className="k">Dimensions</span><span className="v">{img.shape.c} ch · {img.shape.z} slices · {img.shape.t} frames</span>
        <span className="k">Type</span><span className="v">{img.bit_depth}{img.binary_hint ? " (binary mask)" : ""}</span>
        <span className="k">Value range</span><span className="v">{fmt(img.value_range[0])} – {fmt(img.value_range[1])}</span>
        <span className="k">Pixel size</span><span className="v">{fmt(cal.pixel_width)} × {fmt(cal.pixel_height)} {cal.unit}{img.shape.z > 1 ? ` · z ${fmt(cal.pixel_depth)}` : ""}</span>
        {cal.frame_interval > 0 && <><span className="k">Frame interval</span><span className="v">{cal.frame_interval} {cal.time_unit}</span></>}
        <span className="k">Memory</span><span className="v">{(img.nbytes / 1048576).toFixed(1)} MB</span>
        {img.path && <><span className="k">Path</span><span className="v" style={{ fontSize: 10.5 }}>{img.path}</span></>}
        {/* provenance (E7-16): what this image was derived from and the file it came from, ahead of the raw metadata */}
        {img.metadata.derived_from !== undefined && <><span className="k">Derived from</span><span className="v">{String(img.metadata.derived_from)}</span></>}
        {img.metadata.source_path !== undefined && <><span className="k">Source</span><span className="v" style={{ fontSize: 10.5 }}>{String(img.metadata.source_path)}</span></>}
        {Object.entries(img.metadata).filter(([k]) => k !== "derived_from" && k !== "source_path").slice(0, 12).map(([k, v]) => <React.Fragment key={k}><span className="k">{k}</span><span className="v">{String(v)}</span></React.Fragment>)}
      </div>
      <div className="btn-row">
        <button className="btn sm" onClick={() => invoke("image.properties")}><Settings2 size={12} /> Calibration…</button>
        <button className="btn sm" onClick={() => invoke("image.rename")}>Rename…</button>
        <button className="btn sm" onClick={() => invoke("color.channel_names")} disabled={img.rgb || img.shape.c < 2}>Channel names…</button>
      </div>
      {hist && hist.undo.length > 0 && <div className="hint-text">{hist.undo.length} undoable step{hist.undo.length > 1 ? "s" : ""}.</div>}
    </div>
  );
}

// ------------------------------------------------------------------ rois
function RoiPanel() {
  const state = useStore((s) => s.state)!;
  const img = useStore((s) => s.activeImage());
  const selection = useStore((s) => (img ? s.selections[img.id] : null));
  const setSelection = useStore((s) => s.setSelection);
  const selectedIds = useStore((s) => s.selectedRoiIds);
  const setSelectedIds = useStore((s) => s.setSelectedRoiIds);
  const showAll = useStore((s) => s.showManagerRois);
  const setShowAll = useStore((s) => s.setShowManagerRois);
  const setDisplay = useStore((s) => s.setDisplay);
  const refresh = useStore((s) => s.refresh);
  const toast = useStore((s) => s.toast);
  const [editing, setEditing] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const thisOnly = useStore((s) => s.roisThisImageOnly);
  const setThisOnly = useStore((s) => s.setRoisThisImageOnly);
  const allRois = state.rois;
  // "This image only" (default, E5-11 / E6-3): entries of other images stay in the manager but out of this list
  const rois = useMemo(() => (thisOnly && img ? allRois.filter((r) => !r.image_id || r.image_id === img.id) : allRois), [allRois, thisOnly, img?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const hiddenCount = allRois.length - rois.length;
  const imageName = (id: string | undefined) => (id ? state.images.find((im) => im.id === id)?.name : undefined);

  const add = async () => {
    if (!img || !selection) return toast("Draw a selection first", "error");
    await api.addRoi(selection, img.id, { t: img.display.t, z: img.display.z, c: img.display.c });
    await refresh();
  };
  const pick = (r: Roi, e: React.MouseEvent) => {
    let ids: string[];
    if (e.shiftKey && selectedIds.length) {
      const a = rois.findIndex((x) => x.id === selectedIds[0]);
      const b = rois.findIndex((x) => x.id === r.id);
      ids = rois.slice(Math.min(a, b), Math.max(a, b) + 1).map((x) => x.id!);
    } else if (e.metaKey || e.ctrlKey) ids = selectedIds.includes(r.id!) ? selectedIds.filter((i) => i !== r.id) : [...selectedIds, r.id!];
    else ids = [r.id!];
    setSelectedIds(ids);
    if (img && ids.length === 1) {
      setSelection(img.id, r);
      if (r.position) setDisplay(img.id, { t: Math.min(r.position.t, img.shape.t - 1), z: Math.min(r.position.z, img.shape.z - 1) });
    }
  };
  const del = async () => {
    const ids = selectedIds.length ? selectedIds : [];
    if (!ids.length) return;
    const r = (await api.deleteRois(ids)) as { deleted?: number; restorable?: boolean } | undefined;
    setSelectedIds([]);
    await refresh();
    const n = r?.deleted ?? ids.length;
    const restorable = r?.restorable ?? !!useStore.getState().ops["rois.restore_cleared"];
    toast(`Deleted ${n} ROI${n === 1 ? "" : "s"}`, "info", restorable ? { actions: [{ label: "Undo delete", run: () => void restoreClear("rois") }] } : undefined);
  };
  const measure = async () => {
    const r = await api.measureRois(selectedIds);
    await refresh();
    useStore.getState().togglePanel("results", true);
    toast(`Measured ${r.rows} ROI${r.rows === 1 ? "" : "s"}`, "success");
  };
  const combine = async (mode: string) => {
    if (!img) return;
    try {
      const r = await api.combineRois(selectedIds, mode, img.id);
      setSelection(img.id, r);
    } catch (e: any) {
      toast(e.message, "error");
    }
  };
  const rename = async (id: string, name: string) => {
    setEditing(null);
    if (name.trim()) {
      await api.patchRoi(id, { name: name.trim() });
      await refresh();
    }
  };
  return (
    <div>
      <div className="btn-row" style={{ marginTop: 0, marginBottom: 6 }}>
        <button className="btn sm primary" onClick={() => void add()} title="Add current selection (T)"><Plus size={12} /> Add</button>
        <button className="btn sm" onClick={() => void measure()} disabled={!rois.length} title="Measure selected (or all) ROIs">Measure</button>
        <button className="btn sm" onClick={() => void del()} disabled={!selectedIds.length} title="Delete selected"><Trash2 size={12} /></button>
        <button className={"btn sm" + (showAll ? " primary" : "")} onClick={() => setShowAll(!showAll)} title="Draw every listed ROI on the image" aria-pressed={showAll}><Layers size={12} /> Show all</button>
      </div>
      <label className="switch this-image" title="List only the ROIs that belong to the active image (the manager keeps every image's ROIs)">
        <input type="checkbox" checked={thisOnly} onChange={(e) => setThisOnly(e.target.checked)} /> This image only{thisOnly && hiddenCount > 0 ? <span className="dim"> · {hiddenCount} of other images hidden</span> : null}
      </label>
      <div className="roi-list">
        {rois.length === 0 && allRois.length === 0 && <div className="hint-text" style={{ padding: 8 }}>No ROIs yet. Draw a selection and press <span className="kbd">T</span>, or run Analyze Particles.</div>}
        {rois.length === 0 && allRois.length > 0 && <div className="hint-text" style={{ padding: 8 }}>No ROIs of this image. {allRois.length} ROI{allRois.length === 1 ? "" : "s"} of other images are in the manager – untick “This image only” to list them.</div>}
        {rois.map((r, i) => {
          const other = img && r.image_id && r.image_id !== img.id ? imageName(r.image_id) : undefined;
          const shown = roiDisplayName(r);
          return (
            <div key={r.id} className={"roi-item" + (selectedIds.includes(r.id!) ? " sel" : "") + (other ? " foreign" : "")} onClick={(e) => pick(r, e)} onDoubleClick={() => setEditing(r.id!)} title={`${r.name || r.id}${other ? ` – belongs to ${other}` : ""}`}>
              <span className="idx">{i + 1}</span>
              {editing === r.id ? (
                <input autoFocus defaultValue={r.name} aria-label="ROI name" onBlur={(e) => void rename(r.id!, e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") void rename(r.id!, (e.target as HTMLInputElement).value); if (e.key === "Escape") setEditing(null); }} onClick={(e) => e.stopPropagation()} />
              ) : (
                <span className="nm">{shown}{other && <span className="roi-img" title={`ROI of ${other}`}>{other}</span>}</span>
              )}
              <span className="ty">{r.type}{r.position && (r.position.z > 0 || r.position.t > 0) ? ` z${r.position.z + 1}${r.position.t > 0 ? ` t${r.position.t + 1}` : ""}` : ""}</span>
            </div>
          );
        })}
      </div>
      <div className="btn-row">
        <button className="btn sm" onClick={() => void combine("or")} disabled={rois.length < 1} title="Union of the selected ROIs as a selection">OR</button>
        <button className="btn sm" onClick={() => void combine("and")} disabled={rois.length < 2}>AND</button>
        <button className="btn sm" onClick={() => void combine("xor")} disabled={rois.length < 2}>XOR</button>
        <button className="btn sm" onClick={() => void useStore.getState().runOp("overlay.from_manager", {})} disabled={!rois.length || !img} title="Burn as overlay">To overlay</button>
        <span style={{ flex: 1 }} />
        <button className="btn sm" onClick={() => fileRef.current?.click()} title="Import ROIs (JSON)">Import</button>
        <button className="btn sm" onClick={() => doAction("exportRois")} disabled={!allRois.length} title="Export every ROI as JSON" aria-label="Export ROIs"><Download size={12} /></button>
        <button className="btn sm danger" onClick={() => void useStore.getState().clearRoiManager()} disabled={!allRois.length} title={`Delete all ${allRois.length} ROIs of every image (asks first; Undo clear in the toast)`}>Clear…</button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void api.importRois(f).then(refresh); e.target.value = ""; }} />
      </div>
      <div className="hint-text">Click to select, {kbd("⇧")}/{kbd("⌘")}-click for multiple, double-click to rename. Selecting a ROI jumps to its slice.</div>
    </div>
  );
}

// ------------------------------------------------------------------ results
/** Section-header button: open the Results / last table maximised (ResultsModal). */
function ExpandButton({ kind, disabled }: { kind: "results" | "table"; disabled?: boolean }) {
  const setTableModal = useStore((s) => s.setTableModal);
  return (
    <button className="icon-btn" disabled={disabled} title="Open in a large window: sort, hide columns, copy / CSV" onClick={() => setTableModal(kind)}><Maximize2 size={13} /></button>
  );
}

function ResultsPanel() {
  const results = useStore((s) => s.state!.results);
  const measurements = useStore((s) => s.state!.measurements);
  const invoke = useStore((s) => s.invoke);
  const toast = useStore((s) => s.toast);
  const decimals = useStore((s) => s.state!.measure_options.decimals ?? 3);
  const img = useStore((s) => s.activeImage());
  const thisOnly = useStore((s) => s.resultsThisImageOnly);
  const setThisOnly = useStore((s) => s.setResultsThisImageOnly);
  const [selRow, setSelRow] = useState<number | null>(null);
  const { ref: wrapRef, overflow } = useHorizontalOverflow<HTMLDivElement>();
  // "This image only" (E5-11 / E6-3): rows whose Label names the active image; null = no Label column (cannot filter)
  const mine = useMemo(() => (img ? rowsOfImage(results.columns, results.rows, img.name) : null), [results, img?.name]); // eslint-disable-line react-hooks/exhaustive-deps
  const filtering = thisOnly && mine !== null && !!img;
  const rowIdx = useMemo(() => (filtering ? mine! : results.rows.map((_, i) => i)), [filtering, mine, results.rows]);
  const hiddenRows = results.rows.length - rowIdx.length;
  const copy = () => {
    const lines = [[" ", ...results.columns].join("\t"), ...results.rows.map((r, i) => [i + 1, ...r.map((v) => fmtCell(v, decimals))].join("\t"))];
    navigator.clipboard.writeText(lines.join("\n"));
    toast("Results copied (tab-separated)", "success");
  };
  const pickRow = (i: number) => {
    setSelRow(i);
    selectResultsRow(i); // ROI-manager name or Analyze-Particles "<image>:<n>" label → select + jump to its slice
  };
  return (
    <div>
      <div className="btn-row" style={{ marginTop: 0, marginBottom: 6 }}>
        <button className="btn sm" onClick={() => invoke("analyze.measure")} title="Measure (M)">Measure</button>
        <button className="btn sm" onClick={() => invoke("analyze.set_measurements")} title="Choose columns"><Settings2 size={12} /> Columns</button>
        <button className="btn sm" onClick={copy} disabled={!results.rows.length}><Copy size={12} /></button>
        <button className="btn sm" onClick={() => doAction("exportCsv")} disabled={!results.rows.length}><Download size={12} /> CSV</button>
        <button className="btn sm" onClick={() => invoke("analyze.summarize")} disabled={!results.rows.length} title="Summarize (mean, SD, min, max of every column)">Σ</button>
        <button className="btn sm danger" onClick={() => void useStore.getState().clearResultsTable()} disabled={!results.rows.length} title={`Delete all ${results.rows.length} rows (asks first; Undo clear in the toast)`}>Clear…</button>
      </div>
      {mine !== null && results.rows.length > 0 && (
        <label className="switch this-image" title="Show only the rows whose Label names the active image (the table keeps every image's rows; CSV and the large view export all of them)">
          <input type="checkbox" checked={thisOnly} onChange={(e) => setThisOnly(e.target.checked)} /> This image only{filtering && hiddenRows > 0 ? <span className="dim"> · {hiddenRows} row{hiddenRows === 1 ? "" : "s"} of other images hidden</span> : null}
        </label>
      )}
      {results.rows.length === 0 ? (
        <div className="hint-text">Empty. Press <span className="kbd">M</span> to measure the selection. Columns: {measurements.join(", ")}.</div>
      ) : rowIdx.length === 0 ? (
        <div className="hint-text">No rows of this image. {results.rows.length} row{results.rows.length === 1 ? "" : "s"} of other images – untick “This image only” to see them.</div>
      ) : (
        <div className={"table-wrap" + (overflow ? " scrollable" : "")} ref={wrapRef} tabIndex={0} aria-label="Results table (scrolls sideways)">
          <table className="results">
            <thead><tr><th>#</th>{results.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {rowIdx.slice(-2000).map((idx) => {
                const r = results.rows[idx];
                return (
                  <tr key={idx} className={selRow === idx ? "sel" : ""} onClick={() => pickRow(idx)}>
                    <td>{idx + 1}</td>
                    {r.map((v, k) => <td key={k}>{fmtCell(v, decimals)}</td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {rowIdx.length > 2000 && <div className="hint-text">Showing the last 2000 of {rowIdx.length} rows — <a href="#" onClick={(e) => { e.preventDefault(); useStore.getState().setTableModal("results"); }}>open the full table</a>.</div>}
        </div>
      )}
      {overflow && rowIdx.length > 0 && <div className="hint-text table-scroll-hint">More columns to the right – scroll sideways (⇧ wheel) or open the large view ⤢.</div>}
    </div>
  );
}

function fmtCell(v: unknown, decimals: number): string {
  if (v === null || v === undefined) return "NaN";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(decimals);
  return String(v);
}

function GenericTable({ table }: { table: { columns: string[]; rows: any[][] } }) {
  return (
    <div className="table-wrap">
      <table className="results">
        <thead><tr>{table.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{table.rows.map((r, i) => <tr key={i}>{r.map((v, k) => <td key={k}>{fmtCell(v, 3)}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ history
/**
 * History (E5-8 / E7-16): every undoable step with its parameters and scope (newer servers send `undo_detail`,
 * e.g. "sigma=5, scope=slice"), reachable with the keyboard (role=button, Tab, Enter / Space).  Clicking a step
 * undoes back to before it; a redo row re-applies up to it.
 */
function HistoryPanel({ img }: { img: ImageSummary }) {
  const hist = useStore((s) => s.state?.history[img.id]);
  const refresh = useStore((s) => s.refresh);
  if (!hist) return null;
  const undoTo = async (n: number) => {
    for (let i = 0; i < n; i++) await api.undo(img.id);
    await refresh();
  };
  const redoTo = async (n: number) => {
    for (let i = 0; i < n; i++) await api.redo(img.id);
    await refresh();
  };
  const key = (run: () => void) => (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      run();
    }
  };
  const undoDetail = hist.undo_detail || [];
  const redoDetail = hist.redo_detail || [];
  const redo = [...hist.redo].reverse();
  const redoDet = [...redoDetail].reverse();
  return (
    <div role="list" aria-label="Undo history">
      {hist.undo.length === 0 && hist.redo.length === 0 && <div className="hint-text">No changes yet. Every applied command can be undone.</div>}
      {hist.undo.map((l, i) => {
        const d = undoDetail[i] || "";
        const n = hist.undo.length - i;
        const title = `Undo back to before “${l}”${d ? ` (${d})` : ""} – ${n} step${n === 1 ? "" : "s"}`;
        return (
          <div key={"u" + i} className="hist-item" role="button" tabIndex={0} onClick={() => void undoTo(n)} onKeyDown={key(() => void undoTo(n))} title={title} aria-label={title}>
            <span className="n">{i + 1}</span><span className="hist-label">{l}{d && <span className="hist-detail">{d}</span>}</span>
          </div>
        );
      })}
      <div className="hist-item current" role="listitem" aria-current="step"><span className="n">●</span>current state</div>
      {redo.map((l, i) => {
        const d = redoDet[i] || "";
        const title = `Redo “${l}”${d ? ` (${d})` : ""}`;
        return (
          <div key={"r" + i} className="hist-item redo" role="button" tabIndex={0} onClick={() => void redoTo(i + 1)} onKeyDown={key(() => void redoTo(i + 1))} title={title} aria-label={title}>
            <span className="n">↻</span><span className="hist-label">{l}{d && <span className="hist-detail">{d}</span>}</span>
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ script: see ScriptPanel.tsx (Python / ImageJ macro / JavaScript editor + recorder)

function LogPanel() {
  const log = useStore((s) => s.state!.log);
  return (
    <div>
      {log.slice(-30).reverse().map((l, i) => (
        <div key={i} className={"log-line " + l.level}><span className="t">{new Date(l.t * 1000).toLocaleTimeString()}</span>{l.msg}</div>
      ))}
      {log.length === 0 && <div className="hint-text">Nothing logged yet.</div>}
    </div>
  );
}
