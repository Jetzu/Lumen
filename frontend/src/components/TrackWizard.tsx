/**
 * The TrackMate wizard inside Inspector ▸ Tracks: Detector → Initial threshold (quality histogram) →
 * Spot filters → Linker (nearest neighbour / simple LAP / LAP with gaps, splits, merges) → Track filters →
 * Display → Export, with Back / Next and a summary.  Every step runs the ordinary tracking command with the
 * values of its form (tracking.detect_spots, filter_spots, link_spots, filter_tracks, set_display, the
 * exporters), so the recorder / script see the same calls as the command palette.  The forms are built from
 * the commands' parameter specs; progress and values live in trackStore so the panel can be collapsed.
 */
import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Play, RotateCcw } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import type { ImageSummary, OpSpec, ParamSpec } from "../types";
import { fetchHistogram, trackSignature, useTrackStore, type SpotHistogram } from "../trackStore";
import NumberInput from "./NumberInput";

export const WIZARD_STEPS = ["Detector", "Initial threshold", "Spot filters", "Linker", "Track filters", "Display", "Export"];
const DETECT_FIELDS = ["detector", "radius", "threshold", "subpixel", "median_filter", "units", "z_mode"];
const LINK_FIELDS: Record<string, string[]> = {
  "Nearest neighbour": ["max_displacement", "min_track_length", "units"],
  "Simple LAP tracker": ["max_displacement", "gap_closing_frames", "max_gap_distance", "min_track_length", "units", "alternative_cost_factor", "linking_penalties", "gap_penalties"],
  "LAP tracker": ["max_displacement", "gap_closing_frames", "max_gap_distance", "allow_splitting", "split_max_distance", "allow_merging", "merge_max_distance", "min_track_length", "units", "alternative_cost_factor", "linking_penalties", "gap_penalties", "split_penalties", "merge_penalties"],
};
const TRACK_FILTER_FIELDS = ["min_length", "min_duration", "min_displacement", "max_speed", "feature", "threshold", "above"];
const DISPLAY_FIELDS = ["color_by", "colormap", "display_mode", "tail_length", "spot_radius", "show_names"];

function defaults(spec: OpSpec | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of spec?.params || []) out[p.name] = p.default;
  return out;
}

/** One parameter control, built from the command's ParamSpec (compact version of the operation sheet's fields). */
function Field({ spec, value, onChange }: { spec: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  const id = `tw-${spec.name}`;
  if (spec.type === "bool") {
    return (
      <div className="field check tw-field" title={spec.help}>
        <input id={id} type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        <label htmlFor={id}>{spec.label}</label>
      </div>
    );
  }
  if (spec.type === "choice") {
    return (
      <div className="field tw-field" title={spec.help}>
        <label htmlFor={id}>{spec.label}</label>
        <div className="ctrl"><select id={id} value={String(value ?? spec.default ?? "")} onChange={(e) => onChange(e.target.value)}>{(spec.choices || []).map((c) => <option key={c} value={c}>{c}</option>)}</select></div>
      </div>
    );
  }
  if (spec.type === "int" || spec.type === "float") {
    // a text field with a decimal point whatever the browser locale (E7-26: "2,5" for the spot radius)
    const n = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
    return (
      <div className="field tw-field" title={spec.help}>
        <label htmlFor={id}>{spec.label}{spec.unit ? <span className="unit">{spec.unit}</span> : null}</label>
        <div className="ctrl"><NumberInput id={id} value={Number.isFinite(n) ? n : null} min={spec.min} max={spec.max} step={spec.step ?? (spec.type === "int" ? 1 : "any")} integer={spec.type === "int"} onChange={(v) => onChange(v)} onInvalidBlur={() => onChange(spec.default)} /></div>
      </div>
    );
  }
  return (
    <div className="field tw-field" title={spec.help}>
      <label htmlFor={id}>{spec.label}</label>
      <div className="ctrl"><input id={id} type="text" value={String(value ?? "")} placeholder={spec.help ? spec.help.slice(0, 60) : undefined} onChange={(e) => onChange(e.target.value)} /></div>
    </div>
  );
}

function ParamForm({ opId, names, hidden }: { opId: string; names: string[]; hidden?: (name: string) => boolean }) {
  const spec = useStore((s) => s.ops[opId]);
  const stored = useTrackStore((s) => s.wizard.params[opId]);
  const values = useMemo(() => ({ ...defaults(spec), ...(stored || {}) }), [spec, stored]);
  if (!spec) return <div className="wf-req">Command {opId} is not available.</div>;
  const byName = new Map(spec.params.map((p) => [p.name, p] as const));
  return (
    <div className="tw-form">
      {names.map((n) => {
        const p = byName.get(n);
        if (!p || (hidden && hidden(n))) return null;
        return <Field key={n} spec={p} value={values[n]} onChange={(v) => useTrackStore.getState().setWizardParams(opId, { [n]: v })} />;
      })}
    </div>
  );
}

/** The values a step runs with: the command's defaults overridden by the form. */
function paramsFor(opId: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  const spec = useStore.getState().ops[opId];
  return { ...defaults(spec), ...(useTrackStore.getState().wizard.params[opId] || {}), ...extra };
}

/** "1 spots in 1 frames" → "1 spot in 1 frame", "1 point ROIs added" → "1 point ROI added" (E7-24). */
export function fixPlurals(msg: string): string {
  return msg
    .replace(/\b(\d+)\s+(point\s+)?(ROIs)\b/g, (_, n: string, pt: string | undefined, word: string) => (n === "1" ? `${n} ${pt || ""}${word.slice(0, -1)}` : `${n} ${pt || ""}${word}`))
    .replace(/\b(\d+)\s+(spots|frames|tracks|edges|links|particles|rows|points)\b/g, (_, n: string, word: string) => (n === "1" ? `${n} ${word.slice(0, -1)}` : `${n} ${word}`));
}

/** A step's result message that means "nothing to go on" (0 spots / 0 tracks): the step is not done (E5-13). */
export function stepFailed(msg: string | undefined): boolean {
  if (!msg) return false;
  return /\b0 (spots?|tracks?)\b/.test(msg) || /^no (spots|tracks)/i.test(msg);
}

async function runStep(img: ImageSummary, step: number, opId: string, params: Record<string, unknown>): Promise<string | null> {
  const st = useStore.getState();
  if (!st.ops[opId]) {
    st.toast(`${opId} is not available`, "error");
    return null;
  }
  const res = await st.runOp(opId, params, { silent: true, roi: null, imageId: img.id });
  if (!res) return null;
  const msg = fixPlurals(res.message || "done");
  const w = useTrackStore.getState();
  w.setWizard({ results: { ...w.wizard.results, [step]: msg } });
  if (stepFailed(msg)) st.toast(`${msg} – lower the quality threshold or the radius, then detect again`, "info");
  // the detection preview (Viewport.SpotPreview) draws the model's unlinked spots of the current frame: refetch the
  // model after every wizard step, also while the image has no track overlays yet (the signature-based refetch in
  // trackStore only fires for images that already show tracks)
  const cur = useStore.getState().image(img.id);
  if (cur) void useTrackStore.getState().fetchModel(img.id, trackSignature(cur.overlay as unknown[]), true);
  return msg;
}

// ------------------------------------------------------------------ quality histogram (initial threshold)
function QualityHistogram({ img, feature, refreshKey }: { img: ImageSummary; feature: string; refreshKey: string }) {
  const [hist, setHist] = useState<SpotHistogram | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const threshold = useTrackStore((s) => s.wizard.threshold);
  useEffect(() => {
    let alive = true;
    fetchHistogram(img.id, feature, 48).then((h) => {
      if (!alive) return;
      setHist(h);
      setErr(null);
      const w = useTrackStore.getState();
      if (w.wizard.threshold === null && h.min !== null) w.setWizard({ threshold: h.min });
    }).catch((e) => alive && setErr(e.message || String(e)));
    return () => {
      alive = false;
    };
  }, [img.id, feature, refreshKey]);
  if (err) return <div className="wf-req">{err}</div>;
  if (!hist || hist.n === 0 || hist.min === null || hist.max === null) return <div className="hint-text">No spots yet – run the detector first.</div>;
  const W = 260;
  const H = 90;
  const maxc = Math.max(1, ...hist.counts);
  const bw = W / hist.counts.length;
  const thr = threshold ?? hist.min;
  const lo = hist.min;
  const hi = hist.max;
  const xOf = (v: number) => (hi > lo ? ((v - lo) / (hi - lo)) * W : 0);
  let kept = 0;
  hist.counts.forEach((c, i) => {
    if (hist.edges[i + 1] > thr || (i === hist.counts.length - 1 && hist.edges[i] >= thr)) kept += c;
  });
  const step = hi > lo ? (hi - lo) / 200 : 1;
  return (
    <div className="tw-hist">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
        {hist.counts.map((c, i) => {
          const h = (c / maxc) * (H - 14);
          const on = hist.edges[i + 1] > thr;
          return <rect key={i} x={i * bw} y={H - 12 - h} width={Math.max(bw - 0.5, 0.5)} height={h} fill={on ? "var(--accent)" : "var(--bg4)"} />;
        })}
        <line x1={xOf(thr)} x2={xOf(thr)} y1={0} y2={H - 12} stroke="#ffd54a" strokeWidth={1.5} />
        <text x={2} y={H - 2} fontSize={8} fill="var(--text-dim)" fontFamily="var(--mono)">{lo.toPrecision(4)}</text>
        <text x={W - 2} y={H - 2} fontSize={8} textAnchor="end" fill="var(--text-dim)" fontFamily="var(--mono)">{hi.toPrecision(4)}</text>
      </svg>
      <div className="tw-hist-ctl">
        <input type="range" min={lo} max={hi} step={step} value={thr} aria-label={`${feature} threshold`} onChange={(e) => useTrackStore.getState().setWizard({ threshold: Number(e.target.value) })} />
        <NumberInput value={Number(thr.toPrecision(5))} min={lo} max={hi} step={step} aria-label={`${feature} threshold value`} onChange={(v) => useTrackStore.getState().setWizard({ threshold: v })} />
      </div>
      <div className="hint-text" style={{ margin: "2px 0 0" }}>{feature} ≥ {thr.toPrecision(4)}: keeps ≈ {kept} of {hist.n} spots ({hist.source === "tracks" ? "track spots" : "detected spots"}; median {hist.median?.toPrecision(4)})</div>
    </div>
  );
}

// ------------------------------------------------------------------ the wizard
export default function TrackWizard({ img }: { img: ImageSummary }) {
  const wizard = useTrackStore((s) => s.wizard);
  const ops = useStore((s) => s.ops);
  const tracksTable = useStore((s) => s.state?.tables?.Tracks);
  const [busy, setBusy] = useState(false);
  const [histKey, setHistKey] = useState("0");
  const step = wizard.step;
  const go = (s: number) => useTrackStore.getState().setWizard({ step: Math.max(0, Math.min(WIZARD_STEPS.length - 1, s)) });
  const run = async (opId: string, params: Record<string, unknown>) => {
    setBusy(true);
    try {
      const msg = await runStep(img, step, opId, params);
      if (msg) setHistKey(String(Date.now()));
      return msg;
    } finally {
      setBusy(false);
    }
  };
  const nTracks = tracksTable?.rows.length ?? 0;
  // a step counts as done only when it produced something (E5-13): no tick for "0 spots", and Next waits for the
  // detector (step 1) and the linker (step 4) to have succeeded
  const done = (i: number) => !!wizard.results[i] && !stepFailed(wizard.results[i]);
  const nextBlocked = (step === 0 && !done(0)) || (step === 3 && !done(3) && !nTracks);
  const nextReason = step === 0 ? "Detect spots first (a run with 0 spots does not count)" : step === 3 ? "Link the spots into tracks first" : undefined;
  const linkParams = paramsFor("tracking.link_spots");
  const tracker = String(linkParams.tracker || "Simple LAP tracker");
  /**
   * Detect spots.  The detections are previewed by the viewport on the frame each spot belongs to (Viewport ▸
   * SpotPreview reads the track model's unlinked spots, E8-4) – nothing is added to the shared ROI Manager any more
   * (E7-24 / E8-4: one "Spots t…" point ROI per frame showed every frame's spots at once and filled the manager).
   * "Spots t…" ROIs an earlier build left in the manager for this image are removed.
   */
  const detectSpots = async () => {
    const st = useStore.getState();
    const stale = (st.state?.rois || []).filter((r) => r.image_id === img.id && /^Spots t\d+$/.test(r.name || "")).map((r) => r.id!).filter(Boolean);
    if (stale.length) {
      try {
        await api.deleteRois(stale);
      } catch {
        /* keep going: the preview does not depend on them */
      }
    }
    await run("tracking.detect_spots", paramsFor("tracking.detect_spots", { add_to_manager: false, add_to_results: false }));
    useTrackStore.getState().setWizard({ threshold: null }); // step 2 seeds its threshold from the new detection, not the old one (E10-5)
  };
  const filterFeatures = ops["tracking.filter_spots"]?.params.find((p) => p.name === "feature")?.choices || ["Quality"];

  const body = (() => {
    switch (step) {
      case 0:
        return (
          <>
            <div className="wf-help">Detect spots on every frame of the current channel. The radius is the expected spot radius; threshold 0 picks the quality threshold automatically (refine it in the next step).</div>
            <ParamForm opId="tracking.detect_spots" names={DETECT_FIELDS} hidden={(n) => n === "z_mode" && img.shape.z <= 1} />
            {/* the detected spots are drawn on the image, frame by frame (Viewport ▸ SpotPreview), so a run is visible before Next (E7-24 / E8-4) */}
            <div className="wf-actions"><button className="btn sm primary" disabled={busy} onClick={() => void detectSpots()}><Play size={12} /> Detect spots</button></div>
            {done(0) && <div className="hint-text" style={{ margin: "2px 0 0" }}>The spots of the current frame are drawn on the image (step through the frames to check the others); nothing is added to the ROI Manager.</div>}
          </>
        );
      case 1:
        return (
          <>
            <div className="wf-help">Set the initial quality threshold on the histogram of the detected spots' quality; spots below it are discarded (undoable with Undo Track Edit).</div>
            <QualityHistogram img={img} feature="Quality" refreshKey={histKey} />
            <div className="wf-actions"><button className="btn sm primary" disabled={busy || wizard.threshold === null} onClick={() => void run("tracking.filter_spots", { feature: "Quality", threshold: wizard.threshold ?? 0, above: true })}><Play size={12} /> Apply threshold</button></div>
          </>
        );
      case 2:
        return (
          <>
            <div className="wf-help">Filter the spots on any feature (mean / median intensity, contrast, SNR, radius, position …). Each filter is applied with Filter Spots; add as many as needed.</div>
            {wizard.filters.map((f, i) => (
              <div key={i} className="tw-filter">
                <select value={f.feature} onChange={(e) => patchFilter(i, { feature: e.target.value, applied: undefined })}>{filterFeatures.map((c) => <option key={c} value={c}>{c}</option>)}</select>
                <select value={f.above ? "above" : "below"} onChange={(e) => patchFilter(i, { above: e.target.value === "above", applied: undefined })}><option value="above">≥</option><option value="below">≤</option></select>
                <NumberInput value={f.threshold} step="any" aria-label={`${f.feature} filter threshold`} onChange={(v) => patchFilter(i, { threshold: v, applied: undefined })} />
                <button className="btn sm" disabled={busy} title={f.applied ? f.applied : "Apply this filter"} onClick={() => void run("tracking.filter_spots", { feature: f.feature, threshold: f.threshold, above: f.above }).then((m) => m && patchFilter(i, { applied: m }))}>{f.applied ? "✓" : "Apply"}</button>
                <button className="icon-btn" title="Remove" onClick={() => useTrackStore.getState().setWizard({ filters: wizard.filters.filter((_, j) => j !== i) })}>×</button>
              </div>
            ))}
            <div className="wf-actions"><button className="btn sm" onClick={() => useTrackStore.getState().setWizard({ filters: [...wizard.filters, { feature: "SNR", threshold: 0, above: true }] })}>+ Add filter</button></div>
          </>
        );
      case 3:
        return (
          <>
            <div className="wf-help">Link the spots into tracks: the nearest-neighbour tracker, the simple LAP tracker (frame-to-frame + gap closing) or the LAP tracker with gap closing, track splitting and merging (lineage trees), optionally with feature penalties.</div>
            <ParamForm opId="tracking.link_spots" names={["tracker", ...(LINK_FIELDS[tracker] || LINK_FIELDS["Simple LAP tracker"])]} />
            <div className="wf-actions"><button className="btn sm primary" disabled={busy} onClick={() => void run("tracking.link_spots", paramsFor("tracking.link_spots"))}><Play size={12} /> Link spots</button></div>
          </>
        );
      case 4:
        return (
          <>
            <div className="wf-help">Keep only the tracks that pass the criteria: spots per track, duration, displacement, speed, or a threshold on any track feature (N splits, N gaps, Mean quality …).</div>
            <ParamForm opId="tracking.filter_tracks" names={TRACK_FILTER_FIELDS} />
            <div className="wf-actions"><button className="btn sm primary" disabled={busy || !nTracks} onClick={() => void run("tracking.filter_tracks", paramsFor("tracking.filter_tracks"))}><Play size={12} /> Filter tracks</button></div>
          </>
        );
      case 5:
        return (
          <>
            <div className="wf-help">Display settings for the viewer and the TrackScheme: colour by a track or spot feature through a colormap, how much of each track is drawn, tail length, spot radius and names.</div>
            <ParamForm opId="tracking.set_display" names={DISPLAY_FIELDS} />
            <div className="wf-actions"><button className="btn sm primary" disabled={busy} onClick={() => void run("tracking.set_display", paramsFor("tracking.set_display"))}><Play size={12} /> Apply display</button></div>
          </>
        );
      default:
        return (
          <>
            <div className="wf-help">Export the tracks: TrackMate XML (spots, lineage edges, features – opens in Fiji's TrackMate), the Spots / Edges / Tracks statistics as CSV, or show them as tables.</div>
            <div className="tw-form">
              <div className="field tw-field"><label>TrackMate XML file</label><div className="ctrl"><input type="text" value={wizard.exportXml} onChange={(e) => useTrackStore.getState().setWizard({ exportXml: e.target.value })} /><button className="btn sm" disabled={busy || !nTracks} onClick={() => void run("tracking.export_trackmate_xml", { path: wizard.exportXml, include_unlinked: false })}>Export</button></div></div>
              <div className="field tw-field"><label>CSV statistics prefix</label><div className="ctrl"><input type="text" value={wizard.exportCsv} onChange={(e) => useTrackStore.getState().setWizard({ exportCsv: e.target.value })} /><button className="btn sm" disabled={busy || !nTracks} onClick={() => void run("tracking.export_feature_tables", { path: wizard.exportCsv, which: "All", include_unlinked: false, trackmate_headers: false })}>Export</button></div></div>
            </div>
            <div className="wf-actions">
              <button className="btn sm" disabled={busy || !nTracks} onClick={() => void run("tracking.feature_tables", { include_unlinked: false, max_rows: 50000 })}>Feature tables</button>
              <button className="btn sm" disabled={busy || !nTracks} onClick={() => useTrackStore.getState().setSchemeOpen(true)}>TrackScheme</button>
            </div>
            <div className="tw-summary">
              <div className="wf-help" style={{ fontWeight: 600 }}>Summary</div>
              {WIZARD_STEPS.map((name, i) => <div key={i} className={"wf-out" + (wizard.results[i] ? "" : " dimmed")}>{name}: {wizard.results[i] || "not run"}</div>)}
            </div>
          </>
        );
    }
  })();

  function patchFilter(i: number, patch: Partial<{ feature: string; threshold: number; above: boolean; applied: string | undefined }>) {
    const w = useTrackStore.getState().wizard;
    useTrackStore.getState().setWizard({ filters: w.filters.map((f, j) => (j === i ? { ...f, ...patch } : f)) });
  }

  return (
    <div className="tw">
      <div className="tw-steps">
        {WIZARD_STEPS.map((name, i) => {
          // a chip cannot jump past a gate the Next button enforces (E9-5): detection must have found spots before
          // steps 2–7, and linking must have produced tracks before steps 5–7
          const gateIdx = [0, 3].find((g) => g < i && ((g === 0 && !done(0)) || (g === 3 && !done(3) && !nTracks)));
          const gated = gateIdx !== undefined;
          const gateReason = gateIdx === 0 ? "Run the detection first (step 1)" : gateIdx === 3 ? "Link the spots first (step 4)" : undefined;
          return (
          <button key={i} className={"tw-step" + (i === step ? " current" : "") + (done(i) ? " done" : "") + (wizard.results[i] && !done(i) ? " failed" : "") + (gated ? " gated" : "")} onClick={() => { if (!gated) go(i); }} disabled={gated} aria-disabled={gated || undefined} title={gated ? gateReason : wizard.results[i] || name} aria-current={i === step ? "step" : undefined}>
            <span className="wf-mark">{done(i) ? "✓" : wizard.results[i] ? "!" : i + 1}</span><span className="tw-step-name">{name}</span>
          </button>
          );
        })}
      </div>
      <div className="tw-body">
        <div className="wf-card-title">{step + 1}. {WIZARD_STEPS[step]}</div>
        {body}
        {wizard.results[step] && <div className="wf-out">{wizard.results[step]}</div>}
      </div>
      <div className="wf-foot">
        <button className="btn sm" disabled={step === 0} onClick={() => go(step - 1)}><ChevronLeft size={12} /> Back</button>
        <button className="btn sm" disabled={step === WIZARD_STEPS.length - 1 || nextBlocked} title={nextBlocked ? nextReason : undefined} onClick={() => go(step + 1)}>Next <ChevronRight size={12} /></button>
        {nextBlocked && <span className="wf-req">{nextReason}</span>}
        <span style={{ flex: 1 }} />
        <button className="btn sm" onClick={() => useTrackStore.getState().resetWizard()} title="Forget the wizard's values and results"><RotateCcw size={12} /> Reset</button>
      </div>
    </div>
  );
}
