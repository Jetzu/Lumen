/**
 * Shared state of the tracking UI (Inspector ▸ Tracks list, the TrackMate-style wizard and the TrackScheme):
 * the track model of the active image as served by GET /api/tracking/model (spots with features, lineage
 * edges, display settings, undo depth), the spot selection of the TrackScheme, and the wizard's progress.
 * Nothing here is persisted: the model lives on the server (session.settings) and is refetched whenever the
 * image's track overlays change or a tracking command completes (see the store subscription at the end).
 */
import { create } from "zustand";
import { useStore } from "./store";

export interface ModelSpot {
  id: number; t: number; x: number; y: number; z: number; r: number;
  quality: number | null; intensity: number | null; median: number | null; min: number | null; max: number | null; total: number | null;
  std: number | null; contrast: number | null; snr: number | null; visibility: number | null;
  name?: string; manual?: boolean;
}
export interface ModelTrack { id: string; name: string; color: string; manual: boolean; overlay_id: string; stats: Record<string, number | null>; spots: ModelSpot[]; edges: [number, number][] }
export interface DisplayFeature { kind: "track" | "spot"; name: string; key: string; lo: number; hi: number }
export interface TrackDisplay { color_by: string; colormap: string; display_mode: string; tail_length: number; spot_radius: boolean; show_names: boolean; feature: DisplayFeature | null }
export interface TrackModel {
  image: string; version: number; nt: number; calibration: Record<string, unknown>; tracks: ModelTrack[]; free_spots: ModelSpot[]; n_free_spots: number; n_spots: number;
  display: TrackDisplay; undo_depth: number; spot_features: string[]; track_features: string[]; color_by: string[]; display_modes: string[]; colormaps: string[];
}
export interface SpotHistogram { feature: string; counts: number[]; edges: number[]; min: number | null; max: number | null; median: number | null; n: number; source: string }

export interface SpotFilter { feature: string; threshold: number; above: boolean; applied?: string }
export interface WizardState {
  /** 0 Detector · 1 Initial threshold · 2 Spot filters · 3 Linker · 4 Track filters · 5 Display · 6 Export */
  step: number;
  /** Parameter values per command id, as edited in the step forms. */
  params: Record<string, Record<string, unknown>>;
  /** The message of the last run of each step (summary). */
  results: Record<number, string>;
  filters: SpotFilter[];
  threshold: number | null;
  exportXml: string;
  exportCsv: string;
}

const EMPTY_WIZARD: WizardState = { step: 0, params: {}, results: {}, filters: [], threshold: null, exportXml: "tracks.xml", exportCsv: "tracks" };

interface TrackStore {
  model: TrackModel | null;
  /** Signature of the overlay the model was fetched for (see trackSignature) – a changed signature triggers a refetch. */
  modelKey: string | null;
  modelLoading: boolean;
  modelError: string | null;
  fetchModel: (imageId: string, key: string, force?: boolean) => Promise<TrackModel | null>;
  schemeOpen: boolean;
  setSchemeOpen: (v: boolean) => void;
  /** Spots picked in the TrackScheme (ids of the track model). */
  selectedSpotIds: number[];
  setSelectedSpotIds: (ids: number[]) => void;
  panelMode: "list" | "wizard";
  setPanelMode: (m: "list" | "wizard") => void;
  wizard: WizardState;
  setWizard: (patch: Partial<WizardState>) => void;
  setWizardParams: (opId: string, patch: Record<string, unknown>) => void;
  resetWizard: () => void;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = j.error || j.detail || msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export function fetchHistogram(imageId: string, feature: string, bins = 48): Promise<SpotHistogram> {
  return fetch(`/api/tracking/histogram?image_id=${encodeURIComponent(imageId)}&feature=${encodeURIComponent(feature)}&bins=${bins}`).then((r) => handle<SpotHistogram>(r));
}

/** The track overlay id ("track-3") a branch item ("track-3-b1") belongs to. */
export function trackKeyOfOverlayId(id: string): string {
  return id.replace(/-b\d+$/, "");
}

/** Numeric track id the tracking ops take, from an overlay item id ("track-12" / "manual-3-b1" → 12 / 3). */
export function numericTrackId(id: string): number | null {
  const m = /^(?:track|manual)-(\d+)/.exec(id);
  return m ? Number(m[1]) : null;
}

/**
 * A cheap signature of the track overlays (ids, lengths, colours): the de-duplication key of the model fetch.  It
 * changes when tracks appear, vanish, grow or are recoloured – not on display settings, a replaced point or the undo
 * depth, which is why a completed tracking command refetches the model regardless (subscription below).
 */
export function trackSignature(overlay: unknown[]): string {
  const parts: string[] = [];
  for (const o of overlay as { type?: string; id?: string; color?: string; points?: unknown[] }[]) {
    if (o.type === "track") parts.push(`${o.id}:${o.points?.length ?? 0}:${o.color ?? ""}`);
  }
  return parts.join("|");
}

let seq = 0;
let inflight: { imageId: string; key: string; seq: number; promise: Promise<TrackModel | null> } | null = null;

export const useTrackStore = create<TrackStore>((set, get) => ({
  model: null,
  modelKey: null,
  modelLoading: false,
  modelError: null,
  fetchModel: (imageId, key, force = false) => {
    const st = get();
    if (!force && st.model && st.model.image === imageId && st.modelKey === key) return Promise.resolve(st.model);
    // the same model is already on its way (the forced refetch after a tracking command, then the panel's own
    // effect with the same signature): share that request instead of fetching twice
    if (!force && inflight && inflight.imageId === imageId && inflight.key === key) return inflight.promise;
    const my = ++seq;
    set({ modelLoading: true, modelError: null });
    const promise = (async () => {
      try {
        const model = await fetch(`/api/tracking/model?image_id=${encodeURIComponent(imageId)}`).then((r) => handle<TrackModel>(r));
        if (my !== seq) return get().model; // a newer fetch is on its way
        const valid = new Set(model.tracks.flatMap((t) => t.spots.map((s) => s.id)));
        set({ model, modelKey: key, modelLoading: false, selectedSpotIds: get().selectedSpotIds.filter((id) => valid.has(id)) });
        return model;
      } catch (e: any) {
        if (my === seq) set({ modelLoading: false, modelError: e?.message || String(e) });
        return null;
      } finally {
        if (inflight && inflight.seq === my) inflight = null;
      }
    })();
    inflight = { imageId, key, seq: my, promise };
    return promise;
  },
  schemeOpen: false,
  setSchemeOpen: (v) => set({ schemeOpen: v }),
  selectedSpotIds: [],
  setSelectedSpotIds: (ids) => set({ selectedSpotIds: ids }),
  panelMode: "list",
  setPanelMode: (m) => set({ panelMode: m }),
  wizard: EMPTY_WIZARD,
  setWizard: (patch) => set({ wizard: { ...get().wizard, ...patch } }),
  setWizardParams: (opId, patch) => {
    const w = get().wizard;
    set({ wizard: { ...w, params: { ...w.params, [opId]: { ...(w.params[opId] || {}), ...patch } } } });
  },
  resetWizard: () => set({ wizard: { ...EMPTY_WIZARD, params: {}, results: {}, filters: [] } }),
}));

/**
 * The model changes on the server in ways the overlay signature cannot see – display settings (tracking.set_display:
 * display mode, tail length, spot radius, names), a replaced point with the same count and colour, the undo depth,
 * the free-spot count – so a completed tracking command refetches it.  store.runOp records the command in `lastRun`
 * before the refresh that carries the new overlays: act on that refresh.  The Tracks panel / TrackScheme effects,
 * which key their fetch on the same signature, join the in-flight request instead of fetching a second time.
 */
let handledRun = 0;
useStore.subscribe((s, prev) => {
  if (s.state === prev.state || !s.lastRun || s.lastRun.seq === handledRun || !s.lastRun.opId.startsWith("tracking.")) return;
  handledRun = s.lastRun.seq;
  const im = s.activeImage();
  if (!im) return;
  const ts = useTrackStore.getState();
  const key = trackSignature(im.overlay as unknown[]);
  // a model is shown for this image, or it has tracks someone may be looking at (the viewer reads the display mode
  // from the model even while the Tracks section is collapsed): refetch; an image without either has nothing to show
  if (ts.model?.image === im.id || key !== "") void ts.fetchModel(im.id, key, true);
});
