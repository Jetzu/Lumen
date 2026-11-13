import type { ImageSummary, Meta, OpRunResult, PlaneData, Roi, SessionState, TypedArray } from "./types";

/**
 * Client id of this browser TAB (GUI fix rounds 2 and 3, E7-4 / E8-2): the server keeps one macro recorder per client,
 * keyed by the `X-Lumen-Client` header, so other tabs / users never mix their lines into ours.  The id is kept in
 * `sessionStorage` (per tab, survives a reload and a browser crash-restore) so a reload – or an accidental ⌘R – comes
 * back to the same recorder instead of starting an empty one and orphaning the session's recording on the server; a
 * new tab gets an id of its own.  Every same-origin `/api/` request made on the main thread carries the id (fetch is
 * wrapped once, below), including the calls of modules that do not go through `api`; GET /api/state and GET
 * /api/recorder therefore return this tab's lines right after the reload.
 */
const CLIENT_ID_KEY = "lumen-client-id";

function freshClientId(): string {
  try {
    const c = globalThis.crypto as Crypto | undefined;
    if (c?.randomUUID) return c.randomUUID().replace(/-/g, "").slice(0, 16);
  } catch {
    /* no crypto */
  }
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export const CLIENT_ID: string = (() => {
  // storage may be unavailable (private mode, disabled cookies, quota): fall back to a per-load id
  try {
    const kept = globalThis.sessionStorage?.getItem(CLIENT_ID_KEY);
    if (kept && /^[A-Za-z0-9_-]{8,64}$/.test(kept)) return kept;
  } catch {
    /* no storage */
  }
  const id = freshClientId();
  try {
    globalThis.sessionStorage?.setItem(CLIENT_ID_KEY, id);
  } catch {
    /* no storage */
  }
  return id;
})();
export const CLIENT_HEADER = "X-Lumen-Client";

if (typeof window !== "undefined" && typeof window.fetch === "function" && !(window as any).__lumenClientFetch) {
  const native = window.fetch.bind(window);
  (window as any).__lumenClientFetch = true;
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url.startsWith("/api/") || url.startsWith(`${location.origin}/api/`)) {
      const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
      if (!headers.has(CLIENT_HEADER)) headers.set(CLIENT_HEADER, CLIENT_ID);
      return native(input, { ...init, headers });
    }
    return native(input, init);
  };
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = j.error || j.detail || msg;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

const json = (body: unknown) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** Options of POST /api/images/open: `virtual: true` keeps the pixels in the file (planes on demand), `false` reads the
 *  file into memory, undefined = automatic (a virtual stack when the decoded size exceeds the auto_virtual_mb preference). */
export interface OpenOptions {
  virtual?: boolean;
  series?: number | null;
  level?: number;
}

const openImage = (path: string, opts: OpenOptions = {}) =>
  fetch("/api/images/open", json({ path, ...opts })).then((r) => handle<{ id: string; image?: ImageSummary }>(r));

export const api = {
  meta: () => fetch("/api/meta").then((r) => handle<Meta>(r)),
  luts: () => fetch("/api/luts").then((r) => handle<Record<string, number[][]>>(r)),
  /** LUTs grouped for the menu / select (lumen/server/extra/lut_groups.py); 404 on an older server → the client's own table. */
  lutGroups: () => fetch("/api/luts/groups").then((r) => handle<{ groups: { name: string; luts: string[] }[]; aliases: Record<string, string> }>(r)),
  state: () => fetch("/api/state").then((r) => handle<SessionState>(r)),
  fs: (path: string) => fetch(`/api/fs?path=${encodeURIComponent(path)}`).then((r) => handle<{ path: string; parent: string | null; entries: { name: string; path: string; dir: boolean; size: number; image: boolean }[] }>(r)),
  /** Open a server path (see OpenOptions); the response carries the new image's summary (`image.virtual` when it opened lazily). */
  open: openImage,
  openPath: (path: string, opts?: OpenOptions) => openImage(path, opts),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/images/upload", { method: "POST", body: fd }).then((r) => handle<{ id: string }>(r));
  },
  save: (id: string, path: string, format?: string) => fetch(`/api/images/${id}/save`, json({ path, format })).then((r) => handle<{ path: string }>(r)),
  close: (id: string) => fetch(`/api/images/${id}`, { method: "DELETE" }).then((r) => handle(r)),
  select: (id: string) => fetch(`/api/images/${id}/select`, { method: "POST" }).then((r) => handle(r)),
  display: (id: string, patch: Record<string, unknown>) => fetch(`/api/images/${id}/display`, json(patch)).then((r) => handle(r)),
  resetDisplay: (id: string) => fetch(`/api/images/${id}/reset_display`, { method: "POST" }).then((r) => handle(r)),
  undo: (id: string) => fetch(`/api/images/${id}/undo`, { method: "POST" }).then((r) => handle<{ label: string | null }>(r)),
  redo: (id: string) => fetch(`/api/images/${id}/redo`, { method: "POST" }).then((r) => handle<{ label: string | null }>(r)),
  value: (id: string, x: number, y: number, t: number, z: number) => fetch(`/api/images/${id}/value?x=${x}&y=${y}&t=${t}&z=${z}`).then((r) => handle<{ values: (number | null)[] | null }>(r)),
  run: (op: string, params: Record<string, unknown>, image_id?: string | null, roi?: Roi | null, scope = "stack") =>
    fetch(`/api/ops/${op}/run`, json({ params, image_id, roi, scope })).then((r) => handle<OpRunResult>(r)),
  preview: async (op: string, params: Record<string, unknown>, image_id: string, roi: Roi | null, scope: string, stride: number): Promise<PlaneData> => {
    const res = await fetch(`/api/ops/${op}/preview?stride=${stride}`, json({ params, image_id, roi, scope }));
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.error || res.statusText);
    }
    return decodePlanes(res);
  },
  planes: async (id: string, t: number, z: number, stride: number): Promise<PlaneData> => {
    const res = await fetch(`/api/images/${id}/planes?t=${t}&z=${z}&stride=${stride}`);
    if (!res.ok) throw new Error("plane fetch failed");
    return decodePlanes(res);
  },
  rois: () => fetch("/api/rois").then((r) => handle<Roi[]>(r)),
  addRoi: (roi: Roi, image_id: string | null, position?: { t: number; z: number; c: number }) => fetch("/api/rois", json({ roi, image_id, position })).then((r) => handle<Roi>(r)),
  patchRoi: (id: string, patch: Partial<Roi>) => fetch(`/api/rois/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patch }) }).then((r) => handle<Roi>(r)),
  deleteRois: (ids: string[]) => fetch("/api/rois/delete", json({ ids })).then((r) => handle(r)),
  /** Empties the ROI Manager; newer servers answer `{cleared, restorable}` and keep the entries for `restoreClear("rois")`. */
  clearRois: () => fetch("/api/rois", { method: "DELETE" }).then((r) => handle<{ ok: boolean; cleared?: number; restorable?: boolean }>(r)),
  /** One-step undo of the last ROI Manager clear / delete or Results clear (E7-6); `restored` = 0 when there is nothing. */
  restoreClear: (kind?: "rois" | "results") => fetch("/api/session/restore_clear", json({ kind: kind ?? null })).then((r) => handle<{ kind: string | null; restored: number }>(r)),
  combineRois: (ids: string[], mode: string, image_id: string | null) => fetch("/api/rois/combine", json({ ids, mode, image_id })).then((r) => handle<Roi>(r)),
  measureRois: (ids: string[]) => fetch("/api/rois/measure", json({ ids })).then((r) => handle<{ rows: number }>(r)),
  importRois: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/rois/import", { method: "POST", body: fd }).then((r) => handle<{ added: number }>(r));
  },
  clearResults: () => fetch("/api/results", { method: "DELETE" }).then((r) => handle<{ ok: boolean; cleared?: number; restorable?: boolean }>(r)),
  settings: (body: Record<string, unknown>) => fetch("/api/settings", json(body)).then((r) => handle(r)),
  /** Preferences (lumen/server/extra/settings.py). PUT is used because the core app owns POST /api/settings. */
  getSettings: () => fetch("/api/settings").then((r) => handle<Record<string, unknown>>(r)),
  putSettings: (patch: Record<string, unknown>) => fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch) }).then((r) => handle<Record<string, unknown>>(r)),
  /** Raw markdown of a bundled document (README | COMMANDS | FIJI_PARITY | RESEARCH | DESIGN). */
  doc: async (name: string): Promise<string> => {
    const res = await fetch(`/api/docs/${encodeURIComponent(name)}`);
    if (!res.ok) {
      let msg = res.statusText;
      try {
        const j = await res.json();
        msg = j.error || j.detail || msg;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }
    return res.text();
  },
  clearThreshold: (id: string) => fetch(`/api/threshold/${id}`, { method: "DELETE" }).then((r) => handle(r)),
  script: (code: string) => fetch("/api/script/run", json({ code })).then((r) => handle<{ stdout: string; error: string | null }>(r)),
  clearMacro: () => fetch("/api/macro", { method: "DELETE" }).then((r) => handle(r)),
  /** This client's recorder (per `X-Lumen-Client`): its lines, when it started and the image its last line targeted. */
  recorder: () => fetch("/api/recorder").then((r) => handle<{ client: string; started: number; started_iso: string; active_image: string | null; lines: string[]; count: number }>(r)),
  /** Empty this client's recorder (a fresh session marker is written); falls back to the session-wide DELETE /api/macro. */
  clearRecorder: async (): Promise<{ cleared?: number }> => {
    const res = await fetch("/api/recorder", { method: "DELETE" });
    if (res.status === 404 || res.status === 405) return fetch("/api/macro", { method: "DELETE" }).then((r) => handle<{ cleared?: number }>(r));
    return handle<{ cleared?: number }>(res);
  },
  /** Exactly the lines Apply Recorded Macro to Folder would run (open / save / select / display / undo lines skipped). */
  recorderPreview: () => fetch("/api/recorder/preview").then((r) => handle<{ lines: string[]; skipped: string[]; script: string; undone?: string[]; summary?: string }>(r)),
  /** The recorder as an ImageJ macro (`→ IJM`), when the server offers it: {lines, script}; a 404 means it does not. */
  recorderIjm: () => fetch("/api/recorder/ijm").then((r) => handle<{ lines?: string[]; script?: string; macro?: string; converted?: number; count?: number; skipped?: { line: string; reason: string }[]; notes?: string[]; undone?: string[]; removed?: number }>(r)),
  /** Plain text files on the server (lumen/server/extra/textfiles.py): the script editor's Open… / Save… and the Text Window's Save. */
  readTextFile: (path: string) => fetch(`/api/textfile?path=${encodeURIComponent(path)}`).then((r) => handle<{ path: string; name: string; text: string }>(r)),
  writeTextFile: (path: string, text: string) => fetch("/api/textfile", json({ path, text })).then((r) => handle<{ path: string; bytes: number }>(r)),
  /** Long-running operations: the jobs currently running (progress) and cancellation. */
  jobs: () => fetch("/api/jobs").then((r) => handle<JobInfo[]>(r)),
  cancelJob: (id: string) => fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }).then((r) => handle<{ ok: boolean; id: string }>(r)),
};

/** A running (or just finished) operation as reported by GET /api/jobs and the `progress` WebSocket events. */
export interface JobInfo {
  id: string;
  op: string;
  name: string;
  started: number;
  fraction: number;
  message: string;
  cancelled: boolean;
  elapsed: number;
  /** Only on the final event of a job. */
  done?: boolean;
  status?: "ok" | "error" | "cancelled";
}

async function decodePlanes(res: Response): Promise<PlaneData> {
  const dtype = res.headers.get("x-dtype") || "uint8";
  const shape = (res.headers.get("x-shape") || "").split(",").map(Number);
  const stride = Number(res.headers.get("x-stride") || 1);
  const buf = await res.arrayBuffer();
  const [c, h, w] = shape.length === 3 ? shape : [1, shape[0], shape[1]];
  const n = h * w;
  const channels: TypedArray[] = [];
  for (let i = 0; i < c; i++) {
    const off = i * n;
    if (dtype === "uint8") channels.push(new Uint8Array(buf, off, n));
    else if (dtype === "uint16") channels.push(new Uint16Array(buf, off * 2, n));
    else channels.push(new Float32Array(buf, off * 4, n));
  }
  return { dtype, shape: [c, h, w], stride, channels };
}
