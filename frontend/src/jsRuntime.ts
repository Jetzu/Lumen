/**
 * JavaScript scripting runtime for the script editor.
 *
 * User code runs in a sandboxed Web Worker (no DOM, no network of its own: fetch / XMLHttpRequest / WebSocket /
 * importScripts are removed) that only owns a `lumen` object.  Every `lumen.*` method is a promise that is
 * answered by this module over the worker's message channel, which then talks to the REST API — so scripts are
 * written with async/await:
 *
 *     const img = await lumen.getImage();
 *     await lumen.run("filters.gaussian", { sigma: 2 });
 *     const rows = await lumen.measure();
 *     lumen.log(rows);
 *
 * lumen API (worker side): run(id, params, {roi, scope, image}) · getImage() · getImages() · getState() · log(...)
 * · openSample(name) · open(path) · save(path, format) · select(nameOrId) · close(nameOrId) · measure(params)
 * · getResults() · setResults(rows) · clearResults() · getRois() · addRoi(roi) · setRoi(roi) · clearRoi()
 * · macro(ijmText) · python(code) · commands(search) · help(id) · wait(ms).
 * `print(...)` is an alias of `lumen.log`, console.log/warn/error go to the output console too.
 *
 * ImageJ 1.x API (Fiji JavaScript scripts): the worker also offers `importClass` / `importPackage`, `Packages.ij.*`
 * / `Packages.java.*` and the auto-imported class names (`IJ`, `ImagePlus`, `WindowManager`, `RoiManager`,
 * `ResultsTable`, `GenericDialog`, `Roi` …, as ImageJ's JavaScript evaluator does), implemented by the Python shim
 * in lumen/core/ijshim through a synchronous bridge (one XMLHttpRequest per call to POST /api/ijshim/call), so
 *
 *     importClass(Packages.ij.IJ);
 *     var imp = IJ.getImage(); IJ.run(imp, "Gaussian Blur...", "sigma=2"); IJ.log(imp.getTitle());
 *
 * runs unchanged with its ordinary synchronous semantics. Objects (ImagePlus, ImageProcessor, ResultsTable …) are
 * proxies over server-side handles: methods are calls, bean fields (`stats.mean`, `rect.width`, `cal.pixelWidth`)
 * are fetched / stored on access, class constants (`Measurements.AREA`) are read once per class. The bridge
 * session (and the script's selection) is released when the script ends; the network access of user code stays
 * limited to this one endpoint.
 */
import { api } from "./api";
import { useStore } from "./store";
import type { ImageSummary, OpRunResult, Roi } from "./types";

export interface JsRunHandle {
  /** Resolves when the script finished (error = null) or failed / was stopped. */
  done: Promise<{ error: string | null; result?: string }>;
  stop: () => void;
}

/** Source of the sandbox worker (kept as a raw string so that `\n` reaches the worker unchanged). */
const WORKER_SOURCE = String.raw`"use strict";
const XHR = self.XMLHttpRequest; // kept privately for the ImageJ bridge below; user code gets no network access of its own
self.fetch = undefined; self.XMLHttpRequest = undefined; self.WebSocket = undefined; self.importScripts = undefined;
let seq = 0;
const pending = new Map();
function rpc(method, args) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    self.postMessage({ type: "rpc", id, method, args: args || [] });
  });
}
function fmt(v) {
  if (v === undefined) return "undefined";
  if (typeof v === "string") return v;
  if (v instanceof Error) return v.stack || v.message || String(v);
  try { return JSON.stringify(v, null, 1); } catch (e) { return String(v); }
}
const log = (...a) => self.postMessage({ type: "log", text: a.map(fmt).join(" ") });
const wait = (ms) => new Promise((r) => setTimeout(r, Math.max(0, Number(ms) || 0)));
const lumen = {
  run: (id, params, opts) => rpc("run", [id, params || {}, opts || {}]),
  getImage: () => rpc("getImage"),
  getImages: () => rpc("getImages"),
  getState: () => rpc("getState"),
  log,
  print: log,
  openSample: (name) => rpc("run", ["file.sample", { name }, {}]),
  open: (path) => rpc("open", [path]),
  save: (path, format) => rpc("save", [path, format]),
  select: (key) => rpc("select", [key]),
  close: (key) => rpc("close", [key]),
  undo: (key) => rpc("undo", [key]),
  redo: (key) => rpc("redo", [key]),
  display: (key, patch) => rpc("display", [key, patch || {}]),
  measure: (params) => rpc("measure", [params || {}]),
  getResults: () => rpc("getResults"),
  setResults: (rows) => rpc("setResults", [rows]),
  clearResults: () => rpc("clearResults"),
  getRois: () => rpc("getRois"),
  addRoi: (roi) => rpc("addRoi", [roi]),
  setRoi: (roi) => rpc("setRoi", [roi]),
  clearRoi: () => rpc("setRoi", [null]),
  macro: (text) => rpc("run", ["script.run_ij_macro", { macro: String(text) }, {}]),
  python: (code) => rpc("python", [String(code)]),
  commands: (search) => rpc("commands", [search || ""]),
  help: (id) => rpc("help", [id]),
  wait,
};
Object.freeze(lumen);
console.log = console.info = console.debug = log;
console.warn = (...a) => log("⚠", ...a);
console.error = (...a) => log("✖", ...a);

// ---- ImageJ 1.x API bridge: synchronous calls to the Python shim (lumen/core/ijshim/bridge.py) ----
let ijBase = "", ijToken = "", ijRoi = null, ijFirst = true, ijIndex = null;
const ijDescr = new Map(), ijClasses = new Map();
const H = Symbol("ijHandle"), C = Symbol("ijClass");
function ijEncode(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isNaN(v) ? { __nan__: true } : v === Infinity ? { __inf__: 1 } : v === -Infinity ? { __inf__: -1 } : v;
  if (typeof v === "boolean" || typeof v === "string") return v;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "function" || typeof v === "object") {
    if (v[H] !== undefined) return { __handle__: v[H] };
    if (v[C] !== undefined) return { __classref__: v[C] };
    if (typeof v === "function") return null;
    if (Array.isArray(v)) return v.map(ijEncode);
    if (ArrayBuffer.isView(v)) return Array.from(v, ijEncode);
    const o = {};
    for (const k of Object.keys(v)) o[k] = ijEncode(v[k]);
    return o;
  }
  return String(v);
}
function ijDecode(v) {
  if (v === null || typeof v !== "object") return v;
  if (Array.isArray(v)) return v.map(ijDecode);
  if (v.__nan__) return NaN;
  if (v.__inf__ !== undefined) return v.__inf__ > 0 ? Infinity : -Infinity;
  if (v.__handle__ !== undefined) return ijHandle(v.__handle__, v.__class__);
  if (v.__classref__ !== undefined) return ijClass(v.__classref__);
  const o = {};
  for (const k of Object.keys(v)) o[k] = ijDecode(v[k]);
  return o;
}
function ijRpc(target, method, args) {
  if (!ijToken) throw new Error("The ImageJ API is not available in this run");
  const x = new XHR();
  x.open("POST", ijBase + "/api/ijshim/call", false);
  x.setRequestHeader("Content-Type", "application/json");
  const body = { token: ijToken, target: target, method: method, args: (args || []).map(ijEncode) };
  if (ijFirst) { body.roi = ijRoi; ijFirst = false; }
  x.send(JSON.stringify(body));
  let j = null;
  try { j = JSON.parse(x.responseText); } catch (e) { j = null; }
  if (j && j.log) for (const line of j.log) log(line);
  if (!j) throw new Error("ImageJ bridge: HTTP " + x.status + " for " + target + "." + method);
  if (j.error) throw new Error(j.error);
  return ijDecode(j.result);
}
function ijDescribe(path) {
  let d = ijDescr.get(path);
  if (!d) { d = ijRpc(path, "__describe__", []); ijDescr.set(path, d); }
  return d;
}
function ijClass(path) {
  let cls = ijClasses.get(path);
  if (cls) return cls;
  cls = new Proxy(function () {}, {
    get(t, name) {
      if (name === C) return path;
      if (name === H) return undefined;
      if (typeof name !== "string" || name === "then" || name === "toJSON") return undefined;
      if (name === "toString" || name === "valueOf") return () => "[JavaClass " + path + "]";
      const d = ijDescribe(path);
      if (Object.prototype.hasOwnProperty.call(d.consts || {}, name)) return ijDecode(d.consts[name]);
      if (d.nested && d.nested.includes(name)) return ijClass(path + "." + name);
      if (d.statics && d.statics.includes(name)) return ijRpc(path, "__getattr__", [name]);
      return (...args) => ijRpc(path, name, args);
    },
    set(t, name, v) { ijRpc(path, "__setattr__", [name, v]); return true; },
    construct(t, args) { return ijRpc(path, "__new__", args); },
    has() { return true; },
  });
  ijClasses.set(path, cls);
  return cls;
}
function ijHandle(id, cls) {
  return new Proxy({}, {
    get(t, name) {
      if (name === H) return id;
      if (name === C) return undefined;
      if (name === Symbol.toPrimitive) return () => ijRpc(id, "toString", []);
      if (typeof name !== "string" || name === "then" || name === "toJSON") return undefined;
      if (name === "toString" || name === "valueOf") return () => ijRpc(id, "toString", []);
      const d = cls ? ijDescribe(cls) : { props: [], consts: {} };
      if (d.props && d.props.includes(name)) return ijRpc(id, "__getattr__", [name]);
      if (Object.prototype.hasOwnProperty.call(d.consts || {}, name)) return ijDecode(d.consts[name]);
      return (...args) => ijRpc(id, name, args);
    },
    set(t, name, v) { ijRpc(id, "__setattr__", [name, v]); return true; },
    has() { return true; },
  });
}
const P = Symbol("ijPackage");
function ijPackage(path) {
  return new Proxy({}, {
    get(t, name) {
      if (name === P) return path;
      if (typeof name !== "string" || name === "then" || name === "toJSON") return undefined;
      if (name === "toString") return () => "[JavaPackage " + path + "]";
      const p = path ? path + "." + name : name;
      return /^[A-Z]/.test(name) ? ijClass(p) : ijPackage(p);
    },
  });
}
const Packages = ijPackage("");
const ijImported = new Map();
const ijPackages = [];
function importClass(cls) {
  const p = cls && cls[C];
  if (!p) throw new Error("importClass expects a class, e.g. importClass(Packages.ij.IJ)");
  ijImported.set(p.split(".").pop(), cls);
}
function importPackage(pkg) {
  const p = typeof pkg === "string" ? pkg : pkg && pkg[P];
  if (!p) throw new Error("importPackage expects a package, e.g. importPackage(Packages.ij.plugin)");
  if (!ijPackages.includes(p)) ijPackages.unshift(p);
}
const JS_GLOBALS = new Set(["Math", "JSON", "Object", "Array", "String", "Number", "Boolean", "Date", "RegExp", "Error", "TypeError", "RangeError", "SyntaxError", "ReferenceError",
  "EvalError", "URIError", "AggregateError", "Map", "Set", "WeakMap", "WeakSet", "WeakRef", "Promise", "Symbol", "Infinity", "NaN", "Proxy", "Reflect", "Function", "Uint8Array",
  "Uint8ClampedArray", "Int8Array", "Uint16Array", "Int16Array", "Uint32Array", "Int32Array", "Float32Array", "Float64Array", "BigInt64Array", "BigUint64Array", "ArrayBuffer",
  "SharedArrayBuffer", "DataView", "BigInt", "Intl", "Atomics", "TextEncoder", "TextDecoder", "URL", "URLSearchParams", "Blob", "Worker", "AbortController", "Iterator",
  "Headers", "Request", "Response", "FinalizationRegistry", "Packages"]);
function ijLookup(name) {
  if (ijImported.has(name)) return ijImported.get(name);
  if (!/^[A-Z]/.test(name) || JS_GLOBALS.has(name) || !ijToken) return undefined;
  for (const p of ijPackages) {
    const d = ijDescribe(p);
    if (d.classes && d.classes.includes(name)) return ijClass(p + "." + name);
  }
  if (!ijIndex) ijIndex = ijRpc("", "__index__", []);
  const path = ijIndex[name];
  return path ? ijClass(path) : undefined;
}
// the scope user code runs in (a with-block): ImageJ's auto-imported class names resolve here, everything else falls through
const SCOPE_NAMES = new Set(["Packages", "java", "importClass", "importPackage"]);
const ijScope = new Proxy({}, {
  has(t, name) { return typeof name === "string" && (SCOPE_NAMES.has(name) || ijLookup(name) !== undefined); },
  get(t, name) {
    if (typeof name !== "string") return undefined;
    if (name === "Packages") return Packages;
    if (name === "java") return ijPackage("java");
    if (name === "importClass") return importClass;
    if (name === "importPackage") return importPackage;
    return ijLookup(name);
  },
  set(t, name, v) { ijImported.set(name, v); return true; },
});

self.onmessage = async (ev) => {
  const m = ev.data || {};
  if (m.type === "rpc-result") {
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.ok) p.resolve(m.result); else p.reject(new Error(m.error || "failed"));
    return;
  }
  if (m.type !== "run") return;
  ijBase = m.ijBase || ""; ijToken = m.ijToken || ""; ijRoi = m.ijRoi === undefined ? null : m.ijRoi; ijFirst = true;
  try {
    // sloppy-mode function so that the with-block (the ImageJ name space) is allowed; user code keeps its own strictness
    const fn = new Function("lumen", "print", "wait", "__scope", "return (async () => { with (__scope) {\n" + m.code + "\n} })();");
    const result = await fn(lumen, log, wait, ijScope);
    self.postMessage({ type: "done", error: null, result: result === undefined ? undefined : fmt(result) });
  } catch (e) {
    self.postMessage({ type: "done", error: fmt(e) });
  }
};
`;

let workerUrl: string | null = null;
function getWorkerUrl(): string {
  if (!workerUrl) workerUrl = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: "text/javascript" }));
  return workerUrl;
}

interface HostContext {
  /** Selection used by subsequent run() calls (lumen.setRoi), null = the viewport's selection. */
  roi: Roi | null | undefined;
}

function findImage(key: unknown): ImageSummary | undefined {
  const st = useStore.getState();
  if (key === undefined || key === null || key === "") return st.activeImage();
  const k = String(key);
  return st.image(k) || st.state?.images.find((im) => im.name === k);
}

function rowsAsObjects(results: { columns: string[]; rows: any[][] } | undefined): Record<string, unknown>[] {
  if (!results) return [];
  return results.rows.map((r) => Object.fromEntries(results.columns.map((c, i) => [c, r[i]])));
}

/** The side effects the UI applies after a command (selection, tables, plots), as `store.runOp` does. */
function applyRunResult(res: OpRunResult, img: ImageSummary | undefined) {
  const st = useStore.getState();
  const target = res.new_images && res.new_images.length ? st.image(res.new_images[res.new_images.length - 1]) : img;
  if (res.roi && target) st.setSelection(target.id, res.roi);
  if (res.clear_roi && target) st.setSelection(target.id, null);
  if (res.plot) {
    st.setPlot(res.plot);
    st.togglePanel("plot", true);
  }
  if (res.histogram && res.histogram.edges) {
    st.setHistogram(res.histogram);
    st.togglePanel("histogram", true);
  }
  if (res.table) {
    useStore.setState({ table: res.table });
    st.togglePanel("table", true);
  }
  if (res.results_added) st.togglePanel("results", true);
}

const handlers: Record<string, (ctx: HostContext, ...args: any[]) => Promise<unknown>> = {
  run: async (ctx, id: string, params: Record<string, unknown>, opts: { roi?: Roi | null; scope?: string; image?: string }) => {
    const st = useStore.getState();
    const spec = st.ops[id] || Object.values(st.ops).find((o) => o.name.toLowerCase() === String(id).toLowerCase());
    if (!spec) throw new Error(`Unknown command ${id}`);
    const img = findImage(opts.image);
    const roi = opts.roi !== undefined ? opts.roi : ctx.roi !== undefined ? ctx.roi : img ? st.selections[img.id] ?? null : null;
    await st.flushDisplay();
    const res = await api.run(spec.id, params || {}, img?.id ?? null, roi, opts.scope || "stack");
    await useStore.getState().refresh();
    applyRunResult(res, img);
    return res;
  },
  getImage: async () => {
    await useStore.getState().refresh();
    return useStore.getState().activeImage() ?? null;
  },
  getImages: async () => {
    await useStore.getState().refresh();
    return useStore.getState().state?.images ?? [];
  },
  getState: async () => api.state(),
  open: async (_ctx, path: string) => {
    const r = await api.openPath(String(path));
    await useStore.getState().refresh();
    return useStore.getState().image(r.id) ?? r;
  },
  save: async (_ctx, path: string, format?: string) => {
    const img = useStore.getState().activeImage();
    if (!img) throw new Error("No image is open");
    const r = await api.save(img.id, String(path), format);
    await useStore.getState().refresh();
    return r.path;
  },
  select: async (_ctx, key: string) => {
    const img = findImage(key);
    if (!img) throw new Error(`No image ${key}`);
    await api.select(img.id);
    await useStore.getState().refresh();
    return img;
  },
  close: async (_ctx, key?: string) => {
    const img = findImage(key);
    if (!img) throw new Error(`No image ${key ?? ""}`);
    await api.close(img.id);
    await useStore.getState().refresh();
    return true;
  },
  // recorded lumen.undo("title") / lumen.redo("title") lines (the recorder writes them, GUI fix round 2)
  undo: async (_ctx, key?: string) => {
    const img = findImage(key);
    if (!img) throw new Error(`No image ${key ?? ""}`);
    await useStore.getState().flushDisplay();
    const r = await api.undo(img.id);
    await useStore.getState().refresh();
    return r.label;
  },
  redo: async (_ctx, key?: string) => {
    const img = findImage(key);
    if (!img) throw new Error(`No image ${key ?? ""}`);
    await useStore.getState().flushDisplay();
    const r = await api.redo(img.id);
    await useStore.getState().refresh();
    return r.label;
  },
  /** lumen.display("title", {channel: 2, min, max, lut, visible, inverted, gamma, mode, t, z, c}) – channel is 1-based like the recorder. */
  display: async (_ctx, key: string, patch: Record<string, unknown>) => {
    const img = findImage(key);
    if (!img) throw new Error(`No image ${key ?? ""}`);
    const p = patch || {};
    const out: Record<string, unknown> = {};
    for (const k of ["mode", "t", "z", "c"]) if (p[k] !== undefined) out[k] = p[k];
    const chanKeys = ["min", "max", "lut", "visible", "inverted", "gamma", "name"];
    if (chanKeys.some((k) => p[k] !== undefined)) {
      const n = img.display.channels.length;
      const idx = p.channel === undefined ? img.display.c : Math.max(0, Math.min(n - 1, Number(p.channel) - 1));
      const ch: Record<string, unknown> = {};
      for (const k of chanKeys) if (p[k] !== undefined) ch[k] = p[k];
      out.channels = Array.from({ length: n }, (_, i) => (i === idx ? ch : {}));
    }
    await useStore.getState().flushDisplay();
    await api.display(img.id, out);
    await useStore.getState().refresh();
    return useStore.getState().image(img.id)?.display ?? null;
  },
  measure: async (ctx, params: Record<string, unknown>) => {
    const before = useStore.getState().state?.results.rows.length ?? 0;
    const res = (await handlers.run(ctx, "analyze.measure", params || {}, {})) as OpRunResult;
    const results = useStore.getState().state?.results;
    const n = res.results_added ?? Math.max(0, (results?.rows.length ?? 0) - before);
    return rowsAsObjects(results).slice(-n);
  },
  getResults: async () => {
    const st = await api.state();
    return rowsAsObjects(st.results);
  },
  setResults: async (_ctx, rows: unknown) => {
    const list = Array.isArray(rows) ? rows : rows && typeof rows === "object" && Array.isArray((rows as any).rows) ? rowsAsObjects(rows as any) : null;
    if (!list) throw new Error("setResults expects an array of row objects");
    const literal = JSON.stringify(JSON.stringify(list)); // a JSON string literal is also a valid Python string literal
    const code = `import json as _json\n_rows = _json.loads(${literal})\nlumen.clear_results()\nlumen.session.results.append(_rows)\nlumen.session._changed("results")\n`;
    const r = await api.script(code);
    if (r.error) throw new Error(r.error);
    await useStore.getState().refresh();
    useStore.getState().togglePanel("results", true);
    return list.length;
  },
  clearResults: async () => {
    await api.clearResults();
    await useStore.getState().refresh();
    return true;
  },
  getRois: async () => api.rois(),
  addRoi: async (_ctx, roi: Roi) => {
    const img = useStore.getState().activeImage();
    const r = await api.addRoi(roi, img?.id ?? null, img ? { t: img.display.t, z: img.display.z, c: img.display.c } : undefined);
    await useStore.getState().refresh();
    return r;
  },
  setRoi: async (ctx, roi: Roi | null) => {
    ctx.roi = roi;
    const img = useStore.getState().activeImage();
    if (img) useStore.getState().setSelection(img.id, roi);
    return roi;
  },
  python: async (_ctx, code: string) => {
    const r = await api.script(code);
    await useStore.getState().refresh();
    if (r.error) throw new Error(r.error);
    return r.stdout;
  },
  commands: async (_ctx, search: string) => {
    const s = String(search || "").toLowerCase();
    return Object.values(useStore.getState().ops)
      .filter((o) => o.id.toLowerCase().includes(s) || o.name.toLowerCase().includes(s))
      .map((o) => ({ id: o.id, name: o.name, menu: o.menu }));
  },
  help: async (_ctx, id: string) => {
    const o = useStore.getState().ops[id];
    if (!o) throw new Error(`Unknown command ${id}`);
    return [`${o.name}  [${o.id}]  – ${o.description}`, ...o.params.map((p) => `  ${p.name}: ${p.type} = ${JSON.stringify(p.default)}${p.choices ? ` choices=${JSON.stringify(p.choices)}` : ""}  ${p.help || ""}`)].join("\n");
  },
};

/** A fresh token for the ImageJ bridge session of one script run. */
function newToken(): string {
  const c = (globalThis as any).crypto;
  if (c?.randomUUID) return c.randomUUID();
  return `js-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** Run JavaScript in the sandbox worker; `onLog` receives every lumen.log / console line as it happens. */
export function runJs(code: string, opts: { onLog?: (text: string) => void } = {}): JsRunHandle {
  const ctx: HostContext = { roi: undefined };
  let worker: Worker;
  try {
    worker = new Worker(getWorkerUrl());
  } catch (e: any) {
    return { done: Promise.resolve({ error: `Cannot start the script worker: ${e?.message || e}` }), stop: () => undefined };
  }
  const ijToken = newToken();
  const startImg = useStore.getState().activeImage();
  const ijRoi = startImg ? useStore.getState().selections[startImg.id] ?? null : null;
  // end the bridge session on the server; when the script used the ImageJ API, its final selection reaches the viewport
  const release = async () => {
    try {
      const { releaseIjSession } = await import("./ijScript");
      const r = await releaseIjSession(ijToken);
      if (r.found) {
        const st = useStore.getState();
        const img = st.activeImage();
        if (img) st.setSelection(img.id, r.roi ?? null);
      }
    } catch {
      /* the session was never created or the server is gone */
    }
  };
  let finish!: (r: { error: string | null; result?: string }) => void;
  let finished = false;
  const done = new Promise<{ error: string | null; result?: string }>((resolve) => {
    finish = (r) => {
      if (finished) return;
      finished = true;
      worker.terminate();
      void release().finally(() => resolve(r));
    };
  });
  worker.onmessage = (ev: MessageEvent) => {
    const m = ev.data || {};
    if (m.type === "log") opts.onLog?.(String(m.text));
    else if (m.type === "done") finish({ error: m.error ?? null, result: m.result });
    else if (m.type === "rpc") {
      const h = handlers[m.method];
      const reply = (ok: boolean, payload: unknown) => {
        if (finished) return;
        try {
          worker.postMessage(ok ? { type: "rpc-result", id: m.id, ok: true, result: payload ?? null } : { type: "rpc-result", id: m.id, ok: false, error: String(payload) });
        } catch (e: any) {
          worker.postMessage({ type: "rpc-result", id: m.id, ok: false, error: `result is not transferable: ${e?.message || e}` });
        }
      };
      if (!h) return reply(false, `lumen.${m.method} is not available`);
      h(ctx, ...(Array.isArray(m.args) ? m.args : [])).then((r) => reply(true, r), (e) => reply(false, e?.message || e));
    }
  };
  worker.onerror = (e) => finish({ error: e.message || "worker error" });
  worker.postMessage({ type: "run", code, ijBase: window.location.origin, ijToken, ijRoi });
  return { done, stop: () => finish({ error: "Stopped" }) };
}

// ---------------------------------------------------------------- recorder → JavaScript
/** Python literal (as the recorder writes it: repr) → JavaScript / JSON source. */
export function pyLiteralToJs(src: string): string {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const ch = src[i];
    if (ch === "'" || ch === '"') {
      // a Python string: re-quote as a JSON string
      let j = i + 1;
      let s = "";
      while (j < n && src[j] !== ch) {
        if (src[j] === "\\" && j + 1 < n) {
          const e = src[j + 1];
          if (e === "n") s += "\n";
          else if (e === "t") s += "\t";
          else if (e === "r") s += "\r";
          else if (e === "\\") s += "\\";
          else if (e === "'" || e === '"') s += e;
          else if (e === "x" && j + 3 < n) {
            s += String.fromCharCode(parseInt(src.slice(j + 2, j + 4), 16));
            j += 2;
          } else s += e;
          j += 2;
        } else s += src[j++];
      }
      out += JSON.stringify(s);
      i = j + 1;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      out += word === "True" ? "true" : word === "False" ? "false" : word === "None" ? "null" : word === "inf" ? "Infinity" : word === "nan" ? "NaN" : word;
      i = j;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

/** Split `k=v, k2=v2` at top-level commas (brackets and quotes respected). */
function splitKwargs(s: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | null = null;
  let cur = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      cur += ch;
      if (ch === "\\" && i + 1 < s.length) cur += s[++i];
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"') quote = ch;
    else if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    if (ch === "," && depth === 0) {
      parts.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

const RUN_OPTS = new Set(["roi", "scope", "image"]);

/**
 * One recorder line → JavaScript: `lumen.run("id", a=1, roi={...})`, `lumen.open(...)`, `lumen.save(...)`,
 * `lumen.select("title")` / `close` / `undo` / `redo`, and `lumen.display("title", channel=2, min=1, lut='Fire')`
 * (→ `await lumen.display("title", { channel: 2, min: 1, lut: "Fire" });`).  A trailing `  # → "title"` note
 * (the name a new image got) is dropped first.
 */
export function recorderLineToJs(line: string): string {
  let t = line.trim();
  if (!t) return "";
  if (t.startsWith("#")) return "//" + t.slice(1);
  t = t.replace(/\s+#\s*→.*$/, "").trim();
  const disp = /^lumen\.display\(\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*(?:,\s*([\s\S]*))?\)\s*$/.exec(t);
  if (disp) {
    const title = pyLiteralToJs(disp[1]);
    const fields: string[] = [];
    for (const kv of splitKwargs(disp[2] || "")) {
      const eq = kv.indexOf("=");
      if (eq < 0) continue;
      const key = kv.slice(0, eq).trim();
      fields.push(`${/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ? key : JSON.stringify(key)}: ${pyLiteralToJs(kv.slice(eq + 1).trim())}`);
    }
    return `await lumen.display(${title}, { ${fields.join(", ")} });`;
  }
  const m = /^lumen\.run\(\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*(?:,\s*([\s\S]*))?\)\s*$/.exec(t);
  if (m) {
    const id = pyLiteralToJs(m[1]);
    const params: string[] = [];
    const opts: string[] = [];
    for (const kv of splitKwargs(m[2] || "")) {
      const eq = kv.indexOf("=");
      if (eq < 0) continue;
      const key = kv.slice(0, eq).trim();
      const val = pyLiteralToJs(kv.slice(eq + 1).trim());
      const entry = `${/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ? key : JSON.stringify(key)}: ${val}`;
      (RUN_OPTS.has(key) ? opts : params).push(entry);
    }
    return `await lumen.run(${id}, { ${params.join(", ")} }${opts.length ? `, { ${opts.join(", ")} }` : ""});`;
  }
  const simple = /^lumen\.(open|save|select|close|undo|redo)\(([\s\S]*)\)\s*$/.exec(t);
  if (simple) return `await lumen.${simple[1]}(${splitKwargs(simple[2]).map((a) => pyLiteralToJs(a)).join(", ")});`;
  return "// " + t;
}

/** The recorder's Python lines as a runnable JavaScript script. */
export function recorderToJs(lines: string[]): string {
  const body = lines.map(recorderLineToJs).filter((l) => l !== "");
  return "// Recorded by Lumen – JavaScript (async; every lumen call returns a promise)\n" + body.join("\n") + "\n";
}
