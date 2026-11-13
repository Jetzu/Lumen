/**
 * State of the script editor (Inspector ▸ Script & Recorder, Plugins ▸ Scripting ▸ Script Editor): the code,
 * its language, the server path it came from, the output console and two "requests" other parts of the UI
 * can make (focus the editor, run the code).  Lives outside the main store so that the Text Window, the
 * menus and the shortcuts can push code into the editor without rendering the whole app.
 */
import { create } from "zustand";
import type { ScriptParamDescriptor } from "./ijScript";
import { useStore } from "./store";
import type { Roi } from "./types";

/** "jython" is Python run by the same server-side runner, with the ImageJ 1.x classes (`from ij import IJ`) – see lumen/core/ijshim.
 *  "groovy" / "beanshell" / "rhino" (and the opt-in "jruby" / "clojure") are Fiji's JVM languages: they need the optional Java
 *  bridge (`pip install "lumen-imaging[jvm]"` + a JDK; lumen/core/ijshim/jvm_bridge.py) and run on the server's JVM with the
 *  real ImageJ 1.x classes. */
export type ScriptLang = "python" | "jython" | "ijm" | "js" | "groovy" | "beanshell" | "rhino" | "jruby" | "clojure";

export const SCRIPT_LANGS: { id: ScriptLang; label: string; ext: string; jvm?: boolean }[] = [
  { id: "python", label: "Python", ext: ".py" },
  { id: "jython", label: "Jython (ImageJ 1.x API)", ext: ".py" },
  { id: "ijm", label: "ImageJ macro", ext: ".ijm" },
  { id: "js", label: "JavaScript", ext: ".js" },
  // short labels: the inspector's select is narrow (E5-15 / E7-14); the panel's hint explains the Java bridge
  { id: "groovy", label: "Groovy (JVM bridge)", ext: ".groovy", jvm: true },
  { id: "beanshell", label: "BeanShell (JVM bridge)", ext: ".bsh", jvm: true },
  { id: "rhino", label: "Rhino JavaScript (JVM bridge)", ext: ".js", jvm: true },
  { id: "jruby", label: "JRuby (JVM bridge, opt-in)", ext: ".rb", jvm: true },
  { id: "clojure", label: "Clojure (JVM bridge, opt-in)", ext: ".clj", jvm: true },
];

/** The languages run by the server's optional JVM bridge (POST /api/script/run with `language`). */
export const JVM_LANGS = new Set<ScriptLang>(SCRIPT_LANGS.filter((l) => l.jvm).map((l) => l.id));
export const isJvmLang = (lang: ScriptLang): boolean => JVM_LANGS.has(lang);

export interface ScriptOutput {
  stdout: string;
  error: string | null;
}

export const DEFAULT_CODE: Record<ScriptLang, string> = {
  python: 'img = lumen.image\nprint(img.name, img.shape)\nlumen.run("filters.gaussian", sigma=2)\n',
  jython: '#@ Double (label="Sigma", value=2.0) sigma\nfrom ij import IJ\nimp = IJ.getImage()\nIJ.run(imp, "Gaussian Blur...", "sigma=%g" % sigma)\nIJ.log("%s %dx%d" % (imp.getTitle(), imp.getWidth(), imp.getHeight()))\n',
  ijm: 'run("Gaussian Blur...", "sigma=2");\nrun("Measure");\nprint("done: " + nResults);\n',
  js: '// JavaScript runs in a sandboxed worker; every lumen.* call is async\nconst img = await lumen.getImage();\nlumen.log(img ? `${img.name} ${img.shape.x}×${img.shape.y}` : "no image");\nawait lumen.run("filters.gaussian", { sigma: 2 });\nconst rows = await lumen.measure();\nlumen.log(rows);\n',
  groovy: '#@ Double (label="Sigma", value=2.0) sigma\n// Groovy on the server\'s JVM with the real ImageJ 1.x (Java bridge); the active image is bound as imp\nimport ij.IJ\nIJ.run(imp, "Gaussian Blur...", "sigma=" + sigma)\nIJ.log(imp.getTitle() + " " + imp.getWidth() + "x" + imp.getHeight() + " mean " + imp.getStatistics().mean)\n',
  beanshell: '// BeanShell on the server\'s JVM (Java bridge); ij.* is imported, the active image is bound as imp\nIJ.run(imp, "Invert", "");\nIJ.log("inverted " + imp.getTitle());\n',
  rhino: '// Rhino JavaScript on the server\'s JVM (Java bridge): importClass / Packages work, every ImageJ call is direct\nimportClass(Packages.ij.IJ);\nIJ.run(imp, "Invert", "");\nIJ.log("inverted " + imp.getTitle());\n',
  jruby: '# JRuby on the server\'s JVM (Java bridge; start Lumen with LUMEN_JVM_LANGUAGES=jruby)\njava_import "ij.IJ"\nIJ.run(imp, "Invert", "")\nIJ.log("inverted #{imp.getTitle}")\n',
  clojure: ';; Clojure on the server\'s JVM (Java bridge; start Lumen with LUMEN_JVM_LANGUAGES=clojure)\n(import \'ij.IJ)\n(IJ/run imp "Invert" "")\n(IJ/log (str "inverted " (.getTitle imp)))\n',
};

const KEY_CODE = "lumen.script";
const KEY_LANG = "lumen.script.lang";

/** Language from a file name / path (.py → Python, or Jython when the text imports the ImageJ classes; .jy → Jython;
 *  .ijm / .txt → ImageJ macro, .js → JavaScript; .groovy / .gvy → Groovy, .bsh → BeanShell, .rb → JRuby, .clj → Clojure
 *  (the Java bridge). A .js file stays the browser's JavaScript – pick Rhino by hand when it needs the JVM). */
export function guessLang(nameOrPath: string, fallback: ScriptLang = "python", text?: string): ScriptLang {
  const n = nameOrPath.toLowerCase();
  if (/\.(ijm|txt)$/.test(n)) return "ijm";
  if (/\.(m?js)$/.test(n)) return fallback === "rhino" ? "rhino" : "js";
  if (/\.(groovy|gvy)$/.test(n)) return "groovy";
  if (/\.bsh$/.test(n)) return "beanshell";
  if (/\.rb$/.test(n)) return "jruby";
  if (/\.clj$/.test(n)) return "clojure";
  if (/\.jy$/.test(n)) return "jython";
  if (/\.py$/.test(n)) return text !== undefined && /^\s*(from|import)\s+(ij|java|jarray|loci|fiji)\b/m.test(text) ? "jython" : fallback === "jython" ? "jython" : "python";
  return fallback;
}

export function langExt(lang: ScriptLang): string {
  return SCRIPT_LANGS.find((l) => l.id === lang)?.ext || ".txt";
}

interface ScriptStore {
  code: string;
  lang: ScriptLang;
  /** Server path the script was opened from or saved to (Save… reuses the folder and name). */
  path: string | null;
  output: ScriptOutput | null;
  running: boolean;
  /** Bumped by "Script Editor" (menu, `[`): the panel opens its section and focuses the textarea. */
  focusTick: number;
  /** Bumped by run requests from outside the panel (Text Window ▸ Run as …). */
  runTick: number;
  setCode: (code: string) => void;
  setLang: (lang: ScriptLang) => void;
  setPath: (p: string | null) => void;
  setOutput: (o: ScriptOutput | null) => void;
  appendOutput: (text: string) => void;
  setRunning: (v: boolean) => void;
  /** Put code (and its language) into the editor and show it; `run` also runs it right away. */
  load: (code: string, lang?: ScriptLang, opts?: { run?: boolean; path?: string | null }) => void;
  /** Open the inspector's Script section and focus the editor. */
  focus: () => void;
  /** The editor shown as a large dialog over the image ("Open in a large view", like the 3-D view; E7-14). */
  largeOpen: boolean;
  setLargeOpen: (v: boolean) => void;
  /** The editor text differs from what was loaded / saved last (guards "→ Python" & co, E7-15). */
  dirty: boolean;
  setDirty: (v: boolean) => void;
}

function savedLang(): ScriptLang {
  try {
    const v = localStorage.getItem(KEY_LANG);
    return SCRIPT_LANGS.some((l) => l.id === v) ? (v as ScriptLang) : "python";
  } catch {
    return "python";
  }
}

function savedCode(): string {
  try {
    return localStorage.getItem(KEY_CODE) ?? DEFAULT_CODE[savedLang()];
  } catch {
    return DEFAULT_CODE.python;
  }
}

function showPanel() {
  const st = useStore.getState();
  st.setInspectorOpen(true);
  st.togglePanel("script", true);
}

export const useScriptStore = create<ScriptStore>((set, get) => ({
  code: savedCode(),
  lang: savedLang(),
  path: null,
  output: null,
  running: false,
  focusTick: 0,
  runTick: 0,
  setCode: (code) => {
    set({ code });
    try {
      localStorage.setItem(KEY_CODE, code);
    } catch {
      /* quota / private mode */
    }
  },
  setLang: (lang) => {
    set({ lang });
    try {
      localStorage.setItem(KEY_LANG, lang);
    } catch {
      /* ignore */
    }
  },
  setPath: (path) => set({ path }),
  setOutput: (output) => set({ output }),
  appendOutput: (text) => {
    const o = get().output || { stdout: "", error: null };
    set({ output: { ...o, stdout: o.stdout + text + (text.endsWith("\n") ? "" : "\n") } });
  },
  setRunning: (running) => set({ running }),
  load: (code, lang, opts = {}) => {
    get().setCode(code);
    if (lang) get().setLang(lang);
    set({ path: opts.path === undefined ? get().path : opts.path, output: null });
    showPanel();
    if (opts.run) set({ runTick: get().runTick + 1 });
    else set({ focusTick: get().focusTick + 1 });
  },
  focus: () => {
    showPanel();
    set({ focusTick: get().focusTick + 1 });
  },
  largeOpen: false,
  setLargeOpen: (v) => set({ largeOpen: v, focusTick: v ? get().focusTick + 1 : get().focusTick }),
  dirty: false,
  setDirty: (v) => set({ dirty: v }),
}));

/**
 * E8-2: an accidental ⌘R or tab close must not throw away editor text that is neither saved to a file nor one of the
 * default snippets – the browser asks first.  (The text itself is also mirrored in localStorage; the guard covers the
 * cases where that is not enough: a closed tab, cleared site data.)  The recorder needs no guard: it lives on the
 * server under this tab's client id and is back after a reload (api.ts CLIENT_ID).
 */
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", (e) => {
    const s = useScriptStore.getState();
    if (!s.dirty || !s.code.trim() || Object.values(DEFAULT_CODE).includes(s.code)) return;
    e.preventDefault();
    e.returnValue = ""; // older browsers show their prompt only with a value set
  });
}

/** What POST /api/script/run answers for a JVM language (lumen.core.ijshim.jvm_bridge.run_jvm_script). A missing bridge
 *  (no `[jvm]` extra / no JDK) is reported in `error` with the install steps, not as an HTTP error. */
export interface JvmRunResult {
  stdout: string;
  error: string | null;
  language: string;
  /** Present when an image was bound as `imp`: the script's final selection (null = none). */
  roi?: Roi | null;
  parameters?: ScriptParamDescriptor[];
  /** Ids of the images the script created and show()-ed (already added to the session). */
  new_images?: string[];
  /** The bound image's pixels were replaced (an undo entry was recorded). */
  changed?: boolean;
  results_rows?: number;
  roi_manager?: boolean;
  rois_copied?: number;
  /** e.g. the ROI Manager being unavailable on a headless JVM. */
  note?: string;
}

/** Run a Groovy / BeanShell / Rhino / JRuby / Clojure script through the server's Java bridge (the `#@` values in `params`,
 *  the viewport selection in `roi`). */
export async function runJvmScript(code: string, opts: { language: ScriptLang; params?: Record<string, unknown>; roi?: Roi | null }): Promise<JvmRunResult> {
  const res = await fetch("/api/script/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, params: opts.params ?? null, roi: opts.roi ?? null, language: opts.language }),
  });
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
  return (await res.json()) as JvmRunResult;
}
