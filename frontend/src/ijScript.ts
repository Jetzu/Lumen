/**
 * Fiji-script support for the Script panel: the Python / Jython runner with `#@` script parameters and the
 * viewport selection (POST /api/script/run), the parameter header (POST /api/script/params) and the release of a
 * JavaScript run's ImageJ bridge session (DELETE /api/ijshim/session/{token}; see jsRuntime.ts and
 * lumen/core/ijshim).
 */
import type { Roi } from "./types";

export interface ScriptParamDescriptor {
  name: string;
  /** int | float | bool | string | choice | file | directory | color | image | table | roimanager | service | message | other */
  type: string;
  java_type?: string;
  io?: "input" | "output" | "both";
  label: string;
  default: unknown;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  choices?: string[];
  style?: string;
  description?: string;
  visibility?: string;
}

export interface ScriptRunResult {
  stdout: string;
  error: string | null;
  /** Present once the script touched the ImageJ shim: its final selection (null = none). */
  roi?: Roi | null;
  parameters?: ScriptParamDescriptor[];
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

/** The parameter kinds the panel asks for before a run; the others (images, services …) are supplied automatically. */
export const PROMPT_TYPES = new Set(["int", "float", "bool", "string", "choice", "file", "directory", "color"]);

export const hasScriptParams = (code: string): boolean => /^\s*#\s*@/m.test(code);

/** A `.py` text that imports the ImageJ classes is a Jython script. */
export const looksLikeJython = (text: string): boolean => /^\s*(from|import)\s+(ij|java|jarray|loci|fiji)\b/m.test(text);

export const scriptParams = (code: string) =>
  fetch("/api/script/params", json({ code })).then((r) => handle<{ parameters: ScriptParamDescriptor[] }>(r)).then((j) => j.parameters);

export const runPythonScript = (code: string, opts: { params?: Record<string, unknown>; roi?: Roi | null; language?: "python" | "jython" } = {}) =>
  fetch("/api/script/run", json({ code, params: opts.params ?? null, roi: opts.roi ?? null, language: opts.language ?? "python" })).then((r) => handle<ScriptRunResult>(r));

export const releaseIjSession = (token: string) =>
  fetch(`/api/ijshim/session/${encodeURIComponent(token)}`, { method: "DELETE" }).then((r) => handle<{ found: boolean; roi: Roi | null }>(r));

export const JYTHON_EXAMPLE = `#@ Double (label="Sigma", value=2.0) sigma
# Jython (ImageJ 1.x API) – runs on Lumen's Python, no JVM
from ij import IJ, WindowManager
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable

imp = IJ.getImage()
IJ.log("%s: %dx%d, %d slices" % (imp.getTitle(), imp.getWidth(), imp.getHeight(), imp.getNSlices()))
IJ.run(imp, "Duplicate...", "title=work")
work = IJ.getImage()
IJ.run(work, "Gaussian Blur...", "sigma=%g" % sigma)
IJ.setAutoThreshold(work, "Otsu dark")
IJ.run(work, "Convert to Mask", "")
IJ.run(work, "Analyze Particles...", "size=20-Infinity display add")
rm = RoiManager.getInstance()
rt = ResultsTable.getResultsTable()
for i in range(rt.size()):
    print(i, rm.getName(i), rt.getValue("Area", i))
print("%d particles" % rm.getCount())
`;

export const JS_IJ_EXAMPLE = `// Fiji JavaScript (ImageJ 1.x API): synchronous calls through Lumen's ij bridge
importClass(Packages.ij.IJ);
importClass(Packages.ij.plugin.frame.RoiManager);
var imp = IJ.getImage();
IJ.log(imp.getTitle() + " " + imp.getWidth() + "x" + imp.getHeight());
IJ.run(imp, "Duplicate...", "title=work");
var work = IJ.getImage();
IJ.run(work, "Gaussian Blur...", "sigma=2");
IJ.setAutoThreshold(work, "Otsu dark");
IJ.run(work, "Convert to Mask", "");
IJ.run(work, "Analyze Particles...", "size=20-Infinity display add");
var rm = RoiManager.getInstance();
var rt = ResultsTable.getResultsTable();
for (var i = 0; i < rt.size(); i++) IJ.log(i + " " + rm.getName(i) + " " + rt.getValue("Area", i));
IJ.log(rm.getCount() + " particles");
`;
