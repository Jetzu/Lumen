/**
 * A small syntax highlighter for the script editor (no dependency): Python / Jython, the ImageJ macro language,
 * JavaScript / Rhino, Groovy / BeanShell, JRuby and Clojure.  It tokenises line by line into comments, strings,
 * numbers, keywords, built-ins (the `lumen` API, ImageJ's run()/macro functions, `IJ`), and calls; the editor draws the
 * result in a <pre> layer under a transparent <textarea>, so typing, selection, undo and every shortcut stay the
 * browser's own.  Multi-line strings / block comments carry their state from line to line.
 */
import type { ScriptLang } from "./scriptStore";

export type TokenKind = "kw" | "str" | "com" | "num" | "fn" | "bi" | "op" | "txt";
export interface Token { kind: TokenKind; text: string }

const KW: Record<string, string[]> = {
  py: ["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "print"],
  ijm: ["var", "function", "macro", "if", "else", "for", "while", "do", "return", "true", "false", "NaN", "PI", "requires", "eval", "exit", "break", "continue"],
  js: ["var", "let", "const", "function", "return", "if", "else", "for", "while", "do", "break", "continue", "new", "this", "true", "false", "null", "undefined", "typeof", "instanceof", "in", "of", "try", "catch", "finally", "throw", "class", "extends", "async", "await", "importClass", "importPackage", "switch", "case", "default"],
  groovy: ["def", "import", "class", "return", "if", "else", "for", "while", "in", "new", "true", "false", "null", "try", "catch", "finally", "throw", "static", "void", "int", "double", "float", "boolean", "String", "as", "instanceof", "switch", "case", "default", "break", "continue"],
  rb: ["def", "end", "do", "if", "elsif", "else", "unless", "while", "until", "for", "in", "return", "class", "module", "require", "true", "false", "nil", "and", "or", "not", "begin", "rescue", "ensure", "yield", "then", "puts", "java_import"],
  clj: ["def", "defn", "fn", "let", "if", "do", "when", "loop", "recur", "import", "ns", "require", "doseq", "dotimes", "nil", "true", "false"],
};

const BUILTINS: Record<string, string[]> = {
  py: ["lumen", "np", "numpy", "scipy", "skimage", "ndi", "IJ", "ImagePlus", "WindowManager", "len", "range", "list", "dict", "int", "float", "str", "sum", "min", "max", "abs", "enumerate", "zip", "open"],
  ijm: ["run", "open", "close", "selectWindow", "selectImage", "getTitle", "setAutoThreshold", "setThreshold", "setMinAndMax", "makeRectangle", "makeOval", "makePolygon", "makeLine", "roiManager", "nResults", "getResult", "setResult", "print", "newImage", "saveAs", "getDirectory", "getFileList", "File", "Stack", "Array", "Math", "String", "IJ", "Ext", "getPixel", "setPixel", "getWidth", "getHeight", "nSlices", "nImages", "setBatchMode", "waitForUser", "getNumber", "getString", "Dialog", "List", "Overlay", "Roi", "Table", "Property",
        "setSlice", "getSliceNumber", "getDimensions", "getVoxelSize", "getPixelSize", "getInfo", "getMetadata", "getValue", "getStatistics", "getRawStatistics", "getHistogram",
        "getThreshold", "resetThreshold", "resetMinAndMax", "getMinAndMax", "setColor", "setForegroundColor", "setBackgroundColor", "setLineWidth", "fill", "drawLine", "drawString",
        "drawRect", "fillRect", "getSelectionBounds", "getSelectionCoordinates", "selectionType", "isOpen", "getImageID", "getList", "lengthOf", "substring", "indexOf", "replace",
        "split", "toString", "d2s", "round", "floor", "parseInt", "parseFloat", "isNaN", "pow", "sqrt", "abs", "minOf", "maxOf", "random", "showProgress", "showStatus", "wait",
        "beep", "call", "exec", "getTime", "setOption", "updateDisplay", "rename", "getArgument", "setPasteMode", "makePoint", "makeSelection", "getBoolean", "getDateAndTime",
        "nImages", "IJ.log", "updateResults", "String.append", "Color"],
  js: ["lumen", "IJ", "ImagePlus", "WindowManager", "console", "Math", "JSON", "Array", "Object", "Promise"],
};

function family(lang: ScriptLang): "py" | "ijm" | "js" | "groovy" | "rb" | "clj" {
  if (lang === "python" || lang === "jython") return "py";
  if (lang === "ijm") return "ijm";
  if (lang === "js" || lang === "rhino") return "js";
  if (lang === "groovy" || lang === "beanshell") return "groovy";
  if (lang === "jruby") return "rb";
  return "clj";
}

/** State carried between lines: inside a block comment or a triple-quoted string. */
type State = null | { until: string; kind: TokenKind };

function lineComment(fam: string): string[] {
  if (fam === "py" || fam === "rb") return ["#"];
  if (fam === "clj") return [";"];
  return ["//"];
}

function tokenizeLine(line: string, fam: string, kw: Set<string>, bi: Set<string>, state: State): { tokens: Token[]; state: State } {
  const out: Token[] = [];
  let i = 0;
  const push = (kind: TokenKind, text: string) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ kind, text });
  };
  if (state) {
    const end = line.indexOf(state.until);
    if (end < 0) return { tokens: [{ kind: state.kind, text: line }], state };
    push(state.kind, line.slice(0, end + state.until.length));
    i = end + state.until.length;
    state = null;
  }
  const comments = lineComment(fam);
  while (i < line.length) {
    const rest = line.slice(i);
    const c = line[i];
    if (comments.some((m) => rest.startsWith(m))) {
      push("com", rest);
      break;
    }
    if (fam !== "py" && fam !== "rb" && fam !== "clj" && rest.startsWith("/*")) {
      const end = line.indexOf("*/", i + 2);
      if (end < 0) {
        push("com", rest);
        return { tokens: out, state: { until: "*/", kind: "com" } };
      }
      push("com", line.slice(i, end + 2));
      i = end + 2;
      continue;
    }
    if (fam === "py" && (rest.startsWith('"""') || rest.startsWith("'''"))) {
      const q = rest.slice(0, 3);
      const end = line.indexOf(q, i + 3);
      if (end < 0) {
        push("str", rest);
        return { tokens: out, state: { until: q, kind: "str" } };
      }
      push("str", line.slice(i, end + 3));
      i = end + 3;
      continue;
    }
    if (c === '"' || c === "'" || (c === "`" && fam === "js")) {
      let j = i + 1;
      while (j < line.length && line[j] !== c) j += line[j] === "\\" ? 2 : 1;
      push("str", line.slice(i, Math.min(j + 1, line.length)));
      i = j + 1;
      continue;
    }
    const num = /^(?:0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)/.exec(rest);
    if (num && !/[\w$]/.test(line[i - 1] || "")) {
      push("num", num[0]);
      i += num[0].length;
      continue;
    }
    const word = /^[A-Za-z_$][\w$]*/.exec(rest);
    if (word) {
      const w = word[0];
      const after = line.slice(i + w.length);
      const kind: TokenKind = kw.has(w) ? "kw" : bi.has(w) ? "bi" : /^\s*\(/.test(after) ? "fn" : "txt";
      push(kind, w);
      i += w.length;
      continue;
    }
    push(/[=+\-*/%<>!&|^~?:]/.test(c) ? "op" : "txt", c);
    i++;
  }
  return { tokens: out, state };
}

/** Tokens per line for the whole text (lines split on \n; the editor renders one row per line). */
export function highlight(code: string, lang: ScriptLang): Token[][] {
  const fam = family(lang);
  const kw = new Set(KW[fam] || []);
  const bi = new Set(BUILTINS[fam === "groovy" || fam === "rb" || fam === "clj" ? "js" : fam] || []);
  let state: State = null;
  return code.split("\n").map((line) => {
    const r = tokenizeLine(line, fam, kw, bi, state);
    state = r.state;
    return r.tokens;
  });
}
