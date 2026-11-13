/**
 * The desktop app (desktop/lumen_desktop): native file dialogs and a few hooks the shell calls.
 *
 * Inside the desktop window pywebview exposes the shell's `DesktopApi` as `window.pywebview.api`.  When it is there,
 * every file dialog of the app (File ▸ Open / Save As, every "Browse…" of a path parameter, script Open / Save) goes
 * to the system dialog instead of the in-app server file browser – the server runs on this computer, so a native
 * path is a server path.  In a plain browser `nativeApi()` is null and nothing here changes behaviour.
 */
import { api } from "./api";

export interface NativeApi {
  native_open_images(initial?: string | null, multiple?: boolean): Promise<string[] | null>;
  native_pick_open(title?: string | null, extensions?: string[] | null, initial?: string | null): Promise<string | null>;
  native_pick_save(title?: string | null, filename?: string | null, initial?: string | null): Promise<string | null>;
  native_pick_folder(title?: string | null, initial?: string | null): Promise<string | null>;
  set_unsaved(count: number): Promise<boolean>;
}

export function nativeApi(): NativeApi | null {
  const pw = (window as unknown as { pywebview?: { api?: Partial<NativeApi> } }).pywebview;
  const a = pw?.api;
  return a && typeof a.native_open_images === "function" && typeof a.native_pick_save === "function" ? (a as NativeApi) : null;
}

/** The request shape of `Store.setFileDialog` (kept structural so this module does not import the store). */
export interface FileDialogRequest {
  mode: "open" | "save" | "pick-open" | "pick-save" | "pick-folder";
  imageId?: string;
  format?: string;
  title?: string;
  filename?: string;
  extensions?: string[];
  initialPath?: string;
  onPick?: (path: string) => void;
  inApp?: boolean;
}

interface StoreLike {
  image: (id: string | null | undefined) => { id: string; name: string; path?: string | null } | undefined;
  prefs: { default_save_format?: string };
  refresh: () => Promise<void>;
  markSaved: (id: string) => void;
  toast: (msg: string, kind?: "info" | "success" | "error") => void;
  setFileDialog: (d: FileDialogRequest | null) => void;
}

const SAVE_EXT: [string, string][] = [
  [".ome.tiff", "ome.tif"], [".ome.tif", "ome.tif"], [".tiff", "tif"], [".tif", "tif"], [".png", "png"], [".jpeg", "jpg"], [".jpg", "jpg"], [".gif", "gif"], [".npy", "npy"],
];

/** The save format a chosen file name implies: a SAVE_FORMATS id, or undefined to let the server decide from the
 * extension (.h5, .nii, .zarr … are written by extension too). */
function formatFromName(name: string): string | undefined {
  const low = name.toLowerCase();
  return SAVE_EXT.find(([ext]) => low.endsWith(ext))?.[1];
}

function defaultSaveName(name: string, format: string): string {
  const ext = format === "ome.tif" ? ".ome.tif" : "." + format;
  const stem = name.replace(/\.(ome\.)?(tiff?|png|jpe?g|gif|npy|czi|nd2|lif|lsm|ims|h5|hdf5|zvi|dm3|dm4|nii(\.gz)?|mrc|fits?)$/i, "");
  return stem + ext;
}

const dirOf = (p?: string | null) => (p ? p.replace(/[\\/][^\\/]*$/, "") : null);

/**
 * Run a file-dialog request through the system dialog.  Returns false when there is no native bridge (the caller
 * shows the in-app dialog); a failure inside the bridge falls back to the in-app dialog by itself.
 */
export async function runNativeFileDialog(d: FileDialogRequest, getStore: () => StoreLike): Promise<boolean> {
  const n = nativeApi();
  if (!n || d.inApp) return false;
  const st = getStore();
  try {
    if (d.mode === "open") {
      const files = await n.native_open_images(d.initialPath ?? null, true);
      if (!files || !files.length) return true;
      let ok = 0;
      for (const f of files) {
        try {
          const r = await api.open(f);
          ok++;
          if (r.image?.virtual) st.toast(`Opened ${f.split(/[\\/]/).pop()} as a virtual stack – planes are read on demand`, "info");
        } catch (e: any) {
          st.toast(`${f.split(/[\\/]/).pop()}: ${e?.message || e}`, "error");
        }
      }
      await getStore().refresh();
      if (ok) st.toast(ok === 1 ? `Opened ${files[0].split(/[\\/]/).pop()}` : `Opened ${ok} images`, "success");
      return true;
    }
    if (d.mode === "save") {
      const img = st.image(d.imageId);
      if (!img) return true;
      const fmt = d.format || st.prefs.default_save_format || "tif";
      const target = await n.native_pick_save(d.title ?? `Save “${img.name}”`, defaultSaveName(img.name, fmt), dirOf(img.path));
      if (!target) return true;
      try {
        const r = await api.save(img.id, target, formatFromName(target));
        getStore().markSaved(img.id);
        await getStore().refresh();
        st.toast(`Saved ${r.path}`, "success");
      } catch (e: any) {
        st.toast(e?.message || String(e), "error");
      }
      return true;
    }
    if (d.mode === "pick-open") {
      const p = await n.native_pick_open(d.title ?? null, d.extensions ?? null, d.initialPath ?? null);
      if (p) d.onPick?.(p);
      return true;
    }
    if (d.mode === "pick-save") {
      const p = await n.native_pick_save(d.title ?? null, d.filename ?? null, d.initialPath ?? null);
      if (p) d.onPick?.(p);
      return true;
    }
    if (d.mode === "pick-folder") {
      const p = await n.native_pick_folder(d.title ?? null, d.initialPath ?? null);
      if (p) d.onPick?.(p);
      return true;
    }
  } catch (e) {
    console.warn("native file dialog failed – using the in-app dialog", e);
    getStore().setFileDialog({ ...d, inApp: true });
    return true;
  }
  return false;
}

/**
 * Hooks for the desktop shell: `window.__lumenApp.openDocs()` / `.openShortcuts()` (the native Help menu) and the
 * unsaved-image count the shell's quit confirmation follows.  Safe to call in a browser (does nothing there).
 */
export function startNativeHooks(opts: { openDocs: () => void; openShortcuts: () => void; unsavedCount: () => number; subscribe: (fn: () => void) => () => void }): void {
  (window as unknown as { __lumenApp?: unknown }).__lumenApp = { openDocs: opts.openDocs, openShortcuts: opts.openShortcuts };
  let last = -1;
  const push = () => {
    const n = nativeApi();
    if (!n) return;
    const c = opts.unsavedCount();
    if (c === last) return;
    last = c;
    n.set_unsaved(c).catch(() => { last = -1; });
  };
  opts.subscribe(push);
  window.addEventListener("pywebviewready", () => { last = -1; push(); });
  push();
}
