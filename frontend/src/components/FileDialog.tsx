import { useEffect, useRef, useState } from "react";
import { Folder, FileImage, File as FileIcon, ArrowUp, Home, Upload, X } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import { normalizeFormat } from "../prefs";
import { fmtBytes } from "./MemoryMonitor";
import { useFocusTrap } from "../focusTrap";

interface Entry { name: string; path: string; dir: boolean; size: number; image: boolean }

export default function FileDialog() {
  const dlg = useStore((s) => s.fileDialog)!;
  const close = () => useStore.getState().setFileDialog(null);
  const meta = useStore((s) => s.meta)!;
  const img = useStore((s) => (dlg.imageId ? s.image(dlg.imageId) : s.activeImage()));
  const [path, setPath] = useState<string>(localStorage.getItem("lumen.dir") || meta.cwd || meta.home);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [parent, setParent] = useState<string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filename, setFilename] = useState(dlg.filename ?? (img ? img.name.replace(/\.[^.]+$/, "") : "image"));
  const [format, setFormat] = useState(normalizeFormat(dlg.format || useStore.getState().prefs.default_save_format));
  // "Open as virtual stack": keep the pixels in the file and read planes on demand (data larger than RAM; TIFF, HDF5 /
  // Imaris, Zarr / OME-Zarr, npy).  Off = automatic: the server opens lazily when the file exceeds auto_virtual_mb.
  const [virtual, setVirtual] = useState(false);
  // "pick-open" / "pick-save" / "pick-folder": hand the chosen server path to the caller (text files for the script
  // editor / text windows, path and folder parameters of the operation sheet)
  const pickOpen = dlg.mode === "pick-open";
  const pickSave = dlg.mode === "pick-save";
  const pickFolder = dlg.mode === "pick-folder";
  const matches = (name: string) => !dlg.extensions || dlg.extensions.some((ext) => name.toLowerCase().endsWith(ext.toLowerCase()));
  const fileRef = useRef<HTMLInputElement>(null);
  // Focus management (M24): the file-name field (save) or the folder-path field (open / pick) takes focus once
  // on open, Tab cycles inside (quick links, path, file rows, buttons), Esc closes and the opener (menu trigger,
  // a sheet's Browse… button) gets the focus back.  The hook runs once, so typing in the path / filename inputs
  // never re-runs focus().
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, {
    onEscape: close,
    initialFocus: () => dialogRef.current?.querySelector<HTMLElement>(dlg.mode === "save" || pickSave ? "#file-dialog-name" : 'input[aria-label="Folder path"]'),
  });
  const sep = path.includes("\\") ? "\\" : "/";

  const load = async (p: string) => {
    try {
      const r = await api.fs(p);
      setEntries(r.entries);
      setParent(r.parent);
      setPath(r.path);
      setErr(null);
      localStorage.setItem("lumen.dir", r.path);
    } catch (e: any) {
      setErr(e.message);
    }
  };
  useEffect(() => {
    // start in the caller's folder (Browse… next to a filled-in path) when it exists, else where the user was last
    void (async () => {
      const start = dlg.initialPath?.trim();
      if (start) {
        try {
          await api.fs(start);
          await load(start);
          return;
        } catch {
          /* not a readable folder: fall back */
        }
      }
      await load(path);
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const openSel = async (p?: string) => {
    const target = p || sel;
    if (!target) return;
    const st = useStore.getState();
    if (pickOpen) {
      st.setFileDialog(null);
      dlg.onPick?.(target);
      return;
    }
    try {
      st.setFileDialog(null);
      const r = await api.open(target, virtual ? { virtual: true } : {});
      await st.refresh();
      const name = target.split(sep).pop();
      if (r.image?.virtual) {
        // opened lazily (checkbox, or automatically because the file exceeds auto_virtual_mb): say so once, up front
        st.toast(`Opened ${name} as a virtual stack – ${fmtBytes(r.image.full_bytes ?? 0)} decoded, planes read on demand${r.image.virtual_source ? ` (${r.image.virtual_source})` : ""}; the first pixel-modifying command reads it into memory`, "info");
      } else {
        st.toast(`Opened ${name}`, "success");
      }
    } catch (e: any) {
      st.toast(e.message, "error");
    }
  };
  const save = async () => {
    const st = useStore.getState();
    const full = path + sep + filename.trim();
    if (pickSave) {
      st.setFileDialog(null);
      dlg.onPick?.(full);
      return;
    }
    if (!img) return;
    try {
      const r = await api.save(img.id, full, format);
      st.setFileDialog(null);
      st.markSaved(img.id); // no longer "modified": closing will not ask
      await st.refresh();
      st.toast(`Saved ${r.path}`, "success");
    } catch (e: any) {
      st.toast(e.message, "error");
    }
  };
  const quick = [
    { label: "Home", p: meta.home },
    { label: "Desktop", p: meta.home + sep + "Desktop" },
    { label: "Documents", p: meta.home + sep + "Documents" },
    { label: "Downloads", p: meta.home + sep + "Downloads" },
    { label: "Working dir", p: meta.cwd },
  ];
  return (
    <div className="modal-backdrop" onMouseDown={close}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="file-dialog-title" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <h3 id="file-dialog-title">{dlg.title ?? (dlg.mode === "open" ? "Open image" : pickOpen ? "Open file" : pickSave ? "Save file" : pickFolder ? "Choose folder" : `Save “${img?.name}”`)}</h3>
          <span style={{ flex: 1 }} />
          {dlg.mode === "open" && <button className="btn sm" onClick={() => fileRef.current?.click()}><Upload size={12} /> Upload from this computer</button>}
          <button className="icon-btn" title="Close" aria-label="Close" onClick={close}><X size={15} /></button>
        </div>
        <div className="dialog-body">
          <div className="quick-links" role="group" aria-label="Quick links">
            {quick.map((q) => <button key={q.label} className="btn sm" onClick={() => void load(q.p)}><Home size={11} /> {q.label}</button>)}
          </div>
          <div className="path-bar">
            <button className="btn sm" onClick={() => parent && void load(parent)} disabled={!parent} title="Up" aria-label="Up one folder"><ArrowUp size={12} /></button>
            <input value={path} aria-label="Folder path" onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") void load(path); }} spellCheck={false} />
          </div>
          {err && <div className="sheet-error" role="alert" style={{ margin: 0 }}>{err}</div>}
          <div className="file-list" role="listbox" aria-label="Files">
            {entries.map((en) => (
              <div key={en.path} className={"file-item" + (sel === en.path ? " sel" : "") + (!en.dir && (pickFolder || (pickOpen || pickSave ? !matches(en.name) : !en.image)) ? " dim" : "")} role="option" aria-selected={sel === en.path} tabIndex={0} onClick={() => (en.dir ? void load(en.path) : pickFolder ? undefined : pickSave ? setFilename(en.name) : setSel(en.path))} onDoubleClick={() => (en.dir || pickSave || pickFolder ? undefined : void openSel(en.path))} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); if (en.dir) void load(en.path); else if (pickFolder) return; else if (pickSave) setFilename(en.name); else if (sel === en.path) void openSel(en.path); else setSel(en.path); } }}>
                {en.dir ? <Folder size={14} style={{ color: "var(--warn)" }} /> : en.image ? <FileImage size={14} style={{ color: "var(--accent)" }} /> : <FileIcon size={14} style={(pickOpen || pickSave) && matches(en.name) ? { color: "var(--accent)" } : undefined} />}
                <span>{en.name}</span>
                {!en.dir && <span className="sz">{en.size > 1048576 ? (en.size / 1048576).toFixed(1) + " MB" : (en.size / 1024).toFixed(0) + " KB"}</span>}
              </div>
            ))}
            {entries.length === 0 && !err && <div className="hint-text" style={{ padding: 10 }}>Empty folder</div>}
          </div>
          {dlg.mode === "open" && (
            <div className="row">
              <label className="virtual-check" title="Keep the pixels in the file and read planes on demand – for TIFF, HDF5 / Imaris, Zarr / OME-Zarr and npy files larger than RAM. Off = automatic: files bigger than the auto_virtual_mb preference open as virtual stacks anyway.">
                <input type="checkbox" checked={virtual} onChange={(e) => setVirtual(e.target.checked)} />
                Open as virtual stack <span className="dim">(planes read on demand; TIFF, HDF5, Zarr, npy)</span>
              </label>
            </div>
          )}
          {(dlg.mode === "save" || pickSave) && (
            <div className="row">
              <label htmlFor="file-dialog-name">File name</label>
              <input id="file-dialog-name" type="text" value={filename} onChange={(e) => setFilename(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && filename.trim()) void save(); }} style={{ flex: 1 }} />
              {dlg.mode === "save" && <select value={format} aria-label="File format" onChange={(e) => setFormat(e.target.value)}>
                <option value="tif">.tif (ImageJ hyperstack)</option>
                <option value="ome.tif">.ome.tif (OME-TIFF)</option>
                <option value="png">.png (current view, RGB)</option>
                <option value="jpg">.jpg (current view)</option>
                <option value="gif">.gif (animated, all slices)</option>
                <option value="npy">.npy (numpy)</option>
              </select>}
            </div>
          )}
        </div>
        <div className="dialog-foot">
          <span className="dim small">{dlg.mode === "open" ? "Double-click an image or select it and press Open. Folders of images open as a stack (select the folder path and press Open)." : pickOpen ? `Select a file and press Open${dlg.extensions ? ` (${dlg.extensions.join(", ")})` : ""}.` : pickSave ? "Choose a folder, type a name and press Save." : pickFolder ? "Open the folder you want (or type its path), then press Choose this folder." : "Images are saved with calibration, LUTs and display ranges."}</span>
          <span className="grow" />
          <button className="btn" onClick={close}>Cancel</button>
          {dlg.mode === "open" ? (
            <>
              <button className="btn" onClick={() => void openSel(path)} title="Open every image in this folder as one stack">Open folder as stack</button>
              <button className="btn primary" disabled={!sel} onClick={() => void openSel()}>Open</button>
            </>
          ) : pickOpen ? (
            <button className="btn primary" disabled={!sel} onClick={() => void openSel()}>Open</button>
          ) : pickFolder ? (
            <button className="btn primary" disabled={!path.trim() || !!err} onClick={() => { useStore.getState().setFileDialog(null); dlg.onPick?.(path); }}>Choose this folder</button>
          ) : (
            <button className="btn primary" disabled={!filename.trim()} onClick={() => void save()}>Save</button>
          )}
        </div>
        <input ref={fileRef} type="file" multiple hidden accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.gif,.npy,.nd2,.czi,.lif" onChange={(e) => { if (e.target.files?.length) { close(); void useStore.getState().openFiles(e.target.files); } }} />
      </div>
    </div>
  );
}
