import React, { useEffect, useId, useRef, useState } from "react";
import { X, RotateCcw } from "lucide-react";
import { useStore } from "../store";
import { ARROW_STYLES, DENSITIES, FONT_FAMILIES, PASTE_MODES, PREF_DEFAULTS, SAVE_FORMATS, defaultAutoVirtualMb, toHex6, type Prefs } from "../prefs";
import { kbd } from "../platform";
import { useFocusTrap } from "../focusTrap";
import NumberInput from "./NumberInput";

/** Edit ▸ Options ▸ Preferences… (⌘,): every change is PUT to /api/settings at once. */
export default function PreferencesSheet() {
  const prefs = useStore((s) => s.prefs);
  const loaded = useStore((s) => s.prefsLoaded);
  const setPrefs = useStore((s) => s.setPrefs);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  // Focus management (M24–M26): the first control (Theme) takes focus once when the sheet opens, Tab cycles
  // inside, Esc closes and the opener (menu trigger, whatever had the focus before ⌘,) gets the focus back.
  // The hook runs once, not on every render: every committed change re-renders the sheet (it subscribes to
  // prefs), and re-running focus() would pull the focus off the control being edited (a NumField's debounced
  // commit, a select, a colour …).
  const dialogRef = useRef<HTMLDivElement>(null);
  const close = () => useStore.getState().setPrefsOpen(false);
  const trap = useFocusTrap(dialogRef, { onEscape: close });
  const set = (patch: Partial<Prefs>) => void setPrefs(patch);
  const reset = () => {
    // auto_virtual_mb's server default is half of the physical RAM (PREF_DEFAULTS holds a 0 placeholder, which
    // would mean "never"): recompute it from the RAM the server reports; keep the current value if unknown
    const autoVirtual = defaultAutoVirtualMb(useStore.getState().state?.memory_limit_bytes) ?? prefs.auto_virtual_mb;
    set({ ...PREF_DEFAULTS, auto_virtual_mb: autoVirtual });
    useStore.getState().toast("Preferences reset to defaults", "info");
  };
  return (
    <div className="modal-backdrop" onMouseDown={close}>
      <div className="dialog prefs" role="dialog" aria-modal="true" aria-labelledby="prefs-title" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <h3 id="prefs-title">Preferences</h3>
          <span className="dim small">{loaded ? "saved on the server as you change them" : "loading…"}</span>
          <span style={{ flex: 1 }} />
          <span className="kbd">{kbd("⌘,")}</span>
          <button className="icon-btn" onClick={close} title="Close (Esc)" aria-label="Close (Esc)"><X size={15} /></button>
        </div>
        <div className="dialog-body prefs-body">
          <Group title="Appearance">
            <Row label="Theme">
              <select value={theme} onChange={(e) => setTheme(e.target.value as "dark" | "light")}>
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </Row>
            <Row label="Density" hint="spacing of the inspector sections, tables and sheets">
              <select value={prefs.ui_density === "compact" ? "compact" : "comfortable"} onChange={(e) => set({ ui_density: e.target.value })}>
                {DENSITIES.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
              </select>
            </Row>
            <Row label="Selection colour" hint="ROI outlines and handles">
              <input type="color" value={toHex6(prefs.roi_color)} onChange={(e) => set({ roi_color: e.target.value })} />
            </Row>
            <Row label="Selection stroke" hint="screen px">
              <NumField value={prefs.roi_stroke_width} min={0.5} max={20} step={0.5} onCommit={(v) => set({ roi_stroke_width: v })} />
            </Row>
            <Row label="Interpolate zoomed images" hint="smooth instead of pixelated when zoomed in">
              <input type="checkbox" checked={prefs.interpolate_zoom} onChange={(e) => set({ interpolate_zoom: e.target.checked })} />
            </Row>
            <Row label="Antialiased text" hint="overlay labels, scale bars, time stamps">
              <input type="checkbox" checked={prefs.antialiased_text} onChange={(e) => set({ antialiased_text: e.target.checked })} />
            </Row>
          </Group>
          <Group title="Fonts">
            <Row label="Font size" hint="image px · default for the Text tool and labels">
              <NumField value={prefs.font_size} min={1} max={500} step={1} onCommit={(v) => set({ font_size: v })} />
            </Row>
            <Row label="Font family">
              <select value={FONT_FAMILIES.some((f) => f.id === prefs.font_family) ? prefs.font_family : "custom"} onChange={(e) => { if (e.target.value !== "custom") set({ font_family: e.target.value }); }}>
                {FONT_FAMILIES.map((f) => <option key={f.id || "default"} value={f.id}>{f.label}</option>)}
                {!FONT_FAMILIES.some((f) => f.id === prefs.font_family) && <option value="custom">{prefs.font_family}</option>}
              </select>
            </Row>
          </Group>
          <Group title="Points">
            <Row label="Point size" hint="screen px (Point tool)">
              <NumField value={prefs.point_size} min={1} max={100} step={1} onCommit={(v) => set({ point_size: v })} />
            </Row>
            <Row label="Point colour">
              <input type="color" value={toHex6(prefs.point_color)} onChange={(e) => set({ point_color: e.target.value })} />
            </Row>
          </Group>
          <Group title="Arrows">
            <Row label="Style">
              <select value={ARROW_STYLES.includes(prefs.arrow_style) ? prefs.arrow_style : ARROW_STYLES[0]} onChange={(e) => set({ arrow_style: e.target.value })}>
                {ARROW_STYLES.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </Row>
            <Row label="Head size" hint="image px">
              <NumField value={prefs.arrow_head_size} min={1} max={500} step={1} onCommit={(v) => set({ arrow_head_size: v })} />
            </Row>
            <Row label="Line width" hint="image px">
              <NumField value={prefs.arrow_width} min={0.5} max={200} step={0.5} onCommit={(v) => set({ arrow_width: v })} />
            </Row>
          </Group>
          <Group title="Colours">
            <Row label="Foreground value" hint="Fill, Draw, Brush, Bucket">
              <NumField value={prefs.foreground} step={1} onCommit={(v) => set({ foreground: v })} />
            </Row>
            <Row label="Background value" hint="Clear, Clear Outside, erase">
              <NumField value={prefs.background} step={1} onCommit={(v) => set({ background: v })} />
            </Row>
          </Group>
          <Group title="Plots">
            <Row label="Y range" hint="min / max · equal = automatic">
              <NumField value={prefs.plot_y_min} step={1} aria-label="Y range minimum" onCommit={(v) => set({ plot_y_min: v })} />
              <NumField value={prefs.plot_y_max} step={1} aria-label="Y range maximum" onCommit={(v) => set({ plot_y_max: v })} />
            </Row>
            <Row label="Interpolate line profiles" hint="sub-pixel sampling along lines">
              <input type="checkbox" checked={prefs.plot_interpolate} onChange={(e) => set({ plot_interpolate: e.target.checked })} />
            </Row>
          </Group>
          <Group title="Saving & editing">
            <Row label="Default save format">
              <select value={SAVE_FORMATS.some((f) => f.id === prefs.default_save_format) ? prefs.default_save_format : "tif"} onChange={(e) => set({ default_save_format: e.target.value })}>
                {SAVE_FORMATS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
              </select>
            </Row>
            <Row label="JPEG quality" hint="1–100">
              <NumField value={prefs.jpeg_quality} min={1} max={100} step={1} onCommit={(v) => set({ jpeg_quality: v })} />
            </Row>
            <Row label="Ask before closing a modified image" hint="File ▸ Close, ⌘W, Close All and the tab ×; the dialog's “Don't ask again” switches this off">
              <input type="checkbox" checked={prefs.confirm_close_modified} onChange={(e) => set({ confirm_close_modified: e.target.checked })} />
            </Row>
            <Row label="Paste mode" hint="how Edit ▸ Paste blends the clipboard">
              <select value={PASTE_MODES.includes(prefs.paste_mode) ? prefs.paste_mode : "Copy"} onChange={(e) => set({ paste_mode: e.target.value })}>
                {PASTE_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Row>
          </Group>
          <Group title="DICOM">
            <Row label="Open as 32-bit" hint="apply rescale slope/intercept">
              <input type="checkbox" checked={prefs.dicom_open_as_32bit} onChange={(e) => set({ dicom_open_as_32bit: e.target.checked })} />
            </Row>
            <Row label="Ignore rescale slope/intercept">
              <input type="checkbox" checked={prefs.dicom_ignore_rescale} onChange={(e) => set({ dicom_ignore_rescale: e.target.checked })} />
            </Row>
            <Row label="Calibrate images" hint="pixel spacing, slice thickness">
              <input type="checkbox" checked={prefs.dicom_calibrate_images} onChange={(e) => set({ dicom_calibrate_images: e.target.checked })} />
            </Row>
            <Row label="Apply window/level on open">
              <input type="checkbox" checked={prefs.dicom_window_level_on_open} onChange={(e) => set({ dicom_window_level_on_open: e.target.checked })} />
            </Row>
          </Group>
          <Group title="Advanced">
            <Row label="Rounded rectangle corner radius" hint="px">
              <NumField value={prefs.rounded_rect_radius} min={0} max={10000} step={1} onCommit={(v) => set({ rounded_rect_radius: v })} />
            </Row>
            <Row label="Undo memory budget" hint="MB">
              <NumField value={prefs.undo_budget_mb} min={50} max={1000000} step={50} onCommit={(v) => set({ undo_budget_mb: v })} />
            </Row>
            <Row label="Max worker threads" hint="0 = automatic">
              <NumField value={prefs.max_threads} min={0} max={4096} step={1} onCommit={(v) => set({ max_threads: v })} />
            </Row>
            <Row label="HTTP proxy" hint="host:port">
              <TextField value={prefs.proxy} placeholder="none" onCommit={(v) => set({ proxy: v })} />
            </Row>
            <Row label="No-proxy hosts" hint="comma-separated">
              <TextField value={prefs.no_proxy} placeholder="none" onCommit={(v) => set({ no_proxy: v })} />
            </Row>
            <Row label="Virtual-stack plane cache (MB)" hint="planes of virtual stacks kept in memory (16–1000000); the Memory Monitor shows its fill">
              <NumField value={prefs.virtual_cache_mb} min={16} max={1000000} step={16} onCommit={(v) => set({ virtual_cache_mb: v })} />
            </Row>
            <Row label="Open files larger than this as virtual stacks (MB; 0 = never)" hint="decoded size above which File ▸ Open reads planes on demand (TIFF, HDF5 / Imaris, Zarr, npy); server default: half of the physical RAM">
              <NumField value={prefs.auto_virtual_mb} min={0} max={100000000} step={256} onCommit={(v) => set({ auto_virtual_mb: v })} />
            </Row>
          </Group>
        </div>
        <div className="dialog-foot">
          <button className="btn sm" onClick={reset}><RotateCcw size={12} /> Reset to defaults</button>
          <span className="dim small">Per-tool overrides stay in the tool-options bar above the image.</span>
          <span className="grow" />
          <button className="btn primary" onClick={close}>Done</button>
        </div>
      </div>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="prefs-group">
      <h4>{title}</h4>
      {children}
    </div>
  );
}

/** Accessibility attributes a Row injects into its controls (NumField / TextField spread them onto their input). */
type A11y = { id?: string; "aria-label"?: string; "aria-labelledby"?: string; "aria-describedby"?: string };

/**
 * One preference row (M26, label association).  The row label names its first control (`aria-labelledby` on
 * the label text, `htmlFor` for click-to-focus) and the hint describes every control (`aria-describedby`).  A
 * control that brings its own `aria-label` (the Y-range pair) keeps that name and only gets the description.
 */
function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  const base = useId();
  const labelId = `${base}-label`;
  const hintId = hint ? `${base}-hint` : undefined;
  const kids = React.Children.toArray(children).filter((k): k is React.ReactElement<A11y> => React.isValidElement(k));
  const controlId = kids[0]?.props.id ?? `${base}-ctrl`;
  return (
    <div className="prefs-row">
      <label htmlFor={controlId}>
        <span id={labelId}>{label}</span>
        {hint && <span id={hintId} className="hint">{hint}</span>}
      </label>
      <div className="ctrl">
        {kids.map((k, i) =>
          React.cloneElement(k, {
            id: i === 0 ? controlId : k.props.id,
            "aria-labelledby": k.props["aria-labelledby"] ?? (k.props["aria-label"] ? undefined : labelId),
            "aria-describedby": k.props["aria-describedby"] ?? hintId,
          }),
        )}
      </div>
    </div>
  );
}

/** Text input that keeps what the user types and commits (trimmed) on blur/Enter, so a PUT is not sent per keystroke. */
function TextField({ value, placeholder, onCommit, ...a11y }: { value: string; placeholder?: string; onCommit: (v: string) => void } & A11y) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = (s: string) => {
    const v = s.trim();
    if (v !== value) onCommit(v);
    else setText(v);
  };
  return (
    <input
      type="text"
      {...a11y}
      value={text}
      placeholder={placeholder}
      spellCheck={false}
      onChange={(e) => setText(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit((e.target as HTMLInputElement).value);
      }}
    />
  );
}

/**
 * Number input that keeps what the user types: values that are already valid as typed save (debounced)
 * while typing; blur / Enter clamps to [min, max], saves and shows the stored value.  The debounce never
 * clamps: an out-of-range prefix ("2" on the way to "200" in a min-50 field) used to be clamped and
 * written back under the caret, so the remaining keystrokes were appended to the clamped value.
 * A NumberInput (text field, E7-26): the value shows with a decimal point whatever the browser locale and
 * "," is accepted while typing; the typed text ("1.50") survives its own round-trip because the field only
 * rewrites it when the committed number differs from what the text parses to.
 */
function NumField({ value, min, max, step, onCommit, ...a11y }: { value: number; min?: number; max?: number; step?: number; onCommit: (v: number) => void } & A11y) {
  const [shown, setShown] = useState(value); // the (unclamped) number the field shows while typing
  const timer = useRef<number | null>(null);
  const latest = useRef(value); // current prop: the timer must not compare against a stale closure
  latest.current = value;
  useEffect(() => {
    setShown(value); // a committed value coming back, Reset to defaults, reload, reverted PUT
  }, [value]);
  const clearTimer = () => {
    if (timer.current) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => clearTimer, []);
  const clamp = (n: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, n));
  const commit = (n: number) => {
    // blur / Enter: clamp, save and show the stored value
    clearTimer();
    const v = clamp(n);
    if (v !== latest.current) onCommit(v);
    setShown(v);
  };
  return (
    <NumberInput
      {...a11y}
      min={min}
      max={max}
      step={step}
      value={shown}
      onChange={(n) => {
        setShown(n);
        clearTimer();
        timer.current = window.setTimeout(() => {
          timer.current = null;
          // live-save only a value that needs no clamping; anything else waits for more digits or blur / Enter
          if (clamp(n) !== n || n === latest.current) return;
          onCommit(n);
        }, 350);
      }}
      onBlur={() => commit(shown)}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit(shown);
      }}
    />
  );
}
