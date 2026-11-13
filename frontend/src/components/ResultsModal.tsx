import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { X, Copy, Download, Columns3, Minimize2 } from "lucide-react";
import { useStore } from "../store";
import { useFocusTrap } from "../focusTrap";

/** Format a table cell the way the inspector does (numbers to `decimals`, null → NaN). */
export function fmtCell(v: unknown, decimals: number): string {
  if (v === null || v === undefined) return "NaN";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(decimals);
  return String(v);
}

/**
 * Select the ROI a Results row refers to: a ROI-manager entry with the row's label, or the
 * Analyze-Particles convention (row "<image>:<n>", ROI "<n:04d>-<y>-<x>").  Jumps to its slice.
 */
export function selectResultsRow(i: number): void {
  const st = useStore.getState();
  const results = st.state?.results;
  if (!results || !results.rows[i]) return;
  const label = String(results.rows[i][0] ?? "");
  const rois = st.state?.rois || [];
  let roi = rois.find((r) => r.name === label || `${st.activeImage()?.name}:${r.name}` === label);
  if (!roi) {
    const m = /:(\d+)$/.exec(label);
    if (m) {
      const prefix = m[1].padStart(4, "0") + "-";
      roi = rois.find((r) => (r.name || "").startsWith(prefix));
    }
  }
  if (!roi) return;
  st.setSelectedRoiIds([roi.id!]);
  const img = st.activeImage();
  if (img) {
    st.setSelection(img.id, roi);
    if (roi.position) st.setDisplay(img.id, { t: Math.min(roi.position.t, img.shape.t - 1), z: Math.min(roi.position.z, img.shape.z - 1) });
  }
}

function csvField(s: string): string {
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function downloadText(name: string, text: string, type: string) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const OVERSCAN = 12;
type Sort = { col: number; dir: 1 | -1 } | null; // col = -1 sorts by the row index

function compareCells(a: unknown, b: unknown): number {
  const an = typeof a === "number" ? a : a === null || a === undefined ? NaN : Number(a);
  const bn = typeof b === "number" ? b : b === null || b === undefined ? NaN : Number(b);
  const aNum = Number.isFinite(an) && (typeof a === "number" || (typeof a === "string" && a.trim() !== ""));
  const bNum = Number.isFinite(bn) && (typeof b === "number" || (typeof b === "string" && b.trim() !== ""));
  if (aNum && bNum) return an - bn;
  if (aNum !== bNum) return aNum ? -1 : 1; // numbers before text; NaN / empty last
  const as = a === null || a === undefined ? "" : String(a);
  const bs = b === null || b === undefined ? "" : String(b);
  if (!as && bs) return 1;
  if (as && !bs) return -1;
  return as.localeCompare(bs, undefined, { numeric: true });
}

/** Large, sortable, column-toggling view of the Results table or the last generic table. */
export default function ResultsModal() {
  const kind = useStore((s) => s.tableModal);
  const setTableModal = useStore((s) => s.setTableModal);
  const results = useStore((s) => s.state?.results);
  const table = useStore((s) => s.table);
  const decimalsPref = useStore((s) => s.state?.measure_options.decimals ?? 3);
  const toast = useStore((s) => s.toast);
  const close = useCallback(() => setTableModal(null), [setTableModal]);

  const isResults = kind === "results";
  const columns = useMemo(() => (isResults ? results?.columns : table?.columns) || [], [isResults, results, table]);
  const rows = useMemo(() => (isResults ? results?.rows : table?.rows) || [], [isResults, results, table]);
  const decimals = isResults ? decimalsPref : 3;
  const title = isResults ? "Results" : table?.name || "Table";

  const [sort, setSort] = useState<Sort>(null);
  const [hidden, setHidden] = useState<string[]>([]);
  const [colMenu, setColMenu] = useState(false);
  const [selRow, setSelRow] = useState<number | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(600);
  const [rowH, setRowH] = useState(25);
  const scrollRef = useRef<HTMLDivElement>(null);
  const colMenuRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  // Focus management (M24): the Columns button takes focus on open, Tab cycles inside (toolbar, column menu
  // while it is open, Esc is handled by the capture listener below so the column menu closes first) and the
  // opener (the inspector's expand button) gets the focus back.  isolate: a single-letter shortcut typed on a
  // focused button must not reach the global handler (this modal is not in its modalOpen list).
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, { isolate: true });

  // Escape closes (capture phase: the global shortcut handler must not also act on it)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        if (colMenu) setColMenu(false);
        else close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [close, colMenu]);

  // close the column menu on an outside click
  useEffect(() => {
    if (!colMenu) return;
    const onDown = (e: PointerEvent) => {
      if (colMenuRef.current && !colMenuRef.current.contains(e.target as Node)) setColMenu(false);
    };
    document.addEventListener("pointerdown", onDown, true);
    return () => document.removeEventListener("pointerdown", onDown, true);
  }, [colMenu]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewH(el.clientHeight));
    ro.observe(el);
    setViewH(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // forget a sort / hidden columns / selection that no longer apply – keyed on the column *names*,
  // since every server refresh hands us a new results object with the same columns
  const colKey = columns.join("\u0001");
  useEffect(() => {
    const cols = colKey ? colKey.split("\u0001") : [];
    setSort((s) => (s && s.col >= cols.length ? null : s));
    setHidden((h) => h.filter((c) => cols.includes(c)));
    setSelRow(null);
  }, [colKey]);

  const visibleCols = useMemo(() => columns.map((c, k) => ({ c, k })).filter(({ c }) => !hidden.includes(c)), [columns, hidden]);
  const order = useMemo(() => {
    const idx = rows.map((_, i) => i);
    if (!sort) return idx;
    const { col, dir } = sort;
    const cmp = col < 0 ? (a: number, b: number) => a - b : (a: number, b: number) => compareCells(rows[a][col], rows[b][col]) || a - b;
    return idx.sort((a, b) => dir * cmp(a, b));
  }, [rows, sort]);

  const onScroll = () => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      if (scrollRef.current) setScrollTop(scrollRef.current.scrollTop);
    });
  };
  // measure the real row height once rows are in the DOM (border-collapse makes it hard to predict)
  useLayoutEffect(() => {
    const tr = scrollRef.current?.querySelector<HTMLTableRowElement>("tbody tr[data-row]");
    if (tr && tr.offsetHeight > 0 && Math.abs(tr.offsetHeight - rowH) > 0.5) setRowH(tr.offsetHeight);
  });

  const n = order.length;
  const start = Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN);
  const end = Math.min(n, Math.ceil((scrollTop + viewH) / rowH) + OVERSCAN);
  const slice = order.slice(start, end);

  const toggleSort = (col: number) => setSort((s) => (!s || s.col !== col ? { col, dir: 1 } : s.dir === 1 ? { col, dir: -1 } : null));
  const sortMark = (col: number) => (sort && sort.col === col ? (sort.dir === 1 ? "▲" : "▼") : "");
  const lines = (sep: string, quote: (s: string) => string) => {
    const head = ["#", ...visibleCols.map(({ c }) => quote(c))].join(sep);
    const body = order.map((i) => [i + 1, ...visibleCols.map(({ k }) => quote(fmtCell(rows[i][k], decimals)))].join(sep));
    return [head, ...body].join("\n");
  };
  const copy = () => {
    void navigator.clipboard.writeText(lines("\t", (s) => s));
    toast(`${n} rows copied (tab-separated)`, "success");
  };
  const csv = () => downloadText((isResults ? "Results" : (table?.name || "table").replace(/[^\w.-]+/g, "_")) + ".csv", lines(",", csvField), "text/csv");
  const pick = (i: number) => {
    setSelRow(i);
    if (isResults) selectResultsRow(i);
  };

  if (!kind) return null;
  return (
    <div className="modal-backdrop vol-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="dialog results-modal" role="dialog" aria-modal="true" aria-labelledby="results-modal-title" tabIndex={-1} ref={dialogRef} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <h3 id="results-modal-title">{title}</h3>
          <span className="dim small" style={{ marginLeft: 4 }}>{n} row{n === 1 ? "" : "s"} · {visibleCols.length}/{columns.length} columns</span>
          <span style={{ flex: 1 }} />
          <button className="icon-btn" onClick={close} title="Back to the inspector (Esc)" aria-label="Back to the inspector (Esc)"><Minimize2 size={15} /></button>
          <button className="icon-btn" onClick={close} title="Close (Esc)" aria-label="Close (Esc)"><X size={16} /></button>
        </div>
        <div className="results-toolbar">
          <div className="col-menu-wrap" ref={colMenuRef}>
            <button className={"btn sm" + (hidden.length ? " primary" : "")} onClick={() => setColMenu((v) => !v)} title="Show / hide columns"><Columns3 size={12} /> Columns{hidden.length ? ` (${hidden.length} hidden)` : ""}</button>
            {colMenu && (
              <div className="col-menu">
                <div className="col-menu-head">
                  <button className="btn sm" onClick={() => setHidden([])}>All</button>
                  <button className="btn sm" onClick={() => setHidden(columns.slice(1))} title="Keep only the first column">First only</button>
                </div>
                {columns.map((c) => (
                  <label key={c}><input type="checkbox" checked={!hidden.includes(c)} onChange={(e) => setHidden((h) => (e.target.checked ? h.filter((x) => x !== c) : [...h, c]))} /> {c}</label>
                ))}
              </div>
            )}
          </div>
          <button className="btn sm" onClick={() => setSort(null)} disabled={!sort} title="Restore the original row order">Unsort</button>
          <span className="grow" />
          <button className="btn sm" onClick={copy} disabled={!n} title="Copy the visible columns (tab-separated)"><Copy size={12} /> Copy</button>
          <button className="btn sm" onClick={csv} disabled={!n} title="Download the visible columns as CSV"><Download size={12} /> CSV</button>
        </div>
        <div className="results-scroll" ref={scrollRef} onScroll={onScroll}>
          {n === 0 ? (
            <div className="hint-text" style={{ padding: 16 }}>The table is empty.</div>
          ) : (
            <table className="results big">
              <thead>
                <tr>
                  <th className={sort?.col === -1 ? "sorted" : ""} onClick={() => toggleSort(-1)} title="Sort by row number">#<span className="sort">{sortMark(-1)}</span></th>
                  {visibleCols.map(({ c, k }) => (
                    <th key={c} className={sort?.col === k ? "sorted" : ""} onClick={() => toggleSort(k)} title={`Sort by ${c}`}>{c}<span className="sort">{sortMark(k)}</span></th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {start > 0 && <tr className="spacer"><td colSpan={visibleCols.length + 1} style={{ height: start * rowH }} /></tr>}
                {slice.map((i) => (
                  <tr key={i} data-row={i} className={selRow === i ? "sel" : ""} onClick={() => pick(i)}>
                    <td>{i + 1}</td>
                    {visibleCols.map(({ k }) => <td key={k}>{fmtCell(rows[i][k], decimals)}</td>)}
                  </tr>
                ))}
                {end < n && <tr className="spacer"><td colSpan={visibleCols.length + 1} style={{ height: (n - end) * rowH }} /></tr>}
              </tbody>
            </table>
          )}
        </div>
        <div className="results-foot">
          <span>{n} rows</span>
          <span>{visibleCols.length} of {columns.length} columns</span>
          {sort && <span>sorted by {sort.col < 0 ? "#" : columns[sort.col]} {sort.dir === 1 ? "ascending" : "descending"}</span>}
          <span style={{ flex: 1 }} />
          <span>{isResults ? "click a row to select its ROI · " : ""}click a header to sort · first column and header stay pinned</span>
        </div>
      </div>
    </div>
  );
}
