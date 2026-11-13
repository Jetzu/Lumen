import { useEffect, useMemo, useRef, useState } from "react";
import { Search, CornerDownLeft } from "lucide-react";
import { needsSelectionShort, useStore } from "../store";
import type { OpSpec } from "../types";
import { CLIENT_ACTIONS, doAction } from "../actions";
import { HIDDEN_MENU_OPS, commandCount, displayName, prettyShortcut } from "../menus";
import { kbd } from "../platform";
import { getAliases, type AliasMap } from "../paletteAliases";
import { lutGroupOf } from "../lutGroups";

interface Entry {
  id: string;
  name: string;
  category: string;
  description: string;
  kbd?: string;
  op?: OpSpec;
  /** Run `op` with these params straight away (a LUT picked from the palette: "LUT: Fire"). */
  params?: Record<string, any>;
  action?: string;
  arg?: any;
  keywords: string[];
  /** Fiji names of the command (docs/FIJI_PARITY.md rows, the macro-layer run("…") names): searchable, shown when matched. */
  aliases?: string[];
}

const ACTIONS: Entry[] = CLIENT_ACTIONS.map((a) => ({ id: a.id, name: a.name, category: a.category, description: a.description, kbd: a.kbd, action: a.action, arg: a.arg, keywords: a.keywords }));

/** Plugins ▸ Utilities ▸ Search…: these entries always close the result list and search the web for what was typed. */
const WEB_SEARCH_IDS = new Set(["a:searchWiki", "a:searchForum"]);

/** A name without its trailing ellipsis, lower-cased, for comparisons. */
function norm(s: string): string {
  return s.toLowerCase().replace(/(…|\.\.\.)\s*$/, "").trim();
}

/** Damerau–Levenshtein distance capped at `max + 1` (typo tolerance: "clahee" → "clahe", "gausian" → "gaussian"). */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const prev = new Array(b.length + 1).fill(0).map((_, j) => j);
  let prev2: number[] = [];
  let cur: number[] = [];
  for (let i = 1; i <= a.length; i++) {
    cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev2 = prev.slice();
    prev.splice(0, prev.length, ...cur);
  }
  return prev[b.length];
}

interface Scored {
  e: Entry;
  s: number;
  /** The Fiji alias that matched (shown next to the name). */
  via?: string;
  /** Only fuzzy / typo evidence: dropped when the query has real matches (E7-10: "clahe" → Colocalization…). */
  weak: boolean;
}

/**
 * Ranking (E5-3 / E5-4 / E6-1 / E7-10): the whole query against the name first (exact > prefix > substring),
 * then every word: name exact > alias exact > name prefix > word start > substring > keyword / alias / category /
 * id / description > a close typo (edit distance ≤ 1, ≤ 2 for long words) > subsequence.  Score 0 = no match.
 */
function score(e: Entry, q: string): Scored | null {
  const full = norm(q);
  if (!full) return null;
  const name = norm(e.name);
  const aliases = (e.aliases || []).map(norm);
  const words = full.split(/\s+/).filter(Boolean);
  let total = 0;
  let via: string | undefined;
  let weak = true;
  for (const w of words) {
    let s = 0;
    const aliasIdx = aliases.findIndex((a) => a === w);
    if (name === w) s = 100;
    else if (aliasIdx >= 0) { s = 95; via = e.aliases![aliasIdx]; }
    else if (name.startsWith(w)) s = 80;
    else if (name.split(/[^a-z0-9]+/).some((t) => t.startsWith(w))) s = 65;
    else if (name.includes(w)) s = 50;
    else {
      const ai = aliases.findIndex((a) => a.startsWith(w) || a.split(/[^a-z0-9]+/).some((t) => t.startsWith(w)));
      if (ai >= 0) { s = 48; via = e.aliases![ai]; }
      else if (e.keywords.some((k) => k.toLowerCase().includes(w))) s = 40;
      else {
        const ai2 = aliases.findIndex((a) => a.includes(w));
        if (ai2 >= 0) { s = 38; via = e.aliases![ai2]; }
        else if (e.category.toLowerCase().includes(w)) s = 30;
        else if (e.id.toLowerCase().includes(w)) s = 28;
        else if (e.description.toLowerCase().includes(w)) s = 20;
      }
    }
    if (s === 0 && w.length >= 4) {
      // typo tolerance against the words of the name and the aliases
      const max = w.length >= 7 ? 2 : 1;
      const cands = [...name.split(/[^a-z0-9]+/), ...aliases.flatMap((a) => a.split(/[^a-z0-9]+/))].filter((t) => t.length >= 3);
      if (cands.some((t) => editDistance(t, w, max) <= max)) s = 15;
      else if (subsequence(name, w)) s = 8;
    }
    if (s === 0) return null;
    if (s > 15) weak = false;
    total += s;
  }
  // the whole query as typed: "open sample" ranks Open Sample above Open ImageJ Sample…, "fire" the LUT above Proxy Settings
  if (name === full) total += 200;
  else if (aliases.includes(full)) total += 150;
  else if (name.startsWith(full)) total += 120;
  else if (name.includes(full)) total += 60;
  else if (aliases.some((a) => a.startsWith(full))) total += 50;
  return { e, s: total, via, weak };
}

function subsequence(s: string, q: string): boolean {
  let i = 0;
  for (const ch of s) if (ch === q[i]) i++;
  return i === q.length;
}

export default function CommandPalette() {
  const ops = useStore((s) => s.meta?.ops || []);
  const lutNames = useStore((s) => s.meta?.luts || []);
  const lutGrouping = useStore((s) => s.lutGrouping);
  const recent = useStore((s) => s.recent);
  const setOpen = useStore((s) => s.setPaletteOpen);
  const invoke = useStore((s) => s.invoke);
  const img = useStore((s) => s.activeImage());
  const selection = useStore((s) => (img ? s.selections[img.id] : null));
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  // Fiji names as aliases (docs/FIJI_PARITY.md); the static map until the document has been parsed once
  const [aliases, setAliases] = useState<AliasMap>(() => getAliases());
  useEffect(() => {
    setAliases(getAliases((m) => setAliases(m)));
  }, []);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const entries = useMemo<Entry[]>(() => {
    const lutOp = ops.find((o) => o.id === "color.lut");
    const luts: Entry[] = lutOp
      ? lutNames.map((n) => ({
          id: `lut:${n}`, name: `LUT: ${n}`, category: "Image/Lookup Tables", op: lutOp, params: { lut: n },
          description: `Set the current channel's lookup table to ${n} (display only – pixel values unchanged)${lutGrouping.aliases[n] ? " · " + lutGrouping.aliases[n] : ""}`,
          keywords: ["lut", "lookup table", "colormap", "colour map", n.toLowerCase(), (lutGroupOf(n, lutGrouping) || "").toLowerCase()].filter(Boolean),
          aliases: [n],
        }))
      : [];
    // one entry per name: a command whose label duplicates a built-in action (same thing, two routes) is listed once,
    // and every name ends in the same ellipsis character (E7-13)
    const actionNames = new Set(ACTIONS.map((a) => norm(a.name)));
    const tidy = (s: string) => s.replace(/\.\.\.\s*$/, "…");
    return [
      ...ACTIONS.map((a) => ({ ...a, name: tidy(a.name) })),
      ...ops.filter((o) => !HIDDEN_MENU_OPS.has(o.id) && !actionNames.has(norm(displayName(o)))).map((o) => ({ id: o.id, name: tidy(displayName(o)), category: o.menu, description: o.description, kbd: prettyShortcut(o.shortcut), op: o, keywords: o.keywords, aliases: aliases[o.id] })),
      ...luts,
    ];
  }, [ops, lutNames, lutGrouping, aliases]);

  const results = useMemo(() => {
    const query = q.trim();
    if (!query) {
      const rec = recent.map((id) => entries.find((e) => e.id === id)).filter(Boolean) as Entry[];
      const rest = entries.filter((e) => !recent.includes(e.id));
      return { groups: [{ title: "Recent", items: rec }, { title: "All commands", items: rest }].filter((g) => g.items.length), empty: false, via: new Map<string, string>() };
    }
    let scored = entries.filter((e) => !WEB_SEARCH_IDS.has(e.id)).map((e) => score(e, query)).filter((x): x is Scored => !!x);
    // fuzzy noise only when nothing better exists (E7-10)
    if (scored.some((x) => !x.weak)) scored = scored.filter((x) => !x.weak);
    scored.sort((a, b) => b.s - a.s || a.e.name.localeCompare(b.e.name));
    scored = scored.slice(0, 80);
    const via = new Map<string, string>();
    for (const x of scored) if (x.via && norm(x.via) !== norm(x.e.name)) via.set(x.e.id, x.via);
    // the typed words become the query of the ImageJ wiki / image.sc forum search (opened in a new tab)
    const web = entries.filter((e) => WEB_SEARCH_IDS.has(e.id)).map((e) => ({ ...e, name: e.name.replace(/…$/, ` “${query}”`), arg: query }));
    return { groups: [{ title: `${scored.length} result${scored.length === 1 ? "" : "s"}`, items: scored.map((x) => x.e) }, { title: "Search the web", items: web }].filter((g) => g.items.length), empty: scored.length === 0, via };
  }, [q, entries, recent]);
  const flat = results.groups.flatMap((g) => g.items);
  const query = q.trim();
  // the same count in the header, the footer and the live region (E7-10): matches, excluding the two web-search rows
  const nMatches = query ? flat.filter((e) => !WEB_SEARCH_IDS.has(e.id)).length : flat.length;
  // one "commands" number everywhere (E8-5): the registry commands, as on the empty screen and in the command
  // reference; the palette's own UI actions and the LUT entries are counted separately
  const isLut = (e: Entry) => e.id.startsWith("lut:");
  const isCommand = (e: Entry) => !!e.op && !isLut(e);
  const isAction = (e: Entry) => !e.op && !isLut(e) && !WEB_SEARCH_IDS.has(e.id);
  const nCommands = commandCount(ops);
  const nActions = entries.filter(isAction).length;
  const nLuts = entries.filter(isLut).length;
  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const footText = query
    ? `${flat.filter(isCommand).length} of ${nCommands} commands` + (flat.some(isAction) ? ` + ${plural(flat.filter(isAction).length, "UI action")}` : "") + (flat.some(isLut) ? ` + ${plural(flat.filter(isLut).length, "LUT")}` : "")
    : `${nCommands} commands + ${plural(nActions, "UI action")} + ${plural(nLuts, "LUT")}`;
  // announced to screen readers (aria-live): the result count follows every keystroke, the highlight follows ↑↓
  const liveText = query ? `${nMatches} result${nMatches === 1 ? "" : "s"} for ${query}` : `${nCommands} commands`;
  const optionId = (i: number) => `palette-opt-${i}`;

  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  useEffect(() => setSel(0), [q]);
  useEffect(() => {
    const el = listRef.current?.querySelector(".palette-item.sel") as HTMLElement | null;
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  /** Why a command cannot run right now (the menu greys it out the same way, E5-10 / E7-18); undefined = enabled. */
  const disabledReason = (e: Entry): string | undefined => {
    if (!e.op) return undefined;
    if (e.op.requires_image && !img) return "needs an open image";
    // derived from the op's requires_roi kind (area / line / point / any) with the tool keys that draw it (E7-18)
    if (e.op.requires_roi && !selection) return needsSelectionShort(e.op);
    return undefined;
  };

  const run = (e: Entry) => {
    setOpen(false);
    if (e.op && e.params) {
      const st = useStore.getState();
      if (e.op.requires_image && !st.activeImage()) return st.toast("Open an image first", "error");
      void st.runOp(e.op.id, e.params);
    } else if (e.op) invoke(e.op.id);
    else if (e.action) doAction(e.action, e.arg);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { setSel((s) => Math.min(s + 1, flat.length - 1)); e.preventDefault(); }
    else if (e.key === "ArrowUp") { setSel((s) => Math.max(s - 1, 0)); e.preventDefault(); }
    else if (e.key === "Enter") { if (flat[sel]) run(flat[sel]); }
    else if (e.key === "Escape") setOpen(false);
  };

  let idx = -1;
  let group = -1;
  return (
    <div className="modal-backdrop" onMouseDown={() => setOpen(false)}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-input">
          <Search size={16} className="dim" aria-hidden="true" />
          <input
            ref={inputRef}
            role="combobox"
            aria-label="Search commands"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-haspopup="listbox"
            aria-autocomplete="list"
            aria-activedescendant={flat[sel] ? optionId(sel) : undefined}
            autoComplete="off"
            spellCheck={false}
            placeholder="Type a command or a Fiji name… e.g. gaussian, threshold, CLAHE, fire, z project"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKey}
          />
          <span className="kbd" aria-hidden="true">esc</span>
        </div>
        <div className="sr-only" aria-live="polite" aria-atomic="true">{liveText}</div>
        <div className="palette-list" ref={listRef} role="listbox" id="palette-list" aria-label="Commands">
          {results.empty && <div className="palette-empty" role="presentation">No command matches “{q}” – search the ImageJ wiki or the image.sc forum below</div>}
          {results.groups.map((g) => {
            group++;
            const groupId = `palette-group-${group}`;
            return (
            <div key={g.title} role="group" aria-labelledby={groupId}>
              <div className="palette-group" id={groupId} role="presentation">{g.title}</div>
              {g.items.map((e) => {
                idx++;
                const i = idx;
                const reason = disabledReason(e);
                const via = results.via.get(e.id);
                return (
                  <div key={e.id} id={optionId(i)} role="option" aria-selected={i === sel} aria-disabled={reason ? true : undefined} className={"palette-item" + (i === sel ? " sel" : "") + (reason ? " disabled" : "")} title={reason ? `${e.name}: ${reason}` : undefined} onMouseEnter={() => setSel(i)} onClick={() => run(e)}>
                    <div className="pi-text">
                      <span className="pi-name"><Highlight text={e.name} q={q} /></span>
                      <span className="pi-cat">{e.category}</span>
                      {via && <span className="pi-alias" title="Fiji name">Fiji: {via}</span>}
                      {reason && <span className="pi-reason">{reason}</span>}
                      {e.description && <div className="pi-desc">{e.description}</div>}
                    </div>
                    {e.kbd && <span className="kbd pi-kbd">{kbd(e.kbd)}</span>}
                    {i === sel && <CornerDownLeft size={13} className="dim" aria-hidden="true" />}
                  </div>
                );
              })}
            </div>
            );
          })}
        </div>
        <div className="palette-foot" aria-hidden="true"><span>↑↓ navigate</span><span>↵ run</span><span>esc close</span><span style={{ marginLeft: "auto" }}>{footText}</span></div>
      </div>
    </div>
  );
}

function Highlight({ text, q }: { text: string; q: string }) {
  const full = q.trim().toLowerCase();
  const w = full.split(/\s+/)[0];
  if (!w) return <>{text}</>;
  const low = text.toLowerCase();
  let i = low.indexOf(full);
  let len = full.length;
  if (i < 0) {
    i = low.indexOf(w);
    len = w.length;
  }
  if (i < 0) return <>{text}</>;
  return <>{text.slice(0, i)}<mark>{text.slice(i, i + len)}</mark>{text.slice(i + len)}</>;
}
