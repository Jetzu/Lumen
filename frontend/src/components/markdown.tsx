/**
 * Minimal Markdown → React renderer for the bundled docs (README, COMMANDS …): headings, fenced
 * code, tables (rows that mention a command id get `data-op` so the viewer can scroll to them),
 * lists, block quotes, rules, paragraphs; inline code, bold and links.  Links to other bundled
 * documents switch the viewer, http(s) links open in a new tab, images are dropped.
 */
import React from "react";
import type { DocName } from "../store";

export interface MdOptions {
  onDoc?: (name: DocName) => void;
  /** Case-insensitive text filter applied to table rows (headings and prose stay). */
  filter?: string;
  /** Per command id, its ImageJ-macro equivalents (OpSpec.ij_commands): shown under the command's reference row (E7-20). */
  ijCommands?: Record<string, string[] | undefined>;
}

/** An ij_commands entry as a macro call (a bare "Gaussian Blur..." → run("Gaussian Blur...")); kept local so this module stays free of UI imports. */
function macroCall(c: string): string {
  const s = String(c).trim();
  return /^[A-Za-z_][A-Za-z0-9_.]*\s*\(/.test(s) ? s : `run("${s.replace(/"/g, '\\"')}")`;
}

const DOC_NAMES = new Set(["README", "COMMANDS", "FIJI_PARITY", "RESEARCH", "DESIGN"]);

export function docOfHref(href: string): DocName | null {
  const m = /(?:^|\/)([A-Za-z_]+)\.md(?:#.*)?$/.exec(href.trim());
  if (!m) return null;
  const name = m[1].toUpperCase();
  return DOC_NAMES.has(name) ? (name as DocName) : null;
}

export function slug(text: string): string {
  return text.toLowerCase().replace(/[`*_]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// code, **bold**, *italic* (not a lone "*" or a "2 * 3"), images (dropped) and links
const INLINE_RE = /(`[^`]+`|\*\*[^*]+\*\*|(?<![\w*])\*[^*\s](?:[^*]*[^*\s])?\*(?![\w*])|!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))/g;

export function inline(text: string, opts: MdOptions): React.ReactNode {
  const parts = text.split(INLINE_RE);
  return parts.map((p, k) => {
    if (!p) return null;
    if (/^`[^`]+`$/.test(p)) return <code key={k}>{p.slice(1, -1)}</code>;
    if (/^\*\*[^*]+\*\*$/.test(p)) return <strong key={k}>{p.slice(2, -2)}</strong>;
    if (/^\*[^*\s](?:[^*]*[^*\s])?\*$/.test(p)) return <em key={k}>{p.slice(1, -1)}</em>;
    if (/^!\[/.test(p)) return null;
    const m = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(p);
    if (m) {
      const label = m[1];
      const href = m[2];
      const doc = docOfHref(href);
      if (doc) return <a key={k} href="#" onClick={(e) => { e.preventDefault(); opts.onDoc?.(doc); }}>{label}</a>;
      if (/^https?:\/\//i.test(href)) return <a key={k} href={href} target="_blank" rel="noopener noreferrer">{label}</a>;
      return <span key={k} title={href}>{label}</span>;
    }
    return <React.Fragment key={k}>{p}</React.Fragment>;
  });
}

function splitRow(line: string): string[] {
  const s = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return s.split(/(?<!\\)\|/).map((c) => c.replace(/\\\|/g, "|").trim());
}

const ID_CELL = /^`([A-Za-z0-9_.:-]+)`$/;

function table(rows: string[], key: number, opts: MdOptions): React.ReactNode {
  const header = splitRow(rows[0]);
  const hasSep = /^\s*\|?\s*:?-{2,}/.test(rows[1] || "");
  const body = rows.slice(hasSep ? 2 : 1).map(splitRow);
  const filter = (opts.filter || "").trim().toLowerCase();
  const opOf = (r: string[]) => r.map((c) => ID_CELL.exec(c)?.[1]).find(Boolean);
  const ijOf = (op: string | undefined) => (op && opts.ijCommands ? opts.ijCommands[op] || [] : []);
  // the filter also matches the ImageJ names ("Gaussian Blur..." finds filters.gaussian)
  const visible = filter ? body.filter((r) => (r.join(" ") + " " + ijOf(opOf(r)).join(" ")).toLowerCase().includes(filter)) : body;
  if (filter && !visible.length) return null;
  return (
    <table key={key} className="md-table">
      <thead><tr>{header.map((h, k) => <th key={k}>{inline(h, opts)}</th>)}</tr></thead>
      <tbody>
        {visible.map((r, k) => {
          const op = opOf(r);
          const ij = ijOf(op);
          const last = r.length - 1;
          return (
            <tr key={k} data-op={op} id={op ? "cmd-" + op : undefined}>
              {r.map((c, j) => (
                <td key={j}>
                  {inline(c, opts)}
                  {j === last && ij.length > 0 && <div className="md-ij" title="The equivalent call in an ImageJ macro">ImageJ: {ij.map((x, i) => <code key={i}>{macroCall(x)}</code>)}</div>}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

const LIST_RE = /^\s*(?:[-*+]|\d+[.)])\s+(.*)$/;

export function renderMarkdown(md: string, opts: MdOptions = {}): React.ReactNode[] {
  const lines = md.replace(/\r\n?/g, "\n").split("\n");
  const out: React.ReactNode[] = [];
  let key = 0;
  let i = 0;
  const para: string[] = [];
  const flush = () => {
    if (para.length) {
      out.push(<p key={key++}>{inline(para.join(" "), opts)}</p>);
      para.length = 0;
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(<pre key={key++}>{buf.join("\n")}</pre>);
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flush();
      const level = Math.min(h[1].length + 1, 6);
      const text = h[2].replace(/\s+#+\s*$/, "");
      out.push(React.createElement(`h${level}`, { key: key++, id: slug(text) }, inline(text, opts)));
      i++;
      continue;
    }
    if (/^\s*\|/.test(line)) {
      flush();
      const rows: string[] = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const t = table(rows, key++, opts);
      if (t) out.push(t);
      continue;
    }
    if (LIST_RE.test(line)) {
      flush();
      const ordered = /^\s*\d/.test(line);
      const items: string[] = [];
      // an item continues on indented lines that are not items themselves (wrapped prose under a bullet)
      while (i < lines.length) {
        if (LIST_RE.test(lines[i])) items.push(lines[i++].replace(/^\s*(?:[-*+]|\d+[.)])\s+/, ""));
        else if (items.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*\|/.test(lines[i])) items[items.length - 1] += " " + lines[i++].trim();
        else break;
      }
      out.push(React.createElement(ordered ? "ol" : "ul", { key: key++ }, items.map((t, k) => <li key={k}>{inline(t, opts)}</li>)));
      continue;
    }
    if (/^\s*$/.test(line)) {
      flush();
      i++;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      flush();
      out.push(<hr key={key++} />);
      i++;
      continue;
    }
    if (/^>\s?/.test(line)) {
      flush();
      const quote: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, "").trim());
      out.push(<blockquote key={key++}>{inline(quote.join(" "), opts)}</blockquote>);
      continue;
    }
    para.push(line.trim());
    i++;
  }
  flush();
  return out;
}
