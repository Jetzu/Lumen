/**
 * Lookup tables grouped for Image ▸ Lookup Tables and the Display section's LUT select (GUI fix round 2:
 * E5-1 / E6-9 / E7-13).  The server's table (`lumen.core.luts.LUT_GROUPS`, served by `GET /api/luts/groups`)
 * is authoritative; `DEFAULT_GROUPS` is the same table for an older server without the router, and any name
 * the groups do not know (a table imported at runtime) lands under "Imported".
 */
import { api } from "./api";

export interface LutGroup {
  name: string;
  luts: string[];
}

export interface LutGrouping {
  groups: LutGroup[];
  /** Near-duplicate names explained ("mpl-viridis" → "Fiji's mpl-viridis.lut (= Viridis, truncated)"). */
  aliases: Record<string, string>;
}

export const DEFAULT_GROUPS: LutGrouping = {
  groups: [
    { name: "ImageJ", luts: ["Grays", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Orange", "Fire", "Ice", "Spectrum", "3-3-2 RGB", "Red/Green", "HiLo"] },
    { name: "Fiji", luts: ["16 colors", "Thermal", "Phase", "Red Hot", "Orange Hot", "Yellow Hot", "Cyan Hot", "Magenta Hot", "Green Fire Blue", "Rainbow RGB", "Royal", "Sepia", "Thal", "Thallium", "Union Jack", "ICA", "ICA2", "ICA3", "brgbcmyw", "5 ramps", "6 shades", "blue orange icb", "cool", "edges", "gem", "glow", "physics", "smart"] },
    { name: "Perceptual (matplotlib)", luts: ["Viridis", "Magma", "Inferno", "Plasma", "Cividis", "Turbo", "Jet", "Hot", "Cool", "Rainbow", "mpl-viridis", "mpl-magma", "mpl-inferno", "mpl-plasma"] },
    { name: "Labels", luts: ["Glasbey", "Glasbey on dark", "Glasbey inverted", "Random"] },
  ],
  aliases: {
    "mpl-viridis": "Fiji's mpl-viridis.lut (= Viridis, truncated)", "mpl-magma": "Fiji's mpl-magma.lut (= Magma, truncated)",
    "mpl-inferno": "Fiji's mpl-inferno.lut (= Inferno, truncated)", "mpl-plasma": "Fiji's mpl-plasma.lut (= Plasma, truncated)",
    Thallium: "NIH Image table, like Thal", cool: "Fiji's cool.lut (not matplotlib's Cool)", Cool: "matplotlib (not Fiji's cool.lut)",
    Glasbey: "index 0 is white (Fiji's glasbey.lut)", "Glasbey on dark": "index 0 is black – label images",
  },
};

/** `GET /api/luts/groups`; the built-in table when the server predates the router. */
export async function fetchLutGroups(): Promise<LutGrouping> {
  try {
    const r = await api.lutGroups();
    if (r && Array.isArray(r.groups) && r.groups.length) return { groups: r.groups, aliases: r.aliases || {} };
  } catch {
    /* older server: the default table below */
  }
  return DEFAULT_GROUPS;
}

/** The registered LUT names arranged by group (names the table does not list go to "Imported", in registration order). */
export function groupLuts(names: string[], grouping: LutGrouping): LutGroup[] {
  const known = new Set(names);
  const placed = new Set<string>();
  const out: LutGroup[] = [];
  for (const g of grouping.groups) {
    const luts = g.luts.filter((n) => known.has(n));
    for (const n of luts) placed.add(n);
    if (luts.length) out.push({ name: g.name, luts });
  }
  const rest = names.filter((n) => !placed.has(n));
  if (rest.length) out.push({ name: "Imported", luts: rest });
  return out;
}

/** Label of a LUT in a select / menu: the name plus a short note for the near-duplicates ("mpl-viridis – = Viridis"). */
export function lutLabel(name: string, grouping: LutGrouping): string {
  const a = grouping.aliases[name];
  return a ? `${name} – ${a}` : name;
}

/** The group a LUT belongs to ("ImageJ" …), for palette keywords. */
export function lutGroupOf(name: string, grouping: LutGrouping): string | undefined {
  return grouping.groups.find((g) => g.luts.includes(name))?.name;
}
