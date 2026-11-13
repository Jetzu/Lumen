export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || (navigator as any).userAgentData?.platform || "");

/** Render a shortcut label for the current platform (⌘ → Ctrl, ⌥ → Alt, ⇧ → Shift on non-Mac). */
export function kbd(s: string | undefined | null): string | undefined {
  if (!s) return undefined;
  if (isMac) return s;
  return s.replace(/⌘/g, "Ctrl+").replace(/⌥/g, "Alt+").replace(/⇧/g, "Shift+").replace(/\+\+/g, "+");
}
