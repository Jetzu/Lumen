/**
 * The Agent panel: Agent Chat for Lumen (frontend/public/agent – the ChemDraw MCP's Agent Chat, served by Lumen's
 * /api/agent bridge) docked on the right of the window.  Opened from the top bar (✻ Agent) or ⌘J; mounted on first
 * open and then kept alive while hidden, so a running answer keeps streaming.  The theme follows Lumen's
 * (postMessage); the panel's own × closes it.  The left edge drags to resize (width kept per browser).
 */
import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";

const W_KEY = "lumen.agentWidth";
const W_DEFAULT = 380; // small by default: the image keeps most of the window
const W_MIN = 280;

function loadWidth(): number {
  try {
    const v = Number(localStorage.getItem(W_KEY));
    return v >= W_MIN && v <= 1400 ? v : W_DEFAULT;
  } catch {
    return W_DEFAULT;
  }
}

export default function AgentDock() {
  const open = useStore((s) => s.agentOpen);
  const theme = useStore((s) => s.theme);
  const [mounted, setMounted] = useState(open);
  const [width, setWidth] = useState(loadWidth);
  const [src] = useState(() => `/agent/index.html?theme=${useStore.getState().theme}`);
  const frame = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (open) setMounted(true);
    document.documentElement.style.setProperty("--agent-w", open ? `${width}px` : "0px");
    window.dispatchEvent(new Event("resize")); // the viewport re-fits
  }, [open, width]);

  useEffect(() => {
    frame.current?.contentWindow?.postMessage({ type: "lumen-theme", theme }, window.location.origin);
  }, [theme, mounted]);

  useEffect(() => {
    const onMsg = (ev: MessageEvent) => {
      if (ev.origin === window.location.origin && ev.data && ev.data.type === "lumen-agent-close") useStore.getState().setAgentOpen(false);
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  useEffect(() => {
    if (open) window.setTimeout(() => frame.current?.contentWindow?.document.getElementById("prompt")?.focus(), 60);
  }, [open]);

  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = width;
    document.body.classList.add("agent-resizing");
    const move = (ev: PointerEvent) => setWidth(Math.max(W_MIN, Math.min(Math.round(window.innerWidth * 0.6), w0 + (x0 - ev.clientX))));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("agent-resizing");
      setWidth((w) => {
        try {
          localStorage.setItem(W_KEY, String(w));
        } catch {
          /* storage blocked */
        }
        return w;
      });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  if (!mounted) return null;
  return (
    <aside className={"agent-dock" + (open ? "" : " closed")} aria-label="Agent Chat" aria-hidden={!open}>
      <div className="agent-splitter" role="separator" aria-orientation="vertical" aria-label="Resize the Agent panel (drag; double-click resets)" title="Drag to resize · double-click resets" onPointerDown={startDrag} onDoubleClick={() => setWidth(W_DEFAULT)}><span className="grip" aria-hidden="true" /></div>
      <iframe ref={frame} src={src} title="Agent Chat for Lumen" onLoad={() => frame.current?.contentWindow?.postMessage({ type: "lumen-theme", theme: useStore.getState().theme }, window.location.origin)} />
    </aside>
  );
}
