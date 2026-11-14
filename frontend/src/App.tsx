import React, { useEffect, useState } from "react";
import { useStore, clampInspectorWidth, INSPECTOR_DEFAULT_W } from "./store";
import { useShortcuts } from "./shortcuts";
import TopBar from "./components/TopBar";
import ImageTabs from "./components/ImageTabs";
import ToolRail from "./components/ToolRail";
import Viewport from "./components/Viewport";
import SplitView from "./components/SplitView";
import Inspector from "./components/Inspector";
import ResultsModal from "./components/ResultsModal";
import StatusBar from "./components/StatusBar";
import Toasts from "./components/Toasts";
import CommandPalette from "./components/CommandPalette";
import FileDialog from "./components/FileDialog";
import HelpOverlay from "./components/HelpOverlay";
import PreferencesSheet from "./components/PreferencesSheet";
import DocsModal from "./components/DocsModal";
import ShortcutsDialog from "./components/ShortcutsDialog";
import AgentDock from "./components/AgentDock";
import { useMaterialiseWarning } from "./components/MemoryMonitor";

export default function App() {
  const ready = useStore((s) => s.ready);
  const fatal = useStore((s) => s.fatal);
  const theme = useStore((s) => s.theme);
  const paletteOpen = useStore((s) => s.paletteOpen);
  const fileDialog = useStore((s) => s.fileDialog);
  const helpOpen = useStore((s) => s.helpOpen);
  const inspectorOpen = useStore((s) => s.inspectorOpen);
  const inspectorWidth = useStore((s) => s.inspectorWidth);
  const prefsOpen = useStore((s) => s.prefsOpen);
  const docsOpen = useStore((s) => s.docsOpen);
  const shortcutsOpen = useStore((s) => s.shortcutsOpen);
  const tableModal = useStore((s) => s.tableModal);
  const splitOn = useStore((s) => s.split.on);
  const roiColor = useStore((s) => s.prefs.roi_color);
  const fontFamily = useStore((s) => s.prefs.font_family);
  const [resizing, setResizing] = useState(false);
  useShortcuts();
  // "about to read a virtual stack into memory" toast (data larger than RAM): fires whether or not the Memory
  // Monitor panel is on; the hook de-duplicates per image, so the monitor's own call never doubles the toast
  useMaterialiseWarning();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  // number inputs format with a decimal point whatever the OS locale (E5-14 / E6-6 / E7-26): the browser takes the
  // number locale from the element's `lang`, so the document and the app root both say "en"
  useEffect(() => {
    if (!document.documentElement.lang) document.documentElement.lang = "en";
  }, []);
  // preferences that are pure CSS: selection colour (--roi) and the overlay font
  useEffect(() => {
    document.documentElement.style.setProperty("--roi", roiColor || "#ffd54a");
    document.documentElement.style.setProperty("--overlay-font", fontFamily ? fontFamily : "var(--font)");
  }, [roiColor, fontFamily]);
  useEffect(() => {
    void useStore.getState().init();
  }, []);
  useEffect(() => {
    const prevent = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);
  // the inspector may never take more than 60 % of the window: re-clamp (without persisting) on resize
  useEffect(() => {
    const onResize = () => {
      const st = useStore.getState();
      const c = clampInspectorWidth(st.inspectorWidth);
      if (c !== st.inspectorWidth) useStore.setState({ inspectorWidth: c });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  if (fatal) return <div style={{ padding: 40, color: "var(--danger)" }}>Cannot reach the Lumen server: {fatal}. Start it with <code>lumen</code> (or <code>python -m lumen</code>).</div>;
  if (!ready) return <div style={{ padding: 40, color: "var(--text-muted)" }}>Loading Lumen…</div>;
  return (
    <div className={"app" + (inspectorOpen ? "" : " no-inspector") + (resizing ? " resizing" : "")} lang="en" style={{ "--inspector-w": `${inspectorWidth}px` } as React.CSSProperties}>
      <TopBar />
      <ImageTabs />
      <ToolRail />
      {splitOn ? <SplitView /> : <Viewport />}
      {inspectorOpen && <InspectorSplitter onDragChange={setResizing} />}
      {inspectorOpen && <Inspector />}
      <AgentDock />
      <StatusBar />
      <Toasts />
      {paletteOpen && <CommandPalette />}
      {fileDialog && <FileDialog />}
      {helpOpen && <HelpOverlay />}
      {prefsOpen && <PreferencesSheet />}
      {shortcutsOpen && <ShortcutsDialog />}
      {docsOpen && <DocsModal />}
      {tableModal && <ResultsModal />}
    </div>
  );
}

/** Drag handle on the inspector's left edge: 260 px … 60 % of the window; double-click resets; focusable, ← → resize by 24 px, Home resets (E7-19). */
function InspectorSplitter({ onDragChange }: { onDragChange: (dragging: boolean) => void }) {
  const setInspectorWidth = useStore((s) => s.setInspectorWidth);
  const width = useStore((s) => s.inspectorWidth);
  const [dragging, setDragging] = useState(false);
  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 96 : 24;
    let next: number | null = null;
    if (e.key === "ArrowLeft") next = width + step; // the inspector grows to the left
    else if (e.key === "ArrowRight") next = width - step;
    else if (e.key === "Home") next = INSPECTOR_DEFAULT_W;
    else return;
    e.preventDefault();
    e.stopPropagation();
    setInspectorWidth(next);
  };
  const max = clampInspectorWidth(1e9);
  const start = (d: boolean) => {
    setDragging(d);
    onDragChange(d);
  };
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    start(true);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging) return;
    // the inspector is flush with the window's right edge
    useStore.setState({ inspectorWidth: clampInspectorWidth(window.innerWidth - e.clientX) });
  };
  const onPointerUp = () => {
    if (!dragging) return;
    start(false);
    setInspectorWidth(useStore.getState().inspectorWidth); // persists
  };
  return (
    <div className={"splitter" + (dragging ? " dragging" : "")} role="separator" aria-orientation="vertical" aria-label="Inspector width" aria-valuemin={260} aria-valuemax={max} aria-valuenow={Math.round(width)} aria-valuetext={`${Math.round(width)} px`} tabIndex={0}
      title="Drag to resize the inspector · double-click to reset · ← → when focused"
      onKeyDown={onKeyDown} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onDoubleClick={() => setInspectorWidth(INSPECTOR_DEFAULT_W)} />
  );
}
