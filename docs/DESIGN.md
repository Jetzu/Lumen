# Lumen — design notes

Lumen is a re-imagining of Fiji / ImageJ: the same core scientific image-processing
toolbox, behind a single-window, keyboard-first interface.

## What the research says

**Fiji's documented UX problems**

- More than 1,100 commands live in deeply nested dropdown menus (JoVE 2026, EasyFiji
  method article). Users without a programming background struggle to find anything.
- The Plugins menu "can become easily cluttered after the installation of several
  plugins" (official ImageJ user guide).
- "Window soup": every image, dialog, results table, histogram and ROI manager opens as
  its own floating window. Nothing is docked; state is scattered across the desktop.
- Dialogs are modal and blind: most filters have no live preview, and the ones that do
  (Gaussian, Median...) hide it behind a checkbox.
- Brightness/Contrast, Threshold, Channels and Color are four separate floating tools
  that must be re-opened per image.
- Inconsistent terminology ("Make Binary" vs "Convert to Mask", "Black background"
  option that silently inverts every binary op).
- Only one undo level.

**What the better tools do**

- *napari*: one window; canvas + layer list + per-layer controls; sliders for every
  extra dimension; plugins dock as side widgets instead of popping windows.
- *QuPath*: single workspace, project/file tree on the left, viewer in the middle,
  annotation/measurement panels on the right; a command list with live search.
- *ImageJ2 search bar*: the single most-loved ImageJ2 feature is the search bar that
  finds commands by name.
- *Modern creative tools (Figma, Photoshop, VS Code)*: command palette (⌘K), tabs
  instead of windows, inspector with collapsible sections, non-destructive previews,
  deep undo history.

## Lumen's design principles

1. **One window, three regions.** Tool rail on the left, viewport in the middle,
   inspector on the right. Images open as tabs. Results, ROIs, histogram, history and
   the script console live in inspector sections, never in floating windows.
2. **Command palette first.** ⌘K / Ctrl+K opens a fuzzy search over every command,
   with category, description and shortcut. Menus still exist for discoverability but
   the palette is the primary way to run anything.
3. **Every command is a schema.** Operations declare their parameters (type, range,
   default, choices). The UI auto-generates a parameter sheet, so every command looks
   and behaves the same, and scripts call the same registry.
4. **Live preview by default.** Any per-plane operation previews on the current slice
   while you drag a slider; nothing is written until *Apply*.
5. **Display is non-destructive and per-channel.** Brightness/contrast, LUT,
   visibility and composite mode are properties of the view, adjusted with a
   histogram-backed range slider that updates instantly on the client.
6. **Deep undo** (memory-bounded), with a visible history list you can click back into.
7. **Recorder always on.** Every applied command is recorded as a Python line you can
   copy into the built-in script console and re-run on another image.
8. **Honest binary semantics.** A mask is always foreground = 255, background = 0.
   No "black background" switch.

## Architecture

```
lumen/            Python package (FastAPI server + image engine)
  core/image.py   LumenImage: 5-D (T, Z, C, Y, X) numpy data + calibration + display
  core/registry.py  @op decorator: parameter schema, plane/image/analysis kinds, ROI masking
  core/ops/*.py   the command library (filters, math, threshold, binary, stacks, analyze...)
  core/roi.py     ROI shapes → masks, ROI manager
  core/measure.py ImageJ-compatible measurements (Area, Mean, ..., Feret, shape descriptors)
  core/threshold.py 16 auto-threshold methods on 256-bin histograms
  core/luts.py    look-up tables shared by server rendering and the client compositor
  core/session.py open images, results table, ROI manager, undo, recorder
  server/app.py   REST + WebSocket API, serves the built frontend
frontend/         React + TypeScript + Vite single-page app
  render/         client-side compositor: raw planes + LUT + min/max → canvas
  components/     TopBar (with menus), ToolRail, Viewport, OverlayItems, Inspector, VolumeViewer, OrthoViews, CommandPalette, OperationSheet
```

Planes travel from server to client as raw typed arrays; brightness, contrast, LUT and
channel compositing happen in the browser, so display adjustments never hit the server.
Processing happens in numpy / scipy / scikit-image on the server, which is the same
scientific stack napari, CellProfiler and scikit-image users already trust.

## Evidence and further reading

The claims in "What the research says" are documented, with links to every source, in
[RESEARCH.md](RESEARCH.md) (a sourced review of Fiji/ImageJ, ImageJ2, napari, QuPath, Icy,
CellProfiler, ilastik, the commercial packages, the browser viewers, 3D Slicer, general
interaction references and the usability literature, with a comparison matrix and a table
that traces each principle above to its evidence). Two caveats that came out of that review:

- "More than 1,100 commands" is the figure given by the EasyFiji article (JoVE 2026), which is
  paywalled and could not be re-verified. What can be verified: ImageJ 1.54's `IJ_Props.txt`
  declares 348 built-in submenu commands (the top-level ones are hard-coded on top of that), and
  the ImageJ wiki describes Fiji as bundling "nearly 1000 additional plugins".
- The "Plugins menu can become easily cluttered" sentence was not found on the current ImageJ
  User Guide pages; the documented statement is Rueden's 2018 forum summary that the legacy UI
  "is the result of many cycles of iteration and the existing user-base generally prefers it",
  which is why Lumen keeps the vocabulary and changes the window model.

The Fiji menu tree, command by command, is mapped to Lumen's registry in
[FIJI_PARITY.md](FIJI_PARITY.md); the full parameter list is in [COMMANDS.md](COMMANDS.md).


## Phase 2: closing the functionality gap

After the first build, the capabilities Fiji users rely on most were added as ordinary
registry commands, so they inherit the palette, parameter sheets, undo and the recorder:
tracking (`tracking.*`), registration and stitching (`registration.*`), trainable and
classical segmentation (`segment.*`), batch processing (`batch.*`), file formats and
ImageJ ROI files (`file.import_*` / `file.export_*`), the remaining ImageJ core commands
(`adjust.*`, `analyze.*`, `stacks.*`, `overlay.*`), a WebGL 3-D viewer with orthogonal
views, drawing/annotation tools, and an ImageJ-macro compatibility layer
(`lumen/core/ij_compat.py`) so existing Fiji macros can be pasted into the console.

Two adversarial review passes (reviewers plus two independent verifiers per finding)
produced 103 confirmed defects, all fixed with regression tests. The parity matrix in
`FIJI_PARITY.md` and the command reference in `COMMANDS.md` are generated from the
registry and describe the final state.
