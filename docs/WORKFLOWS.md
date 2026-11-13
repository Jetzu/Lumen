# Lumen workflows

Step-by-step recipes for microscopists. Every recipe gives the goal, a sample image to try it on, the
clicks in the single-window UI, the equivalent Python script for the **Script & Recorder** section
(every script below was run through the console runner before it was written down), the output you
should see, the pitfalls, and what you would have done in Fiji.

Conventions:

- **⌘K** (Ctrl+K on Linux/Windows) opens the command palette; type part of a command name.
  Command *names* are given in the UI steps, command *ids* (what scripts use) in the scripts.
  The full table of ids and parameters is in [COMMANDS.md](COMMANDS.md).
- *Inspector* sections (right-hand side): **Display**, **Threshold**, **3-D View**, **Orthogonal Views**,
  **ROI Manager**, **Results**, **Plot**, **Histogram**, **Image Info**, **History**, **Script & Recorder**, **Log**,
  **Workflows**. A **Table** section appears when a command produces a named table (Particle Summary, Tracks,
  Branches, Batch …). The **Workflows** section walks through guided versions of several recipes below
  (`count_nuclei`, `intensity_per_cell`, `colocalization`, `track_cells`, `zstack_figure`, `stitch_tiles`,
  `batch_folder`, `pixel_classifier`, `register_timelapse`) step by step and can emit the equivalent script.
- Commands with parameters open a *sheet* at the top-right of the viewport. **Preview** shows the result on
  the current plane while you drag; **Apply** commits it (one undo step).
- Scripts run in **Script & Recorder ▸ Run** (**⌘↵**). `lumen.run("<id>", **params)` returns, in this
  order of precedence, a plot (dict), the Results rows it added (list of dicts), the new images it
  created (list of `LumenImage`), or the raw payload (dict with `message`, `table`, `new_images` …).
  When a command does several of these, use `lumen.image`, `lumen.results`, `lumen.rois` and
  `lumen.session.tables` to reach the rest. See [SCRIPTING.md](SCRIPTING.md) for the API.
- Samples: **File ▸ Open Sample** lists Lumen's bundled samples (`file.sample`, generated from
  scikit-image, no download) and the 29 classic ImageJ samples (`file.open_imagej_sample`, downloaded
  once into `~/.lumen/samples`).
- Output paths in the scripts (`~/lumen-demo`) are created by the script; change `out` to taste.

## Contents

1. [Count nuclei in a fluorescence image](#1-count-nuclei-in-a-fluorescence-image)
2. [Measure per-cell intensity in a second channel](#2-measure-per-cell-intensity-in-a-second-channel)
3. [Colocalization of two channels (Costes + Manders)](#3-colocalization-of-two-channels-costes--manders)
4. [Track moving cells in a time-lapse and export to TrackMate](#4-track-moving-cells-in-a-time-lapse-and-export-to-trackmate)
5. [Handle a z-stack: projection, orthogonal views, 3-D view, surface export](#5-handle-a-z-stack-projection-orthogonal-views-3-d-view-surface-export)
6. [Stitch a grid of tiles](#6-stitch-a-grid-of-tiles)
7. [Batch-process a folder with a recorded macro](#7-batch-process-a-folder-with-a-recorded-macro)
8. [Manage ROIs: manager, combine, ImageJ .roi/.zip interchange](#8-manage-rois-manager-combine-imagej-roizip-interchange)
9. [Calibrate and measure lengths and areas](#9-calibrate-and-measure-lengths-and-areas)
10. [Make a figure: LUTs, composite, scale bar, calibration bar, annotations, flatten, export](#10-make-a-figure-luts-composite-scale-bar-calibration-bar-annotations-flatten-export)
11. [Run an existing ImageJ macro in Lumen](#11-run-an-existing-imagej-macro-in-lumen)
12. [Train a pixel classifier (Weka-style)](#12-train-a-pixel-classifier-weka-style)
13. [Analyze skeletons (neurites)](#13-analyze-skeletons-neurites)
14. [Register / align a drifting time-lapse](#14-register--align-a-drifting-time-lapse)
15. [FFT filtering of periodic noise](#15-fft-filtering-of-periodic-noise)

---

## 1. Count nuclei in a fluorescence image

**Goal.** Segment the DAPI channel of a multi-channel image, split touching nuclei, count them, and get
one Results row (area, mean …) and one ROI per nucleus.

**Sample.** ImageJ sample **Fluorescent Cells** (3 channels, 8-bit: Red = actin, Green = tubulin, Blue =
nuclei). Offline alternative: bundled **Synthetic nuclei (3 channels)** (channel 1 = DAPI, 16-bit,
0.325 µm/px).

**UI steps.**

1. **File ▸ Open Sample ▸ Fluorescent Cells** (or ⌘K → *Open ImageJ Sample…*).
2. In **Display**, click the third channel (Blue) so that it is the *current* channel: commands act on
   the current channel of the current plane unless their *apply to* selector says otherwise.
3. ⌘K → **Gaussian Blur**, sigma 1, *Apply to: Current plane only (channel 3)*, **Apply** (optional; it
   suppresses noise before thresholding). The *Apply to* selector of every plane filter offers *Current
   plane only*, *All channels of this slice*, *Channel n, all slices/frames* and *Entire stack*.
4. In **Threshold**, choose **Otsu**, keep *dark bg* on, click **Auto**: the in-range pixels are shown in
   red. Click **Apply → mask**: for a multi-channel image this creates a new 8-bit image *Mask of channel 3*
   (a single-channel image is converted in place).
5. ⌘K → **Fill Holes**, then **Watershed** (method *Distance transform (h-maxima)*; raise *tolerance* if
   nuclei get over-split).
6. ⌘K → **Analyze Particles**: *Min size* 50, *Exclude on edges* on, *Add to manager* on, *Summarize* on,
   *Show: Outlines*. **Apply**.
7. Read the count in the toast / **Log** ("17 particles"), the per-nucleus rows in **Results**, the
   **Particle Summary** table (Count, Total Area, Average Size, %Area, Mean) and the outlines in
   **ROI Manager** (click **Show all** to draw them; select a row to jump to that nucleus).

**Script.**

```python
# recipe: count_nuclei
lumen.run("file.open_imagej_sample", name="Fluorescent Cells")   # 3 channels: Red, Green, Blue (nuclei)
img = lumen.image
img.display.c = 2                                                 # channel 3 (0-based index) = DAPI
lumen.run("filters.gaussian", sigma=1.0, scope="plane")
lumen.run("threshold.apply", method="Otsu", dark_background=True) # -> new image "Mask of channel 3"
lumen.run("binary.fill_holes")
lumen.run("binary.watershed")                                     # distance-transform watershed
rows = lumen.run("analyze.particles", min_size=50, exclude_on_edges=True, show="Outlines")
print(len(rows), "nuclei")
summary = lumen.session.tables["Particle Summary"]
print(summary["columns"], summary["rows"][0])
print(lumen.results[0])
print("ROIs in manager:", len(lumen.rois), lumen.rois[0]["name"])
```

**Expected output.** `17 nuclei`; Particle Summary row `['Mask of FluorescentCells.tif', 17, 6657.0, 391.6, 2.54, 254.4]`;
first Results row `{'Label': 'Mask of FluorescentCells.tif:1', 'Area': 642.0, 'Mean': 255.0, …}`;
17 polygon ROIs named `0001-<y>-<x>`; a new image *Outlines of Mask of FluorescentCells.tif*.
With *Synthetic nuclei* and `min_size=20` you get 39 nuclei.

**Pitfalls.**

- The Results columns come from **Set Measurements** (`analyze.set_measurements`); the default is
  Area, Mean, Min, Max. Measured *on the mask* these are 255 — see recipe 2 for intensities.
- `min_size` / `max_size` are in calibrated units² when the image is calibrated (µm² for the synthetic
  sample), in pixels² otherwise. `max_size=0` means no upper limit.
- A thresholded image that is not 0/255 is auto-thresholded with the *Default* method inside
  Analyze Particles; set a threshold first (Threshold panel or `threshold.compute`) to control it.
- `include_holes=True` (default) fills holes before labelling, as ImageJ's *Include holes* does.

**Fiji.** *Image ▸ Color ▸ Split Channels*, *Process ▸ Filters ▸ Gaussian Blur…*, *Image ▸ Adjust ▸ Threshold…* (Otsu, Apply),
*Process ▸ Binary ▸ Fill Holes* / *Watershed*, *Analyze ▸ Analyze Particles…* (`size=50-Infinity exclude summarize add`).
Lumen's *Black background* is always on (masks are 255 = object), so no *Process ▸ Binary ▸ Options* step.

---

## 2. Measure per-cell intensity in a second channel

**Goal.** Segment nuclei on channel 1 and read, for every nucleus, the mean intensity (and integrated
density) of channel 2.

**Sample.** Bundled **Synthetic nuclei (3 channels)**: DAPI, Cytoplasm, Spots (16-bit, 0.325 µm/px).
Also works on *Fluorescent Cells* (segment Blue, measure Red or Green).

**UI steps.**

1. **File ▸ Open Sample ▸ Synthetic nuclei (3 channels)**.
2. ⌘K → **Split Channels**: three single-channel images *C1-*, *C2-*, *C3-synthetic_cells.tif* appear as tabs.
3. On *C1-…*: **Gaussian Blur** (sigma 1), **Threshold ▸ Otsu ▸ Apply → mask** (single channel → converted in place),
   **Watershed**.
4. ⌘K → **Set Measurements**: Area, Mean, Integrated density, Centroid.
5. ⌘K → **Analyze Particles** with *Measure intensities on* = **C2-synthetic_cells.tif**, *Min size* 20,
   *Exclude on edges* on. The Results rows now carry the gray values of C2 inside each nucleus.
6. Alternative without splitting: segment as in recipe 1 (the ROIs land in the **ROI Manager**), click the
   original image tab, pick channel 2 in **Display**, then ⌘K → **Multi Measure (ROI Manager)** with
   *one row per slice* off: one row per ROI measured on the *current channel*.

**Script.**

```python
# recipe: second_channel
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")   # DAPI, Cytoplasm, Spots
src = lumen.image
lumen.run("color.split")                                         # C1-, C2-, C3-synthetic_cells.tif
lumen.select("C1-synthetic_cells.tif")
lumen.run("filters.gaussian", sigma=1.0)
lumen.run("threshold.apply", method="Otsu")
lumen.run("binary.watershed")
lumen.run("analyze.set_measurements", measurements=["area", "mean", "integrated_density", "centroid"])
# A) read the gray values of every nucleus from the cytoplasm channel while segmenting
rows = lumen.run("analyze.particles", min_size=20, exclude_on_edges=True, clear_results=True,
                 intensity_image="C2-synthetic_cells.tif")
print(len(rows), "cells; first row:", rows[0])
# B) the same ROIs measured on any channel of the original image (ROI Manager ▸ Multi Measure)
lumen.select(src.name)
src.display.c = 1                                                # channel 2 = Cytoplasm (0-based)
res = lumen.run("analyze.multi_measure", axis="Z", one_row_per_slice=False)
print("multi-measure rows:", len(res), res[0])
# C) one ROI at a time
r = lumen.rois[0]
print("single ROI on C2:", lumen.run("analyze.measure", roi=r)[0])
```

**Expected output.** 39 cells; row A `{'Label': 'C1-synthetic_cells.tif:1', 'Area': 64.33, 'Mean': 643.3, 'IntDen': 41383.7, 'RawIntDen': 391798.0, 'X': 102.5, 'Y': 8.3}`;
B and C give the same Mean/IntDen for the same nucleus (`Label` differs: the ROI name `0001-25-315`).

**Pitfalls.**

- `intensity_image` reads the plane of the other image at the *same channel index* as the mask plane.
  A mask has one channel, so it reads channel 1 of a multi-channel intensity image — split the channels
  first (as above) or use Multi Measure / `analyze.measure` with the display set to the wanted channel.
- The **Measure** button in the ROI Manager measures each ROI on the plane it was created on (its stored
  *position*: t, z, c of the mask). On the original image that is channel 1. **Multi Measure** and the
  script's `analyze.measure` use the *current* channel instead.
- `clear_results=True` empties the Results table before adding the new rows (ImageJ *Clear results*).
- `analyze.label_objects` (**Measure Labels**) does the same for a label image (e.g. from
  **Label Connected Components** or **Expand Labels** to grow nuclei into the cytoplasm).

**Fiji.** *Analyze ▸ Analyze Particles…* with *Add to Manager*, then select the other channel and
*ROI Manager ▸ Measure* / *More ▸ Multi Measure*; or *Image ▸ Adjust ▸ Threshold…* on C1 and
*Analyze ▸ Set Measurements… ▸ Redirect to: C2*. `intensity_image` is Lumen's *Redirect to*.

---

## 3. Colocalization of two channels (Costes + Manders)

**Goal.** Pearson, Spearman and Manders coefficients between two channels, with Costes auto-thresholds,
threshold-based Manders (tM1/tM2), a Costes randomisation p-value and a 2-D scatter histogram.

**Sample.** Bundled **Synthetic nuclei (3 channels)** (channel 1 DAPI vs channel 2 Cytoplasm overlap
partially, Pearson ≈ 0.55). ImageJ **Fluorescent Cells** (Red vs Green, Pearson ≈ 0.40) works too.
Note that **Cells (3-D, 2 channels)** (membrane vs nuclei) is *anti*-correlated and gives a Costes warning.

**UI steps.**

1. **File ▸ Open Sample ▸ Synthetic nuclei (3 channels)**.
2. Optional: draw a rectangle (tool **1**) around the cells of interest — all coefficients are computed
   inside the selection.
3. ⌘K → **Colocalization (2 channels)**: channel A 1, channel B 2, thresholds 0 (= Otsu). One Results
   row: Pearson, Spearman, Manders M1/M2, thresholds.
4. ⌘K → **Colocalization (Costes)**: channel A 1, channel B 2, *Randomisation iterations* 100,
   *Block size* 5 px (≈ PSF width), *Scatter plot image* on. One Results row (Pearson overall / above /
   below thresholds, Spearman, tM1, tM2, Costes thresholds, regression, % pixels above, p-value, scrambled
   r mean ± SD) and a new 32-bit image *Scatter A vs B of …* (log counts, *Fire* LUT; axis ranges and the
   regression line are stored in its metadata).
5. **Results ▸** export (File ▸ Export ▸ Results as CSV, or ⌘K → **Results Table...**).

**Script.**

```python
# recipe: coloc
import os
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")
img = lumen.image
rows = lumen.run("analyze.colocalization", channel_a=1, channel_b=2)       # Otsu thresholds
print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[0].items()})
rows = lumen.run("analyze.coloc_costes", channel_a=1, channel_b=2, scramble_iterations=100,
                 block_size=5, show_scatter=True, seed=1)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[0].items()})
print([i.name for i in lumen.images])                                      # + "Scatter A vs B of …"
# restrict to a region
lumen.select(img.name)
lumen.rect(100, 100, 250, 250)
rows = lumen.run("analyze.colocalization", channel_a=1, channel_b=2)
print("in ROI:", round(rows[0]["Pearson"], 3), round(rows[0]["Manders M1"], 3))
lumen.set_roi(None)
lumen.run("file.export_results", path=f"{out}/coloc.csv", format="csv")
```

**Expected output.** `analyze.colocalization`: Pearson 0.547, Spearman 0.333, Manders M1 0.623, M2 0.192, Threshold A 1378 / B 451 (Otsu).
`analyze.coloc_costes`: Costes thresholds A 943 / B 456, Pearson above thresholds 0.007 (below −0.001), Manders tM1 0.622,
tM2 0.215, 10.5 % of the pixels above both thresholds, Costes p-value 0.0 (none of the 100 block-scrambled images reaches
the observed r; scrambled r = 0.001 ± 0.010), plus the scatter image. Inside the rectangle: Pearson 0.567, M1 0.627.

**Pitfalls.**

- Coefficients are computed on the *current* z-slice / frame (set the slice first, or Z-project).
- For anti-correlated channels the Costes threshold search fails (Pearson above thresholds = NaN, tM1 =
  tM2 = 0) and the message carries a warning — that is the expected answer, not a bug.
- `block_size` should match the PSF (in pixels); too small a block gives an optimistic p-value.
- `threshold_a` / `threshold_b` of `analyze.colocalization` are raw intensities; 0 = Otsu.

**Fiji.** *Analyze ▸ Colocalization ▸ Coloc 2* (Costes threshold, Manders, Costes significance test, 2-D
histogram) and the older *Colocalization Threshold*. Lumen does not do the Li ICQ or the
object-based methods of JACoP.

---

## 4. Track moving cells in a time-lapse and export to TrackMate

**Goal.** Detect spots in every frame (LoG), link them into tracks (LAP with gap closing), filter, get
per-track statistics, and write TrackMate XML / CSV that Fiji's TrackMate can open.

**Sample.** Bundled **Nuclei timelapse (synthetic)**: 12 frames, 15 nuclei of radius ≈ 9 px, 0.65 µm/px,
5 min per frame. ImageJ **Mitosis** (5-D) also works (track on one channel).

**UI steps.**

1. **File ▸ Open Sample ▸ Nuclei timelapse (synthetic)**. The T slider appears under the image
   (play button, **,** / **.** step frames).
2. ⌘K → **Track (detect + link)**: *Detector* LoG, *Spot radius* 9 (pixels), *Quality threshold* 0
   (automatic), *Tracker* **Simple LAP tracker** (frame-to-frame LAP linking + gap closing) or **LAP
   tracker** with *Allow track splitting* / *Allow track merging* for lineage trees (cell divisions,
   fusions), *Max linking distance* 12, *Max frame gap* 2, *Min spots per track* 3. **Apply**.
   Tracks are drawn as time-coloured tails; the **Tracks** table (one row per track: N spots, start/end
   frame, duration, net displacement, total distance, mean/max speed, mean intensity, confinement ratio)
   opens in the inspector. Step by step instead: the inspector's **Tracks** panel ▸ **Wizard** tab walks
   through TrackMate's stages (Detector → Initial threshold → Spot filters → Linker → Track filters →
   Display → Export) with feature histograms; the panel's list tab shows every track, and the display
   step's *Display mode* (full tracks, local tails, up to the current frame, fade) and *Tail length* set
   how much of each track the viewer draws around the current frame.
3. ⌘K → **Filter Tracks** (e.g. *Min spots per track* 6) to drop short tracks.
4. ⌘K → **Export Tracks (TrackMate XML)** → path `tracks.xml`; **Export Tracks (CSV)** (layout TrackMate:
   `ID, TRACK_ID, FRAME, POSITION_X …` in calibrated units).
5. ⌘K → **Tracks to ROIs** turns every track into a polyline ROI; **Kymograph** needs a line selection
   (tool **5**) and gives a (T × length) image.
6. Manual tracking: tool **X** — click adds this frame's point and advances to the next frame, ⌥-click
   deletes it (commands *Manual Tracking: New Track / Add Point / Delete Point*).
7. **Import Tracks (TrackMate XML)** loads a TrackMate file (positions in physical units are divided by
   the file's pixel size).

**Script.**

```python
# recipe: tracking
import os
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Nuclei timelapse (synthetic)")       # 12 frames, 0.65 µm, 5 min/frame
res = lumen.run("tracking.track", detector="LoG", radius=9, threshold=0, subpixel=True,
                max_displacement=12, gap_closing_frames=2, max_gap_distance=15, min_track_length=3)
print(res["message"])
tracks = res["table"]                                               # also lumen.session.tables["Tracks"]
print(tracks["columns"])
print(len(tracks["rows"]), "tracks; first:", tracks["rows"][0])
print(lumen.run("tracking.filter_tracks", min_length=6)["message"])
print(lumen.run("tracking.export_trackmate_xml", path=f"{out}/tracks.xml")["message"])
print(lumen.run("tracking.export_tracks", path=f"{out}/tracks.csv", layout="TrackMate")["message"])
print(lumen.run("tracking.tracks_to_rois", width=1.0)["message"])
lumen.set_roi({"type": "line", "x1": 10, "y1": 128, "x2": 246, "y2": 128})
ky = lumen.run("tracking.kymograph", axis="T")
print("kymograph:", ky[0].name, ky[0].shape)
lumen.select("timelapse.tif")
lumen.run("tracking.clear_tracks", forget_spots=True)
print(lumen.run("tracking.import_trackmate_xml", path=f"{out}/tracks.xml")["message"])
```

**Expected output.** `161 spots (quality ≥ 35.31); 16 tracks from 159 of 161 spots (speeds in µm/min)`
(when the linker closed gaps the bracket reads `(N gaps closed; speeds in µm/min)`, and the LAP tracker with
splitting / merging prepends `S splits, M merges`);
Tracks columns `Label, N spots, Start frame, End frame, Duration (frames), Duration (min), Net displacement, Total distance, Mean speed, Max speed, Mean intensity, Confinement ratio, Color`;
`Kept 13 of 16 tracks (3 removed)`; `Wrote 147 spots of 13 tracks to …/tracks.xml (TrackMate XML)`;
`13 track ROIs added to the ROI Manager`; kymograph `(1, 1, 1, 12, 237)`; `Imported 13 tracks (147 linked of 147 spots)`.

**Pitfalls.**

- *Spot radius* is in the chosen *units* (pixels by default; `units="calibrated"` for µm). Speeds and
  displacements in the Tracks table are calibrated when the image has a pixel size and a frame interval.
- `threshold=0` picks the quality threshold automatically (Otsu on the detector response); raise it if
  background blobs are tracked.
- On a Z stack the detector works on the current slice (`z_mode="Current slice"`) or a per-frame max
  projection (`"Max projection"`); there is no 3-D (z-aware) detection or linking.
- The TrackMate XML carries spots (with intensity / contrast / SNR features), lineage edges (splits,
  merges, gap-closing links) and the image dimensions; TrackMate's filter settings are not round-tripped.

**Fiji.** *Plugins ▸ Tracking ▸ TrackMate* (LoG detector → LAP tracker → filters → *Export tracks to XML
file* / *Export all spots statistics*), *Plugins ▸ Tracking ▸ Manual Tracking*, *Analyze ▸ Multi Kymograph*.

---

## 5. Handle a z-stack: projection, orthogonal views, 3-D view, surface export

**Goal.** Look at a volume from every side: maximum projection, XZ/YZ sections, rotating 3-D projection,
interactive volume rendering, and a mesh you can open in Blender / MeshLab / a slicer.

**Sample.** Bundled **Cells (3-D, 2 channels)**: 60 slices × 2 channels (Membrane, Nuclei), 256 × 256,
0.26 × 0.26 × 0.29 µm. ImageJ **T1 Head** and **Bat Cochlea Volume** are classic single-channel volumes.

**UI steps.**

1. **File ▸ Open Sample ▸ Cells (3-D, 2 channels)**. The Z slider appears (⇧ wheel or **,** / **.** scroll slices).
2. ⌘K → **Z Project**, *method* Max Intensity (also Min / Average / Sum / SD / Median; *start* / *stop*
   limit the range; *axis* T projects a time series). A new image *MAX_cells3d.tif* opens.
3. Back on the volume: open **Orthogonal Views** in the inspector — XZ and YZ sections follow the cursor.
   The *command* **Orthogonal Views** (x, y, *scale_z*) writes them out as two images instead.
4. ⌘K → **3D Project** (Brightest Point, Y-axis, 10° steps) renders a rotation series as a time series
   (play the T slider).
5. Open **3-D View** in the inspector: modes **MIP**, **Alpha** (translucent), **Iso** (iso-surface) and
   **Mesh** (marching cubes). Per channel: *Cut* / *Level*, *Opacity*, *Gain*, visibility. Drag to orbit,
   wheel zooms, ⇧-drag / right-drag pans, double-click resets; **Spin** auto-rotates; **Snapshot** opens
   the view as a new RGB image (for figures). In **Mesh** mode: *Wire* shows edges, the *step* and
   *smoothing* controls trade detail for speed, **Export OBJ/STL/PLY…** downloads the visible surfaces and
   **Measure** writes surface area / enclosed volume per component into Results.
6. ⌘K → **Export Surface (OBJ/STL)** does the mesh export as a command (path, format, iso *level*
   `auto` / `otsu` / a raw value, *channel*, *smooth*, *calibrated* units); **Measure Surface (3D)** gives
   the statistics without a file.
7. ⌘K → **Make Substack** (e.g. slices `20-40`) to keep part of the stack; **Reslice** rotates the volume.

**Script.**

```python
# recipe: zstack
import os
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Cells (3-D, 2 channels)")          # (Z=60, C=2) 0.26 x 0.26 x 0.29 µm
vol = lumen.image
print(vol.shape, vol.calibration.pixel_depth, vol.calibration.unit)
proj = lumen.run("stacks.z_project", method="Max Intensity")[0]
print("projection:", proj.name, proj.shape)
lumen.select(vol.name)
views = lumen.run("stacks.orthogonal_views", x=128, y=128, scale_z=True)
print("ortho:", [(v.name, v.shape) for v in views])
lumen.select(vol.name)
rot = lumen.run("stacks.project3d", method="Brightest Point", axis="Y-Axis", angle_step=30, total_rotation=360)[0]
print("3D project:", rot.name, rot.shape)
lumen.select(vol.name)
lumen.run("stacks.substack", slices="20-40", axis="Z")
print("substack:", lumen.image.name, lumen.image.shape)
lumen.select(vol.name)
vol.display.c = 1                                                  # nuclei channel (0-based)
res = lumen.run("mesh.export", path=f"{out}/nuclei.stl", format="stl", level="otsu", smooth=10, step=2)
print(res["message"])
rows = lumen.run("mesh.measure", level="otsu", step=2, smooth=10, min_faces=500)
print(len(rows), "surface components; largest:", rows[0])
```

**Expected output.** `(1, 60, 2, 256, 256) 0.29 µm`; `MAX_cells3d.tif (1, 1, 2, 256, 256)`;
ortho `XZ cells3d.tif (1, 1, 2, 67, 256)` and `YZ cells3d.tif (1, 1, 2, 256, 67)` (Z stretched by 0.29/0.26);
`Projections of cells3d.tif (12, 1, 2, 256, 265)`; `Substack (20-40) (1, 21, 2, 256, 256)`;
`Wrote 73727 vertices / 145302 faces (level 10495.8, µm) → …/nuclei.stl`; `19 surface components` (with
`min_faces=500`), the largest `{'Label': 'cells3d.tif:Nuclei #13', 'Channel': 2, 'Level': 10495.8, 'Vertices': 3802, 'Faces': 7470,
'Area': 612.9, 'Volume': 1982.2, 'Orientation': 'outward', 'Unit': 'µm', 'X': 62.4, 'Y': 34.9, 'Z': 10.3, 'BX': …, 'Width': 9.46, 'Height': 24.58, 'Depth': 8.48}`.

**Pitfalls.**

- Every Lumen image is a 5-D (T, Z, C, Y, X) hyperstack; stack commands take `axis="Z"` or `"T"`
  instead of distinguishing stacks from hyperstacks.
- The 3-D View works on a downsampled copy (resolution selector) — exports and `mesh.export` use the
  full volume unless `max_dim` is set.
- `mesh.measure` returns at most `max_rows` (200) components, largest first; `min_faces` drops specks.
- Every `mesh.measure` row carries an `Orientation` column – `outward` for a shell around bright material, `inward`
  for a cavity or dark inclusion – and the volume in its summary message is the material volume (cavities subtracted).
- `stacks.orthogonal_views` is a snapshot; the inspector section is the live, cursor-following version.

**Fiji.** *Image ▸ Stacks ▸ Z Project…*, *Image ▸ Stacks ▸ Orthogonal Views*, *Image ▸ Stacks ▸ 3D Project…*,
*Plugins ▸ 3D Viewer* / *Volume Viewer*, *File ▸ Export ▸ Wavefront .OBJ* from the 3D Viewer,
*Image ▸ Stacks ▸ Tools ▸ Make Substack…*, *Reslice*.

---

## 6. Stitch a grid of tiles

**Goal.** Assemble overlapping tiles stored as files in a folder into one image, refining the grid
positions by phase correlation and blending the seams.

**Sample.** There is no bundled tile set; the script cuts four overlapping 300 × 300 tiles (29 % overlap,
2 × 2 grid) out of the bundled **Moon** sample and writes them to `~/lumen-demo/tiles/tile_01.tif …`.
Use your own `tile_*.tif` folder the same way.

**UI steps.**

1. ⌘K → **Grid Stitching** (Plugins ▸ Registration): *Folder* = the tile folder (the server-side file
   browser lets you pick it), *Pattern* `tile_*.tif` (glob or regex), *Grid size X/Y* 2 × 2, *Tile overlap*
   29 %, *Tile order* Row-by-row (Column-by-column, Snake by rows/columns), *Refine* on, *Fusion* Linear blend
   (Max, Average, or *Overlay (channels)* to inspect seams), *Min correlation* 0.3. **Apply**.
2. The stitched image opens as a new tab; the **Tile positions** table and one Results row per neighbour
   pair (dx, dy, correlation, used) tell you which pairs were refined and which fell back to the grid.
3. For two open images use ⌘K → **Pairwise Stitching** (image1, image2, fusion); **Phase Correlation**
   only reports the shift between two open images.

**Script.**

```python
# recipe: stitch
import os
import numpy as np
out = os.path.expanduser("~/lumen-demo"); tiles = f"{out}/tiles"; os.makedirs(tiles, exist_ok=True)
lumen.run("file.sample", name="Moon")                  # 512 x 512
full = lumen.plane
size, step = 300, 212                                  # 300-px tiles, 88 px (29 %) overlap, 2 x 2 grid
k = 0
for gy in range(2):
    for gx in range(2):
        k += 1
        lumen.new_image(full[gy * step: gy * step + size, gx * step: gx * step + size], name=f"tile_{k:02d}")
        lumen.save(f"{tiles}/tile_{k:02d}.tif")
        lumen.close()
rows = lumen.run("registration.grid_stitch", folder=tiles, pattern="tile_*.tif", grid_x=2, grid_y=2,
                 overlap_percent=29, order="Row-by-row", refine=True, fusion="Linear blend", title="moon_stitched")
print("pairs:", rows)                                  # one Results row per neighbour pair
st = lumen.image                                       # the stitched image is the new active image
print(st.name, st.shape, list(lumen.session.tables))
lumen.open(f"{tiles}/tile_01.tif"); lumen.open(f"{tiles}/tile_02.tif")
pair = lumen.run("registration.pairwise_stitch", image1="tile_01.tif", image2="tile_02.tif", fusion="Linear blend", title="pair")
print(pair[0], lumen.image.name, lumen.image.shape)
```

**Expected output.** Four rows such as `{'Label': 'tile_01.tif → tile_02.tif', 'Tile A': 1, 'Tile B': 2, 'dx': 212, 'dy': 0, 'Correlation': 1.0, 'Used': True}`;
`moon_stitched (1, 1, 1, 512, 512)` (the original reassembled exactly); table `Tile positions`;
pairwise `{'Label': 'tile_02.tif → tile_01.tif', 'dx': 212, 'dy': 0, 'Correlation': 1.0}` and `pair (1, 1, 1, 300, 512)`.

**Pitfalls.**

- Tiles are taken in *natural sort order* of the file names: `tile_1, tile_2 … tile_10` sorts correctly,
  but mixed naming schemes do not. The pattern can be a glob (`*.tif`) or a regular expression.
- The overlap you give is only the starting guess; with `refine=True` every neighbour pair is corrected
  by phase correlation unless its correlation is below `min_correlation` (then the grid position is kept).
- Featureless overlaps (empty background) refine badly: lower `min_correlation` or turn `refine` off.
- `title` names the stitched image (default: the folder name); the Results rows (one per neighbour pair)
  and the **Tile positions** table are the record of what was done — export them with **Results Table...**.

**Fiji.** *Plugins ▸ Stitching ▸ Grid/Collection stitching* (type *Grid: row-by-row*, overlap, *compute
overlap*, *Linear Blending*) and *Pairwise stitching*.

---

## 7. Batch-process a folder with a recorded macro

**Goal.** Apply the same commands to every image in a folder, save the processed images, and collect
the measurements of all files in one CSV.

**Sample.** The script writes three bundled samples (Blobs, Coins, Human mitosis) into
`~/lumen-demo/batch_in/` and processes that folder.

**UI steps.**

1. Open one representative image and apply your commands normally (e.g. **Median** radius 2, **Apply
   Threshold → Mask** Otsu, **Analyze Particles**). Every applied command is appended to the
   **Script & Recorder** section as a `lumen.run(…)` line — the recorder is always on.
2. ⌘K → **Apply Recorded Macro to Folder...** (Plugins ▸ Batch): *Input folder*, *File pattern*
   (regex, default matches tif/png/jpg/nd2/czi/lif), *Include subfolders*, *Output folder* (default
   `<input>/lumen_output`), *Output format* (tif, ome.tif, png, jpg, npy or *none*), *Save Results table*,
   *Close images after each file*, *Continue on error*, *Max files*. The recorded lines that open, save
   or close files are stripped automatically; the rest runs on every file.
3. Or ⌘K → **Process Folder...** and paste / edit the script in the *Script* box (`lumen.image` is the
   file being processed). **Run Script on All Open Images...** runs a script once per open tab.
4. The **Batch** table lists every file with status, time, output name and the last printed line;
   `Results.csv` in the output folder has every measurement with a *File* column; the rows are also
   appended to the session Results.
5. ⌘K → **Batch Measure...** (Process ▸ Batch) measures every file with the current **Set Measurements**
   after an optional auto-threshold, without a script.
6. To reuse a script later: **Script & Recorder ▸ → editor**, edit, and keep it as a `.py` file; or turn it
   into a command (see [SCRIPTING.md ▸ Plugins](SCRIPTING.md#writing-op-plugins)).

**Script.**

```python
# recipe: batch
import os
out = os.path.expanduser("~/lumen-demo"); folder = f"{out}/batch_in"; os.makedirs(folder, exist_ok=True)
for name in ("Blobs", "Coins", "Human mitosis"):               # make a folder of test images
    lumen.run("file.sample", name=name)
    lumen.save(f"{folder}/{name.replace(' ', '_').lower()}.tif")
    lumen.close()
script = '''
lumen.run("filters.gaussian", sigma=2)
lumen.run("threshold.apply", method="Otsu")
lumen.run("analyze.particles", min_size=30, summarize=False, add_to_manager=False)
'''
res = lumen.run("batch.process_folder", input_folder=folder, pattern=r"\.tif$", output_folder=f"{folder}/out",
                output_format="png", script=script, save_results=True, close_after=True)
print(res["message"])
print(res["table"]["rows"])
print(sorted(os.listdir(f"{folder}/out")))
print(open(f"{folder}/out/Results.csv").readline().strip())
# the recorder: commands applied in the UI are recorded as lumen.run(...) lines (simulated here);
# commands run from a script are not recorded
lumen.run("file.sample", name="Blobs")
lumen.session.run_op("filters.median", {"radius": 2})
lumen.session.run_op("process.enhance_contrast", {"saturated": 0.35})
print("recorder:", lumen.session.macro[-2:])
res = lumen.run("batch.apply_recorded_macro", input_folder=folder, pattern=r"\.tif$", output_folder=f"{folder}/out2", output_format="tif")
print(res["message"])
rows = lumen.run("batch.measure", input_folder=folder, pattern=r"\.tif$", threshold="Otsu", dark_background=True)
print(rows[0])
```

**Expected output.** `Batch: 3 ok, 0 failed of 3 file(s) in 0.1 s → …/batch_in/out`;
rows `['blobs.tif', 'ok', 0.05, 'blobs.png', '']` …; files `Results.csv, blobs.png, coins.png, human_mitosis.png`;
CSV header `File,Label,Area,Mean,Min,Max`; recorder lines
`lumen.run("filters.median", radius=2.0, shape='disk')`, `lumen.run("process.enhance_contrast", saturated=0.35, …)`;
`Macro batch: 3 ok, 0 failed …`; batch-measure row `{'File': 'blobs.tif', 'Label': 'blobs.tif', 'Area': 17017.0, 'Mean': 185.6, …}`.

**Pitfalls.**

- The file pattern is a **regular expression** (`\.tif$`), not a glob; a plain glob such as `*.tif` is
  translated for you, but `*` alone is not.
- The image that is *saved* is whatever the script leaves active, provided the script created it (a
  projection, a mask …); otherwise the opened file itself. If the script closes everything, nothing is saved.
- `output_format="none"` runs the script without writing images (measure-only batches).
- The output folder is skipped when it lies inside the input folder, so re-running a batch never
  processes its own output.
- Recorder lines run by a batch are *not* re-recorded; the session recorder is restored afterwards.

**Fiji.** *Process ▸ Batch ▸ Macro…* (paste the recorded macro, choose input/output/format),
*Process ▸ Batch ▸ Measure…*, or a macro with `getFileList` + `open` + `saveAs` in a loop.

---

## 8. Manage ROIs: manager, combine, ImageJ .roi/.zip interchange

**Goal.** Collect selections in the ROI Manager, name and recolour them, combine them (OR/AND/XOR),
measure them, convert to and from label images, and exchange them with Fiji as `.roi` / `.zip` files.

**Sample.** Bundled **Blobs** (8-bit, 256 × 256).

**UI steps.**

1. **File ▸ Open Sample ▸ Blobs**; **Threshold ▸ Otsu ▸ Apply → mask**; ⌘K → **Analyze Particles**
   (*Add to manager* on): one polygon ROI per blob in **ROI Manager**.
2. Draw a selection (tools **1** rectangle, **2** oval, **3** polygon, **4** freehand, **5** line, **6**
   segmented line, **7** points, **8** wand) and press **T** (**Add**) to add it.
3. Select rows in the manager (click, ⇧/⌘-click): **Measure** measures the selected (or all) ROIs on the
   plane each was made on, **Delete** removes them, **Show all** draws all on the image, **OR** / **AND** /
   **XOR** turn the selected ROIs into one combined selection, **To overlay** burns them into the overlay,
   **Import** / the download icon exchange Lumen's JSON format, **Clear** empties the manager.
   Double-click a name to rename; ⌘K → **Properties…** (**Y** with a selection) sets name, colour,
   stroke width and plane position.
4. **File ▸ Import ▸ ImageJ ROIs (.roi/.zip)...** loads a Fiji ROI Manager zip or a single `.roi`;
   **File ▸ Export ▸ ImageJ ROIs (.roi/.zip)...** writes the manager (zip) or the current selection (`.roi`),
   byte-compatible with ImageJ.
5. Selection commands (Edit ▸ Selection): **Enlarge**, **Make Band**, **Convex Hull**, **Fit Ellipse /
   Circle / Rectangle / Spline**, **Make Inverse**, **Create Mask**, **Create Selection** (from a mask),
   **Rotate / Scale Selection**, **Line to Area**, **Specify**.
6. ⌘K → **ROIs to Labels** rasterises the manager into a label image; **Labels to ROIs** does the reverse
   (names `L<label>`); **Multi Measure (ROI Manager)** measures every ROI on every slice/frame.

**Script.**

```python
# recipe: rois
import os
from lumen.core import roi as roimod
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Blobs")
img = lumen.image
lumen.run("threshold.apply", method="Otsu")
lumen.run("analyze.particles", min_size=30, add_to_manager=True)
print(len(lumen.rois), "ROIs;", lumen.rois[0]["name"], lumen.rois[0]["type"])
lumen.rect(10, 10, 60, 40)                                # a selection (the current ROI of the script)
lumen.run("selection.add_to_manager")                     # = pressing T
lumen.add_roi({"type": "oval", "x": 100, "y": 100, "w": 40, "h": 30, "name": "my oval"})
lumen.session.roi_manager.update(lumen.rois[-1]["id"], {"name": "nucleus 7", "color": "#ff0000"})
lumen.run("file.export_ij_rois", path=f"{out}/blobs_rois.zip", which="ROI Manager")   # for Fiji
lumen.session.roi_manager.clear()
print(lumen.run("file.import_ij_rois", path=f"{out}/blobs_rois.zip", clear_first=True)["message"])
print(len(lumen.rois), lumen.rois[-1]["name"])
lumen.run("file.export_rois_json", path=f"{out}/blobs_rois.json")                      # Lumen's own format
union = roimod.combine(lumen.rois[:5], (img.height, img.width), "or")                  # "or" | "and" | "xor"
lumen.set_roi(union)
print(lumen.run("analyze.measure")[0])
lumen.set_roi(None)
lab = lumen.run("segment.rois_to_labels")[0]
print(lab.name, int(lab.data.max()), "labels")
lumen.select(img.name)
lumen.run("overlay.from_manager")
lumen.run("analyze.multi_measure", one_row_per_slice=False)
print(len(lumen.results), "result rows")
lumen.set_roi(lumen.rois[0])
lumen.run("file.export_ij_rois", path=f"{out}/one.roi", which="Current selection")
```

**Expected output.** `55 ROIs; 0001-5-96 polygon`; after import `Imported 57 ROI(s)` and `57 nucleus 7`
(names and colours survive the ImageJ round trip); union `{'Label': 'OR', 'Area': 1583.0, 'Mean': 254.8, …}`;
`Labels from ROIs (blobs.tif) 57 labels`; `57 result rows`; files `blobs_rois.json`, `blobs_rois.zip`, `one.roi`.

**Pitfalls.**

- ROI dicts: `{"type": "rect"|"oval", x, y, w, h}`, `{"type": "polygon"|"freehand"|"polyline"|"point",
  "points": [[x, y], …]}`, `{"type": "line", x1, y1, x2, y2, "width"}`; every manager entry also has
  `id`, `name`, `image_id` and `position` (t, z, c). A combined selection has `type: "mask"`.
- ROI names starting with a class name are what the pixel classifier uses as training labels (recipe 12).
- Adding a ROI that came from the manager (same `id`) always creates a new, independent entry.
- `file.export_ij_rois` writes a `.zip` for the manager (`which="ROI Manager"`) and a single `.roi` for the
  current selection (`which="Current selection"`); the ROI Manager's own **Import** / download buttons use
  Lumen's JSON format (`file.import_rois_json` / `file.export_rois_json`).

**Fiji.** *Analyze ▸ Tools ▸ ROI Manager…* (Add [t], Measure, Combine, AND, XOR, Rename, Properties,
More ▸ Save / Open, Multi Measure, Labels), *Edit ▸ Selection ▸ …*, LabelMap ↔ ROI from MorphoLibJ or
the *ROI Manager ▸ More ▸ Labels* trick.

---

## 9. Calibrate and measure lengths and areas

**Goal.** Give the image a pixel size (from the metadata, by hand, or from a line of known length) and
measure areas, perimeters, lengths and angles in µm.

**Sample.** Bundled **Blobs** (uncalibrated). Microscope files (ND2, CZI, LIF, OME-TIFF …) arrive
calibrated; check **Image Info**.

**UI steps.**

1. **File ▸ Open Sample ▸ Blobs**. **Image Info** shows *pixel* as the unit.
2. ⌘K → **Properties / Set Scale**: *pixel width* 0.5, *pixel height* 0.5, *unit* µm (also voxel depth,
   frame interval and time unit for stacks and time series). **Apply**.
3. Or draw a line (tool **5**) along a scale bar / stage micrometer and ⌘K → **Set Scale from Line**:
   *known distance* 100, *unit* µm.
4. ⌘K → **Set Measurements**: tick Area, Mean, Perimeter, Feret, Shape descriptors, Centroid; *decimals* 3.
5. Draw a rectangle / oval / polygon and press **M** (**Measure**): Area in µm², Perimeter in µm, Feret
   diameters in µm, centroid in µm. A line selection gives *Length* (and *Angle*); a segmented line with
   three points plus ⌘K → **Angle** gives the angle at the vertex (tool **Q** draws an angle directly).
6. **Results ▸ Columns** reopens Set Measurements; **File ▸ Export ▸ Results as CSV** or ⌘K →
   **Results Table...** (csv / tsv / xlsx) saves the table.

**Script.**

```python
# recipe: calibrate
import os
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Blobs")
img = lumen.image
print("before:", img.calibration.pixel_width, img.calibration.unit)
lumen.run("image.properties", pixel_width=0.5, pixel_height=0.5, voxel_depth=1.0, unit="µm")   # a) by hand
print("after properties:", img.calibration.pixel_width, img.calibration.unit)
lumen.set_roi({"type": "line", "x1": 10, "y1": 20, "x2": 210, "y2": 20})                      # b) 200 px = 100 µm
print(lumen.run("image.set_scale_from_line", known_distance=100, unit="µm")["message"])
lumen.run("analyze.set_measurements", measurements=["area", "mean", "perimeter", "feret", "shape_descriptors", "centroid"], decimals=3)
lumen.rect(20, 20, 100, 60)                                   # 100 x 60 px = 50 x 30 µm
print("rect:", lumen.run("analyze.measure")[0])
lumen.set_roi({"type": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 0})                          # 100 px = 50 µm
print("line:", lumen.run("analyze.measure")[0])
lumen.set_roi({"type": "polyline", "points": [[0, 0], [30, 40], [60, 0]]})
print("angle:", lumen.run("analyze.angle")[0])
lumen.set_roi(None)
lumen.run("file.export_results", path=f"{out}/measurements.csv")
```

**Expected output.** `Scale: 2 px/µm`; rect row `Area 1500.0, Perim. 160.0, Feret 58.31, MinFeret 30.0, X 35.0, Y 25.0, Circ. 0.736, AR 1.667, Solidity 1.0`;
line row `Length 50.0, Angle 0.0`; angle row `Angle 73.74`.

**Pitfalls.**

- Calibration is per image; **Duplicate**, projections and crops inherit it, `lumen.new_image(..., like=img)`
  copies it to an array you made yourself.
- `measurements` ids: `area, mean, std, mode, min, max, centroid, center_of_mass, perimeter, bounding_rect,
  fit_ellipse, shape_descriptors, feret, integrated_density, median, skewness, kurtosis, area_fraction,
  stack_position`.
- Measurements of a mask (0/255) report Mean 255; measure the original image or use `intensity_image`.
- *Limit to threshold* (`limit_to_threshold=True` in Set Measurements) measures only the thresholded
  pixels of the selection, like ImageJ.

**Fiji.** *Analyze ▸ Set Scale…*, *Image ▸ Properties…*, *Analyze ▸ Set Measurements…*, *Analyze ▸ Measure*
[m], the angle tool, *File ▸ Save As ▸ Results…*.

---

## 10. Make a figure: LUTs, composite, scale bar, calibration bar, annotations, flatten, export

**Goal.** Turn a multi-channel image into a publication panel: per-channel colours, composite view,
scale bar, intensity calibration bar, a label and an arrow, then flatten to RGB and export PNG — and keep
the editable hyperstack as TIFF.

**Sample.** Bundled **Synthetic nuclei (3 channels)** (calibrated, so the scale bar is in µm).

**UI steps.**

1. **File ▸ Open Sample ▸ Synthetic nuclei (3 channels)**.
2. **Display**: per channel pick the LUT — ImageJ's (Grays, Red, Green, Blue, Cyan, Magenta, Yellow, Fire, Ice,
   Spectrum, 3-3-2 RGB, Red/Green, HiLo, 16 colors, Glasbey, Phase …), Fiji's (Red Hot, Green Fire Blue, Royal,
   Sepia, Thallium, ICA, physics, gem, glow, smart …) and the perceptual maps (Viridis, Magma, Inferno, Plasma,
   Cividis, Turbo …); `lumen.session.state()["lut_names"]` lists all 60 — drag the histogram handles or type
   min/max, **Auto** (0.35 % saturation), **Inv** inverts the LUT, the eye hides a channel; **Composite** /
   **Color** / **Gray** set the mode. Display settings never change pixel values.
3. ⌘K → **Scale Bar** (*length* in calibrated units, 0 = automatic; thickness, position, colour, label, font
   size), **Calibration Bar** (position, LUT, size, decimals, ticks), **Add Text** (tool **Y** places one by
   click), **Add Arrow** (tool **J** drags one; the command needs a line selection), **Time Stamp** for movies.
   All of these are overlay items: ⌘K → **List Overlay Elements**, **Remove Overlay**, or **Overlay to ROI
   Manager** to edit them.
4. ⌘K → **Flatten**: a new RGB image with the composite rendering and the overlay burnt in (**Burn In
   Overlay** burns into the pixels of the current image type instead; **Bake LUT into RGB** / **RGB Color**
   convert the current channel / composite to RGB without the overlay).
5. **File ▸ Export ▸ PNG (current view)** downloads exactly what the viewport shows; **Save As…** (⇧⌘S)
   with `.png` / `.jpg` writes the rendered RGB, with `.tif` the full hyperstack (channels, calibration,
   LUTs, display ranges and overlay in ImageJ-compatible metadata), with `.ome.tif` OME-TIFF.
6. For separate single-channel images, ⌘K → **Merge Channels** (C1 red, C2 green, C3 blue, C4 gray, C5 cyan,
   C6 magenta, C7 yellow, *create composite*).

**Script.**

```python
# recipe: figure
import os
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")
img = lumen.image
for c, lut in enumerate(["Cyan", "Magenta", "Yellow"]):
    img.display.c = c
    lumen.run("color.lut", lut=lut)                           # or color.lut(lut=..., all_channels=True)
img.display.mode = "composite"
img.display.channels[1].max = 1500                            # display range of channel 2
lumen.run("overlay.scale_bar", length=20, thickness=6, position="Lower Right", color="#ffffff", show_label=True, font_size=14)
lumen.run("overlay.text", text="DAPI / Cytoplasm / Spots", x=10, y=10, font_size=18, color="#ffffff", at_selection=False)
lumen.set_roi({"type": "line", "x1": 300, "y1": 300, "x2": 260, "y2": 260})
lumen.run("overlay.arrow", color="#ffffff", stroke_width=2, head_size=10)
lumen.set_roi(None)
img.display.c = 0
lumen.run("overlay.calibration_bar", position="Upper Right", lut="Current", width=16, height=100, font_size=10, decimals=0)
print("overlay items:", [o["type"] for o in img.overlay])
flat = lumen.run("overlay.flatten")[0]                         # RGB with the overlay burnt in
print(flat.name, flat.shape, flat.rgb)
print(lumen.save(f"{out}/figure.png"))
lumen.select(img.name)
print(lumen.save(f"{out}/figure.tif"))                         # editable hyperstack (LUTs, ranges, overlay)
lumen.run("color.split")
merged = lumen.run("color.merge", c1="C2-synthetic_cells.tif", c3="C1-synthetic_cells.tif", create_composite=True)[0]
print(merged.name, merged.shape)
```

**Expected output.** `overlay items: ['scalebar', 'text', 'arrow', 'calibration_bar']`;
`synthetic_cells.tif (flat) (1, 1, 3, 512, 512) True`; `…/figure.png` and `…/figure.tif`; `Composite (1, 1, 2, 512, 512)`.

**Pitfalls.**

- `overlay.scale_bar` needs a calibrated image for a µm label; on an uncalibrated image the length is in pixels.
- The calibration bar shows the LUT and display range of the *current* channel (`lut="Current"`).
- Flatten renders with the current display settings (mode, visible channels, min/max, gamma); set them
  first. Saving a multi-channel image as PNG/JPEG writes the same rendering.
- `overlay.text` places the text at the selection's bounding box when a selection exists and
  `at_selection=True` (the default); pass `at_selection=False` to use `x`, `y`.

**Fiji.** *Image ▸ Lookup Tables*, *Image ▸ Color ▸ Channels Tool… / Merge Channels…*, *Analyze ▸ Tools ▸ Scale Bar… /
Calibration Bar…*, the text and arrow tools + *Image ▸ Overlay ▸ Add Selection…*, *Image ▸ Overlay ▸ Flatten*, *File ▸ Save As ▸ PNG / Tiff*.

---

## 11. Run an existing ImageJ macro in Lumen

**Goal.** Reuse `.ijm` macro code: `run("…", "options")` calls, `makeRectangle`, `setAutoThreshold`,
`roiManager`, `nResults` / `getResult`, loops and functions — from the UI or from Python.

**Sample.** Any; the example uses `run("Blobs (25K)")` (ImageJ sample names are mapped to Lumen's samples).

**UI steps.**

1. ⌘K → **Run ImageJ Macro** (Plugins ▸ Scripting): paste the macro into the text box, **Apply**. Output
   of `print()` goes to the **Log** section; `run("Measure")` rows go to **Results**, `roiManager("add")`
   to the **ROI Manager**; the macro's final selection becomes the current selection.
2. ⌘K → **ImageJ Command List** shows the table of every supported `run("…")` command, the Lumen command
   it maps to and how the option keys are translated (also in [FIJI_PARITY.md](FIJI_PARITY.md#imagej-macro-compatibility-layer)).
3. In **Script & Recorder** the macro functions are available directly in Python: `run("Invert")`,
   `makeRectangle(10, 10, 50, 50)`, `nResults()`, `getResult("Mean", 0)`, `IJ.run(cmd, options)`,
   `IJ.macro(text)` (returns the Log text), `IJ.transpile(text)` (shows the generated Python).

**Script.**

```python
# recipe: ij_macro
macro = '''
run("Blobs (25K)");
run("Gaussian Blur...", "sigma=2");
setAutoThreshold("Otsu dark");
run("Convert to Mask");
run("Analyze Particles...", "size=20-Infinity display clear summarize add");
print("particles: " + nResults);
n = roiManager("count");
for (i = 0; i < 3; i++) {
    roiManager("select", i);
    run("Measure");
}
print("ROIs: " + n + ", mean of row 0: " + getResult("Mean", 0));
'''
res = lumen.run("script.run_ij_macro", macro=macro)         # what the Run ImageJ Macro command does
print(res["message"]); print(res["log"])
print(lumen.ij.macro('run("Blobs (25K)"); print(getTitle() + " " + getWidth() + "x" + getHeight());'))
lumen.ij.run("Gaussian Blur...", "sigma=1")                  # IJ.run from Python
lumen.ij.run("Enhance Contrast...", "saturated=0.35")
run("Invert")                                                # macro functions injected into the console
makeRectangle(10, 10, 50, 50)
run("Measure")
print("nResults:", nResults(), "mean:", getResult("Mean", nResults() - 1))
print(IJ.transpile('for (i = 0; i < 2; i++) { print("i=" + i); }'))
try:
    lumen.ij.run("Bio-Formats Importer", "open=x.czi")       # unsupported commands fail loudly
except Exception as e:
    print("error:", e)
try:
    lumen.ij.macro('waitForUser("click");')
except Exception as e:
    print("error:", e)
```

**Expected output.** `Macro finished – 2 line(s) of output`, Log `particles: 54` and `ROIs: 54, mean of row 0: 255`;
`blobs.tif-1 256x256`; `nResults: 58 mean: 195.57`; the transpiled `while` loop;
`error: ImageJ command 'Bio-Formats Importer' is not supported by Lumen`;
`error: Unsupported macro function or undefined variable: waitForUser`.

**What is supported.** The ImageJ-1 macro *language* (variables, `if`/`for`/`while`, user functions,
arrays and the `Array.*`, `Math.*`, `String.*`, `File.*`, `List.*`, `Fit.*`, `Stack.*`, `Roi.*`, `Overlay.*`,
`Table.*`, `Color.*`, `Property.*` namespaces, `eval`, `runMacro(path, arg)` / `getArgument`, `exit`),
171 macro functions (selection, threshold, ROI manager, results, image info, pixels, strings, maths,
dialogs that return their defaults) and ~190 `run()` command rows covering File, Edit, Image, Process,
Analyze, the sample images and common Fiji plugins (CLAHE, Auto Threshold, Auto Local Threshold,
Time Stamper, Gel Analyzer …). Option strings are parsed like ImageJ's (`key=value`, `[bracketed]`,
`"quoted"`, bare flags, `size=20-Infinity` ranges, the `stack` flag, `channels=`/`slices=`/`frames=` on
hyperstacks). `run("<lumen.command.id>", "param=value")` also works, so a macro can call any Lumen command.

**What is not.** Commands with no Lumen equivalent stop the macro with *not supported* (Bio-Formats
Importer; a handful listed in FIJI_PARITY.md); every window-management and preferences dialog
(Brightness/Contrast…, Tile, Cascade, Memory & Threads…, Record… and ~35 more) is accepted as a no-op;
nothing can wait for a user: `waitForUser` is unsupported, `Dialog.*`, `getNumber`, `getString` and
`getBoolean` return their defaults without showing anything, `showMessage` prints its text to the Log,
`showStatus` / `showProgress` / `setBatchMode` are no-ops, and there is no file chooser — pass paths
explicitly; JavaScript, Jython, Groovy
and BeanShell scripts are not macros and are not translated; Java plugins (`call("…")`) only for the few
`ij.Prefs`/`ij.IJ` calls the layer knows. Unknown functions raise *Unsupported macro function or
undefined variable*, and macro constructs the transpiler cannot handle raise *Unsupported macro construct*
— nothing is skipped silently.

**Fiji.** *Plugins ▸ Macros ▸ Run…* / the Script Editor with language IJ1 Macro.

---

## 12. Train a pixel classifier (Weka-style)

**Goal.** Teach a random forest to separate classes from a few hand-drawn training regions, get
probability maps and a label image, save the model and apply it to other images.

**Sample.** Bundled **Synthetic nuclei (3 channels)** (train on channel 1). ImageJ **Fluorescent Cells** or
**Neuron** work the same way.

**UI steps.**

1. Open the image, select the channel to train on in **Display** (or choose *All channels* later).
2. Draw training regions with the brush-like tools (**4** freehand, **2** oval, **1** rectangle) and press
   **T** after each. Name them after their class: double-click the name in the **ROI Manager** and type
   `background 1`, `background 2`, `nuclei 1`, … — a ROI whose name *starts with* a class name is a training
   region for that class (`background-2`, `nuclei_3` work too).
3. ⌘K → **Train Pixel Classifier** (Plugins ▸ Segmentation): *classes* `background,nuclei` (first class =
   label 0), *channels* Current channel / All channels, *features* (Gaussian, Difference of Gaussians, Sobel,
   Laplacian, Hessian, Variance, Entropy, Membrane projections), *max sigma* (scales 1, 2, 4 … up to it),
   *number of trees*, *max depth*, *apply now*, *output* Probabilities / Labels / Both. **Apply**.
4. One Results row reports training pixels per class and the out-of-bag accuracy; new images
   *Probabilities of …* (one channel per class) and *Classification of …* (label image, 0 = first class) open.
5. Add or fix training ROIs and train again until the classification is right; ⌘K → **Save Classifier**
   (`.joblib`), later **Load Classifier** and **Apply Pixel Classifier** (*smooth* the probabilities, *mask
   class* + *probability threshold* for an extra binary mask, *all planes* for stacks) on any image.
6. ⌘K → **Labels to ROIs** on the classification image turns objects into ROIs; **Filter Labels** /
   **Analyze Particles** (on the mask) measure them.

**Script.**

```python
# recipe: classifier
import os
import numpy as np
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")
img = lumen.image
img.display.c = 0                                                     # DAPI
p = lumen.plane
lumen.add_roi({"type": "rect", "x": 2, "y": 2, "w": 30, "h": 30, "name": "background 1"})      # name = class name
lumen.add_roi({"type": "rect", "x": 470, "y": 470, "w": 30, "h": 30, "name": "background 2"})
ys, xs = np.where(p > np.percentile(p, 99.5))                         # a few bright nucleus pixels
for k in range(3):
    y, x = int(ys[k * len(ys) // 3]), int(xs[k * len(xs) // 3])
    lumen.add_roi({"type": "oval", "x": x - 3, "y": y - 3, "w": 6, "h": 6, "name": f"nuclei {k + 1}"})
rows = lumen.run("segment.train_classifier", classes="background,nuclei", channels="Current channel",
                 features=["Gaussian", "Difference of Gaussians", "Sobel", "Hessian"], max_sigma=4,
                 n_trees=50, apply_now=True, output="Both")
print(rows[0])                                                        # training statistics
print([i.name for i in lumen.images])
lumen.select(img.name)
print(lumen.run("segment.save_classifier", path=f"{out}/nuclei.joblib")["message"])
print(lumen.run("segment.load_classifier", path=f"{out}/nuclei.joblib")["message"])
lumen.select(img.name)
labels = lumen.run("segment.apply_classifier", output="Labels", smooth=1.0)[0]
print(labels.name, labels.dtype, np.unique(labels.data), int((labels.data == 1).sum()), "nucleus pixels")
lumen.select(labels.name)
print(lumen.run("segment.labels_to_rois", min_size=20, clear_manager=True)["message"])
```

**Expected output.** `{'Label': 'Classifier (synthetic_cells.tif)', 'Classes': 2, 'Features': 16, 'Trees': 50, 'Training pixels': 1864, 'OOB accuracy': 0.92, "Pixels 'background'": 1768, "Pixels 'nuclei'": 96}`;
images `Probabilities of synthetic_cells.tif`, `Classification of synthetic_cells.tif`;
`Saved classifier (background, nuclei) to …/nuclei.joblib`; `Loaded classifier (background, nuclei; 16 features, …)`;
a uint8 label image with values `[0 1]` and ≈ 18 700 nucleus pixels; the classification is one connected
blob of touching nuclei here (`1 ROIs added`) — apply **Watershed** to the mask before Labels to ROIs for
single nuclei.

**Pitfalls.**

- Training ROIs must belong to the image you train on (ROIs carry an `image_id`); on a stack they are
  taken from the plane stored in their position.
- Few training pixels (here 96 nucleus pixels) give a noisy classifier; draw freehand regions *inside* the
  objects and the background, not boxes that straddle both.
- `channels="All channels"` computes the features on every channel — more informative, slower; apply such
  a classifier to images with the same channel layout (the **Load Classifier** message lists its classes,
  feature count and training image).
- The classifier lives in the session until **Save Classifier**; restarting Lumen forgets it.

**Fiji.** *Plugins ▸ Segmentation ▸ Trainable Weka Segmentation* (add traces to classes, *Train classifier*,
*Create result*, *Get probability*, *Save classifier* / *Load classifier*, *Apply classifier*). Lumen's
trainer has no trace editor, no 3-D features and uses scikit-learn's random forest instead of Weka's FastRandomForest.

---

## 13. Analyze skeletons (neurites)

**Goal.** Segment a filamentous structure, thin it to a one-pixel skeleton, prune spurs and get per-skeleton
branch / junction / end-point counts, branch lengths and a Branches table.

**Sample.** ImageJ **Neuron** (rat hippocampal neuron, 5 channels, 16-bit, 0.16 µm/px); the script uses
channel 5, the densest neurite staining. Any binary skeleton (e.g. **Blobs** after *Skeletonize*) works for
the measurement step.

**UI steps.**

1. **File ▸ Open Sample ▸ Neuron**; in **Display** make channel 5 current; ⌘K → **Duplicate**
   (*what: Current plane*) to work on a single-channel copy.
2. ⌘K → **Subtract Background** (rolling ball radius 50), **Gaussian Blur** (sigma 1). For thin, faint
   processes **Ridge / Vessel Filter** (Frangi) before thresholding helps; for bright, thick ones it is not needed.
3. **Threshold ▸ Li** (or Triangle — both suit sparse foreground) ▸ **Apply → mask**; ⌘K → **Remove Small
   Objects** (min size 200).
4. ⌘K → **Analyze Skeleton** (Analyze): *Skeletonize first* on, *Prune end branches shorter than* 1 µm,
   *Prune cycles* none / shortest branch, *Show* Tagged skeleton (end points 1, slabs 2, junctions 3) or
   Longest branch mask. **Apply**.
5. **Results** gets one row per skeleton (`# Branches`, `# Junctions`, `# End-point voxels`, `# Junction
   voxels`, `# Slab voxels`, `Average Branch Length`, `# Triple points`, `# Quadruple points`, `Maximum Branch
   Length`, `Longest Shortest Path`); the **Branches** table lists every branch with its length, end points
   and Euclidean distance; the tagged skeleton opens as a label image.

**Script.**

```python
# recipe: skeleton
lumen.run("file.open_imagej_sample", name="Neuron")               # 5 channels, 0.16 µm
img = lumen.image
img.display.c = 4                                                  # channel 5 (0-based)
lumen.run("image.duplicate", title="neurites", what="Current plane")
lumen.run("process.subtract_background", radius=50)
lumen.run("filters.gaussian", sigma=1)
lumen.run("threshold.apply", method="Li")
lumen.run("binary.remove_small", min_size=200)
rows = lumen.run("analyze.skeleton", skeletonize_first=True, prune_ends=1.0, prune_cycles="none", show="Tagged skeleton")
rows = sorted(rows, key=lambda r: -r["# Branches"])
print(len(rows), "skeletons; largest:", {k: round(v, 2) if isinstance(v, float) else v for k, v in rows[0].items()})
branches = lumen.session.tables["Branches"]
print(branches["columns"], len(branches["rows"]), branches["rows"][0])
print([i.name for i in lumen.images][-1])
```

**Expected output.** `10 skeletons`; the largest (the dendritic arbor) `{'Label': 'neurites#1', 'Skeleton ID': 1, '# Branches': 1057,
'# Junctions': 605, '# End-point voxels': 108, '# Junction voxels': 2442, '# Slab voxels': 7589, 'Average Branch Length': 1.53,
'# Triple points': 454, '# Quadruple points': 116, 'Maximum Branch Length': 9.5, 'Longest Shortest Path': 132.28}` (µm);
a `Branches` table with columns `Skeleton ID, Branch, Length, Euclidean distance, V1 x, V1 y, V2 x, V2 y` and 1169 rows;
the label image *Tagged skeleton of neurites*.

**Pitfalls.**

- Lengths are calibrated (µm) when the image is; `prune_ends` is in the same unit.
- A mask that is not 0/255 is thresholded with the current threshold first; a float Frangi response must
  be thresholded explicitly (Otsu on a Frangi output keeps very few pixels — use Li / Triangle, or set the
  threshold by hand in the Threshold panel).
- Loops (cycles) inflate branch counts; `prune_cycles="shortest branch"` breaks each cycle like Fiji's
  *shortest branch* option. The longest-shortest-path search is skipped for skeletons with more than
  2000 vertices.
- 2-D only (per plane; `all_planes=True` processes a stack plane by plane); no 3-D skeleton graph.

**Fiji.** *Process ▸ Binary ▸ Skeletonize*, *Plugins ▸ Skeleton ▸ Analyze Skeleton (2D/3D)* (prune cycle
method, *prune ends*, *calculate largest shortest path*, *show detailed info*), *Plugins ▸ Analyze ▸ Ridge
Detection* / *Frangi Vesselness* for the segmentation.

---

## 14. Register / align a drifting time-lapse

**Goal.** Remove stage drift (translation, optionally rotation / affine) from a time series so that cells
stay in place; correct Z drift in 4-D data; handle deformation with elastic alignment.

**Sample.** Bundled **Nuclei timelapse (synthetic)**; the script adds an artificial drift of (3, 2) px per
frame to show the correction. Your own drifting stack works unchanged.

**UI steps.**

1. Open the time series. ⌘K → **Align Stack (StackReg)** (Plugins ▸ Registration): *axis* T, *transformation*
   Translation (Rigid body adds rotation, Affine adds scaling/shear), *reference* First slice (Previous slice
   for slow continuous drift, Middle slice), *channel* 0 = current, *sub-pixel factor* 10, *interpolation*
   Linear, *create new image* on to keep the original. **Apply**.
2. The **Alignment** table lists dx, dy, angle and scale per frame; the aligned stack opens as
   *Aligned <name>*. Play the T slider to check.
3. For 4-D (T, Z) data ⌘K → **Correct Drift (3D)** also estimates Z shifts (*Also correct Z drift*).
4. For non-rigid motion ⌘K → **Elastic Alignment (Optical Flow)** (TV-L1 / Farnebäck / iLK flow between
   consecutive frames; *displacement magnitude stack* shows how much moved) or, for one pair, **Elastic
   Registration (bUnwarpJ-style)** + **Apply Deformation**.
5. ⌘K → **Phase Correlation** reports the residual shift between two open images; **Register to Image**
   aligns one image onto another; **Align Channels** removes chromatic shift between channels;
   **Register Virtual Stack Slices (Folder)** aligns a folder of files and writes the result to disk.

**Script.**

```python
# recipe: register
import numpy as np
lumen.run("file.sample", name="Nuclei timelapse (synthetic)")
src = lumen.image
data = lumen.data                                                      # (T, Z, C, Y, X)
drift = np.stack([np.roll(data[t], (3 * t, 2 * t), axis=(-2, -1)) for t in range(data.shape[0])])
lumen.new_image(drift, name="drifting.tif", axes="TZCYX", like=src)   # like= copies the calibration
aligned = lumen.run("registration.align_stack", axis="T", mode="Translation", reference="First slice",
                    upsample=10, create_new=True)[0]
print("aligned:", aligned.name, aligned.shape)
shifts = lumen.session.tables["Alignment"]
print(shifts["columns"], shifts["rows"][:3])
lumen.select(aligned.name); lumen.run("stacks.substack", slices="1", axis="T");  first = lumen.image
lumen.select(aligned.name); lumen.run("stacks.substack", slices="12", axis="T"); last = lumen.image
print("residual shift last→first:", lumen.run("registration.phase_correlation", target=first.name)[0])
lumen.select("drifting.tif")
print(lumen.run("registration.drift_correct", reference="First frame", create_new=True)[0].name)
lumen.select("drifting.tif")
print(lumen.run("registration.elastic_align", axis="T", reference="Previous slice", create_new=True)[0].name)
```

**Expected output.** `aligned: Aligned drifting.tif (12, 1, 1, 256, 256)`; Alignment rows
`['drifting.tif:T2', 2, -3.0, -3.2, 0.0, 1.0], ['drifting.tif:T3', 3, -6.2, -6.5, 0.0, 1.0] …` (the per-frame
drift of 3 px/frame in y and ≈ 2 px/frame in x, recovered to within the cell motion); residual shift
`dx 0.1, dy 0.7`; `Drift corrected drifting.tif`; `Elastic drifting.tif`.

**Pitfalls.**

- Registration estimates the shift on *one* channel (`channel`, 0 = current) and applies it to all.
  Pick a channel with static structure (nuclei, beads), not one with moving signal.
- *Previous slice* accumulates errors over long series; *First slice* cannot follow large drifts (more
  than half the field); combine with `registration.align_folder` for huge data.
- Shifting frames into place leaves empty borders on the moved frames; crop afterwards (**Auto Crop** or a
  rectangle + **Crop**).
- In the synthetic example the cells themselves move, so the recovered shift is the *median* motion;
  on real data with a static background the table reproduces the drift exactly.

**Fiji.** *Plugins ▸ Registration ▸ StackReg* / *TurboReg*, *Correct 3D drift*, *Linear Stack Alignment with
SIFT*, *bUnwarpJ*, *Register Virtual Stack Slices*, *Image ▸ Stacks ▸ Tools ▸ ...*.

---

## 15. FFT filtering of periodic noise

**Goal.** Remove stripes / interference patterns by editing the Fourier spectrum: either the bandpass
filter with stripe suppression, or masking the offending peaks by hand and inverse-transforming — and a
reusable custom filter for whole stacks.

**Sample.** The script adds vertical stripes (period 8 px) to the bundled **Blobs**; the ImageJ **TEM Filter**
sample is the classic real-world case.

**UI steps.**

1. Open the striped image. Quick route: ⌘K → **Bandpass Filter (FFT)** (*filter large structures down to*
   40 px, *small structures up to* 3 px, *suppress stripes* Vertical / Horizontal, *autoscale*) with
   **Preview** on. Good for stripes aligned with the axes.
2. Precise route: ⌘K → **FFT**. The power spectrum opens as an 8-bit image (origin in the centre; the
   complex transform is kept behind it). Periodic noise shows as bright spots away from the centre.
3. Paint the spots black with the oval tool (**2**) + **Fill** (**F**, after ⌘K → **Set Foreground Value** 0)
   or the brush (**9**) with foreground 0. Black = blocked; painting white (255) instead passes *only* those
   regions.
4. ⌘K → **Inverse FFT** (*masking* on, output *Original type* or 32-bit): the filtered image opens.
   **Redisplay Power Spectrum** restores the spectrum if you painted too much.
5. For a stack, duplicate the (painted) spectrum, ⌘K → **Set** value 255 and paint the blocked spots black
   again — this is a filter image (255 = pass, 0 = block); ⌘K → **Custom Filter (FFT)** applies it to every
   plane. **FD Math** correlates / convolves / deconvolves two images in Fourier space; **Swap Quadrants**
   moves the origin between the corner and the centre.

**Script.**

```python
# recipe: fft
import numpy as np
lumen.run("file.sample", name="Blobs")
p = lumen.plane.astype(np.float32)
yy, xx = np.mgrid[:p.shape[0], :p.shape[1]]
stripes = 40 * np.sin(2 * np.pi * xx / 8)                              # vertical stripes, period 8 px
lumen.new_image(np.clip(p + stripes, 0, 255).astype(np.uint8), name="blobs_stripes.tif")
noisy = lumen.image
# A) bandpass with stripe suppression
lumen.run("image.duplicate", title="bandpass")
lumen.run("process.fft_bandpass", large=40, small=3, suppress="Vertical", autoscale=True)
bp = lumen.image
# B) manual FFT mask: black out the two peaks in the power spectrum, inverse FFT
lumen.select(noisy.name)
spec = lumen.run("process.fft")[0]                                      # 8-bit power spectrum, origin centred
n = spec.width
cx, cy, f = n // 2, n // 2, n // 8                                      # period 8 px -> peaks at ± n/8 from the centre
for dx in (-f, f):
    lumen.set_roi({"type": "oval", "x": cx + dx - 5, "y": cy - 5, "w": 11, "h": 11})
    lumen.run("edit.fill", value=0)                                     # black = blocked
lumen.set_roi(None)
clean = lumen.run("process.inverse_fft", output="Original type", masking=True)[0]
def stripe_power(a):
    return float(np.abs(np.fft.fft2(a.astype(np.float32)))[0, n // 8])
print("stripe power  original:%.0f  noisy:%.0f  bandpass:%.0f  fft-mask:%.0f" % (
    stripe_power(p), stripe_power(noisy.data[0, 0, 0]), stripe_power(bp.data[0, 0, 0]), stripe_power(clean.data[0, 0, 0])))
# C) a filter image (255 = pass, 0 = block) applied to every plane
lumen.select(spec.name)
lumen.run("image.duplicate", title="filter")
lumen.run("math.set", value=255)
for dx in (-f, f):
    lumen.set_roi({"type": "oval", "x": cx + dx - 5, "y": cy - 5, "w": 11, "h": 11})
    lumen.run("edit.fill", value=0)
lumen.set_roi(None)
lumen.select(noisy.name)
print(lumen.run("process.fft_custom_filter", filter="filter", all_planes=True)["message"])
print("stripe power after custom filter: %.0f" % stripe_power(lumen.image.data[0, 0, 0]))
```

**Expected output.** `FFT of blobs_stripes.tif (1, 1, 1, 256, 256) uint8`, `Inverse FFT of blobs_stripes.tif … uint8`;
`stripe power  original:9753  noisy:1304271  bandpass:837259  fft-mask:166` — the hand mask removes the
stripes almost completely while the bandpass (which keeps structures between 3 and 40 px) only attenuates a
period-8 pattern; `Custom filter filter applied to 1 plane(s)` and ≈ 116 afterwards.

**Pitfalls.**

- The spectrum image remembers which image it came from; **Inverse FFT** needs that source still open and
  unchanged in size. Painting must stay in the spectrum image (0 or 255 values are the mask; grey is ignored).
- Mask both symmetric peaks (the spectrum of a real image is point-symmetric); the script does, with `± f`.
- Images whose size is not a power of two are transformed as they are (no padding), so frequencies map to
  `width / period` pixels from the centre exactly.
- **FFT Options** (window function, raw power spectrum, complex output) change what **FFT** produces.

**Fiji.** *Process ▸ FFT ▸ FFT*, paint with the brush/oval + *Edit ▸ Fill* (black), *Process ▸ FFT ▸ Inverse FFT*,
*Process ▸ FFT ▸ Bandpass Filter…* (*Suppress stripes*), *Custom Filter…*, *FD Math…*, *Swap Quadrants*.
