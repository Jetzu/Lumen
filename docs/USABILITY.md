# Lumen vs Fiji/ImageJ: a comparative usability assessment

*November 2026. Seven heuristic evaluations plus a keystroke-level task comparison. The first
four (one general pass and three role-played persona passes - a Fiji power user, an occasional
wet-lab biologist, an accessibility reviewer) were performed by the AI assistant that wrote this
document; the other three were performed afterwards, on the fixed build, by three other models of
the same family (Claude Sonnet 5.5, Haiku 5.5 and Opus 5.5) running as separate agents that had
not read this document or each other's output. All seven are AI evaluations, stated plainly in
section 1.1; none is a user. The Fiji side is derived from ImageJ 1.54's own documentation (every
step links to its source); the Lumen side was measured and walked on the running application with
Playwright scripts in headless Google Chrome. Section 3.9 merges the 41 problems of the first
round and 3.10 ranks them; sections 3.12-3.13 do the same for the 42 problems of the second
round, 5.3-5.5 say what was fixed after each, and 3.14 holds the re-walks (E8-E10) in which the
evaluators who found the problems went back to the fixed builds. No user study has been run yet;
section 5 proposes one and `docs/usability_study/` holds the kit to run it. Written for scientists who know
Fiji. Nothing here is a benchmark of analysis quality: it compares the interfaces only.*

**Headline.** Over the ten tasks of section 2, the shortest documented Fiji path needs 84 actions
(clicks, key chords and typed strings) and opens 20 separate windows, 7 of them modal dialogs;
the measured Lumen path needs 54 actions, opens no separate window, and uses 15 in-window panels
(7 of them the transient command palette). Lumen is shorter on 6 tasks, equal on 3 (open and
adjust a multi-channel image, Z-projection, find a command) and one action longer on 1 (measuring
a drawn rectangle, because Fiji's default tool is the rectangle). The large differences are
tracking (7 vs 21-27), batch from a recording (11 vs 20), threshold-and-particles (4 vs 7, 0 vs 5
windows) and side-by-side comparison (1 vs 2-6). The honest counterweights are in section 3.4:
Fiji's plugin ecosystem, Bio-Formats, script languages, large-data handling, community
documentation and twenty years of validated algorithms; Lumen's browser constraints, its novelty,
and the parity rows that were still partial at the time. The first four heuristic evaluations
(section 3) found 41 distinct problems, 10 of them major (severity 3) and none catastrophic; the
major ones were concentrated in error prevention around destructive or implicit actions and in
keyboard / screen-reader access, both areas the original pass had not exercised (sections 3.9,
3.10). After those were fixed, the three evaluations by other models (sections 3.11-3.13) found
42 problems, 30 of them new, 9 of them major and one rated catastrophic: a recording that does
not say which image each command ran on. Eight of the nine majors are about the recorder and the
scripting API - areas the first round had rated minor - and they, too, have since been fixed
(section 5.4). The evaluators also confirmed, without having been told about them, that the first
round's fixes hold, and caught two that did not (section 3.12). The re-walks of section 3.14 then
checked the second round's fixes the same way, found a severity-4 regression that a scripted check had
missed, and after the third round confirmed the recorder, the validation, the clears and the menus.

---

## 1. Method

### 1.1 Heuristic evaluation

The two interfaces were walked through against Nielsen's ten usability heuristics
([Nielsen 1994, reviewed 2024](https://www.nngroup.com/articles/ten-usability-heuristics/))
plus five criteria specific to scientific image software, chosen from the requirements that recur
in the bioimage-analysis surveys reviewed in [RESEARCH.md §12](RESEARCH.md): *reproducibility and
recording*, *provenance*, *batch and scripting*, *large-data handling*, and *accessibility*.
Problems are rated on Nielsen's 0-4 severity scale (0 not a problem, 1 cosmetic, 2 minor, 3 major,
4 catastrophe; severity combines frequency, impact and persistence -
[Nielsen 1994, severity ratings](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/)).
A heuristic evaluation is "a method for identifying design problems in a user interface" in which
"evaluators judge the design against a set of guidelines"; NN/g recommends that "three to five
people should independently evaluate the same interface" and then merge and rate the findings
([Moran & Gordon 2023](https://www.nngroup.com/articles/how-to-conduct-a-heuristic-evaluation/)).

**Four evaluations.** This document contains four heuristic evaluations of Lumen's GUI, written
in separate sections so that they can be compared:

| | Evaluator | Perspective | Method | Section |
|---|---|---|---|---|
| E1 | original evaluator | general, knows Lumen's source | comparative tables Fiji vs Lumen, one rating per heuristic | 3.1-3.5 |
| E2 | *Fiji power user who scripts everything* | recorder, script editor, macros, shortcuts, batch, stack scope, API | 3.6 |
| E3 | *wet-lab biologist who uses Fiji twice a month* | menus rather than the palette, labels and hints, errors, closing unsaved work, file dialog, laptop viewport (1366×768) | 3.7 |
| E4 | *accessibility / HCI reviewer* | keyboard reachability, ARIA semantics, focus management, contrast in both themes, target sizes, motion, reflow | 3.8 |

**All four evaluations were performed by an AI assistant (Claude), not by four people.** E1 is
the original pass. E2-E4 are role-played personas: each was run as a separate pass over the live
application with its own probe set, and each table was written before the other two were
re-read, but the three share one model's knowledge and habits. They are therefore not independent
in the sense NN/g intends, and the agreement statistics in section 3.9 are weaker evidence than
the same numbers from four human evaluators would be. They are reported because the exercise
still does what multiple evaluators are for: E2-E4 found severity-3 problems that E1 had not
rated (section 3.9), mostly in areas E1 did not exercise.

**How the persona walks were done.** Each persona walked the running build (served from
`lumen/static`, built 7 Nov 2026, 488 registered commands) in headless Google Chrome through
`tests/e2e/usability/heuristic_walkthrough.py` (Playwright, channel `chrome`, SwiftShader WebGL
as in `tests/e2e`). The script visits the UI locations an evaluator would - menus, the command
palette, parameter sheets, every inspector section, Preferences, the help overlay, the split view,
the script editor, the pixel classifier, the tracks panel and the 3-D view - performs the
persona's probes, and records evidence only: screenshots, DOM facts (roles, labels, contrast
ratios, focus order, target sizes, hint and error texts) and the server-side outcome of each step
read back through the REST API. The judgements - what is a problem, and its severity - were made
by the evaluator from that evidence and the screenshots; the script asserts nothing. Its structured
output (`evidence.json` per persona) is committed under `tests/e2e/usability/walkthrough/`;
the screenshots named in the tables are written next to it when the script is re-run
(`python heuristic_walkthrough.py --url http://host:port`). Evidence for Fiji remains the ImageJ
User Guide, menu reference and plugin pages (linked inline); evidence for E1's Lumen column is the
build of 7 Nov 2026 (407 registered commands at the time) and the measurements of section 2.

The session the walks ran in was shared with other work on the same server: another image was
already open, so the empty-viewport screen could not be photographed and is described from
`Viewport.tsx` instead, and section open/closed states carried over within each persona's browser
context (noted where it matters).

**Three further evaluations by other models (E5-E7).** After the first round's problems had
been fixed (§5.3), three more evaluations were run on the build of 8 Nov 2026 by three *other*
models of the Claude family - Sonnet 5.5, Haiku 5.5 and Opus 5.5 - each as a separate agent with
its own headless Chrome, its own persona and no access to this document, to the earlier
evaluations, to Lumen's source or to each other. They received one brief: the same list of tasks
to walk (open and adjust a multi-channel sample; threshold and Analyze Particles; rectangle and
Measure; Z Project; LUTs; Gaussian Blur with preview and undo; a time series and the Tracks
wizard; palette searches; Preferences, the help overlay, the editor, the colour picker, the split
view, the 3-D view; closing a modified image; the menubar from the keyboard; commands without a
selection), the heuristics and severity scale of this section, and the instruction to report
only what a screenshot or a DOM fact shows. Their structured output is committed as
`tests/e2e/usability/walkthrough/{e5_sonnet,e6_haiku,e7_opus}/evaluation.json`; §3.11 reproduces
it, §3.12 measures their agreement with each other and with E1-E4, and §3.13 ranks the merged
list. They remain AI evaluations - same vendor, same brief, models that share training - and
§3.12 says how far that limits the agreement figures. What they add over E1-E4 is independence
of *authorship*: three minds with one brief instead of one mind with four briefs, and a check on
the first round's fixes by evaluators who did not know what had been fixed.

| | Evaluator (model) | Perspective | Section |
|---|---|---|---|
| E5 | Claude Sonnet 5.5 | *microscopy core-facility manager who trains biologists*: novice time-to-productivity, irreversible-mistake risk | 3.11 |
| E6 | Claude Haiku 5.5 | *cell biologist, two years of Fiji through the menus, never scripted, impatient* | 3.11 |
| E7 | Claude Opus 5.5 | *image-analysis specialist who writes macros and Python daily*: reproducibility, provenance, ImageJ-identical results; knows napari, QuPath, Icy | 3.11 |

### 1.2 Task-based comparison

Ten tasks that a microscopy user performs weekly were specified (section 2.1). For each tool and
task the following were counted: separate windows opened, modal dialogs among them, clicks plus
key presses, whether a live preview exists, undo depth, and whether the step ends up in a
reproducible record.

**Counting rules (identical for both tools).**

| Rule | Count |
|---|---|
| Mouse click, or one click-drag (press, move, release) | 1 click |
| Menu navigation | 1 click per menu level (Image ▸ Stacks ▸ Z Project = 3) |
| Key press or chord (Ctrl+Shift+T, M, Enter, ⌘K) | 1 key |
| Typed string (a search query, a number, a path), regardless of length | 1 text entry |
| Choice from a dropdown | 2 clicks (open the list, pick the entry) |
| Checkbox toggle, OK / Apply / Run / Next button | 1 click |
| A documented keyboard shortcut is used when the command has one | that key, not the menu path |
| Dialog defaults are accepted when they fit the task; a checkbox is counted only when the task needs it and the documented default is off or undocumented (stated per task) | |
| **Windows**: separate OS windows opened during the task (Fiji: every image, dialog, table, plot, manager is one). Lumen runs in one browser window, so this is always 0; its in-window panels are counted separately as *dialogs*: the command palette (a transient overlay that blocks the UI until Esc or a run: counted as modal) and parameter sheets (non-modal, docked at the top right of the viewport) | |
| The image is already open at the start of every task except task 1. The Lumen setup actions (opening the sample through the UI) are counted but reported in their own column | |
| First-time use: no remembered dialog state, shortest documented path, expert who knows the names | |

**Fiji counts** are derived, not measured: each step is a documented control on a documented
menu path (section 2.3 lists the derivation and the source for every task). Where the
documentation does not state a default, the assumption is written down. ImageJ remembers dialog
settings between uses, so a second run of the same task in Fiji costs fewer clicks than the
first-time count given here; Lumen's inspector sections likewise remember whether they were open.

**Lumen counts** were measured on the running server (`http://127.0.0.1:8766`) with
`measure_lumen.py` in headless Google Chrome (Playwright, channel `chrome`, viewport 1440×900,
SwiftShader WebGL as in `tests/e2e`). Every pointer and keyboard action the script performs goes
through one `Counter` object, so the counts are complete by construction; the per-task action
logs are in section 2.4, and the script verifies each outcome through the REST API (results
rows, ROIs, history, recorded macro lines, written files). The script and its outputs live in
the repository:
`tests/e2e/usability/measure_lumen.py` (outputs `lumen_counts.json` / `lumen_counts.md` next to it)
(outputs `lumen_counts.json`, `lumen_counts.md`, `run2.log` in the same folder; the batch input
folder `batch_in/` holds three 128×128 8-bit TIFFs). It uses the same selectors and helpers as
`tests/e2e/conftest.py` (`palette`, `open_sample`, `section`, `drag`) and can be re-run against
any Lumen server with `python measure_lumen.py http://host:port`.

Sample images: Lumen's bundled samples *Synthetic nuclei (3 channels)* (512×512×3),
*Blobs* (`blobs.tif`, 8-bit), *Cells (3-D, 2 channels)* (256×256, z-stack), *Nuclei timelapse
(synthetic)* and *Coins*; the Fiji paths assume the equivalent ImageJ samples
*Fluorescent Cells (400K)* (3-channel composite), *Blobs (25K)*, *Confocal Series* and
*mitosis.tif*, all listed at [imagej.net/ij/images](https://imagej.net/ij/images/).

---

## 2. Task-based comparison

### 2.1 The ten tasks

1. Open a multi-channel image and adjust the display range of each channel.
2. Threshold the image and run Analyze Particles with results and ROIs.
3. Draw a rectangular ROI and measure it.
4. Z-project a stack (maximum intensity).
5. Make a 3-channel composite with one LUT per channel: (a) on an open composite, (b) by merging three single-channel images.
6. Run a filter (Gaussian blur, sigma 3) with a preview, then undo it.
7. Track cells in a time-lapse and export the tracks to a file.
8. Find and launch a command whose menu location you do not know (CLAHE).
9. Batch-process a folder with the steps you just recorded (a Gaussian blur).
10. Compare two open images side by side.

### 2.2 Results

Actions = clicks + keys + text entries. Fiji windows include the dialog itself and every window
the command opens (image, Results, Summary, ROI Manager, …). "Recorded" means the step appears in
a machine-readable record without the user doing anything extra beyond having the recorder open
(Fiji) or nothing at all (Lumen).

| # | Task | Fiji actions | Fiji windows (modal) | Fiji preview | Fiji undo | Fiji recorded | Lumen actions (setup) | Lumen windows / dialogs (modal) | Lumen preview | Lumen undo | Lumen recorded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Open 3-channel image, adjust each channel | 9 | 2 (0) | yes (display range) | Reset button; not an undo step | Auto / Set yes, slider drags no (see 2.3) | **9** (0) | 0 / 2 (1) | yes, instant on the client | Reset / Reset all; not an undo step | open yes; display changes **no** |
| 2 | Threshold + Analyze Particles | 7 | 5 (1) | threshold yes; particles no | n/a (pixels untouched without Apply) | yes | **4** (6) | 0 / 1 (0) | threshold yes; particles no | n/a | yes (2 lines) |
| 3 | Measure a drawn ROI | **2** | 1 (0) | n/a | n/a | yes | 3 (6) | 0 / 0 (0) | n/a | n/a | yes (ROI geometry in the line) |
| 4 | Z-projection | 4 | 2 (1) | no | n/a (new window) | yes | **4** (6) | 0 / 2 (1) | no | n/a (new tab) | yes |
| 5a | Composite: LUT per channel | 9 | 1 (0) | yes | n/a | yes (menu commands) | **6** (6) | 0 / 0 (0) | yes, instant | Reset all | **no** (display state) |
| 5b | Composite: merge 3 images | 10 | 2 (1) | no | n/a | yes | **10** (9) | 0 / 2 (1) | no | n/a (new image) | yes |
| 6 | Filter with preview, undo | 7 (+1 on stacks) | 1 (1) (+1 on stacks) | yes, checkbox in a modal dialog | 1 step, no redo | yes | **6** (6) | 0 / 2 (1) | yes, on by default, non-modal | measured 3 steps undone + redo; memory-bounded | yes |
| 7 | Track cells, export tracks | 21-27 | 2 (1) + 3 if statistics tables | yes (detector Preview, overlays) | none (Previous button; initial filter "irreversible") | **no** (IJ1 recorder cannot record TrackMate; XML save instead) | **7** (6) | 0 / 3 (1) | **no** detection preview | tracks table (delete / clear) | yes (2 lines) |
| 8 | Find a command | 3 | 1 (0) | n/a | n/a | yes once run | **3** (6) | 0 / 1 (1) | n/a | n/a | yes once run |
| 9 | Batch a folder from the recording | 20 | 5 (3) | Test button | n/a (new files) | is the recording | **11** (6) | 0 / 4 (2) | no | n/a (new files) | yes (batch line too) |
| 10 | Compare two images side by side | 2 (6 with synchronized cursor/zoom) | 0 (1 with sync) | n/a | n/a | tile: yes | **1** (12) | 0 / 0 (0) | n/a | n/a | no (view state) |
| | **Total (5a path)** | **84** | **20 (7)** | | | | **54** | **0 / 15 (7)** | | | |

Bold marks the shorter path. Totals use the lower TrackMate bound (defaults accepted) and
*Tile* alone for task 10.

### 2.3 Fiji: derivation of each count, with sources

The ImageJ User Guide pages are the 1.46r edition that imagej.net still serves as the reference
manual; the menu reference and the Java sources are current (1.54).

**Task 1.** File ▸ Open Samples ▸ Fluorescent Cells (3 clicks; "Opens example images hosted on
the ImageJ Web site", [Guide §26](https://imagej.net/ij/docs/guide/146-26.html)); the image
opens "in a separate window" (same page). Brightness/Contrast with its shortcut [C] (1 key): a
window with "Minimum and Maximum sliders", "Brightness and Contrast sliders" and "Auto, Reset, Set,
and Apply" buttons; "Holding Shift adjusts all channels of a composite image at once", otherwise
the current channel ([Guide §28.2.1](https://imagej.net/ij/docs/guide/146-28.html)). Channel 1:
Auto (1 click); channels 2 and 3: move the hyperstack c-scrollbar (1 click each; hyperstacks "are
displayed in a window with three labelled scrollbars",
[Guide §8](https://imagej.net/ij/docs/guide/146-8.html)) then Auto (1 click each). Total 9
(8 clicks, 1 key); windows: image + B&C = 2. Preview: adjusting the sliders "changes the LUT - and
not the pixel values … unless you press Apply"
([Bioimage Book](https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html));
Apply "applies the current display range mapping function to the pixel data" (Guide §28.2.1).
Recording: Auto, Set and Apply emit recorder lines, plain slider drags do not - this is the
evaluator's experience, **not verified** in the documentation.

**Task 2.** Threshold with its shortcut [T] (1 key): "upper slider to adjust the minimum threshold
value and the lower one to adjust the maximum", Method dropdown with 16 methods, Dark background,
Auto, Apply, Reset, Set ([Guide §28.2.4](https://imagej.net/ij/docs/guide/146-28.html),
[menu reference](https://imagej.net/ij/docs/menus/image.html)); Apply is not needed because
Analyze Particles "counts and measures objects in binary or thresholded images"
([Analyze menu reference](https://imagej.net/ij/docs/menus/analyze.html)). Analyze ▸ Analyze
Particles (2 clicks) opens a modal dialog with Size, Circularity, Show, and the checkboxes
Display results, Clear results, Summarize, Add to Manager, Exclude on edges, Include holes,
Record starts, In situ Show ([Guide §30](https://imagej.net/ij/docs/guide/146-30.html)); the
task needs Display results, Summarize and Add to Manager (3 clicks; defaults are not documented,
ImageJ remembers them) and OK (1). The command opens "the Results Table, the Summary table, the
ROI Manager" (Guide §30; Summarize shows results "in a separate Summary table"). Total 7
(6 clicks, 1 key); windows 5 (Threshold, dialog, Results, Summary, ROI Manager), 1 modal.

**Task 3.** The rectangle is ImageJ's default tool; click-drag (1): "location, width, height, and
aspect ratio appear in the status bar while drawing"
([Guide §19.1.1](https://imagej.net/ij/docs/guide/146-19.html)). Measure [m] (1 key)
"calculates and displays on the Results Table either area statistics, line lengths and angles, or
point coordinates" ([Guide §30](https://imagej.net/ij/docs/guide/146-30.html)). Total 2;
windows 1 (Results).

**Task 4.** Image ▸ Stacks ▸ Z Project (3 clicks) builds a dialog with `Start slice`, `Stop slice`
and a `Projection type` choice from `{"Average Intensity", "Max Intensity", "Min Intensity",
"Sum Slices", "Standard Deviation", "Median"}`, plus `All time frames` for hyperstacks
([ZProjector.java](https://imagej.net/ij/developer/source/ij/plugin/ZProjector.java.html)).
Defaults accepted (if the remembered method is not Max Intensity, add 2 clicks for the
dropdown), OK (1). Total 4; windows 2 (modal dialog + the new `MAX_` image).

**Task 5a.** Channels Tool with its shortcut Shift+Z (1 key) "opens the Channels widget" to
"switch display modes (composite, color or grayscale) and enable/disable channels"; its "More≫"
button "gives quick access to some Image▷Color commands and a list of primary colors (red, green,
blue, cyan, magenta, yellow) for pseudocoloring composites"
([Guide §28.7.5](https://imagej.net/ij/docs/guide/146-28.html),
[menu reference](https://imagej.net/ij/docs/menus/image.html)). Channel 1: More≫ then a colour
(2 clicks); channels 2 and 3: c-scrollbar (1) + More≫ (1) + colour (1) each. Total 9 (8 clicks,
1 key); windows 1. Recorded: these are menu commands and "each menu command you use generates a
macro run() function call" when the recorder is open
([Guide §31](https://imagej.net/ij/docs/guide/146-31.html)).

**Task 5b.** Image ▸ Color ▸ Merge Channels (3 clicks): "merges 2-7 images into an RGB or
multi-channel composite", one C1-C7 dropdown per channel, "Create composite" checkbox (and Keep
source images, Ignore source LUTs) ([Guide §28.5.2](https://imagej.net/ij/docs/guide/146-28.html)).
Three dropdowns (6 clicks), Create composite assumed already ticked (its default is not
documented; +1 if not), OK (1). Total 10; windows 2 (modal dialog + Composite).

**Task 6.** Process ▸ Filters ▸ Gaussian Blur (3 clicks); the dialog is built with
`gd.addNumericField("Sigma (Radius):", …)`, `gd.addCheckbox("Scaled Units …")` and
`gd.addPreviewCheckbox(pfr)`
([GaussianBlur.java](https://imagej.net/ij/developer/source/ij/plugin/filter/GaussianBlur.java.html)):
sigma (1 text), Preview (1 click), OK (1). On a stack ImageJ first asks "Choose 'Yes' to process
entire stack or 'No' to process only the active slice" unless "Hide 'Process Stack?' dialog" is set
([Guide §29](https://imagej.net/ij/docs/guide/146-29.html),
[Guide §27, Misc options](https://imagej.net/ij/docs/guide/146-27.html)). Undo [z] (1 key).
Total 7 (5 clicks, 1 key, 1 text), +1 click on stacks. Undo "is currently limited to the most
recent image editing / filtering operation" and "the equivalent to 'Redo' is the Process▷Repeat
Command [R], that re-runs the previous used command"
([Guide §6](https://imagej.net/ij/docs/guide/146-6.html)); multi-image undo needs "Keep multiple
undo buffers" ([Guide §27](https://imagej.net/ij/docs/guide/146-27.html)).

**Task 7.** Plugins ▸ Tracking ▸ TrackMate (3 clicks; menu path from TrackMate's
[plugins.config](https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/resources/plugins.config)).
The [getting-started tutorial](https://imagej.net/plugins/trackmate/tutorials/getting-started)
walks through, in order: start panel (calibration), detector choice, detector configuration
("Estimated blob diameter", threshold, median filter, sub-pixel localization, **Preview** button),
detection log, initial spot filtering (quality histogram; the tutorial calls this step
irreversible), view choice, spot filters, tracker choice, tracker configuration (Simple LAP: three
fields), tracking log, track filters, display options (with the TrackScheme button), plot panel,
actions panel, save. Pressing Next through those 13 panels (13 clicks), then in the actions panel
choosing "Export tracks to XML file" (2 clicks; action name and its `IOUtils.askForFileForSaving`
save prompt from
[ExportTracksToXML.java](https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/java/fiji/plugin/trackmate/action/ExportTracksToXML.java)),
Execute (1) and the save dialog (filename 1 text, Save 1 click) gives 21 actions with every
default accepted; typing the blob diameter, pressing Preview, dragging the initial quality
threshold and typing the three LAP distances brings it to 27. Windows: the wizard and the modal
save dialog (the image window already exists; "Export statistics to tables" would "export all
statistics to 3 tables" -
[ExportStatsTablesAction.java](https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/java/fiji/plugin/trackmate/action/ExportStatsTablesAction.java)).
Recording: "The macro language is too limited to work with such awesome things as TrackMate"; the
documented reproducible route is a Jython script or the saved XML
([TrackMate scripting](https://imagej.net/plugins/trackmate/scripting/scripting)).

**Task 8.** The Command Finder: "Evoke the prompt by pressing L", "If you type part of a command
name, the list will only show commands that match that substring", "If only a single command
matches then that command can be run by pressing Enter", otherwise "Double clicking on a command
will run that command"; "Show full information" displays "the location of the listed menu
entries" ([Guide §31](https://imagej.net/ij/docs/guide/146-31.html)). L (1 key), text (1),
Enter (1) = 3 for a unique match, 4 with an arrow key. The ImageJ2 search bar "integrated with
the main ImageJ window" searches commands and, when ticked, the wiki and forum
([ImageJ news, 29 Dec 2017](https://imagej.net/news/2017-12-29-imagej2-and-fiji-updated-now-with-search-bar));
same count.

**Task 9.** Plugins ▸ Macros ▸ Record (3 clicks): "To create a macro, open the recorder, use one or
more ImageJ commands, then click Create" ([Guide §31](https://imagej.net/ij/docs/guide/146-31.html)).
Run the Gaussian: 3 clicks + OK (1) (dialog, modal). Copy the recorded line: click in the recorder
text (1), Ctrl+A (1 key), Ctrl+C (1 key). Process ▸ Batch ▸ Macro (3 clicks): "Input…" selects the
source folder, "Output…" the destination, "Output Format" TIFF by default, "Add Macro Code"
snippets, "Test" runs the macro on the first image
([Guide §29](https://imagej.net/ij/docs/guide/146-29.html); "The fastest way to start batch
conversion is via the Process › Batch › Macro… command" and "The macro recorder is an excellent
way to generate macro code", [imagej.net/scripting/batch](https://imagej.net/scripting/batch)).
Input… (1) + choose in the folder dialog (1), Output… (1) + choose (1), click in the code area (1),
Ctrl+V (1 key), Process (1). Total 20 (17 clicks, 3 keys); windows 5 (Recorder, Gaussian dialog,
Batch dialog, two folder choosers), 3 modal; the optional Test adds an image window.

**Task 10.** Window ▸ Tile (2 clicks) "shrinks all open image windows and repositions them to fit on
the screen without overlapping" ([Window menu](https://imagej.net/ij/docs/menus/window.html));
no linked navigation. Analyze ▸ Tools ▸ Synchronize Windows (3 clicks) "synchronizes mouse motion
and input between multiple windows" with Sync cursor / channels / z-slices / t-frames options
([Guide §30](https://imagej.net/ij/docs/guide/146-30.html)) plus "Synchronize All" (1) = 6 with
one extra window.

### 2.4 Lumen: measured action logs

Each line is one counted action as the script performed it (full logs with verification strings
in `run2.log` / `lumen_counts.json`). "dialog" lines are not actions; they are the panels that
appeared.

**Task 1 - 9 actions (7 clicks, 1 key, 1 text).** Ctrl+K · type "open sample" · click *Open
Sample* · sheet: sample dropdown (2) · Apply · Display section (open by default; all three
channels' histogram sliders visible at once): Auto ×3. Verified: per-channel ranges changed on all
three channels, mode = composite. Recorded: only `lumen.run("file.sample", …)`; the display
changes produced no recorder line.

**Task 2 - 4 clicks.** Expand *Threshold* section · Auto · Particles… · Run. Verified: threshold
118-255; 55 ROIs listed in the docked ROI Manager, 55 rows in the docked Results, summary in the
Table section. Recorded: `threshold.compute(method='Default', …)` and `analyze.particles(…)` with
all parameters.

**Task 3 - 3 actions (1 drag, 2 keys).** `1` (rectangle tool) · drag on the canvas · `m`.
Verified: one Results row (`blobs.tif`, area 5696, mean 84.76). Recorded:
`analyze.measure(roi={'type': 'rect', 'x': 75, 'y': 84, 'w': 89, 'h': 64})`.

**Task 4 - 4 actions.** Ctrl+K · "z proj" · Enter (the selected entry was *Z Project*) · Apply.
The sheet has no Preview switch (whole-image command). Verified: `MAX_cells3d.tif`, shape
z = 1, c = 2, opened as a second tab. Recorded: `stacks.z_project(method='Max Intensity', …)`.

**Task 5a - 6 clicks.** Display section: LUT dropdown ×3 (Cyan, Magenta, Yellow). Verified
through the API; composite mode is the default for multi-channel images. Not recorded.

**Task 5b - 10 actions (7 clicks, 2 keys, 1 text).** Setup additionally ran *Split Channels*
(3 actions). Ctrl+K · "merge" · Enter · C1, C2, C3 image dropdowns (6) · Apply. Verified:
`Composite`, c = 3, LUTs Red/Green/Blue. Recorded: `color.merge(c1=…, c2=…, c3=…,
create_composite=True, keep_source=False)`.

**Task 6 - 6 actions (1 click, 3 keys, 2 texts).** Ctrl+K · "gauss" · Enter · sigma "3" (the
Preview switch was already on; a preview request fired while typing) · Apply · Ctrl+Z. Verified:
history `['Gaussian Blur']` → undo → redo `['Gaussian Blur']`. Undo-depth probe (8 further
actions, not in the task count): Shift+S, Shift+S, Shift+I pushed `['Smooth', 'Smooth', 'Invert']`
onto the stack, the History section listed them, three Ctrl+Z emptied it with three redo entries,
Ctrl+Shift+Z redid one. Recorded: `filters.gaussian(sigma=3.0, scope='slice')`.

**Task 7 - 7 actions (3 clicks, 2 keys, 2 texts).** Ctrl+K · "track" · Enter · Run (defaults:
LoG, radius 2.5, automatic threshold, LAP linking, gap closing 2) · Tracks section appeared
with 113 tracks and time-coloured overlays · Export CSV… · path · Run. Verified: 946-line CSV
with TrackMate columns (`ID, TRACK_ID, FRAME, POSITION_X, …`). No detection preview in the sheet.
Recorded: `tracking.track(…)` and `tracking.export_tracks(path=…, layout='TrackMate', …)`.

**Task 8 - 3 actions (2 keys, 1 text).** Ctrl+K · "clahe" · Enter. The first entry was
*CLAHE (Local Contrast)* with category *Process* and the description "Contrast-limited adaptive
histogram equalization."; Enter opened its sheet. Uncounted probes: "local contrast" listed
CLAHE, Normalize Local Contrast, Auto Local Threshold, Retinex and a *Search ImageJ wiki for
"local contrast"* entry; a nonsense query showed "No command matches … - search the ImageJ wiki
or the image.sc forum below".

**Task 9 - 11 actions (4 clicks, 4 keys, 3 texts).** Expand *Script & Recorder* · Clear (fresh
recording; the recorder is always on, so this replaces "open the recorder") · Ctrl+K · "gauss" ·
Enter · Apply · Ctrl+K · "recorded macro" · Enter (*Apply Recorded Macro to Folder...*) · input
folder path · Run. Verified: `cell_0.tif`, `cell_1.tif`, `cell_2.tif` written to
`batch_in/lumen_output`, log "Macro batch: 3 ok, 0 failed of 3 file(s)". The replayed script was
`lumen.run("filters.gaussian", sigma=2.0, scope='slice')` (open/save lines are skipped
automatically).

**Task 10 - 1 key.** Ctrl+Shift+\ (setup opened *Blobs* and *Coins*). Verified: a `.split-view`
with two panes showing `coins.png` and `blobs.tif` in the same window; the compare pane follows the
active pane's pan and zoom.

### 2.5 Reading the numbers

- **Where the counts are equal** (tasks 1, 4, 5b, 8) the dialogs are structurally the same: one
  command, one parameter form, one confirmation. Lumen's advantage there is not fewer actions but
  no window management (0 vs 2 windows in task 1 and 4), descriptions and categories in the
  launcher (task 8), and a recorder line without opening a recorder.
- **Where Lumen is clearly shorter** the reason is docking or a single command replacing a
  wizard: the Threshold section with its *Particles…* button (task 2: 4 vs 7, and Results, Summary
  and ROI Manager appear as inspector sections instead of three windows); one `tracking.track`
  command with the Tracks table docked (task 7: 7 vs 21-27); the always-on recorder and a path
  field instead of two folder choosers (task 9: 11 vs 20); a split view on one key (task 10).
- **Where Fiji is shorter or equal with a simpler model**: the default tool is the rectangle, so
  a measurement is drag + M (task 3); Fiji's B&C `Shift`-drag adjusts all channels at once; and
  ImageJ remembers every dialog's settings, so repeated runs cost less than the first-time counts
  above (Lumen sheets also open with the last parameters, but this was not measured).
- **Preview.** Lumen previews plane operations by default (task 6, switch on without a click)
  but has no preview for whole-image or analysis commands (tasks 4, 5b, 7, 9); Fiji's preview is a
  checkbox inside a modal dialog for filters, and TrackMate's detector has a dedicated Preview
  button, which Lumen's tracking sheet lacks.
- **Recording.** Lumen recorded every command in every task without any setup, including tracking
  and the batch run itself, but it does **not** record display changes (tasks 1 and 5a: ranges and
  LUTs are view state), whereas Fiji's recorder captures LUT menu commands and B&C Auto/Set. For a
  figure-preparation workflow this is a real gap.
- **Undo.** Measured: Fiji one step, no redo; Lumen three steps undone and redone, listed in the
  History section, bounded by memory rather than by count.

---

## 3. Heuristic evaluation

Sections 3.1-3.5 are Evaluator 1's original comparative pass and are unchanged except where a later
evaluator contested an entry (marked). Sections 3.6-3.8 are the three persona evaluations, 3.9 the
agreement between all four and 3.10 the merged, ranked list. Sections 3.11-3.13 are the second
round: the three evaluations by other models on the fixed build, their agreement, and their merged
list.

### 3.1 Evaluator 1: Nielsen's ten heuristics (comparative)

Severity is the evaluator's single rating of the most important problem under that heuristic
(0-4, section 1.1). Sources for Fiji statements are in [RESEARCH.md §1](RESEARCH.md) unless
linked here; Lumen statements name the component or the measured task.

| Heuristic | Fiji / ImageJ 1.54 | Sev. | Lumen | Sev. |
|---|---|---|---|---|
| 1 Visibility of system status | Status line in the main window (coordinates, values), progress bar for long commands. State is scattered: which of many windows is active, where results went (the "Window soup" complaints in RESEARCH §1). | 2 | Status bar with dimensions, selection and cursor value (`role="status"`), busy dot, progress chip with percentage, message, elapsed time and Cancel (`ProgressBar.tsx`), HUD chips for *preview*, *threshold* and *1/n downsampled*, toasts (`aria-live="polite"`). Weak spot: the preview covers only the current plane and the HUD chip is the only hint. | 1 |
| 2 Match between system and real world | Its vocabulary *is* the field's standard. But "Make Binary" and "Convert to Mask" coexist, and the "Black background" preference silently inverts binary operations ([Process menu](https://imagej.net/ij/docs/menus/process.html)). | 2 | Keeps ImageJ names, measurement columns and letter shortcuts (M, T, H, K, …; `shortcuts.ts`), so Fiji users' habits transfer; renames a few commands (*Apply Threshold → Mask*) and drops the black-background switch, which a long-time user must unlearn. | 1 |
| 3 User control and freedom | One undo step, no redo ([Guide §6](https://imagej.net/ij/docs/guide/146-6.html)); modal dialogs; B&C *Apply* rewrites pixels. | 3 | Esc closes any overlay in a fixed order; Cancel on long jobs; deep undo/redo with a clickable History section (task 6). Display changes are reversible only through *Reset*, not undo. | 1 |
| 4 Consistency and standards | `GenericDialog` forms are uniform; the big plugins (TrackMate wizard, Trainable Weka, 3D Viewer) each bring their own GUI model. | 2 | All 407 commands share one schema-generated sheet (title, help button, description, fields, *Advanced options*, Preview switch, Cancel, Apply/Run - `OperationSheet.tsx`). The hand-built Display, Threshold, ROI Manager and Tracks panels follow a different, denser layout, and the primary button reads *Apply* for image commands but *Run* for analysis commands. | 1 |
| 5 Error prevention | Analyze Particles on an unthresholded 8-bit image errors out after the dialog; "Process Stack?" asks every time; the persistent black-background preference is the documented trap. | 3 | Parameters are typed with ranges and choices, so invalid values cannot be entered; buttons are disabled until applicable (*Particles…* without a threshold, *Measure* without ROIs, split view with one image); masks are always foreground 255. Not evaluated: confirmation before closing an unsaved image. | 1 |
| 6 Recognition rather than recall | 348 built-in commands plus Fiji's plugins in nested menus (RESEARCH §1); the Command Finder exists but must be known; the EasyFiji authors built a GUI because of this. | 3 | The palette shows name, category, description and shortcut for every command and falls back to a wiki/forum search (task 8); menus mirror ImageJ's; tool rail buttons carry titles and keys. Fourteen inspector sections and the 1-9 / letter tool keys still have to be learnt. | 1 |
| 7 Flexibility and efficiency | Single-letter shortcuts, user shortcuts (Plugins ▸ Shortcuts), macros, ten script languages, update sites. | 1 | Same letter shortcuts, custom shortcuts, ⌘K, Repeat Last (⇧R), Python console, REST/WebSocket API; one key for split view (task 10). Python only. | 1 |
| 8 Aesthetic and minimalist design | A tiny toolbar window; each dialog is minimal; the desktop fills with windows. | 2 | One window, but the inspector lists up to fourteen sections and a parameter sheet overlays the top-right of the viewport; on a 13-inch laptop the inspector needs scrolling (section state is remembered). | 2 |
| 9 Help users recognise, diagnose, recover from errors | Java exceptions surface as stack traces in the Log/Exception window. | 2 | Sheet errors are plain sentences in a `role="alert"` region (e.g. "The recorder is empty - apply some commands first"); toasts; Log section; batch reports per-file errors and continues. Script errors in the console show Python tracebacks. | 1 |
| 10 Help and documentation | The User Guide, imagej.net, forum, and a large body of tutorials and videos; the Guide is the 1.46r edition. | 1 | `?` shortcut overlay, Help menu with README, command reference, parity and design notes rendered in-app (`DocsModal.tsx`), a help button on every sheet, hint text under panels ("Display changes never modify pixel values…"), guided Workflows. No community, no tutorials, no videos. | 2 |

### 3.2 Evaluator 1: scientific-software criteria (comparative)

| Criterion | Fiji / ImageJ 1.54 | Sev. | Lumen | Sev. |
|---|---|---|---|---|
| Reproducibility / recording | Recorder is opt-in (Plugins ▸ Macros ▸ Record) and records IJ1 menu commands; wizard-style plugins such as TrackMate are outside it (section 2.3, task 7). | 2 | Recorder is always on and recorded every command in every task, including tracking and the batch run; the recorded line is runnable Python (`lumen.run(...)`) with every parameter. Display state (ranges, LUTs, composite mode) is not recorded (tasks 1, 5a). | 1 |
| Provenance | Image ▸ Show Info; no per-image history of applied operations beyond what was recorded. | 2 | The History section is the per-image list of applied operations (undo stack) and the Image Info section shows calibration and metadata; exported CSVs carry TrackMate's column layout. Whether the history is written into saved files was not evaluated. | 2 |
| Batch / scripting | Process ▸ Batch ▸ Macro, Script Editor with ten languages, headless mode, SciJava batch processor ([imagej.net/scripting/batch](https://imagej.net/scripting/batch)). | 1 | `batch.process_folder`, `batch.apply_recorded_macro`, `batch.run_script_on_open_images`, a Python console, a REST API and an ImageJ-macro compatibility runner; Python only, no job queue (RESEARCH §14). | 2 |
| Large-data handling | Virtual stacks, Bio-Formats readers, BigDataViewer for datasets larger than RAM. | 1 | Downsampled fetching for display and virtual folder sequences; a single large TIFF is read fully; OME-Zarr is loaded into memory ([FIJI_PARITY.md](FIJI_PARITY.md) "TIFF Virtual Stack" row; RESEARCH §14). Data larger than RAM is not supported. | 3 |
| Accessibility | Java AWT/Swing; HiDPI support was "experimental" in Fiji 2.7.0 and the setting did not persist (RESEARCH §1). Screen-reader support not evaluated. | 3 | ARIA roles on the tool rail (`toolbar`, `aria-pressed`), sliders (`role="slider"` with ±1 % / ±10 % keyboard steps, covered by `tests/e2e`), status bar, dialogs (`aria-modal`), six `:focus-visible` rules, light and dark themes; HiDPI comes free with the browser. No screen-reader test, no `prefers-reduced-motion` rule, contrast ratios not audited. | 2 |

### 3.3 Evaluator 1: where Lumen is better (with the evidence)

- **One window.** 0 separate windows over ten tasks against 20 (section 2.2); results, ROIs,
  tracks, histogram, history and the console are inspector sections that appear when they have
  content (tasks 2, 3, 7).
- **Palette.** Category, description and shortcut per entry, word-order-free matching, and a
  web-search fallback (task 8) - the Command Finder shows names and menu paths only.
- **Previews.** On by default for plane operations and non-modal: the image, the inspector and
  the keyboard stay usable while a sheet is open (task 6); Fiji's preview is a checkbox inside a
  modal dialog.
- **Undo.** Multi-step undo and redo with a visible list (task 6) against one step and
  "Repeat Command" as the only redo.
- **Recording.** Zero-setup recorder that also covers tracking and batch (tasks 7, 9).
- **Docked panels.** Threshold with *Particles…* (task 2, 4 actions and 0 windows against 7 and
  5); Tracks table with filter/export/delete beside the image (task 7); split view with linked
  navigation on one key (task 10).
- **Keyboard model.** ImageJ's letters kept, ⌘K/Esc/Enter for everything else, custom shortcuts,
  keyboard-operable sliders.

### 3.4 Evaluator 1: where Fiji is better, or Lumen is weaker

- **Ecosystem.** Fiji bundles "nearly 1000 additional plugins" with update sites (RESEARCH §1);
  Lumen's plugin model is `@op` files in `~/.lumen/plugins` with no in-app manager and no
  third-party catalogue.
- **Bio-Formats.** Lumen reads TIFF/OME, ND2, CZI, LIF, OIB/OIF, DICOM, NIfTI, MRC, HDF5/IMS,
  OME-Zarr and more natively, but "not the full Bio-Formats list (e.g. VSI, NDPI, SVS, SLD) and no
  series chooser dialog" ([FIJI_PARITY.md](FIJI_PARITY.md), Bio-Formats Importer row); the
  Bio-Formats macro extensions are the one ❌ row.
- **Script languages and headless use.** Groovy, BeanShell, Jython, JavaScript, Clojure, Ruby,
  Scala, the Script Editor with Git integration and templates, headless and cluster runs; Lumen
  is Python only (FIJI_PARITY Script Editor row) and has no job queue. *Contested by E2 (M40):
  the Script & Recorder editor offers Python, ImageJ macro and JavaScript; the gap is Groovy /
  BeanShell / Jython and script-language plugins, not "Python only".*
- **Validated algorithms.** Several parity rows are "a different algorithm": StackReg/TurboReg
  (phase correlation + features instead of pyramid intensity optimisation), bUnwarpJ
  (optical flow instead of B-splines), SIFT stack alignment, descriptor-based registration
  ([FIJI_PARITY.md](FIJI_PARITY.md) Registration rows). Numbers from those commands will differ
  from Fiji's and have not been cross-validated here.
- **Large data.** Virtual stacks and BigDataViewer have no equivalent (section 3.2).
- **Multi-window workflows.** Some users tile several images and tools across two monitors; Lumen
  panels cannot be torn off into separate OS windows, and the split view compares exactly two
  images.
- **Browser constraints.** File access goes through Lumen's own server-side file browser
  (`FileDialog.tsx`), drag-and-drop or upload rather than the OS file picker; macOS asks whether
  *Python* may access Desktop/Documents (README); clipboard image access needs a browser
  permission; printing is the browser's print dialog; the palette blocks the UI while open.
- **Default tool.** Fiji's default rectangle tool makes the commonest measurement one action
  shorter (task 3). *Withdrawn after E3 (M41): a drag with Lumen's default pointer tool draws a
  rectangle, so the extra key is not needed; the measured count in section 2 pressed `1` and is
  left as measured.*
- **Novelty and learning curve.** Fiji users know where things are, and there is no community,
  tutorial corpus or forum for Lumen; the in-app documentation (section 3.1, heuristic 10) is all
  there is. Discoverability of fourteen inspector sections and of the single-letter tool keys
  has not been tested with anyone.
- **Detection preview.** TrackMate's detector Preview button has no counterpart in Lumen's
  tracking sheet (task 7).
- **Display state is not recorded** (tasks 1, 5a), while Fiji's recorder captures LUT commands.

### 3.5 Evaluator 1: still missing per FIJI_PARITY.md

Of 309 Fiji menu rows, 30 are 🟡 partial and 1 is ❌ missing in
[FIJI_PARITY.md](FIJI_PARITY.md). The partial rows that affect everyday GUI work: TrackMate's
TrackScheme, track splitting/merging and spot-feature filters; Trainable Weka's per-class trace
GUI and classifier choice (random forest only, 2-D and 3-D); the 3D Viewer's landmarks,
orthoslices, animation recording and multi-image scenes; MorphoLibJ's interactive Label Edition
window; Grid/Collection stitching's "positions from file" / unknown positions / 3-D tiles; the
Script Editor's languages and Git integration; TIFF virtual stacks; the Bio-Formats series chooser
and the formats listed above; interactive transform handles (parameter sheets with preview
instead); Capture Screen (a browser cannot capture the desktop); the Color Picker's wheel; Overlay
Options' default fill/label switches; Elastic Montaging, Phase Map, MOPS and
moving-least-squares transforms.

---

### 3.6 Evaluator 2: Fiji power user who scripts everything

*Persona.* Ten years of Fiji; writes IJ1 macros and Jython daily; expects the recorder, a real
script editor, exact command names, and keyboard everything. Walked the app at 1440×900 with the
`power` probe set: palette queries by Fiji name and op id, Gaussian with preview, the typing guard,
the Script & Recorder section (three languages, a macro line, a failing Python line, ⌘R, ⇧R),
custom shortcuts, a filter on a z-stack, Z Project, Process Folder, the Help menu and REST docs,
tracking and the classifier panel, split view, 3-D view, then every menu, section, Preferences and
the help overlay. Evidence: `tests/e2e/usability/walkthrough/power/evidence.json` (43 observations).

| ID | Heuristic | Where | Observation (what the evaluator saw) | Sev. | Evidence |
|---|---|---|---|---|---|
| E2-1 | 5 Error prevention, 4 Consistency | File menu ▸ Close; `?` overlay | The File menu shows **⌘W** next to *Close*, but `shortcuts.ts` has no ⌘W binding and the `?` overlay does not list it: in a browser tab the chord closes the tab (the images survive on the server; the page must be reopened). ⌘W is the Fiji user's most frequent chord. | 3 | `power/18_menu_file.png`; `facts.menus.File` |
| E2-2 | 6 Recognition | ⌘K palette | Fiji's spelling with dots, "Gaussian Blur...", matches nothing except the wiki/forum fallback; without the dots it is first. Op ids ("filters.gaussian") match. "set scale" ranks *Set Scale from Line* above *Properties / Set Scale*; "bin" returns 79 entries. | 2 | `facts.palette_queries` |
| E2-3 | 7 Efficiency, 8 Minimalism | Inspector ▸ Script & Recorder | The editor is a `<textarea>` with a line-number gutter: no syntax highlighting, no completion, eight rows. In the 330 px inspector column the *Recorder* label is clipped to "Recorde" and the language select shows "Py". (Three languages: Python, ImageJ macro, JavaScript; Run / Run selection / ⌘R; Open and Save on the server; download.) | 2 | `power/04_script_panel.png` |
| E2-4 | 10 Help | Help menu | Fifteen entries (Keyboard Shortcuts, Lumen README, Command Reference, Fiji Parity, …) but no link to the REST API documentation the server serves at `/docs` and `/openapi.json`; the script panel's *API help* button inserts a snippet rather than opening a reference. | 1 | `facts.menus.Help` |
| E2-5 | 3 User control | Plugins ▸ Shortcuts ▸ Add Shortcut… | Focus lands in the key-capture field, which swallows Esc (clears the combo and blurs); the dialog is still open after Esc and needs a backdrop click or a second Esc. | 1 | `power/07_shortcuts_dialog.png` |
| E2-6 | Reproducibility | Display section → recorder | Changing a channel's LUT to *Fire* added no recorder line (`.macro-lines` unchanged), while every command did, including `tracking.track`. | 2 | `facts.lut_recorded`, `facts.tracks_recorded` |
| E2-7 | 6 Recognition, 4 Consistency | Process Folder… sheet | *Input folder* and *Output folder* are bare text fields without a Browse… button although the file dialog exists; the pattern is a regex; the script is a plain textarea. | 2 | `power/10_sheet_batch.png` |
| E2-8 | 1 Visibility | Z Project sheet; Track (detect + link) sheet | No Preview switch on whole-image and analysis sheets; the tracking sheet has fourteen fields and *Advanced options* but no detector preview (TrackMate has one). | 2 | `power/09_sheet_zproject.png`, `power/12_sheet_track.png` |
| E2-9 | 9 Errors | Script console | `lumen.run("filters.nope")` prints five frames of Lumen internals before `KeyError: "Unknown command 'filters.nope'"`; no suggestion of similar names. | 1 | `power/05_script_error.png` |
| E2-10 | Reproducibility | Recorder | The recorder is session-wide: lines from other browser tabs or clients (sample opens, measurements) appear in the macro; *Clear* is the only filter. | 2 | `power/04_script_panel.png` |
| E2-11 | 6 Recognition, 8 Minimalism | Tool rail | Four sample "plugin tools" (Remove Label at Point, Classifier Trace Tool, Spot Click Counter, Measure Distance Tool) with paragraph-long developer tooltips ("Shows how a tool=True command receives clicks (x, y, x2, y2, button, …)"). | 2 | `facts.tool_rail` (E3 screenshot `biologist/12_tool_rail.png`) |
| E2-12 | 8 Minimalism | Parameter sheets | Sheets dock over the top-right of the viewport and hide part of the image while a preview is being judged. | 1 | `power/10_sheet_batch.png` |

What worked for this evaluator: the palette matches op ids and ImageJ names; ⇧R repeats the last
command; ⌘R inside the editor runs the script without reloading the page; a letter typed into a
sheet field does not fire the single-letter shortcuts; stack sheets offer an *Apply to* scope with
plane counts ("Entire stack (120 planes)"); custom shortcuts can be bound; the macro line
`run("Gaussian Blur...", "sigma=2")` ran unchanged; every command including tracking was
recorded; the History section lists each step and undo/redo are deep.

### 3.7 Evaluator 3: wet-lab biologist who uses Fiji twice a month

*Persona.* Counts nuclei and measures intensities for the lab's figures; learned Fiji from a
colleague; navigates by menus and remembered dialogs, never scripts; works on a 13-inch laptop.
Walked the app at 1366×768 with the `biologist` probe set: open a sample from the File menu, look
for Brightness/Contrast under Image ▸ Adjust, read the Display section, threshold and count via the
menu and via the section, draw and measure with the default tool, filter from the Process menu with
preview, undo from the Edit menu, close the modified image, Save As, the Workflows section, Help ▸
README, Plugins ▸ Tracking and the Tracks section with its wizard, the 3-D and orthogonal views and
the classifier panel, then every menu, section, Preferences and the help overlay. Evidence:
`tests/e2e/usability/walkthrough/biologist/evidence.json` (39 observations).

| ID | Heuristic | Where | Observation (what the evaluator saw) | Sev. | Evidence |
|---|---|---|---|---|---|
| E3-1 | 6 Recognition, 2 Real world | Image ▸ Adjust submenu | Twelve entries (Color Threshold…, Window/Level…, …, Canvas Size…) but **no Brightness/Contrast…**. The command exists in the palette (category *Image/Adjust*, ⇧⌘C) and opens the Display section. A Fiji user who navigates menus does not find the most-used adjust command. | 3 | `biologist/04_menu_image_adjust.png` |
| E3-2 | 5 Error prevention, 1 Visibility | Analyze ▸ Analyze Particles… | On the unthresholded 3-channel 16-bit image the sheet opened without a warning and *Run* reported "41 particles": an automatic *Default* threshold was applied silently (the code path in `lumen/core/ops/analyze.py`). After an explicit *Auto* threshold the same image gave 82. The Threshold section's own *Particles…* button is correctly disabled without a threshold, so the two entry points disagree. | 3 | `biologist/09_particles_error.png`, `biologist/11_after_particles.png` |
| E3-3 | 5 Error prevention, 3 User control | File ▸ Close; tab × | A blurred image (modified, one undo step) closed immediately: no confirmation, no toast. The only hint of the unsaved state is a small green dot on the tab, with "● modified" in its tooltip. | 3 | `biologist/17_close_modified.png` |
| E3-4 | 8 Minimalism | Inspector at 1366×768 | After Analyze Particles nine sections were open (Display, Threshold, Workflows, 3-D View, Orthogonal Views, ROI Manager, Pixel Classifier, Results, Particle Summary); fully expanded the inspector is 5345 px = 8 screens; the Results table was below the fold right after the command that filled it. | 2 | `biologist/11_after_particles.png`, `biologist/45_inspector_all_open.png` |
| E3-5 | 6 Recognition | Tool rail | 24 icon-only tools whose names appear only in tooltips; the last four are sample plugin tools described in developer language. | 2 | `biologist/12_tool_rail.png` |
| E3-6 | 2 Real world | Pixel Classifier; Threshold; status bar | "background (background) 0 / foreground 0", *Trace mode*, *Train*, *Get probability*, *Create result*, `.joblib`; *Apply → mask*; "fg 255 · bg 0", "mem 4.5 MB", "117 %" in the status bar. The classifier panel has no first-step hint. | 2 | `biologist/27_classifier_panel.png` |
| E3-7 | 4 Consistency | Analyze Particles sheet | Seventeen fields with ImageJ jargon (*Record starts*, *In situ show*, *Composite rois*); the help text explains each; the primary button reads *Run* here and *Apply* on filters. | 1 | `biologist/10_particles_sheet.png` |
| E3-8 | 6 Recognition, 4 Consistency | File ▸ Save As… | Lumen's own file browser opens on the server's working directory (here the source checkout) with Home / Desktop / Documents / Downloads quick links; not the OS dialog, and the default folder means nothing to a biologist. | 2 | `biologist/18_save_as_dialog.png` |
| E3-9 | 3 User control | Display ▸ Auto, then Undo | The Undo button stayed disabled after a display change; *Reset* / *Reset all* are the way back. | 1 | `evidence.json` |
| E3-10 | 6 Recognition, 10 Help | Workflows section | Ten guided workflows (Count nuclei, Measure intensity per cell, Track cells, Train a pixel classifier, Colocalization, …) with step lists - but the section is collapsed by default and nothing on screen points to it. | 1 | `biologist/19_workflows.png`, `biologist/20_workflow_open.png` |
| E3-11 | 8 Minimalism | Toasts | "Undo Gaussian Blur" and "Redo Gaussian Blur" were still stacked at the bottom of the viewport while the Save dialog was open. | 1 | `biologist/18_save_as_dialog.png` |
| E3-12 | 8 Minimalism | Plugins ▸ Tracking | 25 entries (Detect Spots, Link Spots, Track (detect + link), three Manual Tracking items, Split / Merge / Link / Unlink / Add / Delete …); the docked Tracks section with its wizard button is the easier path but appears only on time-lapses. | 1 | `biologist/22_menu_tracking.png` |
| E3-13 | 7 Efficiency (disagreement) | Tool rail, default tool | A drag on the image with the default *Select / move* tool **draws a rectangle** (status bar: "rect x=87 y=149 w=169 h=106"), so Fiji's drag-then-M habit works without pressing `1`. This contradicts Evaluator 1's "default tool" entry in section 3.4; the measured task 3 pressed `1` anyway, so its count is unchanged. | 0 | `biologist/13_drag_default_tool.png` |

What worked for this evaluator: the menus mirror ImageJ's and File ▸ Open Sample lists the
samples; the Threshold section's *Particles…* button is disabled until a threshold exists; the hint
"Display changes never modify pixel values. Use Apply LUT / RGB Color to bake them in."; Edit ▸
Undo / Redo work from the menu; Help ▸ Lumen README renders in-app; the preview is on by
default with a "preview" HUD chip; the empty viewport (from `Viewport.tsx`, not reachable during
the walk because another image was open) offers "Drop an image here", three sample buttons and
"Press ⌘K to search all … commands".

### 3.8 Evaluator 4: accessibility / HCI reviewer

*Persona.* Audits web applications against WCAG 2.2 AA and the ARIA Authoring Practices; uses the
keyboard first and a screen reader second; reads DOM semantics, not pixels. Walked the app at
1440×900 (and once at 640×400) with the `a11y` probe set: landmarks and headings, a 70-stop Tab walk,
menu-bar keyboard operation, the palette's ARIA, sliders and section heads with an image open,
contrast of 29 sampled text styles in both themes, focus management of sheets, the help overlay,
Preferences and the file dialog, the toolbar and tab patterns, motion and the 3-D canvas, reflow,
then every menu, section, Preferences and the help overlay. No screen reader was run; the findings
are from the accessibility tree and computed styles. Evidence:
`tests/e2e/usability/walkthrough/a11y/evidence.json` (33 observations).

| ID | Criterion | Where | Observation (what the evaluator saw) | Sev. | Evidence |
|---|---|---|---|---|---|
| E4-1 | Keyboard (WCAG 2.1.1), ARIA menubar pattern | Menu bar | `role="menubar"` / `menuitem` / `aria-haspopup` are present and Enter on a focused trigger opens the menu, but ArrowDown / ArrowRight do nothing and none of the `.menu-item` elements is focusable (0 of them): the menus are mouse-only. | 3 | `a11y/03_menu_keyboard.png` |
| E4-2 | Name/role/value (4.1.2), keyboard | Inspector section heads | `<div class="section-head" onClick>` with no role, `tabindex` or `aria-expanded`; in a 140-Tab walk the heads were never focused, so a collapsed section cannot be opened from the keyboard; the badge ("82") is plain text, not announced as a count. | 3 | `facts.landmarks_image.sectionHeads`, `facts.tab_walk_image_regions` |
| E4-3 | Name/role/value, keyboard | Image tabs | `<div class="tab" tabindex="-1">` without `role="tab"` / `aria-selected`; no keyboard shortcut switches images (none in `shortcuts.ts` or the `?` overlay); the × button has only `title="Close"`. | 3 | `facts.tab_walk_image_regions`, `toolbar_and_tabs` observation |
| E4-4 | Name/role/value, ARIA combobox/listbox pattern | ⌘K palette | No role on the container, the input or the list; no `aria-activedescendant` or `aria-selected`; ArrowDown moves the highlight while the input keeps focus, so a screen reader hears nothing; the background is not inert; Esc drops focus to `body`. | 3 | `a11y/04_palette_aria.png`, `facts.palette_aria` |
| E4-5 | Contrast (1.4.3) | Inspector, palette, sheets (both themes) | 7 of 29 dark and 8 of 27 light sampled text styles are below AA: hint text 3.83 / 3.07, results cells 3.53 / 2.87, palette category 2.55 / 2.52, palette footer 3.53 / 2.87, input placeholder 3.62 / 4.30, menu triggers 4.43 (dark), primary button 4.36 (light). | 3 | `facts.contrast`; `a11y/07_light_theme.png` |
| E4-6 | Focus order (2.4.3) | Parameter sheet | `role="dialog"` with `aria-labelledby`, but focus stays on `body` when a sheet opens and goes nowhere on Esc; keyboard users Tab from the top of the page to reach the fields. | 2 | `dialogs_focus` observation |
| E4-7 | Modal dialog pattern | Help overlay (`?`) | `role="dialog" aria-modal="true"` and focused on open, but no focus trap: Tab leaves to the menu bar while the overlay is shown. | 2 | `a11y/10_help_focus.png` |
| E4-8 | Info and relationships (1.3.1), 4.1.2 | Preferences (⌘,) | The container has no `role="dialog"` and no accessible name; eight selects / inputs have no label association (visual row labels only). | 2 | `a11y/11_prefs_a11y.png` |
| E4-9 | Keyboard (2.1.1) | File dialog (⌘O) | `role="dialog" aria-modal aria-labelledby` (good), but none of the file / folder rows is focusable; keyboard users must type into the *Folder path* field. | 2 | `a11y/12_file_dialog_a11y.png` |
| E4-10 | Target size (2.5.8) | Everywhere | 161 of 208 visible interactive elements are smaller than 24×24 CSS px (menu triggers 22 px tall, tab × 16×16, small buttons 17-22 px). | 2 | `facts.targets_image` |
| E4-11 | Landmarks / headings (1.3.1, 2.4.1) | Page | No `main` / `banner` / `complementary` / `navigation` landmark, no heading outside overlays, no skip link (`lang="en"` and the title are set). | 2 | `facts.landmarks_empty` |
| E4-12 | Non-text content (1.1.1) | Viewport, histograms, 3-D view | Canvases carry no role or `aria-label`; the image's only text equivalent is the status bar; the 3-D canvas has `tabindex="-1"` and no keyboard orbit. | 2 | `facts.landmarks_image.canvases`, `a11y/14_volume_a11y.png` |
| E4-13 | Animation (2.3.3), 4.1.2 | Stylesheet; 3-D View | No `prefers-reduced-motion` rule in the built CSS; *Auto-rotate* is a plain button without `aria-pressed`. | 2 | `a11y/14_volume_a11y.png` |
| E4-14 | Reflow (1.4.10) | Layout at 640 CSS px | Horizontal scrolling (1018 px document) with a 331 px inspector and a 262 px canvas; nothing reflows. | 2 | `a11y/15_narrow_640.png` |
| E4-15 | Name/role/value | Display, ROI Manager, Results | Three channel-name inputs and two small buttons have no accessible name. | 2 | `facts.unlabeled_image` |
| E4-16 | Toolbar pattern; separators | Tool rail; splitters; tabs | `role="toolbar"` with `aria-pressed`, but arrow keys do not move inside it (24 Tab stops); the inspector splitter and the split-view divider are `role="separator"` with `tabindex="-1"` and no `aria-valuenow`; the modified state is a colour dot whose text is only in the tooltip. | 1 | `toolbar_and_tabs` observation |

What worked for this evaluator: the range sliders (`role="slider"`, `aria-valuenow` /
`aria-valuetext`, ±1 % / ±10 % keys); a visible focus outline on 69 of 70 Tab stops; `aria-live`
on the status bar and the toasts; sheet errors in `role="alert"`, console output in `role="log"`,
the busy dot in `role="status"`; the file dialog's dialog semantics; light and dark themes; HiDPI
rendering from the browser.

### 3.9 Agreement between the four evaluations

Problems were merged by hand: two findings are the same problem when they name the same control
and the same consequence (E1's §3.1-3.4 entries, E2-1…E2-12, E3-1…E3-13, E4-1…E4-16). Items that
E1 listed in section 3.4 without a severity are counted as found ("listed") but do not enter the
merged severity. Two of E1's entries are **contested** by a later evaluator and are listed last.

**Counts.** 41 distinct problems. Found by one evaluator only: 31; by two: 8; by three: 0; by
all four: 2. Each evaluator found between 12 and 16 of the 41 (29 %-39 %; E1 15, E2 12, E3 12,
E4 16), in line with the rule of thumb that a single evaluator finds roughly a third of the
problems a group finds, and the reason NN/g asks for three to five. 13 of the 31
single-evaluator problems are E4's accessibility findings: the specialist found what the others
did not look for, which is the usual argument for including a "double expert".

**Pairwise overlap** (problems both found / problems either found, Jaccard index):

| Pair | Both | Either | Jaccard |
|---|---|---|---|
| E1–E2 | 5 | 22 | 0.23 |
| E1–E3 | 5 | 22 | 0.23 |
| E1–E4 | 3 | 28 | 0.11 |
| E2–E3 | 3 | 21 | 0.14 |
| E2–E4 | 2 | 26 | 0.08 |
| E3–E4 | 2 | 26 | 0.08 |

**Severity.** 10 problems carry a severity-3 rating (M1, M2, M3, M4, M5, M6, M7, M8, M9, M10); none was rated 4. E1 had
rated one of them (M10, large data). The other 9 come from the personas, and all but one of
those (M9) sit in two areas E1 did not exercise: *error prevention around destructive or implicit
actions* (M1, M2, M3) and *keyboard and screen-reader access* (M4-M8). No severity-3 problem was
found by more than one evaluator, which is the expected pattern for problems that need a specific
perspective, and also a reminder that each of them rests on one rating.

**Where the evaluators agree.** The two problems found by three or more evaluators were found
by all four, and both are design decisions rather than defects: the density of the inspector
(M11) and the server-side file browser (M12). The eight problems found by two evaluators are
M13-M20: the sample plugin tools, the recorder's blind spot for display state, missing previews,
vocabulary, motion, undo of display changes, tracebacks and sheet wording.

**Where they disagree.** E3 found that a drag with the default pointer tool draws a rectangle, so
E1's "default tool" entry (M41) is withdrawn; E2 found Python, ImageJ-macro and JavaScript modes in
the editor, so E1's "Python only" (M40) is narrowed to "no Groovy / BeanShell / Jython and no
script-language plugins". E1's ratings are generally one point lower than the personas' for the
same problem (M11, M14, M15, M16): the author of the comparison tables rated with Fiji as the
baseline, the personas rated Lumen on its own.

### 3.10 Merged, severity-ranked problem list

Merged severity is the mean of the ratings given by the evaluators who found the problem (a
post-hoc round in which every evaluator rates every problem, as NN/g recommends, was not done:
with one mind behind all four tables it would add no information). Ranking: merged severity, then
the number of evaluators who found the problem, then ID. "listed" = found by E1 in section 3.4
without a rating; "–" = not found. The top ten are all severity 3; within that group the order is
the editor's judgement of impact (data loss, silent wrong results, a chord every Fiji user types,
then the keyboard and contrast barriers).

| Rank | ID | Problem | Heuristic | Where | E1 | E2 | E3 | E4 | Found by | Merged sev. |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M1 | Closing a modified image discards it without any confirmation (E1 had listed this as *not evaluated*) | 5 | File ▸ Close, tab × | – | – | 3 | – | 1 | 3.0 |
| 2 | M2 | Analyze Particles runs on an unthresholded image with a silent automatic threshold (the Threshold section's own *Particles…* button is correctly disabled) | 5, 1 | Analyze ▸ Analyze Particles… sheet | – | – | 3 | – | 1 | 3.0 |
| 3 | M3 | File ▸ Close advertises ⌘W, which the browser uses to close the tab (no ⌘W binding exists in `shortcuts.ts`) | 5, 4 | File menu, help overlay | – | 3 | – | – | 1 | 3.0 |
| 4 | M4 | Menus are mouse-only: no arrow-key navigation, items not focusable | A11y (WCAG 2.1.1) | menu bar | – | – | – | 3 | 1 | 3.0 |
| 5 | M5 | Inspector section heads are not keyboard-reachable and carry no expanded/collapsed semantics | A11y (4.1.2) | inspector | – | – | – | 3 | 1 | 3.0 |
| 6 | M6 | Image tabs are not keyboard-reachable, have no tab semantics, and no shortcut switches images | A11y (4.1.2) | image tabs | – | – | – | 3 | 1 | 3.0 |
| 7 | M7 | Command palette has no ARIA roles or active-descendant; results are silent for screen readers | A11y (4.1.2) | ⌘K palette | – | – | – | 3 | 1 | 3.0 |
| 8 | M8 | Text contrast below WCAG AA for hints, results cells, palette category/footer and placeholders, in both themes (E1 had noted *contrast ratios not audited*) | A11y (1.4.3) | inspector, palette, sheets | – | – | – | 3 | 1 | 3.0 |
| 9 | M9 | Brightness/Contrast is missing from Image ▸ Adjust (palette and ⇧⌘C only) | 6, 2 | Image ▸ Adjust submenu | – | – | 3 | – | 1 | 3.0 |
| 10 | M10 | Data larger than RAM is not supported (no virtual stacks, no BigDataViewer equivalent) (not exercised by the persona walks) | Large data | file readers | 3 | – | – | – | 1 | 3.0 |
| 11 | M12 | Server-side file browser instead of the OS picker; batch sheets take bare path text fields without a Browse button | 4, 6 | File dialog, Process Folder… sheet | listed | 2 | 2 | 2 | 4 | 2.0 |
| 12 | M13 | Sample plugin tools sit in the tool rail with developer-prose tooltips; 24 icon-only tools | 6, 8 | tool rail | – | 2 | 2 | – | 2 | 2.0 |
| 13 | M17 | No prefers-reduced-motion rule; 3-D auto-rotate state not exposed; 3-D canvas not keyboard-operable | A11y (2.3.3) | stylesheet, 3-D View | 2 | – | – | 2 | 2 | 2.0 |
| 14 | M21 | Script editor is a plain textarea (no highlighting or completion), eight rows in a 330 px column with clipped labels | 7, 8 | Script & Recorder | – | 2 | – | – | 1 | 2.0 |
| 15 | M22 | Palette misses Fiji names typed with a trailing '…' and ranks some ImageJ synonyms low | 6 | ⌘K palette | – | 2 | – | – | 1 | 2.0 |
| 16 | M23 | Recorder is session-wide: commands from other tabs or clients appear in the macro | Reproducibility | recorder | – | 2 | – | – | 1 | 2.0 |
| 17 | M24 | Parameter sheet does not take focus on open nor return it on close; palette Esc drops focus to body | A11y (2.4.3) | sheets, palette | – | – | – | 2 | 1 | 2.0 |
| 18 | M25 | Help overlay is aria-modal but has no focus trap | A11y | ? overlay | – | – | – | 2 | 1 | 2.0 |
| 19 | M26 | Preferences dialog has no dialog role or name and eight unlabeled controls; channel-name inputs and two inspector buttons unlabeled | A11y (1.3.1, 4.1.2) | Preferences, Display, ROI Manager, Results | – | – | – | 2 | 1 | 2.0 |
| 20 | M27 | File dialog rows are not focusable (keyboard users must type paths) | A11y (2.1.1) | File dialog | – | – | – | 2 | 1 | 2.0 |
| 21 | M28 | 161 of 208 interactive elements are smaller than 24×24 CSS px | A11y (2.5.8) | everywhere | – | – | – | 2 | 1 | 2.0 |
| 22 | M29 | No landmarks, headings or skip link; canvases have no text alternative | A11y (1.3.1, 1.1.1) | page, viewport | – | – | – | 2 | 1 | 2.0 |
| 23 | M30 | Layout does not reflow below ~1000 CSS px (horizontal scroll at 200 % zoom) | A11y (1.4.10) | layout | – | – | – | 2 | 1 | 2.0 |
| 24 | M31 | No plugin manager or catalogue; no community corpus, tutorials or videos | 10 | Help, plugins | 2 | – | – | – | 1 | 2.0 |
| 25 | M32 | No job queue for batch runs | Batch | batch commands | 2 | – | – | – | 1 | 2.0 |
| 26 | M11 | Inspector density: up to 13 sections, 7-8 screens when expanded, Results below the fold on a laptop, sheets overlay the viewport | 8 | inspector, sheets | 2 | 1 | 2 | 2 | 4 | 1.8 |
| 27 | M14 | Display state (ranges, LUTs, composite mode) is never recorded | Reproducibility | Display section, recorder | 1 | 2 | – | – | 2 | 1.5 |
| 28 | M15 | No preview for whole-image or analysis commands; no spot-detection preview before tracking | 1 | Z Project, Track sheets | 1 | 2 | – | – | 2 | 1.5 |
| 29 | M16 | Expert vocabulary: Apply → mask, LUT, traces / estimator / .joblib, 'fg 255 · bg 0', 'mem' | 2 | Threshold, Pixel Classifier, status bar | 1 | – | 2 | – | 2 | 1.5 |
| 30 | M18 | Display changes are not undo steps (Reset is the only way back) | 3 | Display section, Undo | 1 | – | 1 | – | 2 | 1.0 |
| 31 | M19 | Script errors are raw Python tracebacks with Lumen internals | 9 | script console | 1 | 1 | – | – | 2 | 1.0 |
| 32 | M20 | Apply vs Run wording and 17-field analysis sheets with ImageJ jargon | 4 | parameter sheets | 1 | – | 1 | – | 2 | 1.0 |
| 33 | M34 | No link to the REST API docs (/docs) from the Help menu | 10 | Help menu | – | 1 | – | – | 1 | 1.0 |
| 34 | M35 | Add Shortcut: Esc in the key-capture field clears it instead of closing the dialog | 3 | Plugins ▸ Shortcuts ▸ Add Shortcut… | – | 1 | – | – | 1 | 1.0 |
| 35 | M36 | Guided Workflows are collapsed by default and nothing on screen points to them | 6, 10 | Workflows section | – | – | 1 | – | 1 | 1.0 |
| 36 | M37 | Toasts stack and persist behind dialogs | 8 | toasts | – | – | 1 | – | 1 | 1.0 |
| 37 | M38 | Plugins ▸ Tracking lists 25 commands | 8 | Plugins ▸ Tracking | – | – | 1 | – | 1 | 1.0 |
| 38 | M39 | Tool rail ignores arrow keys; splitters not keyboard-resizable; modified state is a colour dot with text only in the tooltip | A11y | tool rail, splitters, tabs | – | – | – | 1 | 1 | 1.0 |
| 39 | M33 | Panels cannot be torn off to a second monitor; the split view holds exactly two images | 7 | layout | listed | – | – | – | 1 | n/r |
| 40 | M40 | Python-only scripting (**contested**: E2 found Python, ImageJ-macro and JavaScript modes in the editor) | 7 | Script & Recorder | 1 | – | – | – | 1 | 1.0 |
| 41 | M41 | Pointer is the default tool, so measuring costs one more action than in Fiji (**contested**: E3 found that a drag with the pointer tool draws a rectangle) | 7 | tool rail | listed | – | – | – | 1 | n/r |

### 3.11 Evaluators 5-7: three further evaluations by other models

The three evaluations in this section were added after the fixes of §5.3, on the build of 8 Nov
2026 (493 registered commands). They differ from E1-E4 in one respect that matters for the
agreement figures: each was performed by a *different model* of the Claude family, running as a
separate agent that had not read this document, the earlier evaluations or Lumen's source.
They are still AI evaluations, not people (§1.1 and §5.1 say what that means), and all three
received the same brief - the same task list, Nielsen's ten heuristics plus the five criteria of
§1.1, severity 0-4, a persona to adopt, and the instruction to write down only what they could
show in a screenshot or a DOM fact. Each drove the running server in its own headless Google
Chrome (Playwright, channel `chrome`, 1440×900) and wrote a structured file, committed as
`tests/e2e/usability/walkthrough/{e5_sonnet,e6_haiku,e7_opus}/evaluation.json` (problems with
heuristic, location, observation, severity and the screenshot names; positives; the overall
verdict). Their observations are reproduced below, shortened to one sentence each; the files
hold the full text.

Two conditions of the runs are stated by the evaluators themselves and should be read with the
tables. The server was shared, so E5 and E7 started with another session's images, a global
threshold and 41 ROIs already present ("a true first-run experience could not be evaluated",
E5); E7 began by closing everything. And the tracking step used a noise movie made with
File ▸ New, so the wizard's detection results mean nothing beyond the wizard's behaviour.

| | Evaluator (model) | Persona | Severities given | Problems | Positives |
|---|---|---|---|---|---|
| E5 | Claude Sonnet 5.5 | microscopy core-facility manager who trains biologists on Fiji; judges novice time-to-productivity and irreversible-mistake risk | 3 ×5, 2 ×10, 1 ×5 | 20 | 10 |
| E6 | Claude Haiku 5.5 | PhD cell biologist, two years of Fiji through the menus, never scripted, impatient | 3 ×1, 2 ×5, 1 ×8 | 14 | 14 |
| E7 | Claude Opus 5.5 | image-analysis specialist in a bioimaging core facility who writes Fiji macros and Python daily, cares about reproducibility, provenance and ImageJ-identical results; has used napari, QuPath and Icy | 4 ×1, 3 ×10, 2 ×10, 1 ×6 | 27 | 14 |

#### E5 (Sonnet): core-facility manager

| ID | Sev. | Heuristic | Where | Problem |
|---|---|---|---|---|
| E5-1 | 3 | 2, 6 | Image ▸ Lookup Tables; palette | The submenu lists only operations and no LUT names; LUTs are chosen from a ~75-entry native select without swatches; "fire" in the palette runs *Proxy Settings…* on Enter and "viridis" finds nothing. A trainee following "Image ▸ Lookup Tables ▸ Fire" cannot. |
| E5-2 | 3 | 4, 5 | palette; Lookup Tables | Two commands named *Apply LUT*: one changes the channel's display LUT, the other bakes the display range into the pixels; only the description differs. "LUT…" appears twice (Import, Export). |
| E5-3 | 2 | 5, 9 | palette | Fuzzy ranking puts irrelevant hits first and Enter runs the first hit without a weak-match hint; no typo tolerance ("clahee" finds nothing although "clahe" and "contrast limited adaptive" work). |
| E5-4 | 2 | 2, 4 | palette | "Open Sample" ranks *Open ImageJ Sample…* (a network download) above *Open Sample* (bundled); the opened sample is called "synthetic_cells.tif-1", which says nothing about provenance. |
| E5-5 | 3 | 6, 1 | Analyze Particles sheet; Threshold section | The missing-threshold error names a fix (*Auto-threshold if needed*) that is hidden under the collapsed Advanced options, is drawn over the sheet's Cancel button and fades; the Threshold section never says which channel is thresholded - the red overlay covered a channel that was hidden in the composite. |
| E5-6 | 2 | 5 | Display section | A channel maximum below its minimum is accepted silently: the sliders cross, the ramp inverts and the channel disappears from the composite. |
| E5-7 | 3 | 1, reproducibility | Script & Recorder | Commands are recorded with complete parameters, but display edits (ranges, LUT, visibility, composite mode) and the drawn rectangle are not; lines carry no image identity; one global log across images and reloads with only *Clear*. The replayed script will not reproduce the saved figure. |
| E5-8 | 2 | 1, provenance | History; Undo | Undo is deep (30 of 30 inverts undone, memory-budgeted) but History shows command names only (no sigma, no scope); display changes never enter History and leave Undo disabled. |
| E5-9 | 1 | 5 | Gaussian Blur sheet | On a 10-slice stack the default scope is *Entire stack* while the preview shows one plane; Fiji asks. The stack run was recorded without its scope. |
| E5-10 | 2 | 9 | toasts for selection-dependent commands | "Crop needs a area selection" (grammar), no next step, fades; the palette lists Crop, Fit Ellipse, Straighten as enabled although the menu greys them; toasts accumulate in the live region. |
| E5-11 | 2 | 4, 1 | ROI Manager; Results | Both are global: 41 polygons and 42 rows from one image are shown on an unrelated stack with no source column; names like "0001-0025-0315"; numeric columns are truncated in the narrow inspector. |
| E5-12 | 2 | 1, 8 | threshold badge and overlay | Threshold state survives tab switches, reloads and the split view (a solid red rectangle on a fresh load, "threshold 148-3328"); the pink badge text is under 3:1 in the light theme; on a shared server a new user sees someone else's leftover state. |
| E5-13 | 2 | 1 | Tracks wizard | "0 spots in 0 frames" still earns a green tick and *Next* walks through every later step; steps are not gated; the radius field shows "2,5". |
| E5-14 | 1 | 4 | Preferences; Tracks wizard | Decimal commas in some fields with an en-US browser ("1,5" reads as 1500) against points elsewhere. |
| E5-15 | 3 | 7, keyboard | Script & Recorder editor | "[" moves focus into the editor and Esc does not leave it, so every later shortcut is typed into the script (*Kimg = lumen.image* in the run); the language select is truncated to "Pyt" and the *Recorder* label is clipped. |
| E5-16 | 2 | accessibility | tool rail; tabs; chips; New Image | Tab order passes ~25 tool buttons (no roving tabindex); 81 targets under 24 px; five unnamed controls (three channel-name inputs, two icon buttons); *Slices* and *Frames* unlabelled; menubar text at 4.43:1. |
| E5-17 | 1 | 7 | menubar via F10 | Esc closes the menu but leaves focus on the trigger with no one-key way back to the image; the Image menu has 118 items across submenus. |
| E5-18 | 2 | 5, large data | New Image sheet | 20000×20000 32-bit shows no size estimate or warning before Apply (1.6 GB); the progressive-preview badge "1/8" on an 8000×8000 image is unexplained; *Slices* / *Frames* have no range hint. |
| E5-19 | 1 | 10, 8 | inspector; help overlay; 3-D view | Up to 14 sections, no first-run guidance, *Workflows* collapsed by default; the overlay clips its last group; the 3-D controls are icon-only in a 305 px canvas. |
| E5-20 | 1 | reflow | window narrower than 1000 px | Horizontal scroll, clipped Preferences / Help icons, the image is not re-fitted after a resize. |

*Positives (E5).* The close-modified dialog (alertdialog, Cancel focused, destructive button in red,
"don't ask" preference); deep memory-budgeted undo; preview on by default with progressive tiling
on an 8000×8000 image; the palette as "the strongest discovery tool" (CLAHE in one keystroke
group, synonyms, "5 of 493", ARIA combobox); consistent non-modal sheets; the fully parameterised
recorder and *Apply Recorded Macro to Folder*; clear multi-channel handling with the sentence
"Display changes never modify pixel values"; Z Project as a derived image; the keyboard menubar;
visible focus rings; the help overlay; the server-side file browser with shortcuts and upload.
*Verdict (E5, shortened):* "unusually capable and coherent for a browser Fiji … I would be
comfortable letting a novice use it for basic measure-and-threshold work because the
irreversible-mistake surface is small. The problems cluster in discoverability and traceability
rather than in core function."

#### E6 (Haiku): cell biologist who never scripts

| ID | Sev. | Heuristic | Where | Problem |
|---|---|---|---|---|
| E6-1 | 2 | 4, 2 | palette | Two near-identical sample commands; "Open Sample" ranks the networked *Open ImageJ Sample…* first, so Enter opens the wrong one. |
| E6-2 | 2 | 5, 9 | Analyze Particles sheet | The error text is useful but the red toast covers the sheet's Cancel button and the fix is prose only - no inline button. |
| E6-3 | 3 | 1, 5, reproducibility | ROI Manager; Results | The lists are not scoped to the active image: a blank 20-frame image still shows "ROI MANAGER (41)" and rows labelled synthetic_cells.tif-1, so a table can silently belong to another image. |
| E6-4 | 1 | 2, 9 | close-image dialog | "It has 1 unsaved change (the undo history)" calls the undo history an unsaved change; the dialog itself is clear and does not default to the destructive action. |
| E6-5 | 1 | 7, 6 | tab strip | Close buttons appear only on hover; the modified marker is a small dot. |
| E6-6 | 2 | 4 | numeric fields | Decimal commas in an English UI ("0,01", "2,5", "1,5") against "." in the status bar. |
| E6-7 | 2 | 1, 6 | Edit ▸ Undo | The item says "Undo ⌘Z" without naming the step. |
| E6-8 | 1 | 8, 4 | palette, query "LUT" | Duplicate entries ("LUT…" twice, "Apply LUT" twice) with no grouping, "which makes the palette look broken". |
| E6-9 | 2 | 7, 8 | Display LUT select | A flat native select of about 70 entries without groups; the palette is no faster. |
| E6-10 | 1 | WCAG 1.4.3 | menubar triggers | 13 px light grey on grey at 4.43:1; the only group below AA in a computed pass over 477 text nodes. |
| E6-11 | 1 | 8 | Results table | Wider than the inspector; later columns clipped without a scroll cue. |
| E6-12 | 1 | 1, 7 | Tracks section | On a time series the panel is below the fold and nothing in the viewport points to it. |
| E6-13 | 1 | 10, 6 | shortcut overlay | Small key chips, dim descriptions; scrolling needed for the last groups. |
| E6-14 | 1 | 9 | missing-selection toast | "Plot Profile needs a selection" without saying how (line tool, `5`); no persistent log entry. |

*Positives (E6).* Every core task completed without scripting; the palette finds CLAHE at once;
one sheet layout everywhere; Z Project keeps the original; Measure fills a row with the rectangle
shown live in the status bar; undo with History and the recorder "supports reproducibility";
split view; the time-series scrubber and the Track wizard's defaults; F10 keyboard menus and 23
labelled toolbar buttons; the unsaved-changes dialog; toast errors instead of modals; autosaved
preferences; the status bar. *Verdict (E6, shortened):* "well organized for a Fiji user … The
most serious problem is state scope: the ROI Manager and Results tables are global rather than per
image." Not evaluated by E6: large data, undo depth beyond one step, batch, Analyze Particles on a
thresholded image.

#### E7 (Opus): image-analysis specialist who scripts daily

| ID | Sev. | Heuristic | Where | Problem |
|---|---|---|---|---|
| E7-1 | 4 | reproducibility, 5 | recorder | Tab switches are not recorded and `file.close` has no target: after *file.new → detect spots → close blobs → click MAX_t1-head → Invert, Smooth, Median, Subtract Background* nothing in the recording selects MAX_t1-head, so a replay or folder batch runs the filters on whichever image is active. Fiji records selectImage() / close("title"). |
| E7-2 | 3 | reproducibility, provenance | Undo vs recorder | Undo and History jumps are not recorded: after undoing a Gaussian the script still ends with it. |
| E7-3 | 3 | reproducibility, 4 | Display section vs Apply LUT sheet | Inspector display changes (DAPI max 1500, Auto, hide Spots, LUT select Viridis) are recorded nowhere, while the sheet route records `color.lut`; same action, two provenance outcomes. |
| E7-4 | 3 | reproducibility, 1 | recorder pane | One global server-persisted log: about 65 foreign lines after a clean start; "→ Python" put 104 lines in the editor; no session marker, timestamps or auto-scroll. |
| E7-5 | 3 | 5, batch | Apply Recorded Macro to Folder | The sheet never shows the commands it will run; with E7-1/2/4 a batch would apply undone and foreign steps to every file. |
| E7-6 | 3 | 5, 3 | ROI Manager / Results *Clear* | Deletes all 41 ROIs or 43 rows without confirmation, undo or a recorded line; Fiji asks and offers to save. |
| E7-7 | 3 | 5, 1 | numeric limits | Out-of-range values are clamped silently: sigma 500 ran as 200 (visible only in the recorder), width 30000 became 20000; the field is HTML-invalid but Apply stays enabled. Fiji accepts sigma 500. |
| E7-8 | 3 | 5 (API), 9 | Python editor | A misspelled keyword (`sigmaa`) runs silently with the default; an unknown command id gives a raw traceback with internal paths and no "did you mean". |
| E7-9 | 3 | 4, 2 | Lookup Tables; palette | Two commands named *Apply LUT…*, one destructive; the submenu lists no LUTs; the LUT sheet has no preview. |
| E7-10 | 2 | 2, 6 | palette | Fiji's own names are unknown ("enhance local contrast", "bleach"); "Open Sample" ranks the ImageJ sample first; the header says 4 results and the footer "6 of 493"; fuzzy noise ("clahe" → Colocalization). |
| E7-11 | 2 | 1, reproducibility | Threshold; Analyze Particles; Measure | Nothing says which channel is acted on; result rows carry no channel; the recorded `threshold.compute` has no channel argument. |
| E7-12 | 2 | 8, 4, contrast | menubar | The `.menubar > .menu-trigger` rule never applies (each trigger is wrapped in a div), so the triggers render as default buttons with no hover or open state and 4.43:1 text. |
| E7-13 | 2 | 4, 8 | File menu; LUT select | Duplicates and mixed hierarchy ("Open Recent" submenu and "Open Recent…", "Export" twice, "New…" plus a "New" submenu); the 60-entry LUT list is unsorted with near-duplicates (Viridis / mpl-viridis, Cool / cool). |
| E7-14 | 3 | 7 | script editor | A 283×147 px textarea: no highlighting, completion or tabs; language select ("Pyth") and filename truncated; horizontal scroll on focus; the splitter is an invisible 9 px mouse-only strip; the recorder cannot emit ImageJ macro language. |
| E7-15 | 2 | 3 | → Python / → JavaScript / API help | Replace the editor's content without asking; ⌘Z does not bring the text back. |
| E7-16 | 2 | provenance | History; saved TIFF; Image Info | History shows names only (two identical "Gaussian Blur" rows), rows are not keyboard-reachable; the saved TIFF carries no processing log or source and the ImageJ version string is misleading; Image Info lists *Path* but not where the image came from. |
| E7-17 | 2 | 4, macro parity | Save As | Tab, Image Info and `getTitle()` keep the old title after Save As; Fiji renames the window. |
| E7-18 | 2 | 4, 5, 9 | palette vs menu | The palette offers Crop as enabled while the menu greys it; "a area selection"; "needs a selection" does not say which kind or which tool key. |
| E7-19 | 2 | ARIA / keyboard | statusbar; Display; splitter; 3-D view | The cursor readout sits in an aria-live status region (constant announcements); sliders and eye toggles lack channel names and aria-pressed; aria-valuenow exposes floats like 3068.599999999999; the splitter is not focusable; the large 3-D view is not a dialog. |
| E7-20 | 2 | 10 | sheet "?"; overlay | Command help lists parameter names only - no meaning, default, unit or macro equivalent; internal Design / Research documents are exposed in the same dialog; no conceptual help ("Composite rois", "scope"). |
| E7-21 | 3 | 5, deployment | server-side state | ROIs, Results and preferences persist on the server for everyone; the file dialog browses the server's home; Python is executed on the server. "Fine for one person on a laptop, risky for my use case." |
| E7-22 | 1 | 1 | split view | The compare pane opens at its own 149 % zoom instead of fitting the half-width pane. |
| E7-23 | 1 | 4 | "[" key | Opens the editor or changes the brush size depending on the tool; the overlay lists both without saying so. |
| E7-24 | 1 | 4, 1 | Tracks wizard | "1 spots in 1 frame"; different quality figures in steps 1 and 2; the detected spot is not drawn; seven steps squeezed into 330 px. |
| E7-25 | 1 | 2 (Fiji) | Z Project; Gaussian Blur | *Stop* shows 0 with "0 = last slice" instead of the slice count; no *Scaled units* option. |
| E7-26 | 1 | 4 | number inputs | Browser-locale decimal commas against "." in Results, recorder and status bar. |
| E7-27 | 1 | 8 | tool-options bar | Choosing a drawing tool inserts a ~30 px bar and pushes the canvas down, so the click target moves. |

*Positives (E7).* Fiji-familiar dialogs (the full Analyze Particles option set, MAX_ naming,
Particle Summary); explicit keyword-argument recorder lines; ImageJ macro language "really runs in
the editor" (getTitle, run("Gaussian Blur…"), setAutoThreshold, nResults); per-image multi-level
undo with History, persistence across reloads and a modified dot; preview by default; close
dialogs with Cancel focused and a single *Close 4 images?* for Close All; the ARIA combobox
palette with web fallback; a menubar that "works fully from the keyboard" (F10, roving tabindex,
submenus, type-ahead, Home/End, Esc one level at a time, focus returned to the trigger);
aria-modal dialogs with focus traps; the status bar with calibrated coordinates and all channel
values; channel handling; inline 3-D, Orthogonal Views, a TrackMate-style wizard and guided
Workflows without plugin installs; batch reach; ImageJ-hyperstack TIFFs with LUTs and ranges.

*E7's comparison.* Better than Fiji: palette discoverability, multi-level visual undo, consistent
preview and one tabbed window, readable recorder output, keyboard / ARIA support, built-in 3-D,
tracking, classifier and workflows, Python and IJM in one editor. Worse than Fiji: recorder
fidelity (image switches, close targets, undo, display changes, clears - "the core of
reproducibility"), no IJM export of the recording, unguarded clears, silent clamping and silently
ignored keywords, Fiji vocabulary that does not carry over, a weak script editor, no plugin
ecosystem or update sites. Against napari: Lumen has the operations library, ROI Manager, Results,
recorder and IJM interpreter out of the box and needs no Python environment; napari has an
in-process console with live arrays, a layer model that keeps derived data attached to its source,
and first-class 3-D. *Verdict (E7, shortened):* "impressive and pleasant … For my facility
persona, though, it fails the property I care about most: the recorded script is not a faithful
account of what I did. … Fix image targeting and undo in the recorder, scope the recorder per
session, record or explicitly mark display changes, validate parameters loudly, guard the
destructive clears, and resolve the naming collisions. Then Lumen would be better than Fiji for
most of my daily interactive work."

### 3.12 Agreement among E5-E7, and with the first round

The 61 observations were merged by the same rule as in §3.9 (same control and same consequence):
42 distinct problems, listed as N1-N42 in §3.13. Where one evaluator's observation covers two
problems (E5-5, E5-7, E5-16, E7-9, E7-10, E7-13, E7-19, E7-21) it is counted under each, marked
"part" in the table.

**Counts.** Found by one evaluator only: 23; by two: 12; by all three: 7. E5 found 25 of the 42
(60 %), E6 12 (29 %), E7 31 (74 %). The seven problems all three found are the LUT discoverability
pair (N1, N2), the "Open Sample" ranking (N4), the wording and palette state of selection-dependent
commands (N18), ROI Manager and Results being global across images (N19), decimal commas (N22)
and the menubar triggers (N26).

**Pairwise overlap** (Jaccard index, as in §3.9):

| Pair | Both | Either | Jaccard |
|---|---|---|---|
| E5–E6 | 9 | 28 | 0.32 |
| E5–E7 | 17 | 39 | 0.44 |
| E6–E7 | 7 | 36 | 0.19 |

The overlap is higher than between E1-E4 (0.08-0.23), and seven problems were found by everyone
against two in the first round. Three reasons, in decreasing order of weight: the three received
one brief with one task list, so they looked at the same screens in the same order; the build
they saw had already lost the first round's problems, so what remained was more concentrated;
and the same vendor's models share training and habits - E5 and E7 produced near-identical
sentences about the two *Apply LUT* commands. The figures therefore still overstate the
independence of human evaluators, but less than §3.9's did, because the shared mind of §5.1 is
now three minds with one brief rather than one mind with four briefs.

**Severity.** Nine merged problems are rated 3 or higher: N9 (3.5, the only severity-4 rating of
the whole assessment, E7-1), N8, N10, N11, N12, N13, N14, N15 and N23 (3.0). Eight of the nine
are *reproducibility and API strictness* problems that the first round had either rated minor
(display state never recorded, M14, 1.5; the session-wide recorder, M23, 2.0) or not found at all
(image targeting, undo, batch preview, unguarded clears, silent clamping, silently ignored
keywords). Five of the nine rest on E7 alone. This repeats the §3.9 pattern: the evaluator with the
specialist perspective finds the specialist's problems, and the ratings that matter most are
single ratings.

**Verification of the first round's fixes.** None of the three evaluators had read §3.10 or
§5.3, so their positives are independent evidence that the fixes of 7 Nov hold: the
close-modified dialog (M1: E5, E6, E7), Analyze Particles refusing an unthresholded image (M2:
E5, E6 - and the new complaint N5 is about how that refusal is presented), ⌘W (M3: E5, E6), the
keyboard menubar with roles (M4: all three; E7 confirms type-ahead, Home/End and Esc returning
focus), inspector section buttons (M5: not mentioned as a problem by anyone), image tabs with
ARIA state (M6: E6), the palette as an ARIA combobox (M7: E5, E7) and Brightness/Contrast under
Image ▸ Adjust (M9: not mentioned). Two claims in §5.3 did **not** survive: M8 ("every text and
UI contrast pair now meets WCAG AA") missed the menubar triggers, which all three evaluators
measured at 4.43:1 because, as E7-12 found, the menubar's CSS rule never applied and the
contrast script had checked the token pair rather than the rendered button; and M26 ("every
control … carries an accessible name") covered Preferences and the sheets but not the Display
section's channel-name inputs or two inspector icon buttons (E5-16, E7-19). Twelve of the 42
problems are re-findings of first-round items (N3, N8, N10, N15, N16, N23, N25, N26, N29, N30,
N32, N38); 30 are new.

**Where they disagree.** E5-17 reports no evidence of type-ahead in the menubar; E7 used it. E5
counts the Esc behaviour (focus stays on the trigger) as a problem, E7 lists the same behaviour
as correct ARIA practice; both are right, and the fix is a second Esc that returns to the image.
E6 rates the shared ROI Manager as the most serious problem of the whole interface (3); E5 rates
the same observation 2 and E7 folds it into the shared-server problem (N38). The severity-3
ratings of E5 (LUT names, Apply LUT, the Analyze Particles toast, the recorder, the editor focus
trap) are two-thirds about discoverability; E7's eleven ratings of 3 or more are nine-tenths about reproducibility.
Same screens, different stakes - which is what a persona brief is for.

### 3.13 Merged, severity-ranked list of the second round

Merged severity is the mean of the ratings given by the evaluators who found the problem (§3.10's
rule). Ranking: merged severity, then number of evaluators, then ID. The last column names the
first-round item the problem repeats, if any; "–" means the problem is new. Which of these were
fixed afterwards is in §5.4.

| Rank | ID | Problem | Where | E5 | E6 | E7 | Found by | Merged sev. | First round |
|---|---|---|---|---|---|---|---|---|---|
| 1 | N9 | Recorded lines carry no image identity; tab switches and close targets are not recorded, so a replay can process the wrong image | recorder | 3 (E5-7 part) | – | 4 (E7-1) | 2 | 3.5 | – |
| 2 | N8 | Display edits (ranges, LUT, visibility, composite mode) are never recorded | Display section; recorder | 3 (E5-7 part) | – | 3 (E7-3) | 2 | 3.0 | M14 (1.5) |
| 3 | N10 | The recorder is one global, server-persisted log: no session marker, start / stop or auto-scroll | recorder pane | 3 (E5-7 part) | – | 3 (E7-4) | 2 | 3.0 | M23 (2.0) |
| 4 | N23 | Script editor: "[" enters it and Esc does not leave; cramped textarea with truncated labels, no highlighting, no IJM export | Script & Recorder | 3 (E5-15) | – | 3 (E7-14) | 2 | 3.0 | M21 (2.0) |
| 5 | N11 | Undo and History jumps are not recorded | Undo; recorder | – | – | 3 (E7-2) | 1 | 3.0 | – |
| 6 | N12 | *Apply Recorded Macro to Folder* never shows the lines it will run | batch sheet | – | – | 3 (E7-5) | 1 | 3.0 | – |
| 7 | N13 | ROI Manager / Results *Clear* without confirmation, undo or recording | inspector | – | – | 3 (E7-6) | 1 | 3.0 | – |
| 8 | N14 | Out-of-range parameters are silently clamped | sheets; API | – | – | 3 (E7-7) | 1 | 3.0 | – |
| 9 | N15 | Misspelled keywords are silently ignored; unknown commands give raw tracebacks | scripting API | – | – | 3 (E7-8) | 1 | 3.0 | M19 (1.0) |
| 10 | N1 | No LUT names in Image ▸ Lookup Tables; "fire" / "viridis" fail in the palette; the Display select is flat and unsorted with near-duplicates | Lookup Tables; palette; Display | 3 (E5-1) | 2 (E6-9) | 3 (E7-9 part, E7-13 part) | 3 | 2.7 | – |
| 11 | N19 | ROI Manager and Results are global across images, with cryptic ROI names and a clipped table | inspector | 2 (E5-11) | 3 (E6-3, E6-11) | 3 (E7-21 part) | 3 | 2.7 | – |
| 12 | N5 | The Analyze Particles error hides its fix under Advanced options, covers Cancel and offers no inline action | Analyze Particles sheet | 3 (E5-5 part) | 2 (E6-2) | – | 2 | 2.5 | M2 (fixed; presentation) |
| 13 | N6 | Threshold, Measure and Analyze Particles never say which channel they act on, and the recorded line has no channel | Threshold section; sheets | 3 (E5-5 part) | – | 2 (E7-11) | 2 | 2.5 | – |
| 14 | N38 | Server-global state (ROIs, Results, threshold, preferences, file system, code execution) on a shared server | server | 2 (E5-12 part) | – | 3 (E7-21) | 2 | 2.5 | M12 (2.0) |
| 15 | N2 | Two commands named *Apply LUT*, one destructive; duplicate "LUT…" palette entries | palette; Lookup Tables | 3 (E5-2) | 1 (E6-8) | 3 (E7-9 part) | 3 | 2.3 | – |
| 16 | N4 | "Open Sample" ranks the networked *Open ImageJ Sample…* first | palette | 2 (E5-4) | 2 (E6-1) | 2 (E7-10 part) | 3 | 2.0 | – |
| 17 | N3 | Palette ranking: weak fuzzy hits run on Enter, no typo tolerance, Fiji names are not aliases, header and footer counts differ | palette | 2 (E5-3) | – | 2 (E7-10 part) | 2 | 2.0 | M22 (2.0) |
| 18 | N16 | History shows no parameters; display changes are not undoable; rows not keyboard-reachable; saved files carry no provenance | History; Image Info; TIFF | 2 (E5-8) | – | 2 (E7-16) | 2 | 2.0 | M18 (1.0) |
| 19 | N25 | Tool rail without roving tabindex; targets under 24 px; unnamed channel-name inputs, icon buttons, Slices / Frames; unnamed per-channel sliders | tool rail; Display; New Image | 2 (E5-16) | – | 2 (E7-19 part) | 2 | 2.0 | M26, M28, M39 |
| 20 | N7 | A display maximum below the minimum is accepted silently | Display section | 2 (E5-6) | – | – | 1 | 2.0 | – |
| 21 | N20 | Threshold state persists across tabs, reloads and the split view as a solid overlay; badge contrast in the light theme | viewport | 2 (E5-12) | – | – | 1 | 2.0 | – |
| 22 | N28 | New Image gives no size estimate or warning for huge images; Slices / Frames without range hints | New Image sheet | 2 (E5-18) | – | – | 1 | 2.0 | – |
| 23 | N33 | Edit ▸ Undo does not name the step | Edit menu | – | 2 (E6-7) | – | 1 | 2.0 | – |
| 24 | N24 | → Python / → JavaScript / API help overwrite the editor without asking | Script & Recorder | – | – | 2 (E7-15) | 1 | 2.0 | – |
| 25 | N34 | File menu duplicates (Open Recent, Export, New) | File menu | – | – | 2 (E7-13 part) | 1 | 2.0 | – |
| 26 | N35 | Save As keeps the old title, so getTitle() differs from Fiji | Save As | – | – | 2 (E7-17) | 1 | 2.0 | – |
| 27 | N36 | Cursor readout inside an aria-live region; float aria-valuenow; splitter not focusable; large 3-D view not a dialog | statusbar; splitter; 3-D view | – | – | 2 (E7-19 part) | 1 | 2.0 | – |
| 28 | N37 | Command help lists parameter names only; internal documents exposed; no conceptual help | sheet "?" | – | – | 2 (E7-20) | 1 | 2.0 | – |
| 29 | N18 | Selection-dependent commands: palette shows them enabled while the menu greys them; "a area selection"; no next step in the toast | palette; toasts | 2 (E5-10) | 1 (E6-14) | 2 (E7-18) | 3 | 1.7 | – |
| 30 | N26 | Menubar triggers: the CSS rule never applies, no hover / open state, 4.43:1 text | menubar | 2 (E5-16 part) | 1 (E6-10) | 2 (E7-12) | 3 | 1.7 | M8 (claimed fixed) |
| 31 | N21 | Tracks wizard: steps not gated on zero spots; grammar; detected spots not drawn; cramped | Tracks wizard | 2 (E5-13) | – | 1 (E7-24) | 2 | 1.5 | – |
| 32 | N22 | Decimal commas in number inputs under an English UI | Preferences; wizard; sheets | 1 (E5-14) | 2 (E6-6) | 1 (E7-26) | 3 | 1.3 | – |
| 33 | N29 | Inspector density: up to 14 sections, no first-run guidance, Workflows collapsed, Tracks below the fold, a dense clipped help overlay | inspector; overlay | 1 (E5-19) | 1 (E6-12, E6-13) | – | 2 | 1.0 | M11, M36 |
| 34 | N17 | Gaussian Blur defaults to the entire stack; the recorded line omitted the scope | Gaussian Blur sheet | 1 (E5-9) | – | – | 1 | 1.0 | – |
| 35 | N27 | Menubar: no second Esc back to the image; 118 items under Image | menubar | 1 (E5-17) | – | – | 1 | 1.0 | – (type-ahead part contested by E7) |
| 36 | N30 | No reflow below ~1000 px; image not re-fitted after resize | layout | 1 (E5-20) | – | – | 1 | 1.0 | M30 (2.0) |
| 37 | N31 | The close dialog calls the undo history an "unsaved change" | close dialog | – | 1 (E6-4) | – | 1 | 1.0 | – |
| 38 | N32 | Tab close buttons only on hover; modified marker is a dot | tab strip | – | 1 (E6-5) | – | 1 | 1.0 | M39 part |
| 39 | N39 | Split-view compare pane opens at its own zoom, cropped | split view | – | – | 1 (E7-22) | 1 | 1.0 | – |
| 40 | N40 | "[" opens the editor or changes the brush size depending on the tool, undocumented | shortcuts | – | – | 1 (E7-23) | 1 | 1.0 | – |
| 41 | N41 | Z Project *Stop* shows 0 instead of the slice count; Gaussian lacks *Scaled units* | sheets | – | – | 1 (E7-25) | 1 | 1.0 | – |
| 42 | N42 | The tool-options bar pushes the canvas down | viewport | – | – | 1 (E7-27) | 1 | 1.0 | – |

---

### 3.14 Re-evaluation of the fixed build by the evaluators who found the problems

The fixes of §5.4 were checked the only way that counts: the evaluators who found the problems went
back to the application. E8 is the Opus persona of E7 (the image-analysis specialist), run as a new
agent of the same model family that was given only its own earlier file - not this document, not
the fix list - and asked to re-test every one of its 27 problems with the original control and
sequence, rate each *fixed / partly fixed / not fixed*, and look for regressions. E9 does the same for
E5's 20 problems (the core-facility manager, Sonnet). Both files are committed next to the originals
(`tests/e2e/usability/walkthrough/e8_opus_recheck/`, `e9_sonnet_recheck/`). The E8 walk happened
*before* the fixes of §5.5; E9 ran after them, so the two also bracket that round.

**E8 (Opus, E7's 27 problems): 15 fixed, 10 partly fixed, 2 not fixed, 0 could not test.** E8
reproduced E7's exact sequences - the tab switch that lost its target, the σ = 500 blur, the misspelled
keyword, the Save As rename, the compare-pane zoom - and reports each with screenshots, recorder lines
and REST responses.

| ID | Status | Sev. now | What E8 saw (shortened) |
|---|---|---|---|
| E7-1 | fixed | 0 | E7's sequence replayed: the recorder now has `lumen.select("MAX_t1-head.tif")` before the filters and `lumen.close("blobs.tif")` with its title; a replay runs on the right image. |
| E7-2 | fixed | 0 | ⌘Z adds `lumen.undo("MAX_t1-head.tif")`; a History jump of four steps adds four undo lines. |
| E7-3 | fixed | 0 | Inspector edits (DAPI max 1500, Auto, eye toggle, Viridis) are recorded as `lumen.display(...)` lines per channel. |
| E7-4 | fixed | 0 | The recorder is per browser client, starts with a session marker, holds only this session's 31 lines and auto-scrolls. |
| E7-5 | partly | 3 | The batch sheet shows an editable list, but its default still contains the steps that were undone: the undo lines are dropped and the commands they cancelled stay. |
| E7-6 | fixed | 0 | Clear… is an alertdialog with the count ("Delete 41 ROIs"), Cancel focused, and the toast offers *Undo clear*, which restores. |
| E7-7 | fixed | 0 | σ = 500 runs as 500 and is recorded so; width 200 000 is refused inline with the limit named. |
| E7-8 | fixed | 0 | `sigmaa=1.0` stops the script with "unknown parameter 'sigmaa' … did you mean 'sigma'?"; an unknown command id suggests the right one; no traceback paths. |
| E7-9 | fixed | 0 | "apply lut" lists distinct names: *Set LUT*, *Apply Display Range to Pixels (Apply LUT)*, *… (B&C Apply)*, the destructive ones described as such; the submenu lists the LUTs. |
| E7-10 | partly | 1 | "enhance local contrast" finds CLAHE first, "clahe" has no noise, counts agree; "bleach" still finds nothing. |
| E7-11 | fixed | 0 | "Channel 1 · DAPI is thresholded"; `threshold.compute(…, channel=1)` and `analyze.particles(…, channel=1)` are recorded. |
| E7-12 | fixed | 0 | Flat triggers at 7.16:1 (dark) / 6.11:1 (light), hover and open states. |
| E7-13 | partly | 1 | The duplicate Open Recent…, the second Export and the top-level New… are gone; the LUT dropdown is grouped with the near-duplicates explained; some duplicates remain in the File menu's New / Export submenus. |
| E7-14 | partly | 2 | Full-width language select, readable file name, line-number gutter, a large-view editor dialog, → IJM, keyboard splitter; still a plain textarea, and the recorder header overflows at the default width. |
| E7-15 | fixed | 0 | → Python / → IJM / API help ask *Replace* or *Append below* when the editor holds unsaved lines. |
| E7-16 | partly | 2 | History rows show parameters, are buttons, tooltips name the step; Image Info shows *Derived from* / *Source*; the saved TIFF still has no processing log and says "ImageJ=1.11a". |
| E7-17 | fixed | 0 | Save As renames the tab, Image Info and `getTitle()`; the save is recorded. |
| E7-18 | partly | 1 | The palette greys Crop with the reason; the toast names the area tools; the Plot Profile hint suggests an oval, which the command then refuses. |
| E7-19 | partly | 1 | Status bar no longer live; sliders and eye toggles named per channel with rounded values; min/max inputs labelled; the split view's image selects still have no accessible name. |
| E7-20 | partly | 1 | The sheet's *?* opens an inline table with defaults, units and ranges; no ImageJ-macro equivalent is shown. |
| E7-21 | not fixed | 3 | Preferences, ROI Manager (61 ROIs survived Close All), Results, recent files, the Save dialog's folder and Python execution are still server-global. |
| E7-22 | fixed | 0 | The compare pane fits the image at 100 % with no offset. |
| E7-23 | fixed | 0 | The overlay states the brush dependency of "[". |
| E7-24 | partly | 1 | Steps 1 and 2 report consistent counts and the slider starts at the threshold; "20 point ROIs added" has become grammar of its own, and the wizard is still squeezed into the inspector. |
| E7-25 | partly | 1 | Z Project opens with Stop = 10, the real slice count; Gaussian Blur had no *Scaled units* yet (added afterwards, §5.4). |
| E7-26 | not fixed | 1 | Spot radius "2,5" and Selection stroke "1,5" in the host locale: `lang="en"` does not change how Chrome formats number inputs. |
| E7-27 | fixed | 0 | The canvas box is identical before and after choosing a tool. |

**New problems E8 found**, with severities, and what became of them (§5.5):

| ID | Sev. | Problem | Cause / fix |
|---|---|---|---|
| E8-1 | **4** | **Menu dropdowns invisible** (regression): `.menubar` had become `overflow-x: auto` and `.topbar` `overflow: hidden` in the narrow-window item of §5.4, and the absolutely positioned popover inside them was clipped to the 27 px strip; from the keyboard the strip scrolled one item at a time. Only ⌘K remained. | The two overflow rules were the cause; removed (the menubar wraps instead), and the popovers now render through a portal with fixed coordinates so no ancestor can clip them. The §5.4 browser check had read the menu items from the DOM, where they were present, and did not test visibility - a lesson recorded in §5.1. |
| E8-2 | 3 | A page reload starts an empty recorder; the previous recording is still on the server but nothing shows or restores it. | The client id was generated per page load; it is now kept per tab in `sessionStorage`, so a reload keeps its recorder and a new tab gets its own. |
| E8-3 | 2 | The IJM export wrote four `run("Undo")` lines in a row (ImageJ's undo is single-level, so the macro ends on a different image); its notes were only in the API response; a change to max alone was dropped. | Undo is now resolved before export and before a folder batch (the undone command and the undo line both disappear); notes are `// note:` lines in the script; a min- or max-only change becomes `getMinAndMax(lo, hi); setMinAndMax(lo, 1500);`. |
| E8-4 | 2 | Tracks detection turned on *Show all* and drew the spots of all 20 frames on frame 1, and added 20 point ROIs to the shared ROI Manager. | The detection preview is drawn per frame only, without entering the ROI Manager. |
| E8-5 | 1 | Three command counts: 497 (empty screen, the server that was still running the previous build), 597 (palette footer, counting LUT entries), 498 (command reference). | The palette footer counts commands only and names the LUT entries separately; the empty screen uses the same number. |
| E8-6 | 1 | The first image opened into an empty workspace came up at 149 % with its bottom behind the channel strip; the same image later fitted at 136 %. | The fit was computed before the channel chips had their height; the viewport re-fits when its container resizes until the user zooms or pans. |

*E8's verdict (shortened):* "The recorder now gives a faithful account of a session, with image targeting,
undo, display changes and clears, plus an ImageJ-macro export. Parameter validation, guarded clears and Save
As now behave as a Fiji user expects. I still wouldn't hand it to facility users until the invisible menubar
(E8-1), the recording lost on reload (E8-2), the batch list that includes undone steps (E7-5) and the shared
server state (E7-21) are fixed."

**E9 (Sonnet, E5's 20 problems, after the round-3 fixes): 16 fixed, 4 partly fixed, 0 not fixed, 0 could
not test.** E9 re-ran E5's probes - the "fire" and "apply lut" palette queries, a maximum typed below the
minimum, Analyze Particles without a threshold, the 20-frame noise movie in the Tracks wizard, "[" followed
by ⇧K, F10 and Esc, a 20000 × 20000 New Image, a 900 × 650 window - and found the recipe path a trainee
would follow now works: LUT names where Fiji users look, the destructive commands labelled, errors that
say what to do, ROI Manager and Results scoped to the image, the editor releasing focus on Esc, decimal
points in both fields it re-checked.

| ID | Status | Sev. now | What E9 saw (shortened) |
|---|---|---|---|
| E5-1 | fixed | 0 | Image ▸ Lookup Tables lists the ImageJ LUTs and the Fiji / Perceptual / Labels submenus; clicking Fire set the channel's LUT. |
| E5-2 | fixed | 1 | "apply lut" lists *Apply Display Range to Pixels (Apply LUT)*, *… (B&C Apply)* ("DESTRUCTIVE …") and *Set LUT*. |
| E5-3 | fixed | 1 | "fire" → LUT: Fire; "clahee" → CLAHE; "gausian blur" → Gaussian Blur; "watershed", "scale bar", "z project" put the obvious command first. |
| E5-4 | fixed | 0 | "Open Sample" first, the ImageJ download second; the opened sample is named without a "-1". |
| E5-5 | fixed | 0 | The missing-threshold error is an inline, persistent box in the sheet with *Auto-threshold and run* and *Open the Threshold panel*. |
| E5-6 | fixed | 1 | A maximum below the minimum is refused inline and the state stays; same for a minimum above the maximum. |
| E5-7 | partly | 1 | Display edits are recorded with the image named, image switches as `lumen.select`; the LUT line did not name its channel (it does now, §5.5). |
| E5-8 | partly | 2 | History shows parameters, but only the non-default ones, and display edits are still not undoable (both changed afterwards, §5.5). |
| E5-9 | fixed | 0 | Gaussian Blur on a stack defaults to *Current plane only* with a hint; the recorder writes the scope. |
| E5-10 | fixed | 1 | Selection-dependent commands are greyed in the palette with the reason; the toast names the tools. |
| E5-11 | fixed | 1 | *This image only* on by default: "No ROIs of this image. 41 ROIs of other images are in the manager …". |
| E5-12 | fixed | 1 | Threshold state is per image; a new image and the compare pane show no overlay. |
| E5-13 | fixed | 1 | A 0-spot detection gets a "!" chip and a disabled *Next* with the reason (the step chips could still be clicked - E9-5, fixed in §5.5). |
| E5-14 | fixed | 0 | Selection stroke "1.5" and Spot radius "2.5" with a decimal point under en-US. |
| E5-15 | fixed | 0 | "[" focuses the editor, Esc returns focus to the canvas, ⇧K then opens the colour picker without typing into the script. |
| E5-16 | partly | 1 | 0 unnamed controls (was 5), roving tabindex on the rail, 7.16:1 menubar, 24 px tab buttons; 24 of 80 targets still under 24 px. |
| E5-17 | fixed | 1 | F10, ↓, first Esc closes the menu, second Esc returns to the canvas; type-ahead works. |
| E5-18 | fixed | 0 | New Image shows "Size: 1.5 GB in memory (6 % of 24 GB RAM) – large: a confirmation follows …". |
| E5-19 | partly | 1 | A welcome screen and richer Workflows cards; still 13 sections, Workflows collapsed, the help overlay 36 px past a 900 px window (overlay fixed, §5.5). |
| E5-20 | fixed | 1 | No horizontal scroll at 900 × 650 and 760 × 600, the image re-fits, the inspector keeps its width. |

**New problems E9 found**, all severity 1: E9-1 the Display section's new validation message overlapped the
next row (wrapped now); E9-2 the particles toast and Results label do not say which channel was analysed (the
messages now do: "41 particles on channel 1 (DAPI)"); E9-3 at 900 × 650 the top-bar icons are dropped and the
tool rail is cut off after Fill (the rail now shows a thin scrollbar; the icons stay hidden below 1000 px by
design); E9-4 "threshhold" ranks *Apply Threshold → Mask* above *Threshold* and "fiji lut ice" finds nothing
(open); E9-5 the wizard's step chips could jump past the gate *Next* enforces (chips are gated now).

*E9's verdict (shortened):* "I would now let a trainee follow a Fiji recipe for LUTs, threshold plus Analyze
Particles, Measure and Gaussian blur without standing next to them. … What still blocks fully hands-off use is
traceability and accessibility polish: display edits are not undoable or in History, History hides default-valued
parameters, the recorder is one global log, and a few layout rough edges remain." The first two of those were
fixed in the same cycle (§5.5); the recorder is one log per browser tab by design, as Fiji's is one per
session.

**E10 (Opus, E8's 18 open items, after the round-3 fixes): 11 fixed, 6 partly fixed, 1 not fixed.** E10
re-ran E8's sequences: the clipped menus at four widths, the reload with a live recorder, the σ = 500 blur
undone before the macro export, the noise movie in the wizard, the three command counts, the first image
into an empty workspace, "bleach" in the palette, a Save As followed by a dump of the TIFF's tags, "2,5"
typed into a numeric field.

| ID | Status | Sev. now | What E10 saw (shortened) |
|---|---|---|---|
| E8-1 | fixed | 0 | The dropdown is a fixed-position popover at the end of `<body>`; File shows whole at 1000, 1280, 1440 and 1920 px, submenus included. |
| E8-2 | fixed | 0 | The client id lives in `sessionStorage`; after a reload the recorder still shows its 17 lines and marker. |
| E8-3 | fixed | 0 | E8's sequence: the macro export has no `run("Undo")`; the undone steps and their undo lines are dropped and the script says so. |
| E8-4 | fixed | 0 | Detection draws only the current frame's spots and adds nothing to the ROI Manager. |
| E8-5 | fixed | 0 | "499 commands" on the empty screen, in the palette footer ("499 commands + 41 UI actions + 60 LUTs") and in the reference. |
| E8-6 | fixed | 0 | The first sample into an empty workspace opened at 136 %, whole, above the channel strip. |
| E7-5 | partly | 2 | The batch preview no longer contains the five undone commands; it still replayed a ROI Manager clear and a Results clear that "Undo clear" had taken back (fixed afterwards, §5.6). |
| E7-10 | fixed | 0 | "bleach" finds *Bleach Correction…* with the three methods. |
| E7-13 | partly | 1 | Menus are clean; the palette still listed the duplicates and mixed "..." with "…" (fixed afterwards, §5.6). |
| E7-14 | partly | 1 | The recorder header fits; the editor stays a plain textarea. |
| E7-16 | fixed | 0 | Tags after Save As and ⌘S: `Software = Lumen 0.1.0`, `ImageJ=1.54p …`, a processing log with the steps and parameters. |
| E7-18 | partly | 1 | Plot Profile's hint said "line or rect selection (1–7)" (the client did not know the new kind; it does now, §5.6). |
| E7-19 | fixed | 0 | Auto / Reset / Inv are named per channel; the split-view selects are named. |
| E7-20 | partly | 1 | The sheet's help ends with `ImageJ run("Gaussian Blur...") · filters.gaussian`; no option string. |
| E7-21 | not fixed | 3 | Server-global preferences, ROIs, Results, recent files, file dialog and script execution - by design (§5.4). |
| E7-24 | partly | 1 | "1 spot in 1 frame", only the current frame's spots; the wizard still lives in the inspector. |
| E7-25 | fixed | 0 | *Scaled units* under Advanced options, with the ImageJ explanation. |
| E7-26 | fixed | 0 | "2.5" and "1.5" in the host locale; "2,5" typed on a German keyboard is read as 2.5. |

**Regressions and new problems E10 found**, and what became of them (§5.6):

| ID | Sev. | Problem | Cause / fix |
|---|---|---|---|
| E10-1 | 2 | Keyboard use of the menus broke: F10 ↓ opened a menu but focus stayed on the trigger; ← → dropped focus to the body. | The portaled popover is `visibility: hidden` for one layout pass until it is positioned, and a hidden element cannot take focus; the first item is now focused once the popover is visible. The menu guard checks keyboard focus too. |
| E10-2 | 1 | The Design and Research documents are back under Help ▸ Design & Research Notes and as tabs of the docs dialog. | Deliberate: they are documentation and belong in Help; they stay out of a sheet's *?* help, which is what E7-20 asked. |
| E10-3 | 2 | "abc" or "2,5,1" in a numeric field turned it red but Apply stayed enabled and the command ran with the default. | The field reports unparsable text to the sheet, which disables Apply and says "Radius is not a number". |
| E10-4 | 1 | The processing log's derivation line had no parameters; after a second save the Source line named the file itself. | *Derived from* carries the command's parameters; the first Save As keeps the opened file as the source. |
| E10-5 | 1 | Step 2 of the wizard kept the old threshold after a new detection. | A detection re-seeds the threshold. |
| E10-6 | 2 | Z Project on a single-slice 20-frame series pre-filled Stop 20 and then refused ("needs a stack with more than one slice"); Fiji projects the frames. | Like ImageJ's ZProjector, a single-slice time series is projected over its frames when the axis is left at Z. |

*E10's verdict (shortened):* "Most of round 2's problems are fixed, including both blockers (menus and
recorder reload), so I would now give Lumen to Fiji users on single-user facility workstations, with warnings
about the batch replaying clears (E7-5) and non-numeric input running silently (E10-3). A shared facility
server is still blocked by the missing separation between users (E7-21), and keyboard users have lost the
menus (E10-1)." The two warnings and the keyboard regression were fixed in the same cycle (§5.6); the shared
server stays a deployment rule (one process per user, PLATFORMS.md).

**E11 (Opus, E10's seven open items, after the round-4 fixes; 13 Nov 2026): 4 fixed, 2 partly fixed, 1 not fixed.**

| ID | Status | What E11 saw (shortened) |
|---|---|---|
| E10-1 | fixed | F10, ↓, Home, End, Enter, Esc and ← / → in submenus move real focus at 1440 and 1000 px; a mouse-opened menu keeps focus on ← / →. |
| E10-3 | partly | Apply is blocked with "Radius is not a number", but Enter in the field still ran Median with the default (fixed afterwards, §5.7). |
| E10-4 | partly | *Derived from* carries its parameters and an opened file stays the source; a projection re-saved with ⌘S named its own file as the source (fixed afterwards, §5.7). |
| E10-5 | fixed | A new detection re-seeds step 2's threshold; with 0 spots *Next* stays disabled. |
| E10-6 | fixed | Z Project on a 20-frame single-slice series projects the frames; the result equals numpy's maximum over time. The recorded line still says `axis='Z'` (the replay gives the same result). |
| E7-5 | fixed | Neither the batch preview nor the IJM export replays a ROI Manager or Results clear that *Undo clear* took back. |
| E10-2 | not fixed | Help ▸ Design & Research Notes is still there: E11 judged it a developer brief rather than help, rendered badly (bullets split in two, literal asterisks). The menu item now says *(background)* and the Markdown renderer was fixed (§5.7); the notes stay in Help as documentation. |

**New problems E11 found**, both severity 2, both fixed (§5.7): E11-1 ImageJ's Blobs sample opened as an RGB image with
its inverting palette baked into the pixels, so a whole-image Measure gave a mean of 151.73 instead of ImageJ's 103.27 -
every Fiji Blobs tutorial would disagree; E11-2 the folder-batch list kept a line that opens the Blobs sample.

*E11's verdict (shortened):* "Round 4 fixed the important things: keyboard menus, clears that were taken back, Z Project
on a time-lapse, and Tracks re-seeding all work now, so I'd hand Lumen to Fiji users on a single-user workstation.
Before that, Enter must stop running the default on non-numeric input, the batch list must drop the sample-open line,
and Blobs must open as 8-bit like in ImageJ; shared-server separation (E7-21) is still unsolved." The three named
conditions were fixed the same day (§5.7).

**E12 (Opus, E11's three conditions, the provenance leftover and the new editor highlighting; 13 Nov 2026): 4 fixed, 1
partly fixed.** Enter with "abc" or "2,5,1" in a sheet field runs nothing and says "Radius is not a number"; the batch
list leaves the Blobs line out ("1 … line left out"); Blobs opens as 8-bit with an inverting LUT and measures Mean
103.269, Min 8, Max 248 - identical to ImageJ and to the GIF's raw indices (Gel.gif too); a re-saved projection no
longer names itself as its source. The highlighting was right in the docked editor, but **E12-1 (severity 3)**: in the
large view the text area used 13 px type and the coloured layer 12 px, so below line 7 a click edited the line above;
E12-2 (severity 1): after a page reload, images saved with ⌘S showed as modified again; E12-3 (severity 0): `setSlice`
was coloured as a user function. All three were fixed the same day: one `--code-fs` size for both layers (a click on
visible line 15 now puts the caret on line 15), the saved marks kept per tab across reloads, the macro built-in list
extended. *E12's verdict:* "All three conditions from the last verdict, and the provenance leftover, now hold, so I
would hand Lumen to Fiji users on a single-user facility workstation." 

---

## 4. Traceability: GUI decisions, principles and sources

| GUI decision in Lumen | [DESIGN.md](DESIGN.md) principle | Evidence in [RESEARCH.md](RESEARCH.md) | Observed in this assessment |
|---|---|---|---|
| One window; images as tabs; results, ROIs, histogram, history, console as inspector sections | 1 One window, three regions | §1 "difficulty in managing multiple windows is a general problem" (Image.sc 68201); the Window menu exists only to tile and cascade; multi-screen placement threads. Precedents: napari docks (§3), QuPath Analysis pane (§4), VS Code areas and tabs (§11), 3D Slicer module panel (§10) | 0 OS windows in ten tasks vs 20 (section 2.2); Results/ROI Manager/Tracks sections populated without user action (tasks 2, 3, 7) |
| ⌘K command palette with category, description, shortcut and web fallback | 2 Command palette first | §1 Command Finder "the quickest way to find a command without having to navigate through all the menus"; §2 the ImageJ2 search bar was the one new-UI feature that shipped; §4 QuPath `Ctrl+L` "much easier than navigating menus"; §3 napari 0.6 palette; §6 CellProfiler module search; §11 Figma Quick Actions ("everything must be in it"), VS Code ⇧⌘P | Task 8: 3 actions, parity with the Command Finder, plus description and category per entry; every task launched its command from the palette |
| Every command is a parameter schema; the sheet is generated | 3 Every command is a schema | §2 ImageJ2 "parameterized modules" whose dialogs are framework-built (Rueden 2017); §3 napari #5358 controls "not presented in the same place, same order" | Identical sheet layout for Open Sample, Z Project, Merge, Gaussian, Track, Export, Batch (tasks 1-9); heuristic 4 |
| Live preview on the current plane, on by default, non-modal | 4 Live preview by default | §7 ilastik "Live update"; §4 QuPath "Live prediction"; §8 Imaris "preview of the results at each point", GA3 "Preview" of the whole graph, ZEN Preview, arivis 4D preview; §1 Rueden on legacy real-time plugins breaking outside AWT | Task 6: preview on without a click, request fired while typing; no preview for whole-image/analysis commands (tasks 4, 5b, 7, 9) |
| Non-destructive per-channel display with histogram sliders, all channels visible in composite mode | 5 Display is non-destructive and per-channel | §1 B&C *Apply* rewrites pixels, teaching material says do not press it; five separate Adjust dialogs; §3 napari contrast limits "very difficult" to fine-tune; §11 Photoshop adjustment layers; §9 OMERO.iviewer settings that "NEVER change the original Image data", Viv client-side compositing | Tasks 1 and 5a: all three channels adjusted without switching channels (9 and 6 actions, 0 windows); display changes not recorded (gap) |
| Deep undo with a visible History section | 6 Deep undo | §1 "ImageJ does not have a large undo/redo buffer"; §4 QuPath undo "turns itself off … when many objects are present"; §12 EasyFiji advertises "undoable processing"; §11 Affinity/Photoshop history | Task 6: three steps undone and redone, listed in History |
| Recorder always on, output is runnable Python | 7 Recorder always on | §4 QuPath Workflow tab → script; §12 surveys: ¾ of users also script but want a no-code entry; §2 IJ1 menu commands bypass ImageJ2's module events ("ModuleEvents not fired"); §6 CellProfiler: recorder as the bridge from exploration to batch | Every task produced recorder lines (section 2.4); task 9 replayed them over a folder in 11 actions |
| Masks are foreground = 255, no black-background switch | 8 Honest binary semantics | §1 "Binary commands assume that objects are black … unless 'Black Background' is checked", persistent `Prefs.blackBackground` | Task 2 ran Analyze Particles straight on the threshold; heuristics 2 and 5 |
| Browser UI on a Python engine | Architecture | §1 Rueden 2018 "I would favor developing in JavaScript over JavaFX"; the Swing UI "was never fully finished"; §9 Viv, Neuroglancer, iviewer render in the browser; §12 installation as the first usability barrier | Measurable in headless Chrome; constraints listed in section 3.4 |
| Threshold as a docked section with a red overlay and a *Particles…* button | 1 + 5 | §1 four separate adjust tools re-opened per image | Task 2: 4 actions, 0 windows vs 7 and 5 |
| Tracks as a docked table with overlays, filter, export (TrackMate column layout) | 1 + 3 | §8 Imaris/arivis keep objects in panels beside the viewer; §1 TrackMate's wizard is a separate window | Task 7: 7 actions vs 21-27; CSV header matches TrackMate's columns |
| Guided Workflows section | 2 + 7 | §8 Aivia Launchpad "Guided Sequences", Imaris 11 Workflows; §12 surveys asking for "easy-to-follow … beginner's guides" | Present (Workflows section); not measured |
| Split view with linked pan/zoom | 1 | §9 OMERO.iviewer "multiple side-by-side viewers and the ability to synchronize settings between them"; §5 Icy synchronised viewers; §1 Synchronize Windows as a separate tool | Task 10: 1 key vs 2-6 clicks and an extra window |
| Progress chip with Cancel, job list, memory monitor | Nielsen 1 | §3 napari 0.5 asynchronous slicing; §14 "batch job queue with progress" listed as a gap | Progress chip observed during tracking and batch; no queue |
| In-app docs, help-on-command, hint text, `?` overlay | Nielsen 10 | §12 "respondents' overarching needs are documentation, detailed tutorials"; §6 CellProfiler "context-dependent warnings" | Heuristic 10: present, but no community corpus |

---

## 5. Limitations and a proposed user study

### 5.1 Limitations of this assessment

- **No users.** Nothing here measures learnability, error rates, time or satisfaction. The action
  counts are keystroke-level counts for an expert who knows the names of things and the shortest
  path; novices in Fiji use menus rather than `L`, and novices in Lumen do not know that ⌘K
  exists until told.
- **Seven evaluations, no people.** The original evaluator also knows Lumen's source, and the
  three persona passes were role-played by the same AI assistant (section 1.1): they share
  knowledge, habits and blind spots, so the agreement figures in section 3.9 overstate
  independence, and every severity-3 rating in section 3.10 rests on a single evaluator's
  judgement. The second round (E5-E7) was written by three other models that had not read this
  document, which removes the shared *authorship* but not the shared vendor, training and brief;
  §3.12 shows their overlap is higher than E1-E4's and says why that is only partly reassuring.
  Five of the nine major problems of §3.13 rest on E7 alone. NN/g's three to five *independent*
  evaluators have still not been achieved, and no AI evaluation replaces a person at the screen.
- **A scripted check is not a walk.** The 17-check script of §5.4 read the menu items from the DOM and
  passed while the dropdowns were invisible on screen (E8-1, §3.14): a DOM fact is not a visual fact.
  The re-evaluations of §3.14 exist because of that; the script remains useful only for what it
  measures (text, attributes, contrast ratios), never as evidence that something can be seen or used.
- **Fiji counts are derived from documentation**, not measured on a running Fiji. Where the
  documentation does not give a default (Analyze Particles checkboxes, Create composite, Z Project
  method, Preview checkbox state) the assumption is stated; a Fiji session that remembers settings
  from previous use costs fewer clicks. The ImageJ User Guide is the 1.46r edition.
- **Lumen counts come from a script**, which never mis-clicks, never scrolls the inspector, never
  misreads a label, and ran on synthetic sample images at 1440×900. Real tasks include reading,
  deciding and correcting, which the counts ignore. Section state (which inspector sections are
  open) changes the count by one per section.
- **Windows are not all equal.** A non-modal Results window in Fiji and a Results section in Lumen
  both show the same table; the count records the difference in window management, not in
  information. The palette was counted as modal although it is dismissed by one key.
- **Not compared:** analysis correctness, speed, memory, file-format breadth (section 3.4 only
  lists them), stability over long sessions, screen-reader use, and non-English users.

### 5.2 Proposed protocol for a small user study

**Aim.** Test whether scientists who use Fiji complete common tasks in Lumen with equal or higher
completion rate and satisfaction, and find the specific points where Lumen's model confuses them.

**Design.** Within-subject, counterbalanced (half start with Fiji, half with Lumen), one 75-minute
session per participant, think-aloud, screen and audio recorded, moderator present.

**Participants.** 8-12 researchers who use Fiji at least weekly (the primary population), plus 4
who have never used Fiji (learnability control). Record years of Fiji use, scripting experience,
operating system and screen size. With these numbers the study detects large differences only;
it is a usability study, not an A/B test.

**Materials.** The same image set in both tools: a 3-channel fluorescence image, an 8-bit image
with blob-like objects, a 2-channel z-stack, a nuclei time-lapse, a folder of ten TIFFs. Fiji
1.54 as installed, Lumen as served from this repository with an empty `~/.lumen`. Written task
cards; no hints about menu locations or the palette beyond a two-minute tour of each tool. **The
ready-to-run kit is in [`docs/usability_study/`](usability_study/README.md)**: participant
briefing, consent text, the ten task sheets with success criteria, the SUS questionnaire, CSV
logging templates and `analyze_results.py` (pure Python: SUS scoring with confidence intervals,
completion / time / error summaries, paired Wilcoxon tests, pass-criteria check).

**Tasks.** The ten tasks of section 2.1, in the same order for both tools, each with a defined
end state (e.g. "a CSV with one row per track exists on the Desktop"). Task 9 follows task 6 so
that something has been recorded.

**Measures.**
- Task completion (complete / complete with moderator assist / abandoned after 5 min).
- Time on task from the card being read to the end state.
- Errors: wrong command, wrong window, destructive action needing undo, dialog cancelled.
- Clicks and keys: Lumen from a browser-side event log (the counting rules of section 1.2);
  Fiji from the screen recording.
- Where they looked for a command first (menu, search, memory, help), from the think-aloud.
- SUS after each tool: the ten standard items on a 5-point scale, scored 0-100 (odd items
  score minus one, even items five minus score, sum times 2.5); the published average across
  studies is 68 ([Sauro, measuringu.com/sus](https://measuringu.com/sus/)). SUS works with
  small samples but the estimate is imprecise, so report the confidence interval (same source).
- Two open questions per tool: "what would stop you using this for your own data?" and "what did
  you miss?".

**Analysis.** Per task, paired comparison of completion, time and errors (Wilcoxon signed-rank,
report effect sizes and intervals, not only p-values); SUS means with 95 % confidence intervals;
qualitative coding of think-aloud notes into Nielsen's heuristics, with severity from frequency
across participants. Pre-register the task list, measures and analysis before recruiting.

**Pass criteria for Lumen** (stated in advance): completion ≥ 90 % without assist on tasks 1-6
and 8-10; task 7 and 9 completion ≥ 75 %; SUS not lower than Fiji's for the same participants;
no severity-4 finding. Each severity-3 finding becomes an issue with the task card and the
recording timestamp attached.

**Expected weak points to watch** (from sections 2.5, 3.4 and 3.10): discovering ⌘K; the file
browser instead of the OS picker; the meaning of *Apply* vs *Run*; the number of inspector
sections; looking for Brightness/Contrast under Image ▸ Adjust (M9); pressing ⌘W to close an
image (M3); running Analyze Particles before thresholding (M2); closing a modified image (M1);
expecting display changes to be undoable or recorded; wanting a detection preview before tracking;
wanting a second OS window on a second monitor. Participants who rely on the keyboard or a screen
reader will hit M4-M8 immediately; the study should either fix those first or recruit sighted
mouse users only and say so.

---

## 5.3 Fixes made after the evaluation

The merged list in §3.10 was acted on in the same build cycle (7 Nov 2026, evening). Fixed and re-verified by
type check and unit tests, not yet by a repeat walkthrough: M1 (closing a modified image asks Cancel / Save /
Close without saving, with a "don't ask again" preference), M2 (Analyze Particles refuses an unthresholded,
non-binary image unless *Auto-threshold if needed* is ticked; the macro layer keeps ImageJ's behaviour),
M3 (⌘W / ⇧⌘W close the image / all images where the browser lets the shortcut through), M4 (the menubar is
keyboard-operable: F10 or ⌥ focuses it, arrows move, Enter runs, Esc closes; menubar / menu / menuitem roles),
M5 (inspector section heads are buttons with `aria-expanded`), M6 (image tabs are a `tablist` with arrow keys
and ⌘1–9), M7 (the palette is a combobox with a listbox, `aria-activedescendant` and a live result count),
M8 (every text and UI contrast pair now meets WCAG AA in both themes; 19 pairs per theme checked by a script),
M9 (Image ▸ Adjust lists Brightness/Contrast… and Threshold… first), M11 (a Compact / Comfortable density
preference), M12 (a Browse… button next to every path or folder parameter). M10 (data larger than RAM) was addressed afterwards: `lumen.core.lazy` generalises the TIFF virtual stack to HDF5 /
Imaris, Zarr / OME-Zarr and npy; File ▸ Open opens any such file above `auto_virtual_mb` (default: half of the physical
RAM) as a virtual stack and says so (log, toast, Open-dialog checkbox to force it); projections, statistics, profiles,
substacks and exports stream plane by plane with progress / cancel; the Memory Monitor shows per-image resident vs
full size and warns once before a command materialises such a stack. Measured: a 1.95 GB synthetic OME-Zarr opens in
0.01 s with 4 MB resident; max / SD projections and Stack Statistics run at +44 / +181 MB peak RSS without
materialising it; a zlib OME-TIFF export streams at +6 MB. Still open: no multi-resolution (BigDataViewer-style)
rendering of single planes larger than RAM. M24–M26 were fixed afterwards as well: every dialog-like surface
(Preferences, Help, Docs, Shortcuts, File dialog, Results modal, script-parameter and confirm dialogs, and the
operation sheet when opened from the keyboard) now traps Tab, closes on Esc and returns focus to its opener through
a shared `useFocusTrap` hook, and every control in the Preferences and operation sheets carries an accessible name,
hint association and slider value attributes.

## 5.4 Fixes made after the cross-model evaluations

The list in §3.13 was acted on in the same build cycle (8 Nov 2026), in two packages: the engine
and API package below, and the interface package after it. Each fix is covered by unit tests
named here and by the type check and build; none has been re-walked by an evaluator, so the
tables of §3.11 describe the build *before* these changes.

**Engine and API** (`tests/test_recorder.py`, 18 tests, plus adjusted assertions in
`test_batch.py` and `test_ij_compat.py`):

- N9 (image targeting): the recorder emits `lumen.select("<title>")` whenever the image a command
  runs on differs from its last target (a tab switch between commands, or a command aimed at a
  non-active image), records closing as `lumen.close("<title>")`, appends `# → "<resulting
  title>"` to every line that creates an image, and takes the title *before* the command so Save As
  and Rename record the name the user saw. `lumen.select` / `lumen.close` replay. The closed
  image is forgotten as a target.
- N11 (undo in the recording): Undo, Redo and History jumps record `lumen.undo("<title>")` /
  `lumen.redo("<title>")`, which replay; scripted undo is not re-recorded.
- N8 (display state): the display endpoint records only the *effective* change, as
  `lumen.display("<title>", channel=2, min=100, max=1500, lut='Viridis', visible=False)`,
  `lumen.display("<title>", mode='single')` or `lumen.display("<title>", z=12)`; consecutive
  edits of the same channel collapse into one line, so a dragged slider leaves one line, not
  fifty; display changes made by commands (`color.lut`) are not recorded twice. The recording also
  exports as an ImageJ macro now (E7-14's second half; `→ IJM`, `GET /api/recorder/ijm`,
  `lumen/core/ij_export.py`): the macro layer's own import tables are inverted, so
  `run("Gaussian Blur...", "sigma=3 stack")`, `makeRectangle`, `setAutoThreshold("Otsu dark")`,
  `Stack.setChannel(c); setMinAndMax(a, b); run("Fire")`, `selectWindow`, `close("title")`,
  `newImage`, `roiManager("reset")` come out; a line with no ImageJ equivalent stays in the script
  as a `// not converted (reason)` comment and is listed, nothing is dropped silently. A test
  records a session, exports it, runs the macro through the macro layer and compares the pixels
  (`tests/test_ij_export.py`).
- N10 (global recorder): one recorder per browser client (an `X-Lumen-Client` header the page
  generates on load; requests without it share a default recorder, so scripts and tests behave
  as before), each with a start marker (`# Lumen recorder – session started 2026-11-08 04:00:01`)
  kept out of the line count, `GET /api/recorder`, `DELETE /api/recorder` and
  `GET /api/recorder/preview`; the state payload reports the active target and line count. The
  32 most recent clients are kept.
- N12 (batch preview): `Apply Recorded Macro to Folder` takes an explicit, editable list of lines
  (empty = the recording); the preview endpoint returns the lines that will run, the lines that
  are skipped (open / save / sample, and now select / undo / display / restore lines, which name
  images of the recording session) and the resulting script.
- N13 (clears): clearing or deleting ROIs and clearing Results stash what they removed (one step)
  and return the counts; `rois.restore_cleared` / `results.restore_cleared` (also
  `POST /api/session/restore_clear`) bring them back; the clears are recorded as
  `lumen.run("rois.clear")`, `lumen.run("rois.delete", indices='1')`,
  `lumen.run("analyze.clear_results")`.
- N14 (clamping): a value outside a parameter's range is rejected with
  `sigma must be between 0 and 200 (got 500)` instead of being clamped; the limits were audited
  at the same time - Gaussian sigma is now unbounded as in ImageJ, rank-filter radii go to 1000,
  New Image to 100 000 px a side behind a memory guard that says what the image would need
  ("A 20000×20000 32-bit image needs 1.5 GB – more than the 1.0 GB a new image may use on this
  machine").
- N15 (API strictness): `unknown parameter 'sigmaa' for filters.gaussian – did you mean 'sigma'?`
  and `Unknown command 'filters.gaussia' – did you mean filters.gaussian, filters.gaussian3d,
  filters.unsharp?`; script errors keep only the script's own frames; HTTP errors carry the
  message alone and the traceback goes to the server log.
- N6 (channel): Threshold, Measure and Analyze Particles take a `channel` parameter (1-based,
  0 = current) that is written into the recorded line on multi-channel images and omitted on
  single-channel or RGB images; the Results table's `Ch` column comes with *stack position*.
- N35 (Save As): the image is renamed to the saved file, as in ImageJ, so `getTitle()` follows.
- N16 (provenance, engine half): every image derived from another carries
  `derived_from` ("t1-head.tif ▸ Z Project") and `source_path` in its metadata, shown by Image
  Info and embedded in OME-TIFF exports; the History payload carries each step's non-default
  parameters (`sigma=5.0, scope=slice`) for the interface to show.
- N38 (shared server): not fixed. What is server-global is now stated in
  [PLATFORMS.md](PLATFORMS.md): the image list, ROI Manager, Results, tables, thresholds,
  preferences, undo histories, the file dialog's view of the file system and script execution.
  The per-client header separates recorders only and is not an authentication token; a shared
  facility needs one server process per user.

**Interface** (type check clean, `npm run build`, the e2e suite, `tests/test_luts.py`,
`tests/test_extras.py` and `tests/test_api.py` for the server-side pieces, and a scripted walk of
the fixed build in headless Chrome, `tests/e2e/usability/verify_fix_round2.py`: 17 checks - the
palette's "fire", "viridis", "enhance local contrast", "open sample" and "apply lut" hits, the
menubar's measured contrast of 7.2:1, the LUT submenu, the Threshold section's channel label,
the ROI Manager's *This image only* switch, History parameters, the named Edit ▸ Undo, the
recorder marker and line, the IJM export landing in the editor, the Tracks section and wizard on
a new time series, the classifier panel, and the inline out-of-range message with Apply
disabled - all pass; the one browser console entry is Chrome's own note of the 400 the server
answers to the deliberately invalid value, which is the mechanism behind the inline message):

- N1, N2 (LUTs): Image ▸ Lookup Tables now lists every LUT - the twelve ImageJ built-ins flat,
  then *Fiji*, *Perceptual*, *Labels* and *Imported* submenus - and the palette has a
  `LUT: Fire` entry per table, so "fire" and "viridis" find them; the Display select is grouped
  with the near-duplicates labelled (`mpl-viridis – Fiji's file, = Viridis truncated`); the
  non-destructive command is called *Set LUT* and previews a gradient, the destructive one is
  labelled *Apply Display Range to Pixels (Apply LUT)* (its server name stays ImageJ's "Apply
  LUT" for macros); *Import LUT…* / *Export LUT…* are named; the File menu's duplicates (Open
  Recent, Export, New) are merged into one submenu each (N34).
- N3, N4 (palette): ranking is exact name > Fiji alias > prefix > word start > substring >
  keyword / category > typo (Damerau-Levenshtein ≤ 2) > subsequence, weak hits are dropped when
  real matches exist, the header and footer share one count, and the Fiji names of
  [FIJI_PARITY.md](FIJI_PARITY.md) are searchable aliases shown as "Fiji: …", so "Enhance Local
  Contrast" finds CLAHE and "Open Sample" is first for "Open Sample".
- N5, N18 (errors): toasts sit bottom-left of the image area, never over a sheet, and carry
  action buttons; a sheet shows its error inline with *Auto-threshold and run* / *Open the
  Threshold panel*; selection errors read "Crop needs an area selection – draw a rectangle (1),
  oval (2) … first" and the palette greys selection-dependent commands with the reason, like the
  menu.
- N6 (channel): the Threshold section names the channel it acts on and warns, with a *Show
  channel N* button, when that channel is hidden in the composite; the red overlay follows the
  channel's visibility in the viewport and the compare pane.
- N7: a channel maximum at or below the minimum is refused inline (aria-invalid, blur restores).
- N10, N12, N24, N23 (recorder and editor): every request carries the client id; the recorder
  pane shows the session marker, the line count, auto-scrolls and has *Clear*; *→ Python* /
  *→ JavaScript* / *→ IJM* / *API help* ask *Replace* or *Append* when the editor holds unsaved
  text; *Apply Recorded Macro to Folder* opens with the exact lines from the preview endpoint,
  editable, and says how many were skipped; Esc or ⌘[ leaves the editor to the canvas, the
  editor is at least ten rows and grows to forty (resizable), the language select and file name
  are no longer truncated, and *Open in a large view* gives it a focus-trapped dialog.
- N13: *Clear* confirms with counts ("Delete 41 ROIs") and the toast offers *Undo clear*;
  *Delete selected* offers *Undo delete*.
- N14, N28 (values): an out-of-range value shows its message under the field (the server's
  `sigma must be between …` for the rest), Apply is disabled while a field is invalid, Gaussian's
  sigma is a plain number field with a `≥ 0` hint; New Image shows a live size estimate and asks
  before creating more than 512 MB or a quarter of the RAM; plane filters default to *Current
  plane* on stacks, with a hint (N17).
- N16, N33 (history): History rows show the parameters inline and in the tooltip, are buttons
  (Enter / Space), and Edit ▸ Undo / Redo and the toolbar buttons name the step; Image Info has
  *Derived from* and *Source* rows.
- N19: *This image only* switches (default on, remembered) on the ROI Manager and the Results
  table, with the hidden count in the badge ("12 / 41"); ROIs are named "Particle 1 (x 315,
  y 25)" and foreign ones carry their image; the Results table signals horizontal overflow and
  scrolls from the keyboard.
- N21, N29 (wizard, layout): the Tracks wizard ticks only steps that succeeded (a "!" on zero
  spots), blocks *Next* with a reason, says "1 spot in 1 frame" and draws the detected spots as
  point ROIs; on a time series the Tracks section moves up below Threshold and opens once by
  itself; the shortcut overlay is taller, in three columns and brighter.
- N22: `lang="en"` on the document and the app root, and on every sheet, display and threshold
  number input, so "2.5" is "2.5" whatever the browser locale.
- N25, N26, N36 (accessibility): the tool rail has a roving tabindex with arrow / Home / End
  keys; channel eye toggles, name inputs, LUT selects and sliders are named with their channel
  and carry `aria-pressed`; the menubar's selector is fixed, triggers have hover and open
  states and 13 px 500-weight text at ≥ 7:1; the status bar's cursor readout is no longer a
  live region; the inspector splitter is focusable and resizes with ← / →; tab close buttons
  are 24 px and visible on inactive tabs (N32); the large 3-D view is a dialog with a focus
  trap; slider `aria-valuenow` is rounded.
- N37: the sheet's *?* opens an inline parameter help panel (label, help, default, unit, range,
  choices) with a *Command reference…* link, and the Design / Research documents are hidden from
  it.
- Small items: the compare pane fits its image on open (N39); the overlay says what "[" does
  with the brush active and that Esc leaves the editor (N40); Z Project's *Stop* is seeded with
  the slice count (N41); the tool-options bar is always present so the canvas does not move
  (N42); Esc on the menubar returns focus to where it was (N27); the top bar shrinks and the
  inspector keeps 420 px for the image below 1000 px (N30); the close dialog no longer calls
  the undo history an "unsaved change" (N31); HUD chips keep their dark glass in the light theme,
  so the threshold badge reads (N20, badge part).

Gaussian's *Scaled units* option (N41, second half) was added afterwards as a real parameter
(`scaled_units`: sigma divided by the pixel width and height, as ImageJ's GaussianBlur does), with the
macro flag `scaled` mapped both ways and three cross-validation cases that are pixel-identical to
ImageJ 1.54p on a 0.5 × 0.25 µm calibration.

**Not done**, and why: the "-1" suffix on a re-opened
sample's name (N4, note) - it is how unique titles work, and the name now appears in the
recorder's `# → "…"` comments; threshold state surviving reloads and the shared ROI Manager on a
shared server (N20, N38) - server-global by design, see the deployment note above; the Preferences
and colour-picker number inputs get their decimal point only through the inherited `lang`.

## 5.5 Fixes made after the re-evaluation (round 3)

E8's re-walk (§3.14) was acted on in the same build cycle (8 Nov 2026, early morning), again in two
packages, each with its own tests; a third pass of the browser checks and the e2e suite followed, and
E9 / E10 re-walked the result (§3.14).

**Engine and API** (`tests/test_undo_resolution.py`, `test_tiff_provenance.py`, `test_bleach.py`, plus
updated `test_ij_export.py`, `test_recorder.py`, `test_batch.py`):

- E7-5 and E8-3 (undone steps in a batch, `run("Undo")` in the macro): one helper,
  `lumen/core/recorder_lines.py`, resolves undo and redo before anything replays a recording. It keeps
  one undo and one redo stack per image title (a `lumen.select`, a `# → "title"` tail, a `lumen.display`
  line or a Save As rename moves the target), pushes only commands that create undo entries (pixel
  commands of kind *plane* or *image*; measurements, ROI Manager, batch and file commands never), and on
  `lumen.undo` drops both the undo line and the step it cancels; `lumen.redo` puts the step back. The
  folder batch's preview and run use it (the summary says "K undone steps removed" and names them), and
  so does the macro export, which never writes `run("Undo")` any more - an undo that cancels nothing stays
  as a comment that explains why. A test records *gaussian → median → undo → undo → redo* and checks that
  only the gaussian survives, and that an undo after a measurement cancels the last pixel command, not the
  measurement; another checks the helper against a real session's History.
- E8-3 (notes, min/max-only): notes are `// note:` lines at the top of the exported script, de-duplicated;
  a display change of only min or only max becomes `getMinAndMax(__lo, __hi); setMinAndMax(__lo, 1500);`,
  round-tripped through the macro layer in a test.
- E7-16 (provenance in the file): the ImageJ-hyperstack TIFF now says `ImageJ=1.54p` (the version the
  cross-validation targets) and `Software = Lumen <version>`, and its *Info* property - what Fiji shows
  under Image ▸ Show Info - carries a processing log: the Lumen version and date, *Derived from*, *Source*,
  and one numbered line per History step with its parameters, appended to whatever Info the file had
  (undone steps are not listed). Lumen reads it back into the image's metadata. The session passes the
  image's History to the writer.
- E7-20 (macro equivalent in the help): every command's JSON carries `ij_commands`, the `run("…")` names
  the macro layer maps to it, inverted from the layer's own table.
- E7-10 (Bleach Correction): `adjust.bleach_correction` implements CorrectBleach's three methods - Simple
  Ratio with background, Exponential Fit through the project's port of ImageJ's CurveFitter ("Exponential
  with Offset", frames scaled by fit(0)/fit(t), half-life reported) and Histogram Matching to the first
  frame - on the T axis (a plain Z stack is treated as time, as the plugin does), per channel, optionally
  inside an area ROI; it produces a new image, a per-frame table and a mean-versus-frame plot, and
  `run("Bleach Correction", "correction=[Exponential Fit] background=0")` maps to it. Tests use a synthetic
  decay `100·e^(−0.1t) + 10` and check that both corrections bring every frame's mean within 2 % of the first.

**Interface** (type check clean, `npm run build`, the e2e suite, `verify_fix_round2.py` and the new
`verify_menus_visible.py`):

- E8-1 (menus invisible): the two overflow rules were removed, and the dropdowns are now rendered through
  a portal to `document.body` with fixed coordinates measured from the trigger (re-measured on resize and
  scroll, flipped at the viewport edges, capped in height with their own scrollbar when taller than the
  room), so no ancestor can clip them again. Submenus are measured and flipped the same way. Narrow windows
  get smaller trigger padding and hidden secondary icons instead of an overflow. The new guard script asks
  the browser which element sits at the centre of each open popover and of its first item
  (`elementFromPoint`) at 1280, 1440 and 1920 px, with the mouse and with F10 - the check the DOM read of
  §5.4 could not make.
- E8-2 (recording lost on reload): the client id lives in the tab's `sessionStorage`, so a reload keeps
  its recorder and a new tab gets its own; the editor warns before unload only when it holds unsaved text.
- E8-4 (detection preview): detection no longer adds ROIs or switches *Show all* on; the wizard draws the
  model's spots of the current frame only, as a preview layer that disappears when the wizard moves past
  the detection steps, with a "N spots on frame k" chip. The ROI Manager count does not change.
- E8-5 (counts): one number everywhere - the registry's command count - and the palette footer names the
  UI actions and LUT entries separately ("498 commands + 51 UI actions + 60 LUTs" at the time).
- E8-6 (first image cut off): the auto-fit view is remembered per image and re-applied when the viewport
  container resizes, until the user zooms or pans; Fit re-arms it.
- E7-26 (decimal commas, found by four evaluators): the numeric fields are no longer `<input type="number">`
  (Chrome formats those in the browser's locale whatever `lang` says) but a shared text input with
  `inputmode="decimal"` that shows a point, accepts a comma, validates, steps with ↑/↓ and keeps the
  `aria-value*` attributes, the inline range message and the disabled Apply - in the operation sheet, the
  Display min/max and threshold fields, Preferences, the Track wizard and the classifier.
- E7-14: the recorder header wraps its buttons; the large editor view takes focus and returns it.
  E7-13: registry items whose label duplicates a client item in the same submenu are dropped from the
  menu (they stay in the palette), "..." becomes "…", and Open Recent shows the folder for duplicate
  names. E7-18: the hint for a missing selection follows the command's selection kind - line commands
  name the line tools, and Plot Profile now declares *line or rectangle*, which the engine validates.
  E7-19: the split view's selects and buttons, and the per-channel Auto / Reset / Inv buttons, have
  accessible names; Esc returns focus from the large 3-D and editor views. E7-20: the sheet help and the
  command reference show `ImageJ: run("Gaussian Blur...")`. E7-24: plurals.

**After E9** (same cycle; `tests/test_display_undo.py`): display edits - a channel's range, LUT, visibility,
the composite mode - are undo steps of their own (a slider drag merges into one; a plane change is none), with
the recorder's `lumen.undo` lines and the undo resolution of the batch and macro export following suit; the
History detail lists every parameter, defaults included, and the scope of a plane command on a stack
("sigma=2.0, scaled_units=False, scope=stack (all planes)"); *Set LUT* takes and records its channel on
multi-channel images; the particles and Measure messages say "on channel 1 (DAPI)"; the Display section's
validation message wraps instead of overlapping; the tool rail shows a thin scrollbar in short windows; the
help overlay fits a 900 px window; the wizard's step chips are gated like its *Next* button.

**Not done**, and why: E7-21 stays by design (§5.4); the editor is still a textarea without a highlighter;
Chrome's *Duplicate tab* copies `sessionStorage`, so a duplicated tab shares its parent's recorder; the
palette's handling of extra words in a query ("fiji lut ice") and of some typos (E9-4) is unchanged; 24 of 80
interactive targets are still under 24 px (E5-16); the inspector keeps its 13 sections.

*Verification of this round:* the unit suite (1209 passed, 281 opt-in skipped), the type check and
build, the e2e suite (10 passed), the 17-check script of §5.4 (all pass) and the new menu guard
(every top-level menu and its first submenu visible and hit-testable at 1280, 1440 and 1920 px, with
the mouse and from the keyboard; the only non-hit is a disabled item, which has `pointer-events: none`
by design). The re-walks by E9 and E10 in §3.14 are the evaluators' own verdict on the same build.

## 5.6 Fixes made after the second re-check (round 4)

E10's list (§3.14) was closed in the same cycle, without agents:

- E10-1: the portaled menu focuses its first item once the popover is visible (it is hidden for one layout
  pass while it is positioned, and a hidden element cannot take focus); `verify_menus_visible.py` now also
  checks that after F10 and ↓ the focus is on a menu item, that ↓ moves it, that → opens the next menu with
  its first item focused and that Esc returns to the trigger.
- E10-3: `NumberInput` reports whether its text parses; the operation sheet disables Apply and says
  "Radius is not a number" until it does.
- E10-6: `stacks.z_project` projects the frames of a single-slice time series when the axis is left at Z,
  as ImageJ's ZProjector does.
- E7-5 (rest): a `rois.restore_cleared` / `results.restore_cleared` line cancels the clear it took back in
  the batch list and the macro export, like an undo.
- E10-4: *Derived from* carries the command's parameters ("t1-head.tif ▸ Z Project (method='Max
  Intensity', start=1, stop=0, axis='Z')"); the first Save As keeps the opened file as *Source*.
- E10-5: a new detection re-seeds the wizard's initial threshold. E7-18: the client knows the
  *line or rectangle* kind and names the line tools and the rectangle. E7-13: the palette lists each name
  once and with one ellipsis style.
- Not done: E10-2 is deliberate (documentation under Help); E7-14, E7-20, E7-21 and E7-24 as before.

*Verification:* type check and build, the unit suite (1215 passed, 281 opt-in skipped), the e2e suite (10 passed), both browser
guards including the keyboard check (20 of 20 checks pass), and a final short re-check by the Opus evaluator of
E10's six items (E11, §3.14).

## 5.7 Fixes made after the final re-check (round 5)

- E11-1: palette (indexed) GIF / PNG / BMP images open the way ImageJ opens them - the 8-bit indices are the pixel
  values and the palette becomes the display LUT: a grey ramp is Grays, a reversed ramp an inverting LUT (Blobs: mean
  103.27 as in ImageJ), any other palette a runtime LUT (`tests/test_indexed_images.py`).
- E11-2: every `file.open_…` line (ImageJ sample, URL, recent file) is left out of the folder batch, like the other
  open lines (`tests/test_e11_fixes.py`).
- E10-3 (rest): Enter in a sheet obeys the same gate as the Apply button.
- E10-4 (rest): the processing log never names the file being written as its own source.
- E10-2: the in-app Markdown renderer joins indented continuation lines to their list item, renders `*italic*`, and
  merges consecutive quote lines; the menu item reads *Design & Research Notes (background)*.
- E7-14 (rest): the script editor has syntax highlighting (Python / Jython, ImageJ macro, JavaScript / Rhino, Groovy /
  BeanShell, JRuby, Clojure: comments, strings, numbers, keywords, calls, the `lumen` / ImageJ built-ins) drawn in a
  layer under the text area, so typing, selection, undo and every shortcut stay the browser's own
  (`frontend/src/highlight.ts`). It is still not an IDE: no completion, no tabs.
- Ecosystem (E7's "no plugin ecosystem or update sites", §3.11): Lumen now reads **Fiji update sites** and installs
  their plugin jars; each jar's commands appear under *Plugins ▸ Fiji* and run with the real ImageJ through the Java
  bridge, also from ImageJ macros by name. Fiji's bundled plugins folder alone declares 349 commands Lumen does not
  implement natively; with a JDK and Fiji's jars present, any of them can now be installed and run
  (`tests/test_fiji_plugins.py` runs Fiji's Auto_Threshold - identical to Lumen's Otsu - and Kuwahara jars from a local
  update site). Interactive Swing plugins and ImageJ2 / SciJava commands remain out of reach.

*Verification:* the unit suite (1227 passed, 281 opt-in skipped), the e2e suite (10 passed), both browser guards (all
checks pass).

## 6. Conclusion

The measured comparison stands: over the ten tasks Lumen needs 54 actions to Fiji's documented
84 and opens no separate window to Fiji's 20, and it is shorter on six tasks and equal on three.
The first heuristic pass (E1) rated Lumen's worst problem under every Nielsen heuristic at 1 or 2
and found nothing worse than large-data handling.

The three persona passes change that picture in one direction. They found 9 further
severity-3 problems, none of which contradicts the counts and none of which E1 had seen: a modified
image closes without confirmation (M1); Analyze Particles silently thresholds an unthresholded
image and reports a plausible count (M2); the File menu advertises ⌘W, which closes the browser
tab (M3); Brightness/Contrast is absent from Image ▸ Adjust (M9); and, for anyone who does not
use a mouse, the menus, the inspector sections, the image tabs and the command palette are
unreachable or silent (M4-M7), with text contrast below AA in both themes (M8). The "keyboard-first"
claim therefore holds for the palette and the single-letter shortcuts and not yet for the rest of
the interface. The problems the evaluators agree on most (inspector density, the server-side file
browser) are consequences of the one-window, browser-hosted design and are rated minor by all.

The second round, by three other models on the fixed build (sections 3.11-3.13), moves the
picture again. The first round's major problems are gone, confirmed by evaluators who did not
know they had existed, with two exceptions caught in the same way (the menubar's contrast and a
handful of unnamed controls, section 3.12). What the three found instead is a different class of
problem: the recording that Lumen's reproducibility story rests on was not faithful - it did not
say which image each command ran on (N9, the only severity-4 rating in the assessment), did not
record undo, display changes or clears, mixed every session into one log, and would have batched
that log blindly over a folder; the engine clamped out-of-range values and swallowed misspelled
keywords without a word. The first round had rated the recorder's gaps minor; the evaluator who
scripts for a living rated them major, and was right. Those problems were fixed (section 5.4), and
then - the step the first round never had - the evaluators went back (section 3.14). The specialist
confirmed 15 of its 27 problems gone and 10 narrowed, and found a severity-4 regression the fix round's
own script had missed: the menus were invisible. The core-facility manager, after the next round,
confirmed 16 of its 20 gone and 4 narrowed and said it would now let a trainee follow a Fiji recipe
alone. Each re-walk left a shorter list, and each list was fixed before the next walk (section 5.5).
The specialist's last re-check (E10) found the round-3 blockers gone and three new severity-2 items
- a keyboard regression of the portaled menus, non-numeric text accepted by a sheet, Z Project refusing a
single-slice time series - all fixed in section 5.6 and covered by the menu guard, which now checks keyboard
focus as well as visibility. The final re-check (E11) confirmed those fixes and said it would hand Lumen to Fiji users
on a single-user workstation once three things were fixed - Enter on a non-numeric value, a sample-open line in the
batch list, and ImageJ's Blobs sample opening as RGB - all three fixed the same day (section 5.7), and a last re-check
(E12) confirmed them and repeated the verdict without conditions. What remains open is listed, not hidden: the shared-server design, the plain editor, the option strings in the help, the inspector's
density.

What this assessment still cannot say is how real users fare. Seven AI evaluations - four by one
assistant role-playing four perspectives, three by other models with one brief - are more than
one evaluation and less than seven evaluators; the single-rating severity-3 findings above are
each one judgement; and no human has used Lumen under observation. The kit in `docs/usability_study/` exists so that the proposed study (section 5.2)
can be run as soon as participants are available; until then the honest summary is: *measurably
fewer actions and windows, two rounds of major problems that were concrete and have been fixed,
and no evidence yet about learnability, errors or satisfaction.*

---

## Sources used in this document

ImageJ / Fiji documentation
- ImageJ User Guide 1.46r: §6 Undo and Redo https://imagej.net/ij/docs/guide/146-6.html ; §8 Stacks and Hyperstacks https://imagej.net/ij/docs/guide/146-8.html ; §19 Tools https://imagej.net/ij/docs/guide/146-19.html ; §26 File menu https://imagej.net/ij/docs/guide/146-26.html ; §27 Edit menu https://imagej.net/ij/docs/guide/146-27.html ; §28 Image menu https://imagej.net/ij/docs/guide/146-28.html ; §29 Process menu https://imagej.net/ij/docs/guide/146-29.html ; §30 Analyze menu https://imagej.net/ij/docs/guide/146-30.html ; §31 Plugins menu https://imagej.net/ij/docs/guide/146-31.html
- ImageJ menu reference: Image https://imagej.net/ij/docs/menus/image.html ; Process https://imagej.net/ij/docs/menus/process.html ; Analyze https://imagej.net/ij/docs/menus/analyze.html ; Window https://imagej.net/ij/docs/menus/window.html
- ImageJ sources: GaussianBlur.java https://imagej.net/ij/developer/source/ij/plugin/filter/GaussianBlur.java.html ; ZProjector.java https://imagej.net/ij/developer/source/ij/plugin/ZProjector.java.html
- ImageJ sample images: https://imagej.net/ij/images/
- ImageJ2 search bar announcement: https://imagej.net/news/2017-12-29-imagej2-and-fiji-updated-now-with-search-bar
- Batch processing: https://imagej.net/scripting/batch
- TrackMate: getting started https://imagej.net/plugins/trackmate/tutorials/getting-started ; scripting https://imagej.net/plugins/trackmate/scripting/scripting ; actions index https://imagej.net/plugins/trackmate/actions ; plugins.config https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/resources/plugins.config ; ExportTracksToXML.java https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/java/fiji/plugin/trackmate/action/ExportTracksToXML.java ; ExportStatsTablesAction.java https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/java/fiji/plugin/trackmate/action/ExportStatsTablesAction.java
- Bioimage Book, ImageJ chapter: https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html

Usability method
- Nielsen, 10 usability heuristics (1994, reviewed 2024): https://www.nngroup.com/articles/ten-usability-heuristics/
- Nielsen, Severity ratings for usability problems (1994): https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/
- Moran & Gordon, How to conduct a heuristic evaluation (2023): https://www.nngroup.com/articles/how-to-conduct-a-heuristic-evaluation/
- Sauro, Measuring usability with the System Usability Scale: https://measuringu.com/sus/
- Brooke, SUS: a "quick and dirty" usability scale (1996), as reproduced at https://www.usability.gov/how-to-and-tools/methods/system-usability-scale.html
- W3C, Web Content Accessibility Guidelines 2.2 (2023): https://www.w3.org/TR/WCAG22/ (1.1.1 Non-text content, 1.3.1 Info and relationships, 1.4.3 Contrast, 1.4.10 Reflow, 2.1.1 Keyboard, 2.3.3 Animation from interactions, 2.4.3 Focus order, 2.5.8 Target size, 4.1.2 Name, role, value)
- W3C, ARIA Authoring Practices Guide patterns (menubar, toolbar, combobox, listbox, dialog, tabs): https://www.w3.org/WAI/ARIA/apg/patterns/

Lumen
- Measurement script and outputs: `tests/e2e/usability/` (`measure_lumen.py`, `lumen_counts.json`, `lumen_counts.md`); the script writes its temporary batch folder next to itself
- Heuristic-walkthrough evidence collector and its outputs: `tests/e2e/usability/heuristic_walkthrough.py`, `tests/e2e/usability/walkthrough/{power,biologist,a11y}/evidence.json` (screenshots are regenerated next to them by re-running the script)
- Re-walks of the fixed builds: `tests/e2e/usability/walkthrough/{e8_opus_recheck,e9_sonnet_recheck,e10_opus_recheck,e11_opus_recheck,e12_opus_recheck}/evaluation.json`
- Second-round evaluations by other models: `tests/e2e/usability/walkthrough/{e5_sonnet,e6_haiku,e7_opus}/evaluation.json` (problems with heuristic, location, observation, severity and screenshot names; positives; verdict; E7 adds a comparison with Fiji and napari)
- User-study kit: `docs/usability_study/` (README with the protocol, briefing, consent, task sheets, SUS, CSV templates, `analyze_results.py`)
- Browser test helpers the script mirrors: `tests/e2e/conftest.py`, `tests/e2e/test_smoke.py`
- Design and evidence: [DESIGN.md](DESIGN.md), [RESEARCH.md](RESEARCH.md), [FIJI_PARITY.md](FIJI_PARITY.md), [PLATFORMS.md](PLATFORMS.md)
