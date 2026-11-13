# Task sheets

Ten cards, one per task, each with the same wording for both tools and the tool-specific end state
the observer checks. Print one card per page; show one card at a time; read it aloud; start the
clock when you finish reading. The participant may re-read the card at any time. Do not show the
"end state" or "moderator notes" parts to the participant.

## 0. Materials (prepare before the first session)

Put these five items on the Desktop of the study laptop, in a folder named `study`:

| Name | Content | Lumen source | Fiji source |
|---|---|---|---|
| `cells_3ch.tif` | 3-channel fluorescence image, 512×512, 16-bit | Open Sample ▸ *Synthetic nuclei (3 channels)*, then File ▸ Save As | the same file (Lumen writes an ImageJ hyperstack TIFF) |
| `blobs.tif` | 8-bit image with blob-like objects | Open Sample ▸ *Blobs* | File ▸ Open Samples ▸ Blobs (25K) or the same file |
| `stack_2ch.tif` | 2-channel z-stack, 256×256×60 | Open Sample ▸ *Cells (3-D, 2 channels)* | the same file |
| `timelapse.tif` | nuclei time-lapse, 6+ frames | Open Sample ▸ *Nuclei timelapse (synthetic)* | the same file |
| `folder10/` | ten small single-plane TIFFs | ten crops of `blobs.tif` (any 128×128 windows) | the same folder |

Both tools must read the same files; check once that every file opens with the expected
dimensions in both before the first participant. Reset between participants: close all images,
clear results and ROIs, empty the Lumen recorder (Script & Recorder ▸ Clear) and the Fiji Recorder
window, delete everything the previous participant wrote to `study/out/`.

Conventions below: **Goal** is read to the participant. **End state** is what the observer verifies
(do not read it). **Counts** are the expert path lengths from USABILITY.md §2.2 for reference only.

---

## Task 1 - Open a multi-channel image and adjust each channel

**Goal (read aloud).** "Open `cells_3ch.tif` from the study folder. Make each of the three channels
look reasonable on screen - adjust the brightness or display range of every channel separately so
that none is saturated and none is too dark. Tell me when you are happy with how it looks."

**Materials.** `cells_3ch.tif`. Start with no image open.

**End state.** The image is open and the display range of all three channels has been changed
from the file's defaults (Lumen: three different ranges in the Display section; Fiji: B&C changed
on each channel of the composite). The pixel data are unchanged (Fiji: *Apply* was not pressed;
Lumen: no history entry).

**Success.** `complete` if all three channels were adjusted without help. If only one channel was
adjusted and the participant believes the task is done, ask "are all three channels adjusted?"
once; log `assist` if that was needed.

**Moderator notes.** Watch where they look first (menu, search, memory). Fiji's B&C adjusts the
current channel; Shift-drag adjusts all. Lumen shows all three channels' sliders at once.
Counts: Fiji 9, Lumen 9.

---

## Task 2 - Threshold and count the particles

**Goal.** "Open `blobs.tif`. Threshold it so that the blobs are selected, then count and measure
the blobs so that you get a table with one row per blob and the blob outlines are kept as regions
you could reuse."

**Materials.** `blobs.tif`.

**End state.** A results table with one row per blob (Fiji: Results window; Lumen: Results section)
and the outlines in the ROI manager (Fiji: ROI Manager window with Add to Manager ticked; Lumen:
ROI Manager section). Any sensible threshold counts; expect roughly 55-65 blobs.

**Success.** `complete` if both the table and the ROIs exist. Running Analyze Particles without a
threshold and accepting the result is an `err_wrong_command` (Fiji refuses; Lumen thresholds
silently - note which it was).

**Moderator notes.** Counts: Fiji 7, Lumen 4. In Lumen the Threshold section has a *Particles…*
button; via Analyze ▸ Analyze Particles the sheet opens too.

---

## Task 3 - Measure a drawn rectangle

**Goal.** "On the blobs image, draw a rectangle anywhere that contains a few blobs and measure the
area and mean intensity inside it."

**Materials.** `blobs.tif` still open from task 2 (clear the results first, or ask the participant
to ignore the earlier rows).

**End state.** One new results row with Area and Mean for a rectangular selection.

**Success.** `complete` if the row exists. Measuring the whole image because no selection was
active is an `err_wrong_command`.

**Moderator notes.** Counts: Fiji 2, Lumen 3 as measured (Lumen's pointer tool also draws a
rectangle; see USABILITY.md E3-13). Watch whether they look for the rectangle tool or just drag.

---

## Task 4 - Maximum-intensity projection

**Goal.** "Open `stack_2ch.tif`, which is a z-stack with two channels. Make a maximum-intensity
projection of the whole stack."

**Materials.** `stack_2ch.tif`.

**End state.** A new image (Fiji: `MAX_stack_2ch.tif` window; Lumen: `MAX_…` tab) with z = 1 and
two channels.

**Success.** `complete` if the projection exists with the maximum method. Average or sum
projection counts as `assist` only if the participant noticed and asked; otherwise log `complete`
with a note (the card says maximum).

**Moderator notes.** Counts: Fiji 4, Lumen 4.

---

## Task 5 - Composite with one colour per channel

**Goal.** "Go back to the 3-channel image from task 1. Make it a colour composite in which the
three channels are shown in cyan, magenta and yellow, all overlaid in one view."

**Materials.** `cells_3ch.tif` (reopen if closed; the display changes of task 1 may persist).

**End state.** Composite display mode with the three channel LUTs set to Cyan, Magenta, Yellow
(Fiji: Channels Tool More ▸ colours, or Image ▸ Lookup Tables per channel; Lumen: Display section
LUT dropdowns).

**Success.** `complete` if all three LUTs are set and the view is composite. Converting to RGB first
(so that the LUTs can no longer be changed) is an `err_destructive`.

**Moderator notes.** This is variant 5a of USABILITY.md (counts: Fiji 9, Lumen 6). Variant 5b
(merge three single-channel images) is not used here to keep the session at 75 minutes.

---

## Task 6 - Filter with a preview, then undo

**Goal.** "On the blobs image, apply a Gaussian blur with sigma 3. Before you apply it, look at a
preview of the result. After you have applied it, undo it so that the image is sharp again."

**Materials.** `blobs.tif`.

**End state.** The history shows the blur applied and then undone (Lumen: History section empty
again or the undo entry gone; Fiji: Edit ▸ Undo used once, the image looks sharp). The participant
saw a preview before applying (Fiji: Preview checkbox; Lumen: the preview is on by default - ask
"did you see a preview?" and log the answer in the notes).

**Success.** `complete` if blurred with sigma 3 and undone. Blurring twice and undoing once is an
`err_destructive` (the image stays blurred).

**Moderator notes.** Counts: Fiji 7, Lumen 6. In Fiji the Preview checkbox must be ticked; in
Lumen the Preview switch is on. This task must precede task 9.

---

## Task 7 - Track the nuclei and export the tracks

**Goal.** "Open `timelapse.tif`. Track the nuclei over time with the default settings and export
the tracks to a file in `study/out/` - any text format, as long as there is one row per detected
spot with its track ID, frame and position."

**Materials.** `timelapse.tif`; the empty folder `study/out/`.

**End state.** A file in `study/out/` with columns like `TRACK_ID, FRAME, POSITION_X, POSITION_Y`
(Lumen: Tracks ▸ Export CSV; Fiji: TrackMate ▸ Export tracks to XML, or Export statistics to
tables then save the spots table as CSV).

**Success.** `complete` if the file exists with per-spot rows. Abandon after 5 minutes is expected
for some participants in Fiji's wizard; log it, then show the path.

**Moderator notes.** Counts: Fiji 21-27, Lumen 7. Pass criterion for this task is 75 % (README §6).

---

## Task 8 - Find a command you do not know the location of

**Goal.** "Find and open the command called CLAHE - a local contrast enhancement - without me
telling you where it is in the menus. Stop when its dialog or settings are on screen; you do not
need to run it."

**Materials.** Any image open.

**End state.** The CLAHE dialog (Fiji: *Enhance Local Contrast (CLAHE)* dialog; Lumen: the *CLAHE
(Local Contrast)* sheet) is visible.

**Success.** `complete` if reached by any means (menu browsing, Command Finder / search bar,
palette). Log `first_lookup`: this task is the main measure of search discoverability.

**Moderator notes.** Counts: Fiji 3, Lumen 3. Fiji: `L` opens the Command Finder; Lumen: ⌘K / Ctrl+K
opens the palette, also the search field in the top bar.

---

## Task 9 - Batch-process a folder with what you just recorded

**Goal.** "Apply the same Gaussian blur you used in task 6 to every image in `study/folder10/` and
save the results into `study/out/` - without opening the ten images one by one. The recording of
your earlier steps may help."

**Materials.** `folder10/` with ten TIFFs; `study/out/`; the recorder has task 6's blur in it
(Fiji: the Recorder window was open during task 6 - the moderator opens it at the start of the
session and says so in the tour; Lumen: the recorder is always on).

**End state.** Ten blurred images in `study/out/` (Fiji: Process ▸ Batch ▸ Macro with the recorded
line; Lumen: *Apply Recorded Macro to Folder…* or *Process Folder…* with the recorded line).

**Success.** `complete` if ten output files exist and are blurred. Processing with the wrong sigma
is `complete` with a note. Writing into the input folder is an `err_destructive`.

**Moderator notes.** Counts: Fiji 20, Lumen 11. Pass criterion 75 % (README §6). Lumen's folder
fields are typed paths (no Browse button): watch for this.

---

## Task 10 - Compare two images side by side

**Goal.** "Show the blobs image and the 3-channel image side by side so that you can compare them,
and zoom into both at once."

**Materials.** `blobs.tif` and `cells_3ch.tif` open.

**End state.** Both images visible at the same time without overlapping (Fiji: Window ▸ Tile, plus
Analyze ▸ Tools ▸ Synchronize Windows for linked zoom; Lumen: split view, ⌘⇧\ or the split button
next to the tabs) and a zoom applied to both.

**Success.** `complete` if both are visible side by side and zoomed. Tiling without synchronised
zoom in Fiji is `complete` with a note ("zoomed separately") because the card says "at once" but
many Fiji users do not know Synchronize Windows; record it.

**Moderator notes.** Counts: Fiji 2 (6 with sync), Lumen 1.

---

## After the last task (both tools)

Hand over the SUS sheet (`sus_questionnaire.md`), then ask the two open questions and write the
answers verbatim in `sus_responses.csv`:

1. "What would stop you from using this tool for your own data?"
2. "What did you miss?"
