"""Guided, step-by-step workflows (the wizard GUIs Fiji users know from TrackMate, the Trainable
Weka Segmentation, CellProfiler pipelines or the Imaris creation wizards), defined as *data*.

A workflow is an ordered checklist of steps.  Every step points at one registered command (``op``)
with the parameter defaults a novice should start from; the Workflows panel opens the ordinary
operation sheet (with live preview) for it, so a workflow never duplicates UI – it only orders and
explains the commands that already exist.  The same definition is replayed as a Python script
(``build_script``) so a workflow finished interactively becomes a batch pipeline, exactly like the
recorder does for free-form sessions (RESEARCH.md §6–8: CellProfiler test mode, ilastik applets,
Imaris wizards, Aivia "Guided Sequences" – guided entry points for novices on top of one engine).

Workflow entry::

    {"id": str, "title": str, "description": str, "requires": Requirement, "tags": [str],
     "steps": [Step, ...]}

Step entry::

    {"id": str,                 unique inside the workflow
     "title": str,              short imperative label ("Smooth", "Count and measure")
     "help": str,               what the step does / what to look at in the preview
     "op": str | None,          registered op id; None = a manual instruction step
     "params": {name: value},   defaults merged over the op's own Param defaults
     "preview": bool,           open the sheet with live preview switched on
     "optional": bool,          the user may skip it
     "requires": Requirement,   precondition shown as a hint and used to grey the step out
     "outputs_hint": str,       what the step produces ("a new 8-bit mask tab", "Results rows")
     "action": str | None,      manual steps only: a client action to trigger (e.g. "saveAs")
     "script": str | None}      manual steps only: Python line(s) that stand in for the step in a script

Requirement (string, or list of strings that must all hold):

    "none"          nothing needed (the step opens or creates images)
    "image"         an open image
    "selection"     an area/line selection on the active image
    "rois"          entries in the ROI manager
    "threshold"     a threshold set on the active image (Threshold panel / threshold.compute)
    "binary"        the active image is a 0/255 mask
    "labels"        the active image is a label image (the client cannot verify this; it checks "image")
    "stack"         a Z stack (nz > 1)
    "timelapse"     a time series (nt > 1)
    "multichannel"  at least two channels (not RGB)
    "spots"         spots detected in this session (tracking.detect_spots)
    "tracks"        tracks linked in this session (a Tracks table exists)
    "classifier"    a trained or loaded pixel classifier

Parameter values of type ``image`` may use the tokens ``"$original"`` (the image that was active when
the workflow was started) and ``"$active"`` (the active image at the time the step runs); scripts
turn them into ``source.id`` / ``lumen.image.id``.
"""
from __future__ import annotations

import json
from typing import Any

REQUIREMENTS = ("none", "image", "selection", "rois", "threshold", "binary", "labels", "stack", "timelapse",
                "multichannel", "spots", "tracks", "classifier")

_SEG_NUCLEI: list[dict[str, Any]] = [
    {"id": "smooth", "title": "Smooth", "op": "filters.gaussian", "params": {"sigma": 2.0}, "preview": True, "requires": "image",
     "help": "A small Gaussian blur suppresses noise so the threshold follows nuclei, not single pixels. Use a sigma of about a fifth of the nuclear radius; watch the preview – touching nuclei should stay separable.",
     "outputs_hint": "The image is smoothed in place (undo with ⌘Z)."},
    {"id": "background", "title": "Subtract background", "op": "process.subtract_background", "params": {"radius": 50.0, "light_background": False}, "preview": True, "optional": True, "requires": "image",
     "help": "Rolling-ball background subtraction flattens uneven illumination. The radius must be larger than the largest nucleus. Skip it when the background is already flat.",
     "outputs_hint": "Flattened image (in place)."},
    {"id": "threshold", "title": "Choose a threshold", "op": "threshold.compute", "params": {"method": "Otsu", "dark_background": True}, "requires": "image",
     "help": "An automatic threshold separates nuclei from background; Otsu suits bimodal histograms, Li or Triangle dim objects. The red overlay shows what will become foreground – nothing is changed yet.",
     "outputs_hint": "A threshold range on this image (Threshold panel)."},
    {"id": "mask", "title": "Make a mask", "op": "threshold.apply", "params": {"method": "Current"}, "preview": True, "requires": "threshold",
     "help": "Convert the thresholded pixels into a binary mask (255 inside, 0 outside).",
     "outputs_hint": "The image becomes an 8-bit 0/255 mask."},
    {"id": "fill", "title": "Fill holes", "op": "binary.fill_holes", "params": {}, "preview": True, "requires": "binary",
     "help": "Dim nucleoli leave holes in the mask; fill them so every nucleus is one solid object.",
     "outputs_hint": "Mask without holes."},
    {"id": "split", "title": "Split touching nuclei", "op": "binary.watershed", "params": {"method": "Distance transform (h-maxima)", "tolerance": 0.5}, "preview": True, "requires": "binary",
     "help": "A distance-transform watershed cuts touching nuclei apart. Raise the tolerance if objects are over-segmented, lower it if clumps stay joined.",
     "outputs_hint": "Mask with one-pixel cuts between touching objects."},
]


WORKFLOWS: list[dict[str, Any]] = [
    {
        "id": "count_nuclei",
        "title": "Count nuclei",
        "description": "Segment nuclei in a fluorescence image and count and measure them: smooth, threshold, clean up the mask, split touching nuclei, then Analyze Particles.",
        "requires": "image",
        "tags": ["segmentation", "counting", "nuclei", "DAPI"],
        "steps": [
            {"id": "copy", "title": "Work on a copy", "op": "image.duplicate", "params": {"title": "nuclei", "what": "Current plane", "crop_to_selection": True}, "optional": True, "requires": "image",
             "help": "Duplicate the current plane (the nuclear channel) so the original stays untouched for intensity measurements later. Draw a rectangle first to crop to a region of interest.",
             "outputs_hint": "A new tab 'nuclei' that becomes the active image."},
            *_SEG_NUCLEI,
            {"id": "count", "title": "Count and measure", "op": "analyze.particles", "params": {"min_size": 20.0, "max_size": 0.0, "min_circularity": 0.0, "show": "Outlines", "exclude_on_edges": True, "summarize": True, "add_to_manager": True}, "requires": "binary",
             "help": "Find every object of the mask, discard specks below the minimum size and (optionally) objects cut by the image border, then measure them. The count is in the Particle Summary table, one row per nucleus goes to Results, and the outlines go to the ROI manager.",
             "outputs_hint": "Results rows, a Particle Summary table, outline ROIs and an Outlines image."},
            {"id": "summary", "title": "Summarize", "op": "analyze.summarize", "params": {}, "optional": True, "requires": "none",
             "help": "Mean, SD, min and max of every Results column (area, mean intensity …) for a quick sanity check of the population.",
             "outputs_hint": "A Summary table."},
        ],
    },
    {
        "id": "intensity_per_cell",
        "title": "Measure intensity per cell",
        "description": "Segment nuclei on a copy, turn them into labelled objects, optionally grow each label into the cytoplasm, and measure the original image's intensity inside every object.",
        "requires": "image",
        "tags": ["segmentation", "intensity", "per-cell", "labels"],
        "steps": [
            {"id": "copy", "title": "Copy the nuclear channel", "op": "image.duplicate", "params": {"title": "nuclei", "what": "Current plane", "crop_to_selection": True}, "requires": "image",
             "help": "Show the nuclear channel, then duplicate the current plane. The segmentation happens on the copy; intensities are read from the original image you started from, so it must stay open.",
             "outputs_hint": "A new tab 'nuclei' (active); the original remains the intensity source."},
            *_SEG_NUCLEI,
            {"id": "label", "title": "Label the objects", "op": "binary.label", "params": {"connectivity": "8"}, "requires": "binary",
             "help": "Give every connected object its own integer label (shown with a categorical LUT).",
             "outputs_hint": "A new label image (active)."},
            {"id": "expand", "title": "Grow into the cytoplasm", "op": "segment.expand_labels", "params": {"distance": 3.0}, "preview": True, "optional": True, "requires": "labels",
             "help": "Expand every label outwards by a few pixels without overlapping neighbours – a cheap cytoplasm proxy when there is no membrane stain. Skip it to measure the nuclei only.",
             "outputs_hint": "Labels grown by the distance (in place)."},
            {"id": "measure", "title": "Measure every object", "op": "analyze.label_objects", "params": {"intensity_image": "$original"}, "requires": "labels",
             "help": "One Results row per label with area, mean, min, max … of the chosen intensity image (the original image by default – change it in the sheet to measure another channel).",
             "outputs_hint": "Results rows, one per cell."},
        ],
    },
    {
        "id": "track_cells",
        "title": "Track cells",
        "description": "TrackMate-style tracking of a time series: detect spots on every frame, link them with the LAP tracker, drop short tracks and export the tracks as CSV.",
        "requires": "timelapse",
        "tags": ["tracking", "time-lapse", "TrackMate", "spots"],
        "steps": [
            {"id": "detect", "title": "Detect spots", "op": "tracking.detect_spots", "params": {"detector": "LoG", "radius": 2.5, "threshold": 0.0, "subpixel": True, "add_to_manager": True, "add_to_results": False}, "requires": "timelapse",
             "help": "Laplacian-of-Gaussian detection on every frame of the current channel. Set the radius to the typical object radius; the quality threshold 0 picks one automatically – raise it if background specks are detected. Detected spots appear in the ROI manager.",
             "outputs_hint": "Spots (per frame) in the ROI manager."},
            {"id": "link", "title": "Link spots into tracks", "op": "tracking.link_spots", "params": {"max_displacement": 10.0, "gap_closing_frames": 2, "max_gap_distance": 15.0, "min_track_length": 2}, "requires": "spots",
             "help": "Frame-to-frame linking plus gap closing (LAP). The maximum displacement must exceed the largest movement between consecutive frames.",
             "outputs_hint": "Track overlays and a Tracks table (length, duration, speed …)."},
            {"id": "filter", "title": "Filter tracks", "op": "tracking.filter_tracks", "params": {"min_length": 3, "min_duration": 0, "min_displacement": 0.0, "max_speed": 0.0}, "optional": True, "requires": "tracks",
             "help": "Remove tracks that are too short or move too fast to be real cells.",
             "outputs_hint": "Updated overlays and Tracks table."},
            {"id": "export", "title": "Export tracks", "op": "tracking.export_tracks", "params": {"path": "tracks.csv", "layout": "TrackMate"}, "requires": "tracks",
             "help": "Write one row per spot of every track to a CSV file in the TrackMate column layout (readable by TrackMate, napari, pandas).",
             "outputs_hint": "A CSV file."},
        ],
    },
    {
        "id": "pixel_classifier",
        "title": "Train a pixel classifier",
        "description": "Trainable (Weka-style) segmentation: scribble a few regions per class, train a random forest on texture features, apply it, then count the objects.",
        "requires": "image",
        "tags": ["machine learning", "Weka", "ilastik", "segmentation"],
        "steps": [
            {"id": "annotate", "title": "Draw training regions", "op": None, "params": {}, "requires": "image",
             "help": "Easiest in Inspector ▸ Pixel Classifier: pick a class and paint a few strokes on it with the brush (the strokes become class-named ROIs). Or draw a region on pure background with the freehand or rectangle tool, add it to the ROI manager (T) and rename it 'background' in the ROI panel; draw regions on the objects and name them 'foreground'. Two or three small regions per class are enough; the ROI name only needs to start with the class name ('foreground 2').",
             "outputs_hint": "ROIs named background… and foreground… in the ROI manager."},
            {"id": "train", "title": "Train the classifier", "op": "segment.train_classifier", "params": {"classes": "background,foreground", "channels": "Current channel", "max_sigma": 8.0, "n_trees": 100, "apply_now": True, "output": "Probabilities"}, "requires": "rois",
             "help": "Learn a random forest from the pixels under the class ROIs (Gaussian, DoG, Sobel, Hessian … features at several scales) and show the class probabilities for the current plane. Add ROIs where the result is wrong and train again – this is the ilastik 'live update' loop.",
             "outputs_hint": "A Probabilities image (one channel per class); the classifier stays in the session."},
            {"id": "apply", "title": "Apply and make a mask", "op": "segment.apply_classifier", "params": {"all_planes": True, "output": "Labels", "smooth": 1.0, "mask_class": "foreground", "probability_threshold": 0.5}, "requires": "classifier",
             "help": "Classify the whole image (every plane) with the trained classifier and threshold the foreground probability into a binary mask. Select the original image first if a probability image is active.",
             "outputs_hint": "A label image and a 'foreground mask' image (active)."},
            {"id": "count", "title": "Count the objects", "op": "analyze.particles", "params": {"min_size": 20.0, "show": "Outlines", "exclude_on_edges": False, "summarize": True}, "requires": "binary",
             "help": "Analyze Particles on the foreground mask: discard specks, measure the rest and add outlines to the ROI manager.",
             "outputs_hint": "Results rows, a Particle Summary table and outline ROIs."},
        ],
    },
    {
        "id": "colocalization",
        "title": "Colocalization",
        "description": "Coloc 2-style colocalization of two channels: optionally flatten the background, compute Costes thresholds with Pearson/Manders coefficients and a significance test, and the classic Manders split within the selection.",
        "requires": "multichannel",
        "tags": ["colocalization", "Pearson", "Manders", "Costes", "two channels"],
        "steps": [
            {"id": "background", "title": "Flatten the background", "op": "process.subtract_background", "params": {"radius": 50.0, "light_background": False}, "preview": True, "optional": True, "requires": "multichannel",
             "help": "Uneven background inflates Pearson's coefficient. Rolling-ball subtraction on every channel (the default scope is the whole stack) removes it; skip for images that are already flat.",
             "outputs_hint": "Background-corrected channels (in place)."},
            {"id": "costes", "title": "Costes thresholds and coefficients", "op": "analyze.coloc_costes", "params": {"channel_a": 1, "channel_b": 2, "scramble_iterations": 100, "block_size": 5, "show_scatter": True}, "requires": "multichannel",
             "help": "Draw a selection around the cell first to exclude empty background. Costes automatic thresholds, Pearson above/below them, thresholded Manders tM1/tM2, Spearman and a scrambled-block significance test (p-value), plus the 2-D scatter plot.",
             "outputs_hint": "Results rows and a 2-D histogram image."},
            {"id": "manders", "title": "Manders with Otsu thresholds", "op": "analyze.colocalization", "params": {"channel_a": 1, "channel_b": 2, "threshold_a": 0, "threshold_b": 0}, "optional": True, "requires": "multichannel",
             "help": "The classic Pearson/Spearman/Manders coefficients with Otsu (or manual) thresholds per channel, for comparison with the Costes result.",
             "outputs_hint": "Results rows."},
        ],
    },
    {
        "id": "stitch_tiles",
        "title": "Stitch tiles",
        "description": "Grid/Collection stitching: fuse a folder of overlapping tiles acquired in a regular grid into one image, add a scale bar and save it.",
        "requires": "none",
        "tags": ["stitching", "tiles", "mosaic", "grid"],
        "steps": [
            {"id": "stitch", "title": "Stitch the grid", "op": "registration.grid_stitch", "params": {"folder": "", "pattern": "*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 10.0, "order": "Row-by-row", "refine": True, "fusion": "Linear blend"}, "requires": "none",
             "help": "Point at the folder of tiles, give the grid size, acquisition order and nominal overlap. Every neighbour pair is refined by phase correlation; pairs with a poor correlation fall back to the grid position.",
             "outputs_hint": "One stitched image (new tab)."},
            {"id": "scalebar", "title": "Add a scale bar", "op": "overlay.scale_bar", "params": {"length": 0, "position": "Lower Right"}, "optional": True, "requires": "image",
             "help": "Needs the pixel size (Image ▸ Properties) – tiles usually carry it. Length 0 picks a round number automatically.",
             "outputs_hint": "A scale bar in the overlay."},
            {"id": "save", "title": "Save the mosaic", "op": None, "action": "saveAs", "script": 'lumen.save("stitched.tif")', "requires": "image",
             "help": "File ▸ Save As… (⇧⌘S) writes the stitched image as an ImageJ-compatible TIFF (or OME-TIFF / PNG).",
             "outputs_hint": "A file on disk."},
        ],
    },
    {
        "id": "zstack_figure",
        "title": "Z-stack to figure",
        "description": "Turn a Z stack into a publication panel: maximum-intensity projection, contrast, scale bar and LUT calibration bar, burn them in and export.",
        "requires": "stack",
        "tags": ["figure", "projection", "scale bar", "export"],
        "steps": [
            {"id": "project", "title": "Project the stack", "op": "stacks.z_project", "params": {"method": "Max Intensity", "start": 1, "stop": 0}, "requires": "stack",
             "help": "A maximum-intensity projection shows every bright structure of the stack in one plane; use Average or Sum for quantitative panels. Limit the slice range to leave out blurry top/bottom slices.",
             "outputs_hint": "A new projected image (active)."},
            {"id": "contrast", "title": "Set the contrast", "op": "process.enhance_contrast", "params": {"saturated": 0.35, "normalize": False}, "optional": True, "requires": "image",
             "help": "Stretch the display range so 0.35 % of the pixels saturate (pixel values stay untouched unless 'normalize' is on). Fine-tune per channel in the Display panel.",
             "outputs_hint": "An adjusted display range."},
            {"id": "scalebar", "title": "Add a scale bar", "op": "overlay.scale_bar", "params": {"length": 0, "thickness": 6, "position": "Lower Right", "show_label": True}, "requires": "image",
             "help": "Needs a calibrated pixel size (Image ▸ Properties). Length 0 picks a round number; keep the label for the figure legend or hide it and state the length in the caption.",
             "outputs_hint": "A scale bar in the overlay."},
            {"id": "calbar", "title": "Add a calibration bar", "op": "overlay.calibration_bar", "params": {"position": "Upper Right", "lut": "Current", "width": 20, "height": 128}, "optional": True, "requires": "image",
             "help": "A LUT colour bar with the current channel's display range – useful for pseudo-coloured intensity or ratio images, unnecessary for plain fluorescence panels.",
             "outputs_hint": "A calibration bar in the overlay."},
            {"id": "flatten", "title": "Flatten", "op": "overlay.flatten", "params": {}, "requires": "image",
             "help": "Burn the overlay (scale bar, calibration bar, annotations) and the current LUTs/composite into a new RGB image – exactly what you see.",
             "outputs_hint": "A new RGB image (active)."},
            {"id": "export", "title": "Export the panel", "op": None, "action": "saveAs", "script": 'lumen.save("figure.png")', "requires": "image",
             "help": "File ▸ Save As… (⇧⌘S) as PNG or TIFF for the figure assembly tool.",
             "outputs_hint": "A PNG/TIFF file."},
        ],
    },
    {
        "id": "batch_folder",
        "title": "Batch over a folder",
        "description": "Turn an interactive analysis into a batch run: do it once on an example image while the recorder watches, then replay the recorded commands on every file of a folder.",
        "requires": "none",
        "tags": ["batch", "folder", "recorder", "macro"],
        "steps": [
            {"id": "example", "title": "Analyse one example image", "op": None, "params": {}, "requires": "none",
             "help": "Open a representative image and run the analysis interactively – a workflow from this panel, or any commands. Every applied command is recorded as a Python line (Script & Recorder panel). Open/save lines are left out of the replay automatically; clear the recorder first if it contains earlier experiments.",
             "outputs_hint": "Recorded lumen.run(...) lines in the Script & Recorder panel."},
            {"id": "batch", "title": "Replay on a folder", "op": "batch.apply_recorded_macro", "params": {"input_folder": "", "pattern": "\\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$", "recursive": False, "output_folder": "", "output_format": "tif", "save_results": True, "close_after": True, "continue_on_error": True}, "requires": "none",
             "help": "Every matching file is opened, the recorded commands run on it, the processed image is saved to the output folder (default <input>/lumen_output) and measurements are appended to Results.csv with a File column.",
             "outputs_hint": "Processed images and Results.csv in the output folder."},
        ],
    },
    {
        "id": "register_timelapse",
        "title": "Register a time-lapse",
        "description": "Remove stage drift from a time series, then track the objects: drift correction by phase correlation, one-click detect + link, optional filtering and CSV export.",
        "requires": "timelapse",
        "tags": ["registration", "drift", "time-lapse", "tracking"],
        "steps": [
            {"id": "drift", "title": "Correct drift", "op": "registration.drift_correct", "params": {"channel": 0, "reference": "Previous frame", "correct_z": False, "upsample": 10, "interpolation": "Linear", "create_new": False}, "requires": "timelapse",
             "help": "Estimates the XY shift of every frame (from the max projection, by phase correlation) and shifts all slices and channels back. 'Previous frame' follows slow drift; 'First frame' is better for jumps. The shifts go to the Alignment table.",
             "outputs_hint": "A registered stack (in place or as a new tab) and an Alignment table."},
            {"id": "track", "title": "Detect and link", "op": "tracking.track", "params": {"detector": "LoG", "radius": 2.5, "threshold": 0.0, "max_displacement": 10.0, "gap_closing_frames": 2, "min_track_length": 2}, "requires": "timelapse",
             "help": "One-click TrackMate: LoG spot detection on every frame and LAP linking with gap closing. Set the radius to the object radius and the maximum displacement above the largest frame-to-frame movement.",
             "outputs_hint": "Track overlays and a Tracks table."},
            {"id": "filter", "title": "Filter tracks", "op": "tracking.filter_tracks", "params": {"min_length": 3}, "optional": True, "requires": "tracks",
             "help": "Keep only tracks of at least N spots (and/or below a maximum speed).",
             "outputs_hint": "Updated overlays and Tracks table."},
            {"id": "export", "title": "Export tracks", "op": "tracking.export_tracks", "params": {"path": "tracks.csv", "layout": "TrackMate"}, "optional": True, "requires": "tracks",
             "help": "One row per spot of every track as CSV (TrackMate column layout).",
             "outputs_hint": "A CSV file."},
        ],
    },
    {
        "id": "edit_labels",
        "title": "Edit labels",
        "description": "MorphoLibJ's Label Edition as a guided session: pick a label with the value picker, select, remove or merge labels, grow or shrink them, fill holes, drop small or border labels, keep the largest – every step previews and undoes with ⌘Z.",
        "requires": "labels",
        "tags": ["labels", "MorphoLibJ", "editing", "segmentation"],
        "steps": [
            {"id": "pick", "title": "Pick a label", "op": None, "params": {}, "requires": "labels",
             "help": "Choose the value picker (I) and click a label: its value becomes the foreground value (status bar). The steps below use that picked label whenever their label value is 0 or their label list is empty – or type the values in their sheets. The point tool (shift-click adds points) marks several labels for merging.",
             "outputs_hint": "The picked label value (foreground value)."},
            {"id": "select", "title": "Select the label", "op": "morpho.select_label", "params": {"label": 0, "add_to_manager": False}, "optional": True, "requires": "labels",
             "help": "Outline the picked label as the active selection – to check which object you picked, measure it, or add it to the ROI manager.",
             "outputs_hint": "A selection around the label."},
            {"id": "remove", "title": "Remove labels", "op": "morpho.replace_label", "params": {"labels": "", "new_value": 0}, "preview": True, "optional": True, "requires": "labels",
             "help": "Set the picked label (or a list such as '3, 5-7') to 0. The same sheet renumbers labels when the new value is not 0. Remove Label at Point (the ⌫ tool) deletes labels with one click each.",
             "outputs_hint": "Labels removed (in place)."},
            {"id": "merge", "title": "Merge labels", "op": "morpho.merge_labels", "params": {"labels": "", "new_value": 0}, "preview": True, "optional": True, "requires": "labels",
             "help": "Give several labels one value: list them, or click them with the point tool first (shift-click adds points) and leave the list empty. The merged object takes the smallest value unless you set one.",
             "outputs_hint": "Merged labels (in place)."},
            {"id": "grow", "title": "Grow or shrink", "op": "morpho.label_morphology", "params": {"operation": "Dilation", "radius": 1}, "preview": True, "optional": True, "requires": "labels",
             "help": "Dilate every label by n pixels (labels never overlap each other) or erode them; opening and closing smooth the outlines. Watch the preview and raise the radius step by step.",
             "outputs_hint": "Labels grown / shrunk (in place)."},
            {"id": "fill", "title": "Fill holes", "op": "morpho.fill_label_holes", "params": {}, "preview": True, "optional": True, "requires": "labels",
             "help": "Background holes enclosed by a label get its value (a hole between labels is split between them).",
             "outputs_hint": "Labels without holes (in place)."},
            {"id": "size", "title": "Drop small labels", "op": "morpho.label_size_opening", "params": {"min_size": 20, "max_size": 0}, "preview": True, "optional": True, "requires": "labels",
             "help": "Remove every label with fewer pixels than the minimum (and, optionally, the ones above a maximum) – specks left by the segmentation.",
             "outputs_hint": "Small labels removed (in place)."},
            {"id": "border", "title": "Drop border labels", "op": "segment.remove_border_labels", "params": {"buffer": 0}, "preview": True, "optional": True, "requires": "labels",
             "help": "Delete the labels touching the image edge (objects cut by the field of view bias size measurements).",
             "outputs_hint": "Border labels removed (in place)."},
            {"id": "largest", "title": "Keep the largest", "op": "morpho.keep_largest_label", "params": {}, "preview": True, "optional": True, "requires": "labels",
             "help": "Keep only the largest label, for example the cell of interest among debris.",
             "outputs_hint": "One label left (in place)."},
            {"id": "undo", "title": "Undo a step", "op": None, "action": "undo", "script": "# undo (⌘Z) is interactive – nothing to replay here", "requires": "image",
             "help": "Every edit is one undo step: ⌘Z (Edit ▸ Undo) takes the last one back, ⇧⌘Z redoes it. Reopen a step to run it again with other values.",
             "outputs_hint": "The previous labels."},
            {"id": "rois", "title": "Labels to ROIs", "op": "segment.labels_to_rois", "params": {"min_size": 0, "simplify": True, "clear_manager": False}, "optional": True, "requires": "labels",
             "help": "Add one outline ROI per label to the ROI manager for measurements on other channels, or render the result with Labels To RGB / Label Overlay.",
             "outputs_hint": "ROIs named L<label> in the ROI manager."},
        ],
    },
]

_STEP_DEFAULTS: dict[str, Any] = {"op": None, "params": {}, "preview": False, "optional": False, "requires": "image",
                                  "outputs_hint": "", "help": "", "action": None, "script": None}


def _norm_requires(req: Any) -> list[str]:
    if req is None:
        return ["image"]
    if isinstance(req, str):
        return [req]
    return [str(r) for r in req]


def normalized(wf: dict[str, Any]) -> dict[str, Any]:
    """A workflow with every step key filled in (what the API serves)."""
    out = {"id": wf["id"], "title": wf["title"], "description": wf.get("description", ""),
           "requires": _norm_requires(wf.get("requires", "image")), "tags": list(wf.get("tags", [])), "steps": []}
    for st in wf["steps"]:
        s = {**_STEP_DEFAULTS, **st}
        s["requires"] = _norm_requires(s["requires"])
        s["params"] = dict(s["params"] or {})
        out["steps"].append(s)
    return out


def all_workflows() -> list[dict[str, Any]]:
    return [normalized(w) for w in WORKFLOWS]


def get_workflow(workflow_id: str) -> dict[str, Any]:
    for w in WORKFLOWS:
        if w["id"] == workflow_id:
            return normalized(w)
    raise KeyError(f"Unknown workflow {workflow_id!r}")


def summaries() -> list[dict[str, Any]]:
    out = []
    for w in all_workflows():
        out.append({"id": w["id"], "title": w["title"], "description": w["description"], "requires": w["requires"], "tags": w["tags"],
                    "n_steps": len(w["steps"]), "n_required": sum(1 for s in w["steps"] if not s["optional"]),
                    "ops": [s["op"] for s in w["steps"] if s["op"]]})
    return out


def with_op_info(wf: dict[str, Any]) -> dict[str, Any]:
    """The full workflow with the op name/kind/preview of every step resolved from the registry."""
    from .registry import REGISTRY
    out = dict(wf)
    steps = []
    for s in wf["steps"]:
        s = dict(s)
        if s["op"]:
            spec = REGISTRY.get(s["op"])
            s["op_name"] = spec.name
            s["op_kind"] = spec.kind
            s["op_preview"] = bool(spec.preview and spec.kind != "analysis")
        else:
            s["op_name"] = None
            s["op_kind"] = None
            s["op_preview"] = False
        steps.append(s)
    out["steps"] = steps
    return out


# ------------------------------------------------------------------ validation
def validate(workflows: list[dict[str, Any]] | None = None) -> list[str]:
    """Check every workflow against the registry; returns a list of problems (empty = valid)."""
    from .registry import REGISTRY
    problems: list[str] = []
    seen_ids: set[str] = set()
    for raw in workflows if workflows is not None else WORKFLOWS:
        wid = raw.get("id")
        if not wid or not isinstance(wid, str):
            problems.append(f"workflow without id: {raw!r}"[:120])
            continue
        if wid in seen_ids:
            problems.append(f"{wid}: duplicate workflow id")
        seen_ids.add(wid)
        try:
            wf = normalized(raw)
        except KeyError as e:
            problems.append(f"{wid}: missing key {e}")
            continue
        if not wf["title"]:
            problems.append(f"{wid}: empty title")
        for r in wf["requires"]:
            if r not in REQUIREMENTS:
                problems.append(f"{wid}: unknown requirement {r!r}")
        if not wf["steps"]:
            problems.append(f"{wid}: no steps")
        step_ids: set[str] = set()
        for s in wf["steps"]:
            sid = s.get("id")
            where = f"{wid}/{sid}"
            if not sid:
                problems.append(f"{wid}: step without id")
                continue
            if sid in step_ids:
                problems.append(f"{where}: duplicate step id")
            step_ids.add(sid)
            if not s.get("title"):
                problems.append(f"{where}: empty title")
            if not s.get("help"):
                problems.append(f"{where}: empty help")
            for r in s["requires"]:
                if r not in REQUIREMENTS:
                    problems.append(f"{where}: unknown requirement {r!r}")
            if s["op"] is None:
                if s["params"]:
                    problems.append(f"{where}: manual step has params")
                continue
            if s["action"]:
                problems.append(f"{where}: only manual steps may carry an action")
            try:
                spec = REGISTRY.get(s["op"])
            except KeyError:
                problems.append(f"{where}: unknown op {s['op']!r}")
                continue
            if spec.id != s["op"]:
                problems.append(f"{where}: op {s['op']!r} should be referenced by its id {spec.id!r}")
            if s["preview"] and not (spec.preview and spec.kind != "analysis"):
                problems.append(f"{where}: {spec.id} cannot preview")
            by_name = {p.name: p for p in spec.params}
            for k, v in s["params"].items():
                p = by_name.get(k)
                if p is None:
                    problems.append(f"{where}: {spec.id} has no parameter {k!r}")
                    continue
                if p.type == "image":
                    if v is not None and not (isinstance(v, str) and v):
                        problems.append(f"{where}: {k} must be an image id/name or a $token")
                    continue
                try:
                    coerced = p.coerce(v)
                except ValueError as e:
                    problems.append(f"{where}: {e}")
                    continue
                if p.type in ("int", "float") and isinstance(v, (int, float)) and not isinstance(v, bool) and coerced != v:
                    problems.append(f"{where}: {k}={v!r} is outside {p.min}–{p.max}")
                if p.type == "bool" and not isinstance(v, bool):
                    problems.append(f"{where}: {k} must be a bool")
                if p.type == "choice" and p.choices and v not in p.choices:
                    problems.append(f"{where}: {k}={v!r} is not one of {p.choices}")
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    problems.append(f"{where}: {k} is not JSON-serialisable")
    return problems


# ------------------------------------------------------------------ script generation
def step_params(step: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full, coerced parameter dict of a step: op defaults ← workflow defaults ← overrides."""
    from .registry import REGISTRY
    spec = REGISTRY.get(step["op"])
    merged = {**step.get("params", {}), **(overrides or {})}
    out: dict[str, Any] = {}
    for p in spec.params:
        v = merged.get(p.name, p.default)
        if p.type == "image" and isinstance(v, str) and v.startswith("$"):
            out[p.name] = v  # token, resolved by the script / the client
        else:
            out[p.name] = p.coerce(v)
    return out


def _literal(name: str, value: Any, ptype: str) -> str:
    from .session import _py
    if ptype == "image" and isinstance(value, str) and value.startswith("$"):
        return {"$original": "source.id", "$active": "lumen.image.id"}.get(value, _py(value))
    return _py(value)


def build_script(workflow_id: str, params: dict[str, dict[str, Any]] | None = None, skip: list[str] | None = None,
                 include_optional: bool = True) -> dict[str, Any]:
    """A Python script that replays the workflow with ``lumen.run`` lines like the recorder.

    ``params`` maps step id → parameter overrides, ``skip`` lists step ids to leave out (optional steps
    only; a required step in ``skip`` is still emitted but commented), ``include_optional=False`` leaves
    every optional step out unless it has overrides.
    """
    from .registry import REGISTRY
    wf = get_workflow(workflow_id)
    params = params or {}
    skipped = set(skip or [])
    lines: list[str] = [f"# {wf['title']} — a Lumen guided workflow", f"# {wf['description']}",
                        "# Run it from the Script & Recorder panel, or on every file of a folder with Process ▸ Batch ▸ Process Folder…"]
    needs_source = any(any(v == "$original" for v in {**s["params"], **(params.get(s["id"]) or {})}.values()) for s in wf["steps"])
    if needs_source or any(r != "none" for r in wf["requires"]):
        lines.append("source = lumen.image  # the image that is active when the script starts")
    emitted: list[dict[str, Any]] = []
    n = 0
    for s in wf["steps"]:
        sid = s["id"]
        overrides = params.get(sid) or {}
        leave_out = (sid in skipped or (not include_optional and not overrides)) and s["optional"]
        n += 1
        head = f"# {n}. {s['title']}"
        if leave_out:
            lines.append("")
            lines.append(f"{head} — skipped")
            emitted.append({"id": sid, "included": False})
            continue
        lines.append("")
        lines.append(head + (" (optional)" if s["optional"] else ""))
        if s["op"] is None:
            if s.get("script"):
                lines.append(s["script"])
            else:
                for h in _wrap(s["help"]):
                    lines.append(f"#    {h}")
            emitted.append({"id": sid, "included": True, "manual": True})
            continue
        spec = REGISTRY.get(s["op"])
        p = step_params(s, overrides)
        types = {q.name: q.type for q in spec.params}
        args = ", ".join(f"{k}={_literal(k, v, types[k])}" for k, v in p.items())
        lines.append(f'lumen.run("{spec.id}"{", " if args else ""}{args})')
        emitted.append({"id": sid, "included": True, "op": spec.id, "params": p})
    return {"id": wf["id"], "title": wf["title"], "script": "\n".join(lines) + "\n", "steps": emitted}


def _wrap(text: str, width: int = 96) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]
