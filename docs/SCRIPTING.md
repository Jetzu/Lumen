# Lumen scripting and plugin reference

Python is Lumen's only scripting language. The same command registry drives the palette, the menus,
the parameter sheets, the macro recorder and the API, so anything you can click you can script with
`lumen.run("<command id>", …)`, and anything you write as an `@op` function becomes a command with a
sheet, a live preview and undo. Command ids and parameters are listed in [COMMANDS.md](COMMANDS.md);
worked examples are in [WORKFLOWS.md](WORKFLOWS.md). Every snippet marked `# example:` below was run
through the console runner before it was written down.

## Contents

- [Where scripts run](#where-scripts-run)
- [The `lumen` object](#the-lumen-object)
  - [Images](#images)
  - [Running commands](#running-commands)
  - [Results, ROIs, log](#results-rois-log)
  - [`LumenImage`, `Calibration`, `Display`](#lumenimage-calibration-display)
  - [The session](#the-session)
- [ImageJ compatibility layer: `IJ`, `run()` and the macro functions](#imagej-compatibility-layer-ij-run-and-the-macro-functions)
- [Running Fiji scripts (Jython, JavaScript, macros)](#running-fiji-scripts-jython-javascript-macros)
  - [JVM languages through the Java bridge (Groovy, BeanShell, Rhino, JRuby, Clojure)](#jvm-languages-through-the-java-bridge-groovy-beanshell-rhino-jruby-clojure)
- [Writing `@op` plugins](#writing-op-plugins)
  - [`Param` types](#param-types)
  - [Kinds of command](#kinds-of-command)
  - [`OpContext`](#opcontext)
  - [`OpResult`](#opresult)
  - [Previews, ROI awareness, stack scope, undo](#previews-roi-awareness-stack-scope-undo)
  - [Progress and cancellation](#progress-and-cancellation)
  - [A complete plugin file](#a-complete-plugin-file)
  - [The plugins folder and startup scripts](#the-plugins-folder-and-startup-scripts)
- [Batch commands](#batch-commands)
- [Headless use (no server)](#headless-use-no-server)
- [REST API](#rest-api)
  - [Endpoints](#endpoints)
  - [Pixel data](#pixel-data)
  - [WebSocket events and jobs](#websocket-events-and-jobs)
  - [A client in 30 lines](#a-client-in-30-lines)
- [Examples](#examples)

---

## Where scripts run

| Place | How | Namespace |
|---|---|---|
| **Script & Recorder** section of the inspector | type or paste, **Run** (⌘↵). Output and tracebacks appear below the editor. **API help** inserts `lumen.commands(…)` / `lumen.help(…)` calls; the **Recorder** list shows every command applied in the UI as a `lumen.run(…)` line (copy, **→ editor**, **Clear**) | `lumen`, `np` / `numpy`, `scipy`, `skimage`, the ImageJ macro functions (`run`, `makeRectangle`, `nResults` …) and `IJ` |
| Batch commands | `batch.process_folder` / `batch.run_script_on_open_images` run a script once per file / image with `lumen.image` bound to it (see [Batch commands](#batch-commands)) | same |
| REST | `POST /api/script/run {"code": "…"}` → `{"stdout": "…", "error": null \| "traceback"}` | same |
| Plugin files | `~/.lumen/plugins/*.py` are *imported* when the server starts; they register `@op` commands and may configure the session (see [plugins folder](#the-plugins-folder-and-startup-scripts)) | plain Python modules |
| Your own Python | `from lumen.core.session import Session` … (see [Headless use](#headless-use-no-server)) | plain Python |

`exit()` / `quit()` / `sys.exit()` end the *script* (ImageJ `exit` semantics), never the server or a batch;
a message or non-zero status is reported as the script's error. `print()` goes to the script output
(and to the Batch table's *Message* column); `lumen.log()` goes to the **Log** section.

---

## The `lumen` object

`lumen` is a `LumenAPI` façade over the session (`lumen/core/scripting.py`).

### Images

| Member | Meaning |
|---|---|
| `lumen.image` | the active `LumenImage` (or `None`) |
| `lumen.images` | list of all open `LumenImage`s, in tab order |
| `lumen.data` | the active image's pixels as a **(T, Z, C, Y, X)** numpy array — a view: edit it in place, then call `lumen.update()` |
| `lumen.plane` | the current 2-D plane (`display.t`, `display.z`, `display.c`) |
| `lumen.open(path)` | open a file (any supported format) → `LumenImage`; also added to *Recent* and recorded |
| `lumen.save(path, fmt=None)` | save the active image; the format follows the extension (`.tif` ImageJ hyperstack, `.ome.tif`, `.png` / `.jpg` rendered RGB, `.npy`, `.nii`, `.h5`, `.zarr`, `.avi` / `.mp4` with the video extra …) unless `fmt` is given → the path written |
| `lumen.select(name_or_id)` | make an image active (by tab name or id) → the image |
| `lumen.close(name_or_id=None)` | close an image (default: the active one) |
| `lumen.undo(name_or_id=None)` · `lumen.redo(name_or_id=None)` | one undo / redo step on an image (default: the active one) → the label of the step, or `None` |
| `lumen.display(name_or_id=None, channel=None, **settings)` | change display state without touching pixels: per channel (1-based) `min`, `max`, `lut`, `visible`; or `mode` (`composite` / `single` / `grayscale`) and the plane `t`, `z`, `c` (0-based) → the display dict. These are the lines the recorder writes for inspector changes |
| `lumen.new_image(array, name="Untitled", axes=None, like=None)` | add a numpy array as a new image. `axes` names the array's dimensions with the letters `T Z C Y X` (e.g. `"ZYX"`, `"CYX"`, `"TZCYX"`; `S` for RGB samples); without it 2-D is `YX`, 3-D is `ZYX` (or `YXS`, i.e. RGB, for a `uint8` array whose last dimension is 3 or 4), 4-D `ZCYX`, 5-D `TZCYX`. `like=img` copies that image's calibration |
| `lumen.update(array=None, label="Script edit")` | commit a numpy edit of the active image as one undo step: with `array`, replace the pixels (any axes order accepted by `to_canonical`); without, keep the in-place edits of `lumen.data` and refresh the display |

```python
# example: numpy_edit
import numpy as np
lumen.run("file.sample", name="Blobs")
img = lumen.image
print(img.shape, img.dtype, lumen.plane.shape)            # (1, 1, 1, 256, 256) uint8 (256, 256)
lumen.data[0, 0, 0, :50, :] = 0                           # edit the view in place …
lumen.update(label="black top rows")                      # … and commit (undoable)
ratio = lumen.plane.astype(np.float32) / 255.0
lumen.new_image(ratio, name="ratio", like=img)            # a new 32-bit image with the same calibration
mip = np.random.default_rng(0).random((5, 64, 64)).astype(np.float32)
stack = lumen.new_image(mip, name="stack", axes="ZYX")
print(stack.shape, stack.nz)                              # (1, 5, 1, 64, 64) 5
lumen.select(img.name)
print(lumen.session.histories[img.id].labels())           # {'undo': ['black top rows'], 'redo': []}
lumen.session.undo(img.id)
print(int(lumen.plane[:50].max()))                        # back to the original values
```

### Running commands

```python
lumen.run(command, scope="stack", roi=None, image=None, **params)
```

- `command` is a command id (`"filters.gaussian"`) — the display name (`"Gaussian Blur"`) is accepted too.
- `scope` (plane commands only): `"plane"` (current plane; all channels of an RGB image), `"slice"` (all
  channels of the current slice/frame), `"channel"` (the current channel on every slice and frame), `"stack"`
  (every plane, the default). This is the *Apply to* selector of the sheet.
- `roi`: a ROI dict for this call; otherwise the script's current selection (`lumen.roi`, set with
  `lumen.set_roi()` / `lumen.rect()`) is used. Area ROIs mask plane filters and restrict analyses; commands
  with `requires_roi` raise without one.
- `image`: run on another open image (name or id) without making it active.
- Parameters are checked and coerced exactly like the sheet does: `int`/`float` outside the declared range
  raise `ValueError("sigma must be between 0 and 200 (got 500)")` rather than being clamped, `choice` is
  matched case-insensitively, and an unknown name raises
  `unknown parameter 'sigmaa' for filters.gaussian – did you mean 'sigma'?`; an unknown command id raises
  `Unknown command 'filters.gaussia' – did you mean filters.gaussian, …?`. Missing parameters take their
  defaults.

**Return value** — the most useful part of the command's payload, in this order of precedence:

1. the **plot** dict if the command produced one (`{"title", "x", "series": [{"name", "values"}], "x_label", "y_label"}`),
2. the **Results rows it added** (list of dicts) if it measured something,
3. the **new images** it created (list of `LumenImage`),
4. otherwise the raw payload dict: `op`, `new_images` (ids), `results_added`, `rois_added`, `message`, `log`,
   `table`, `roi`, `clear_roi`, `threshold`, `histogram`, `elapsed_ms` — whichever apply.

So `lumen.run("analyze.particles", …)` returns rows, `lumen.run("stacks.z_project")` returns `[image]`, and
after a command that both measures and creates images (Grid Stitching, Train Pixel Classifier) the rows come
back while the image is reached through `lumen.image` (new images become active) or `lumen.images`. For the
complete payload call `lumen.session.run_op(id, params_dict, …)` directly.

Commands run from a script are **not** written to the recorder (the UI's and the REST API's are); a
script that creates files or windows behaves exactly as the command does in the UI (new tabs, toasts in
the Log, undo steps on the image).

**What the recorder writes.** Every UI action that changes an image or the session becomes one line of the
same API: `lumen.run("filters.gaussian", sigma=2.0, scope='slice')`; `lumen.select("t1-head.tif")` whenever
the image a command runs on differs from the previous line's; `lumen.close("blobs.tif")`;
`lumen.undo("t1-head.tif")` / `lumen.redo(...)`; `lumen.display("cells.tif", channel=2, min=100, max=1500,
lut='Viridis', visible=False)` for inspector display changes (consecutive edits of one channel collapse into
one line; commands that change the display, such as `color.lut`, are not recorded twice);
`lumen.run("rois.clear")`, `lumen.run("rois.delete", indices='1,3-5')`, `lumen.run("analyze.clear_results")`
and their `rois.restore_cleared` / `results.restore_cleared` counterparts. Lines that create images carry
the resulting title as a trailing comment (`# → "MAX_t1-head.tif"`). The recorder is per browser client
(the page sends an `X-Lumen-Client` id; requests without it share a default recorder) and starts with a
marker line (`# Lumen recorder – session started 2026-11-08 04:00:01`) that is not counted as a command.

**Three export formats.** The recorder pane turns the recording into Python (the lines as they are), JavaScript
(`→ JavaScript`, async `lumen` calls for the browser runtime) or an ImageJ-1 macro (`→ IJM`, `GET /api/recorder/ijm`,
`lumen.core.ij_export.recorder_to_ijm`). The macro export inverts the macro layer's own tables, so every command the
layer imports with `run("Name", "key=value")` exports the same way (`run("Gaussian Blur...", "sigma=3 stack")`,
`makeRectangle(…)` before a measurement, `setAutoThreshold("Otsu dark")`, `Stack.setChannel` / `setMinAndMax` /
`run("Fire")` for display lines, `selectWindow` / `close("title")`, `newImage`, `roiManager("reset")` …). Undo and
redo are resolved first (`lumen.core.recorder_lines.resolve_undo`): a step the user undid is left out together with
its `lumen.undo` line, a redo puts it back, so the macro reproduces the image the user ended with and never contains
`run("Undo")` (ImageJ's undo is single-level); the folder batch (`batch.apply_recorded_macro`, `GET
/api/recorder/preview`) uses the same resolution and says how many undone steps it removed. A line that has no ImageJ
equivalent – a bundled sample, a batch command, a parameter ImageJ's dialog does not have – is kept as `// not
converted (reason): …` and listed in `skipped`; caveats are `// note: …` lines at the top of the script (and `notes`
in the response); a display change of only min or max becomes `getMinAndMax(__lo, __hi); setMinAndMax(__lo, 1500);`.
The tests run an exported macro through the macro layer and compare the pixels with the recorded session
(`tests/test_ij_export.py`, `tests/test_undo_resolution.py`).

| Member | Meaning |
|---|---|
| `lumen.commands(search="")` | list of `"<id>  <name>"` strings whose id or name contains `search` |
| `lumen.help(command)` | one line per parameter: name, type, default, choices, help |

### Results, ROIs, log

| Member | Meaning |
|---|---|
| `lumen.results` | the Results table as a list of row dicts (`{"Label": …, "Area": …}`); columns are added as they appear |
| `lumen.clear_results()` | empty it |
| `lumen.rois` | the ROI Manager as a list of ROI dicts (each with `id`, `name`, `image_id`, `position`) |
| `lumen.add_roi(roi)` | add a ROI to the manager, attached to the active image → the stored entry (always a new entry, even for a dict that already carries a manager `id`) |
| `lumen.set_roi(roi_or_None)` | set the script's current selection (used by subsequent `run()` calls) |
| `lumen.rect(x, y, w, h)` | set and return a rectangular selection |
| `lumen.roi` | the current selection dict (or `None`) |
| `lumen.log(*values)` | append a line to the **Log** section |
| `lumen.session` | the underlying `Session` (below) |
| `lumen.ij` | the ImageJ façade (below) |

ROI dictionaries:

| type | keys |
|---|---|
| `rect`, `oval` | `x, y, w, h` (pixels, floats allowed) |
| `polygon`, `freehand`, `polyline`, `point` | `points: [[x, y], …]` |
| `line` | `x1, y1, x2, y2`, optional `width` |
| `mask` | produced by combining ROIs (`lumen.core.roi.combine(rois, (h, w), "or" \| "and" \| "xor")`) and by *Create Selection* |

Optional keys on any ROI: `name`, `color` (`"#rrggbb"`), `stroke_width`, `position` (`{"t", "z", "c"}` — the
plane it belongs to; set automatically when added from an image). `lumen.core.roi` has the helpers
`is_area`, `is_line`, `to_mask(roi, (h, w))`, `from_mask(mask)`, `label_to_rois(labels)`, `bounding_box`,
`line_length`, `polygon_points`, `translate`, `scale_roi`, `combine`.

### `LumenImage`, `Calibration`, `Display`

`lumen.core.image.LumenImage(data, name="Untitled", axes=None, rgb=False, calibration=None, metadata=None, path=None, channel_names=None)`

| Attribute / method | Meaning |
|---|---|
| `id`, `name`, `path` | 10-character id, tab name (unique per session), source file or `None` |
| `data` | `(T, Z, C, Y, X)` numpy array, dtype `uint8`, `uint16` or `float32` (RGB: `uint8` with `C = 3` and `rgb=True`) |
| `shape`, `nt`, `nz`, `nc`, `height`, `width`, `dtype`, `bit_depth` (`"8-bit"` …), `is_stack`, `is_binary`, `nbytes` | derived |
| `rgb` | the three channels are R, G, B samples (converted with `type.rgb`, split with `type.rgb_stack`) |
| `calibration` | `Calibration(pixel_width, pixel_height, pixel_depth, unit, frame_interval, time_unit, value_unit)`; `.scaled` is true when the unit is not pixels; `.to_dict()` |
| `display` | `Display(channels=[ChannelDisplay(name, lut, min, max, visible, inverted, gamma)], mode="composite" \| "single" \| "grayscale", t, z, c)` — the current position **and** the rendering; change it directly (`img.display.c = 1`) or through `session.update_display(id, patch)` |
| `overlay` | list of overlay items (`{"type": "scalebar" \| "text" \| "arrow" \| "calibration_bar" \| "timestamp" \| ROI types …}`); replaced by `OpResult.overlay` |
| `metadata` | free dict (slice labels, OME/ImageJ metadata, `source`, tracking data …) |
| `rev` | bumped whenever the pixels change (cache key for clients) |
| `current_plane()` | the 2-D plane at `display.t/z/c` |
| `iter_planes()` | every `(t, z, c)` index |
| `replace_data(data, rgb=None)` | set a new 5-D array (resyncs channels, clamps the position, bumps `rev`) |
| `copy(name=None, data=None)` | a copy (same calibration, display and metadata), optionally with other pixels |
| `reset_display(channel_names=None, keep_luts=False)` | auto range per channel |
| `summary()` | the JSON the UI shows (`/api/images/{id}`) |
| `LumenImage.to_canonical(array, axes=None, rgb=False)` | static: any array → `(T, Z, C, Y, X)` |

### The session

`lumen.session` is the `Session` (`lumen/core/session.py`) the UI, the REST API and the scripts share:

| Member | Meaning |
|---|---|
| `images` (`OrderedDict` id → image), `active`, `active_id`, `get_image(name_or_id)`, `select(id)`, `add_image(img, activate=True)`, `close_image(id)`, `open_file(path)`, `save_image(id, path, fmt=None)` | image management |
| `run_op(op_id, params=None, image_id=None, roi=None, scope="stack", preview=False, record=True)` | run a command and get the full payload (`preview=True` returns `{"preview": planes}` instead) |
| `results` (`ResultsTable`: `rows`, `columns`, `append(rows)`, `clear()`, `to_csv()`), `tables` (name → `{"columns", "rows", …}` for Particle Summary, Tracks, Branches, Batch …), `last_plot` | tables |
| `roi_manager` (`RoiManager`: `rois`, `add(roi, image_id=None, position=None)`, `remove(ids)`, `clear()`, `get(id)`, `update(id, patch)`, `to_list()`) | ROIs |
| `measurements` (list of ids), `measure_options` (`limit_to_threshold`, `decimals`, `display_label`, `invert_y`, …), `thresholds` (image id → `{"lower", "upper"}`), `settings` (foreground/background values, preferences, internal state) | what *Set Measurements*, the Threshold panel and Preferences edit |
| `histories` (image id → `History` with `labels()`), `undo(id)`, `redo(id)` | undo |
| `macro` (the recorder's lines), `record(line)`, `log`, `add_log(msg, level="info")`, `recent` | recorder, log |
| `update_display(id, patch)` | patch `t`, `z`, `c`, `mode`, `channels: [{name, lut, min, max, visible, inverted, gamma}]` |
| `state()`, `list_jobs()`, `cancel_job(id)`, `listeners` | what `/api/state`, `/api/jobs` return; `listeners` are callables receiving every event dict |

---

## JavaScript in the browser

The Script & Recorder section of the inspector (shortcut `[`) runs Python (on the server), ImageJ macros
(through the compatibility layer) and **JavaScript**. JavaScript runs inside the browser in a sandboxed Web
Worker (no network access except the Lumen API) with an asynchronous `lumen` object; use `await` for every
call. Open… / Save… read and write script files on the server (`/api/textfile`) or in the browser, ⌘R /
Ctrl+R runs the script, ⇧⌘R runs the selection, and the recorder can export the recorded steps as JavaScript.

```javascript
const img = await lumen.openSample("Blobs");
await lumen.run("filters.gaussian", { sigma: 2 });
await lumen.run("threshold.apply", { method: "Otsu" });
await lumen.run("analyze.particles", { min_size: 20, show: "Outlines" });
const rows = await lumen.getResults();
lumen.log(`${rows.length} particles, mean area ${rows.reduce((s, r) => s + r.Area, 0) / rows.length}`);
```

| Call | Meaning |
|---|---|
| `run(id, params, {roi, scope, image})` | run any command (same ids as Python's `lumen.run`) |
| `getImage()`, `getImages()`, `getState()` | the active image summary, all images, the session state |
| `openSample(name)`, `open(path)`, `save(path)`, `select(id)`, `close(id)` | image management |
| `measure()`, `getResults()`, `setResults(rows)`, `clearResults()` | the Results table |
| `getRois()`, `addRoi(roi)`, `setRoi(roi)`, `clearRoi()` | selections and the ROI Manager |
| `macro(code)`, `python(code)` | run an ImageJ macro or Python snippet on the server |
| `commands()`, `help(id)`, `log(text)`, `print(text)`, `wait(ms)` | discovery and output |

## ImageJ compatibility layer: `IJ`, `run()` and the macro functions

`lumen/core/ij_compat.py` implements ImageJ-1's macro world on top of the session:

- `IJ` (also `lumen.ij`) — `IJ.run("Gaussian Blur...", "sigma=2")` executes an ImageJ menu command: the
  command name is looked up in a table of ~190 rows (214 names with aliases) and its option string is
  translated into the parameters of the Lumen command (`stack` → every plane, `size=20-Infinity`,
  `channels=1-2`, `[bracketed values]`, …). A Lumen command id works as well: `run("filters.gaussian", "sigma=3")`.
  LUT names (`run("Fire")`) apply that LUT. Unknown or unsupported commands raise `ValueError`.
- `IJ.macro(text, arg="")` — transpiles ImageJ-1 macro code to Python and executes it; returns what the
  macro printed (its Log). `arg` is what `getArgument()` returns inside. `IJ.transpile(text)` shows the Python.
- `IJ.roi` — the current selection, shared with `lumen.roi`.
- The macro functions are methods of `IJ` **and** are injected into every script's namespace:
  `run`, `open`, `selectWindow`, `selectImage`, `getTitle`, `getImageID`, `getWidth`, `getHeight`,
  `getDimensions`, `nImages`, `nSlices`, `setSlice`, `getSliceNumber`, `makeRectangle`, `makeOval`,
  `makeLine`, `makePolygon`, `makePoint`, `makeSelection`, `getSelectionBounds`, `getSelectionCoordinates`,
  `selectionType`, `setThreshold`, `setAutoThreshold`, `getThreshold`, `resetThreshold`, `roiManager`,
  `nResults`, `getResult`, `getResultLabel`, `setResult`, `updateResults`, `getStatistics`, `getRawStatistics`,
  `getHistogram`, `getPixel`, `setPixel`, `getMinAndMax`, `setMinAndMax`, `resetMinAndMax`, `getPixelSize`,
  `setVoxelSize`, `getVoxelSize`, `setColor`, `setForegroundColor`, `fill`, `drawLine`, `drawString`,
  `newImage`, `rename`, `close`, `save`, `saveAs`, `print`, `d2s`, `pad`, `split`, `replace`, `matches`,
  `substring`, `indexOf`, `parseInt`, `parseFloat`, `newArray`, `lengthOf`, `getTime`, `getDateAndTime`,
  `getDirectory`, `getFileList`, `eval`, `runMacro`, `getArgument`, `exit`, … (171 in total — the list is in
  [FIJI_PARITY.md](FIJI_PARITY.md#imagej-macro-compatibility-layer)) plus the namespaces `Array`, `Math`,
  `String`, `File`, `Dialog`, `List`, `Fit`, `Stack`, `Roi`, `Overlay`, `Table`, `Color`, `Property`.
  Names that would shadow a Python builtin (`open`, `round`, `abs`, `print` …) are *not* injected — use
  `IJ.open(path)` etc. for those; `nResults`, `nSlices` and `nImages` are functions (`nResults()`).
- Dialog-type functions cannot wait for a user: `Dialog.*`, `getNumber`, `getString`, `getBoolean` return
  their defaults, `showMessage` prints to the Log, `showStatus` / `showProgress` / `setBatchMode` /
  `waitForUser`-free macros run unattended; `waitForUser` itself is unsupported and stops the macro.
- The commands **Run ImageJ Macro** (`script.run_ij_macro`, parameter `macro`) and **ImageJ Command List**
  (`script.ij_command_list`) expose the same layer in the UI; `lumen.core.ij_compat.command_table()`
  returns the mapping table and `parse_options("sigma=2 stack")` the option parser.

```python
# example: ij
lumen.run("file.sample", name="Blobs")
IJ.run("Gaussian Blur...", "sigma=2")                     # ImageJ command + option string
IJ.run("filters.median", "radius=1")                      # a Lumen id works too
makeRectangle(20, 20, 60, 40)
run("Measure")
print(nResults(), round(getResult("Mean", nResults() - 1), 1))
log = IJ.macro('''
setAutoThreshold("Otsu dark");
run("Convert to Mask");
run("Analyze Particles...", "size=30-Infinity add");
print("count=" + roiManager("count"));
''')
print(log.strip(), "|", len(lumen.rois), "ROIs")
print(IJ.transpile('a = newArray(1, 2, 3); print(a.length);').strip())
```

---

## Running Fiji scripts (Jython, JavaScript, macros)

Fiji's *Script Editor* runs seven languages on the JVM. Lumen has no JVM of its own: the three that do not
need one run here, unchanged, with the ImageJ 1.x classes provided by the shim in `lumen/core/ijshim`; the JVM
languages run through the optional Java bridge (`pip install "lumen-imaging[jvm]"` + a JDK, see below):

| Fiji language | Lumen | How |
|---|---|---|
| **ImageJ 1.x macro** (`.ijm`) | ✅ | the compatibility layer above (`run("…")`, ~190 commands, 171 macro functions) |
| **Jython** (`.py`, `from ij import IJ`) | ✅ | Lumen's Python with importable `ij.*` / `java.*` / `jarray` modules (Script panel language *Python* or *Jython (ImageJ 1.x API)*, `POST /api/script/run`, batch commands, headless `lumen.core.ijshim.bind(session)`) |
| **JavaScript** (`.js`, `importClass(Packages.ij.IJ)`) | ✅ | the browser's sandboxed worker, with ImageJ's auto-imported class names and a synchronous bridge to the same shim (`POST /api/ijshim/call`) |
| **Groovy** (`.groovy`), **BeanShell** (`.bsh`), **JavaScript on Rhino** | ✅ | the [Java bridge](#jvm-languages-through-the-java-bridge-groovy-beanshell-rhino-jruby-clojure): a `java` subprocess over a Fiji installation's jars with the real `ij.jar` in ImageJ's batch mode – needs a JDK and Fiji, no Python package (Script panel languages marked *needs the Java bridge*, `POST /api/script/run` with `language`, command `script.run_jvm_script`); alternatively the in-process JPype bridge on Linux / Windows |
| **JRuby** (`.rb`) | ✅ | the same subprocess bridge with Fiji's `jruby-core` + `jruby-stdlib` jars (gems disabled); in-process: `LUMEN_JVM_LANGUAGES=jruby` |
| **Clojure** (`.clj`) | ✅ | the same bridges; Fiji ships `clojure-1.12.jar` without `spec.alpha` / `core.specs.alpha` – put those two jars (Maven Central, ~640 KB) on `LUMEN_JVM_CLASSPATH`. Verified 13 Nov 2026: an Invert and a radius-2 Median from Clojure are bit-identical to Lumen's own, `#@` parameters and the Results table come back |
| **Compiled ImageJ-1 plugins** (`.jar` / `.class`) | ✅ | through the same bridge: name the jar (or class folder) on `LUMEN_JVM_CLASSPATH` and call `IJ.runPlugIn(imp, "the.PlugIn", "")` from a Groovy / BeanShell / Rhino script; a plugin that shows a `GenericDialog` reads its options from `Macro.setOptions("key=value")` when called from a thread named `Run$_…` (example below); verified with Fiji's `Auto_Threshold-1.18.0.jar` – its Otsu mask equals Lumen's on every pixel |
| **Compiled Java plugins**, SciJava / ImageJ2 API | ❌ | not runnable; port them to Python (the `lumen` API or the `ij` shim) |

Also not available: SciJava `OpService` / `CommandService` (`ij.op()…`), ImgLib2 / ImageJ2 (`net.imglib2`,
`net.imagej`) and arbitrary Java classes – `import` of an unknown `ij.*` / `java.*` module raises an `ImportError`
that names the modules the shim provides.

### Jython

```python
# example: jython
#@ Double (label="Sigma", value=2.0) sigma
from ij import IJ, ImagePlus, WindowManager
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.gui import OvalRoi

IJ.run("Blobs (25K)")
imp = IJ.getImage()
IJ.run(imp, "Gaussian Blur...", "sigma=%g" % sigma)      # any command of the compatibility layer
ip = imp.getProcessor()
print(imp.getTitle(), imp.getWidth(), ip.getPixel(10, 10), imp.getStatistics().mean > 0)
imp.setRoi(OvalRoi(20, 20, 60, 60))
IJ.setAutoThreshold(imp, "Otsu dark")
IJ.run(imp, "Convert to Mask", "")
IJ.run(imp, "Analyze Particles...", "size=20-Infinity display add")
rm = RoiManager.getInstance()
rt = ResultsTable.getResultsTable()
for i in range(rt.size()):
    IJ.log("%s %.1f" % (rm.getName(i), rt.getValue("Area", i)))
print(rm.getCount(), WindowManager.getImageTitles())
```

What the shim covers (every class is a Python class over the session; `imp.img` is the `LumenImage`):

- `ij.IJ` – `run(cmd[, options])`, `run(imp, cmd, options)`, `getImage`, `open`, `openImage` (no tab until
  `imp.show()`), `createImage` / `createHyperStack` / `newImage`, `selectWindow`, `log`, `showMessage`,
  `showMessageWithCancel`, `error`, `showStatus` / `showProgress` (no-ops), `getNumber` / `getString` /
  `getDirectory` / `getFilePath` (answered from the script parameters, see below), `saveAs` / `save` /
  `saveAsTiff`, `setThreshold` / `setAutoThreshold` / `resetThreshold`, `setMinAndMax` / `resetMinAndMax`,
  `makeRectangle` / `makeOval` / `makeLine` / `makePoint` / `makePolygon` / `doWand`, `runMacro` /
  `runMacroFile`, `runPlugIn("ij.plugin.frame.RoiManager")`, `d2s`, `pad`, `freeMemory`, `getVersion`,
  `isResultsWindow`, `renameResults`, `deleteRows`, `setTool` / `getToolName`, `setForegroundColor`,
  `isMacOSX` / `isWindows` / `isLinux`, `getLuts`, `beep` / `wait` (no-ops), `exit`.
- `ij.ImagePlus` – title, id, size and dimension getters, `getType` / `getBitDepth`, `getProcessor` (a live view
  of the current plane), `getStack`, `setStack` / `setProcessor`, `setSlice` / `setPosition` / `setC` / `setZ` /
  `setT` and the getters, `getCalibration` (live) / `setCalibration`, `getRoi` / `setRoi` / `deleteRoi` /
  `killRoi`, `getStatistics` / `getAllStatistics`, `getOverlay` / `setOverlay`, `duplicate`, `crop`, `resize`,
  `flatten`, display range and mode, `setActiveChannels`, `setDimensions`, `getProperty` / `setProperty` /
  `getProp`, `getOriginalFileInfo`, `show` / `hide` / `close` / `isVisible` / `getWindow`, `updateAndDraw`;
  `ImagePlus(title, ip | stack)`, `CompositeImage`.
- `ij.ImageStack` – `getSize`, `getProcessor(n)`, `getPixels(n)`, `setPixels`, `getSliceLabel` /
  `setSliceLabel`, `addSlice(label, ip)`, `deleteSlice`, `getVoxel` / `setVoxel`, `duplicate`, `crop`; stacks built
  from processors become images with `ImagePlus(title, stack)`.
- `ij.process.ImageProcessor` (+ `ByteProcessor`, `ShortProcessor`, `FloatProcessor`, `ColorProcessor`) –
  `getPixel` / `putPixel` / `get` / `set` / `getf` / `setf` / `getPixelValue` / `putPixelValue` /
  `getPixels` (a numpy view, writes go through) / `setPixels` / `getRow` / `getColumn` / `getLine`,
  `getWidth` / `getHeight` / `getBitDepth`, `getStatistics` / `getStats` / `getHistogram`, `getMin` / `getMax` /
  `setMinAndMax` / `resetMinAndMax`, `setRoi` / `getRoi` / `getMask` / `resetRoi`, `setColor` / `setValue` /
  `fill` / `fillOutside` / `drawLine` / `drawRect` / `fillRect` / `drawOval` / `fillOval` / `drawString` /
  `fillPolygon`, `add` / `subtract` / `multiply` / `divide` / `min` / `max` / `and` / `or` / `xor` / `gamma` /
  `log` / `exp` / `sqr` / `sqrt` / `abs` / `invert` / `noise` / `threshold` / `applyTable`,
  `setThreshold` / `setAutoThreshold` / `getMinThreshold` / `getMaxThreshold` / `getAutoThreshold`,
  `blurGaussian` / `smooth` / `sharpen` / `findEdges` / `medianFilter` / `erode` / `dilate` / `skeletonize` /
  `outline` / `convolve` / `convolve3x3` / `filter` (Lumen commands on that plane), `translate` / `rotate` /
  `flipHorizontal` / `flipVertical` / `scale` / `resize` / `crop` / `duplicate` / `createProcessor`,
  `convertToByte` / `convertToShort` / `convertToFloat` / `convertToRGB` / `toFloat`, `snapshot` / `reset`,
  `insert` / `copyBits` (all `Blitter` modes), `isBinary` / `isInvertedLut` / `invertLut`, the `ISODATA`, `BLUR_MORE`
  … constants; `ImageStatistics` (`mean`, `min`, `max`, `stdDev`, `area`, `pixelCount`, `histogram`, `median`,
  `mode`, `xCentroid`, `yCentroid`, `skewness`, `kurtosis`, `major` / `minor` / `angle` …), `StackStatistics`,
  `AutoThresholder` (`getThreshold(Method.Otsu, histogram)`), `ImageConverter` / `StackConverter`, `FloatPolygon`.
- `ij.measure` – `ResultsTable` (`getResultsTable()` is the Results section; `ResultsTable()` is private until
  `show(title)`): `incrementCounter`, `addValue`, `addLabel`, `setValue`, `getValue`, `getStringValue`,
  `getLabel`, `getColumn`, `getColumnIndex`, `columnExists`, `getHeadings` / `getColumnHeadings`, `size` /
  `getCounter`, `deleteRow` / `deleteColumn`, `sort`, `reset`, `show`, `saveAs`, `ResultsTable.open(path)`;
  `Calibration` (`pixelWidth`, `pixelHeight`, `pixelDepth`, `frameInterval`, `getUnit` / `setUnit`, `scaled`,
  `getX` / `getRawX` …); the `Measurements` bit flags.
- `ij.gui` – `Roi`, `OvalRoi`, `EllipseRoi`, `PolygonRoi`, `Line`, `PointRoi`, `TextRoi`, `ShapeRoi` (`or` / `and` /
  `xor` / `not`), with `getBounds` / `getFloatPolygon` / `getPolygon` / `contains` / `getLength` /
  `getStatistics` / `getMask` / `getContainedPoints` / `getContourCentroid` / `getFeretValues` / names, colours,
  stroke widths, positions and properties; `Overlay`; `GenericDialog` / `NonBlockingGenericDialog` /
  `GenericDialogPlus` (non-interactive: `getNext…` return the script parameters by field label or the defaults,
  `wasCanceled()` is false); `WaitForUserDialog`, `YesNoCancelDialog`, `MessageDialog` (log and continue);
  `Plot` (`addPoints` / `add` / `setColor` / `setLimits` / `addLegend` / `show` → the Plot section,
  `getImagePlus()` renders it); `Toolbar`.
- `ij.plugin` – `Duplicator`, `ZProjector`, `ChannelSplitter`, `ImageCalculator`, `HyperStackConverter`,
  `RGBStackMerge`, `SubstackMaker`, `Concatenator`, `ContrastEnhancer`, `Scaler`, `Resizer`, `Straightener`,
  `Colors`, `Macro_Runner`, `ImageInfo`; `ij.plugin.frame.RoiManager` (`getInstance`, `getCount`, `getRoi`,
  `getRoisAsArray`, `getName`, `rename`, `addRoi` / `add`, `select`, `deselect`, `setSelectedIndexes`,
  `getSelectedIndexes`, `runCommand` with every `roiManager()` command, `reset`, `multiMeasure`,
  `moveRoisToOverlay`, `translate`, `sort`); `ij.plugin.filter` – `ParticleAnalyzer` (options, measurements,
  private or live table, `getOutputImage`), `Analyzer` (`setMeasurements` / `getMeasurements` /
  `measure`), `GaussianBlur`, `RankFilters`, `BackgroundSubtracter`, `Convolver`, `ThresholdToSelection`,
  `MaximumFinder`, `EDM`, `Binary`, `Filters`, the `PlugInFilter` flags.
- `ij.io` – `FileSaver` (`saveAsTiff` / `saveAsPng` / `saveAsJpeg` / `saveAsGif` / `saveAsZip` / `saveAsRaw` /
  `saveAsText`), `Opener`, `OpenDialog` / `SaveDialog` / `DirectoryChooser` (answered from the parameters or
  `None`), `FileInfo`; `ij.text.TextWindow` (a text window); `ij.util.Tools`; `ij.Prefs`, `ij.Macro`,
  `ij.WindowManager` (`getImageTitles`, `getIDList`, `getImage`, `getCurrentImage`, `getImageCount`,
  `setTempCurrentImage`, `closeAllWindows`, `getUniqueName`).
- `java.lang` (`System.out.println`, `System.currentTimeMillis`, `Math`, `Integer`, `Double`, `Float`, `Long`,
  `Boolean`, `String.format` / `valueOf`, `StringBuilder`, `Thread.sleep`, `Runtime`, the exceptions),
  `java.io` (`File` – also a Python path –, `FileWriter`, `BufferedWriter`, `PrintWriter`, `FileReader`,
  `BufferedReader`), `java.util` (`ArrayList`, `HashMap`, `HashSet`, `Arrays`, `Collections`, `Random`, `Date`),
  `java.awt` (`Color`, `Rectangle`, `Point`, `Polygon`, `Dimension`, `Font`), `jarray.array` / `zeros`,
  `fiji.util.gui.GenericDialogPlus`, `loci.plugins.BF.openImagePlus` / `loci.plugins.in.ImporterOptions`
  (every series of a file through Lumen's readers), `loci.formats.ImageReader`.

Semantics worth knowing: an image a script creates (`IJ.createImage`, `IJ.openImage`, `imp.duplicate()`,
`ImagePlus(title, ip)`) is private until `imp.show()` – or until a command runs on it (`IJ.run(imp, …)`), which
adds it as a tab like ImageJ would open a window; `imp.hide()` / `close()` remove the tab. The selection is
script-wide (`imp.setRoi` sets what the next commands use, like the macro language's), it starts as the
viewport's selection and the script's final selection goes back to the viewport. `IJ.log` / `System.out`
print to the script output *and* the Log section. Pixel edits through processors are immediate (no undo step;
`lumen.update()` records one); commands run through `IJ.run` are undoable as usual.

**`#@` script parameters** (SciJava): the header is parsed and its variables assigned before the script runs –

```python
#@ String (label="Prefix", value="cells") prefix
#@ Integer (label="Threshold", min=0, max=255, value=128, style="slider") threshold
#@ File (label="Output folder", style="directory") folder
#@ Boolean show
#@ ImagePlus imp
#@ LogService log
#@output String result
```

In the Script panel a dialog asks for the promptable values (numbers, booleans, strings, choices, files,
folders; the last values are remembered); `POST /api/script/run {"code", "params": {"threshold": 90}}` supplies
them directly, `POST /api/script/params {"code"}` lists them. `ImagePlus` / `Dataset` is the current image,
`File` a `java.io.File` (also usable as a path), `LogService` / `UIService` / `StatusService` /
`DatasetIOService` / `ImageJ` small stand-ins (`log.info`, `ui.show(imp)`, `ij.io().open(path)`); `OpService`
and the other SciJava services raise a clear error when used. The same `params` answer `GenericDialog`
fields (by label), `IJ.getNumber` / `getString` (by prompt) and `OpenDialog` / `DirectoryChooser` (`file` /
`directory` / the title); a dialog that gets no value returns its default or `None` instead of waiting.

### JavaScript (ImageJ API)

```javascript
importClass(Packages.ij.IJ);
importClass(Packages.ij.plugin.frame.RoiManager);   // optional: ij.*, ij.gui.*, ij.process.*, ij.measure.*,
var imp = IJ.getImage();                              // ij.plugin.*, ij.io.*, java.lang.*, java.awt.*, java.util.* are pre-imported
IJ.run(imp, "Gaussian Blur...", "sigma=2");
var stats = imp.getStatistics();
IJ.log(imp.getTitle() + " mean " + stats.mean);
var rt = ResultsTable.getResultsTable();
for (var i = 0; i < rt.size(); i++) IJ.log(rt.getValue("Area", i));
```

Every ImageJ call is synchronous (one request to `/api/ijshim/call` each; the `lumen.*` API stays asynchronous),
objects are proxies over server-side handles, bean fields (`stats.mean`, `rect.width`, `cal.pixelWidth`) and
class constants (`Measurements.AREA`, `ImagePlus.GRAY8`) work, `new GenericDialog("…")` constructs. The covered
classes are those of the Jython list (the names are resolved through the same shim). Not available: Java
callbacks / listeners, `java.lang.Class` reflection, threads, Swing; `IJ.getNumber` & co. return their
defaults (JavaScript runs have no parameter prompt).

### JVM languages through the Java bridge (Groovy, BeanShell, Rhino, JRuby, Clojure)

Fiji's Groovy, BeanShell and Rhino-JavaScript scripts (and JRuby / Clojure) run with the **real ImageJ 1.x**
(`ij.jar`) in ImageJ's macro *batch mode* through `lumen/core/ijshim/jvm_bridge.py`, which has two modes
(`LUMEN_JVM_MODE=auto|subprocess|inprocess`, default `auto`):

| Mode | How | Needs | Picked by `auto` |
|---|---|---|---|
| **subprocess** (`lumen/core/ijshim/jvm_subprocess.py`) | one `java` process per script over a **Fiji installation's jars** (read-only) – or `ij.jar` + the language jar on `LUMEN_JVM_CLASSPATH` | a JDK (`JAVA_HOME`, `java` on `PATH`, Homebrew `openjdk`, Temurin / Zulu under `/Library/Java/JavaVirtualMachines`, `/usr/lib/jvm`, `C:\Program Files\Eclipse Adoptium`, …) and Fiji (`/Applications/Fiji`, `~/Fiji.app`, `/opt/fiji`, `C:\Fiji.app`, … or `LUMEN_FIJI_JARS`); **no Python package** | on macOS (an in-process JVM cannot run ImageJ commands there: AWT wants the process's main thread) and wherever `jpype` is not installed |
| **inprocess** | a JVM started inside Lumen's process (scyjava + JPype), headless, jars from Maven or `LUMEN_IJ_JAR` / `LUMEN_JVM_CLASSPATH` | `pip install "lumen-imaging[jvm]"` + a JDK | Linux / Windows with the extra installed |

Nothing Java-related is imported until the first such script runs; without a JDK / jars the Script panel,
`POST /api/script/run` and `script.run_jvm_script` answer with the install steps instead of failing, and
`Plugins ▸ Scripting ▸ JVM Script Bridge Status` (`script.jvm_bridge_status`, the panel's **Bridge status** button)
shows both modes, the JDK and the jars that were found.

**Subprocess mode – what happens per run** (about one second of overhead; verified on macOS with Fiji's
`ij-1.54p.jar`, `groovy-4.0.23`, `bsh-2.1.1`, `rhino-1.7.14`, `jruby-core-9.1.17` + `jruby-stdlib-9.4.12` and OpenJDK 21):

1. The active image goes to a temp ImageJ TIFF (calibration, hyperstack dimensions, display range), the viewport
   selection to a `.roi`, the session's ROI Manager entries to a `RoiSet.zip` and its Results rows to a CSV.
2. A generated runner sets `ij.macro.Interpreter.batchMode = true` (what `-batch` does – `imp.show()` registers
   images without windows), installs a *hidden* `RoiManager` as the instance (so `RoiManager.getRoiManager()`
   never opens a window), seeds the Results table / ROI Manager, opens the image (`IJ.openImage` + `show()`, the
   selection restored), binds `imp`, `IJ`, `WindowManager` and the `#@` parameters as variables, and runs your
   script file (`GroovyShell` / BeanShell `source()` / Rhino `load()` with ES6 / Ruby `eval` with the runner's binding /
   `load-file`). ImageJ's prefs are isolated (`-Duser.home`, `-Dij.dir`, an empty `-Dplugins.dir`: default settings,
   Fiji's plugins folder is not scanned – install plugins from an update site instead, see below); Fiji's whole `jars` folder is on the classpath, so `loci.formats` works too.
3. A language-independent teardown **macro** saves every open image as TIFF, the bound image's selection, the Results
   table (9 decimals), the ROI Manager and the Log window; the runner then exits (`System.exit(0)`).
4. Lumen reads everything back: the bound image in place with an **undo** entry (`Groovy script` …) when its pixels,
   calibration or title changed, other `show()`-ed images as new images, the final selection, Results rows and ROI
   Manager entries when they changed, and `println` / `print` / `puts` / `IJ.log` lines into the output and the Log
   section (ImageJ without a Log window prints `IJ.log` to stdout and echoes the Results table when `Measure` runs).

| Variable | Meaning |
|---|---|
| `LUMEN_JVM_MODE` | `auto` (default), `subprocess` or `inprocess` |
| `LUMEN_FIJI_JARS` | a Fiji `jars` folder (or the Fiji folder) when it is not in a standard place; an invalid value means *no* Fiji |
| `LUMEN_JVM_CLASSPATH` | extra jars / folders / globs (`os.pathsep`-separated) – both modes; subprocess: an ImageJ-only setup is `ij.jar` + `groovy-4.x.jar` here |
| `LUMEN_IJ_JAR` | a local `ij.jar` (both modes; in-process: instead of the Maven artifact) |
| `LUMEN_JVM_HEAP` | the heap (`-Xmx`), e.g. `4g`; subprocess default: 1 GiB or 3 × the image size + 512 MiB (≤ 8 GiB) |
| `LUMEN_JVM_TIMEOUT` | subprocess: seconds a script may run (600) |
| `LUMEN_JVM_KEEP_TEMP` | subprocess: `1` keeps the temp folder (runner, inputs, outputs) for debugging |
| `LUMEN_JVM_LANGUAGES` | in-process: `jruby`, `clojure` or `all` – enable the opt-in languages (before the JVM starts) |
| `LUMEN_JVM_EXTRA_ENDPOINTS` | in-process: more Maven coordinates (comma-separated) |
| `LUMEN_JVM_HEADLESS` | in-process: `0` starts a non-headless AWT (Linux / Windows) – needed for the ROI Manager there |

**What a script sees, and what comes back** (both modes)

| In | Out |
|---|---|
| the active image as `imp` – an `ij.ImagePlus` hyperstack (8 / 16 / 32-bit or RGB, calibration, position, display range), also `IJ.getImage()` / `WindowManager.getCurrentImage()` | pixel, calibration and title changes of `imp` go back *in place*, with an **undo** entry (`Groovy script` …) |
| the viewport selection as `imp.getRoi()` | the script's final `imp.getRoi()` becomes the viewport selection (none → cleared) |
| `#@` parameters (prompted for in the panel; `#@ ImagePlus`, `#@ ResultsTable`, `#@ RoiManager`, `#@ File` are supplied, SciJava services are `null`) | – |
| the Results table (`ResultsTable.getResultsTable()`), seeded from the session's | copied back when the script changed it (`rt.reset()` clears it) |
| the ROI Manager, seeded from the session's (subprocess: always, hidden; in-process: only on a non-headless JVM, `LUMEN_JVM_HEADLESS=0`) | copied back when changed |
| – | images the script `show()`s are added as new images; `println` / `print` / `puts` / `IJ.log` lines are the output and go to the Log section |

Groovy and Rhino scripts also get `IJ` and `WindowManager` bound (no import needed); BeanShell scripts start with
`import ij.*; import ij.gui.*; …` and Rhino with `importPackage(Packages.ij)…`, like ImageJ's own evaluators.

```groovy
#@ Double (label="Sigma", value=2.0) sigma
import ij.IJ
IJ.run(imp, "Gaussian Blur...", "sigma=" + sigma)              // bit-exact with Lumen's filters.gaussian
IJ.setAutoThreshold(imp, "Otsu dark")
IJ.run(imp, "Analyze Particles...", "size=30-Infinity display")   // → the Results table
new ij.plugin.Duplicator().run(imp).show()                        // → a new image
IJ.log(imp.getTitle() + ": " + ij.measure.ResultsTable.getResultsTable().size() + " particles")
```

Limits: ImageJ windows, dialogs and `GenericDialog` prompts do not exist (use `#@` parameters; a script that
creates `new RoiManager()` itself does open a window); `System.exit` / `IJ.run("Quit")` inside a script ends the
run before the results are collected; `IJ.runMacro` inside a script resets ImageJ's batch-mode image list (ImageJ
behaviour); one JVM script runs at a time. Subprocess mode: every run pays the JVM start (~1 s) and writes the image
to a temp TIFF; JRuby runs with gems disabled (Fiji's `jruby-core` 9.1 and `jruby-stdlib` 9.4 do not match
otherwise); Clojure needs `spec.alpha` + `core.specs.alpha` on `LUMEN_JVM_CLASSPATH` (verified with both: bit-identical Invert and Median, parameters and Results).

**Fiji update sites.** *Plugins ▸ Utilities ▸ Fiji Update Site…* lists the files of an update site (`site`: a known name such as `ImageJ`, `Fiji`, `BIG-EPFL`, `IJPB-plugins (MorphoLibJ)`, `CSBDeep`, or any URL / local folder with a `db.xml.gz`); *Install from Fiji Update Site…* (`files="plugins/Foo_.jar, …"`) downloads them and their missing dependencies into `~/.lumen/jars`, which is on the bridge's classpath. Each command a jar declares in `plugins.config` becomes a Lumen command under *Plugins ▸ Fiji ▸ …* with one parameter, `options` (the plugin dialog's macro options), and the ImageJ macro layer reaches it by name: `run("Kuwahara Filter", "sampling=5");` runs Fiji's plugin when Lumen has no command of that name (Lumen's own ports win when it has one). *Run Fiji Plugin Class…* runs any ImageJ 1.x plugin class on the classpath. Plugin jars are Java code and run with your permissions. Verified: the index reader against the real IJPB-plugins (MorphoLibJ, 1 jar), BIG-EPFL (13 jars) and ImageJ (117 files) sites on 13 Nov 2026; installing and running against a local update site built from Fiji's own jars (`tests/test_fiji_plugins.py`); and a real install: MorphoLibJ 1.6.5 from the IJPB-plugins site (1.1 MB, size-checked; its ij.jar / jama.jar dependencies recognised as already present in Fiji), whose 91 commands appear under Plugins ▸ Fiji. Running its Morphological Filters next to Lumen's own MorphoLibJ port found three discrepancies in the port – the disk element (ImageJ's r²+1 rule), the off-centre octagon and diamond, and the Laplacian – now fixed and covered by a test against the real plugin.

**Compiled plugins.** A Fiji plugin jar runs through the same subprocess when it is on the classpath. With
`LUMEN_JVM_CLASSPATH=/Applications/Fiji/plugins/Auto_Threshold-1.18.0.jar` this Groovy script thresholds the bound
image with the real Fiji plugin and the result comes back into Lumen like any other bridge run (undo entry included):

```groovy
import ij.Macro
// GenericDialog-based plugins read macro options only from a thread whose name starts with "Run$_"
def t = new Thread({ Macro.setOptions("method=Otsu white"); IJ.runPlugIn(imp, "fiji.threshold.Auto_Threshold", "") } as Runnable, "Run\$_lumen")
t.start(); t.join()
```

`IJ.run(imp, "Auto Threshold", …)` by menu name does not work in the bridge (ImageJ's command table is not built from
Fiji's plugins folder); call the class. Plugins that need a display (Swing windows, interactive tools) stay out of
reach – the JVM is headless.
In-process mode: one JVM per process – whichever bridge starts it first (this one or the Bio-Formats reader) fixes
the classpath; the ROI Manager needs `LUMEN_JVM_HEADLESS=0`; it cannot run ImageJ commands on macOS.

---

## Writing `@op` plugins

```python
from lumen.core.registry import op, Param, OpResult

@op(id, name, category, *, kind="plane", description="", params=None, preview=True, roi_aware=True,
    requires_image=True, requires_roi=None, shortcut=None, menu=None, keywords=None, scope_param=True,
    supports_rgb=True, icon="", undo_label=None)
```

| Argument | Meaning |
|---|---|
| `id` | unique, `"<group>.<snake_name>"` (`"plugins.my_blur"`); a duplicate id raises at import |
| `name` | what the palette and menus show |
| `category` | the menu path (`"Plugins/My Tools"`, `"Process/Filters"`); `menu=` overrides the menu while `category` stays the palette's category |
| `kind` | `"plane"`, `"image"` or `"analysis"` (below) |
| `description` | the palette's subtitle and `lumen.help()` text; defaults to the function's docstring |
| `params` | list of `Param` — these become the sheet, the recorder arguments and the `**params` of your function |
| `preview` | live preview while the sheet is open (plane commands always can; image commands only if they are side-effect free; analysis never) |
| `roi_aware` | plane commands: restrict the result to the area selection; others: whether the selection is passed as `ctx.roi` |
| `requires_image` | `False` for generators / importers that work without an open image |
| `requires_roi` | `None`, `"area"`, `"line"` or `"any"` — the command refuses to run without such a selection |
| `shortcut` | single key (`"M"`, `"Shift+T"`) shown in the palette |
| `keywords` | extra search words for the palette |
| `scope_param` | plane commands: offer the *Apply to* selector (set `False` if the command must see the whole stack) |
| `supports_rgb` | `False` makes the command refuse RGB images with a clear message |
| `undo_label` | label in the History list (default: the name) |

### `Param` types

`Param(name, type="float", default=None, label="", min=None, max=None, step=None, choices=None, help="", advanced=False, unit="")`

| type | value your function receives | sheet widget |
|---|---|---|
| `int` | `int` (rounded, clamped to `min`/`max`) | slider / number |
| `float` | `float` (clamped) | slider / number (`step`) |
| `bool` | `bool` (`"1"`, `"true"`, `"yes"`, `"on"` are true) | switch |
| `choice` | one of `choices` (matched case-insensitively) | drop-down |
| `multichoice` | list of strings (from a list or a comma-separated string) | check list |
| `string` | `str` | text field |
| `text` | `str` (multi-line) | text area |
| `image` | the *name or id* of an open image as a string (`None` if none chosen) — resolve it with `ctx.get_image(value)` | drop-down of open images |
| `color` | `"#rrggbb"` string | colour picker |

`label` defaults to the capitalised name; `advanced=True` hides the field behind *Advanced*; `help` is the
tooltip; `unit` is shown after the value.

### Kinds of command

**`plane`** — `fn(plane2d, ctx, **params) -> 2-D array` of the same shape. The registry calls it once per
plane selected by the scope, masks the result with the area selection, casts it back to the image's
dtype (clipped and rounded for integers; a boolean result becomes 0/255) and installs one undo step.
When the function returns `float32` for *every* plane of the stack the image becomes 32-bit; `uint8`
/ `uint16` results change the type likewise. Use `ctx.preview` to skip expensive extras while previewing.

**`image`** — `fn(ctx, **params) -> OpResult | None`. Full control: read `ctx.img.data` (5-D), return
`OpResult(data=new_5d_array, rgb=…)` to replace the pixels (any shape / dtype), `new_images=[…]` to open
images, and anything else from the table below. One undo step is recorded if the command changed the
image (data, name, calibration, display or overlay). A failed command is rolled back.

**`analysis`** — `fn(ctx, **params) -> OpResult`. Never modifies pixels (no undo snapshot, no preview);
returns results rows, ROIs, a plot, a histogram, a table, a selection, a threshold or just a message.

### Plugin tools (`tool=True`)

A command declared with `tool=True` appears on the tool rail (after the built-in tools) and is run every time
the user clicks or drags on the image while it is selected. It receives the interaction as parameters:
`x`, `y` (image coordinates), `x2`, `y2` (drag end), `button` (0 left, 2 right), `shift`, `alt`, `ctrl`
and `event` (`click`, `drag-start` or `drag-end`); its ordinary `Param`s become the tool-options bar.

```python
@op("plugins.click_marker", "Click Marker", "Plugins", kind="analysis", tool=True, tool_icon="✚",
    tool_cursor="crosshair", description="Mark every click with a circle",
    params=[Param("radius", "float", 6, min=1, max=100)])
def click_marker(ctx, x=0.0, y=0.0, event="click", radius=6.0, **_):
    if event != "click":
        return OpResult()
    d = ctx.img.display
    return OpResult(overlay=[{"type": "oval", "x": x - radius, "y": y - radius, "w": 2 * radius, "h": 2 * radius,
                              "position": {"t": d.t, "z": d.z, "c": None}}], message=f"marked ({x:.0f}, {y:.0f})")
```

`plugins.new_template` with kind "Plugin Tool" writes a complete example into the plugins folder; the built-in
`tools.spot_counter` and `tools.measure_distance` tools are further references. Interactions are recorded
(`lumen.run("plugins.click_marker", x=…, y=…, event="click")`), so a tool session replays as a script.

### `OpContext`

| Member | Meaning |
|---|---|
| `ctx.session` | the `Session` (results, ROI manager, other images, settings) |
| `ctx.image` / `ctx.img` | the target image (`image` may be `None` when `requires_image=False`; `img` raises a clear error) |
| `ctx.roi` | the normalised selection dict or `None` |
| `ctx.mask` | boolean `(H, W)` mask of the *area* selection, or `None` (cached) |
| `ctx.scope` | `"plane"` / `"slice"` / `"channel"` / `"stack"` (plane commands; image/analysis always `"stack"`) |
| `ctx.planes()` | the `(t, z, c)` indices the scope selects |
| `ctx.preview` | `True` while the sheet previews |
| `ctx.get_image(name_or_id)` | resolve an `image`-type parameter |
| `ctx.progress(fraction, message="")`, `ctx.check_cancel()` | progress bar and cancellation (below) |
| `ctx.job`, `ctx.nested` | the job record; `nested` is true when run from inside another command (batch, script) |

### `OpResult`

| Field | Effect |
|---|---|
| `data` (+ `rgb`) | replace the image's pixels with this `(T, Z, C, Y, X)` array (image commands) |
| `new_images` | `LumenImage`s to open (the last one becomes active unless `select_image` says otherwise) |
| `results`, `results_replace` | rows appended to the Results table (replace it first if `results_replace`) |
| `table` | `{"name": "My table", "columns": [...], "rows": [[...]], "summary": {...}}` → a Table section in the inspector and `session.tables[name]` |
| `plot` | `{"title", "x": [...], "series": [{"name", "values", "lut"?}], "x_label", "y_label"}` → the Plot section |
| `histogram` | `{"counts", "edges", "stats", "title"}` → the Histogram section |
| `roi`, `clear_roi` | set / clear the user's selection |
| `rois_add` | ROI dicts for the manager (each may carry `name`, `color`, `position`, `image_id`) |
| `overlay` | the image's new overlay list |
| `message` | toast + the `message` field of the payload |
| `log` | a Log line |
| `rename`, `calibration` (dict patch), `display` (patch), `threshold` (`{"lower", "upper", …}`), `close_image`, `select_image` (id) | image state |

Raise `ValueError("…")` for user errors; the message is shown as a toast and returned as `{"error": "…"}`
(HTTP 400) by the API.

### Previews, ROI awareness, stack scope, undo

- Plane commands get previews for free: the sheet calls the function on the current plane (all channels
  of the slice when the scope is slice/stack) through `POST /api/ops/{id}/preview` and shows the result
  without touching the image.
- An image command with `preview=True` is *called* during previews with the image data restored
  afterwards; it must therefore compute a new array and **never write into `ctx.img.data` in place**
  (set `preview=False` for commands that have side effects, create images or take long).
- `roi_aware=True` plane commands only change pixels inside the selection; analysis commands read
  `ctx.mask` / `ctx.roi` themselves. `requires_roi` makes the selection mandatory.
- `scope_param=True` plane commands loop over `ctx.planes()`; the recorder writes `scope=` when it is not
  the whole stack.
- Undo is automatic for plane and image commands (memory-bounded history per image; the History section
  lists the labels). Analysis commands are not undoable; put pixel changes in an image command.
- Dtype: Lumen images are `uint8`, `uint16` or `float32`. Return other dtypes from an image command only
  through `LumenImage.to_canonical` / `cast_to` conventions (ints are clipped, bools become 0/255).

### Progress and cancellation

Every command run through `Session.run_op` (UI, REST, script) is a *job*: `GET /api/jobs` lists the
running ones and `POST /api/jobs/{id}/cancel` asks one to stop. Inside a command:

```python
for i, (t, z, c) in enumerate(ctx.planes()):
    ctx.check_cancel()                                # raises OperationCancelled (a ValueError) when asked to stop
    ...
    ctx.progress((i + 1) / n, f"Plane {i + 1}/{n}")  # 0–1 and a status message; throttled to 20 events/s
```

Both are no-ops during previews, direct calls and nested commands (a command run from a batch or a script
reports to the outer job), so they can always be called. Plane commands report per-plane progress
automatically. The client sees `{"type": "progress", "job": {...}}` events (below); a cancelled command is
rolled back like a failed one.

### A complete plugin file

This file was loaded through `LUMEN_PLUGINS` and all three commands were run (with and without a
selection, with a preview) before it was written down.

```python
# ~/.lumen/plugins/demo_plugin.py – three kinds of command
import numpy as np
from scipy import ndimage as ndi
from lumen.core.image import LumenImage
from lumen.core.registry import op, Param, OpResult

CAT = "Plugins/Demo"


# 1. plane command: fn(plane2d, ctx, **params) -> plane2d. Lumen loops over the planes selected by the
#    "apply to" scope, masks the ROI, keeps the dtype and gives it a live preview and undo for free.
@op("demo.local_contrast", "Demo: Local Contrast", CAT,
    description="Divide by a large Gaussian blur (flat-field style)",
    params=[Param("sigma", "float", 20.0, min=1, max=200, step=1, help="Blur radius in pixels"),
            Param("mode", "choice", "divide", choices=["divide", "subtract"]),
            Param("clip", "bool", True, label="Clip to the input range"),
            Param("note", "string", "", advanced=True)])
def local_contrast(plane, ctx, sigma=20.0, mode="divide", clip=True, note=""):
    p = plane.astype(np.float32)
    bg = ndi.gaussian_filter(p, sigma)
    out = p / np.maximum(bg, 1e-6) * p.mean() if mode == "divide" else p - bg + p.mean()
    if clip:
        out = np.clip(out, p.min(), p.max())
    return out                       # cast back to the image dtype by the registry


# 2. image command: fn(ctx, **params) -> OpResult. Full control over the 5-D array; may change
#    shape/dtype, create images, write results.
@op("demo.channel_ratio", "Demo: Channel Ratio", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="New 32-bit image = channel A / channel B",
    params=[Param("channel_a", "int", 1, min=1), Param("channel_b", "int", 2, min=1),
            Param("create_new", "bool", True)])
def channel_ratio(ctx, channel_a=1, channel_b=2, create_new=True):
    img = ctx.img                                   # raises if no image is open
    if max(channel_a, channel_b) > img.nc:
        raise ValueError(f"The image has only {img.nc} channel(s)")
    a = img.data[:, :, channel_a - 1:channel_a].astype(np.float32)
    b = img.data[:, :, channel_b - 1:channel_b].astype(np.float32)
    ratio = a / np.maximum(b, 1e-6)
    if create_new:
        out = LumenImage(ratio, name=f"Ratio C{channel_a}/C{channel_b} of {img.name}", axes="TZCYX",
                         calibration=img.calibration)
        return OpResult(new_images=[out], message="ratio image created")
    return OpResult(data=ratio, rgb=False)          # replaces the pixels in place (undoable)


# 3. analysis command: never modifies pixels; returns results / ROIs / a plot / a table.
@op("demo.bright_spots", "Demo: Bright Spots", CAT, kind="analysis", preview=False,
    description="Count pixels above a percentile inside the selection and add them as ROIs",
    params=[Param("percentile", "float", 99.0, min=0, max=100, step=0.5)])
def bright_spots(ctx, percentile=99.0):
    img = ctx.img
    plane = img.current_plane()
    mask = ctx.mask                                 # boolean (H, W) of the area selection, or None
    vals = plane[mask] if mask is not None else plane.ravel()
    thr = float(np.percentile(vals, percentile))
    spots = plane >= thr
    if mask is not None:
        spots &= mask
    lab, n = ndi.label(spots)
    rois = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y, x = sl
        rois.append({"type": "rect", "x": x.start, "y": y.start, "w": x.stop - x.start, "h": y.stop - y.start, "name": f"spot {i}"})
    row = {"Label": img.name, "Threshold": thr, "Spots": n, "Pixels": int(spots.sum())}
    hist, edges = np.histogram(vals, bins=32)
    return OpResult(results=[row], rois_add=rois, message=f"{n} spots ≥ {thr:.1f}",
                    plot={"title": "Intensity histogram", "x": edges[:-1].tolist(), "series": [{"name": "count", "values": hist.tolist()}],
                          "x_label": "intensity", "y_label": "pixels"},
                    table={"name": "Spots", "columns": ["Spot", "x", "y"], "rows": [[r["name"], r["x"], r["y"]] for r in rois]})
```

After a restart (or **Plugins ▸ Utilities ▸ Reload Plugins** and a page reload) the three commands are in
the palette under *Plugins/Demo*, in the Plugins menu, in the recorder and in the API:

```python
lumen.run("demo.local_contrast", sigma=30, mode="divide", scope="plane")   # inside lumen.roi if set
ratio = lumen.run("demo.channel_ratio", channel_a=2, channel_b=1)[0]       # the new image
plot = lumen.run("demo.bright_spots", percentile=99.5)                      # the plot; rows in lumen.results,
                                                                            # ROIs in lumen.rois, table in lumen.session.tables["Spots"]
```

### The plugins folder and startup scripts

- Plugin files are every `*.py` in `~/.lumen/plugins` and in the folders listed in the `LUMEN_PLUGINS`
  environment variable (separated by `os.pathsep`: `:` on macOS/Linux, `;` on Windows). They are imported
  in name order *after* Lumen's own command modules, when the server starts (`lumen.core.ops.load_user_plugins()`).
  An import error is printed to the terminal and skipped; the other plugins still load.
- **Plugins ▸ Utilities ▸ Reload Plugins** (`help.reload_plugins`) re-imports them and replaces their
  commands; reload the browser page to see new menu entries. **Help ▸ About Plugins** (`help.about_plugins`)
  and `help.compiler_info` list the folders, files and the commands each registered.
- There is no separate *startup macro*: a plugin file is ordinary Python executed at import, so it can
  configure the session the UI is about to use. This file was verified to take effect:

```python
# ~/.lumen/plugins/startup.py – configure the session at start-up
from lumen.core.session import SESSION
SESSION.measurements = ["area", "mean", "integrated_density", "centroid"]
SESSION.measure_options["decimals"] = 2
SESSION.add_log("startup.py ran")
```

- Files given on the command line are opened at start: `lumen image.tif other.nd2`.
- To ship a plugin to colleagues: one `.py` file, numpy / scipy / scikit-image / scikit-learn are always
  available; other imports must be installed into Lumen's Python environment.

---

## Batch commands

| Command | What it does |
|---|---|
| `batch.process_folder` (**Process Folder...**) | for every file matching `pattern` (a regex, e.g. `\.tif$`; `recursive`) in `input_folder`: open it, run `script` with `lumen.image` bound to it, save the image the script leaves active (if the script created it) or the opened file to `output_folder` (default `<input>/lumen_output`) in `output_format` (`tif`, `ome.tif`, `png`, `jpg`, `npy`, `none`), append the Results rows the script produced to `Results.csv` there (with a *File* column; `save_results`), close what was opened (`close_after`). `continue_on_error`, `max_files`. Returns the **Batch** table (File, Status, Time, Output, Message) and a summary |
| `batch.apply_recorded_macro` (**Apply Recorded Macro to Folder...**) | the same with the recorder's lines as the script (or the `lines` parameter, which the sheet pre-fills from `GET /api/recorder/preview` so you see and edit exactly what will run), minus the lines that open / save / close files, run batch commands, or name images of the recording session (`lumen.select` / `undo` / `redo` / `display` and the restore lines) |
| `batch.run_script_on_open_images` (**Run Script on All Open Images...**) | run `script` once per open image (each becomes `lumen.image` in turn); nothing is saved or closed |
| `batch.measure` (**Batch Measure...**) | open every matching file, optionally auto-threshold it (`threshold`, `dark_background`), run Measure with the current Set Measurements on the current plane or on `all_planes`, append the rows with a *File* column |

Inside a batch script `print()`'s last line lands in the table's *Message* column, errors are per file,
`lumen.run(...)` works as usual, the recorder is left untouched and the previously active image is
re-selected afterwards. See [WORKFLOWS.md ▸ Batch](WORKFLOWS.md#7-batch-process-a-folder-with-a-recorded-macro).

---

## Headless use (no server)

The engine is a plain Python package; nothing needs the browser or FastAPI. This is how the recipes in
WORKFLOWS.md were verified and how you run Lumen on a cluster node:

```python
# example: headless
from lumen.core.session import Session
from lumen.core import ops                      # registers every command (and your plugins)
from lumen.core.scripting import LumenAPI, run_script

s = Session()                                   # an independent session (the server uses lumen.core.session.SESSION)
out = s.run_op("file.sample", {"name": "Blobs"})
img = s.active
s.run_op("filters.gaussian", {"sigma": 2}, scope="plane")
s.run_op("threshold.apply", {"method": "Otsu"})
out = s.run_op("analyze.particles", {"min_size": 30, "add_to_manager": False})
print(out["message"], len(s.results.rows), s.results.columns)
print(s.results.to_csv().splitlines()[0])

lumen = LumenAPI(s)                             # the console's `lumen` object for this session
print(len(lumen.run("analyze.measure", roi=lumen.rect(0, 0, 50, 50))))
res = run_script(s, 'print(lumen.image.name, lumen.image.shape)')   # exactly what the console does
print(res)                                      # {'stdout': 'blobs.tif (1, 1, 1, 256, 256)\n', 'error': None}
```

`python -m lumen` / `lumen` starts the server (`--port`, `--host`, `--no-browser`, `--window`, files to open).

---

## REST API

Base URL `http://127.0.0.1:8765` (the port is printed at start; Lumen only listens on localhost unless
started with `--host`). JSON in, JSON out; errors are `{"error": "message"}` with status 400 (404 for
unknown jobs). The interactive schema is at `/docs` (FastAPI).

### Endpoints

| Method and path | Body / query | Returns |
|---|---|---|
| `GET /api/meta` | | `version`, `ops` (every command spec: id, name, category, kind, params with types/defaults/choices, preview, roi_aware, requires_roi, shortcut, menu, keywords, scope_param, supports_rgb), `luts`, `threshold_methods`, `measurements`, `home`, `cwd` |
| `GET /api/ops` | | the `ops` list alone |
| `POST /api/ops/{id}/run` | `{"params": {...}, "image_id": "…" \| null, "roi": {...} \| null, "scope": "stack"}` | the command payload (`op`, `new_images`, `results_added`, `rois_added`, `message`, `log`, `table`, `plot`, `roi`, `threshold`, `elapsed_ms` …) |
| `POST /api/ops/{id}/preview?stride=1` | same body | the previewed planes as raw bytes (see [Pixel data](#pixel-data)) |
| `GET /api/state` | | `version`, `images` (summaries), `active`, `results {columns, rows}`, `rois`, `history`, `macro` (last 200 lines), `log`, `measurements`, `measure_options`, `thresholds`, `last_plot`, `tables`, `recent`, `lut_names`, `memory_bytes` |
| `GET /api/images`, `GET /api/images/{id}` | | image summaries (id, name, shape, dtype, calibration, display, overlay, metadata …) |
| `POST /api/images/open` | `{"path": "/data/cells.tif", "format": null}` | `{"id", "image"}` |
| `POST /api/images/upload` | multipart `file` | `{"id", "image"}` (for drag & drop / remote clients) |
| `POST /api/images/{id}/save` | `{"path": "...", "format": null}` | `{"path"}` |
| `GET /api/images/{id}/download?format=tif` | | the file (`tif`, `ome.tif`, `png`, `jpg`, `npy`, …) |
| `DELETE /api/images/{id}` · `POST /api/images/{id}/select` · `POST …/undo` · `POST …/redo` · `POST …/reset_display` | | |
| `POST /api/images/{id}/display` | `{"t", "z", "c", "mode", "channels": [{"name","lut","min","max","visible","inverted","gamma"}]}` (any subset) | the display dict |
| `GET /api/images/{id}/planes?t=0&z=0&stride=1` | | all channels of one plane, raw bytes |
| `GET /api/images/{id}/plane?t=0&z=0&c=0&stride=1` | | one channel plane, raw bytes |
| `GET /api/images/{id}/value?x=&y=&t=&z=` | | `{"values": [per channel]}` |
| `GET /api/images/{id}/rgb.png?t=&z=` | | the rendered composite as PNG |
| `GET /api/images/{id}/volume?t=0&max_dim=256&channels=all` | | uint8 volume per channel (headers `X-Shape`, `X-Channels`, `X-Spacing`, `X-Downsample`) for the 3-D view |
| `GET /api/images/{id}/ortho?x=&y=&t=` | | XZ / YZ sections |
| `GET /api/images/{id}/mesh?c=&t=&level=&step=&max_dim=&smooth=&min_faces=` · `GET …/mesh.obj` / `.stl` / `.ply` | | iso-surface (binary buffer with `X-Vertices`, `X-Faces`, `X-Level`, `X-Bounds`, `X-Extent` …) or a mesh file |
| `GET /api/rois` · `POST /api/rois {"roi", "image_id", "position"}` · `PATCH /api/rois/{id} {"patch"}` · `POST /api/rois/delete {"ids"}` · `DELETE /api/rois` · `POST /api/rois/combine {"ids", "mode"}` · `POST /api/rois/measure {"ids"}` · `GET /api/rois/export` · `POST /api/rois/import` | | ROI Manager |
| `GET /api/results` · `DELETE /api/results` · `GET /api/results.csv` | | Results table |
| `DELETE /api/threshold/{image_id}` | | clear the threshold overlay |
| `POST /api/script/run` | `{"code": "..."}` | `{"stdout", "error"}` |
| `GET /api/macro` · `DELETE /api/macro` | | the recorder as text (marker line first) / clear it → `{"ok", "cleared"}` |
| `GET /api/recorder` · `DELETE /api/recorder` · `GET /api/recorder/preview` · `GET /api/recorder/ijm` | header `X-Lumen-Client: <id>` selects the client's recorder | `{client, started, started_iso, marker, active_image, lines, count}` / clear / the lines a folder batch would run (`{lines, skipped, script}`) / the recording as an ImageJ macro (`{script, lines, converted, count, skipped: [{line, reason}], notes}`) |
| `POST /api/session/restore_clear` | `{"kind": "rois" \| "results" \| null}` | bring back the last cleared ROIs or Results → `{kind, restored}`; `DELETE /api/rois`, `POST /api/rois/delete` and `DELETE /api/results` answer with the counts and `restorable` |
| `GET /api/jobs` · `POST /api/jobs/{id}/cancel` | | running commands / cancel one |
| `GET /api/settings` · `PUT` / `PATCH /api/settings {…}` | | preferences and measurement settings (`measurements`, `measure_options`, `recent_clear`) |
| `GET /api/docs` · `GET /api/docs/{README\|COMMANDS\|FIJI_PARITY\|RESEARCH\|DESIGN}` | | the bundled documentation as markdown |
| `GET /api/workflows` · `GET /api/workflows/{id}` · `GET` / `POST /api/workflows/{id}/script` | `{"params": {step_id: {name: value}}, "skip": [step_id], "include_optional": true}` | the guided workflows (`count_nuclei`, `intensity_per_cell`, `track_cells`, `pixel_classifier`, `colocalization`, `stitch_tiles`, `zstack_figure`, `batch_folder`, `register_timelapse`: steps with op, params, preview and requirements) and a `lumen.run` script that replays one |
| `GET /api/luts` · `GET /api/fs?path=` | | LUT tables (256 × 3) / the server-side file browser |

### Pixel data

Planes travel as raw little-endian arrays, not JSON: the body of `/planes`, `/plane`, `/preview`,
`/volume` and `/ortho` is `array.tobytes()` and the headers say how to read it — `X-Dtype` (`uint8`,
`uint16`, `float32`), `X-Shape` (comma-separated, `C,H,W` for `/planes`), `X-Stride` (the subsampling
applied when `stride > 1`). In Python: `np.frombuffer(body, dtype).reshape(shape)`.

### WebSocket events and jobs

`ws://127.0.0.1:8765/api/events` pushes one JSON object per change; poll `/api/state` (or the specific
endpoint) after an event:

| Event | When |
|---|---|
| `{"type": "images", "version": n}` | images opened / closed / changed (pixels, display, overlay, name) |
| `{"type": "active", "version": n}` | another image was selected |
| `{"type": "results", …}`, `{"type": "rois", …}`, `{"type": "recent", …}`, `{"type": "state", …}` | the Results table, ROI Manager, recent files or anything else changed |
| `{"type": "progress", "job": {"id", "op", "name", "started", "fraction", "message", "cancelled", "elapsed"}}` | a command started or reported progress; the final event adds `"done": true` and `"status": "ok" \| "error" \| "cancelled"` |

`version` increases monotonically (also in `/api/state`), so a client can skip stale refreshes.

### A client in 30 lines

```python
# example-rest: client (needs a running Lumen; verified against a throwaway server)
import json, urllib.request
import numpy as np

BASE = "http://127.0.0.1:8765"

def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def plane(image_id, t=0, z=0, stride=1):
    with urllib.request.urlopen(f"{BASE}/api/images/{image_id}/planes?t={t}&z={z}&stride={stride}") as r:
        shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
        return np.frombuffer(r.read(), dtype=r.headers["X-Dtype"]).reshape(shape)   # (C, H, W)

meta = call("GET", "/api/meta")
print(meta["version"], len(meta["ops"]), "commands")
out = call("POST", "/api/ops/file.sample/run", {"params": {"name": "Blobs"}})
iid = out["new_images"][0]
call("POST", "/api/ops/filters.gaussian/run", {"params": {"sigma": 2}, "image_id": iid, "scope": "plane"})
rows = call("POST", "/api/ops/analyze.measure/run", {"image_id": iid, "roi": {"type": "rect", "x": 10, "y": 10, "w": 50, "h": 50}})
print(rows["results_added"], call("GET", "/api/state")["results"]["rows"][-1])
arr = plane(iid)
print(arr.dtype, arr.shape, float(arr.mean()))
print(call("POST", "/api/script/run", {"code": "print(lumen.image.name)"}))
call("DELETE", f"/api/images/{iid}")
```

---

## Examples

**Process every open image** (the console equivalent of *Run Script on All Open Images*):

```python
# example: loop_images
for name in ("Blobs", "Coins", "Moon"):
    lumen.run("file.sample", name=name)
for img in list(lumen.images):
    lumen.select(img.name)
    lumen.run("process.subtract_background", radius=30)
    lumen.run("threshold.apply", method="Otsu")
    rows = lumen.run("analyze.particles", min_size=20, add_to_manager=False, summarize=False)
    lumen.log(f"{img.name}: {len(rows)} objects")
print([l["msg"] for l in lumen.session.log[-3:]])
```

**Per-ROI measurements on every channel of a hyperstack:**

```python
# example: roi_channels
lumen.run("file.sample", name="Synthetic nuclei (3 channels)")
img = lumen.image
img.display.c = 0
lumen.run("threshold.apply", method="Otsu")                     # mask of channel 1 as a new image
lumen.run("analyze.particles", min_size=20, add_to_manager=True, display_results=False)
lumen.select(img.name)
lumen.run("analyze.set_measurements", measurements=["area", "mean"])
table = {}
for c in range(img.nc):
    img.display.c = c
    for r in lumen.rois:
        row = lumen.run("analyze.measure", roi=r)[0]
        table.setdefault(r["name"], {})[img.display.channels[c].name] = round(row["Mean"], 1)
first = next(iter(table))
print(len(table), "nuclei;", first, table[first])
```

**Add a numpy result as a new calibrated image and save it:**

```python
# example: save_result
import os
import numpy as np
out = os.path.expanduser("~/lumen-demo"); os.makedirs(out, exist_ok=True)
lumen.run("file.sample", name="Cells (3-D, 2 channels)")
vol = lumen.image
nuclei = vol.data[0, :, 1]                                      # (Z, Y, X) of channel 2
mip = nuclei.max(axis=0)
lumen.new_image(mip, name="nuclei_mip", like=vol)              # keeps 0.26 µm/px
print(lumen.image.calibration.unit, lumen.image.dtype)
print(lumen.save(f"{out}/nuclei_mip.tif"))
```

**A one-line command from a lambda-style plugin** (anything callable works):

```python
# ~/.lumen/plugins/quick.py
from lumen.core.registry import op, Param

@op("quick.gamma_sqrt", "Square-root gamma", "Plugins/Quick", description="sqrt(x / max) * max")
def gamma_sqrt(plane, ctx):
    import numpy as np
    m = float(plane.max()) or 1.0
    return np.sqrt(plane.astype("float32") / m) * m
```

More: [WORKFLOWS.md](WORKFLOWS.md) has fifteen verified end-to-end scripts (nuclei counting, per-cell
intensities, colocalization, tracking + TrackMate export, z-stacks and meshes, stitching, batches, ROI
interchange, calibration, figures, ImageJ macros, pixel classification, skeletons, registration, FFT).
