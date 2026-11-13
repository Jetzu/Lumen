# Scientific image-analysis GUIs: a sourced review for Lumen

*Compiled November 2026 from primary documentation, release notes, papers, GitHub issues and
Image.sc forum threads. Every factual claim links to its source; the full list is at the end.
Quotations are verbatim from the linked page. Where a claim could not be verified online it is
marked **(not verified)**.*

Lumen re-implements the Fiji/ImageJ toolbox behind a single-window, keyboard-first UI
([DESIGN.md](DESIGN.md)). This document is the evidence base for that design: for each tool
we describe the UI model, what works, what users complain about, and the lesson for Lumen.
Section 13 is a comparison matrix; section 14 traces each Lumen design principle to sources.

---

## 1. Fiji / ImageJ 1.x

**UI model.** ImageJ 1.x is a toolbar window plus one floating window per image, dialog,
table, plot and histogram. Commands live in eight menus (File, Edit, Image, Process, Analyze,
Plugins, Window, Help) documented in the [ImageJ menu reference](https://imagej.net/ij/docs/menus/)
and the [User Guide](https://imagej.net/ij/docs/guide/146-26.html). The
[Window menu](https://imagej.net/ij/docs/menus/window.html) exists only to manage window
soup: *Show All*, *Put Behind* (Tab cycles images), *Cascade*, *Tile*. Counting the
`xxxNN=` menu entries in ImageJ 1.54's
[`IJ_Props.txt`](https://raw.githubusercontent.com/imagej/ImageJ/master/IJ_Props.txt)
gives 348 built-in commands (own count, Nov 2026); Fiji then adds plugins on top -
the ImageJ wiki describes "nearly 1000 additional plugins"
([Plugin tiers](https://imagej.net/imagej-wiki-static/Plugin)). The "more than 1,100
commands" figure quoted in [DESIGN.md](DESIGN.md) comes from the EasyFiji article
([JoVE 2026, doi:10.3791/69441](https://doi.org/10.3791/69441)), which is paywalled; we could
not re-verify the exact number **(not verified)**.

Fiji is "a distribution of the popular open-source software ImageJ focused on
biological-image analysis" that "uses modern software engineering practices to combine
powerful software libraries with a broad range of scripting languages" and ships "an
integrated update system" ([Schindelin et al. 2012, Nature Methods](https://europepmc.org/abstract/MED/22743772)).
The name expands to "Fiji Is Just ImageJ" ([Wikipedia](https://en.wikipedia.org/wiki/Fiji_(software)));
imagej.net calls it a "batteries-included" distribution of ImageJ2 that also includes the original
ImageJ ([imagej.net/software/fiji](https://imagej.net/software/fiji/)). Fiji-specific commands are
added under existing menus by each plugin's `plugins.config` - for example
`Plugins>Tracking, "TrackMate"` ([TrackMate plugins.config](https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/resources/plugins.config)),
`Plugins>Segmentation, "Trainable Weka Segmentation"` ([Trainable_Segmentation](https://raw.githubusercontent.com/fiji/Trainable_Segmentation/master/src/main/resources/plugins.config)),
`Plugins>Stitching, "Grid/Collection stitching"` ([Stitching](https://raw.githubusercontent.com/fiji/Stitching/master/src/main/resources/plugins.config)),
`Plugins>Registration, "Linear Stack Alignment with SIFT"` and `Process, "Enhance Local Contrast (CLAHE)"`
([mpicbg](https://raw.githubusercontent.com/axtimwalde/mpicbg/master/mpicbg_/src/main/resources/plugins.config)),
`Plugins, "3D Viewer"` ([3D_Viewer](https://raw.githubusercontent.com/fiji/3D_Viewer/master/src/main/resources/plugins.config)).

**What works.**
- The toolbox itself: the Process/Analyze menus are the de-facto standard for 2-D
  measurement (the 18 measurement check-boxes of *Set Measurements*, *Analyze Particles*,
  *Set Scale*, *Calibrate* - [User Guide, Analyze menu](https://imagej.net/ij/docs/guide/146-30.html)).
- Single-letter shortcuts (`m` measure, `t` add to manager, `h` histogram, `k` plot profile,
  `b` add to overlay) ([Analyze](https://imagej.net/ij/docs/guide/146-30.html),
  [Image](https://imagej.net/ij/docs/guide/146-28.html)).
- The macro recorder (*Plugins ▸ Macros ▸ Record…*) and the Command Finder (`L`), which the guide
  calls "The quickest way to find a command without having to navigate through all the menus"
  ([User Guide, Plugins menu](https://imagej.net/ij/docs/guide/146-31.html)). It was still being
  improved in 1.54m (Dec 2024): filtering "now splits the input into individual words and returns
  commands where all words are found in either the command name or the menu path"
  ([ImageJ release notes](https://imagej.net/ij/notes.html)).
- Display adjustments are non-destructive until *Apply*: "Adjusting the sliders changes the LUT -
  and not the pixel values. That is, unless you press Apply"
  ([Bioimage Book, ImageJ chapter](https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html)).

**What users complain about (with sources).**
- *Undo.* "ImageJ does not have a large undo/redo buffer." Undo "is currently limited to the most
  recent image editing / filtering operation"; multi-image undo needs "Keep multiple undo buffers"
  in *Edit ▸ Options ▸ Memory & Threads*
  ([User Guide §Undo](https://imagej.net/ij/docs/guide/146-6.html)). The ImageJ2 team's own
  tracker says "ideally, we would like ImageJ to support multiple undo and redo as efficiently as
  possible" ([imagej-common #3](https://github.com/imagej/imagej-common/issues/3)).
- *Window management.* "I think difficulty in managing multiple windows is a general problem
  when many images are opened with ImageJ … I'm always confused with the massive number of
  windows" ([Image.sc: Manage multiple windows in ImageJ, 2022](https://forum.image.sc/t/manage-multiple-windows-in-imagej/68201)).
  Multi-monitor users report windows landing on the wrong screen
  ([Image.sc 1418](https://forum.image.sc/t/window-placement-with-multiple-screens/1418),
  [Image.sc 24990](https://forum.image.sc/t/better-support-for-multi-screen-setups-in-imagej-fiji/24990)).
- *Four separate adjust tools.* *Image ▸ Adjust* holds Brightness/Contrast `[C]`, Window/Level,
  Color Balance, Threshold `[T]` and Color Threshold as separate floating dialogs, plus
  *Image ▸ Color ▸ Channels Tool* `[Z]` ([User Guide, Image menu](https://imagej.net/ij/docs/guide/146-28.html)).
- *Apply destroys data.* The B&C *Apply* button "applies the current display range mapping
  function to the pixel data", and teaching material warns "in most cases you should not press
  Apply" ([ImageJ docs](https://imagej.net/ij/docs/guide/146-28.html),
  [Bioimage Book](https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html)).
- *Binary semantics.* "Binary commands assume that objects are black and background is white
  unless 'Black Background' is checked in the Process>Binary>Options dialog box"
  ([Process menu](https://imagej.net/ij/docs/menus/process.html)); the setting is a persistent
  preference (`Prefs.blackBackground`, [Binary.java](https://imagej.net/developer/source/ij/plugin/filter/Binary.java.html)),
  so a value left over from another session silently swaps erode/dilate. *Make Binary* and
  *Convert to Mask* coexist in the same submenu ([Process menu](https://imagej.net/ij/docs/menus/process.html)).
- *Discoverability.* EasyFiji's authors built a whole GUI plugin because the raw menus are hard
  for life scientists; a core-facility manager replied "The folk who would use this most are my
  core imaging facility users, including undergrad students and they don't read ImageSc!"
  ([Image.sc: Announcing EasyFiji, Nov 2025](https://forum.image.sc/t/announcing-easyfiji-a-user-friendly-gui-plugin-for-fiji/117617)).
- *HiDPI / look-and-feel.* Fiji 2.7.0 (Sept 2022) added "experimental support for the FlatLaf Look
  and Feel (L&F), which makes Fiji behave better on HiDPI displays. There is still more to be done
  to make Fiji fully HiDPI-aware"; the preference did not persist across restarts
  ([Image.sc: Fiji 2.7.0 released](https://forum.image.sc/t/fiji-2-7-0-released/71652)).

**State of the UI, 2024-2026.** ImageJ 1.x is still maintained by Wayne Rasband; the release
notes top out at 1.54p (17 Feb 2025) and consist of bug fixes and small dialog improvements
([ImageJ release notes](https://imagej.net/ij/notes.html)). Fiji's development effort in this
period went into the runtime, not the UI: the Jaunch native launcher (1.0.0 in March 2025,
"code-signed … run natively on Apple silicon"), "Fiji-Latest bundles now available with Java 21"
(14 May 2025), Jaunch 2.0.3 (Nov 2025), and a Script Editor console "revamped … (views, colors,
dark mode)" in Oct 2026 ([Image.sc: Fiji & friends dev update thread](https://forum.image.sc/t/fiji-friends-dev-update-thread/103718)).
No new GUI is planned. The definitive discussion is the 2018 *Modern mode future* thread:
"the modern mode was a trial to re-build a graphical user interface (GUI) for ImageJ2 from
scratch, but the current focus of development is on the IJ2 backend" (J. Eglinger); "The
current/legacy ImageJ UI is the result of many cycles of iteration and the existing user-base
generally prefers it. When we released the first few betas of ImageJ2, which shipped with the
Swing UI active by default, the bug reports flooded in for things like the brightness/contrast
dialog … While the modern Swing UI has some technical advantages, it was never fully finished";
"a new ImageJ UI project should have dedicated funding and people … I would favor developing in
JavaScript over JavaFX" (C. Rueden); "building a new UI is overdue for some 15 years"
(R. Haase) ([Image.sc: Modern mode future](https://forum.image.sc/t/modern-mode-future/20639)).
Rueden also explains *why* the IJ1 UI resists replacement: "Some legacy plugins assume/hardcode
UI implementation details, such as there being a visible ij.ImageJ main window … Some legacy
plugins try to do real-time updates - e.g. Plot Profile in live mode, or Brightness/Contrast"
(same thread).

**Lesson for Lumen.** Keep the toolbox and its vocabulary (measurement names, shortcut letters,
command names) because that is what users value; replace the window model, the undo model and
the four-dialog display model, which are the documented pain points. Do not try to be
pixel-compatible with the IJ1 dialogs - that is the trap the ImageJ2 Swing UI fell into.
Building the UI in a browser stack rather than a JVM toolkit follows Rueden's own 2018 reasoning.

## 2. ImageJ2 and the search bar

**UI model.** ImageJ2 "separates concerns, fully decoupling the data model from the user
interface" and defines "parameterized modules: executable routines with declared input and
output parameters of specific types", so the framework, not the plugin author, builds input
dialogs. It "includes a user interface written in Java's Swing toolkit which is modeled closely
after the ImageJ 1.x design" and a legacy layer in which "ImageJ2 intercepts an ImageJ 1.x
request and attempts to delegate to its own routines"
([Rueden et al. 2017, BMC Bioinformatics](https://europepmc.org/article/PMC/PMC5708080)).

**The search bar.** Added 29 Dec 2017 (ImageJ 2.0.0-rc-64): "the search bar finds commands,
and can search the ImageJ wiki as well as the ImageJ Forum"; queries starting with `!` run code
snippets; "Developers can extend the capabilities of the search bar by writing Searcher
plugins" ([ImageJ news 2017-12-29](https://imagej.net/news/2017-12-29-imagej2-and-fiji-updated-now-with-search-bar)).
It is embedded in the main toolbar window and is the one piece of ImageJ2 UI that shipped to all
Fiji users ([ImageJ2 page](https://imagej.net/ImageJ2)).

**Lesson for Lumen.** Two ideas carried over directly: (1) every command is a declared
parameter schema from which the UI is generated; (2) search is the primary launcher. Lumen
goes one step further by making search a palette with fuzzy matching, category, description and
shortcut, and by making commands recordable as Python because they go through one registry
(the same problem Rueden identified: IJ1 commands run from the menu bypass the module system,
"ModuleEvents not fired for IJ1 command execution",
[Modern mode future #20](https://forum.image.sc/t/modern-mode-future/20639)).

## 3. napari

**UI model.** One Qt window: canvas (vispy) in the middle, layer list and per-layer controls
docked on the left, dimension sliders under the canvas, plugin widgets docked at the sides. The
Python model (`Viewer`, layers, dims) is separated from the Qt view and connected through
events ([napari architecture guide](https://napari.org/stable/developers/architecture/index.html));
the Qt viewer is a `QSplitter` holding the layer list, layer buttons and a layer-controls
container ([napari.qt.QtViewer](https://napari.org/0.4.19/api/napari.qt.QtViewer.html)).
Seven layer types (Image, Labels, Points, Shapes, Surface, Tracks, Vectors)
([napari layers guide](https://napari.org/stable/howtos/layers/index.html)).

**0.5 (11 Jul 2024).** "the beginning of an architectural overhaul of napari": asynchronous
slicing/rendering, menus and commands moved onto *app-model*, which "allows plugins to organize
their commands in defined menus in the napari menubar and application" (NAP-6); the plugin
manager was moved out of the core repository; layer-list conveniences (alt-click eye to show
only that layer, context menu to hide unselected layers, a transform button "allowing you to
resize and rotate layers") ([napari 0.5.0 release notes](https://napari.org/stable/release/release_0_5_0.html)).

**0.6 (1 May 2025).** A command palette: "Press Ctrl/Command+Shift+P and start typing the name
of the action you want to use", including plugin actions; axis-orientation ("handedness")
controls; right-click indicators on buttons that hide extra menus; a clearer 2D/3D toggle;
colormap button in layer controls; multi-layer status bar; grid spacing; log viewer
([napari 0.6.0 release notes](https://napari.org/stable/release/release_0_6_0.html),
[feature highlights](https://napari.org/stable/getting_started/features.html)).
The app-model docs still say "the application model is currently not a part of the public
napari interface" ([app-model](https://napari.org/stable/developers/architecture/app_model.html)).

**Plugin manager.** `napari-plugin-manager` "used to be part of the napari codebase before the
0.5.x release series"; it "knows how to detect if napari was installed using" conda or pip,
accepts "the name of the package, a url to the resource" or a dragged file
([napari-plugin-manager README](https://github.com/napari/napari-plugin-manager)); users open
*Plugins ▸ Install/Uninstall Plugins…*, search tiles, or paste a pip/conda specifier
([Finding and installing plugins](https://napari.org/stable/plugins/start_using_plugins/finding_and_installing_plugins.html)).

**What works.** Layers as the data model; N-D sliders for free; non-destructive per-layer
contrast/colormap/gamma/blending; a Python console in the viewer; plugins dock instead of
spawning windows; a command palette since 0.6.

**Known UX pain points (sources).**
- Layer controls: "Layer controls don't seem to have any indication of what level they are
  operating on", "For layer controls that impact one another, there is no indication that they
  are related", "These controls are not presented in the same place, same order, or sometimes
  even with the same tool tip" - design issue open since Nov 2022
  ([napari #5358](https://github.com/napari/napari/issues/5358)).
- Contrast limits: "the default behavior of napari is setting the max pixel value as the vmax,
  which makes the control of values in the interesting region very difficult"; a maintainer
  replied that "make contrast limits widget more discoverable" had long been on the to-do list
  ([napari #4203](https://github.com/napari/napari/issues/4203)); only the contrast slider has a
  right-click fine-control pop-up ([#7297](https://github.com/napari/napari/issues/7297)); reset
  behaviour is inconsistent ([#8604](https://github.com/napari/napari/issues/8604)); no way to
  copy limits between layers ([#9198](https://github.com/napari/napari/issues/9198)); accidental
  edits motivated a request to lock controls ([#6889](https://github.com/napari/napari/issues/6889)).
- Performance: "if I add some Points layers, the viewer gets very sluggish and slow … gets very
  annoying when creating ~10 Points layers" ([napari #6658](https://github.com/napari/napari/issues/6658));
  3-D viewing "a lot of stuttering when moving the image"
  ([Image.sc 68103](https://forum.image.sc/t/poor-3d-viewer-performance-in-napari/68103)); "we
  haven't optimised the points layer much at all"
  ([Image.sc 45512](https://forum.image.sc/t/visualisation-of-images-and-million-of-dots-in-napari/45512)).
- Dock widgets are developer-built: widgets that do not refresh when layers change
  ([Image.sc 40985](https://forum.image.sc/t/napari-auto-updating-dock-widget-min-max-properties-made-with-magicgui/40985))
  and "napari does not provide a good way for displaying temporary interactive overlays on top
  of layers" ([Image.sc 67059](https://forum.image.sc/t/new-widget-control-napari/67059)).
- No ImageJ-style processing toolbox or results table in core; analysis lives in plugins and
  there is no global undo for image operations.

**Lesson for Lumen.** Adopt the single window, the per-layer (per-channel) non-destructive
display controls, docked widgets, the Python console and the 0.6 command palette. Avoid the
documented failure modes: give contrast controls a histogram and typed min/max from the start,
keep every control in the same place and order for every command (schema-generated sheets), and
ship the processing/measurement toolbox in core so that users are not dependent on dock widgets
of varying quality.

## 4. QuPath

**UI model.** Single workspace: toolbar on top, viewer in the middle, an *Analysis pane*
(`Shift+A`) that "contains the main tabs for managing projects, images and objects" - Project,
Image, Annotations, Hierarchy, Workflow ([QuPath 0.6 command reference](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html),
[First steps](https://qupath.readthedocs.io/en/0.6/docs/starting/first_steps.html)).
*View ▸ Show command list* (`Ctrl+L`) is documented as "Show the command list (much easier than
navigating menus…)" ([commands reference](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html)).
Brightness/Contrast (`Shift+C`) "changes how the image is displayed, not the image itself"
(same page). The *Workflow* tab logs commands run through the GUI; double-clicking an entry
reopens its dialog with the same settings, and "workflows can be turned into scripts" for batch
runs ([Workflows](https://qupath.readthedocs.io/en/0.6/docs/scripting/workflows.html)); the
history "is useful to help automatically generate batch-processing scripts"
([commands reference](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html)).
Pixel classifiers train with live feedback: "Press Live prediction and QuPath should already
start showing its predicted classifications"
([Pixel classification tutorial](https://qupath.readthedocs.io/en/0.6/docs/tutorials/pixel_classification.html)).

**0.6.0 (27 Jun 2025).** Extension Manager; InstanSeg cell segmentation; OME-Zarr read/write; a
built-in *QuPath tour* of the interface; a toolbar redesign whose goal was "to make the most
important buttons visible and easy to access, even on small laptop screens"; redesigned class
list with per-class eye icons; measurement tables with a toolbar and an interactive
scatter-plot; "new navigation controls for browsing through time series and z-stacks, replacing
the old sliders" with z-projection overlays; an ImageJ script runner
([Image.sc: QuPath v0.6.0 now available](https://forum.image.sc/t/qupath-v0-6-0-now-available/114104)).
v0.7.0 followed ([GitHub releases](https://github.com/qupath/qupath/releases),
[call for testing, Feb 2026](https://forum.image.sc/t/call-for-help-testing-qupath-v0-7-0/119313)).

**Complaints.** "QuPath's undo is limited, and turns itself off (for performance reasons) when
many objects are present" ([commands reference](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html));
it "works best if you have plenty of RAM (≥ 16 GB)"
([FAQ 0.4](https://qupath.readthedocs.io/en/0.4/docs/reference/faqs.html)); hangs on some
Bio-Formats images outside a project ([qupath #894](https://github.com/qupath/qupath/issues/894)).
It is a whole-slide tool: the object hierarchy and project model are heavier than a biologist
with a 5-D microscopy stack needs.

**Lesson for Lumen.** The three-region layout (navigator/project left, viewer centre, objects
and measurements right), the `Ctrl+L` command list, and the command history that becomes a script
are directly validated. Lumen's recorder-always-on and History inspector are the same idea
applied to a Python API. QuPath's limited undo is a warning: undo must be designed in from the
start, not bolted onto an object model.

## 5. Icy

**UI model.** Icy is "a collaborative bioimage informatics platform that combines a community
website for contributing and sharing tools and material, and software with a high-end visual
programming framework" ([de Chaumont et al. 2012, Nature Methods](https://europepmc.org/abstract/MED/22743774)).
Its main window is a Microsoft-Office-style ribbon: functions are "organized in tabs and
subgroups" and a full ImageJ runs "in its own ribbon tab"
([BIII: Bioimage analysis with Icy](https://biii.eu/bioimage-analysis-icy)); the GUI "is
organised into a ribbon menu and side panels"
([NEUBIAS webinar Q&A](https://forum.image.sc/t/neubias-academy-home-webinar-bioimage-analysis-with-icy-questions-answers/37460)).
"With protocols you can graphically design the workflow of plugins and scripts to be executed";
"Easily write, share, search and install any plugin from the community directly from Icy";
"Easily write scripts in javascript or python with the integrated script editor"; it "Works
natively on Windows, Linux and MacOS (Java required)" and lists "Interactive widgets, interface
builder, OpenCL, 3D data visualization, workspaces" ([icy.bioimageanalysis.org](https://icy.bioimageanalysis.org/)).
The site still lists new plugins in 2026 (multiTools, 30 Apr 2026, same page).

**What works.** In-app plugin search/installation (years before napari), visual Protocols for
non-programmers, multiple synchronised viewers of one sequence.

**Complaints.** Little independent usability literature; the ribbon trades menu depth for
tab-switching and still spawns separate windows for sequences. Java dependency.
**(Specific user complaints not verified in this review.)**

**Lesson for Lumen.** In-app plugin discovery and a graphical pipeline are valuable but are
second-order; the first-order need is that a single command is fast to find and run. Lumen's
`~/.lumen/plugins` drop-in model is the minimal version of Icy's plugin store.

## 6. CellProfiler

**UI model.** A pipeline is an ordered list of modules on the left, the selected module's
settings on the right, and every module can pop its own output window. CellProfiler 2.0 was
"engineered to meet the needs of its growing user base. It is more robust and user friendly"
([Kamentsky et al. 2011](https://europepmc.org/abstract/MED/21349861)) with "drag-and-drop
operations, context-sensitive menus, undo capabilities, user-friendly error reporting, and
context-dependent warnings" and a "newly-designed test mode [that] allows a researcher to step
through a pipeline and repeatedly adjust settings to optimize image analysis"
([Kamentsky et al. 2011, full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3072555)).
CellProfiler 4: "Guided by feedback from biologists, we have made several improvements to the
CellProfiler user interface"; "a new search filter in the 'Add module' popup allows users to more
easily find desired modules"; "new icons in the Test Mode pipeline interface provide a stronger
visual indication of which module" runs next; "We added an interface to visualize which modules
produce inputs needed by, or use outputs from, a module of interest"; and "In an independent
analysis of 15 free image analysis tools CellProfiler scored highly in both usability and
functionality" ([Stirling et al. 2021, BMC Bioinformatics](https://europepmc.org/article/PMC/PMC8431850)).
A 2023 PLOS Biology perspective singles it out: its interface "allows users to construct their
image analysis pipeline easily without requiring coding skills"
([Paul-Gilloteaux 2023](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002213)).

**What works.** Batch is the default mental model; test mode is a form of live preview; module
search was added for the same reason ImageJ added the Command Finder.

**Complaints.** The pipeline model is poor for exploratory single-image work; module output
windows are, again, window soup; settings are long forms without instant preview.

**Lesson for Lumen.** Search over an ordered catalogue, show only relevant settings, and
preview before committing. Lumen's recorder (every applied command becomes a Python line) is the
bridge from exploration to a batch pipeline without forcing the pipeline model on interactive use.

## 7. ilastik

**UI model.** "an easy-to-use interactive tool that brings machine-learning-based (bio)image
analysis to end users without substantial computational expertise … Users adapt the workflows to
the problem at hand by interactively providing sparse training annotations for a nonlinear
classifier … Its computational back end runs operations on-demand wherever possible, allowing for
interactive prediction on data larger than RAM. Once the classifiers are trained, ilastik
workflows can be applied to new data from the command line"
([Berg et al. 2019, Nature Methods](https://europepmc.org/abstract/MED/31570887)). Workflows are a
fixed vertical sequence of applets (Input Data, Feature Selection, Training, Prediction Export);
in Training you "Select a class and scribble over pixels which belong to it", then "press the
Live update button" to see predictions "as an overlay on the image"; "the user can view an
uncertainty map, this indicates areas where the classifier is unsure"; the classifier "can be
applied to unseen images as batch processing (without further training)"
([ilastik pixel classification docs](https://www.ilastik.org/documentation/pixelclassification/pixelclassification)).

**What works.** Interactive training with immediate overlay feedback is the canonical example
of "live preview" for an expensive operation: compute lazily, show the current view, let the user
iterate. Uncertainty display tells the user where to annotate next.

**Complaints.** The applet sequence is rigid; the viewer is a bespoke volumina widget with
its own conventions; projects are single-purpose files.

**Lesson for Lumen.** Expensive commands (ML segmentation, tracking) should still preview on the
current plane and stay non-modal; Lumen's preview-on-current-slice principle generalises
ilastik's *Live update*.

## 8. Commercial packages

### Imaris (Oxford Instruments)
"step-by-step wizards with a preview of the results at each point in the workflow"
([Imaris for Cell Biologists](https://imaris.oxinst.com/products/imaris-for-cell-biologists)).
Object creation (Spots, Surfaces, Cells, Filaments) runs through Creation Wizards in the Surpass
view; "All creation steps can be validated on a slicer (intermediate segmentation steps, region
threshold, seed points, etc.)" ([Imaris 9.8 notes](https://imaris.oxinst.com/support/imaris-release-notes/9-8-0));
batch is set up by "starting the 'New Batch' wizard from the main toolbar … batch pipelines
consisting of multiple image processing steps followed by detection of spots, surfaces, cells or
filaments" ([Imaris 9.3 notes](https://imaris.oxinst.com/support/imaris-release-notes/9-3-0));
10.1 put machine-learning segmentation inside the Surfaces wizard with the classifier "stored as
part of the Surfaces creation parameters, which can then be used for batch processing"
([Imaris 10.1 notes](https://imaris.oxinst.com/support/imaris-release-notes/10-1-0)); Imaris 11
adds Workflows, "a step-by-step protocol system designed to simplify and streamline analysis
while enabling batch processing and protocol sharing"
([Imaris 11.0 notes](https://imaris.oxinst.com/support/imaris-release-notes/11-0-0)). A core
facility describes it as having "a reasonably clean user interface" usable "with relatively
little training" ([Microscopy Today NetNotes](https://www.cambridge.org/core/journals/microscopy-today/article/netnotes/21D7980A3B3F6DDCE4A3D82BD2A2E577)).
*Lesson:* every parameter in a wizard has an immediate visual consequence; parameters are stored
with the result so they can be re-applied.

### arivis Pro (Zeiss, formerly Vision4D)
"ZEISS arivis Pro 4.2 (formerly Vision4D)"; the Analysis Pipeline "can directly show a preview in
the 4D Viewer for object creation operations" ([arivis Pro 4.2 notes](https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-2));
4.3's "High precision preview in the Analysis Pipeline can work on 3D data"
([release notes index](https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes)). The
pipeline panel has two areas, "The Pipeline area and the analysis operations list area", and
operators are added by double-click or drag
([arivis Pro help: building the analysis pipeline](https://knowledge.zeiss.com/rms/en/arivis-pro/scripting-and-interaction-with-other-programs/performing-objects-density-heat-map-and-object-distribution-gradient-plotting-profile/building-the-analysis-pipeline)).
*Lesson:* a docked operation list with search beside a pipeline, previewed in the same viewer.

### Aivia (Leica)
Aivia 15 (May 2025): "With the new Aivia Launchpad, users are guided through analysis setup using
an intuitive interface featuring pre-built Guided Sequences and embedded tutorials"; "The new
Workflow Creator simplifies batching setup"; "Segment by Example, which enables users to paint
directly on 2D or 3D sample cells to deploy an AI model"
([Leica news](https://www.leica-microsystems.com/company/news/news-details/latest-ai-software-simplifies-image-analysis-and-speeds-up-insights-for-scientists/)).
"Aivia's Workflow Processor is a powerful way to chain core parts of Aivia together and run the
sequence of steps on single or batches of files" ([Aivia docs](https://aivia-software.atlassian.net/wiki/x/AYCIc)).
*Lesson:* guided entry points for novices on top of the same engine; batch = replay of an
interactive session.

### NIS-Elements GA3 (Nikon)
"A graphically enhanced user interface displayed as a 'decision tree' allows users to readily
create, modify, and interlink customized workflows"; "'Preview' mode lets the user see the
results of all preprocessing and analysis options chosen within the workflow in real-time"
([Western University Biotron, GA3](https://uwo.ca/sci/research/biotron/integrated_microscopy/image_analysis/nikon_ga3_analysis.html));
a recipe "is graphically represented as an oriented graph"
([protocols.io, Nikon UK 2025](https://www.protocols.io/view/nikon-automated-microscopy-import-and-run-ga3-reci-g3dzbyi77.pdf)); recipes run on clusters through a batch interface
([Nikon cluster computing](https://www.microscope.healthcare.nikon.com/products/software/nis-elements/cluster-computing)).
*Lesson:* real-time preview of a whole chain, not only one filter.

### ZEN (Zeiss)
In the Processing tab users "select Single or Batch mode, and single processing is the default";
the *Preview* button shows the effect before commit; "Clicking Apply processes the image into a new
container, and the source stays unchanged"; in batch, settings "apply only to the selected image
but can be copied to other images in the list"
([ZEN Image Processing](https://knowledge.zeiss.com/rms/en/zen/basic-concepts/image-processing),
[Applying Batch Processing](https://knowledge.zeiss.com/rms/en/zen/basic-concepts/image-processing/applying-batch-processing)).
*Lesson:* Single/Batch as two tabs of the same operation sheet; non-destructive by default.

**Common thread across the commercial tools:** one workspace, docked panels, preview before
apply, parameters stored with results, and batch as a replay of the interactive setup. None of
them is open or scriptable in the Fiji sense, which is the gap Lumen aims at.

## 9. Web viewers: OMERO.iviewer, Neuroglancer, Viv/Avivator

**OMERO.iviewer.** "An OMERO.web app for visualizing images in OMERO", viewer "based on
OpenLayers" ([omero-iviewer README](https://github.com/ome/omero-iviewer)). Features: "Toggle
channels on and off, change colors and adjust levels"; "Draw shapes and select colors and line
intensity"; "Duplicate ROIs across multiple images"; "Support for multiple side-by-side viewers
and the ability to synchronize settings between them"
([openmicroscopy.org/omero/iviewer](https://www.openmicroscopy.org/omero/iviewer/)). Rendering
settings are saved server-side and "This NEVER changes the original Image data and can be
reverted at any time" ([OME workshop slides 2019](https://downloads.openmicroscopy.org/presentations/2019/ELMI/ELMI_workshop_2019.pdf)).

**Neuroglancer.** "Neuroglancer is a WebGL-based viewer for volumetric data … purely a
client-side program, but it depends on data being accessible via HTTP in a suitable format";
"The four-pane view consists of 3 orthogonal cross-sectional views as well as a 3-D view";
sources include precomputed, N5, "Zarr v2/v3 and OME-Zarr 0.4/0.5"; "Click on a layer name to
toggle its visibility"; requires "WebGL (2.0) and the EXT_color_buffer_float extension"
([Neuroglancer README](https://github.com/google/neuroglancer)).

**Viv / Avivator.** "Viv is a JavaScript library for rendering OME-TIFF and OME-NGFF (Zarr)
directly in the browser. The rendering components of Viv are packaged as deck.gl layers";
Avivator is "A lightweight viewer for local and remote datasets"
([Viv README](https://github.com/hms-dbmi/viv); paper:
[Manz et al. 2022, Nature Methods, doi:10.1038/s41592-022-01482-7](https://doi.org/10.1038/s41592-022-01482-7)).

**What works.** Zero-install, URL-shareable viewing; per-channel colour/range controls as the
primary UI; GPU compositing in the browser; chunked remote data.

**Complaints.** They are viewers, not analysis tools: little measurement beyond ROI shapes
(iviewer), no processing, no undo, and (Neuroglancer) a steep keyboard-driven learning curve.

**Lesson for Lumen.** Lumen's client-side compositor (raw planes + LUT + min/max in the
browser, [DESIGN.md](DESIGN.md)) is the Viv/iviewer model; the server keeps the numpy engine.
OME-Zarr input and a shareable view state are the natural next steps.

## 10. 3D Slicer

**UI model.** A module panel on the left shows the controls of the current module; a toolbar
with module search ("Ctrl + f or click on magnify glass icon"), module history (back/forward),
and a Favorite-modules bar; a layout drop-down of slice/3-D/chart/table arrangements - "When
Slicer is exited normally, the selected layout is saved and restored next time the application
is started"; every view has a pin-out controller bar; a Data Probe reports values under the
cursor; a Python console toggles with ``Ctrl+3``
([3D Slicer user interface docs](https://slicer.readthedocs.io/en/latest/user_guide/user_interface.html)).
The docs describe no global undo (same page).

**Lesson for Lumen.** Module search with history and favourites is the command palette in
another costume; persistent layout and a data probe (pixel value under cursor) are cheap wins.

## 11. General interaction references

- **Photoshop adjustment layers**: "apply color and tonal corrections (like brightness, contrast,
  or vibrance) without permanently changing pixel data"
  ([Adobe Help](https://helpx.adobe.com/photoshop/using/adjustment-fill-layers.html)). The model
  for Lumen's view-level display settings.
- **Affinity Photo live filters**: "Live filters allow you to apply filter effects such as
  blurring, sharpening, noise and distortion non-destructively"; double-click the layer to
  change settings later ([Affinity Help](https://affinity.help/photo2/en-US.lproj/pages/Layers/livefilters.html)).
  Lumen's preview-then-apply sheet is the scientific-image compromise: preview is live, the
  commit is explicit, and undo history restores the previous pixels.
- **Figma Quick Actions**: `⌘/` / `Ctrl+/` opens a searchable menu of "recent actions, Figma
  plugins, menu items, and more" ([Figma forum announcement](https://forum.figma.com/t/new-quick-actions-menu/1788));
  users complain when plugins fail to appear in it
  ([Figma forum](https://forum.figma.com/t/plugins-not-always-appearing-in-quick-actions-bar-search/25639)) -
  i.e. once a palette exists, *everything* must be in it.
- **VS Code**: "The most important key combination to know is ⇧⌘P … which brings up the Command
  Palette. From here, you have access to all functionality within VS Code"; the workbench "is
  divided into six main areas" (Editor, Primary Side Bar, Secondary Side Bar, Status Bar,
  Activity Bar, Panel) and "shows open items with tabs"
  ([VS Code UI docs](https://code.visualstudio.com/docs/getstarted/userinterface)). Lumen's
  tool rail / viewport tabs / inspector / palette mirror this.

## 12. HCI and usability literature on scientific image software

- **Kizic & Borovac 2001, Chalmers Lindholmen University College**: bachelor's thesis "Image/J -
  graphical user interface improvement", run March-June 2001 with the Center for Image
  Processing in Education; the aim was "to evaluate usability of an image processing software
  (Image/J), and then to improve it", using heuristic evaluation and cognitive walkthrough
  ([mirror of the thesis PDF](https://corpora.tika.apache.org/base/docs/govdocs1/600/600521.pdf)).
  The full text could not be downloaded during this review, so its specific findings are
  **(not verified)**; the point stands that ImageJ's UI was already being usability-tested in
  2001 and the basic window model has not changed since.
- **Carpenter, Kamentsky & Eliceiri 2012**, "A call for bioimaging software usability":
  "Bioimaging software developed in a research setting often is not widely used by the
  scientific community. We suggest that … usability should be a more highly valued goal"
  ([Nature Methods 9:666](https://europepmc.org/abstract/MED/22743771)).
- **2020 BioImage Analysis Survey** (COBA; 484 participants): the "vast majority" use
  "open-source point and click software" (ImageJ, Fiji, CellProfiler, Icy); respondents asked for
  "easy-to-follow (at your own pace) beginner's guides", "a centralized location to find
  available image analysis tools", example workflows and interoperable platforms; "some users
  find the image.sc forum intimidating"
  ([Jamali et al. 2021, Biological Imaging](https://www.cambridge.org/core/journals/biological-imaging/article/2020-bioimage-analysis-survey-community-experiences-and-needs-for-the-future/9E824DC0C27568FE5B9D12FB59B1BB90)).
- **2023 "Bridging Imaging Users to Imaging Analysis"** (493 participants): "respondents'
  overarching needs are documentation, detailed tutorials on the usage of image analysis tools",
  user-friendly software and better segmentation; "The respondents expressed their needs for more
  user-friendly tools/plugins"; yet "approximately ¾ of participants reporting using computational
  libraries and scripts" and "No-code-required tools are not necessarily needed by every member of
  the community" ([Sivagurunathan et al. 2023, J. Microscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC10274673/)).
- **Paul-Gilloteaux 2023**, "Bioimage informatics: Investing in software usability is
  essential": methods "often have software for implementation, but it is not usable because of a
  lack of interface and guidance"; success depends on "the user interface and user experience,
  ease of installation, shallowness of the learning curve"
  ([PLOS Biology](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002213)).
- **EasyFiji 2026** (Li & Taylor, St. Jude): a Fiji plugin with "Task-based panels for Display,
  Process, Save, and Image Info", "Channel-specific and undoable processing" and a "Command
  recorder for reproducibility" ([Image.sc announcement](https://forum.image.sc/t/announcing-easyfiji-a-user-friendly-gui-plugin-for-fiji/117617);
  [JoVE article](https://doi.org/10.3791/69441)). Its existence - and the choice of exactly those
  four panels plus per-channel undo - is independent confirmation of Lumen's priorities.
- **Fiji 2012 / ImageJ2 2017 papers** both criticise ImageJ 1.x architecture rather than its UI
  ([Schindelin 2012](https://europepmc.org/abstract/MED/22743772),
  [Rueden 2017](https://europepmc.org/article/PMC/PMC5708080)); the UI critique lives in the
  forum and in user-facing plugins, not in the literature.

**Synthesis.** Across surveys the two dominant requests are *user-friendly point-and-click
tools* and *scripting that is not a separate world*. The literature says what, not how; the
how comes from the tools reviewed above: palette, docked panels, live preview, non-destructive
display, deep undo, and a recorder that turns clicks into code.

## 13. Comparison matrix

Legend: ✅ yes · 🟡 partial · ❌ no · n/v = not verified in this review. Sources are the
sections above.

| Tool | Single window | Command search / palette | Live preview | Undo depth | Per-channel display controls | Docked results / ROI panels | Scripting + recorder | Plugin model | Batch | 3-D |
|---|---|---|---|---|---|---|---|---|---|---|
| Fiji / ImageJ 1.x | ❌ one window per image/dialog/table | 🟡 Command Finder `L`; ImageJ2 search bar in toolbar | 🟡 "Preview" checkbox on some filters only | 1 step ("most recent … operation") | 🟡 B&C + Channels Tool as separate floating dialogs | ❌ ROI Manager, Results, histogram are windows | ✅ IJ macro recorder; 10 script languages | ✅ jars/scripts in `plugins/`, update sites | 🟡 Process ▸ Batch, macros | 🟡 3D Viewer (Java 3D), Volume Viewer, BDV plugin |
| ImageJ2 "modern mode" | 🟡 Swing UI, never finished | ✅ search bar | 🟡 | n/v | n/v | n/v | ✅ SciJava scripts | ✅ SciJava plugins | 🟡 headless | n/v |
| napari 0.6 | ✅ | ✅ `Ctrl/⌘⇧P` palette (0.6) | 🟡 depends on plugin widget | ❌ no global undo (Labels painting only, n/v) | ✅ per-layer contrast/colormap/gamma/blending | ✅ dock widgets | ✅ IPython console; no recorder | ✅ npe2 manifests, plugin manager | 🟡 via scripts | ✅ vispy 3-D |
| QuPath 0.6 | ✅ | ✅ Command list `Ctrl+L` | 🟡 Live prediction for classifiers | 🟡 "limited, and turns itself off … when many objects" | ✅ B&C dialog per channel | ✅ Analysis pane tabs | ✅ Groovy + Workflow history → script | ✅ Extension Manager (0.6) | ✅ Run for project | 🟡 z-stack navigation + projections; no volume rendering |
| Icy | 🟡 ribbon main window + sequence windows | 🟡 ribbon search (n/v) | 🟡 | n/v | n/v | 🟡 side panels | ✅ JS/Python editor; Protocols | ✅ in-app plugin store | 🟡 Protocols | ✅ VTK 3-D (n/v) |
| CellProfiler 4 | 🟡 main window + module output windows | ✅ module search in "Add module" | 🟡 Test Mode step-through | 🟡 pipeline-edit undo | ❌ | ❌ | 🟡 pipeline file; Python API | ✅ modules/plugins | ✅ core model | 🟡 3-D pipelines |
| ilastik | ✅ | ❌ | ✅ Live update | 🟡 label edits (n/v) | 🟡 | 🟡 applets | 🟡 headless CLI; no recorder | ❌ fixed workflows | ✅ batch export / headless | ✅ 3-D rendering |
| Imaris | ✅ Surpass | ❌ | ✅ "preview of the results at each point" | n/v | ✅ (n/v) | ✅ | 🟡 XTensions (MATLAB/Python); Workflows (11) | 🟡 XTensions | ✅ Batch / Workflows | ✅ |
| arivis Pro | ✅ | 🟡 operations list | ✅ preview in 4D viewer | n/v | ✅ | ✅ | 🟡 Python scripting | 🟡 | ✅ pipelines | ✅ |
| Aivia | ✅ | 🟡 | 🟡 | n/v | ✅ | ✅ | 🟡 recipes / Workflow Processor | 🟡 Bioimage.io models | ✅ | ✅ |
| NIS-Elements GA3 | ✅ | 🟡 | ✅ "Preview" mode of whole graph | n/v | ✅ | ✅ | 🟡 recipes; macros | 🟡 | ✅ incl. cluster | ✅ |
| ZEN | ✅ | 🟡 | ✅ Preview button | n/v | ✅ (n/v) | ✅ | 🟡 macros | 🟡 modules | ✅ Batch tab | ✅ |
| OMERO.iviewer | ✅ (browser) | ❌ | ✅ display only | ❌ | ✅ | ✅ ROI/Info/Settings panel | ❌ | ❌ | ❌ | 🟡 z-projection |
| Neuroglancer | ✅ (browser) | ❌ keyboard-driven | ✅ display only | ❌ | ✅ per layer, shader code | 🟡 layer side panel | ✅ Python API; view state in URL (n/v) | ❌ | ❌ | ✅ 4-pane |
| Viv / Avivator | ✅ (browser) | ❌ | ✅ display only | ❌ | ✅ | 🟡 | ❌ (library) | ❌ | ❌ | ✅ volume rendering |
| 3D Slicer | ✅ | ✅ module search `Ctrl+F` | 🟡 per module | ❌ not documented | 🟡 per volume | ✅ module panel, Data Probe | ✅ Python console | ✅ extensions manager | 🟡 | ✅ |
| **Lumen (v0.1)** | ✅ tabs + inspector | ✅ `⌘K` palette over every registered command | ✅ default for plane ops | ✅ unlimited, visible history | ✅ histogram-backed per-channel min/max, LUT | ✅ ROI Manager, Results, Plot, Histogram, History, Script, Log sections | ✅ Python console + always-on recorder | ✅ `@op` files in `~/.lumen/plugins` | 🟡 `batch.process_folder` and recorded-macro replay over a folder; no job queue or progress GUI | 🟡 3-D View section (MIP / alpha / iso-surface / mesh rendering, OBJ / STL / PLY export, surface measurement) plus projections, reslice and orthogonal views; no multi-image scenes, landmarks or animation recording |

## 14. Design decisions in Lumen, traced to evidence

| Lumen principle ([DESIGN.md](DESIGN.md)) | Evidence it answers | Precedent it borrows |
|---|---|---|
| **1. One window, three regions; images as tabs; results/ROIs/histogram/history docked** | "difficulty in managing multiple windows is a general problem" ([Image.sc 68201](https://forum.image.sc/t/manage-multiple-windows-in-imagej/68201)); the IJ1 *Window* menu exists only to tile/cascade ([Window menu](https://imagej.net/ij/docs/menus/window.html)); multi-screen placement threads ([24990](https://forum.image.sc/t/better-support-for-multi-screen-setups-in-imagej-fiji/24990)) | napari dock widgets; QuPath Analysis pane ([commands](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html)); VS Code "six main areas" + tabs ([VS Code](https://code.visualstudio.com/docs/getstarted/userinterface)); Slicer module panel |
| **2. Command palette first** | Command Finder is "the quickest way to find a command without having to navigate through all the menus" ([User Guide](https://imagej.net/ij/docs/guide/146-31.html)); the ImageJ2 search bar was the one new-UI feature that shipped ([news 2017](https://imagej.net/news/2017-12-29-imagej2-and-fiji-updated-now-with-search-bar)); CellProfiler 4 added module search "to more easily find desired modules" ([Stirling 2021](https://europepmc.org/article/PMC/PMC8431850)) | QuPath `Ctrl+L` "much easier than navigating menus"; napari 0.6 `Ctrl+Shift+P`; VS Code; Figma `⌘/` |
| **3. Every command is a schema** | ImageJ2's "parameterized modules" whose dialogs are generated by the framework ([Rueden 2017](https://europepmc.org/article/PMC/PMC5708080)); napari's #5358: controls "not presented in the same place, same order" ([napari #5358](https://github.com/napari/napari/issues/5358)) | app-model actions with command/menu/keybinding rules ([napari app-model](https://napari.org/stable/developers/architecture/app_model.html)) |
| **4. Live preview by default** | IJ1 filters preview only via a checkbox; Rueden: legacy plugins "try to do real-time updates - e.g. Plot Profile in live mode, or Brightness/Contrast" and break outside the AWT window ([Modern mode future](https://forum.image.sc/t/modern-mode-future/20639)) | ilastik "Live update"; QuPath "Live prediction"; Imaris "preview of the results at each point"; GA3 "Preview" of the whole graph; ZEN Preview button; arivis preview in the 4D viewer |
| **5. Non-destructive, per-channel display with a histogram slider** | B&C *Apply* permanently rescales pixels and tutorials say "do not press this button" ([Bioimage Book](https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html)); five separate Adjust dialogs ([Image menu](https://imagej.net/ij/docs/guide/146-28.html)); napari contrast limits "very difficult" to fine-tune, widget "more discoverable" long on the to-do list ([napari #4203](https://github.com/napari/napari/issues/4203)) | Photoshop adjustment layers; OMERO.iviewer settings that "NEVER change the original Image data"; Viv client-side compositing |
| **6. Unlimited undo with a visible history** | "ImageJ does not have a large undo/redo buffer" ([User Guide](https://imagej.net/ij/docs/guide/146-6.html)); QuPath undo "turns itself off … when many objects are present" ([QuPath](https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html)); EasyFiji advertises "undoable processing" as a feature ([Image.sc](https://forum.image.sc/t/announcing-easyfiji-a-user-friendly-gui-plugin-for-fiji/117617)) | Affinity/Photoshop history; CellProfiler 2 "undo capabilities" |
| **7. Recorder always on (Python)** | Surveys: ¾ of users also script, but want no-code entry ([2023 survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC10274673/)); IJ1 menu commands bypass the module event system so a unified history is impossible in Fiji ([Modern mode future #20](https://forum.image.sc/t/modern-mode-future/20639)) | IJ macro recorder; QuPath Workflow → script; Imaris parameters stored with results; napari IPython console |
| **8. Honest binary semantics (mask = 255 foreground, no Black-background switch)** | "Binary commands assume that objects are black and background is white unless 'Black Background' is checked" and the preference persists across sessions ([Process menu](https://imagej.net/ij/docs/menus/process.html), [Binary.java](https://imagej.net/developer/source/ij/plugin/filter/Binary.java.html)) | scikit-image / napari Labels convention (foreground non-zero) |
| **Browser UI on a Python engine** | Rueden 2018: "I would favor developing in JavaScript over JavaFX … embedded in a web site, standalone desktop client"; the Swing UI "was never fully finished" ([Modern mode future](https://forum.image.sc/t/modern-mode-future/20639)); installation is the first usability barrier ([Paul-Gilloteaux 2023](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002213)) | Viv/Avivator, Neuroglancer, OMERO.iviewer render in the browser; napari/scikit-image give the numpy stack |

**What Lumen still lacks relative to the field** (from the matrix): a batch job queue with progress
and resumable runs (ZEN/Imaris/CellProfiler — `batch.process_folder` runs a script over a folder but
has no job GUI), multi-image 3-D scenes with landmarks and animation recording (Imaris, Fiji 3D Viewer —
Lumen's 3-D View renders one image as MIP / alpha / iso-surface / mesh), lazy multi-resolution streaming of remote data
(Neuroglancer, Viv, QuPath 0.6 — Lumen reads OME-Zarr into memory), an in-app plugin/extension
manager (napari, QuPath 0.6, Icy), and a shareable view state (Neuroglancer URL). These are tracked in
[FIJI_PARITY.md](FIJI_PARITY.md) and the project README.

---

## Sources

ImageJ / Fiji
- ImageJ menu reference: https://imagej.net/ij/docs/menus/ (File, Edit, Image, Process, Analyze, Plugins, Window, Help pages)
- ImageJ User Guide 1.46r: https://imagej.net/ij/docs/guide/146-6.html (Undo), 146-26 (File), 146-27 (Edit), 146-28 (Image), 146-29 (Process), 146-30 (Analyze), 146-31 (Plugins)
- ImageJ release notes: https://imagej.net/ij/notes.html
- ImageJ 1.54 IJ_Props.txt: https://raw.githubusercontent.com/imagej/ImageJ/master/IJ_Props.txt
- Binary.java source: https://imagej.net/developer/source/ij/plugin/filter/Binary.java.html
- Plugin tiers (ImageJ wiki): https://imagej.net/imagej-wiki-static/Plugin
- Fiji overview: https://imagej.net/software/fiji/
- Fiji (software), Wikipedia: https://en.wikipedia.org/wiki/Fiji_(software)
- Schindelin et al. 2012, Fiji, Nature Methods 9:676: https://europepmc.org/abstract/MED/22743772
- Rueden et al. 2017, ImageJ2, BMC Bioinformatics 18:529: https://europepmc.org/article/PMC/PMC5708080
- ImageJ2 search bar announcement (2017-12-29): https://imagej.net/news/2017-12-29-imagej2-and-fiji-updated-now-with-search-bar
- ImageJ2 page: https://imagej.net/ImageJ2
- imagej-common issue #3 (undo/redo): https://github.com/imagej/imagej-common/issues/3
- Image.sc, Modern mode future (2018): https://forum.image.sc/t/modern-mode-future/20639
- Image.sc, Manage multiple windows in ImageJ (2022): https://forum.image.sc/t/manage-multiple-windows-in-imagej/68201
- Image.sc, Window placement with multiple screens: https://forum.image.sc/t/window-placement-with-multiple-screens/1418
- Image.sc, Better support for multi-screen setups: https://forum.image.sc/t/better-support-for-multi-screen-setups-in-imagej-fiji/24990
- Image.sc, Fiji 2.7.0 released (2022): https://forum.image.sc/t/fiji-2-7-0-released/71652
- Image.sc, Fiji & friends dev update thread (2024-2026): https://forum.image.sc/t/fiji-friends-dev-update-thread/103718
- Image.sc, Announcing EasyFiji (2025): https://forum.image.sc/t/announcing-easyfiji-a-user-friendly-gui-plugin-for-fiji/117617
- Li & Taylor 2026, EasyFiji, JoVE: https://doi.org/10.3791/69441 ; code: https://github.com/chenli-git/FijiGui
- Bioimage Book, ImageJ images & pixels: https://bioimagebook.github.io/chapters/1-concepts/1-images_and_pixels/imagej.html
- Fiji plugins.config files: TrackMate https://raw.githubusercontent.com/fiji/TrackMate/master/src/main/resources/plugins.config ; Trainable Segmentation https://raw.githubusercontent.com/fiji/Trainable_Segmentation/master/src/main/resources/plugins.config ; Stitching https://raw.githubusercontent.com/fiji/Stitching/master/src/main/resources/plugins.config ; mpicbg https://raw.githubusercontent.com/axtimwalde/mpicbg/master/mpicbg_/src/main/resources/plugins.config ; 3D Viewer https://raw.githubusercontent.com/fiji/3D_Viewer/master/src/main/resources/plugins.config
- TrackMate getting started: https://imagej.net/plugins/trackmate/tutorials/getting-started
- Trainable Weka Segmentation: https://imagej.net/plugins/tws/
- MorphoLibJ: https://imagej.net/plugins/morpholibj
- Bio-Formats in Fiji: https://imagej.net/formats/bio-formats
- Script Editor: https://imagej.net/scripting/script-editor

napari
- Architecture guide: https://napari.org/stable/developers/architecture/index.html
- App-model: https://napari.org/stable/developers/architecture/app_model.html
- QtViewer API (0.4.19): https://napari.org/0.4.19/api/napari.qt.QtViewer.html
- Layers guide: https://napari.org/stable/howtos/layers/index.html
- 0.5.0 release notes: https://napari.org/stable/release/release_0_5_0.html
- 0.6.0 release notes: https://napari.org/stable/release/release_0_6_0.html
- Feature highlights: https://napari.org/stable/getting_started/features.html
- Finding and installing plugins: https://napari.org/stable/plugins/start_using_plugins/finding_and_installing_plugins.html
- napari-plugin-manager: https://github.com/napari/napari-plugin-manager
- Issues: #5358 https://github.com/napari/napari/issues/5358 ; #4203 https://github.com/napari/napari/issues/4203 ; #7297 https://github.com/napari/napari/issues/7297 ; #8604 https://github.com/napari/napari/issues/8604 ; #9198 https://github.com/napari/napari/issues/9198 ; #6889 https://github.com/napari/napari/issues/6889 ; #6658 https://github.com/napari/napari/issues/6658
- Image.sc threads: https://forum.image.sc/t/poor-3d-viewer-performance-in-napari/68103 ; https://forum.image.sc/t/visualisation-of-images-and-million-of-dots-in-napari/45512 ; https://forum.image.sc/t/napari-auto-updating-dock-widget-min-max-properties-made-with-magicgui/40985 ; https://forum.image.sc/t/new-widget-control-napari/67059

QuPath
- 0.6 command reference: https://qupath.readthedocs.io/en/0.6/docs/reference/commands.html
- 0.6 first steps: https://qupath.readthedocs.io/en/0.6/docs/starting/first_steps.html
- Workflows: https://qupath.readthedocs.io/en/0.6/docs/scripting/workflows.html
- Pixel classification: https://qupath.readthedocs.io/en/0.6/docs/tutorials/pixel_classification.html
- FAQ 0.4: https://qupath.readthedocs.io/en/0.4/docs/reference/faqs.html
- v0.6.0 announcement (2025-06-27): https://forum.image.sc/t/qupath-v0-6-0-now-available/114104
- GitHub releases: https://github.com/qupath/qupath/releases ; issue #894 https://github.com/qupath/qupath/issues/894
- v0.7.0 testing call: https://forum.image.sc/t/call-for-help-testing-qupath-v0-7-0/119313

Icy
- de Chaumont et al. 2012, Nature Methods 9:690: https://europepmc.org/abstract/MED/22743774
- Icy website: https://icy.bioimageanalysis.org/
- BIII entry: https://biii.eu/bioimage-analysis-icy
- NEUBIAS webinar Q&A: https://forum.image.sc/t/neubias-academy-home-webinar-bioimage-analysis-with-icy-questions-answers/37460

CellProfiler
- Kamentsky et al. 2011, Bioinformatics 27:1179: https://europepmc.org/abstract/MED/21349861 ; full text https://pmc.ncbi.nlm.nih.gov/articles/PMC3072555
- Stirling et al. 2021, CellProfiler 4, BMC Bioinformatics 22:433: https://europepmc.org/article/PMC/PMC8431850

ilastik
- Berg et al. 2019, Nature Methods 16:1226: https://europepmc.org/abstract/MED/31570887
- Pixel classification docs: https://www.ilastik.org/documentation/pixelclassification/pixelclassification

Commercial
- Imaris for Cell Biologists: https://imaris.oxinst.com/products/imaris-for-cell-biologists
- Imaris release notes: 9.3 https://imaris.oxinst.com/support/imaris-release-notes/9-3-0 ; 9.8 https://imaris.oxinst.com/support/imaris-release-notes/9-8-0 ; 10.1 https://imaris.oxinst.com/support/imaris-release-notes/10-1-0 ; 11.0 https://imaris.oxinst.com/support/imaris-release-notes/11-0-0
- Microscopy Today NetNotes: https://www.cambridge.org/core/journals/microscopy-today/article/netnotes/21D7980A3B3F6DDCE4A3D82BD2A2E577
- arivis Pro 4.2 release notes: https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-2 ; index https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes ; pipeline help https://knowledge.zeiss.com/rms/en/arivis-pro/scripting-and-interaction-with-other-programs/performing-objects-density-heat-map-and-object-distribution-gradient-plotting-profile/building-the-analysis-pipeline
- Aivia 15 news: https://www.leica-microsystems.com/company/news/news-details/latest-ai-software-simplifies-image-analysis-and-speeds-up-insights-for-scientists/ ; Aivia Workflow Processor docs https://aivia-software.atlassian.net/wiki/x/AYCIc
- NIS-Elements GA3 (Western University Biotron): https://uwo.ca/sci/research/biotron/integrated_microscopy/image_analysis/nikon_ga3_analysis.html ; protocols.io recipe guide https://www.protocols.io/view/nikon-automated-microscopy-import-and-run-ga3-reci-g3dzbyi77.pdf ; Nikon cluster computing https://www.microscope.healthcare.nikon.com/products/software/nis-elements/cluster-computing
- ZEN image processing: https://knowledge.zeiss.com/rms/en/zen/basic-concepts/image-processing ; batch https://knowledge.zeiss.com/rms/en/zen/basic-concepts/image-processing/applying-batch-processing

Web viewers and 3D Slicer
- OMERO.iviewer: https://www.openmicroscopy.org/omero/iviewer/ ; README https://github.com/ome/omero-iviewer ; OME workshop slides https://downloads.openmicroscopy.org/presentations/2019/ELMI/ELMI_workshop_2019.pdf
- Neuroglancer README: https://github.com/google/neuroglancer
- Viv README: https://github.com/hms-dbmi/viv ; Manz et al. 2022, Nature Methods: https://doi.org/10.1038/s41592-022-01482-7
- 3D Slicer user interface: https://slicer.readthedocs.io/en/latest/user_guide/user_interface.html

General interaction references
- Adobe Photoshop adjustment layers: https://helpx.adobe.com/photoshop/using/adjustment-fill-layers.html
- Affinity Photo live filters: https://affinity.help/photo2/en-US.lproj/pages/Layers/livefilters.html
- Figma Quick Actions announcement: https://forum.figma.com/t/new-quick-actions-menu/1788 ; plugin visibility complaint https://forum.figma.com/t/plugins-not-always-appearing-in-quick-actions-bar-search/25639
- VS Code user interface: https://code.visualstudio.com/docs/getstarted/userinterface

Usability literature
- Kizic & Borovac 2001, Image/J - graphical user interface improvement (Chalmers Lindholmen): https://corpora.tika.apache.org/base/docs/govdocs1/600/600521.pdf
- Carpenter, Kamentsky & Eliceiri 2012, A call for bioimaging software usability, Nature Methods 9:666: https://europepmc.org/abstract/MED/22743771
- Jamali et al. 2021, 2020 BioImage Analysis Survey, Biological Imaging: https://www.cambridge.org/core/journals/biological-imaging/article/2020-bioimage-analysis-survey-community-experiences-and-needs-for-the-future/9E824DC0C27568FE5B9D12FB59B1BB90
- Sivagurunathan et al. 2023, Bridging Imaging Users to Imaging Analysis, J. Microscopy: https://pmc.ncbi.nlm.nih.gov/articles/PMC10274673/
- Paul-Gilloteaux 2023, Investing in software usability is essential, PLOS Biology 21:e3002213: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002213

---

## 15. Usability evidence

The design principles traced in section 14 were checked against Fiji in a separate assessment,
[USABILITY.md](USABILITY.md): a heuristic evaluation (Nielsen's ten heuristics plus
reproducibility, provenance, batch/scripting, large-data and accessibility criteria, with 0-4
severity ratings) and a keystroke-level comparison of ten everyday tasks. The Fiji side of each
task is derived step by step from ImageJ 1.54's documentation; the Lumen side was measured on the
running application with a Playwright script in headless Chrome. Result: 84 actions and 20
separate windows (7 modal) for the documented Fiji paths against 54 actions and no separate window
for Lumen, with Lumen shorter on six tasks, equal on three and one action longer on one. The same
document records where Fiji remains ahead (plugin ecosystem, Bio-Formats, script languages, data
larger than RAM, community documentation, validated algorithms), which Lumen gaps the measurement
exposed (display changes are not recorded, no detection preview for tracking, no window tear-off),
and a protocol for the user study that has not been run yet (SUS, completion, time, errors, 8-12
Fiji users). Its section 4 is the per-decision traceability table back to this review.
