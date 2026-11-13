# Fiji / ImageJ ▸ Lumen parity map

*Generated November 2026 against Lumen's command registry (503 registered commands, dumped with
`REGISTRY.ops`) and the frontend menu/tool definitions. The ImageJ side is ImageJ 1.54's
[`IJ_Props.txt`](https://raw.githubusercontent.com/imagej/ImageJ/master/IJ_Props.txt) (every
submenu entry), the hard-coded top-level commands documented in the
[ImageJ User Guide](https://imagej.net/ij/docs/guide/146-26.html) and
[menu reference](https://imagej.net/ij/docs/menus/), and the Fiji additions declared in each
plugin's `plugins.config` (TrackMate, Trainable_Segmentation, Stitching, mpicbg, 3D_Viewer,
Volume_Viewer, Manual_Tracking, Descriptor_based_registration, bUnwarpJ,
register_virtual_stack_slices, Auto_Threshold, Auto_Local_Threshold, AnalyzeSkeleton,
Colocalisation_Analysis, MorphoLibJ, 3D_Objects_Counter, Fiji_Plugins, Directionality, Time_Lapse).*

**Status legend**

| | |
|---|---|
| ✅ covered | a Lumen command (id in backticks) does what the Fiji command does |
| 🟡 partial | exists but with a different algorithm, fewer options, or a different workflow — see Notes |
| ❌ missing | no equivalent yet |
| 🖥️ UI feature | not a command in Lumen: it is a panel, a tool, a tab, a key or a menu action of the single-window UI (counted as covered) |

Every Lumen command is reachable from the ⌘K palette, the menu bar and `lumen.run("<id>", …)`
in the script console; UI features are named by their panel/menu. Fiji commands that are
*deprecated* submenus (e.g. `Plugins › Stitching › deprecated`) are omitted.

---

## File

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| New ▸ Image… | `file.new` | ✅ | 8/16/32-bit/RGB, fill Black/White/Ramp/Noise, any C/Z/T |
| New ▸ Hyperstack… | `file.new` | ✅ | same command (channels/slices/frames) |
| New ▸ Text Window | `file.new_text` (⇧N), `text.append`, `text.set`, `file.save_text`, `text.to_log`; text windows are editable text areas in the inspector's table section | ✅ | a named single-column table shown as an editable text area with Save / Append / Save to file / Download / → Editor and Run as Python / ImageJ macro / JavaScript; `print("[window]", …)` in macros appends to it; `file.import_text` opens text files the same way, `file.export_text` writes Results / tables / Log |
| New ▸ Internal Clipboard | `file.new_from_clipboard` | ✅ | opens the pixels copied with Copy / Cut as a new image; a non-rectangular copy becomes its selection |
| New ▸ System Clipboard | File ▸ New from System Clipboard; ⌘V in the viewport | 🖥️ | the image on the OS clipboard (a screenshot, a copy from another app) opens as a new image through the browser clipboard API, which asks for permission once |
| Open… | File ▸ Open (⌘O): server-side file browser, drag & drop, upload | 🖥️ | readers listed under Import below |
| Open Next | `file.open_next` (⇧O), `file.open_previous` (⌥⇧O) | ✅ | next / previous file by name in the current image's folder, closing the current one like ImageJ |
| Open Samples ▸ (30 ImageJ samples) | `file.open_imagej_sample`; `file.sample` | ✅ | 29 classic ImageJ samples (Blobs, Boats, Bridge, Clown, T1 Head, Mitosis, Flybrain, Bat Cochlea, …) downloaded from imagej.net/images on first use and cached in `~/.lumen/samples`; plus 11 bundled scikit-image samples |
| Open Recent ▸ | File ▸ Open Recent submenu (from `session.recent`); `file.open_recent`; Clear Recent Files | ✅ | newest first, by index or path |
| Import ▸ Image Sequence… | `file.import_sequence` | ✅ | pattern, start/count/step, scale, sort, as Z/T/C, recursive, virtual |
| Import ▸ Raw… | `file.import_raw` | ✅ | type, width/height, offset, n, gap, endian, white-is-zero |
| Import ▸ LUT… | `file.import_lut`, `file.import_luts_folder` | ✅ | ImageJ / Fiji `.lut` files (or a whole `luts/` folder) register as LUTs for every LUT command; optionally applied to the current channel |
| Import ▸ Text Image… | `file.import_text_image` | ✅ | |
| Import ▸ Text File… | `file.import_text` | ✅ | one row per line, as a table named after the file, echoed to the Log |
| Import ▸ Results… | `file.import_results` | ✅ | CSV / TSV / Excel and ImageJ's tab-separated `Results.xls` into the Results table (replace or append) |
| Import ▸ Table… | `file.import_table` | ✅ | CSV / TSV / Excel as a named table |
| Import ▸ URL… | `file.open_url` | ✅ | |
| Import ▸ Stack From List… | `file.import_stack_from_list` | ✅ | one path per line → Z stack, T series or channels |
| Import ▸ TIFF Virtual Stack… | `file.import_tiff_virtual`; `file.bioformats_import(virtual=True)`; File ▸ Open / `POST /api/images/open` ("Open as virtual stack" checkbox, automatic above `auto_virtual_mb`); `file.import_sequence(virtual=True)` for folders | ✅ | TIFF / OME-TIFF / LSM / STK (memmap or page reader), HDF5 datasets + Imaris resolution levels, Zarr / OME-Zarr and npy open as lazy 5-D arrays (`lumen.core.lazy`): planes on demand through a shared LRU cache (`virtual_cache_mb`, 512 MB), copy-on-write on the first pixel write, undo / Revert stay lazy; Z/T projection (max / min / mean / sum / SD / median), Stack Statistics, Z/T profile, Make Substack and the OME-TIFF / ImageJ TIFF / npy / h5 / zarr / sequence exports stream plane by plane with progress + cancel; File ▸ Open goes virtual automatically above half the RAM (`auto_virtual_mb`); one series / level at a time, ranges and crops read the selection into memory |
| Import ▸ AVI… | File ▸ Open | ✅ | avi/mp4/mov/mkv/webm/wmv via imageio-ffmpeg |
| Import ▸ XY Coordinates… | `file.import_xy_coordinates` | ✅ | polygon, segmented line or multi-point selection; invert Y |
| *Fiji* Plugins ▸ Bio-Formats ▸ Bio-Formats Importer | `file.bioformats_import`, `file.bioformats_series`, `file.bioformats_config`; File ▸ Open (`lumen.core.io`, `lumen.core.readers`, `lumen.core.readers.bioformats_bridge` modes `fiji-jars` / `bioio`, `lumen.core.readers.fiji_jars`) | ✅ | full importer dialog (series / all series, colour mode, view + stack order, autoscale, split C/Z/T, group files, virtual, range, crop, swap dimensions, pyramid level, metadata / OME-XML tables) over native readers for TIFF / OME-TIFF (multi-series, pyramids SVS / NDPI / SCN / BIF), LSM, STK, raster formats, NPY, ND2 positions, CZI scenes, LIF, OIF/OIB, DICOM, NIfTI, MRC, HDF5/IMS, Zarr/OME-Zarr, FITS, MHA/MHD, NRRD, video, Gatan DM3/DM4, Zeiss ZVI and Leica LEI (pure Python), a bioio fallback, and every Java-only format through the Bio-Formats jars of an installed Fiji plus a JDK with no Python package (verified on this Mac with Bio-Formats 8.1.1 / OpenJDK 21: multi-series OME-TIFF, ICS, PNG – series listing, calibration, channel names, resolution levels, RGB; the `[bioformats]` extra is the alternative); needs a JDK and Fiji (or the extra) installed by the user, one `java` run (~1 s) per listing / read; the DM3 / ZVI / LEI readers were validated on synthetic files only |
| *Fiji* Bio-Formats Exporter | `file.bioformats_export` (File ▸ Export ▸ Bio-Formats Exporter…); `write_image(fmt="ome.zarr")` | ✅ | OME-TIFF, OME-Zarr or ImageJ TIFF with the exporter's options: compression (Uncompressed, LZW, zlib, JPEG, JPEG-2000, PackBits, Zstd), one file per Z section / timepoint / channel (`name_Z0_C0_T0.ome.tif`, optionally zero-padded), a C/Z/T save range and embedded OME-XML (channel names and colours, physical sizes, time increment, description) |
| *Fiji* Bio-Formats Remote Importer | `file.bioformats_remote_import`; `file.open_url` | ✅ | downloads the URL to a temporary file and opens it with the importer options (series, colour mode, view, autoscale, splitting, ranges, crop, metadata) |
| *Fiji* Bio-Formats Windowless Importer | `lumen.open(path)` in scripts; `file.bioformats_import` with explicit parameters (also `run("Bio-Formats Importer", "open=…")` in macros) | ✅ | every importer option is a parameter, so a recorded call runs without a dialog |
| *Fiji* Bio-Formats Macro Extensions / Stack Slicer / Plugins Configuration | `Ext.*` macro functions (`run("Bio-Formats Macro Extensions")`), `file.stack_slicer`, `file.bioformats_config` | ✅ | `Ext.setId` / `setSeries` / `getSeriesCount` / `getSeriesName` / `getSizeX` … `getSizeT` / `getImageCount` / `getMetadataValue` / `getPlaneTimingDeltaT` / `getIndex` / `getZCTCoords` / `openImage` / `openImagePlus` … over the native readers; Stack Slicer splits the hyperstack by channel / Z / T (stack order, keep original); Plugins Configuration = a table of every format the importer handles, its extensions, the Python reader behind it and whether it is installed, plus the formats that still need Java Bio-Formats |
| Close | `file.close` (⌘W) | ✅ | |
| Close All | `file.close_all` | ✅ | |
| Save | File ▸ Save (⌘S) | 🖥️ | ImageJ-compatible TIFF hyperstack with calibration, LUTs, ranges |
| Save As ▸ Tiff… | File ▸ Save As | ✅ | |
| Save As ▸ Gif… | `write_image(fmt="gif")` / Save As | ✅ | animated for stacks |
| Save As ▸ Jpeg… / PNG… / BMP… / PGM… | Save As | ✅ | plus WebP, PPM, TGA |
| Save As ▸ Text Image… | Save As `.txt` | ✅ | |
| Save As ▸ ZIP… | `file.export_zip` | ✅ | ImageJ TIFF inside a `.zip` |
| Save As ▸ Raw Data… | Save As `.raw` | ✅ | |
| Save As ▸ Image Sequence… | `file.export_sequence` | ✅ | format, name pattern, split axis |
| Save As ▸ AVI… | `file.export_video` | ✅ | mp4/avi/gif/mov/mkv/webm, fps, along Z or T, current display settings |
| Save As ▸ FITS… | Save As `.fits` | ✅ | |
| Save As ▸ LUT… | `file.export_lut` | ✅ | 768-byte ImageJ `.lut` of the current channel |
| Save As ▸ Selection… | `file.export_ij_rois` | ✅ | ImageJ `.roi` / `.zip`, also `file.export_rois_json`; read back with `file.import_ij_rois` / `file.import_rois_json` (ImageJ: File ▸ Open on a `.roi`) |
| Save As ▸ XY Coordinates… | `file.save_xy_coordinates` | ✅ | |
| Save As ▸ Results… | `file.export_results`; File ▸ Export ▸ Results as CSV | ✅ | |
| Save As ▸ Text… | `file.export_text` | ✅ | Results, a named table, the Log or the recorded macro |
| Revert | `file.revert` | ✅ | |
| Page Setup… / Print… | File ▸ Page Setup (Portrait / Landscape, Fit to Page / Fill Page Width / Actual Size), File ▸ Print… | 🖥️ | prints the current view with its name, size, calibration and a scale bar through the browser's print dialog (paper or PDF); the setup is remembered per browser |
| Quit | close the browser tab / stop `lumen` | 🖥️ | |

## Edit

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Undo | ⌘Z, History panel | 🖥️ | unlimited undo/redo per image with a clickable history list (ImageJ: one level) |
| Cut | `edit.cut` (⌘X) | ✅ | copies all channels of the current plane's selection and clears it to the background value |
| Copy | `edit.copy` (⌘C) | ✅ | internal clipboard |
| Copy to System | Edit ▸ Copy to System | 🖥️ | current view or the selection's bounding box to the system clipboard as PNG (browser clipboard API) |
| Paste | `edit.paste` (⌘V) | ✅ | at the selection or centred |
| Paste Control… | `edit.paste_control`, `edit.paste_with_mode` | ✅ | 14 transfer modes: Copy, Blend, Average, Difference, Transparent-zero, AND, OR, XOR, Add, Subtract, Multiply, Divide, Min, Max — set once or passed per paste |
| Clear | `edit.clear` | ✅ | |
| Clear Outside | `edit.clear_outside` | ✅ | |
| Fill | `edit.fill` (F) | ✅ | |
| Draw | `edit.draw` | ✅ | |
| Invert | `edit.invert` (⇧I) | ✅ | |
| Selection ▸ Select All | `selection.all` (A) | ✅ | |
| Selection ▸ Select None | `selection.none` (⇧A) | ✅ | |
| Selection ▸ Restore Selection | Edit ▸ Selection ▸ Restore Selection (⇧E) | 🖥️ | brings back the previous selection of the image; the ROI Manager keeps named ROIs |
| Selection ▸ Fit Spline | `selection.fit_spline` | ✅ | closed cubic spline through the polygon / freehand vertices (open spline for segmented lines), `points` per segment; `selection.interpolate(smooth)` still smooths in place |
| Selection ▸ Fit Circle | `selection.fit_circle` | ✅ | same area and centroid for area selections; least-squares circle through points / lines |
| Selection ▸ Fit Ellipse | `selection.fit_ellipse` | ✅ | ImageJ's EllipseFitter → 72-vertex EllipseRoi; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Selection ▸ Fit Rectangle | `selection.fit_rectangle` | ✅ | minimum-area rotated rectangle (rotating calipers); `selection.bounding_box` stays axis-aligned |
| Selection ▸ Interpolate | `selection.interpolate` | ✅ | |
| Selection ▸ Convex Hull | `selection.convex_hull` | ✅ | |
| Selection ▸ Make Inverse | `selection.inverse` | ✅ | |
| Selection ▸ Create Selection | `selection.create` | ✅ | from the current mask/threshold |
| Selection ▸ Create Mask | `selection.create_mask` | ✅ | always 255 inside / 0 outside |
| Selection ▸ Properties… | `selection.properties` | ✅ | name, stroke colour, line width, fill colour and C/Z/T position of the active selection |
| Selection ▸ Scale… | `selection.scale` | ✅ | |
| Selection ▸ Rotate… | `selection.rotate` | ✅ | |
| Selection ▸ Translate… | drag with the pointer tool (V) | 🖥️ | no numeric dx/dy dialog |
| Selection ▸ Enlarge… | `selection.enlarge` | ✅ | |
| Selection ▸ Make Band… | `selection.band` | ✅ | |
| Selection ▸ Specify… | `selection.specify` | ✅ | |
| Selection ▸ Straighten… | `stacks.straighten` | ✅ | spline-resampled, all planes |
| Selection ▸ To Bounding Box | `selection.bounding_box` | ✅ | |
| Selection ▸ Line to Area | `selection.line_to_area` | ✅ | |
| Selection ▸ Area to Line | `selection.area_to_line` | ✅ | |
| Selection ▸ Image to Selection… | `selection.image_to_selection` | ✅ | non-zero pixels of another open image (opacity, all channels) |
| Selection ▸ Add to Manager | `selection.add_to_manager` (T) | ✅ | |
| *Fiji* Selection ▸ Select Bounding Box (guess background) / Fit Circle to Image / Points from Mask / Make rectangular selection rounded | `selection.select_bounding_box`, `selection.fit_circle_to_image`, `selection.points_from_mask`, `selection.rounded_rect` | ✅ | background guessed from the border (tolerance); circle fitted to the thresholded / binary foreground (least squares on the outline or on all pixels, or from area and centroid), restricted to the selection; one point per mask pixel (step / max points, as a selection or into the ROI Manager); rounded rectangle with corner radius |
| *(Lumen extra)* | `selection.wand` | — | wand with tolerance / connectivity / add mode (ImageJ: Wand tool) |
| *(Lumen extras)* | `edit.flood_fill`, `edit.paint_stroke`; Brush (9) and Bucket (U) tools | — | ImageJ's Paintbrush / Flood Fill toolbar tools: brush strokes with the foreground value (`[` `]` size, ⌥-drag erases) and tolerance flood fill |
| Options ▸ Line Width… | `selection.line_width` | ✅ | |
| Options ▸ Input/Output… | `edit.options_io`; Preferences sheet (⌘,) | ✅ | JPEG quality, default save format |
| Options ▸ Fonts… | `edit.options_fonts` | ✅ | font family, size, antialiasing for text tools and labels; `font_size` is still a per-command parameter |
| Options ▸ Profile Plot Options… | `edit.options_plots` | ✅ | plot size, fixed y-range, interpolation; `analyze.plot_profile(vertical, all_channels)` |
| Options ▸ Rounded Rect Tool… | `edit.options_rounded_rect` | ✅ | corner radius, stroke width |
| Options ▸ Arrow Tool… | `edit.options_arrow_tool` | ✅ | style, head size, width, double-headed defaults for the Arrow tool (J); `overlay.arrow` takes them per call |
| Options ▸ Point Tool… | `edit.options_point_tool` | ✅ | marker size, colour, type |
| Options ▸ Wand Tool… | `selection.wand(tolerance, connectivity)` | ✅ | per call instead of a global option |
| Options ▸ Colors… | `edit.set_colors`, `edit.set_foreground`, `edit.set_background`; `edit.options_roi_defaults` | ✅ | foreground / background values (both, or one at a time from the value picker) and the selection colour / line width |
| Options ▸ Appearance… | `edit.options_appearance`; View ▸ Toggle Theme | ✅ | interpolate zoomed images, antialiasing, stack auto-contrast; light / dark theme |
| Options ▸ Conversions… | `scale` parameter of `type.8bit` / `type.16bit` | ✅ | per call |
| Options ▸ Plots… / Roi Defaults… / Startup… / Misc… / Fresh Start / Reset… | `edit.options_plots`, `edit.options_roi_defaults`, `macros.edit_startup`, `edit.options_misc`, `help.reset`, `edit.options_reset`; `edit.options` = the whole Preferences sheet (⌘,) | ✅ | Startup = `~/.lumen/startup.py` (Python API) then `startup.ijm` (ImageJ macro), both run at start-up and shown by Edit Startup Script; Misc = divide-by-zero value, shortcut modifier, beep, paste mode; Fresh Start = `help.reset(close_images=True)` (Results, ROI Manager, tables, plots, thresholds, recorder, Log, images); Reset restores every preference |
| Options ▸ Memory & Threads… / Proxy Settings… / Compiler… / DICOM… | `edit.options_memory`, `edit.options_proxy`, `help.compiler_info`, `edit.options_dicom` | ✅ | undo memory budget per image and the numpy / scipy thread count (threadpoolctl), optional garbage collection; HTTP(S) proxy for sample downloads and URLs; Compiler… lists the Python plugin folders and files (plugins are `@op` Python files hot-reloaded by `plugins.reload` — nothing to compile); DICOM: open as 32-bit, ignore rescale slope / intercept, calibrate, window / level on open — honoured by the DICOM reader |
| *Fiji* Options ▸ Look and Feel… | View ▸ Toggle Theme | 🖥️ | |

## Image

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Type ▸ 8-bit | `type.8bit` | ✅ | `scale` = "Scale when converting"; RGB → luminance |
| Type ▸ 16-bit | `type.16bit` | ✅ | |
| Type ▸ 32-bit | `type.32bit` | ✅ | |
| Type ▸ 8-bit Color | `type.8bit_color` | ✅ | median-cut palette of N colours, stored as an indexed LUT |
| Type ▸ RGB Color | `type.rgb` | ✅ | |
| Type ▸ RGB Stack / HSB Stack / Lab Stack | `type.rgb_stack`, `type.hsb_stack`, `type.lab_stack` | ✅ | |
| Adjust ▸ Brightness/Contrast… | Display section (⇧⌘C): histogram-backed per-channel min/max, gamma, Auto, Reset, "Same range for all"; `adjust.brightness_contrast_apply` = the *Apply* button | 🖥️ | display is non-destructive and per channel; Apply is an explicit, undoable command |
| Adjust ▸ Window/Level… | `adjust.window_level` | ✅ | |
| Adjust ▸ Color Balance… | Display section, per channel | 🖥️ | for RGB images: convert to RGB Stack to adjust channels separately |
| Adjust ▸ Threshold… | Threshold section (⇧⌘T): 16 auto methods, red overlay, dark background, Auto / Apply / Create selection / Measure / Analyze Particles; `threshold.compute`, `threshold.set`, `threshold.apply`, `threshold.histogram` | ✅ | the 16 auto methods (Default … Yen) are exact ports of ImageJ's AutoThresholder / setAutoThreshold: levels on 8/16/32-bit, open-ended ranges, ImageJ's +1 rule — pixel-exact vs ImageJ 1.54p (cross-validated); the panel has no "Over/Under" / "B&W" display modes and no stack histogram option (`analyze.stack_histogram`) |
| Adjust ▸ Color Threshold… | `adjust.color_threshold` | ✅ | HSB / RGB / Lab / YUV, pass/stop filters, Mask / Selection / Overlay output |
| Adjust ▸ Size… | `image.scale(width, height)` | ✅ | |
| Adjust ▸ Canvas Size… | `image.canvas_size` | ✅ | |
| Adjust ▸ Line Width… | `selection.line_width` | ✅ | |
| Adjust ▸ Coordinates… | `adjust.coordinates` | ✅ | origin x/y/z |
| *Fiji* Adjust ▸ Bleach Correction | `adjust.bleach_correction` | ✅ | CorrectBleach's three methods – Simple Ratio (with background), Exponential Fit (ImageJ CurveFitter "Exponential with Offset", frames scaled by fit(0)/fit(t), half-life reported) and Histogram Matching (to frame 1) – on the T axis (a plain Z stack is treated as time, like the plugin), per channel, optional area ROI for the means; new `Bleach_corrected_<title>` image, a per-frame table and a mean-vs-frame plot; `run("Bleach Correction", "correction=[Simple Ratio] background=0")` maps 1:1 |
| *Fiji* Adjust ▸ Auto Threshold | `threshold.apply(method=…, add_one=…)` | ✅ | same 16 methods (exact ports, cross-validated vs ImageJ 1.54p); per-plane or stack histogram; whole stack or only the current plane (`only_current_plane`, the other planes are kept) |
| *Fiji* Adjust ▸ Auto Local Threshold | `threshold.local` | ✅ | all nine Fiji methods (Bernsen, Contrast, Mean, Median, MidGrey, Niblack, Otsu, Phansalkar, Sauvola) plus Gaussian; radius, parameter 1, white objects |
| *Fiji* Adjust ▸ Auto Crop | `image.auto_crop`, `image.auto_crop_selection` | ✅ | crops (or selects) the bounding box of everything that differs from the border background; tolerance, whole stack |
| Show Info… | Image Info section (Name, Path, Type, Dimensions, Pixel size, Frame interval, Memory, metadata) | 🖥️ | |
| Properties… | `image.properties` | ✅ | pixel size, voxel depth, unit, frame interval |
| Color ▸ Split Channels | `color.split` | ✅ | |
| Color ▸ Merge Channels… | `color.merge` | ✅ | up to 7 channels, composite or RGB |
| Color ▸ Arrange Channels… | `color.arrange` | ✅ | |
| Color ▸ Invert LUTs | `color.invert_lut` | ✅ | |
| Color ▸ Channels Tool… | channel chips + Mode (Composite / Single / Grayscale) in the Display section | 🖥️ | |
| Color ▸ Stack to RGB | `color.bake_lut` | ✅ | renders the composite with the current LUTs/ranges |
| Color ▸ Make Composite | Display mode = Composite (default for multi-channel) | 🖥️ | |
| Color ▸ Show LUT / Display LUTs / Edit LUT… | `color.show_lut`, `color.display_luts`, `color.edit_lut` | ✅ | LUT card with display range and R/G/B curves; montage of every LUT (built-in + imported); new LUT from an RGB text table or a two-colour gradient, registered and applied |
| Color ▸ Color Picker… | Color Picker popover (⇧K, anchored to the foreground / background swatches of the tool rail): HSB square + hue strip, RGB and hex inputs and presets for RGB images, value + slider over the bit depth for grayscale; `edit.set_foreground`, `edit.set_background`, `edit.set_colors`; Value picker tool (I) reads values from the image | ✅ | colour and value stay in sync (colour ↔ luminance); Fill / Clear / Draw and the brush / bucket use the picked colour per channel on RGB images |
| *Fiji* Color ▸ Retinex / Replace Red with Magenta | `process.retinex`; Magenta LUT per channel | ✅ | multi-scale retinex with colour restoration (level, scale, scale division, dynamic) |
| *(Lumen extra)* | `color.channel_names` | — | |
| Stacks ▸ Add Slice / Delete Slice | `stacks.add_slice`, `stacks.delete_slice` | ✅ | Z, T or C (adding a channel inserts a blank channel with its own LUT and display range) |
| Stacks ▸ Next Slice / Previous Slice / Set Slice… | Z and T sliders, `,` `.` `<` `>` keys, ⇧-wheel | 🖥️ | |
| Stacks ▸ Images to Stack / Stack to Images | `stacks.images_to_stack`, `stacks.to_images` | ✅ | |
| Stacks ▸ Make Montage… | `stacks.montage` | ✅ | Z/T/C, scale, border |
| Stacks ▸ Reslice… | `stacks.reslice`, `stacks.reslice_line` | ✅ | Top (XZ) / Left (YZ), output spacing, slice count, rotate 90°, interpolate; with a line or polyline selection the slice(s) along the line (parallel slices for straight lines) |
| Stacks ▸ Orthogonal Views | Orthogonal Views section (live) + `stacks.orthogonal_views` | ✅ | |
| Stacks ▸ Z Project… | `stacks.z_project` | ✅ | Max/Min/Average/Sum/SD/Median, start/stop, Z or T axis |
| Stacks ▸ 3D Project… | `stacks.project3d` | ✅ | Brightest/Nearest/Mean, axis, angle step, thresholds, surface and interior depth cueing |
| Stacks ▸ Plot Z-axis Profile | `stacks.z_profile` | ✅ | Z or T |
| Stacks ▸ Measure Stack… | `analyze.measure_stack` | ✅ | |
| Stacks ▸ Label… | `stacks.label_slices` | ✅ | format string, start / interval, position, colour, font; also `overlay.time_stamp` (start time, interval) |
| Stacks ▸ Statistics | `stacks.statistics`; `analyze.stack_histogram` | ✅ | voxel count, mean, SD, min, max, mode, median over the whole stack per channel → Stack Statistics table |
| Stacks ▸ Tools ▸ Combine… | `stacks.combine` | ✅ | |
| Stacks ▸ Tools ▸ Concatenate… | `stacks.concatenate` | ✅ | |
| Stacks ▸ Tools ▸ Reduce… | `stacks.reduce` | ✅ | |
| Stacks ▸ Tools ▸ Reverse | `stacks.reverse` | ✅ | |
| Stacks ▸ Tools ▸ Insert… | `stacks.insert` | ✅ | |
| Stacks ▸ Tools ▸ Montage to Stack… | `stacks.montage_to_stack` | ✅ | columns × rows tiles, border |
| Stacks ▸ Tools ▸ Make Substack… | `stacks.substack` | ✅ | |
| Stacks ▸ Tools ▸ Grouped Z Project… | `stacks.grouped_z_project` | ✅ | |
| Stacks ▸ Tools ▸ Remove Slice Labels | `stacks.remove_slice_labels`; `stacks.set_slice_label`, `stacks.slice_labels_to_results` | ✅ | per-plane labels live in the image metadata and are used by Make Montage |
| Stacks ▸ Tools ▸ Start/Stop Animation, Animation Options… | Play button in the dimension bar (Z or T) | 🖥️ | fixed 120 ms per frame; no frame-rate / loop-back dialog — `file.export_video(fps=…)` for a movie at a chosen rate |
| *Fiji* Stacks ▸ Dynamic Reslice / Radial Reslice | Orthogonal Views section (live); `stacks.radial_reslice` | ✅ | radial reslice around a point (or the image centre / a line's end): number of angles, total rotation, length |
| Hyperstacks ▸ New Hyperstack… | `file.new` | ✅ | |
| Hyperstacks ▸ Stack to Hyperstack… | `stacks.to_hyperstack` | ✅ | all six dimension orders |
| Hyperstacks ▸ Hyperstack to Stack | `stacks.hyperstack_to_stack` | ✅ | ImageJ order (channels fastest, then slices, then frames) with `c:… z:… t:…` slice labels |
| Hyperstacks ▸ Reduce Dimensionality… | `stacks.reduce_dimensionality` | ✅ | keep channels / slices / frames, keep source |
| Hyperstacks ▸ Make Subset… | `stacks.make_subset` | ✅ | channel, slice and frame ranges applied together |
| Hyperstacks ▸ Channels Tool… | Display section | 🖥️ | |
| *Fiji* Hyperstacks ▸ Re-order Hyperstack… | `stacks.reorder` | ✅ | |
| Crop | `image.crop` (⇧X) | ✅ | |
| Duplicate… | `image.duplicate` (⇧D) | ✅ | whole stack / plane / channel / slice, crop to selection |
| Rename… | `image.rename` | ✅ | |
| Scale… | `image.scale` | ✅ | x/y/z factors or target size, None/Bilinear/Bicubic, average when downsizing |
| Transform ▸ Flip Horizontally / Vertically / Flip Z | `image.flip_h`, `image.flip_v`, `image.flip_z` | ✅ | |
| Transform ▸ Rotate 90 Degrees Right / Left | `image.rotate90` | ✅ | |
| Transform ▸ Rotate… | `image.rotate` | ✅ | arbitrary angle, enlarge |
| Transform ▸ Translate… | `image.translate` | ✅ | |
| Transform ▸ Bin… | `image.bin` | ✅ | |
| Transform ▸ Image to Results / Results to Image | `image.to_results`, `image.from_results` | ✅ | current plane (or the selection's bounding box) ↔ numeric Results columns, 32-bit |
| Zoom ▸ In / Out / Original Scale / View 100% / To Selection / Scale to Fit / Set… / Maximize | View menu: Zoom In (+), Zoom Out (−), Fit to Window (0), Actual Size, Zoom to Selection; wheel zooms at the cursor; Zoom tool (Z) | 🖥️ | no numeric "Set…" zoom |
| Overlay ▸ Add Selection… | `overlay.add` (B) | ✅ | colour, stroke width |
| Overlay ▸ Add Image… | `overlay.add_image`, `overlay.burn_image` | ✅ | another open image as a non-destructive overlay item at x/y with opacity and scale, drawn in the viewport; burn-in to an RGB copy |
| Overlay ▸ Hide / Show / Toggle Overlay | View ▸ Show Overlay (⇧O) | 🖥️ | |
| Overlay ▸ From ROI Manager / To ROI Manager | `overlay.from_manager`, `overlay.to_roi_manager` | ✅ | |
| Overlay ▸ Remove Overlay | `overlay.remove` | ✅ | |
| Overlay ▸ List Elements | `overlay.list` | ✅ | |
| Overlay ▸ Flatten | `overlay.flatten`, `overlay.burn_in` | ✅ | burn-in for all planes |
| Overlay ▸ Labels… | `analyze.label` | ✅ | index / name / measurement / value labels for the overlay's or the ROI Manager's items at the centroid or bounding box with ImageJ's options (use names, font size, bold, colour, background); `run("Labels...", "color=… font=… show use bold draw")` maps 1:1 |
| Overlay ▸ Measure Overlay | `overlay.measure` | ✅ | every ROI-shaped overlay item (areas, lines, points) measured with the Set Measurements configuration, each on its own plane when pinned to one; one Results row per item labelled image:name |
| Overlay ▸ Overlay Options… | `edit.options_overlay` (Edit ▸ Options ▸ Overlay…); `edit.options_roi_defaults`; `selection.properties` | ✅ | default stroke width and fill colour of overlay items plus the label switches (show labels, use names, font size, colour, background), honoured by the viewport and by Flatten / burn-in; per-ROI colour / stroke / fill through Properties |
| *(Lumen extras)* | `overlay.text`, `overlay.arrow`, `overlay.scale_bar`, `overlay.calibration_bar`, `overlay.time_stamp`, `overlay.add_item` / `overlay.remove_item` / `overlay.clear_type`; Text (Y) and Arrow (J) tools | — | ImageJ keeps Scale Bar / Calibration Bar under Analyze ▸ Tools |
| Lookup Tables ▸ Invert LUT | `color.invert_lut`; "Invert LUT" toggle in Display | ✅ | |
| Lookup Tables ▸ Apply LUT | `type.apply_lut` | ✅ | bakes the display range into the pixels (ContrastAdjuster.apply semantics); shown in the UI as *Apply Display Range to Pixels (Apply LUT)* so it cannot be confused with *Set LUT* |
| Lookup Tables ▸ Fire, Grays, Ice, Spectrum, 3-3-2 RGB, Red, Green, Blue, Cyan, Magenta, Yellow, Red/Green | LUT drop-down per channel; `color.lut` | ✅ | all 12 ImageJ built-ins |
| *Fiji* luts folder (glasbey, 16 colors, phase, thermal, HiLo, mpl-viridis…, physics, royal, sepia, …) | 60 LUTs (`lumen.core.luts`; the 12 ImageJ built-ins reproduce LutLoader exactly and all 36 Fiji `.lut` files are byte-identical): the 12 ImageJ built-ins plus Fiji's `luts/` set — 16 colors, 5 ramps, 6 shades, blue orange icb, brgbcmyw, cool, edges, gem, glow, physics, smart, Royal, Sepia, Thal, Thallium, Union Jack, ICA / ICA2 / ICA3, HiLo, Phase, Thermal, Glasbey (+ inverted, on dark), Red / Orange / Yellow / Cyan / Magenta Hot, Green Fire Blue, Rainbow RGB, mpl-viridis / magma / plasma / inferno — plus Orange, Viridis, Magma, Inferno, Plasma, Cividis, Turbo, Jet, Hot, Cool, Rainbow and Random; `file.import_lut` / `file.import_luts_folder` for any other `.lut` | ✅ | LUT drop-down per channel, `color.lut`, `color.display_luts` montage |

## Process

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Smooth | `process.smooth` (⇧S) | ✅ | 3×3 mean; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Sharpen | `process.sharpen` | ✅ | pixel-exact vs ImageJ 1.54p (cross-validated) |
| Find Edges | `process.find_edges` | ✅ | Sobel; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Find Maxima… | `analyze.find_maxima` | ✅ | full MaximumFinder port: prominence, strict, exclude edges, light background, above lower threshold; outputs Single Points / Maxima Within Tolerance / Segmented Particles / Count / Point Selection / List — pixel-exact vs ImageJ 1.54p (cross-validated) |
| Enhance Contrast… | `process.enhance_contrast` | ✅ | saturated %, normalize, equalize, process all, use stack histogram; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Noise ▸ Add Noise / Add Specified Noise… / Salt and Pepper / Despeckle / Remove Outliers… / Remove NaNs… | `noise.add`, `noise.add_specified`, `noise.salt_pepper`, `noise.despeckle`, `noise.remove_outliers`, `noise.remove_nans` | ✅ | Despeckle / Remove Outliers / Remove NaNs use ImageJ's rank-filter kernel (dx²+dy² ≤ r²+1, the kernel cross-validated for the Filters) |
| *Fiji* Noise ▸ ROF Denoise | `filters.rof_denoise` | ✅ | Rudin–Osher–Fatemi total-variation denoising (theta, iterations); `filters.nlmeans`, `filters.bilateral` as alternatives |
| Shadows ▸ North … Northwest | `process.shadows(direction)` | ✅ | all eight directions; no Shadows Demo |
| Binary ▸ Make Binary | `binary.make` | ✅ | uses the current threshold, an auto method or ImageJ's legacy ISODATA when no threshold is set; `dark_background` replaces the global Black-background option — pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Convert to Mask | `threshold.apply` | ✅ | mask is always 255 = foreground; whole stack (`calculate` = per-slice thresholds) or only the current plane with the other planes kept (`only_current_plane`) — pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Erode / Dilate / Open / Close | `binary.erode`, `binary.dilate`, `binary.open`, `binary.close` | ✅ | `iterations`, `count` and `pad_edges` per call (ImageJ's Options ▸ iterations/count/pad edges) — pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Median | `binary.median` | ✅ | majority filter over a circular neighbourhood (radius) |
| Binary ▸ Outline | `binary.outline` | ✅ | ImageJ mode (8-connected 1-px boundary); pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Fill Holes | `binary.fill_holes` | ✅ | pixel-exact vs ImageJ 1.54p (cross-validated on single images; the stack case in docs/CROSSVALIDATION.md is an ImageJ batch-mode artefact) |
| Binary ▸ Skeletonize | `binary.skeletonize` | ✅ | `method=ImageJ` is ImageJ's lookup-table thinning (pixel-exact vs ImageJ 1.54p, cross-validated); Zhang-Suen, Lee, Guo-Hall or medial axis as alternatives |
| Binary ▸ Distance Map | `binary.distance_map` | ✅ | `output` = Overwrite (8-bit EDM, ImageJ's default) / 8-bit / 16-bit / 32-bit; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Ultimate Points | `binary.ultimate_points` | ✅ | overwrites the image with the UEP values (ImageJ's default) or a point selection / 32-bit UEP image (`output`); pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Watershed | `binary.watershed` | ✅ | `method` = ImageJ EDM/UEP (the default; pixel-exact vs ImageJ 1.54p, cross-validated) or distance-transform watershed seeded by h-maxima (`tolerance`, `min_distance`) |
| Binary ▸ Voronoi | `binary.voronoi` | ✅ | lines carry the EDM value like ImageJ; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Binary ▸ Dither… | `binary.dither` | ✅ | Floyd–Steinberg error diffusion |
| Binary ▸ Options… | per-command parameters; no Black-background switch | 🖥️ | see "Deliberately different" |
| *(Lumen extras)* | `binary.remove_small`, `binary.label`, `binary.convex_hull` | — | MorphoLibJ-style helpers |
| Math ▸ Add… Subtract… Multiply… Divide… AND… OR… XOR… Min… Max… Gamma… Set… Log Exp Square Square Root Reciprocal NaN Background Abs | `math.add`, `math.subtract`, `math.multiply`, `math.divide`, `math.and`, `math.or`, `math.xor`, `math.min`, `math.max`, `math.gamma`, `math.set`, `math.log`, `math.exp`, `math.square`, `math.sqrt`, `math.reciprocal`, `math.nan_background`, `math.abs` | ✅ | 18 commands, per plane or whole stack; pixel-exact vs ImageJ 1.54p on 8/16/32-bit (cross-validated: ImageJ's truncation and 16-bit min–max ranges) |
| Math ▸ Macro… | `math.macro`; `math.expression` | ✅ | ImageJ macro code with v, x, y, z, w, h, a, d; `math.expression` for numpy syntax|
| FFT ▸ FFT | `process.fft`, `process.fft_options` | ✅ | 8-bit power spectrum (padded to a power of two like ImageJ) with the complex transform stored for Inverse FFT; options: window function, raw power spectrum, complex output — pixel-exact vs ImageJ 1.54p (cross-validated) |
| FFT ▸ Inverse FFT / Redisplay Power Spectrum / FFT Options… / Custom Filter… / FD Math… / Swap Quadrants / Make Circular Selection… | `process.inverse_fft`, `process.fft_redisplay_power_spectrum`, `process.fft_options`, `process.fft_custom_filter`, `process.fd_math`, `process.fft_swap_quadrants`; `selection.specify(oval, centered)` | ✅ | inverse FFT honours black (block) / white (pass) regions painted on the spectrum, like ImageJ (FFT → Inverse FFT round trip pixel-exact vs ImageJ 1.54p, cross-validated); FD Math = correlate / convolve / deconvolve with regularisation; custom filter = 255-pass / 0-block image on every plane |
| FFT ▸ Bandpass Filter… | `process.fft_bandpass` | ✅ | large/small structures, stripe suppression, tolerance, autoscale, saturate; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Filters ▸ Convolve… | `filters.convolve` | ✅ | text kernel, normalize; pixel-exact vs ImageJ 1.54p (cross-validated) |
| Filters ▸ Gaussian Blur… / Median… / Mean… / Minimum… / Maximum… / Unsharp Mask… / Variance… / Top Hat… | `filters.gaussian`, `filters.median`, `filters.mean`, `filters.minimum`, `filters.maximum`, `filters.unsharp`, `filters.variance`, `filters.tophat` | ✅ | ImageJ's circular rank-filter kernel (dx²+dy² ≤ r²+1), Gaussian kernel accuracy, Gaussian *Scaled units* (sigma in calibrated units, per axis) and Top Hat "don't subtract"; pixel-exact vs ImageJ 1.54p on 8/16/32-bit (cross-validated, scaled units included) |
| Filters ▸ Gaussian Blur 3D… | `filters.gaussian3d` | ✅ | |
| Filters ▸ Median 3D… / Mean 3D… / Minimum 3D… / Maximum 3D… / Variance 3D… | `filters.rank3d(filter=…)` | ✅ | one command with a `filter` choice |
| Filters ▸ Show Circular Masks… | `filters.show_circular_masks` | ✅ | kernels for radii 0.5 … 10 as a stack |
| *Fiji* Enhance Local Contrast (CLAHE) | `process.clahe` | ✅ | block size, clip limit, bins |
| *(Lumen extras)* | `filters.dog`, `process.laplacian`, `filters.bilateral`, `filters.nlmeans`, `filters.frangi` | — | |
| Batch ▸ Measure… | `batch.measure` | ✅ | folder + pattern, optional threshold / dark background, all planes; rows appended with a File column using the current Set Measurements |
| Batch ▸ Convert… | `batch.process_folder(output_format=…)` | ✅ | |
| Batch ▸ Macro… | `batch.apply_recorded_macro`, `batch.run_script_on_open_images` | ✅ | replays the recorder's Python instead of an IJ macro |
| Batch ▸ Virtual Stack… | `batch.virtual_stack` | ✅ | folder + pattern, recursive: each file is opened, the script run, the result saved (tif / ome.tif / png / jpg / npy) and closed before the next one is read, so the stack is never held in memory; Results rows kept per file; continue on error |
| Image Calculator… | `process.calculator` | ✅ | Add/Subtract/Multiply/Divide/AND/OR/XOR/Min/Max/Average/Difference/Copy/Transparent-zero, 32-bit result |
| Subtract Background… | `process.subtract_background` | ✅ | rolling ball, sliding paraboloid or fast top-hat; light background, create background, smoothing; pixel-exact vs ImageJ 1.54p on 8/16/32-bit (cross-validated) |
| Repeat Command | Edit ▸ Repeat Last Command (⇧R) | 🖥️ | |
| *Fiji* Plugins ▸ Process ▸ Exact (Signed) Euclidean Distance Transform (3D) | `binary.signed_distance_map`, `binary.distance_map_3d` | ✅ | signed EDT (positive inside, negative outside) and 3-D EDT over X/Y/Z in calibrated units with anisotropic voxels; `binary.distance_map` stays 2-D per plane |
| *Fiji* Plugins ▸ Process ▸ Smooth (3D) | `filters.gaussian3d` | ✅ | |
| *Fiji* Plugins ▸ Integral Image Filters ▸ Mean / Sample Variance / Standard Deviation / Remove Outliers / Normalize Local Contrast | `filters.normalize_local_contrast`; `filters.mean`, `filters.variance` (+ `math.sqrt`), `noise.remove_outliers` | ✅ | Normalize Local Contrast with block radius x/y, standard deviations, center, stretch; the others via the ordinary (non integral-image) filters |

## Analyze

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Measure | `analyze.measure` (M) | ✅ | 19 measurement sets incl. Feret, shape descriptors, fit ellipse, skewness/kurtosis, stack position — ImageJ-exact Analyzer / ImageStatistics / EllipseFitter / Feret / perimeter / line / point / limit-to-threshold / invert-Y (cross-validated vs ImageJ 1.54p) |
| Analyze Particles… | `analyze.particles` | ✅ | size/circularity, Show = Nothing/Outlines/Bare Outlines/Masks ("Mask of …")/Count Masks (16-bit)/Labels/Ellipses, display/clear/summarize/add to manager/exclude edges/include holes, all planes, redirect (`intensity_image`), pixel units, record starts, in situ show, composite ROIs — ImageJ's ParticleAnalyzer scan and measurements (cross-validated vs ImageJ 1.54p); Overlay Outlines / Masks via `overlay.from_manager` |
| Summarize | `analyze.summarize` | ✅ | |
| Distribution… | `analyze.distribution` | ✅ | |
| Label | `analyze.label` | ✅ | |
| Clear Results | `analyze.clear_results` | ✅ | |
| Set Measurements… | `analyze.set_measurements` | ✅ | the 18 ImageJ check-boxes + limit to threshold, decimals, display label, invert Y, scientific notation, add to overlay, NaN empty cells; Redirect is per call (`intensity_image`) |
| Set Scale… | `image.properties`, `image.set_scale_from_line` | ✅ | no "Global" flag |
| Calibrate… | `analyze.calibrate`, `analyze.calibrated_values`, `analyze.measure_calibrated` | ✅ | ImageJ's function list: straight line, 2nd / 3rd polynomial, exponential (+ offset, recovery), power, log, Rodbard, Rodbard (NIH Image), gamma variate, Chapman-Richards, uncalibrated OD |
| Histogram | `analyze.histogram` (H) + live Histogram section | ✅ | ImageJ's bin assignment on 8/16/32-bit (cross-validated vs ImageJ 1.54p) |
| Plot Profile | `analyze.plot_profile` (K) + Plot section | ✅ | line or rectangle, vertical, all channels; `getProfile()` = ImageJ's ProfilePlot incl. wide lines (cross-validated vs ImageJ 1.54p) |
| Surface Plot… | `analyze.surface_plot` | ✅ | |
| Gels ▸ Select First Lane / Select Next Lane / Plot Lanes / Re-plot Lanes / Reset / Label Peaks / Gel Analyzer Options… | `analyze.gel_select_lane`, `analyze.gel_plot_lanes`, `analyze.gel_label_peaks`; `analyze.gel_lanes` | ✅ | lanes are numbered rectangles in the ROI Manager (Select First / Next Lane); stacked lane profiles (Plot Lanes / Re-plot, optional plot image); bands integrated between baseline minima with value labels (Label Peaks); Reset = delete the lane ROIs; orientation / dark bands / smoothing are parameters |
| Tools ▸ Save XY Coordinates… | `file.save_xy_coordinates` | ✅ | |
| Tools ▸ Fractal Box Count… | `analyze.fractal_box_count` | ✅ | |
| Tools ▸ Analyze Line Graph | `analyze.line_graph` | ✅ | digitises the curves of a plot image grouped by colour (tolerance, max curves, invert Y, axis ranges) → plot + Results |
| Tools ▸ Curve Fitting… | `analyze.curve_fit` | ✅ | ImageJ's CurveFitter list: straight line, polynomials 2nd–8th, exponential (+ linear regression, with offset, recovery), power (+ linear regression), log, log2, Gaussian (± offset), sigmoid (Boltzmann), error function, Rodbard and Rodbard (NIH Image), rational, Michaelis-Menten, gamma variate, Chapman-Richards and a user-defined equation `y = f(x, a…f)`; data from two Results columns, the last plot or typed values; initial parameters, restarts, residual plot (`lumen.core.fitting`) |
| Tools ▸ ROI Manager… | ROI Manager section: Add (T), Delete, Rename, Measure, Measure selected, Combine AND/OR/XOR, Show all, To overlay, Import / Export (`file.import_ij_rois`, `file.export_ij_rois`, `file.import_rois_json`, `file.export_rois_json`); `analyze.multi_measure`; `selection.properties` (= More ▸ Properties) | 🖥️ | missing More ▸ Split, Sort, Fill / Draw (use `edit.fill` / `edit.draw`), Remove Slice Info, Multi Plot |
| Tools ▸ Scale Bar… | `overlay.scale_bar` | ✅ | |
| Tools ▸ Calibration Bar… | `overlay.calibration_bar` | ✅ | |
| Tools ▸ Synchronize Windows | View ▸ Synchronize Views | 🖥️ | |
| Tools ▸ Grid… | `overlay.grid` | ✅ | lines / horizontal lines / crosses / points / combined; area per point or spacing, random offset, colour |
| Tools ▸ Compare Images… | `analyze.compare_images` | ✅ | Pearson, RMSE, MAE, PSNR, SSIM of two images → Results + Compare Images table; optional 32-bit difference image |
| *Fiji* Analyze ▸ Skeleton ▸ Analyze Skeleton (2D/3D) | `analyze.skeleton`, `analyze.skeleton_longest_path` | ✅ | 2-D per plane or 3-D on the whole Z stack: tagged skeleton (end = 1, slab = 2, junction = 3), per-skeleton branches / junctions / end points / branch lengths / longest shortest path plus a Branches table, end-branch pruning by calibrated length, cycle pruning, longest-branch and longest-shortest-path masks; optional skeletonize-first |
| *Fiji* Analyze ▸ Colocalization ▸ Coloc 2 / Colocalization Threshold / Colocalization Test | `analyze.coloc_costes`, `analyze.colocalization` | ✅ | Coloc 2 equivalent within the selection: Costes auto-threshold (orthogonal regression), Pearson whole / above / below thresholds, thresholded Manders tM1/tM2, Spearman, Costes block-scrambling significance test (p-value, block size, iterations, seed) and a 2-D scatter-histogram image; `analyze.colocalization` = manual thresholds (Colocalization Threshold); Li's ICQ and Kendall's tau are not computed |
| *Fiji* Analyze ▸ 3D Objects Counter / 3D OC Options | `segment3d.label_3d`, `segment3d.measure_labels_3d`, `segment3d.filter_labels_3d` | ✅ | 6 / 18 / 26-connected labelling of a binary stack or the current threshold (= the Objects map, per frame), per-object voxel count, calibrated volume, centroid, bounding box, intensity statistics, surface area (marching cubes or Crofton) and sphericity → Results + a Labels 3D table; volume, edge and mean-intensity filters; no surface / centroid / centre-of-mass map images |
| *Fiji* Analyze ▸ Directionality / Shape Index Map | `analyze.directionality`, `filters.shape_index` | ✅ | local gradient orientation (structure tensor) or Fourier components, bins / bin start, Gaussian fit → Direction, Dispersion, Amount, Goodness + plot; Koenderink shape index at scale sigma |
| *(Lumen extras)* | `analyze.angle`, `analyze.label_objects` (measure labels), `analyze.stack_histogram` | — | |

## Plugins

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Macros ▸ Record… | Script & Recorder section (always on); `macros.record_to_file` | 🖥️ | records Python `lumen.run(...)` lines with image targets, undo, display changes and clears, per browser tab; exports as Python, JavaScript or an ImageJ macro (`→ IJM`, lines without an ImageJ equivalent become `//` comments that say why); Save Recorded Macro… writes a script that `macros.run_file` replays or `macros.install` turns into a command |
| Macros ▸ Run… | `macros.run_file`; `script.run_ij_macro` | ✅ | runs a Python script (lumen API, numpy / scipy / skimage) or an ImageJ-1 `.ijm` macro file through the compatibility layer (`lumen.core.ij_compat`): 323 `run()` rows mapped (423 command names including aliases), 171 macro functions, 34 dialog / window commands accepted as no-ops, none declared unsupported — full table in the section below |
| Macros ▸ Install… / Edit… / Startup Macros… | `macros.install`, `macros.edit_startup`; `~/.lumen/plugins/*.py` (`@op` commands) | ✅ | Install… copies a `.py` plugin, an `.ijm` macro or a plain Python script into the plugins folder and registers it as a command at once (menu of your choice); Startup Macros… shows `~/.lumen/startup.py` and `startup.ijm`, which run at start-up; Edit… = the reported path in your editor or the Script console (no built-in text editor) |
| Macros ▸ Interactive Interpreter… | Python console (Script section) | ✅ | |
| Shortcuts ▸ Add Shortcut… / Add Shortcut by Name… / Remove Shortcut… | Plugins ▸ Shortcuts ▸ Add Shortcut… / Remove Shortcut… (Shortcuts dialog) | 🖥️ | bind any key combination to any command, or manage custom bindings; defaults still come from the `@op` decorator |
| Shortcuts ▸ List Shortcuts / List Commands | Help ▸ Keyboard Shortcuts (?); `utilities.list_commands`, `script.ij_command_list`, ⌘K palette | ✅ | every registered command with id, menu, kind, shortcut, parameters and description as a table |
| Utilities ▸ Find Commands… / Commands… | ⌘K command palette; `utilities.command_finder`, `help.search_commands` | ✅ | fuzzy search over name, id, keywords, category, description and shortcut (subsequences and misspellings match) |
| Utilities ▸ Control Panel… | menu bar | 🖥️ | |
| Utilities ▸ Search… | `help.search_files` (Plugins ▸ Utilities ▸ Search…); Search ImageJ Wiki… / Search image.sc Forum… (also offered by the ⌘K palette); `help.search_commands` | ✅ | greps the plugins / macros / scripts folders, the open images' names and metadata, the Log and the recorded script for a string (case option) → a Search Files table of where / file / line / text, like ImageJ's Search…; command search and the web search (Fiji's search bar) stay separate |
| Utilities ▸ Monitor Memory… | Plugins ▸ Utilities ▸ Monitor Memory (live sparkline + status chip); Image Info ▸ Memory | 🖥️ | memory held by open images and undo history |
| Utilities ▸ Capture Screen / Capture Delayed… / Capture Image | Plugins ▸ Utilities ▸ Capture Screen (⇧G) / Capture Delayed… / Capture Image (⇧H); `overlay.flatten`, `color.bake_lut` | ✅ | Capture Screen grabs one frame of the screen, window or tab you pick in the browser's screen-share dialog as a new RGB image, Capture Delayed does the same after a countdown; Capture Image flattens the current view (LUTs, display ranges, threshold, overlay) into a new RGB image |
| Utilities ▸ Monitor Events… / ImageJ Properties / Threads / Benchmark / Reset… | `help.monitor_events`, `help.properties`, `help.threads`, `help.benchmark`, `help.reset` | ✅ | session events (commands with run time, opened / saved files, errors; live in the Log); versions, platform, CPU count, process memory, plugin folders, open images; live Python threads with the line each executes; a timed set of 13 operations on a 512×512 image; Reset clears Results, ROI Manager, tables, plots, thresholds and the recorder (optionally Log and images) |
| Utilities ▸ DICOM Metadata… | `utilities.dicom_metadata`; Image Info (metadata from the DICOM reader) | ✅ | every header tag (tag, name, VR, value) re-read from the `.dcm` file or folder with pydicom |
| *Fiji* Utilities ▸ Repeat a Recent Command | History section; ⇧R | 🖥️ | |
| New ▸ Macro / JavaScript / Plugin / Plugin Filter | `plugins.new_template` | ✅ | documented templates written into the plugins folder and loaded at once: a Python command, a plane filter, an ImageJ macro (installed as a command) or a script with an auto-generated dialog; one language (Python) plus `.ijm` — the README shows a 10-line plugin |
| New ▸ Macro Tool / Plugin Tool / Plugin Frame / Text Window… / Table… | ImageJ macro-language tools: `macro "Name Tool - C00cO11cc" { getCursorLoc(x, y, z, flags); … }` sets in plugins/, macros/, startup.ijm or installed with Plugins ▸ Macros ▸ Install… go on the tool rail (`macrotool.*`; getCursorLoc press / drag / release with ImageJ flag bits, isKeyDown / setKeyDown, toolID / setTool / IJ.getToolName, call("ij.Prefs.*"), "Tool Options" / "Tool Selected" blocks, persistent `var` state, icon colour / glyph from the icon string); Python plugin tools `@op(..., tool=True)` (`tools.spot_counter`, `tools.measure_distance`); plugin frames `@op(..., frame=True)` / `kind="frame"` shown as docked sub-panels in the inspector section "Plugin Frames" (`frames.image_statistics`); templates `plugins.new_template` kinds Macro Tool / Plugin Tool / Plugin Frame, also via `run("Macro Tool"\|"Plugin Frame"\|…, "name=…")` and `run("New...", "type=[Plugin Frame]")`; `file.new_text` (⇧N), `table.new` | ✅ | "Plugin Frame" in Lumen = a command whose output (table / plot / log) lives in its own docked, refreshable inspector sub-panel rather than a floating window; Menu Tool blocks and the drawn 16×16 icons are not reproduced (glyph + colour instead); Options macros run headless (Dialog.* returns defaults) |
| Compile and Run… / Install… | `plugins.reload`, `macros.install`, `plugins.open_folder`, `plugins.list` | ✅ | hot reload: every plugin file in `~/.lumen/plugins` and `$LUMEN_PLUGINS` is re-imported and its commands re-registered without a restart (also Help ▸ Update Menus = `help.reload_plugins`); Install… copies and registers a file; List Plugins shows each file's commands and load status |
| *Fiji* Scripting ▸ Script Editor (`[`) / Script Interpreter | Script & Recorder section (`[`): Python / Jython (ImageJ 1.x API) / ImageJ-1 macro / JavaScript / Groovy / BeanShell / Rhino JavaScript / JRuby editor with Run (⌘R) / Run selection (⇧⌘R) / Stop / Clear, an output console, a `#@` script-parameter prompt, open / save of `.py` / `.jy` / `.ijm` / `.js` / `.groovy` / `.bsh` / `.rb` files and the recorder's steps as Python or JavaScript; Python console (Script Interpreter); `lumen.core.ijshim` (`ij.*`, `java.*`, `jarray`, `loci.plugins.BF`), `lumen.core.ijshim.jvm_subprocess` / `jvm_bridge` (`LUMEN_JVM_MODE`), `lumen.core.readers.fiji_jars`, `script.run_jvm_script`, `script.jvm_bridge_status`, `POST /api/script/run` (`params`, `roi`, `language`), `POST /api/script/params`, `POST /api/ijshim/call` | ✅ | every Fiji scripting style runs: ImageJ-1 macros, Jython (`from ij import IJ …` through the `ij` shim) and browser JavaScript natively; Groovy / BeanShell / Rhino JavaScript / JRuby with the real `ij.jar` in ImageJ batch mode as a `java` subprocess over an installed Fiji's jars plus a JDK, no Python package (verified bit-exact against Lumen's own filters on this Mac: `imp` in / out with undo, selection, `#@` params, shown images, output / IJ.log, Results table and ROI Manager both ways; the in-process JPype bridge `[jvm]` serves Linux / Windows); compiled ImageJ-1 plugins (`.jar` / `.class`) run through the same bridge: install them from a Fiji update site (Plugins ▸ Utilities ▸ Install from Fiji Update Site…, their commands appear under Plugins ▸ Fiji) or put the jar on `LUMEN_JVM_CLASSPATH` and call `IJ.runPlugIn(imp, "the.PlugIn", "")` from a script (verified with Fiji's own `Auto_Threshold-1.18.0.jar`: its Otsu mask is identical to Lumen's on every pixel); residue: Clojure needs `spec.alpha` / `core.specs.alpha` jars Fiji lacks (not exercised), Fiji's plugins folder is not scanned into a menu (name the class), SciJava OpService / ImgLib2 / ImageJ2 API are absent, Java Swing / GenericDialog dialogs do not exist on the JVM (use `#@`, or macro options as the plugin example does), no Git integration |
| **Segmentation** (Fiji) | | | |
| Segmentation ▸ Trainable Weka Segmentation | `segment.train_classifier`, `segment.apply_classifier`, `segment.apply_classifier_folder`, `segment.save_classifier`, `segment.load_classifier`, `segment.classifier_info`, `segment.classifier_overlay`, `segment.trace_tool`; Inspector ▸ Pixel Classifier | ✅ | the Weka GUI as an inspector section: classes (add / rename / colour / delete), traces drawn with the selection tools, the brush or the Classifier Trace Tool pen on the tool rail (stored as `class:<name>` ROI-Manager entries), Train / Apply (plane, stack, other image, folder) / Toggle overlay / Get probability / Create result / Save / Load / Settings / Info; features Gaussian, DoG, Sobel, Laplacian, Hessian, variance, entropy, membrane projections, mean / min / max / median, anisotropic diffusion, bilateral, Lipschitz, Kuwahara, Gabor, derivatives, structure, neighbours; classifiers random forest, extra trees, gradient boosting, SVM, k-NN, logistic regression, MLP (scikit-learn instead of Weka); probability maps, labels and a transparent class overlay; `.joblib` classifier files instead of Weka `.model`; `run("Trainable Weka Segmentation", …)` maps classes / features / sigma / classifier |
| Segmentation ▸ Trainable Weka Segmentation 3D / Color Clustering | `segment3d.train_classifier`, `segment3d.apply_classifier`, `segment3d.save_classifier`, `segment3d.load_classifier` (same Pixel Classifier panel, Trace Tool and folder batch); `segment.color_clustering` | ✅ | 3-D classifier on volumetric features (3-D Gaussians scaled for the z spacing, DoG, Sobel, Hessian eigenvalues, variance, Laplacian, mean / min / max / median, structure) from traces drawn on any slice, the same seven classifier types, applied per frame in z chunks under a memory cap, `.joblib` save / load; k-means colour clustering (RGB / Lab / HSB or channel values, optional spatial weight) → label image |
| MorphoLibJ ▸ Segmentation ▸ Morphological Segmentation / Classic Watershed / Marker-controlled Watershed | `morpho.morphological_segmentation`, `morpho.marker_watershed`, `segment.seeded_watershed`, `segment3d.seeded_watershed_3d` | ✅ | object or gradient input, morphological / Sobel gradient, tolerance (dynamic) for the extended-minima markers, 4 / 8 connectivity, dams; outputs Overlaid basins / Overlaid dams / Catchment basins (labels) / Watershed lines, with the sheet's live preview in place of the plugin's slider; marker-controlled watershed from a marker image (labels or binary), point or area ROIs or local maxima, optionally inside a mask, with or without dams, in 2-D and 3-D (smoothing, gradient, compactness, threshold / mask limit) |
| MorphoLibJ ▸ Binary Images ▸ Distance Transform Watershed / Connected Components Labeling / Area Opening / Keep Largest Region | `binary.watershed`, `binary.label`, `binary.remove_small`, `morpho.keep_largest_label`; 3-D: `segment3d.watershed_3d`, `segment3d.label_3d`, `segment3d.filter_labels_3d`, `binary.distance_map_3d` | ✅ | |
| MorphoLibJ ▸ Analyze ▸ Analyze Regions / Analyze Regions 3D / Intensity Measurements 2D/3D / Microstructure Analysis | `morpho.analyze_regions`, `morpho.analyze_regions_3d`, `morpho.intensity_measurements`, `morpho.microstructure_analysis`, `segment3d.measure_labels_3d`; `analyze.label_objects` | ✅ | per label: area, Crofton perimeter, circularity, Euler number, bounding box, centroid, equivalent ellipse, convexity, max Feret, oriented box, geodesic diameter (chamfer 3-4), tortuosity, largest inscribed circle, average thickness, with ellipse / box / circle / geodesic-path overlays; intensity mean / SD / max / min / median / mode / skewness / kurtosis / voxels / volume inside every label of a 2-D plane or a 3-D stack; Analyze Regions 3D per label of the z-stack: volume, Crofton surface area (3 / 13 directions), sphericity, Euler number (6 / 26), bounding box, centroid, equivalent ellipsoid (centre, radii, azimuth / elevation / roll), ellipsoid elongations, largest inscribed sphere and mean breadth in calibrated units; Microstructure Analysis = densities of the intrinsic volumes (area / volume, perimeter / surface, mean breadth, Euler number) with edge correction for the whole image or per label, 2-D and 3-D |
| MorphoLibJ ▸ Label Images ▸ Keep / Remove Largest Label, Remove / Replace Label(s), Crop Label, Label Boundaries, Dilate Labels, Label Morphological Filters, Expand Labels, Label Size Filtering, Remove Border Labels, Merge Labels, Label Map to ROIs, Region Adjacency Graph, Label Overlap Measures | `morpho.keep_largest_label`, `morpho.remove_largest_label`, `morpho.replace_label`, `morpho.crop_label`, `morpho.labels_to_boundaries`, `morpho.dilate_labels`, `morpho.label_morphology`, `segment.expand_labels`, `segment.shrink_labels`, `segment.label_filter`, `segment.remove_border_labels`, `segment.merge_touching_labels`, `segment.labels_to_rois`, `segment.rois_to_labels`, `morpho.region_adjacency_graph`, `morpho.label_overlap_measures`; 3-D: `segment3d.expand_labels_3d`, `segment3d.filter_labels_3d`, `segment3d.labels_to_rois_3d` | ✅ | boundaries inner / outer / thick; label dilation without overlap, optionally within a mask; dilation / erosion / opening / closing of label images; RAG as a table plus centroid lines; overlap, Jaccard, Dice, target / source overlap, volume similarity, FN / FP per label pair |
| MorphoLibJ ▸ Label Images ▸ Label Edition / Labels To RGB / Assign Measure to Label | `morpho.select_label`, `morpho.remove_label_at` (tool-rail click-to-remove), `morpho.merge_labels`, `morpho.fill_label_holes`, `morpho.label_size_opening`, `morpho.labels_to_rgb`, `morpho.assign_measure_to_label`; Glasbey LUTs (`color.lut`) | ✅ | Label Edition's operations as commands and a tool: select a label by value / click, remove the label under every click, merge listed or clicked labels, fill the holes enclosed by labels, size opening; Labels To RGB with Glasbey / Glasbey on dark / golden angle / random or any LUT, background colour and shuffling; Assign Measure to Label = a 32-bit parametric map of any Results / table column |
| MorphoLibJ ▸ Filters ▸ Morphological Filters (2D / 3D), Directional Filtering / Minima & Maxima ▸ Regional, Extended, Impose Min & Max (2D / 3D) / Binary Images ▸ Chamfer & Geodesic Distance Map, Kill Borders, Fill Holes, Area Opening, Binary Overlay / Morphological Reconstruction (2D / 3D) / Label Images ▸ Label Overlay | `morpho.filters`, `morpho.filters_3d`, `morpho.directional_filter`, `morpho.regional_min_max`, `morpho.extended_min_max`, `morpho.impose_min_max`, `morpho.chamfer_distance_map`, `morpho.geodesic_distance_map`, `morpho.kill_borders`, `morpho.fill_holes`, `morpho.area_opening`, `morpho.reconstruct`, `morpho.binary_overlay`, `morpho.label_overlay` | ✅ | erosion / dilation / opening / closing / white and black top hat / gradients / Laplacian with disk, square, diamond, octagon or line elements (ball, cube or cross with x / y / z radii in 3-D); directional filtering over N line orientations (max / min of opening, closing, erosion, dilation, mean, median); regional and extended (dynamic) minima / maxima and minima / maxima imposition; chessboard, city-block, quasi-Euclidean, Borgefors and chessknight chamfer distances (normalised, 16- or 32-bit) and geodesic distances inside a mask; kill borders, fill holes and area opening for binary and gray-level images; reconstruction by dilation / erosion (4 / 8 connectivity, on the plane or the whole stack); binary and label overlays as new RGB images; the MorphoLibJ macro names (`run("Morphological Filters", …)`, `run("Extended Min & Max", …)`, …) map 1:1 |
| *(update sites)* StarDist / Cellpose | `segment.stardist`, `segment.cellpose` | ✅ | optional packages; 2-D models |
| *(Lumen extras)* | `segment.random_walker`, `segment3d.random_walker_3d`, `segment.active_contour`, `segment.morph_snakes`, `segment.superpixels`, `segment.multi_otsu` | — | scikit-image methods with no Fiji-core equivalent |
| **Tracking** (Fiji) | | | |
| Tracking ▸ TrackMate | `tracking.detect_spots`, `tracking.filter_spots`, `tracking.link_spots`, `tracking.track`, `tracking.filter_tracks`, `tracking.feature_tables`, `tracking.export_feature_tables`, `tracking.export_tracks`, `tracking.export_trackmate_xml`, `tracking.import_trackmate_xml`, `tracking.split_track`, `tracking.merge_tracks`, `tracking.link_two_spots`, `tracking.unlink_spots`, `tracking.add_spot`, `tracking.delete_spot`, `tracking.undo_edit`, `tracking.set_display`, `tracking.tracks_to_rois`, `tracking.clear_tracks`; Inspector ▸ Tracks: wizard + TrackScheme | ✅ | the wizard (Detector → initial threshold on the quality histogram → spot filters → linker → track filters → display → export) runs the ordinary commands: LoG / DoG / local-maxima detectors with sub-pixel localisation and TrackMate's spot features (quality, intensity statistics, contrast, SNR); spot filters on any feature; nearest-neighbour, simple LAP or full LAP linking (gap closing, splitting, merging, alternative-cost factor, feature penalties) → lineage trees; track filters on any track feature; Spots / Edges / Track statistics tables in TrackMate's conventions and as CSV; TrackMate XML in / out with lineage edges and features; display settings (colour by feature with a colormap, full / local-tail / fade modes, spot radius, names); TrackScheme = the lineage graph with the selection synced to the viewer, split / link / unlink / delete / undo and SVG export; the Workflows panel's *Track cells* orders the same steps; no Kalman / overlap trackers, no Hessian / thresholding / label-image detectors, no 3-D linking (detection on the current slice or the max projection) |
| Tracking ▸ Load a TrackMate file / Manual tracking with TrackMate | `tracking.import_trackmate_xml`, `tracking.export_trackmate_xml`; Manual tracking tool (X) | ✅ | AllSpots / AllTracks / edges → overlays, stored spots, lineage links and the spot / edge / track features, all of which round-trip through export (a Model that TrackMate opens); splits and merges are kept as lineage edges and shown in TrackScheme |
| Tracking ▸ Manual Tracking | Manual tracking tool (X): click adds this frame's point and advances, ⌥-click deletes; `tracking.manual_new_track`, `tracking.manual_add_point`, `tracking.manual_delete_point` | ✅ | manual tracks share the Tracks table, overlays, filters and exports with automatic tracking |
| *(update site)* KymographBuilder / Multi Kymograph | `tracking.kymograph` | ✅ | along a line selection, Z or T |
| **Registration** (Fiji) | | | |
| Registration ▸ Linear Stack Alignment with SIFT (+ MultiChannel) | `registration.align_stack` (estimator SIFT feature matching); `registration.extract_correspondences` | ✅ | the dialog's parameters: initial Gaussian blur, steps per scale octave, minimum / maximum image size, feature descriptor size and orientation bins, closest / next-closest ratio, maximal alignment error, inlier ratio, expected transformation (translation / rigid / similarity = scaled rotation / affine), interpolate, show info and transformation matrix; the registration channel drives all channels (MultiChannel); per-slice transforms in an Alignment table; `run("Linear Stack Alignment with SIFT", …)` maps 1:1 |
| Registration ▸ Elastic Stack Alignment / Elastic Montaging | `registration.elastic_align`, `registration.elastic_montage` | ✅ | dense optical-flow (TV-L1 / iterative Lucas-Kanade, multi-scale) alignment of slices or frames onto the previous / first / middle slice with optional displacement-field output; Elastic Montaging places the tiles (grid, positions file, unknown positions or the open images) by phase correlation + global optimisation, aligns every overlapping pair with an optical-flow field, relaxes the per-tile deformations on spring meshes (mesh spacing, stiffness, minimum correlation, maximum displacement) and blends the warped tiles into one montage, with a Montage table and pair residuals before / after in Results; a different estimator from mpicbg's block matching |
| Registration ▸ Descriptor-based registration (2d/3d) / series registration / Manual landmark selection | `registration.descriptor_registration`, `registration.descriptor_series`, `registration.landmark_pairs`; also `registration.register_to_image`, `registration.feature_correspondences`, `image.transform_landmarks`, `registration.phase_correlation`; 3-D: `registration.register_3d`, `registration.align_time_3d` | ✅ | Preibisch's method: DoG detections (bright / dark / both, size, threshold) described by the geometry of their nearest neighbours (translation / rigid / similarity / affine invariant descriptors with redundancy), matched by descriptor similarity (significance ratio) and filtered by RANSAC; 2-D or 3-D (two Z stacks, in calibrated space), applied to every channel and frame; series registration onto the previous (chained) or first frame with per-frame transforms and detection / candidate / inlier counts in an Alignment table; manual landmarks = two multi-point ROIs from the ROI Manager (point i ↔ point i) fitted by least squares to translation / rigid / similarity / affine / projective; inliers → Matches ROIs, Correspondences table and the transform in Results; `run("Descriptor-based registration (2d/3d)", …)` maps the dialog keys |
| Registration ▸ bUnwarpJ | `registration.bunwarpj`, `registration.bunwarpj_apply`, `registration.bunwarpj_load`, `registration.bunwarpj_save`, `registration.bunwarpj_invert`, `registration.bunwarpj_compare`, `registration.bunwarpj_convert`; `registration.elastic_pair`, `registration.apply_deformation` | ✅ | cubic B-spline free-form deformation with the plugin's dialog: Fast / Accurate / Mono, subsampling, initial and final deformation (very coarse … super fine), divergence / curl / landmark / image / consistency weights, stop threshold, landmark ROIs; registered source (and target), deformation field and transform files in bUnwarpJ's elastic / raw formats, which the I/O commands apply, load, save, invert, compare (warping index) and convert like the plugin's I/O menu; `registration.elastic_pair` is the optical-flow alternative |
| Registration ▸ Register Virtual Stack Slices | `registration.align_folder` | ✅ | folder in → registered images out (tif / png / npy); translation / rigid / affine / elastic onto the first or previous image; transforms in the Alignment table and `transforms.csv` |
| Registration ▸ Correct 3D drift | `registration.drift_correct`, `registration.align_time_3d` | ✅ | xy (and optional z) per frame from projections, or full 3-D phase correlation (sub-voxel, optional crop to the region valid in every frame), applied to all channels / slices; shifts reported in an Alignment table |
| *(BIG update site)* StackReg / TurboReg | `registration.align_stack` (estimator Intensity, pyramid optimisation) | ✅ | translation / rigid body / scaled rotation / affine / bilinear by multi-resolution intensity optimisation (pyramid levels, iterations) onto the previous / current slice, like StackReg; `run("StackReg", "transformation=…")` maps 1:1 |
| *(Lumen extras)* | `registration.align_channels`, `registration.multi_channel_3d`, `registration.phase_correlation` | — | chromatic-shift correction in 2-D and 3-D (per-channel dx / dy / dz in an Alignment table); phase-correlation shift / peak between two images |
| Plugins ▸ Transform ▸ Interactive Rigid / Similarity / Affine / Perspective / MLS / Landmark Correspondences | `image.transform_landmarks`, `image.transform_mls`, `image.transform_landmark_correspondences`, `image.perspective`, `image.shear`, `image.warp_polar` | ✅ | landmark-driven translation / rigid / similarity / affine / projective warps and moving-least-squares deformation (locally affine / similarity / rigid, alpha) from point selections, ROI-Manager points, typed lists or the extracted correspondences; Landmark Correspondences transforms the source into a target image's frame (least squares or MLS); four-corner perspective rectification, shear, polar / log-polar; parameter sheets with live preview instead of dragged handles |
| Plugins ▸ Feature Extraction ▸ SIFT / MOPS / Block Matching correspondences | `registration.extract_correspondences`, `registration.feature_correspondences`, `registration.block_matching_2d` | ✅ | SIFT, MOPS (multi-scale oriented patches), ORB or block-matching correspondences between two images with the dialog's parameters (initial blur, octave steps, image sizes, descriptor size / orientation bins, closest / next-closest ratio, maximal alignment error, inlier ratio) filtered by RANSAC on translation / rigid / similarity / affine / projective → Matches point ROIs on both images, a Correspondences table and the transform in Results (reused by Landmark Transform); block matching on a grid (search radius, sub-pixel peak) → arrow overlay, Block matching table and optionally a dense warp onto the target |
| **Stitching** (Fiji) | | | |
| Stitching ▸ Pairwise stitching | `registration.pairwise_stitch` | ✅ | phase correlation with peak checking; Linear blend / Max / Average / Overlay fusion |
| Stitching ▸ Grid/Collection stitching | `registration.stitch_collection`; `registration.grid_stitch` | ✅ | Fiji's dialog: grid row-by-row / column-by-column / snake (folder + pattern, grid size, overlap), positions from a TileConfiguration.txt, unknown positions (all-pairs phase correlation) or the open images; compute overlap, regression / max-avg / absolute displacement thresholds, peak checking, sub-pixel accuracy, downsampling; 2-D and 3-D tiles, multi-channel and time-series tiles (same positions for all frames or per frame); fusion linear blending / average / median / max / min / overlay into composite / none (writes TileConfiguration.registered.txt); open the result or write it to file; `run("Grid/Collection stitching", …)` maps the dialog keys; no "Filename defined position" / "Sequential images" types and no multi-series Bio-Formats input |
| **3-D** (Fiji) | | | |
| Plugins ▸ 3D Viewer | 3-D View section: MIP / alpha / iso-surface / orthoslice / mesh modes, multi-image scenes (any open image with its own colour, transparency, threshold / iso level, channel and transform), view presets, perspective / orthographic, background, lights, clipping box, landmarks (click on a surface; named; to Results / ROI Manager), turntable and freehand animation recorded into an RGB stack / GIF / MP4, supersampled snapshots; `scene.save`, `scene.load`, `scene.landmarks_to_results`, `mesh.export`, `mesh.measure` | ✅ | scenes saved / loaded as JSON (3D Viewer ▸ Save / Load view); marching-cubes surfaces (step, Taubin smoothing) exported as OBJ / STL / PLY and measured (area, volume, centroid, bounding box per component); MP4 needs the opt-in `[video]` extra (GIF always works); no landmark-based registration of objects; MIP / alpha volumes are blended over surfaces without per-ray depth termination |
| Plugins ▸ Volume Viewer | 3-D View section | ✅ | MIP / alpha-blended / iso-surface rendering of the current stack |
| Stacks ▸ 3D Project (covered above) | `stacks.project3d` | ✅ | |
| **Other Fiji plugins** | | | |
| Plugins ▸ Time Lapse ▸ (Circle/LOI Interpolator, Phase Map, Gaussian stack focuser…) | `stacks.interpolate_rois`, `timelapse.phase_map`, `registration.stack_focus` | ✅ | LOI / shape interpolator: ROI Manager ROIs on different frames or slices are keyframes and the ROIs in between are interpolated; Phase Map = per pixel the dominant periodic component over the frames (FFT within a period range, detrending, window, phase relative to the mean signal or a selection) as an amplitude / phase / period 3-channel 32-bit image plus the mean power spectrum plot and a Results summary; stack focuser = extended depth of field (sharpest slice per pixel by Sobel energy / local variance / Laplacian energy, optional height map) |
| Plugins ▸ Optic Flow ▸ … | `process.optic_flow` | ✅ | dense TV-L1 or iterative Lucas-Kanade flow between consecutive frames: magnitude / direction / component stacks and a vector overlay on a grid (different estimators from Fiji's Gaussian-window MSE / integral-block LK / PMCC) |
| Image ▸ Drawing ▸ Radial / Linear Gradient | `edit.linear_gradient`, `edit.radial_gradient`; `file.new(fill="Ramp")` | ✅ | foreground → background gradient in the selection or the whole plane: linear along a line selection (or across an area's bounding box), radial from the centre to a radius |

## Window

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| Show All | single window | 🖥️ | nothing can be hidden behind another window |
| Put Behind [tab] | image tabs (click) | 🖥️ | no Tab-key cycling |
| Cascade / Tile | tabs; View ▸ Split View (⇧⌘\\) shows two images side by side, View ▸ Synchronize Views | 🖥️ | two panes at most — no free tiling or cascading of N windows |
| *(list of open windows)* | image tabs | 🖥️ | |
| *Fiji* Tasks window (progress) | progress chip in the top bar (name, percentage, message, elapsed time, Cancel), toasts + status bar | 🖥️ | long commands report progress over the WebSocket and can be cancelled (`POST /api/jobs/{id}/cancel`); a cancelled command rolls its image back |

## Help

| Fiji command | Lumen equivalent | Status | Notes |
|---|---|---|---|
| ImageJ Website… / News… / Documentation… / Search Website… / List Archives… / Plugins… / Macros… / Source Code… / Macro Language… / Macro Functions… | `help.links`; Help ▸ Lumen README / Command Reference / Fiji Parity (in-app documentation viewer); `help.command`, `lumen.help("<id>")` | ✅ | a table of Lumen / ImageJ / Fiji / Python-stack links, the docs rendered in-app, and per-command help with parameters, types, defaults and choices |
| Update ImageJ… / *Fiji* Update… | `help.update`, `help.check_updates`, `plugins.fiji_update_site`, `plugins.fiji_install`, `plugins.fiji_installed`, `plugins.run_fiji_plugin` | ✅ | Lumen updates itself through Check for Updates / Update Lumen… (git pull or pip); **Fiji update sites** are read directly: Plugins ▸ Utilities ▸ Fiji Update Site… lists a site's files from its `db.xml.gz` (ImageJ, Fiji, BIG-EPFL, IJPB, CSBDeep or any URL / folder), Install from Fiji Update Site… downloads the chosen plugin jars and their missing dependencies (size-checked) into `~/.lumen/jars`, and every command a jar declares in its `plugins.config` becomes a Lumen command under Plugins ▸ Fiji – run by the real ImageJ 1.x through the Java bridge on the active image, with macro options and an undo step, and reachable from ImageJ macros by its name (verified with Fiji's Auto_Threshold jar from a local update site: identical to Lumen's Otsu mask; `run("Kuwahara Filter", "sampling=5")` runs Fiji's Kuwahara plugin, which Lumen does not implement); needs a JDK and Fiji's jars; interactive Swing plugins and ImageJ2 / SciJava commands stay out of reach |
| Update Menus | `help.reload_plugins`, `plugins.reload` (Plugins ▸ Utilities) | ✅ | re-imports every plugin file and re-registers its commands without a restart; the menus refetch the command list |
| About Plugins ▸ | `help.about_plugins` (Help ▸ About Plugins) | ✅ | every loaded command module (built-in and `~/.lumen/plugins`) with its command count, plus modules that failed to import |
| About ImageJ… | Help ▸ About Lumen | 🖥️ | |
| *Fiji* Help on Menu Item | `help.command`; palette shows each command's description | ✅ | id, menu, kind, shortcut and a parameter table |
| *(Lumen)* | Help ▸ Keyboard Shortcuts (?) | — | |

---

## Summary

| Fiji menu | ✅ covered | 🟡 partial | ❌ missing | 🖥️ UI feature | Fiji rows | Covered or partial | Lumen-only extras |
|---|---|---|---|---|---|---|---|
| File | 41 | 0 | 0 | 5 | 46 | 46 (100%) | 0 |
| Edit | 46 | 0 | 0 | 5 | 51 | 51 (100%) | 2 |
| Image | 76 | 0 | 0 | 10 | 86 | 86 (100%) | 2 |
| Process | 40 | 0 | 0 | 2 | 42 | 42 (100%) | 2 |
| Analyze | 25 | 0 | 0 | 2 | 27 | 27 (100%) | 1 |
| Plugins | 43 | 0 | 0 | 5 | 48 | 48 (100%) | 2 |
| Window | 0 | 0 | 0 | 5 | 5 | 5 (100%) | 0 |
| Help | 5 | 0 | 0 | 1 | 6 | 6 (100%) | 1 |
| **All menus** | **276** | **0** | **0** | **35** | **311** | **311 (100%)** | **10** |

*Rows are Fiji commands or tightly related command groups as listed above (a row such as
"Math ▸ Add… … Abs" counts once even though it maps to 18 Lumen commands), so the table measures
menu coverage, not command counts. Lumen registers 503 commands in total.*

**Overall:** of 311 Fiji/ImageJ menu rows, 276 are covered by an equivalent Lumen command and 35 are
covered by a built-in UI feature (together 311, 100%); none are partial or missing. The ImageJ 1.x core (File, Edit, Image, Process, Analyze) is complete for 2-D, stack and
hyperstack work, including the FFT submenu, text windows and text / table import-export, every Options dialog, printing,
Batch ▸ Virtual Stack, TIFF virtual stacks and the Fiji LUT set; the Plugins menu covers the Bio-Formats importer / exporter /
macro extensions over native readers, MorphoLibJ (segmentation, filters, minima / maxima, distance maps, reconstruction, region
analysis, label-image tools and Label Edition), Trainable Weka Segmentation in 2-D and 3-D with its GUI, TrackMate with the wizard
and TrackScheme, SIFT / descriptor-based / intensity (StackReg) / elastic / bUnwarpJ registration, Elastic Montaging, Grid/Collection
stitching, the 3D Viewer's scenes, landmarks and animations, the Utilities submenu, plugin tools, hot-reloaded plugins and macro /
plugin templates. The remaining limits are not missing functions but deployment and scope notes: (1) the Bio-Formats long tail of
proprietary formats and the JVM script languages (Groovy, BeanShell, Rhino, JRuby) run through `java` subprocesses over
an installed Fiji's jars plus a JDK (verified on this Mac; the pure-Python `[bioformats]` / `[jvm]` extras are the
alternative on Linux / Windows), so a machine without Java falls back to the native readers and to Python / Jython /
JavaScript / macros, and compiled ImageJ-1 plugins run the same way when their jar is named on `LUMEN_JVM_CLASSPATH`; (2) Clojure scripts (two jars Fiji does not ship) and the SciJava / ImgLib2 API are out of reach without
Fiji itself; (3) the single-window compromises marked 🖥️ (two panes instead of free window tiling, a fixed animation
rate, screen capture through the browser's screen-share dialog). Within the ✅ rows the notes name the dialog options that differ or
are missing — TrackMate's Kalman / overlap trackers and 3-D linking, the stitching "Filename defined position" / "Sequential
images" layouts, the 3D Viewer's landmark-based registration, and estimators that differ from mpicbg's (optical flow instead
of block matching for the elastic alignments).

## Cross-validation against ImageJ 1.54p

Parity rows say a command *exists*; [docs/CROSSVALIDATION.md](CROSSVALIDATION.md) says how closely it
*computes the same thing*. 194 ImageJ macros (277 cases over 8/16/32-bit images, a Z stack, a 2-channel
composite and a calibrated TIFF) were run unchanged by the real ImageJ 1.54p (`ij.jar` in batch mode on
OpenJDK 21) and by Lumen's macro layer, and the pixels, Results rows and printed values were compared:
**275 of the 277 cases are exact, 1 is within a stated tolerance and 1 is a known mismatch** (the first
run found 115 exact / 33 within tolerance / 126 mismatches; the fix round that followed ported ImageJ's
rank-filter kernel, rolling ball, AutoThresholder, binary operations, Analyzer / EllipseFitter / Feret /
ParticleAnalyzer, ZProjector, transforms, TypeConverter, FFT padding, MaximumFinder, histogram binning,
ProfilePlot and the ROI geometry bit-exactly). The residue: the Stack Statistics Results row that ImageJ prints with 3
decimals, and a Fill-Holes-on-a-stack case that is an ImageJ batch-mode artefact (the Convert Stack to
Binary dialog resets black-background, so ImageJ's own Fill Holes ran with foreground 0). The tests live in
`tests/crossval` (opt-in: `pytest tests/crossval -m crossval`).


## ImageJ macro compatibility layer

`lumen/core/ij_compat.py` (dumped 7 Nov 2026 via `command_table()`) lets `script.run_ij_macro` and
`macros.run_file` execute ImageJ-1 macro code: `run("…", "key=value")` calls are translated to Lumen commands with
their option strings mapped to explicit parameters, and the macro built-ins below are
implemented on top of the session (images, ROI Manager, Results table, threshold, display).
`script.ij_command_list` shows this table inside Lumen. Handlers honour ImageJ's `stack` flag,
slice/channel/frame ranges (Duplicate, Make Substack, Z Project) and calibrated units (Enlarge, Make Band,
Specify, Set Scale) the way the ImageJ dialogs do.

**324 `run()` rows: 323 rows map to Lumen commands (423 command names including aliases); one row covers 34 dialog/window commands accepted as no-ops. No command is declared unsupported any more.**

| ImageJ `run("…")` command | Lumen command | Option mapping / notes |
|---|---|---|
| "Gaussian Blur..." | `filters.gaussian` | sigma (or legacy radius/2.5) → sigma; 'scaled' → sigma in calibrated units; 'stack' → all planes |
| "Median..." | `filters.median` |  |
| "Mean..." | `filters.mean` |  |
| "Minimum..." | `filters.minimum` |  |
| "Maximum..." | `filters.maximum` |  |
| "Variance..." | `filters.variance` |  |
| "Unsharp Mask..." | `filters.unsharp` | radius → radius, mask → weight |
| "Convolve..." | `filters.convolve` | text1 → kernel |
| "Top Hat..." | `filters.tophat` |  |
| "Gaussian Blur 3D..." | `filters.gaussian3d` |  |
| "Median 3D..." | `filters.rank3d` | x → radius_xy, z → radius_z |
| "Mean 3D..." | `filters.rank3d` | x → radius_xy, z → radius_z |
| "Minimum 3D..." | `filters.rank3d` | x → radius_xy, z → radius_z |
| "Maximum 3D..." | `filters.rank3d` | x → radius_xy, z → radius_z |
| "Variance 3D..." | `filters.rank3d` | x → radius_xy, z → radius_z |
| "Smooth" | `process.smooth` |  |
| "Sharpen" | `process.sharpen` |  |
| "Find Edges" | `process.find_edges` |  |
| "Enhance Contrast" / "Enhance Contrast..." | `process.enhance_contrast` |  |
| "Enhance Local Contrast (CLAHE)" / "CLAHE" | `process.clahe` | blocksize, histogram, maximum/100 → clip_limit |
| "Subtract Background..." | `process.subtract_background` | rolling → radius; light; create; sliding → Sliding paraboloid; disable → no smoothing |
| "Bandpass Filter..." | `process.fft_bandpass` |  |
| "FFT" | `process.fft` |  |
| "Add Noise" | `noise.add` |  |
| "Add Specified Noise..." | `noise.add_specified` | standard → sd |
| "Salt and Pepper" | `noise.salt_pepper` |  |
| "Despeckle" | `noise.despeckle` |  |
| "Remove Outliers..." | `noise.remove_outliers` |  |
| "Remove NaNs..." | `noise.remove_nans` |  |
| "North" | `process.shadows` | Process ▸ Shadows |
| "Northeast" | `process.shadows` | Process ▸ Shadows |
| "East" | `process.shadows` | Process ▸ Shadows |
| "Southeast" | `process.shadows` | Process ▸ Shadows |
| "South" | `process.shadows` | Process ▸ Shadows |
| "Southwest" | `process.shadows` | Process ▸ Shadows |
| "West" | `process.shadows` | Process ▸ Shadows |
| "Northwest" | `process.shadows` | Process ▸ Shadows |
| "Add..." | `math.add` |  |
| "Subtract..." | `math.subtract` |  |
| "Multiply..." | `math.multiply` |  |
| "Divide..." | `math.divide` |  |
| "Min..." | `math.min` |  |
| "Max..." | `math.max` |  |
| "Gamma..." | `math.gamma` |  |
| "Set..." | `math.set` |  |
| "AND..." | `math.and` |  |
| "OR..." | `math.or` |  |
| "XOR..." | `math.xor` |  |
| "Log" | `math.log` |  |
| "Exp" | `math.exp` |  |
| "Square" | `math.square` |  |
| "Square Root" | `math.sqrt` |  |
| "Reciprocal" | `math.reciprocal` |  |
| "Abs" | `math.abs` |  |
| "NaN Background" | `math.nan_background` |  |
| "Invert" | `edit.invert` |  |
| "Macro..." | `math.macro` | Process ▸ Math ▸ Macro: code=[v=v+50*sin(d/10)] with v x y z w h a d (ImageJ's pixel-wise macro); 'stack' → all planes |
| "Fill" | `edit.fill` |  |
| "Clear" | `edit.clear` |  |
| "Clear Outside" | `edit.clear_outside` |  |
| "Draw" | `edit.draw` |  |
| "Crop" | `image.crop` |  |
| "Rotate 90 Degrees Right" | `image.rotate90` |  |
| "Rotate 90 Degrees Left" | `image.rotate90` |  |
| "Flip Horizontally" | `image.flip_h` | 'stack' → every plane, otherwise only the current plane (IJ.setupDialog) |
| "Flip Vertically" | `image.flip_v` | 'stack' → every plane, otherwise only the current plane (IJ.setupDialog) |
| "Flip Z" | `image.flip_z` |  |
| "Rotate... " | `image.rotate` | Image ▸ Transform ▸ Rotate (note the trailing space, as in ImageJ); 'stack' → every plane, otherwise only the current plane (IJ.setupDialog); 'enlarge' grows the canvas of every slice |
| "Translate..." | `image.translate` | 'stack' → every plane, otherwise only the current plane (IJ.setupDialog) |
| "Canvas Size..." | `image.canvas_size` |  |
| "Bin..." | `image.bin` |  |
| "Rename..." | `image.rename` |  |
| "Coordinates..." | `adjust.coordinates` |  |
| "8-bit" | `type.8bit` |  |
| "16-bit" | `type.16bit` |  |
| "32-bit" | `type.32bit` |  |
| "RGB Color" / "Stack to RGB" | `type.rgb` |  |
| "RGB Stack" | `type.rgb_stack` |  |
| "HSB Stack" | `type.hsb_stack` |  |
| "Lab Stack" | `type.lab_stack` |  |
| "Split Channels" | `color.split` |  |
| "Arrange Channels..." | `color.arrange` |  |
| "Invert LUT" | `color.invert_lut` |  |
| "Apply LUT" | `type.apply_lut` | bakes the display range into the pixels |
| "Erode" | `binary.erode` |  |
| "Dilate" | `binary.dilate` |  |
| "Open" | `binary.open` |  |
| "Close-" | `binary.close` |  |
| "Outline" | `binary.outline` |  |
| "Fill Holes" | `binary.fill_holes` |  |
| "Skeletonize" | `binary.skeletonize` | method=[Zhang-Suen\|Lee\|Thin (Guo-Hall)\|Medial axis] is a Lumen extension |
| "Distance Map" | `binary.distance_map` |  |
| "Ultimate Points" | `binary.ultimate_points` | output=Image → 32-bit UEP image holding the EDM value (ImageJ's output); default: the UEPs as a point selection |
| "Watershed" | `binary.watershed` | method=[ImageJ EDM/UEP] / tolerance= are Lumen extensions |
| "Voronoi" | `binary.voronoi` |  |
| "Auto Local Threshold" | `threshold.local` | Fiji: method=Bernsen\|Contrast\|Mean\|Median\|MidGrey\|Niblack\|Otsu\|Phansalkar\|Sauvola radius= parameter_1= white (parameter_2 is ignored) |
| "Z Project..." | `stacks.z_project` | projection start stop; nSlices==1 → projects the frames (ImageJ ZProjector); 'all' → every frame, else only the current frame |
| "Grouped Z Project..." | `stacks.grouped_z_project` |  |
| "Reverse" | `stacks.reverse` |  |
| "Delete Slice" | `stacks.delete_slice` |  |
| "Add Slice" | `stacks.add_slice` | add=channel/slice/frame (ImageJ hyperstack dialog) → C/Z/T |
| "Make Montage..." | `stacks.montage` | first/last/increment/label are ignored |
| "Stack to Images" | `stacks.to_images` |  |
| "Images to Stack" | `stacks.images_to_stack` |  |
| "Concatenate..." | `stacks.concatenate` | image3+ are ignored |
| "Stack to Hyperstack..." | `stacks.to_hyperstack` |  |
| "Plot Z-axis Profile" | `stacks.z_profile` |  |
| "Reduce..." | `stacks.reduce` |  |
| "3D Project..." | `stacks.project3d` |  |
| "Orthogonal Views" | `stacks.orthogonal_views` |  |
| "Combine..." | `stacks.combine` |  |
| "Straighten..." | `stacks.straighten` |  |
| "Label..." | `stacks.label_slices` | format=0/0000/Text [text=] starting= interval=; numbers the frames (nFrames>1) else the slices, like ImageJ's StackLabeler (label = starting + (n-1)*interval when the op supports start/interval); x/y/range ignored |
| "Time Stamper" | `overlay.time_stamp` | starting= (start time) interval= (time per frame) decimal= or= (suffix) font=; x/y/anti-aliased ignored, stamp is an overlay (Burn In to make it permanent) |
| "Select All" | `selection.all` |  |
| "Select None" | `selection.none` |  |
| "Enlarge..." | `selection.enlarge` | enlarge= is in calibrated units unless 'pixel' is given (ImageJ RoiEnlarger: pixels = n / pixelWidth) |
| "Make Band..." | `selection.band` | band= is always in calibrated units (ImageJ: round(size / pixelWidth)) |
| "Convex Hull" | `selection.convex_hull` |  |
| "Fit Ellipse" | `selection.fit_ellipse` |  |
| "To Bounding Box" | `selection.bounding_box` |  |
| "Rotate..." | `selection.rotate` | Edit ▸ Selection ▸ Rotate (no trailing space); falls back to Image ▸ Rotate without a selection |
| "Scale... " | `selection.scale` | Edit ▸ Selection ▸ Scale (trailing space) |
| "Interpolate" | `selection.interpolate` |  |
| "Line Width..." | `selection.line_width` |  |
| "Specify..." | `selection.specify` | width height x y oval centered; 'scaled' → width/x in units ÷ pixel_width, height/y ÷ pixel_height (ImageJ SpecifyROI) |
| "Create Mask" | `selection.create_mask` |  |
| "Create Selection" | `selection.create` |  |
| "Make Inverse" | `selection.inverse` |  |
| "Line to Area" | `selection.line_to_area` |  |
| "Area to Line" | `selection.area_to_line` |  |
| "Flatten" | `overlay.flatten` |  |
| "Remove Overlay" | `overlay.remove` |  |
| "From ROI Manager" | `overlay.from_manager` |  |
| "To ROI Manager" | `overlay.to_roi_manager` |  |
| "List Elements" | `overlay.list` |  |
| "Scale Bar..." | `overlay.scale_bar` |  |
| "Calibration Bar..." | `overlay.calibration_bar` |  |
| "Burn In Overlay" | `overlay.burn_in` |  |
| "Measure" | `analyze.measure` |  |
| "Clear Results" | `analyze.clear_results` |  |
| "Summarize" | `analyze.summarize` |  |
| "Plot Profile" | `analyze.plot_profile` |  |
| "Distribution..." | `analyze.distribution` |  |
| "Find Maxima..." | `analyze.find_maxima` |  |
| "Curve Fitting..." | `analyze.curve_fit` |  |
| "Fractal Box Count..." | `analyze.fractal_box_count` |  |
| "Surface Plot..." | `analyze.surface_plot` |  |
| "Label" | `analyze.label` |  |
| "Multi Measure" | `analyze.multi_measure` |  |
| "Angle" | `analyze.angle` |  |
| "Gel Analyzer" | `analyze.gel_lanes` | Lumen's one-step gel analysis: lanes = rectangles in the ROI Manager |
| "Calibrate..." | `analyze.calibrate` | function=[None\|Straight Line\|2nd Degree Polynomial\|3rd Degree Polynomial\|Exponential\|Exponential with Offset\|Exponential Recovery\|Power\|Log\|Rodbard\|Rodbard (NIH Image)\|Gamma Variate\|Chapman-Richards\|Uncalibrated OD] unit= text1=[pixel values] text2=[standards] |
| "Close" | `file.close` |  |
| "Close All" | `file.close_all` |  |
| "Revert" | `file.revert` |  |
| "AuPbSn 40 (56K)" / "Bat Cochlea Volume (19K)" / "Bat Cochlea Renderings (449K)" / "Blobs (25K)" / "Boats (356K)" / "Bridge (174K)" / "Cardio (768K, RGB DICOM)" / "Cell Colony (31K)" / "Clown (14K)" / "Confocal Series (2.2MB)" / "Dot Blot (7K)" / "Embryos (42K)" / "Fluorescent Cells (400K)" / "Fly Brain (1MB)" / "Gel (105K)" / "HeLa Cells (1.3M, 48-bit RGB)" / "Image with Overlay (1.3M)" / "Leaf (36K)" / "Line Graph (21K)" / "M51 Galaxy (177K, 16-bits)" / "Mitosis (26MB, 5D stack)" / "Neuron (1.6M, 5 channels)" / "Nile Bend (1.9M)" / "Organ of Corti (2.8M, 4D stack)" / "Particles (75K)" / "T1 Head (2.4M, 16-bits)" / "T1 Head Renderings (736K)" / "TEM Filter (112K)" / "Tree Rings (48K)" / "AuPbSn 40" / "Bat Cochlea Volume" / "Bat Cochlea Renderings" / "Boats" / "Bridge" / "Cardio (RGB DICOM)" / "Cell Colony" / "Clown" / "Confocal Series" / "Dot Blot" / "Embryos" / "Fluorescent Cells" / "Flybrain" / "Gel" / "HeLa Cells" / "Image with Overlay" / "Leaf" / "Line Graph" / "M51 Galaxy" / "Mitosis" / "Neuron" / "Nile Bend" / "Organ of Corti" / "Particles" / "T1 Head" / "T1 Head Renderings" / "TEM Filter" / "Tree Rings" / "Blobs" / "Fly Brain" / "Cardio" | `file.sample / file.open_imagej_sample` | the 29 ImageJ sample names (with or without the size suffix): Blobs, T1 Head, Mitosis, Fluorescent Cells, HeLa Cells, Confocal Series, Clown and Cell Colony open the bundled offline Lumen stand-ins (file.sample); the others are downloaded from imagej.net/images (cached in ~/.lumen/samples); 'download' forces the original ImageJ file |
| "Duplicate..." | `image.duplicate` | 'duplicate'/'stack' → whole stack; else the current plane (all channels of the current slice for a composite-mode hyperstack, as ImageJ's Duplicator); range=first-last (plain stacks) or channels=/slices=/frames= (hyperstacks) with 'duplicate'; title= |
| "Scale..." | `image.scale` | x y z width height depth interpolation average create title |
| "Size..." | `image.scale` | width height depth → in-place resize |
| "Properties..." | `image.properties / stacks.to_hyperstack` | pixel_width pixel_height voxel_depth unit frame=[5 sec]; channels/slices/frames re-shape the stack; with a selection and only ROI keys (name stroke width fill position) it is the selection dialog ('Properties... ') |
| "Set Scale..." | `image.properties` | distance known pixel unit → pixel size = known/distance |
| "Threshold..." | `threshold.compute` | opens the threshold panel in ImageJ; here computes the Default threshold |
| "Auto Threshold" | `threshold.apply` | method=Otsu white [stack \| use_stack_histogram] → binary mask ('white' = white objects on black background); without 'stack' only the current slice is thresholded |
| "Convert to Mask" / "Make Binary" | `threshold.apply` | uses the current threshold (setThreshold/setAutoThreshold) or method=; background=Dark/Light; calculate → per plane |
| "Options..." | `binary.* ` | Process ▸ Binary ▸ Options: do=Erode/Dilate/Open/Close/Outline/Fill Holes/Skeletonize is executed, iterations → iterations |
| "Set Measurements..." | `analyze.set_measurements` | area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack limit decimal= display nan scientific add invert |
| "Analyze Particles..." | `analyze.particles` | size=min-max [pixel] circularity=min-max show= display clear summarize add exclude include stack |
| "Histogram" | `analyze.histogram / analyze.stack_histogram` | bins=; 'stack' → whole stack |
| "Image Calculator..." | `process.calculator` | image1= operation= image2= create 32-bit |
| "Merge Channels..." | `color.merge` | c1= … c7= create keep |
| "Make Composite" | `display mode` | display=Composite/Color/Grayscale → view mode |
| "Make Substack..." | `stacks.substack` | slices= channels= frames= (all given ranges apply together, like ImageJ's SubHyperstackMaker; slices= means frames on a time-lapse); delete (single range only) |
| "Re-order Hyperstack ..." / "Re-order Hyperstack..." | `stacks.reorder` | channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)] → the swapped pair |
| "Insert..." | `stacks.insert` | source= destination= x= y= |
| "Add Selection..." | `overlay.add / overlay.arrow` | stroke= width=; an arrow selection (makeArrow) becomes an arrow item |
| "Next Slice [>]" | `setSlice(n+1)` |  |
| "Previous Slice [<]" | `setSlice(n-1)` |  |
| "First Slice" | `setSlice(1)` |  |
| "Last Slice" | `setSlice(nSlices)` |  |
| "Save" | `session.save_image` | saves to the image's path (TIFF) |
| "Open..." | `session.open_file` | path= |
| "Tiff..." / "PNG..." / "Jpeg..." | `session.save_image` | path= (or save=) |
| "Color Threshold..." | `adjust.color_threshold` | best effort: space/method=HSB\|RGB\|Lab\|YUV, <component>_min/_max (hue, saturation, brightness / red, green, blue / l, a, b / y, u, v or c1..c3), <component>_filter=pass\|stop (or <component>=stop), output=Mask\|Selection\|Overlay |
| "Fit Spline" | `selection.interpolate` | smooth=True; interval → interval (default 5 px) |
| "Image Sequence..." | `file.import_sequence` | open=[folder or first file] / dir=[folder] → folder, file= / filter= → pattern, number → count, starting → start, increment → step, scale, use → virtual |
| "Cut" | `edit.cut` | all channels of the current plane; the selection is cleared to the background value |
| "Copy" | `edit.copy` | internal clipboard |
| "Paste" | `edit.paste_with_mode` | at the selection or centred; honours the Paste Control transfer mode (setPasteMode) |
| "Paste Control..." | `edit.paste_control` | transfer=Copy/Blend/Average/Difference/Transparent-white/Transparent-zero/AND/OR/XOR/Add/Subtract/Multiply/Divide/Min/Max (all 15 ImageJ transfer modes, Transparent-white included) |
| "Internal Clipboard" | `file.new_from_clipboard` | File ▸ New ▸ Internal Clipboard: the pixels copied with Copy / Cut become a new image |
| "Fit Circle" | `selection.fit_circle` | Edit ▸ Selection ▸ Fit Circle (no options) |
| "Fit Rectangle" | `selection.fit_rectangle` | Edit ▸ Selection ▸ Fit Rectangle: minimum-area rotated rectangle |
| "Select Bounding Box" / "Select Bounding Box (guess background color)" | `selection.select_bounding_box` | Fiji Edit ▸ Selection ▸ Select Bounding Box; the background is guessed from the image border |
| "Image to Selection..." | `selection.image_to_selection` | image= (title of an open mask / image) opacity= |
| "Restore Selection" | `ij.last_roi` | re-activates the previous selection (the one removed by Select None, Fill, Measure of another image …); frontend-only in ImageJ |
| "Properties... " | `selection.properties` | Edit ▸ Selection ▸ Properties (trailing space, as in ImageJ): name= stroke= (colour) width= fill= position=none\|n\|c,z,t; group= is stored on the ROI |
| "Colors..." | `setForegroundColor / setBackgroundColor / edit.options_roi_defaults` | Edit ▸ Options ▸ Colors: foreground= background= selection= (colour names) |
| "Input/Output..." | `edit.options_io` | Edit ▸ Options ▸ Input/Output: jpeg= → jpeg_quality (gif= file= use_file copy_column copy_row save_column save_row are ignored) |
| "Point Tool..." | `edit.options_point_tool` | Edit ▸ Options ▸ Point Tool: type=Hybrid\|Cross\|Dot\|Circle color= size=Tiny\|Small\|Medium\|Large\|[Extra Large]\|XXL\|XXXL (label show counter= auto-measure auto-next add are ignored) |
| "Arrow Tool..." | `edit.options_arrow_tool` | Edit ▸ Options ▸ Arrow Tool: width= size= (head size) style=Filled\|Notched\|Open\|Headless\|Bar double; color= outline keep are ignored |
| "Fonts..." | `edit.options_fonts` | Edit ▸ Options ▸ Fonts is a non-modal panel in ImageJ (recorded as setFont); accepts font= size= style=[bold italic antialiased] antialiased |
| "Appearance..." | `edit.options_appearance` | Edit ▸ Options ▸ Appearance: interpolate → interpolate_zoom, auto (auto contrast stacks), antialiased; open black no use menu= gui= 16-bit= are ignored |
| "Plots..." / "Profile Plot Options..." | `edit.options_plots` | Edit ▸ Options ▸ Plots: width= height= minimum= maximum= interpolate (font= draw sub-pixel auto-close list vertical fixed are ignored) |
| "Rounded Rect Tool..." | `edit.options_rounded_rect` | Edit ▸ Options ▸ Rounded Rect Tool: corner= is the corner *diameter* (→ radius = corner/2), width= stroke width; stroke= fill= colours are ignored |
| "Misc..." | `edit.options_misc` | Edit ▸ Options ▸ Misc: divide=Infinity\|max\|NaN\|0 → divide_by_zero, require (command key for shortcuts); use hide move run enhanced reverse debug are ignored |
| "Open Next" | `file.open_next` | File ▸ Open Next: the next image file of the current image's folder replaces it |
| "LUT... " / "LUT..." | `file.import_lut / file.export_lut` | open=[file.lut] → File ▸ Import ▸ LUT (ImageJ: 'LUT... ' with a trailing space); save=[file.lut] → File ▸ Save As ▸ LUT |
| "Text File... " / "Text File..." | `file.import_text` | File ▸ Import ▸ Text File: open=[file.txt] → table + Log |
| "Results... " / "Results..." | `file.import_results / file.export_results` | open=[file.csv] → File ▸ Import ▸ Results (replaces the table); save=[file.csv\|.tsv\|.xlsx] → File ▸ Save As ▸ Results |
| "Table... " / "Table..." | `file.import_table / table.new` | File ▸ Import ▸ Table: open=[file.csv] (name= optional); without open= it is File ▸ New ▸ Table: name= headings=[A,B,C] |
| "XY Coordinates... " / "XY Coordinates..." | `file.import_xy_coordinates / file.save_xy_coordinates` | open=[file.txt] → File ▸ Import ▸ XY Coordinates (polygon selection; type=polyline\|point); save=[file] → File ▸ Save As ▸ XY Coordinates |
| "Stack From List..." | `file.import_stack_from_list` | File ▸ Import ▸ Stack From List: open=[list.txt] (use → virtual stack is ignored) |
| "ZIP..." | `file.export_zip` | File ▸ Save As ▸ ZIP: save=[file.zip] (an ImageJ TIFF inside the archive) |
| "Text..." | `file.export_text` | File ▸ Save As ▸ Text: save=[file.txt]; saves the Results table when it has rows, else the Log (source=Results\|Table\|Log\|Macro to choose) |
| "Raw..." | `file.import_raw` | File ▸ Import ▸ Raw: open= image=[16-bit Unsigned] width= height= offset= number= gap= little-endian white open (all files) use |
| "Text Image... " / "Text Image..." | `file.import_text_image` | File ▸ Import ▸ Text Image: open=[file.txt] (saving a text image is not available) |
| "URL..." | `file.open_url` | File ▸ Import ▸ URL: url= |
| "Image Sequence... " | `file.export_sequence` | File ▸ Save As ▸ Image Sequence (trailing space): dir=[folder] format=TIFF\|PNG\|JPEG\|BMP\|PGM name= digits= start=; GIF → PNG, ZIP → TIFF |
| "AVI... " | `file.export_video` | File ▸ Save As ▸ AVI (trailing space): save=[file.avi] frame= (fps); compression= is ignored; needs the optional [video] extra |
| "8-bit Color" | `type.8bit_color / type.8bit` | number= colours (median cut) for RGB images; grayscale images are converted to 8-bit |
| "Auto Crop" / "Auto Crop (guess background color)" | `image.auto_crop` | Fiji Image ▸ Crop ▸ Auto Crop: trims to the bounding box of everything that differs from the border colour |
| "Show LUT" | `color.show_lut` | Image ▸ Color ▸ Show LUT (no options) |
| "Display LUTs" | `color.display_luts` | Image ▸ Color ▸ Display LUTs (no options) |
| "Edit LUT..." | `color.edit_lut` | interactive in ImageJ (no options → nothing happens); name= start= end= or table=[r g b lines] build and apply a LUT |
| "Montage to Stack..." | `stacks.montage_to_stack` | Image ▸ Stacks ▸ Tools ▸ Montage to Stack: columns= rows= (ImageJ's StackMaker: images_per_row= images_per_column=) border= |
| "Remove Slice Labels" | `stacks.remove_slice_labels` | Image ▸ Stacks ▸ Tools ▸ Remove Slice Labels |
| "Set Label..." | `stacks.set_slice_label` | Image ▸ Stacks ▸ Tools ▸ Set Label: label= → the current plane's label |
| "Statistics" | `stacks.statistics` | Image ▸ Stacks ▸ Statistics: whole-stack voxel statistics (table 'Stack Statistics' + Results row) |
| "Radial Reslice" | `stacks.radial_reslice` | Fiji Image ▸ Stacks ▸ Radial Reslice: angle= degrees_per_slice= (→ n_angles = angle/degrees_per_slice); direction= and rotate are ignored (slices run counter-clockwise from the line / point selection) |
| "Reslice [/]..." / "Reslice [/]... " / "Reslice..." | `stacks.reslice / stacks.reslice_line` | output= (spacing) start=Top\|Bottom\|Left\|Right avoid (no Z interpolation) flip rotate; with a line / polyline selection: slice_count= rotate (ImageJ Slicer) |
| "Hyperstack to Stack" | `stacks.hyperstack_to_stack` | flattens channels/slices/frames into one Z stack (ImageJ order c, z, t) with 'c:… z:… t:…' slice labels; a plain stack or RGB image is left as it is |
| "Reduce Dimensionality..." | `stacks.reduce_dimensionality` | channels slices frames keep (ImageJ HyperStackReducer: a missing keyword drops that dimension to the current position; keep = keep the source) |
| "Make Subset..." | `stacks.make_subset` | ImageJ SubHyperstackMaker: channels=1-2 slices=1-5-2 frames=1,3 (all ranges apply together) |
| "Image to Results" | `image.to_results` | Image ▸ Transform ▸ Image to Results: pixel values of the current plane (or selection bounds) → Results |
| "Results to Image" | `image.from_results` | Image ▸ Transform ▸ Results to Image: numeric Results columns → 32-bit image |
| "Interactive Affine" / "Interactive Rigid" / "Interactive Similarity" / "Interactive Perspective" | `image.transform_landmarks` | Fiji's interactive landmark transforms have no macro options; here source=[x1,y1; x2,y2; …] (or the point selection / source_roi=) target=[…] (or target_roi=) interpolation= create |
| "Polar Transformer" | `image.warp_polar` | Fiji Polar Transformer: method=Polar (Cartesian is not available) number= (angle samples) default_center \| polar=[x y] / center_x= center_y= radius= log; degrees= is ignored (always 360°) |
| "Add Image..." | `overlay.add_image` | Image ▸ Overlay ▸ Add Image: image= x= y= opacity= (zero / transparent background is ignored) |
| "Grid..." | `overlay.grid` | Analyze ▸ Tools ▸ Grid: grid=Lines\|[Horizontal Lines]\|Crosses\|Points\|Combined\|None area= color= bold random (center is ignored); None removes the grid |
| "Dither" | `binary.dither` | Floyd–Steinberg dithering of the current plane ('stack' → all planes) |
| "Show Circular Masks" | `filters.show_circular_masks` | Process ▸ Filters ▸ Show Circular Masks (no options) |
| "Normalize Local Contrast" | `filters.normalize_local_contrast` | Fiji Process ▸ Filters: block_radius_x= block_radius_y= standard_deviations= center stretch stack |
| "ROF Denoise" | `filters.rof_denoise` | Fiji Process ▸ Filters ▸ ROF Denoise: theta= |
| "Retinex" | `process.retinex` | Fiji Process ▸ Retinex: level=Uniform\|Low\|High scale= scale_division= dynamic= |
| "Shape Index Map" | `filters.shape_index` | Fiji Process ▸ Shape Index Map: sigma= (gaussian= / radius= accepted) |
| "Exact Euclidean Distance Transform (3D)" / "Exact Signed Euclidean Distance Transform (3D)" | `binary.distance_map_3d` | Fiji Process ▸ Binary: 3-D EDT (32-bit, calibrated); the signed variant is approximated by the unsigned transform |
| "Measure..." / "Batch Measure" | `batch.measure` | Process ▸ Batch ▸ Measure: input=[folder] (dir= / folder= / open=) file= (name filter) threshold= (Lumen: auto-threshold before measuring) |
| "FFT Options..." | `process.fft_options` | Process ▸ FFT ▸ FFT Options: fft (display 8-bit FFT window) raw (raw power spectrum) complex (complex output) do (do forward transform now); fast (FHT) is ignored, window = Hann window (Lumen) |
| "Redisplay Power Spectrum" | `process.fft_redisplay_power_spectrum` | Process ▸ FFT ▸ Redisplay Power Spectrum (no options) |
| "Inverse FFT" | `process.inverse_fft` | Process ▸ FFT ▸ Inverse FFT of a power-spectrum image (black / white painted regions filter); '32-bit' keeps a float result (Lumen) |
| "Swap Quadrants" | `process.fft_swap_quadrants` | Process ▸ FFT ▸ Swap Quadrants ('stack' → all planes) |
| "FD Math..." | `process.fd_math` | Process ▸ FFT ▸ FD Math: image1= operation=Correlate\|Convolve\|Deconvolve image2= result= do (inverse transform; without it the result stays in the frequency domain) |
| "Custom Filter..." | `process.fft_custom_filter` | Process ▸ FFT ▸ Custom Filter: filter=[filter image title] ('stack' / process → every plane) |
| "Analyze Line Graph" | `analyze.line_graph` | Analyze ▸ Tools ▸ Analyze Line Graph (no options); a rectangle selection limits the plot area |
| "Directionality" | `analyze.directionality` | Fiji Analyze ▸ Directionality: method=[Local gradient orientation]\|[Fourier components] nbins= histogram_start= (histogram=) ; histogram_end display_table are ignored; stacks are processed plane by plane |
| "Gel Analyzer Options..." | `(ij.gel_options)` | Analyze ▸ Gels ▸ Gel Analyzer Options: vertical / horizontal (lane direction) invert (peaks) label (with percentages); uncalibrated outline are ignored |
| "Select First Lane" / "Select Next Lane" | `analyze.gel_select_lane` | Analyze ▸ Gels: the rectangle selection becomes lane 1 (Select First Lane drops the previous lanes of the image) / the next lane |
| "Plot Lanes" | `analyze.gel_plot_lanes` | Analyze ▸ Gels ▸ Plot Lanes: the lanes selected with Select First/Next Lane (honours Gel Analyzer Options) |
| "Label Peaks" | `analyze.gel_label_peaks` | Analyze ▸ Gels ▸ Label Peaks: band areas (or percentages with 'Label with percentages') as Results rows + labels |
| "Analyze Skeleton (2D/3D)" / "Analyze Skeleton" | `analyze.skeleton` | Fiji AnalyzeSkeleton: prune=[none\|shortest branch\|lowest intensity voxel\|lowest intensity branch] (intensity variants → shortest branch) prune_0 (prune ends) ; calculate show display exclude are ignored; stacks are analysed plane by plane; skeletonize (Lumen) thins the mask first |
| "Coloc 2" | `analyze.coloc_costes` | Fiji Coloc 2: channel_1= channel_2= (image titles; two images are combined into a temporary 2-channel image, one multi-channel image uses channel_a=/channel_b=) psf= (→ block size) costes_randomisations= ; roi_or_mask= threshold_regression= and the display check-boxes are ignored |
| "Elastic Stack Alignment" | `registration.elastic_align` | Fiji Plugins ▸ Registration ▸ Elastic Stack Alignment: best effort – the slices/frames are aligned onto the previous one with dense optical flow; Fiji's block-matching parameters (layer_scale search_radius block_radius …) have no equivalent and are ignored |
| "bUnwarpJ" | `registration.bunwarpj` | Fiji bUnwarpJ: source_image= (made current) target_image= registration=[Fast\|Accurate\|Mono] image_subsampling_factor= initial_deformation=[Very Coarse\|Coarse\|Fine\|Very Fine] final_deformation=[Very Coarse\|Coarse\|Fine\|Very Fine\|Super Fine] divergence_weight= curl_weight= landmark_weight= image_weight= consistency_weight= stop_threshold= save_transformations (→ deformation field image) save_direct= save_inverse= (transform files); verbose is ignored |
| "Register Virtual Stack Slices" | `registration.align_folder` | Fiji Plugins ▸ Registration ▸ Register Virtual Stack Slices: source=[folder] output=[folder] registration=[Translation\|Rigid\|Similarity\|Affine\|Elastic\|Moving least squares …] (feature= when registration= is absent; Similarity → Affine, Moving least squares → Elastic); advanced shrinkage save are ignored |
| "Phase Correlation" | `registration.phase_correlation` | Fiji Stitching ▸ Phase Correlation: first_image= (made current) second_image= (target); subpixel_accuracy → 10× sub-pixel, else integer shifts; number_of_peaks= and the fusion options are ignored |
| "Pairwise stitching" | `registration.pairwise_stitch` | Fiji Stitching ▸ Pairwise stitching: first_image= second_image= fusion_method=[Linear Blending\|Max. Intensity\|Average\|Overlay into composite image] fused_image= (title) check_peaks= |
| "Extract SIFT Correspondences" / "Extract MOPS Correspondences" | `registration.extract_correspondences` | Fiji Plugins ▸ Feature Extraction: source_image= (made current) target_image= initial_gaussian_blur= steps_per_scale_octave= minimum_image_size= maximum_image_size= feature_descriptor_size= feature_descriptor_orientation_bins= closest/next_closest_ratio= maximal_alignment_error= inlier_ratio= expected_transformation=[Translation\|Rigid\|Similarity\|Affine\|Perspective] → SIFT / MOPS matches as point ROIs on both images and a Correspondences table; filter is implied |
| "Load a TrackMate file" | `tracking.import_trackmate_xml` | Fiji Plugins ▸ Tracking ▸ Load a TrackMate file: open=[file.xml] → tracks of the current image |
| "Color Clustering" | `segment.color_clustering` | Fiji Plugins ▸ Segmentation ▸ Color Clustering is interactive; number_of_clusters= (clusters=) color_space=RGB\|Lab\|HSB → k-means label image + 'Clusters' table |
| "About Plugins" / "About Plugins..." | `help.about_plugins` | Help ▸ About Plugins: table of the loaded command modules |
| "Bio-Formats Importer" / "Bio-Formats" / "Bio-Formats (Windowless)" / "Bio-Formats Windowless Importer" | `file.bioformats_import` | open=[file] color_mode=Default\|Composite\|Colorized\|Grayscale\|Custom view=Hyperstack\|[Standard ImageJ]\|[Data Browser] stack_order= series_N (1-based) / series_list= / open_all_series autoscale split_channels split_focal_planes split_timepoints group_files (name=[pattern]) virtual\|use_virtual_stack specify_range c_begin_1= c_end_1= c_step_1= z_* t_* crop x_coordinate_1= y_coordinate_1= width_1= height_1= display_metadata display_ome-xml swap_dimensions z_1= c_1= t_1= |
| "Bio-Formats Remote Importer" | `file.bioformats_remote_import` | url=[http://…] (or open=) plus the Bio-Formats Importer options |
| "Bio-Formats Exporter" | `file.bioformats_export` | save=[file.ome.tif] compression=Uncompressed\|LZW\|zlib\|JPEG\|JPEG-2000\|PackBits\|Zstd write_each_z_section write_each_timepoint write_each_channel padded; .zarr → OME-Zarr, any .tif → OME-TIFF |
| "TIFF Virtual Stack..." | `file.import_tiff_virtual` | File ▸ Import ▸ TIFF Virtual Stack: open=[file.tif] (series= level= optional) |
| "Stack Slicer" | `file.stack_slicer` | Plugins ▸ Bio-Formats ▸ Stack Slicer: split_channels split_z split_timepoints stack_order= keep_original |
| "Bio-Formats Plugins Configuration" | `file.bioformats_config` | → the 'Bio-Formats Plugins Configuration' table (formats, readers, availability) |
| "Bio-Formats Macro Extensions" | `Ext.*` | enables the Ext.setId / getSeriesCount / setSeries / getSizeX … / openImage macro functions (see IJ.Ext) |
| "AVI..." | `session.open_file / stacks.substack` | File ▸ Import ▸ AVI: open=[file.avi] first= last= (frame range), convert → 8-bit grayscale; use (virtual) is accepted |
| "Fit Circle to Image" | `selection.fit_circle_to_image` | Edit ▸ Selection ▸ Fit Circle to Image: circle through the thresholded / non-zero pixels (method= optional) |
| "Points from Mask" | `selection.points_from_mask` | Edit ▸ Selection ▸ Points from Mask: step= max_points= → multi-point selection of the mask pixels |
| "Labels..." | `analyze.label` | Image ▸ Overlay ▸ Labels…: color= font= show use (names as labels) bold draw (label background) → index / name labels of the overlay items |
| "Measure Overlay" | `overlay.measure` | Image ▸ Overlay ▸ Measure Overlay: one Results row per overlay item |
| "Overlay Options..." | `edit.options_overlay` | Image ▸ Overlay ▸ Overlay Options…: width= fill= show (labels) use (names) font= color= background; the stroke colour and 'apply' are ignored |
| "Search..." | `help.search_files` | Plugins ▸ Utilities ▸ Search…: search= (or _=) case (sensitive); files, images, the Log and the script are all searched |
| "Linear Gradient" | `edit.linear_gradient` | Image ▸ Drawing ▸ Linear Gradient: angle= (foreground → background colour) |
| "Radial Gradient" | `edit.radial_gradient` | Image ▸ Drawing ▸ Radial Gradient: radius= (0 = to the corners) |
| "Update ImageJ..." / "Update..." / "Update Fiji..." | `help.update` | Help ▸ Update…: check (only report what is available) dry (dry run) → upgrades the Lumen package with pip |
| "Text Window" / "Text Window..." | `file.new_text` | File ▸ New ▸ Text Window: name= text=; print("[name]", s) appends to an open text window |
| "Labels To RGB" | `morpho.labels_to_rgb` | MorphoLibJ Labels ▸ Labels To RGB: colormap=[Glasbey\|Glasbey Dark\|Golden Angle\|Mixed Colors\|Grays\|Fire\|Ice\|Spectrum\|…] background=White\|Black shuffle |
| "Assign Measure to Label" | `morpho.assign_measure_to_label` | MorphoLibJ Labels ▸ Assign Measure to Label: results=[table] column=[name] → the label image coloured by the measurement |
| "Microstructure Analysis" / "Microstructure Analysis 3D" | `morpho.microstructure_analysis` | MorphoLibJ Analyze ▸ Microstructure Analysis (3D): area\|volume perimeter\|surface mean_breadth euler flags (none = all), directions=2\|4 (3\|13 in 3D) connectivity=4\|8 (6\|26) |
| "Analyze Regions 3D" | `morpho.analyze_regions_3d` | MorphoLibJ Analyze ▸ Analyze Regions 3D: volume surface_area sphericity euler_number bounding_box centroid equivalent_ellipsoid ellipsoid_elongations max._inscribed mean_breadth flags (none = all), surface_area_method=[Crofton (3 dirs.)\|Crofton (13 dirs.)] euler_connectivity=C6\|C26 |
| "Morphological Filters" | `morpho.filters` | MorphoLibJ Filters ▸ Morphological Filters: operation=[Erosion\|Dilation\|Opening\|Closing\|White Top Hat\|Black Top Hat\|Gradient\|Laplacian\|Internal Gradient\|External Gradient] element=[Disk\|Square\|Diamond\|Octagon\|Horizontal Line\|Vertical Line\|Line 45 degrees\|Line 135 degrees] radius= (stack → every plane) |
| "Morphological Filters (3D)" | `morpho.filters_3d` | MorphoLibJ Filters ▸ Morphological Filters (3D): operation= element=[Ball\|Cube\|Cross] x-radius= y-radius= z-radius= |
| "Regional Min & Max" / "Regional Min & Max 3D" | `morpho.regional_min_max` | MorphoLibJ Minima & Maxima ▸ Regional Min & Max (3D): operation=[Regional Minima\|Regional Maxima] connectivity=4\|8 (6\|26) |
| "Extended Min & Max" / "Extended Min & Max 3D" | `morpho.extended_min_max` | MorphoLibJ Minima & Maxima ▸ Extended Min & Max (3D): operation=[Extended Minima\|Extended Maxima] dynamic= connectivity= |
| "Impose Min & Max" / "Impose Min & Max 3D" | `morpho.impose_min_max` | MorphoLibJ Minima & Maxima ▸ Impose Min & Max (3D): original= (made current) marker= operation=[Impose Minima\|Impose Maxima] connectivity= |
| "Chamfer Distance Map" / "Chamfer Distance Map 3D" | `morpho.chamfer_distance_map` | MorphoLibJ Binary Images ▸ Chamfer Distance Map (3D): distances=[Chessboard (1,1)\|City-Block (1,2)\|Quasi-Euclidean (1,1.41)\|Borgefors (3,4)\|Chessknight (5,7,11)] output=[16 bits\|32 bits] normalize |
| "Geodesic Distance Map" | `morpho.geodesic_distance_map` | MorphoLibJ Binary Images ▸ Geodesic Distance Map: marker= mask= (made current) distances=[…] normalize |
| "Marker-controlled Watershed" | `morpho.marker_watershed` | MorphoLibJ Segmentation ▸ Marker-controlled Watershed: input= (made current) marker= mask= compactness= binary calculate (dams) use (diagonal connectivity); 3-D when the input is a Z stack |
| "Morphological Reconstruction" / "Morphological Reconstruction 3D" | `morpho.reconstruct` | MorphoLibJ Morphological Reconstruction (3D): marker= mask= (made current) type=[By Dilation\|By Erosion] connectivity= |
| "Kill Borders" | `morpho.kill_borders` | MorphoLibJ Binary Images ▸ Kill Borders (stack → every plane) |
| "Fill Holes (Binary/Gray)" | `morpho.fill_holes` | MorphoLibJ Binary Images ▸ Fill Holes (Binary/Gray): grayscale hole filling by reconstruction |
| "Area Opening" | `morpho.area_opening` | MorphoLibJ Binary Images ▸ Area Opening: pixel= (minimum area) |
| "Directional Filtering" | `morpho.directional_filter` | MorphoLibJ Filters ▸ Directional Filtering: type=Max\|Min operation=[Opening\|Closing\|Erosion\|Dilation\|Mean\|Median] line= (length) direction= (number of directions) |
| "Binary Overlay" | `morpho.binary_overlay` | MorphoLibJ Binary Images ▸ Binary Overlay: reference= (made current) binary= overlay=Red\|Green\|… (colour) |
| "Label Overlay" | `morpho.label_overlay` | MorphoLibJ Labels ▸ Label Overlay: reference= (made current) labels= opacity= (%) |
| "Merge Labels" | `morpho.merge_labels` | MorphoLibJ Label Edition ▸ Merge Labels (interactive in Fiji): label(s)=1,2,3 (or labels=) merged into the first one (new_value= optional) |
| "Label Size Opening" | `morpho.label_size_opening` | MorphoLibJ Labels ▸ Label Size Opening: min_size= (pixels / voxels) |
| "Select Label(s)" | `morpho.select_label` | MorphoLibJ Labels ▸ Select Label(s): label(s)=1,2,3 → the outline of each listed label becomes a selection (several → all added to the ROI Manager, the last one stays selected) |
| "Remove Label(s)" | `morpho.replace_label` | MorphoLibJ Labels ▸ Remove Label(s): label(s)=1,2,3 → set to 0 |
| "Linear Stack Alignment with SIFT" / "Linear Stack Alignment with SIFT MultiChannel" | `registration.align_stack` | Fiji Plugins ▸ Registration ▸ Linear Stack Alignment with SIFT: initial_gaussian_blur= steps_per_scale_octave= minimum_image_size= maximum_image_size= feature_descriptor_size= feature_descriptor_orientation_bins= closest/next_closest_ratio= maximal_alignment_error= inlier_ratio= expected_transformation=[Translation\|Rigid\|Similarity\|Affine] interpolate show_info show_transformation_matrix registration_channel= → SIFT alignment onto the previous slice as a new stack |
| "StackReg" | `registration.align_stack` | Fiji Plugins ▸ Registration ▸ StackReg: transformation=[Translation\|Rigid Body\|Scaled Rotation\|Affine\|Bilinear] → intensity-based pyramid alignment onto the current slice, in place |
| "Descriptor-based registration (2d/3d)" | `registration.descriptor_registration` | Fiji Plugins ▸ Registration ▸ Descriptor-based registration (2d/3d): first_image= (made current) second_image= (target) brightness_of_objects=[Very low\|Low\|Medium\|Strong\|Very strong] approximate_size=[5 px] type_of_detections=[Maxima only\|Minima only\|Minima & Maxima] transformation_model=[Translation\|Rigid\|Similarity\|Affine (2d)] number_of_neighbors= redundancy= significance= allowed_error_for_ransac= choose_registration_channel_for_image_1= → registered copy of the first image |
| "Descriptor-based series registration (2d/3d + t)" | `registration.descriptor_series` | Fiji Plugins ▸ Registration ▸ Descriptor-based series registration (2d/3d + t): the detection / model / RANSAC keys of the pairwise command plus register=[… first frame …] (else onto the previous frame) choose_registration_channel= → the stack is aligned in place |
| "Phase Map" | `timelapse.phase_map` | Fiji Plugins ▸ Time Lapse ▸ Phase Map: min_period= max_period= detrend=[Mean\|Linear\|None] window phase=[Degrees\|Radians] output=[Period (frames)\|Frequency (cycles/frame)\|Frequency (Hz)\|Period (seconds)] reference=[None\|Mean signal\|Selection] |
| "Trainable Weka Segmentation" / "Trainable Weka Segmentation 3D" | `segment.train_classifier / segment3d.train_classifier` | Fiji Plugins ▸ Segmentation ▸ Trainable Weka Segmentation (3D): interactive in Fiji – with options (classes= features=[Gaussian_blur Sobel_filter …] min_sigma= max_sigma= classifier=[…FastRandomForest] n_trees= output=) it trains on the class-named ROIs of the manager right away; without options it only opens the Weka session for the call("trainableSegmentation.Weka_Segmentation.*") statics (createNewClass, changeClassName, addTrace, setFeature, setMinimumSigma, setMaximumSigma, setClassifier, trainClassifier, getResult, getProbability, loadClassifier, saveClassifier, applyClassifier, toggleOverlay, setOpacity) |
| "Grid/Collection stitching" | `registration.stitch_collection` | Fiji Plugins ▸ Stitching ▸ Grid/Collection stitching: type=[Grid: row-by-row\|Grid: column-by-column\|Grid: snake by rows\|Grid: snake by columns\|Positions from file\|Unknown position] grid_size_x= grid_size_y= tile_overlap= directory= file_names=tile_{ii}.tif layout_file= fusion_method=[Linear Blending\|Average\|Median\|Max. Intensity\|Min. Intensity\|Overlay into composite image\|Do not fuse images …] regression_threshold= max/avg_displacement_threshold= absolute_displacement_threshold= compute_overlap subpixel_accuracy downsample_tiles image_output=[Fuse and display\|Write to disk] output_directory=; computation_parameters= use_virtual_input_images ignore_z_stage invert_x invert_y are ignored (no 'Filename defined position' / 'Sequential images' types) |
| "Elastic Montaging" | `registration.elastic_montage` | Fiji Plugins ▸ Registration ▸ Elastic Montaging (on the open images): stiffness= minimal_pmcc_r= (→ min_correlation) layer_scale= (→ downscale) resolution= (→ mesh spacing); search_radius= block_radius= maximal_curvature_ratio= maximal_second_best_r/best_r= maximal_stretch= maximal_iterations= maximal_plateauwidth= are ignored |
| "Landmark Correspondences" | `image.transform_landmark_correspondences` | Fiji Plugins ▸ Transform ▸ Landmark Correspondences: source_image= (made current) template_image= (point selections on both, or source=[x1,y1; …] target=[…] / source_roi= target_roi=) transformation_method=[Translation\|Rigid\|Similarity\|Affine\|Perspective\|Moving Least Squares (non-linear)] transformation_class=[Rigid\|Similarity\|Affine] (MLS) alpha= interpolate; mesh_resolution= is ignored |
| "Moving Least Squares" / "Interactive Moving Least Squares" | `image.transform_mls` | Fiji Plugins ▸ Transform ▸ Interactive Moving Least Squares (no macro options in Fiji): source=[x1,y1; x2,y2; …] (or the point selection / source_roi=) target=[…] (or target_roi=) model=[Affine\|Similarity\|Rigid] alpha= interpolation= create |
| "<LUT name>" (Grays, Red, Green, Blue, Cyan, Magenta …) | `color.lut` | Any Lumen LUT name |

**Accepted as no-ops** (interactive dialogs and window management that have no meaning in a
single-window UI, or whose job another Lumen feature does — Manual Tracking → the `tracking.manual_*` commands, 3D Viewer → the 3-D View section, ROI Manager… / Threshold Panel → the inspector sections, Capture Screen / Capture Image → the Plugins ▸ Utilities menu entries in the browser): "Brightness/Contrast..." / "Window/Level..." / "Color Balance..." / "Channels Tool..." / "Show Overlay" / "Hide Overlay" / "Tile" / "Cascade" / "Put Behind [tab]" / "Show Info..." / "Conversions..." / "Line Width... " / "Memory & Threads..." / "Proxy Settings..." / "Compiler..." / "Wand Tool..." / "Collect Garbage" / "Monitor Memory..." / "ImageJ [NEW]..." / "Record..." / "Startup..." / "Reset..." / "Select Window..." / "Animation Options..." / "Synchronize Windows" / "Manual Tracking" / "3D Viewer" / "Capture Screen" / "Capture Image" / "Help on Menu Item" / "Help on Menu Item..." / "ROI Manager..." / "Threshold Panel" / "Control Panel...".

**Declared unsupported:** none. The Bio-Formats Importer / Exporter / Remote Importer, TIFF Virtual Stack and AVI dialogs that
used to stop a macro now run through `file.bioformats_import`, `file.bioformats_export`, `file.bioformats_remote_import`,
`file.import_tiff_virtual` and `session.open_file` with their option strings mapped (series, colour mode, view, autoscale,
splitting, ranges, crop, virtual, compression, per-plane files, frame range), and `run("Bio-Formats Macro Extensions")` enables
the `Ext.*` functions (`Ext.setId`, `getSeriesCount`, `setSeries`, `getSizeX` … `getSizeT`, `getImageCount`, `getMetadataValue`,
`getPlaneTimingDeltaT`, `getIndex`, `getZCTCoords`, `openImage`, `openImagePlus` …) over the native readers. A format that still
needs Java Bio-Formats (see the Importer row under File) fails with a clear error at `open=`.

**171 macro functions implemented:** `abs`, `acos`, `asin`, `atan`, `atan2`, `beep`, `bitDepth`, `call`, `changeValues`, `charCodeAt`, `checksum`, `close`, `cos`, `currentMemory`, `d2s`, `deleteRows`, `doCommand`, `drawLine`, `drawOval`, `drawRect`, `drawString`, `endsWith`, `eval`, `exit`, `exp`, `fill`, `fillOval`, `fillRect`, `floodFill`, `floor`, `freeMemory`, `fromCharCode`, `getArgument`, `getBoolean`, `getBoundingRect`, `getCursorLoc`, `getDateAndTime`, `getDimensions`, `getDir`, `getDirectory`, `getFileList`, `getFullVersion`, `getHeight`, `getHistogram`, `getImageID`, `getImageInfo`, `getInfo`, `getLine`, `getList`, `getLocationAndSize`, `getLut`, `getMetadata`, `getMinAndMax`, `getNumber`, `getPixel`, `getPixelSize`, `getRawStatistics`, `getResult`, `getResultLabel`, `getResultString`, `getSelectionBounds`, `getSelectionCoordinates`, `getSliceNumber`, `getStatistics`, `getString`, `getStringWidth`, `getThreshold`, `getTime`, `getTitle`, `getToolName`, `getValue`, `getVersion`, `getVoxelSize`, `getWidth`, `getZoom`, `indexOf`, `is`, `isActive`, `isKeyDown`, `isNaN`, `isOpen`, `lastIndexOf`, `lengthOf`, `log`, `makeArrow`, `makeEllipse`, `makeLine`, `makeOval`, `makePoint`, `makePolygon`, `makeRectangle`, `makeRotatedRectangle`, `makeSelection`, `makeText`, `matches`, `maxMemory`, `maxOf`, `minOf`, `nImages`, `nResults`, `nSlices`, `newArray`, `newImage`, `open`, `pad`, `parseFloat`, `parseInt`, `pow`, `print`, `random`, `redirectErrorMessages`, `rename`, `renameResults`, `replace`, `requires`, `reset`, `resetMinAndMax`, `resetThreshold`, `restoreSelection`, `roiManager`, `round`, `run`, `runMacro`, `save`, `saveAs`, `selectImage`, `selectWindow`, `selectionName`, `selectionType`, `setAutoThreshold`, `setBackgroundColor`, `setBatchMode`, `setColor`, `setFont`, `setForegroundColor`, `setJustification`, `setKeyDown`, `setLineWidth`, `setLocation`, `setLut`, `setMetadata`, `setMinAndMax`, `setOption`, `setPasteMode`, `setPixel`, `setResult`, `setSelectionName`, `setSlice`, `setThreshold`, `setTool`, `setVoxelSize`, `showMessage`, `showMessageWithCancel`, `showProgress`, `showStatus`, `sin`, `snapshot`, `split`, `sqrt`, `startsWith`, `substring`, `tan`, `toBinary`, `toHex`, `toLowerCase`, `toString`, `toUpperCase`, `toolID`, `updateDisplay`, `updateResults`, `wait`.

Design note: ImageJ's option strings depend on global preferences (black background,
scale-when-converting, measurement set), which is exactly what Lumen's per-command parameters
remove; the layer therefore maps each option string to explicit parameters, and anything without
a Lumen equivalent (an option no command takes, a format that still needs Java Bio-Formats) fails loudly rather than skipping the step.


## Deliberately different

These are not gaps; they are the places where Lumen's design ([DESIGN.md](DESIGN.md),
[RESEARCH.md](RESEARCH.md)) intentionally departs from Fiji.

| Fiji | Lumen | Why |
|---|---|---|
| **Window soup** – every image, dialog, table, plot and the ROI Manager is a floating window; the *Window* menu exists to tile and cascade them | **Tabs + inspector** – images are tabs in one viewport; Results, ROI Manager, Histogram, Plot, History, Script & Recorder, Log, Orthogonal Views and 3-D View are inspector sections; parameter sheets dock at the top-right of the viewport | "difficulty in managing multiple windows is a general problem" ([Image.sc 68201](https://forum.image.sc/t/manage-multiple-windows-in-imagej/68201)); see RESEARCH.md §1 and §14 |
| **Black background option** (`Process ▸ Binary ▸ Options`) silently decides what erode/dilate/watershed act on, and it persists across sessions | **Masks are always foreground = 255, background = 0.** Commands that need to know what is "object" take an explicit `dark_background` / `white_objects` parameter per call | the ImageJ docs themselves warn that binary commands "assume that objects are black and background is white unless 'Black Background' is checked" ([Process menu](https://imagej.net/ij/docs/menus/process.html)) |
| **ImageJ macro language** with a recorder that emits `run("Command", "key=value")` | **Python** console with a recorder that emits `lumen.run("command.id", param=value)`; numpy / scipy / scikit-image are in scope; plus an **IJ1 compatibility layer** (`script.run_ij_macro`) for existing macros | surveys: ¾ of users already script ([Sivagurunathan 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10274673/)); one registry drives palette, menus, recorder and API |
| **Brightness/Contrast, Window/Level, Color Balance, Threshold, Channels Tool** are five floating dialogs that must be reopened per image; *Apply* rewrites pixels | **Display section** of the inspector: per-channel histogram slider, LUT, gamma, visibility, composite/single/grayscale mode, always visible for the active image; *Apply* is a separate, undoable command (`adjust.brightness_contrast_apply`) | non-destructive by default, like Photoshop adjustment layers and OMERO.iviewer rendering settings (RESEARCH.md §11, §9) |
| **LUT panel / Image ▸ Lookup Tables menu** applies a LUT to the whole image | **Per-channel LUT drop-down** in the Display section; `color.lut(all_channels=…)` for scripts | multi-channel images are the norm in microscopy |
| **One undo level**; `Edit ▸ Options ▸ Memory & Threads ▸ Keep multiple undo buffers` | **Unlimited undo/redo** per image with a clickable History list | ([User Guide §Undo](https://imagej.net/ij/docs/guide/146-6.html)) |
| **Modal dialogs**, preview only where a plugin implements the *Preview* checkbox | **Live preview on the current plane** for every `plane` command while you drag; nothing is written until *Apply* | RESEARCH.md §14, principle 4 |
| **Menus + Command Finder (`L`)** | **⌘K palette first**, menus second; the palette shows category, description and shortcut | RESEARCH.md §2 |
| **Global options dialogs** (`Edit ▸ Options ▸ …`) that change the behaviour of later commands | **Parameters belong to the command** (iterations, radius, scale-when-converting, wand tolerance…), so a recorded line reproduces the result without hidden state. The Preferences sheet (⌘,, `edit.options` and its sub-dialogs) only holds appearance and tool defaults (selection colour, fonts, point / arrow / rounded-rect tools, plots, JPEG quality, save format); behaviour switches such as the paste transfer mode can also be passed per call (`edit.paste_with_mode`) | reproducibility: a recorded script must not depend on preferences |
| **Stack ▸ Tools / Hyperstacks** distinguish stacks from hyperstacks | Every image is a 5-D `(T, Z, C, Y, X)` hyperstack; stack commands take an `axis` parameter (Z or T); `stacks.hyperstack_to_stack` / `stacks.to_hyperstack` exist for ImageJ-style flattening and re-shaping | one data model, fewer commands |
| **Plugins menu** that "can become easily cluttered" as jars are added | **`@op` plugin files** in `~/.lumen/plugins` register with a category, so they land in the palette and the right menu with an auto-generated sheet, preview and undo | RESEARCH.md §3, §5 |
