# Cross-validation against ImageJ 1.54p

*Independent functional evidence for the ImageJ-compatible commands: the same ImageJ macro is run by the real ImageJ 1.54p (`ij.jar`, batch mode) and by Lumen's macro layer, and the two outputs are compared numerically. Second run, 8 November 2026, after the fix round that followed the first run of 7 November 2026 (`tests/crossval/harness.py`; re-run instructions at the end).*

**277 cases: 275 exact, 1 within a stated tolerance, 1 mismatch.** ImageJ needed 51.7 s for 277 JVM launches (≈ 0.19 s each), Lumen 1.3 s; the pytest module runs in about a minute.

The first run found 115 exact / 33 within tolerance / 126 mismatches. Every mismatch was recorded with its numbers and a one-line diagnosis and routed to fix agents, who ported the ImageJ algorithms bit-exactly (`lumen/core/ops/_imagej*.py`, `lumen/core/ij_autothreshold.py`): the circular rank-filter kernel (dx²+dy² ≤ r²+1), the rolling ball / sliding paraboloid, AutoThresholder with ImageJ's 16-/32-bit level mapping and open-ended ranges, the binary operations (8-neighbour erode/dilate with count / iterations / pad edges, Outline, lookup-table Skeletonize, 8-bit EDM, UEP overwrite, Voronoi values, EDM/UEP Watershed), Analyzer / ImageStatistics / EllipseFitter / Feret / PolygonFiller / OvalRoi / line and point measurements / limit-to-threshold / invert-Y, ParticleAnalyzer (scan order, Mask of…, 16-bit Count Masks, Bare Outlines), ZProjector (titles, half-up rounding, float32 median), ImageProcessor.rotate / resize / scale / translate, TypeConverter and the display-range tables, 16-bit Gamma/Log ranges, FFT padding and the bandpass filter, MaximumFinder, ImageJ's histogram binning and 16-bit median, ProfilePlot (`getProfile()` incl. wide lines), RoiEnlarger / makeBand / RoiScaler / RoiRotator / EllipseRoi / line-to-area / draw, `setOption("BlackBackground" | "ScaleConversions")`, and ImageJ's half-up rounding and 8-bit truncation conventions in Math and the calculator. The harness and macros were not changed between the two runs (apart from the expectations file).

## Method

1. **Synthetic inputs** (deterministic numpy, written as ImageJ-flavoured TIFFs with `tifffile` – unit, spacing, hyperstack axes in the ImageJ description):

   * `blobs8` – 160×120 8-bit: 14 Gaussian blobs + 3 rings on background 30, noise σ=6
   * `blobs16` – 160×120 16-bit: the same kind of field ×16 (0–4095)
   * `blobs32` – 160×120 32-bit: (field − 60)/100, values ≈ −0.6 … 2.0 (negative values included)
   * `zstack8` – 96×96×6 8-bit Z stack: blobs drifting 1.5 px/slice with per-slice gain 0.55…1.0
   * `twochan16` – 96×96 2-channel 16-bit composite (different blob fields per channel)
   * `cal8` – blobs8 with a 0.25 µm pixel size stored in the TIFF (ImageJ metadata unit=um, 4 px/µm)
2. **One macro per case** in `tests/crossval/macros/*.ijm`; it runs **unchanged** on both sides. Every macro opens `dir + "/input.tif"` (`dir = getArgument()`), runs the command(s) and saves what it produced into that folder: `saveAs("Tiff", …/result.tif)` for pixels, `saveAs("Results", …/Results.csv)` (with `decimal=9`) for the Results table, and `print("key=value")` lines (`d2s(v, 6)` for non-integers) for thresholds, counts, bounds, histograms and profiles. A macro's header line (`// crossval: title=… | images=… | compare=…`) turns it into one case per listed input image (`gaussian_s2[blobs16]`).
3. **ImageJ side**: `java -Dapple.awt.UIElement=true -Duser.home=<scratch> -Dplugins.dir=<empty> -Xmx1024m -cp ij-1.54p.jar ij.ImageJ -batch <macro> <folder>` – ImageJ's own batch mode (no ImageJ window, no image windows, `print` goes to stdout), with isolated preferences and without Fiji's plugins; the Fiji application is never launched. ImageJ 1.x cannot execute `IJ.run` under `java.awt.headless=true` (its menus are AWT objects), so the JVM runs in the login session with the Dock icon suppressed.
4. **Lumen side**: `IJ(Session()).macro(text, arg=<folder>)` (`lumen.core.ij_compat`), i.e. the same `run("…", "options")` calls through the command table and the same macro functions. Pixels are taken from the session's current image in memory (so Lumen's TIFF writer is not part of the comparison), rows from `session.results`, printed values from the macro's return value.
5. **Comparison**: pixel arrays after mapping both to (T, Z, C, Y, X) – shape, dtype, max |Δ|, fraction of differing pixels, mean |Δ|; Results rows – ImageJ's columns must all exist in Lumen (Lumen-only extras such as `Ch`/`Frame` are noted, not failed), rows are matched by position or by centroid (`X`,`Y`) for particles, every common column is compared; printed values – per key, arrays element-wise. A result difference below 1e-8 (ImageJ's 9-decimal CSV) or a printed difference below 2e-6 (`d2s(v, 6)`) still counts as exact.
6. **Status policy**: *exact* = identical; *within tolerance* = the stated bound holds and the cause is a documented convention (listed below, still a TODO); *mismatch* = any other systematic difference, with the numbers.

## Environment

* ImageJ 1.54p – `/Applications/Fiji/jars/ij-1.54p.jar` (read-only; override with `LUMEN_IJ_JAR`), run on OpenJDK 21.0.12.1 (Homebrew, `/opt/homebrew/opt/openjdk@21/…/bin/java`; `JAVA_HOME` or `java` on PATH also work)
* Lumen on Python 3.14.6 – numpy 2.5.4, scipy 1.18.1, scikit-image 0.26.0, tifffile 2026.10.15; macOS 27.0 (arm64)
* Binary conventions fixed in the macros: `setOption("BlackBackground", true)`, `run("Options...", "iterations=1 count=1 black do=Nothing")`, masks from `setAutoThreshold("Default dark")` + `Convert to Mask`; `Set Measurements…` with every measurement and `decimal=9`

## Summary by area

| Area | cases | ✅ exact | 🟢 within tolerance | ❌ mismatch |
|---|---|---|---|---|
| Analyze ▸ Analyze Particles | 6 | 6 | 0 | 0 |
| Image ▸ Adjust ▸ Threshold (auto methods) | 11 | 11 | 0 | 0 |
| Process ▸ Binary | 13 | 12 | 0 | 1 |
| Process ▸ Image Calculator | 14 | 14 | 0 | 0 |
| Analyze ▸ Measure / Set Measurements | 20 | 20 | 0 | 0 |
| Image ▸ Transform / Scale / Crop | 34 | 34 | 0 | 0 |
| Process ▸ Filters | 30 | 30 | 0 | 0 |
| Process ▸ Enhance Contrast / Subtract Background | 15 | 15 | 0 | 0 |
| Process ▸ FFT | 4 | 4 | 0 | 0 |
| Process | 9 | 9 | 0 | 0 |
| Process ▸ Find Maxima | 11 | 11 | 0 | 0 |
| Analyze ▸ Histogram / Plot Profile / statistics | 14 | 14 | 0 | 0 |
| Process ▸ Math | 38 | 38 | 0 | 0 |
| Image ▸ Lookup Tables | 14 | 14 | 0 | 0 |
| Image ▸ Stacks | 12 | 11 | 1 | 0 |
| Edit ▸ Selection (ROI geometry) | 18 | 18 | 0 | 0 |
| Image ▸ Type | 14 | 14 | 0 | 0 |
| **All** | **277** | **274** | **2** | **1** |

## Remaining differences

| Case | Status | Measured (ImageJ vs Lumen) | Diagnosis / TODO |
|---|---|---|---|
| `binary_fill_holes_stack` (Fill Holes on a binary stack) | ❌ mismatch | pixels uint8 96×96×6: max\|Δ\|=255, 12.08% of pixels differ, mean\|Δ\|=30.8 | ImageJ side: the Convert Stack to Binary dialog resets Prefs.blackBackground (no 'black' key in the options) and in batch mode Fill Holes then ran with foreground = 0 – its output keeps only the border-connected 255 regions (3 slices empty, no hole filled; verified pixel-exactly); Lumen fills the holes of every slice (12% of voxels differ); asserted: none (exact) |
| `stack_statistics` (Stack.getStatistics + Image ▸ Stacks ▸ Statistics) | 🟢 within tolerance | results: 1 vs 1 rows; worst StdDev: \|Δ\|=0.0004402 (ImageJ 52.54, Lumen 52.5404); Lumen adds Label, Volume; log: all 5 printed values identical | ImageJ writes the Stack Statistics Results row with the default 3 decimals (this macro sets no decimal=9): Mean 53.554 / StdDev 52.540 vs Lumen's 53.554326 / 52.540440; the printed Stack.getStatistics values agree to 6 decimals; %Volume, Voxels, Min, Max, Mode, Median identical (Lumen's extra Label / Volume columns are reported, not failed); asserted: results: |Δ| ≤ 0.0005 + 0·|v| |

* `binary_fill_holes_stack` is not a Lumen defect: in ImageJ's batch mode the *Convert Stack to Binary* dialog (no `black` key in its options) resets `Prefs.blackBackground`, so ImageJ's own Fill Holes ran with foreground = 0 and only kept the border-connected 255 regions (3 slices empty, no hole filled – verified pixel-exactly against that reading); Lumen fills the holes of every slice. It stays in the table as an xfail with that reason.
* `stack_statistics`: ImageJ writes the Stack Statistics row with the default 3 decimals (the macro cannot set `decimal=9` for that command); the printed `Stack.getStatistics` values agree to 6 decimals and every other column is identical.

## Not cross-validated

* `Bin…` blocks in ImageJ's batch mode (it waits for a dialog even with `x= y= bin=` options), and `Find Maxima… output=Count` does the same – the Find Maxima counts are validated through `output=List` + `nResults` instead, Bin has no case.
* LUTs that Fiji ships as files in its `luts` folder (e.g. `16 colors`) are not in `ij.jar`; the nine built-in LUTs (Fire, Grays, Ice, Spectrum, Red, Green, Cyan, 3-3-2 RGB, Red/Green) are validated pixel-exactly after `RGB Color` and `Flatten`.
* Lumen-only columns (`Ch`, `Frame`, `Label`, `Value`, `Volume`) are reported in the table but never fail a case; Lumen extensions of ImageJ commands (extra parameters) are not exercised.
* Commands whose ImageJ result is random (Add Noise, Salt and Pepper) or interactive are out of scope.

## Full table

Measured = the numbers of the generating run; Tolerance asserted = what `tests/crossval` checks (none = must be identical).

| # | Case | ImageJ command(s) | Image | Compared | Measured | Tolerance asserted | Status | Note |
|---|---|---|---|---|---|---|---|---|
| | **Analyze ▸ Analyze Particles** | | | | | | | |
| 1 | `analyze_particles_all` | Analyze Particles… size=0-Infinity, every measurement | blobs8 | results, log | results: 11 vs 11 rows; worst X: \|Δ\|=4.988e-10 (ImageJ 32.5094, Lumen 32.5094); Lumen adds Ch, Frame; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 2 | `analyze_particles_calibrated` | Analyze Particles… on a calibrated image (size in µm²) | cal8 | results, log | results: 11 vs 11 rows; worst Feret: \|Δ\|=4.957e-10 (ImageJ 5.70088, Lumen 5.70088); Lumen adds Ch, Frame; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 3 | `analyze_particles_count_masks` | Analyze Particles… show=[Count Masks] | blobs8 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 4 | `analyze_particles_filtered` | Analyze Particles… size=50-Infinity circularity=0.30-1.00 exclude include | blobs8 | results, log | results: 11 vs 11 rows; worst XM: \|Δ\|=4.988e-10 (ImageJ 32.5094, Lumen 32.5094); Lumen adds Ch, Frame; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 5 | `analyze_particles_masks` | Analyze Particles… show=Masks | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 6 | `analyze_particles_stack` | Analyze Particles… stack (every slice) | zstack8 | results, log | results: 13 vs 13 rows; worst Y: \|Δ\|=4.956e-10 (ImageJ 49.2419, Lumen 49.2419); Lumen adds Ch, Frame; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Image ▸ Adjust ▸ Threshold (auto methods)** | | | | | | | |
| 7 | `auto_threshold_dark[blobs8]` | setAutoThreshold(method + ' dark') → getThreshold, 16 methods | blobs8 | log | log: all 16 printed values identical | none (exact) | ✅ exact |  |
| 8 | `auto_threshold_dark[blobs16]` | setAutoThreshold(method + ' dark') → getThreshold, 16 methods | blobs16 | log | log: all 16 printed values identical | none (exact) | ✅ exact |  |
| 9 | `auto_threshold_dark[blobs32]` | setAutoThreshold(method + ' dark') → getThreshold, 16 methods | blobs32 | log | log: all 16 printed values identical | none (exact) | ✅ exact |  |
| 10 | `auto_threshold_light[blobs8]` | setAutoThreshold(method) (objects dark) → getThreshold, 16 methods | blobs8 | log | log: all 16 printed values identical | none (exact) | ✅ exact |  |
| 11 | `auto_threshold_light[blobs16]` | setAutoThreshold(method) (objects dark) → getThreshold, 16 methods | blobs16 | log | log: all 16 printed values identical | none (exact) | ✅ exact |  |
| | **Process ▸ Binary** | | | | | | | |
| 12 | `binary_close` | Close- (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 13 | `binary_dilate` | Dilate (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 14 | `binary_distance_map` | Distance Map (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 15 | `binary_erode` | Erode (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 16 | `binary_erode_iter3` | Erode, iterations=3 count=1 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 17 | `binary_fill_holes` | Fill Holes (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 18 | `binary_fill_holes_stack` | Fill Holes on a binary stack | zstack8 | pixels | pixels uint8 96×96×6: max\|Δ\|=255, 12.08% of pixels differ, mean\|Δ\|=30.8 | none (exact) | ❌ mismatch | ImageJ side: the Convert Stack to Binary dialog resets Prefs.blackBackground (no 'black' key in the options) and in batch mode Fill Holes then ran with foreground = 0 – its output keeps only the border-connected 255 regions (3 slices empty, no hole filled; verified pixel-exactly); Lumen fills the holes of every slice (12% of voxels differ) |
| 19 | `binary_open` | Open (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 20 | `binary_outline` | Outline (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 21 | `binary_skeletonize` | Skeletonize (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 22 | `binary_ultimate_points` | Ultimate Points (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 23 | `binary_voronoi` | Voronoi (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 24 | `binary_watershed` | Watershed (after Default-dark mask) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Image Calculator** | | | | | | | |
| 25 | `calc_add_8` | Image Calculator… operation=Add create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 26 | `calc_and_8` | Image Calculator… operation=AND create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 27 | `calc_average_8` | Image Calculator… operation=Average create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 28 | `calc_difference_8` | Image Calculator… operation=Difference create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 29 | `calc_divide_32` | Image Calculator… operation=Divide create 32-bit | blobs8 | pixels, log | pixels float32 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 30 | `calc_max_8` | Image Calculator… operation=Max create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 31 | `calc_min_8` | Image Calculator… operation=Min create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 32 | `calc_multiply_32` | Image Calculator… operation=Multiply create 32-bit | blobs8 | pixels, log | pixels float32 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 33 | `calc_multiply_8` | Image Calculator… operation=Multiply create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 34 | `calc_subtract_32` | Image Calculator… operation=Subtract create 32-bit | blobs8 | pixels, log | pixels float32 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 35 | `calc_subtract_inplace[blobs8]` | Image Calculator… Subtract in place (image1 modified) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 36 | `calc_subtract_inplace[blobs16]` | Image Calculator… Subtract in place (image1 modified) | blobs16 | pixels, log | pixels uint16 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 37 | `calc_subtract_stack` | Image Calculator… Subtract stack create (stack − blurred stack) | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| 38 | `calc_transparent_zero` | Image Calculator… operation=Transparent-zero create | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Analyze ▸ Measure / Set Measurements** | | | | | | | |
| 39 | `calibrated_file_measure` | Calibration read from the TIFF (0.25 µm/px) + Measure | cal8 | results, log | results: 1 vs 1 rows; worst Major: \|Δ\|=4.696e-10 (ImageJ 16.9257, Lumen 16.9257); Lumen adds Ch, Frame; log: all 4 printed values identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 40 | `canvas_size_center_zero[blobs8]` | Canvas Size… 200×150 Center zero | blobs8 | pixels | pixels uint8 200×150: identical | none (exact) | ✅ exact |  |
| 41 | `canvas_size_center_zero[blobs16]` | Canvas Size… 200×150 Center zero | blobs16 | pixels | pixels uint16 200×150: identical | none (exact) | ✅ exact |  |
| 42 | `canvas_size_topleft` | Canvas Size… 100×80 Top-Left (crop) | blobs8 | pixels | pixels uint8 100×80: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Adjust ▸ Threshold (auto methods)** | | | | | | | |
| 43 | `convert_to_mask_otsu[blobs8]` | setAutoThreshold('Otsu dark') + Convert to Mask | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 44 | `convert_to_mask_otsu[blobs16]` | setAutoThreshold('Otsu dark') + Convert to Mask | blobs16 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 45 | `convert_to_mask_otsu[blobs32]` | setAutoThreshold('Otsu dark') + Convert to Mask | blobs32 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 46 | `convert_to_mask_stack` | Convert to Mask on a stack (calculate threshold for each image) | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 47 | `convolve_3x3_norm[blobs8]` | Convolve… 3×3 binomial, normalize | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 48 | `convolve_3x3_norm[blobs32]` | Convolve… 3×3 binomial, normalize | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 49 | `convolve_5x5_raw[blobs8]` | Convolve… 5×5 Laplacian-like, no normalize | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 50 | `convolve_5x5_raw[blobs32]` | Convolve… 5×5 Laplacian-like, no normalize | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 51 | `crop_oval` | Crop to an oval (outside cleared) | blobs8 | pixels | pixels uint8 70×50: identical | none (exact) | ✅ exact |  |
| 52 | `crop_rect[blobs8]` | Crop to a rectangle | blobs8 | pixels | pixels uint8 60×45: identical | none (exact) | ✅ exact |  |
| 53 | `crop_rect[blobs16]` | Crop to a rectangle | blobs16 | pixels | pixels uint16 60×45: identical | none (exact) | ✅ exact |  |
| 54 | `crop_rect[blobs32]` | Crop to a rectangle | blobs32 | pixels | pixels float32 60×45: identical | none (exact) | ✅ exact |  |
| 55 | `duplicate_rect` | Duplicate… of a rectangle selection | blobs8 | pixels, log | pixels uint8 60×45: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Process ▸ Enhance Contrast / Subtract Background** | | | | | | | |
| 56 | `enhance_contrast_equalize[blobs8]` | Enhance Contrast… equalize | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 57 | `enhance_contrast_equalize[blobs16]` | Enhance Contrast… equalize | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 58 | `enhance_contrast_normalize[blobs8]` | Enhance Contrast… saturated=0.35 normalize | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 59 | `enhance_contrast_normalize[blobs16]` | Enhance Contrast… saturated=0.35 normalize | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 60 | `enhance_contrast_normalize[blobs32]` | Enhance Contrast… saturated=0.35 normalize | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 61 | `enhance_contrast_sat[blobs8]` | Enhance Contrast… saturated=0.35 (display range only) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 62 | `enhance_contrast_sat[blobs16]` | Enhance Contrast… saturated=0.35 (display range only) | blobs16 | pixels, log | pixels uint16 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Process ▸ FFT** | | | | | | | |
| 63 | `fft_bandpass` | Bandpass Filter… large=40 small=3 autoscale saturate | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 64 | `fft_inverse_roundtrip` | FFT → Inverse FFT round trip | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 65 | `fft_power_spectrum[blobs8]` | FFT (8-bit power spectrum) | blobs8 | pixels, log | pixels uint8 256×256: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 66 | `fft_power_spectrum[blobs32]` | FFT (8-bit power spectrum) | blobs32 | pixels, log | pixels uint8 256×256: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Process** | | | | | | | |
| 67 | `find_edges[blobs8]` | Find Edges | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 68 | `find_edges[blobs16]` | Find Edges | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 69 | `find_edges[blobs32]` | Find Edges | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Find Maxima** | | | | | | | |
| 70 | `find_maxima_count[blobs8]` | Find Maxima… prominence=30 output=List → nResults (output=Count blocks ImageJ's batch mode) | blobs8 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 71 | `find_maxima_count[blobs16]` | Find Maxima… prominence=30 output=List → nResults (output=Count blocks ImageJ's batch mode) | blobs16 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 72 | `find_maxima_count[blobs32]` | Find Maxima… prominence=30 output=List → nResults (output=Count blocks ImageJ's batch mode) | blobs32 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 73 | `find_maxima_light` | Find Maxima… prominence=30 light output=List → nResults (minima) | blobs8 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 74 | `find_maxima_list` | Find Maxima… output=List (rows sorted by value) | blobs8 | results | results: 13 vs 13 rows; all 2 common columns identical | none (exact) | ✅ exact |  |
| 75 | `find_maxima_points[blobs8]` | Find Maxima… prominence=30 output=[Point Selection] (n, Σx, Σy) | blobs8 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 76 | `find_maxima_points[blobs32]` | Find Maxima… prominence=30 output=[Point Selection] (n, Σx, Σy) | blobs32 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 77 | `find_maxima_segmented` | Find Maxima… output=[Segmented Particles] | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 78 | `find_maxima_single_points` | Find Maxima… output=[Single Points] | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 79 | `find_maxima_strict_exclude` | Find Maxima… prominence=30 strict exclude output=List → nResults | blobs8 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 80 | `find_maxima_within_tolerance` | Find Maxima… output=[Maxima Within Tolerance] | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 81 | `flip_h[blobs8]` | Flip Horizontally | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 82 | `flip_h[blobs16]` | Flip Horizontally | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 83 | `flip_v[blobs8]` | Flip Vertically | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 84 | `flip_v[blobs32]` | Flip Vertically | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 85 | `flip_z` | Flip Z | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 86 | `gaussian_s0_7[blobs8]` | Gaussian Blur… sigma=0.7 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 87 | `gaussian_s0_7[blobs32]` | Gaussian Blur… sigma=0.7 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 88 | `gaussian_s2[blobs8]` | Gaussian Blur… sigma=2 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 89 | `gaussian_s2[blobs16]` | Gaussian Blur… sigma=2 | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 90 | `gaussian_s2[blobs32]` | Gaussian Blur… sigma=2 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 91 | `gaussian_s6[blobs8]` | Gaussian Blur… sigma=6 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 92 | `gaussian_s6[blobs16]` | Gaussian Blur… sigma=6 | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 93 | `gaussian_scaled[blobs8]` | Gaussian Blur… sigma=1 scaled (pixel 0.5×0.25 µm) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 94 | `gaussian_scaled[blobs16]` | Gaussian Blur… sigma=1 scaled (pixel 0.5×0.25 µm) | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 95 | `gaussian_scaled[blobs32]` | Gaussian Blur… sigma=1 scaled (pixel 0.5×0.25 µm) | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 96 | `gaussian_stack` | Gaussian Blur… sigma=2 stack | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| | **Analyze ▸ Histogram / Plot Profile / statistics** | | | | | | | |
| 97 | `get_pixel[blobs8]` | getPixel at 12 positions (8/16/32-bit) | blobs8 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 98 | `get_pixel[blobs16]` | getPixel at 12 positions (8/16/32-bit) | blobs16 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 99 | `get_pixel[blobs32]` | getPixel at 12 positions (8/16/32-bit) | blobs32 | log | log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 100 | `get_statistics[blobs8]` | getStatistics / getRawStatistics / getValue inside a rectangle | blobs8 | log | log: all 10 printed values identical | none (exact) | ✅ exact |  |
| 101 | `get_statistics[blobs16]` | getStatistics / getRawStatistics / getValue inside a rectangle | blobs16 | log | log: all 10 printed values identical | none (exact) | ✅ exact |  |
| 102 | `get_statistics[blobs32]` | getStatistics / getRawStatistics / getValue inside a rectangle | blobs32 | log | log: all 10 printed values identical | none (exact) | ✅ exact |  |
| 103 | `histogram_16bit_256` | getHistogram 256 bins on 16-bit | blobs16 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 104 | `histogram_32bit_64` | getHistogram 64 bins on 32-bit | blobs32 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 105 | `histogram_8bit_256` | getHistogram 256 bins | blobs8 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 106 | `histogram_roi` | getHistogram inside an oval | blobs8 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| | **Process ▸ Math** | | | | | | | |
| 107 | `invert[blobs8]` | Edit ▸ Invert | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 108 | `invert[blobs16]` | Edit ▸ Invert | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 109 | `invert[blobs32]` | Edit ▸ Invert | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 110 | `invert_stack` | Edit ▸ Invert (stack) | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Lookup Tables** | | | | | | | |
| 111 | `lut_332_rgb_rgb` | '3-3-2 RGB' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 112 | `lut_apply` | setMinAndMax(50, 180) + Apply LUT | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 113 | `lut_cyan_rgb` | 'Cyan' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 114 | `lut_fire_16bit_rgb` | 'Fire' LUT on 16-bit after resetMinAndMax + RGB Color | blobs16 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 115 | `lut_fire_flatten` | 'Fire' LUT + Flatten | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 116 | `lut_fire_rgb` | 'Fire' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 117 | `lut_grays_minmax_rgb` | setMinAndMax(50, 180) + RGB Color (display range baked in) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 118 | `lut_grays_rgb` | 'Grays' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 119 | `lut_green_rgb` | 'Green' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 120 | `lut_ice_rgb` | 'Ice' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 121 | `lut_invert_rgb` | Invert LUT + RGB Color | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 122 | `lut_red_green_rgb` | 'Red/Green' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 123 | `lut_red_rgb` | 'Red' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| 124 | `lut_spectrum_rgb` | 'Spectrum' LUT + RGB Color (pixel-exact table) | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Adjust ▸ Threshold (auto methods)** | | | | | | | |
| 125 | `make_binary_default` | setAutoThreshold('Default dark') + Make Binary | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Math** | | | | | | | |
| 126 | `math_abs` | Math: run("Abs"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 127 | `math_add_50[blobs8]` | Math: run("Add...", "value=50"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 128 | `math_add_50[blobs16]` | Math: run("Add...", "value=50"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 129 | `math_add_50[blobs32]` | Math: run("Add...", "value=50"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 130 | `math_and_11110000` | Math: run("AND...", "value=11110000"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 131 | `math_divide_3[blobs8]` | Math: run("Divide...", "value=3"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 132 | `math_divide_3[blobs32]` | Math: run("Divide...", "value=3"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 133 | `math_exp[blobs8]` | Math: run("Exp"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 134 | `math_exp[blobs32]` | Math: run("Exp"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 135 | `math_gamma_0_5[blobs8]` | Math: run("Gamma...", "value=0.5"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 136 | `math_gamma_0_5[blobs16]` | Math: run("Gamma...", "value=0.5"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 137 | `math_gamma_0_5[blobs32]` | Math: run("Gamma...", "value=0.5"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 138 | `math_log[blobs8]` | Math: run("Log"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 139 | `math_log[blobs16]` | Math: run("Log"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 140 | `math_log[blobs32]` | Math: run("Log"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 141 | `math_macro_v_x[blobs8]` | Math: run("Macro...", "code=[v=v*0.5+x/4]"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 142 | `math_macro_v_x[blobs32]` | Math: run("Macro...", "code=[v=v*0.5+x/4]"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 143 | `math_max_150[blobs8]` | Math: run("Max...", "value=150"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 144 | `math_max_150[blobs16]` | Math: run("Max...", "value=150"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 145 | `math_min_50` | Math: run("Min...", "value=50"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 146 | `math_multiply_1_5[blobs8]` | Math: run("Multiply...", "value=1.5"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 147 | `math_multiply_1_5[blobs16]` | Math: run("Multiply...", "value=1.5"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 148 | `math_multiply_1_5[blobs32]` | Math: run("Multiply...", "value=1.5"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 149 | `math_nan_background` | Math: run("NaN Background"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 150 | `math_reciprocal` | Math: run("Reciprocal"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 151 | `math_set_77` | Math: run("Set...", "value=77"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 152 | `math_sqrt[blobs8]` | Math: run("Square Root"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 153 | `math_sqrt[blobs16]` | Math: run("Square Root"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 154 | `math_sqrt[blobs32]` | Math: run("Square Root"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 155 | `math_square[blobs8]` | Math: run("Square"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 156 | `math_square[blobs16]` | Math: run("Square"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 157 | `math_square[blobs32]` | Math: run("Square"); | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 158 | `math_subtract_50[blobs8]` | Math: run("Subtract...", "value=50"); | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 159 | `math_subtract_50[blobs16]` | Math: run("Subtract...", "value=50"); | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 160 | `maximum_r2[blobs8]` | Maximum… radius=2 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 161 | `maximum_r2[blobs32]` | Maximum… radius=2 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 162 | `mean_r3[blobs8]` | Mean… radius=3 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 163 | `mean_r3[blobs32]` | Mean… radius=3 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Analyze ▸ Measure / Set Measurements** | | | | | | | |
| 164 | `measure_channel2` | Measure on channel 2 of a composite (Ch column) | twochan16 | results | results: 1 vs 1 rows; worst YM: \|Δ\|=4.373e-10 (ImageJ 39.8429, Lumen 39.8429); Lumen adds Slice, Frame | none (exact) | ✅ exact |  |
| 165 | `measure_invert_y` | Measure with 'invert Y coordinates' | blobs8 | results | results: 1 vs 1 rows; worst YM: \|Δ\|=4.407e-10 (ImageJ 73.0864, Lumen 73.0864) | none (exact) | ✅ exact |  |
| 166 | `measure_limit_threshold` | Measure with 'limit to threshold' (Mean, %Area over thresholded pixels) | blobs8 | results | results: 1 vs 1 rows; worst %Area: \|Δ\|=3.333e-10 (ImageJ 16.0521, Lumen 16.0521) | none (exact) | ✅ exact |  |
| 167 | `measure_line[blobs8]` | Measure, straight line, every measurement | blobs8 | results | results: 1 vs 1 rows; worst Circ.: \|Δ\|=4.812e-10 (ImageJ 0.0830567, Lumen 0.0830567); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 168 | `measure_line[blobs16]` | Measure, straight line, every measurement | blobs16 | results | results: 1 vs 1 rows; worst Max: \|Δ\|=5.002e-10 (ImageJ 4612.6, Lumen 4612.6); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 169 | `measure_line_wide` | Measure, straight line of width 5 (makeLine with width) | blobs8 | results | results: 1 vs 1 rows; worst Feret: \|Δ\|=1.051e-06 (ImageJ 153.725, Lumen 153.725); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 170 | `measure_oval[blobs8]` | Measure, oval, every measurement | blobs8 | results | results: 1 vs 1 rows; worst Minor: \|Δ\|=4.564e-10 (ImageJ 50.0786, Lumen 50.0786); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 171 | `measure_oval[blobs16]` | Measure, oval, every measurement | blobs16 | results | results: 1 vs 1 rows; worst IntDen: \|Δ\|=4.657e-10 (ImageJ 3.54266e+06, Lumen 3.54266e+06); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 172 | `measure_point` | Measure a point selection | blobs8 | results | results: 1 vs 1 rows; all 6 common columns identical; Lumen adds Value | none (exact) | ✅ exact |  |
| 173 | `measure_polygon[blobs8]` | Measure, polygon, every measurement | blobs8 | results | results: 1 vs 1 rows; worst AR: \|Δ\|=4.615e-10 (ImageJ 1.40062, Lumen 1.40062); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 174 | `measure_polygon[blobs32]` | Measure, polygon, every measurement | blobs32 | results | results: 1 vs 1 rows; worst Skew: \|Δ\|=4.771e-10 (ImageJ 0.839036, Lumen 0.839036); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 175 | `measure_rect[blobs8]` | Measure, rectangle, every measurement | blobs8 | results | results: 1 vs 1 rows; worst Skew: \|Δ\|=4.529e-10 (ImageJ 1.75335, Lumen 1.75335); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 176 | `measure_rect[blobs16]` | Measure, rectangle, every measurement | blobs16 | results | results: 1 vs 1 rows; worst Circ.: \|Δ\|=4.506e-10 (ImageJ 0.76937, Lumen 0.76937); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 177 | `measure_rect[blobs32]` | Measure, rectangle, every measurement | blobs32 | results | results: 1 vs 1 rows; worst Median: \|Δ\|=4.534e-10 (ImageJ -0.188856, Lumen -0.188856); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 178 | `measure_slice4_zstack` | Measure on slice 4 of a Z stack (Slice column) | zstack8 | results | results: 1 vs 1 rows; worst YM: \|Δ\|=4.493e-10 (ImageJ 37.5308, Lumen 37.5308); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 179 | `measure_whole_image[blobs8]` | Measure without a selection | blobs8 | results | results: 1 vs 1 rows; worst Circ.: \|Δ\|=4.506e-10 (ImageJ 0.76937, Lumen 0.76937); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 180 | `measure_whole_image[blobs16]` | Measure without a selection | blobs16 | results | results: 1 vs 1 rows; worst StdDev: \|Δ\|=4.864e-10 (ImageJ 757.179, Lumen 757.179); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| 181 | `measure_whole_image[blobs32]` | Measure without a selection | blobs32 | results | results: 1 vs 1 rows; worst YM: \|Δ\|=4.573e-10 (ImageJ 41.701, Lumen 41.701); Lumen adds Ch, Frame | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 182 | `median_r2[blobs8]` | Median… radius=2 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 183 | `median_r2[blobs16]` | Median… radius=2 | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 184 | `median_r2[blobs32]` | Median… radius=2 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 185 | `minimum_r2[blobs8]` | Minimum… radius=2 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 186 | `minimum_r2[blobs16]` | Minimum… radius=2 | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| | **Analyze ▸ Histogram / Plot Profile / statistics** | | | | | | | |
| 187 | `profile_line[blobs8]` | getProfile() along a line | blobs8 | log | log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 188 | `profile_line[blobs32]` | getProfile() along a line | blobs32 | log | log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 189 | `profile_line_wide` | getProfile() along a line of width 5 (makeLine with width) | blobs8 | log | log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 190 | `profile_rect` | getProfile() of a rectangle (column averages) | blobs8 | log | log: all 2 printed values identical | none (exact) | ✅ exact |  |
| | **Image ▸ Stacks** | | | | | | | |
| 191 | `reslice_top` | Reslice [/]… (top, output spacing 1) | zstack8 | pixels | pixels uint8 96×6×96: identical | none (exact) | ✅ exact |  |
| | **Edit ▸ Selection (ROI geometry)** | | | | | | | |
| 192 | `roi_band_5` | Make Band… band=5 (oval) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 193 | `roi_bounding_box` | To Bounding Box of a polygon | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 194 | `roi_convex_hull` | Convex Hull of a concave polygon | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 195 | `roi_create_selection` | Create Selection from a mask → bounds, area | blobs8 | log | log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 196 | `roi_draw` | Draw an oval outline (foreground 255, line width 1) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 197 | `roi_enlarge_5` | Enlarge… enlarge=5 (oval) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 198 | `roi_enlarge_minus5` | Enlarge… enlarge=-5 (shrink a polygon) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 199 | `roi_enlarge_rect` | Enlarge… enlarge=7 (rectangle stays a rectangle) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 200 | `roi_fill_clear` | Fill (foreground 200) and Clear Outside (background 10) of an oval | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 201 | `roi_fit_ellipse` | Fit Ellipse to a polygon | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 202 | `roi_interpolate` | Interpolate interval=5 smooth | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 203 | `roi_interpolate_nosmooth` | Interpolate interval=3 (no smoothing) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 3 printed values identical | none (exact) | ✅ exact |  |
| 204 | `roi_line_to_area` | makeLine width 5 + Line to Area | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 205 | `roi_make_inverse` | Make Inverse of an oval | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 206 | `roi_oval_mask` | makeOval → Create Mask (rasterisation) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 207 | `roi_polygon_mask` | makePolygon → Create Mask (rasterisation) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 208 | `roi_rotate_30` | Selection Rotate… angle=30 (rectangle) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| 209 | `roi_scale_1_5` | Selection Scale… x=1.5 y=1.5 centered (oval) | blobs8 | pixels, log | pixels uint8 160×120: identical; log: all 2 printed values identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 210 | `rotate90_left_stack` | Rotate 90 Degrees Left (stack) | zstack8 | pixels | pixels uint8 96×96×6: identical | none (exact) | ✅ exact |  |
| 211 | `rotate90_right[blobs8]` | Rotate 90 Degrees Right | blobs8 | pixels | pixels uint8 120×160: identical | none (exact) | ✅ exact |  |
| 212 | `rotate90_right[blobs16]` | Rotate 90 Degrees Right | blobs16 | pixels | pixels uint16 120×160: identical | none (exact) | ✅ exact |  |
| 213 | `rotate_15_bicubic` | Rotate… angle=15 interpolation=Bicubic | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 214 | `rotate_15_bilinear[blobs8]` | Rotate… angle=15 interpolation=Bilinear | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 215 | `rotate_15_bilinear[blobs16]` | Rotate… angle=15 interpolation=Bilinear | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 216 | `rotate_15_bilinear[blobs32]` | Rotate… angle=15 interpolation=Bilinear | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 217 | `rotate_15_enlarge` | Rotate… angle=15 Bilinear enlarge | blobs8 | pixels | pixels uint8 186×157: identical | none (exact) | ✅ exact |  |
| 218 | `rotate_30_none` | Rotate… angle=30 interpolation=None | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 219 | `scale_0_5_average[blobs8]` | Scale… x=0.5 y=0.5 Bilinear average create | blobs8 | pixels | pixels uint8 80×60: identical | none (exact) | ✅ exact |  |
| 220 | `scale_0_5_average[blobs16]` | Scale… x=0.5 y=0.5 Bilinear average create | blobs16 | pixels | pixels uint16 80×60: identical | none (exact) | ✅ exact |  |
| 221 | `scale_0_5_bicubic` | Scale… x=0.5 y=0.5 Bicubic create | blobs8 | pixels | pixels uint8 80×60: identical | none (exact) | ✅ exact |  |
| 222 | `scale_1_5_bilinear[blobs8]` | Scale… x=1.5 y=1.5 Bilinear create | blobs8 | pixels, log | pixels uint8 240×180: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 223 | `scale_1_5_bilinear[blobs32]` | Scale… x=1.5 y=1.5 Bilinear create | blobs32 | pixels, log | pixels float32 240×180: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 224 | `scale_2_none` | Scale… x=2 y=2 None create | blobs8 | pixels | pixels uint8 320×240: identical | none (exact) | ✅ exact |  |
| | **Analyze ▸ Measure / Set Measurements** | | | | | | | |
| 225 | `set_scale_measure` | Set Scale… distance=4 known=1 unit=um, then Measure an oval in calibrated units | blobs8 | results, log | results: 1 vs 1 rows; worst Skew: \|Δ\|=4.335e-10 (ImageJ 1.62947, Lumen 1.62947); Lumen adds Ch, Frame; log: all 3 printed values identical | none (exact) | ✅ exact |  |
| | **Image ▸ Adjust ▸ Threshold (auto methods)** | | | | | | | |
| 226 | `set_threshold_mask` | setThreshold(80, 255) + Convert to Mask | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Process** | | | | | | | |
| 227 | `sharpen[blobs8]` | Sharpen | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 228 | `sharpen[blobs16]` | Sharpen | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 229 | `sharpen[blobs32]` | Sharpen | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 230 | `size_width_100` | Size… width=100 height=75 Bilinear average | blobs8 | pixels | pixels uint8 100×75: identical | none (exact) | ✅ exact |  |
| | **Process** | | | | | | | |
| 231 | `smooth[blobs8]` | Smooth | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 232 | `smooth[blobs16]` | Smooth | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 233 | `smooth[blobs32]` | Smooth | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Stacks** | | | | | | | |
| 234 | `split_channels_c2` | Split Channels → C2-input.tif | twochan16 | pixels, log | pixels uint16 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 235 | `stack_statistics` | Stack.getStatistics + Image ▸ Stacks ▸ Statistics | zstack8 | results, log | results: 1 vs 1 rows; worst StdDev: \|Δ\|=0.0004402 (ImageJ 52.54, Lumen 52.5404); Lumen adds Label, Volume; log: all 5 printed values identical | results: |Δ| ≤ 0.0005 + 0·|v| | 🟢 within tolerance | ImageJ writes the Stack Statistics Results row with the default 3 decimals (this macro sets no decimal=9): Mean 53.554 / StdDev 52.540 vs Lumen's 53.554326 / 52.540440; the printed Stack.getStatistics values agree to 6 decimals; %Volume, Voxels, Min, Max, Mode, Median identical (Lumen's extra Label / Volume columns are reported, not failed) |
| 236 | `stack_to_rgb_composite` | Stack to RGB (composite 2 channels) | twochan16 | pixels | pixels uint8 96×96×3: identical | none (exact) | ✅ exact |  |
| 237 | `substack_2_4` | Make Substack… slices=2-4 | zstack8 | pixels, log | pixels uint8 96×96×3: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| | **Process ▸ Enhance Contrast / Subtract Background** | | | | | | | |
| 238 | `subtract_bg_create15` | Subtract Background… rolling=15 create (background image) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 239 | `subtract_bg_light30` | Subtract Background… rolling=30 light | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 240 | `subtract_bg_paraboloid15[blobs8]` | Subtract Background… rolling=15 sliding (sliding paraboloid) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 241 | `subtract_bg_paraboloid15[blobs32]` | Subtract Background… rolling=15 sliding (sliding paraboloid) | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 242 | `subtract_bg_rolling15[blobs8]` | Subtract Background… rolling=15 (rolling ball) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 243 | `subtract_bg_rolling15[blobs16]` | Subtract Background… rolling=15 (rolling ball) | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 244 | `subtract_bg_rolling15[blobs32]` | Subtract Background… rolling=15 (rolling ball) | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 245 | `subtract_bg_rolling15_nosmooth` | Subtract Background… rolling=15 disable (no smoothing) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Type** | | | | | | | |
| 246 | `to_16bit_from32_noscale` | 32-bit → 16-bit with setOption('ScaleConversions', false) | blobs32 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 247 | `to_16bit_from32_reset` | 32-bit → 16-bit after resetMinAndMax | blobs32 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 248 | `to_16bit_from8` | 8-bit → 16-bit | blobs8 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 249 | `to_32bit_from16` | 16-bit → 32-bit | blobs16 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 250 | `to_32bit_from8` | 8-bit → 32-bit | blobs8 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 251 | `to_8bit_from16_noscale` | 16-bit → 8-bit with setOption('ScaleConversions', false) | blobs16 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 252 | `to_8bit_from16_range` | 16-bit → 8-bit with setMinAndMax(500, 3000) | blobs16 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 253 | `to_8bit_from16_reset` | 16-bit → 8-bit after resetMinAndMax (scaled to the data range) | blobs16 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 254 | `to_8bit_from32_range` | 32-bit → 8-bit with setMinAndMax(0, 1) | blobs32 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 255 | `to_8bit_from32_reset` | 32-bit → 8-bit after resetMinAndMax | blobs32 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 256 | `to_8bit_from_rgb` | RGB → 8-bit (unweighted) | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 257 | `to_8bit_from_rgb_weighted` | RGB → 8-bit with weighted conversion | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 258 | `to_8bit_stack` | 16-bit stack → 8-bit (stack display range) | twochan16 | pixels | pixels uint8 96×96×2: identical | none (exact) | ✅ exact |  |
| 259 | `to_rgb_from8` | 8-bit (Grays) → RGB Color | blobs8 | pixels | pixels uint8 160×120×3: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 260 | `tophat_r10` | Top Hat… radius=10 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Transform / Scale / Crop** | | | | | | | |
| 261 | `translate_5_3_none[blobs8]` | Translate… x=5 y=3 interpolation=None | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 262 | `translate_5_3_none[blobs16]` | Translate… x=5 y=3 interpolation=None | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 263 | `translate_5_3_none[blobs32]` | Translate… x=5 y=3 interpolation=None | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 264 | `translate_sub_bilinear[blobs8]` | Translate… x=2.5 y=-1.5 interpolation=Bilinear | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 265 | `translate_sub_bilinear[blobs32]` | Translate… x=2.5 y=-1.5 interpolation=Bilinear | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Process ▸ Filters** | | | | | | | |
| 266 | `unsharp_r2_w06[blobs8]` | Unsharp Mask… radius=2 mask=0.60 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 267 | `unsharp_r2_w06[blobs16]` | Unsharp Mask… radius=2 mask=0.60 | blobs16 | pixels | pixels uint16 160×120: identical | none (exact) | ✅ exact |  |
| 268 | `unsharp_r2_w06[blobs32]` | Unsharp Mask… radius=2 mask=0.60 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| 269 | `variance_r2[blobs8]` | Variance… radius=2 | blobs8 | pixels | pixels uint8 160×120: identical | none (exact) | ✅ exact |  |
| 270 | `variance_r2[blobs32]` | Variance… radius=2 | blobs32 | pixels | pixels float32 160×120: identical | none (exact) | ✅ exact |  |
| | **Image ▸ Stacks** | | | | | | | |
| 271 | `zproject_avg` | Z Project… projection=[Average Intensity] | zstack8 | pixels, log | pixels uint8 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 272 | `zproject_max` | Z Project… projection=[Max Intensity] | zstack8 | pixels, log | pixels uint8 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 273 | `zproject_median` | Z Project… projection=[Median] | zstack8 | pixels, log | pixels uint8 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 274 | `zproject_min` | Z Project… projection=[Min Intensity] | zstack8 | pixels, log | pixels uint8 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 275 | `zproject_range_2_4` | Z Project… start=2 stop=4 Max Intensity | zstack8 | pixels, log | pixels uint8 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 276 | `zproject_sd` | Z Project… projection=[Standard Deviation] | zstack8 | pixels, log | pixels float32 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |
| 277 | `zproject_sum` | Z Project… projection=[Sum Slices] | zstack8 | pixels, log | pixels float32 96×96: identical; log: all 1 printed values identical | none (exact) | ✅ exact |  |

## How to re-run

```bash
# the pytest module (opt-in: it starts one JVM per case; ~1 min)
.venv/bin/python -m pytest tests/crossval -m crossval -q
LUMEN_CROSSVAL=1 .venv/bin/python -m pytest tests/crossval -q      # same, via the environment

# the table (markdown) and every number (JSON)
.venv/bin/python tests/crossval/harness.py --markdown report.md --json report.json
.venv/bin/python tests/crossval/harness.py --only gaussian_s2,median_r2 --ignore-expect   # a few cases, zero tolerance
```

* `LUMEN_IJ_JAR` – the ImageJ 1.x jar (default `/Applications/Fiji/jars/ij-1.54p.jar`); `JAVA_HOME`, the Homebrew OpenJDK 21 path or `java` on PATH; `LUMEN_CROSSVAL_TIMEOUT` – seconds per ImageJ process (default 120). Without `java` or the jar the module skips cleanly; without the opt-in (`-m crossval`, `LUMEN_CROSSVAL=1` or `pytest tests/crossval`) its tests are skipped so the default run never starts a JVM. The JVM needs a display session (ImageJ 1.x builds AWT menus even in batch mode; on Linux use Xvfb).
* Every case asserts its recorded status: an exact case must stay exact, a tolerance case must stay within its bound, and a known mismatch is an *xfail* that **fails once it stops reproducing** – fix the defect, then promote the case in `tests/crossval/expectations.py` and in this document.
* A new case is a new macro file: copy the header line, list the input images (`blobs8, blobs16, blobs32, zstack8, twochan16, cal8`) and what to compare (`pixels`, `results`, `log`); `tests/crossval/harness.py` picks it up and the case id is the file stem (plus `[image]`).
