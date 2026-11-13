# Lumen command reference

503 commands, generated from the registry. Every command is reachable from the command palette (⌘K / Ctrl+K), the menu bar, the Python scripting API as `lumen.run("<id>", **params)`, and (for the mapped subset) ImageJ-macro `run("…")` calls via the compatibility layer (see docs/FIJI_PARITY.md).

Notes: *live preview* means the operation sheet renders the result on the current plane while you adjust parameters; *needs … selection* commands are greyed out until a matching ROI exists. Parameters are listed in the order the sheet shows them; `help.command` prints the full table (types, defaults, ranges) for any id.

## File

| Command | id | Parameters | Notes |
|---|---|---|---|
| New ▸ Internal Clipboard | `file.new_from_clipboard` | `title`, `outside` |  |
| Open Next | `file.open_next` | `close_current` | shortcut **Shift+O** |
| Open Previous | `file.open_previous` | `close_current` | shortcut **Alt+Shift+O** |
| Open Recent | `file.open_recent` | `index`, `path` |  |
| Open ImageJ Sample… | `file.open_imagej_sample` | `name`, `use_cache` |  |
| New Image | `file.new` | `title`, `type`, `width`, `height`, `slices`, `channels`, `frames`, `fill`, `period` | shortcut **N** |
| Open Sample | `file.sample` | `name` |  |
| Close | `file.close` | – | shortcut **W** |
| Close All | `file.close_all` | – |  |
| Revert | `file.revert` | – |  |

## File/Import

| Command | id | Parameters | Notes |
|---|---|---|---|
| Bio-Formats Series... | `file.bioformats_series` | `path` |  |
| Bio-Formats... | `file.bioformats_import` | `path`, `series`, `open_all_series`, `color_mode`, `custom_luts`, `view`, `stack_order`, `autoscale`, `split_channels`, `split_focal_planes`, `split_timepoints`, `group_files`, `group_pattern`, `virtual`, `specify_range`, `c_begin`, `c_end`, `c_step`, `z_begin`, `z_end`, `z_step`, `t_begin`, `t_end`, `t_step`, `crop`, `crop_x`, `crop_y`, `crop_width`, `crop_height`, `display_metadata`, `display_ome_xml`, `swap_dimensions`, `swap_order`, `level` |  |
| Bio-Formats Remote... | `file.bioformats_remote_import` | `url`, `series`, `open_all_series`, `color_mode`, `custom_luts`, `view`, `stack_order`, `autoscale`, `split_channels`, `split_focal_planes`, `split_timepoints`, `group_files`, `group_pattern`, `virtual`, `specify_range`, `c_begin`, `c_end`, `c_step`, `z_begin`, `z_end`, `z_step`, `t_begin`, `t_end`, `t_step`, `crop`, `crop_x`, `crop_y`, `crop_width`, `crop_height`, `display_metadata`, `display_ome_xml`, `swap_dimensions`, `swap_order`, `level` |  |
| TIFF Virtual Stack... | `file.import_tiff_virtual` | `path`, `series`, `level` |  |
| LUT… | `file.import_lut` | `path`, `name`, `apply` |  |
| LUTs from Folder… | `file.import_luts_folder` | `folder`, `recursive` |  |
| Results… | `file.import_results` | `path`, `replace` |  |
| Table… | `file.import_table` | `path`, `name` |  |
| Text File… | `file.import_text` | `path`, `to_log` |  |
| XY Coordinates… | `file.import_xy_coordinates` | `path`, `type`, `invert_y` |  |
| Stack From List… | `file.import_stack_from_list` | `path`, `stack_as`, `title` |  |
| Raw... | `file.import_raw` | `path`, `image_type`, `width`, `height`, `offset`, `n_images`, `gap`, `little_endian`, `white_is_zero`, `open_all` |  |
| Image Sequence... | `file.import_sequence` | `folder`, `pattern`, `start`, `count`, `step`, `scale`, `sort`, `stack_as`, `recursive`, `virtual` |  |
| Text Image... | `file.import_text_image` | `path` |  |
| URL... | `file.open_url` | `url` |  |
| ImageJ ROIs (.roi/.zip)... | `file.import_ij_rois` | `path`, `add_to_manager`, `clear_first` |  |
| ROIs from JSON... | `file.import_rois_json` | `path`, `clear_first` |  |

## File/Export

| Command | id | Parameters | Notes |
|---|---|---|---|
| Bio-Formats Exporter... | `file.bioformats_export` | `path`, `format`, `compression`, `write_each_z_section`, `write_each_timepoint`, `write_each_channel`, `padded`, `c_begin`, `c_end`, `z_begin`, `z_end`, `t_begin`, `t_end`, `embed_metadata`, `description` |  |
| LUT… | `file.export_lut` | `path`, `lut` |  |
| ZIP… | `file.export_zip` | `path` |  |
| Text… | `file.export_text` | `path`, `source`, `table`, `row_numbers` |  |
| ImageJ ROIs (.roi/.zip)... | `file.export_ij_rois` | `path`, `which` |  |
| ROIs as JSON... | `file.export_rois_json` | `path`, `which` |  |
| Image Sequence... | `file.export_sequence` | `folder`, `format`, `name_pattern`, `split`, `start_index` |  |
| Video (AVI/MP4/GIF)... | `file.export_video` | `path`, `format`, `fps`, `axis`, `quality` |  |
| XY Coordinates... | `file.save_xy_coordinates` | `path`, `invert_y`, `oval_points` | needs any selection |
| Results Table... | `file.export_results` | `path`, `format`, `table`, `row_numbers` |  |

## File/New

| Command | id | Parameters | Notes |
|---|---|---|---|
| Text Window… | `file.new_text` | `name`, `text` |  |
| Table… | `table.new` | `name`, `columns` |  |

## File/Text

| Command | id | Parameters | Notes |
|---|---|---|---|
| Append Text… | `text.append` | `name`, `text` |  |
| Set Text… | `text.set` | `name`, `text` |  |
| Save Text As… | `file.save_text` | `name`, `path` |  |
| Text to Log | `text.to_log` | `name` |  |

## Edit

| Command | id | Parameters | Notes |
|---|---|---|---|
| Set Foreground Value | `edit.set_foreground` | `value` |  |
| Set Background Value | `edit.set_background` | `value` |  |
| Fill | `edit.fill` | `value` | shortcut **F**; needs area selection |
| Clear | `edit.clear` | – | needs area selection |
| Clear Outside | `edit.clear_outside` | – | needs area selection |
| Draw | `edit.draw` | `line_width` | needs any selection |
| Colour Settings | `edit.set_colors` | `foreground`, `background` |  |
| Copy | `edit.copy` | – | shortcut **Cmd+C** |
| Paste | `edit.paste` | – | shortcut **Cmd+V** |
| Cut | `edit.cut` | – | shortcut **Cmd+X**; needs area selection |
| Paste Control… | `edit.paste_control` | `mode` |  |
| Paste (with transfer mode) | `edit.paste_with_mode` | `mode` |  |
| Invert | `edit.invert` | – | shortcut **Shift+I**; live preview |
| Paint Stroke | `edit.paint_stroke` | `points`, `width`, `erase`, `all_channels` |  |
| Flood Fill | `edit.flood_fill` | `x`, `y`, `tolerance`, `connectivity`, `value` |  |

## Edit/Selection

| Command | id | Parameters | Notes |
|---|---|---|---|
| Select All | `selection.all` | – | shortcut **A** |
| Select None | `selection.none` | – | shortcut **Shift+A** |
| Enlarge | `selection.enlarge` | `pixels` | needs area selection |
| Make Band | `selection.band` | `width` | needs area selection |
| Convex Hull | `selection.convex_hull` | – | needs area selection |
| Fit Ellipse | `selection.fit_ellipse` | – | needs area selection |
| To Bounding Box | `selection.bounding_box` | – | needs any selection |
| Rotate Selection | `selection.rotate` | `angle` | needs any selection |
| Scale Selection | `selection.scale` | `x_scale`, `y_scale`, `centered` | needs any selection |
| Interpolate / Smooth | `selection.interpolate` | `interval`, `smooth`, `adjust` | needs any selection |
| Create Mask | `selection.create_mask` | – | needs area selection |
| Create Selection | `selection.create` | – |  |
| Add to ROI Manager | `selection.add_to_manager` | – | shortcut **T**; needs any selection |
| Specify | `selection.specify` | `width`, `height`, `x`, `y`, `oval`, `centered` |  |
| Line Width | `selection.line_width` | `width` | needs line selection |
| Wand (trace region) | `selection.wand` | `x`, `y`, `tolerance`, `connectivity`, `add` |  |
| Make Inverse | `selection.inverse` | – | needs area selection |
| Line to Area | `selection.line_to_area` | – | needs line selection |
| Area to Line | `selection.area_to_line` | – | needs area selection |
| Fit Circle | `selection.fit_circle` | – | needs any selection |
| Fit Rectangle | `selection.fit_rectangle` | – | needs any selection |
| Fit Spline | `selection.fit_spline` | `points` | needs any selection |
| Image to Selection… | `selection.image_to_selection` | `image`, `opacity`, `all_channels` |  |
| Select Bounding Box (guess background) | `selection.select_bounding_box` | `tolerance`, `all_channels` |  |
| Fit Circle to Image | `selection.fit_circle_to_image` | `method` |  |
| Points from Mask | `selection.points_from_mask` | `step`, `max_points`, `output` |  |
| Properties… | `selection.properties` | `name`, `color`, `stroke_width`, `fill`, `position` | needs any selection |
| Rounded Rectangle… | `selection.rounded_rect` | `x`, `y`, `w`, `h`, `corner_radius`, `arc_points` |  |

## Edit/Options

| Command | id | Parameters | Notes |
|---|---|---|---|
| Preferences… | `edit.options` | `roi_color`, `roi_stroke_width`, `font_size`, `font_family`, `antialiased_text`, `foreground`, `background`, `jpeg_quality`, `default_save_format`, `point_size`, `point_color`, `point_type`, `arrow_style`, `arrow_head_size`, `arrow_width`, `arrow_double`, `plot_y_min`, `plot_y_max`, `plot_interpolate`, `plot_width`, `plot_height`, `interpolate_zoom`, `rounded_rect_radius`, `rounded_rect_stroke_width`, `paste_mode`, `divide_by_zero`, `require_control_key`, `beep_on_error`, `auto_contrast_stacks`, `overlay_stroke_width`, `overlay_default_fill`, `overlay_show_labels`, `overlay_use_names`, `overlay_label_font_size`, `overlay_label_color`, `overlay_label_background` |  |
| Input/Output… | `edit.options_io` | `jpeg_quality`, `default_save_format` |  |
| Point Tool… | `edit.options_point_tool` | `point_size`, `point_color`, `point_type` |  |
| Arrow Tool… | `edit.options_arrow_tool` | `arrow_style`, `arrow_head_size`, `arrow_width`, `arrow_double` |  |
| Fonts… | `edit.options_fonts` | `font_family`, `font_size`, `antialiased_text` |  |
| Appearance… | `edit.options_appearance` | `interpolate_zoom`, `antialiased_text`, `auto_contrast_stacks` |  |
| Plots… | `edit.options_plots` | `plot_width`, `plot_height`, `plot_y_min`, `plot_y_max`, `plot_interpolate` |  |
| ROI Defaults… | `edit.options_roi_defaults` | `roi_color`, `roi_stroke_width` |  |
| Rounded Rect Tool… | `edit.options_rounded_rect` | `rounded_rect_radius`, `rounded_rect_stroke_width` |  |
| Misc… | `edit.options_misc` | `divide_by_zero`, `require_control_key`, `beep_on_error`, `paste_mode` |  |
| Overlay Options… | `edit.options_overlay` | `overlay_stroke_width`, `overlay_default_fill`, `overlay_show_labels`, `overlay_use_names`, `overlay_label_font_size`, `overlay_label_color`, `overlay_label_background` |  |
| Reset Preferences | `edit.options_reset` | – |  |
| Memory & Threads… | `edit.options_memory` | `undo_budget_mb`, `max_threads`, `collect_garbage` |  |
| Proxy Settings… | `edit.options_proxy` | `proxy`, `no_proxy` |  |
| Compiler… (Python plugins) | `help.compiler_info` | – |  |
| DICOM… | `edit.options_dicom` | `open_as_32bit`, `ignore_rescale`, `calibrate_images`, `window_level_on_open` |  |

## Image

| Command | id | Parameters | Notes |
|---|---|---|---|
| Crop | `image.crop` | – | shortcut **Shift+X**; needs area selection |
| Duplicate | `image.duplicate` | `title`, `what`, `crop_to_selection` | shortcut **Shift+D** |
| Scale | `image.scale` | `x_scale`, `y_scale`, `z_scale`, `width`, `height`, `interpolation`, `create_new`, `average_when_downsizing`, `keep_canvas` |  |
| Rotate | `image.rotate` | `angle`, `interpolation`, `enlarge` | live preview |
| Properties / Set Scale | `image.properties` | `pixel_width`, `pixel_height`, `voxel_depth`, `unit`, `frame_interval`, `time_unit` |  |
| Set Scale from Line | `image.set_scale_from_line` | `known_distance`, `unit` | needs line selection |
| Rename | `image.rename` | `title` |  |

## Image/Type

| Command | id | Parameters | Notes |
|---|---|---|---|
| 8-bit Color | `type.8bit_color` | `colors` |  |
| 8-bit | `type.8bit` | `scale`, `weighted` |  |
| 16-bit | `type.16bit` | `scale` |  |
| 32-bit | `type.32bit` | `weighted` |  |
| RGB Color | `type.rgb` | `create_new` |  |
| Stack to RGB (slices as R, G, B) | `type.slices_to_rgb` | `create_new` |  |
| RGB Stack | `type.rgb_stack` | – |  |
| HSB Stack | `type.hsb_stack` | – |  |
| Lab Stack | `type.lab_stack` | – |  |

## Image/Adjust

| Command | id | Parameters | Notes |
|---|---|---|---|
| Bleach Correction... | `adjust.bleach_correction` | `method`, `background`, `keep_type` |  |
| Color Threshold | `adjust.color_threshold` | `space`, `c1_min`, `c1_max`, `c1_filter`, `c2_min`, `c2_max`, `c2_filter`, `c3_min`, `c3_max`, `c3_filter`, `output` |  |
| Window/Level | `adjust.window_level` | `window`, `level`, `all_channels` |  |
| Coordinates (origin) | `adjust.coordinates` | `origin_x`, `origin_y`, `origin_z` |  |
| Apply Display Range to Pixels (B&C Apply) | `adjust.brightness_contrast_apply` | `all_channels` |  |
| Auto Crop | `image.auto_crop` | `tolerance`, `all_planes` |  |
| Auto Crop (selection) | `image.auto_crop_selection` | `tolerance`, `all_planes` |  |
| Compute Auto Threshold | `threshold.compute` | `method`, `dark_background`, `channel` |  |
| Set Threshold | `threshold.set` | `lower`, `upper` | shortcut **Shift+T** |
| Apply Threshold → Mask | `threshold.apply` | `method`, `dark_background`, `per_plane`, `only_current_plane`, `add_one`, `channel` | live preview |
| Auto Local Threshold | `threshold.local` | `method`, `radius`, `parameter_1`, `white_objects` | live preview |
| Threshold Histogram | `threshold.histogram` | – |  |
| Canvas Size | `image.canvas_size` | `width`, `height`, `position`, `zero_fill` |  |

## Image/Color

| Command | id | Parameters | Notes |
|---|---|---|---|
| Split Channels | `color.split` | – |  |
| Merge Channels | `color.merge` | `c1`, `c2`, `c3`, `c4`, `c5`, `c6`, `c7`, `create_composite`, `keep_source` |  |
| Arrange Channels | `color.arrange` | `order` |  |
| Set Channel Names | `color.channel_names` | `names` |  |

## Image/Lookup Tables

| Command | id | Parameters | Notes |
|---|---|---|---|
| Set LUT | `color.lut` | `lut`, `all_channels`, `channel` |  |
| Invert LUT | `color.invert_lut` | – |  |
| Bake LUT into RGB | `color.bake_lut` | – |  |
| Show LUT | `color.show_lut` | `width`, `ticks` |  |
| Display LUTs | `color.display_luts` | `width`, `row_height` |  |
| Edit LUT… | `color.edit_lut` | `name`, `table`, `start`, `end`, `apply` |  |
| Apply LUT | `type.apply_lut` | `all_channels` |  |

## Image/Stacks

| Command | id | Parameters | Notes |
|---|---|---|---|
| Z Project | `stacks.z_project` | `method`, `start`, `stop`, `axis` |  |
| Grouped Z Project | `stacks.grouped_z_project` | `group_size`, `method` |  |
| Make Substack | `stacks.substack` | `slices`, `axis`, `delete_from_original` |  |
| Delete Slice | `stacks.delete_slice` | `axis` |  |
| Add Slice | `stacks.add_slice` | `axis` |  |
| Reverse | `stacks.reverse` | `axis` |  |
| Make Montage | `stacks.montage` | `columns`, `rows`, `scale`, `border`, `axis` |  |
| Stack to Images | `stacks.to_images` | `axis` |  |
| Images to Stack | `stacks.images_to_stack` | `title`, `axis`, `name_filter`, `close_sources` |  |
| Concatenate | `stacks.concatenate` | `image1`, `image2`, `axis`, `title` |  |
| Plot Z-axis Profile | `stacks.z_profile` | `axis` |  |
| Reslice | `stacks.reslice` | `direction`, `interpolate_z`, `output_spacing`, `slice_count`, `rotate` |  |
| Reslice (line) | `stacks.reslice_line` | `output_spacing`, `slice_count`, `rotate`, `interpolate_z` | needs line selection |
| Radial Reslice | `stacks.radial_reslice` | `n_angles`, `total_rotation`, `length`, `interpolate_z` |  |
| Set Slice Label | `stacks.set_slice_label` | `label` |  |
| Remove Slice Labels | `stacks.remove_slice_labels` | – |  |
| Slice Labels to Results | `stacks.slice_labels_to_results` | – |  |
| Montage to Stack | `stacks.montage_to_stack` | `columns`, `rows`, `border` |  |
| Stack Statistics | `stacks.statistics` | – |  |
| 3D Project | `stacks.project3d` | `method`, `axis`, `angle_step`, `total_rotation`, `initial_angle`, `lower_threshold`, `upper_threshold`, `depth_cueing`, `interior_depth_cueing`, `interpolate`, `all_channels`, `max_size` |  |
| Orthogonal Views | `stacks.orthogonal_views` | `x`, `y`, `scale_z` |  |
| Reduce | `stacks.reduce` | `factor`, `axis` |  |
| Label Slices (time stamp) | `stacks.label_slices` | `format`, `position`, `color`, `font_size`, `start_frame`, `start`, `interval` |  |
| Combine | `stacks.combine` | `image1`, `image2`, `vertical` |  |
| Insert | `stacks.insert` | `source`, `x`, `y` |  |
| Straighten | `stacks.straighten` | `line_width`, `all_planes` | needs line selection |

## Image/Stacks/Tools

| Command | id | Parameters | Notes |
|---|---|---|---|
| Interpolate ROIs (keyframes) | `stacks.interpolate_rois` | `axis`, `name_filter`, `n_vertices`, `prefix` |  |

## Image/Hyperstacks

| Command | id | Parameters | Notes |
|---|---|---|---|
| Re-order Hyperstack | `stacks.reorder` | `order` |  |
| Stack to Hyperstack | `stacks.to_hyperstack` | `channels`, `slices`, `frames`, `order` |  |
| Hyperstack to Stack | `stacks.hyperstack_to_stack` | – |  |
| Reduce Dimensionality | `stacks.reduce_dimensionality` | `keep_channels`, `keep_slices`, `keep_frames`, `keep_source` |  |
| Make Subset | `stacks.make_subset` | `channels`, `slices`, `frames` |  |

## Image/Transform

| Command | id | Parameters | Notes |
|---|---|---|---|
| Image to Results | `image.to_results` | `replace` |  |
| Results to Image | `image.from_results` | `title`, `exclude_label` |  |
| Landmark Transform (Rigid / Similarity / Affine / Perspective) | `image.transform_landmarks` | `model`, `target_points`, `source_points`, `source_roi`, `target_roi`, `width`, `height`, `interpolation`, `create_new` |  |
| Perspective Correction (4 corners) | `image.perspective` | `corners`, `width`, `height`, `interpolation`, `create_new` |  |
| Shear | `image.shear` | `shear_x`, `shear_y`, `enlarge`, `interpolation` | live preview |
| Polar Transform | `image.warp_polar` | `scaling`, `center_x`, `center_y`, `radius`, `angles`, `radii`, `interpolation` |  |
| Rotate 90° | `image.rotate90` | `direction` |  |
| Flip Horizontally | `image.flip_h` | – |  |
| Flip Vertically | `image.flip_v` | – |  |
| Flip Z | `image.flip_z` | – |  |
| Translate | `image.translate` | `dx`, `dy`, `interpolation` | live preview |
| Bin | `image.bin` | `x`, `y`, `z`, `method` |  |
| Moving Least Squares Deformation | `image.transform_mls` | `model`, `target_points`, `source_points`, `source_roi`, `target_roi`, `alpha`, `width`, `height`, `interpolation`, `create_new` |  |
| Landmark Correspondences (two images) | `image.transform_landmark_correspondences` | `target`, `model`, `source_points`, `target_points`, `source_roi`, `target_roi`, `alpha`, `interpolation`, `show_matrix` |  |

## Image/Overlay

| Command | id | Parameters | Notes |
|---|---|---|---|
| Add Selection to Overlay | `overlay.add` | `color`, `stroke_width` | shortcut **B**; needs any selection |
| Overlay from ROI Manager | `overlay.from_manager` | – |  |
| Remove Overlay | `overlay.remove` | – |  |
| Flatten | `overlay.flatten` | – |  |
| Scale Bar | `overlay.scale_bar` | `length`, `thickness`, `position`, `color`, `show_label`, `font_size` |  |
| Add Text | `overlay.text` | `text`, `x`, `y`, `font_size`, `color`, `background`, `at_selection` |  |
| Add Arrow | `overlay.arrow` | `color`, `stroke_width`, `head_size`, `double_headed` | needs line selection |
| Calibration Bar | `overlay.calibration_bar` | `position`, `lut`, `width`, `height`, `font_size`, `color`, `decimals`, `ticks`, `fill` |  |
| Time Stamp | `overlay.time_stamp` | `format`, `position`, `color`, `font_size`, `start_frame`, `start_time`, `interval` |  |
| List Overlay Elements | `overlay.list` | – |  |
| Overlay to ROI Manager | `overlay.to_roi_manager` | `remove` |  |
| Burn In Overlay | `overlay.burn_in` | `all_planes`, `keep_overlay` |  |
| Measure Overlay | `overlay.measure` | `append`, `current_plane_only` |  |
| Add Overlay Item | `overlay.add_item` | `item` |  |
| Remove Overlay Item | `overlay.remove_item` | `id` |  |
| Clear Overlay Items of Type | `overlay.clear_type` | `type` |  |
| Add Image to Overlay | `overlay.add_image` | `image`, `x`, `y`, `opacity`, `scale` |  |
| Burn Image Overlay | `overlay.burn_image` | `image`, `x`, `y`, `opacity`, `scale` |  |

## Image/Drawing

| Command | id | Parameters | Notes |
|---|---|---|---|
| Linear Gradient | `edit.linear_gradient` | `angle`, `apply_to`, `reverse`, `fg_value`, `bg_value`, `fg_color`, `bg_color` | live preview |
| Radial Gradient | `edit.radial_gradient` | `center_x`, `center_y`, `radius`, `apply_to`, `reverse`, `fg_value`, `bg_value`, `fg_color`, `bg_color` | live preview |

## Process

| Command | id | Parameters | Notes |
|---|---|---|---|
| Image Calculator | `process.calculator` | `image1`, `operation`, `image2`, `create_new`, `float_result` |  |
| Enhance Contrast | `process.enhance_contrast` | `saturated`, `normalize`, `equalize`, `classic_equalization`, `process_all`, `use_stack_histogram` | live preview |
| CLAHE (Local Contrast) | `process.clahe` | `block_size`, `clip_limit`, `bins` | live preview |
| Subtract Background | `process.subtract_background` | `radius`, `light_background`, `create_background`, `method`, `smooth` | live preview |
| Smooth | `process.smooth` | – | shortcut **Shift+S**; live preview |
| Sharpen | `process.sharpen` | – | live preview |
| Find Edges | `process.find_edges` | – | live preview |
| Laplacian of Gaussian | `process.laplacian` | `sigma` | live preview |
| Shadows | `process.shadows` | `direction` | live preview |
| Retinex | `process.retinex` | `level`, `scale`, `scale_division`, `dynamic` | live preview |

## Process/Filters

| Command | id | Parameters | Notes |
|---|---|---|---|
| Gaussian Blur | `filters.gaussian` | `sigma`, `scaled_units` | live preview |
| Gaussian Blur 3D | `filters.gaussian3d` | `sigma_xy`, `sigma_z` | live preview |
| Median | `filters.median` | `radius`, `shape` | live preview |
| Mean | `filters.mean` | `radius`, `shape` | live preview |
| Minimum | `filters.minimum` | `radius` | live preview |
| Maximum | `filters.maximum` | `radius` | live preview |
| Variance | `filters.variance` | `radius` | live preview |
| Unsharp Mask | `filters.unsharp` | `radius`, `weight` | live preview |
| Convolve | `filters.convolve` | `kernel`, `normalize` | live preview |
| Top Hat | `filters.tophat` | `radius`, `light_background`, `dont_subtract` | live preview |
| Difference of Gaussians | `filters.dog` | `sigma1`, `sigma2` | live preview |
| Bilateral Filter | `filters.bilateral` | `sigma_spatial`, `sigma_color` | live preview |
| Non-local Means Denoise | `filters.nlmeans` | `h`, `patch_size`, `patch_distance` | live preview |
| Ridge / Vessel Filter | `filters.frangi` | `sigma_min`, `sigma_max`, `bright` | live preview |
| 3D Filters (Median / Mean / Min / Max) | `filters.rank3d` | `filter`, `radius_xy`, `radius_z` |  |
| Normalize Local Contrast | `filters.normalize_local_contrast` | `block_radius_x`, `block_radius_y`, `standard_deviations`, `center`, `stretch` | live preview |
| ROF Denoise | `filters.rof_denoise` | `theta`, `iterations` | live preview |
| Shape Index | `filters.shape_index` | `sigma`, `flat_to_zero` |  |
| Show Circular Masks | `filters.show_circular_masks` | `max_radius` |  |

## Process/Binary

| Command | id | Parameters | Notes |
|---|---|---|---|
| Make Binary | `binary.make` | `method`, `dark_background` | live preview |
| Erode | `binary.erode` | `iterations`, `count`, `pad_edges`, `radius` | live preview |
| Dilate | `binary.dilate` | `iterations`, `count`, `pad_edges`, `radius` | live preview |
| Open | `binary.open` | `iterations`, `count`, `pad_edges`, `radius` | live preview |
| Close | `binary.close` | `iterations`, `count`, `pad_edges`, `radius` | live preview |
| Outline | `binary.outline` | `mode` | live preview |
| Fill Holes | `binary.fill_holes` | – | live preview |
| Skeletonize | `binary.skeletonize` | `method` | live preview |
| Distance Map | `binary.distance_map` | `output` |  |
| Ultimate Points | `binary.ultimate_points` | `output` |  |
| Watershed | `binary.watershed` | `method`, `tolerance`, `min_distance` | live preview |
| Voronoi | `binary.voronoi` | `values` | live preview |
| Remove Small Objects | `binary.remove_small` | `min_size` | live preview |
| Label Connected Components | `binary.label` | `connectivity` |  |
| Convex Hull (binary) | `binary.convex_hull` | – | live preview |
| Dither | `binary.dither` | – | live preview |
| Binary Median (majority) | `binary.median` | `radius` | live preview |
| Signed Distance Map | `binary.signed_distance_map` | – |  |
| Distance Map 3D | `binary.distance_map_3d` | `calibrated` |  |

## Process/Math

| Command | id | Parameters | Notes |
|---|---|---|---|
| Add | `math.add` | `value` | live preview |
| Subtract | `math.subtract` | `value` | live preview |
| Multiply | `math.multiply` | `value` | live preview |
| Divide | `math.divide` | `value` | live preview |
| AND | `math.and` | `value` | live preview |
| OR | `math.or` | `value` | live preview |
| XOR | `math.xor` | `value` | live preview |
| Min | `math.min` | `value` | live preview |
| Max | `math.max` | `value` | live preview |
| Gamma | `math.gamma` | `value` | live preview |
| Set | `math.set` | `value` | live preview |
| Log | `math.log` | – | live preview |
| Exp | `math.exp` | – | live preview |
| Square | `math.square` | – | live preview |
| Square Root | `math.sqrt` | – | live preview |
| Reciprocal | `math.reciprocal` | – | live preview |
| NaN Background | `math.nan_background` | – | live preview |
| Abs | `math.abs` | – | live preview |
| Expression | `math.expression` | `expression` | live preview |
| Math ▸ Macro… | `math.macro` | `code` | live preview |

## Process/Noise

| Command | id | Parameters | Notes |
|---|---|---|---|
| Add Noise | `noise.add` | – | live preview |
| Add Specified Noise | `noise.add_specified` | `sd` | live preview |
| Salt and Pepper | `noise.salt_pepper` | `percent` | live preview |
| Despeckle | `noise.despeckle` | – | live preview |
| Remove Outliers | `noise.remove_outliers` | `radius`, `threshold`, `which` | live preview |
| Remove NaNs | `noise.remove_nans` | `radius` | live preview |

## Process/FFT

| Command | id | Parameters | Notes |
|---|---|---|---|
| FFT | `process.fft` | – |  |
| Bandpass Filter (FFT) | `process.fft_bandpass` | `large`, `small`, `suppress`, `tolerance`, `autoscale`, `saturate` | live preview |
| Inverse FFT | `process.inverse_fft` | `output`, `masking` |  |
| Swap Quadrants | `process.fft_swap_quadrants` | – | live preview |
| Redisplay Power Spectrum | `process.fft_redisplay_power_spectrum` | – |  |
| FFT Options | `process.fft_options` | `fft_window`, `raw_power_spectrum`, `complex_output`, `do_forward` |  |
| FD Math | `process.fd_math` | `image1`, `operation`, `image2`, `result`, `do_inverse`, `regularization` |  |
| Custom Filter (FFT) | `process.fft_custom_filter` | `filter`, `all_planes` |  |

## Process/Batch

| Command | id | Parameters | Notes |
|---|---|---|---|
| Batch Measure... | `batch.measure` | `input_folder`, `pattern`, `recursive`, `threshold`, `dark_background`, `all_planes`, `max_files`, `continue_on_error` |  |
| Virtual Stack (process a folder plane by plane)… | `batch.virtual_stack` | `input_folder`, `pattern`, `recursive`, `output_folder`, `script`, `format`, `save_results`, `continue_on_error`, `max_files` |  |

## Analyze

| Command | id | Parameters | Notes |
|---|---|---|---|
| Analyze Skeleton | `analyze.skeleton` | `mode`, `skeletonize_first`, `prune_ends`, `prune_cycles`, `show`, `all_planes` |  |
| Skeleton Longest Shortest Path | `analyze.skeleton_longest_path` | `mode`, `skeletonize_first`, `prune_ends`, `prune_cycles`, `all_planes` |  |
| Colocalization (Costes) | `analyze.coloc_costes` | `channel_a`, `channel_b`, `scramble_iterations`, `block_size`, `show_scatter`, `seed` |  |
| Measure | `analyze.measure` | `channel` | shortcut **M** |
| Measure Stack | `analyze.measure_stack` | `axis` |  |
| Set Measurements | `analyze.set_measurements` | `measurements`, `limit_to_threshold`, `decimals`, `display_label`, `invert_y`, `scientific_notation`, `add_to_overlay`, `nan_empty_cells`, `redirect` |  |
| Clear Results | `analyze.clear_results` | – |  |
| Analyze Particles | `analyze.particles` | `min_size`, `max_size`, `min_circularity`, `max_circularity`, `show`, `display_results`, `clear_results`, `summarize`, `add_to_manager`, `exclude_on_edges`, `include_holes`, `all_planes`, `intensity_image`, `pixel_units`, `record_starts`, `in_situ_show`, `composite_rois`, `auto_threshold`, `channel` |  |
| Histogram | `analyze.histogram` | `bins`, `all_planes`, `x_min`, `x_max`, `y_max`, `log_scale`, `copy_to_results` | shortcut **H** |
| Plot Profile | `analyze.plot_profile` | `vertical`, `all_channels` | shortcut **K**; needs line or rectangle selection |
| Find Maxima | `analyze.find_maxima` | `prominence`, `output`, `exclude_edges`, `light_background`, `strict`, `above_lower_threshold` |  |
| Summarize Results | `analyze.summarize` | `group_by` |  |
| Measure Labels | `analyze.label_objects` | `intensity_image` |  |
| Colocalization (2 channels) | `analyze.colocalization` | `channel_a`, `channel_b`, `threshold_a`, `threshold_b` |  |
| Distribution | `analyze.distribution` | `column`, `auto_binning`, `bins`, `range` |  |
| Calibrate (intensity) | `analyze.calibrate` | `function`, `pixel_values`, `standards`, `unit`, `custom_equation` |  |
| Calibrated Values (32-bit) | `analyze.calibrated_values` | – |  |
| Measure (calibrated values) | `analyze.measure_calibrated` | – |  |
| Angle | `analyze.angle` | – | needs line selection |
| Curve Fitting | `analyze.curve_fit` | `source`, `x_column`, `y_column`, `function`, `series`, `custom_equation`, `initial_params`, `restarts`, `show_residuals` |  |
| Fractal Box Count | `analyze.fractal_box_count` | `box_sizes` |  |
| Surface Plot | `analyze.surface_plot` | `smoothing`, `wireframe`, `lut`, `invert`, `elevation`, `azimuth`, `width`, `height`, `max_points` |  |
| Stack Histogram | `analyze.stack_histogram` | `bins`, `all_channels` |  |
| Multi Measure (ROI Manager) | `analyze.multi_measure` | `axis`, `one_row_per_slice`, `only_this_image`, `append` |  |
| Labels… (ROI names / values as overlay) | `analyze.label` | `what`, `column`, `decimals`, `use_names`, `font_size`, `bold`, `color`, `background`, `at`, `source`, `only_this_image` |  |

## Analyze/Gels

| Command | id | Parameters | Notes |
|---|---|---|---|
| Gel Analyzer (lanes) | `analyze.gel_lanes` | `orientation`, `dark_bands`, `min_prominence`, `smooth`, `only_this_image`, `percent_per_lane` |  |
| Gels ▸ Select Lane | `analyze.gel_select_lane` | `color` | needs area selection |
| Gels ▸ Plot Lanes | `analyze.gel_plot_lanes` | `orientation`, `dark_bands`, `smooth`, `make_image` |  |
| Gels ▸ Label Peaks | `analyze.gel_label_peaks` | `min_prominence`, `orientation`, `dark_bands`, `smooth`, `label`, `font_size`, `color`, `percent_per_lane` |  |

## Analyze/Tools

| Command | id | Parameters | Notes |
|---|---|---|---|
| Analyze Line Graph | `analyze.line_graph` | `tolerance`, `max_curves`, `invert_y`, `x_min`, `x_max`, `y_min`, `y_max` |  |
| Grid | `overlay.grid` | `type`, `area_per_point`, `spacing`, `random_offset`, `color`, `stroke_width` |  |
| Compare Images | `analyze.compare_images` | `image1`, `image2`, `difference_image` |  |
| Directionality | `analyze.directionality` | `method`, `n_bins`, `bin_start`, `sigma`, `all_planes` |  |
| ROI Manager ▸ Delete All | `rois.clear` | – |  |
| ROI Manager ▸ Delete | `rois.delete` | `indices` |  |
| ROI Manager ▸ Restore Cleared | `rois.restore_cleared` | – |  |
| Results ▸ Restore Cleared | `results.restore_cleared` | – |  |

## Plugins/Macros

| Command | id | Parameters | Notes |
|---|---|---|---|
| Edit Startup Script… | `macros.edit_startup` | `kind`, `create` |  |
| Install… | `macros.install` | `path`, `name`, `menu`, `overwrite` |  |
| Run Macro File… | `macros.run_file` | `path` |  |
| Save Recorded Macro… | `macros.record_to_file` | `path`, `title`, `clear_recorder` |  |

## Plugins/Utilities

| Command | id | Parameters | Notes |
|---|---|---|---|
| Fiji Update Site… | `plugins.fiji_update_site` | `site`, `filter` |  |
| Install from Fiji Update Site… | `plugins.fiji_install` | `site`, `files`, `with_dependencies` |  |
| Installed Fiji Plugins | `plugins.fiji_installed` | – |  |
| Run Fiji Plugin Class… | `plugins.run_fiji_plugin` | `plugin_class`, `arg`, `options` |  |
| New Plugin Template… | `plugins.new_template` | `kind`, `name`, `overwrite`, `load` |  |
| Open Plugins Folder | `plugins.open_folder` | `reveal` |  |
| List Plugins | `plugins.list` | – |  |
| Compile and Run / Update Menus | `plugins.reload` | – |  |
| DICOM Metadata | `utilities.dicom_metadata` | `filter`, `include_private` |  |
| List Commands | `utilities.list_commands` | `menu`, `kind` |  |
| Command Finder… | `utilities.command_finder` | `query`, `max_results`, `search_descriptions` |  |
| Install Plugin from File, Folder, Zip or URL… | `plugins.install_from` | `source`, `overwrite`, `menu` |  |
| Lumen Properties | `help.properties` | – |  |
| Threads | `help.threads` | – |  |
| Benchmark | `help.benchmark` | `size`, `repeats` |  |
| Reset… | `help.reset` | `clear_results`, `clear_rois`, `clear_tables`, `clear_thresholds`, `clear_macro`, `clear_log`, `close_images`, `collect_garbage` |  |
| Monitor Events | `help.monitor_events` | `n`, `level` |  |
| Search Commands… | `help.search_commands` | `query`, `max_results` |  |
| Search… | `help.search_files` | `query`, `case_sensitive`, `files`, `images`, `log`, `script`, `max_results` |  |
| Reload Plugins | `help.reload_plugins` | – |  |

## Plugins/Segmentation

| Command | id | Parameters | Notes |
|---|---|---|---|
| Train Pixel Classifier | `segment.train_classifier` | `classes`, `channels`, `features`, `min_sigma`, `max_sigma`, `membrane_thickness`, `membrane_patch_size`, `classifier`, `n_trees`, `max_depth`, `features_per_node`, `balance_classes`, `svm_max_samples`, `k_neighbors`, `hidden_units`, `max_pixels_per_class`, `class_colors`, `apply_now`, `output`, `overlay_opacity` |  |
| Apply Pixel Classifier | `segment.apply_classifier` | `image`, `all_planes`, `output`, `smooth`, `mask_class`, `probability_threshold`, `overlay_opacity` |  |
| Save Classifier | `segment.save_classifier` | `path` |  |
| Load Classifier | `segment.load_classifier` | `path` |  |
| Classifier Info | `segment.classifier_info` | `kind` |  |
| Classifier Overlay | `segment.classifier_overlay` | `action`, `opacity` |  |
| Classifier Trace Tool | `segment.trace_tool` | `class_name`, `width`, `color` |  |
| Apply Pixel Classifier to Folder... | `segment.apply_classifier_folder` | `input_folder`, `pattern`, `recursive`, `output_folder`, `output`, `output_format`, `use_3d`, `all_planes`, `smooth`, `continue_on_error`, `max_files` |  |
| Random Walker | `segment.random_walker` | `classes`, `markers_image`, `beta`, `mode`, `smooth`, `output` |  |
| Active Contour (Chan-Vese) | `segment.active_contour` | `mu`, `lambda1`, `lambda2`, `iterations`, `init`, `tolerance`, `dt`, `dark_background`, `to_selection` |  |
| Morphological Snakes | `segment.morph_snakes` | `method`, `iterations`, `smoothing`, `init`, `lambda1`, `lambda2`, `alpha`, `sigma`, `balloon`, `dark_background`, `to_selection` |  |
| Superpixels | `segment.superpixels` | `method`, `n_segments`, `compactness`, `sigma`, `scale`, `min_size`, `kernel_size`, `max_dist`, `ratio`, `output` |  |
| Multi-Otsu Threshold | `segment.multi_otsu` | `classes`, `output`, `all_planes` |  |
| Cellpose | `segment.cellpose` | `model`, `custom_model`, `diameter`, `cyto_channel`, `nuclei_channel`, `flow_threshold`, `cellprob_threshold`, `use_gpu`, `add_to_manager`, `all_planes` |  |
| StarDist | `segment.stardist` | `model`, `custom_model`, `prob_thresh`, `nms_thresh`, `normalize`, `n_tiles`, `add_to_manager`, `all_planes` |  |
| Labels to ROIs | `segment.labels_to_rois` | `min_size`, `simplify`, `clear_manager` |  |
| ROIs to Labels | `segment.rois_to_labels` | `use_positions` |  |
| Filter Labels | `segment.label_filter` | `min_area`, `max_area`, `min_circularity`, `max_circularity`, `min_solidity`, `max_solidity`, `intensity_image`, `min_mean`, `max_mean`, `exclude_edges`, `all_planes` |  |
| Expand Labels | `segment.expand_labels` | `distance` | live preview |
| Shrink Labels | `segment.shrink_labels` | `distance` | live preview |
| Remove Border Labels | `segment.remove_border_labels` | `buffer` | live preview |
| Merge Touching Labels | `segment.merge_touching_labels` | `connectivity` | live preview |
| Seeded Watershed | `segment.seeded_watershed` | `markers`, `smooth`, `invert`, `use_gradient`, `use_threshold`, `prominence`, `compactness`, `watershed_line`, `add_to_manager` |  |
| Color Clustering | `segment.color_clustering` | `n_clusters`, `color_space`, `spatial_weight`, `max_samples`, `seed`, `color_image` |  |
| Train 3D Pixel Classifier | `segment3d.train_classifier` | `classes`, `channels`, `features`, `min_sigma`, `max_sigma`, `classifier`, `n_trees`, `max_depth`, `features_per_node`, `balance_classes`, `svm_max_samples`, `k_neighbors`, `hidden_units`, `max_voxels_per_class`, `class_colors`, `max_memory_mb`, `apply_now`, `output`, `overlay_opacity` |  |
| Apply 3D Pixel Classifier | `segment3d.apply_classifier` | `image`, `all_frames`, `output`, `smooth`, `mask_class`, `probability_threshold`, `max_memory_mb`, `overlay_opacity` |  |
| Save 3D Classifier | `segment3d.save_classifier` | `path` |  |
| Load 3D Classifier | `segment3d.load_classifier` | `path` |  |
| Label 3D (Connected Components) | `segment3d.label_3d` | `connectivity`, `all_frames` |  |
| Measure Labels 3D | `segment3d.measure_labels_3d` | `intensity_image`, `surface`, `min_voxels`, `clear_results`, `all_frames` |  |
| Filter Labels 3D | `segment3d.filter_labels_3d` | `min_volume`, `max_volume`, `in_voxels`, `exclude_edges`, `intensity_image`, `min_mean`, `max_mean`, `all_frames` |  |
| Distance Watershed 3D | `segment3d.watershed_3d` | `tolerance`, `smooth`, `connectivity`, `watershed_line`, `output`, `all_frames` |  |
| Expand Labels 3D | `segment3d.expand_labels_3d` | `distance` |  |
| Labels to ROIs 3D | `segment3d.labels_to_rois_3d` | `min_voxels`, `max_rois`, `slice_step`, `simplify`, `clear_manager` |  |
| Seeded Watershed 3D | `segment3d.seeded_watershed_3d` | `markers`, `markers_image`, `smooth`, `invert`, `use_gradient`, `use_threshold`, `mask_image`, `prominence`, `connectivity`, `compactness`, `watershed_line`, `add_to_manager`, `max_rois` |  |
| Random Walker 3D | `segment3d.random_walker_3d` | `classes`, `markers_image`, `beta`, `downsample`, `mode`, `smooth`, `output` |  |

## Plugins/MorphoLibJ

| Command | id | Parameters | Notes |
|---|---|---|---|
| Analyze Regions | `morpho.analyze_regions` | `features`, `crofton_directions`, `connectivity`, `overlays`, `all_planes` |  |
| Intensity Measurements 2D/3D | `morpho.intensity_measurements` | `intensity_image`, `mode`, `measurements` |  |
| Keep Largest Label | `morpho.keep_largest_label` | – | live preview |
| Remove Largest Label | `morpho.remove_largest_label` | – | live preview |
| Replace / Remove Labels | `morpho.replace_label` | `labels`, `new_value` | live preview |
| Crop Label | `morpho.crop_label` | `label`, `border` |  |
| Label Boundaries | `morpho.labels_to_boundaries` | `mode`, `connectivity`, `all_planes` |  |
| Dilate Labels | `morpho.dilate_labels` | `radius`, `mask_image`, `all_planes` |  |
| Label Morphological Filters | `morpho.label_morphology` | `operation`, `radius` | live preview |
| Region Adjacency Graph | `morpho.region_adjacency_graph` | `mode`, `connectivity`, `show_overlay` |  |
| Label Overlap Measures | `morpho.label_overlap_measures` | `reference_image`, `matching`, `mode` |  |
| Morphological Segmentation | `morpho.morphological_segmentation` | `input`, `gradient`, `gradient_radius`, `tolerance`, `connectivity`, `calculate_dams`, `output`, `smooth`, `all_planes` |  |
| Labels To RGB | `morpho.labels_to_rgb` | `lut`, `background`, `shuffle`, `seed`, `all_planes` |  |
| Assign Measure to Label | `morpho.assign_measure_to_label` | `column`, `table`, `label_column`, `missing`, `all_planes` |  |
| Microstructure Analysis | `morpho.microstructure_analysis` | `mode`, `regions`, `features`, `directions_2d`, `directions_3d`, `connectivity` |  |
| Analyze Regions 3D | `morpho.analyze_regions_3d` | `features`, `directions`, `connectivity`, `all_frames` |  |
| Select Label | `morpho.select_label` | `label`, `add_to_manager` |  |
| Remove Label at Point | `morpho.remove_label_at` | `x`, `y`, `all_planes` |  |
| Merge Labels | `morpho.merge_labels` | `labels`, `new_value` | live preview |
| Fill Label Holes | `morpho.fill_label_holes` | – | live preview |
| Label Size Opening | `morpho.label_size_opening` | `min_size`, `max_size` | live preview |
| Morphological Filters | `morpho.filters` | `operation`, `element`, `radius`, `line_angle` | live preview |
| Morphological Filters (3D) | `morpho.filters_3d` | `operation`, `element`, `radius_x`, `radius_y`, `radius_z` |  |
| Regional Min & Max | `morpho.regional_min_max` | `operation`, `connectivity`, `mode` |  |
| Extended Min & Max | `morpho.extended_min_max` | `operation`, `dynamic`, `connectivity`, `mode` |  |
| Impose Min & Max | `morpho.impose_min_max` | `operation`, `marker_image`, `connectivity`, `mode` |  |
| Chamfer Distance Map | `morpho.chamfer_distance_map` | `weights`, `normalize`, `output`, `mode`, `all_planes` |  |
| Geodesic Distance Map | `morpho.geodesic_distance_map` | `marker_image`, `weights`, `normalize` |  |
| Marker-controlled Watershed | `morpho.marker_watershed` | `markers`, `marker_image`, `mask_image`, `input`, `connectivity`, `calculate_dams`, `mode`, `add_to_manager` |  |
| Morphological Reconstruction | `morpho.reconstruct` | `operation`, `marker_image`, `connectivity`, `mode` |  |
| Kill Borders | `morpho.kill_borders` | `connectivity` | live preview |
| Fill Holes (Binary / Gray) | `morpho.fill_holes` | `connectivity` | live preview |
| Area Opening | `morpho.area_opening` | `operation`, `min_area`, `connectivity` | live preview |
| Directional Filtering | `morpho.directional_filter` | `type`, `operation`, `line_length`, `n_directions` | live preview |
| Binary Overlay | `morpho.binary_overlay` | `mask_image`, `color`, `opacity`, `all_planes` |  |
| Label Overlay | `morpho.label_overlay` | `label_image`, `lut`, `opacity`, `shuffle`, `all_planes` |  |

## Plugins/Tracking

| Command | id | Parameters | Notes |
|---|---|---|---|
| Optic Flow (Dense) | `process.optic_flow` | `channel`, `method`, `downscale`, `smoothing`, `attachment`, `tightness`, `num_warp`, `num_iter`, `radius`, `magnitude`, `direction`, `components`, `vector_overlay`, `grid`, `vector_scale`, `min_magnitude`, `color` |  |
| Detect Spots | `tracking.detect_spots` | `detector`, `radius`, `threshold`, `subpixel`, `median_filter`, `units`, `z_mode`, `add_to_manager`, `add_to_results` |  |
| Link Spots (LAP) | `tracking.link_spots` | `tracker`, `max_displacement`, `gap_closing_frames`, `max_gap_distance`, `allow_splitting`, `split_max_distance`, `allow_merging`, `merge_max_distance`, `min_track_length`, `alternative_cost_factor`, `linking_penalties`, `gap_penalties`, `split_penalties`, `merge_penalties`, `units` |  |
| Track (detect + link) | `tracking.track` | `detector`, `radius`, `threshold`, `subpixel`, `median_filter`, `units`, `z_mode`, `tracker`, `max_displacement`, `gap_closing_frames`, `max_gap_distance`, `allow_splitting`, `split_max_distance`, `allow_merging`, `merge_max_distance`, `min_track_length`, `alternative_cost_factor`, `linking_penalties`, `gap_penalties`, `split_penalties`, `merge_penalties`, `add_to_manager`, `add_to_results` |  |
| Export Tracks (CSV) | `tracking.export_tracks` | `path`, `layout`, `include_unlinked` |  |
| Kymograph | `tracking.kymograph` | `axis`, `all_channels` | needs line selection |
| Clear Tracks | `tracking.clear_tracks` | `forget_spots` |  |
| Filter Tracks | `tracking.filter_tracks` | `min_length`, `min_duration`, `min_displacement`, `max_speed`, `feature`, `threshold`, `above` |  |
| Manual Tracking: New Track | `tracking.manual_new_track` | `name`, `color` |  |
| Manual Tracking: Add Point | `tracking.manual_add_point` | `track_id`, `x`, `y`, `t`, `radius`, `advance` |  |
| Manual Tracking: Delete Point | `tracking.manual_delete_point` | `track_id`, `t`, `delete_track` |  |
| Export Tracks (TrackMate XML) | `tracking.export_trackmate_xml` | `path`, `include_unlinked` |  |
| Import Tracks (TrackMate XML) | `tracking.import_trackmate_xml` | `path`, `only_filtered`, `units`, `add_to_manager` |  |
| Tracks to ROIs | `tracking.tracks_to_rois` | `width` |  |
| Filter Spots | `tracking.filter_spots` | `feature`, `threshold`, `above` |  |
| Feature Tables (Spots / Edges / Tracks) | `tracking.feature_tables` | `include_unlinked`, `max_rows` |  |
| Export Feature Tables (CSV) | `tracking.export_feature_tables` | `path`, `which`, `include_unlinked`, `trackmate_headers` |  |
| Split Track at Frame | `tracking.split_track` | `track_id`, `frame` |  |
| Merge Two Tracks | `tracking.merge_tracks` | `track_id`, `other_id` |  |
| Link Two Spots | `tracking.link_two_spots` | `spot_a`, `spot_b` |  |
| Unlink Two Spots | `tracking.unlink_spots` | `spot_a`, `spot_b` |  |
| Add Spot | `tracking.add_spot` | `x`, `y`, `t`, `radius`, `link_to`, `track_id` |  |
| Delete Spot | `tracking.delete_spot` | `spot_id`, `track_id`, `t`, `bridge` |  |
| Undo Track Edit | `tracking.undo_edit` | – |  |
| Track Display Settings | `tracking.set_display` | `color_by`, `colormap`, `display_mode`, `tail_length`, `spot_radius`, `show_names` |  |

## Plugins/Registration

| Command | id | Parameters | Notes |
|---|---|---|---|
| Align Stack (StackReg / SIFT) | `registration.align_stack` | `axis`, `mode`, `estimator`, `reference`, `channel`, `upsample`, `interpolation`, `create_new`, `initial_blur`, `steps_per_octave`, `min_image_size`, `max_image_size`, `descriptor_size`, `orientation_bins`, `max_ratio`, `max_alignment_error`, `min_inlier_ratio`, `pyramid_levels`, `iterations`, `show_info`, `show_matrix` |  |
| Align Channels | `registration.align_channels` | `reference_channel`, `estimate_on`, `upsample`, `interpolation`, `create_new` |  |
| Register to Image | `registration.register_to_image` | `target`, `mode`, `upsample`, `interpolation` |  |
| Pairwise Stitching | `registration.pairwise_stitch` | `image1`, `image2`, `fusion`, `check_peaks`, `title` |  |
| Grid Stitching | `registration.grid_stitch` | `folder`, `pattern`, `grid_x`, `grid_y`, `overlap_percent`, `order`, `refine`, `fusion`, `min_correlation`, `title` |  |
| Correct Drift (3D) | `registration.drift_correct` | `channel`, `reference`, `correct_z`, `upsample`, `interpolation`, `create_new` |  |
| Elastic Alignment (Optical Flow) | `registration.elastic_align` | `axis`, `reference`, `channel`, `method`, `downscale`, `smoothing`, `attachment`, `tightness`, `num_warp`, `num_iter`, `radius`, `interpolation`, `output_displacement`, `create_new` |  |
| Elastic Registration (bUnwarpJ-style) | `registration.elastic_pair` | `target`, `method`, `downscale`, `smoothing`, `attachment`, `tightness`, `num_warp`, `num_iter`, `radius`, `interpolation`, `output_field`, `store` |  |
| Apply Deformation | `registration.apply_deformation` | `field`, `interpolation`, `invert` |  |
| Register Virtual Stack Slices (Folder) | `registration.align_folder` | `folder`, `pattern`, `output_folder`, `mode`, `reference`, `channel`, `upsample`, `interpolation`, `method`, `downscale`, `smoothing`, `output_format` |  |
| Phase Correlation | `registration.phase_correlation` | `target`, `upsample`, `large_shifts` |  |
| Extract SIFT Correspondences | `registration.feature_correspondences` | `target`, `detector`, `model`, `max_features`, `max_ratio`, `residual_threshold`, `add_rois` |  |
| Register 3D Volume to Image | `registration.register_3d` | `target`, `mode`, `channel`, `upsample`, `source_points`, `target_points`, `interpolation`, `create_new` |  |
| Correct 3D Drift (phase correlation) | `registration.align_time_3d` | `channel`, `reference`, `upsample`, `max_shift`, `interpolation`, `crop_to_common`, `create_new` |  |
| Align Channels (3D) | `registration.multi_channel_3d` | `reference_channel`, `estimate_on`, `upsample`, `max_shift`, `interpolation`, `create_new` |  |
| Extended Depth of Field (Stack Focuser) | `registration.stack_focus` | `method`, `kernel_size`, `channel`, `smooth_height`, `height_map` |  |
| Block Matching (sparse displacement field) | `registration.block_matching_2d` | `target`, `block_size`, `search_radius`, `grid_spacing`, `min_correlation`, `arrows`, `vector_scale`, `color`, `warp`, `smoothing`, `interpolation`, `output_field`, `store` |  |
| bUnwarpJ (B-spline Elastic Registration) | `registration.bunwarpj` | `target`, `mode`, `subsampling`, `initial_deformation`, `final_deformation`, `divergence_weight`, `curl_weight`, `landmark_weight`, `image_weight`, `consistency_weight`, `stop_threshold`, `source_landmarks`, `target_landmarks`, `interpolation`, `output_field`, `output_target`, `save_direct`, `save_inverse`, `store`, `max_iterations` |  |
| bUnwarpJ ▸ Apply Elastic / Raw Transform | `registration.bunwarpj_apply` | `file`, `direction`, `target`, `interpolation`, `invert` |  |
| bUnwarpJ ▸ Load Transform | `registration.bunwarpj_load` | `file`, `as_inverse`, `width`, `height`, `output_field` |  |
| bUnwarpJ ▸ Save Transform | `registration.bunwarpj_save` | `file`, `inverse_file`, `format` |  |
| bUnwarpJ ▸ Invert Transform | `registration.bunwarpj_invert` | `file`, `width`, `height`, `output_file`, `swap` |  |
| bUnwarpJ ▸ Compare Transforms | `registration.bunwarpj_compare` | `file1`, `file2`, `mode`, `width`, `height` |  |
| bUnwarpJ ▸ Convert Transform (Elastic ↔ Raw) | `registration.bunwarpj_convert` | `file`, `output_file`, `mode`, `intervals`, `width`, `height` |  |
| Descriptor-based Registration (2D/3D) | `registration.descriptor_registration` | `target`, `objects`, `size`, `threshold`, `max_detections`, `model`, `n_neighbors`, `redundancy`, `significance`, `ransac_threshold`, `ransac_iterations`, `min_inlier_ratio`, `min_inliers`, `channel`, `also_apply_to`, `interpolation`, `add_rois`, `create_new` |  |
| Descriptor-based Series Registration (2D/3D + t) | `registration.descriptor_series` | `objects`, `size`, `threshold`, `max_detections`, `model`, `reference`, `n_neighbors`, `redundancy`, `significance`, `ransac_threshold`, `ransac_iterations`, `min_inlier_ratio`, `min_inliers`, `channel`, `interpolation`, `create_new` |  |
| Register with Manual Landmarks | `registration.landmark_pairs` | `model`, `source_roi`, `target_roi`, `target`, `interpolation`, `apply`, `create_new` |  |
| Grid/Collection Stitching (positions, unknown, 3-D) | `registration.stitch_collection` | `layout`, `folder`, `pattern`, `layout_file`, `images`, `positions`, `grid_x`, `grid_y`, `overlap_percent`, `order`, `compute_overlap`, `channel`, `timelapse`, `check_peaks`, `regression_threshold`, `max_avg_displacement`, `abs_displacement`, `subpixel`, `downsample`, `fusion`, `output`, `output_path`, `registered_file`, `title` |  |
| Elastic Montaging | `registration.elastic_montage` | `layout`, `folder`, `pattern`, `layout_file`, `images`, `positions`, `grid_x`, `grid_y`, `overlap_percent`, `order`, `channel`, `method`, `downscale`, `smoothing`, `mesh_spacing`, `stiffness`, `min_correlation`, `max_displacement`, `interpolation`, `output`, `output_path`, `title` |  |

## Plugins/3D

| Command | id | Parameters | Notes |
|---|---|---|---|
| Export Surface (OBJ/STL) | `mesh.export` | `path`, `format`, `level`, `channel`, `step`, `smooth`, `max_dim`, `min_faces`, `calibrated` |  |
| Measure Surface (3D) | `mesh.measure` | `level`, `channel`, `step`, `smooth`, `max_dim`, `min_faces`, `max_rows` |  |
| Save 3D Scene... | `scene.save` | `path`, `scene` |  |
| Load 3D Scene... | `scene.load` | `path` |  |
| Landmarks to Results (3D) | `scene.landmarks_to_results` | `landmarks`, `add_rois`, `distances` |  |

## Plugins/Batch

| Command | id | Parameters | Notes |
|---|---|---|---|
| Process Folder... | `batch.process_folder` | `input_folder`, `pattern`, `recursive`, `output_folder`, `output_format`, `script`, `save_results`, `close_after`, `continue_on_error`, `max_files` |  |
| Apply Recorded Macro to Folder... | `batch.apply_recorded_macro` | `input_folder`, `pattern`, `recursive`, `output_folder`, `output_format`, `lines`, `save_results`, `close_after`, `continue_on_error`, `max_files` |  |
| Run Script on All Open Images... | `batch.run_script_on_open_images` | `script`, `continue_on_error` |  |

## Plugins/Scripting

| Command | id | Parameters | Notes |
|---|---|---|---|
| Run ImageJ Macro | `script.run_ij_macro` | `macro` |  |
| ImageJ Command List | `script.ij_command_list` | – |  |
| Run JVM Script (Groovy / BeanShell / Rhino) | `script.run_jvm_script` | `code`, `language` |  |
| JVM Script Bridge Status | `script.jvm_bridge_status` | – |  |

## Plugins/Bio-Formats

| Command | id | Parameters | Notes |
|---|---|---|---|
| Stack Slicer... | `file.stack_slicer` | `split_channels`, `split_z`, `split_timepoints`, `stack_order`, `keep_original` |  |
| Bio-Formats Plugins Configuration | `file.bioformats_config` | – |  |

## Plugins/Feature Extraction

| Command | id | Parameters | Notes |
|---|---|---|---|
| Extract Correspondences (SIFT / MOPS / ORB / Block matching) | `registration.extract_correspondences` | `target`, `method`, `model`, `initial_blur`, `steps_per_octave`, `min_image_size`, `max_image_size`, `descriptor_size`, `orientation_bins`, `max_ratio`, `max_alignment_error`, `min_inlier_ratio`, `max_features`, `block_size`, `search_radius`, `min_correlation`, `add_rois` |  |

## Plugins/Frames

| Command | id | Parameters | Notes |
|---|---|---|---|
| Image Statistics Frame | `frames.image_statistics` | `plane`, `decimals`, `use_selection` |  |

## Plugins/Time Lapse

| Command | id | Parameters | Notes |
|---|---|---|---|
| Phase Map | `timelapse.phase_map` | `channel`, `min_period`, `max_period`, `detrend`, `window`, `phase_unit`, `frequency_output`, `reference`, `all_slices` |  |

## Plugins/Tools

| Command | id | Parameters | Notes |
|---|---|---|---|
| Spot Click Counter | `tools.spot_counter` | `marker_size`, `color`, `all_planes` |  |
| Measure Distance Tool | `tools.measure_distance` | `add_to_overlay`, `color`, `line_width`, `decimals` |  |

## Help

| Command | id | Parameters | Notes |
|---|---|---|---|
| Lumen / ImageJ Links | `help.links` | – |  |
| About Plugins / Command Modules | `help.about_plugins` | – |  |
| Command Help… | `help.command` | `command` |  |
| Check for Updates… | `help.check_updates` | `check_pypi` |  |
| Update Lumen… | `help.update` | `check_only`, `dry_run`, `timeout` |  |
