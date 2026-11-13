// crossval: title=Gaussian Blur… sigma=1 scaled (pixel 0.5×0.25 µm) | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Properties...", "unit=um pixel_width=0.5 pixel_height=0.25 voxel_depth=1");
run("Gaussian Blur...", "sigma=1 scaled");
saveAs("Tiff", dir + "/result.tif");
