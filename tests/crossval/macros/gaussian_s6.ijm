// crossval: title=Gaussian Blur… sigma=6 | images=blobs8,blobs16 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Gaussian Blur...", "sigma=6");
saveAs("Tiff", dir + "/result.tif");
