// crossval: title=Gaussian Blur… sigma=0.7 | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Gaussian Blur...", "sigma=0.7");
saveAs("Tiff", dir + "/result.tif");
