// crossval: title=Gaussian Blur… sigma=2 | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Gaussian Blur...", "sigma=2");
saveAs("Tiff", dir + "/result.tif");
