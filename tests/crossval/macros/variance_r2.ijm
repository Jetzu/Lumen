// crossval: title=Variance… radius=2 | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Variance...", "radius=2");
saveAs("Tiff", dir + "/result.tif");
