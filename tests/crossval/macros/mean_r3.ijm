// crossval: title=Mean… radius=3 | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Mean...", "radius=3");
saveAs("Tiff", dir + "/result.tif");
