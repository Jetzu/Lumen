// crossval: title=Top Hat… radius=10 | images=blobs8 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Top Hat...", "radius=10");
saveAs("Tiff", dir + "/result.tif");
