// crossval: title=Minimum… radius=2 | images=blobs8,blobs16 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Minimum...", "radius=2");
saveAs("Tiff", dir + "/result.tif");
