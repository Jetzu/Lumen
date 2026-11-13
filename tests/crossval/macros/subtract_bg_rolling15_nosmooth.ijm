// crossval: title=Subtract Background… rolling=15 disable (no smoothing) | images=blobs8 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Subtract Background...", "rolling=15 disable");
saveAs("Tiff", dir + "/result.tif");
