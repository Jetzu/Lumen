// crossval: title=Subtract Background… rolling=30 light | images=blobs8 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Invert");
run("Subtract Background...", "rolling=30 light");
saveAs("Tiff", dir + "/result.tif");
