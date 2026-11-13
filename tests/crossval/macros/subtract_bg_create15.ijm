// crossval: title=Subtract Background… rolling=15 create (background image) | images=blobs8 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Subtract Background...", "rolling=15 create");
saveAs("Tiff", dir + "/result.tif");
