// crossval: title=Subtract Background… rolling=15 sliding (sliding paraboloid) | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Subtract Background...", "rolling=15 sliding");
saveAs("Tiff", dir + "/result.tif");
