// crossval: title=Subtract Background… rolling=15 (rolling ball) | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Subtract Background...", "rolling=15");
saveAs("Tiff", dir + "/result.tif");
