// crossval: title=Math: run("Reciprocal"); | images=blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Reciprocal");
saveAs("Tiff", dir + "/result.tif");
