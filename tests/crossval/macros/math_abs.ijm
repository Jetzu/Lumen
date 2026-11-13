// crossval: title=Math: run("Abs"); | images=blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Abs");
saveAs("Tiff", dir + "/result.tif");
