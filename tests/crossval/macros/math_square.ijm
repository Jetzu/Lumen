// crossval: title=Math: run("Square"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Square");
saveAs("Tiff", dir + "/result.tif");
