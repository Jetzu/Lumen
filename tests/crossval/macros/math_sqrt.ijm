// crossval: title=Math: run("Square Root"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Square Root");
saveAs("Tiff", dir + "/result.tif");
