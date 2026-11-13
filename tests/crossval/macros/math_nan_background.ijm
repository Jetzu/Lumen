// crossval: title=Math: run("NaN Background"); | images=blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
setThreshold(0.3, 10);
run("NaN Background");
saveAs("Tiff", dir + "/result.tif");
