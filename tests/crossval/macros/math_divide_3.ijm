// crossval: title=Math: run("Divide...", "value=3"); | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Divide...", "value=3");
saveAs("Tiff", dir + "/result.tif");
