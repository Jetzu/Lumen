// crossval: title=Math: run("Min...", "value=50"); | images=blobs8 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Min...", "value=50");
saveAs("Tiff", dir + "/result.tif");
