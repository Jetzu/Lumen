// crossval: title=Math: run("Subtract...", "value=50"); | images=blobs8,blobs16 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Subtract...", "value=50");
saveAs("Tiff", dir + "/result.tif");
