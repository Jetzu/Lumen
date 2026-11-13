// crossval: title=Math: run("Set...", "value=77"); | images=blobs8 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Set...", "value=77");
saveAs("Tiff", dir + "/result.tif");
