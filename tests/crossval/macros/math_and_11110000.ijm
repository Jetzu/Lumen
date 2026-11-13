// crossval: title=Math: run("AND...", "value=11110000"); | images=blobs8 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("AND...", "value=11110000");
saveAs("Tiff", dir + "/result.tif");
