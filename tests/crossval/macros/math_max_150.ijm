// crossval: title=Math: run("Max...", "value=150"); | images=blobs8,blobs16 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Max...", "value=150");
saveAs("Tiff", dir + "/result.tif");
