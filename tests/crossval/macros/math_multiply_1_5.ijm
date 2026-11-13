// crossval: title=Math: run("Multiply...", "value=1.5"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Multiply...", "value=1.5");
saveAs("Tiff", dir + "/result.tif");
