// crossval: title=Math: run("Gamma...", "value=0.5"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Gamma...", "value=0.5");
saveAs("Tiff", dir + "/result.tif");
