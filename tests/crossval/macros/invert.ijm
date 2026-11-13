// crossval: title=Edit ▸ Invert | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Invert");
saveAs("Tiff", dir + "/result.tif");
