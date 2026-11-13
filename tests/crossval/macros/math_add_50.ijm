// crossval: title=Math: run("Add...", "value=50"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Add...", "value=50");
saveAs("Tiff", dir + "/result.tif");
