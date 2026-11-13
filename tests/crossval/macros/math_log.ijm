// crossval: title=Math: run("Log"); | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Log");
saveAs("Tiff", dir + "/result.tif");
