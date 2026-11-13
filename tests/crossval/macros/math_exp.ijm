// crossval: title=Math: run("Exp"); | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Exp");
saveAs("Tiff", dir + "/result.tif");
