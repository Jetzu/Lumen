// crossval: title=Math: run("Macro...", "code=[v=v*0.5+x/4]"); | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Macro...", "code=[v=v*0.5+x/4]");
saveAs("Tiff", dir + "/result.tif");
