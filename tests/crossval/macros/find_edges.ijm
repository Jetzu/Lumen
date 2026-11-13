// crossval: title=Find Edges | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process
dir = getArgument();
open(dir + "/input.tif");
run("Find Edges");
saveAs("Tiff", dir + "/result.tif");
