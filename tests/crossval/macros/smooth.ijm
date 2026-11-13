// crossval: title=Smooth | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process
dir = getArgument();
open(dir + "/input.tif");
run("Smooth");
saveAs("Tiff", dir + "/result.tif");
