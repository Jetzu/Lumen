// crossval: title=Flip Horizontally | images=blobs8,blobs16 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Flip Horizontally");
saveAs("Tiff", dir + "/result.tif");
