// crossval: title=Flip Vertically | images=blobs8,blobs32 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Flip Vertically");
saveAs("Tiff", dir + "/result.tif");
