// crossval: title=Crop to a rectangle | images=blobs8,blobs16,blobs32 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(20, 15, 60, 45);
run("Crop");
saveAs("Tiff", dir + "/result.tif");
