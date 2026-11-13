// crossval: title=Rotate 90 Degrees Right | images=blobs8,blobs16 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Rotate 90 Degrees Right");
saveAs("Tiff", dir + "/result.tif");
