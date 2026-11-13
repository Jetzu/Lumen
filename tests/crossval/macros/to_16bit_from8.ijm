// crossval: title=8-bit → 16-bit | images=blobs8 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("16-bit");
saveAs("Tiff", dir + "/result.tif");
