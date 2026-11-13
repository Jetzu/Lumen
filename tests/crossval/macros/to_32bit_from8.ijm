// crossval: title=8-bit → 32-bit | images=blobs8 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("32-bit");
saveAs("Tiff", dir + "/result.tif");
