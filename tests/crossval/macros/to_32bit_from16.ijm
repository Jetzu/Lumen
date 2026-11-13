// crossval: title=16-bit → 32-bit | images=blobs16 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("32-bit");
saveAs("Tiff", dir + "/result.tif");
