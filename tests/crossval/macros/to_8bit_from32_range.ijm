// crossval: title=32-bit → 8-bit with setMinAndMax(0, 1) | images=blobs32 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
setMinAndMax(0, 1);
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
