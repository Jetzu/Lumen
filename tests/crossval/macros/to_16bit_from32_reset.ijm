// crossval: title=32-bit → 16-bit after resetMinAndMax | images=blobs32 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
resetMinAndMax();
run("16-bit");
saveAs("Tiff", dir + "/result.tif");
