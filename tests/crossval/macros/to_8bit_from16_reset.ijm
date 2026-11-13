// crossval: title=16-bit → 8-bit after resetMinAndMax (scaled to the data range) | images=blobs16 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
resetMinAndMax();
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
