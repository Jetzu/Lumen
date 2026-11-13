// crossval: title=8-bit (Grays) → RGB Color | images=blobs8 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
