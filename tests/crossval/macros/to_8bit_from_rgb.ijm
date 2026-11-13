// crossval: title=RGB → 8-bit (unweighted) | images=blobs8 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("Fire");
run("RGB Color");
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
