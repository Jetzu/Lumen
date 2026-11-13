// crossval: title=setMinAndMax(50, 180) + Apply LUT | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
setMinAndMax(50, 180);
run("Apply LUT");
saveAs("Tiff", dir + "/result.tif");
