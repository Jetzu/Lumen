// crossval: title=Invert LUT + RGB Color | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
run("Invert LUT");
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
