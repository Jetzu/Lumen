// crossval: title=setMinAndMax(50, 180) + RGB Color (display range baked in) | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
setMinAndMax(50, 180);
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
