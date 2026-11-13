// crossval: title='Grays' LUT + RGB Color (pixel-exact table) | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
run("Grays");
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
