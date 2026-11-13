// crossval: title='3-3-2 RGB' LUT + RGB Color (pixel-exact table) | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
run("3-3-2 RGB");
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
