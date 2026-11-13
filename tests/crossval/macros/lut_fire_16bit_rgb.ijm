// crossval: title='Fire' LUT on 16-bit after resetMinAndMax + RGB Color | images=blobs16 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
resetMinAndMax();
run("Fire");
run("RGB Color");
saveAs("Tiff", dir + "/result.tif");
