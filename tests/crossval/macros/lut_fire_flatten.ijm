// crossval: title='Fire' LUT + Flatten | images=blobs8 | compare=pixels | group=Image ▸ Lookup Tables
dir = getArgument();
open(dir + "/input.tif");
run("Fire");
run("Flatten");
saveAs("Tiff", dir + "/result.tif");
