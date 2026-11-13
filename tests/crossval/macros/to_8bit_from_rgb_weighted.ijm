// crossval: title=RGB → 8-bit with weighted conversion | images=blobs8 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
run("Fire");
run("RGB Color");
run("Conversions...", "scale weighted");
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
