// crossval: title=16-bit → 8-bit with setMinAndMax(500, 3000) | images=blobs16 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
setMinAndMax(500, 3000);
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
