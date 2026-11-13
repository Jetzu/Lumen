// crossval: title=32-bit → 16-bit with setOption('ScaleConversions', false) | images=blobs32 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
setOption("ScaleConversions", false);
run("16-bit");
saveAs("Tiff", dir + "/result.tif");
