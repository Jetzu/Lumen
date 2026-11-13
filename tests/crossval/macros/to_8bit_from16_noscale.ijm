// crossval: title=16-bit → 8-bit with setOption('ScaleConversions', false) | images=blobs16 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
setOption("ScaleConversions", false);
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
