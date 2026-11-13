// crossval: title=setThreshold(80, 255) + Convert to Mask | images=blobs8 | compare=pixels | group=Image ▸ Adjust ▸ Threshold (auto methods)
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setThreshold(80, 255);
run("Convert to Mask");
saveAs("Tiff", dir + "/result.tif");
