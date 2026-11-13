// crossval: title=Convert to Mask on a stack (calculate threshold for each image) | images=zstack8 | compare=pixels | group=Image ▸ Adjust ▸ Threshold (auto methods)
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate");
saveAs("Tiff", dir + "/result.tif");
