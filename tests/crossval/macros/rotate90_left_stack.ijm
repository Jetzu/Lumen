// crossval: title=Rotate 90 Degrees Left (stack) | images=zstack8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Rotate 90 Degrees Left");
saveAs("Tiff", dir + "/result.tif");
