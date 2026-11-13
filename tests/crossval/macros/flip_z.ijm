// crossval: title=Flip Z | images=zstack8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Flip Z");
saveAs("Tiff", dir + "/result.tif");
