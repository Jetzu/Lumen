// crossval: title=Gaussian Blur… sigma=2 stack | images=zstack8 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Gaussian Blur...", "sigma=2 stack");
saveAs("Tiff", dir + "/result.tif");
