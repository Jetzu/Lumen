// crossval: title=Erode, iterations=3 count=1 | images=blobs8 | compare=pixels | group=Process ▸ Binary
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Options...", "iterations=3 count=1 black do=Nothing");
run("Erode");
saveAs("Tiff", dir + "/result.tif");
