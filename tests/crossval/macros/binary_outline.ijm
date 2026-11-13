// crossval: title=Outline (after Default-dark mask) | images=blobs8 | compare=pixels | group=Process ▸ Binary
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Options...", "iterations=1 count=1 black do=Nothing");
run("Outline");
saveAs("Tiff", dir + "/result.tif");
