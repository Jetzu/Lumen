// crossval: title=Find Maxima… output=[Single Points] | images=blobs8 | compare=pixels | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=[Single Points]");
saveAs("Tiff", dir + "/result.tif");
