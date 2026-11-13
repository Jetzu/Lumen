// crossval: title=Find Maxima… output=[Segmented Particles] | images=blobs8 | compare=pixels | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=[Segmented Particles]");
saveAs("Tiff", dir + "/result.tif");
