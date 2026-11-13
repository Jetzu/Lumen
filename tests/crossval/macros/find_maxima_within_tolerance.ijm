// crossval: title=Find Maxima… output=[Maxima Within Tolerance] | images=blobs8 | compare=pixels | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=[Maxima Within Tolerance]");
saveAs("Tiff", dir + "/result.tif");
