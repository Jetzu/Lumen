// crossval: title=Analyze Particles… show=[Count Masks] | images=blobs8 | compare=pixels | group=Analyze ▸ Analyze Particles
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Analyze Particles...", "size=0-Infinity show=[Count Masks] clear");
saveAs("Tiff", dir + "/result.tif");
