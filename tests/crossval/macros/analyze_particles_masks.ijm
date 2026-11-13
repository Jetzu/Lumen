// crossval: title=Analyze Particles… show=Masks | images=blobs8 | compare=pixels,log | group=Analyze ▸ Analyze Particles
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Analyze Particles...", "size=0-Infinity show=Masks clear");
print("count=" + nResults);
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
