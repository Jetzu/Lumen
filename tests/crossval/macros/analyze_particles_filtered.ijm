// crossval: title=Analyze Particles… size=50-Infinity circularity=0.30-1.00 exclude include | images=blobs8 | compare=results,log | group=Analyze ▸ Analyze Particles | match=X,Y
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
run("Analyze Particles...", "size=50-Infinity circularity=0.30-1.00 exclude include display clear");
print("count=" + nResults);
saveAs("Results", dir + "/Results.csv");
