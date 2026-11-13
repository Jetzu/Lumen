// crossval: title=Analyze Particles… on a calibrated image (size in µm²) | images=cal8 | compare=results,log | group=Analyze ▸ Analyze Particles | match=X,Y
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Convert to Mask");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
getPixelSize(unit, pw, ph);
print("pw=" + d2s(pw, 6));
run("Analyze Particles...", "size=1-Infinity display clear");
print("count=" + nResults);
saveAs("Results", dir + "/Results.csv");
