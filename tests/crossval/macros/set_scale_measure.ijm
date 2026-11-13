// crossval: title=Set Scale… distance=4 known=1 unit=um, then Measure an oval in calibrated units | images=blobs8 | compare=results,log | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Scale...", "distance=4 known=1 unit=um");
getPixelSize(unit, pw, ph);
print("unit=" + unit);
print("pw=" + d2s(pw, 6));
print("ph=" + d2s(ph, 6));
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
makeOval(30, 20, 70, 50);
run("Measure");
saveAs("Results", dir + "/Results.csv");
