// crossval: title=Calibration read from the TIFF (0.25 µm/px) + Measure | images=cal8 | compare=results,log | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
getPixelSize(unit, pw, ph);
print("unit=" + unit);
print("pw=" + d2s(pw, 6));
print("ph=" + d2s(ph, 6));
getVoxelSize(vw, vh, vd, vunit);
print("vd=" + d2s(vd, 6));
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
makeRectangle(20, 15, 60, 45);
run("Measure");
saveAs("Results", dir + "/Results.csv");
