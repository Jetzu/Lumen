// crossval: title=Measure on slice 4 of a Z stack (Slice column) | images=zstack8 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
setSlice(4);
makeRectangle(10, 10, 50, 50);
run("Measure");
saveAs("Results", dir + "/Results.csv");
