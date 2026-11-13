// crossval: title=Measure, polygon, every measurement | images=blobs8,blobs32 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
makePolygon(10, 10, 120, 30, 100, 100, 40, 90, 20, 60);
run("Measure");
saveAs("Results", dir + "/Results.csv");
