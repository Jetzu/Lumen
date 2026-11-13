// crossval: title=Measure, straight line, every measurement | images=blobs8,blobs16 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
makeLine(10, 20, 140, 100);
run("Measure");
saveAs("Results", dir + "/Results.csv");
