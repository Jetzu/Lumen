// crossval: title=Measure a point selection | images=blobs8 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area mean min centroid redirect=None decimal=9");
makePoint(37, 52);
run("Measure");
saveAs("Results", dir + "/Results.csv");
