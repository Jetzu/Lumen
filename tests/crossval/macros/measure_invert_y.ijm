// crossval: title=Measure with 'invert Y coordinates' | images=blobs8 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area centroid center bounding feret's invert redirect=None decimal=9");
makeOval(30, 20, 70, 50);
run("Measure");
saveAs("Results", dir + "/Results.csv");
