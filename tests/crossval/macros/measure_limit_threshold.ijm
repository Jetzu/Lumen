// crossval: title=Measure with 'limit to threshold' (Mean, %Area over thresholded pixels) | images=blobs8 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
setAutoThreshold("Default dark");
run("Set Measurements...", "area mean standard min centroid area_fraction limit redirect=None decimal=9");
run("Select All");
run("Measure");
saveAs("Results", dir + "/Results.csv");
