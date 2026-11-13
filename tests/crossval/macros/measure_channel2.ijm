// crossval: title=Measure on channel 2 of a composite (Ch column) | images=twochan16 | compare=results | group=Analyze ▸ Measure / Set Measurements
dir = getArgument();
open(dir + "/input.tif");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack redirect=None decimal=9");
Stack.setChannel(2);
makeOval(20, 20, 50, 40);
run("Measure");
saveAs("Results", dir + "/Results.csv");
