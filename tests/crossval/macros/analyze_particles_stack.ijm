// crossval: title=Analyze Particles… stack (every slice) | images=zstack8 | compare=results,log | group=Analyze ▸ Analyze Particles
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate");
run("Set Measurements...", "area centroid stack redirect=None decimal=9");
run("Analyze Particles...", "size=10-Infinity display clear stack");
print("count=" + nResults);
saveAs("Results", dir + "/Results.csv");
