// crossval: title=Stack.getStatistics + Image ▸ Stacks ▸ Statistics | images=zstack8 | compare=results,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
Stack.getStatistics(voxelCount, mean, min, max, stdDev);
print("voxels=" + voxelCount);
print("mean=" + d2s(mean, 6));
print("min=" + min);
print("max=" + max);
print("std=" + d2s(stdDev, 6));
run("Statistics");
saveAs("Results", dir + "/Results.csv");
