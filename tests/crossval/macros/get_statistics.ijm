// crossval: title=getStatistics / getRawStatistics / getValue inside a rectangle | images=blobs8,blobs16,blobs32 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(20, 15, 60, 45);
getStatistics(area, mean, min, max, std);
print("area=" + area);
print("mean=" + d2s(mean, 6));
print("min=" + min);
print("max=" + max);
print("std=" + d2s(std, 6));
getRawStatistics(n, rmean, rmin, rmax, rstd);
print("n=" + n);
print("rmean=" + d2s(rmean, 6));
print("median=" + d2s(getValue("Median"), 6));
print("skew=" + d2s(getValue("Skew"), 6));
print("kurt=" + d2s(getValue("Kurt"), 6));
