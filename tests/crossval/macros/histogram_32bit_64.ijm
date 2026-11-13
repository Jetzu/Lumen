// crossval: title=getHistogram 64 bins on 32-bit | images=blobs32 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics
dir = getArgument();
open(dir + "/input.tif");
getHistogram(values, counts, 64);
s = "";
for (i = 0; i < counts.length; i++) s = s + counts[i] + ",";
print("counts=" + s);
print("v0=" + d2s(values[0], 6));
print("vlast=" + d2s(values[values.length - 1], 6));
