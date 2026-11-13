// crossval: title=getHistogram inside an oval | images=blobs8 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics
dir = getArgument();
open(dir + "/input.tif");
makeOval(30, 20, 70, 50);
getHistogram(values, counts, 256);
s = "";
for (i = 0; i < counts.length; i++) s = s + counts[i] + ",";
print("counts=" + s);
print("v0=" + d2s(values[0], 6));
print("vlast=" + d2s(values[values.length - 1], 6));
