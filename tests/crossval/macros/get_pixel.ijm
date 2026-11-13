// crossval: title=getPixel at 12 positions (8/16/32-bit) | images=blobs8,blobs16,blobs32 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics
dir = getArgument();
open(dir + "/input.tif");
s = "";
for (i = 0; i < 12; i++) s = s + d2s(getPixel(13 * i + 3, 9 * i + 2), 6) + ",";
print("pixels=" + s);
