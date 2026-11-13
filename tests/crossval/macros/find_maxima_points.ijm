// crossval: title=Find Maxima… prominence=30 output=[Point Selection] (n, Σx, Σy) | images=blobs8,blobs32 | compare=log | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=[Point Selection]");
getSelectionCoordinates(xs, ys);
sx = 0; sy = 0;
for (i = 0; i < xs.length; i++) { sx += xs[i]; sy += ys[i]; }
print("n=" + xs.length);
print("sumx=" + sx);
print("sumy=" + sy);
