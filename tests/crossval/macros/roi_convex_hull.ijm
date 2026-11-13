// crossval: title=Convex Hull of a concave polygon | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makePolygon(20, 20, 80, 25, 50, 50, 90, 90, 30, 80, 45, 55);
run("Convex Hull");
getSelectionCoordinates(xs, ys);
print("n=" + xs.length);
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
