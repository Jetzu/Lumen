// crossval: title=Enlarge… enlarge=-5 (shrink a polygon) | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makePolygon(10, 10, 120, 30, 100, 100, 40, 90, 20, 60);
run("Enlarge...", "enlarge=-5");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
