// crossval: title=Selection Rotate… angle=30 (rectangle) | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(40, 30, 60, 40);
run("Rotate...", "angle=30");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
