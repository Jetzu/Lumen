// crossval: title=makeLine width 5 + Line to Area | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeLine(10, 20, 140, 100, 5);
run("Line to Area");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
