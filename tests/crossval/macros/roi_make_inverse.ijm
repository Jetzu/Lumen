// crossval: title=Make Inverse of an oval | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeOval(30, 20, 70, 50);
run("Make Inverse");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
