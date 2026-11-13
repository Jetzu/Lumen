// crossval: title=Make Band… band=5 (oval) | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeOval(40, 30, 50, 40);
run("Make Band...", "band=5");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
