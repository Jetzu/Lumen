// crossval: title=Selection Scale… x=1.5 y=1.5 centered (oval) | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeOval(40, 30, 50, 40);
run("Scale... ", "x=1.5 y=1.5 centered");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
