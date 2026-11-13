// crossval: title=Enlarge… enlarge=7 (rectangle stays a rectangle) | images=blobs8 | compare=pixels,log | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(20, 15, 60, 45);
run("Enlarge...", "enlarge=7");
getSelectionBounds(x, y, w, h);
print("bounds=" + x + "," + y + "," + w + "," + h);
print("area=" + d2s(getValue("Area"), 6));
print("type=" + selectionType());
run("Create Mask");
saveAs("Tiff", dir + "/result.tif");
