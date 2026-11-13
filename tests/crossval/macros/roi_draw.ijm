// crossval: title=Draw an oval outline (foreground 255, line width 1) | images=blobs8 | compare=pixels | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
setForegroundColor(255, 255, 255);
makeOval(30, 20, 70, 50);
run("Draw", "slice");
saveAs("Tiff", dir + "/result.tif");
