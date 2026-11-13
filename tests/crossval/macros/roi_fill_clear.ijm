// crossval: title=Fill (foreground 200) and Clear Outside (background 10) of an oval | images=blobs8 | compare=pixels | group=Edit ▸ Selection (ROI geometry)
dir = getArgument();
open(dir + "/input.tif");
setForegroundColor(200, 200, 200);
setBackgroundColor(10, 10, 10);
makeOval(30, 20, 70, 50);
run("Fill", "slice");
run("Clear Outside");
saveAs("Tiff", dir + "/result.tif");
