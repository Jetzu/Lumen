// crossval: title=Canvas Size… 100×80 Top-Left (crop) | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Canvas Size...", "width=100 height=80 position=Top-Left");
saveAs("Tiff", dir + "/result.tif");
