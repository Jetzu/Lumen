// crossval: title=Canvas Size… 200×150 Center zero | images=blobs8,blobs16 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Canvas Size...", "width=200 height=150 position=Center zero");
saveAs("Tiff", dir + "/result.tif");
