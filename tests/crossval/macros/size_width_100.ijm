// crossval: title=Size… width=100 height=75 Bilinear average | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Size...", "width=100 height=75 constrain average interpolation=Bilinear");
saveAs("Tiff", dir + "/result.tif");
