// crossval: title=Translate… x=2.5 y=-1.5 interpolation=Bilinear | images=blobs8,blobs32 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Translate...", "x=2.5 y=-1.5 interpolation=Bilinear");
saveAs("Tiff", dir + "/result.tif");
