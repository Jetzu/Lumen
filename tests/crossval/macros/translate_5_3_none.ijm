// crossval: title=Translate… x=5 y=3 interpolation=None | images=blobs8,blobs16,blobs32 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Translate...", "x=5 y=3 interpolation=None");
saveAs("Tiff", dir + "/result.tif");
