// crossval: title=Rotate… angle=30 interpolation=None | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Rotate... ", "angle=30 grid=1 interpolation=None");
saveAs("Tiff", dir + "/result.tif");
