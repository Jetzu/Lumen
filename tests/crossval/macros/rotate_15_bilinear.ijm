// crossval: title=Rotate… angle=15 interpolation=Bilinear | images=blobs8,blobs16,blobs32 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Rotate... ", "angle=15 grid=1 interpolation=Bilinear");
saveAs("Tiff", dir + "/result.tif");
