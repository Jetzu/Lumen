// crossval: title=Rotate… angle=15 interpolation=Bicubic | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Rotate... ", "angle=15 grid=1 interpolation=Bicubic");
saveAs("Tiff", dir + "/result.tif");
