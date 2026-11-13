// crossval: title=Crop to an oval (outside cleared) | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
makeOval(30, 20, 70, 50);
run("Crop");
saveAs("Tiff", dir + "/result.tif");
