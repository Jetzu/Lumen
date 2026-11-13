// crossval: title=Scale… x=0.5 y=0.5 Bilinear average create | images=blobs8,blobs16 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Scale...", "x=0.5 y=0.5 interpolation=Bilinear average create title=scaled");
saveAs("Tiff", dir + "/result.tif");
