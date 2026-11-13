// crossval: title=Scale… x=0.5 y=0.5 Bicubic create | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Scale...", "x=0.5 y=0.5 interpolation=Bicubic create title=scaled");
saveAs("Tiff", dir + "/result.tif");
