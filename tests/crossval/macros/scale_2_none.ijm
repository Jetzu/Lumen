// crossval: title=Scale… x=2 y=2 None create | images=blobs8 | compare=pixels | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Scale...", "x=2 y=2 interpolation=None create title=scaled");
saveAs("Tiff", dir + "/result.tif");
