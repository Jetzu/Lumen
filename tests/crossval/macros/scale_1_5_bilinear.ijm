// crossval: title=Scale… x=1.5 y=1.5 Bilinear create | images=blobs8,blobs32 | compare=pixels,log | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
run("Scale...", "x=1.5 y=1.5 interpolation=Bilinear average create title=scaled");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
