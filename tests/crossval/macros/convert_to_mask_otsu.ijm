// crossval: title=setAutoThreshold('Otsu dark') + Convert to Mask | images=blobs8,blobs16,blobs32 | compare=pixels | group=Image ▸ Adjust ▸ Threshold (auto methods)
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Otsu dark");
run("Convert to Mask");
saveAs("Tiff", dir + "/result.tif");
