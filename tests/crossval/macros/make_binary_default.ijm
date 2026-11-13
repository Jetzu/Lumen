// crossval: title=setAutoThreshold('Default dark') + Make Binary | images=blobs8 | compare=pixels | group=Image ▸ Adjust ▸ Threshold (auto methods)
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
setAutoThreshold("Default dark");
run("Make Binary");
saveAs("Tiff", dir + "/result.tif");
