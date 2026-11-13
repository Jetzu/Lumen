// crossval: title=Enhance Contrast… saturated=0.35 normalize | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Enhance Contrast...", "saturated=0.35 normalize");
saveAs("Tiff", dir + "/result.tif");
