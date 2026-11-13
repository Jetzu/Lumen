// crossval: title=Enhance Contrast… equalize | images=blobs8,blobs16 | compare=pixels | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Enhance Contrast...", "saturated=0.35 equalize");
saveAs("Tiff", dir + "/result.tif");
