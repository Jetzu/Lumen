// crossval: title=Enhance Contrast… saturated=0.35 (display range only) | images=blobs8,blobs16 | compare=pixels,log | group=Process ▸ Enhance Contrast / Subtract Background
dir = getArgument();
open(dir + "/input.tif");
run("Enhance Contrast...", "saturated=0.35");
getMinAndMax(lo, hi);
print("minmax=" + lo + "," + hi);
saveAs("Tiff", dir + "/result.tif");
