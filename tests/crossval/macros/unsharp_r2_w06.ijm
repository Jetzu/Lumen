// crossval: title=Unsharp Mask… radius=2 mask=0.60 | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Unsharp Mask...", "radius=2 mask=0.60");
saveAs("Tiff", dir + "/result.tif");
