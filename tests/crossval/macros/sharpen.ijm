// crossval: title=Sharpen | images=blobs8,blobs16,blobs32 | compare=pixels | group=Process
dir = getArgument();
open(dir + "/input.tif");
run("Sharpen");
saveAs("Tiff", dir + "/result.tif");
