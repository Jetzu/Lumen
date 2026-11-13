// crossval: title=Convolve… 5×5 Laplacian-like, no normalize | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Convolve...", "text1=[-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1]");
saveAs("Tiff", dir + "/result.tif");
