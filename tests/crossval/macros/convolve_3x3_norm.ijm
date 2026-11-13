// crossval: title=Convolve… 3×3 binomial, normalize | images=blobs8,blobs32 | compare=pixels | group=Process ▸ Filters
dir = getArgument();
open(dir + "/input.tif");
run("Convolve...", "text1=[1 2 1\n2 4 2\n1 2 1] normalize");
saveAs("Tiff", dir + "/result.tif");
