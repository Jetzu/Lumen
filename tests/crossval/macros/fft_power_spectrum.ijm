// crossval: title=FFT (8-bit power spectrum) | images=blobs8,blobs32 | compare=pixels,log | group=Process ▸ FFT
dir = getArgument();
open(dir + "/input.tif");
run("FFT");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
