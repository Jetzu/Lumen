// crossval: title=FFT → Inverse FFT round trip | images=blobs8 | compare=pixels,log | group=Process ▸ FFT
dir = getArgument();
open(dir + "/input.tif");
run("FFT");
run("Inverse FFT");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
