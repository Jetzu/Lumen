// crossval: title=Bandpass Filter… large=40 small=3 autoscale saturate | images=blobs8 | compare=pixels | group=Process ▸ FFT
dir = getArgument();
open(dir + "/input.tif");
run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
saveAs("Tiff", dir + "/result.tif");
