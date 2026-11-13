// crossval: title=16-bit stack → 8-bit (stack display range) | images=twochan16 | compare=pixels | group=Image ▸ Type
dir = getArgument();
open(dir + "/input.tif");
resetMinAndMax();
run("8-bit");
saveAs("Tiff", dir + "/result.tif");
