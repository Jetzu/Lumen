// crossval: title=Stack to RGB (composite 2 channels) | images=twochan16 | compare=pixels | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Stack to RGB");
saveAs("Tiff", dir + "/result.tif");
