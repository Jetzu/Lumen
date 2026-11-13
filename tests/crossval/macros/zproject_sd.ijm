// crossval: title=Z Project… projection=[Standard Deviation] | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Z Project...", "projection=[Standard Deviation]");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
