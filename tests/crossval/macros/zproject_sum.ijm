// crossval: title=Z Project… projection=[Sum Slices] | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Z Project...", "projection=[Sum Slices]");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
