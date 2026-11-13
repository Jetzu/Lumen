// crossval: title=Z Project… projection=[Median] | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Z Project...", "projection=[Median]");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
