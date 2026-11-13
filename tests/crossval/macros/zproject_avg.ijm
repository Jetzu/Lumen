// crossval: title=Z Project… projection=[Average Intensity] | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Z Project...", "projection=[Average Intensity]");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
