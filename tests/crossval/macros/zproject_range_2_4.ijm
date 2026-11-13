// crossval: title=Z Project… start=2 stop=4 Max Intensity | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Z Project...", "start=2 stop=4 projection=[Max Intensity]");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
