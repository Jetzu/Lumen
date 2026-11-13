// crossval: title=Make Substack… slices=2-4 | images=zstack8 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Make Substack...", "slices=2-4");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
