// crossval: title=Split Channels → C2-input.tif | images=twochan16 | compare=pixels,log | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Split Channels");
selectWindow("C2-input.tif");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
