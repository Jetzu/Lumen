// crossval: title=Edit ▸ Invert (stack) | images=zstack8 | compare=pixels | group=Process ▸ Math
dir = getArgument();
open(dir + "/input.tif");
run("Invert", "stack");
saveAs("Tiff", dir + "/result.tif");
