// crossval: title=Fill Holes on a binary stack | images=zstack8 | compare=pixels | group=Process ▸ Binary
dir = getArgument();
open(dir + "/input.tif");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate");
run("Fill Holes", "stack");
saveAs("Tiff", dir + "/result.tif");
