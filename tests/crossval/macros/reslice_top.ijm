// crossval: title=Reslice [/]… (top, output spacing 1) | images=zstack8 | compare=pixels | group=Image ▸ Stacks
dir = getArgument();
open(dir + "/input.tif");
run("Reslice [/]...", "output=1.000 start=Top avoid");
saveAs("Tiff", dir + "/result.tif");
