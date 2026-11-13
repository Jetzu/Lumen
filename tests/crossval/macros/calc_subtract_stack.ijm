// crossval: title=Image Calculator… Subtract stack create (stack − blurred stack) | images=zstack8 | compare=pixels | group=Process ▸ Image Calculator
dir = getArgument();
open(dir + "/input.tif");
rename("a");
run("Duplicate...", "title=b");
run("Flip Horizontally", "stack");
run("Image Calculator...", "image1=a operation=Subtract image2=b create stack");
saveAs("Tiff", dir + "/result.tif");
