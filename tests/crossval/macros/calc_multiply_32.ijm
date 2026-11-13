// crossval: title=Image Calculator… operation=Multiply create 32-bit | images=blobs8 | compare=pixels,log | group=Process ▸ Image Calculator
dir = getArgument();
open(dir + "/input.tif");
rename("a");
run("Duplicate...", "title=b");
run("Flip Horizontally");
run("Image Calculator...", "image1=a operation=Multiply image2=b create 32-bit");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
