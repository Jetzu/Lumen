// crossval: title=Image Calculator… operation=Divide create 32-bit | images=blobs8 | compare=pixels,log | group=Process ▸ Image Calculator
dir = getArgument();
open(dir + "/input.tif");
rename("a");
run("Duplicate...", "title=b");
run("Flip Horizontally");
run("Image Calculator...", "image1=a operation=Divide image2=b create 32-bit");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
