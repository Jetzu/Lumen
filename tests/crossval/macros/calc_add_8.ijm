// crossval: title=Image Calculator… operation=Add create | images=blobs8 | compare=pixels,log | group=Process ▸ Image Calculator
dir = getArgument();
open(dir + "/input.tif");
rename("a");
run("Duplicate...", "title=b");
run("Flip Horizontally");
run("Image Calculator...", "image1=a operation=Add image2=b create");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
