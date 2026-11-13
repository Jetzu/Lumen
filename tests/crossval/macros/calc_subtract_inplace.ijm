// crossval: title=Image Calculator… Subtract in place (image1 modified) | images=blobs8,blobs16 | compare=pixels,log | group=Process ▸ Image Calculator
dir = getArgument();
open(dir + "/input.tif");
rename("a");
run("Duplicate...", "title=b");
run("Flip Horizontally");
run("Image Calculator...", "image1=a operation=Subtract image2=b");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
