// crossval: title=Duplicate… of a rectangle selection | images=blobs8 | compare=pixels,log | group=Image ▸ Transform / Scale / Crop
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(20, 15, 60, 45);
run("Duplicate...", "title=dup");
print("title=" + getTitle());
saveAs("Tiff", dir + "/result.tif");
