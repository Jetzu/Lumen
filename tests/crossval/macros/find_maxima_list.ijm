// crossval: title=Find Maxima… output=List (rows sorted by value) | images=blobs8 | compare=results | group=Process ▸ Find Maxima | match=X,Y
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=List");
saveAs("Results", dir + "/Results.csv");
