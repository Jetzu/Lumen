// crossval: title=Find Maxima… prominence=30 strict exclude output=List → nResults | images=blobs8 | compare=log | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 strict exclude output=List");
print("count=" + nResults);
