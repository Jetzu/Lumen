// crossval: title=Find Maxima… prominence=30 light output=List → nResults (minima) | images=blobs8 | compare=log | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 light output=List");
print("count=" + nResults);
