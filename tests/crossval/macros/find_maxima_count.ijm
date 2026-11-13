// crossval: title=Find Maxima… prominence=30 output=List → nResults (output=Count blocks ImageJ's batch mode) | images=blobs8,blobs16,blobs32 | compare=log | group=Process ▸ Find Maxima
dir = getArgument();
open(dir + "/input.tif");
run("Find Maxima...", "prominence=30 output=List");
print("count=" + nResults);
