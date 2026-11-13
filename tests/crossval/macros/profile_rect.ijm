// crossval: title=getProfile() of a rectangle (column averages) | images=blobs8 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics | fallback=profile
dir = getArgument();
open(dir + "/input.tif");
makeRectangle(20, 15, 60, 45);
profile = getProfile();
s = "";
for (i = 0; i < profile.length; i++) s = s + d2s(profile[i], 4) + ",";
print("n=" + profile.length);
print("profile=" + s);
