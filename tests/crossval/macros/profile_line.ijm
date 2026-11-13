// crossval: title=getProfile() along a line | images=blobs8,blobs32 | compare=log | group=Analyze ▸ Histogram / Plot Profile / statistics | fallback=profile
dir = getArgument();
open(dir + "/input.tif");
makeLine(10, 20, 140, 100);
profile = getProfile();
s = "";
for (i = 0; i < profile.length; i++) s = s + d2s(profile[i], 4) + ",";
print("n=" + profile.length);
print("profile=" + s);
