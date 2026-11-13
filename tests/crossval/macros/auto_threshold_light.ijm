// crossval: title=setAutoThreshold(method) (objects dark) → getThreshold, 16 methods | images=blobs8,blobs16 | compare=log | group=Image ▸ Adjust ▸ Threshold (auto methods)
dir = getArgument();
open(dir + "/input.tif");
methods = newArray("Default", "Huang", "Intermodes", "IsoData", "Li", "MaxEntropy", "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy", "Shanbhag", "Triangle", "Yen");
for (i = 0; i < methods.length; i++) {
    setAutoThreshold(methods[i] + "");
    getThreshold(lower, upper);
    print(methods[i] + "=" + lower + "," + upper);
}
