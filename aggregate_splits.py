import json
import os

if __name__ == "__main__":
    res = dict()
    fnames = filter(lambda fn: fn.startswith("split") and fn.endswith(".json"), os.listdir("outputs/"))
    for fname in fnames:
        with open("outputs/" + fname) as fp:
            js = json.load(fp)
            res = {**res, **js}
    with open("outputs/result.json", "w") as fp:
        json.dump(res, fp)
            
