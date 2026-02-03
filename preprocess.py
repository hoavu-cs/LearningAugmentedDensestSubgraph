import os
import urllib.request
import shutil
import json
import argparse

def save_graph(g_json, fp, gname):
    edges = set()
    nv = 0
    for u, v in g_json:
        u, v = (u, v) if u < v else (v, u)
        nv = max(u, v, nv)
        edges.add((u, v))
    fp.write(f"{nv},{len(edges)},u,{gname},2,Int64,simplegraph\n")
    for (u, v) in edges:
        fp.write(f"{u},{v}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_edge", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    min_edge = args.min_edge
    count = args.count
    # Download and unpack archive
    if not os.path.exists("twitch_egos.zip"):
        urllib.request.urlretrieve("https://snap.stanford.edu/data/twitch_egos.zip")
        shutil.unpack_archive("twitch_egos.zip")
    # Process graphs
    with open("twitch_egos/twitch_edges.json") as fp:
        jo = json.load(fp)
        with open("datasets/graphs.lg", "w") as fp2:
            cnt_tmp = 0
            for k in jo.keys():
                # Filter for graphs with more edges than `min_edge`
                if len(jo[k]) // 2 < min_edge:
                    continue
                # Only process `count` graphs if specified
                if count is not None and cnt_tmp >= count:
                    break
                save_graph(jo[k], fp2, k)
                cnt_tmp += 1
