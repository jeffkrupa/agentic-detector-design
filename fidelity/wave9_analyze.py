#!/usr/bin/env python3
"""Wave 9 preview analysis: per-event core-window stats from event dumps."""
import sys, os, json, math
import numpy as np

LO, HI = 5, 18  # core window, inclusive

def load_dump(path):
    ev, dots, edeps = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            # eventID, sig, tracks, steps, 50 edeps, 50 dots
            ev.append(int(p[0]))
            edeps.append([float(x) for x in p[4:54]])
            dots.append([float(x) for x in p[54:104]])
    return np.array(ev), np.array(edeps), np.array(dots)

def stats(dots):
    W = dots[:, LO:HI+1].sum(axis=1)
    T = dots.sum(axis=1)
    n = len(W)
    return {
        "n": n,
        "mean_W": float(W.mean()),
        "se_W": float(W.std(ddof=1)/math.sqrt(n)),
        "median_W": float(np.median(W)),
        "tm5_W": float(np.mean(np.sort(W)[int(0.05*n):int(0.95*n)])),
        "var_W": float(W.var(ddof=1)),
        "mean_T": float(T.mean()),
        "se_T": float(T.std(ddof=1)/math.sqrt(n)),
        "nonfinite": int((~np.isfinite(dots)).sum()),
    }

def main(base, tags):
    out = {}
    for tag in tags:
        d = os.path.join(base, tag, "dump.txt")
        if not os.path.exists(d):
            print(f"{tag}: missing"); continue
        ev, edeps, dots = load_dump(d)
        out[tag] = stats(dots)
        out[tag]["eventIDs_hash"] = int(ev.sum())
    print(json.dumps(out, indent=1))
    return out

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
