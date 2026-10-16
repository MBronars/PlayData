import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, json, glob, numpy as np
from audit import load_episode, QLO, QHI
for D in sys.argv[1:]:
    for fn in sorted(glob.glob(f"{D}/episode_*.hdf5")):
        e = load_episode(fn)
        ph = [e["phases"][i] for i in e["ph"]]
        # longest run of retreat steps, and joint 5-7 travel during it
        best, k, start = (0, 0, 0), 0, 0
        for t, p in enumerate(ph + ["end"]):
            if p == "retreat":
                if k == 0: start = t
                k += 1
            else:
                if k > best[0]: best = (k, start, t)
                k = 0
        n, a, b = best
        travel = np.abs(np.diff(e["Q"][a:b, 4:], axis=0)).sum(0) if n > 1 else np.zeros(3)
        near = (np.minimum(e["Q"] - QLO, QHI - e["Q"]) < 0.1).any(1).sum()
        flag = " <-- LONG" if n > 50 else ""
        print(f"{D.split('/')[-1]} ep {fn[-8:-5]} longest retreat {n} steps ({a}-{b}) wrist travel {np.round(travel, 1)} near-limit steps {near}{flag}")
