import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, glob, json, numpy as np, h5py, collections, itertools
for D in sys.argv[1:]:
    C = collections.Counter(); big = []
    for fn in sorted(glob.glob(f"{D}/episode_*.hdf5")):
        g = h5py.File(fn)["data/demo_0"]; ph = json.loads(g.attrs["phases"]); pid = g["phase_id"][:]
        segs = json.loads(g.attrs["segments"]); T = len(pid)
        inseg = np.zeros(T, bool)
        for s in segs: inseg[s["start"]:s["end"]] = True
        for k, v in itertools.groupby(range(T), key=lambda t: inseg[t]):
            v = list(v)
            if k: continue
            phs = collections.Counter(ph[pid[t]] for t in v)
            for p, n in phs.items(): C[p] += n
            if len(v) > 40: big.append((len(v), fn[-9:-5], v[0], dict(phs)))
    print(D, dict(C)); [print("   ", b) for b in sorted(big, reverse=True)[:10]]
