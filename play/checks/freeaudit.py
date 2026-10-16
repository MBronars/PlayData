import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
"""Free-space audit: time the hand is neither carrying nor near any object or
fixture-free spot; long stretches without interaction; gaps between actions."""
import sys, glob, json, numpy as np, h5py
for D in sys.argv[1:]:
    tot = free = gap = 0; rot_free = []; worst = []
    for fn in sorted(glob.glob(f"{D}/episode_*.hdf5")):
        g = h5py.File(fn)["data/demo_0"]
        P = g["obs/ee_pos"][:]; A = g["actions"][:]; O = g["obs/object_states"][:]
        n = O.shape[1] // 7
        op = O[:, :n * 7].reshape(len(O), n, 7)[:, :, :3]
        d = np.linalg.norm(op - P[:, None], axis=2).min(1)
        gr = g["obs/gripper_states"][:]
        segs = json.loads(g.attrs["segments"])
        inseg = np.zeros(len(P), bool)
        for s in segs:
            inseg[s["start"]:s["end"]] = True
        phases = json.loads(g.attrs["phases"]); pid = g["phase_id"][:]
        ph = np.array([phases[i] for i in pid])
        isfree = (d > 0.12) & ~np.isin(ph, ["transport", "articulate", "push", "grasp", "release"])
        tot += len(P); free += isfree.sum(); gap += (~inseg).sum()
        rot_free.append(np.linalg.norm(A[isfree, 3:6], axis=1))
        # longest free stretch
        best = cur = 0; st = bs = 0
        for t, f in enumerate(isfree):
            cur = cur + 1 if f else 0
            if cur == 1: st = t
            if cur > best: best, bs = cur, st
        worst.append((best, fn[-9:-5], bs))
    rf = np.concatenate(rot_free)
    print(f"{D}: free-space {free / tot:.3f} of time, outside any action {gap / tot:.3f}; |rot| in free space p50/p90 {np.percentile(rf, 50):.3f}/{np.percentile(rf, 90):.3f}")
    print("   longest free stretches (steps, ep, start):", sorted(worst, reverse=True)[:6])
