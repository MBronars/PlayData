import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, glob, numpy as np
from motion_analysis import load_human, load_play
from scipy.stats import wasserstein_distance
H = load_human(_os.path.join(PROJECT, "libero_ref", "lowdim"))
sets = {"human": H}
for name in sys.argv[1:]:
    sets[name.rstrip("/").split("/")[-1]] = load_play(sorted(glob.glob(f"{name}/episode_*.hdf5")))
def cat(T, f): return np.concatenate([f(t) for t in T])
A = {k: cat(T, lambda t: t["actions"]) for k, T in sets.items()}
D = {k: cat(T, lambda t: np.diff(t["actions"][:, :6], axis=0)) for k, T in sets.items()}
hq = {}
print("steps", {k: len(v) for k, v in A.items()})
names = ["x", "y", "z", "rx", "ry", "rz"]
for k, a in A.items():
    print(f"== {k}")
    for i in range(6):
        q = np.percentile(a[:, i], [0.5, 5, 50, 95, 99.5])
        sat = np.mean(np.abs(a[:, i]) >= 0.99)
        w = wasserstein_distance(a[:, i], A["human"][:, i])
        print(f"  {names[i]:2s} q0.5/5/50/95/99.5 {np.round(q, 2)}  std {a[:, i].std():.3f}  sat {sat:.4f}  W1 {w:.3f}")
    tr = np.linalg.norm(a[:, :3], axis=1); ro = np.linalg.norm(a[:, 3:6], axis=1)
    print(f"  |trans| p50/p90/p99 {np.round(np.percentile(tr, [50, 90, 99]), 3)}  |rot| p50/p90/p99 {np.round(np.percentile(ro, [50, 90, 99]), 3)}")
    d = D[k]; dn = np.linalg.norm(d[:, :3], axis=1); dr = np.linalg.norm(d[:, 3:], axis=1)
    print(f"  |dA trans| p50/p99/p99.9 {np.round(np.percentile(dn, [50, 99, 99.9]), 3)}  |dA rot| p50/p99/p99.9 {np.round(np.percentile(dr, [50, 99, 99.9]), 3)}")
    g = a[:, 6]; tog = np.sum(np.abs(np.diff(np.sign(g))) > 0)
    print(f"  gripper toggles per 100 steps {100 * tog / len(g):.2f}  closed frac {np.mean(g > 0):.2f}")
# fraction of play steps outside the human support (per-dim 0.1-99.9 pct)
lo = np.percentile(A["human"][:, :6], 0.1, axis=0); hi = np.percentile(A["human"][:, :6], 99.9, axis=0)
dlo = np.percentile(np.linalg.norm(D["human"][:, :3], axis=1), 99.9); dro = np.percentile(np.linalg.norm(D["human"][:, 3:], axis=1), 99.9)
for k in A:
    out = np.mean(np.any((A[k][:, :6] < lo) | (A[k][:, :6] > hi), axis=1))
    outd = np.mean((np.linalg.norm(D[k][:, :3], axis=1) > dlo) | (np.linalg.norm(D[k][:, 3:], axis=1) > dro))
    print(f"{k}: steps outside human 0.1-99.9% box {out:.4f}; steps with step-to-step change beyond human p99.9 {outd:.4f}")
