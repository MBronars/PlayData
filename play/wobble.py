"""End-of-approach smoothness: last 1.2 s before each gripper close.
  acc_flips  - sign changes of acceleration along the motion (zig-zag)
  speed_bumps- local maxima in the speed profile after it started decelerating
  rot_wobble - sign changes of angular velocity about the main rotation axis
  final_creep- seconds spent within 1 cm of the grasp point before closing
"""
import sys, glob, json, numpy as np, h5py
from scipy.spatial.transform import Rotation
HZ = 20

def ends(A, P, O, win=24):
    g = np.sign(A[:, 6]); out = []
    for t in np.where(np.diff(g) > 0)[0] + 1:
        a = t - win
        if a < 2: continue
        seg = P[a:t + 1]; V = np.diff(seg, axis=0) * HZ; sp = np.linalg.norm(V, axis=1)
        u = (seg[-1] - seg[0]); u = u / (np.linalg.norm(u) + 1e-9)
        acc = np.diff(V @ u)
        flips = int(np.sum(np.diff(np.sign(acc[np.abs(acc) > 0.002])) != 0))
        imax = int(np.argmax(sp)); bumps = int(np.sum((sp[imax + 1:-1] > sp[imax:-2] + 0.004) & (sp[imax + 1:-1] > sp[imax + 2:] + 0.004)))
        R = Rotation.from_rotvec(O[a:t + 1]); w = (R[1:] * R[:-1].inv()).as_rotvec() * HZ
        ax = w.sum(0); ax = ax / (np.linalg.norm(ax) + 1e-9)
        wa = w @ ax; rot_w = int(np.sum(np.diff(np.sign(wa[np.abs(wa) > 0.05])) != 0))
        creep = float(np.sum(np.linalg.norm(seg - seg[-1], axis=1) < 0.01) / HZ)
        out.append(dict(acc_flips=flips, speed_bumps=bumps, rot_wobble=rot_w, creep=creep))
    return out

def summary(rows, label):
    f = lambda k: np.round(np.percentile([r[k] for r in rows], [50, 90]), 2).tolist()
    print(f"{label:10s} n={len(rows):4d} acc_flips {f('acc_flips')} speed_bumps {f('speed_bumps')} rot_wobble {f('rot_wobble')} final_creep_s {f('creep')}")

if __name__ == "__main__":
    import glob as G
    H = []
    for fn in sorted(G.glob("/data/user_data/mbronars/AgenticExploration/libero_ref/lowdim/libero_90__*.npz"))[:40]:
        z = np.load(fn, allow_pickle=True)
        for d in sorted({k.split("/")[0] for k in z.files if k.startswith("demo_")})[:10]:
            H += ends(z[f"{d}/actions"], z[f"{d}/ee_pos"], z[f"{d}/ee_ori"])
    summary(H, "HUMAN")
    for p in sys.argv[1:]:
        rows = []
        for fn in sorted(G.glob(p + "/episode_*.hdf5")):
            with h5py.File(fn) as f:
                g = f["data/demo_0"]; rows += ends(g["actions"][:], g["obs/ee_pos"][:], g["obs/ee_ori"][:])
        summary(rows, p.split("/")[-1])
