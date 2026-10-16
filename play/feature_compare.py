"""Compare many low-level motion statistics between human demos and play sets."""
import sys, glob, numpy as np
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from motion_analysis import load_human, load_play, HZ
from scipy.spatial.transform import Rotation

def feats(T):
    F = {}
    A = np.concatenate([t["actions"][:, :6] for t in T])
    V = np.concatenate([np.diff(t["ee_pos"], axis=0) * HZ for t in T])
    Acc = np.concatenate([np.diff(t["ee_pos"], 2, axis=0) * HZ * HZ for t in T])
    sp = np.linalg.norm(V, axis=1)
    F["speed"] = sp; F["accel"] = np.linalg.norm(Acc, axis=1)
    F["|a_trans|"] = np.linalg.norm(A[:, :3], axis=1); F["|a_rot|"] = np.linalg.norm(A[:, 3:], axis=1)
    F["|da_trans|"] = np.concatenate([np.linalg.norm(np.diff(t["actions"][:, :3], axis=0), axis=1) for t in T])
    F["|da_rot|"] = np.concatenate([np.linalg.norm(np.diff(t["actions"][:, 3:6], axis=0), axis=1) for t in T])
    F["|d2a_trans|"] = np.concatenate([np.linalg.norm(np.diff(t["actions"][:, :3], 2, axis=0), axis=1) for t in T])
    F["vz/|v|"] = np.abs(V[:, 2]) / (sp + 1e-6)
    F["axis_dominance"] = np.max(np.abs(A[:, :3]), 1) / (np.linalg.norm(A[:, :3], axis=1) + 1e-6)
    # action -> velocity consistency (cosine between action and resulting velocity)
    cs = []
    for t in T:
        a = t["actions"][:-1, :3]; v = np.diff(t["ee_pos"], axis=0)
        m = (np.linalg.norm(a, axis=1) > 0.2)
        cs.append(np.sum(a[m] * v[m], 1) / (np.linalg.norm(a[m], axis=1) * np.linalg.norm(v[m], axis=1) + 1e-9))
    F["cos(a,v)"] = np.concatenate(cs)
    # ratio |v| / |a|
    r = []
    for t in T:
        a = np.linalg.norm(t["actions"][:-1, :3], axis=1); v = np.linalg.norm(np.diff(t["ee_pos"], axis=0), axis=1)
        m = a > 0.2; r.append(v[m] / a[m])
    F["|v|/|a|"] = np.concatenate(r)
    # autocorrelation of action at lag 3 (smoothness)
    ac = []
    for t in T:
        x = t["actions"][:, :3] - t["actions"][:, :3].mean(0)
        if len(x) > 10: ac.append(np.sum(x[3:] * x[:-3]) / (np.sum(x * x) + 1e-9))
    F["act_autocorr_lag3"] = np.array(ac)
    W = []
    for t in T:
        R = Rotation.from_rotvec(t["ee_ori"]); W.append(np.linalg.norm((R[1:] * R[:-1].inv()).as_rotvec(), axis=1) * HZ)
    F["ang_speed"] = np.concatenate(W)
    return F

if __name__ == "__main__":
    H = load_human("/data/user_data/mbronars/AgenticExploration/libero_ref/lowdim", pattern="libero_90__*")
    sets = {p.split("/")[-1]: load_play(sorted(glob.glob(p + "/episode_*.hdf5"))) for p in sys.argv[1:]}
    FH = feats(H); FS = {k: feats(v) for k, v in sets.items()}
    print("feature".ljust(20), "human p10/50/90".ljust(26), "  ".join(k.ljust(26) for k in FS))
    for f in FH:
        h = np.percentile(FH[f], [10, 50, 90])
        row = [np.percentile(FS[k][f], [10, 50, 90]) for k in FS]
        print(f.ljust(20), str(np.round(h, 3)).ljust(26), "  ".join(str(np.round(r, 3)).ljust(26) for r in row))
