"""Motion statistics for LIBERO-format HDF5 files (reference demos or play data)."""
import sys, glob, h5py, numpy as np, json

def stats(files):
    A, dA, V, L, grip_sw, rotmag, pause = [], [], [], [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            for k in f["data"]:
                d = f["data"][k]
                a = d["actions"][:]; p = d["obs"]["ee_pos"][:] if "ee_pos" in d["obs"] else d["obs"]["ee_states"][:, :3]
                A.append(a); dA.append(np.abs(np.diff(a[:, :6], axis=0)))
                v = np.linalg.norm(np.diff(p, axis=0), axis=1) * 20
                V.append(v); L.append(len(a))
                g = np.sign(a[:, 6]); grip_sw.append((np.diff(g) != 0).sum())
                rotmag.append(np.linalg.norm(a[:, 3:6], axis=1))
                pause.append((v < 0.01).mean())
    A = np.concatenate(A); dA = np.concatenate(dA); V = np.concatenate(V); rot = np.concatenate(rotmag)
    tr = np.linalg.norm(A[:, :3], axis=1)
    return {
        "episodes": len(L), "mean_len": float(np.mean(L)),
        "trans |a| p50/p90/max": np.round(np.percentile(tr, [50, 90, 100]), 3).tolist(),
        "per-dim |a| p50": np.round(np.percentile(np.abs(A[:, :6]), 50, axis=0), 3).tolist(),
        "frac any dim saturated(>.95)": float((np.abs(A[:, :6]) > 0.95).any(1).mean()),
        "rot |a| p50/p90": np.round(np.percentile(rot, [50, 90]), 3).tolist(),
        "jerk mean|da| (trans,rot)": [round(float(dA[:, :3].mean()), 4), round(float(dA[:, 3:].mean()), 4)],
        "eef speed m/s p50/p90/max": np.round(np.percentile(V, [50, 90, 100]), 3).tolist(),
        "frac paused (<1cm/s)": float(np.mean(pause)),
        "gripper switches/ep": float(np.mean(grip_sw)),
        "gripper action values": np.unique(np.round(A[:, 6], 2))[:6].tolist(),
    }

if __name__ == "__main__":
    print(json.dumps(stats(sys.argv[1:]), indent=1))
