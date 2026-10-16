"""Motion analysis shared by human LIBERO demos and play data.

Trajectories are dicts with actions (T,7), ee_pos (T,3), ee_ori (T,3 axis-angle),
gripper_states (T,2). Phases are split at gripper command flips."""
import glob
import json
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HZ = 20


def load_human(lowdim_dir, max_files=None, pattern="*"):
    trajs = []
    for fn in sorted(glob.glob(os.path.join(lowdim_dir, pattern + ".npz")))[:max_files]:
        z = np.load(fn, allow_pickle=True)
        task = os.path.basename(fn)[:-4]
        demos = sorted({k.split("/")[0] for k in z.files if k.startswith("demo_")},
                       key=lambda s: int(s.split("_")[1]))
        for d in demos:
            trajs.append(dict(task=task, src="human",
                              **{k: z[f"{d}/{k}"] for k in ["actions", "ee_pos", "ee_ori",
                                                            "gripper_states", "joint_states"]}))
    return trajs


def load_play(files, chunk=None):
    """Play episodes; optionally split into chunks of `chunk` steps so lengths
    are comparable to single-task demos."""
    trajs = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            d = f["data/demo_0"]
            t = dict(task="play", src="play", actions=d["actions"][:],
                     ee_pos=d["obs/ee_pos"][:], ee_ori=d["obs/ee_ori"][:],
                     gripper_states=d["obs/gripper_states"][:],
                     joint_states=d["obs/joint_states"][:],
                     primitive_id=d["primitive_id"][:],
                     segments=json.loads(d.attrs["segments"]))
        trajs.append(t)
    return trajs


def speed(t):
    return np.linalg.norm(np.diff(t["ee_pos"], axis=0), axis=1) * HZ


def ang_speed(t):
    R = Rotation.from_rotvec(t["ee_ori"])
    return np.linalg.norm((R[1:] * R[:-1].inv()).as_rotvec(), axis=1) * HZ


def gripper_events(t):
    g = np.sign(t["actions"][:, 6])
    idx = np.where(np.diff(g) != 0)[0] + 1
    return [(i, "close" if g[i] > 0 else "open") for i in idx]


def phases(t):
    ev = gripper_events(t)
    cuts = [0] + [i for i, _ in ev] + [len(t["actions"])]
    out = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1]
        if b - a < 5:
            continue
        g = np.sign(t["actions"][a, 6])
        kind = "transport" if g > 0 else ("approach" if k < len(cuts) - 2 else "retreat")
        out.append((a, b, kind))
    return out


def phase_metrics(t, a, b):
    p = t["ee_pos"][a:b]
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * HZ
    arc = v.sum() / HZ
    chord = np.linalg.norm(p[-1] - p[0])
    vel = np.diff(p, axis=0) * HZ
    vz, vxy = np.abs(vel[:, 2]), np.linalg.norm(vel[:, :2], axis=1)
    moving = v > 0.02
    diag = float(((vz > 0.03) & (vxy > 0.03)).sum() / max(moving.sum(), 1))
    # number of speed peaks (sub-movements)
    vs = np.convolve(v, np.ones(3) / 3, mode="same")
    peaks = int(((vs[1:-1] > vs[:-2]) & (vs[1:-1] >= vs[2:]) & (vs[1:-1] > 0.05)).sum())
    return dict(dur=(b - a) / HZ, arc=arc, chord=chord, straight=chord / max(arc, 1e-6),
                vpeak=float(v.max()) if len(v) else 0, vmean=float(v.mean()) if len(v) else 0,
                diag=diag, peaks=peaks, dz=float(p[-1, 2] - p[0, 2]),
                rot=float(np.linalg.norm((Rotation.from_rotvec(t["ee_ori"][b - 1]) *
                                          Rotation.from_rotvec(t["ee_ori"][a]).inv()).as_rotvec())))


def event_profiles(trajs, w=20):
    """Mean speed / z-velocity profile in a window around gripper close/open."""
    prof = {"close": [], "open": []}
    for t in trajs:
        v = speed(t)
        vz = np.diff(t["ee_pos"][:, 2]) * HZ
        for i, kind in gripper_events(t):
            if i - w >= 0 and i + w < len(v):
                prof[kind].append(np.stack([v[i - w:i + w], vz[i - w:i + w]]))
    return {k: (np.mean(v, 0) if v else None, len(v)) for k, v in prof.items()}


def dwell_stats(trajs, thr=0.02):
    """Steps the arm stays slow (< thr m/s) right after a gripper command."""
    out = {"close": [], "open": []}
    for t in trajs:
        v = speed(t)
        for i, kind in gripper_events(t):
            j = i
            while j < len(v) and v[j] < thr:
                j += 1
            # slow before
            k = i - 1
            while k >= 0 and v[k] < thr:
                k -= 1
            out[kind].append((i - 1 - k, j - i))
    return {k: np.array(v) for k, v in out.items()}


def global_metrics(trajs):
    A = np.concatenate([t["actions"] for t in trajs])
    dA = np.concatenate([np.abs(np.diff(t["actions"][:, :6], axis=0)) for t in trajs])
    V = np.concatenate([speed(t) for t in trajs])
    W = np.concatenate([ang_speed(t) for t in trajs])
    tr = np.linalg.norm(A[:, :3], axis=1)
    return {
        "trans|a| p25/50/90": np.percentile(tr, [25, 50, 90]).round(3).tolist(),
        "rot|a| p50/90": np.percentile(np.linalg.norm(A[:, 3:6], axis=1), [50, 90]).round(3).tolist(),
        "|a_i| p50": np.percentile(np.abs(A[:, :6]), 50, axis=0).round(3).tolist(),
        "jerk(trans,rot)": [round(float(dA[:, :3].mean()), 4), round(float(dA[:, 3:].mean()), 4)],
        "speed p25/50/90/max": np.percentile(V, [25, 50, 90, 100]).round(3).tolist(),
        "angspeed p50/90": np.percentile(W, [50, 90]).round(3).tolist(),
        "frac<1cm/s": round(float((V < 0.01).mean()), 3),
        "frac<3cm/s": round(float((V < 0.03).mean()), 3),
    }


def phase_table(trajs):
    rows = {}
    for t in trajs:
        for a, b, kind in phases(t):
            rows.setdefault(kind, []).append(phase_metrics(t, a, b))
    out = {}
    for kind, ms in rows.items():
        out[kind] = {k: np.round(np.percentile([m[k] for m in ms], [25, 50, 75]), 3).tolist()
                     for k in ms[0]}
        out[kind]["n"] = len(ms)
    return out
