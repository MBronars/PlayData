"""Episode audit: measurable versions of the things that look wrong in videos.

  overshoot / wobble  - in the ~1.5 s before each gripper close/open, how far the
                        hand goes past its final point along the approach, and how
                        many times its velocity reverses (wobble)
  joint limits        - steps where any joint is within 0.10 rad of its limit
  tilt                - gripper tilt from vertical outside drawer/inspect phases
  close coordination  - hand speed when the close command fires, distance moved
                        while closing, dwell (slow time) before the command
Run on play episodes (with phase labels) and on human demos for reference.

  python audit.py play_dir [--episodes 0,6,12] [--human]
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HZ = 20
QLO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
QHI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def load_episode(fn):
    with h5py.File(fn) as f:
        g = f["data/demo_0"]
        ep = dict(A=g["actions"][:], P=g["obs/ee_pos"][:], O=g["obs/ee_ori"][:], Q=g["obs/joint_states"][:],
                  G=g["obs/gripper_states"][:])
        if "phase_id" in g:
            ep["ph"] = g["phase_id"][:]
            ep["phases"] = json.loads(g.attrs["phases"])
        else:
            ep["ph"] = np.zeros(len(ep["A"]), int)
            ep["phases"] = ["human"]
    return ep


def gripper_events(A):
    g = np.sign(A[:, 6])
    return [(int(i), "close" if g[i] > 0 else "open") for i in np.where(np.diff(g) != 0)[0] + 1]


def arrival_metrics(P, t, win=30):
    """Overshoot and wobble in the window before event t (final point = P[t])."""
    a = max(1, t - win)
    seg = P[a:t + 1]
    if len(seg) < 8:
        return None
    target = P[t]
    d0 = target - seg[0]
    if np.linalg.norm(d0) < 0.01:
        u = None
    else:
        u = d0 / np.linalg.norm(d0)
    # overshoot: how far past the final point along the approach direction
    over = 0.0 if u is None else float(np.max(np.maximum((seg - target) @ u, 0)))
    V = np.diff(seg, axis=0) * HZ
    sp = np.linalg.norm(V, axis=1)
    # wobble: reversals of the velocity direction while still moving (> 1 cm/s)
    rev = 0
    for k in range(1, len(V)):
        if sp[k] > 0.01 and sp[k - 1] > 0.01 and V[k] @ V[k - 1] < 0:
            rev += 1
    # lateral oscillation: residual from a straight approach
    if u is not None:
        lat = (seg - target) - np.outer((seg - target) @ u, u)
        lat_osc = float(np.std(np.diff(np.linalg.norm(lat, axis=1))) * HZ)
    else:
        lat_osc = 0.0
    return dict(overshoot=over, reversals=rev, lat_osc=lat_osc)


def close_metrics(P, G, A, t):
    V = np.linalg.norm(np.diff(P, axis=0), axis=1) * HZ
    sp_cmd = float(V[min(t, len(V) - 1)])
    # fingers settle: width stops changing
    w = G[:, 0] - G[:, 1]
    k = t
    while k + 1 < len(w) and abs(w[k + 1] - w[k]) > 2e-4 and k - t < 30:
        k += 1
    moved = float(np.linalg.norm(P[k] - P[t]))
    # dwell: slow steps right before the command
    j = t - 1
    while j > 0 and V[j] < 0.02:
        j -= 1
    return dict(speed_at_cmd=sp_cmd, moved_while_closing=moved, dwell=(t - 1 - j) / HZ, close_steps=k - t)


def tilt_deg(O):
    R = Rotation.from_rotvec(O).as_matrix()
    return np.degrees(np.arccos(np.clip(-R[:, 2, 2], -1, 1)))


def audit_episode(ep):
    out = dict(arrivals=[], closes=[], opens=[])
    for t, kind in gripper_events(ep["A"]):
        m = arrival_metrics(ep["P"], t)
        if m:
            m.update(t=t, kind=kind, phase=ep["phases"][ep["ph"][max(t - 1, 0)]])
            out["arrivals"].append(m)
        cm = close_metrics(ep["P"], ep["G"], ep["A"], t)
        cm.update(t=t)
        out["closes" if kind == "close" else "opens"].append(cm)
    Q = ep["Q"]
    margin = np.minimum(Q - QLO, QHI - Q)
    near = margin < 0.10
    out["joint_near_steps"] = int(near.any(1).sum())
    out["joint_near_by_joint"] = near.sum(0).tolist()
    out["joint_min_margin"] = float(margin.min())
    tl = tilt_deg(ep["O"])
    excl = {i for i, n in enumerate(ep["phases"]) if n in ("articulate", "inspect", "push")}
    m = np.array([p not in excl for p in ep["ph"]])
    out["tilt"] = tl[m]
    return out


def summarize(audits, label):
    arr = [a for x in audits for a in x["arrivals"]]
    cl = [a for x in audits for a in x["closes"]]
    op = [a for x in audits for a in x["opens"]]
    tl = np.concatenate([x["tilt"] for x in audits]) if audits else np.zeros(1)
    pct = lambda v, q=(50, 90): np.round(np.percentile(v, q), 3).tolist() if len(v) else None
    s = dict(
        n_arrivals=len(arr),
        overshoot_cm=pct([100 * a["overshoot"] for a in arr]),
        reversals=pct([a["reversals"] for a in arr]),
        frac_wobbly=round(float(np.mean([a["reversals"] >= 3 for a in arr])), 2) if arr else None,
        close_speed=pct([c["speed_at_cmd"] for c in cl]),
        moved_while_closing_cm=pct([100 * c["moved_while_closing"] for c in cl]),
        dwell_before_close_s=pct([c["dwell"] for c in cl]),
        open_speed=pct([c["speed_at_cmd"] for c in op]),
        dwell_before_open_s=pct([c["dwell"] for c in op]),
        tilt_deg=pct(tl, (50, 90, 99)),
        frac_tilt_gt_25=round(float(np.mean(tl > 25)), 3),
        joint_near_steps=int(sum(x["joint_near_steps"] for x in audits)),
        joint_near_by_joint=np.sum([x["joint_near_by_joint"] for x in audits], 0).tolist(),
        joint_min_margin=round(min(x["joint_min_margin"] for x in audits), 3),
    )
    print(f"== {label}")
    for k, v in s.items():
        print(f"   {k:24s} {v}")
    return s


def human_reference(n_files=40):
    import glob as _g
    files = sorted(_g.glob("/data/user_data/mbronars/AgenticExploration/libero_ref/lowdim/libero_90__*.npz"))[:n_files]
    audits = []
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        demos = sorted({k.split("/")[0] for k in z.files if k.startswith("demo_")})[:10]
        for d in demos:
            ep = dict(A=z[f"{d}/actions"], P=z[f"{d}/ee_pos"], O=z[f"{d}/ee_ori"], Q=z[f"{d}/joint_states"],
                      G=z[f"{d}/gripper_states"], ph=np.zeros(len(z[f"{d}/actions"]), int), phases=["human"])
            audits.append(audit_episode(ep))
    return audits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("play_dir")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--flag", action="store_true", help="list the worst events per episode")
    args = ap.parse_args()
    if args.human:
        summarize(human_reference(), "HUMAN demos")
    files = sorted(glob.glob(os.path.join(args.play_dir, "episode_*.hdf5")))
    if args.episodes:
        keep = {int(x) for x in args.episodes.split(",")}
        files = [f for f in files if int(f[-11:-5]) in keep]
    audits = []
    for fn in files:
        a = audit_episode(load_episode(fn))
        a["ep"] = int(fn[-11:-5])
        audits.append(a)
        if args.flag:
            bad = sorted(a["arrivals"], key=lambda x: -(x["overshoot"] + 0.01 * x["reversals"]))[:3]
            print(f"ep {a['ep']}: worst arrivals", [(b["t"], b["kind"], b["phase"], round(100 * b["overshoot"], 1), b["reversals"]) for b in bad],
                  "| joint-near steps", a["joint_near_steps"], a["joint_near_by_joint"], "| tilt p99 %.0f" % np.percentile(a["tilt"], 99))
    summarize(audits, f"PLAY {args.play_dir}")
