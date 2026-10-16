"""Extract object-centric motion segments from replayed human demos."""
import os as _os
import glob
import os
import pickle

import numpy as np

REF = _os.environ.get("LIBERO_REF", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "libero_ref"))
HZ = 20


def quat_yaw(q):  # mujoco (w, x, y, z)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _trim_idle_start(eef, a, thr=0.01):
    v = np.linalg.norm(np.diff(eef, axis=0), axis=1) * HZ
    i = a
    while i < len(v) - 1 and v[i] < thr:
        i += 1
    return max(a, i - 2)


def extract(task_pkl):
    with open(task_pkl, "rb") as f:
        R = pickle.load(f)
    segs = []
    for di, d in enumerate(R["demos"]):
        A = d["actions"]
        T = len(A)
        g = np.sign(A[:, 6])
        ev = [(i, "close" if g[i] > 0 else "open") for i in np.where(np.diff(g) != 0)[0] + 1]
        base = dict(task=R["task"], suite=R["suite"], demo=di, table_z=R["table_z"])
        prev = 0
        for k, (i, kind) in enumerate(ev):
            nxt = ev[k + 1][0] if k + 1 < len(ev) else T
            if kind == "close":
                # which object gets lifted after the close?
                w = min(i + 30, T - 1)
                dz = d["obj_pos"][w, :, 2] - d["obj_pos"][i, :, 2]
                dist = np.linalg.norm(d["obj_pos"][i] - d["eef_pos"][i], axis=1)
                j = int(np.argmax(dz))
                lifted = dz[j] > 0.02
                if not lifted:
                    j = int(np.argmin(dist))
                start = _trim_idle_start(d["eef_pos"], prev) if prev == 0 else prev
                segs.append(dict(base, type="grasp_approach", t0=start, t1=i,
                                 obj=R["objects"][j], cat=R["categories"][j], lifted=bool(lifted),
                                 obj_pos=d["obj_pos"][i, j].copy(), obj_quat=d["obj_quat"][i, j].copy(),
                                 first=prev == 0, next_open=int(nxt)))
                # transport: close -> open (or demo end)
                carried = j if lifted else None
                if carried is not None:
                    ts = min(nxt + 25, T - 1)
                    segs.append(dict(base, type="transport", t0=i, t1=nxt, obj=R["objects"][j],
                                     cat=R["categories"][j],
                                     release_obj_pos=d["obj_pos"][min(nxt, T - 1), j].copy(),
                                     settle_obj_pos=d["obj_pos"][ts, j].copy(),
                                     settle_obj_quat=d["obj_quat"][ts, j].copy(),
                                     grasp_obj_pos=d["obj_pos"][i, j].copy(),
                                     grasp_eef_pos=d["eef_pos"][i].copy(),
                                     ends_open=nxt < T))
            prev = i
        # articulation segments
        for jk, jn in enumerate(R["joints"]):
            q = d["joint_q"][:, jk]
            if np.ptp(q) < 0.02:
                continue
            mv = np.where(np.abs(np.diff(q)) > 5e-4)[0]
            if len(mv) == 0:
                continue
            m0, m1 = mv[0], mv[-1] + 1
            # segment starts at the last gripper event before motion (or demo start)
            starts = [i for i, _ in ev if i <= m0 - 5]
            t0 = starts[-1] if starts else _trim_idle_start(d["eef_pos"], 0)
            t1 = min(T, m1 + 12)
            fix = jn.split("_")[0] + "_" + jn.split("_")[1] + "_" + jn.split("_")[2] \
                if jn.count("_") >= 3 else jn
            hp = d["handle_pos"][m0, jk].copy()
            hm = d["handle_mat"][m0, jk].copy()
            fi = [k for k, f in enumerate(R["fixtures"]) if jn.startswith(f + "_")]
            fi = fi[0] if fi else None
            kind = ("drawer" if "cabinet" in jn else "microwave" if "microwave" in jn
                    else "stove" if "stove" in jn else "other")
            level = next((l for l in ["top", "middle", "bottom"] if f"_{l}_" in jn + "_"), None)
            segs.append(dict(base, type="articulation", t0=int(t0), t1=int(t1), joint=jn,
                             kind=kind, level=level,
                             fixture_cat=R["fixture_cats"][fi] if fi is not None else None,
                             fix_pos=d["fix_pos"][m0, fi].copy() if fi is not None else None,
                             fix_quat=d["fix_quat"][m0, fi].copy() if fi is not None else None,
                             q_start=float(q[m0]), q_end=float(q[m1 - 1]), move_t0=int(m0),
                             move_t1=int(m1), anchor_pos=hp,
                             anchor_yaw=float(np.arctan2(hm[1, 0], hm[0, 0])),
                             grip_closed_frac=float(np.mean(g[m0:m1] > 0))))
    segs = [s for s in segs if s["t1"] - s["t0"] >= 12]
    return R, segs


def load_all():
    tasks, segs = {}, []
    for fn in sorted(glob.glob(os.path.join(REF, "replay", "*.pkl"))):
        R, s = extract(fn)
        tasks[R["task"]] = R
        segs += s
    return tasks, segs


if __name__ == "__main__":
    import collections
    tasks, segs = load_all()
    print(len(tasks), "tasks", len(segs), "segments")
    c = collections.Counter((s["type"], s.get("cat", s.get("joint", "").split("_level")[0][-20:]))
                            for s in segs)
    for k, v in sorted(c.items()):
        print(k, v)
