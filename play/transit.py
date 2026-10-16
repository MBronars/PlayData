"""Human free-space transit search (a small motion graph).

All gripper-open frames from every human demo are indexed in the robot-base
frame (plus y-mirrored copies). To move from the robot's current state to the
entry state of a chosen clip, we find a human sub-sequence [a, b] whose frame a
matches the current pose/velocity and whose frame b matches the target, so the
robot travels with genuine human motion instead of a large synthetic warp."""
import json
import os
import re

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BASES = json.load(open(os.path.join(HERE, "robot_bases.json")))
FLIP = np.diag([-1.0, -1.0, 1.0])
MIR = np.diag([1.0, -1.0, 1.0])
W_V = 0.35   # m per (m/s)
W_R = 0.12   # m per rad


def task_base(R):
    try:
        m = re.search(r"\(problem\s+(\S+)\)", open(R["bddl"]).read())
        return np.array(BASES[m.group(1).lower()]["base"])
    except Exception:
        return np.array([-0.66, 0.0, 0.912])


def _sym_angle(Ra, Rb):
    """Angle between gripper orientations modulo the pi flip about z (batched Ra)."""
    t1 = np.einsum("...ij,ij->...", Ra, Rb)
    t2 = np.einsum("...ij,ij->...", Ra @ FLIP, Rb)
    return np.arccos(np.clip((np.maximum(t1, t2) - 1) / 2, -1, 1))


class TransitDB:
    def __init__(self, tasks, min_run=15):
        P, V, Rm, ref = [], [], [], []
        self.clips = []  # (task, demo, mirror)
        for tname, R in tasks.items():
            base = task_base(R)
            for di, d in enumerate(R["demos"]):
                open_ = d["actions"][:, 6] < 0
                p = d["eef_pos"].astype(np.float64) - base
                v = np.gradient(d["eef_pos"].astype(np.float64), axis=0) * 20.0
                Rd = d["eef_mat"].astype(np.float64)
                # runs of consecutive open frames
                T = len(open_)
                run_end = np.full(T, -1)
                i = 0
                while i < T:
                    if open_[i]:
                        j = i
                        while j + 1 < T and open_[j + 1]:
                            j += 1
                        if j - i + 1 >= min_run:
                            run_end[i:j + 1] = j
                        i = j + 1
                    else:
                        i += 1
                idx = np.where(run_end >= 0)[0]
                if len(idx) == 0:
                    continue
                for mirror in (False,):
                    ci = len(self.clips)
                    self.clips.append((tname, di, mirror, base))
                    if mirror:
                        pp, vv = p @ MIR, v @ MIR
                        RR = MIR @ Rd @ MIR @ FLIP
                    else:
                        pp, vv, RR = p, v, Rd
                    P.append(pp[idx]); V.append(vv[idx]); Rm.append(RR[idx])
                    ref.append(np.stack([np.full(len(idx), ci), idx, run_end[idx]], 1))
        self.P = np.concatenate(P)
        self.V = np.concatenate(V)
        self.R = np.concatenate(Rm)
        self.ref = np.concatenate(ref)
        self.tree = cKDTree(np.concatenate([self.P, W_V * self.V], 1))
        # map (clip, frame) -> row, for forward scans
        self._row = {}
        starts = np.r_[0, np.where(np.diff(self.ref[:, 0]) != 0)[0] + 1]
        for s in starts:
            self._row[int(self.ref[s, 0])] = (s, int(self.ref[s, 1]))
        self.tasks = tasks

    def search(self, p, v, Rc, pt, vt, Rt, k=30000, min_len=12, max_len=200):
        """Bidirectional: frames near the start state and frames near the
        target state that lie on the same human clip (a < b <= a + max_len).
        p, pt: positions in robot-base frame. Returns (cost, clip, a, b, ca, cb)."""
        da, ia = self.tree.query(np.r_[p, W_V * v], k=k)
        db, ib = self.tree.query(np.r_[pt, W_V * vt], k=k)
        ca = da + W_R * _sym_angle(self.R[ia], Rc)
        cb = db + W_R * _sym_angle(self.R[ib], Rt)
        def topk_per_clip(rows, cost, k=6):
            cl = self.ref[rows, 0]
            order = np.lexsort((cost, cl))
            cl_s = cl[order]
            first = np.r_[0, np.where(np.diff(cl_s) != 0)[0] + 1]
            grp_start = np.repeat(first, np.diff(np.r_[first, len(cl_s)]))
            keep = order[(np.arange(len(cl_s)) - grp_start) < k]
            return rows[keep], cost[keep]
        ia, ca = topk_per_clip(ia, ca)
        ib, cb = topk_per_clip(ib, cb)
        clip_a, fa, re_a = self.ref[ia, 0], self.ref[ia, 1], self.ref[ia, 2]
        clip_b, fb = self.ref[ib, 0], self.ref[ib, 1]
        order = np.argsort(clip_b, kind="stable")
        cbs, fbs, cls = cb[order], fb[order], clip_b[order]
        lo = np.searchsorted(cls, clip_a, "left"); hi = np.searchsorted(cls, clip_a, "right")
        best = None
        for j in np.where(hi > lo)[0]:
            sl = slice(lo[j], hi[j])
            b = fbs[sl]; c2 = cbs[sl]
            ok = (b >= fa[j] + min_len) & (b <= np.minimum(fa[j] + max_len, re_a[j]))
            if not ok.any():
                continue
            tot = ca[j] + c2[ok] + 0.0008 * (b[ok] - fa[j])
            m = int(np.argmin(tot))
            if best is None or tot[m] < best[0]:
                best = (float(tot[m]), int(clip_a[j]), int(fa[j]), int(b[ok][m]), float(ca[j]), float(c2[ok][m]))
        return best

    def clip_data(self, ci, a, b):
        tname, di, mirror, base = self.clips[ci]
        d = self.tasks[tname]["demos"][di]
        return {k: d[k][a:b + 1] for k in ["eef_pos", "eef_mat", "actions"]}, mirror, base


_DB = None


def get_db(tasks):
    global _DB
    if _DB is None:
        _DB = TransitDB(tasks)
    return _DB
