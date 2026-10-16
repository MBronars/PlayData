"""Human-path play: human trajectories as shape priors, warped onto exact
geometric keyframes.

For a pick-and-place, a human approach+carry clip is chosen by motion matching
(the robot's current position/velocity vs. the clip, and the displacement the
task needs vs. the clip's displacement). The clip's eef path is then warped with
anchors: its start -> the robot's current pose, its gripper-close frame -> the
exact grasp pose computed from object geometry, its release frame -> the exact
place pose. Actions are recomputed by tracking the warped path with velocity
feed-forward (tracking human paths this way is as human-looking as replaying
human actions), and gripper commands follow the human timing, so the robot
closes and releases while moving, exactly where the geometry says."""
import numpy as np
from scipy.spatial.transform import Rotation

from compose import FLIP, rot_err
from composed_play import ComposedPlay
from human_play import GROUP_OF, GEOM
from play_controller import EpisodeTooLong


def _smooth_interp(idx, vals, T):
    """C1-smooth interpolation of anchor corrections over time (min-jerk
    between consecutive anchors, constant outside)."""
    idx = np.asarray(idx); vals = np.asarray(vals, float)
    out = np.zeros((T,) + vals.shape[1:])
    for t in range(T):
        if t <= idx[0]:
            out[t] = vals[0]
        elif t >= idx[-1]:
            out[t] = vals[-1]
        else:
            k = np.searchsorted(idx, t) - 1
            x = (t - idx[k]) / max(idx[k + 1] - idx[k], 1)
            s = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
            out[t] = (1 - s) * vals[k] + s * vals[k + 1]
    return out


def warp(P, R, pos_anchors, rot_anchors):
    """Warp a path so that P'[i] = target at each anchor (i, target), keeping
    the human shape in between. Same for orientations (rotation corrections)."""
    T = len(P)
    pi = [a[0] for a in pos_anchors]
    pc = [a[1] - P[a[0]] for a in pos_anchors]
    Pw = P + _smooth_interp(pi, pc, T)
    ri = [a[0] for a in rot_anchors]
    rc = []
    for i, Rt in rot_anchors:
        Rh = R[i]
        if np.linalg.norm(rot_err(Rt @ FLIP, Rh)) < np.linalg.norm(rot_err(Rt, Rh)):
            Rt = Rt @ FLIP
        rc.append(Rotation.from_matrix(Rt @ Rh.T).as_rotvec())
    # keep consecutive rotation corrections on the same branch
    rv = _smooth_interp(ri, rc, T)
    Rw = np.einsum("tij,tjk->tik", Rotation.from_rotvec(rv).as_matrix(), R)
    return Pw, Rw


class HumanPathPlay(ComposedPlay):
    def __init__(self, pc, rng, bank, tasks, segs, p_explore=0.2):
        super().__init__(pc, rng, bank, p_explore=p_explore)
        self.tasks = tasks
        self.transport = {(s["task"], s["demo"], s["t0"]): s for s in segs if s["type"] == "transport"}
        def carry_rot(tr):
            R = tasks[tr["task"]]["demos"][tr["demo"]]["eef_mat"][tr["t0"]:tr["t1"]]
            return float(np.max(np.linalg.norm(Rotation.from_matrix(R @ R[0].T).as_rotvec(), axis=1))) if len(R) else 9.0
        self.approach = []
        for s in segs:
            if s["type"] != "grasp_approach" or not s["lifted"]:
                continue
            tr = self.transport.get((s["task"], s["demo"], s["t1"]))
            # skip carries with big wrist rotations (tilting a bottle into a rack, pouring...)
            if tr is None or not tr["ends_open"] or carry_rot(tr) > 0.4:
                continue
            self.approach.append(s)

    def _clip(self, s, t0, t1):
        d = self.tasks[s["task"]]["demos"][s["demo"]]
        t1 = min(t1, len(d["actions"]))
        return d["eef_pos"][t0:t1].astype(float), d["eef_mat"][t0:t1].astype(float), d["actions"][t0:t1].astype(float)

    def _select(self, n, g, release, n_cands=150):
        """Motion-matched clip choice. Returns (seg, start_index, transport_seg)."""
        pc = self.pc
        p_cur = pc.eef_pos
        v_cur = getattr(pc, "eef_vel", np.zeros(3))
        grp = GROUP_OF.get(self.cat(n))
        same = [s for s in self.approach if GROUP_OF.get(s["cat"]) == grp]
        pool = same if len(same) >= 20 and self.rng.random() < 0.8 else self.approach
        d_app = g - p_cur
        d_tr = release - g
        tops = self._object_tops()
        best = []
        for i in self.rng.permutation(len(pool))[:n_cands]:
            s = pool[i]
            tr = self.transport[(s["task"], s["demo"], s["t1"])]
            P, _, _ = self._clip(s, s["t0"], s["t1"] + 1)
            if len(P) < 15:
                continue
            V = np.gradient(P, axis=0) * 20
            key = len(P) - 1
            ts_max = max(0, key - 15)
            # remaining displacement of the clip from each candidate start vs what we need
            rem = P[key] - P[:ts_max + 1]
            cost = (np.linalg.norm(rem - d_app, axis=1) + 0.3 * np.linalg.norm(V[:ts_max + 1] - v_cur, axis=1))
            ts = int(np.argmin(cost))
            Pt, _, _ = self._clip(tr, tr["t0"], tr["t1"] + 1)
            c_tr = np.linalg.norm((Pt[-1] - Pt[0]) - d_tr)
            # rough warped approach: does it come in low next to tall things?
            Pa = P[ts:]
            sw = np.linspace(0, 1, len(Pa))[:, None]
            Pa = Pa + (1 - sw) * (p_cur - Pa[0]) + sw * (g - Pa[-1])
            deficit = 0.0
            for xy, top, r in tops:
                near = np.linalg.norm(Pa[:-12, :2] - xy, axis=1) < r + 0.05
                if near.any():
                    far_from_grasp = np.linalg.norm(Pa[:-12, :2] - g[:2], axis=1) > 0.02
                    m = near & (far_from_grasp | (np.linalg.norm(xy - g[:2]) > 0.06))
                    if m.any():
                        deficit = max(deficit, float(np.max(top + 0.015 - Pa[:-12, 2][m])))
            best.append((cost[ts] + 0.8 * c_tr + 3.0 * max(deficit, 0.0), s, ts, tr))
        if not best:
            return None
        best.sort(key=lambda x: x[0])
        return best[self.rng.integers(min(4, len(best)))][1:]

    def _slip_info(self, tr, t_carry):
        if len(tr) < 30:
            return None
        d = np.array([x[1] for x in tr]); P = np.array([x[2] for x in tr])
        base = np.median(d[22:30])
        bad = np.where((d - base > 0.01) & (np.arange(len(d)) < t_carry - 2))[0]
        bad = bad[bad > 25]
        V = np.gradient(P, axis=0) * 20; Acc = np.linalg.norm(np.gradient(V, axis=0) * 20, axis=1)
        amax = float(Acc[22:max(23, t_carry)].max())
        if len(bad) == 0:
            return dict(slip=False, amax=round(amax, 2), vmax=round(float(np.linalg.norm(V[22:t_carry], axis=1).max()), 3))
        k = bad[0]
        return dict(slip=True, at=round(k / t_carry, 2), amax=round(amax, 2), a_at=round(float(Acc[max(0, k - 5):k + 1].max()), 2),
                    v_at=round(float(np.linalg.norm(V[k])), 3), vz_at=round(float(V[k, 2]), 3))

    @staticmethod
    def _lift_first(Pw, t_close, t_open, h=0.035):
        """Carried objects leave the surface (and come down onto it) mostly
        vertically: horizontal motion is blended in with height above the
        grasp / release level, so tall objects don't drag and pivot."""
        Pw = Pw.copy()
        z0, z1 = Pw[t_close, 2], Pw[t_open, 2]
        xy0, xy1 = Pw[t_close, :2].copy(), Pw[t_open, :2].copy()
        mj = lambda x: (lambda c: 10 * c ** 3 - 15 * c ** 4 + 6 * c ** 5)(np.clip(x, 0, 1))
        mid = (t_close + t_open) // 2
        for t in range(t_close + 1, mid):
            w = mj((Pw[t, 2] - z0) / h)
            Pw[t, :2] = xy0 + w * (Pw[t, :2] - xy0)
        for t in range(mid, t_open):
            w = mj((Pw[t, 2] - z1) / h)
            Pw[t, :2] = xy1 + w * (Pw[t, :2] - xy1)
        # smooth the result a little to remove kinks from the blend
        from scipy.ndimage import gaussian_filter1d
        seg = slice(t_close + 1, t_open)
        Pw[seg, :2] = gaussian_filter1d(Pw[:, :2], sigma=1.5, axis=0)[seg]
        return Pw

    def _object_tops(self, exclude=None):
        from human_play import top_center
        out = []
        for m in self.pc.names:
            if m == exclude:
                continue
            p, q = self.obj_pose(m)
            cat = self.cat(m)
            r = 0.5 * max(np.array(GEOM[cat]["hi"][:2]) - np.array(GEOM[cat]["lo"][:2])) if cat in GEOM else 0.05
            out.append((p[:2], top_center(cat, p, q)[2], r))
        return out

    def _approach_clearance(self, Pw, n, g, t_close, margin=0.015, finger_r=0.05):
        """Keep the fingertips above objects (incl. the target) while the hand
        is not yet over the grasp point; descend near-vertically at the end."""
        from scipy.ndimage import gaussian_filter1d
        tops = self._object_tops()
        T = t_close + 1
        deficit = np.zeros(len(Pw))
        for t in range(T - 3):
            need = -np.inf
            for xy, top, r in tops:
                dxy = np.linalg.norm(Pw[t, :2] - xy)
                if dxy < r + finger_r:
                    # the target itself only matters until we're nearly above the grasp point
                    if np.linalg.norm(xy - self.obj_pose(n)[0][:2]) < 1e-6 and np.linalg.norm(Pw[t, :2] - g[:2]) < 0.02:
                        continue
                    need = max(need, top + margin)
            if need > -np.inf:
                deficit[t] = max(0.0, need - Pw[t, 2])
        if deficit.max() <= 0:
            return Pw
        bump = gaussian_filter1d(deficit, sigma=4)
        bump *= deficit.max() / max(bump.max(), 1e-9)
        bump[T - 1:] = 0
        # fade the bump into the final descent so the grasp anchor stays exact
        fade = np.clip((T - 1 - np.arange(len(Pw))) / 15.0, 0, 1)
        Pw = Pw.copy(); Pw[:, 2] += bump * fade
        return Pw

    def _clearance(self, Pw, n, g, obj0, t_close, t_open, margin=0.012):
        """Lift the carry path wherever the carried object's bottom would come
        within `margin` of the table or another object's top (smooth bump);
        anchors at grasp and release stay exact."""
        from scipy.ndimage import gaussian_filter1d
        pc = self.pc
        cat = self.cat(n)
        h_below = (g[2] - obj0[2]) + (-GEOM[cat]["lo"][2] if cat in GEOM else 0.03)  # eef z minus object bottom z
        rad = 0.5 * max(np.array(GEOM[cat]["hi"][:2]) - np.array(GEOM[cat]["lo"][:2])) if cat in GEOM else 0.04
        others = []
        for m in pc.names:
            if m == n:
                continue
            p, q = self.obj_pose(m)
            from human_play import top_center
            others.append((p[:2], top_center(self.cat(m), p, q)[2],
                           0.5 * max(np.array(GEOM[self.cat(m)]["hi"][:2]) - np.array(GEOM[self.cat(m)]["lo"][:2])) if self.cat(m) in GEOM else 0.05))
        T = len(Pw)
        deficit = np.zeros(T)
        lo_t = t_close + 8
        hi_t = max(lo_t + 1, t_open - 6)
        for t in range(lo_t, min(hi_t, T)):
            bottom = Pw[t, 2] - h_below
            need = pc.table_z + margin
            for xy, top, r in others:
                if np.linalg.norm(Pw[t, :2] - xy) < r + rad + 0.01:
                    need = max(need, top + margin)
            deficit[t] = max(0.0, need - bottom)
        if deficit.max() > 0:
            bump = gaussian_filter1d(np.maximum.accumulate(deficit) * 0 + deficit, sigma=6)
            bump *= deficit.max() / max(bump.max(), 1e-9)
            # keep anchors exact
            w = np.ones(T); w[:t_close + 1] = 0; w[t_open:] = 0
            Pw = Pw.copy(); Pw[:, 2] += bump * w
        return Pw

    @staticmethod
    def _local_slow(P, R, grip, center, half=3, factor=2.5):
        """Slow the clip down locally around `center` (the grasp moment, where
        humans are slowest anyway) so the fingers finish closing before the
        lift. Returns resampled P, R, grip and a function mapping old->new idx."""
        T = len(P)
        rate = np.ones(T)
        lo, hi = max(0, center - half), min(T, center + half + 1)
        rate[lo:hi] = 1.0 / factor
        tq = [0.0]
        while tq[-1] < T - 1:
            k = int(min(tq[-1], T - 1))
            tq.append(tq[-1] + rate[k])
        tq = np.clip(np.array(tq), 0, T - 1)
        Pn = np.stack([np.interp(tq, np.arange(T), P[:, i]) for i in range(3)], 1)
        from scipy.spatial.transform import Slerp
        Rn = Slerp(np.arange(T), Rotation.from_matrix(R))(tq).as_matrix()
        gn = grip[np.round(tq).astype(int)]
        def remap(i):
            return int(np.searchsorted(tq, i))
        return Pn, Rn, gn, remap

    def p_pick_place(self, n, explore=False, max_attempts=3):
        pc, c = self.pc, self.c
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        dest = self._place_goal(n)
        if dest is None:
            return None
        kind, goal, label = dest
        attempts, grasped, src = 0, False, ""
        while attempts < max_attempts:
            attempts += 1
            gp = self.grasp_pose(n, explore=explore and attempts == 1)
            if gp is None:
                return None
            g, Rg = gp
            obj0, _ = self.obj_pose(n)
            rel = g - obj0
            h_obj = (GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]) if self.cat(n) in GEOM else 0.05
            drop = self.rng.uniform(0.003, 0.012) if h_obj > 0.08 else self.rng.uniform(0.01, 0.025)
            release = goal + rel + np.r_[0, 0, drop]
            sel = self._select(n, g, release)
            if sel is None:
                return None
            s, ts, tr = sel
            src = f"{s['task']}#{s['demo']}"
            a0 = s["t0"] + ts
            t_close = s["t1"] - a0
            t_open = tr["t1"] - a0
            retreat = int(self.rng.integers(12, 22))
            P, R, A = self._clip(s, a0, tr["t1"] + retreat)
            T = len(P)
            t_open = min(t_open, T - 1)
            rot_keep = float(__import__("os").environ.get("CARRY_ROT", "0.5"))
            # humans close on the way down and keep descending a little: put the
            # exact grasp pose at the lowest point right after the close command
            t_cmd = t_close
            win = P[t_cmd:min(t_cmd + 12, t_open - 5), 2]
            t_close = t_cmd + (int(np.argmin(win)) if len(win) else 0)
            R_place = Rotation.from_rotvec([0, 0, self.rng.normal(0, 0.12) * rot_keep * 2]).as_matrix() @ Rg
            # damp the human's wrist rotation during the carry
            Rr = R.copy()
            rv = Rotation.from_matrix(R[t_close:] @ R[t_close].T).as_rotvec() * rot_keep
            Rr[t_close:] = np.einsum("tij,jk->tik", Rotation.from_rotvec(rv).as_matrix(), R[t_close])
            R = Rr
            Pw, Rw = warp(P, R,
                          [(0, pc.eef_pos), (t_close, g), (t_open, release), (T - 1, P[T - 1] - P[t_open] + release)],
                          [(0, pc.eef_mat), (t_close, Rg), (t_open, R_place), (T - 1, R_place)])
            Pw = self._approach_clearance(Pw, n, g, t_close)
            Pw = self._lift_first(Pw, t_close, t_open)
            Pw = self._clearance(Pw, n, g, obj0, t_close, t_open)
            grip = np.where(A[:, 6] > 0, 1.0, -1.0)
            # start closing at most ~3 steps before the hand reaches the grasp pose
            grip[:max(t_cmd, t_close - 3)] = -1.0
            # let the fingers finish closing before the human lift starts
            Pw, Rw, grip, remap = self._local_slow(Pw, Rw, grip, t_close)
            t_close, t_open, T = remap(t_close), min(remap(t_open), len(Pw) - 1), len(Pw)
            # phase 1: approach up to the close frame
            pc.phase = "approach"
            c.follow(Pw[:t_close + 1], np.gradient(Pw[:t_close + 1], axis=0), Rw[:t_close + 1], grip[:t_close + 1],
                     residual=0, wobble=0, act_noise=0)
            at_close = dict(err=(pc.eef_pos - g).round(3).tolist(), rot=round(float(np.linalg.norm(rot_err(Rg, pc.eef_mat))), 2),
                            yaw_err=round(float(np.degrees(np.arctan2((Rg.T @ pc.eef_mat)[1, 0], (Rg.T @ pc.eef_mat)[0, 0]))), 1),
                            width=round(float(pc.obs["robot0_gripper_qpos"][0] - pc.obs["robot0_gripper_qpos"][1]), 3))
            if np.linalg.norm(pc.eef_pos - g) > 0.006:
                pc.phase = "correct"
                far = np.linalg.norm(pc.eef_pos - g) > 0.015
                c.settle(g, Rg, -1.0 if far else float(grip[t_close]), tol=0.005, max_steps=14 if far else 8)
                if far:
                    c.move([(0.35, g + np.r_[0, 0, -0.002])], Rg, grip=1.0, residual=0.0, wobble=0.0, act_noise=0.0)
            # phase 2: human close-and-carry, warped to the exact release
            obj_g, _ = self.obj_pose(n)
            z0 = obj_g[2]
            state = {"ok": None, "held_rel": None, "err_rel": None}

            state["trace"] = []

            m_, d_ = pc.m, pc.d
            obj_geoms = set(pc._obj_geoms[n])
            finger_ids = set(m_.geom(gn).id for gn in pc.robot.gripper.contact_geoms)
            state["events"] = []

            def check(t):
                op, _ = self.obj_pose(n)
                state["trace"].append((t, float(np.linalg.norm(pc.eef_pos - op)), pc.eef_pos.copy(), bool(pc.is_grasped(n))))
                others = set()
                for k in range(d_.ncon):
                    cc = d_.contact[k]
                    if cc.geom1 in obj_geoms and cc.geom2 not in finger_ids and cc.geom2 not in obj_geoms:
                        others.add(m_.body(m_.geom_bodyid[cc.geom2]).name)
                    elif cc.geom2 in obj_geoms and cc.geom1 not in finger_ids and cc.geom1 not in obj_geoms:
                        others.add(m_.body(m_.geom_bodyid[cc.geom1]).name)
                if t > 8 and t < t_open - t_close - 2 and others:
                    state["events"].append((t, sorted(others)[:2], round(float(state["trace"][-1][1] - state["trace"][8][1]), 3)))
                if t == t_open - t_close - 1:
                    state["held_rel"] = bool(pc.is_grasped(n))
                    state["err_rel"] = float(np.linalg.norm(pc.eef_pos - release))
                    state["obj_rel"] = (self.obj_pose(n)[0] - goal).round(3).tolist()
                if t == 22:
                    state["ok"] = self.obj_pose(n)[0][2] - z0 > 0.015 and pc.is_grasped(n)
                    return not state["ok"]
                return False

            pc.phase = "transport"
            seg = slice(t_close, T)
            Pc = Pw[seg] + (pc.eef_pos - Pw[t_close]) * np.linspace(1, 0, T - t_close)[:, None]
            c.follow(Pc, np.gradient(Pc, axis=0), Rw[seg], grip[seg], residual=0, wobble=0, act_noise=0,
                     stop_fn=check)
            if state["ok"] is False:
                pc.phase = "regrasp"
                c.move([(0.25, pc.eef_pos + np.r_[0, 0, 0.004]), (0.9, pc.eef_pos + np.r_[self.rng.normal(0, 0.01, 2), 0.06])],
                       pc.eef_mat, grip=-1.0, precise_end=False)
                continue
            grasped = True
            break
        if pc.grip > 0:
            c.move([(0.6, pc.eef_pos + np.r_[0, 0, 0.04])], pc.eef_mat, grip=-1.0, precise_end=False)
        p_end, _ = self.obj_pose(n)
        placed = grasped and np.linalg.norm(p_end[:2] - goal[:2]) < 0.05
        if kind == "on":
            lang, prim = f"put the {nice} on the {label.rsplit('_', 1)[0].replace('_', ' ')}", "pick_place_on"
        elif kind == "in_drawer":
            lang, prim = f"put the {nice} in the {label.split(':')[1]} drawer", "pick_place_drawer"
        else:
            lang, prim = f"pick up the {nice} and put it down somewhere else", "pick_place"
        return dict(primitive=prim, target=f"{n}->{label}", success=bool(placed), grasped=grasped,
                    attempts=attempts, language=lang, explore=bool(explore), source=src,
                    diag=dict(place_err=float(np.linalg.norm(p_end[:2] - goal[:2])), upright=bool(self.upright(n)),
                              held_at_release=state.get("held_rel"), eef_err_at_release=state.get("err_rel"),
                              obj_at_release=state.get("obj_rel"), t_carry=int(t_open - t_close),
                              slip=self._slip_info(state.get("trace", []), t_open - t_close),
                              at_close=at_close,
                              contacts=state.get("events", [])[:1], n_contact=len(state.get("events", [])),
                              grasped_trace=[x[3] for x in state.get("trace", [])][8::10]))
