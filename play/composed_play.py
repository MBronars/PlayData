"""Composed play: keyframes from scene geometry, style from human statistics
(see compose.py). Primitives: pick & place (table / onto object / into drawer)
with natural regrasps, drawer open (open-gripper hook, as humans do) and close,
reach-and-hesitate, and deliberate fumbles."""
import numpy as np

from errors import has as _has

from compose import Composer, R_DOWN, rotz, wrap, rot_err
from human_play import GEOM, GROUP_OF, quat_to_mat, top_center, bottom_offset, RECEPTACLE_GROUPS
from play_controller import EpisodeTooLong

RIM = {"bowl", "mug", "plate"}


def small_tilt(rng, max_deg=7.0):
    ax = rng.normal(size=2)
    ax = np.r_[ax / (np.linalg.norm(ax) + 1e-9), 0.0]
    ang = np.radians(abs(rng.normal(0, max_deg / 2)))
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(ax * min(ang, np.radians(max_deg))).as_matrix()


class ComposedPlay:
    p_err = 0.3   # fraction of actions with an injected near-failure (see errors.py)
    err_severity = 0.15   # fraction of those drawn from the large, obvious ranges
    def __init__(self, pc, rng, bank, p_explore=0.2):
        self.pc, self.rng = pc, rng
        self.c = Composer(pc, rng, bank)
        self.c.ik_q7 = self._ik_q7
        self.p_explore = p_explore
        self.segments = []
        pc.drawers = pc._find_drawers(("top", "middle", "bottom"))
        pc.slew = (0.16, 0.05)   # per-step action change limits (~human p99)
        self.fixture_boxes = self._fixture_boxes()
        from collision import CollisionChecker
        self.cc = CollisionChecker(pc)
        self.open_rel = self.cc.gripper_rel()   # gripper is open at episode start
        import os
        self.margin = float(os.environ.get("CC_MARGIN", "0.012"))
        self.plan_log = []
        self._install_contact_log()

    # ----------------------------------------------------------- scene info
    def cat(self, n):
        return self.pc.obj_category(n)

    def obj_pose(self, n):
        m, d = self.pc.m, self.pc.d
        b = m.body(self.pc.e.objects_dict[n].root_body).id
        return d.xpos[b].copy(), d.xquat[b].copy()

    def rel_R(self, n):
        cat = self.cat(n)
        _, q = self.obj_pose(n)
        return quat_to_mat(q) @ quat_to_mat(GEOM[cat]["quat"]).T if cat in GEOM else np.eye(3)

    def upright(self, n):
        return self.rel_R(n)[2, 2] > np.cos(np.radians(25))

    def _fixture_boxes(self):
        m, d, e = self.pc.m, self.pc.d, self.pc.e
        out = []
        for fname, fx in e.fixtures_dict.items():
            if fname.endswith("table") or fname == "floor":
                continue
            bid = m.body(fx.root_body).id
            pts = []
            for g in range(m.ngeom):
                if m.body_rootid[m.geom_bodyid[g]] == bid and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                    cs = c + h * np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)])
                    pts.append(cs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            if pts:
                pts = np.concatenate(pts)
                out.append((fname, pts.min(0), pts.max(0)))
        return out

    def _yaw_for_axis(self, axis_xy):
        """Gripper yaw so that its closing axis is along `axis_xy` (either sign),
        choosing the equivalent closest to the current wrist yaw."""
        phi = np.arctan2(axis_xy[1], axis_xy[0]) - np.pi / 2
        cur = self.pc.eef_yaw()
        return self.pc.pick_yaw([wrap(phi), wrap(phi + np.pi)])

    def _install_contact_log(self):
        """Log unintended contacts: robot (any link) or the held object touching
        an object/fixture that isn't the current target."""
        pc = self.pc
        m = pc.m
        self.contact_log = []
        self.cur_targets = set()
        robot_b = set(b for b in range(m.nbody) if m.body(b).name.startswith(("robot0_", "gripper0_")))
        owner = {}
        for name, obj in list(pc.e.objects_dict.items()) + list(pc.e.fixtures_dict.items()):
            rb = m.body(obj.root_body).id
            for g in range(m.ngeom):
                if m.body_rootid[m.geom_bodyid[g]] == rb:
                    owner[g] = name
        orig = pc.step_raw

        def step_raw(a, _o=orig):
            r = _o(a)
            d = pc.d
            for k in range(d.ncon):
                c = d.contact[k]
                for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    b1 = m.geom_bodyid[g1]
                    o2 = owner.get(g2)
                    if o2 is None or o2.endswith("table") or o2 in self.cur_targets:
                        continue
                    if b1 in robot_b:
                        self.contact_log.append((pc.t, pc.phase, m.body(b1).name, o2))
                    elif owner.get(g1) in self.cur_targets and owner.get(g1) != o2 and pc.phase in ("transport", "release"):
                        self.contact_log.append((pc.t, pc.phase, "held:" + owner.get(g1), o2))
            return r
        pc.step_raw = step_raw

    def grasp_candidates(self, n, explore=False, k=8):
        """Default grasp first, then geometric alternatives: other points on a
        rim, shifted along a box, or across the other axis if it fits."""
        out = []
        d = self.grasp_pose(n, explore=explore)
        if d is not None:
            out.append(d)
        cat = self.cat(n)
        if cat not in GEOM:
            return out
        grp = GROUP_OF.get(cat)
        g = GEOM[cat]
        ext = np.array(g["hi"][:2]) - np.array(g["lo"][:2])
        if grp in RIM:
            for ang in np.linspace(0, 2 * np.pi, 9)[:-1] + self.rng.uniform(0, 0.7):
                o = self.grasp_pose(n, rim_angle=ang)
                if o is not None:
                    out.append(o)
        else:
            for shift in (0.25, -0.25):
                o = self.grasp_pose(n, long_shift=shift)
                if o is not None:
                    out.append(o)
            if max(ext) < 0.072:
                o = self.grasp_pose(n, other_axis=True)
                if o is not None:
                    out.append(o)
        return out[:k + 1]

    def grasp_pose(self, n, explore=False, rim_angle=None, long_shift=0.0, other_axis=False):
        """Exact grasp pose from geometry. Returns (pos, R) or None."""
        cat = self.cat(n)
        if cat not in GEOM:
            return None
        g = GEOM[cat]
        lo, hi = np.array(g["lo"]), np.array(g["hi"])
        p, q = self.obj_pose(n)
        Rr = self.rel_R(n)
        top = top_center(cat, p, q)
        h = hi[2] - lo[2]
        grp = GROUP_OF.get(cat)
        ext = hi[:2] - lo[:2]
        if grp in RIM:
            r = 0.5 * min(ext) - 0.012
            # rim point on the side facing the robot (+/- 50 deg), away from a mug handle
            base_ang = np.pi + self.rng.uniform(-0.9, 0.9)
            if grp == "mug" and rim_angle is None:
                # canonical long horizontal axis holds the handle: pick a rim point across it
                long_ax = Rr @ (np.r_[1, 0, 0] if ext[0] > ext[1] else np.r_[0, 1, 0])
                hdir = np.arctan2(long_ax[1], long_ax[0])
                cands = [hdir + np.pi / 2, hdir - np.pi / 2]
                base_ang = min(cands, key=lambda a: abs(wrap(a - np.pi))) + self.rng.uniform(-0.4, 0.4)
            if rim_angle is not None:
                base_ang = rim_angle
                if grp == "mug":  # stay away from the handle
                    long_ax = Rr @ (np.r_[1, 0, 0] if ext[0] > ext[1] else np.r_[0, 1, 0])
                    hdir = np.arctan2(long_ax[1], long_ax[0])
                    if min(abs(wrap(base_ang - hdir)), abs(wrap(base_ang - hdir - np.pi))) < 0.6:
                        return None
            u = np.r_[np.cos(base_ang), np.sin(base_ang), 0]
            if grp == "plate":
                r = 0.5 * min(ext) - 0.008
            pos = np.r_[top[:2] + r * u[:2], top[2] - (0.01 if grp == "plate" else self.rng.uniform(0.026, 0.034))]
            yaw = self._yaw_for_axis(u[:2])
        else:
            narrow = np.r_[1, 0, 0] if ext[0] < ext[1] else np.r_[0, 1, 0]
            longv = np.r_[0, 1, 0] if ext[0] < ext[1] else np.r_[1, 0, 0]
            if other_axis:
                narrow, longv = longv, narrow
            if (ext @ np.abs(narrow[:2])) > 0.075:
                return None
            ax = Rr @ narrow
            # grip deep enough to hold through the carry: mid-height on flat
            # boxes, well below the cap on tall bottles
            # grip deep: about half the height, up to the finger length
            depth = float(np.clip(0.5 * h, 0.0, 0.045)) if h < 0.035 else float(np.clip(self.rng.uniform(0.45, 0.55) * h, 0.022, 0.045))
            pos = np.r_[top[:2], top[2] - depth]
            if long_shift:
                pos[:2] += (Rr @ longv)[:2] * long_shift * 0.5 * (ext @ np.abs(longv[:2]))
            yaw = self._yaw_for_axis(ax[:2])
        pos[2] = max(pos[2], self.pc.table_z + 0.012)
        if explore:
            pos[:2] += self.rng.normal(0, 0.02, 2)
        R = rotz(yaw) @ R_DOWN @ small_tilt(self.rng, 6.0)
        return pos, R

    # ------------------------------------------------------------ primitives
    def _approach_keys(self, g, app=None):
        """Keyframes toward grasp g; hover / pre-grasp are backed off along the
        grasp's approach axis (straight up for top-down grasps)."""
        app = np.r_[0, 0, -1.0] if app is None else app / np.linalg.norm(app)
        back = -app
        p0 = self.pc.eef_pos
        dist = np.linalg.norm(g - p0)
        T = float(np.clip(1.74 + 4.02 * dist, 1.4, 4.5) * self.rng.uniform(0.85, 1.15))
        keys = []
        hover = g + back * self.rng.uniform(0.07, 0.14) + np.r_[self.rng.normal(0, 0.015, 2), 0.0]
        # hover above first only when coming from somewhere else; if the hand is
        # already roughly above the grasp, go straight down
        if T > 1.5 and np.linalg.norm(p0[:2] - g[:2]) > 0.05:
            if dist > 0.25:
                mid = 0.5 * (p0 + hover)
                mid[2] = max(p0[2], hover[2]) + self.rng.uniform(0.0, 0.04)
                keys.append((0.45 * (T - 1.0), mid))
            keys.append((T - 1.0, hover))
        pre = g + back * self.rng.uniform(0.03, 0.045) + np.r_[self.rng.normal(0, 0.004, 2), 0.0]
        keys.append((T - 0.45, pre))
        keys.append((T, g))
        return keys

    def _clear(self, P, R, moving, ignore):
        # whatever the hand already touches where the path starts (it may have
        # just pushed or put something down) is ignored while moving off it;
        # after the first half second the path has to be clear of it too
        own0 = set()
        for _ in range(3):
            ig0 = lambda t, own: ignore(t, own) or own in own0
            d0, _, o0 = self.cc.path_clearance(P[:1], R[:1], moving, ignore=ig0)
            if d0 >= self.margin or o0 is None:
                break
            own0.add(o0)
        if own0:
            base_ignore = ignore
            ignore = lambda t, own: base_ignore(t, own) or (own in own0 and t < 10)
        dist, t, own = self.cc.path_clearance(P, R, moving, ignore=ignore)
        if dist < -0.01:
            return dist, own
        # joint space: sequential IK along the path; too close to a limit = blocked,
        # a little close = less attractive than a comfortable alternative
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        wd = self.reach.wrist_dist(P[::4], R[::4])
        # never further out than the limit - or than where the arm already is
        # (a plan that starts stretched out may still bring the arm back in)
        if np.max(wd) > max(self.reach.WRIST_MAX + 0.02, wd[0] + 0.005):
            return min(dist, -0.02), "reach-limit"
        jm, _ = self.reach.path_margin(P, R, stride=8)
        self.last_jm = jm
        if jm < 0.08:
            return min(dist, -0.02), "joint-limit"
        if jm < 0.25:
            dist = dist - 0.1 * (0.25 - jm)
        return dist, own

    def _approach_variants(self, g, app=None):
        """Natural (human-sampled) approach first, then higher hovers / a high
        via point that goes over whatever is in the way."""
        yield self._approach_keys(g, app)
        tops = [self.pc.table_z] + [self.obj_pose(m)[0][2] + 0.15 for m in self.pc.names]
        for extra in (0.06, 0.12):
            keys = self._approach_keys(g, app)
            out = []
            for (t, p) in keys[:-2]:
                p = p.copy(); p[2] += extra; out.append((t, p))
            pre = keys[-2]
            out.append((pre[0], pre[1] + np.r_[0, 0, extra * 0.5]))
            out.append(keys[-1])
            yield out
        # straight up first, over, then down
        p0 = self.pc.eef_pos
        high = max(p0[2], g[2] + 0.18)
        dist = np.linalg.norm(g - p0)
        T = float(np.clip(2.2 + 4.0 * dist, 2.0, 5.0))
        back = np.r_[0, 0, 1.0] if app is None else -app / np.linalg.norm(app)
        yield [(0.3 * T, np.r_[p0[:2], high]), (0.7 * T, (g + back * 0.12) * [1, 1, 0] + [0, 0, max(high, (g + back * 0.12)[2])]),
               (T - 0.45, g + back * 0.04), (T, g)]

    def plan_approach(self, n, explore=False):
        """First collision-free approach over grasp alternatives x path variants.
        Returns (g, Rg, P, V, R) or None (then the robot plays with something else)."""
        best = None
        cands = []
        self.grasp_info = []
        cands_ = list(self.grasp_candidates(n, explore=explore))
        err = getattr(self, "err", None) or {}
        if _has(err, "unstable_grasp") and len(cands_) > 1:
            # one of the worst-ranked grasps instead of a good one
            k_ = int(self.rng.integers(max(1, len(cands_) // 2), len(cands_)))
            cands_ = [cands_[k_]] + cands_[:k_] + cands_[k_ + 1:]
            gi = list(getattr(self, "grasp_info", []))
            if len(gi) == len(cands_):
                self.grasp_info = [gi[k_]] + gi[:k_] + gi[k_ + 1:]
        elif _has(err, "grasp_offset"):
            cands_ = [(g_ + R_[:, 0] * err["offset"], R_) for g_, R_ in cands_]
        elif _has(err, "grasp_high"):
            cands_ = [(g_ - R_[:, 2] * err["offset"], R_) for g_, R_ in cands_]
        for gi_, (g, Rg) in enumerate(cands_):
            if tuple(np.round(g, 2)) in getattr(self, "bad_grasps", set()):
                continue
            self.chosen_grasp = self.grasp_info[gi_] if gi_ < len(getattr(self, "grasp_info", [])) else ("other",)
            self._relax(self.chosen_grasp[0] in ("side", "handle"))
            for keys in self._approach_variants(g, Rg[:, 2]):
                P, V, R = self.c.plan(keys, Rg, v_end=Rg[:, 2] * 0.02, rot_end=0.6)
                t_pre = int(round(keys[-2][0] * 20)) if len(keys) >= 2 else len(P)
                sup = self.support_of(n)
                # unusual grasps (by the handle, from the side) come in close
                # to the object itself for longer
                t_own = t_pre - 2 if self.chosen_grasp[0] not in ("side", "handle") else int(0.6 * len(P))
                ign = lambda t, own, t_pre=t_pre, sup=sup, t_own=t_own: (own in ("table", sup) and t >= t_pre - 2) or (own == n and t >= t_own)
                dist, own = self._clear(P, R, self.open_rel + self.cc.forearm_rel(), ign)
                if __import__("os").environ.get("DEBUG_APPR"):
                    d_, t_, o_ = self.cc.path_clearance(P, R, self.open_rel + self.cc.forearm_rel(), ignore=ign)
                    print(f"      APPR {self.chosen_grasp[0]} dist={dist:.3f} own={own} at t={t_}/{len(P)} pair={getattr(self.cc, 'last_pair', None)}", flush=True)
                    if own == "joint-limit":
                        q = self.reach.current_q(); rr = self.reach
                        for t__ in list(range(8, len(P), 8)) + [len(P) - 1]:
                            ep_, er_, q = rr.solve(P[t__], R[t__], q0=q, iters=25)
                            print(f"         t={t__} ep={ep_:.3f} er={er_:.2f} q={np.round(q, 2)} lo={np.round(rr.lo, 2)} hi={np.round(rr.hi, 2)}", flush=True)
                if dist >= 0.003 and self.chosen_grasp[0] in ("side", "handle"):
                    # trying it a different way on purpose: a little closer is fine
                    self.plan_log.append(("approach", "unusual", self.chosen_grasp[0], round(dist, 3)))
                    return g, Rg, P, V, R
                if dist >= 0.025:  # comfortable clearance: take the most natural one
                    self.plan_log.append(("approach", "ok", round(dist, 3)))
                    return g, Rg, P, V, R
                sev = (getattr(self, "err", None) or {}).get("severe")
                if (getattr(self, "careless", False) or sev) and dist >= (-0.04 if sev else -0.012) and own in self.pc.names:
                    # not bothering to find the clean way in: the hand may clip
                    # a neighbouring object on the way (as people do)
                    self.plan_log.append(("approach", "careless", round(dist, 3), own))
                    return g, Rg, P, V, R
                cands.append((dist, own, (g, Rg, P, V, R)))
                if best is None or dist > best[0]:
                    best = (dist, own, (g, Rg, P, V, R))
        self._relax(False)
        if best is not None and best[0] >= self.margin:
            self.plan_log.append(("approach", "tight", round(best[0], 3)))
            return best[2]
        # nothing clean: accept brushing the obstacle - barely, if it is a
        # fixture or the table; a loose object may get nudged a couple of cm
        graze_ok = -0.02 if (best is not None and best[1] in self.pc.names) else -0.004
        if best is not None and best[0] >= graze_ok and getattr(self, "allow_graze", True):
            self.plan_log.append(("approach", "graze", round(best[0], 3), best[1]))
            return best[2]
        self.plan_log.append(("approach", "blocked", round(best[0], 3) if best else None, best[1] if best else None))
        return None

    def _grasp(self, n, explore=False):
        """Approach + close. Returns grasp eef pos or None."""
        pc, c = self.pc, self.c
        self.cur_targets = {n}
        plan = self.plan_approach(n, explore=explore)
        if plan is None:
            return None
        g, R, P, V, Rp = plan
        self._obj_at_plan = self.obj_pose(n)[0]
        pc.phase = "approach"
        c.end_tol = 0.006
        if __import__("os").environ.get("DEBUG_IK"):
            jm_pred, q_pred = self.reach.path_margin(P, Rp, stride=8)
        err = getattr(self, "err", None) or {}
        fly = self.rng.random() < self.p_fly_close
        if _has(err, "close_early"):
            fly = True
        elif (_has(err, "close_late") or _has(err, "lift_early")):
            fly = False
        if fly and len(P) > 20:
            # humans start closing while the hand is still moving in: extend the
            # path by a few mm of creep and fire the close shortly before arrival
            r = self._fly_close(g, R, P, V, Rp)
            if r == "ok":
                return g, R
            if r == "retry":
                self.bad_grasps = getattr(self, "bad_grasps", set()) | {tuple(np.round(g, 2))}
                return "retry"
            if r == "blocked":
                return None
            # "abort": off the plan near the end, fingers still open: finish below
        else:
            c.follow(P, V, Rp, -1.0)
        if __import__("os").environ.get("DEBUG_IK"):
            q_act = self.reach.current_q()
            sh = pc.d.xpos[pc.m.body("robot0_link2").id]
            print("      SHOULDER dist %.3f horiz %.3f dz %.3f" % (np.linalg.norm(g - sh), np.linalg.norm((g - sh)[:2]), (g - sh)[2]), flush=True)
            print("      IKCHECK pred", np.round(q_pred, 2), "actual", np.round(q_act, 2), "pred margin %.2f actual margin %.2f" % (jm_pred, self.reach.margin(q_act)), flush=True)
        c.end_tol = 0.012
        if c.limit_hit and np.linalg.norm(pc.eef_pos - g) > 0.035:
            self.plan_log.append(("approach", "joint stop", round(float(np.linalg.norm(pc.eef_pos - g)), 3)))
            return None
        pc.phase = "correct"
        c.settle(g, R, -1.0, tol=0.005, max_steps=10)
        if np.linalg.norm(pc.eef_pos - g) > 0.035 and not err.get("severe"):
            # really blocked short of it: back off instead of closing on air
            # (a couple of cm off, people just close and see whether it holds)
            self.bad_grasps = getattr(self, "bad_grasps", set()) | {tuple(np.round(g, 2))}
            pc.phase = "regrasp"
            c.move([(0.7, pc.eef_pos - R[:, 2] * 0.06)], pc.eef_mat, grip=-1.0, precise_end=False)
            return "retry"
        if __import__("os").environ.get("DEBUG_GRASP"):
            from compose import rot_err as _re
            op, _ = self.obj_pose(n)
            print(f"      GRASP {n} t={pc.t} info={getattr(self, 'chosen_grasp', None)} eef_err={np.round(pc.eef_pos - g, 3)} rot_err={np.linalg.norm(_re(R, pc.eef_mat)):.2f} obj_moved_during_approach={np.linalg.norm(op[:2] - self._obj_at_plan[:2]):.3f} grasp_rel_obj={np.round(g - self._obj_at_plan, 3)} app={np.round(R[:, 2], 2)} close={np.round(R[:, 0], 2)}", flush=True)
        if _has(err, "close_late"):
            # hesitates and pushes on past the grasp point before closing
            ov = R[:, 2] * err["overshoot"] + np.r_[self.rng.normal(0, 0.006, 2), 0.0]
            pc.phase = "correct"
            c.move([(self.rng.uniform(0.35, 0.5), pc.eef_pos + ov)], R, grip=-1.0, residual=0.0, precise_end=False)
            c.hold(int(self.rng.integers(3, 7)), -1.0)
        pc.phase = "grasp"
        # humans close while still moving slowly: a short creeping motion while
        # the fingers close (2-5 mm), no stop
        creep = R[:, 2] * self.rng.uniform(0.001, 0.004) + np.r_[self.rng.normal(0, 0.0015, 2), 0.0]
        t_close = self.rng.uniform(0.3, 0.45) if not _has(err, "lift_early") else self.rng.uniform(0.05, 0.1)
        if _has(err, "lift_early") and err.get("severe"):
            # a hurried snatch: closing and pulling up at the same moment, so
            # the fingers shut while the hand is already rising
            up = np.r_[0, 0, self.rng.uniform(0.025, 0.045)]
            c.move([(self.rng.uniform(0.3, 0.4), pc.eef_pos + up)], R, grip=1.0, residual=0.0, precise_end=False)
            return g, R
        c.move([(t_close, pc.eef_pos + creep)], R, grip=1.0, residual=0.0, wobble=0.0, precise_end=False)
        return g, R

    p_fly_close = 0.6

    def _fly_close(self, g, R, P, V, Rp):
        """Follow the approach and close on the way in. Returns False (fingers
        still open, nothing committed) if the hand is off the plan when the close
        would fire; the caller then finishes with a regular settle-and-close."""
        pc, c = self.pc, self.c
        creep = float(self.rng.uniform(0.001, 0.004))
        n_ext = int(self.rng.integers(5, 9))
        s_ = np.linspace(0, 1, n_ext + 1)[1:]
        ease = 3 * s_ ** 2 - 2 * s_ ** 3
        ext = g[None] + np.outer(ease, R[:, 2] * creep)
        P2 = np.vstack([P, ext])
        V2 = np.gradient(P2, axis=0)
        R2 = np.concatenate([Rp, np.repeat(Rp[-1:], n_ext, 0)])
        seg = np.linalg.norm(np.diff(P2, axis=0), axis=1)
        rem = np.r_[np.cumsum(seg[::-1])[::-1], 0.0] - creep  # path distance left to g
        d_close = float(self.rng.uniform(0.005, 0.015))
        early = (getattr(self, "err", None) or {}).get("early")
        if early:
            d_close = early      # injected error: closing well before arriving
        i_close = int(np.argmax(rem < d_close))
        i_close = int(np.clip(i_close, len(P) - ((80 if early > 0.06 else 40) if early else 10), len(P2) - 3))
        grip = np.full(len(P2), -1.0)
        grip[i_close:] = 1.0
        state = {"abort": False}

        def stop(t):
            if early:
                return False   # injected: close regardless of where the hand is
            if t == i_close - 1 and np.linalg.norm(P2[t] - pc.eef_pos) > 0.009:
                state["abort"] = True
                return True
            return False
        c.end_tol = 0.008
        c.follow(P2, V2, R2, grip, stop_fn=stop)
        c.end_tol = 0.012
        if state["abort"]:
            return "abort"
        if np.linalg.norm(pc.eef_pos - g) > 0.035:
            if c.limit_hit:
                # stopped by a joint stop well short of it: not an attempt,
                # the arm just can't get there from here
                self.plan_log.append(("approach", "joint stop", round(float(np.linalg.norm(pc.eef_pos - g)), 3)))
                c.move([(0.6, pc.eef_pos + np.r_[0, 0, 0.04])], pc.eef_mat, grip=-1.0, precise_end=False)
                return "blocked"
            return self._fly_miss(g, R)
        pc.phase = "grasp"
        # linger a moment (varied) while the fingers finish closing
        c.hold(int(self.rng.integers(2, 9)), 1.0, drift=R[:, 2] * 0.0002)
        return "ok"

    def _fly_miss(self, g, R):
        """Closed well short of the grasp: still try to lift (the fingers are
        shut, it doesn't come), then open and back off."""
        pc, c = self.pc, self.c
        pc.phase = "grasp"
        c.hold(int(self.rng.integers(3, 7)), 1.0)
        pc.phase = "transport"
        c.move([(self.rng.uniform(0.5, 0.7), pc.eef_pos + np.r_[0, 0, self.rng.uniform(0.03, 0.05)])], pc.eef_mat,
               grip=1.0, precise_end=False)
        pc.phase = "regrasp"
        c.hold(int(self.rng.integers(4, 8)), -1.0, drift=np.r_[0, 0, 0.002])
        c.move([(0.6, pc.eef_pos + np.r_[0, 0, 0.03])], pc.eef_mat, grip=-1.0, precise_end=False)
        return "retry"

    def _place_goal(self, n):
        pc = self.pc
        cat = self.cat(n)
        recs = [m for m in pc.names if m != n and GROUP_OF.get(self.cat(m)) in RECEPTACLE_GROUPS and self.upright(m)]
        r = self.rng.random()
        if recs and r < 0.3:
            t = recs[self.rng.integers(len(recs))]
            p, q = self.obj_pose(t)
            tc = top_center(self.cat(t), p, q)
            goal = np.r_[tc[:2] + self.rng.normal(0, 0.01, 2), tc[2] + bottom_offset(cat) - 0.005]
            return "on", goal, t
        open_dr = [dr for dr in pc.drawers if pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.1 and dr["level"] == "top"]
        if open_dr and r < 0.5:
            dr = open_dr[0]
            R = pc.d.xmat[dr["bid"]].reshape(3, 3)
            hp = pc.d.geom_xpos[dr["handle"]]
            xy = hp[:2] + R[:2, 1] * 0.11
            return "in_drawer", np.r_[xy, hp[2] - 0.035 + bottom_offset(cat)], f"{dr['fixture']}:{dr['level']}"
        xy = self._free_spot(n)
        if xy is None:
            return None
        return "table", np.r_[xy, pc.table_z + bottom_offset(cat) + 0.003], "table"

    def _free_spot(self, n, tries=60, min_move=0.08):
        pc = self.pc
        from scene_gen import OBJ_XY
        sc = {"kitchen_table": "kitchen", "living_room_table": "living_room", "study_table": "study"}
        tname = next((f for f in pc.e.fixtures_dict if f.endswith("table")), "kitchen_table")
        (xl, xh), (yl, yh) = OBJ_XY.get(sc.get(tname, "kitchen"))
        x0, y0 = pc.e.workspace_offset[:2]
        p_now, _ = self.obj_pose(n)
        for _ in range(tries):
            xy = np.array([x0 + self.rng.uniform(xl, xh), y0 + self.rng.uniform(yl, yh)])
            if np.linalg.norm(xy - p_now[:2]) < min_move:
                continue
            # only places the arm can comfortably reach with a top-down hand
            rb = np.linalg.norm(xy - np.array(pc.robot.base_pos)[:2])
            if not (0.47 < rb < 0.62):
                continue
            if hasattr(self, "reach"):
                wp = np.r_[xy, pc.table_z + 0.03]
                if self.reach.wrist_dist(wp, np.diag([1.0, -1.0, -1.0]))[0] > self.reach.WRIST_MAX - 0.02:
                    continue
            if any(np.all(xy > lo[:2] - 0.07) and np.all(xy < hi[:2] + 0.07) for _, lo, hi in self.fixture_boxes):
                continue
            if all(np.linalg.norm(self.obj_pose(m)[0][:2] - xy) > self._radius(n) + self._radius(m) + 0.045
                   for m in pc.names if m != n):
                return xy
        return None

    def _ik_q7(self, pos, R):
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        ep, er, q = self.reach.solve(pos, R, q0=self.reach.current_q(), iters=30)
        return float(q[6]) if ep < 0.02 else None

    def _relax(self, on):
        """More wrist range while trying an unusual grasp (planning and the
        joint guard); back to the human range when the action is over."""
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        self.reach.relax(on)
        self.c.relax(on)

    def _swap_err(self, err, old, new, **params):
        if err.get("parts"):
            err["parts"] = [new if t == old else t for t in err["parts"]]
            err["type"] = "+".join(err["parts"])
        else:
            err["type"] = new
        err.update(params)

    def _retry_err(self):
        """After a failed injected grasp: retry cleanly, or (in a high-severity
        run) fumble again."""
        e = getattr(self, "err", None) or {}
        if e.get("severe") and self.rng.random() < self.err_severity * 0.6:
            from errors import _combo
            for _ in range(10):
                e2 = _combo(self.rng)
                if any(t in e2["parts"] for t in ("grasp_offset", "grasp_high", "close_early", "lift_early", "unstable_grasp")):
                    break
            lab = getattr(self, "_err_label", None)
            if lab is not None:
                lab.setdefault("retry_errors", []).append(e2["type"])
            return e2
        return None

    def _carry_hit_target(self, n, goal):
        """A standing loose object near the way to `goal` to knock on the way."""
        pc = self.pc
        p0 = self.obj_pose(n)[0]
        best = None
        for m in pc.names:
            if m == n or self.support_of(m) is not None or self.support_of(m) == n or GROUP_OF.get(self.cat(m)) is None:
                continue
            pm = self.obj_pose(m)[0]
            if np.linalg.norm(pm[:2] - np.array(pc.robot.base_pos)[:2]) < 0.45:
                continue
            # distance from the segment start->goal
            d = goal[:2] - p0[:2]
            s_ = float(np.clip((pm[:2] - p0[:2]) @ d / max(d @ d, 1e-9), 0, 1))
            off = np.linalg.norm(p0[:2] + s_ * d - pm[:2])
            if off < 0.25 and (best is None or off < best[0]):
                best = (off, m)
        return None if best is None else best[1]

    def _knock_spot(self, n, goal, overlap):
        """Injected error: a set-down spot that overlaps a standing neighbour
        near the goal, so the object lands half on it."""
        pc = self.pc
        cands = []
        for m in pc.names:
            if m == n or self.support_of(m) is not None or self.support_of(m) == n:
                continue
            pm = self.obj_pose(m)[0]
            dd = np.linalg.norm(pm[:2] - goal[:2])
            if dd < 0.3 and np.linalg.norm(pm[:2] - self.obj_pose(n)[0][:2]) > 0.08:
                cands.append((dd, m, pm))
        if not cands:
            return None
        _, m, pm = min(cands, key=lambda x: x[0])
        u = goal[:2] - pm[:2]
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.r_[1.0, 0.0]
        r = self._radius(n) * 0.7 + self._radius(m) * 0.7 - overlap
        xy = pm[:2] + u * max(r, 0.01)
        if not (0.47 < np.linalg.norm(xy - np.array(pc.robot.base_pos)[:2]) < 0.62):
            return None
        return np.r_[xy, goal[2]], m

    def _radius(self, n):
        cat = self.cat(n)
        if cat not in GEOM:
            return 0.05
        e = np.array(GEOM[cat]["hi"][:2]) - np.array(GEOM[cat]["lo"][:2])
        return 0.5 * float(np.linalg.norm(e))

    def p_pick_place(self, n, explore=False, max_attempts=3, dest=None, lang=None, prim=None, inspect=False, yaw_turn=0.0):
        pc, c = self.pc, self.c
        self.cur_targets = {n}
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        dest = dest if dest is not None else self._place_goal(n)
        if dest is None:
            return None
        kind, goal, label = dest
        p_obj0, _ = self.obj_pose(n)
        goal_true = None
        attempts, grasped = 0, False
        state = {}
        self.put_back = False
        while attempts < max_attempts:
            attempts += 1
            # turning the object in the hand on top of an already twisted
            # grasp (from the side / by a handle) flips the wrist: plain grasp
            self._no_unusual = abs(yaw_turn) > 0
            try:
                r = self._grasp(n, explore=explore and attempts == 1)
            finally:
                self._no_unusual = False
            if r == "retry":
                if getattr(self, "err", None):
                    if self.rng.random() < 0.5:
                        break
                    self.err = self._retry_err()
                continue
            if r is None:
                if attempts > 1:
                    break     # tried and failed already: that's the action
                return None
            g, Rg = r
            obj_g, _ = self.obj_pose(n)
            ehit = getattr(self, "err", None) or {}
            if _has(ehit, "carry_hit") and getattr(self, "_bump_target", None) is None and not inspect \
                    and pc.is_grasped(n):
                m_hit = self._carry_hit_target(n, goal)
                if m_hit is not None:
                    ehit["hit_carry"] = m_hit
                    self._bump_target, self._bump_fast = m_hit, True
                    try:
                        held = self._bump(n, obj_g[2])
                    finally:
                        self._bump_target, self._bump_fast = None, False
                    if not held and not pc.is_grasped(n):
                        grasped = True     # knocked out of the hand: it stays where it fell
                        state = {"ok": True}
                        break
            if inspect and not self._inspect(n, obj_g[2]):
                pc.phase = "regrasp"
                c.move([(0.25, pc.eef_pos + np.r_[0, 0, 0.004]), (0.9, pc.eef_pos + np.r_[0, 0, 0.06])], pc.eef_mat, grip=-1.0, precise_end=False)
                continue
            # lift; abort early if the object doesn't come along (then regrasp)
            z0 = obj_g[2]
            rel = pc.eef_pos - self.obj_pose(n)[0]
            h_obj = (GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]) if self.cat(n) in GEOM else 0.05
            # tall things are set down almost on the surface so they don't topple
            drop = self.rng.uniform(0.003, 0.012) if h_obj > 0.08 else self.rng.uniform(0.01, 0.025)
            err = getattr(self, "err", None) or {}
            if _has(err, "drop_high"):
                drop = err["drop"]
            if err.get("severe") and _has(err, "place_offset") and kind == "table" and goal_true is None:
                # off target on the open table just looks like another spot:
                # come down half on a neighbour instead
                self._swap_err(err, "place_offset", "knock_over", overlap=float(self.rng.uniform(0.02, 0.045)))
            if _has(err, "knock_over") and goal_true is None:
                kn = self._knock_spot(n, goal, err["overlap"])
                if kn is None:
                    # nothing near to land on: drop it from high instead
                    self._swap_err(err, "knock_over", "drop_high", drop=float(self.rng.uniform(0.15, 0.25)))
                    drop = err["drop"]
                else:
                    goal_true = goal.copy()
                    goal, self._knock_m = kn
                    err["hit"] = self._knock_m
            if _has(err, "place_offset") and goal_true is None:
                goal_true = goal.copy()
                goal = goal + np.r_[err["offset_xy"], 0.0]
            release = goal + rel + np.r_[0, 0, drop]
            turn = yaw_turn + self.rng.normal(0, 0.15)
            R_end = rotz(turn) @ Rg
            if abs(yaw_turn) > 0:
                # the object turns about the hand: the hand must end where the
                # turned object's grasp point sits over the goal
                rel = rotz(turn) @ rel
                release = goal + rel + np.r_[0, 0, drop]
            carry = self.plan_carry(n, kind, goal, label, rel, drop, R_end, obj_g)
            keys, goal, release = carry
            state = {"ok": None}

            state["rel0"] = None
            state["carry_len"] = len(keys[0]) if isinstance(keys, tuple) else 0
            # check that it came along once the hand should have risen ~3 cm
            # (a carry that also turns the object rises more slowly)
            Pk = keys[0]
            up_ = np.where(Pk[:, 2] - Pk[0, 2] > 0.03)[0]
            state["t_chk"] = int(np.clip(up_[0] if len(up_) else 16, 12, 45))
            if _has(err, "release_early"):
                state["early_open_t"] = max(20, int(err["frac"] * state["carry_len"]))

            def check(t):
                rz = self.obj_pose(n)[0][2] - pc.eef_pos[2]
                if t == state["t_chk"] + 4:
                    state["rel0"] = rz
                if state.get("rel0") is not None and "mid_drop_t" not in state and t < state["carry_len"] - 14 \
                        and rz < state["rel0"] - 0.03:
                    state["mid_drop_t"] = t
                    state["mid_drop_frac"] = round(t / max(state["carry_len"], 1), 2)
                    state["mid_drop_speed"] = round(float(np.linalg.norm(pc.eef_vel)), 3)
                    state["mid_drop_acc"] = round(float(np.linalg.norm(pc.eef_vel - state.get("v_prev", pc.eef_vel)) * 20), 2)
                state["v_prev"] = pc.eef_vel.copy()
                if t == state["t_chk"]:
                    lifted = self.obj_pose(n)[0][2] - z0
                    state["ok"] = bool(lifted > 0.015 and pc.is_grasped(n))
                    if __import__("os").environ.get("DEBUG_LIFT"): print("      LIFTCHECK lifted %.3f grasped %s ok %s" % (lifted, pc.is_grasped(n), state["ok"]), flush=True)
                    return not state["ok"]
                if state.get("early_open_t") is not None and t == state["early_open_t"] and state.get("ok"):
                    state["early_open"] = True
                    return True
                if t > state["t_chk"] and state.get("ok") and "drop_t" not in state and not pc.is_grasped(n):
                    state["drop_t"] = pc.t
                    state["drop_frac"] = round(t / max(len(keys[0]), 1), 2)
                    state["drop_speed"] = round(float(np.linalg.norm(pc.eef_vel)), 3)
                return False

            pc.phase = "transport"
            P_, V_, R_ = keys
            c.follow(P_, V_, R_, 1.0, stop_fn=check)
            if state.get("early_open"):
                # injected error: the hand opens while still on its way
                pc.phase = "release"
                k0 = state["early_open_t"]
                rest = P_[k0:]
                if len(rest) > 4:
                    c.follow(rest, np.gradient(rest, axis=0), R_[k0:], -1.0, precise_end=False)
                grasped = True
                break
            if state["ok"] is False:
                pc.phase = "regrasp"
                c.hold(int(self.rng.integers(5, 8)), -1.0, drift=np.r_[0, 0, 0.002])
                c.move([(0.8, pc.eef_pos + np.r_[self.rng.normal(0, 0.01, 2), 0.06])], pc.eef_mat, grip=-1.0)
                if getattr(self, "err", None):
                    # after an injected slip: either one clean try more, or give up
                    if self.rng.random() < 0.5:
                        break
                    self.err = self._retry_err()
                    if self.err is not None and (_has(self.err, "shaky")):
                        pc.noise_cfg = dict(self.err, t0=pc.t)
                continue
            grasped = True
            break
        # release and retreat
        state_held = pc.is_grasped(n)
        at_open = dict(eef_err=(pc.eef_pos - release).round(3).tolist() if "release" in dir() else None,
                       obj_vs_goal=(self.obj_pose(n)[0] - goal).round(3).tolist(), attempts=attempts,
                       grasped=grasped)
        pc.phase = "release"
        # open while already starting to move away (no pause)
        g_open = np.r_[np.full(int(self.rng.integers(3, 6)), -1.0)]
        self._unwinding = False
        P_, V_, R_ = self.plan_retreat(n)
        g_was = c.guard
        if self._unwinding:
            c.guard = False    # turning the wrist back toward neutral: only gets better
        try:
            c.follow(P_, V_, R_, -1.0, precise_end=False)
        finally:
            c.guard = g_was
        p_end, _ = self.obj_pose(n)
        goal_chk = goal_true if goal_true is not None else goal
        placed = grasped and np.linalg.norm(p_end[:2] - goal_chk[:2]) < 0.05 and not state.get("early_open")
        if kind == "in":
            placed = grasped and self.inside(n, label)
        if self.put_back:
            placed = False
        self._diag = dict(at_open=at_open, mid_drop=state.get("mid_drop_t") is not None, mid_drop_frac=state.get("mid_drop_frac"),
                          mid_drop_speed=state.get("mid_drop_speed"), carry_steps=state.get("carry_len"),
                          carry_plan=[x for x in self.plan_log if x[0] == "carry"][-1:],
                          grasp=list(getattr(self, "chosen_grasp", ("?",))), moved=float(np.linalg.norm(p_end[:2] - p_obj0[:2])), put_back=bool(self.put_back), drop_t=state.get("drop_t") if isinstance(state, dict) else None,
                          drop_frac=state.get("drop_frac") if isinstance(state, dict) else None,
                          drop_speed=state.get("drop_speed") if isinstance(state, dict) else None,
                          place_err=float(np.linalg.norm(p_end[:2] - goal[:2])), upright=bool(self.upright(n)),
                          z_err=float(p_end[2] - goal[2]), held_at_release=bool(state_held))
        if lang is None:
            if kind == "on":
                lang, prim = f"put the {nice} on the {label.rsplit('_', 1)[0].replace('_', ' ')}", "pick_place_on"
            elif kind == "in":
                lang, prim = f"put the {nice} in the {label.rsplit('_', 1)[0].replace('_', ' ')}", "put_in"
            elif kind == "in_drawer":
                lang, prim = f"put the {nice} in the {label.split(':')[1]} drawer", "pick_place_drawer"
            else:
                lang, prim = f"pick up the {nice} and put it down somewhere else", "pick_place"
        return dict(primitive=prim, target=f"{n}->{label}", success=bool(placed), grasped=grasped,
                    attempts=attempts, language=lang, explore=bool(explore), diag=self._diag)

    def inside(self, n, cont):
        p, _ = self.obj_pose(n)
        cp, cq = self.obj_pose(cont)
        cat = self.cat(cont)
        if cat not in GEOM:
            return False
        lo, hi = np.array(GEOM[cat]["lo"]), np.array(GEOM[cat]["hi"])
        Rr = quat_to_mat(cq) @ quat_to_mat(GEOM[cat]["quat"]).T
        local = Rr.T @ (p - cp)
        return bool(np.all(local[:2] > lo[:2] - 0.01) and np.all(local[:2] < hi[:2] + 0.01) and p[2] < cp[2] + hi[2] + 0.03)

    def support_of(self, n):
        """Container / plate the object currently rests in or on, if any."""
        p, _ = self.obj_pose(n)
        for m in self.pc.names:
            if m == n or GROUP_OF.get(self.cat(m)) not in ("basket", "container", "plate", "bowl"):
                continue
            if self.inside(n, m) and p[2] > self.pc.table_z + 0.012:
                return m
        return None

    def _carry_keys(self, p0, release, extra=0.0):
        dist = np.linalg.norm(release[:2] - p0[:2])
        T = float(np.clip(2.42 + 3.54 * dist, 1.8, 5.0) * self.rng.uniform(0.85, 1.15)) + extra * 4
        lift = p0 + np.r_[(release[:2] - p0[:2]) * 0.04, self.rng.uniform(0.05, 0.08) + extra * 0.5]
        mid = 0.5 * (p0 + release)
        mid[2] = max(p0[2], release[2]) + self.rng.uniform(0.06, 0.13) + extra
        pre = release + np.r_[self.rng.normal(0, 0.004, 2), self.rng.uniform(0.03, 0.05)]
        keys = [(0.9, lift)]
        if T > 2.4 or extra > 0:
            keys.append((0.5 * (T + 0.9) - 0.2, mid))
        keys += [(T - 0.45, pre), (T, release)]
        return keys

    def plan_carry(self, n, kind, goal, label, rel, drop, R_end, obj_g):
        """Collision-free carry for the gripper + held object (+ anything inside
        it). Real destinations always win over putting it back: comfortable
        clearance first, then tight, then grazing; putting it back where it was
        is the last resort (and counts as a failed move)."""
        pc = self.pc
        moving = self.cc.gripper_rel() + self.cc.attach_rel(pc._obj_geoms[n]) + self.cc.forearm_rel()
        for m in pc.names:
            if m != n and self.support_of(m) == n:
                moving += self.cc.attach_rel(pc._obj_geoms[m])
        goals = [goal]
        if kind == "table":
            for _ in range(6):
                xy = self._free_spot(n, min_move=getattr(self, "min_move", 0.08))
                if xy is not None:
                    goals.append(np.r_[xy, goal[2]])
        h_obj = (GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]) if self.cat(n) in GEOM else 0.05
        want = 0.03 if h_obj > 0.08 else 0.025
        sup = self.support_of(n)
        recv = label if kind in ("on", "in") else None
        # whatever the held object already (nearly) touches at the start - a
        # neighbour, the cabinet side - is ignored while lifting off it
        near0 = set()
        for _ in range(3):
            ig0 = lambda t, own: own in near0 or own in ("table", sup)
            d0, _, o0 = self.cc.path_clearance(np.array([pc.eef_pos]), np.array([pc.eef_mat]), moving, ignore=ig0)
            if d0 >= want or o0 is None:
                break
            near0.add(o0)
        tried = []
        for gi, gl in enumerate(goals):
            release = gl + rel + np.r_[0, 0, drop]
            for extra in (0.0, 0.07, 0.15):
                keys = self._carry_keys(pc.eef_pos, release, extra)
                P, V, R = self.c.plan(keys, R_end, rot_end=0.9)
                t_pre = int(round(keys[-2][0] * 20))
                ign = lambda t, own, t_pre=t_pre: (own == "table" and (t < 14 or t >= t_pre - 2)) or \
                    ((own == sup or own in near0) and t < 14) or (own == recv and t >= t_pre - 2) or \
                    (kind == "in_drawer" and own in self.pc.e.fixtures_dict and t >= t_pre - 4) or \
                    (own == getattr(self, "_knock_m", None) and t >= t_pre - 14)
                dist, own = self._clear(P, R, moving, ign)
                if __import__("os").environ.get("DEBUG_CARRY"):
                    d_, t_, o_ = self.cc.path_clearance(P, R, moving, ignore=ign)
                    print(f"      CARRY goal={np.round(gl,3)} start={np.round(pc.eef_pos,3)} fixtures={[(f, np.round(lo[:2],2).tolist(), np.round(hi[:2],2).tolist()) for f, lo, hi in self.fixture_boxes]} gi={gi} extra={extra} dist={dist:.3f} own={own} raw={d_:.3f}@t{t_}/{len(P)} pair={getattr(self.cc, 'last_pair', None)}", flush=True)
                if dist >= want:
                    self.plan_log.append(("carry", "ok", gi, extra, round(dist, 3)))
                    return (P, V, R), gl, release
                sev = (getattr(self, "err", None) or {}).get("severe")
                if (getattr(self, "careless", False) or sev) and dist >= (-0.04 if sev else -0.012) and own in self.pc.names:
                    self.plan_log.append(("carry", "careless", gi, extra, round(dist, 3), own))
                    return (P, V, R), gl, release
                tried.append((dist, gi, (P, V, R), gl, release))
        for thr, tag in ((self.margin, "tight"), (-0.003, "graze")):
            ok = [x for x in tried if x[0] >= thr]
            if ok:
                best = min(ok, key=lambda x: (x[1], -x[0]))  # earliest (preferred) destination
                self.plan_log.append(("carry", tag, best[1], round(best[0], 3)))
                return best[2], best[3], best[4]
        if tried and max(x[0] for x in tried) >= -0.02:
            # still go somewhere new: take the path that comes closest to the
            # clearance threshold, even if it may bump something lightly
            # (never one that plows through an obstacle)
            best = max(tried, key=lambda x: x[0])
            self.plan_log.append(("carry", "bump", best[1], round(best[0], 3)))
            return best[2], best[3], best[4]
        # nothing works: lift a little and set it straight back down where it was
        back = np.r_[obj_g[:2], obj_g[2] + 0.002]
        release = back + rel + np.r_[0, 0, drop]
        keys = [(0.8, pc.eef_pos + np.r_[0, 0, 0.05]), (1.8, release + np.r_[0, 0, 0.02]), (2.2, release)]
        self.plan_log.append(("carry", "put-back"))
        self.put_back = True
        return self.c.plan(keys, R_end, rot_end=0.9), back, release

    def plan_retreat(self, n):
        pc = self.pc
        moving = self.cc.gripper_rel() + self.cc.forearm_rel()
        # after a grasp from the side / by a handle, or with the wrist pushed
        # far round (an object shoving the hand as it was set down): back out
        # the way the hand came in, then rise while turning the hand back to
        # point down - one smooth move, instead of going straight up with the
        # wrist still wound up (which ends in a long posture recovery)
        q = self.reach.current_q() if hasattr(self, "reach") else None
        lo0, hi0 = (self.reach._lo0, self.reach._hi0) if q is not None else (None, None)
        wound = q is not None and (q[4] < lo0[4] + 0.15 or q[4] > hi0[4] - 0.15 or q[5] > hi0[5] - 0.15 or q[5] < lo0[5] + 0.15)
        unusual = getattr(self, "chosen_grasp", ("",))[0] in ("side", "handle")
        if wound or unusual:
            R0 = pc.eef_mat
            app = R0[:, 2]
            cl = R0[:, 0] - np.r_[0, 0, R0[2, 0]]
            cl = cl / np.linalg.norm(cl) if np.linalg.norm(cl) > 1e-3 else np.r_[1.0, 0, 0]
            Rt = np.stack([cl, np.cross(np.r_[0, 0, -1.0], cl), np.r_[0, 0, -1.0]], 1)
            out = pc.eef_pos - app * self.rng.uniform(0.04, 0.06)
            top = out + np.r_[self.rng.normal(0, 0.01, 2), self.rng.uniform(0.06, 0.09)]
            P, V, R = self.c.plan([(0.5, out), (self.rng.uniform(1.4, 1.8), top)], Rt, rot_end=0.9)
            self.plan_log.append(("retreat", "unwind", "wound" if wound else "unusual"))
            self._unwinding = True
            return P, V, R
        # humans drift up ~2 cm (at most ~6 cm) in the second after letting go
        for up in (self.rng.uniform(0.02, 0.06), 0.08, 0.12):
            tgt = pc.eef_pos + np.r_[self.rng.normal(0, 0.01, 2), up]
            P, V, R = self.c.plan([(0.3, pc.eef_pos + np.r_[0, 0, 0.003]), (self.rng.uniform(1.0, 1.4), tgt)], pc.eef_mat, rot_end=0.5)
            ign = lambda t, own: (own in (n, "table") and t < 10)
            dist, own = self._clear(P, R, moving, ign)
            if dist >= self.margin * 0.7:
                return P, V, R
        return P, V, R

    def p_fumble(self, n):
        """Reach for an object a bit off, close on air or a corner, give up."""
        r = self._grasp(n, explore=True)
        if r is None:
            return None
        pc, c = self.pc, self.c
        c.move([(0.6, pc.eef_pos + np.r_[0, 0, 0.03])], pc.eef_mat, grip=1.0)
        c.hold(6, -1.0)
        c.move([(0.8, pc.eef_pos + np.r_[0, 0, 0.05])], pc.eef_mat, grip=-1.0)
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        return dict(primitive="pick_fumble", target=n, success=False, grasped=False, attempts=1,
                    language=f"try to pick up the {nice}")

    def p_reach(self, n):
        """Reach toward an object and change one's mind."""
        self.cur_targets = {n}
        # reach for wherever one would actually grab it (containers included)
        cands = self.grasp_candidates(n)
        if not cands:
            self.plan_log.append(("reach", "no grasp"))
            return None
        g, R = cands[0]
        keys = self._approach_keys(g)
        stop = keys[-3][0] if len(keys) >= 3 else keys[0][0]
        t_stop = stop * self.rng.uniform(0.6, 1.0)
        self.pc.phase = "approach"
        # follow the approach spline only up to t_stop, then drift away
        keys = [k for k in keys if k[0] <= t_stop] or [(t_stop, keys[0][1])]
        if not self.checked_move(keys, R, -1.0, ignore_last=0):
            return None
        # change of mind: a brief pause, then straight on to whatever is next
        # (no drawing back through free space)
        self.c.hold(int(self.rng.integers(3, 8)), -1.0, drift=np.r_[0, 0, 0.001])
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        return dict(primitive="reach_toward", target=n, success=True, language=f"reach toward the {nice}")

    def home_pose(self):
        """End-effector pose of the robot's initial joint configuration."""
        import mujoco
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        rc = self.reach
        rc.d2.qpos[:] = self.pc.d.qpos
        rc.d2.qpos[rc.qi] = rc.q_home
        mujoco.mj_kinematics(rc.m, rc.d2)
        return rc.d2.site_xpos[rc.site].copy(), rc.d2.site_xmat[rc.site].reshape(3, 3).copy()

    def recover_posture(self):
        self.c.guard = False
        try:
            self._recover_posture()
        finally:
            self.c.guard = True

    def _escape_posture(self, tries=3):
        """Look-ahead escape from a contorted arm posture (e.g. the flipped-wrist
        branch): simulate a few candidate re-orienting moves from a snapshot,
        keep the one that leaves the most joint room, and execute it."""
        from scipy.spatial.transform import Rotation as _Rot
        pc, c = self.pc, self.c
        for _ in range(tries):
            m0 = c.joint_margin()
            if m0 > 0.15:
                return
            R0, p0 = pc.eef_mat.copy(), pc.eef_pos.copy()
            hp, hR = self.home_pose()
            cands = []
            for ax in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
                cands.append(("tilt", p0 + [0, 0, 0.04], _Rot.from_rotvec(np.array(ax, float) * 0.5).as_matrix() @ R0))
            for a in (-0.9, 0.9):
                cands.append(("twist", p0 + [0, 0, 0.03], _Rot.from_rotvec(R0[:, 2] * a).as_matrix() @ R0))
            for yy in (0.0, np.pi):
                cands.append(("home", 0.5 * (p0 + hp) + [0, 0, 0.04], rotz(yy) @ hR))
            snap = self.pc.snapshot(); rec = pc.record_fn; corr = (getattr(c, "_corr", np.zeros(3)).copy(), getattr(c, "_rcorr", np.zeros(3)).copy())
            scores = []
            try:
                pc.record_fn = None
                for name, p, R in cands:
                    pc.restore(snap); c._corr, c._rcorr = corr
                    c.move([(1.0, p)], R, grip=-1.0, precise_end=False)
                    scores.append(c.joint_margin())
            finally:
                pc.restore(snap); c._corr, c._rcorr = corr
                pc.record_fn = rec
            k = int(np.argmax(scores))
            if scores[k] < m0 + 0.05:
                return
            name, p, R = cands[k]
            c.move([(1.0, p)], R, grip=-1.0, precise_end=False)
            self.plan_log.append(("escape", name, round(scores[k], 2)))

    def _unbend_elbow(self, tries=5):
        """At (nearly) full stretch the controller can't bend the elbow by
        pulling the hand in. Some motions do bend it (often down, sometimes
        sideways): try a few from a snapshot and repeat the best one."""
        pc, c = self.pc, self.c
        for _ in range(tries):
            q = self.reach.current_q()
            if q[3] < -0.6:
                return
            base = np.array(pc.robot.base_pos)
            p0, R0 = pc.eef_pos.copy(), pc.eef_mat.copy()
            inn = np.r_[(base[:2] - p0[:2]) / (np.linalg.norm(base[:2] - p0[:2]) + 1e-9), 0]
            side = np.cross([0, 0, 1.0], inn)
            cands = [np.r_[0, 0, -1.0], 1.5 * np.r_[0, 0, -1.0], (np.r_[0, 0, -1.0] + 0.4 * inn) / 1.08, side, -side, np.r_[0, 0, 1.0]]
            snap = pc.snapshot(); rec = pc.record_fn
            corr = (getattr(c, "_corr", np.zeros(3)).copy(), getattr(c, "_rcorr", np.zeros(3)).copy())
            scores = []
            try:
                pc.record_fn = None
                for dv in cands:
                    pc.restore(snap); c._corr, c._rcorr = corr
                    tgt = p0 + dv * 0.12
                    tgt[2] = max(tgt[2], pc.table_z + 0.07)
                    c.move([(1.0, tgt)], R0, grip=-1.0, precise_end=False)
                    scores.append(-float(self.reach.current_q()[3]))
            finally:
                pc.restore(snap); c._corr, c._rcorr = corr
                pc.record_fn = rec
            k = int(np.argmax(scores))
            if scores[k] < -q[3] + 0.02:
                return
            tgt = p0 + cands[k] * 0.12
            tgt[2] = max(tgt[2], pc.table_z + 0.07)
            c.move([(self.rng.uniform(0.9, 1.2), tgt)], R0, grip=-1.0, precise_end=False)
            self.plan_log.append(("unbend", k, round(float(self.reach.current_q()[3]), 2)))

    def _recover_posture(self):
        """Get out of an awkward posture the way a person would: one smooth
        1-2 s move that lifts the hand and draws it out a little (unfolding the
        elbow), the hand turning to point down. Only a genuinely flipped wrist
        gets the look-ahead escape first, and at most one try of it."""
        pc, c = self.pc, self.c
        pc.phase = "retreat"
        q = self.reach.current_q()
        stretched = q[3] > -0.7
        if q[3] > -0.5:
            self._unbend_elbow()
            q = self.reach.current_q()
            if q[3] > -0.4:
                # still locked out straight: draw the hand back toward the rest
                # pose (lower and nearer the body), slowly, the way one pulls an
                # outstretched arm back in
                hp, hR = self.home_pose()
                tgt = pc.eef_pos + 0.5 * (hp - pc.eef_pos) + np.r_[0, 0, -0.02]
                c.move([(self.rng.uniform(1.6, 2.0), tgt)], hR, grip=-1.0, precise_end=False)
                q = self.reach.current_q()
                self.plan_log.append(("recover", "drawn in", round(float(q[3]), 2)))
                if q[3] > -0.4:
                    return
        base = np.array(pc.robot.base_pos)[:2]
        p = pc.eef_pos.copy()
        out = p[:2] - base
        r = np.linalg.norm(out)
        out = out / (r + 1e-9)
        hp, hR = self.home_pose()
        # folded up close to the body: draw the hand out; stretched out: bring
        # it in (at full stretch only pulling in bends the elbow again)
        if stretched:
            r_new = float(np.clip(r - 0.15, 0.45, 0.58))
        elif q[3] < -2.6:
            r_new = float(np.clip(r + 0.06, 0.46, 0.6))
        else:
            r_new = float(np.clip(r, 0.46, 0.6))
        xy = base + out * r_new
        xy = 0.75 * xy + 0.25 * hp[:2] + self.rng.normal(0, 0.015, 2)
        # lifting a stretched arm straightens the elbow again: keep the height
        z = p[2] + (self.rng.uniform(-0.01, 0.02) if stretched else self.rng.uniform(0.05, 0.09))
        tgt = np.r_[xy, max(z, pc.table_z + (0.1 if stretched else 0.2))]
        # hand pointing down, keeping roughly the yaw it has (no wrist spin),
        # unless the last joint is close to its stop: then turn it back some
        cur_y = pc.eef_yaw()
        q7 = float(q[6])
        y = cur_y
        if abs(q7) > 2.3:
            y = wrap(cur_y + np.sign(q7) * min(1.0, abs(q7) - 1.5))   # dq7 ~ -dyaw
        R = rotz(y) @ R_DOWN
        T = float(self.rng.uniform(1.2, 1.8))
        flipped = q[4] < -1.6 or q[4] > 2.5 or q[5] > 3.6
        if flipped:
            # getting the wrist off its flipped branch has to pass through
            # worse joint angles first: try the same plain move with a few
            # final hand yaws in simulation and do the one that ends best
            hy = float(np.arctan2(hR[1, 0], hR[0, 0]))
            yaws = [y, wrap(y + np.pi / 2), wrap(y - np.pi / 2), wrap(hy), wrap(hy + np.pi)]
            snap = pc.snapshot(); rec = pc.record_fn
            corr = (getattr(c, "_corr", np.zeros(3)).copy(), getattr(c, "_rcorr", np.zeros(3)).copy())
            scores = []
            try:
                pc.record_fn = None
                c.guard = False
                for yy in yaws:
                    pc.restore(snap); c._corr, c._rcorr = corr
                    c.move([(T, tgt)], rotz(yy) @ R_DOWN, grip=-1.0, precise_end=False)
                    scores.append(self.reach.margin(self.reach.current_q()))
            finally:
                pc.restore(snap); c._corr, c._rcorr = corr
                pc.record_fn = rec
                c.guard = True
            k = int(np.argmax(scores))
            g_was, c.guard = c.guard, False
            try:
                c.move([(T, tgt)], rotz(yaws[k]) @ R_DOWN, grip=-1.0, precise_end=False)
            finally:
                c.guard = g_was
        else:
            # the plain move runs with the joint guard on: it may only make things better
            g_was, c.guard = c.guard, True
            try:
                if not self.checked_move([(T, tgt)], R, -1.0, ignore_last=0, min_clear=-0.005):
                    c.move([(T, tgt)], R, grip=-1.0, precise_end=False)
            finally:
                c.guard = g_was
        # only a wrist still on its flipped branch after that gets the
        # look-ahead escape (it can look odd, so it's the last resort)
        q = self.reach.current_q()
        if q[4] < -1.6 or q[4] > 2.5 or q[5] > 3.6:
            self._escape_posture(tries=1)   # last resort
        self.plan_log.append(("recover", round(self.reach.margin(self.reach.current_q()), 2)))

    def checked_move(self, keys, R, grip, ignore_owner=None, ignore_last=12, v_end=None, moving=None,
                     ignore_first=None, min_clear=None, exact=False, **kw):
        """Execute the first collision-free variant of a keyframe move: as given,
        with raised intermediate keys, or up-over-down. Returns False if none."""
        pc, c = self.pc, self.c
        moving = moving if moving is not None else (self.cc.gripper_rel() + self.cc.forearm_rel())
        end = keys[-1][1]
        variants = [keys]
        for extra in (0.06, 0.12):
            variants.append([(t, p + np.r_[0, 0, extra]) for t, p in keys[:-1]] + [keys[-1]])
        p0 = pc.eef_pos
        high = max(p0[2], end[2]) + 0.12
        T = keys[-1][0] + 1.0
        variants.append([(0.3 * T, np.r_[p0[:2], high]), (0.75 * T, np.r_[end[:2], high]), (T, end)])
        if ignore_first is not None:
            # whatever the hand/held object already touches at the start (a
            # neighbour the object leans on) is ignored while moving off it
            own0 = set(ignore_first[0])
            for _ in range(3):
                ig0 = lambda t, own: own in own0
                d0, _, o0 = self.cc.path_clearance(np.array([pc.eef_pos]), np.array([pc.eef_mat]), moving, ignore=ig0)
                if d0 >= self.margin or o0 is None or o0 in own0:
                    break
                own0.add(o0)
            ignore_first = (tuple(own0), ignore_first[1])
            if __import__("os").environ.get("DEBUG_MOVE"):
                print("      IGNORE_FIRST", ignore_first, "d0 %.3f" % d0, o0, flush=True)
        for keys_ in variants:
            P, V, Rp = c.plan(keys_, R, v_end=v_end, exact=exact)
            n = len(P)
            ign = lambda t, own: (own == ignore_owner and t >= n - ignore_last) or \
                (ignore_first is not None and own in ignore_first[0] and t < ignore_first[1])
            dist, own = self._clear(P, Rp, moving, ign)
            if ignore_first is not None and dist < self.margin:
                # starting in (near) contact: fine as long as it never gets closer
                d0, _, _ = self.cc.path_clearance(P[:1], Rp[:1], moving, ignore=ign)
                if d0 < self.margin and dist >= d0 - 0.002:
                    dist = self.margin
            if __import__("os").environ.get("DEBUG_MOVE"):
                print(f"      CHECKED_MOVE dist {dist:.3f} own {own} pair {getattr(self.cc, 'last_pair', None)} n {n}", flush=True)
            if dist >= (self.margin if min_clear is None else min_clear):
                c.follow(P, V, Rp, grip, **kw)
                return True
        self.plan_log.append(("move", "blocked", own))
        return False

    def _drawer_frame(self, dr):
        d = self.pc.d
        R = d.xmat[dr["bid"]].reshape(3, 3)
        out = -R[:, 1]; out[2] = 0; out /= np.linalg.norm(out)
        return d.geom_xpos[dr["handle"]].copy(), out

    def p_drawer(self, dr, want_open, explore=False, params=None):
        pc, c = self.pc, self.c
        self.cur_targets = {dr["fixture"]}
        adr = pc.m.jnt_qposadr[dr["jid"]]
        q0 = pc.d.qpos[adr]
        hp, out = self._drawer_frame(dr)
        lv = dr["level"]
        if want_open:
            # open gripper, closing axis along the pull direction, one finger hooked behind the bar
            P_ = params or {}
            strat = P_.get("strat", "down" if lv == "top" else "pinch")
            if strat == "down":
                R = rotz(self._yaw_for_axis(out[:2])) @ R_DOWN @ small_tilt(self.rng, 4.0)
            else:
                # human lower-drawer pose (measured in the handle frame from 100 demos):
                # gripper nearly horizontal, pointing into the cabinet ~28 deg down,
                # fingers spread sideways around the bar
                import json, os
                TP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawer_tilt_pose.json")))
                tp = TP["bottom" if lv == "bottom" else "middle"]
                lat = np.cross([0, 0, 1], out)
                F = np.stack([out, lat, [0, 0, 1]], 1)
                R = F @ np.array(tp["R"]) @ small_tilt(self.rng, 4.0)
                if strat == "pinch":
                    # pinch the bar: gripper points into the cabinet, tilted down a
                    # little like the human pose, fingers closing vertically
                    from scipy.spatial.transform import Rotation as _Rot
                    tilt = P_.get("tilt", np.radians(self.rng.uniform(10, 25)))
                    app = -out * np.cos(tilt) - np.r_[0, 0, 1] * np.sin(tilt)
                    cl = np.cross(lat, app); cl /= np.linalg.norm(cl)
                    R = np.stack([cl, np.cross(app, cl), app], 1) @ small_tilt(self.rng, 3.0)
            # rear fingertip (4.3-5.2 cm behind the hand center when open) must drop
            # into the 1.6 cm gap between the bar and the drawer face
            e_ = getattr(self, "err", None) or {}
            if strat == "down":
                hook = hp + out * P_.get("out", self.rng.uniform(0.04, 0.048)) + np.r_[0, 0, P_.get("z", self.rng.uniform(-0.013, -0.008))]
            elif strat == "pinch":
                hook = hp + out * P_.get("out", self.rng.uniform(-0.004, 0.004)) + np.r_[0, 0, P_.get("z", self.rng.uniform(-0.004, 0.003))]
            else:
                hook = hp + out * P_.get("out", 0.0) + np.r_[0, 0, P_.get("z", 0.025)]
            if explore:
                hook[:2] += self.rng.normal(0, 0.01, 2)
            if _has(e_, "drawer_miss"):
                # injected error: the finger lands off the handle
                hook = hook + out * e_["offset"][0] + np.r_[0, 0, e_["offset"][1]]
            # how far can the drawer come out without pushing anything?
            pull = self.rng.uniform(0.11, 0.17)
            dgeoms = [g for g in range(pc.m.ngeom) if pc.m.geom_bodyid[g] == dr["bid"] and pc.m.geom_contype[g]]
            for k in np.linspace(pull, 0.05, 7):
                dist_, own_ = self.cc.swept_clearance(dgeoms, [out * x for x in np.linspace(0, k, 6)], dr["fixture"])
                if dist_ >= 0.01:
                    pull = k
                    break
            else:
                self.plan_log.append(("drawer", "sweep blocked", own_))
                return None
            # the hook pose and the end of the pull must be comfortably reachable
            # with this orientation, otherwise the arm sags onto the handle
            if not hasattr(self, "reach"):
                from reach import Reach
                self.reach = Reach(self.pc)
            Rok = None
            for Rc in (R, R @ np.diag([-1.0, -1.0, 1.0])):
                if self.reach.wrist_dist(hook, Rc)[0] > self.reach.WRIST_MAX:
                    continue
                ep_, er_, q_ = self.reach.solve(hook, Rc, q0=self.reach.current_q())
                if ep_ > 0.006 or er_ > 0.06 or self.reach.margin(q_) < 0.1 or q_[3] > -0.6:
                    continue    # (elbow clearly bent: near full stretch the arm straightens on contact)
                while pull >= 0.08:
                    pe = hook + out * pull
                    ep2, er2, q2 = self.reach.solve(pe, Rc, q0=q_)
                    if ep2 < 0.006 and er2 < 0.06 and self.reach.margin(q2) >= 0.1 and q2[3] <= -0.6 and \
                            self.reach.wrist_dist(pe, Rc)[0] <= self.reach.WRIST_MAX + 0.02:
                        break
                    pull -= 0.02
                if pull >= 0.08:
                    Rok = Rc
                    break
            if Rok is None:
                self.plan_log.append(("drawer", "unreachable", dr["fixture"]))
                return None
            R = Rok
            if __import__("os").environ.get("DEBUG_DRW"):
                _, _, qh = self.reach.solve(hook, R, q0=self.reach.current_q())
                print(f"      DRAWER {dr['fixture']}:{dr['level']} want_open={want_open} strat={strat} hook={np.round(hook, 3)} pull={pull:.3f} q_hook={np.round(qh, 2)} t={pc.t}", flush=True)
            # pre-hook pose: back off along the gripper's approach axis
            above = hook - R[:, 2] * self.rng.uniform(0.04, 0.055)
            dist = np.linalg.norm(above - pc.eef_pos)
            T = float(np.clip(1.74 + 4.02 * dist, 1.5, 4.5) * self.rng.uniform(0.85, 1.15))
            keys = [(T - 0.9, above + np.r_[self.rng.normal(0, 0.012, 2), self.rng.uniform(0.04, 0.08)]), (T, above)]
            pc.phase = "approach"
            if not self.checked_move(keys, R, -1.0, ignore_owner=dr["fixture"], exact=True):
                return None
            pc.phase = "correct"
            c.settle(above, R, -1.0, tol=0.003, max_steps=15)
            pc.phase = "approach"
            c.move([(self.rng.uniform(0.6, 0.9), hook)], R, grip=-1.0, residual=0.0)
            pc.phase = "articulate"
            # top drawers: finger hooked behind the bar; lower drawers: press on the
            # bar and pull (keep a little downward force for friction)
            dz = {"down": 0.004, "pinch": 0.003}.get(strat, P_.get("press", -0.012))
            if strat == "pinch":
                # close on the bar first; start pulling as the fingers bite
                pc.phase = "grasp"
                c.hold(int(self.rng.integers(7, 10)), 1.0)
                pc.phase = "articulate"
            c.move([(self.rng.uniform(1.2, 2.0), hook + out * pull + np.r_[0, 0, dz])], R,
                   grip=1.0 if strat == "pinch" else -1.0, residual=0.4, precise_end=False)
            if strat == "pinch":
                c.hold(int(self.rng.integers(5, 8)), -1.0)
        else:
            R = rotz(self._yaw_for_axis(np.cross([0, 0, 1], out)[:2])) @ R_DOWN @ small_tilt(self.rng, 5.0)
            front = hp + out * 0.05 + np.r_[0, 0, 0.005]
            dist = np.linalg.norm(front - pc.eef_pos)
            T = float(np.clip(1.74 + 4.02 * dist, 1.5, 4.5) * self.rng.uniform(0.85, 1.15))
            keys = [(T - 1.0, front + np.r_[0, 0, self.rng.uniform(0.08, 0.12)]), (T - 0.4, front + np.r_[0, 0, 0.035]), (T, front)]
            pc.phase = "approach"
            if not self.checked_move(keys, R, 1.0, ignore_owner=dr["fixture"], exact=True, v_end=np.r_[0, 0, -0.02]):
                return None
            push = hp - out * (abs(q0) + 0.025)
            pc.phase = "articulate"
            c.move([(self.rng.uniform(1.0, 1.6), push + np.r_[0, 0, 0.005])], R, grip=1.0, precise_end=False)
        pc.phase = "retreat"
        if not self.checked_move([(0.9, pc.eef_pos + out * 0.04 + np.r_[0, 0, self.rng.uniform(0.06, 0.1)])], pc.eef_mat, -1.0,
                                 ignore_owner=dr["fixture"], ignore_last=0):
            c.move([(0.9, pc.eef_pos + np.r_[0, 0, 0.08])], pc.eef_mat, grip=-1.0)
        dq = pc.d.qpos[adr] - q0
        ok = dq < -0.06 if want_open else dq > 0.04
        return dict(primitive=f"drawer_{'open' if want_open else 'close'}", target=f"{lv} drawer",
                    success=bool(ok), language=f"{'open' if want_open else 'close'} the {lv} drawer of the cabinet",
                    explore=bool(explore))

    # ---------------------------------------------------------------- driver
    def choose(self):
        pc = self.pc
        explore = self.rng.random() < self.p_explore
        opts = []
        movable = [m for m in pc.names if GROUP_OF.get(self.cat(m)) not in (None, "basket") and self.upright(m)]
        last = self.segments[-1] if self.segments else None
        if movable:
            if last is not None and last["primitive"].startswith("pick") and not last.get("grasped", True):
                tgt = last["target"].split("->")[0]
                if tgt in movable and self.rng.random() < 0.6:
                    return lambda: self.p_pick_place(tgt)
            recent = [x.get("target", "").split("->")[0] for x in self.segments[-4:]]
            w = np.array([1.0 / (1 + 2 * recent.count(m)) for m in movable])
            n = movable[self.rng.choice(len(movable), p=w / w.sum())]
            opts.append((0.6, lambda: self.p_pick_place(n, explore=explore)))
            opts.append((0.06, lambda: self.p_reach(movable[self.rng.integers(len(movable))])))
            opts.append((0.04, lambda: self.p_fumble(movable[self.rng.integers(len(movable))])))
        if pc.drawers:
            # top drawers are reliable; lower ones are tried less often
            wd = np.array([1.0 if d["level"] == "top" else 0.3 for d in pc.drawers])
            dr = pc.drawers[self.rng.choice(len(pc.drawers), p=wd / wd.sum())]
            is_open = pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.04
            opts.append((0.3, lambda: self.p_drawer(dr, not is_open, explore=explore)))
        if not opts:
            return None
        w = np.array([o[0] for o in opts])
        return opts[self.rng.choice(len(opts), p=w / w.sum())][1]

    def run_episode(self, n_primitives):
        fails = 0
        fails_since_ok, recovered, last_motion_t = 0, 0, self.pc.t
        while len(self.segments) < n_primitives:
            # already at a joint stop (an action ended there): sort the posture
            # out first, rather than letting each new attempt nudge it further
            if hasattr(self, "reach") and self.reach.margin(self.reach.current_q()) < 0.03 and recovered < 1 and \
                    not (self.pc.grip > 0 and any(self.pc.is_grasped(m) for m in self.pc.names)):
                try:
                    self.recover_posture()
                    recovered += 1
                except EpisodeTooLong:
                    break
            fn = self.choose()
            if fn is None:
                break
            t0 = self.pc.t
            before = {m: self.obj_pose(m) for m in self.pc.names}
            before_sup = {m: (self.support_of(m),) for m in self.pc.names}
            self.contact_log = []
            try:
                info = fn()
            except EpisodeTooLong:
                if __import__("os").environ.get("DEBUG_END"):
                    print(f"   EPISODE END: too long, t={self.pc.t} segments={len(self.segments)}", flush=True)
                # the action under way when time ran out: label it too, so the
                # labels cover the whole episode
                kind, tgt = getattr(self, "last_task", ("?", ""))
                tgt = tgt if isinstance(tgt, str) else ""
                if self.pc.t - t0 > 5:
                    what = kind.replace("_", " ") + (f" with the {tgt.rsplit('_', 1)[0].replace('_', ' ')}" if tgt else "")
                    err = getattr(self, "err", None)
                    self.segments.append(dict(primitive=kind, target=tgt, success=False, truncated=True,
                                              language=f"{what} (cut off by the end of the episode)",
                                              start=t0, end=self.pc.t, disturbed=[], contacts=[],
                                              error=dict(err) if err else None))
                break
            if info is None and self.pc.t - t0 > 10:
                # the arm moved and then the action was given up (blocked, a
                # joint stop): still an action, label it as a failed one
                kind, tgt = getattr(self, "last_task", ("?", ""))
                tgt = tgt if isinstance(tgt, str) else ""
                what = kind.replace("_", " ") + (f" the {tgt.rsplit('_', 1)[0].replace('_', ' ')}" if tgt else "")
                info = dict(primitive=kind, target=tgt, success=False, abandoned=True,
                            language=f"start to {what}, then give up", diag=dict(plan=[str(x) for x in self.plan_log[-3:]]))
                if getattr(self, "_last_err", None):
                    info["error"] = self._last_err
            if info is None:
                fails += 1
                streak = self.pc.t - last_motion_t
                if fails_since_ok > 150:
                    if __import__("os").environ.get("DEBUG_END"):
                        import collections as _c
                        print(f"   EPISODE END: {fails_since_ok} failures in a row at t={self.pc.t}; margin {self.reach.margin(self.reach.current_q()):.2f} q {np.round(self.reach.current_q(), 2)}; last plan_logs {self.plan_log[-4:]}", flush=True)
                    break   # nothing in the scene can be done any more
                fails_since_ok += 1
                if fails_since_ok % 20 == 0:
                    recovered = 0   # still stuck: allow fresh attempts to get out of the posture
                if not hasattr(self, "reach"):
                    from reach import Reach
                    self.reach = Reach(self.pc)
                jl = any("joint-limit" in str(x) for x in self.plan_log[-3:])
                # only move when the posture is really the problem (never wander
                # about just because plans failed)
                if (self.reach.margin(self.reach.current_q()) < 0.12 or (jl and self.reach.margin(self.reach.current_q()) < 0.2)) and recovered < 1:
                    try:
                        self.recover_posture()
                        recovered += 1
                        last_motion_t = self.pc.t
                    except EpisodeTooLong:
                        break
                continue
            fails_since_ok = 0
            recovered = 0
            # finishing in an awkward posture: drift back toward a comfortable one
            # as part of this action's retreat, instead of failing the next plan
            # and pausing to recover then
            if not hasattr(self, "reach"):
                from reach import Reach
                self.reach = Reach(self.pc)
            holding = self.pc.grip > 0 and any(self.pc.is_grasped(m) for m in self.pc.names)
            if self.reach.margin(self.reach.current_q()) < 0.1 and not holding:
                try:
                    self.recover_posture()
                    info["posture_reset"] = True
                except EpisodeTooLong:
                    pass
            last_motion_t = self.pc.t
            info.update(start=t0, end=self.pc.t)
            # collateral disturbance: objects other than the target that moved
            tgt = info.get("target", "").split("->") + list(info.get("interact", []))
            # whatever sat in/on the handled object moves along with it: not collateral
            tgt = tgt + [m for m in self.pc.names if m not in tgt and any(
                b_[0] is not None and b_[0] == t_ for t_ in tgt for b_ in [before_sup.get(m, (None,))])]
            moved = []
            for m, (p0, q0) in before.items():
                if m in tgt:
                    continue
                p1, q1 = self.obj_pose(m)
                ang = 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(q0, q1))))))
                if np.linalg.norm(p1 - p0) > 0.01 or ang > 5:
                    moved.append(m)
            info["disturbed"] = moved
            import collections as _c
            info["contacts"] = [list(k) + [v] for k, v in _c.Counter((p, b, o) for _, p, b, o in self.contact_log).items()]
            self.segments.append(info)
        return self.segments
