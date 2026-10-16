"""Playing with the scene's fixtures: turning the stove knob, opening and
closing the microwave door, and the surfaces things can be put on (burner,
microwave top, cabinet top, inside an open drawer).

Everything is computed from the fixture's own geometry and joints at the
current state (knob fin, door handle and hinge, site regions), and every move
is collision-checked like the rest of the play. Mixin for PlayfulPlay."""
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from compose import R_DOWN, rotz, wrap
from composed_play import small_tilt
from human_play import GEOM, bottom_offset


def _nice(n):
    return n.rsplit("_", 1)[0].replace("_", " ")


class FixturePlay:
    # ------------------------------------------------------------ lookup
    def _ensure_reach(self):
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        return self.reach

    def fixture_named(self, kind):
        for f in self.pc.e.fixtures_dict:
            if kind in f:
                return f
        return None

    def _joint_q(self, jname):
        m, d = self.pc.m, self.pc.d
        try:
            j = m.joint(jname).id
        except KeyError:
            return None, None
        return j, float(d.qpos[m.jnt_qposadr[j]])

    def _fixture_geoms(self, f):
        m = self.pc.m
        root = m.body(self.pc.e.fixtures_dict[f].root_body).id
        return [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root
                and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def _box_extent(self, g):
        m, d = self.pc.m, self.pc.d
        Rg = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == 6:
            return np.abs(Rg) @ m.geom_size[g]
        return np.full(3, m.geom_rbound[g])

    # ------------------------------------------------------------ stove
    def stove_on(self):
        f = self.fixture_named("stove")
        if f is None:
            return None
        _, q = self._joint_q(f"{f}_button")
        return None if q is None else q > 0.4

    def p_stove_knob(self, turn_on):
        """Grip the knob's fin from above and twist it, as people do."""
        pc, c, m, d = self.pc, self.c, self.pc.m, self.pc.d
        self._ensure_reach()
        f = self.fixture_named("stove")
        if f is None:
            return None
        jid, q0 = self._joint_q(f"{f}_button")
        bid = m.jnt_bodyid[jid]
        axis_xy = d.xpos[bid][:2].copy()
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == 6]
        base_z = min(d.geom_xpos[g][2] - self._box_extent(g)[2] for g in geoms)
        # the fin: the biggest raised box on the knob
        raised = [g for g in geoms if d.geom_xpos[g][2] > base_z + 0.015]
        fin = max(raised, key=lambda g: np.prod(m.geom_size[g]))
        Rg = d.geom_xmat[fin].reshape(3, 3)
        ext = np.abs(Rg) @ m.geom_size[fin]
        top = d.geom_xpos[fin][2] + ext[2]
        hz = [i for i in range(3) if abs(Rg[2, i]) < 0.7]
        long_i = max(hz, key=lambda i: m.geom_size[fin][i])
        along = Rg[:, long_i].copy(); along[2] = 0; along /= np.linalg.norm(along)
        close = np.cross([0, 0, 1.0], along)
        pos = np.r_[d.geom_xpos[fin][:2] + along[:2] * self.rng.uniform(-0.008, 0.008), top - self.rng.uniform(0.009, 0.015)]
        if turn_on:
            dq = self.rng.uniform(0.8, 1.4)
            dq = min(dq, 2.0 - q0)
        else:
            dq = -(q0 + self.rng.uniform(0.0, 0.03))
        if abs(dq) < 0.2:
            return None
        # of the two equivalent hand yaws, the one leaving the last joint room for the twist
        yaw_c = [wrap(np.arctan2(close[1], close[0]) - np.pi / 2), wrap(np.arctan2(close[1], close[0]) + np.pi / 2)]
        q7 = float(pc.d.qpos[self.reach.qi[6]]) if hasattr(self, "reach") else 0.0
        cur = pc.eef_yaw()
        def room(y):
            q_after = q7 - wrap(y - cur) - dq
            return 2.8973 - max(abs(q7 - wrap(y - cur)), abs(q_after))
        y = max(yaw_c, key=room)
        R_flat = rotz(y) @ R_DOWN @ small_tilt(self.rng, 4.0)
        # close to the robot the hand leans back (rotated about the fin's
        # direction, so the fingers still straddle it); keep the first lean
        # whose grasp and end-of-twist poses the arm can reach comfortably
        base = np.array(pc.robot.base_pos)[:2]
        away = pos[:2] - base; away /= np.linalg.norm(away)
        R = None
        for lean in (0.0, 0.3, 0.5, 0.7):
            for sgn in (1, -1):
                Rl = _Rot.from_rotvec(along * sgn * lean).as_matrix() @ R_flat
                if lean > 0 and Rl[:2, 2] @ away < 0:
                    continue  # lean the hand back (fingers toward the far side)
                ok = True
                for x in (0.0, 1.0):
                    pp = np.r_[axis_xy + rotz(dq * x)[:2, :2] @ (pos[:2] - axis_xy), pos[2]]
                    ep, er, qq = self.reach.solve(pp, rotz(dq * x) @ Rl, q0=self.reach.current_q())
                    if ep > 0.006 or er > 0.06 or self.reach.margin(qq) < 0.12:
                        ok = False
                        break
                if ok:
                    R = Rl
                    break
            if R is not None:
                break
        if R is None:
            self.plan_log.append(("stove", "knob unreachable"))
            return None
        self.cur_targets = {f}
        pc.phase = "approach"
        dist = np.linalg.norm(pos - pc.eef_pos)
        T = float(np.clip(1.74 + 4.02 * dist, 1.5, 4.0) * self.rng.uniform(0.9, 1.1))
        keys = [(T - 0.9, pos + np.r_[self.rng.normal(0, 0.01, 2), self.rng.uniform(0.07, 0.1)]),
                (T - 0.35, pos + np.r_[0, 0, 0.025]), (T, pos)]
        if not self.checked_move(keys, R, -1.0, ignore_owner=f, ignore_last=14, exact=True, v_end=np.r_[0, 0, -0.01]):
            return None
        pc.phase = "correct"
        c.settle(pos, R, -1.0, tol=0.004, max_steps=10)
        pc.phase = "grasp"
        c.hold(int(self.rng.integers(7, 10)), 1.0)
        # twist about the knob's own axis (the hand orbits it slightly)
        pc.phase = "articulate"
        n = 24 + int(abs(dq) * 18)
        s = np.linspace(0, 1, n)
        s = 3 * s ** 2 - 2 * s ** 3
        p0 = pc.eef_pos.copy(); R0 = pc.eef_mat.copy()
        P = np.array([np.r_[axis_xy + rotz(dq * x)[:2, :2] @ (p0[:2] - axis_xy), p0[2] - 0.002] for x in s])
        Rp = np.array([rotz(dq * x) @ R0 for x in s])
        V = np.gradient(P, axis=0)
        c.follow(P, V, Rp, 1.0, residual=0.0, wobble=0.3, precise_end=False)
        c.hold(int(self.rng.integers(2, 5)), 1.0)
        pc.phase = "release"
        c.hold(int(self.rng.integers(5, 8)), -1.0)
        pc.phase = "retreat"
        if not self.checked_move([(self.rng.uniform(0.8, 1.1), pc.eef_pos + np.r_[self.rng.normal(0, 0.01, 2), self.rng.uniform(0.06, 0.1)])],
                                 pc.eef_mat, -1.0, ignore_owner=f, ignore_last=0):
            c.move([(0.9, pc.eef_pos + np.r_[0, 0, 0.08])], pc.eef_mat, grip=-1.0)
        _, q1 = self._joint_q(f"{f}_button")
        ok = q1 > 0.4 if turn_on else q1 < 0.15
        return dict(primitive="stove_on" if turn_on else "stove_off", target=f, success=bool(ok),
                    language=f"turn {'on' if turn_on else 'off'} the stove", diag=dict(q0=q0, q1=q1))

    # ------------------------------------------------------------ microwave
    def microwave_q(self):
        f = self.fixture_named("microwave")
        if f is None:
            return None
        return self._joint_q(f"{f}_microjoint")[1]

    def _microwave_push_shut(self, f, jid, q0, door, hinge, n_out_now, panel):
        """Close the door the way people do: an open hand on the outside of the
        door, swept along the arc until it shuts. The contact point is picked
        along the door where the whole sweep is reachable."""
        pc, c, m, d = self.pc, self.c, self.pc.m, self.pc.d
        pg = d.geom_xpos[panel]
        L_vec = pg[:2] - hinge
        L = 2 * np.linalg.norm(L_vec)          # door length (panel centre is mid-door)
        u = L_vec / np.linalg.norm(L_vec)
        z_door_lo = pg[2] - self._box_extent(panel)[2]
        z = pg[2] + self.rng.uniform(-0.01, 0.03)
        dq = -q0 + self.rng.uniform(0.0, 0.02)
        side = n_out_now
        half_th = m.geom_size[panel][1] if m.geom_size[panel][1] < m.geom_size[panel][0] else m.geom_size[panel][0]
        for frac in list(self.rng.permutation([0.55, 0.7, 0.85])) + [0.95]:
            r = frac * L
            contact = hinge + u * r + side * (half_th + 0.012)
            y = np.arctan2(u[1], u[0]) - np.pi / 2   # fingers in a line along the door
            R = rotz(y) @ R_DOWN @ small_tilt(self.rng, 3.0)
            qe, ok = self.reach.current_q(), True
            for x in np.linspace(0, 1, 7):
                pe = np.r_[hinge + rotz(dq * x)[:2, :2] @ (contact - hinge), z]
                best = None
                for Rc in (R, R @ np.diag([-1.0, -1.0, 1.0])):
                    ep, er, q_ = self.reach.solve(pe, rotz(dq * x) @ Rc, q0=qe, iters=60 if x == 0 else 40)
                    if ep < 0.008 and er < 0.08 and self.reach.margin(q_) > 0.08:
                        best = (Rc, q_)
                        break
                if best is None:
                    if __import__("os").environ.get("DEBUG_FX"):
                        print(f"   PUSH frac={frac} fail at x={x:.2f} pe={np.round(pe,3)} ep={ep:.3f} er={er:.2f} m={self.reach.margin(q_):.2f} wrist={self.reach.wrist_dist(pe, rotz(dq * x) @ R)[0]:.3f} shoulder={np.round(self.reach.shoulder(),3)}", flush=True)
                    ok = False
                    break
                if x == 0:
                    R = best[0]
                qe = best[1]
            if ok:
                break
        else:
            self.plan_log.append(("microwave", "can't reach to push shut"))
            return None
        self.cur_targets = {f}
        pc.phase = "approach"
        start = np.r_[contact + side * 0.03, z]
        dist = np.linalg.norm(start - pc.eef_pos)
        T = float(np.clip(1.74 + 4.02 * dist, 1.5, 4.0) * self.rng.uniform(0.9, 1.1))
        wm = self.reach.WRIST_MAX
        self.reach.WRIST_MAX = 0.76
        try:
            ok = self.checked_move([(T - 0.8, start + np.r_[0, 0, 0.08]), (T, start)], R, -1.0, ignore_owner=f, ignore_last=8)
        finally:
            self.reach.WRIST_MAX = wm
        if not ok:
            return None
        pc.phase = "articulate"
        n = int(30 + abs(dq) * 28 * self.rng.uniform(0.9, 1.2))
        s = np.linspace(0, 1, n); s = 3 * s ** 2 - 2 * s ** 3
        p0 = np.r_[contact - side * 0.008, z]
        R0 = pc.eef_mat.copy()
        P = np.array([np.r_[hinge + rotz(dq * x)[:2, :2] @ (p0[:2] - hinge), z] for x in s])
        P = np.vstack([np.linspace(pc.eef_pos, P[0], 8), P])
        Rp = np.concatenate([np.repeat(R0[None], 8, 0), np.array([rotz(dq * x) @ R0 for x in s])])
        adr = m.jnt_qposadr[jid]
        # stop when it is shut - or when the hand is held up (door stuck on
        # something) and falling behind the plan: pressing on would make the
        # hand jump over the door and fling the arm out straight
        Pf = np.vstack([np.repeat(P[:1], 8, 0), P]) if len(Rp) == len(P) + 8 else P

        def stop_push(t):
            if d.qpos[adr] > -0.03:
                return True
            k = min(t, len(Pf) - 1)
            return t > 10 and np.linalg.norm(Pf[k][:2] - pc.eef_pos[:2]) > 0.035
        c.follow(P, np.gradient(P, axis=0), Rp, -1.0, residual=0.0, wobble=0.3, precise_end=False,
                 stop_fn=stop_push)
        pc.phase = "retreat"
        _, q1 = self._joint_q(f"{f}_microjoint")
        if not self.checked_move([(0.9, pc.eef_pos + np.r_[side * 0.04, 0.07])], pc.eef_mat, -1.0, ignore_owner=f, ignore_last=0):
            c.move([(0.9, pc.eef_pos + np.r_[0, 0, 0.08])], pc.eef_mat, grip=-1.0)
        return dict(primitive="microwave_close", target=f, success=bool(q1 > -0.15),
                    language="close the microwave", diag=dict(q0=q0, q1=q1, how="push"))

    def p_microwave(self, want_open, q_target=None):
        """Take the door handle and swing the door open (or shut) along its arc."""
        pc, c, m, d = self.pc, self.c, self.pc.m, self.pc.d
        self._ensure_reach()
        f = self.fixture_named("microwave")
        if f is None:
            return None
        jid, q0 = self._joint_q(f"{f}_microjoint")
        lo, hi = m.jnt_range[jid]
        door = m.jnt_bodyid[jid]
        hinge = d.xpos[door][:2].copy()
        dg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == door and (m.geom_contype[g] or m.geom_conaffinity[g])]
        caps = [g for g in dg if m.geom_type[g] == 3]
        handle = max(caps, key=lambda g: m.geom_size[g][1])
        panel = max([g for g in dg if m.geom_type[g] == 6], key=lambda g: np.prod(m.geom_size[g]))
        h = d.geom_xpos[handle].copy()
        half = m.geom_size[handle][1]
        n_out = h[:2] - d.geom_xpos[panel][:2]
        n_out = n_out - (n_out @ (h[:2] - hinge)) * (h[:2] - hinge) / np.linalg.norm(h[:2] - hinge) ** 2
        n_out /= np.linalg.norm(n_out) + 1e-9
        tang = np.r_[-n_out[1], n_out[0]]
        if want_open:
            q_end = max(lo + 0.1, -self.rng.uniform(0.9, 1.4) if q_target is None else q_target)
        else:
            q_end = min(hi, 0.0)
        dq = q_end - q0
        if abs(dq) < 0.25:
            return None
        if not want_open and self.rng.random() < 0.75:
            r = self._microwave_push_shut(f, jid, q0, door, hinge, n_out, panel)
            if r is not None or self.plan_log[-1:] != [("microwave", "can't reach to push shut")]:
                return r
        fgeoms = [g for g in self._fixture_geoms(f) if g != handle]
        # grasp the handle bar from above, fingers either side of it along the
        # door; try a few heights and both hand yaws, keep what clears the door
        cands = []
        for dz, lean in ((0.047, 0.35), (0.04, 0.5), (0.047, 0.2), (0.03, 0.65), (0.047, 0.0)):
            pos = np.r_[h[:2] + n_out * self.rng.uniform(-0.002, 0.002), h[2] + dz]
            for sgn in (1, -1):
                y = np.arctan2(sgn * tang[1], sgn * tang[0]) - np.pi / 2
                R = rotz(y) @ R_DOWN @ small_tilt(self.rng, 2.0)
                # lean the palm out, away from the door (pads stay on the bar)
                for ls in (1, -1):
                    Rl = _Rot.from_rotvec(np.r_[tang, 0] * ls * lean).as_matrix() @ R
                    if lean == 0 or Rl[:2, 2] @ n_out < 0:
                        R = Rl
                        break
                d_fix = self.cc.pose_clearance(pos, R, self.open_rel, fgeoms)
                d_h = self.cc.pose_clearance(pos, R, [x for x in self.open_rel if "pad" in m.geom(x[0]).name], [handle])
                if __import__("os").environ.get("DEBUG_FX"):
                    worst = sorted((round(self.cc.pose_clearance(pos, R, [x], fgeoms), 4), m.geom(x[0]).name[8:]) for x in self.open_rel)[:3]
                    print(f"   MW dz={dz} lean={lean} sgn={sgn} d_fix={d_fix:.4f} d_h={d_h:.4f} worst={worst}", flush=True)
                if d_fix < 0.0 or d_h < 0.0005:
                    continue
                # end of the swing must be reachable with this hand yaw
                # the whole swing must be reachable with this hand yaw (IK seeded
                # step by step along the arc)
                # how far along the swing the arm can follow comfortably
                # the gripper is symmetric: of R and its 180-degree twin, the one
                # the arm reaches best at the start of the swing
                # ... judged over the whole swing (the hand turns with the door,
                # which can wind the last joint into its stop)
                best0 = None
                for Rc in (R, R @ np.diag([-1.0, -1.0, 1.0])):
                    ep, er, q_ = self.reach.solve(pos, Rc, q0=self.reach.current_q(), iters=150)
                    if ep >= 0.006 or er >= 0.06:
                        continue
                    w_, xo_, qe_ = 9.0, 0.0, q_
                    for x in np.linspace(0, 1, 9):
                        pe = np.r_[hinge + rotz(dq * x)[:2, :2] @ (pos[:2] - hinge), pos[2]]
                        ep, er, qe_ = self.reach.solve(pe, rotz(dq * x) @ Rc, q0=qe_, iters=40)
                        if ep > 0.008 or er > 0.08 or self.reach.margin(qe_) < 0.08:
                            break
                        w_ = min(w_, self.reach.margin(qe_)); xo_ = x
                    key = (xo_, w_)
                    if best0 is None or key > best0[0]:
                        best0 = (key, Rc)
                if best0 is None:
                    continue
                (x_ok, worst), R = best0
                if x_ok == 0.0:
                    worst = -1.0
                if abs(dq) * x_ok < min(0.7, abs(dq)):
                    worst = -1.0
                if __import__("os").environ.get("DEBUG_FX"):
                    print(f"   MW-ARC dz={dz} lean={lean} sgn={sgn} worst margin {worst:.2f} last x {x:.1f} ep {ep:.3f} er {er:.2f} dq {dq:.2f}", flush=True)
                if worst < 0.08:
                    continue
                cands.append((worst + 0.3 * d_fix + 0.1 * (dz == 0.047) + 0.5 * x_ok, pos, R, x_ok))
        if not cands:
            self.plan_log.append(("microwave", "no handle grasp"))
            return None
        _, pos, R, x_ok = max(cands, key=lambda x: x[0])
        dq *= x_ok
        if want_open:
            # don't fling it further open than one can reach to push it shut again
            pg = d.geom_xpos[panel]
            edge = hinge + 2 * (pg[:2] - hinge) * 0.95
            for k in range(20):
                e_t = np.r_[hinge + rotz(dq)[:2, :2] @ (edge - hinge), pg[2]]
                if self.reach.wrist_dist(e_t, R_DOWN)[0] <= 0.715 or abs(dq) < 0.5:
                    break
                dq *= 0.92
        self.cur_targets = {f}
        pc.phase = "approach"
        dist = np.linalg.norm(pos - pc.eef_pos)
        T = float(np.clip(1.74 + 4.02 * dist, 1.5, 4.0) * self.rng.uniform(0.9, 1.1))
        keys = [(T - 0.9, pos + np.r_[n_out * 0.03, self.rng.uniform(0.06, 0.09)]), (T - 0.35, pos + np.r_[0, 0, 0.025]), (T, pos)]
        # reaching for a handle at arm's length: the IK check above decides,
        # so the coarse straight-arm limit may be a bit looser here
        wm = self.reach.WRIST_MAX
        self.reach.WRIST_MAX = 0.76
        try:
            ok = self.checked_move(keys, R, -1.0, ignore_owner=f, ignore_last=14, exact=True, v_end=np.r_[0, 0, -0.01])
        finally:
            self.reach.WRIST_MAX = wm
        if not ok:
            return None
        pc.phase = "correct"
        c.settle(pos, R, -1.0, tol=0.004, max_steps=10)
        pc.phase = "grasp"
        c.hold(int(self.rng.integers(7, 10)), 1.0)
        # swing along the door's arc, the hand turning with the door
        pc.phase = "articulate"
        n = int(30 + abs(dq) * 30 * self.rng.uniform(0.9, 1.2))
        s = np.linspace(0, 1, n)
        s = 3 * s ** 2 - 2 * s ** 3
        p0 = pc.eef_pos.copy(); R0 = pc.eef_mat.copy()
        P = np.array([np.r_[hinge + rotz(dq * x)[:2, :2] @ (p0[:2] - hinge), p0[2]] for x in s])
        Rp = np.array([rotz(dq * x) @ R0 for x in s])
        c.follow(P, np.gradient(P, axis=0), Rp, 1.0, residual=0.0, wobble=0.3, precise_end=False)
        pc.phase = "release"
        c.hold(int(self.rng.integers(5, 8)), -1.0)
        # back off away from the door, then up
        pc.phase = "retreat"
        _, q1 = self._joint_q(f"{f}_microjoint")
        out = rotz(q1 - q0)[:2, :2] @ n_out
        if not self.checked_move([(0.6, pc.eef_pos + np.r_[out * 0.05, 0.01]), (1.3, pc.eef_pos + np.r_[out * 0.08, 0.08])],
                                 pc.eef_mat, -1.0, ignore_owner=f, ignore_last=0):
            c.move([(0.9, pc.eef_pos + np.r_[0, 0, 0.08])], pc.eef_mat, grip=-1.0)
        ok = q1 < -0.6 if want_open else q1 > -0.15
        return dict(primitive="microwave_open" if want_open else "microwave_close", target=f, success=bool(ok),
                    language=f"{'open' if want_open else 'close'} the microwave", diag=dict(q0=q0, q1=q1))

    # ------------------------------------------------------------ surfaces
    def fixture_surfaces(self):
        """Places on fixtures to put things: (label, owner, center xy, top z,
        half extents xy)."""
        pc, m, d = self.pc, self.pc.m, self.pc.d
        out = []
        f = self.fixture_named("stove")
        if f is not None:
            s = m.site(f"{f}_cook_region").id
            bur = [g for g in self._fixture_geoms(f) if "burner" in m.body(m.geom_bodyid[g]).name]
            top = max(d.geom_xpos[g][2] + self._box_extent(g)[2] for g in bur) if bur else d.site_xpos[s][2] + 0.025
            out.append(("the stove", f, d.site_xpos[s][:2].copy(), top, np.r_[0.035, 0.035]))
        f = self.fixture_named("microwave")
        if f is not None:
            s = m.site(f"{f}_top_side").id
            out.append(("the top of the microwave", f, d.site_xpos[s][:2].copy(), d.site_xpos[s][2] - m.site_size[s][2],
                        m.site_size[s][:2] * 0.6))
        for f in [x for x in pc.e.fixtures_dict if "cabinet" in x]:
            # the cabinet's top panel: the highest collision box spanning most of it
            gs = self._fixture_geoms(f)
            ext = {g: self._box_extent(g) for g in gs}
            top_g = max(gs, key=lambda g: d.geom_xpos[g][2] + ext[g][2])
            ctr = d.geom_xpos[top_g]
            out.append(("the top of the cabinet", f, ctr[:2].copy(), ctr[2] + ext[top_g][2], ext[top_g][:2] * 0.55))
        return out

    def _dest_surface(self, n, surf):
        label, owner, xy, top, half = surf
        cat = self.cat(n)
        if cat not in GEOM:
            return None
        e = (np.array(GEOM[cat]["hi"]) - np.array(GEOM[cat]["lo"]))[:2].max() / 2
        h = GEOM[cat]["hi"][2] - GEOM[cat]["lo"][2]
        if top > self.pc.table_z + 0.1 and h > 0.11:
            return None  # a tall bottle on a cabinet / microwave top falls off
        room = np.maximum(half - 0.5 * e, 0.0)
        rc = self._ensure_reach()
        for _ in range(4):
            g = np.r_[xy + self.rng.uniform(-1, 1, 2) * room, top + bottom_offset(cat) + 0.004]
            # the hand must comfortably reach above it (holding the object)
            hand = g + np.r_[0, 0, 0.5 * h + 0.03]
            if rc.wrist_dist(hand, np.diag([1.0, -1.0, -1.0]))[0] < rc.WRIST_MAX - 0.02:
                return "on", g, owner
        return None

    def _dest_drawer(self, n):
        pc = self.pc
        open_dr = [dr for dr in pc.drawers if pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.1 and dr["level"] == "top"]
        if not open_dr:
            return None
        dr = open_dr[0]
        R = pc.d.xmat[dr["bid"]].reshape(3, 3)
        hp = pc.d.geom_xpos[dr["handle"]]
        q = pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]]
        # middle of the open part of the drawer box
        xy = hp[:2] + R[:2, 1] * (0.08 + self.rng.uniform(-0.015, 0.015)) + R[:2, 0] * self.rng.uniform(-0.03, 0.03)
        return "in_drawer", np.r_[xy, hp[2] - 0.035 + bottom_offset(self.cat(n))], f"{dr['fixture']}:{dr['level']}"
