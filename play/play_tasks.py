import os
"""Playful task layer on top of composed_play: containers are grasped and moved
too, objects go into and come out of containers, things are moved across the
table, lifted and looked at, stacked, put in drawers; targets are chosen by
novelty, and when something is in the way the robot moves it first."""
import numpy as np

from errors import has as _has

from composed_play import ComposedPlay, small_tilt
from fixtures_play import FixturePlay
from interactions import InteractionPlay
from creative import CreativePlay
from compose import R_DOWN, rotz, wrap
from human_play import GEOM, GROUP_OF, top_center, bottom_offset, quat_to_mat

CONTAINERS = {"basket", "container"}
HOLDERS = {"basket", "container", "plate", "bowl"}
STACK_ON = {"box", "book"}
# kinds of play that look alike to a viewer
FAMILY = {"move": "shuffle", "reorient": "shuffle", "push": "shuffle",
          "reach": "idle", "fumble": "idle",
          "push_into": "contact", "bump_with": "contact", "door_sweep": "contact",
          "put_in": "container", "stack": "container", "take_out": "container",
          "put_on_fixture": "fixture_place", "put_in_drawer": "fixture_place",
          "drawer": "fixture", "stove": "fixture", "microwave": "fixture",
          "push_with": "contact", "line_up": "arrange", "tower": "arrange",
          "reach_toward": "idle", "pick_fumble": "idle", "push_into ": "contact",
          "bump": "contact"}


def nice(n):
    return n.rsplit("_", 1)[0].replace("_", " ")


class PlayfulPlay(CreativePlay, InteractionPlay, FixturePlay, ComposedPlay):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.touch = {n: 0 for n in self.pc.names}
        self.fails = {}
        self.graze_failed = set()

    # ------------------------------------------------------ container grasps
    def _wall_grasps(self, n):
        """Grasps on the container's real walls: every thin, tall collision box
        is a wall; try points along it with the fingers closing across it, and
        keep those whose open fingers clear the container's other walls."""
        pc = self.pc
        m, d = pc.m, pc.d
        geoms = list(pc._obj_geoms[n])
        robot = np.array(pc.robot.base_pos)
        p_obj, _ = self.obj_pose(n)
        cands = []
        for g in geoms:
            if m.geom_type[g] != 6:
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            # the box axis that is most vertical is its height; of the other two,
            # the thinner one is the wall thickness (works for rotated containers)
            vert = int(np.argmax(np.abs(Rg[2, :])))
            hz = [i for i in range(3) if i != vert]
            if size[vert] < 0.02 or min(size[hz]) > 0.016:
                continue
            thin = hz[int(np.argmin(size[hz]))]
            along = hz[int(np.argmax(size[hz]))]
            ctr = d.geom_xpos[g]
            top = ctr[2] + size[vert] * abs(Rg[2, vert])
            nrm = Rg[:, thin].copy(); nrm[2] = 0; nrm /= np.linalg.norm(nrm) + 1e-9
            av = Rg[:, along].copy(); av[2] = 0; av /= np.linalg.norm(av) + 1e-9
            for f in (-0.6, -0.3, 0.3, 0.6, 0.0):
                pos = ctr + av * f * size[along]
                pos[2] = top - self.rng.uniform(0.018, 0.026)
                R = rotz(self._yaw_for_axis(nrm[:2])) @ R_DOWN @ small_tilt(self.rng, 3.0)
                others = [o for o in geoms if o != g]
                Rs = [(0.0, R)]
                if getattr(self, "_dump_mode", False):
                    # to pour something out, people grab the container already
                    # tilted, so turning the wrist takes it well past vertical
                    from scipy.spatial.transform import Rotation as _Rot
                    Rs = [(-0.1, _Rot.from_rotvec(nrm * sg * self.rng.uniform(0.6, 0.75)).as_matrix() @ R) for sg in (1, -1)] + Rs
                for bonus, R_ in Rs:
                    if self.cc.pose_clearance(pos, R_, self.open_rel, others) < 0.004:
                        continue
                    score = np.linalg.norm(pos[:2] - robot[:2]) + 0.05 * abs(f) + bonus
                    cands.append((score, pos, R_))
        if not cands:
            return []
        cands.sort(key=lambda x: x[0])
        return [(pos, R) for _, pos, R in cands[:8]]

    def grasp_candidates(self, n, explore=False, k=10):
        if GROUP_OF.get(self.cat(n)) in CONTAINERS and self.cat(n) in GEOM:
            return self._wall_grasps(n)
        from grasp_sampler import GraspSampler
        c = GraspSampler(self).candidates(n, k=k)
        if explore and c:
            pos, R = c[0]
            c = [(pos + np.r_[self.rng.normal(0, 0.015, 2), 0.0], R)] + c[1:]
        return c

    def _radius(self, n):
        return super()._radius(n)

    # ------------------------------------------------------------- dests
    def _dest_far(self, n):
        self._snug_to = None
        if self.rng.random() < 0.22:
            sn = self._dest_snug(n)
            if sn is not None:
                self._snug_to = sn[1]
                return sn[0]
        self.min_move = 0.15
        xy = self._free_spot(n, min_move=0.15)
        if xy is None:
            xy = self._free_spot(n, min_move=0.08)
        if xy is None:
            return None
        return "table", np.r_[xy, self.pc.table_z + bottom_offset(self.cat(n)) + 0.003], "table"

    def _dest_in(self, n, c):
        cat_c = self.cat(c)
        p, q = self.obj_pose(c)
        lo, hi = np.array(GEOM[cat_c]["lo"]), np.array(GEOM[cat_c]["hi"])
        Rr = quat_to_mat(q) @ quat_to_mat(GEOM[cat_c]["quat"]).T
        inner = (hi[:2] - lo[:2]) / 2 * 0.35
        loc = np.r_[(lo[:2] + hi[:2]) / 2 + self.rng.uniform(-1, 1, 2) * inner, 0]
        xy = (p + Rr @ loc)[:2]
        if GROUP_OF.get(cat_c) in CONTAINERS:
            # a spot where the object actually fits on the container's floor
            # (between walls and dividers), from the container's collision geoms
            rim = p[2] + hi[2]
            spot = self._fit_in_container(n, c, rim)
            if spot is not None:
                # lower it in and let go a little above the floor (a long drop
                # onto stiff thin slats can launch the object), not deep between walls
                xy, floor = spot
                z_bot = max(floor + 0.015, rim - 0.04)
            else:
                # doesn't fit between the walls: set it on top anyway and let it
                # rest however it lands (play, not tidiness)
                z_bot = rim + 0.01
            return "in", np.r_[xy, z_bot + bottom_offset(self.cat(n))], c
        tc = top_center(cat_c, p, q)
        return "on", np.r_[xy, tc[2] + bottom_offset(self.cat(n)) - 0.005], c

    def _fit_in_container(self, n, c, rim, tries=40, want=0.008):
        """Random xy inside the container where the object (translated, same
        orientation) sits on the container's floor with clearance to its walls
        and dividers. Returns (xy, floor_z) or None."""
        from collision import _gdist
        pc = self.pc
        m, d = pc.m, pc.d
        cat_c = self.cat(c)
        p, q = self.obj_pose(c)
        lo, hi = np.array(GEOM[cat_c]["lo"]), np.array(GEOM[cat_c]["hi"])
        Rr = quat_to_mat(q) @ quat_to_mat(GEOM[cat_c]["quat"]).T
        og = [g for g in pc._obj_geoms[n] if m.geom_type[g] == 6] or list(pc._obj_geoms[n])
        cg = list(pc._obj_geoms[c])
        bb = pc.obj_bbox(n)
        ctr = 0.5 * (bb["lo"] + bb["hi"])
        saved = [(d.geom_xpos[g].copy()) for g in og]
        best = None
        try:
            for _ in range(tries):
                u = self.rng.uniform(-0.8, 0.8, 2) * (hi[:2] - lo[:2]) / 2
                loc = np.r_[(lo[:2] + hi[:2]) / 2 + u, 0]
                xy = (p + Rr @ loc)[:2]
                floor = self._floor_z(c, xy, rim + 0.05)
                if floor is None or floor > rim - 0.015:
                    continue  # on a divider / wall top, not the floor
                delta = np.r_[xy - ctr[:2], floor + 0.01 - bb["lo"][2]]
                for g, xp in zip(og, saved):
                    d.geom_xpos[g] = xp + delta
                dist = min(_gdist(m, d, a, b, 0.05, None) for a in og for b in cg)
                if os.environ.get("DEBUG_FIT"):
                    print(f"      FIT xy={np.round(xy - p[:2], 3)} floor-rim={floor - rim:+.3f} clearance={dist:+.4f}", flush=True)
                if dist >= want:
                    return xy, floor
                if best is None or dist > best[0]:
                    best = (dist, xy, floor)
        finally:
            for g, xp in zip(og, saved):
                d.geom_xpos[g] = xp
        if best is not None and best[0] >= 0.0:
            return best[1], best[2]
        return None

    def _floor_z(self, c, xy, z_from):
        """Height of the container's own surface under xy (ray cast down)."""
        import mujoco
        m, d = self.pc.m, self.pc.d
        geoms = set(self.pc._obj_geoms[c])
        pnt = np.r_[xy, z_from]
        gid = np.zeros(1, np.int32)
        for _ in range(10):
            dist = mujoco.mj_ray(m, d, pnt, np.array([0, 0, -1.0]), None, 1, -1, gid)
            if dist < 0:
                return None
            hz = pnt[2] - dist
            if gid[0] in geoms:
                return float(hz)
            if hz < self.pc.table_z - 0.01:
                return None
            pnt = np.r_[xy, hz - 1e-4]
        return None

    def fits_in(self, n, c):
        if n == c or self.cat(n) not in GEOM or self.cat(c) not in GEOM:
            return False
        if GROUP_OF.get(self.cat(n)) in CONTAINERS:
            return False
        e_n = np.array(GEOM[self.cat(n)]["hi"][:2]) - np.array(GEOM[self.cat(n)]["lo"][:2])
        e_c = np.array(GEOM[self.cat(c)]["hi"][:2]) - np.array(GEOM[self.cat(c)]["lo"][:2])
        if not (max(e_n) < min(e_c) * 1.05 and self.support_of(n) != c):
            return False
        # tall things don't stand on a plate, bowl or ramekin (they topple)
        h_n = GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]
        if GROUP_OF.get(self.cat(c)) in ("plate", "bowl") and h_n > max(0.075, 1.2 * min(e_n)):
            return False
        return True

    # ------------------------------------------------------------- tasks
    def _dump(self, n, z0):
        """Pick a container up, carry it over open table, tip it until the
        contents fall out, turn it back. The tipping direction is chosen by
        trying a few in simulation first (from a snapshot) and keeping the one
        the wrist can actually do."""
        from scipy.spatial.transform import Rotation as _Rot
        pc, c = self.pc, self.c
        inside = [m for m in pc.names if m != n and self.support_of(m) == n]
        moving = self.cc.gripper_rel() + self.cc.attach_rel(pc._obj_geoms[n]) + self.cc.forearm_rel()
        for m in inside:
            moving += self.cc.attach_rel(pc._obj_geoms[m])
        sup = self.support_of(n)
        pc.phase = "transport"
        # lift, and drift over a free patch of table
        xy = self._free_spot(n, min_move=0.06)
        lift = self.rng.uniform(0.16, 0.22)
        over = pc.eef_pos + np.r_[0, 0, lift] if xy is None else \
            np.r_[pc.eef_pos[:2] + 0.6 * (xy - self.obj_pose(n)[0][:2]), pc.eef_pos[2] + lift]
        if not (self.checked_move([(0.9, pc.eef_pos + np.r_[0, 0, 0.07]), (2.0, over)], pc.eef_mat, 1.0,
                                  moving=moving, ignore_first=(("table", sup, n) + tuple(inside), 16)) or
                self.checked_move([(0.9, pc.eef_pos + np.r_[0, 0, 0.07]), (1.8, pc.eef_pos + np.r_[0, 0, lift])], pc.eef_mat, 1.0,
                                  moving=moving, ignore_first=(("table", sup, n) + tuple(inside), 16))):
            return False
        if self.obj_pose(n)[0][2] - z0 < 0.05 or not pc.is_grasped(n):
            return False
        # try the tipping directions in simulation, keep the best
        R0, p0 = pc.eef_mat.copy(), pc.eef_pos.copy()
        cands = []
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            ax = np.r_[np.cos(a), np.sin(a), 0]
            for mag in (2.0, 2.5):
                cands.append(_Rot.from_rotvec(ax * mag * self.rng.uniform(0.95, 1.05)).as_matrix() @ R0)
        snap = pc.snapshot(); rec = pc.record_fn
        corr = (getattr(c, "_corr", np.zeros(3)).copy(), getattr(c, "_rcorr", np.zeros(3)).copy())
        scores = []
        qi, jlo, jhi = c._jlim if hasattr(c, "_jlim") else (c.joint_margin(), c._jlim)[1]
        true_margin = lambda: float(np.min(np.minimum(pc.d.qpos[qi] - jlo, jhi - pc.d.qpos[qi])))
        c.guard = False   # pouring takes the wrist past the everyday range on purpose
        try:
            pc.record_fn = None
            for R in cands:
                pc.restore(snap); c._corr, c._rcorr = corr
                try:
                    c.move([(1.5, p0 + np.r_[0, 0, 0.02])], R, grip=1.0, precise_end=False)
                except Exception:
                    scores.append(-1.0); continue
                tip = float(np.degrees(np.arccos(np.clip(self.rel_R(n)[2, 2], -1, 1))))
                ok = pc.is_grasped(n) and true_margin() > 0.03
                scores.append(tip if ok else -1.0)
        finally:
            pc.restore(snap); c._corr, c._rcorr = corr
            pc.record_fn = rec
        k = int(np.argmax(scores))
        if scores[k] < 60:
            c.guard = True
            self.plan_log.append(("dump", "can't tip", round(scores[k], 0)))
            return True  # just holds it up for a moment, then puts it down
        self.plan_log.append(("dump", "tip", round(scores[k], 0)))
        pc.phase = "inspect"
        c.move([(self.rng.uniform(1.3, 1.7), p0 + np.r_[0, 0, 0.02])], cands[k], grip=1.0, precise_end=False)
        # shake it out: a couple of quick up-down jolts while it's upside down
        Rt = pc.eef_mat.copy()
        for _ in range(int(self.rng.integers(2, 4))):
            if not [m for m in inside if self.support_of(m) == n]:
                break
            pe = pc.eef_pos.copy()
            c.move([(0.25, pe + np.r_[0, 0, 0.035])], Rt, grip=1.0, precise_end=False, residual=0.0)
            c.move([(0.2, pe - np.r_[0, 0, 0.01])], Rt, grip=1.0, precise_end=False, residual=0.0)
        c.hold(int(self.rng.integers(4, 10)), 1.0)
        pc.phase = "transport"
        c.move([(self.rng.uniform(1.2, 1.6), p0)], R0, grip=1.0, precise_end=False)
        c.guard = True
        return pc.is_grasped(n)

    def _inspect(self, n, z0):
        """Lift it up, turn it around a bit, have a look."""
        if getattr(self, "_dump_mode", False):
            return self._dump(n, z0)
        if getattr(self, "_bump_target", None) is not None:
            return self._bump(n, z0)
        pc = self.pc
        moving = self.cc.gripper_rel() + self.cc.attach_rel(pc._obj_geoms[n]) + self.cc.forearm_rel()
        pc.phase = "transport"
        sup = self.support_of(n)
        # where to hold it up: a random spot above, else straight up / lower /
        # drifting toward the open middle of the table
        x0, y0 = pc.e.workspace_offset[:2]
        mid = np.r_[x0 - 0.05, y0] - pc.eef_pos[:2]
        mid = mid / (np.linalg.norm(mid) + 1e-9)
        ups = [np.r_[self.rng.normal(0, 0.03, 2), self.rng.uniform(0.14, 0.2)],
               np.r_[0, 0, self.rng.uniform(0.12, 0.16)],
               np.r_[mid * 0.08, self.rng.uniform(0.12, 0.16)],
               np.r_[0, 0, 0.09]]
        for du in ups:
            up = pc.eef_pos + du
            if self.checked_move([(0.9, pc.eef_pos + np.r_[0, 0, min(0.06, du[2] * 0.5)]), (1.8, up)], pc.eef_mat, 1.0,
                                 moving=moving, ignore_first=(("table", sup, n), 16)):
                break
        else:
            return False
        if self.obj_pose(n)[0][2] - z0 < 0.04:
            return False
        pc.phase = "inspect"
        R_grasp = pc.eef_mat.copy()
        from scipy.spatial.transform import Rotation as _Rot
        for _ in range(int(self.rng.integers(1, 3))):
            # turn it to look at another side: a yaw plus a tilt toward the robot
            ax = np.r_[self.rng.normal(size=2), 0]; ax /= np.linalg.norm(ax)
            R = _Rot.from_rotvec(ax * self.rng.uniform(0.25, 0.5)).as_matrix() @ rotz(self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 0.7)) @ pc.eef_mat
            tgt = pc.eef_pos + np.r_[self.rng.normal(0, 0.03, 2), self.rng.normal(0, 0.015)]
            self.checked_move([(self.rng.uniform(1.3, 2.0), tgt)], R, 1.0, moving=moving)
            if not pc.is_grasped(n):
                return True  # slipping: stop waving it around and put it down
            if getattr(self.c, "limit_hit", False):
                break  # the wrist won't turn further this way
        # turn it back the way it was held before putting it down
        self.checked_move([(self.rng.uniform(1.0, 1.5), pc.eef_pos + np.r_[0, 0, 0.0])], R_grasp, 1.0, moving=moving)
        return True

    def p_push(self, n, direction=None, into=None):
        """Slide the object across the table with a closed gripper: come down
        behind it, push 8-20 cm toward free space, lift away. Contact only with
        the target; the object's swept footprint must stay clear of others."""
        pc, c = self.pc, self.c
        self.cur_targets = {n}
        cat = self.cat(n)
        if cat not in GEOM:
            return None
        if self.support_of(n) is not None:
            return None
        p, q = self.obj_pose(n)
        # footprint and height as the object lies now (it may be on its side)
        bb = pc.obj_bbox(n)
        h = float(bb["hi"][2] - bb["lo"][2])
        r = 0.5 * float(np.linalg.norm((bb["hi"] - bb["lo"])[:2]))
        p = np.r_[0.5 * (bb["lo"] + bb["hi"])[:2], p[2]]
        geoms = list(pc._obj_geoms[n])
        # whatever sits inside / on it slides along
        for m_ in pc.names:
            if m_ != n and self.support_of(m_) == n:
                geoms += list(pc._obj_geoms[m_])
        cands = []
        hit = set()
        into_p0 = self.obj_pose(into)[0].copy() if into is not None else None
        if into is not None:
            # push it so it runs into `into` and shoves it a little
            bm_ = pc.obj_bbox(into)
            pm = 0.5 * (bm_["lo"] + bm_["hi"])
            dv = np.r_[pm[:2] - p[:2], 0]
            direction = dv / (np.linalg.norm(dv) + 1e-9)
            half_ = lambda b_: 0.5 * float(np.abs((b_["hi"] - b_["lo"])[:2]) @ np.abs(direction[:2]))
            # the faces' gap along the push (bounding diagonals would overstate it)
            gap = np.linalg.norm(dv) - half_(bb) - half_(bm_)
            hit = {into} | {m_ for m_ in pc.names if self.support_of(m_) == into}
            dist_opts = (max(gap, 0.0) + self.rng.uniform(0.04, 0.09),)
        dirs = [direction] if direction is not None else [np.r_[np.cos(a), np.sin(a), 0] for a in self.rng.permutation(np.linspace(0, 2 * np.pi, 12, endpoint=False))]
        for d in dirs:
            for dist in (dist_opts if into is not None else (self.rng.uniform(0.12, 0.2), self.rng.uniform(0.08, 0.12))):
                end = p + d * dist
                x0, y0 = pc.e.workspace_offset[:2]
                if not (-0.3 < end[0] - x0 < 0.18 and -0.33 < end[1] - y0 < 0.33):
                    continue
                # keep it where it can still be played with afterwards
                base = np.array(pc.robot.base_pos)[:2]
                if not (0.48 < np.linalg.norm(end[:2] - base) < 0.62):
                    continue
                sw = self.cc.swept_clearance(geoms, [d * x for x in np.linspace(0, dist, 6)], {n} | hit | {m_ for m_ in pc.names if self.support_of(m_) == n})
                if sw[0] < 0.008:
                    continue
                cands.append((d, dist))
                break
        if not cands:
            self.plan_log.append(("push", "no free direction"))
            return None
        # for every free direction, the clearest way of getting the hand down
        # behind the object (hover heights, or up-over-down); then push along
        # the clearest of them, brushing a loose neighbour at most
        moving = self.cc.gripper_rel() + self.cc.forearm_rel()
        # how to push it: low down, in the middle or up high on the object
        # (high on a tall thing can tip it over), fingers closed or open, with
        # the flat of the hand or its narrow side
        where = self.rng.choice(["low", "middle", "high"], p=[0.35, 0.35, 0.3]) if into is None else "low"
        frac = {"low": 0.2, "middle": 0.5, "high": 0.85}[where]
        zc = pc.table_z + max(0.012, min(frac * h, h - 0.006)) + 0.004
        grip_push = 1.0 if self.rng.random() < 0.7 else -1.0
        side_on = self.rng.random() < 0.3
        self._push_style = dict(height=where, fingers="closed" if grip_push > 0 else "open", hand="side" if side_on else "flat")
        best = None
        for d, dist in cands:
            start = np.r_[p[:2] - d[:2] * (r + 0.035), zc]
            e_ = getattr(self, "err", None) or {}
            if _has(e_, "push_offset"):
                # off-centre: the object spins away rather than sliding straight
                start[:2] += np.r_[-d[1], d[0]] * e_["offset"]
            # the hand stays out of the folded-up zone next to the robot
            base_ = np.array(pc.robot.base_pos)[:2]
            if np.linalg.norm(start[:2] - base_) < 0.47:
                self.plan_log.append(("push", "too close to the body"))
                continue
            # nor out at full stretch (the arm ends up straight and stuck)
            if np.linalg.norm(start[:2] - base_) > 0.64:
                self.plan_log.append(("push", "too far"))
                continue
            # fingers closed, palm face perpendicular to the push
            ax_ = d[:2] if side_on else np.cross([0, 0, 1], d)[:2]
            R = rotz(self._yaw_for_axis(ax_)) @ R_DOWN @ small_tilt(self.rng, 4.0)
            travel = dist + r + 0.035 - 0.01   # close the gap, then push the centre by ~dist
            # the hand is symmetric: of the two ways to hold it, take the one
            # that leaves the last joint room all the way to the end of the
            # stroke (moving across the table winds it on its own)
            R = self._push_flip(R, start, start + d * travel)
            Tp = travel / self.rng.uniform(0.06, 0.1)
            Pp, Vp, Rp = c.plan([(Tp, start + d * travel)], R, p0=start, R0=R, v0=np.zeros(3), exact=True)
            dp = self._clear(Pp, Rp, moving, lambda t, own: own in (n, "table") or own in hit or (own in pc.names and self.support_of(own) == n))
            if dp[0] < 0.008:
                self.plan_log.append(("push", "path blocked", round(dp[0], 3), dp[1]))
                continue
            p0 = pc.eef_pos
            T = float(np.clip(1.74 + 4.02 * np.linalg.norm(start - p0), 1.5, 4.5))
            variants = []
            for hov in (self.rng.uniform(0.06, 0.1), 0.14):
                variants.append([(T - 0.9, start + np.r_[0, 0, hov]), (T - 0.35, start + np.r_[0, 0, 0.02]), (T, start)])
            high = max(p0[2], zc) + 0.14
            variants.append([(0.3 * T, np.r_[p0[:2], high]), (0.7 * T, np.r_[start[:2], high]), (T - 0.4, start + np.r_[0, 0, 0.03]), (T, start)])
            for keys in variants:
                Pa, Va, Ra = c.plan(keys, R, v_end=np.r_[0, 0, -0.01], exact=True)
                da = self._clear(Pa, Ra, moving, lambda t, own: own == "table" and t >= len(Pa) - 8)
                tol = -0.02 if da[1] in pc.names else -0.004
                if da[0] >= 0.015:
                    best = (da[0], d, dist, start, R, Pa, Va, Ra, Pp, Vp, Rp)
                    break
                if da[0] >= tol and (best is None or da[0] > best[0]):
                    best = (da[0], d, dist, start, R, Pa, Va, Ra, Pp, Vp, Rp)
            if best is not None and best[0] >= 0.015:
                break
        if best is None:
            self.plan_log.append(("push", "approach blocked"))
            return None
        if best[0] < 0.015:
            self.plan_log.append(("push", "graze", round(best[0], 3)))
        _, d, dist, start, R, Pa, Va, Ra, Pp, Vp, Rp = best
        travel = dist + r + 0.035 - 0.01
        if True:
            pc.phase = "approach"
            c.follow(Pa, Va, Ra, grip_push)
            if np.linalg.norm(pc.eef_pos - start) > 0.02:
                # didn't get down behind it (joint stop / contact): don't shove blindly
                self.plan_log.append(("push", "approach short", round(float(np.linalg.norm(pc.eef_pos - start)), 3)))
                pc.phase = "retreat"
                c.move([(0.8, pc.eef_pos + np.r_[0, 0, 0.06])], pc.eef_mat, grip=-1.0, precise_end=False)
                # it was an attempt all the same
                prim = "push_into" if into is not None else "push"
                return dict(primitive=prim, target=n, interact=[into] if into is not None else [], success=False,
                            language=f"push the {nice(n)}" + (f" into the {nice(into)}" if into is not None else ""),
                            diag=dict(push_dist=float(dist), moved=float(np.linalg.norm(self.obj_pose(n)[0][:2] - p[:2])),
                                      note="couldn't get down behind it"))
            pc.phase = "push"
            if os.environ.get("DEBUG_PUSH"):
                m_, d_ = pc.m, pc.d
                con = sorted({(m_.body(m_.geom_bodyid[x.geom1]).name, m_.geom(x.geom2).name, round(float(x.pos[2] - pc.table_z), 3)) for x in d_.contact[:d_.ncon]})
                print(f"   PUSH start eef {np.round(pc.eef_pos - start, 3)} z-table {pc.eef_pos[2] - pc.table_z:.3f} d {np.round(d, 2)} r {r:.3f} h {h:.3f} obj {np.round(p - start, 3)}", [x for x in con if 'gripper' in x[0] + x[1] or 'robot' in x[0] + x[1]][:6], flush=True)
            c.follow(Pp, Vp, Rp, grip_push, residual=0.3, wobble=0.5, precise_end=False)
            if os.environ.get("DEBUG_PUSH"):
                con = sorted({(m_.body(m_.geom_bodyid[x.geom1]).name, m_.geom(x.geom2).name, round(float(x.pos[2] - pc.table_z), 3)) for x in d_.contact[:d_.ncon]})
                print(f"   PUSH end eef {np.round(pc.eef_pos - (start + d * travel), 3)} z-table {pc.eef_pos[2] - pc.table_z:.3f}", [x for x in con if 'gripper' in x[0] + x[1] or 'robot' in x[0] + x[1]][:6], flush=True)
            pc.phase = "retreat"
            self.checked_move([(0.9, pc.eef_pos + np.r_[-d[0] * 0.03, -d[1] * 0.03, 0.09])], pc.eef_mat, -1.0, ignore_owner=n, ignore_last=0)
            moved = np.linalg.norm(self.obj_pose(n)[0][:2] - p[:2])
            ang = np.degrees(np.arctan2(d[1], d[0]))
            word = ("away" if -45 <= ang < 45 else "to the left" if 45 <= ang < 135 else "to the right" if -135 <= ang < -45 else "closer")
            if into is not None:
                shoved = float(np.linalg.norm(self.obj_pose(into)[0][:2] - into_p0[:2]))
                return dict(primitive="push_into", target=n, interact=[into], success=bool(shoved > 0.01),
                            language=f"push the {nice(n)} into the {nice(into)}",
                            diag=dict(push_dist=float(dist), moved=float(moved), shoved=shoved))
            hword = {"low": "", "middle": " from the middle", "high": " near its top"}[where]
            return dict(primitive="push", target=n, success=bool(moved > 0.6 * dist), language=f"push the {nice(n)} {word}{hword}",
                        diag=dict(push_dist=float(dist), moved=float(moved), style=self._push_style,
                                  tipped=bool(not self.upright(n))))
        return None

    def _near_dist(self, n, xy):
        """Distance from xy to the nearest point of n's footprint."""
        bb = self.pc.obj_bbox(n)
        c = np.clip(xy, bb["lo"][:2], bb["hi"][:2])
        return float(np.linalg.norm(c - xy))

    def _push_flip(self, R, start, end):
        if not hasattr(self, "reach"):
            from reach import Reach
            self.reach = Reach(self.pc)
        rc = self.reach
        best = None
        for Rc in (R, R @ np.diag([-1.0, -1.0, 1.0])):
            q = rc.current_q()
            ok = True
            room = np.inf
            for p in (start, end):
                ep, er, q = rc.solve(p, Rc, q0=q)
                ok = ok and ep < 0.01
                room = min(room, 2.8973 - abs(q[6]))
            # prefer the one with room; among those, the smaller turn from now
            turn = np.linalg.norm(__import__("compose").rot_err(Rc, self.pc.eef_mat))
            key = (ok and room > 0.4, room > 0.4, -turn if room > 0.4 else room)
            if best is None or key > best[0]:
                best = (key, Rc)
        return best[1]

    def task_list(self):
        pc = self.pc
        # anything on the table is fair game (a can lying on its side can be
        # grasped across its body or pushed); only things tipped over inside
        # a container are left alone
        base_xy = np.array(pc.robot.base_pos)[:2]
        objs = [n for n in pc.names if GROUP_OF.get(self.cat(n)) is not None
                and (self.support_of(n) is None or self.rel_R(n)[2, 2] > 0.5)
                # right up against the robot: reaching there folds the arm up
                and np.linalg.norm(self.obj_pose(n)[0][:2] - base_xy) > 0.44
                # pushed out of reach: only at full stretch (where the arm gets stuck)
                and self._near_dist(n, base_xy) < 0.62
                # fallen off the table
                and self.obj_pose(n)[0][2] > pc.table_z - 0.03]
        last = self.segments[-1]["target"] if self.segments else ""
        novelty = lambda n: (1.0 / (1 + self.touch[n])) * (0.3 if n in last else 1.0)
        surfaces = self.fixture_surfaces()
        mq_ = self.microwave_q()
        drawer_open = any(pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.1 and dr["level"] == "top" for dr in pc.drawers)
        T = []
        for n in objs:
            nv = novelty(n)
            grp = GROUP_OF.get(self.cat(n))
            is_cont = grp in CONTAINERS
            sup = self.support_of(n)
            T.append((0.8 * nv, "move", n, None))
            if sup is not None:
                T.append((3.0 * nv, "take_out", n, sup))
            if not is_cont:
                for c in objs:
                    if c == n:
                        continue
                    if GROUP_OF.get(self.cat(c)) in HOLDERS and self.fits_in(n, c):
                        T.append((1.6 * nv, "put_in", n, c))
                    elif GROUP_OF.get(self.cat(c)) in STACK_ON and self.can_stack(n, c):
                        T.append((1.2 * nv, "stack", n, c))
                # the fixtures' surfaces: the burner, the microwave and cabinet tops
                for sf in surfaces:
                    if self._dest_surface(n, sf) is None:
                        continue
                    w = 0.7
                    if "stove" in sf[0]:
                        w = 3.0 if grp in ("pot", "pan") else 0.9
                    T.append((w * nv, "put_on_fixture", n, sf))
                if drawer_open:
                    # a tall bottle stands up out of the drawer and jams it when
                    # it's closed: that happens, but mostly put in shorter things
                    h_n = GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2] if self.cat(n) in GEOM else 0.05
                    T.append(((1.8 if h_n < 0.08 else 0.25) * nv, "put_in_drawer", n, None))
                T.append((0.5 * nv, "reorient", n, None))
            if sup is None:
                T.append((0.6 * nv, "push", n, None))
            # things hitting things: shove a neighbour by pushing this into it,
            # knock one with this while carrying it, sweep the door into it
            pn = self.obj_pose(n)[0]
            for m2 in objs:
                if m2 == n or self.support_of(m2) is not None:
                    continue
                dd = np.linalg.norm(self.obj_pose(m2)[0][:2] - pn[:2])
                if dd < 0.28 and sup is None and not is_cont:
                    T.append((0.55 * nv, "push_into", n, m2))
                if dd < 0.35 and not is_cont:
                    T.append((0.25 * nv, "bump_with", n, m2))
                    T.append((0.3 * nv, "push_with", n, m2))
            if mq_ is not None and sup is None:
                T.append((1.4 * nv, "door_sweep", n, None))
            T.append((0.05 * nv, "reach", n, None))
            T.append((0.025 * nv, "fumble", n, None))
        # arrangements of several things: a row, a little tower
        loose = [n for n in objs if self.support_of(n) is None and GROUP_OF.get(self.cat(n)) not in CONTAINERS]
        rowable = [n for n in loose if GROUP_OF.get(self.cat(n)) not in ("plate", "bowl")]
        if len(rowable) >= 2 and len(loose) >= 3:
            for _ in range(3):
                k_ = 2 if (len(rowable) < 3 or self.rng.random() < 0.7) else 3
                pick = list(self.rng.choice(rowable, size=min(len(rowable), k_), replace=False))
                anchor = [x for x in loose if x not in pick]
                if anchor:
                    T.append((0.9, "line_up", anchor[int(self.rng.integers(len(anchor)))], tuple(pick)))
        flat = [c_ for c_ in loose if GROUP_OF.get(self.cat(c_)) in STACK_ON]
        for c_ in flat:
            ups = [n for n in loose if n != c_ and self.can_stack(n, c_)]
            if len(ups) >= 2:
                a_, b_ = self.rng.choice(ups, size=2, replace=False)
                T.append((0.8, "tower", c_, (str(a_), str(b_))))
        for dr in pc.drawers:
            is_open = pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.04
            w = 1.2 if dr["level"] == "top" else 0.3
            T.append((w, "drawer", dr, not is_open))
        on = self.stove_on()
        if on is not None:
            T.append((1.6 if not on else 1.2, "stove", None, not on))
        mq = self.microwave_q()
        if mq is not None:
            T.append((1.6, "microwave", None, mq > -0.3))
        # fixtures are a big part of the scene: when there are any, about a
        # third of the choices are about them (otherwise the many object tasks
        # drown them out)
        FIX = ("drawer", "stove", "microwave", "put_on_fixture", "put_in_drawer", "door_sweep")
        w_fix = sum(w for w, k, _, _ in T if k in FIX)
        w_obj = sum(w for w, k, _, _ in T if k not in FIX)
        if w_fix > 0 and w_obj > 0:
            f_ = 0.35 / 0.65 * w_obj / w_fix
            T = [(w * f_ if k in FIX else w, k, a, b) for w, k, a, b in T]
        # variety: kinds of play not done yet in this episode are preferred;
        # don't keep retrying what keeps failing; don't repeat the last task
        import collections as _col
        done = _col.Counter(s_.get("kind", s_["primitive"]) for s_ in self.segments)
        fam_done = _col.Counter(FAMILY.get(s_.get("kind", s_["primitive"]), "other") for s_ in self.segments)
        last_task = getattr(self, "last_task", None)
        follow = self._follow_ups()
        n_seg = len(self.segments)
        out = []
        for w, k, a, b in T:
            key = (k, a if isinstance(a, str) else id(a))
            fam = FAMILY.get(k, "other")
            # variety: the same kind of play, and anything that looks alike
            # (moving / turning / pushing things about), gets less likely
            w = w * 0.3 ** self.fails.get(key, 0) * 0.55 ** done.get(k, 0) * 0.7 ** fam_done.get(fam, 0)
            # scenes with fixtures: start with them rather than shuffling things
            if fam in ("fixture", "fixture_place") and n_seg < 2:
                w *= 1.8
            # follow-through: what naturally comes next after the last actions
            w *= follow(k, a, b)
            if last_task == key:
                w *= 0.05
            if (k == "reach" and done.get(k, 0) >= 1) or (k == "fumble" and done.get(k, 0) >= 2):
                w = 0.0     # a reach-and-change-your-mind twice an episode is plenty
            out.append((w, k, a, b))
        if sum(o[0] for o in out) <= 0:
            return []
        return out

    def _follow_ups(self):
        """Weight multipliers for what naturally follows the recent actions:
        put something in an opened drawer and close it again, put the pot on a
        stove that was just switched on (and later switch it off), shut a
        microwave that was opened, take something back out of a container."""
        segs = [s_ for s_ in self.segments if s_.get("success")]
        last = segs[-1] if segs else None
        recent = [s_["primitive"] for s_ in segs[-3:]]
        lp = last["primitive"] if last else None
        since = lambda prim: next((i for i, s_ in enumerate(reversed(segs)) if s_["primitive"] == prim), None)

        def f(k, a, b):
            m = 1.0
            if lp == "drawer_open":
                if k == "put_in_drawer":
                    m *= 4.0
            if lp == "put_in_drawer":
                if k == "drawer" and b is False:
                    m *= 6.0          # close it again
                if k == "put_in_drawer":
                    m *= 1.5
            if k == "drawer" and b is False and since("put_in_drawer") is not None and since("put_in_drawer") < 3:
                m *= 2.0
            if lp == "stove_on" and k == "put_on_fixture" and isinstance(b, tuple) and "stove" in b[0]:
                m *= 4.0
            if lp == "put_on_stove" and k == "stove" and b is True:
                m *= 3.0
            on_for = since("stove_on")
            if k == "stove" and b is False and on_for is not None and on_for >= 2:
                m *= 2.5
            if lp == "microwave_open" and k in ("door_sweep", "put_on_fixture"):
                m *= 2.0
            opened = since("microwave_open")
            if k == "microwave" and b is False and opened is not None and opened >= 1:
                m *= 3.0
            if lp in ("put_in", "stack") and k == "take_out" and isinstance(a, str) and last["target"].startswith(a):
                m *= 1.5
            if k in ("reach", "fumble") and any(r_ in ("reach_toward", "pick_fumble") for r_ in recent):
                m *= 0.2
            return m
        return f

    def p_reorient(self, n):
        """Pick it up, turn it 60-150 degrees, set it down (where it was or a
        little way off)."""
        pc = self.pc
        # turn the way the last wrist joint has room for (dq7 ~ -dyaw)
        q7 = float(pc.d.qpos[pc.robot._ref_joint_pos_indexes[6]])
        mag = self.rng.uniform(np.radians(60), np.radians(150))
        turn = mag if q7 - mag > -2.6 else -mag
        if q7 + mag < 2.6 and self.rng.random() < 0.5:
            turn = -mag
        p, _ = self.obj_pose(n)
        near_base = np.linalg.norm(p[:2] - np.array(pc.robot.base_pos)[:2]) < 0.45
        if self.rng.random() < 0.5 and not near_base:
            xy = p[:2] + self.rng.normal(0, 0.015, 2)
        else:
            xy = self._free_spot(n, min_move=0.06)
            if xy is None:
                if near_base:
                    return None   # turning it right next to the robot folds the arm up
                xy = p[:2]
            elif not near_base:
                xy = p[:2] + (xy - p[:2]) * min(1.0, 0.12 / max(np.linalg.norm(xy - p[:2]), 1e-6))
        cat = self.cat(n)
        dest = ("table", np.r_[xy, pc.table_z + bottom_offset(cat) + 0.003], "table")
        word = "around" if abs(turn) > np.radians(120) else "a bit"
        info = self.p_pick_place(n, dest=dest, lang=f"turn the {nice(n)} {word}", prim="reorient", yaw_turn=turn)
        if info is not None:
            info["diag"] = dict(info.get("diag", {}), turn=float(np.degrees(turn)))
        return info

    def can_stack(self, n, c):
        """Can `n` sit on top of `c` (a box or book lying flat)?"""
        if n == c or self.cat(n) not in GEOM or self.cat(c) not in GEOM:
            return False
        if self.rel_R(c)[2, 2] < 0.9 or self.support_of(n) == c:
            return False
        e_n = np.array(GEOM[self.cat(n)]["hi"][:2]) - np.array(GEOM[self.cat(n)]["lo"][:2])
        e_c = np.array(GEOM[self.cat(c)]["hi"][:2]) - np.array(GEOM[self.cat(c)]["lo"][:2])
        h_c = GEOM[self.cat(c)]["hi"][2] - GEOM[self.cat(c)]["lo"][2]
        # something to stand on: the top is not much smaller than the object's
        # footprint, and not too high a tower
        h_n = GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]
        steady = h_n < max(0.075, 1.2 * min(e_n))     # a bottle on a box falls over
        return min(e_c) > 0.6 * min(e_n) and e_c.prod() > 0.5 * e_n.prod() and h_c < 0.09 and steady

    def run_task(self, kind, a, b, explore=False):
        if kind == "move":
            dest = self._dest_far(a)
            if dest is None:
                return None
            return self.p_pick_place(a, explore=explore, dest=dest,
                                     lang=f"move the {nice(a)} to another spot", prim="move")
        if kind == "take_out":
            dest = self._dest_far(a)
            if dest is None:
                return None
            return self.p_pick_place(a, dest=dest, lang=f"take the {nice(a)} out of the {nice(b)}", prim="take_out")
        if kind == "put_in":
            dest = self._dest_in(a, b)
            word = "in" if dest[0] == "in" else "on"
            return self.p_pick_place(a, dest=dest, lang=f"put the {nice(a)} {word} the {nice(b)}", prim=f"put_{word}")
        if kind == "stack":
            dest = self._dest_in(a, b)
            return self.p_pick_place(a, dest=dest, lang=f"stack the {nice(a)} on the {nice(b)}", prim="stack")
        if kind == "put_on_fixture":
            dest = self._dest_surface(a, b)
            if dest is None:
                return None
            prim = "put_on_stove" if "stove" in b[0] else "put_on_fixture"
            return self.p_pick_place(a, dest=dest, lang=f"put the {nice(a)} on {b[0]}", prim=prim)
        if kind == "put_in_drawer":
            dest = self._dest_drawer(a)
            if dest is None:
                return None
            return self.p_pick_place(a, dest=dest, lang=f"put the {nice(a)} in the drawer", prim="put_in_drawer")
        if kind == "reorient":
            return self.p_reorient(a)
        if kind == "push_into":
            return self.p_push(a, into=b)
        if kind == "push_with":
            return self.p_push_with(a, b)
        if kind == "line_up":
            return self.p_line_up(a, list(b))
        if kind == "tower":
            return self.p_tower(a, list(b))
        if kind == "bump_with":
            return self.p_bump_with(a, b)
        if kind == "door_sweep":
            return self.p_door_sweep(a)
        if kind == "stove":
            return self.p_stove_knob(b)
        if kind == "microwave":
            return self.p_microwave(b)
        if kind == "dump":
            dest = self._dest_far(a)
            if dest is None:
                return None
            inside = [m for m in self.pc.names if m != a and self.support_of(m) == a]
            self._dump_mode = True
            try:
                info = self.p_pick_place(a, dest=dest, inspect=True, lang=f"pick up the {nice(a)} and dump it out", prim="dump")
            finally:
                self._dump_mode = False
            if info is not None:
                out = [m for m in inside if self.support_of(m) != a]
                info["success"] = bool(info.get("grasped")) and len(out) > 0
                info["dumped"] = out
            return info
        if kind == "push":
            return self.p_push(a)
        if kind == "reach":
            return self.p_reach(a)
        if kind == "fumble":
            return self.p_fumble(a)
        if kind == "drawer":
            return self.p_drawer(a, b, explore=explore)
        return None

    def choose(self):
        T = self.task_list()
        if not T:
            return None
        w = np.array([t[0] for t in T])
        i = self.rng.choice(len(T), p=w / w.sum())
        _, kind, a, b = T[i]
        self.last_task = (kind, a if isinstance(a, str) else id(a))
        explore = self.rng.random() < self.p_explore

        def fn():
            import os
            if os.environ.get("DEBUG_TASKS"):
                tl = sorted(self.task_list(), key=lambda x: -x[0])[:6]
                print("CHOOSE", kind, a if isinstance(a, str) else (a["level"] if isinstance(a, dict) else b), "| top:", [(round(w, 3), k, x if isinstance(x, str) else "drawer") for w, k, x, _ in tl], flush=True)
            self.plan_log = []
            # injected near-failures (a fraction of the actions)
            from errors import sample_error
            self.err = sample_error(self.rng, kind, p_err=self.p_err, severity=self.err_severity)
            err0 = self.err   # same dict: hooks may add to it (e.g. what got knocked)
            self._last_err = err0
            self._knock_m = None
            self._err_label = err0
            if self.err and (_has(self.err, "shaky") or _has(self.err, "jitter")):
                self.pc.noise_cfg = dict(self.err, t0=self.pc.t)
            # sometimes not bothering with the cleanest way in or the roomiest
            # place: the hand or the object may clip a neighbour
            self.careless = kind in ("move", "put_in", "take_out", "stack", "reorient", "put_on_fixture") \
                and self.rng.random() < 0.3
            # after a grazing grasp failed on this object, clear the obstacle instead
            self.allow_graze = not (isinstance(a, str) and a in self.graze_failed)
            info = self.run_task(kind, a, b, explore=explore)
            self.allow_graze = True
            if info is not None and isinstance(a, str) and not info.get("success", True) and \
                    any(x[0] == "approach" and x[1] == "graze" for x in self.plan_log):
                self.graze_failed.add(a)
            if info is None and isinstance(a, str) and a in self.graze_failed:
                # graze not allowed -> the approach was reported blocked: fall through to clearing
                pass
            if info is None and kind in ("move", "take_out", "put_in", "inspect"):
                # blocked? move whatever is in the way first, then try again
                blk = [x for x in self.plan_log if x[0] == "approach" and x[1] == "blocked"]
                blocker = blk[-1][3] if blk and len(blk[-1]) > 3 else None
                if blocker in self.pc.names and blocker != a and GROUP_OF.get(self.cat(blocker)) is not None:
                    first = self.run_task("move", blocker, None)
                    if first is not None:
                        self.touch[blocker] += 1
                        second = self.run_task(kind, a, b, explore=False)
                        if second is not None:
                            second["language"] = f"move the {nice(blocker)} out of the way, then " + second["language"]
                            second["primitive"] = "clear_then_" + second["primitive"]
                            second["cleared"] = blocker
                            self.pc.noise_cfg = None
                            self.err = None
                            if err0 is not None:
                                second["error"] = err0
                            return second
                        key = (kind, a if isinstance(a, str) else id(a))
                        self.fails[key] = self.fails.get(key, 0) + 1
                        self.pc.noise_cfg = None
                        self.err = None
                        if err0 is not None:
                            first["error"] = err0
                        return first
                # still can't pick it: push it somewhere freer instead
                if kind in ("move", "take_out"):
                    if err0 is not None and err0["type"] not in ("shaky", "jitter"):
                        # a pick error means nothing for a push (a shaky hand still does)
                        if _has(err0, "shaky"):
                            self.err = err0 = {"type": "shaky", "sigma": err0["sigma"], "severe": True}
                        else:
                            self.err = err0 = None
                            self.pc.noise_cfg = None
                    info = self.p_push(a)
            if info is None:
                key = (kind, a if isinstance(a, str) else id(a))
                self.fails[key] = self.fails.get(key, 0) + 1
            elif not info.get("success", True):
                key = (kind, a if isinstance(a, str) else id(a))
                self.fails[key] = self.fails.get(key, 0) + 0.5
            if info is not None and isinstance(a, str):
                self.touch[a] += 1
            self.pc.noise_cfg = None
            self.err = None
            self._knock_m = None
            self._relax(False)
            if info is not None and err0 is not None:
                info["error"] = err0
            if info is not None:
                info.setdefault("kind", kind)
                if self.careless:
                    info["careless"] = True
                if kind in ("move", "take_out") and getattr(self, "_snug_to", None):
                    info["snug_to"] = self._snug_to
                    info["language"] = f"set the {nice(a)} down right next to the {nice(self._snug_to)}"
            self.careless = False
            if os.environ.get("DEBUG_TASKS"):
                rc = getattr(self, "reach", None)
                print("  RESULT", info and (info["primitive"], info["success"]), self.plan_log[-3:],
                      "q", None if rc is None else np.round(rc.current_q(), 2), "margin", None if rc is None else round(rc.margin(rc.current_q()), 2), flush=True)
            return info
        return fn
