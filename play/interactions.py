"""Exploring what happens when things hit each other: knocking or shoving
another object with the one being carried, and swinging the microwave door
into something standing in its way. (Pushing one object into another is
`p_push(n, into=m)`.)

Everything up to the intended contact is collision-planned like the rest of
the play; only the contact itself is deliberate. The object that gets hit is
recorded in the action's `interact` list (not counted as an accidental
disturbance). Mixin for PlayfulPlay."""
import numpy as np

from compose import rotz
from human_play import GEOM, bottom_offset
from scene_gen import REACH


def _nice(n):
    return n.rsplit("_", 1)[0].replace("_", " ")


class InteractionPlay:
    # ------------------------------------------------------- bump while carrying
    def p_bump_with(self, n, m):
        """Pick `n` up, knock or shove `m` with it on the way, then put `n`
        down somewhere."""
        dest = self._dest_far(n)
        if dest is None:
            return None
        pm0 = self.obj_pose(m)[0].copy()
        self._bump_target = m
        self._bump_note = "not reached (grasp/lift failed)"
        try:
            info = self.p_pick_place(n, dest=dest, inspect=True, prim="bump_with",
                                     lang=f"knock the {_nice(m)} with the {_nice(n)}")
        finally:
            self._bump_target = None
        if info is not None:
            shoved = float(np.linalg.norm(self.obj_pose(m)[0][:2] - pm0[:2]))
            info["interact"] = [m]
            info["success"] = bool(info.get("grasped")) and shoved > 0.005
            info["diag"] = dict(info.get("diag", {}), shoved=shoved, bump_style=getattr(self, "_bump_style", None),
                                bump_note=getattr(self, "_bump_note", None))
        return info

    def _bump(self, n, z0):
        """Called with `n` in the hand, before it is carried to its goal."""
        pc, c = self.pc, self.c
        m = self._bump_target
        self.cur_targets = {n, m}
        moving = self.cc.gripper_rel() + self.cc.attach_rel(pc._obj_geoms[n]) + self.cc.forearm_rel()
        sup = self.support_of(n)
        pc.phase = "transport"
        up = pc.eef_pos + np.r_[self.rng.normal(0, 0.01, 2), self.rng.uniform(0.05, 0.09)]
        if not self.checked_move([(self.rng.uniform(0.8, 1.0), up)], pc.eef_mat, 1.0, moving=moving,
                                 ignore_first=(("table", sup, n), 16)):
            # no clean way up: lift straight up anyway (that's the attempt)
            c.move([(self.rng.uniform(0.7, 0.9), pc.eef_pos + np.r_[0, 0, 0.05])], pc.eef_mat, grip=1.0,
                   precise_end=False)
            if not pc.is_grasped(n):
                return False
            self.plan_log.append(("bump", "no clean lift"))
            self._bump_note = "no clean lift"
            return True      # carry on to where it was going
        if self.obj_pose(n)[0][2] - z0 < 0.02 or not pc.is_grasped(n):
            return False
        bn, bm = pc.obj_bbox(n), pc.obj_bbox(m)
        po = 0.5 * (bn["lo"] + bn["hi"])
        pm = 0.5 * (bm["lo"] + bm["hi"])
        u0 = np.r_[pm[:2] - po[:2], 0.0]
        if np.linalg.norm(u0) < 1e-6:
            u0 = np.r_[1.0, 0, 0]
        u0 /= np.linalg.norm(u0)
        a0 = np.arctan2(u0[1], u0[0])
        style = "slide" if self.rng.random() < 0.5 else "side"
        self._bump_style = style
        h_m = bm["hi"][2] - pc.table_z
        if style == "slide":
            # skim along the table and run into it
            bottom = self.rng.uniform(0.008, 0.016)
        else:
            # come in a little higher, still overlapping the other object's height
            bottom = self.rng.uniform(0.014, max(0.016, min(0.04, h_m - 0.015)))
        z_obj = po[2] - (bn["lo"][2] - pc.table_z) + bottom
        # approach the other object from the most natural side that is free:
        # straight on from where this one is first, then turning further round
        offs = [0.0] + [s_ * k for k in (0.4, 0.8, 1.2, 1.6, 2.2) for s_ in (1, -1)]
        done = False
        for off in offs[:9]:
            u = np.r_[np.cos(a0 + off), np.sin(a0 + off), 0.0]
            half = lambda bb: 0.5 * float(np.abs((bb["hi"] - bb["lo"])[:2]) @ np.abs(u[:2]))
            gap0 = self.rng.uniform(0.03, 0.05)
            pre_obj = np.r_[pm[:2] - u[:2] * (half(bn) + half(bm) + gap0), z_obj]
            shift = pre_obj - po
            # the real touching distance from there (collision geometry, not boxes)
            hit_d = self._contact_offset(n, m, u, 0.16, shift=shift)
            if __import__("os").environ.get("DEBUG_BUMP"):
                d_pre, own_pre = self._clear(np.array([pc.eef_pos + shift]), np.array([pc.eef_mat]), moving, lambda t, o: o == m)
                print(f"   BUMPDIR off={off:+.1f} hit_d={hit_d} pre-clearance={d_pre:.3f} ({own_pre}) pre-z-tbl={pre_obj[2]-pc.table_z:.3f}", flush=True)
            if hit_d is None:
                continue
            pre = pc.eef_pos + shift
            # keep the hand out of the folded-up zone next to the robot
            if np.linalg.norm(pre[:2] - np.array(pc.robot.base_pos)[:2]) < REACH[0] - 0.07:
                continue
            dist = np.linalg.norm(pre - pc.eef_pos)
            T = float(np.clip(1.2 + 3.5 * dist, 1.2, 3.5))
            keys = [(T - 0.5, pre + np.r_[0, 0, 0.05]), (T, pre)]
            # knocking things about is the point here: brushing another loose
            # thing on the way in is fine (fixtures and the table stay strict,
            # they own the joint/reach checks' -0.02 too)
            ok = self.checked_move(keys, pc.eef_mat, 1.0, moving=moving, ignore_owner=m, ignore_last=6,
                                   ignore_first=(("table", sup, n), 10), min_clear=-0.01)
            if not ok and style == "slide":
                # skimming the table: only the last part comes close to it
                ok = self.checked_move(keys, pc.eef_mat, 1.0, moving=moving, ignore_owner=m, ignore_last=6,
                                       ignore_first=(("table", sup, n), 999), min_clear=-0.01)
            if ok:
                done = True
                break
        if not done:
            self.plan_log.append(("bump", "no way to it"))
            self._bump_note = "no way to it"
            return True     # just carry it on to where it was going
        # re-measure from where it actually is now
        hit_d = self._contact_offset(n, m, u, 0.18)
        if hit_d is None:
            self.plan_log.append(("bump", "would miss"))
            self._bump_note = "would miss"
            return True
        gap = hit_d
        # the hit: through the gap and a little further, at a walking pace -
        # or, using it as a tool, a slow long shove that keeps contact
        if getattr(self, "_bump_mode", None) == "push":
            push = self.rng.uniform(0.07, 0.14)
            speed = self.rng.uniform(0.04, 0.07)
        elif getattr(self, "_bump_fast", False):
            # injected error: swinging the held thing into something, hard
            push = self.rng.uniform(0.05, 0.09)
            speed = self.rng.uniform(0.18, 0.28)
        else:
            push = self.rng.uniform(0.02, 0.06)
            speed = self.rng.uniform(0.08, 0.16)
        through = pc.eef_pos + u * (gap + push) + np.r_[0, 0, 0.002]
        pc.phase = "push"
        c.move([((gap + push) / speed + 0.3, through)], pc.eef_mat, grip=1.0, precise_end=False, residual=0.2)
        if __import__("os").environ.get("DEBUG_BUMP"):
            print(f"   BUMP after: eef moved {np.linalg.norm(pc.eef_pos - pre):.3f} of {gap + push:.3f}; target moved {np.linalg.norm(self.obj_pose(m)[0][:2] - pm[:2]):.3f}", flush=True)
        self._bump_note = "executed"
        c.hold(int(self.rng.integers(2, 6)), 1.0)
        pc.phase = "transport"
        c.move([(0.7, pc.eef_pos - u * 0.03 + np.r_[0, 0, self.rng.uniform(0.04, 0.07)])], pc.eef_mat, grip=1.0,
               precise_end=False)
        self.cur_targets = {n}
        return pc.is_grasped(n)

    def _contact_offset(self, n, m, u, max_d, step=0.004, shift=None):
        """How far the held object `n` can move along u before touching `m`
        (None if it would pass by)."""
        from collision import _gdist
        pc = self.pc
        mm, d = pc.m, pc.d
        gn, gm = list(pc._obj_geoms[n]), list(pc._obj_geoms[m])
        saved = [d.geom_xpos[g].copy() for g in gn]
        try:
            sh = np.zeros(3) if shift is None else shift
            for k in np.arange(0.0, max_d, step):
                for g, xp in zip(gn, saved):
                    d.geom_xpos[g] = xp + sh + u * k
                if min(_gdist(mm, d, a, b, 0.02, None) for a in gn for b in gm) < 0.001:
                    return float(k)
        finally:
            for g, xp in zip(gn, saved):
                d.geom_xpos[g] = xp
        return None

    # ------------------------------------------------------- door into things
    def door_sweep_spot(self, m, opening):
        """A table spot in the microwave door's swing where `m` will be hit."""
        pc, m_, d = self.pc, self.pc.m, self.pc.d
        f = self.fixture_named("microwave")
        if f is None:
            return None
        jid, q = self._joint_q(f"{f}_microjoint")
        door = m_.jnt_bodyid[jid]
        hinge = d.xpos[door][:2].copy()
        dg = [g for g in range(m_.ngeom) if m_.geom_bodyid[g] == door and m_.geom_type[g] == 6
              and (m_.geom_contype[g] or m_.geom_conaffinity[g])]
        panel = max(dg, key=lambda g: np.prod(m_.geom_size[g]))
        pg = d.geom_xpos[panel]
        door_bottom = pg[2] - self._box_extent(panel)[2]
        cat = self.cat(m)
        if cat not in GEOM:
            return None
        h = GEOM[cat]["hi"][2] - GEOM[cat]["lo"][2]
        if pc.table_z + h < door_bottom + 0.012:
            return None  # too low: the door would pass over it
        r_m = self._radius(m)
        # where the door will pass: part-way along the swing that is coming
        # (opening from shut, or shutting from open), out toward the free edge
        th = -self.rng.uniform(0.35, 0.55) if opening else q * self.rng.uniform(0.3, 0.6)
        # the door's actual pose at that angle (scratch copy of the simulator,
        # so the joint's sign conventions don't matter)
        import mujoco
        rc = self._ensure_reach()
        rc.d2.qpos[:] = d.qpos
        rc.d2.qpos[m_.jnt_qposadr[jid]] = th
        mujoco.mj_kinematics(m_, rc.d2)
        hinge_t = rc.d2.xpos[door][:2].copy()
        pg_t = rc.d2.geom_xpos[panel][:2].copy()
        frac = self.rng.uniform(0.6, 0.85)
        xy = hinge_t + 2 * (pg_t - hinge_t) * frac
        # must be on the table, reachable, and clear of everything but the door
        base = np.array(pc.robot.base_pos)[:2]
        if not (REACH[0] < np.linalg.norm(xy - base) < REACH[1]):
            return None
        return xy, th

    def p_door_sweep(self, m):
        """Put `m` where the microwave door will swing, then swing it into it."""
        pc = self.pc
        mq = self.microwave_q()
        if mq is None:
            return None
        opening = mq > -0.3
        spot = None
        for _ in range(8):
            spot = self.door_sweep_spot(m, opening)
            if spot is not None:
                break
        if spot is None:
            self.plan_log.append(("door_sweep", "no spot"))
            return None
        xy, th = spot
        cat = self.cat(m)
        # exactly there (no substitute spot): it has to stand in the door's way
        dest = ("spot", np.r_[xy, pc.table_z + bottom_offset(cat) + 0.003], "table")
        f = self.fixture_named("microwave")
        self.min_move = 0.0
        place = self.p_pick_place(m, dest=dest, prim="move", lang=f"put the {_nice(m)} in front of the microwave door")
        if place is None or not place.get("success"):
            return place
        pm0 = self.obj_pose(m)[0].copy()
        swing = self.p_microwave(opening, q_target=(th - self.rng.uniform(0.3, 0.5)) if opening else None)
        if swing is None:
            place["kind"] = "door_sweep"
            return place
        shoved = float(np.linalg.norm(self.obj_pose(m)[0][:2] - pm0[:2]))
        verb = "open" if opening else "close"
        swing.update(primitive="door_sweep", interact=[m], success=bool(shoved > 0.01),
                     language=f"put the {_nice(m)} in front of the microwave door and {verb} the door into it",
                     diag=dict(swing.get("diag", {}), shoved=shoved, placed_start=place.get("start")))
        swing["start"] = place.get("start")
        return swing
