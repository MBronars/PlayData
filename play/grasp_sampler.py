"""Geometry-driven grasp generation with exact verification.

For the object's collision geoms at their current pose, fit an oriented box
(PCA over geom bounding-box corners). Candidate parallel-jaw grasps close
across any box axis that fits in the gripper (any direction around the axis of
round objects), with approach directions swept in the plane perpendicular to
the closing axis (up to 70 deg from straight down). Rim grasps for bowls / mugs
/ plates use the object's own up axis. Every candidate is verified with the
real gripper collision geometry: open fingers must clear the object (so they
straddle it), the hand must clear the table and neighbours, and the pre-grasp
pose backed off along the approach axis must be clear too."""
import numpy as np

from human_play import GEOM, GROUP_OF

GRIP_MAX = 0.066        # comfortable opening between the fingers' collision hulls
GRIP_HARD = 0.072       # widest closing width tried (humans grab 6.4 cm cans); exact clearance check decides
FINGER_REACH = 0.027    # how far the object may extend into the hand past the pad centre (palm hull is 3.1 cm up)
ROUND = {"can", "bowl", "plate"}
RIM = {"bowl", "mug", "plate"}


def _obb(pc, geoms):
    m, d = pc.m, pc.d
    pts = []
    for g in geoms:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        cs = c + h * np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)])
        pts.append(cs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    pts = np.concatenate(pts)
    ctr = pts.mean(0)
    U, S, Vt = np.linalg.svd(pts - ctr, full_matrices=False)
    axes = Vt  # rows
    proj = (pts - ctr) @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    ctr = ctr + axes.T @ ((lo + hi) / 2)
    return ctr, axes, hi - lo


def _frame(close, app):
    """Gripper rotation: col0 = closing axis, col2 = approach axis (palm->tips)."""
    app = app / np.linalg.norm(app)
    close = close - (close @ app) * app
    close /= np.linalg.norm(close)
    return np.stack([close, np.cross(app, close), app], 1)


class GraspSampler:
    def __init__(self, play):
        self.p = play
        self.pc = play.pc
        self.cc = play.cc
        if not hasattr(play, "reach"):
            from reach import Reach
            play.reach = Reach(play.pc)
        self.reach = play.reach

    def _verify(self, n, pos, R, open_rel):
        pc, cc = self.pc, self.cc
        m = pc.m
        # big tilts only when the object itself is tilted
        tilt = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
        max_tilt = getattr(self, "_max_tilt", None) or (70 if self.obj_tilt(n) > 15 else 45)
        if tilt > max_tilt:
            self._why = "tilt"
            return None
        obj = list(pc._obj_geoms[n])
        sup_in = self.p.support_of(n) is not None
        # fingers straddle the object: pads and hand must clear it; the finger
        # meshes' convex hulls are wider than the real fingers, allow some overlap
        hulls = [x for x in open_rel if m.geom(x[0]).name.endswith(("finger1_collision", "finger2_collision"))]
        rest = [x for x in open_rel if x not in hulls]
        d_obj = cc.pose_clearance(pos, R, rest, obj)
        d_hull = cc.pose_clearance(pos, R, hulls, obj)
        if d_obj < (0.0 if sup_in else 0.0015) or d_hull < -0.045:
            if __import__("os").environ.get("DEBUG_VERIFY"):
                per = sorted((round(cc.pose_clearance(pos, R, [x], obj), 4), self.pc.m.geom(x[0]).name[8:]) for x in open_rel)[:3]
                self._why = "obj " + str(per)
            else:
                self._why = "obj"
            return None
        others = [g for g in cc.obst_geoms if g not in obj and cc.owner[g] != "table"]
        table = cc.table_geoms
        d_tab = cc.pose_clearance(pos, R, open_rel, table) if table else 1.0
        if d_tab < 0.002:
            self._why = "table"
            return None
        # an object inside a container: its walls may be close (fingers slide
        # down beside it into the slot), other things keep the full margin
        sup = self.p.support_of(n)
        sup_g = [g for g in others if cc.owner[g] == sup] if sup is not None else []
        rest_g = [g for g in others if g not in sup_g]
        d_sup = cc.pose_clearance(pos, R, open_rel, sup_g) if sup_g else 1.0
        # if it is already half out of the container (resting on a rim or a
        # divider), brushing the container a little on the way is fine
        allow = 0.005
        if sup is not None and getattr(self, "_sticking", {}).get(n) is None:
            from human_play import GEOM as _G
            cs = self.p.cat(sup)
            top = self.p.obj_pose(sup)[0][2] + (_G[cs]["hi"][2] if cs in _G else 0.05)
            self.__dict__.setdefault("_sticking", {})[n] = bool(pc.obj_bbox(n)["hi"][2] > top + 0.01)
        if sup is not None and self._sticking.get(n):
            allow = 0.023
        d_oth = min(cc.pose_clearance(pos, R, open_rel, rest_g) if rest_g else 1.0, d_sup + allow)
        if d_oth < 0.008:
            self._why = "others"
            if __import__("os").environ.get("DEBUG_VERIFY"):
                self._why = f"others sup={d_sup:.4f} rest={(cc.pose_clearance(pos, R, open_rel, rest_g) if rest_g else 1.0):.4f} worst={sorted((round(cc.pose_clearance(pos, R, [x], sup_g or rest_g), 4), self.pc.m.geom(x[0]).name[8:]) for x in open_rel)[:2]} close={np.round(R[:, 0], 2)}"
            return None
        pre = pos - R[:, 2] * 0.05
        d_pre = min(cc.pose_clearance(pre, R, open_rel, obj + others), cc.pose_clearance(pre, R, open_rel, table) if table else 1.0)
        if d_pre < 0.004:
            self._why = "pre"
            return None
        # the arm must actually be able to get there (grasp and pre-grasp), with
        # some room left at the joint limits
        if self.reach.wrist_dist(pos, R)[0] > self.reach.WRIST_MAX:
            self._why = "wrist"
            return None
        bf = self.reach.best_flip(pos, R)
        if bf is None or bf[0] < 0.08 or not self.reach.ok(pre, bf[1]):
            self._why = "ik"
            return None
        self._last_flip = bf
        return min(d_obj, d_oth)

    @staticmethod
    def _jpen(jm):
        """Prefer grasps that leave room at the joint limits (soft)."""
        return 2.0 * max(0.0, 0.35 - jm)

    def obj_tilt(self, n):
        try:
            return float(np.degrees(np.arccos(np.clip(self.p.rel_R(n)[2, 2], -1, 1))))
        except Exception:
            return 0.0

    def _score(self, n, pos, R, width, depth_frac):
        base = np.array(self.pc.robot.base_pos)
        app = R[:, 2]
        tilt = np.degrees(np.arccos(np.clip(-app[2], -1, 1)))
        # approach coming from the robot's side is easier and more natural
        horiz = app[:2] / (np.linalg.norm(app[:2]) + 1e-9)
        away = (pos[:2] - base[:2]) / (np.linalg.norm(pos[:2] - base[:2]) + 1e-9)
        side = 0.0 if tilt < 10 else (1 - horiz @ away) * 0.5
        yaw = np.arctan2(R[1, 0], R[0, 0])
        # people grasp from (nearly) straight above unless the object itself is
        # tilted: penalize tilt relative to the object's own up axis, quadratically
        try:
            up = self.p.rel_R(n)[:, 2]
            rel = np.degrees(np.arccos(np.clip(-app @ up, -1, 1)))
            tilt = min(tilt, rel)
        except Exception:
            pass
        # people don't twist the hand round more than they need to: among
        # comparable grasps, prefer the one closest to how the hand is held now
        from compose import rot_err, FLIP
        cur = self.pc.eef_mat
        turn = min(np.linalg.norm(rot_err(R, cur)), np.linalg.norm(rot_err(R @ FLIP, cur)))
        # narrow closing widths hold better than near-max openings
        return 0.45 * turn ** 2 + (tilt / 22.0) ** 2 + 0.1 * tilt / 22.0 + side + 0.5 * abs(depth_frac - 0.5) + 1.2 * width / GRIP_MAX \
            + 4.0 * max(0.0, width - GRIP_MAX) / (GRIP_HARD - GRIP_MAX)

    def box_grasps(self, n, n_app=7):
        pc = self.pc
        obj = list(pc._obj_geoms[n])
        ctr, axes, ext = _obb(pc, obj)
        cat = self.p.cat(n)
        grp = GROUP_OF.get(cat)
        open_rel = self.p.open_rel
        out = []
        close_axes = []
        for i in range(3):
            if ext[i] <= GRIP_HARD:
                close_axes.append((axes[i], ext[i]))
        if grp in ROUND or cat in ("alphabet_soup", "tomato_sauce", "wine_bottle"):
            # round cross-section: any direction around the long axis
            k = int(np.argmax(ext)) if grp not in ("bowl", "plate") else int(np.argmin(ext))
            u = axes[k]
            o1 = axes[(k + 1) % 3]; o2 = np.cross(u, o1)
            w = max(ext[(k + 1) % 3], ext[(k + 2) % 3])
            if w <= GRIP_HARD:
                for a in np.linspace(0, np.pi, 6, endpoint=False):
                    close_axes.append((np.cos(a) * o1 + np.sin(a) * o2, w))
        for cax, width in close_axes:
            # approach directions perpendicular to the closing axis, pointing down-ish
            e1 = np.cross(cax, [0, 0, 1.0])
            if np.linalg.norm(e1) < 1e-3:
                continue  # closing axis vertical: can't come from above
            e1 /= np.linalg.norm(e1)
            down = np.cross(e1, cax); down /= np.linalg.norm(down)
            if down[2] > 0:
                down = -down
            for ang in np.radians([0, 20, -20, 40, -40, 60, -60])[:n_app]:
                app = np.cos(ang) * down + np.sin(ang) * e1
                if app[2] > -np.cos(np.radians(70)):
                    continue
                # extent of the object along the approach axis and along the free axis
                proj_app = abs(axes @ app) @ ext / 1.0
                ext_app = float(np.abs(axes.T @ (axes @ app * ext)).sum()) if False else float(np.sum(np.abs(axes @ app) * ext))
                free = np.cross(app, cax); free /= np.linalg.norm(free)
                for slide in (0.0, 0.25, -0.25):
                    depth = min(0.5 * ext_app, FINGER_REACH)
                    pos = ctr - app * (0.5 * ext_app - depth)
                    ext_free = float(np.sum(np.abs(axes @ free) * ext))
                    pos = pos + free * slide * max(ext_free - 0.03, 0) * 0.5
                    R = _frame(cax, app)
                    # the gripper is symmetric: pick the closing direction nearer the current wrist
                    if np.trace(R.T @ pc.eef_mat) < np.trace((R @ np.diag([-1, -1, 1])).T @ pc.eef_mat):
                        R = R @ np.diag([-1.0, -1.0, 1.0])
                    pos[2] = max(pos[2], pc.table_z + 0.011)
                    v = self._verify(n, pos, R, open_rel)
                    if v is None:
                        continue
                    jm, R = self._last_flip
                    out.append((self._score(n, pos, R, width, depth / max(ext_app, 1e-3)) + self._jpen(jm), pos, R, "box", width, depth))
        return out

    def rim_grasps(self, n):
        pc = self.pc
        cat = self.p.cat(n)
        if cat not in GEOM:
            return []
        g = GEOM[cat]
        p, q = self.p.obj_pose(n)
        Rr = self.p.rel_R(n)
        up = Rr[:, 2]
        lo, hi = np.array(g["lo"]), np.array(g["hi"])
        top = p + Rr @ np.r_[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]]
        ext = hi[:2] - lo[:2]
        grp = GROUP_OF.get(cat)
        r = 0.5 * min(ext) - (0.008 if grp == "plate" else 0.006)   # on the wall
        h = hi[2] - lo[2]
        depths = (0.01, 0.007) if grp == "plate" else tuple(d_ for d_ in (0.025, 0.019, 0.014, 0.01) if d_ < 0.75 * h)
        e1 = np.cross(up, [1.0, 0, 0]);
        if np.linalg.norm(e1) < 1e-3:
            e1 = np.cross(up, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        hdir = None
        if grp == "mug":
            long_ax = Rr @ (np.r_[1, 0, 0] if ext[0] > ext[1] else np.r_[0, 1, 0])
            hdir = long_ax / np.linalg.norm(long_ax)
        out = []
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            u = np.cos(a) * e1 + np.sin(a) * e2
            if hdir is not None and abs(u @ hdir) > 0.7:
                continue  # mug handle side
            for blend in (0.0, 0.35):
                app = -up * (1 - blend) + np.r_[0, 0, -1.0] * blend
                if app[2] > -np.cos(np.radians(70)):
                    continue
                R = _frame(u, app)
                if np.trace(R.T @ pc.eef_mat) < np.trace((R @ np.diag([-1, -1, 1])).T @ pc.eef_mat):
                    R = R @ np.diag([-1.0, -1.0, 1.0])
                for depth in depths:
                    pos2 = top + u * r - up * depth
                    pos2[2] = max(pos2[2], pc.table_z + 0.011)
                    v = self._verify(n, pos2, R, self.p.open_rel)
                    if v is not None:
                        break
                if v is None:
                    continue
                jm, R = self._last_flip
                base = np.array(pc.robot.base_pos)
                facing = -(u[:2] @ ((base[:2] - top[:2]) / (np.linalg.norm(base[:2] - top[:2]) + 1e-9)))
                out.append((self._score(n, pos2, R, 0.02, 0.5) + 0.3 * facing + self._jpen(jm), pos2, R, "rim", 0.0, depth))
        return out

    def part_grasps(self, n, max_parts=16):
        """Grasp individual collision parts (a wall slat = rim grasp, a handle
        piece = handle grasp, a neck = neck grasp)."""
        pc = self.pc
        m, d = pc.m, pc.d
        geoms = [g for g in pc._obj_geoms[n] if m.geom_type[g] == 6]
        # thin slats are only meaningful as walls (mugs, bowls); on a bottle they
        # are decoration around the cap - grab the cap / neck itself instead
        thin_ok = GROUP_OF.get(self.p.cat(n)) in RIM
        out = []
        for g in self.p.rng.permutation(geoms)[:max_parts]:
            Rg = d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            ctr = d.geom_xpos[g]
            order = np.argsort(size)
            axes_ok = [order[0]] + ([order[1]] if size[order[0]] > 0.5 * size[order[1]] else [])
            for i in axes_ok:
                if 2 * size[i] > GRIP_MAX - 0.006 or (not thin_ok and 2 * size[i] < 0.015):
                    continue
                cax = Rg[:, i]
                e1 = np.cross(cax, [0, 0, 1.0])
                if np.linalg.norm(e1) < 1e-3:
                    continue
                e1 /= np.linalg.norm(e1)
                down = np.cross(e1, cax); down /= np.linalg.norm(down)
                if down[2] > 0:
                    down = -down
                for ang in np.radians([0, 25, -25, 50, -50]):
                    app = np.cos(ang) * down + np.sin(ang) * e1
                    if app[2] > -np.cos(np.radians(70)):
                        continue
                    ext_app = float(np.sum(np.abs(Rg.T @ app) * 2 * size))
                    if ext_app < 0.03:
                        continue  # too little material to hold on to
                    R0 = _frame(cax, app)
                    if np.trace(R0.T @ pc.eef_mat) < np.trace((R0 @ np.diag([-1, -1, 1])).T @ pc.eef_mat):
                        R0 = R0 @ np.diag([-1.0, -1.0, 1.0])
                    # deepest hold the palm allows (rims and lips stop it early)
                    for depth in (min(0.5 * ext_app, FINGER_REACH, 0.025), 0.019, 0.014):
                        pos = ctr - app * (0.5 * ext_app - depth)
                        pos[2] = max(pos[2], pc.table_z + 0.011)
                        if self._verify(n, pos, R0, self.p.open_rel) is not None:
                            break
                    else:
                        continue
                    jm, R = self._last_flip
                    out.append((self._score(n, pos, R, 2 * size[i], 0.5) + 0.2 + self._jpen(jm), pos, R, "part", 2 * size[i], depth))
        return out

    # ------------------------------------------------------- unusual grasps
    def handle_grasps(self, n):
        """Mug by its handle: pinching the handle's bar from above, at an
        angle, or from the outside (the hand nearly horizontal)."""
        pc = self.pc
        m, d = pc.m, pc.d
        cat = self.p.cat(n)
        if GROUP_OF.get(cat) != "mug" or cat not in GEOM:
            return []
        g = GEOM[cat]
        p, _ = self.p.obj_pose(n)
        Rr = self.p.rel_R(n)
        up = Rr[:, 2]
        lo, hi = np.array(g["lo"]), np.array(g["hi"])
        ext = hi[:2] - lo[:2]
        ctr = p + Rr @ np.r_[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
        long_ax = Rr @ (np.r_[1, 0, 0] if ext[0] > ext[1] else np.r_[0, 1, 0])
        r_body = 0.5 * min(ext)
        geoms = list(pc._obj_geoms[n])
        offs = [(float((d.geom_xpos[x] - ctr) @ long_ax), x) for x in geoms]
        # the handle is on the side whose geoms stick out furthest
        sgn = 1.0 if max(o for o, _ in offs) > -min(o for o, _ in offs) else -1.0
        hout = sgn * long_ax
        hg = [x for o, x in offs if sgn * o > r_body * 0.85]
        if not hg:
            return []
        cax = np.cross(up, hout); cax /= np.linalg.norm(cax)
        out = []
        self._max_tilt = 88
        was = getattr(self.reach, "relaxed", False)
        self.reach.relax(True)
        try:
            for x in hg:
                c = d.geom_xpos[x].copy()
                for t in (0, 35, 65, 85):
                    app = -np.cos(np.radians(t)) * up - np.sin(np.radians(t)) * hout
                    R = _frame(cax, app)
                    if np.trace(R.T @ pc.eef_mat) < np.trace((R @ np.diag([-1, -1, 1])).T @ pc.eef_mat):
                        R = R @ np.diag([-1.0, -1.0, 1.0])
                    pos = c.copy()
                    pos[2] = max(pos[2], pc.table_z + 0.012)
                    if self._verify(n, pos, R, self.p.open_rel) is None:
                        continue
                    jm, R = self._last_flip
                    out.append((0.2 * t / 85 + 0.45 * self._turn(R) ** 2 + self._jpen(jm), pos, R, "handle", 0.01, 0.0))
                    self._add_flip(out, pos, R)
        finally:
            self._max_tilt = None
            self.reach.relax(was)
        return out

    def side_grasps(self, n):
        """From the side: the hand comes in (nearly) horizontally and closes
        around the body, as people grab a can or a bottle."""
        pc = self.pc
        cat = self.p.cat(n)
        grp = GROUP_OF.get(cat)
        if grp in ("plate", "bowl", "basket", "container") or self.obj_tilt(n) > 20:
            return []
        obj = list(pc._obj_geoms[n])
        ctr, axes, ext = _obb(pc, obj)
        bb = pc.obj_bbox(n)
        h = bb["hi"][2] - pc.table_z
        if h < 0.05:
            return []    # too flat: the fingers would hit the table
        base = np.array(pc.robot.base_pos)
        away = (ctr[:2] - base[:2]); away /= np.linalg.norm(away) + 1e-9
        out = []
        self._max_tilt = 88
        was = getattr(self.reach, "relaxed", False)
        self.reach.relax(True)
        try:
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                ah = np.r_[np.cos(a), np.sin(a), 0.0]
                if ah[:2] @ away < -0.3:
                    continue     # reaching around the far side of it
                for t, ci in [(t_, c_) for t_ in (45, 55, 65) for c_ in (0, 1)]:
                    app = -np.cos(np.radians(t)) * np.r_[0, 0, 1.0] + np.sin(np.radians(t)) * ah
                    cax = np.r_[-ah[1], ah[0], 0.0]
                    if ci == 1:
                        # fingers closing front-to-back instead of left-to-right
                        cax = np.cross(app, cax); cax /= np.linalg.norm(cax)
                    width = float(np.sum(np.abs(axes @ cax) * ext))
                    if width > GRIP_HARD:
                        continue
                    ext_app = float(np.sum(np.abs(axes @ app) * ext))
                    depth = min(0.5 * ext_app, FINGER_REACH)
                    for zf in (0.55, 0.4, 0.7):
                        pos = ctr - app * (0.5 * ext_app - depth)
                        pos[2] = pc.table_z + max(zf * h, 0.03)
                        R = _frame(cax, app)
                        if np.trace(R.T @ pc.eef_mat) < np.trace((R @ np.diag([-1, -1, 1])).T @ pc.eef_mat):
                            R = R @ np.diag([-1.0, -1.0, 1.0])
                        if self._verify(n, pos, R, self.p.open_rel) is None:
                            continue
                        jm, R = self._last_flip
                        out.append((0.8 * width / GRIP_MAX + 0.3 * abs(ah[:2] @ away) + 0.45 * self._turn(R) ** 2 + self._jpen(jm), pos, R, "side", width, depth))
                        self._add_flip(out, pos, R)
                        break
        finally:
            self._max_tilt = None
            self.reach.relax(was)
        return out

    def _turn(self, R):
        from compose import rot_err, FLIP
        cur = self.pc.eef_mat
        return min(np.linalg.norm(rot_err(R, cur)), np.linalg.norm(rot_err(R @ FLIP, cur)))

    def _add_flip(self, out, pos, R):
        """The same grasp with the hand turned 180 deg about its axis: which of
        the two the wrist can reach without winding up depends on the path."""
        Rf = R @ np.diag([-1.0, -1.0, 1.0])
        if self.reach.ok(pos, Rf):
            out.append((out[-1][0] + 0.05, pos, Rf) + tuple(out[-1][3:]))

    p_unusual = 0.5

    def candidates(self, n, k=10):
        grp = GROUP_OF.get(self.p.cat(n))
        near_body = np.linalg.norm(self.p.obj_pose(n)[0][:2] - np.array(self.pc.robot.base_pos)[:2]) < 0.48
        if self.p.rng.random() < self.p_unusual and not getattr(self.p, "_no_unusual", False) and not near_body:
            # trying it a different way: by the handle, or from the side
            un = self.handle_grasps(n) + self.side_grasps(n)
            if un:
                un.sort(key=lambda x: x[0])
                fams = sorted({x[3] for x in un})
                fam = fams[int(self.p.rng.integers(len(fams)))]
                un = [x for x in un if x[3] == fam]
                k_ = int(self.p.rng.integers(min(3, len(un))))
                first = [un[k_]]
                rest = self._usual(n, grp)
                c = first + [x for x in un if x is not un[k_]][:5] + rest
                self.p.grasp_info = [x[3:] for x in c[:k]]
                return [(x[1], x[2] if x[3] in ("handle", "side") else self._jitter(n, x[1], x[2])) for x in c[:k]]
        c = self._usual(n, grp)
        self.p.grasp_info = [x[3:] for x in c[:k]]
        return [(x[1], self._jitter(n, x[1], x[2])) for x in c[:k]]

    def _usual(self, n, grp):
        c = self.rim_grasps(n) if grp in RIM else []
        box = self.box_grasps(n)
        c += box
        # parts (single wall slats, handles) for hollow objects, or when nothing
        # else fits; solid objects are held much better across their whole body
        # (a bottle by its neck from above, a box by one face)
        c += [(x[0] + (0.0 if grp in RIM or not box else 0.35),) + tuple(x[1:]) for x in self.part_grasps(n)]
        c.sort(key=lambda x: x[0])
        # grasp the same thing different ways: about half the time, one of the
        # near-equally good grasps (another side, a lower or higher hold, the
        # other face) goes first instead of always the single best one
        if len(c) > 1 and self.p.rng.random() < 0.5:
            n_near = sum(1 for x in c if x[0] <= c[0][0] + 0.6)
            if n_near > 1:
                k_ = int(self.p.rng.integers(n_near))
                c = [c[k_]] + c[:k_] + c[k_ + 1:]
        return c

    def _jitter(self, n, pos, R):
        """People never come in perfectly plumb: tilt the hand a few degrees
        about the closing axis (fingers stay on the same contact line) and a
        little about the other axis; keep it only if it still verifies."""
        from scipy.spatial.transform import Rotation as _Rot
        rng = self.p.rng
        if __import__("os").environ.get("NO_JITTER"):
            return R
        a = np.radians(np.clip(rng.normal(0, 9.0), -18, 18))
        b = np.radians(np.clip(rng.normal(0, 3.0), -6, 6))
        for f in (1.0, 0.5):
            Rj = _Rot.from_rotvec(R[:, 0] * a * f + R[:, 1] * b * f).as_matrix() @ R
            if self._verify(n, pos, Rj, self.p.open_rel) is not None:
                return Rj
        return R
