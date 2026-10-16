"""Exact-geometry collision checking for planned end-effector paths.

The gripper's collision geoms (hand, fingers, last wrist link) are rigidly
attached to the eef site, so for a planned eef pose we can place them exactly
(using the current finger opening) and query MuJoCo's signed distance
(mj_geomDistance) to every obstacle geom: other objects, fixtures, the table.
A carried object's geoms can be attached the same way."""
import mujoco
import numpy as np



_NATIVECCD = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)


def _gdist(m, d, g1, g2, distmax, fromto):
    """mj_geomDistance, cross-checked between MuJoCo's two convex colliders.
    In MuJoCo 3.3 the native CCD returns spurious exact 0.0 distances for many
    separated pairs, while the legacy (libccd) path returns bogus deep
    penetrations for some separated thin boxes. Neither has been seen to report
    a positive distance for overlapping geoms, so: a positive native distance
    is trusted; an exact 0 is re-measured with the legacy collider."""
    M = m._model if hasattr(m, "_model") else m
    dn = mujoco.mj_geomDistance(M, d, g1, g2, distmax, fromto)
    if dn != 0.0:
        return dn
    old = M.opt.disableflags
    M.opt.disableflags = old | _NATIVECCD
    try:
        return mujoco.mj_geomDistance(M, d, g1, g2, distmax, fromto)
    finally:
        M.opt.disableflags = old


class CollisionChecker:
    def __init__(self, pc):
        self.pc = pc
        m, d = pc.m, pc.d
        self.m, self.d = m, d
        # bodies in the subtree of the last wrist link
        root = m.body("robot0_link7").id
        sub = set()
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == root:
                    sub.add(b)
                    break
                x = m.body_parentid[x]
        self.robot_bodies = sub
        self.grip_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub
                           and (m.geom_contype[g] or m.geom_conaffinity[g])]
        # obstacle geoms per owner (object / fixture / table)
        e = pc.e
        self.owner = {}
        self.table_geoms = []
        for name, obj in list(e.objects_dict.items()) + list(e.fixtures_dict.items()):
            rb = m.body(obj.root_body).id
            gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == rb
                  and (m.geom_contype[g] or m.geom_conaffinity[g])]
            if name.endswith("table"):
                self.table_geoms += gs
            else:
                for g in gs:
                    self.owner[g] = name
        self.obst_geoms = list(self.owner) + self.table_geoms
        for g in self.table_geoms:
            self.owner[g] = "table"

    # -------------------------------------------------------------- helpers
    def _aabb(self, g):
        m, d = self.m, self.d
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        ctr = d.geom_xpos[g] + R @ c
        ext = np.abs(R) @ h
        return ctr - ext, ctr + ext

    def attach_rel(self, geoms):
        """Pose of `geoms` relative to the current eef site frame."""
        pc = self.pc
        pe, Re = pc.eef_pos, pc.eef_mat
        out = []
        for g in geoms:
            out.append((g, Re.T @ (self.d.geom_xpos[g] - pe), Re.T @ self.d.geom_xmat[g].reshape(3, 3)))
        return out

    def gripper_rel(self):
        return self.attach_rel(self.grip_geoms)

    @staticmethod
    def swing_frame(R):
        """Frame that follows the gripper's approach axis but not its twist about
        it (the twist is joint 7, which link 6 does not see)."""
        z = R[:, 2]
        ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = ref - (ref @ z) * z; x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z], 1)

    def forearm_rel(self):
        """Link-6 collision geom relative to the eef position and the swing frame."""
        m = self.m
        pe, Re = self.pc.eef_pos, self.pc.eef_mat
        S = self.swing_frame(Re)
        out = []
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name == "robot0_link6" and m.geom_contype[g]:
                out.append((g, S.T @ (self.d.geom_xpos[g] - pe), S.T @ self.d.geom_xmat[g].reshape(3, 3), "swing"))
        return out

    def pose_clearance(self, p, R, moving, obstacles):
        """Distance between `moving` geoms placed at one pose and given geoms."""
        m, d = self.m, self.d
        moving = [mv if len(mv) == 4 else (mv[0], mv[1], mv[2], "rigid") for mv in moving]
        saved = [(d.geom_xpos[g].copy(), d.geom_xmat[g].copy()) for g, *_ in moving]
        best = np.inf
        try:
            for g, rp, rm, _ in moving:
                d.geom_xpos[g] = p + R @ rp
                d.geom_xmat[g] = (R @ rm).ravel()
            for g, *_ in moving:
                for og in obstacles:
                    best = min(best, _gdist(m, d, g, og, 0.05, None))
        finally:
            for (g, *_), (xp, xm) in zip(moving, saved):
                d.geom_xpos[g] = xp
                d.geom_xmat[g] = xm
        return best

    def swept_clearance(self, geoms, offsets, ignore_owner):
        """Translate `geoms` (e.g. a drawer body) by each offset; min distance to
        obstacles not owned by `ignore_owner`."""
        m, d = self.m, self.d
        saved = [(d.geom_xpos[g].copy(), d.geom_xmat[g].copy()) for g in geoms]
        best = (np.inf, None)
        try:
            for off in offsets:
                for g, (xp, _) in zip(geoms, saved):
                    d.geom_xpos[g] = xp + off
                ign = ignore_owner if isinstance(ignore_owner, (set, frozenset, list, tuple)) else {ignore_owner}
                for og in self.obst_geoms:
                    own = self.owner[og]
                    if own in ign or own == "table" or og in geoms:
                        continue
                    for g in geoms:
                        dist = _gdist(m, d, g, og, 0.05, None)
                        if dist < best[0]:
                            best = (dist, own)
        finally:
            for g, (xp, xm) in zip(geoms, saved):
                d.geom_xpos[g] = xp
                d.geom_xmat[g] = xm
        return best

    # -------------------------------------------------------------- query
    def path_clearance(self, P, R, moving, ignore=lambda t, owner: False, stride=2, distmax=0.08):
        """Minimum signed distance between `moving` geoms (list of (geom, rel_pos,
        rel_mat) attached to the eef) placed along the path and all obstacles,
        except those `ignore(t, owner)` says to skip. Returns (min_dist, t, owner)."""
        m, d = self.m, self.d
        moving = [mv if len(mv) == 4 else (mv[0], mv[1], mv[2], "rigid") for mv in moving]
        mov_ids = [g for g, _, _, _ in moving]
        saved = [(d.geom_xpos[g].copy(), d.geom_xmat[g].copy()) for g in mov_ids]
        obst = [g for g in self.obst_geoms if g not in mov_ids]
        boxes = {g: self._aabb(g) for g in obst}
        best = (np.inf, -1, None)
        idxs = list(range(0, len(P), stride))
        if idxs[-1] != len(P) - 1:
            idxs.append(len(P) - 1)
        try:
            for t in idxs:
                p, Rt = P[t], R[t]
                St = None
                for g, rp, rm, kind in moving:
                    if kind == "swing":
                        if St is None:
                            St = self.swing_frame(Rt)
                        d.geom_xpos[g] = p + St @ rp
                        d.geom_xmat[g] = (St @ rm).ravel()
                    else:
                        d.geom_xpos[g] = p + Rt @ rp
                        d.geom_xmat[g] = (Rt @ rm).ravel()
                mlo = np.min([d.geom_xpos[g] for g in mov_ids], 0) - 0.12 - distmax
                mhi = np.max([d.geom_xpos[g] for g in mov_ids], 0) + 0.12 + distmax
                for og in obst:
                    own = self.owner[og]
                    if ignore(t, own):
                        continue
                    lo, hi = boxes[og]
                    if np.any(hi < mlo) or np.any(lo > mhi):
                        continue
                    for g in mov_ids:
                        dist = _gdist(m, d, g, og, distmax, None)
                        if dist < best[0]:
                            best = (dist, t, own)
                            self.last_pair = (m.geom(g).name or m.body(m.geom_bodyid[g]).name, m.geom(og).name or m.body(m.geom_bodyid[og]).name)
        finally:
            for g, (xp, xm) in zip(mov_ids, saved):
                d.geom_xpos[g] = xp
                d.geom_xmat[g] = xm
        return best
