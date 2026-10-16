"""Kinematic reachability via damped-least-squares IK on a scratch MjData copy
(the real simulation is never touched)."""
import mujoco
import numpy as np


# human demo joint range (0.5-99.5 pct) padded by ~0.35 rad, for the wrist joints
COMFORT = {4: (-1.1, 2.1), 5: (0.65, 3.4)}


class Reach:
    def __init__(self, pc):
        self.pc = pc
        self.m = pc.m
        self.d2 = mujoco.MjData(self.m)
        r = pc.robot
        self.qi = np.array(r._ref_joint_pos_indexes)
        self.vi = np.array(r._ref_joint_vel_indexes)
        jids = [self.m.dof_jntid[v] for v in self.vi]
        self.lo = self.m.jnt_range[jids, 0] + 0.03
        self.hi = self.m.jnt_range[jids, 1] - 0.03
        # wrist joints 5/6: stay in the range people use (outside it the arm is
        # on the flipped-wrist branch the controller can't get back from)
        self.lo[4], self.hi[4] = max(self.lo[4], COMFORT[4][0]), min(self.hi[4], COMFORT[4][1])
        self.lo[5], self.hi[5] = max(self.lo[5], COMFORT[5][0]), min(self.hi[5], COMFORT[5][1])
        # elbow: never (nearly) straight, as in the human demos
        self.hi[3] = min(self.hi[3], -0.4)
        self._lo0, self._hi0 = self.lo.copy(), self.hi.copy()
        self.site = pc.eef_site
        # the OSC controller pulls the arm's redundancy toward its initial posture;
        # do the same in the IK so predicted joint paths match executed ones
        try:
            self.q_home = np.array(pc.robot.controller.initial_joint, float)
        except Exception:
            self.q_home = np.array(pc.robot.init_qpos, float)
        self.null_gain = float(__import__('os').environ.get('IK_NULL', '0.0'))

    def solve(self, pos, R, q0=None, iters=60, damp=0.05):
        """Returns (pos_err, rot_err, q) for the closest joint configuration."""
        m, d2, pc = self.m, self.d2, self.pc
        d2.qpos[:] = pc.d.qpos
        if q0 is not None:
            d2.qpos[self.qi] = q0
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_comPos(m, d2)
            ep = pos - d2.site_xpos[self.site]
            Rc = d2.site_xmat[self.site].reshape(3, 3)
            Re = R @ Rc.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.01:
                break
            mujoco.mj_jacSite(m, d2, jp, jr, self.site)
            J = np.vstack([jp[:, self.vi], jr[:, self.vi]])
            e = np.r_[ep, 0.5 * er]
            Jp = J.T @ np.linalg.inv(J @ J.T + damp ** 2 * np.eye(6))
            dq = Jp @ e
            # null-space pull toward the controller's home posture
            dq += (np.eye(len(self.vi)) - Jp @ J) @ (self.null_gain * (self.q_home - d2.qpos[self.qi]))
            q = np.clip(d2.qpos[self.qi] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
            d2.qpos[self.qi] = q
        mujoco.mj_kinematics(m, d2)
        ep = np.linalg.norm(pos - d2.site_xpos[self.site])
        Rc = d2.site_xmat[self.site].reshape(3, 3)
        ang = np.arccos(np.clip((np.trace(R @ Rc.T) - 1) / 2, -1, 1))
        return ep, ang, d2.qpos[self.qi].copy()

    WRIST_MAX = 0.70   # m from the shoulder: beyond this the arm goes nearly straight

    def shoulder(self):
        return self.pc.d.xpos[self.pc.m.body("robot0_link2").id]

    def wrist_dist(self, P, R):
        """Distance from shoulder to the wrist (21 cm back along the approach axis)."""
        P = np.atleast_2d(P); R = np.asarray(R).reshape(-1, 3, 3)
        W = P - R[:, :, 2] * 0.21
        return np.linalg.norm(W - self.shoulder(), axis=1)

    RELAXED = {4: (-1.5, 2.4), 5: (0.45, 3.5)}

    def relax(self, on=True):
        """Unusual grasps (from the side, by a handle) need more of the wrist's
        range than people use for top-down picks."""
        self.lo, self.hi = self._lo0.copy(), self._hi0.copy()
        if on:
            jids = [self.m.dof_jntid[v] for v in self.vi]
            for j, (a, b) in self.RELAXED.items():
                self.lo[j] = max(self.m.jnt_range[jids[j], 0] + 0.03, a)
                self.hi[j] = min(self.m.jnt_range[jids[j], 1] - 0.03, b)
        self.relaxed = bool(on)

    def margin(self, q):
        return float(np.min(np.minimum(q - (self.lo - 0.03), (self.hi + 0.03) - q)))

    def current_q(self):
        return self.pc.d.qpos[self.qi].copy()

    def path_margin(self, P, R, stride=6, q0=None):
        """Sequential IK along a planned path (seeded from the current joints):
        the smallest distance to any joint limit, and whether every waypoint
        was reachable."""
        q = self.current_q() if q0 is None else q0
        mn = np.inf
        idx = list(range(stride, len(P), stride)) + [len(P) - 1]
        for t in idx:
            ep, er, q = self.solve(P[t], R[t], q0=q, iters=25)
            if ep > 0.02:
                return -1.0, q
            mn = min(mn, self.margin(q))
        return mn, q

    def best_flip(self, pos, R):
        """Between the two equivalent gripper orientations, the one whose IK
        solution stays farther from joint limits."""
        best = None
        for Rc in (R, R @ np.diag([-1.0, -1.0, 1.0])):
            ep, er, q = self.solve(pos, Rc, q0=self.current_q())
            if ep < 0.008 and er < 0.08:
                mg = self.margin(q)
                if best is None or mg > best[0]:
                    best = (mg, Rc)
        return best

    def ok(self, pos, R, tol_p=0.008, tol_r=0.08):
        ep, er, _ = self.solve(pos, R)
        if ep < tol_p and er < tol_r:
            return True
        # gripper is symmetric under pi about its own axis
        ep2, er2, _ = self.solve(pos, R @ np.diag([-1.0, -1.0, 1.0]))
        return ep2 < tol_p and er2 < tol_r
