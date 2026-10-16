"""Compose human-styled motions from keyframes.

Keyframes (exact grasp / place / handle poses) come from scene geometry; the
style comes from statistics of the human LIBERO demos: where the hand is 1 s and
0.5 s before a grasp, how long moves take as a function of distance, lift and
carry heights, release heights, and the ~2 mm of high-frequency wobble that
human paths have on top of a smooth spline (sampled from real human residuals).
Paths are clamped cubic splines in time; the robot tracks them with velocity
feed-forward plus light feedback, so the commanded actions are smooth like the
human ones and the hand actually arrives where it should."""
import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.spatial.transform import Rotation, Slerp

HZ = 20
M_PER_ACTION = 0.011
RAD_PER_ACTION = 0.11
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # closing axis = world y at yaw 0
FLIP = np.diag([-1.0, -1.0, 1.0])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_err(Rt, Rc):
    return Rotation.from_matrix(Rt @ Rc.T).as_rotvec()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def soft_limit(x, knee, top):
    ax = np.abs(x)
    y = np.where(ax > knee, knee + (top - knee) * np.tanh((ax - knee) / (top - knee)), ax)
    return np.sign(x) * y


class ResidualBank:
    """High-frequency deviations of human eef paths from a smoothing spline."""

    def __init__(self, tasks, segs, n=400, seed=0):
        rng = np.random.default_rng(seed)
        pool = [s for s in segs if s["type"] in ("grasp_approach", "transport") and s["t1"] - s["t0"] >= 30]
        self.res, self.ares, self.wob = [], [], []
        for i in rng.permutation(len(pool))[:n]:
            s = pool[i]
            d = tasks[s["task"]]["demos"][s["demo"]]
            P = d["eef_pos"][s["t0"]:s["t1"]].astype(float)
            A = d["actions"][s["t0"]:s["t1"], :6].astype(float)
            Rm = d["eef_mat"][s["t0"]:s["t1"]].astype(float)
            t = np.arange(len(P)) / HZ
            fit = np.stack([make_smoothing_spline(t, P[:, k], lam=0.02)(t) for k in range(3)], 1)
            self.res.append(P - fit)
            # high-frequency part of the human action signal
            afit = np.stack([make_smoothing_spline(t, A[:, k], lam=0.005)(t) for k in range(6)], 1)
            self.ares.append(A - afit)
            # slow wrist wander: orientation relative to the start, minus its linear trend
            rv = Rotation.from_matrix(Rm @ Rm[0].T).as_rotvec()
            rv -= np.outer(np.linspace(0, 1, len(rv)), rv[-1])
            self.wob.append(rv)
        self.rng = rng

    def _chain(self, pool, T):
        out = []
        n = 0
        while n < T:
            r = pool[self.rng.integers(len(pool))]
            out.append(r[: T - n]); n += len(out[-1])
        return np.concatenate(out)[:T]

    def sample_actions(self, T, scale=1.0):
        return self._chain(self.ares, T) * scale

    def sample_wobble(self, T, scale=1.0):
        out = np.zeros((T, 3)); k = 0
        while k < T:
            r = self.wob[self.rng.integers(len(self.wob))]
            m = min(len(r), T - k)
            out[k:k + m] = r[:m] + (out[k - 1] if k > 0 else 0)
            k += m
        out -= np.outer(np.linspace(0, 1, T), out[-1])  # back to zero at the end
        return out * scale

    def sample(self, T, scale=1.0):
        out = np.zeros((T, 3))
        k = 0
        while k < T:
            r = self.res[self.rng.integers(len(self.res))]
            n = min(len(r), T - k)
            seg = r[:n] - r[0]  # continuous start
            if k > 0:
                seg = seg + out[k - 1]
            out[k:k + n] = seg
            k += n
        # remove slow drift so the residual never moves the endpoints
        t = np.arange(T) / HZ
        if T > 4:
            out -= np.stack([make_smoothing_spline(t, out[:, i], lam=0.5)(t) for i in range(3)], 1)
        return out * scale


def spline_path(times, points, v0=None, v1=None):
    """Clamped cubic spline through keyframes -> per-step positions & velocities."""
    times = np.asarray(times, float)
    points = np.asarray(points, float)
    bc0 = (1, np.zeros(3) if v0 is None else np.asarray(v0))
    bc1 = (1, np.zeros(3) if v1 is None else np.asarray(v1))
    cs = CubicSpline(times, points, bc_type=(bc0, bc1), axis=0)
    T = int(round(times[-1] * HZ)) + 1
    tq = np.arange(T) / HZ
    return cs(tq), cs(tq, 1) / HZ  # positions, m/step


Q7_LIM = 2.8973


def pick_flip(R0, R1, q7=None):
    """Of the two equivalent gripper orientations (R1 and R1 turned 180 deg about
    its axis), the nearer one - unless that would wind the last joint (which
    turns opposite to the hand's yaw) into its stop."""
    cands = [R1, R1 @ FLIP]
    ang = [np.linalg.norm(rot_err(R, R0)) for R in cands]
    if q7 is None:
        return cands[int(np.argmin(ang))]
    room = []
    for R in cands:
        rz = Rotation.from_matrix(R @ R0.T).as_rotvec()[2]
        room.append(Q7_LIM - abs(q7 - rz))
    ok = [i for i in (0, 1) if room[i] > 0.35]
    if ok:
        return cands[min(ok, key=lambda i: ang[i])]
    return cands[int(np.argmax(room))]


def orient_path(R0, R1, T, start=0.0, end=0.75, q7=None, exact=False):
    """Orientation from R0 to R1 (gripper-symmetric) with a min-jerk profile."""
    if not exact:
        R1 = pick_flip(R0, R1, q7)
    x = np.clip((np.arange(T) / max(T - 1, 1) - start) / max(end - start, 1e-3), 0, 1)
    s = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
    sl = Slerp([0, 1], Rotation.from_matrix(np.stack([R0, R1])))
    return sl(s).as_matrix()


class Composer:
    def __init__(self, pc, rng, bank):
        self.pc, self.rng, self.bank = pc, rng, bank

    def follow(self, P, V, R, grip, kp=6.0, kr=1.5, stop_fn=None, residual=0.8, wobble=1.0,
               act_noise=1.0, precise_end=True, mode=None):
        """Track a planned path. P,V: (T,3) positions & m/step velocities,
        R: (T,3,3), grip: scalar or (T,) gripper command."""
        pc = self.pc
        T = len(P)
        if residual > 0 and T > 8:
            P = P + self.bank.sample(T, residual)
            V = np.gradient(P, axis=0)
        g = np.broadcast_to(np.asarray(grip, float), (T,))
        if wobble > 0 and T > 8:
            W = self.bank.sample_wobble(T, wobble)
            if precise_end:  # keep the final pose exact (grasp / hook)
                fade = np.clip((T - 1 - np.arange(T)) / max(0.25 * T, 1), 0, 1)
                W = W * fade[:, None]
            R = np.einsum("tij,tjk->tik", Rotation.from_rotvec(W).as_matrix(), R)
        AN = self.bank.sample_actions(T, act_noise) if act_noise > 0 else np.zeros((T, 6))
        dR = Rotation.from_matrix(R[1:] @ np.transpose(R[:-1], (0, 2, 1))).as_rotvec()
        if (mode or self.mode) == "ff":
            return self._follow_ff(P, V, R, dR, g, stop_fn, precise_end)
        for t in range(T - 1):
            a = np.zeros(7)
            a[:3] = V[t + 1] / M_PER_ACTION + kp * (P[t] - pc.eef_pos)
            a[3:6] = dR[t] / RAD_PER_ACTION + kr * rot_err(R[t], pc.eef_mat)
            a[:6] += AN[t]
            a[:3] = soft_limit(a[:3], 0.75, 0.97)
            a[3:6] = soft_limit(a[3:6], 0.2, 0.36)
            a[6] = g[t]
            pc.grip = a[6]
            pc.step_raw(a)
            if stop_fn is not None and stop_fn(t):
                return t
        return T

    mode = "ff"
    VMAX = 0.27       # m/s: peak eef speed; keeps actions below saturation (human p90 speed ~0.23)
    LEAD = 2          # actions lead the motion (arm responds as a ~2-step lag)
    TAU = 6.0         # steps: position error is removed through velocity over ~TAU steps

    def _follow_ff(self, P, V, R, dR, g, stop_fn, precise_end):
        """Human-like action generation: the action is the planned velocity a
        couple of steps ahead (so the arm, a smooth 2nd-order lag, follows it),
        plus a slow low-passed correction expressed as extra velocity. The
        action direction therefore stays consistent with the motion, as in
        teleoperation."""
        pc = self.pc
        T = len(P)
        corr = getattr(self, "_corr", np.zeros(3)); rcorr = getattr(self, "_rcorr", np.zeros(3))
        self.limit_hit = False
        self.joint_margin()
        jm0v = self.joint_margins()
        for t in range(T - 1):
            # never drive a joint into its limit: stop the move early instead
            # (each joint on its own: one already at its bound must not hide
            # another one swinging toward its own)
            if t >= (1 if np.min(jm0v) < self.JOINT_STOP else 3) and self.guard:
                jmv = self.joint_margins()
                tol = np.where(jm0v < self.JOINT_STOP, 0.003, 0.01)
                bad = (jmv < self.JOINT_STOP) & (jmv < jm0v - tol)
                jm, jm0 = float(np.min(jmv)), float(np.min(jm0v))
                if bad.any():
                    self.limit_hit = True
                    if __import__("os").environ.get("DEBUG_GUARD"):
                        print(f"      GUARD stop t={t}/{T} phase={pc.phase} jm={jm:.3f} jm0={jm0:.3f} q={np.round(pc.d.qpos[self._jlim[0]], 2)}", flush=True)
                    return t
            k = min(t + self.LEAD, T - 1)
            kr_ = min(t + self.LEAD, T - 2)
            # tighter correction near the end of precise moves
            tau = self.TAU if not (precise_end and T - t < 12) else 3.0
            e = P[t] - pc.eef_pos
            corr = 0.7 * corr + 0.3 * (e / tau)
            er = rot_err(R[t], pc.eef_mat)
            rcorr = 0.7 * rcorr + 0.3 * (er / tau)
            a = np.zeros(7)
            a[:3] = (V[k] + corr) / M_PER_ACTION
            a[3:6] = (dR[kr_] + rcorr) / RAD_PER_ACTION
            a[:3] = soft_limit(a[:3], 0.75, 0.97)
            a[3:6] = soft_limit(a[3:6], 0.2, 0.36)
            a[6] = g[t]
            pc.grip = a[6]
            pc.step_raw(a)
            self._corr, self._rcorr = corr, rcorr
            # stuck (a joint at its stop, pinned against something): the hand
            # falls behind and stops moving - don't keep pushing harder
            ph_ = getattr(self, "_ph", None)
            if t == 0 or ph_ is None:
                ph_ = self._ph = []
            ph_.append(pc.eef_pos.copy())
            if t >= 10 and self.guard and np.linalg.norm(P[t] - pc.eef_pos) > 0.04 and \
                    np.linalg.norm(ph_[-1] - ph_[-9]) < 0.008:
                self.limit_hit = True
                if __import__("os").environ.get("DEBUG_GUARD"):
                    print(f"      STALL stop t={t}/{T} phase={pc.phase} lag={np.linalg.norm(P[t] - pc.eef_pos):.3f}", flush=True)
                return t
            if stop_fn is not None and stop_fn(t):
                return t
        # finish the move: keep tracking until the final keyframe is reached
        if precise_end:
            tol = getattr(self, "end_tol", 0.012)
            hist = []
            for k in range(30):
                e = P[-1] - pc.eef_pos
                en = float(np.linalg.norm(e))
                if en < tol:
                    break
                hist.append(en)
                if len(hist) > 5 and hist[-6] - en < 0.001:
                    break  # not getting closer (blocked / in contact): don't fight it
                if self.guard and np.any((self.joint_margins() < self.JOINT_STOP) & (self.joint_margins() < jm0v - 0.003)):
                    self.limit_hit = True
                    break  # don't chase the last centimetres into a joint stop
                a = np.zeros(7)
                a[:3] = np.clip(e / 0.011 / 4.0, -0.45, 0.45)
                a[3:6] = np.clip(rot_err(R[-1], pc.eef_mat) / RAD_PER_ACTION / 5.0, -0.2, 0.2)
                a[6] = g[-1]
                pc.grip = a[6]
                pc.step_raw(a)
                if stop_fn is not None and stop_fn(T - 1 + k):
                    return T + k
        return T

    JOINT_STOP = 0.05   # rad from a joint limit
    guard = True        # off while recovering from an awkward posture

    def joint_margin(self):
        pc = self.pc
        if not hasattr(self, "_jlim"):
            r = pc.robot
            jids = [pc.m.dof_jntid[v] for v in r._ref_joint_vel_indexes]
            self._jlim = (np.array(r._ref_joint_pos_indexes), pc.m.jnt_range[jids, 0], pc.m.jnt_range[jids, 1])
        qi, lo, hi = self._jlim
        q = pc.d.qpos[qi]
        lo, hi = lo.copy(), hi.copy()
        # wrist: a looser version of the human range (outside it the arm is on
        # the flipped-wrist branch, which the controller can't come back from)
        for j, (a, b) in self.WRIST_BOX.items():
            lo[j], hi[j] = max(lo[j], a), min(hi[j], b)
        # wrist joints, plus the elbow straightening out (people never fully
        # straighten it, and at full stretch the controller can't bend it back);
        # the elbow folding up close to the base is allowed
        return float(np.min(self.joint_margins()))

    def joint_margins(self):
        """Per-joint room: wrist joints 5-7 (within the human box) and the
        elbow straightening out."""
        pc = self.pc
        if not hasattr(self, "_jlim"):
            self.joint_margin()
        qi, lo, hi = self._jlim
        q = pc.d.qpos[qi]
        lo, hi = lo.copy(), hi.copy()
        for j, (a, b) in self.WRIST_BOX.items():
            lo[j], hi[j] = max(lo[j], a), min(hi[j], b)
        mg = np.minimum(q - lo, hi - q)
        return np.r_[mg[4:], self.ELBOW_MAX - q[3]]

    ELBOW_MAX = -0.4   # joint 4: human demos stay below about -0.23

    WRIST_BOX = {4: (-1.15, 2.15), 5: (0.6, 3.4)}   # the human demos' range: stop well before the flipped branch
    WRIST_BOX_RELAXED = {4: (-1.55, 2.45), 5: (0.4, 3.55)}   # during side / handle grasps

    def relax(self, on=True):
        self.WRIST_BOX = self.WRIST_BOX_RELAXED if on else type(self).WRIST_BOX

    def settle(self, target, R, grip, tol=0.005, max_steps=12):
        """Short, smooth corrective sub-movement until within tol (no jitter)."""
        pc = self.pc
        e0 = target - pc.eef_pos
        if np.linalg.norm(e0) < tol:
            return 0
        n = int(np.clip(np.linalg.norm(e0) / 0.004, 4, max_steps))
        P, V = spline_path([0.0, n / HZ], [pc.eef_pos, target], v0=np.zeros(3), v1=np.zeros(3))
        Rr = orient_path(pc.eef_mat, R, len(P), end=1.0, q7=self._q7())
        self.follow(P, V, Rr, grip, residual=0.0, wobble=0.0, precise_end=False)
        return n

    def hold(self, n, grip, drift=None):
        pc = self.pc
        for i in range(n):
            a = np.zeros(7)
            if drift is not None:
                a[:3] = drift / M_PER_ACTION
            a[:3] += self.rng.normal(0, 0.02, 3)
            a[6] = grip
            pc.step_raw(a)

    def plan(self, keys, R_end, v_end=None, rot_end=0.75, p0=None, R0=None, v0=None, exact=False):
        """Plan (without executing) a spline through keyframes from the current
        (or given) pose. Returns P (T,3), V (T,3) m/step, R (T,3,3)."""
        pc = self.pc
        p0 = pc.eef_pos if p0 is None else p0
        R0 = pc.eef_mat if R0 is None else R0
        v0 = np.clip(getattr(pc, "eef_vel", np.zeros(3)), -0.3, 0.3) if v0 is None else v0
        times = np.array([0.0] + [k[0] for k in keys])
        pts = [p0] + [k[1] for k in keys]
        # carry over only the part of the current velocity that points where we
        # are going next (avoids an overshoot-and-come-back bounce at splices)
        d1 = np.asarray(keys[0][1]) - p0
        if np.linalg.norm(d1) > 1e-6:
            u = d1 / np.linalg.norm(d1)
            along = max(float(v0 @ u), 0.0)
            v0 = along * u + 0.3 * (v0 - (v0 @ u) * u)
        P, V = spline_path(times, pts, v0=v0, v1=v_end)
        # never ask for more speed than the arm can follow: stretch the timing
        vmax = float(np.max(np.linalg.norm(V, axis=1))) * HZ if len(V) else 0.0
        if vmax > self.VMAX:
            times = times * (vmax / self.VMAX)
            P, V = spline_path(times, pts, v0=v0, v1=v_end)
        # nor turn the hand faster than people do (human rotation actions:
        # p90 0.18, p99 0.38): a big turn takes its time
        q7 = self._q7() if R0 is pc.eef_mat or np.allclose(R0, pc.eef_mat) else None
        if not exact:
            R_end = self._flip_ik(R0, R_end, pts[-1], q7)
        k = self._rot_stretch(R0, R_end, len(P), rot_end)
        if k > 1.0:
            times = times * k
            P, V = spline_path(times, pts, v0=v0, v1=v_end)
        R = orient_path(R0, R_end, len(P), end=rot_end, q7=q7, exact=True)
        return P, V, R

    ik_q7 = None   # set by the play layer: (pos, R) -> last-joint angle there (IK)

    def _flip_ik(self, R0, R1, p1, q7):
        """pick_flip, but judging the last joint at the end of the move by IK
        (moving the hand across the table winds it too, not only turning it)."""
        if q7 is None or self.ik_q7 is None:
            return pick_flip(R0, R1, q7)
        cands = [R1, R1 @ FLIP]
        ang = [np.linalg.norm(rot_err(R, R0)) for R in cands]
        room = []
        for R in cands:
            q7e = self.ik_q7(p1, R)
            room.append(-1.0 if q7e is None else Q7_LIM - abs(q7e))
        ok = [i for i in (0, 1) if room[i] > 0.35]
        if ok:
            return cands[min(ok, key=lambda i: ang[i])]
        return cands[int(np.argmax(room))]

    def _q7(self):
        try:
            return float(self.pc.d.qpos[self.pc.robot._ref_joint_pos_indexes[6]])
        except Exception:
            return None

    WSTEP = 0.038   # rad/step peak turning rate of a planned move (action ~0.24)

    @classmethod
    def _rot_stretch(cls, R0, R1, T, rot_end):
        ang = np.linalg.norm(rot_err(R1, R0))
        peak = 1.875 * ang / max(rot_end * (T - 1), 1)
        return peak / cls.WSTEP

    def move(self, keys, R_end, grip, v_end=None, stop_fn=None, rot_end=0.75, residual=0.8, **kw):
        """keys: list of (time_s, position) after the current pose (t=0)."""
        pc = self.pc
        times = [0.0] + [k[0] for k in keys]
        pts = [pc.eef_pos] + [k[1] for k in keys]
        v0 = np.clip(getattr(pc, "eef_vel", np.zeros(3)), -0.3, 0.3)
        P, V = spline_path(times, pts, v0=v0, v1=v_end)
        q7 = self._q7()
        R_end = self._flip_ik(pc.eef_mat, R_end, pts[-1], q7)
        k = self._rot_stretch(pc.eef_mat, R_end, len(P), rot_end)
        if k > 1.0:
            times = [x * k for x in times]
            P, V = spline_path(times, pts, v0=v0, v1=v_end)
        R = orient_path(pc.eef_mat, R_end, len(P), end=rot_end, q7=q7, exact=True)
        return self.follow(P, V, R, grip, stop_fn=stop_fn, residual=residual, **kw)
