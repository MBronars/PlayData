"""Retarget human LIBERO motion segments into new scenes.

A segment (eef positions/orientations/actions over [t0, t1)) is mapped through
a rigid transform G (optional y-mirror, yaw rotation about an anchor,
translation) and its start is blended into the robot's current pose. The robot
executes the transformed human actions plus a feedforward for the warp and a
light feedback term, so the human timing and micro-structure are preserved."""
import numpy as np
from robosuite.utils.transform_utils import mat2quat, quat2axisangle

M_PER_ACTION = 0.011
RAD_PER_ACTION = 0.11
MIRROR = np.diag([1.0, -1.0, 1.0])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def minjerk(x):
    x = np.clip(x, 0, 1)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def rot_err(Rt, Rc):
    return quat2axisangle(mat2quat(Rt @ Rc.T))


class Transform:
    """x' = Rz(yaw) M (x - a_src) + a_dst"""

    def __init__(self, a_src, a_dst, yaw, mirror=False):
        self.a_src, self.a_dst = np.asarray(a_src, float), np.asarray(a_dst, float)
        self.M = MIRROR if mirror else np.eye(3)
        self.L = rotz(yaw) @ self.M
        self.mirror = mirror
        self.Rz = rotz(yaw)

    def pos(self, x):
        return (x - self.a_src) @ self.L.T + self.a_dst

    def mat(self, R):  # R: (..., 3, 3)
        if self.mirror:
            # mirrored gripper == same gripper rotated pi about its own z axis
            return self.Rz @ self.M @ R @ self.M @ np.diag([-1.0, -1.0, 1.0])
        return self.Rz @ R

    def act(self, a):
        a = np.array(a, float)
        a[..., :3] = a[..., :3] @ self.L.T
        a[..., 3:6] = (a[..., 3:6] @ self.L.T) * (-1.0 if self.mirror else 1.0)
        return a


def soft_limit(x, knee, top):
    """Identity below `knee`, smooth compression toward `top` above it."""
    ax = np.abs(x)
    over = ax > knee
    y = ax.copy()
    y[over] = knee + (top - knee) * np.tanh((ax[over] - knee) / (top - knee))
    return np.sign(x) * y


def time_stretch(seg_data, k, key_t=None):
    """Play a human clip k times slower (k >= 1): positions/orientations are
    resampled, translational/rotational actions scaled by 1/k."""
    if k <= 1.01:
        return seg_data, key_t
    from scipy.spatial.transform import Rotation, Slerp
    P, Rm, A = seg_data["eef_pos"], seg_data["eef_mat"], seg_data["actions"]
    T = len(P)
    n = int(round((T - 1) * k)) + 1
    tq = np.linspace(0, T - 1, n)
    Pn = np.stack([np.interp(tq, np.arange(T), P[:, i]) for i in range(3)], 1)
    Rn = Slerp(np.arange(T), Rotation.from_matrix(Rm))(tq).as_matrix()
    idx = np.clip(np.floor(tq).astype(int), 0, T - 1)
    An = A[idx].copy()
    An[:, :6] = np.stack([np.interp(tq, np.arange(T), A[:, i]) for i in range(6)], 1) / k
    An[:, 6] = A[np.clip(np.round(tq).astype(int), 0, T - 1), 6]
    out = dict(seg_data, eef_pos=Pn, eef_mat=Rn, actions=An)
    return out, (None if key_t is None else int(round(key_t * k)))


def plan(seg_data, G, p_cur, R_cur, blend_frac=0.75, end_goal=None, end_blend_start=0.0,
         key_t=None, auto_stretch=False, max_stretch=1.15, wrist_yaw=0.0, wrist_const=False):
    """Return desired positions, rotations and feed-forward actions.

    seg_data: dict(eef_pos (T,3), eef_mat (T,3,3), actions (T,7))
    blend_frac: start offset is removed by this fraction of the path's arc length
        up to `key_t` (contact / grasp index; default = segment end).
    end_goal: if given, the final position is warped to it (transport)."""
    if auto_stretch:
        P0 = G.pos(seg_data["eef_pos"])
        arc0 = np.linalg.norm(np.diff(P0, axis=0), axis=1).sum()
        warp = np.linalg.norm(p_cur - P0[0])
        if end_goal is not None:
            warp += np.linalg.norm(end_goal - P0[-1])
        k = float(np.clip(1 + 1.2 * warp / max(arc0, 0.05), 1.0, max_stretch))
        seg_data, key_t = time_stretch(seg_data, k, key_t)
    P = G.pos(seg_data["eef_pos"])
    Rm = G.mat(seg_data["eef_mat"])
    A = G.act(seg_data["actions"])
    T = len(P)
    # wrist yaw offset (orientation only; the translational path stays human):
    # blended in during the approach like a human twisting the wrist, or
    # constant (e.g. while carrying an object grasped with that offset)
    if abs(wrist_yaw) > 1e-4:
        arcw = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        kk = T - 1 if key_t is None else int(np.clip(key_t, 1, T - 1))
        sw = np.ones(T) if wrist_const else minjerk((arcw / max(arcw[kk], 1e-6) - 0.05) / 0.8)
        psi = wrist_yaw * sw
        Rm = np.stack([rotz(psi[t]) @ Rm[t] for t in range(T)])
        A = A.copy()
        A[:-1, 5] += np.diff(psi) / RAD_PER_ACTION
    # gripper is symmetric under pi about its own z: use the closer variant
    FLIP = np.diag([-1.0, -1.0, 1.0])
    if np.linalg.norm(rot_err(R_cur, Rm[0] @ FLIP)) < np.linalg.norm(rot_err(R_cur, Rm[0])):
        Rm = Rm @ FLIP
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s_arc = arc / max(arc[-1], 1e-6)
    k = T - 1 if key_t is None else int(np.clip(key_t, 1, T - 1))
    s_key = arc / max(arc[k], 1e-6)
    # start blend
    d0 = p_cur - P[0]
    w0 = 1 - minjerk(s_key / blend_frac)
    delta = d0[None] * w0[:, None]
    # end warp
    if end_goal is not None:
        e = end_goal - (P[-1] + delta[-1])
        w1 = minjerk((s_arc - end_blend_start) / (1 - end_blend_start))
        delta = delta + e[None] * w1[:, None]
    Pd = P + delta
    # rotation start blend (axis-angle offset fading out)
    r0 = rot_err(R_cur, Rm[0])
    from scipy.spatial.transform import Rotation
    Rd = np.stack([Rotation.from_rotvec(r0 * w0[t]).as_matrix() @ Rm[t] for t in range(T)])
    ff = np.zeros((T, 6))
    ff[:-1, :3] = np.clip(np.diff(delta, axis=0) / M_PER_ACTION, -0.35, 0.35)
    # rotation feed-forward for the orientation start-blend (rotation beyond
    # what the human clip itself does), so the wrist doesn't lag and get
    # corrected by feedback
    from scipy.spatial.transform import Rotation as _R
    dRd = _R.from_matrix(Rd[1:] @ np.transpose(Rd[:-1], (0, 2, 1))).as_rotvec()
    dRh = _R.from_matrix(Rm[1:] @ np.transpose(Rm[:-1], (0, 2, 1))).as_rotvec()
    ff[:-1, 3:6] = np.clip((dRd - dRh) / RAD_PER_ACTION, -0.25, 0.25)
    return dict(P=Pd, R=Rd, A=A, ff=ff, T=T, key_t=k,
                stretch=float(len(P) / max(len(seg_data["eef_pos"]), 1)) if False else None)


def execute(pc, pl, kp=8.0, kr=1.0, ki=0.3, grip_override=None, stop_fn=None,
            precise_window=15, smooth=0.6, cap=0.3, precise_cap=0.6):
    """Run a planned segment: transformed human action + warp feed-forward +
    a low-passed, capped feedback correction toward the planned pose at the
    *same* time index. The integrator is only active in the precision window
    before the key (contact / grasp) frame."""
    integ = np.zeros(3)
    corr = np.zeros(6)
    for t in range(pl["T"] - 1):
        p, Rc = pc.eef_pos, pc.eef_mat
        e = pl["P"][t] - p
        precise = 0 <= pl["key_t"] - t <= precise_window
        if precise:
            integ = np.clip(integ + e, -0.1, 0.1)
        else:
            integ *= 0.9
        c = np.zeros(6)
        c[:3] = kp * e + ki * integ
        c[3:6] = kr * rot_err(pl["R"][t], Rc)
        corr = smooth * corr + (1 - smooth) * c
        lim = precise_cap if precise else cap
        a = pl["A"][t].copy()
        a[:3] += pl["ff"][t, :3] + np.clip(corr[:3], -lim, lim)
        a[3:6] += pl["ff"][t, 3:6] + np.clip(corr[3:6], -0.4 if precise else -0.2, 0.4 if precise else 0.2)
        # keep actions inside the human range (warp/correction terms must not
        # create saturated or unusually large commands)
        a[:3] = soft_limit(a[:3], 0.75, 0.97)
        a[3:6] = soft_limit(a[3:6], 0.22, 0.42)
        if grip_override is not None:
            a[6] = grip_override
        pc.grip = a[6]
        pc.step_raw(a)
        pl["max_err"] = max(pl.get("max_err", 0.0), float(np.linalg.norm(e)))
        if t + 1 == pl["key_t"]:
            pl["key_err"] = float(np.linalg.norm(pl["P"][t + 1] - pc.eef_pos))
            pl["key_eef"] = pc.eef_pos.copy()
            pl["key_rot_err"] = float(np.linalg.norm(rot_err(pl["R"][t + 1], pc.eef_mat)))
        if stop_fn is not None and stop_fn():
            return t
    return pl["T"]


def match_start(P, Rm, p_cur, v_cur, R_cur, t_max, w_v=0.35, w_r=0.08):
    """Motion matching: index in [0, t_max] whose pose & velocity best match
    the robot's current state. Returns (index, cost)."""
    t_max = int(max(0, min(t_max, len(P) - 1)))
    if len(P) < 3:
        return 0, float(np.linalg.norm(P[0] - p_cur))
    V = np.gradient(P, axis=0) * 20.0
    dp = np.linalg.norm(P[:t_max + 1] - p_cur, axis=1)
    dv = np.linalg.norm(V[:t_max + 1] - v_cur, axis=1)
    # rotation angle to current, accounting for the gripper's pi symmetry
    Rs = Rm[:t_max + 1]
    tr1 = np.einsum("tij,ij->t", Rs, R_cur)
    tr2 = np.einsum("tij,ij->t", Rs @ np.diag([-1.0, -1.0, 1.0]), R_cur)
    ang = np.arccos(np.clip((np.maximum(tr1, tr2) - 1) / 2, -1, 1))
    cost = dp + w_v * dv + w_r * ang + 0.3 * np.maximum(ang - np.radians(60), 0)
    i = int(np.argmin(cost))
    return i, float(cost[i])


def correct_to(pc, target, R_target=None, tol=0.006, max_steps=8, gain=8.0, rng=None):
    """Short human-like corrective sub-movement toward `target` (bell-shaped,
    small amplitude). Returns the number of steps taken."""
    n = 0
    prev_phase, pc.phase = pc.phase, "correct"
    for n in range(max_steps):
        e = target - pc.eef_pos
        if np.linalg.norm(e) < tol:
            break
        a = np.zeros(7)
        a[:3] = np.clip(gain * e, -0.5, 0.5)
        if R_target is not None:
            a[3:6] = np.clip(1.5 * rot_err(R_target, pc.eef_mat), -0.3, 0.3)
        if rng is not None:
            a[:3] += rng.normal(0, 0.02, 3)
        a[6] = pc.grip
        pc.step_raw(a)
    pc.phase = prev_phase
    return n


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def choose_transform(a_src, face_src, a_dst, face_dst, rng=None, max_yaw=np.radians(60),
                     sym=None):
    """Pick mirror / yaw so the source facing direction maps onto the target's
    with the smallest rotation. `sym`: rotational symmetry order of the target
    (e.g. 2 for boxes, None = no symmetry, 'inf' = round objects -> yaw 0).
    Returns (Transform, yaw) or (None, None) if rotation would be too large."""
    opts = []
    for mirror in (False, True):
        fs = -face_src if mirror else face_src
        if sym == "inf":
            dy = 0.0
        elif sym:
            base = wrap(face_dst - fs)
            per = 2 * np.pi / sym
            dy = min((wrap(base + k * per) for k in range(sym)), key=abs)
        else:
            dy = wrap(face_dst - fs)
        opts.append((abs(dy), mirror, dy))
    opts = [o for o in opts if o[0] <= max_yaw]
    if not opts:
        return None, None
    if rng is not None and len(opts) == 2 and abs(opts[0][0] - opts[1][0]) < np.radians(20):
        o = opts[rng.integers(2)]
    else:
        o = min(opts)
    return Transform(a_src, a_dst, o[2], mirror=o[1]), o[2]
