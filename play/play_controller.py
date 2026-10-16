"""Scripted 'play' controller for LIBERO. Uses privileged sim state to chain
random manipulation primitives (pick/place, stack, push, lift-and-wave, drawer
open/close, free reaching) with noise, while recording every step."""
import numpy as np
import mujoco
from robosuite.utils.transform_utils import quat2mat, mat2quat, quat2axisangle, quat_slerp

from scene_gen import RECEPTACLES

RIM_OBJECTS = {"akita_black_bowl", "white_bowl", "glazed_rim_porcelain_ramekin",
               "red_coffee_mug", "porcelain_mug", "white_yellow_mug", "basket",
               "plate", "bowl_drainer", "wooden_tray"}
GRIPPER_MAX_WIDTH = 0.075
# Top-down gripper orientation at yaw=0 (fingers close along world y).
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


CTRL_HZ = 20
M_PER_ACTION = 0.012  # steady-state eef travel per step per unit OSC action (measured)
KP_POS = 10.0
KP_ROT = 2.0


def _minjerk(x):
    """Min-jerk position s(x) and derivative ds/dx on x in [0, 1]."""
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, 30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4


class EpisodeTooLong(Exception):
    pass


class PlayController:
    def __init__(self, env, object_names, rng, max_steps=1500, action_noise=0.05,
                 record_fn=None):
        self.env = env  # OffScreenRenderEnv
        self.e = env.env
        self.sim = self.e.sim
        self.m = self.sim.model._model
        self.d = self.sim.data._data
        self.rng = rng
        self.names = list(object_names)
        self.max_steps = max_steps
        self.noise = action_noise
        self._noise_scale = np.array([1, 1, 1, 0.15, 0.15, 0.15]) * action_noise
        self._ou = np.zeros(6)
        self._a_prev = np.zeros(6)
        self.smooth = 0.5
        self.eef_vel = np.zeros(3)
        self.phase = "other"
        self.speed = 0.15      # m/s, resampled per primitive
        self.rot_speed = 1.0   # rad/s
        self.record_fn = record_fn
        self.t = 0
        self.grip = -1.0
        self.obs = None
        self.robot = self.e.robots[0]
        self.eef_site = self.m.site(self.robot.controller.eef_name).id
        self.table_z = self._measure_table_z()
        self.segments = []
        self._obj_geoms = {}
        for n in self.names:
            root = self.e.objects_dict[n].root_body
            bid = self.m.body(root).id
            self._obj_geoms[n] = [g for g in range(self.m.ngeom)
                                  if self.m.body_rootid[self.m.geom_bodyid[g]] == bid
                                  and self.m.geom_contype[g] != 0]
        self.drawers = self._find_drawers()

    # ------------------------------------------------------------------ state
    @property
    def eef_pos(self):
        return self.d.site_xpos[self.eef_site].copy()

    @property
    def eef_mat(self):
        return self.d.site_xmat[self.eef_site].reshape(3, 3).copy()

    def fix_sunk_objects(self, settle_fn, tol=0.004):
        """LIBERO sometimes initializes an object partly inside the table (it is
        then held fast and can't be lifted). Raise such objects back onto the
        surface. Returns the names that were fixed."""
        m, d = self.m, self.d
        fixed = []
        for n in self.names:
            geoms = [g for g in self._obj_geoms[n] if m.geom_type[g] == 6]   # exact extents for boxes only
            if not geoms:
                continue
            lo = min(d.geom_xpos[g][2] - (np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g])[2] for g in geoms)
            if lo < self.table_z - tol:
                jn = self.e.objects_dict[n].joints[0]
                adr = m.jnt_qposadr[m.joint(jn).id]
                d.qpos[adr + 2] += self.table_z - lo + 0.002
                d.qvel[m.jnt_dofadr[m.joint(jn).id]:m.jnt_dofadr[m.joint(jn).id] + 6] = 0
                fixed.append((n, round(float(self.table_z - lo), 3)))
        if fixed:
            mujoco.mj_forward(m, d)
            settle_fn()
        return fixed

    def _measure_table_z(self):
        """Height of the table surface, ray-cast onto the table fixture's own
        geoms (LIBERO's workspace offset is off by ~3 cm in some scenes)."""
        m, d = self.m, self.d
        tb = [b for b in range(m.nbody) if "table" in m.body(b).name]
        roots = {m.body_rootid[b] for b in tb}
        x0, y0, z0 = self.e.workspace_offset
        hits = []
        geomid = np.zeros(1, np.int32)
        for dx in np.linspace(-0.2, 0.1, 5):
            for dy in np.linspace(-0.25, 0.25, 5):
                pnt = np.array([x0 + dx, y0 + dy, z0 + 1.0])
                # ray down through everything; keep the first table hit below
                for _ in range(8):
                    dist = mujoco.mj_ray(m, d, pnt, np.array([0, 0, -1.0]), None, 1, -1, geomid)
                    if dist < 0:
                        break
                    hz = pnt[2] - dist
                    if m.body_rootid[m.geom_bodyid[geomid[0]]] in roots and m.geom_contype[geomid[0]]:
                        hits.append(hz)
                        break
                    pnt = np.array([pnt[0], pnt[1], hz - 1e-4])
        if not hits:
            return float(z0)
        return float(np.median(hits))

    def eef_yaw(self):
        R = self.eef_mat @ R_DOWN.T
        return np.arctan2(R[1, 0], R[0, 0])

    def obj_category(self, name):
        return self.e.objects_dict[name].category_name if hasattr(
            self.e.objects_dict[name], "category_name") else name.rsplit("_", 1)[0]

    def obj_bbox(self, name):
        """World AABB + yaw-aligned extents of an object's collision geoms."""
        bid = self.m.body(self.e.objects_dict[name].root_body).id
        q = self.d.xquat[bid]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        pts = []
        for g in self._obj_geoms[name]:
            c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
            corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1)
                                        for sy in (-1, 1) for sz in (-1, 1)])
            R = self.d.geom_xmat[g].reshape(3, 3)
            pts.append(corners @ R.T + self.d.geom_xpos[g])
        pts = np.concatenate(pts)
        lo, hi = pts.min(0), pts.max(0)
        local = (pts[:, :2] - pts[:, :2].mean(0)) @ rotz(yaw)[:2, :2]
        ext = local.max(0) - local.min(0)
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
        return dict(lo=lo, hi=hi, center=center, yaw=yaw, ext=ext)

    def _find_drawers(self, levels=("top", "middle")):
        """List of (fixture, level, joint id, drawer body, handle geom)."""
        out = []
        for fname, fx in self.e.fixtures_dict.items():
            if "cabinet" not in fname:
                continue
            for level in levels:  # scripted mode: bottom handle too low for its grasp
                jn = f"{fname}_{level}_level"
                bn = f"{fname}_cabinet_{level}"
                try:
                    jid, bid = self.m.joint(jn).id, self.m.body(bn).id
                except KeyError:
                    continue
                geoms = [g for g in range(self.m.ngeom)
                         if self.m.geom_bodyid[g] == bid and self.m.geom_contype[g] != 0]
                # handle = collision geom furthest out along -y in body frame
                h = min(geoms, key=lambda g: self.m.geom_pos[g][1])
                out.append(dict(fixture=fname, level=level, jid=jid, bid=bid, handle=h))
        return out

    # ------------------------------------------------------------- low level
    def step(self, action):
        if self.t >= self.max_steps:
            raise EpisodeTooLong
        a = np.asarray(action, dtype=np.float64).copy()
        # temporally correlated (OU) noise, mostly on translation
        self._ou = 0.9 * self._ou + np.sqrt(1 - 0.81) * self.rng.normal(0, 1, 6) * self._noise_scale
        a[:6] += self._ou
        # low-pass filter for human-like smoothness
        a[:6] = self.smooth * self._a_prev + (1 - self.smooth) * a[:6]
        a[:6] = np.clip(a[:6], -1, 1)
        self._a_prev = a[:6].copy()
        if self.record_fn is not None and self.obs is not None:
            self.record_fn(self.obs, a, self)
        self.obs, _, _, _ = self.env.step(a)
        self.t += 1
        return self.obs

    def snapshot(self):
        """Everything needed to undo a trial motion (for look-ahead)."""
        return dict(state=self.sim.get_state().flatten().copy(), t=self.t, grip=self.grip,
                    a_out=getattr(self, "_a_out", np.zeros(7)).copy(), a_prev=self._a_prev.copy(),
                    eef_vel=self.eef_vel.copy(), obs=self.obs, phase=self.phase)

    def restore(self, s):
        self.sim.set_state_from_flattened(s["state"])
        self.sim.forward()
        self.t, self.grip, self._a_out, self._a_prev = s["t"], s["grip"], s["a_out"].copy(), s["a_prev"].copy()
        self.eef_vel, self.obs, self.phase = s["eef_vel"].copy(), s["obs"], s["phase"]

    def step_raw(self, a):
        """Step with an already human-like action (no added noise/filtering).
        If `self.slew` is set, the per-step change of the arm action is limited
        (like a human hand on a teleop device: no instantaneous jumps)."""
        if self.t >= self.max_steps:
            raise EpisodeTooLong
        a = np.asarray(a, dtype=np.float64).copy()
        nz = getattr(self, "noise_cfg", None)
        if nz is not None:
            # injected error: a shaky hand, or a short burst of jitter. The noise
            # is smooth in time (a hand wobbling, not white noise) and goes
            # through the same rate limit as everything else, so the commands
            # never jump from one step to the next more than a person's do
            k = self.t - nz["t0"]
            on = nz.get("len") is None or nz["start"] <= k < nz["start"] + nz["len"]
            st = getattr(self, "_nz_state", None)
            if st is None or getattr(self, "_nz_cfg", None) is not nz:
                st = np.zeros(6)
                self._nz_cfg = nz
            rho = 0.8
            st = rho * st + np.sqrt(1 - rho ** 2) * self.rng.normal(0, nz["sigma"], 6)
            self._nz_state = st
            if on:
                # fades out as the elbow nears straight (pushing on there
                # locks the arm out at full stretch)
                try:
                    el = float(self.d.qpos[self.robot._ref_joint_pos_indexes[3]])
                    fade = float(np.clip((-el - 0.45) / 0.4, 0.0, 1.0))
                except Exception:
                    fade = 1.0
                a[:6] = np.clip(a[:6] + fade * st * np.r_[1, 1, 1, 0.25, 0.25, 0.25], -1, 1)
        if getattr(self, "slew", None) is not None:
            prev = getattr(self, "_a_out", np.zeros(7))
            lt, lr = self.slew
            d = a[:3] - prev[:3]
            n = np.linalg.norm(d)
            if n > lt:
                a[:3] = prev[:3] + d * (lt / n)
            d = a[3:6] - prev[3:6]
            n = np.linalg.norm(d)
            if n > lr:
                a[3:6] = prev[3:6] + d * (lr / n)
            self._a_out = a.copy()
        if self.record_fn is not None and self.obs is not None:
            self.record_fn(self.obs, a, self)
        p0 = self.eef_pos
        self.obs, _, _, _ = self.env.step(a)
        self._a_prev = a[:6].copy()
        self.eef_vel = (self.eef_pos - p0) * CTRL_HZ
        self.t += 1
        return self.obs

    def move_to(self, pos, yaw=None, grip=None, tol=0.008, max_t=120,
                max_speed=1.0, R=None):
        """Move the eef to `pos` along a minimum-jerk trajectory (bell-shaped
        speed profile), then settle. Orientation is top-down with the given yaw
        about world z, or the full rotation `R` if provided. `max_speed` scales
        the per-primitive nominal speed (use < 1 for careful approaches)."""
        if grip is not None:
            self.grip = grip
        if R is not None:
            Rt = R
        else:
            if yaw is None:
                yaw = self.eef_yaw()
            Rt = rotz(yaw) @ R_DOWN
        pos = np.asarray(pos, dtype=np.float64)
        p0 = self.eef_pos
        q0, qt = mat2quat(self.eef_mat), mat2quat(Rt)
        dist = np.linalg.norm(pos - p0)
        ang = np.linalg.norm(quat2axisangle(mat2quat(Rt @ self.eef_mat.T)))
        v = self.speed * max_speed / CTRL_HZ  # mean speed, m/step
        T = int(np.clip(max(dist / v, ang / (self.rot_speed / CTRL_HZ), 6), 6, max_t))
        for k in range(1, T + 1):
            s, sd = _minjerk(k / T)
            p_des = p0 + s * (pos - p0)
            v_des = sd / T * (pos - p0)
            R_des = quat2mat(quat_slerp(q0, qt, s))
            a_pos = v_des / M_PER_ACTION + KP_POS * (p_des - self.eef_pos)
            aa = quat2axisangle(mat2quat(R_des @ self.eef_mat.T))
            self.step(np.concatenate([np.clip(a_pos, -1, 1), np.clip(KP_ROT * aa, -0.6, 0.6),
                                      [self.grip]]))
        # settle phase: gentle P control until within tolerance
        for _ in range(max(10, max_t - T)):
            err = pos - self.eef_pos
            aa = quat2axisangle(mat2quat(Rt @ self.eef_mat.T))
            if np.linalg.norm(err) < tol and np.linalg.norm(aa) < 0.06:
                return True
            self.step(np.concatenate([np.clip(KP_POS * err, -0.4, 0.4),
                                      np.clip(KP_ROT * aa, -0.4, 0.4), [self.grip]]))
            if np.linalg.norm(err) < tol * 2.5 and np.linalg.norm(self._a_prev[:3]) < 0.05:
                break  # stalled (e.g. contact); don't hang here
        return np.linalg.norm(pos - self.eef_pos) < tol * 2

    def gripper_action(self, grip, n=8):
        self.phase = "gripper_only"
        self.grip = grip
        for _ in range(int(n + self.rng.integers(-2, 3))):
            self.step(np.concatenate([np.zeros(6), [grip]]))

    def hold(self, n):
        for _ in range(n):
            self.step(np.concatenate([np.zeros(6), [self.grip]]))

    def closest_yaw(self, yaw):
        """Gripper is symmetric under pi rotation: pick the closer equivalent."""
        cur = self.eef_yaw()
        y = wrap_pi(yaw)
        alt = wrap_pi(y + np.pi)
        return self.pick_yaw([y, alt])

    Q7_LIM = 2.8973

    def pick_yaw(self, cands, lim=np.pi / 2 + 0.4):
        """Among equivalent gripper yaws, the one nearest the current wrist yaw
        that keeps the last joint (which turns with the gripper's yaw: dq7 ~
        -dyaw) away from its limits; never clip a yaw (that would turn the
        closing axis off its target)."""
        cur = self.eef_yaw()
        q7 = float(self.robot._joint_positions[6]) if hasattr(self.robot, "_joint_positions") else 0.0
        def room(v):
            q = q7 - wrap_pi(v - cur)
            return self.Q7_LIM - abs(q)
        ok = [v for v in cands if abs(v) <= lim and room(v) > 0.35]
        if ok:
            return float(min(ok, key=lambda v: abs(wrap_pi(v - cur))))
        return float(max(cands, key=room))

    @property
    def safe_z(self):
        return self.table_z + 0.22

    def is_grasped(self, name):
        geoms = set(self._obj_geoms[name])
        fingers = set()
        for g in self.robot.gripper.contact_geoms:
            try:
                fingers.add(self.m.geom(g).id)
            except KeyError:
                pass
        touch = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 in geoms and c.geom2 in fingers:
                touch.add(c.geom2)
            elif c.geom2 in geoms and c.geom1 in fingers:
                touch.add(c.geom1)
        return len(touch) >= 2 and self.grip > 0

    # ------------------------------------------------------------ primitives
    def grasp_pose(self, name):
        cat = self.obj_category(name)
        bb = self.obj_bbox(name)
        h = bb["hi"][2] - bb["lo"][2]
        if cat in RIM_OBJECTS:
            r = 0.5 * min(bb["hi"][0] - bb["lo"][0], bb["hi"][1] - bb["lo"][1])
            theta = self.rng.uniform(np.pi / 2, 3 * np.pi / 2)  # near side of rim
            rr = max(r - 0.012, 0.0)
            xy = bb["center"] + rr * np.array([np.cos(theta), np.sin(theta)])
            z = bb["hi"][2] - 0.02
            yaw = theta - np.pi / 2
        else:
            # close across the narrower horizontal extent
            ax = int(np.argmin(bb["ext"]))
            if bb["ext"][ax] > GRIPPER_MAX_WIDTH:
                return None
            phi = bb["yaw"] + (0 if ax == 0 else np.pi / 2)
            xy = bb["center"]
            z = bb["hi"][2] - min(0.03, 0.5 * h)
            yaw = phi - np.pi / 2
        z = max(z, self.table_z + 0.012)
        return np.array([xy[0], xy[1], z]), self.closest_yaw(yaw)

    def pick(self, name):
        gp = self.grasp_pose(name)
        if gp is None:
            return False
        pos, yaw = gp
        pos[:2] += self.rng.normal(0, 0.004, 2)
        z_hover = max(pos[2] + 0.10, self.table_z + 0.15)
        self.move_to([pos[0], pos[1], max(z_hover, self.eef_pos[2])], yaw, grip=-1, tol=0.02)
        self.move_to([pos[0], pos[1], z_hover], yaw, tol=0.01)
        self.move_to(pos, yaw, tol=0.006, max_speed=0.5)
        self.gripper_action(1.0, n=10)
        self.move_to([pos[0], pos[1], z_hover + 0.05], tol=0.02, max_speed=0.6)
        return self.is_grasped(name) and self.obj_bbox(name)["lo"][2] > self.table_z + 0.02

    def place_at(self, xy, z_release, yaw=None):
        cur = self.eef_pos
        travel_z = max(cur[2], z_release + 0.08)
        self.move_to([xy[0], xy[1], travel_z], yaw, tol=0.02)
        self.move_to([xy[0], xy[1], z_release], yaw, tol=0.01, max_speed=0.4)
        self.gripper_action(-1.0, n=7)
        self.move_to([xy[0], xy[1], z_release + 0.10], tol=0.02)

    def free_table_xy(self, exclude=(), margin=0.07):
        others = [self.obj_bbox(n) for n in self.names if n not in exclude]
        x0 = self.e.workspace_offset[0]
        y0 = self.e.workspace_offset[1]
        for _ in range(100):
            xy = np.array([x0 + self.rng.uniform(-0.22, 0.12), y0 + self.rng.uniform(-0.25, 0.25)])
            if self._near_fixture(xy):
                continue
            if all(np.any(xy < b["lo"][:2] - margin) or np.any(xy > b["hi"][:2] + margin)
                   for b in others):
                return xy
        return None

    def _near_fixture(self, xy):
        for fname in self.e.fixtures_dict:
            if fname.endswith("table"):
                continue
            bid = self.m.body(self.e.fixtures_dict[fname].root_body).id
            fp = self.d.xpos[bid]
            if abs(xy[1] - fp[1]) < 0.17 and abs(xy[0] - fp[0]) < 0.25:
                return True
        return False

    def _held_height_below_eef(self, name):
        bb = self.obj_bbox(name)
        return self.eef_pos[2] - bb["lo"][2]

    # High-level play primitives. Each returns (success, language).
    def p_pick_place_table(self, name):
        lang = f"pick up the {self._nice(name)} and move it somewhere else on the table"
        if not self.pick(name):
            self.gripper_action(-1, 8)
            return False, lang
        xy = self.free_table_xy(exclude=[name])
        if xy is None:
            xy = self.obj_bbox(name)["center"]
        dz = self._held_height_below_eef(name)
        self.place_at(xy, self.table_z + dz + 0.015, self.closest_yaw(self.rng.uniform(-np.pi, np.pi)))
        nb = self.obj_bbox(name)
        return (np.linalg.norm(nb["center"] - xy) < 0.05 and not self.is_grasped(name)
                and nb["lo"][2] < self.table_z + 0.03), lang

    def p_pick_place_on(self, name, target):
        lang = f"put the {self._nice(name)} on the {self._nice(target)}"
        if not self.pick(name):
            self.gripper_action(-1, 8)
            return False, lang
        tb = self.obj_bbox(target)
        dz = self._held_height_below_eef(name)
        xy = tb["center"] + self.rng.normal(0, 0.01, 2)
        self.place_at(xy, tb["hi"][2] + dz + 0.02)
        nb = self.obj_bbox(name)
        ok = (np.linalg.norm(nb["center"] - tb["center"]) < 0.08
              and nb["lo"][2] > tb["lo"][2] + 0.005)
        return ok, lang

    def p_lift_and_wave(self, name):
        lang = f"pick up the {self._nice(name)}, wave it around and put it back down"
        if not self.pick(name):
            self.gripper_action(-1, 8)
            return False, lang
        base = self.eef_pos.copy()
        base[2] = max(base[2], self.table_z + 0.2)
        for _ in range(self.rng.integers(2, 5)):
            tgt = base + np.array([self.rng.uniform(-0.08, 0.08), self.rng.uniform(-0.12, 0.12),
                                   self.rng.uniform(-0.03, 0.08)])
            self.move_to(tgt, self.closest_yaw(self.eef_yaw() + self.rng.uniform(-0.8, 0.8)),
                         tol=0.02, max_t=40)
        held = self.is_grasped(name) and self.obj_bbox(name)["lo"][2] > self.table_z + 0.04
        dz = self._held_height_below_eef(name)
        xy = self.free_table_xy(exclude=[name])
        if xy is None:
            xy = self.eef_pos[:2]
        self.place_at(xy, self.table_z + dz + 0.015)
        return held, lang

    def p_push(self, name):
        bb = self.obj_bbox(name)
        ang = self.rng.uniform(-np.pi, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        dist = self.rng.uniform(0.06, 0.14)
        r = 0.5 * np.linalg.norm(bb["hi"][:2] - bb["lo"][:2]) + 0.03
        start = bb["center"] - d * r
        z = self.table_z + min(0.03, 0.5 * (bb["hi"][2] - bb["lo"][2])) + 0.01
        dirs = ["forward", "left", "backward", "right"]
        lang = f"push the {self._nice(name)} {dirs[int(((ang + np.pi / 4) % (2 * np.pi)) // (np.pi / 2))]}"
        yaw = self.closest_yaw(ang)  # fingers perpendicular to push -> flat contact face
        self.move_to([start[0], start[1], max(self.eef_pos[2], self.table_z + 0.12)], yaw, grip=1.0, tol=0.02)
        self.move_to([start[0], start[1], z + 0.05], yaw, tol=0.01)
        self.move_to([start[0], start[1], z], yaw, tol=0.008, max_speed=0.4)
        end = bb["center"] + d * (dist - r + 0.02)
        self.move_to([end[0], end[1], z], yaw, tol=0.01, max_speed=0.35, max_t=80)
        self.move_to([end[0], end[1], z + 0.1], tol=0.02)
        moved = np.linalg.norm(self.obj_bbox(name)["center"] - bb["center"])
        return moved > 0.03, lang

    def p_drawer(self, drawer, open_it):
        jid = drawer["jid"]
        adr = self.m.jnt_qposadr[jid]
        lo, hi = self.m.jnt_range[jid]
        q0 = self.d.qpos[adr]
        g = drawer["handle"]
        R = self.d.xmat[drawer["bid"]].reshape(3, 3)
        out_dir = -R[:, 1]  # handle sits on -y side of the drawer body
        out_dir[2] = 0
        out_dir /= np.linalg.norm(out_dir)
        hp = self.d.geom_xpos[g].copy()
        verb = "open" if open_it else "close"
        lang = f"{verb} the {drawer['level']} drawer of the cabinet"
        yaw = self.closest_yaw(np.arctan2(out_dir[1], out_dir[0]) + np.pi / 2)
        # horizontal gripper pointing into the drawer, fingers closing vertically
        app = -out_dir
        close_ax = np.array([0.0, 0.0, 1.0])
        Rh = np.stack([close_ax, np.cross(app, close_ax), app], axis=1)
        if open_it:
            # side grasp the handle bar, then pull outward
            pre = hp + out_dir * 0.12
            self.move_to([pre[0], pre[1], max(self.eef_pos[2], hp[2] + 0.08)], grip=-1, tol=0.03)
            self.move_to(pre, R=Rh, tol=0.01, max_t=150)
            self.move_to(hp + out_dir * 0.005, R=Rh, tol=0.006, max_speed=0.4)
            self.gripper_action(1.0, 12)
            pull = hp + out_dir * self.rng.uniform(0.10, 0.16)
            self.move_to(pull, R=Rh, tol=0.01, max_speed=0.3, max_t=100)
            self.gripper_action(-1.0, 8)
            back = self.eef_pos + out_dir * 0.08
            self.move_to(back, R=Rh, tol=0.02, max_t=40)
            self.move_to(back + np.array([0, 0, 0.1]), yaw=self.closest_yaw(0.0), tol=0.03, max_t=80)
        else:
            # push the handle bar back to closed with a horizontal closed fist
            pre = hp + out_dir * 0.10
            self.move_to([pre[0], pre[1], max(self.eef_pos[2], hp[2] + 0.08)], grip=1.0, tol=0.03)
            self.move_to(pre, R=Rh, tol=0.01, max_t=150)
            push = hp - out_dir * (abs(q0 - hi) + 0.03)
            self.move_to(push, R=Rh, tol=0.008, max_speed=0.4, max_t=100)
            back = self.eef_pos + out_dir * 0.10
            self.move_to(back, R=Rh, tol=0.02, max_t=40)
            self.move_to(back + np.array([0, 0, 0.1]), yaw=self.closest_yaw(0.0), tol=0.03, max_t=80)
        q1 = self.d.qpos[adr]
        return abs(q1 - q0) > 0.03, lang

    def p_reach(self):
        x0, y0 = self.e.workspace_offset[:2]
        tgt = np.array([x0 + self.rng.uniform(-0.25, 0.1), y0 + self.rng.uniform(-0.25, 0.25),
                        self.table_z + self.rng.uniform(0.08, 0.3)])
        self.move_to(tgt, self.closest_yaw(self.rng.uniform(-np.pi, np.pi)),
                     grip=float(self.rng.choice([-1.0, 1.0])), tol=0.02, max_t=60)
        self.hold(int(self.rng.integers(0, 10)))
        return True, "move the arm around"

    def _nice(self, name):
        return name.rsplit("_", 1)[0].replace("_", " ")

    # ---------------------------------------------------------------- driver
    def choose_primitive(self):
        graspable = [n for n in self.names if self.obj_category(n) not in RECEPTACLES]
        opts = []
        if graspable:
            n = self.rng.choice(graspable)
            opts += [(0.30, lambda: self.p_pick_place_table(n), "pick_place", n),
                     (0.15, lambda: self.p_lift_and_wave(n), "lift_wave", n),
                     (0.15, lambda: self.p_push(n), "push", n)]
            targets = [t for t in self.names if t != n and (
                self.obj_category(t) in RECEPTACLES
                or self.obj_category(t) in {"akita_black_bowl", "white_bowl", "plate"})]
            if targets:
                t = self.rng.choice(targets)
                opts.append((0.25, lambda: self.p_pick_place_on(n, t), "pick_place_on", f"{n}->{t}"))
        if self.drawers:
            dr = self.drawers[self.rng.integers(len(self.drawers))]
            q = self.d.qpos[self.m.jnt_qposadr[dr["jid"]]]
            lo, hi = self.m.jnt_range[dr["jid"]]
            is_open = abs(q) > 0.03
            opts.append((0.25, lambda: self.p_drawer(dr, not is_open),
                         "drawer_close" if is_open else "drawer_open",
                         f"{dr['fixture']}:{dr['level']}"))
        opts.append((0.07, self.p_reach, "reach", ""))
        w = np.array([o[0] for o in opts])
        o = opts[self.rng.choice(len(opts), p=w / w.sum())]
        return o[1], o[2], o[3]

    def run_episode(self, n_primitives):
        for _ in range(n_primitives):
            fn, kind, arg = self.choose_primitive()
            self.speed = self.rng.uniform(0.10, 0.20)
            self.rot_speed = self.rng.uniform(0.7, 1.3)
            t0 = self.t
            try:
                ok, lang = fn()
            except EpisodeTooLong:
                break
            self.segments.append(dict(primitive=kind, target=arg, start=t0, end=self.t,
                                      success=bool(ok), language=lang))
            # occasionally drop anything still held
            if self.grip > 0:
                try:
                    self.gripper_action(-1.0, 6)
                except EpisodeTooLong:
                    break
        return self.segments
