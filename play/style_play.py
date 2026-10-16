"""Play driven by the learned human motor-style policy.

Keyframes (grasp, release, retreat, drawer hook / pull end) come from scene
geometry exactly as in composed_play; the motion between keyframes and the
gripper timing come from the goal-conditioned policy trained on human demos
(style_policy.py). Light gates keep it honest: the gripper may only close when
the hand is within ~1.2 cm of the grasp pose, and a small correction is
blended in very close to a keyframe so it lands where the geometry says."""
import os
from collections import deque

import numpy as np

from compose import R_DOWN, rotz, rot_err, soft_limit
from composed_play import ComposedPlay, small_tilt
from human_play import GEOM, GROUP_OF, RECEPTACLE_GROUPS
from style_policy import StylePolicy, features, H

HERE = os.path.dirname(os.path.abspath(__file__))
_POL = None


def get_policy():
    global _POL
    if _POL is None:
        _POL = StylePolicy(os.path.join(HERE, "style_policy.pt"))
    return _POL


class StylePlay(ComposedPlay):
    def __init__(self, pc, rng, bank, p_explore=0.2, sample_scale=0.3, precise_radius=0.03):
        super().__init__(pc, rng, bank, p_explore=p_explore)
        self.pol = get_policy()
        self.sample_scale = sample_scale
        self.precise_radius = precise_radius
        self.base = np.array(pc.robot.base_pos, float)
        self.pos_hist = deque([pc.eef_pos.copy()] * (H + 1), maxlen=H + 1)
        self.act_hist = deque([np.zeros(7)] * H, maxlen=H)
        self.noise = np.zeros(6)

    def _act(self, goal, Rg, typ):
        pc = self.pc
        P = np.array(self.pos_hist)
        vel = np.diff(P, axis=0)[-H:]
        gw = float(pc.obs["robot0_gripper_qpos"][0] - pc.obs["robot0_gripper_qpos"][1])
        x = features(pc.eef_pos, pc.eef_mat, vel, np.array(self.act_hist), gw, goal, Rg, typ, self.base)
        mu, sd, pg = self.pol(x)
        # temporally correlated sampling (human variability without jitter)
        self.noise = 0.85 * self.noise + np.sqrt(1 - 0.85 ** 2) * self.rng.normal(size=6)
        a = np.zeros(7)
        a[:6] = mu + self.sample_scale * sd * self.noise
        return a, pg

    def _step(self, a):
        pc = self.pc
        a[:3] = soft_limit(a[:3], 0.75, 0.97)
        a[3:6] = soft_limit(a[3:6], 0.22, 0.42)
        pc.grip = a[6]
        pc.step_raw(a)
        self.pos_hist.append(pc.eef_pos.copy())
        self.act_hist.append(a.copy())

    def go(self, goal, Rg, typ, grip_rule, max_steps=160, tol=0.006, done_rule=None):
        """Drive toward a keyframe with the style policy.
        grip_rule(dist, pg, t) -> gripper command; done_rule(dist, t, state) -> bool."""
        pc = self.pc
        state = {"closed_for": 0, "opened_for": 0}
        for t in range(max_steps):
            dist = np.linalg.norm(goal - pc.eef_pos)
            a, pg = self._act(goal, Rg, typ)
            # very close to the keyframe, blend in a small precise correction
            if dist < self.precise_radius:
                w = 1 - dist / self.precise_radius
                a[:3] += w * np.clip((goal - pc.eef_pos) / 0.011 * 0.25, -0.4, 0.4)
                a[3:6] += w * np.clip(rot_err(Rg, pc.eef_mat) / 0.11 * 0.2, -0.2, 0.2)
            a[6] = grip_rule(dist, pg, t)
            state["closed_for"] = state["closed_for"] + 1 if a[6] > 0 else 0
            state["opened_for"] = state["opened_for"] + 1 if a[6] < 0 else 0
            self._step(a)
            if done_rule is not None and done_rule(dist, t, state):
                return True, t
            if done_rule is None and dist < tol:
                return True, t
        return False, max_steps

    # ------------------------------------------------------------ primitives
    def p_pick_place(self, n, explore=False, max_attempts=3):
        pc = self.pc
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        dest = self._place_goal(n)
        if dest is None:
            return None
        kind, goal, label = dest
        attempts, grasped = 0, False
        while attempts < max_attempts:
            attempts += 1
            gp = self.grasp_pose(n, explore=explore and attempts == 1)
            if gp is None:
                return None
            g, Rg = gp
            obj0, _ = self.obj_pose(n)
            pc.phase = "approach"
            # close only when the policy wants to and the hand is really there
            ok, _ = self.go(g, Rg, "close",
                            grip_rule=lambda dist, pg, t: 1.0 if (dist < 0.012 and pg > 0.5) or dist < 0.005 else -1.0,
                            done_rule=lambda dist, t, st: st["closed_for"] >= 5, max_steps=220)
            if not ok:  # never closed: close in place
                pc.phase = "grasp"
                for _ in range(6):
                    a, _ = self._act(g, Rg, "close"); a[6] = 1.0; self._step(a)
            # carry to the release keyframe; open only near it
            rel = g - obj0
            h_obj = (GEOM[self.cat(n)]["hi"][2] - GEOM[self.cat(n)]["lo"][2]) if self.cat(n) in GEOM else 0.05
            drop = self.rng.uniform(0.003, 0.012) if h_obj > 0.08 else self.rng.uniform(0.01, 0.025)
            release = goal + rel + np.r_[0, 0, drop]
            R_rel = rotz(self.rng.normal(0, 0.12)) @ Rg
            z0 = obj0[2]
            st = {"checked": False, "fail": False}

            def done_carry(dist, t, s):
                if t == 22 and not st["checked"]:
                    st["checked"] = True
                    if not (self.obj_pose(n)[0][2] - z0 > 0.015 and pc.is_grasped(n)):
                        st["fail"] = True
                        return True
                return s["opened_for"] >= 3

            pc.phase = "transport"
            self.go(release, R_rel, "open",
                    grip_rule=lambda dist, pg, t: -1.0 if (dist < 0.012 and pg < 0.5) or dist < 0.005 else 1.0,
                    done_rule=done_carry, max_steps=260)
            if st["fail"]:
                pc.phase = "regrasp"
                up = pc.eef_pos + np.r_[0, 0, 0.06]
                self.go(up, pc.eef_mat, "end", grip_rule=lambda *a: -1.0, max_steps=25, tol=0.01)
                continue
            grasped = True
            break
        # retreat with the policy toward a point above
        pc.phase = "retreat"
        up = pc.eef_pos + np.r_[self.rng.normal(0, 0.02, 2), self.rng.uniform(0.06, 0.12)]
        self.go(up, pc.eef_mat, "end", grip_rule=lambda *a: -1.0, max_steps=30, tol=0.015)
        p_end, _ = self.obj_pose(n)
        placed = grasped and np.linalg.norm(p_end[:2] - goal[:2]) < 0.05
        if kind == "on":
            lang, prim = f"put the {nice} on the {label.rsplit('_', 1)[0].replace('_', ' ')}", "pick_place_on"
        elif kind == "in_drawer":
            lang, prim = f"put the {nice} in the {label.split(':')[1]} drawer", "pick_place_drawer"
        else:
            lang, prim = f"pick up the {nice} and put it down somewhere else", "pick_place"
        return dict(primitive=prim, target=f"{n}->{label}", success=bool(placed), grasped=grasped,
                    attempts=attempts, language=lang, explore=bool(explore),
                    diag=dict(place_err=float(np.linalg.norm(p_end[:2] - goal[:2])), upright=bool(self.upright(n))))

    def p_reach(self, n):
        gp = self.grasp_pose(n)
        if gp is None:
            return None
        g, R = gp
        hover = g + np.r_[0, 0, self.rng.uniform(0.04, 0.1)]
        self.pc.phase = "approach"
        self.go(hover, R, "close", grip_rule=lambda *a: -1.0, max_steps=int(self.rng.integers(20, 50)), tol=0.02)
        away = self.pc.eef_pos + np.r_[self.rng.normal(0, 0.06, 2), self.rng.uniform(0.04, 0.1)]
        self.go(away, self.pc.eef_mat, "end", grip_rule=lambda *a: -1.0, max_steps=30, tol=0.02)
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        return dict(primitive="reach_toward", target=n, success=True, language=f"reach toward the {nice}")

    def p_fumble(self, n):
        gp = self.grasp_pose(n, explore=True)
        if gp is None:
            return None
        g, R = gp
        g = g + np.r_[self.rng.normal(0, 0.02, 2), 0.01]
        self.pc.phase = "approach"
        self.go(g, R, "close", grip_rule=lambda dist, pg, t: 1.0 if dist < 0.015 else -1.0,
                done_rule=lambda d, t, st: st["closed_for"] >= 6, max_steps=200)
        up = self.pc.eef_pos + np.r_[0, 0, 0.05]
        self.go(up, R, "end", grip_rule=lambda dist, pg, t: 1.0 if t < 8 else -1.0, max_steps=30, tol=0.015)
        nice = n.rsplit("_", 1)[0].replace("_", " ")
        return dict(primitive="pick_fumble", target=n, success=False, grasped=False, attempts=1,
                    language=f"try to pick up the {nice}")

    def p_drawer(self, dr, want_open, explore=False, params=None):
        lv = dr["level"]
        if lv != "top":
            return super().p_drawer(dr, want_open, explore=explore, params=params)
        pc = self.pc
        adr = pc.m.jnt_qposadr[dr["jid"]]
        q0 = pc.d.qpos[adr]
        hp, out = self._drawer_frame(dr)
        if want_open:
            R = rotz(self._yaw_for_axis(out[:2])) @ R_DOWN @ small_tilt(self.rng, 4.0)
            hook = hp + out * self.rng.uniform(0.04, 0.048) + np.r_[0, 0, self.rng.uniform(-0.013, -0.008)]
            above = hook + np.r_[0, 0, self.rng.uniform(0.04, 0.055)]
            pc.phase = "approach"
            self.go(above, R, "artic_start", grip_rule=lambda *a: -1.0, max_steps=200, tol=0.006)
            pc.phase = "correct"
            self.c.settle(above, R, -1.0, tol=0.003, max_steps=12)
            self.c.move([(self.rng.uniform(0.6, 0.9), hook)], R, grip=-1.0, residual=0.0)
            pc.phase = "articulate"
            end = hook + out * self.rng.uniform(0.11, 0.17) + np.r_[0, 0, 0.004]
            self.go(end, R, "artic_end", grip_rule=lambda *a: -1.0, max_steps=80, tol=0.01)
        else:
            R = rotz(self._yaw_for_axis(np.cross([0, 0, 1], out)[:2])) @ R_DOWN @ small_tilt(self.rng, 5.0)
            front = hp + out * 0.05 + np.r_[0, 0, 0.005]
            pc.phase = "approach"
            self.go(front, R, "artic_start", grip_rule=lambda *a: 1.0, max_steps=200, tol=0.01)
            pc.phase = "articulate"
            push = hp - out * (abs(q0) + 0.025) + np.r_[0, 0, 0.005]
            self.go(push, R, "artic_end", grip_rule=lambda *a: 1.0, max_steps=60, tol=0.01)
        pc.phase = "retreat"
        up = pc.eef_pos + out * 0.04 + np.r_[0, 0, self.rng.uniform(0.06, 0.1)]
        self.go(up, pc.eef_mat, "end", grip_rule=lambda *a: -1.0, max_steps=30, tol=0.015)
        dq = pc.d.qpos[adr] - q0
        ok = dq < -0.06 if want_open else dq > 0.04
        return dict(primitive=f"drawer_{'open' if want_open else 'close'}", target=f"{lv} drawer",
                    success=bool(ok), language=f"{'open' if want_open else 'close'} the {lv} drawer of the cabinet",
                    explore=bool(explore))
