"""RoboTwin 2.0 play task: random tabletop scenes and a robot that plays with
them (pick and place, put things in containers, stack, ...), recorded through
RoboTwin's own pipeline so the data has exactly the format and action
statistics of RoboTwin's scripted expert data.

Linked into RoboTwin as envs/play_table.py (see robotwin_play/README.md)."""
import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *

# tabletop objects with annotated grasp points (contact_points_pose)
GRASPABLE = [
    "001_bottle", "005_french-fries", "006_hamburg", "021_cup", "023_tissue-box", "035_apple", "039_mug",
    "046_alarm-clock", "047_mouse", "048_stapler", "049_shampoo", "050_bell", "057_toycar",
    "058_markpen", "068_boxdrink", "071_can", "073_rubikscube", "075_bread", "077_phone", "079_remotecontrol",
    "080_pillbottle", "081_playingcards", "093_brush-pen", "095_glue", "100_seal", "101_milk-tea",
    "105_sauce-can", "107_soap", "109_hydrating-oil", "111_callbell", "112_tea-box", "113_coffee-box",
    "114_bottle", "115_perfume", "118_tooth-paste",
]
# things to put things in / on (annotated functional points)
CONTAINERS = ["002_bowl", "003_plate", "008_tray", "110_basket", "062_plasticbox", "076_breadbasket"]

UPRIGHT = np.array([0.707107, 0.707107, 0.0, 0.0])   # model y-up -> world z-up
TABLE_Z = 0.741


def _model_ids(name):
    import glob, os
    return sorted(int(os.path.basename(p)[len("model_data"):-5]) for p in glob.glob(f"assets/objects/{name}/model_data*.json"))


class play_table(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        # a separate stream for scene and play choices, seeded from the episode seed
        self.prng = np.random.default_rng(kwags.get("seed", 0) * 7919 + 17)
        self.play_cfg = kwags.get("play", {})
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ scene
    def _free_xy(self, r, placed, xlim=(-0.24, 0.24), ylim=(-0.16, 0.08), tries=200):
        for _ in range(tries):
            xy = np.array([self.prng.uniform(*xlim), self.prng.uniform(*ylim)])
            if all(np.linalg.norm(xy - p) > r + q + 0.03 for p, q in placed):
                return xy
        return None

    def _spawn(self, name, xy, yaw):
        ids = _model_ids(name)
        mid = int(self.prng.choice(ids)) if ids else 0
        qz = t3d.quaternions.axangle2quat([0, 0, 1], yaw)
        q = t3d.quaternions.qmult(qz, UPRIGHT)
        actor = create_actor(self, pose=sapien.Pose([xy[0], xy[1], TABLE_Z + 0.005], q), modelname=name,
                             model_id=mid, convex=True)
        return actor, mid

    def load_actors(self):
        cfg = self.play_cfg
        self.objects, self.containers = [], []
        placed = []
        n_obj = int(self.prng.integers(cfg.get("min_objects", 3), cfg.get("max_objects", 5) + 1))
        names = list(self.prng.choice(GRASPABLE, size=n_obj, replace=False))
        if self.prng.random() < cfg.get("p_container", 0.7):
            names = [str(self.prng.choice(CONTAINERS))] + names
        for name in names:
            is_cont = name in CONTAINERS
            r = 0.09 if is_cont else 0.05
            xy = self._free_xy(r, placed)
            if xy is None:
                continue
            actor, mid = self._spawn(name, xy, self.prng.uniform(-np.pi, np.pi))
            actor.set_mass(0.5 if is_cont else 0.05)
            placed.append((xy, r))
            (self.containers if is_cont else self.objects).append(actor)
            self.add_prohibit_area(actor, padding=0.02)
        self.scene_desc = [dict(name=a.get_name(), container=a in self.containers) for a in self.containers + self.objects]

    # --------------------------------------------------------------- helpers
    def _aabb(self, actor):
        import sapien.physx as px
        comp = actor.actor.find_component_by_type(px.PhysxRigidDynamicComponent)
        lo, hi = comp.compute_global_aabb_tight()
        return np.asarray(lo), np.asarray(hi)

    def _name(self, actor):
        return actor.get_name().split("_", 1)[-1].replace("-", " ").replace("_", " ")

    def _holds(self, arm, actor):
        """Both fingers of this hand touch the object."""
        g = self.robot.left_gripper if arm == "left" else self.robot.right_gripper
        fingers = {j.get_child_link().get_name() for j, _, _ in g}
        name, touch = actor.get_name(), set()
        for c in self.scene.get_contacts():
            n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
            if n0 == name and n1 in fingers:
                touch.add(n1)
            elif n1 == name and n0 in fingers:
                touch.add(n0)
        return len(touch) >= 2

    def _arm_for(self, xy):
        return ArmTag("left" if xy[0] < 0 else "right")

    def _home_other(self, arm):
        """The idle arm goes home at the same time (as in the expert data)."""
        other = arm.opposite
        cur = np.array(self.robot.get_left_ee_pose() if other == "left" else self.robot.get_right_ee_pose())
        home = np.array(self.robot.left_original_pose if other == "left" else self.robot.right_original_pose)
        if np.linalg.norm(cur[:3] - home[:3]) > 0.03:
            return self.back_to_origin(arm_tag=other)
        return None

    def _do(self, a, b=None):
        if b is None:
            return self.move(a)
        return self.move(a, b)

    def _recover(self, arm):
        """After a failed plan: open the hand and bring the arm home."""
        self.plan_success = True
        if not (self.is_left_gripper_open() if arm == "left" else self.is_right_gripper_open()):
            self.move(self.open_gripper(arm_tag=arm))
        self._exec(self.move_by_displacement(arm_tag=arm, z=0.06))
        self._exec(self.back_to_origin(arm_tag=arm))
        self.plan_success = True

    def _table_spot(self, arm, actor, min_move=0.08):
        """A free table spot on this arm's side."""
        xs = (-0.25, -0.06) if arm == "left" else (0.06, 0.25)
        p0 = actor.get_pose().p[:2]
        others = [a for a in self.objects + self.containers if a is not actor]
        for _ in range(100):
            xy = np.array([self.prng.uniform(*xs), self.prng.uniform(-0.15, 0.08)])
            if np.linalg.norm(xy - p0) < min_move:
                continue
            if all(np.linalg.norm(xy - o.get_pose().p[:2]) > 0.09 for o in others):
                return xy
        return None

    # ---------------------------------------------------- plan before acting
    def _plan(self, arm, pose, constraint=None, q=None):
        f = self.robot.left_plan_path if arm == "left" else self.robot.right_plan_path
        try:
            return f(list(pose), constraint_pose=constraint, last_qpos=q)
        except Exception as e:
            if __import__("os").environ.get("PLAY_DEBUG"):
                import traceback; traceback.print_exc()
            return None

    def _end_q(self, arm, res, q=None):
        ent = self.robot.left_entity if arm == "left" else self.robot.right_entity
        names = [j.get_name() for j in ent.get_active_joints()]
        arm_names = self.robot.left_arm_joints_name if arm == "left" else self.robot.right_arm_joints_name
        q = np.array(ent.get_qpos() if q is None else q, dtype=np.float32).copy()
        for k, n in enumerate(arm_names):
            q[names.index(n)] = res["position"][-1][k]
        return q

    JOINT_MARGIN = 0.1   # rad beyond the range the expert data uses

    def _joint_box(self, arm):
        """Per-joint range of RoboTwin's released expert data for this robot
        (robotwin_play/expert_joint_ranges_<robot>.json); None if unknown."""
        if not hasattr(self, "_jbox"):
            import json, os
            emb = str(self.play_cfg.get("embodiment_name", "aloha-agilex"))
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"expert_joint_ranges_{emb}.json")
            self._jbox = json.load(open(path)) if os.path.exists(path) else None
        if self._jbox is None:
            return None
        b = self._jbox[str(arm)]
        return np.array(b["lo"]) - self.JOINT_MARGIN, np.array(b["hi"]) + self.JOINT_MARGIN

    def _feasible(self, arm, chain, times=2):
        """Can the arm do these moves one after the other (list of (pose,
        constraint))? cuRobo is randomized, so a borderline plan has to pass
        `times` times in a row. Plans whose joints leave the range of the
        released expert data are rejected (keeps the actions in-distribution)."""
        box = self._joint_box(arm)
        for _ in range(times):
            q = None
            for pose, cons in chain:
                r = self._plan(arm, pose, cons, q)
                if r is None or r.get("status") != "Success":
                    return False
                if box is not None:
                    P = np.asarray(r["position"])[:, :len(box[0])]
                    if np.any(P < box[0]) or np.any(P > box[1]):
                        return False
                q = self._end_q(arm, r, q)
        return True

    def _sync(self, arm, tol=0.05, rate=0.004):
        """If the arm has been pushed away from where it was told to be (it is
        pressing on something), bring the command smoothly to where it really
        is before planning from there - otherwise the recorded command jumps."""
        tgt = np.array((self.robot.get_left_arm_jointState() if arm == "left" else self.robot.get_right_arm_jointState())[:-1])
        real = np.array((self.robot.get_left_arm_real_jointState() if arm == "left" else self.robot.get_right_arm_real_jointState())[:-1])
        gap = float(np.abs(real - tgt).max())
        if gap <= tol:
            return
        n = int(np.ceil(gap / rate))
        s = np.linspace(0, 1, n + 1)[1:]
        s = 3 * s ** 2 - 2 * s ** 3
        pos = tgt[None] + s[:, None] * (real - tgt)[None]
        vel = np.gradient(pos, axis=0) * 250.0
        seq = {"left_arm": None, "left_gripper": None, "right_arm": None, "right_gripper": None}
        seq[f"{arm}_arm"] = {"position": pos, "velocity": vel}
        self.take_dense_action(seq)
        self.n_syncs = getattr(self, "n_syncs", 0) + 1

    def _exec(self, a, b=None, tries=3):
        """Execute a move; a failed plan moves nothing, so just plan again."""
        for k in range(tries):
            self.plan_success = True
            for x in (a, b):
                if x is not None and x[0] is not None:
                    self._sync(x[0])
            self._do(a, b)
            if self.plan_success:
                return True
        return False

    def _grasp_options(self, actor, arm, pre_dis=0.1, k=4):
        """Grasp points ranked like RoboTwin's chooser (prefer top-down, then the
        arm's preferred side), with some randomness so the same object is not
        always grabbed the same way."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
        from envs.utils.transforms import cal_quat_dis
        pref = self.robot.get_grasp_perfect_direction(arm)
        opts = []
        for i, _ in actor.iter_contact_points():
            pre = self.get_grasp_pose(actor, arm, contact_point_id=i, pre_dis=pre_dis)
            if pre is None:
                continue
            pre = np.array(pre, dtype=np.float64)
            pose = pre.copy()
            pose[:3] += np.array([pre_dis, 0, 0]) @ np.linalg.inv(t3d.quaternions.quat2mat(pre[3:]))
            td = cal_quat_dis(pose[3:], GRASP_DIRECTION_DIC["top_down_little_left" if arm == "right" else "top_down_little_right"])
            sd = cal_quat_dis(pose[3:], GRASP_DIRECTION_DIC[pref])
            opts.append((0.7 * td + 0.3 * sd + self.prng.normal(0, 0.08), i, pre, pose))
        opts.sort(key=lambda x: x[0])
        return opts[:k]

    def _grasp_and_lift(self, actor, arm, lift=0.1, other_arm_ok=True, exclude=(), home_other=True):
        """Grasp with a checked-feasible grasp point (this arm, else the other
        one) and lift. Returns (lifted, arm) or (None, arm) if no grasp can be
        planned (nothing has moved then)."""
        z0 = actor.get_pose().p[2]
        for arm_ in ((arm, arm.opposite) if other_arm_ok else (arm,)):
            for _, cid, pre, pose in self._grasp_options(actor, arm_):
                if cid in exclude:
                    continue
                up = pose.copy(); up[2] += lift
                if not self._feasible(arm_, [(pre, None), (pose, [1, 1, 1, 0, 0, 0]), (up, None)]):
                    continue
                home = self._home_other(arm_) if home_other else None
                acts = (arm_, [Action(arm_, "move", target_pose=list(pre)),
                               Action(arm_, "move", target_pose=list(pose), constraint_pose=[1, 1, 1, 0, 0, 0]),
                               Action(arm_, "close", target_gripper_pos=0.0)])
                self._exec(acts, home)
                if not self.plan_success:
                    if __import__("os").environ.get("PLAY_DEBUG"):
                        print(f"   EXEC-FAIL grasp {actor.get_name()} arm={arm_} cid={cid} with_home={home is not None}", flush=True)
                    return False, arm_
                if lift > 0:
                    self._exec(self.move_by_displacement(arm_tag=arm_, z=lift))
                self.last_grasp = dict(contact_point=int(cid), arm=str(arm_))
                return bool(self.plan_success and (lift <= 0 or actor.get_pose().p[2] - z0 > 0.03)), arm_
        return None, arm

    def _place(self, actor, arm, target, pre_dis, dis, constrain="free", **kw):
        """Place at target if that can be planned; returns True if it moved."""
        pre = self.get_place_pose(actor, arm, target, constrain=constrain, pre_dis=pre_dis, **kw)
        pose = self.get_place_pose(actor, arm, target, constrain=constrain, pre_dis=dis, **kw)
        if not self._feasible(arm, [(pre, None), (pose, None)]):
            return False
        self._exec((arm, [Action(arm, "move", target_pose=list(pre)), Action(arm, "move", target_pose=list(pose)),
                          Action(arm, "open", target_gripper_pos=1.0)]))
        if not self.plan_success and __import__("os").environ.get("PLAY_DEBUG"):
            print(f"   EXEC-FAIL place {actor.get_name()} arm={arm}", flush=True)
        return self.plan_success

    # ------------------------------------------------------------- actions
    def _reaches(self, arm, xy, z=0.9):
        """Can this arm get its hand above xy (pointing down)?"""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
        q = GRASP_DIRECTION_DIC["top_down_little_left" if arm == "right" else "top_down_little_right"]
        return self._feasible(arm, [([xy[0], xy[1], z] + list(q), None)], times=1)

    def _start(self, actor, lang, lift=0.1, dest=None):
        arm = self._arm_for(actor.get_pose().p)
        if dest is not None:
            # the arm that can reach where it is going, if it can also pick it up
            for a in (self._arm_for(dest), arm):
                if self._reaches(a, dest) and self._reaches(a, actor.get_pose().p[:2]):
                    arm = a
                    break
        ok, arm = self._grasp_and_lift(actor, arm, lift=lift)
        if ok is None:
            return None, arm     # no plannable grasp: not attempted
        if not ok:
            self._recover(arm)
            return False, arm
        return True, arm

    def _finish(self, arm, placed, ok, lang):
        if placed:
            self._exec(self.move_by_displacement(arm_tag=arm, z=0.07))
        if not placed or not self.plan_success:
            # couldn't plan the put-down: set it down where the hand is
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=-0.05))
            self._recover(arm)
        return dict(success=bool(ok and placed), language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))

    def p_move(self, actor):
        lang = f"move the {self._name(actor)} to another spot"
        st, arm = self._start(actor, lang)
        if st is None:
            return None
        if not st:
            return dict(success=False, language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        placed, xy = False, None
        for _ in range(4):
            xy = self._table_spot(arm, actor)
            if xy is not None and self._place(actor, arm, [xy[0], xy[1], TABLE_Z, 1, 0, 0, 0], 0.05, 0.01):
                placed = True
                break
        ok = placed and np.linalg.norm(actor.get_pose().p[:2] - xy) < 0.05
        return self._finish(arm, placed, ok, lang)

    def p_put_in(self, actor, cont):
        lang = f"put the {self._name(actor)} in the {self._name(cont)}"
        dest = np.array(cont.get_functional_point(0)[:2])
        p = actor.get_pose().p[:2]
        can = [a for a in (self._arm_for(dest), self._arm_for(dest).opposite)
               if self._reaches(a, dest) and self._reaches(a, p)]
        if not can:
            if self._reaches(self._arm_for(dest), dest) and self._reaches(self._arm_for(p), p) \
                    and self._arm_for(dest) != self._arm_for(p):
                # one hand can't do both: pick it up with one, pass it, put it in with the other
                res = self.p_handover(actor, dest=cont.get_functional_point(0))
                if res is not None:
                    lo, hi = self._aabb(cont)
                    q = actor.get_pose().p
                    res.update(language=f"pass the {self._name(actor)} over and put it in the {self._name(cont)}",
                               success=bool(np.all(q[:2] > lo[:2]) and np.all(q[:2] < hi[:2])), via="handover")
                return res
            return None
        st, arm = self._start(actor, lang, lift=0.12, dest=dest)
        if st is None:
            return None
        if not st:
            return dict(success=False, language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        placed = self._place(actor, arm, cont.get_functional_point(0), 0.08, 0.03)
        lo, hi = self._aabb(cont)
        p = actor.get_pose().p
        ok = np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2])
        return self._finish(arm, placed, ok, lang)

    def p_stack(self, actor, base):
        lang = f"put the {self._name(actor)} on the {self._name(base)}"
        st, arm = self._start(actor, lang, lift=0.12, dest=base.get_pose().p[:2])
        if st is None:
            return None
        if not st:
            return dict(success=False, language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        lo, hi = self._aabb(base)
        bp = base.get_pose().p
        placed = self._place(actor, arm, [bp[0], bp[1], hi[2], 1, 0, 0, 0], 0.06, 0.02)
        p = actor.get_pose().p
        alo, _ = self._aabb(actor)
        ok = alo[2] > hi[2] - 0.02 and np.linalg.norm(p[:2] - bp[:2]) < 0.06
        return self._finish(arm, placed, ok, lang)

    def p_handover(self, actor, dest=None):
        """Pick it up with one arm, pass it to the other in the middle, put it
        down on the other side (or in a container there)."""
        a1 = self._arm_for(actor.get_pose().p)
        a2 = a1.opposite
        what = self._name(actor)
        lang = f"pass the {what} from the {a1} hand to the {a2} hand"
        ok, a1 = self._grasp_and_lift(actor, a1, lift=0.12, other_arm_ok=False)
        if ok is None:
            return None
        if not ok:
            self._recover(a1)
            return dict(success=False, language=lang, arm=str(a1), grasp=getattr(self, "last_grasp", None))
        cid1 = self.last_grasp["contact_point"]
        mid = [self.prng.uniform(-0.03, 0.03), self.prng.uniform(-0.1, -0.02), 0.86, 1, 0, 0, 0]
        if not self._place_hold(actor, a1, mid):
            self._finish(a1, False, False, lang)
            return dict(success=False, language=lang, arm=str(a1), grasp=getattr(self, "last_grasp", None))
        ok2, _ = self._grasp_and_lift(actor, a2, lift=0.0, other_arm_ok=False, exclude=(cid1,), home_other=False)
        if ok2 and not self._holds(a2, actor):
            # closed but not really holding it: let go again before the other hand does
            self.move(self.open_gripper(arm_tag=a2))
            self._exec(self.move_by_displacement(arm_tag=a2, z=0.05, move_axis="arm"))
            self._exec(self.back_to_origin(arm_tag=a2))
            ok2 = False
        if not ok2:
            # the other hand can't take it: set it down in the middle
            self.plan_success = True
            self._place(actor, a1, [mid[0], mid[1], TABLE_Z, 1, 0, 0, 0], 0.05, 0.01)
            self._finish(a1, True, False, lang)
            return dict(success=False, language=lang, arm=str(a1), grasp=getattr(self, "last_grasp", None))
        self.move(self.open_gripper(arm_tag=a1))
        self._exec(self.move_by_displacement(arm_tag=a1, z=0.06, move_axis="arm"))
        home = self.back_to_origin(arm_tag=a1)
        self._exec(self.move_by_displacement(arm_tag=a2, z=0.05), home)
        target = dest if dest is not None else self._table_spot(a2, actor)
        placed = False
        if target is not None:
            tp = list(target) if len(target) == 7 else [target[0], target[1], TABLE_Z, 1, 0, 0, 0]
            placed = self._place(actor, a2, tp, 0.06, 0.02)
        return dict(self._finish(a2, placed, placed, lang), arm=f"{a1}->{a2}")

    def _place_hold(self, actor, arm, target):
        """Bring the held object to target without letting go."""
        pose = self.get_place_pose(actor, arm, target, constrain="free", pre_dis=0.0)
        if not self._feasible(arm, [(pose, None)]):
            return False
        return self._exec((arm, [Action(arm, "move", target_pose=list(pose))]))

    def p_push(self, actor):
        """Push it a little along the table with the closed hand."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
        p = actor.get_pose().p
        lo, hi = self._aabb(actor)
        h = hi[2] - lo[2]
        r = 0.5 * np.linalg.norm((hi - lo)[:2])
        arm = self._arm_for(p)
        q = GRASP_DIRECTION_DIC["top_down_little_left" if arm == "right" else "top_down_little_right"]
        others = [o for o in self.objects + self.containers if o is not actor]
        for _ in range(8):
            ang = self.prng.uniform(-np.pi, np.pi)
            d = np.array([np.cos(ang), np.sin(ang)])
            dist = self.prng.uniform(0.06, 0.12)
            end = p[:2] + d * dist
            if not (-0.28 < end[0] < 0.28 and -0.2 < end[1] < 0.1):
                continue
            if any(np.linalg.norm(end - o.get_pose().p[:2]) < 0.08 for o in others):
                continue
            z = lo[2] + np.clip(0.5 * h, 0.015, 0.05) + 0.11   # ee is ~11 cm above the fingertips
            start = p[:2] - d * (r + 0.04)
            above = [start[0], start[1], z + 0.08] + list(q)
            down = [start[0], start[1], z] + list(q)
            through = [end[0] - d[0] * r, end[1] - d[1] * r, z] + list(q)
            if not self._feasible(arm, [(above, None), (down, [1, 1, 1, 0, 0, 0]), (through, [1, 1, 1, 0, 0, 0])]):
                continue
            words = ["right", "away", "left", "closer"][int(((ang + np.pi / 4) % (2 * np.pi)) // (np.pi / 2))] \
                if True else ""
            lang = f"push the {self._name(actor)} {words}"
            home = self._home_other(arm)
            self._exec(self.close_gripper(arm_tag=arm), home)
            self._exec((arm, [Action(arm, "move", target_pose=above),
                              Action(arm, "move", target_pose=down, constraint_pose=[1, 1, 1, 0, 0, 0]),
                              Action(arm, "move", target_pose=through, constraint_pose=[1, 1, 1, 0, 0, 0])]))
            self._exec(self.move_by_displacement(arm_tag=arm, z=0.08))
            self.move(self.open_gripper(arm_tag=arm))
            moved = float(np.linalg.norm(actor.get_pose().p[:2] - p[:2]))
            if not self.plan_success:
                self._recover(arm)
            return dict(success=bool(moved > 0.5 * dist), language=lang, arm=str(arm), moved=moved)
        return None

    def p_reorient(self, actor):
        """Pick it up, turn it round, put it back down about where it was."""
        lang = f"turn the {self._name(actor)} around"
        st, arm = self._start(actor, lang, lift=0.1)
        if st is None:
            return None
        if not st:
            return dict(success=False, language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        ee = np.array(self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose())
        yaw0 = None
        for ang in self.prng.permutation([0.8, -0.8, 1.4, -1.4, 2.0, -2.0]):
            qz = t3d.quaternions.axangle2quat([0, 0, 1], ang)
            tgt = list(ee[:3]) + list(t3d.quaternions.qmult(qz, ee[3:]))
            if self._feasible(arm, [(tgt, None)]):
                yaw0 = actor.get_pose().q
                self._exec((arm, [Action(arm, "move", target_pose=tgt)]))
                break
        p = actor.get_pose().p
        placed = self._place(actor, arm, [p[0], p[1], TABLE_Z, 1, 0, 0, 0], 0.04, 0.01)
        turned = yaw0 is not None and abs(t3d.quaternions.qmult(t3d.quaternions.qinverse(yaw0), actor.get_pose().q)[0]) < 0.97
        return self._finish(arm, placed, turned, lang)

    # ---------------------------------------------------------------- loop
    def _choose(self):
        objs = [a for a in self.objects if TABLE_Z - 0.05 < a.get_pose().p[2] < TABLE_Z + 0.3]
        opts = []
        last = getattr(self, "_last_kind", None)
        for a in objs:
            opts.append((0.8, "move", (a,)))
            opts.append((0.4, "push", (a,)))
            opts.append((0.35, "reorient", (a,)))
            opts.append((0.35, "handover", (a,)))
            for c in self.containers:
                opts.append((1.0 / max(len(self.containers), 1), "put_in", (a, c)))
            for b in objs:
                if b is not a:
                    opts.append((0.3 / max(len(objs) - 1, 1), "stack", (a, b)))
        # the same kind of thing twice in a row is less likely
        opts = [(w * (0.4 if k == last else 1.0), k, x) for w, k, x in opts]
        w = np.array([o[0] for o in opts])
        return opts[int(self.prng.choice(len(opts), p=w / w.sum()))] if opts else None

    def play_once(self):
        cfg = self.play_cfg
        segs = []
        tries = 0
        while len(segs) < cfg.get("max_actions", 10) and tries < cfg.get("max_tries", 40):
            tries += 1
            if self.FRAME_IDX >= cfg.get("max_frames", 900):
                break
            ch = self._choose()
            if ch is None:
                break
            _, kind, args = ch
            f0 = self.FRAME_IDX
            res = getattr(self, "p_" + kind)(*args)
            self.plan_success = True
            self._last_kind = kind
            if res is None:
                continue
            res.update(primitive=kind, target=[a.get_name() for a in args], start=f0, end=self.FRAME_IDX)
            segs.append(res)
        self.info["info"] = {"scene": self.scene_desc, "segments": segs}
        return self.info

    def check_success(self):
        return True
