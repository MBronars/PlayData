"""RoboTwin 2.0 play task: random tabletop scenes and a robot that plays with
them, recorded through RoboTwin's own pipeline so the data has exactly the
format and action statistics of RoboTwin's scripted expert data.

Every motion is planned first with RoboTwin's planner (cuRobo), checked (plan
succeeded, joints stay within the range the released expert data uses), and
then that same plan is executed through RoboTwin's `move()` in its replay
mode, so what was checked is what gets recorded.

Skills (one per family in RoboTwin's 50-task suite): move, put in a
container, stack, line up, hand over between arms, push, turn around, press
down on something, shake, knock one object with another, pick two things at
once, lift something with both hands, tip a container out.

Linked into RoboTwin as envs/play_table.py (see robotwin_play/README.md)."""
import glob
import json
import os

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC, ROOT_PATH
from .robot import Robot
from .utils import *
from .utils.transforms import cal_quat_dis

# tabletop objects with annotated grasp points (contact_points_pose)
GRASPABLE = [
    "001_bottle", "005_french-fries", "006_hamburg", "021_cup", "023_tissue-box", "035_apple", "039_mug",
    "046_alarm-clock", "047_mouse", "048_stapler", "049_shampoo", "050_bell", "057_toycar",
    "058_markpen", "068_boxdrink", "071_can", "073_rubikscube", "075_bread", "077_phone", "079_remotecontrol",
    "080_pillbottle", "081_playingcards", "093_brush-pen", "095_glue", "100_seal", "101_milk-tea",
    "105_sauce-can", "107_soap", "109_hydrating-oil", "111_callbell", "112_tea-box", "113_coffee-box",
    "114_bottle", "115_perfume", "118_tooth-paste", "020_hammer", "084_woodenmallet",
]
# things to put things in / on (annotated functional points)
CONTAINERS = ["002_bowl", "003_plate", "008_tray", "110_basket", "062_plasticbox", "076_breadbasket"]
OPEN_TOP = ["002_bowl", "110_basket", "062_plasticbox"]   # can be tipped out
TOOLS = ["020_hammer", "084_woodenmallet", "100_seal", "058_markpen", "093_brush-pen"]
# flat-topped things something can stand on (as in RoboTwin's block / bowl stacking)
STACK_BASES = ["023_tissue-box", "112_tea-box", "113_coffee-box", "081_playingcards", "073_rubikscube", "107_soap",
               "077_phone", "079_remotecontrol", "068_boxdrink", "071_can", "105_sauce-can", "101_milk-tea",
               "047_mouse", "075_bread", "006_hamburg", "002_bowl", "003_plate"]

UPRIGHT = np.array([0.707107, 0.707107, 0.0, 0.0])   # model y-up -> world z-up
TABLE_Z = 0.741
TIP = 0.156      # closed fingertips: this far along the hand's approach axis (its x) from the ee frame
CONS = [1, 1, 1, 0, 0, 0]   # keep the hand's orientation on this leg
TOPDOWN = {"left": GRASP_DIRECTION_DIC["top_down_little_right"], "right": GRASP_DIRECTION_DIC["top_down_little_left"]}
SIDEWAYS = {"left": GRASP_DIRECTION_DIC["front_right"], "right": GRASP_DIRECTION_DIC["front_left"]}


def _model_ids(name):
    return sorted(int(os.path.basename(p)[len("model_data"):-5]) for p in glob.glob(f"assets/objects/{name}/model_data*.json"))


def _footprint_radius(name, mid):
    """Half the horizontal diagonal of the model's bounding box (model y is up)."""
    j = json.load(open(f"assets/objects/{name}/model_data{mid}.json"))
    e, sc = np.asarray(j["extents"], dtype=float), np.asarray(j["scale"], dtype=float)
    return 0.5 * float(np.hypot(e[0] * sc[0], e[2] * sc[2]))


def _rot(q, axis, ang):
    """quaternion q turned by ang about a world axis"""
    return list(t3d.quaternions.qmult(t3d.quaternions.axangle2quat(axis, ang), np.asarray(q, dtype=float)))


def _hand_x(q):
    return t3d.quaternions.quat2mat(np.asarray(q, dtype=float))[:, 0]


def _is_tool(actor):
    return any(actor.get_name().startswith(t) for t in TOOLS)


class play_table(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        # a separate stream for scene and play choices, seeded from the episode seed
        self.prng = np.random.default_rng(kwags.get("seed", 0) * 7919 + 17)
        self.play_cfg = kwags.get("play", {})
        self.n_plans = 0
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ robot
    def load_robot(self, **kwags):
        """As Base_Task.load_robot, but the planners use the play configs whose
        joint limits are the expert data's range (env/make_play_planner_cfg.py)."""
        self.robot = Robot(self.scene, self.need_topp, **kwags)
        play = os.path.join(os.path.dirname(self.robot.left_curobo_yml_path), "play_curobo.yml")
        if os.path.exists(os.path.join(ROOT_PATH, play.replace("curobo.yml", "curobo_left.yml"))):
            self.robot.left_curobo_yml_path = self.robot.right_curobo_yml_path = play
            self.play_planner = True
        else:
            self.play_planner = False
        self.robot.set_planner(self.scene)
        self.robot.init_joints()
        for ent in (self.robot.left_entity, self.robot.right_entity):
            for link in ent.get_links():
                link.set_mass(1)

    # ------------------------------------------------------------------ scene
    def _free_xy(self, r, placed, xlim=(-0.24, 0.24), ylim=(-0.16, 0.08), tries=200):
        for _ in range(tries):
            xy = np.array([self.prng.uniform(*xlim), self.prng.uniform(*ylim)])
            if all(np.linalg.norm(xy - p) > r + q + 0.03 for p, q in placed):
                return xy
        return None

    def _spawn(self, name, mid, xy, yaw):
        qz = t3d.quaternions.axangle2quat([0, 0, 1], yaw)
        q = t3d.quaternions.qmult(qz, UPRIGHT)
        actor = create_actor(self, pose=sapien.Pose([xy[0], xy[1], TABLE_Z + 0.005], q), modelname=name,
                             model_id=mid, convex=True)
        return actor

    def load_actors(self):
        cfg = self.play_cfg
        self.objects, self.containers = [], []
        placed = []
        n_obj = int(self.prng.integers(cfg.get("min_objects", 3), cfg.get("max_objects", 5) + 1))
        names = list(self.prng.choice(GRASPABLE, size=n_obj, replace=False))
        if self.prng.random() < cfg.get("p_tool", 0.4) and not any(n in TOOLS for n in names):
            names[-1] = str(self.prng.choice(TOOLS))
        if self.prng.random() < cfg.get("p_container", 0.7):
            names = [str(self.prng.choice(CONTAINERS))] + names
        for name in names:
            is_cont = name in CONTAINERS
            ids = _model_ids(name)
            mid = int(self.prng.choice(ids)) if ids else 0
            r = _footprint_radius(name, mid)    # real size: overlapping spawns get flung off the table
            xy = self._free_xy(r, placed)
            if xy is None:
                continue
            actor = self._spawn(name, mid, xy, self.prng.uniform(-np.pi, np.pi))
            actor.set_mass((0.25 if any(name == n for n in OPEN_TOP) else 0.5) if is_cont else 0.05)
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

    def _fingers(self, arm):
        g = self.robot.left_gripper if arm == "left" else self.robot.right_gripper
        return [j.get_child_link() for j, _, _ in g]

    def _holds(self, arm, actor, n=2):
        """n fingers of this hand touch the object."""
        fingers = {L.get_name() for L in self._fingers(arm)}
        name, touch = actor.get_name(), set()
        for c in self.scene.get_contacts():
            n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
            if n0 == name and n1 in fingers:
                touch.add(n1)
            elif n1 == name and n0 in fingers:
                touch.add(n0)
        return len(touch) >= n

    def _touching(self, a, b):
        return self.check_actors_contact(a.get_name(), b.get_name())

    def _finger_xyz(self, arm):
        return np.mean([L.entity.get_pose().p for L in self._fingers(arm)], axis=0)

    def _ee(self, arm):
        return np.array(self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose(), dtype=float)

    def _gripper_open(self, arm):
        return self.is_left_gripper_open() if arm == "left" else self.is_right_gripper_open()

    def _arm_for(self, xy):
        return ArmTag("left" if xy[0] < 0 else "right")

    def _inside(self, obj, cont):
        lo, hi = self._aabb(cont)
        p = obj.get_pose().p
        return bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] > lo[2] - 0.01)

    def _on_table(self, actor):
        return TABLE_Z - 0.05 < actor.get_pose().p[2] < TABLE_Z + 0.3

    def _free_spot(self, actor, side=None, min_move=0.08, r=0.09):
        """A free table spot (on `side` if given)."""
        xs = {"left": (-0.25, -0.06), "right": (0.06, 0.25), None: (-0.25, 0.25)}[side]
        p0 = actor.get_pose().p[:2]
        others = [a for a in self.objects + self.containers if a is not actor]
        for _ in range(100):
            xy = np.array([self.prng.uniform(*xs), self.prng.uniform(-0.15, 0.08)])
            if np.linalg.norm(xy - p0) < min_move:
                continue
            if all(np.linalg.norm(xy - o.get_pose().p[:2]) > r for o in others):
                return xy
        return None

    # ------------------------------------------------------------- planning
    def _plan(self, arm, pose, constraint=None, q=None):
        f = self.robot.left_plan_path if arm == "left" else self.robot.right_plan_path
        self.n_plans += 1
        try:
            return f([float(x) for x in pose], constraint_pose=constraint, last_qpos=q)
        except Exception:
            if os.environ.get("PLAY_DEBUG"):
                import traceback
                traceback.print_exc()
            return None

    def _end_q(self, arm, res, q=None):
        ent = self.robot.left_entity if arm == "left" else self.robot.right_entity
        names = [j.get_name() for j in ent.get_active_joints()]
        arm_names = self.robot.left_arm_joints_name if arm == "left" else self.robot.right_arm_joints_name
        q = np.array(ent.get_qpos() if q is None else q, dtype=np.float32).copy()
        for k, n in enumerate(arm_names):
            q[names.index(n)] = res["position"][-1][k]
        return q

    JOINT_MARGIN = 0.15   # rad beyond the range the expert data uses (same as the planning URDF's limits)

    def _joint_box(self, arm):
        """Per-joint range of RoboTwin's released expert data for this robot
        (robotwin_play/expert_joint_ranges_<robot>.json); None if unknown."""
        if not hasattr(self, "_jbox"):
            emb = str(self.play_cfg.get("embodiment_name", "aloha-agilex"))
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"expert_joint_ranges_{emb}.json")
            self._jbox = json.load(open(path)) if os.path.exists(path) else None
        if self._jbox is None:
            return None
        b = self._jbox[str(arm)]
        return np.array(b["lo"]) - self.JOINT_MARGIN, np.array(b["hi"]) + self.JOINT_MARGIN

    def _plan_chain(self, arm, legs, tries=2, q0=None, flips=(np.pi,)):
        """Plan the legs [(pose, constraint), ...] one after the other. Every
        leg must plan and stay within the expert joint range. If that fails,
        the same legs are tried with the hand turned about its own axis by
        each angle in `flips` (180 deg is the same grasp, the gripper being
        symmetric; +-90 deg is fine for a free put-down). Returns the list of
        plans, or None (self._why says why)."""
        plans = self._plan_chain1(arm, legs, tries, q0)
        for ang in (flips if plans is None else ()):
            flip = t3d.quaternions.axangle2quat([1, 0, 0], ang)
            flipped = [(list(np.asarray(p, dtype=float)[:3]) + list(t3d.quaternions.qmult(np.asarray(p, dtype=float)[3:], flip)), c)
                       for p, c in legs]
            plans = self._plan_chain1(arm, flipped, tries, q0)
            if plans is not None:
                break
        return plans

    @staticmethod
    def _yawed(legs, about, ang):
        """The same hand poses turned by ang about the vertical through `about`:
        the held object lands in the same place with another yaw."""
        Rz = t3d.quaternions.axangle2quat([0, 0, 1], ang)
        M = t3d.quaternions.quat2mat(Rz)
        out = []
        for p, c in legs:
            p = np.asarray(p, dtype=float)
            xyz = np.asarray(about, dtype=float) + M @ (p[:3] - np.asarray(about, dtype=float))
            out.append((list(xyz) + list(t3d.quaternions.qmult(Rz, p[3:])), c))
        return out

    def _plan_about(self, arm, legs, about, q0=None, angs=(0.0, np.pi, np.pi / 2, -np.pi / 2)):
        """Plan legs that carry an object to `about`, trying the object's yaw
        as given, then turned (the hand swings rigidly about the target)."""
        for ang in angs:
            plans = self._plan_chain(arm, legs if ang == 0.0 else self._yawed(legs, about, ang), q0=q0, flips=())
            if plans is not None:
                return plans
        return None

    def _q_after(self, arm, plans):
        q = None
        for p in plans:
            q = self._end_q(arm, p, q)
        return q

    def _plan_chain1(self, arm, legs, tries=2, q0=None):
        box = self._joint_box(arm)
        for _ in range(tries):
            q, plans, ok = q0, [], True
            for k, (pose, cons) in enumerate(legs):
                r = self._plan(arm, pose, cons, q)
                if r is None or r.get("status") != "Success":
                    self._why = f"leg {k}: {None if r is None else r.get('status')}"
                    ok = False
                    break
                if box is not None:
                    P = np.asarray(r["position"])[:, :len(box[0])]
                    if np.any(P < box[0]) or np.any(P > box[1]):
                        j = int(np.argmax(np.maximum(P - box[1], box[0] - P).max(0)))
                        self._why = f"leg {k}: joint {j} outside expert range ({P[:, j].min():.2f}..{P[:, j].max():.2f})"
                        ok = False
                        break
                plans.append(r)
                q = self._end_q(arm, r, q)
            if ok:
                return plans
            if os.environ.get("PLAY_DEBUG"):
                print(f"   INFEASIBLE {arm} {self._why} target {np.round(np.asarray(legs[k][0], dtype=float)[:3], 3)}", flush=True)
        return None

    def _home_plan(self, arm):
        """A plan taking this arm back to its rest pose, or None if it is there."""
        home = np.array(self.robot.left_original_pose if arm == "left" else self.robot.right_original_pose, dtype=float)
        if np.linalg.norm(self._ee(arm)[:3] - home[:3]) < 0.03:
            return None
        plans = self._plan_chain(arm, [(home, None)], tries=1)
        return plans[0] if plans else None

    # ------------------------------------------------------------ executing
    def _sync(self, arm, tol=0.05, rate=0.004):
        """If the arm has been pushed away from where it was told to be (it is
        pressing on something), bring the command smoothly to where it really
        is before the next plan starts from there - else the recorded command
        jumps."""
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

    def _set_paths(self, arm, plans):
        if arm == "left":
            self.left_joint_path, self.left_cnt = list(plans), 0
        else:
            self.right_joint_path, self.right_cnt = list(plans), 0

    def _run(self, steps):
        """Execute pre-planned steps through RoboTwin's move() in replay mode.
        steps: ("move", arm, plan) | ("grip", arm, value) |
               ("both", (arm, plan_or_None, grip_or_None), (arm, plan_or_None, grip_or_None))"""
        for st in steps:
            self.plan_success = True
            if st[0] == "move":
                _, arm, plan = st
                self._set_paths(arm, [plan])
                self.need_plan = False
                try:
                    self.move((arm, [Action(arm, "move", target_pose=[0.0] * 7)]))
                finally:
                    self.need_plan = True
            elif st[0] == "grip":
                _, arm, val = st
                self.move((arm, [Action(arm, "open" if val > 0.5 else "close", target_gripper_pos=float(val))]))
            else:
                acts = []
                for arm, plan, g in st[1:]:
                    if plan is not None:
                        self._set_paths(arm, [plan])
                        acts.append((arm, [Action(arm, "move", target_pose=[0.0] * 7)]))
                    elif g is not None:
                        acts.append((arm, [Action(arm, "open" if g > 0.5 else "close", target_gripper_pos=float(g))]))
                if not acts:
                    continue
                self.need_plan = False
                try:
                    self.move(acts[0], acts[1] if len(acts) > 1 else None)
                finally:
                    self.need_plan = True
            if not self.plan_success:
                return False
        return True

    def _run_touch(self, steps, a, b):
        """Run steps; True if a and b were in contact at the end or b was moved by it."""
        p0 = b.get_pose().p.copy()
        self._run(steps)
        return bool(self._touching(a, b) or np.linalg.norm(b.get_pose().p - p0) > 0.004)

    def _go(self, arm, legs, grip_after=None):
        """Plan a chain for one arm and run it (optionally a gripper value at the end)."""
        plans = self._plan_chain(arm, legs)
        if plans is None:
            return False
        self._sync(arm)
        steps = [("move", arm, p) for p in plans]
        if grip_after is not None:
            steps.append(("grip", arm, grip_after))
        return self._run(steps)

    def _hold(self, n=2):
        """Keep still for n gripper cycles (recorded, unlike RoboTwin's delay)."""
        self.delay(n, save_freq=self.save_freq)

    def _home(self, arm):
        if not self._gripper_open(arm):
            self._run([("grip", arm, 1.0)])
        p = self._home_plan(arm)
        if p is not None:
            self._sync(arm)
            self._run([("move", arm, p)])

    # --------------------------------------------------------------- grasps
    def _grasp_options(self, actor, arm, pre_dis=0.1, k=6, exclude=(), avoid=None, pref=None, below=None, high_first=False):
        """Grasp points ranked like RoboTwin's chooser (prefer top-down, then
        the arm's preferred side; or a given orientation), with some randomness
        so the same object is not always grabbed the same way."""
        pref_dir = self.robot.get_grasp_perfect_direction(arm)
        opts = []
        for i, cp in actor.iter_contact_points("list"):
            if i in exclude:
                continue
            if avoid is not None and np.linalg.norm(np.asarray(cp[:3]) - avoid) < 0.04:
                continue
            if below is not None and cp[2] > below:
                continue
            pre = self.get_grasp_pose(actor, arm, contact_point_id=i, pre_dis=pre_dis)
            if pre is None:
                continue
            pre = np.array(pre, dtype=np.float64)
            pose = pre.copy()
            pose[:3] += np.array([pre_dis, 0, 0]) @ np.linalg.inv(t3d.quaternions.quat2mat(pre[3:]))
            # the gripper is symmetric: the hand turned 180 deg about its own
            # axis is the same grasp, and often a much easier wrist pose
            flip = t3d.quaternions.axangle2quat([1, 0, 0], np.pi)
            for pre_, pose_ in ((pre, pose), (np.r_[pre[:3], t3d.quaternions.qmult(pre[3:], flip)],
                                             np.r_[pose[:3], t3d.quaternions.qmult(pose[3:], flip)])):
                if pref is not None:
                    score = cal_quat_dis(pose_[3:], pref)
                else:
                    td = cal_quat_dis(pose_[3:], TOPDOWN[str(arm)])
                    sd = cal_quat_dis(pose_[3:], GRASP_DIRECTION_DIC[pref_dir])
                    score = 0.7 * td + 0.3 * sd
                if high_first:
                    score -= 6.0 * float(cp[2])
                opts.append((score + self.prng.normal(0, 0.08), i, pre_, pose_))
        opts.sort(key=lambda x: x[0])
        return opts[:k]

    def _then_pose(self, grasp_pose, actor, then):
        """The hand pose (grasp orientation kept) that holds the object's origin
        at `then`, 5 cm up: used to check that a put-down there is reachable."""
        off = np.asarray(grasp_pose[:3]) - actor.get_pose().p
        return list(np.asarray(then, dtype=float) + off + [0, 0, 0.05]) + list(grasp_pose[3:])

    def _pick(self, actor, arm, lift=0.1, then=None, arms=None, exclude=(), avoid=None, pref=None, home_other=True,
              below=None, high_first=False):
        """Plan a whole pick (approach, grasp, lift, and if `then` is given a
        hand pose it must be able to reach while holding), then execute it.
        Tries this arm, then the other. Returns (held, arm), or (None, arm) if
        nothing could be planned (nothing moved)."""
        z0 = actor.get_pose().p[2]
        for arm_ in (arms or (arm, arm.opposite)):
            for _, cid, pre, pose in self._grasp_options(actor, arm_, exclude=exclude, avoid=avoid, pref=pref,
                                                          below=below, high_first=high_first):
                legs = [(pre, None), (pose, CONS)]
                if lift > 0:
                    up = pose.copy()
                    up[2] += lift
                    legs.append((up, None))
                plans = self._plan_chain(arm_, legs, flips=())   # (the options already include the flipped hand)
                if plans is None:
                    continue
                if then is not None and self._plan_about(arm_, [(self._then_pose(pose, actor, then), None)], then,
                                                         q0=self._q_after(arm_, plans)) is None:
                    continue
                home = self._home_plan(arm_.opposite) if home_other else None
                self._sync(arm_)
                steps = [("both", (arm_, plans[0], None), (arm_.opposite, home, None)) if home is not None else ("move", arm_, plans[0]),
                         ("move", arm_, plans[1]), ("grip", arm_, 0.0)]
                if lift > 0:
                    steps.append(("move", arm_, plans[2]))
                self._run(steps)
                self.last_grasp = dict(contact_point=int(cid), arm=str(arm_))
                held = self._holds(arm_, actor) and (lift <= 0 or actor.get_pose().p[2] - z0 > 0.5 * lift)
                if os.environ.get("PLAY_DEBUG"):
                    ee = self._ee(arm_)
                    print(f"   PICK {actor.get_name()} {arm_} cid={cid} holds={self._holds(arm_, actor)} rose={actor.get_pose().p[2] - z0:.3f} "
                          f"ee-vs-grasp={np.linalg.norm(ee[:3] - (pose[:3] if lift <= 0 else up[:3])):.3f} grip={self.robot.get_left_gripper_val() if arm_ == 'left' else self.robot.get_right_gripper_val():.2f}", flush=True)
                return bool(held), arm_
        return None, arm

    def _place(self, actor, arm, target, pre_dis, dis, up=0.07, release=True, **kw):
        """Put the held object down at target (7-pose or xyz) if that can be
        planned; returns True if it was executed."""
        target = list(target) + [1, 0, 0, 0] if len(target) == 3 else list(target)
        kw.setdefault("constrain", "free")
        pre = self.get_place_pose(actor, arm, target, pre_dis=pre_dis, **kw)
        pose = self.get_place_pose(actor, arm, target, pre_dis=dis, **kw)
        legs = [(pre, None), (pose, None)]
        if up:
            upp = list(pose)
            upp[2] += up
            legs.append((upp, None))
        plans = self._plan_about(arm, legs, target[:3], angs=(0.0, np.pi) if kw["constrain"] == "align" else (0.0, np.pi, np.pi / 2, -np.pi / 2))
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PLACE {actor.get_name()} {arm} target={np.round(target[:3], 3)} pre={np.round(pre[:3], 3)} pose={np.round(pose[:3], 3)} "
                  f"planned={plans is not None} {'' if plans else self._why}", flush=True)
        if plans is None:
            return False
        self._sync(arm)
        steps = [("move", arm, plans[0]), ("move", arm, plans[1])]
        if release:
            steps.append(("grip", arm, 1.0))
        if up:
            steps.append(("move", arm, plans[2]))
        ok = self._run(steps)
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PLACED {actor.get_name()} now at {np.round(actor.get_pose().p, 3)} (target {np.round(target[:3], 3)}) ran={ok}", flush=True)
        return ok

    def _put_down_here(self, actor, arm):
        """Couldn't plan the put-down: set it down right where the hand is."""
        p = actor.get_pose().p
        if not self._place(actor, arm, [p[0], p[1], TABLE_Z], 0.03, 0.005):
            self._run([("grip", arm, 1.0)])   # at least let go of it

    # ------------------------------------------------------------- results
    def _res(self, ok, lang, arm, **extra):
        d = dict(success=bool(ok), language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        d.update(extra)
        return d

    def _start(self, actor, lang, lift=0.1, then=None):
        """Common opening of a pick-and-do action: (True, arm) if the object
        is in the hand, (False, arm) if the grasp failed (arm sent home),
        (None, arm) if nothing could be planned."""
        self.last_grasp = None
        arm = self._arm_for(actor.get_pose().p)
        if then is not None:
            arm = self._arm_for(then)
        ok, arm = self._pick(actor, arm, lift=lift, then=then)
        if ok is None:
            return None, arm
        if not ok:
            self._home(arm)
            return False, arm
        return True, arm

    def _finish(self, actor, arm, placed, ok, lang, **extra):
        if not placed:
            self._put_down_here(actor, arm)
            self._home(arm)
        elif not self._gripper_open(arm):
            self._run([("grip", arm, 1.0)])
        return self._res(ok and placed, lang, arm, **extra)

    # ------------------------------------------------------------- actions
    def p_move(self, actor):
        lang = f"move the {self._name(actor)} to another spot"
        for _ in range(4):
            xy = self._free_spot(actor)
            if xy is None:
                return None
            st, arm = self._start(actor, lang, then=[xy[0], xy[1], TABLE_Z])
            if st is not None:
                break
        else:
            return None
        if not st:
            return self._res(False, lang, arm)
        placed = self._place(actor, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01)
        ok = placed and np.linalg.norm(actor.get_pose().p[:2] - xy) < 0.05
        return self._finish(actor, arm, placed, ok, lang)

    def p_put_in(self, actor, cont):
        lang = f"put the {self._name(actor)} in the {self._name(cont)}"
        fp = cont.get_functional_point(0)
        st, arm = self._start(actor, lang, lift=0.12, then=fp[:3])
        if st is None:
            # no arm can both pick it up and reach the container: pass it over
            if self._arm_for(actor.get_pose().p) != self._arm_for(fp):
                res = self.p_handover(actor, dest=fp)
                if res is not None:
                    res.update(language=f"pass the {self._name(actor)} over and put it in the {self._name(cont)}",
                               success=bool(res["success"] and self._inside(actor, cont)), via="handover")
                return res
            return None
        if not st:
            return self._res(False, lang, arm)
        placed = self._place(actor, arm, fp, 0.08, 0.03)
        return self._finish(actor, arm, placed, self._inside(actor, cont), lang)

    def p_stack(self, actor, base):
        lang = f"put the {self._name(actor)} on the {self._name(base)}"
        lo, hi = self._aabb(base)
        alo, ahi = self._aabb(actor)
        if not any(base.get_name().startswith(n) for n in STACK_BASES) or np.prod((hi - lo)[:2]) < 0.6 * np.prod((ahi - alo)[:2]):
            return None   # nothing stands on a bottle or on something much smaller
        bp = base.get_pose().p
        st, arm = self._start(actor, lang, lift=0.12, then=[bp[0], bp[1], hi[2]])
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        placed = self._place(actor, arm, [bp[0], bp[1], hi[2] + 0.005], 0.06, 0.015)
        p = actor.get_pose().p
        alo2, _ = self._aabb(actor)
        lo2, hi2 = self._aabb(base)
        # resting on the base: above its bottom and within its footprint (a plate's inside is below its rim)
        ok = alo2[2] > lo2[2] + 0.004 and np.all(p[:2] > lo2[:2] - 0.01) and np.all(p[:2] < hi2[:2] + 0.01)
        return self._finish(actor, arm, placed, ok, lang)

    def p_line_up(self, actor, anchor):
        """Put it next to the anchor, lined up with it."""
        lang = f"put the {self._name(actor)} next to the {self._name(anchor)} in a row"
        ap = anchor.get_pose().p
        R = t3d.quaternions.quat2mat(anchor.get_pose().q)
        lo, hi = self._aabb(anchor)
        alo, ahi = self._aabb(actor)
        others = [a for a in self.objects + self.containers if a not in (actor, anchor)]
        cands = []
        for u in (R[:, 0], -R[:, 0], R[:, 2], -R[:, 2]):
            u = np.array([u[0], u[1]])
            if np.linalg.norm(u) < 0.5:
                continue
            u /= np.linalg.norm(u)
            d = 0.5 * float(np.abs((hi - lo)[:2]) @ np.abs(u)) + 0.5 * float(np.abs((ahi - alo)[:2]) @ np.abs(u)) + 0.025
            xy = ap[:2] + u * d
            if not (-0.28 < xy[0] < 0.28 and -0.2 < xy[1] < 0.1):
                continue
            if any(np.linalg.norm(xy - o.get_pose().p[:2]) < 0.08 for o in others):
                continue
            cands.append(xy)
        if not cands:
            return None
        xy = cands[int(self.prng.integers(len(cands)))]
        st, arm = self._start(actor, lang, then=[xy[0], xy[1], TABLE_Z])
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        ax = R[:, 0] if abs(R[0, 0]) + abs(R[1, 0]) > 0.5 else R[:, 2]
        placed = self._place(actor, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01, constrain="align",
                             align_axis=[float(ax[0]), float(ax[1]), 0.0])
        ok = placed and np.linalg.norm(actor.get_pose().p[:2] - xy) < 0.04
        return self._finish(actor, arm, placed, ok, lang)

    def p_handover(self, actor, dest=None):
        """Pick it up with one hand, hold it up in the middle, let the other
        hand take it from the side, and put it down on the other side."""
        lo, hi = self._aabb(actor)
        cz = [cp[2] for _, cp in actor.iter_contact_points("list")]
        if hi[2] - lo[2] < 0.09 or not cz or max(cz) - min(cz) < 0.045:
            return None    # the second hand needs a grasp point well below the first hand's
        a1 = self._arm_for(actor.get_pose().p)
        a2 = a1.opposite
        what = self._name(actor)
        lang = f"pass the {what} from the {a1} hand to the {a2} hand"
        self.last_grasp = None
        ok, a1 = self._pick(actor, a1, lift=0.1, arms=(a1,), high_first=True)
        if ok is None:
            return None
        if not ok:
            self._home(a1)
            return self._res(False, lang, a1)
        g1 = self.last_grasp
        # to the middle, still upright, the hand at a comfortable height
        ee = self._ee(a1)
        went = False
        for hz in (0.95, 0.99, 0.91):
            hand_mid = [0.0, float(self.prng.uniform(-0.08, -0.02)), hz] + list(ee[3:])
            if self._go(a1, [(hand_mid, None)]):
                went = True
                break
        mid = actor.get_pose().p.copy()
        if not went:
            self._put_down_here(actor, a1)
            self._home(a1)
            return self._res(False, lang, a1, grasp=g1)
        held1 = self._holds(a1, actor)
        if os.environ.get("PLAY_DEBUG"):
            cps = sorted(round(float(cp[2]), 3) for _, cp in actor.iter_contact_points("list"))
            print(f"   HANDOVER mid reached, {a1} holds={held1}, object at {np.round(actor.get_pose().p, 3)} fingers {np.round(self._finger_xyz(a1), 3)} contact z {cps}", flush=True)
        # the other hand comes in from its side, horizontally
        pref = GRASP_DIRECTION_DIC["right" if a2 == "left" else "left"]
        ok2, _ = self._pick(actor, a2, lift=0.0, arms=(a2,), exclude=(g1["contact_point"],), avoid=self._finger_xyz(a1),
                            pref=pref, home_other=False, below=float(self._finger_xyz(a1)[2]) - 0.04)
        g2 = self.last_grasp
        if os.environ.get("PLAY_DEBUG"):
            print(f"   HANDOVER second grasp ok2={ok2} holds2={self._holds(a2, actor)} g2={g2} why={getattr(self, '_why', '')}", flush=True)
        if not ok2 or not self._holds(a2, actor):
            # the other hand didn't get it: let go again, set it down in the middle
            if ok2 is not None:
                self._run([("grip", a2, 1.0)])
                ee = self._ee(a2)
                self._go(a2, [(list(ee[:3] - 0.08 * _hand_x(ee[3:])) + list(ee[3:]), None)])
                self._home(a2)
            if held1:
                if not self._place(actor, a1, [mid[0], mid[1], TABLE_Z], 0.05, 0.01):
                    self._put_down_here(actor, a1)
            self._home(a1)
            return self._res(False, lang, f"{a1}->{a2}", grasp=g1, grasp2=g2)
        self._run([("grip", a1, 1.0)])
        ee = self._ee(a1)
        self._go(a1, [(list(ee[:3] - 0.08 * _hand_x(ee[3:])) + list(ee[3:]), None)])
        # the receiving hand brings it over to its side while the other goes home
        ee2 = self._ee(a2)
        over = list(ee2[:3] + [(-0.12 if a2 == "left" else 0.12), 0.0, 0.03]) + list(ee2[3:])
        p2 = self._plan_chain(a2, [(over, None)])
        hp = self._home_plan(a1)
        self._sync(a1)
        self._sync(a2)
        self._run([("both", (a2, p2[0] if p2 else None, None), (a1, hp, None))])
        target = dest
        if target is None:
            xy = self._free_spot(actor, side=str(a2))
            target = None if xy is None else [xy[0], xy[1], TABLE_Z]
        placed = False
        if target is not None:
            placed = self._place(actor, a2, list(target), 0.06, 0.02)
        res = self._finish(actor, a2, placed, placed and self._on_table(actor), lang, grasp2=g2)
        res.update(arm=f"{a1}->{a2}", grasp=g1)
        return res

    def p_push(self, actor):
        """Push it a little along the table with the closed hand."""
        p = actor.get_pose().p
        lo, hi = self._aabb(actor)
        h = hi[2] - lo[2]
        r = 0.5 * np.linalg.norm((hi - lo)[:2])
        arm = self._arm_for(p)
        q = TOPDOWN[str(arm)]
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
            z = lo[2] + np.clip(0.5 * h, 0.015, 0.05) + TIP   # fingertips at that height on the object
            start = p[:2] - d * (r + 0.04)
            above = [start[0], start[1], z + 0.08] + list(q)
            down = [start[0], start[1], z] + list(q)
            through = [end[0] - d[0] * r, end[1] - d[1] * r, z] + list(q)
            plans = self._plan_chain(arm, [(above, None), (down, CONS), (through, CONS), (above, None)])
            if plans is None:
                continue
            words = ["right", "away", "left", "closer"][int(((ang + np.pi / 4) % (2 * np.pi)) // (np.pi / 2))]
            lang = f"push the {self._name(actor)} {words}"
            home = self._home_plan(arm.opposite)
            self._sync(arm)
            self._run([("both", (arm, None, 0.0), (arm.opposite, home, None)) if home is not None else ("grip", arm, 0.0),
                       ("move", arm, plans[0]), ("move", arm, plans[1]), ("move", arm, plans[2]), ("move", arm, plans[3]),
                       ("grip", arm, 1.0)])
            moved = float(np.linalg.norm(actor.get_pose().p[:2] - p[:2]))
            self.last_grasp = None
            return self._res(moved > 0.5 * dist, lang, arm, moved=moved)
        return None

    def p_reorient(self, actor):
        """Pick it up, turn it, put it back down about where it was."""
        lang = f"turn the {self._name(actor)} around"
        p0 = actor.get_pose().p.copy()
        q0 = actor.get_pose().q.copy()
        st, arm = self._start(actor, lang, lift=0.08)
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        ee = self._ee(arm)
        turned = False
        for ang in self.prng.permutation([0.8, -0.8, 1.3, -1.3, 1.8, -1.8]):
            if self._go(arm, [(list(ee[:3]) + _rot(ee[3:], [0, 0, 1], float(ang)), None)]):
                turned = True
                break
        placed = self._place(actor, arm, [p0[0], p0[1], TABLE_Z], 0.04, 0.01)
        dq = t3d.quaternions.qmult(t3d.quaternions.qinverse(q0), actor.get_pose().q)
        ok = turned and abs(dq[0]) < 0.96 and placed
        return self._finish(actor, arm, placed, ok, lang)

    def p_press(self, actor):
        """Press down on top of it with the closed hand (a click)."""
        lo, hi = self._aabb(actor)
        c = 0.5 * (lo + hi)
        arm = self._arm_for(c)
        q = TOPDOWN[str(arm)]
        top = hi[2]
        above = [c[0], c[1], top + TIP + 0.04] + list(q)
        down = [c[0], c[1], top + TIP - 0.012] + list(q)
        plans = self._plan_chain(arm, [(above, None), (down, CONS), (above, CONS)])
        if plans is None:
            return None
        lang = f"press down on the {self._name(actor)}"
        home = self._home_plan(arm.opposite)
        self._sync(arm)
        self._run([("both", (arm, None, 0.0), (arm.opposite, home, None)) if home is not None else ("grip", arm, 0.0),
                   ("move", arm, plans[0]), ("move", arm, plans[1])])
        touched = self._holds(arm, actor, n=1)
        self._run([("move", arm, plans[2]), ("grip", arm, 1.0)])
        self.last_grasp = None
        return self._res(touched, lang, arm)

    def p_shake(self, actor):
        lang = f"shake the {self._name(actor)}"
        st, arm = self._start(actor, lang, lift=0.12)
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        ee = self._ee(arm)
        a = float(self.prng.uniform(0.35, 0.55))
        legs = []
        for _ in range(3):
            legs.append((list(ee[:3] + [0, 0, 0.03]) + _rot(ee[3:], [1, 0, 0], a), None))
            legs.append((list(ee[:3] - [0, 0, 0.03]) + _rot(ee[3:], [1, 0, 0], -a), None))
        legs.append((list(ee), None))
        shaken = self._go(arm, legs)
        xy = actor.get_pose().p[:2]
        placed = self._place(actor, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01)
        return self._finish(actor, arm, placed, shaken, lang)

    def p_knock_with(self, tool, target):
        """Pick one thing up and tap another with it (a hammer on a block, a
        stamp on a card)."""
        lang = f"tap the {self._name(target)} with the {self._name(tool)}"
        tlo, thi = self._aabb(target)
        tp = target.get_pose().p
        bottom = float(self._aabb(tool)[0][2] - tool.get_pose().p[2])   # tool origin vs its lowest point
        over = [tp[0], tp[1], thi[2] + 0.03 - bottom]
        st, arm = self._start(tool, lang, lift=0.1, then=over)
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        ee = self._ee(arm)
        off = ee[:3] - tool.get_pose().p
        hover = list(np.asarray(over) + off) + list(ee[3:])
        tap = list(np.asarray(over) + off - [0, 0, 0.045]) + list(ee[3:])
        plans = self._plan_chain(arm, [(hover, None), (tap, CONS), (hover, CONS)])
        hit = False
        if plans is not None:
            self._sync(arm)
            self._run([("move", arm, plans[0])])
            hit = self._run_touch([("move", arm, plans[1])], tool, target)
            if os.environ.get("PLAY_DEBUG"):
                print(f"   KNOCK tool bottom z {self._aabb(tool)[0][2]:.3f} target top {thi[2]:.3f} tool xy {np.round(tool.get_pose().p[:2], 3)} target xy {np.round(tp[:2], 3)} hit={hit} holds={self._holds(arm, tool)}", flush=True)
            self._run([("move", arm, plans[2])])
        elif os.environ.get("PLAY_DEBUG"):
            print(f"   KNOCK tap chain infeasible: {self._why}", flush=True)
        xy = self._free_spot(tool, side=str(arm))
        placed = xy is not None and self._place(tool, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01)
        res = self._finish(tool, arm, placed, True, lang)
        res["success"] = bool(hit)   # the tap is the action; where the tool ends up is not
        return res

    def p_dual_pick(self, a, b):
        """Pick two things up at the same time, one in each hand, and set them
        down somewhere else at the same time."""
        if a.get_pose().p[0] > 0:
            a, b = b, a
        if a.get_pose().p[0] > 0 or b.get_pose().p[0] < 0:
            return None
        lang = f"pick up the {self._name(a)} and the {self._name(b)} at the same time"
        L, R_ = ArmTag("left"), ArmTag("right")
        spots, chains = {}, {}
        for arm, obj in ((L, a), (R_, b)):
            xy = self._free_spot(obj, side=str(arm))
            if xy is None:
                self._why = f"no free spot on the {arm}"
                return None
            spots[str(arm)] = xy
            opts = self._grasp_options(obj, arm)
            if os.environ.get("PLAY_DEBUG"):
                print(f"   DUAL {arm} {obj.get_name()} options {len(opts)} spot {np.round(xy, 3)}", flush=True)
            for _, cid, pre, pose in opts:
                up = pose.copy()
                up[2] += 0.1
                plans = self._plan_chain(arm, [(pre, None), (pose, CONS), (up, None)], flips=())
                if plans is not None and self._plan_about(arm, [(self._then_pose(pose, obj, [xy[0], xy[1], TABLE_Z]), None)],
                                                          [xy[0], xy[1], TABLE_Z], q0=self._q_after(arm, plans)) is None:
                    plans = None
                if plans is not None:
                    chains[str(arm)] = (cid, plans)
                    break
            if str(arm) not in chains:
                self._why = f"{arm}: no plannable grasp+carry ({getattr(self, '_why', '')})"
                return None
        self.last_grasp = dict(contact_point=[chains["left"][0], chains["right"][0]], arm="both")
        self._sync(L)
        self._sync(R_)
        pl, pr = chains["left"][1], chains["right"][1]
        self._run([("both", (L, pl[0], None), (R_, pr[0], None)), ("both", (L, pl[1], None), (R_, pr[1], None)),
                   ("both", (L, None, 0.0), (R_, None, 0.0)), ("both", (L, pl[2], None), (R_, pr[2], None))])
        held = {"left": self._holds(L, a), "right": self._holds(R_, b)}
        # put both down at the same time
        tl = spots["left"].tolist() + [TABLE_Z, 1, 0, 0, 0]
        tr = spots["right"].tolist() + [TABLE_Z, 1, 0, 0, 0]
        pre_l, pos_l = self.get_place_pose(a, L, tl, pre_dis=0.05), self.get_place_pose(a, L, tl, pre_dis=0.01)
        pre_r, pos_r = self.get_place_pose(b, R_, tr, pre_dis=0.05), self.get_place_pose(b, R_, tr, pre_dis=0.01)
        cl = self._plan_chain(L, [(pre_l, None), (pos_l, None), (list(np.r_[pos_l[:2], pos_l[2] + 0.07, pos_l[3:]]), None)])
        cr = self._plan_chain(R_, [(pre_r, None), (pos_r, None), (list(np.r_[pos_r[:2], pos_r[2] + 0.07, pos_r[3:]]), None)])
        placed = cl is not None and cr is not None
        if placed:
            self._sync(L)
            self._sync(R_)
            self._run([("both", (L, cl[0], None), (R_, cr[0], None)), ("both", (L, cl[1], None), (R_, cr[1], None)),
                       ("both", (L, None, 1.0), (R_, None, 1.0)), ("both", (L, cl[2], None), (R_, cr[2], None))])
        else:
            for arm, obj in ((L, a), (R_, b)):
                self._put_down_here(obj, arm)
                self._home(arm)
        ok = placed and all(held.values()) and np.linalg.norm(a.get_pose().p[:2] - spots["left"]) < 0.05 \
            and np.linalg.norm(b.get_pose().p[:2] - spots["right"]) < 0.05
        if os.environ.get("PLAY_DEBUG"):
            print(f"   DUAL held={held} placed={placed} a at {np.round(a.get_pose().p[:2], 3)} spot {np.round(spots['left'], 3)} "
                  f"b at {np.round(b.get_pose().p[:2], 3)} spot {np.round(spots['right'], 3)}", flush=True)
        return self._res(ok, lang, "both", held=held)

    def p_lift_together(self, cont):
        """Lift a container with both hands, hold it up, set it down again."""
        lang = f"lift the {self._name(cont)} with both hands"
        c = cont.get_pose().p
        L, R_ = ArmTag("left"), ArmTag("right")
        pts = list(cont.iter_contact_points("list"))
        if len(pts) < 2:
            return None
        chains = {}
        home = {"left": np.array(self.robot.left_original_pose[:3]), "right": np.array(self.robot.right_original_pose[:3])}
        # the two grasp points farthest apart; each hand takes the one nearer to it
        far = max(((i, j) for i, _ in pts for j, _ in pts if i < j),
                  key=lambda ij: np.linalg.norm(np.asarray(dict(pts)[ij[0]][:3]) - np.asarray(dict(pts)[ij[1]][:3])))
        if np.linalg.norm(np.asarray(dict(pts)[far[0]][:3]) - np.asarray(dict(pts)[far[1]][:3])) < 0.08:
            return None
        order = sorted(far, key=lambda i: np.linalg.norm(np.asarray(dict(pts)[i][:3]) - home["left"]) - np.linalg.norm(np.asarray(dict(pts)[i][:3]) - home["right"]))
        for arm, first in ((L, order[0]), (R_, order[1])):
            side = [(first, dict(pts)[first])] + [(i, cp) for i, cp in pts if i not in far and
                                                   np.linalg.norm(np.asarray(cp[:3]) - np.asarray(dict(pts)[first][:3])) < 0.05]
            for i, cp in side[:3]:
                pre = self.get_grasp_pose(cont, arm, contact_point_id=i, pre_dis=0.04)
                if pre is None:
                    continue
                pre = np.array(pre, dtype=float)
                pose = pre.copy()
                pose[:3] += np.array([0.04, 0, 0]) @ np.linalg.inv(t3d.quaternions.quat2mat(pre[3:]))
                lift = min(0.08, 1.08 - pose[2])    # the hands can't go much above 1.1 m
                if lift < 0.03:
                    continue
                up = pose.copy()
                up[2] += lift
                plans = self._plan_chain(arm, [(pre, None), (pose, CONS), (up, CONS), (pose, CONS)], flips=())
                if os.environ.get("PLAY_DEBUG"):
                    print(f"   LIFT2 {arm} contact {i} at {np.round(cp[:3], 3)} pre {np.round(pre[:3], 3)} approach {np.round(_hand_x(pre[3:]), 2)} planned={plans is not None} {'' if plans else self._why}", flush=True)
                if plans is not None:
                    chains[str(arm)] = (i, plans)
                    break
            if str(arm) not in chains:
                self._why = f"{arm}: no plannable two-hand grasp"
                return None
        self.last_grasp = dict(contact_point=[chains["left"][0], chains["right"][0]], arm="both")
        pl, pr = chains["left"][1], chains["right"][1]
        z0 = c[2]
        self._sync(L)
        self._sync(R_)
        self._run([("both", (L, pl[0], None), (R_, pr[0], None)), ("both", (L, pl[1], None), (R_, pr[1], None)),
                   ("both", (L, None, 0.0), (R_, None, 0.0)), ("both", (L, pl[2], None), (R_, pr[2], None))])
        lifted = cont.get_pose().p[2] - z0
        both = self._holds(L, cont, n=1) and self._holds(R_, cont, n=1)
        self._hold(2)
        self._run([("both", (L, pl[3], None), (R_, pr[3], None)), ("both", (L, None, 1.0), (R_, None, 1.0))])
        hl, hr = self._home_plan(L), self._home_plan(R_)
        self._sync(L)
        self._sync(R_)
        self._run([("both", (L, hl, None), (R_, hr, None))])
        return self._res(lifted > 0.04 and both, lang, "both", lifted=float(lifted))

    def p_pour(self, cont):
        """Pick up a container with something in it and tip it out."""
        inside = [o for o in self.objects if self._inside(o, cont)]
        if not inside:
            return None
        lang = f"tip the {self._name(inside[0])} out of the {self._name(cont)}"
        self.last_grasp = None
        arm = self._arm_for(cont.get_pose().p)
        ok, arm = self._pick(cont, arm, lift=0.14, high_first=True)
        st = None if ok is None else bool(ok)
        if st is False:
            self._home(arm)
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        ee = self._ee(arm)
        tilted = False
        for ang in (1.4, 1.1, 0.8):
            for ax in ([1, 0, 0], [0, 1, 0]):
                for sgn in (1, -1):
                    legs = [(list(ee[:3] + [0, 0, 0.04]) + _rot(ee[3:], ax, sgn * ang), None), (list(ee), None)]
                    if self._plan_chain(arm, legs, flips=()) is not None:
                        self._sync(arm)
                        tilted = self._go(arm, legs[:1])
                        self.pour_angle = float(sgn * ang)
                        if os.environ.get("PLAY_DEBUG"):
                            up = t3d.quaternions.quat2mat(cont.get_pose().q)[:, 1]   # model y = the container's up
                            print(f"   POUR angle {sgn * ang:.2f} about {ax}: container up axis now {np.round(up, 2)}, held={self._holds(arm, cont)}", flush=True)
                        self._hold(2)
                        self._go(arm, legs[1:])
                        break
                if tilted:
                    break
            if tilted:
                break
        if os.environ.get("PLAY_DEBUG"):
            print(f"   POUR tilted={tilted} why={getattr(self, '_why', '')} still inside={[o.get_name() for o in inside if self._inside(o, cont)]} cont at {np.round(cont.get_pose().p, 3)}", flush=True)
        xy = cont.get_pose().p[:2]
        placed = self._place(cont, arm, [xy[0], xy[1], TABLE_Z], 0.06, 0.015)
        out = not any(self._inside(o, cont) for o in inside)
        return self._finish(cont, arm, placed, tilted and out, lang)

    # ---------------------------------------------------------------- loop
    def _choose(self):
        objs = [a for a in self.objects if self._on_table(a)]
        conts = [c for c in self.containers if self._on_table(c)]
        done = getattr(self, "_done", {})
        touched = getattr(self, "_touched", {})
        last = getattr(self, "_last_kind", None)
        opts = []
        nov = lambda a: 1.0 / (1 + touched.get(a.get_name(), 0))
        for a in objs:
            w = nov(a)
            opts.append((0.8 * w, "move", (a,)))
            opts.append((0.45 * w, "push", (a,)))
            opts.append((0.4 * w, "reorient", (a,)))
            opts.append((0.4 * w, "handover", (a,)))
            opts.append((0.35 * w, "press", (a,)))
            opts.append((0.3 * w, "shake", (a,)))
            for c in conts:
                opts.append((1.0 * w / len(conts), "put_in", (a, c)))
            for b in objs:
                if b is not a:
                    opts.append((0.35 * w / max(len(objs) - 1, 1), "stack", (a, b)))
                    opts.append((0.3 * w / max(len(objs) - 1, 1), "line_up", (a, b)))
                    if _is_tool(a):
                        opts.append((0.9 * w / max(len(objs) - 1, 1), "knock_with", (a, b)))
            for c in conts:
                if any(c.get_name().startswith(n) for n in STACK_BASES):
                    opts.append((0.25 * w / len(conts), "stack", (a, c)))
                    if a.get_pose().p[0] < 0 < b.get_pose().p[0]:
                        opts.append((0.5 / max(len(objs) - 1, 1), "dual_pick", (a, b)))
        for c in conts:
            opts.append((0.35, "lift_together", (c,)))
            if any(c.get_name().startswith(n) for n in OPEN_TOP) and any(self._inside(o, c) for o in objs):
                opts.append((1.0, "pour", (c,)))
        # variety: what has been done already is less likely, the same kind twice in a row even less
        opts = [(w / (1 + done.get(k, 0)) * (0.3 if k == last else 1.0), k, x) for w, k, x in opts]
        w = np.array([o[0] for o in opts])
        return opts[int(self.prng.choice(len(opts), p=w / w.sum()))] if opts else None

    def play_once(self):
        cfg = self.play_cfg
        segs = []
        self._done, self._touched = {}, {}
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
            self.plan_success = True
            res = getattr(self, "p_" + kind)(*args)
            self.plan_success = True
            self._last_kind = kind
            if res is None:
                continue
            self._done[kind] = self._done.get(kind, 0) + 1
            for a in args:
                self._touched[a.get_name()] = self._touched.get(a.get_name(), 0) + 1
            res.update(primitive=kind, target=[a.get_name() for a in args], start=f0, end=self.FRAME_IDX)
            segs.append(res)
        self.info["info"] = {"scene": self.scene_desc, "segments": segs, "plans": self.n_plans,
                             "syncs": getattr(self, "n_syncs", 0)}
        return self.info

    def check_success(self):
        return True
