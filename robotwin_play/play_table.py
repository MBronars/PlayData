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
import subprocess

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC, ROOT_PATH
from .robot import Robot
from .utils import *
from .utils.transforms import cal_quat_dis

import importlib.util as _ilu
_errors = _ilu.spec_from_file_location("rt_errors", os.path.join(os.path.dirname(os.path.realpath(__file__)), "errors.py"))
errors = _ilu.module_from_spec(_errors)
_errors.loader.exec_module(errors)

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
# Articulated fixtures, placed the way RoboTwin's own tasks place them
# (open_microwave, put_object_cabinet, open_laptop, turn_switch). ids index
# the model folders; cps are the handle contact points the hand grasps.
FIXTURES = {
    "044_microwave": dict(ids=[0, 1], x=(-0.12, -0.02), y=(0.15, 0.2), z=(0.8, 0.8), q=[0.707, 0, 0, 0.707],
                          yaw=0.0, cps=[0, 1], props=(0.0, 0.0), word="microwave door"),
    # y: RoboTwin uses 0.155; a little closer so the open drawer is within reach. Some joint friction:
    # it stays where it is left
    "036_cabinet": dict(ids=[46653], x=(-0.05, 0.05), y=(0.09, 0.13), z=(0.741, 0.741), q=[1, 0, 0, 1],
                        yaw=0.0, cps=[0, 1], props=(1.0, 0.0, 0.3), word="drawer"),
    "015_laptop": dict(ids=list(range(11)), x=(-0.05, 0.05), y=(-0.1, 0.05), z=(0.741, 0.741), q=[0.7, 0, 0, 0.7],
                       yaw=np.pi / 3, cps=[0, 1, 2], props=(1.0, 0.0), word="laptop"),
    "056_switch": dict(ids=list(range(8)), x=(-0.25, 0.25), y=(0.0, 0.1), z=(0.81, 0.84), q=[0.704141, 0, 0, 0.71006],
                       yaw=np.pi / 4, cps=[0], props=None, word="switch"),
}
SLIDE = ["058_markpen", "093_brush-pen", "079_remotecontrol", "118_tooth-paste"]   # uniform along their length
SHAKEABLE = ["001_bottle", "114_bottle", "049_shampoo", "101_milk-tea", "105_sauce-can", "109_hydrating-oil",
             "115_perfume", "080_pillbottle", "071_can", "068_boxdrink"]   # things one shakes (RoboTwin: shake_bottle)
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


N_OBSTACLES = 20        # cuRobo obstacle cache: the table + up to 15 scene objects


def _patch_planner_cache():
    """RoboTwin builds its cuRobo planners with a world holding only the table.
    Give them an obstacle cache so the scene objects can be loaded as cuboids
    before each plan (see _world)."""
    from .robot import planner as pm
    cfg = pm.MotionGenConfig
    if getattr(cfg, "_play_patched", False):
        return
    orig = cfg.load_from_robot_config
    def load(*a, **k):
        k.setdefault("collision_cache", {"obb": N_OBSTACLES})
        return orig(*a, **k)
    cfg.load_from_robot_config = staticmethod(load)
    cfg._play_patched = True


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
        self._plan_seed = int(kwags.get("seed", 0)) * 7919 + 29
        self.play_cfg = kwags.get("play", {})
        self.n_plans = 0
        self.err = None
        self.action_steps, self.action_len = 0, 2500
        super()._init_task_env_(**kwags)
        self._viz = None
        self._viz_cam = self._add_viz_camera() if self.save_data and kwags.get("play", {}).get("viz_video", True) else None

    # -------------------------------------------------------------- showcase
    VIZ_W, VIZ_H = 640, 480

    def _add_viz_camera(self):
        """The agent's view (RoboTwin's head camera pose) with a wide lens, so
        both arms and the whole tabletop stay in frame. For showcase videos
        only: not part of the recorded observations (the head camera itself
        is a 37 deg D435 that crops the arms)."""
        names = self.cameras.static_camera_name
        head = self.cameras.static_camera_list[names.index("head_camera")] if "head_camera" in names else None
        cam = self.scene.add_camera(name="play_viz_camera", width=self.VIZ_W, height=self.VIZ_H,
                                    fovy=np.deg2rad(75), near=0.05, far=10)
        if head is not None:
            cam.entity.set_pose(head.entity.get_pose())
        else:
            pos, look = np.array([-0.032, -0.45, 1.35]), np.array([0.0, 0.0, 0.75])
            f = (look - pos) / np.linalg.norm(look - pos)
            l = np.cross([0, 0, 1], f); l /= np.linalg.norm(l)
            m = np.eye(4); m[:3, :3] = np.stack([f, l, np.cross(f, l)], 1); m[:3, 3] = pos
            cam.entity.set_pose(sapien.Pose(m))
        return cam

    def _take_picture(self):
        super()._take_picture()
        if not self.save_data or self._viz_cam is None:
            return
        if self._viz is None:
            d = os.path.join(self.save_dir, "viz")
            os.makedirs(d, exist_ok=True)
            self._viz = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.VIZ_W}x{self.VIZ_H}", "-r", f"{250 / 15:.4f}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "medium", "-movflags", "+faststart",
                 os.path.join(d, f"episode{self.ep_num}.mp4")], stdin=subprocess.PIPE)
        self._viz_cam.take_picture()
        im = (self._viz_cam.get_picture("Color")[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        self._viz.stdin.write(np.ascontiguousarray(im).tobytes())

    def close_viz(self):
        if getattr(self, "_viz", None) is not None:
            self._viz.stdin.close()
            self._viz.wait()
            self._viz = None

    # ------------------------------------------------------------------ robot
    def load_robot(self, **kwags):
        """As Base_Task.load_robot, but the planners use the play configs whose
        joint limits are the expert data's range (env/make_play_planner_cfg.py)."""
        self.robot = Robot(self.scene, self.need_topp, **kwags)
        _patch_planner_cache()
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
            if all(np.linalg.norm(xy - p) > r + q + 0.06 for p, q in placed):
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
        self.objects, self.containers, self.fixtures = [], [], []
        placed = []
        fx = None
        if self.prng.random() < cfg.get("p_fixture", 0.45):
            fx = self._spawn_fixture(str(self.prng.choice(cfg.get("fixtures", list(FIXTURES)))))
            if fx is not None:
                lo, hi = self._aabb(fx)
                placed.append((0.5 * (lo + hi)[:2], 0.5 * float(np.linalg.norm((hi - lo)[:2]))))
        n_obj = int(self.prng.integers(cfg.get("min_objects", 3), cfg.get("max_objects", 5) + 1))   # plenty to do besides the fixture
        names = list(self.prng.choice(GRASPABLE, size=n_obj, replace=False))
        if cfg.get("objects"):          # for tests: these objects (plus random ones up to n_obj)
            forced = list(cfg["objects"])
            names = forced + [n for n in names if n not in forced][:max(0, n_obj - len(forced))]
        if self.prng.random() < cfg.get("p_tool", 0.4) and not any(n in TOOLS for n in names):
            names[-1] = str(self.prng.choice(TOOLS))
        if self.prng.random() < cfg.get("p_container", 0.7):
            names = [str(self.prng.choice(CONTAINERS))] + names
        for name in names:
            is_cont = name in CONTAINERS
            ids = _model_ids(name)
            mid = int(self.prng.choice(ids)) if ids else 0
            r = _footprint_radius(name, mid)    # real size: overlapping spawns get flung off the table
            # containers often in the middle, where both hands reach them (lift_pot)
            xy = self._free_xy(r, placed, xlim=(-0.08, 0.08)) if is_cont and self.prng.random() < 0.5 else None
            xy = self._free_xy(r, placed) if xy is None else xy
            if xy is None:
                continue
            actor = self._spawn(name, mid, xy, self.prng.uniform(-np.pi, np.pi))
            actor.set_mass((0.25 if any(name == n for n in OPEN_TOP) else 0.5) if is_cont else 0.05)
            placed.append((xy, r))
            (self.containers if is_cont else self.objects).append(actor)
            self.add_prohibit_area(actor, padding=0.02)
        self.scene_desc = ([dict(name=f.get_name(), fixture=True) for f in self.fixtures] +
                           [dict(name=a.get_name(), container=a in self.containers) for a in self.containers + self.objects])

    def _spawn_fixture(self, name):
        spec = FIXTURES[name]
        mid = int(self.prng.choice(spec["ids"]))
        np.random.seed(int(self.prng.integers(2 ** 31)))    # rand_pose draws from np.random
        ylim = list(spec["y"])
        if name == "036_cabinet" and os.environ.get("PLAY_CAB_Y"):
            ylim = [float(os.environ["PLAY_CAB_Y"])] * 2
        fx = rand_create_sapien_urdf_obj(scene=self, modelname=name, modelid=mid, xlim=list(spec["x"]),
                                         ylim=ylim, zlim=list(spec["z"]), rotate_rand=spec["yaw"] > 0,
                                         rotate_lim=[0, 0, spec["yaw"]], qpos=spec["q"], fix_root_link=True)
        fx.set_mass(0.01)
        props = spec["props"]
        if name == "036_cabinet" and os.environ.get("PLAY_CAB_PROPS"):
            props = tuple(float(x) for x in os.environ["PLAY_CAB_PROPS"].split(","))
        if props is not None:
            fx.set_properties(*props)
        fx.play_name, fx.play_mid = name, mid
        lo, hi = fx.get_qlimits()[self._fixture_joint(fx)[1]]
        start = {"044_microwave": (0.0, 0.0), "036_cabinet": (0.0, 0.0), "015_laptop": (0.15, 0.25),
                 "056_switch": (0.0, 1.0)}[name]
        q = fx.get_qpos().copy()
        q[self._fixture_joint(fx)[1]] = lo + (hi - lo) * float(self.prng.uniform(*start))
        if name == "056_switch":
            q[self._fixture_joint(fx)[1]] = lo if self.prng.random() < 0.5 else hi
        fx.set_qpos(q)
        self.add_prohibit_area(fx, padding=0.02)
        self.fixtures.append(fx)
        return fx

    # --------------------------------------------------------------- helpers
    def _aabb(self, actor):
        import sapien.physx as px
        if isinstance(actor.actor, px.PhysxArticulation):
            boxes = [L.compute_global_aabb_tight() for L in actor.actor.get_links() if L.get_collision_shapes()]
            return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
        comp = actor.actor.find_component_by_type(px.PhysxRigidDynamicComponent)
        lo, hi = comp.compute_global_aabb_tight()
        return np.asarray(lo), np.asarray(hi)

    def _xy(self, actor):
        """Where the object is on the table: the middle of its box (a pen's
        origin is at one end)."""
        lo, hi = self._aabb(actor)
        return 0.5 * (lo[:2] + hi[:2])

    def _is_fixture(self, a):
        return a in getattr(self, "fixtures", [])

    def _fixture_joint(self, fx, cp=None):
        """(joint, index in qpos) of the joint that moves the handle (contact
        point `cp`, default the fixture's first)."""
        cp = FIXTURES[fx.play_name]["cps"][0] if cp is None else cp
        link = fx.link_dict[fx.config["contact_points"][cp]["base"]]
        active = [a.get_name() for a in fx.actor.get_active_joints()]
        L = link
        while L is not None:
            j = L.get_joint()
            if j is not None and j.get_name() in active:
                return j, active.index(j.get_name())
            L = L.get_parent()
        if len(active) == 1:    # the switch's press point sits on its fixed base: the one joint is the rocker
            return fx.actor.get_active_joints()[0], 0
        raise ValueError(f"no active joint above {link.get_name()}")

    def _fixture_frac(self, fx):
        j, i = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[i]
        return float((fx.get_qpos()[i] - lo) / max(hi - lo, 1e-6))

    def _moved_by(self, fx, T, dq):
        """World transform T carried along by the fixture's joint moving by dq."""
        j, _ = self._fixture_joint(fx)
        J = j.get_global_pose().to_transformation_matrix()
        axis, o = J[:3, 0], J[:3, 3]
        M = np.eye(4)
        if j.type == "prismatic":
            M[:3, 3] = axis * dq
        else:
            R = t3d.axangles.axangle2mat(axis, dq)
            M[:3, :3] = R
            M[:3, 3] = o - R @ o
        return M @ T

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
        p = np.r_[self._xy(obj), self._aabb(obj)[0][2]]
        return bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] > lo[2] - 0.01)

    def _on_table(self, actor):
        return TABLE_Z - 0.05 < actor.get_pose().p[2] < TABLE_Z + 0.3

    def _clear(self, xy, actor, others, gap=0.03):
        """No neighbour's box within the actor's own half-diagonal + gap of xy."""
        lo, hi = self._aabb(actor)
        need = 0.5 * float(np.linalg.norm((hi - lo)[:2])) + gap
        xy = np.asarray(xy, dtype=float)
        for o in others:
            olo, ohi = self._aabb(o)
            if np.linalg.norm(np.clip(xy, olo[:2], ohi[:2]) - xy) < need:
                return False
        return True

    def _free_spot(self, actor, side=None, min_move=0.08, r=0.09, near=None):
        """A free table spot (on `side` if given; within near=(xy, radius) if given)."""
        xs = {"left": (-0.25, -0.06), "right": (0.06, 0.25), None: (-0.25, 0.25)}[side]
        p0 = self._xy(actor)
        others = [a for a in self.objects + self.containers if a is not actor and self._on_table(a)] + self.fixtures
        for i in range(100):
            if near is not None:
                d = near[1] * (0.0 if i == 0 else np.sqrt(self.prng.uniform(0, 1)))
                a = self.prng.uniform(-np.pi, np.pi)
                xy = np.asarray(near[0], dtype=float) + d * np.array([np.cos(a), np.sin(a)])
                if not (-0.28 < xy[0] < 0.28 and -0.2 < xy[1] < 0.1):
                    continue
            else:
                xy = np.array([self.prng.uniform(*xs), self.prng.uniform(-0.15, 0.08)])
            if np.linalg.norm(xy - p0) < min_move:
                continue
            if self._clear(xy, actor, others):
                return xy
        return None

    # ------------------------------------------------------------- planning
    def _planner_frame(self, planner, arm, p):
        """A world point in the planner's base frame (the same transform RoboTwin
        applies to target poses in CuroboPlanner.plan_path)."""
        base = planner.robot_origion_pose
        Rb = t3d.quaternions.quat2mat(np.asarray(base.q, dtype=float)).T
        rel = Rb @ (np.asarray(p, dtype=float) - np.asarray(base.p)) + np.asarray(planner.frame_bias, dtype=float)
        R = np.eye(3)
        if "aloha-agilex" in planner.yml_path:
            R = t3d.axangles.axangle2mat([0, 0, 1], -0.02 if str(arm) == "left" else -0.01)
            rel = R @ rel
        return rel, R @ Rb

    def _world(self, arm, near=None):
        """Load the scene objects (except the ones this action handles, what is
        inside those, and neighbours within 15 cm of the plan's target, where
        the hand has to get close anyway) into the arm's planner as boxes, so
        paths between actions don't sweep through them. RoboTwin's planners
        know only the table."""
        planner = self.robot.left_planner if str(arm) == "left" else self.robot.right_planner
        if not hasattr(planner, "motion_gen") or os.environ.get("PLAY_NO_OBSTACLES"):
            return
        from curobo.geom.types import WorldConfig
        ign = set(getattr(self, "_ignore", ()))
        conts = [c for c in ign if c in self.containers]
        base = planner.robot_origion_pose
        cub = {"table": {"dims": [0.7, 2, 0.04], "pose": [float(base.p[1]), 0.0, 0.74 - float(base.p[2]), 1, 0, 0, 0.0]}}
        for a in self.objects + self.containers + self.fixtures:
            if a in ign or not (self._is_fixture(a) or self._on_table(a)) or any(self._inside(a, c) for c in conts):
                continue
            if self._is_fixture(a):
                # a box per part (door, drawer, body), so the hand can reach into an open drawer
                for L in a.actor.get_links():
                    if not L.get_collision_shapes() or (a, L.get_name()) in getattr(self, "_ignore_links", set()):
                        continue
                    lo, hi = (np.asarray(x) for x in L.compute_global_aabb_tight())
                    p, R = self._planner_frame(planner, arm, 0.5 * (lo + hi))
                    cub[f"{a.get_name()}/{L.get_name()}"] = {"dims": [float(x) for x in (hi - lo)],
                                                            "pose": [float(x) for x in p] + [float(x) for x in t3d.quaternions.mat2quat(R)]}
                continue
            lo, hi = self._aabb(a)
            small = max((hi - lo)[:2]) < 0.15 and (hi - lo)[2] < 0.12   # big things stay obstacles even next to the target
            if near is not None and small and np.linalg.norm(np.clip(np.asarray(near[:2], dtype=float), lo[:2], hi[:2]) - np.asarray(near[:2], dtype=float)) < 0.12:
                continue
            p, R = self._planner_frame(planner, arm, 0.5 * (lo + hi))
            q = t3d.quaternions.mat2quat(R)
            cub[a.get_name()] = {"dims": [float(x) for x in (hi - lo)], "pose": [float(x) for x in p] + [float(x) for x in q]}
            if len(cub) >= N_OBSTACLES:
                break
        key = json.dumps(cub, sort_keys=True)
        if getattr(planner, "_play_world_key", None) == key:
            return
        planner.motion_gen.update_world(WorldConfig.from_dict({"cuboid": cub}))
        planner._play_world_key = key

    def _plan(self, arm, pose, constraint=None, q=None):
        f = self.robot.left_plan_path if arm == "left" else self.robot.right_plan_path
        self.n_plans += 1
        self._world(arm, near=pose[:3])
        # cuRobo draws its optimisation seeds from torch's generator: seed it per
        # plan so an episode replays identically from its seed
        import torch
        s = (self._plan_seed * 1000003 + self.n_plans) % (2 ** 31)
        torch.manual_seed(s)
        np.random.seed(s)
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

    JOINT_MARGIN = 0.06   # rad beyond the range the expert data uses (same as the planning URDF's limits)

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

    @staticmethod
    def _merge_plans(plans, squeeze=0.8):
        """Several planned segments (each starting and ending at rest) as one
        continuous motion: the joint path concatenated and re-timed with one
        smooth speed profile (ease in, cruise, ease out), so a pull along a
        drawer's slide or a door's arc doesn't stop at every waypoint."""
        P = np.concatenate([np.asarray(p["position"], float)[(1 if i else 0):] for i, p in enumerate(plans)], 0)
        if len(P) < 4:
            return dict(plans[0], position=P, velocity=np.zeros_like(P))
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] < 1e-6:
            return dict(plans[0], position=P[:2], velocity=np.zeros_like(P[:2]))
        n = max(4, int(squeeze * len(P)))                  # the stops are what made it slow
        u = np.linspace(0, 1, n)
        s_ = 0.5 - 0.5 * np.cos(np.pi * u)                 # smooth start and stop
        s_ = 0.6 * s_ + 0.4 * np.clip((u - 0.15) / 0.7, 0, 1) if n > 20 else s_   # mostly steady in the middle
        s_ = (s_ - s_[0]) / (s_[-1] - s_[0])
        Q = np.stack([np.interp(s_ * d[-1], d, P[:, k]) for k in range(P.shape[1])], 1)
        return dict(plans[0], position=Q, velocity=np.gradient(Q, axis=0) * 250.0)

    def _set_paths(self, arm, plans):
        if arm == "left":
            self.left_joint_path, self.left_cnt = list(plans), 0
        else:
            self.right_joint_path, self.right_cnt = list(plans), 0

    def _noisy(self, arm, plan):
        """Injected shaky hand / jitter: smooth noise added to the joint targets
        of this plan (kept inside the expert joint range)."""
        e = getattr(self, "err", None)
        if not e or not (errors.has(e, "shaky") or errors.has(e, "jitter")):
            return plan
        P = np.asarray(plan["position"], dtype=np.float64).copy()
        n = len(P)
        if n < 3:            # nothing to shake on a path this short
            return plan
        if errors.has(e, "shaky"):
            P[:, :6] += errors.smooth_noise(self.prng, n, e["sigma"], 6)
        if errors.has(e, "jitter"):
            t0 = int(e["frac"] * self.action_len) - self.action_steps   # burst placed in the action's own time
            k = int(e["len"] * 250)
            a, b = max(0, t0), min(n, max(0, t0) + k)
            if b > a and t0 < n:
                P[a:b, :6] += errors.smooth_noise(self.prng, b - a, e["sigma"], 6, cutoff=3.0)
        box = self._joint_box(arm)
        if box is not None:
            P[:, :6] = np.clip(P[:, :6], box[0], box[1])
        out = dict(plan)
        out["position"] = P
        out["velocity"] = np.gradient(P, axis=0) * 250.0
        return out

    def _count(self, plan):
        self.action_steps += len(plan["position"])

    def _run(self, steps):
        """Execute pre-planned steps through RoboTwin's move() in replay mode.
        steps: ("move", arm, plan) | ("grip", arm, value) |
               ("both", (arm, plan_or_None, grip_or_None), (arm, plan_or_None, grip_or_None))"""
        for st in steps:
            self.plan_success = True
            if st[0] == "move":
                _, arm, plan = st
                plan = self._noisy(arm, plan)
                self._count(plan)
                self._set_paths(arm, [plan])
                self.need_plan = False
                try:
                    self.move((arm, [Action(arm, "move", target_pose=[0.0] * 7)]))
                finally:
                    self.need_plan = True
            elif st[0] == "grip":
                _, arm, val = st
                self.move((arm, [Action(arm, "open" if val > 0.5 else "close", target_gripper_pos=float(val))]))
            elif st[0] == "snatch":
                _, arm, plan = st
                plan = self._noisy(arm, plan)
                self._count(plan)
                seq = {"left_arm": None, "left_gripper": None, "right_arm": None, "right_gripper": None}
                seq[f"{arm}_arm"] = plan
                seq[f"{arm}_gripper"] = self.set_gripper(set_tag=str(arm), **{f"{arm}_pos": 0.0})
                self.take_dense_action(seq)
            else:
                acts = []
                for arm, plan, g in st[1:]:
                    if plan is not None:
                        plan = self._noisy(arm, plan)
                        self._count(plan)
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
    def _grasp_options(self, actor, arm, pre_dis=0.1, k=6, exclude=(), avoid=None, pref=None, below=None, high_first=False,
                       avoid_r=0.04, ends_first=False):
        """Grasp points ranked like RoboTwin's chooser (prefer top-down, then
        the arm's preferred side; or a given orientation), with some randomness
        so the same object is not always grabbed the same way."""
        pref_dir = self.robot.get_grasp_perfect_direction(arm)
        lo, hi = self._aabb(actor)
        ext = np.sort((hi - lo)[:2])
        thin = ext[0] < 0.035 and ext[1] > 0.1      # a pen held by its end pivots in the fingers
        mid = 0.5 * (lo + hi)
        opts = []
        points = [(i, np.asarray(cp[:3]), np.zeros(3)) for i, cp in actor.iter_contact_points("list")]
        if ends_first and any(actor.get_name().startswith(n) for n in SLIDE):
            # a uniform long thing can be held anywhere along it: the annotated
            # grasps slid along its long axis
            ext = np.asarray(actor.config["extents"], float) * np.asarray(actor.config["scale"], float)
            k_ = int(np.argmax(ext))
            ax = t3d.quaternions.quat2mat(np.asarray(actor.get_pose().q, float))[:, k_]
            if abs(ax[2]) < 0.4:
                half = 0.5 * ext[k_] - 0.05
                for i, c, _ in list(points):
                    for t in np.linspace(-half, half, 5):
                        sh = ax * (t - float((c - mid) @ ax))
                        points.append((i, c + sh, sh))
        for i, cp, shift in points:
            if i in exclude and not shift.any():
                continue
            if avoid is not None and np.linalg.norm(cp - avoid) < avoid_r:
                continue
            if below is not None and cp[2] > below:
                continue
            pre = self.get_grasp_pose(actor, arm, contact_point_id=i, pre_dis=pre_dis)
            if pre is None:
                continue
            pre = np.array(pre, dtype=np.float64)
            pre[:3] += shift
            pose = pre.copy()
            R = t3d.quaternions.quat2mat(pre[3:])
            pose[:3] += np.array([pre_dis, 0, 0]) @ np.linalg.inv(R)
            # fingertips off the table: an annotated grasp on something low (a
            # pen) puts them at table height, they pinch it from below and roll it
            tip_z = pose[2] + TIP * R[2, 0]
            clear = float(os.environ.get("PLAY_TIP_CLEAR", 0.0035))
            if tip_z < TABLE_Z + clear and R[2, 0] < -0.3:
                back = (TABLE_Z + clear - tip_z) / -R[2, 0]
                pose[:3] -= R[:, 0] * back
                pre[:3] -= R[:, 0] * back
            e = getattr(self, "err", None)
            if errors.has(e, "grasp_offset"):      # off-centre, along the finger plane (hand z)
                pre[:3] += R[:, 2] * e["offset"]
                pose[:3] += R[:, 2] * e["offset"]
            if errors.has(e, "grasp_high"):        # fingertips only: stop short along the approach
                pose[:3] -= R[:, 0] * e["offset"]
            if errors.has(e, "close_late"):        # overshoot into the object
                pose[:3] += R[:, 0] * e["overshoot"]
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
                if ends_first:     # hold it by one end (a hand-over end to end)
                    score -= 6.0 * float(np.linalg.norm(np.asarray(cp[:2]) - mid[:2]))
                elif thin:
                    score += 4.0 * float(np.linalg.norm(np.asarray(cp[:2]) - mid[:2]))
                opts.append((score + self.prng.normal(0, 0.08), i, pre_, pose_))
        opts.sort(key=lambda x: x[0])
        if errors.has(getattr(self, "err", None), "unstable_grasp") and len(opts) > 1:
            opts = opts[len(opts) // 2:] + opts[:len(opts) // 2]   # one of the worse-ranked ones first
        return opts[:k]

    def _then_pose(self, grasp_pose, actor, then, dzs=(0.1, 0.05)):
        """The hand poses (grasp orientation kept) that hold the object's origin
        over `then` at the carry-over and the pre-place height: used to check
        that a put-down there is reachable before the object is picked up."""
        lo, hi = self._aabb(actor)
        off = np.asarray(grasp_pose[:3]) - [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]]   # as _place sets it down
        return [list(np.asarray(then, dtype=float) + off + [0, 0, dz]) + list(grasp_pose[3:]) for dz in dzs]

    def _pick(self, actor, arm, lift=0.1, then=None, arms=None, exclude=(), avoid=None, pref=None, home_other=True,
              below=None, high_first=False, then_dz=(0.1, 0.05), **gkw):
        """Plan a whole pick (approach, grasp, lift, and if `then` is given a
        hand pose it must be able to reach while holding), then execute it.
        Tries this arm, then the other. Returns (held, arm), or (None, arm) if
        nothing could be planned (nothing moved)."""
        z0 = actor.get_pose().p[2]
        actor._play_q0 = np.array(actor.get_pose().q, dtype=float)
        ign0 = set(getattr(self, "_ignore", ()))
        self._ignore = {actor}
        try:
            self._then_dz = then_dz
            return self._pick1(actor, arm, lift, then, arms, exclude, avoid, pref, home_other, below, high_first, z0, gkw)
        finally:
            self._ignore = ign0

    def _pick1(self, actor, arm, lift, then, arms, exclude, avoid, pref, home_other, below, high_first, z0, gkw):
        for arm_ in (arms or (arm, arm.opposite)):
            for _, cid, pre, pose in self._grasp_options(actor, arm_, exclude=exclude, avoid=avoid, pref=pref,
                                                          below=below, high_first=high_first, **gkw):
                legs = [(pre, None), (pose, CONS)]
                if lift > 0:
                    up = pose.copy()
                    up[2] += lift
                    legs.append((up, None))
                plans = self._plan_chain(arm_, legs, flips=())   # (the options already include the flipped hand)
                if plans is None:
                    continue
                if then is not None and self._plan_about(arm_, [(p_, None) for p_ in self._then_pose(pose, actor, then, self._then_dz)], then,
                                                         q0=self._q_after(arm_, plans)) is None:
                    continue
                home = self._home_plan(arm_.opposite) if home_other else None
                self._sync(arm_)
                e = getattr(self, "err", None)
                first = ("both", (arm_, plans[0], None), (arm_.opposite, home, None)) if home is not None else ("move", arm_, plans[0])
                if errors.has(e, "close_early"):
                    # the fingers shut `early` before the grasp point, and the hand pushes on with them shut
                    P1 = np.asarray(plans[1]["position"])
                    k = max(1, len(P1) - int(e["early"] / 0.1 * len(P1)))   # approach leg is pre_dis = 0.1 long
                    a_, b_ = dict(plans[1]), dict(plans[1])
                    a_["position"], a_["velocity"] = P1[:k], np.asarray(plans[1]["velocity"])[:k]
                    b_["position"], b_["velocity"] = P1[k - 1:], np.asarray(plans[1]["velocity"])[k - 1:]
                    steps = [first, ("move", arm_, a_), ("grip", arm_, 0.0), ("move", arm_, b_)]
                elif errors.has(e, "lift_early") and lift > 0:
                    steps = [first, ("move", arm_, plans[1]), ("snatch", arm_, plans[2])]
                else:
                    steps = [first, ("move", arm_, plans[1]), ("grip", arm_, 0.0)]
                if lift > 0 and not errors.has(e, "lift_early"):
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

    def _place(self, actor, arm, target, pre_dis, dis, up=0.07, release=True, over_extra=0.08, yaw=None, **kw):
        """Put the held object down at target (7-pose or xyz) if that can be
        planned; returns True if it was executed."""
        target = list(target) + [1, 0, 0, 0] if len(target) == 3 else list(target)
        kw.setdefault("constrain", "free")
        if self._tilt(actor) > 0.45:
            self._why = "tilted in the hand"   # re-uprighting it would swing the arm across the table
            if os.environ.get("PLAY_DEBUG"):
                print(f"   PLACE {actor.get_name()} {arm}: tilted {self._tilt(actor):.2f} rad in the hand", flush=True)
            return False
        e = getattr(self, "err", None)
        if errors.has(e, "knock_over"):
            kn = self._knock_spot(actor, target[:3], e["overlap"])
            if kn is not None:
                target = list(kn) + target[3:]
                e["hit"] = self._knock_m
        if errors.has(e, "place_offset"):
            target[0] += e["offset_xy"][0]
            target[1] += e["offset_xy"][1]
        if errors.has(e, "drop_high"):
            dis = dis + e["drop"]
        # the object goes down as it is held: its middle over the spot, its
        # bottom onto the surface (RoboTwin's get_place_pose turns it back to
        # its model's upright pose about its origin: a pen lying on its side,
        # whose origin is at one end, would be stood on end)
        pre, pose = self._down_poses(actor, arm, target, pre_dis, dis)
        if kw["constrain"] == "align" and kw.get("align_axis") is not None:
            R = t3d.quaternions.quat2mat(np.asarray(actor.get_pose().q, dtype=float))
            ax = max((R[:, 0], R[:, 2], R[:, 1]), key=lambda v: np.linalg.norm(v[:2]))
            want = np.asarray(kw["align_axis"], dtype=float)
            yaw = (np.arctan2(want[1], want[0]) - np.arctan2(ax[1], ax[0]) + np.pi / 2) % np.pi - np.pi / 2
            (pre, _), (pose, _) = self._yawed([(pre, None), (pose, None)], target[:3], yaw)
        elif yaw is not None:      # turned by the caller (to fit a slot)
            (pre, _), (pose, _) = self._yawed([(pre, None), (pose, None)], target[:3], yaw)
        # carry over at the lifted height, then come down onto the spot (the
        # planner knows nothing but the table: a diagonal path cuts through walls)
        over = list(pre)
        over[2] = max(float(pre[2]), min(float(self._ee(arm)[2]), float(pre[2]) + over_extra))
        legs = [(over, None), (pre, None), (pose, CONS)]
        if errors.has(e, "carry_hit"):
            hit = self._carry_hit_pose(actor, arm, target[:3])
            if hit is not None:
                legs = [(hit, None)] + legs[1:]
        if up:
            upp = list(pose)
            upp[2] += up
            legs.append((upp, CONS))
        plans = self._plan_about(arm, legs, target[:3], angs=(0.0, np.pi) if (kw["constrain"] == "align" or yaw is not None) else (0.0, np.pi, np.pi / 2, -np.pi / 2))
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PLACE {actor.get_name()} {arm} target={np.round(target[:3], 3)} pre={np.round(pre[:3], 3)} pose={np.round(pose[:3], 3)} "
                  f"planned={plans is not None} {'' if plans else self._why}", flush=True)
        if plans is None:
            return False
        self._sync(arm)
        steps = [("move", arm, p) for p in plans[:len(legs) - (1 if up else 0)]]
        if errors.has(e, "release_early") and release:
            # the hand opens part-way through the carry and carries on with it open
            P0 = np.asarray(plans[0]["position"])
            k = max(2, int(e["frac"] * len(P0)))
            a_, b_ = dict(plans[0]), dict(plans[0])
            a_["position"], a_["velocity"] = P0[:k], np.asarray(plans[0]["velocity"])[:k]
            b_["position"], b_["velocity"] = P0[k - 1:], np.asarray(plans[0]["velocity"])[k - 1:]
            steps = [("move", arm, a_), ("grip", arm, 1.0), ("move", arm, b_)] + steps[1:]
            release = False
        if release:
            steps.append(("grip", arm, 1.0))
        if up:
            steps.append(("move", arm, plans[-1]))
        ok = self._run(steps)
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PLACED {actor.get_name()} now at {np.round(actor.get_pose().p, 3)} (target {np.round(target[:3], 3)}) ran={ok}", flush=True)
        return ok

    def _down_poses(self, actor, arm, target, pre_dis, dis):
        """Hand poses (above, at) that put the held object down as it is held:
        its middle over target xy, its bottom dis above target z."""
        lo, hi = self._aabb(actor)
        ee = self._ee(arm)
        d = np.array([target[0] - 0.5 * (lo[0] + hi[0]), target[1] - 0.5 * (lo[1] + hi[1]), target[2] - lo[2]])
        return list(ee[:3] + d + [0, 0, pre_dis]) + list(ee[3:]), list(ee[:3] + d + [0, 0, dis]) + list(ee[3:])

    def _knock_spot(self, actor, target, overlap):
        """Injected error: a set-down spot half on top of a standing neighbour
        near the target, so something topples."""
        best = None
        for m in self.objects + self.containers:
            if m is actor or not self._on_table(m):
                continue
            pm = m.get_pose().p
            d = np.linalg.norm(pm[:2] - np.asarray(target[:2]))
            if d < 0.3 and (best is None or d < best[0]):
                best = (d, m)
        if best is None:
            return None
        m = best[1]
        lo, hi = self._aabb(m)
        alo, ahi = self._aabb(actor)
        u = np.asarray(target[:2]) - m.get_pose().p[:2]
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0])
        r = 0.5 * float(np.abs((hi - lo)[:2]) @ np.abs(u)) + 0.5 * float(np.abs((ahi - alo)[:2]) @ np.abs(u)) - overlap
        xy = m.get_pose().p[:2] + u * max(r, 0.01)
        self._knock_m = m.get_name()
        self._ignore = set(getattr(self, "_ignore", ())) | {m}
        return [float(xy[0]), float(xy[1]), float(hi[2]) + 0.01]

    def _carry_hit_pose(self, actor, arm, target):
        """Injected error: a hand pose on the way that swings the held object
        through a standing neighbour."""
        p0 = actor.get_pose().p
        best = None
        for m in self.objects + self.containers:
            if m is actor or not self._on_table(m):
                continue
            pm = m.get_pose().p
            d = np.asarray(target[:2]) - p0[:2]
            s_ = float(np.clip((pm[:2] - p0[:2]) @ d / max(d @ d, 1e-9), 0.05, 0.95))
            off = np.linalg.norm(p0[:2] + s_ * d - pm[:2])
            if off < 0.2 and (best is None or off < best[0]):
                best = (off, m)
        if best is None:
            return None
        m = best[1]
        self._ignore = set(getattr(self, "_ignore", ())) | {m}
        lo, hi = self._aabb(m)
        alo, _ = self._aabb(actor)
        ee = self._ee(arm)
        # held object's bottom passes 2 cm below the neighbour's top, right over it
        z = hi[2] - 0.02 + (ee[2] - alo[2])
        self.err["hit_carry"] = m.get_name()
        return [float(m.get_pose().p[0]), float(m.get_pose().p[1]), float(z)] + list(ee[3:])

    def _tilt(self, actor):
        """How far the object has tipped since it was last grasped (the world's
        up axis seen from the object's frame then and now; yaw doesn't count)."""
        q0 = getattr(actor, "_play_q0", None)
        if q0 is None:
            return 0.0
        u0 = t3d.quaternions.quat2mat(np.asarray(q0, dtype=float)).T @ [0, 0, 1]
        u1 = t3d.quaternions.quat2mat(np.asarray(actor.get_pose().q, dtype=float)).T @ [0, 0, 1]
        return float(np.arccos(np.clip(u0 @ u1, -1, 1)))

    def _put_down_here(self, actor, arm):
        """Couldn't plan the put-down: set it down right where the hand is."""
        p = actor.get_pose().p
        if self._tilt(actor) < 0.45 and self._place(actor, arm, [p[0], p[1], TABLE_Z], 0.03, 0.005):
            return
        # tilted in the hand (or no plan): lower it straight down as it is and let go
        ee = self._ee(arm)
        drop = float(self._aabb(actor)[0][2] - TABLE_Z - 0.003)
        down = [ee[0], ee[1], ee[2] - max(drop, 0.0)] + list(ee[3:])
        plans = self._plan_chain(arm, [(down, CONS)]) if drop > 0.01 else []
        if plans is None:
            self._run([("grip", arm, 1.0)])   # at least let go of it
            return
        self._sync(arm)
        self._run([("move", arm, pl) for pl in plans] + [("grip", arm, 1.0)])
        up = [ee[0], ee[1], ee[2] - max(drop, 0.0) + 0.07] + list(ee[3:])
        self._go(arm, [(up, None)])

    # ------------------------------------------------------------- results
    def _res(self, ok, lang, arm, **extra):
        d = dict(success=bool(ok), language=lang, arm=str(arm), grasp=getattr(self, "last_grasp", None))
        d.update(extra)
        return d

    def _start(self, actor, lang, lift=0.1, then=None, then_dz=(0.1, 0.05)):
        """Common opening of a pick-and-do action: (True, arm) if the object
        is in the hand, (False, arm) if the grasp failed (arm sent home),
        (None, arm) if nothing could be planned."""
        self.last_grasp = None
        arm = self._arm_for(actor.get_pose().p)
        if then is not None:
            arm = self._arm_for(then)
        ok, arm = self._pick(actor, arm, lift=lift, then=then, then_dz=then_dz)
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
        if not placed:
            xy2 = self._free_spot(actor, side=str(arm))    # changes its mind: a spot on its own side
            if xy2 is not None:
                xy = xy2
                placed = self._place(actor, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01)
        ok = placed and np.linalg.norm(self._xy(actor) - xy) < 0.05
        return self._finish(actor, arm, placed, ok, lang)

    def p_put_in(self, actor, cont):
        lang = f"put the {self._name(actor)} in the {self._name(cont)}"
        fp = list(cont.get_functional_point(0)[:3])   # position only: the point's own frame may point down
        pre_in = max(0.08, float(self._aabb(cont)[1][2]) - fp[2] + 0.05)     # bottom clears the rim
        st, arm = self._start(actor, lang, lift=max(0.12, float(self._aabb(cont)[1][2]) - TABLE_Z + 0.06), then=fp[:3],
                              then_dz=(pre_in + 0.05, pre_in))
        if st is None:
            # no arm can both pick it up and reach the container: pass it over
            if self._arm_for(actor.get_pose().p) != self._arm_for(fp):
                res = self.p_handover(actor, dest=fp + [1, 0, 0, 0])
                if res is not None:
                    res.update(language=f"pass the {self._name(actor)} over and put it in the {self._name(cont)}",
                               success=bool(res["success"] and self._inside(actor, cont)), via="handover")
                return res
            return None
        if not st:
            return self._res(False, lang, arm)
        fp = list(cont.get_functional_point(0)[:3])   # re-read: the pick may have nudged it
        rim = float(self._aabb(cont)[1][2])
        placed = self._place(actor, arm, fp, max(0.08, rim - fp[2] + 0.05), 0.03)
        return self._finish(actor, arm, placed, self._inside(actor, cont), lang)

    def p_stack(self, actor, base):
        lang = f"put the {self._name(actor)} on the {self._name(base)}"
        lo, hi = self._aabb(base)
        alo, ahi = self._aabb(actor)
        if not any(base.get_name().startswith(n) for n in STACK_BASES) or np.prod((hi - lo)[:2]) < 0.6 * np.prod((ahi - alo)[:2]):
            return None   # nothing stands on a bottle or on something much smaller
        bp = base.get_pose().p
        st, arm = self._start(actor, lang, lift=0.12, then=[bp[0], bp[1], hi[2]], then_dz=(0.11, 0.06))
        if st is None:
            return None
        if not st:
            return self._res(False, lang, arm)
        bp, (lo, hi) = base.get_pose().p, self._aabb(base)   # re-read: the pick may have nudged it
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
        others = [a for a in self.objects + self.containers if a not in (actor, anchor)] + self.fixtures
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
            if not self._clear(xy, actor, [o for o in others if self._on_table(o)]):
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
        ok = placed and np.linalg.norm(self._xy(actor) - xy) < 0.04
        return self._finish(actor, arm, placed, ok, lang)

    def p_handover(self, actor, dest=None):
        """Pick it up with one hand, hold it up in the middle, let the other
        hand take it from the side, and put it down on the other side."""
        lo, hi = self._aabb(actor)
        cps = [np.asarray(cp[:3]) for _, cp in actor.iter_contact_points("list")]
        if not cps:
            return None
        cz = [c[2] for c in cps]
        spread = max(np.linalg.norm(a[:2] - b[:2]) for a in cps for b in cps)
        if hi[2] - lo[2] >= 0.09 and max(cz) - min(cz) >= 0.045:
            mode = "stacked"     # tall: the second hand takes it below the first
        elif spread >= 0.09 or any(actor.get_name().startswith(n) for n in SLIDE):
            mode = "ends"        # long: held by one end, the second hand takes the other
        else:
            return None
        a1 = self._arm_for(self._xy(actor))
        a2 = a1.opposite
        what = self._name(actor)
        lang = f"pass the {what} from the {a1} hand to the {a2} hand"
        self.last_grasp = None
        ok, a1 = self._pick(actor, a1, lift=0.1, arms=(a1,), high_first=mode == "stacked", ends_first=mode == "ends")
        if ok is None:
            return None
        if not ok:
            self._home(a1)
            return self._res(False, lang, a1)
        g1 = self.last_grasp
        # to the middle, still upright, the hand at a comfortable height
        ee = self._ee(a1)
        went = False
        hang = float(ee[2] - self._aabb(actor)[0][2])   # how far the object reaches below the hand
        for hz in (0.95, 0.99, 0.91):
            hz = max(hz, TABLE_Z + 0.05 + hang)            # its bottom stays clear of the table
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
        # the other hand comes in from its side, horizontally (tall things), or
        # takes the free end (long things)
        pref = GRASP_DIRECTION_DIC["right" if a2 == "left" else "left"] if mode == "stacked" else None
        if errors.has(getattr(self, "err", None), "handover_miss"):
            # the receiving hand closes early, beside the object, and the first hand lets go anyway
            self.err["early"] = 0.06
            self.err["parts"] = list(self.err["parts"]) + ["close_early"]
        if mode == "stacked":
            ok2, _ = self._pick(actor, a2, lift=0.0, arms=(a2,), exclude=(g1["contact_point"],), avoid=self._finger_xyz(a1),
                                pref=pref, home_other=False, below=float(self._finger_xyz(a1)[2]) - 0.04)
        else:
            ok2, _ = self._pick(actor, a2, lift=0.0, arms=(a2,), exclude=(g1["contact_point"],), avoid=self._finger_xyz(a1),
                                avoid_r=0.08, home_other=False, ends_first=True)
        if errors.has(getattr(self, "err", None), "handover_miss") and ok2 is not None:
            self._run([("grip", a1, 1.0)])   # ... lets go regardless of whether the other hand has it
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
            xy = None
            for r in (0.1, 0.18):
                xy = self._free_spot(actor, min_move=0.0, near=(self._xy(actor), r))
                if xy is not None:
                    break
            if xy is None:
                xy = self._free_spot(actor, side=str(a2), min_move=0.0)
            target = None if xy is None else [xy[0], xy[1], TABLE_Z]
        placed = False
        if target is not None:
            placed = self._place(actor, a2, list(target), 0.06, 0.02)
        if os.environ.get("PLAY_DEBUG"):
            print(f"   HANDOVER end placed={placed} on_table={self._on_table(actor)} z={actor.get_pose().p[2]:.3f} target={target}", flush=True)
        res = self._finish(actor, a2, placed, True, lang, grasp2=g2)
        if os.environ.get("PLAY_DEBUG"):
            print(f"   HANDOVER final on_table={self._on_table(actor)} z={actor.get_pose().p[2]:.3f} holds1={self._holds(a1, actor)} holds2={self._holds(a2, actor)}", flush=True)
        # the task is the transfer: the other hand had it, and it is back on the table
        res.update(arm=f"{a1}->{a2}", grasp=g1, success=bool(self._on_table(actor) and not self._holds(a1, actor)),
                   placed=bool(placed))
        return res

    def p_push(self, actor):
        """Push it a little along the table with the closed hand."""
        lo, hi = self._aabb(actor)
        c0 = self._xy(actor)
        p = np.r_[c0, lo[2]]
        h = hi[2] - lo[2]
        arm = self._arm_for(p)
        q = TOPDOWN[str(arm)]
        others = [o for o in self.objects + self.containers if o is not actor]
        if any(self._clear(p[:2], actor, [f]) is False for f in self.fixtures):
            return None   # right against a fixture: no room for the hand
        for _ in range(8):
            ang = self.prng.uniform(-np.pi, np.pi)
            d = np.array([np.cos(ang), np.sin(ang)])
            r = 0.5 * float(np.abs((hi - lo)[:2]) @ np.abs(d))   # its half-extent along the push
            dist = self.prng.uniform(0.06, 0.12)
            end = p[:2] + d * dist
            if not (-0.28 < end[0] < 0.28 and -0.2 < end[1] < 0.1):
                continue
            if np.linalg.norm(end - [0, -0.04]) > np.linalg.norm(p[:2] - [0, -0.04]) + 0.02:
                continue   # towards the middle: a can that rolls on shouldn't roll off the table
            if any(np.linalg.norm(end - o.get_pose().p[:2]) < 0.08 for o in others):
                continue
            z = lo[2] + np.clip(0.5 * h, 0.01, 0.05) + TIP    # fingertips at that height on the object
            start = p[:2] - d * (r + 0.04)
            if errors.has(getattr(self, "err", None), "push_offset"):
                start = start + np.array([-d[1], d[0]]) * self.err["offset"]
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
            moved = float(np.linalg.norm(self._xy(actor) - c0))
            self.last_grasp = None
            if os.environ.get("PLAY_DEBUG"):
                print(f"   PUSH {actor.get_name()} {arm} dist={dist:.3f} r={r:.3f} z={z - TIP - lo[2]:.3f} moved={moved:.3f} now={np.round(actor.get_pose().p, 3)} tilt={self._tilt(actor):.2f}", flush=True)
            return self._res(moved > 0.5 * dist, lang, arm, moved=moved)
        return None

    def p_reorient(self, actor):
        """Pick it up, turn it, put it back down about where it was."""
        lang = f"turn the {self._name(actor)} around"
        p0 = np.r_[self._xy(actor), TABLE_Z]
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
        ok = turned and abs(dq[0]) < 0.99 and placed      # turned by at least ~0.28 rad (it can slip back a little)
        return self._finish(actor, arm, placed, ok, lang)

    # ------------------------------------------------------------ fixtures
    def _operate(self, fx, target_frac, lang, again=1, arms=None, hold=False):
        """Grasp the fixture's handle and move the hand along the joint's own
        arc (door, lid) or slide (drawer) until the joint is at target_frac of
        its range, in short planned segments; let go and back off."""
        j, qi = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[qi]
        q0 = float(fx.get_qpos()[qi])
        dq_total = lo + (hi - lo) * target_frac - q0
        p = fx.get_pose().p
        best = None
        for arm in (arms or (self._arm_for(p), self._arm_for(p).opposite)):
            for cp, pre, pose in self._handle_grasps(fx, arm):
                plans = self._plan_chain(arm, [(pre, None), (pose, CONS)], flips=())
                if plans is None:
                    continue
                # the hand rides along with the handle
                Th = np.eye(4)
                Th[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
                Th[:3, 3] = pose[:3]
                n = int(np.clip(np.ceil(abs(dq_total) / (0.12 if j.type != "prismatic" else 0.025)), 2, 12))
                way = []
                for k in range(1, n + 1):
                    T = self._moved_by(fx, Th, dq_total * k / n)
                    way.append(list(T[:3, 3]) + list(t3d.quaternions.mat2quat(T[:3, :3])))
                # as far along as can be planned (at least a third of the way)
                q_end = self._q_after(arm, plans)
                for m in range(n, max(1, n // 3) - 1, -1):
                    moves = self._plan_chain(arm, [(w, None) for w in way[:m]], q0=q_end, flips=())
                    if moves is not None:
                        best = (arm, cp, plans, moves, m / n)
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is None:
            return None
        arm, cp, plans, moves, frac_planned = best
        home = self._home_plan(arm.opposite)
        self._sync(arm)
        e = getattr(self, "err", None)
        if errors.has(e, "handle_slip") and len(moves) > 1:
            moves = moves[:max(1, int(round(e["frac"] * len(moves))))]    # the hand loses the handle here
            again = again if not e.get("severe") else 0
            self.err = None                                                 # a re-grasp goes cleanly
        moves = [self._merge_plans(moves)]                                  # one steady pull, not stop-and-go
        self._run([("both", (arm, plans[0], 1.0), (arm.opposite, home, None)) if home is not None else ("move", arm, plans[0]),
                   ("move", arm, plans[1]), ("grip", arm, 0.0)] + [("move", arm, mv) for mv in moves] +
                  ([] if hold else [("grip", arm, 1.0)]))
        if hold:                                   # keeps holding it (the other hand has something to do)
            q1 = float(fx.get_qpos()[qi])
            moved = (q1 - q0) / dq_total if abs(dq_total) > 1e-6 else 0.0
            self.last_grasp = dict(contact_point=cp if isinstance(cp, str) else int(cp), arm=str(arm))
            return self._res(moved > 0.6, lang, arm, moved=float(moved), fixture=fx.get_name(), held=True)
        ee = self._ee(arm)
        self._go(arm, [(list(ee[:3] - 0.08 * _hand_x(ee[3:])) + list(ee[3:]), None)])
        self.last_grasp = dict(contact_point=cp if isinstance(cp, str) else int(cp), arm=str(arm))
        lo_, hi_ = fx.get_qlimits()[qi]
        if fx.play_name == "015_laptop" and dq_total < 0 and self._fixture_frac(fx) > target_frac + 0.08:
            for a_ in (arm, arm.opposite):
                pressed = self._press_lid_down(fx, a_, lo_ + (hi_ - lo_) * target_frac - float(fx.get_qpos()[qi]))
                if os.environ.get("PLAY_DEBUG"):
                    print(f"   PRESS LID {a_} ran={pressed} lid now {self._fixture_frac(fx):.2f}", flush=True)
                if pressed:
                    if a_ != arm:
                        self._home(a_)
                    break
        q1 = float(fx.get_qpos()[qi])
        moved = (q1 - q0) / dq_total if abs(dq_total) > 1e-6 else 0.0
        if os.environ.get("PLAY_DEBUG"):
            print(f"   OPERATE {fx.get_name()} {arm} cp={cp} {j.type} q {q0:.3f}->{q1:.3f} (target {q0 + dq_total:.3f}, "
                  f"planned {frac_planned:.2f} of the way) moved {moved:.2f}", flush=True)
        if moved < 0.6 and again > 0 and abs(q1 - q0) < abs(dq_total):
            # it slipped out of the hand part-way: grab it again and keep going
            res2 = self._operate(fx, target_frac, lang, again=again - 1)
            if res2 is not None:
                q2 = float(fx.get_qpos()[qi])
                moved = (q2 - q0) / dq_total if abs(dq_total) > 1e-6 else 0.0
                return self._res(moved > 0.6, lang, arm, moved=float(moved), fixture=fx.get_name(), regrasped=True)
        return self._res(moved > 0.6, lang, arm, moved=float(moved), fixture=fx.get_name())

    def _press_lid_down(self, fx, arm, dq):
        """The grasp got the lid most of the way down: press its top edge the
        rest of the way with the closed hand from above."""
        T0 = fx.get_contact_point(0, "matrix")
        j, _ = self._fixture_joint(fx)
        J = j.get_global_pose().to_transformation_matrix()
        a, o = J[:3, 0], J[:3, 3]
        q = TOPDOWN[str(arm)]
        # pushing down on the edge must turn the lid shut (not when the edge is past the hinge)
        if float(np.cross(T0[:3, 3] - o, [0, 0, -1.0]) @ a) * dq <= 0:
            return False
        def tip_on(T):      # fingertips 1 cm in from the edge, on the lid's outer face
            E = T[:3, 3]
            u = (E - o) - ((E - o) @ a) * a
            u /= np.linalg.norm(u)
            return list(E - 0.01 * u + [0, 0, TIP + 0.004]) + list(q)
        legs = [(list(np.asarray(tip_on(T0)[:3]) + [0, 0, 0.05]) + list(q), None), (tip_on(T0), CONS)]
        for k in range(1, 4):
            legs.append((tip_on(self._moved_by(fx, T0, dq * k / 3)), None))
        # getting above the edge: straight up out of the lid's box first, then
        # across high above it, so the hand doesn't sweep the lid back open
        ee = self._ee(arm)
        top = float(self._aabb(fx)[1][2]) + 0.06 + TIP
        up = list(ee[:3]) + list(ee[3:])
        up[2] = max(up[2], top)
        across = list(np.asarray(legs[0][0][:3]) * [1, 1, 0] + [0, 0, max(top, legs[0][0][2])]) + list(q)
        first = self._plan_chain(arm, [(up, None), (across, None), legs[0]], flips=())
        if first is None:
            return False
        first = [first[0], first[1], first[2]]
        plans = None
        for m in range(len(legs), 2, -1):
            plans = self._plan_chain(arm, legs[1:m], q0=self._q_after(arm, first), flips=())
            if plans is not None:
                break
        if plans is None:
            return False
        self._sync(arm)
        self._run([("move", arm, f_) for f_ in first] + [("grip", arm, 0.0)] + [("move", arm, pl) for pl in plans])
        ee = self._ee(arm)
        self._go(arm, [(list(ee[:3] + [0, 0, 0.06]) + list(ee[3:]), None)], grip_after=1.0)
        return True

    def _handle_grasps(self, fx, arm):
        """(label, pre, pose) grasps of the fixture's handle: the annotated
        contact points (each also with the hand turned about its own axis),
        and for a lid also a grasp of its top edge from above, fingers across
        the lid, which stays reachable when the lid stands open."""
        out = []
        fq = t3d.quaternions.axangle2quat([1, 0, 0], np.pi)
        for cp in FIXTURES[fx.play_name]["cps"]:
            pre = self.get_grasp_pose(fx, arm, contact_point_id=cp, pre_dis=0.08)
            pose = self.get_grasp_pose(fx, arm, contact_point_id=cp, pre_dis=0.0)
            if pre is None or pose is None:
                continue
            # somewhere along the handle, not always its middle
            Rz = t3d.quaternions.quat2mat(pose[3:])[:, 2]
            for sh in self.prng.permutation([0.0, 0.02, -0.02]) if fx.play_name in ("036_cabinet", "044_microwave") else [0.0]:
                pre_s = list(np.asarray(pre[:3]) + sh * Rz) + list(pre[3:])
                pose_s = list(np.asarray(pose[:3]) + sh * Rz) + list(pose[3:])
                out.append((cp, pre_s, pose_s))
                out.append((cp, list(pre_s[:3]) + list(t3d.quaternions.qmult(pre_s[3:], fq)),
                            list(pose_s[:3]) + list(t3d.quaternions.qmult(pose_s[3:], fq))))
        if fx.play_name == "015_laptop":
            j, _ = self._fixture_joint(fx)
            J = j.get_global_pose().to_transformation_matrix()
            a, o = J[:3, 0], J[:3, 3]
            E = np.asarray(fx.get_contact_point(0, "list")[:3])
            v = E - o
            u = v - (v @ a) * a
            u /= np.linalg.norm(u)                      # hinge -> lid edge, in the lid
            for sgn in (1.0, -1.0):
                x = -u                                  # hand points down the lid
                y = sgn * np.cross(a, u)                # fingers close across it
                R = np.stack([x, y, np.cross(x, y)], 1)
                q = list(t3d.quaternions.mat2quat(R))
                c = E - 0.02 * u                        # 2 cm down from the edge
                ee = c - 0.12 * x
                out.append(("edge", list(ee - 0.08 * x) + q, list(ee) + q))
        return out

    def _drawer_geom(self, fx):
        """Drawer front (handle front y), inside of its front wall, top, x middle,
        and the unit vector it slides out along (towards the robot)."""
        j, _ = self._fixture_joint(fx)
        link = fx.link_dict[fx.config["contact_points"][0]["base"]]
        dlo, dhi = (np.asarray(x) for x in link.compute_global_aabb_tight())
        J = j.get_global_pose().to_transformation_matrix()
        lo, hi = fx.get_qlimits()[self._fixture_joint(fx)[1]]
        out = J[:3, 0] * (1 if hi > lo else -1)
        out = out if out[1] < 0 else -out
        return dict(front=float(dlo[1]), inner=float(dlo[1]) + 0.05, top=float(dhi[2]), x=0.5 * float(dlo[0] + dhi[0]),
                    zmid=0.5 * float(dlo[2] + dhi[2]), out=out, lo=dlo, hi=dhi)

    def _slide_hand(self, fx, arm, tip0, q, dq, approach, grip, lang, label):
        """Fingertips at tip0 (world), hand orientation q, gripper value grip;
        come in from tip0 + approach, then ride along with the drawer by dq;
        let go and back off. One smooth motion along the slide."""
        x = _hand_x(q)
        ee = lambda tip: list(np.asarray(tip) - TIP * x) + list(q)
        T0 = np.eye(4)
        T0[:3, 3] = tip0
        n = int(np.clip(np.ceil(abs(dq) / 0.025), 2, 8))
        path = [ee(self._moved_by(fx, T0, dq * k / n)[:3, 3]) for k in range(1, n + 1)]
        legs = [(ee(np.asarray(tip0) + approach), None), (ee(tip0), CONS)]
        first = self._plan_chain(arm, legs, flips=())
        if os.environ.get("PLAY_DEBUG"):
            print(f"   {label} try {arm} tip {np.round(tip0, 3)} approach planned={first is not None} {'' if first else self._why}", flush=True)
        if first is None:
            return None
        moves = None
        for m in range(n, max(1, n // 2) - 1, -1):
            moves = self._plan_chain(arm, [(w, None) for w in path[:m]], q0=self._q_after(arm, first), flips=())
            if moves is not None:
                break
        if moves is None:
            return None
        j, qi = self._fixture_joint(fx)
        q0 = float(fx.get_qpos()[qi])
        home = self._home_plan(arm.opposite)
        self._sync(arm)
        self._run([("both", (arm, None, grip), (arm.opposite, home, None)) if home is not None else ("grip", arm, grip),
                   ("move", arm, first[0]), ("move", arm, first[1]), ("move", arm, self._merge_plans(moves)), ("grip", arm, 1.0)])
        e2 = self._ee(arm)
        self._go(arm, [(list(e2[:3] + 0.6 * np.asarray(approach)) + list(e2[3:]), None)])
        q1 = float(fx.get_qpos()[qi])
        moved = (q1 - q0) / dq if abs(dq) > 1e-6 else 0.0
        if os.environ.get("PLAY_DEBUG"):
            print(f"   {label} {fx.get_name()} {arm} q {q0:.3f}->{q1:.3f} (wanted {q0 + dq:.3f}) moved {moved:.2f}", flush=True)
        self.last_grasp = dict(contact_point=label.lower(), arm=str(arm))
        return self._res(moved > 0.6, lang, arm, moved=float(moved), fixture=fx.get_name(), strategy=label.lower())

    def _hook_drawer(self, fx, target_frac, lang):
        """A drawer that's open a little: fingers in behind its front wall from
        above, and slide it out the rest of the way."""
        g = self._drawer_geom(fx)
        j, qi = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[qi]
        dq = lo + (hi - lo) * target_frac - float(fx.get_qpos()[qi])
        body_front = min(np.asarray(L.compute_global_aabb_tight()[0])[1] for L in fx.actor.get_links()
                         if L.get_collision_shapes() and L.get_name() != fx.config["contact_points"][0]["base"])
        if os.environ.get("PLAY_DEBUG"):
            print(f"   HOOK geom front {g['front']:.3f} inner {g['inner']:.3f} body_front {body_front:.3f} top {g['top']:.3f} out {np.round(g['out'], 2)} dq {dq:.3f}", flush=True)
        if g["inner"] + 0.012 > body_front - 0.01:
            return None                            # not open far enough to get the fingers in
        dy, dz = float(os.environ.get("HOOK_DY", 0.065)), float(os.environ.get("HOOK_DZ", 0.05))
        tip = np.array([g["x"] + float(self.prng.uniform(-0.04, 0.04)), g["front"] + dy, g["top"] - dz])
        p = fx.get_pose().p
        def pitched(deg, roll):
            # pointing forward-down into the gap behind the drawer front (lower wrist than straight down)
            a = np.deg2rad(deg)
            x = np.array([0.0, np.cos(a), -np.sin(a)])
            y = np.array([1.0, 0.0, 0.0]) * (1 if roll == 0 else -1)
            return list(t3d.quaternions.mat2quat(np.stack([x, y, np.cross(x, y)], 1)))
        for arm in (self._arm_for(p), self._arm_for(p).opposite):
            for q in [pitched(d, r) for d in (45, 60) for r in (0, 1)] + [TOPDOWN[str(arm)]]:
                res = self._slide_hand(fx, arm, tip, q, dq, -0.05 * _hand_x(q), 0.0, lang, "HOOK")
                if res is not None:
                    return res
        return None

    def _push_drawer(self, fx, target_frac, lang):
        """Push the drawer shut with the closed hand on its front."""
        g = self._drawer_geom(fx)
        j, qi = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[qi]
        dq = lo + (hi - lo) * target_frac - float(fx.get_qpos()[qi])
        hz = float(fx.get_contact_point(0, "list")[2])        # at handle height, like the expert's grasp
        tip = np.array([g["x"] + float(self.prng.uniform(-0.05, 0.05)), g["front"] - 0.004, hz])
        p = fx.get_pose().p
        q_fwd = [0.5, -0.5, -0.5, 0.5]           # horizontal, pointing at the drawer (as in the expert demos)
        for arm in (self._arm_for(p), self._arm_for(p).opposite):
            # the flat of the fingers against its front, hand pointing down, in from above
            for yaw in (0.0, np.pi / 2):
                qd = _rot(TOPDOWN[str(arm)], [0, 0, 1], yaw)
                tip_d = np.array([tip[0], g["front"] - 0.016, hz - 0.035])
                res = self._slide_hand(fx, arm, tip_d, qd, dq, np.array([0, 0, 0.06]), 0.0, lang, "PUSH")
                if res is not None:
                    return res
            # or fingertips first, hand level (as the expert grasps the handle)
            for q in [list(t3d.quaternions.qmult(q_fwd, t3d.quaternions.axangle2quat([1, 0, 0], r))) for r in (0.0, np.pi)]:
                res = self._slide_hand(fx, arm, tip, q, dq, -0.03 * _hand_x(q), 0.0, lang, "PUSH")
                if res is not None:
                    return res
        return None

    def _push_door(self, fx, target_frac, lang):
        """Swing a door shut by pushing on its outside near the free edge
        (behind the handle), not by grasping the handle."""
        j, qi = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[qi]
        dq = lo + (hi - lo) * target_frac - float(fx.get_qpos()[qi])
        T = fx.get_contact_point(FIXTURES[fx.play_name]["cps"][0], "matrix")
        P0 = T[:3, 3]
        t = self._moved_by(fx, T, 0.05 * np.sign(dq))[:3, 3] - P0
        t /= np.linalg.norm(t) + 1e-9                     # the way the handle moves as it shuts
        tip = P0 - 0.03 * t + [0, 0, float(self.prng.uniform(-0.03, 0.02))]
        p = fx.get_pose().p
        for arm in (self._arm_for(p), self._arm_for(p).opposite):
            cands = [_rot(TOPDOWN[str(arm)], [0, 0, 1], y) for y in (0.0, np.pi / 2)]
            x = np.array([t[0], t[1], 0.0]) / (np.linalg.norm(t[:2]) + 1e-9)          # fingertips first, level
            for y in (np.array([0, 0, 1.0]), np.array([0, 0, -1.0])):
                cands.append(list(t3d.quaternions.mat2quat(np.stack([x, y, np.cross(x, y)], 1))))
            for q in cands:
                appr = np.array([0, 0, 0.06]) if abs(_hand_x(q)[2]) > 0.7 else -0.04 * _hand_x(q)
                tip_q = tip if abs(_hand_x(q)[2]) <= 0.7 else tip - [0, 0, 0.03]
                res = self._slide_hand(fx, arm, tip_q, q, dq, appr, 0.0, lang, "PUSH")
                if res is not None:
                    return res
        return None

    def p_open(self, fx):
        word = FIXTURES[fx.play_name]["word"]
        if self._fixture_frac(fx) > (0.6 if fx.play_name == "036_cabinet" else 0.5):
            return None
        hi = {"015_laptop": (0.5, 0.65), "036_cabinet": (0.97, 1.0)}.get(fx.play_name, (0.6, 0.85))   # the handle slips ~20%   # a lid past vertical can't be closed
        frac = self._fixture_frac(fx)
        force = os.environ.get("PLAY_STRATEGY")    # tests: hook / crack / push / grasp
        if fx.play_name == "036_cabinet":
            if frac > 0.2 and (force == "hook" or (not force and self.prng.random() < 0.7)):
                res = self._hook_drawer(fx, float(self.prng.uniform(*hi)), f"slide the {word} out the rest of the way")
                if res is not None:
                    return res
            if frac < 0.2 and (force == "crack" or (not force and self.prng.random() < 0.35)):     # just crack it open
                return self._operate(fx, float(self.prng.uniform(0.62, 0.72)), f"pull the {word} open a little")
        return self._operate(fx, float(self.prng.uniform(*hi)), f"open the {word}")

    def p_close(self, fx):
        word = FIXTURES[fx.play_name]["word"]
        if self._fixture_frac(fx) < 0.3:
            return None
        force = os.environ.get("PLAY_STRATEGY")
        if fx.play_name == "044_microwave" and (force == "push" or (not force and self.prng.random() < 0.5)):
            res = self._push_door(fx, 0.0, f"push the {word} shut")
            if res is not None and not res["success"] and self._fixture_frac(fx) > 0.25:
                # it only swung part-way: take the handle and finish
                j, qi = self._fixture_joint(fx)
                q_before = res.get("moved", 0.0)
                res2 = self._operate(fx, float(self.prng.uniform(0.0, 0.05)), res["language"], again=0)
                if res2 is not None:
                    res2.update(strategy="push+grasp", success=bool(self._fixture_frac(fx) < 0.15))
                    return res2
            if res is not None:
                return res
        if fx.play_name == "036_cabinet" and (force == "push" or (not force and self.prng.random() < 0.6)):
            res = self._push_drawer(fx, 0.0, f"push the {word} shut")
            if res is not None:
                return res
        return self._operate(fx, float(self.prng.uniform(0.0, 0.08)), f"close the {word}")

    def _drawer_slot(self, fx):
        """The open part of the drawer: (center xy, floor z, wall top z, size xy)."""
        j, _ = self._fixture_joint(fx)
        link = fx.link_dict[fx.config["contact_points"][0]["base"]]
        dlo, dhi = (np.asarray(x) for x in link.compute_global_aabb_tight())
        body_front = min(np.asarray(L.compute_global_aabb_tight()[0])[1] for L in fx.actor.get_links()
                         if L.get_collision_shapes() and L.get_name() != link.get_name())
        y0, y1 = dlo[1] + 0.025, min(dhi[1], body_front) - 0.025
        x0, x1 = dlo[0] + 0.025, dhi[0] - 0.025
        floor = float(fx.get_functional_point(0, "list")[2])
        return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)]), floor, float(dhi[2]), np.array([x1 - x0, y1 - y0])

    def p_stow(self, actor, fx):
        """Put something into the open drawer (RoboTwin's put_object_cabinet)."""
        if fx.play_name != "036_cabinet" or self._fixture_frac(fx) < 0.6:
            return None
        c, floor, top, size = self._drawer_slot(fx)
        lo, hi = self._aabb(actor)
        if os.environ.get("PLAY_DEBUG"):
            print(f"   STOW {actor.get_name()} size {np.round(hi - lo, 3)} slot {np.round(size, 3)} frac {self._fixture_frac(fx):.2f}", flush=True)
        if np.any(np.sort((hi - lo)[:2]) > np.sort(size) - 0.005) or hi[2] - lo[2] > 0.16:
            self._why = "doesn't fit"
            return None    # doesn't fit in the open part
        lang = f"put the {self._name(actor)} in the drawer"
        link = fx.link_dict[fx.config["contact_points"][0]["base"]].get_name()
        self._ignore_links = {(fx, link)}
        try:
            pre = top - floor + 0.02          # its bottom just clears the drawer walls
            target = [c[0], c[1], floor + 0.004]
            st, arm = self._start(actor, lang, lift=top - TABLE_Z + 0.025, then=target, then_dz=(pre,))
            if st is None:
                return None
            if not st:
                return self._res(False, lang, arm)
            # turned so its long side runs along the drawer's long side
            R = t3d.quaternions.quat2mat(np.asarray(actor.get_pose().q, float))
            ext = np.asarray(actor.config["extents"], float) * np.asarray(actor.config["scale"], float)
            horiz = [k for k in range(3) if abs(R[2, k]) < 0.5]
            k = max(horiz, key=lambda k: ext[k]) if horiz else 0
            long_ax = R[:, k]
            slot_ax = np.array([1.0, 0, 0]) if size[0] >= size[1] else np.array([0, 1.0, 0])
            yaw = (np.arctan2(slot_ax[1], slot_ax[0]) - np.arctan2(long_ax[1], long_ax[0]) + np.pi / 2) % np.pi - np.pi / 2
            lo2, hi2 = self._aabb(actor)
            fits_as_is = (hi2 - lo2)[0] <= size[0] - 0.005 and (hi2 - lo2)[1] <= size[1] - 0.005
            placed = self._place(actor, arm, target, pre, 0.02, over_extra=0.0, yaw=None if fits_as_is else yaw)
            dl, dh = (np.asarray(x) for x in fx.link_dict[link].compute_global_aabb_tight())
            xy, bot = self._xy(actor), self._aabb(actor)[0][2]
            ok = bool(np.all(xy > dl[:2]) and np.all(xy < dh[:2]) and bot < top - 0.02)
            if os.environ.get("PLAY_DEBUG"):
                print(f"   STOWED xy {np.round(xy, 3)} drawer {np.round(dl[:2], 3)}..{np.round(dh[:2], 3)} bottom {bot:.3f} top {top:.3f}", flush=True)
            return self._finish(actor, arm, placed, ok, lang, fixture=fx.get_name())
        finally:
            self._ignore_links = set()

    def p_put_cabinet(self, actor, fx):
        """RoboTwin's put_object_cabinet: one hand pulls the drawer open and
        holds it, the other puts something in, then lets go."""
        if fx.play_name != "036_cabinet" or self._fixture_frac(fx) > 0.2:
            return None
        c, floor, top, size = self._drawer_slot(fx)
        lo, hi = self._aabb(actor)
        if np.any(np.sort((hi - lo)[:2]) > np.array([0.11, 0.17])) or hi[2] - lo[2] > 0.16:
            return None
        obj_arm = self._arm_for(self._xy(actor))
        hold_arm = obj_arm.opposite
        lang = f"open the drawer and put the {self._name(actor)} in it"
        r1 = self._operate(fx, float(self.prng.uniform(0.85, 0.95)), lang, again=0, arms=(hold_arm,), hold=True)
        if r1 is None:
            return None
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PUTCAB pulled moved={r1.get('moved', 0):.2f} drawer {self._fixture_frac(fx):.2f} slot {np.round(self._drawer_slot(fx)[3], 3)}", flush=True)
        def let_go():
            self._run([("grip", hold_arm, 1.0)])
            ee = self._ee(hold_arm)
            self._go(hold_arm, [(list(ee[:3] - 0.08 * _hand_x(ee[3:])) + list(ee[3:]), None)])
            self._home(hold_arm)
        if not r1["success"]:
            let_go()
            return self._res(False, lang, f"{hold_arm}+{obj_arm}")
        c, floor, top, size = self._drawer_slot(fx)
        link = fx.link_dict[fx.config["contact_points"][0]["base"]].get_name()
        self._ignore_links = {(fx, link)}
        try:
            pre = top - floor + 0.008          # its bottom just over the drawer's walls (the reach is tight here)
            target = [c[0], c[1], floor + 0.004]
            self.last_grasp = None
            ok, _ = self._pick(actor, obj_arm, lift=top - TABLE_Z + 0.025, then=target, then_dz=(pre,), arms=(obj_arm,),
                               home_other=False)
            placed, inside = False, False
            if ok:
                placed = self._place(actor, obj_arm, target, pre, 0.02, over_extra=0.0)
                dl, dh = (np.asarray(x) for x in fx.link_dict[link].compute_global_aabb_tight())
                xy, bot = self._xy(actor), self._aabb(actor)[0][2]
                inside = bool(np.all(xy > dl[:2]) and np.all(xy < dh[:2]) and bot < top - 0.02)
                if not placed:
                    self._put_down_here(actor, obj_arm)
            self._home(obj_arm)
        finally:
            self._ignore_links = set()
        let_go()
        if os.environ.get("PLAY_DEBUG"):
            print(f"   PUTCAB picked={ok} placed={placed} inside={inside} drawer {self._fixture_frac(fx):.2f}", flush=True)
        if ok is None and not placed:
            return self._res(False, lang, f"{hold_arm}+{obj_arm}", opened=True)
        return self._res(inside, lang, f"{hold_arm}+{obj_arm}", fixture=fx.get_name())

    def p_toggle(self, fx):
        """Flip the switch with the closed hand, like RoboTwin's turn_switch:
        press the annotated point; if the rocker doesn't go over, press the
        other half of it (the point mirrored across the rocker's axis)."""
        j, qi = self._fixture_joint(fx)
        lo, hi = fx.get_qlimits()[qi]
        q0 = float(fx.get_qpos()[qi])
        p = fx.get_pose().p
        J = j.get_global_pose().to_transformation_matrix()
        axis, o = J[:3, 0], J[:3, 3]
        lang = "turn the switch " + ("off" if q0 > 0.5 * (lo + hi) else "on")
        face = -fx.get_pose().to_transformation_matrix()[:3, 0]       # RoboTwin's turn_switch arm rule
        first = ArmTag("right" if face[0] > 0 else "left")
        for arm, back in ((first, 0.06), (first, 0.03), (first.opposite, 0.06)):
            pre = self.get_grasp_pose(fx, arm, contact_point_id=0, pre_dis=back)
            pose = self.get_grasp_pose(fx, arm, contact_point_id=0, pre_dis=-0.015)
            if pre is None or pose is None:
                continue
            if errors.has(getattr(self, "err", None), "press_miss"):      # beside the rocker
                d = np.asarray(self.err["offset_xy"], dtype=float)
                off = np.r_[d / (np.linalg.norm(d) + 1e-9) * (0.035 + np.linalg.norm(d)), 0.0]
                pre, pose = list(np.asarray(pre[:3]) + off) + list(pre[3:]), list(np.asarray(pose[:3]) + off) + list(pose[3:])
            n = np.cross(axis, _hand_x(pose[3:]))
            n /= np.linalg.norm(n) + 1e-9
            mirror = lambda x: list(np.asarray(x[:3]) - 2 * float((np.asarray(x[:3]) - o) @ n) * n) + list(x[3:])
            pressed = False
            presses = ((pre, pose),) if errors.has(getattr(self, "err", None), "press_miss") else ((pre, pose), (mirror(pre), mirror(pose)))
            for P_, Q_ in presses:
                plans = self._plan_chain(arm, [(P_, None), (Q_, CONS), (P_, CONS)])
                if plans is None:
                    continue
                home = self._home_plan(arm.opposite) if not pressed else None
                self._sync(arm)
                self._run(([("both", (arm, None, 0.0), (arm.opposite, home, None))] if home is not None else [("grip", arm, 0.0)]) +
                          [("move", arm, plans[0]), ("move", arm, plans[1]), ("move", arm, plans[2])])
                pressed = True
                if abs(float(fx.get_qpos()[qi]) - q0) > 0.5 * (hi - lo):
                    break
            if not pressed:
                continue
            self._run([("grip", arm, 1.0)])
            q1 = float(fx.get_qpos()[qi])
            if os.environ.get("PLAY_DEBUG"):
                print(f"   TOGGLE {fx.get_name()} {arm} q {q0:.3f}->{q1:.3f} range {lo:.2f}..{hi:.2f}", flush=True)
            self.last_grasp = None
            return self._res(abs(q1 - q0) > 0.5 * (hi - lo), lang, arm)
        return None

    def p_press(self, actor):
        """Press down on top of it with the closed hand (a click)."""
        lo, hi = self._aabb(actor)
        c = 0.5 * (lo + hi)
        if errors.has(getattr(self, "err", None), "press_miss"):
            # comes down beside it: the offset is measured from the object's edge, so the hand misses it
            d = np.asarray(self.err["offset_xy"], dtype=float)
            u = d / (np.linalg.norm(d) + 1e-9)
            c = c + np.r_[u * (0.5 * float(np.abs((hi - lo)[:2]) @ np.abs(u)) + np.linalg.norm(d)), 0.0]
            hi = np.r_[hi[:2], TABLE_Z + 0.01]
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
        # like RoboTwin's shake_bottle, gentler: tip it back and forth about the
        # hand's side axis (a tilt, not a spin) with a bob, twice, then level it
        ee = self._ee(arm)
        R = t3d.quaternions.quat2mat(ee[3:])
        a0 = float(self.prng.uniform(0.3, 0.5))
        shaken = False
        # about the hand's side axis or its other horizontal axis, smaller if the wrist can't
        for ax_i, a in [(i, a0 * f) for f in (1.0, 0.6) for i in (1, 2)]:
            ax = R[:, ax_i] - R[2, ax_i] * np.array([0, 0, 1.0])     # horizontal part: a tilt, not a spin
            if np.linalg.norm(ax) < 0.5:
                continue
            ax /= np.linalg.norm(ax)
            tilt = lambda ang: list(t3d.quaternions.qmult(t3d.quaternions.axangle2quat(ax, ang), ee[3:]))
            legs = []
            for _ in range(2):
                legs.append((list(ee[:3] + [0, 0, 0.04]) + tilt(a), None))
                legs.append((list(ee[:3] - [0, 0, 0.02]) + tilt(-a), None))
            legs.append((list(ee), None))
            if self._go(arm, legs):
                shaken = True
                break
        xy = self._xy(actor)
        placed = self._place(actor, arm, [xy[0], xy[1], TABLE_Z], 0.05, 0.01)
        return self._finish(actor, arm, placed, shaken, lang)

    def p_knock_with(self, tool, target):
        """Pick one thing up and tap another with it (a hammer on a block, a
        stamp on a card)."""
        lang = f"tap the {self._name(target)} with the {self._name(tool)}"
        tlo, thi = self._aabb(target)
        tp = target.get_pose().p.copy()
        if errors.has(getattr(self, "err", None), "tap_miss"):
            tp[:2] += self.err["offset_xy"]
            thi = np.r_[thi[:2], TABLE_Z]   # beside it: down to the table
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
                if plans is not None and self._plan_about(arm, [(p_, None) for p_ in self._then_pose(pose, obj, [xy[0], xy[1], TABLE_Z])],
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
        pre_l, pos_l = self._down_poses(a, L, tl, 0.05, 0.01)
        pre_r, pos_r = self._down_poses(b, R_, tr, 0.05, 0.01)
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
        ok = placed and all(held.values()) and np.linalg.norm(self._xy(a) - spots["left"]) < 0.05 \
            and np.linalg.norm(self._xy(b) - spots["right"]) < 0.05
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
        clo, chi = self._aabb(cont)
        if abs(0.5 * (clo[0] + chi[0])) > 0.12:
            self._why = "off to one side: the other hand can't reach it"
            return None
        chains = {}
        home = {"left": np.array(self.robot.left_original_pose[:3]), "right": np.array(self.robot.right_original_pose[:3])}
        cands = {"left": [], "right": []}
        if len(pts) >= 2:
            # the two grasp points farthest apart; each hand takes the one nearer to it
            far = max(((i, j) for i, _ in pts for j, _ in pts if i < j),
                      key=lambda ij: np.linalg.norm(np.asarray(dict(pts)[ij[0]][:3]) - np.asarray(dict(pts)[ij[1]][:3])))
            if np.linalg.norm(np.asarray(dict(pts)[far[0]][:3]) - np.asarray(dict(pts)[far[1]][:3])) >= 0.08:
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
                        cands[str(arm)].append((i, cp, pre, pose))
        if chi[2] - clo[2] > 0.045:
            # its rim on each side, from above, fingers across the wall
            cy = 0.5 * (clo[1] + chi[1])
            for arm, xw in ((L, clo[0] + 0.006), (R_, chi[0] - 0.006)):
                P_ = np.array([xw, cy, chi[2] - 0.012])
                x, y = np.array([0, 0, -1.0]), np.array([1.0, 0, 0])
                ee = P_ - 0.12 * x
                ee[2] = max(ee[2], TABLE_Z + 0.004 + TIP)       # fingertips off the table
                for yy in (y, -y):                               # and the hand turned about its axis
                    q = list(t3d.quaternions.mat2quat(np.stack([x, yy, np.cross(x, yy)], 1)))
                    cands[str(arm)].insert(0, ("rim", P_, np.r_[ee - 0.04 * x, q], np.r_[ee, q]))   # opposite each other: first
        if not cands["left"] or not cands["right"]:
            return None
        for arm in (L, R_):
            for i, cp, pre, pose in cands[str(arm)][:4]:
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
        if os.environ.get("PLAY_DEBUG"):
            print(f"   LIFT2 lifted {lifted:.3f} both={both} grasps {chains['left'][0]}/{chains['right'][0]}", flush=True)
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
        for ang in (2.0, 1.7, 1.4, 1.1):
            for ax in ([1, 0, 0], [0, 1, 0]):
                for sgn in (1, -1):
                    tip = list(ee[:3] + [0, 0, 0.04]) + _rot(ee[3:], ax, sgn * ang)
                    # tipped over, two little shakes to get it sliding, then level again
                    shake = [(list(ee[:3] + [0, 0, 0.04]) + _rot(ee[3:], ax, sgn * (ang - d)), None) for d in (0.3, 0.0, 0.3, 0.0)]
                    legs = [(tip, None)] + shake + [(list(ee), None)]
                    if self._plan_chain(arm, legs, flips=()) is not None:
                        self._sync(arm)
                        tilted = self._go(arm, legs[:-1])
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
        xy = self._xy(cont)
        placed = self._place(cont, arm, [xy[0], xy[1], TABLE_Z], 0.06, 0.015)
        out = not any(self._inside(o, cont) for o in inside)
        res = self._finish(cont, arm, placed, True, lang)
        res.update(success=bool(tilted and out), placed=bool(placed))   # the task is the emptying
        return res

    # ---------------------------------------------------------------- loop
    def _stack_weight(self, a, b):
        """Stable stacks most of the time, a wobbly one now and then: full
        weight when b's top is at least as big as a's footprint and a isn't
        tall and narrow."""
        alo, ahi = self._aabb(a)
        blo, bhi = self._aabb(b)
        da, db = ahi - alo, bhi - blo
        stable = db[0] * db[1] >= da[0] * da[1] and da[2] <= 1.6 * min(da[0], da[1])
        return 1.0 if stable else 0.25

    def _choose(self):
        objs = [a for a in self.objects if self._on_table(a)]
        conts = [c for c in self.containers if self._on_table(c)]
        done = getattr(self, "_done", {})
        touched = getattr(self, "_touched", {})
        last = getattr(self, "_last_kind", None)
        opts = []
        nov = lambda a: 1.0 / (1 + touched.get(a.get_name(), 0))
        fx_actions = sum(done.get(k, 0) for k in ("open", "close", "toggle"))
        for f in getattr(self, "fixtures", []):
            frac = self._fixture_frac(f)
            stowed = done.get("stow", 0) > getattr(self, "_stow_mark", 0)    # something went in since it opened
            if f.play_name == "036_cabinet" and frac < 0.2 and done.get("put_cabinet", 0) < 2 and self.play_cfg.get("put_cabinet"):
                # off by default: the held drawer slides back and the drop point is out of reach (see README)
                for a in objs:
                    opts.append((1.0 / max(1, len(objs)), "put_cabinet", (a, f)))
            if f.play_name == "036_cabinet" and frac > 0.6:
                # an open drawer invites putting something in it
                for a in objs:
                    opts.append((2.5 / max(1, len(objs)), "stow", (a, f)))
            w = 1.6 * nov(f) if fx_actions < 3 else 0.0     # rarer in the scene, so likelier there; not the whole episode
            if f.play_name == "056_switch":
                if done.get("toggle", 0) < 3:
                    opts.append((w, "toggle", (f,)))
            elif frac < (0.6 if f.play_name == "036_cabinet" else 0.5):
                if done.get("open", 0) < 2:                  # not an open-close loop
                    opts.append((w * (0.25 if done.get("close", 0) else 1.0), "open", (f,)))
                    self._stow_mark = done.get("stow", 0)
            elif done.get("close", 0) < 2:
                empty_drawer = f.play_name == "036_cabinet" and not stowed and objs
                opts.append((w * (0.35 if empty_drawer else 1.0), "close", (f,)))
        for a in objs:
            w = nov(a)
            opts.append((0.7 * w, "move", (a,)))
            opts.append((0.4 * w, "push", (a,)))
            opts.append((0.35 * w, "reorient", (a,)))
            opts.append((0.6 * w, "handover", (a,)))
            opts.append((0.2 * w, "press", (a,)))
            if any(a.get_name().startswith(n) for n in SHAKEABLE):
                opts.append((0.15 * w, "shake", (a,)))
            for c in conts:
                opts.append((1.0 * w / len(conts), "put_in", (a, c)))
            for b in objs:
                if b is not a:
                    if any(b.get_name().startswith(n) for n in STACK_BASES):
                        opts.append((0.6 * w * self._stack_weight(a, b) / max(len(objs) - 1, 1), "stack", (a, b)))
                    opts.append((0.35 * w / max(len(objs) - 1, 1), "line_up", (a, b)))
                    if _is_tool(a):
                        opts.append((0.9 * w / max(len(objs) - 1, 1), "knock_with", (a, b)))
                    if self._xy(a)[0] < 0 < self._xy(b)[0]:
                        opts.append((0.8 / max(len(objs) - 1, 1), "dual_pick", (a, b)))
            for c in conts:
                if any(c.get_name().startswith(n) for n in STACK_BASES):
                    opts.append((0.25 * w / len(conts), "stack", (a, c)))
        for c in conts:
            opts.append((0.6, "lift_together", (c,)))
            if any(c.get_name().startswith(n) for n in OPEN_TOP) and any(self._inside(o, c) for o in objs):
                opts.append((1.0, "pour", (c,)))
        # variety: what has been done already is less likely, the same kind twice in a row even less
        skip = getattr(self, "_skip", set())    # couldn't be planned earlier in this episode
        opts = [(w / (1 + done.get(k, 0)) * (0.3 if k == last else 1.0), k, x) for w, k, x in opts if w > 0
                if (k, tuple(a.get_name() for a in x)) not in skip]
        w = np.array([o[0] for o in opts])
        return opts[int(self.prng.choice(len(opts), p=w / w.sum()))] if opts else None

    def play_once(self):
        cfg = self.play_cfg
        segs = []
        self._done, self._touched, self._skip, self._fails = {}, {}, set(), {}
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
            self.err = errors.sample_error(self.prng, kind, p_err=cfg.get("p_err", 0.3), severity=cfg.get("err_severity", 0.15))
            self.action_steps, self.action_len = 0, 2500
            err0 = self.err
            self._ignore = set(args)
            res = getattr(self, "p_" + kind)(*args)
            self.plan_success = True
            self._last_kind = kind
            if res is not None and err0 is not None and not res["success"] and kind in errors.PICK and self.prng.random() < 0.5:
                # after an injected failure: one clean try more (or move on)
                self.err = None
                res2 = getattr(self, "p_" + kind)(*args)
                self.plan_success = True
                if res2 is not None:
                    res2["retry_after_error"] = True
                    res = dict(res2, language=res["language"])
            self.err = None
            if res is None:
                self._skip = getattr(self, "_skip", set()) | {(kind, tuple(a.get_name() for a in args))}
                continue
            if err0 is not None:
                res["error"] = err0
            self._done[kind] = self._done.get(kind, 0) + 1
            for a in args:
                self._touched[a.get_name()] = self._touched.get(a.get_name(), 0) + 1
            res.update(primitive=kind, target=[a.get_name() for a in args], start=f0, end=self.FRAME_IDX)
            segs.append(res)
            key = (kind, tuple(a.get_name() for a in args))
            if not res["success"]:
                self._fails[key] = self._fails.get(key, 0) + 1
                if self._fails[key] >= 2:           # tried twice: leave it be for this episode
                    self._skip = getattr(self, "_skip", set()) | {key}
        self.info["info"] = {"scene": self.scene_desc, "segments": segs, "plans": self.n_plans,
                             "syncs": getattr(self, "n_syncs", 0)}
        return self.info

    def check_success(self):
        return True
