"""Random LIBERO scene generation: writes a BDDL file with a random table, an
optional articulated fixture, and a random set of graspable objects."""
import os
import numpy as np

# Small, graspable objects (all fit the Panda gripper from above).
# kitchenware that comes along with a stove
STOVEWARE = ["moka_pot", "chefmate_8_frypan"]
GRASPABLE = [
    "alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cookies",
    "cream_cheese", "ketchup", "macaroni_and_cheese", "milk", "orange_juice",
    "popcorn", "salad_dressing", "tomato_sauce", "akita_black_bowl",
    "white_bowl", "glazed_rim_porcelain_ramekin", "red_coffee_mug",
    "porcelain_mug", "white_yellow_mug", "black_book", "yellow_book",
]
# Larger objects that serve as place targets (containers / surfaces).
RECEPTACLES = ["plate", "basket", "wooden_tray", "bowl_drainer"]

SCENES = {
    # table fixture name, region x/y bounds (table frame), allowed fixtures, style xml
    "kitchen": dict(table="kitchen_table", problem="LIBERO_Kitchen_Tabletop_Manipulation",
                    fixtures=[None, "wooden_cabinet", "white_cabinet", "flat_stove", "microwave"]),
    "living_room": dict(table="living_room_table", problem="LIBERO_Living_Room_Tabletop_Manipulation",
                        fixtures=[None]),
    "study": dict(table="study_table", problem="LIBERO_Study_Tabletop_Manipulation",
                  fixtures=[None, "wooden_cabinet", "white_cabinet"]),
    # LIBERO's floor arena with a fixed small object set (adaptive_ttc's custom calibration scene, without a plate):
    # three graspables and a basket spawned anywhere in the reachable workspace
    "floor": dict(table="floor", problem="LIBERO_Floor_Manipulation", fixtures=[None],
                  objects=["alphabet_soup", "cream_cheese", "milk", "basket"]),
}

# Object area in table frame (robot sits at -x looking toward +x): broad, only
# limited by comfortable reach (play doesn't need to mirror demo layouts).
OBJ_X = (-0.20, 0.12)
OBJ_Y = (-0.25, 0.25)
# robot base in the table (workspace) frame, and how far from it objects may spawn
BASE_XY = {"kitchen": (-0.66, 0.0), "living_room": (-0.51, 0.0), "study": (-0.55, 0.0), "floor": (-0.6, 0.0)}
REACH_MAX = 0.62
OBJ_XY = {"kitchen": ((-0.22, 0.10), (-0.28, 0.28)),
          "living_room": ((-0.25, 0.12), (-0.28, 0.25)),
          "study": ((-0.08, 0.18), (-0.28, 0.22)),
          "floor": ((-0.28, 0.08), (-0.28, 0.28))}      # the calibration scene's spawn region
# Comfortable distance band from the robot base for objects and destinations. The floor scene uses the whole region
# pi0.5 is evaluated on (objects 0.32-0.74 m from the base) instead of the table scenes' top-down sweet spot.
REACH_BAND = {"floor": (0.33, 0.70)}
REACH = list(REACH_BAND.get(None, (0.47, 0.62)))     # [lo, hi] of the current episode (set_reach)


def set_reach(scene):
    REACH[:] = REACH_BAND.get(scene, (0.47, 0.62))
# Fixtures go to the side/back so they don't eat the whole workspace.
# LIBERO fixture placements (table frame); human clips transfer with translation only.
_PI = np.pi
FIXTURE_POSES = {
    "wooden_cabinet": [((0.0, -0.30), _PI), ((0.03, -0.24), _PI), ((0.03, -0.27), np.radians(155))],
    "white_cabinet": [((0.0, 0.30), 0.0)],
    # (LIBERO also uses x=-0.41, but there the knob is 25 cm from the robot's
    # base at shoulder height: unreachable)
    "flat_stove": [((-0.2, -0.2), 0.0), ((-0.2, 0.2), 0.0), ((-0.2, 0.3), 0.0)],
    # (LIBERO also has it at (0.0, -0.26) facing the other way; there the hinge
    # is on the far side and an opened door can't be reached to shut it again)
    "microwave": [((0.0, 0.34), 0.0)],
}
FIXTURE_YKEEPOUT = 0.17  # objects stay this far in y from a side fixture


def _region(name, target, x, y, half, yaw):
    return f"""      ({name}
          (:target {target})
          (:ranges (({x-half:.4f} {y-half:.4f} {x+half:.4f} {y+half:.4f})))
          (:yaw_rotation (({yaw[0]:.4f} {yaw[1]:.4f})))
      )
"""


def sample_scene(rng, n_objects=None, scene=None):
    scene = scene or rng.choice([k for k in SCENES if "objects" not in SCENES[k]])   # the table scenes
    spec = SCENES[scene]
    fixture = spec["fixtures"][rng.integers(len(spec["fixtures"]))]
    if "objects" in spec:
        objs = list(spec["objects"])
    else:
        n_obj = n_objects or int(rng.integers(3, 7))
        n_rec = int(rng.integers(0, 2)) if n_obj >= 3 else 0
        objs = list(rng.choice(GRASPABLE, size=n_obj - n_rec, replace=False))
        objs += list(rng.choice(RECEPTACLES, size=n_rec, replace=False))
    if fixture == "flat_stove" and rng.random() < 0.85:
        # something to cook with
        objs = ["moka_pot"] + objs[:-1]

    fixture_pose = None
    if fixture is not None:
        fixture_pose = FIXTURE_POSES[fixture][rng.integers(len(FIXTURE_POSES[fixture]))]

    # Rejection-sample object centers with min spacing.
    placed, kept = [], []
    lo, hi = REACH_BAND.get(scene, (0.47, REACH_MAX))
    for o in objs:
        r = 0.09 if o in RECEPTACLES else 0.06
        for _ in range(200):
            xr, yr = OBJ_XY.get(scene, (OBJ_X, OBJ_Y))
            x, y = rng.uniform(*xr), rng.uniform(*yr)
            if fixture_pose is not None:
                fx, fy = fixture_pose[0]
                if fixture == "flat_stove":
                    if abs(y - fy) < 0.12 and abs(x - fx) < 0.12:
                        continue
                elif abs(y - fy) < FIXTURE_YKEEPOUT:
                    continue
            bx, by = BASE_XY[scene]
            if not (lo < np.hypot(x - bx, y - by) < hi):
                continue  # beyond a comfortable top-down reach (or under the robot)
            if all(np.hypot(x - px, y - py) > r + pr for px, py, pr in placed):
                placed.append((x, y, r))
                kept.append(o)
                break
    return dict(scene=scene, fixture=fixture, fixture_pose=fixture_pose,
                objects=kept, positions=[(p[0], p[1]) for p in placed])


def write_bddl(cfg, path):
    spec = SCENES[cfg["scene"]]
    table = spec["table"]
    regions, inits, fixtures = "", [], [f"    {table} - {table}"]
    counts = {}
    obj_lines = {}
    names = []
    for cat, (x, y) in zip(cfg["objects"], cfg["positions"]):
        counts[cat] = counts.get(cat, 0) + 1
        inst = f"{cat}_{counts[cat]}"
        names.append(inst)
        obj_lines.setdefault(cat, []).append(inst)
        rname = f"{inst}_init_region"
        yaw = (0.0, 2 * np.pi) if cat not in RECEPTACLES else (0.0, 0.0)
        regions += _region(rname, table, x, y, 0.01, yaw)
        inits.append(f"    (On {inst} {table}_{rname})")
    if cfg["fixture"] is not None:
        f = cfg["fixture"]
        (fx, fy), fyaw = cfg["fixture_pose"]
        regions += _region(f"{f}_init_region", table, fx, fy, 0.005, (fyaw, fyaw))
        fixtures.append(f"    {f}_1 - {f}")
        inits.append(f"    (On {f}_1 {table}_{f}_init_region)")
        if "cabinet" in f:
            for r in ["top_region", "middle_region", "bottom_region"]:
                regions += f"      ({r}\n          (:target {f}_1)\n      )\n"
    objects = "\n".join(f"    {' '.join(v)} - {k}" for k, v in obj_lines.items())
    text = f"""(define (problem {spec['problem']})
  (:domain robosuite)
  (:language play)
    (:regions
{regions}    )

  (:fixtures
{chr(10).join(fixtures)}
  )

  (:objects
{objects}
  )

  (:obj_of_interest
    {names[0]}
  )

  (:init
{chr(10).join(inits)}
  )

  (:goal
    (And (On {names[0]} {table}_{names[0]}_init_region))
  )

)
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return names
