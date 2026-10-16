"""Robot base pose per LIBERO problem class (needed to express human free-space
motion in the robot's own frame)."""
import glob, json, os, re
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_bases.json")
classes = {}
for f in glob.glob(os.path.join(get_libero_path("bddl_files"), "*", "*.bddl")):
    m = re.search(r"\(problem\s+(\S+)\)", open(f).read())
    if m and m.group(1) not in classes:
        classes[m.group(1)] = f
res = {}
for cls, f in classes.items():
    try:
        env = OffScreenRenderEnv(bddl_file_name=f, camera_heights=32, camera_widths=32)
        env.reset()
        e = env.env
        res[cls.lower()] = dict(base=np.array(e.robots[0].base_pos).round(4).tolist(),
                                ws=np.array(e.workspace_offset).round(4).tolist())
        print(cls, res[cls.lower()], flush=True)
        env.close()
    except Exception as ex:
        print("fail", cls, ex)
json.dump(res, open(OUT, "w"), indent=1)
