"""Planning-only robot configs for the play task.

RoboTwin's aloha-agilex URDF gives every arm joint placeholder limits of
+-10 rad, so cuRobo freely winds the wrist to poses the released expert data
never uses. This writes a copy of the URDF whose joint limits are the expert
data's per-joint range (expert_joint_ranges_<robot>.json) plus a margin, and
planner ymls pointing at it. The play task plans with these; the simulated
robot itself is unchanged.

    python env/make_play_planner_cfg.py [--robot aloha-agilex] [--margin 0.15]
"""
import argparse
import json
import os
import re

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/user_data/mbronars/packages/RoboTwin")
HERE = os.path.dirname(os.path.abspath(__file__))

PREFIX = {"aloha-agilex": ("fl", "fr")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="aloha-agilex")
    p.add_argument("--margin", type=float, default=0.15)
    a = p.parse_args()
    ranges = json.load(open(os.path.join(HERE, "..", f"expert_joint_ranges_{a.robot}.json")))
    emb = os.path.join(ROBOTWIN_ROOT, "assets", "embodiments", a.robot)
    cfg = open(os.path.join(emb, "config.yml")).read()
    urdf_rel = re.search(r'urdf_path:\s*"([^"]+)"', cfg).group(1)
    urdf_path = os.path.normpath(os.path.join(emb, urdf_rel))
    urdf = open(urdf_path).read()
    n = 0
    for side, prefix in zip(("left", "right"), PREFIX[a.robot]):
        for j in range(6):
            name = f"{prefix}_joint{j + 1}"
            lo, hi = ranges[side]["lo"][j] - a.margin, ranges[side]["hi"][j] + a.margin
            pat = re.compile(rf'(<joint name="{name}" type="revolute">.*?<limit )lower="[^"]*" upper="[^"]*"', re.S)
            urdf, k = pat.subn(rf'\g<1>lower="{lo:.4f}" upper="{hi:.4f}"', urdf, count=1)
            n += k
    assert n == 12, f"only {n} joint limits replaced"
    play_urdf = urdf_path.replace(".urdf", "_play.urdf")
    open(play_urdf, "w").write(urdf)
    for side in ("left", "right"):
        yml = open(os.path.join(emb, f"curobo_{side}.yml")).read()
        assert os.path.basename(urdf_path) in yml
        yml = yml.replace(os.path.basename(urdf_path), os.path.basename(play_urdf))
        open(os.path.join(emb, f"play_curobo_{side}.yml"), "w").write(yml)
    print("wrote", play_urdf, "and", os.path.join(emb, "play_curobo_{left,right}.yml"), f"(margin {a.margin} rad)")


if __name__ == "__main__":
    main()
