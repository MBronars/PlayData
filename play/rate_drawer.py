import sys, numpy as np, collections
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController
from composed_play import ComposedPlay
from compose import ResidualBank
from segments import load_all
from scene_gen import write_bddl
from libero.libero.envs import OffScreenRenderEnv
T = "/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all(); bank = ResidualBank(tasks, segs, n=150)
cab = sys.argv[1]; N = int(sys.argv[2])
cfg = dict(scene="kitchen", fixture="wooden_cabinet", fixture_pose=((0.0, -0.30), np.pi), objects=["butter"], positions=[(-0.1, 0.15)]) if cab == "wooden" else \
      dict(scene="kitchen", fixture="white_cabinet", fixture_pose=((0.0, 0.30), 0.0), objects=["butter"], positions=[(-0.1, -0.15)])
p = f"{T}/rd_{cab}.bddl"; names = write_bddl(cfg, p)
env = OffScreenRenderEnv(bddl_file_name=p, camera_heights=32, camera_widths=32, ignore_done=True, horizon=10**6)
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
res = collections.defaultdict(list)
for i in range(N):
    env.reset()
    for _ in range(5): env.step(np.r_[np.zeros(6), -1])
    pc = PlayController(env, names, rng, max_steps=6000); pc.obs = env.env._get_observations()
    cp = ComposedPlay(pc, rng, bank)
    import os, json
    prm = json.loads(os.environ.get("DPARAMS", "{}"))
    levels = os.environ.get("LEVELS", "top,middle,bottom").split(",")
    for lv in levels:
        dr = [d for d in pc.drawers if d["level"] == lv][0]
        a = cp.p_drawer(dr, True, params=dict(prm) if prm else None); res[(lv, "open")].append(a["success"])
        if a["success"]:
            b = cp.p_drawer(dr, False); res[(lv, "close")].append(b["success"])
print(cab, os.environ.get("DPARAMS"), {f"{k[0]}-{k[1]}": f"{sum(v)}/{len(v)}" for k, v in res.items()}, flush=True)
