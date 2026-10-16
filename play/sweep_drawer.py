import sys, numpy as np, itertools
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController
from composed_play import ComposedPlay
from compose import ResidualBank
from segments import load_all
from scene_gen import write_bddl
from libero.libero.envs import OffScreenRenderEnv
T = "/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all(); bank = ResidualBank(tasks, segs, n=100)
cab = sys.argv[1]; strat = sys.argv[2]
cfg = dict(scene="kitchen", fixture="wooden_cabinet", fixture_pose=((0.0, -0.30), np.pi), objects=["butter"], positions=[(-0.1, 0.15)]) if cab == "wooden" else \
      dict(scene="kitchen", fixture="white_cabinet", fixture_pose=((0.0, 0.30), 0.0), objects=["butter"], positions=[(-0.1, -0.15)])
p = f"{T}/sw_{cab}.bddl"; names = write_bddl(cfg, p)
env = OffScreenRenderEnv(bddl_file_name=p, camera_heights=32, camera_widths=32, ignore_done=True, horizon=10**6)
rng = np.random.default_rng(0)
outs = [-0.006, 0.0, 0.006, 0.012]
zs = [-0.006, 0.0, 0.006]
tilt = float(sys.argv[3]) if len(sys.argv) > 3 else 55
levels = ["top", "middle", "bottom"]
for lv in levels:
    row = []
    for o, z in itertools.product(outs, zs):
        env.reset()
        for _ in range(5): env.step(np.r_[np.zeros(6), -1])
        pc = PlayController(env, names, rng, max_steps=4000); pc.obs = env.env._get_observations()
        cp = ComposedPlay(pc, rng, bank)
        dr = [d for d in pc.drawers if d["level"] == lv][0]
        info = cp.p_drawer(dr, True, params=dict(strat=strat, out=o, z=z, tilt=np.radians(tilt)))
        q = pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]]
        row.append(f"o{o:.3f} z{z:.3f}:{q:+.2f}")
    print(cab, strat, tilt if strat == "tilt" else "", lv, " ".join(row), flush=True)
