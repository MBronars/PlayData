import sys, numpy as np, json
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController
from composed_play import ComposedPlay
from compose import ResidualBank
from segments import load_all
from collect import randomize_object_yaws
from scene_gen import write_bddl, sample_scene
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
T = "/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all(); bank = ResidualBank(tasks, segs)
mode = sys.argv[1]
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
res = []
if mode == "pick":
    cats = sys.argv[3].split(",")
    for cat in cats:
        cfg = dict(scene="kitchen", fixture=None, fixture_pose=None, objects=[cat, "akita_black_bowl", "alphabet_soup"],
                   positions=[(-0.05, 0.05), (-0.12, -0.2), (-0.15, 0.22)])
        p = f"{T}/cp_{cat}.bddl"; names = write_bddl(cfg, p)
        env = OffScreenRenderEnv(bddl_file_name=p, camera_heights=128, camera_widths=128, ignore_done=True, horizon=10**6)
        ok = []
        for tr in range(6):
            env.reset(); randomize_object_yaws(env, rng, names)
            for _ in range(10): env.step(np.r_[np.zeros(6), -1])
            frames = []
            pc = PlayController(env, names, rng, max_steps=3000, record_fn=lambda o, a, c: frames.append(np.concatenate([o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]], 0)))
            pc.obs = env.env._get_observations()
            if len(sys.argv) > 4 and sys.argv[4] == "play":
                from play_tasks import PlayfulPlay
                cp = PlayfulPlay(pc, rng, bank)
            elif len(sys.argv) > 4 and sys.argv[4] == "style":
                from style_play import StylePlay
                cp = StylePlay(pc, rng, bank)
            elif len(sys.argv) > 4 and sys.argv[4] == "hp":
                from human_path_play import HumanPathPlay
                cp = HumanPathPlay(pc, rng, bank, tasks, segs)
            else:
                cp = ComposedPlay(pc, rng, bank)
            info = cp.p_pick_place(names[0])
            if info is None: print("none"); continue
            ok.append((info["grasped"], info["success"], bool(info["diag"].get("held_at_release"))))
            print("   ", cat, info["success"], {k: info["diag"].get(k) for k in ["at_close", "grasped_trace"]}, info.get("source", "")[:50])
            if not info["success"] and not globals().get("_saved"):
                globals()["_saved"] = True
                idx = np.linspace(len(frames) * 0.25, len(frames) - 1, 12).astype(int)
                Image.fromarray(np.concatenate([np.concatenate([frames[i] for i in idx[:6]], 1), np.concatenate([frames[i] for i in idx[6:]], 1)], 0)).save(f"{T}/fail_{cat}.png")
                print("    saved failure frames", info["diag"].get("slip"))
        print(cat, "grasped %d placed %d held-at-release %d / %d" % (sum(o[0] for o in ok), sum(o[1] for o in ok), sum(o[2] for o in ok), len(ok)), flush=True)
        env.close()
elif mode == "drawer":
    cfg = dict(scene="kitchen", fixture="wooden_cabinet", fixture_pose=((0.0, -0.30), np.pi), objects=["butter"], positions=[(-0.1, 0.15)])
    if len(sys.argv) > 3 and sys.argv[3] == "white":
        cfg = dict(scene="kitchen", fixture="white_cabinet", fixture_pose=((0.0, 0.30), 0.0), objects=["butter"], positions=[(-0.1, -0.15)])
    p = f"{T}/cp_drawer.bddl"; names = write_bddl(cfg, p)
    env = OffScreenRenderEnv(bddl_file_name=p, camera_heights=128, camera_widths=128, ignore_done=True, horizon=10**6)
    env.reset()
    frames = []
    pc = PlayController(env, names, rng, max_steps=5000, record_fn=lambda o, a, c: frames.append(o["agentview_image"][::-1]))
    pc.obs = env.env._get_observations()
    cp = ComposedPlay(pc, rng, bank)
    for lv in ["top", "middle", "bottom"]:
        dr = [d for d in pc.drawers if d["level"] == lv][0]
        for op in [True, False]:
            info = cp.p_drawer(dr, op)
            print(lv, info["primitive"], info["success"], round(float(pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]]), 3), flush=True)
    idx = np.linspace(0, len(frames) - 1, 12).astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in idx], 1)).save(f"{T}/cp_drawer.png")
