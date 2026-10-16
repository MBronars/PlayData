import sys, numpy as np
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
import retarget, human_play
from play_controller import PlayController
from human_play import HumanPlay
from segments import load_all
from collect import randomize_object_yaws
from scene_gen import sample_scene, write_bddl
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
tasks, segs = load_all()
T="/home/mbronars/.claude/jobs/642d8e55/tmp/s"
saved=0
for ep in range(int(sys.argv[1]), int(sys.argv[1])+6):
    rng=np.random.default_rng(500+ep); cfg=sample_scene(rng); p=f"{T}/dbg{ep}.bddl"; names=write_bddl(cfg,p)
    env=OffScreenRenderEnv(bddl_file_name=p,camera_heights=160,camera_widths=160,ignore_done=True,horizon=10**6)
    env.reset(); randomize_object_yaws(env,rng,names)
    for _ in range(10): env.step(np.r_[np.zeros(6),-1])
    frames=[]; gws=[]; ezs=[]
    pc=PlayController(env,names,rng,max_steps=900,record_fn=lambda o,a,c: (frames.append(np.concatenate([o["agentview_image"][::-1],o["robot0_eye_in_hand_image"][::-1]],0)), gws.append(o["robot0_gripper_qpos"][0]-o["robot0_gripper_qpos"][1]), ezs.append(o["robot0_eef_pos"][2])))
    pc.obs=env.env._get_observations()
    hp=HumanPlay(pc,tasks,segs,rng)
    try: hp.run_episode(8)
    except Exception as e: print("exc", e)
    for s in hp.segments:
        if s["primitive"].startswith("pick_place") and not s["grasped"] and 0<=s["diag"]["key_err"]<0.02 and saved<4 and s["diag"]["donor"]==s["target"].split("->")[0].rsplit("_",1)[0]:
            k=s["start"]+s["diag"]["T"]
            idx=[i for i in range(k-8,k+28,4) if 0<=i<len(frames)]
            print("grip widths", [round(float(gw),3) for gw in np.array(gws)[idx]], "eef z", [round(float(z),3) for z in np.array(ezs)[idx]], "obj top z?", )
            Image.fromarray(np.concatenate([frames[i] for i in idx],1)).save(f"{T}/fail_{saved}.png")
            print("saved", saved, ep, s["target"], s["source"][:50], s["diag"]["donor"], "keyerr %.3f rot %.2f ncorr %d"%(s["diag"]["key_err"],s["diag"]["key_rot"],s["diag"]["n_corr"]))
            saved+=1
    env.close()
