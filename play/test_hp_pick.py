import sys, numpy as np, imageio, json
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController, EpisodeTooLong
from human_play import HumanPlay
from segments import load_all
from collect import randomize_object_yaws
from scene_gen import write_bddl
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
T="/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all()
target=sys.argv[1]; ntr=int(sys.argv[2]); yawrand=int(sys.argv[3])
cfg=dict(scene="kitchen",fixture=None,fixture_pose=None,
         objects=[target,"akita_black_bowl","alphabet_soup"],positions=[(-0.08,0.0),(-0.12,-0.2),(-0.15,0.2)])
p=f"{T}/hp_{target}.bddl"; names=write_bddl(cfg,p)
env=OffScreenRenderEnv(bddl_file_name=p,camera_heights=128,camera_widths=128,ignore_done=True,horizon=10**6)
rng=np.random.default_rng(int(sys.argv[4]) if len(sys.argv)>4 else 0)
res=[]
for tr in range(ntr):
    env.reset()
    if yawrand: randomize_object_yaws(env,rng,names)
    for _ in range(10): env.step(np.r_[np.zeros(6),-1])
    frames=[]
    pc=PlayController(env,names,rng,max_steps=3000,record_fn=lambda o,a,c: frames.append(np.concatenate([o["agentview_image"][::-1],o["robot0_eye_in_hand_image"][::-1]],1)))
    pc.obs=env.env._get_observations()
    import json as _j; hp=HumanPlay(pc,tasks,segs,rng,quality=_j.load(open('/data/user_data/mbronars/AgenticExploration/play/clip_quality.json')))
    pre = sys.argv[5] if len(sys.argv)>5 else "none"
    if pre=="hover": hp.p_hover(names[1])
    if pre=="pick": hp.p_pick_place(names[2])
    from human_play import rel_yaw
    _p,_q=hp.obj_pose(names[0]); oy=rel_yaw("butter" if "butter" in names[0] else names[0].rsplit("_",1)[0], _q)
    orig=hp.pc.step_raw
    kk={}
    def sr(a,_o=orig):
        R=hp.pc.eef_mat; kk["yaw"]=np.degrees(np.arctan2(R[1,0],R[0,0])); return _o(a)
    hp.pc.step_raw=sr
    info=hp.p_pick_place(names[0])
    if info is None: print("none"); continue
    d=info["diag"]; res.append(info["grasped"])
    print("objyaw %.0f"%np.degrees(oy), "key_rot %.2f"%info["diag"].get("key_rot",-1), tr, info["primitive"], info["grasped"], info["success"], d["donor"], "dyaw %.2f keyerr %.3f ts %d ncorr %d warp %.3f"%(d["dyaw"],d["key_err"],d["t_start"],d["n_corr"],d.get("end_warp",-1)), info["source"][:50], flush=True)
    idx=np.linspace(0,len(frames)-1,10).astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in idx],1)).save(f"{T}/hp_{target}_{tr}.png")
print("grasp rate", np.mean(res))
