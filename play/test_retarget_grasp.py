import sys, numpy as np, imageio
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController, EpisodeTooLong
from retarget import plan, execute, choose_transform
from segments import load_all
from scene_gen import write_bddl
from libero.libero.envs import OffScreenRenderEnv
T="/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all()
SYM={"akita_black_bowl":"inf","white_bowl":"inf","plate":"inf","glazed_rim_porcelain_ramekin":"inf",
     "alphabet_soup":"inf","tomato_sauce":"inf","butter":2,"cream_cheese":2,"chocolate_pudding":2,"black_book":2}
cfg=dict(scene="kitchen",fixture=None,fixture_pose=None,
         objects=["akita_black_bowl","butter","alphabet_soup","cream_cheese","porcelain_mug"],
         positions=[(-0.12,-0.15),(0.0,0.1),(-0.15,0.15),(0.05,-0.12),(-0.05,0.0)])
p=f"{T}/grasp_scene.bddl"; names=write_bddl(cfg,p)
env=OffScreenRenderEnv(bddl_file_name=p,camera_heights=128,camera_widths=128,ignore_done=True,horizon=10**6)
rng=np.random.default_rng(0)
stats={}
for name in names:
    cat=name.rsplit("_",1)[0]
    cands=[s for s in segs if s["type"]=="grasp_approach" and s["cat"]==cat and s["lifted"]]
    succ=[]
    for trial in range(6):
        env.reset()
        for _ in range(5): env.step(np.r_[np.zeros(6),-1])
        frames=[]
        pc=PlayController(env,names,rng,max_steps=3000,record_fn=(lambda o,a,c: frames.append(o["agentview_image"][::-1])) if trial==0 else None)
        pc.obs=env.env._get_observations()
        bid=pc.m.body(pc.e.objects_dict[name].root_body).id
        q=pc.d.xquat[bid]; yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        opos=pc.d.xpos[bid].copy()
        for _ in range(20):
            s=cands[rng.integers(len(cands))]
            G,dy=choose_transform(s["anchor_pos"], s["anchor_yaw"], opos, yaw, rng=rng, sym=SYM.get(cat))
            if G is not None: break
        d=tasks[s["task"]]["demos"][s["demo"]]
        # approach + the following transport's first 25 steps (close + lift)
        t1=min(s["t1"]+25, len(d["actions"]))
        sd={k:d[k][s["t0"]:t1] for k in ["eef_pos","eef_mat","actions"]}
        pl=plan(sd,G,pc.eef_pos,pc.eef_mat,blend_frac=0.8,key_t=s["t1"]-s["t0"])
        z0=opos[2]
        try: execute(pc,pl)
        except EpisodeTooLong: pass
        lift=pc.d.xpos[bid][2]-z0
        succ.append(lift>0.03 and pc.is_grasped(name))
        if trial==0: imageio.mimsave(f"{T}/grasp_{cat}.mp4",frames,fps=20)
        print(cat, trial, s["task"][:45], "mir" if G.mirror else "", "dyaw %.2f"%dy, "lift %.3f"%lift, succ[-1], flush=True)
    stats[cat]=f"{sum(succ)}/{len(succ)}"
print(stats)
