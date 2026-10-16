import sys, numpy as np, imageio, pickle
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
from play_controller import PlayController, EpisodeTooLong
from retarget import Transform, plan, execute, choose_transform
from segments import load_all
from libero.libero.envs import OffScreenRenderEnv
T="/home/mbronars/.claude/jobs/642d8e55/tmp/s"
tasks, segs = load_all()
level = sys.argv[1] if len(sys.argv)>1 else "top"
want_open = (sys.argv[2] if len(sys.argv)>2 else "open")=="open"
cands=[s for s in segs if s["type"]=="articulation" and "cabinet" in s["joint"] and s["joint"].endswith(f"{level}_level")
       and ((s["q_end"]<s["q_start"]-0.05) if want_open else (s["q_end"]>s["q_start"]+0.05))]
fixcat = sys.argv[3] if len(sys.argv)>3 else "wooden_cabinet"
cands=[c for c in cands if c["joint"].startswith(fixcat)]
print(len(cands), "candidate segments; src yaws", sorted(set(round(c["anchor_yaw"],2) for c in cands)))
env=OffScreenRenderEnv(bddl_file_name=f"{T}/scene0.bddl",camera_heights=128,camera_widths=128,ignore_done=True,horizon=10**6)
names=['cream_cheese_1', 'alphabet_soup_1', 'black_book_1', 'bbq_sauce_1', 'chocolate_pudding_1']
rng=np.random.default_rng(0)
res={}
for variant in ["auto"]:
    ok=[]
    for trial in range(16):
        env.reset()
        for _ in range(5): env.step(np.r_[np.zeros(6),-1])
        frames=[]
        pc=PlayController(env,names,rng,max_steps=2000,record_fn=(lambda o,a,c: frames.append(o["agentview_image"][::-1])) if trial==0 else None)
        pc.obs=env.env._get_observations()
        dr=[d for d in pc.drawers if d["level"]==level][0]
        adr=pc.m.jnt_qposadr[dr["jid"]]
        if not want_open:
            pc.d.qpos[adr]=-0.15; pc.sim.forward()
        q0=pc.d.qpos[adr]
        hp=pc.d.geom_xpos[dr["handle"]].copy(); Rb=pc.d.xmat[dr["bid"]].reshape(3,3); yaw_n=np.arctan2(Rb[1,0],Rb[0,0])
        s=cands[rng.integers(len(cands))]
        d=tasks[s["task"]]["demos"][s["demo"]]
        sd={k:d[k][s["t0"]:s["t1"]] for k in ["eef_pos","eef_mat","actions"]}
        # anchor the human handle at its position when the drawer is closed vs ours
        G,dy=choose_transform(s["anchor_pos"], s["anchor_yaw"]-np.pi/2, hp, yaw_n-np.pi/2)
        if G is None: print("skip"); continue
        pl=plan(sd,G,pc.eef_pos,pc.eef_mat,blend_frac=0.8,key_t=s['move_t0']-s['t0'])
        mt=s["move_t0"]-s["t0"]
        traj=[]
        orig=pc.step_raw
        def sr(a, _o=orig):
            traj.append((pc.eef_pos.copy(), pc.eef_mat.copy())); return _o(a)
        pc.step_raw=sr
        try: execute(pc,pl,stop_fn=None)
        except EpisodeTooLong: pass
        q1=pc.d.qpos[adr]; ok.append(abs(q1-q0))
        k=min(mt, len(traj)-1)
        hum_rel = G.pos(d["eef_pos"][s["move_t0"]]) - hp
        our_rel = traj[k][0]-hp
        track = np.linalg.norm(traj[k][0]-pl["P"][k])
        maxtrack = max(np.linalg.norm(traj[i][0]-pl["P"][i]) for i in range(len(traj)))
        from retarget import rot_err
        print("   human rel %s  ours rel %s  track@contact %.3f max %.3f roterr %.2f" % (hum_rel.round(3), our_rel.round(3), track, maxtrack, np.linalg.norm(rot_err(pl["R"][k], traj[k][1]))))
        if trial==0: imageio.mimsave(f"{T}/rt_{level}_{variant}.mp4",frames,fps=20)
        print(variant, "mirror" if G.mirror else "", "dyaw %.2f"%dy, trial, s["task"][:50], "demo",s["demo"], "dq %.3f"%(q1-q0), "steps", pl["T"], flush=True)
    res[variant]=np.mean(np.array(ok)>0.08)
print(res)
