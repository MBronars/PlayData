"""Builds the showcase page (artifact_composed/index.html + media/) from
showcase_play/ and showcase_hierr/. Run: python artifact/build_page.py"""
import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
_os.chdir(PLAY)   # paths below are relative to play/
import sys, json, glob, numpy as np, h5py, imageio
from motion_analysis import load_human, load_play, global_metrics, speed
OUT = _os.path.join(PROJECT, "artifact_composed")
logs = {json.loads(l)["ep"]: json.loads(l) for l in open("../showcase_play/log.jsonl")}
# episodes chosen to show each kind of play (fixtures, drawers, stacking, ...)
chosen = [ep for ep in (0, 2, 7, 15, 4, 1, 17, 20) if ep in logs]
eps = []
for ep in chosen:
    r = logs[ep]
    with h5py.File(f"../showcase_play/episode_{ep:06d}.hdf5") as f:
        g = f["data/demo_0"]
        av = g["obs/agentview_rgb"][:]; wr = g["obs/eye_in_hand_rgb"][:]
        A = g["actions"][:]; P = g["obs/ee_pos"][:]
        phases = json.loads(g.attrs["phases"]); ph = g["phase_id"][:]
        scene = json.loads(g.attrs["scene"])
    frames = [np.concatenate([a[::-1], b[::-1]], 1) for a, b in zip(av, wr)]
    imageio.mimsave(f"{OUT}/media/play_{ep}.mp4", frames, fps=20, quality=7, macro_block_size=8)
    v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) * 20]
    eps.append(dict(ep=ep, scene=r["scene"], fixture=r["fixture"], objects=scene["objects"], T=len(A),
                    segments=[dict(s=x["start"], e=x["end"], p=x["primitive"], ok=x["success"], lang=x["language"],
                                   src=x.get("source", ""), att=x.get("attempts", 1), dist=x.get("disturbed", []), cleared=x.get("cleared"), gr=((x.get("diag") or {}).get("grasp") or [None])[0], tr=bool(x.get("truncated")), ab=bool(x.get("abandoned")), err=(((x.get("error") or {}).get("type") or "").replace("+", " + ") + (" (severe)" if (x.get("error") or {}).get("severe") else "") + (" · knocked " + (x.get("error") or {})["hit_carry"].rsplit("_", 1)[0].replace("_", " ") + " on the way" if (x.get("error") or {}).get("hit_carry") else "")) or None) for x in r["segments"]],
                    speed=np.round(v[::2], 3).tolist(), grip=np.round(A[::2, 6], 0).tolist(),
                    phase=[phases[i] for i in ph[::2]]))
# high-error episodes (--p_err 1.0 --err_severity 1.0), shown separately; not in the stats
hlogs = {json.loads(l)["ep"]: json.loads(l) for l in open("../showcase_hierr/log.jsonl")}
for ep in (5, 4, 2, 3, 1, 0, 6, 7):
    r = hlogs[ep]
    with h5py.File(f"../showcase_hierr/episode_{ep:06d}.hdf5") as f:
        g = f["data/demo_0"]
        av = g["obs/agentview_rgb"][:]; wr = g["obs/eye_in_hand_rgb"][:]
        A = g["actions"][:]; P = g["obs/ee_pos"][:]
        phases = json.loads(g.attrs["phases"]); ph = g["phase_id"][:]
        scene = json.loads(g.attrs["scene"])
    frames = [np.concatenate([a[::-1], b[::-1]], 1) for a, b in zip(av, wr)]
    imageio.mimsave(f"{OUT}/media/hierr_{ep}.mp4", frames, fps=20, quality=7, macro_block_size=8)
    v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) * 20]
    eps.append(dict(ep=ep, hi=True, file=f"media/hierr_{ep}.mp4", scene=r["scene"], fixture=r["fixture"], objects=scene["objects"], T=len(A),
                    segments=[dict(s=x["start"], e=x["end"], p=x["primitive"], ok=x["success"], lang=x["language"],
                                   src=x.get("source", ""), att=x.get("attempts", 1), dist=x.get("disturbed", []), cleared=x.get("cleared"), gr=((x.get("diag") or {}).get("grasp") or [None])[0], tr=bool(x.get("truncated")), ab=bool(x.get("abandoned")),
                                   err=(((x.get("error") or {}).get("type") or "").replace("+", " + ") + (" (severe)" if (x.get("error") or {}).get("severe") else "") + (" · knocked " + (x.get("error") or {})["hit_carry"].rsplit("_", 1)[0].replace("_", " ") + " on the way" if (x.get("error") or {}).get("hit_carry") else "")) or None) for x in r["segments"]],
                    speed=np.round(v[::2], 3).tolist(), grip=np.round(A[::2, 6], 0).tolist(),
                    phase=[phases[i] for i in ph[::2]]))
# human demo videos
hum = []
for fn, name in [("libero_90__KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet", "open the top drawer of the cabinet"),
                 ("libero_90__LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket", "pick up the butter and put it in the basket"),
                 ("libero_10__KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "turn on the stove and put the moka pot on it")]:
    z = np.load(f"../libero_ref/lowdim/{fn}.npz", allow_pickle=True)
    av = z["demo_0/agentview_rgb"]
    imageio.mimsave(f"{OUT}/media/human_{len(hum)}.mp4", [a[::-1] for a in av], fps=20, quality=7, macro_block_size=8)
    hum.append(dict(file=f"media/human_{len(hum)}.mp4", task=name, T=len(av)))
# motion statistics
H = load_human("../libero_ref/lowdim")
S = load_play(sorted(glob.glob("../pilot_v1/episode_*.hdf5")))
F = load_play(sorted(glob.glob("../showcase_play/episode_*.hdf5")))
def stats(T):
    A = np.concatenate([t["actions"] for t in T]); V = np.concatenate([speed(t) for t in T])
    dA = np.concatenate([np.abs(np.diff(t["actions"][:, :6], axis=0)) for t in T])
    tr = np.linalg.norm(A[:, :3], axis=1); ro = np.linalg.norm(A[:, 3:6], axis=1)
    return dict(speed_p50=float(np.median(V)), speed_p90=float(np.percentile(V, 90)),
                act_p50=float(np.median(tr)), act_p90=float(np.percentile(tr, 90)),
                rot_p90=float(np.percentile(ro, 90)), jerk=float(dA[:, :3].mean()), jerk_rot=float(dA[:, 3:].mean()),
                sat=float((np.abs(A[:, :3]) >= 0.999).any(1).mean()), slow=float((V < 0.03).mean()))
# speed histograms
bins = np.linspace(0, 0.4, 21)
def hist(T):
    V = np.concatenate([speed(t) for t in T]); h, _ = np.histogram(np.clip(V, 0, 0.399), bins); return (h / h.sum()).round(4).tolist()
# primitive success in showcase
import collections
c = collections.defaultdict(lambda: [0, 0])
for r in logs.values():
    for s in r["segments"]:
        c[s["primitive"]][0] += s["success"]; c[s["primitive"]][1] += 1
picks = [s for r in logs.values() for s in r["segments"] if "grasped" in s and s["primitive"] != "pick_fumble"]
data = dict(episodes=eps, human=hum,
            stats=dict(human=stats(H), scripted=stats(S), play=stats(F)),
            hist=dict(bins=bins.round(3).tolist(), human=hist(H), scripted=hist(S), play=hist(F)),
            prim={k: v for k, v in c.items()},
            grasp=dict(n=len(picks), grasped=sum(s["grasped"] for s in picks),
                       first=sum(1 for s in picks if s["grasped"] and s.get("attempts", 1) == 1)),
            n_human=len(H), n_show=len(logs), steps_show=int(sum(r["T"] for r in logs.values())),
            n_seg=sum(len(r["segments"]) for r in logs.values()),
            n_dist=sum(bool(x.get("disturbed")) for r in logs.values() for x in r["segments"]))
json.dump(data, open(f"{OUT}/data.json", "w"))
print(json.dumps(data["stats"], indent=0)); print(data["grasp"], data["prim"])
# assemble the page: the template with the data inlined
tpl = open(_os.path.join(PLAY, "artifact", "template.html")).read()
open(f"{OUT}/index.html", "w").write(tpl.replace("__DATA__", open(f"{OUT}/data.json").read()))
