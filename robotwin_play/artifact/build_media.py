"""Videos + data for the RoboTwin play page: 2x2 camera grid (head, front,
left wrist, right wrist) at the recording rate, labels, and stats."""
import glob, json, os, subprocess, sys
import cv2, h5py, numpy as np
OUT = "/data/user_data/mbronars/AgenticExploration/artifact_robotwin"
# python build_media.py RUN_DIR[:ep,ep,...] ...   (default: every episode of each run)
RUNS = []
for a in sys.argv[1:]:
    run, _, ids = a.partition(":")
    RUNS.append((run, [int(x) for x in ids.split(",")] if ids else None))
os.makedirs(f"{OUT}/media", exist_ok=True)
FPS = 250 / 15   # one recorded frame per 15 sim steps at 250 Hz
CAMS = ["head_camera", "front_camera", "left_camera", "right_camera"]

def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # legacy RGB bytes -> RGB array

def grid_video(h5, mp4):
    with h5py.File(h5) as f:
        n = len(f["joint_action/vector"])
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                              "-r", f"{FPS:.4f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                              "-preset", "slow", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
        for t in range(n):
            ims = [dec(f[f"observation/{c}/rgb"][t]) for c in CAMS]
            fr = np.concatenate([np.concatenate(ims[:2], 1), np.concatenate(ims[2:], 1)], 0)
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
        p.stdin.close(); p.wait()
        q = f["joint_action/vector"][:]
    return n, q

eps = []
for run, ids in RUNS:
    log = {json.loads(l)["ep"]: json.loads(l) for l in open(f"{run}/log.jsonl")}
    for ep in (sorted(log) if ids is None else ids):
        r = log.get(ep)
        if r is None or "segments" not in r:
            continue
        key = f"{os.path.basename(run)}_{ep}"
        n, q = grid_video(f"{run}/data_released/episode{ep}.hdf5", f"{OUT}/media/{key}.mp4")
        eps.append(dict(key=key, run=os.path.basename(run), ep=ep, frames=n, kind=r.get("kind", os.path.basename(run).split("_")[-1]),
                        objects=[s["name"].split("_", 1)[1].replace("-", " ").replace("_", " ") for s in r["scene"]],
                        containers=[s["name"].split("_", 1)[1].replace("-", " ") for s in r["scene"] if s["container"]],
                        segments=[dict(s=s["start"], e=min(s["end"], n), p=s["primitive"], ok=bool(s["success"]),
                                       lang=s["language"], arm=s["arm"], via=s.get("via"),
                                       err=(s.get("error") or {}).get("type"), severe=bool((s.get("error") or {}).get("severe"))) for s in r["segments"]],
                        grip=np.round(q[:, [6, 13]], 2)[::2].tolist()))
        print("done", key, n, flush=True)

# one released expert episode for comparison
ref = "/data/user_data/mbronars/robotwin_ref/place_object_basket/aloha-agilex_clean_50/aloha-agilex_clean_50/data/episode0.hdf5"
n, _ = grid_video(ref, f"{OUT}/media/expert_place_object_basket.mp4")
expert = dict(file="media/expert_place_object_basket.mp4", task="place object basket (released expert demo, episode 0)", frames=n)

# action-distribution stats (same numbers as checks/actdist.py)
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/robotwin_play/checks")
import actdist
play = sorted(f for run, _ in RUNS for f in glob.glob(f"{run}/data_released/*.hdf5"))
refs = sorted(glob.glob("/data/user_data/mbronars/robotwin_ref/*/aloha-agilex_clean_50/**/data/*.hdf5", recursive=True))
P, R = actdist.stats(actdist.load(play)), actdist.stats(actdist.load(refs))
ARM = actdist.ARM
lo, hi = np.percentile(R["Q"][:, ARM], 0.1, axis=0) - 0.02, np.percentile(R["Q"][:, ARM], 99.9, axis=0) + 0.02
lim = np.percentile(R["dq"], 99.9, axis=0)
bins = np.linspace(0, 0.12, 25)
def hist(dq):
    v = dq.max(1); v = v[v > 1e-4]; h, _ = np.histogram(np.clip(v, 0, 0.1199), bins); return (h / h.sum()).round(4).tolist()
stats = dict(
    n_play=len(play), n_ref=len(refs), frames_play=int(P["frames"]), frames_ref=int(R["frames"]),
    rows=[["Median joint step while moving (rad)", P["dq_p50"], R["dq_p50"]],
          ["99th percentile joint step (rad)", P["dq_p99"], R["dq_p99"]],
          ["Largest joint step (rad)", P["dq_max"], R["dq_max"]],
          ["Median hand step (m)", P["ee_step_p50"], R["ee_step_p50"]],
          ["99th percentile hand step (m)", P["ee_step_p99"], R["ee_step_p99"]],
          ["Frames with both arms still", P["idle_frac"], R["idle_frac"]],
          ["Gripper open/close switches per 100 frames", P["grip_toggles_per_100"], R["grip_toggles_per_100"]],
          ["Frames with a joint outside the expert range", float(np.mean(np.any((P["Q"][:, ARM] < lo) | (P["Q"][:, ARM] > hi), 1))),
           float(np.mean(np.any((R["Q"][:, ARM] < lo) | (R["Q"][:, ARM] > hi), 1)))],
          ["Frames with a joint step beyond the expert 99.9th pct", float(np.mean(np.any(P["dq"] > lim, 1))),
           float(np.mean(np.any(R["dq"] > lim, 1)))]],
    hist=dict(bins=bins.round(3).tolist(), play=hist(P["dq"]), expert=hist(R["dq"])))
json.dump(dict(episodes=eps, expert=expert, stats=stats, fps=FPS), open(f"{OUT}/data.json", "w"))
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")).read()
open(f"{OUT}/index.html", "w").write(tpl.replace("__DATA__", open(f"{OUT}/data.json").read()))
print("ok", len(eps), "episodes ->", f"{OUT}/index.html")
