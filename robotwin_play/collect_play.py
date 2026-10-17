"""Collect RoboTwin play episodes.

    python collect_play.py --out OUT --episodes N --workers W --seed S [--config play_clean] [--embodiment aloha-agilex]

Every episode runs in its own process (plan and record in one pass) and is
written by RoboTwin's own recorder: OUT/data/episode_XXXXXXX.hdf5 (+ video/,
_traj_data/ joint paths) plus OUT/log.jsonl with the play labels.
Run with the RoboTwin python env; ROBOTWIN_ROOT points at the RoboTwin checkout."""
import argparse
import json
import os
import queue
import sys
import time
import traceback

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/user_data/mbronars/packages/RoboTwin")
HERE = os.path.dirname(os.path.abspath(__file__))


def build_args(config, embodiment, out):
    import yaml
    cfg_dir = os.path.join(ROBOTWIN_ROOT, "env_cfg", "task_config")
    path = os.path.join(HERE, "configs", f"{config}.yml")
    args = yaml.safe_load(open(path if os.path.exists(path) else os.path.join(cfg_dir, f"{config}.yml")))
    emb_types = yaml.safe_load(open(os.path.join(cfg_dir, "_embodiment_config.yml")))
    emb = [embodiment] if embodiment else args["embodiment"]

    def emb_file(e):
        return emb_types[e]["file_path"]

    if len(emb) == 1:
        args["left_robot_file"] = args["right_robot_file"] = emb_file(emb[0])
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"], args["right_robot_file"] = emb_file(emb[0]), emb_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    for side in ("left", "right"):
        args[f"{side}_embodiment_config"] = yaml.safe_load(open(os.path.join(args[f"{side}_robot_file"], "config.yml")))
    args["embodiment_name"] = "+".join(map(str, emb[:2])) if len(emb) == 3 else str(emb[0])
    args.setdefault("play", {})["embodiment_name"] = args["embodiment_name"]
    args["task_name"] = "play_table"
    args["task_config"] = config
    args["save_path"] = out
    args["need_plan"] = True
    args["save_data"] = True
    args["render_freq"] = 0
    return args


def run_episode(ep, opts):
    os.chdir(ROBOTWIN_ROOT)
    sys.path.insert(0, ROBOTWIN_ROOT)
    import numpy as np
    from envs.play_table import play_table
    args = build_args(opts.config, opts.embodiment, os.path.abspath(opts.out))
    seed = opts.seed * 100003 + ep
    from envs.utils import UnStableError
    t0 = time.time()
    for k in range(20):
        # objects that don't settle (a tall thing stood up topples): next sub-seed
        task = play_table()
        try:
            task.setup_demo(now_ep_num=ep, seed=seed * 31 + k, **args)
            break
        except UnStableError:
            task.close_env()
    seed = seed * 31 + k
    info = task.play_once()
    task.save_traj_data(ep)
    task.close_env(clear_cache=True)
    segs = info["info"].get("segments", [])
    if task.FRAME_IDX == 0:
        raise RuntimeError("nothing was done in this scene (no plannable action)")
    # the released-dataset layout too (same recordings), then RoboTwin's current one
    sys.path.insert(0, HERE)
    from legacy_format import write_legacy
    write_legacy(task.folder_path["cache"], os.path.join(args["save_path"], "data_released", f"episode{ep}.hdf5"))
    task.merge_pkl_to_hdf5_video(instructions=[s.get("language", "") for s in segs] or ["play"])
    task.remove_data_cache()
    return dict(ep=ep, seed=seed, frames=int(task.FRAME_IDX), scene=info["info"].get("scene"), segments=segs,
                plans=info["info"].get("plans"), syncs=info["info"].get("syncs"), seconds=round(time.time() - t0, 1))


def worker(ep, opts, q):
    try:
        q.put((ep, "ok", run_episode(ep, opts)))
    except Exception:
        q.put((ep, "error", traceback.format_exc()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="play_clean")
    p.add_argument("--embodiment", default=None)
    opts = p.parse_args()
    os.makedirs(opts.out, exist_ok=True)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    pending = list(range(opts.start, opts.start + opts.episodes))
    procs, done, reported = [], 0, set()
    log = open(os.path.join(opts.out, "log.jsonl"), "a")

    def launch():
        while pending and sum(pr.is_alive() for pr in procs) < opts.workers:
            ep = pending.pop(0)
            pr = ctx.Process(target=worker, args=(ep, opts, q))
            pr.start()
            pr.ep = ep
            procs.append(pr)
    launch()
    while done < opts.episodes:
        try:
            ep, status, res = q.get(timeout=1.0)
        except queue.Empty:
            dead = [pr for pr in procs if not pr.is_alive() and pr.exitcode not in (0, None) and pr.ep not in reported]
            if not dead:
                launch()
                continue
            ep, status, res = dead[0].ep, "error", f"worker died, exit code {dead[0].exitcode}"
        reported.add(ep)
        done += 1
        launch()
        if status == "ok":
            print(f"[{done}/{opts.episodes}] ep {ep} frames={res['frames']} actions={len(res['segments'])} "
                  f"ok={sum(s.get('success', False) for s in res['segments'])} {res['seconds']}s", flush=True)
            log.write(json.dumps(res, default=str) + "\n")
        else:
            print(f"[{done}/{opts.episodes}] ep {ep} {status}: {str(res)[-1500:]}", flush=True)
            log.write(json.dumps(dict(ep=ep, status=status, info=str(res))) + "\n")
        log.flush()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
