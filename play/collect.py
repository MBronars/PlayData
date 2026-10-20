"""Collect LIBERO play data with the scripted play controller.

Each episode: a freshly sampled random scene (table, fixture, objects) and a
chain of random primitives. One HDF5 file per episode, LIBERO/robomimic-style:

  data/demo_0/actions             (T, 7)   OSC_POSE delta actions in [-1, 1]
  data/demo_0/states              (T, S)   flattened MuJoCo sim states
  data/demo_0/obs/agentview_rgb   (T, H, W, 3) uint8 (raw render, upside down as in LIBERO)
  data/demo_0/obs/eye_in_hand_rgb (T, H, W, 3) uint8
  data/demo_0/obs/{ee_pos, ee_ori, ee_states, gripper_states, joint_states, object_states}
  data/demo_0/primitive_id        (T,)  index into attrs['primitives'] (-1 = between segments)
  data/demo_0/segment_id          (T,)  index into attrs['segments'] (-1 = none)
  attrs: bddl, model_file (xml), scene (json), segments (json: primitive, target,
         start, end, success, language)

Usage:
  MUJOCO_GL=egl python collect.py --out /path/to/out --episodes 100 --workers 16
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRIMITIVES = ["pick_place", "pick_place_on", "lift_wave", "push",
              "drawer_open", "drawer_close", "reach",
              "pick_place_drawer", "pick_fumble", "pick_touch", "reach_toward", "microwave_open", "microwave_close", "stove_open", "stove_close"]
_LIB = None


def get_library():
    global _LIB
    if _LIB is None:
        from segments import load_all
        _LIB = load_all()
    return _LIB


_BANK = None


def get_bank():
    global _BANK
    if _BANK is None:
        from compose import ResidualBank
        tasks, lib = get_library()
        _BANK = ResidualBank(tasks, lib)
    return _BANK


_Q = None


def get_quality(path):
    global _Q
    if _Q is None:
        _Q = json.load(open(path)) if path and os.path.exists(path) else {}
    return _Q


def randomize_object_yaws(env, rng, names):
    """LIBERO spawns movable objects in a fixed canonical orientation; rotate
    them about world z for variety, then let them settle."""
    from human_play import GROUP_OF
    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    for n in names:
        obj = e.objects_dict[n]
        grp = GROUP_OF.get(n.rsplit("_", 1)[0])
        lim = {"mug": 0.6, "pot": 0.4, "pan": 0.4, None: 0.0, "basket": 0.4}.get(grp, np.pi / 4)
        if lim == 0:
            continue
        jn = obj.joints[0] if obj.joints else None
        if jn is None:
            continue
        adr = m.jnt_qposadr[m.joint(jn).id]
        q = d.qpos[adr + 3:adr + 7].copy()
        a = rng.uniform(-lim, lim)
        qz = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = q
        d.qpos[adr + 3:adr + 7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    e.sim.forward()


def run_episode(ep, args):
    import h5py
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from scene_gen import sample_scene, write_bddl
    from play_controller import PlayController

    seed = args.seed * 100003 + ep
    rng = np.random.default_rng(seed)
    out_h5 = os.path.join(args.out, f"episode_{ep:06d}.hdf5")
    if os.path.exists(out_h5):
        return ep, "exists", None
    cfg = sample_scene(rng, scene=args.scene)
    for _ in range(200):
        if args.fixture is None or cfg["fixture"] == args.fixture:
            break
        cfg = sample_scene(rng, scene=args.scene)
    bddl_path = os.path.join(args.out, "bddl", f"episode_{ep:06d}.bddl")
    names = write_bddl(cfg, bddl_path)

    if args.img > 0:
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=args.img,
                                 camera_widths=args.img, ignore_done=True, horizon=10 ** 6)
    else:                       # no cameras and no GL context: runs on CPU-only nodes (render later: render_from_states.py)
        from libero.libero.envs.env_wrapper import ControlEnv
        env = ControlEnv(bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, ignore_done=True, horizon=10 ** 6)
    env.seed(seed)
    env.reset()
    if args.mode in ("human", "composed", "humanpath"):
        randomize_object_yaws(env, rng, names)
    # let objects settle
    for _ in range(10):
        env.step(np.r_[np.zeros(6), -1.0])

    buf = {k: [] for k in ["actions", "states", "agentview_rgb", "eye_in_hand_rgb",
                           "ee_pos", "ee_ori", "gripper_states", "joint_states",
                           "object_states", "phase"]}

    def record(obs, a, pc):
        buf["actions"].append(a.astype(np.float32))
        buf["states"].append(pc.sim.get_state().flatten())
        if args.img > 0:
            buf["agentview_rgb"].append(obs["agentview_image"])
            buf["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])
        buf["ee_pos"].append(obs["robot0_eef_pos"])
        buf["ee_ori"].append(quat2axisangle(obs["robot0_eef_quat"]))
        buf["gripper_states"].append(obs["robot0_gripper_qpos"])
        buf["joint_states"].append(obs["robot0_joint_pos"])
        buf["object_states"].append(obs["object-state"])
        buf["phase"].append(getattr(pc, "phase", "other"))

    pc = PlayController(env, names, rng, max_steps=args.max_steps,
                        action_noise=args.noise, record_fn=record)
    fixed = pc.fix_sunk_objects(lambda: [env.step(np.r_[np.zeros(6), -1.0]) for _ in range(15)])
    pc.obs = env.env._get_observations()
    if args.mode in ("composed", "humanpath"):
        if args.mode == "humanpath":
            from human_path_play import HumanPathPlay
            tasks_, lib_ = get_library()
            hp = HumanPathPlay(pc, rng, get_bank(), tasks_, lib_, p_explore=args.explore)
        else:
            from play_tasks import PlayfulPlay
            hp = PlayfulPlay(pc, rng, get_bank(), p_explore=args.explore)
            hp.p_err = args.p_err
            hp.err_severity = args.err_severity
        segs = hp.run_episode(args.max_primitives)
        for s in segs:
            if s["primitive"] not in PRIMITIVES:
                PRIMITIVES.append(s["primitive"])
    elif args.mode == "human":
        from human_play import HumanPlay
        tasks, lib = get_library()
        hp = HumanPlay(pc, tasks, lib, rng, p_explore=args.explore, clip_topk=args.clip_topk,
                       quality=get_quality(args.quality))
        segs = hp.run_episode(args.max_primitives)
        for s in segs:
            base = s["primitive"].replace("_explore", "")
            if base not in PRIMITIVES:
                PRIMITIVES.append(base)
    else:
        segs = pc.run_episode(args.max_primitives)
    T = len(buf["actions"])

    prim_id = -np.ones(T, np.int16)
    seg_id = -np.ones(T, np.int16)
    for i, s in enumerate(segs):
        prim_id[s["start"]:s["end"]] = PRIMITIVES.index(s["primitive"].replace("_explore", ""))
        seg_id[s["start"]:s["end"]] = i
        s["target"] = str(s["target"])

    tmp = out_h5 + ".tmp"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps(dict(env_name="LIBERO_play", controller="OSC_POSE",
                                              control_freq=20, camera_size=args.img))
        g.attrs["primitives"] = json.dumps(PRIMITIVES)
        d = g.create_group("demo_0")
        d.attrs["num_samples"] = T
        d.attrs["seed"] = seed
        d.attrs["bddl"] = open(bddl_path).read()
        d.attrs["model_file"] = env.env.sim.model.get_xml()
        d.attrs["scene"] = json.dumps(dict(scene=cfg["scene"], fixture=cfg["fixture"],
                                           objects=names), default=str)
        d.attrs["segments"] = json.dumps(segs)
        d.create_dataset("actions", data=np.array(buf["actions"]))
        d.create_dataset("states", data=np.array(buf["states"]))
        d.create_dataset("primitive_id", data=prim_id)
        d.create_dataset("segment_id", data=seg_id)
        phases = sorted(set(buf["phase"]))
        d.attrs["phases"] = json.dumps(phases)
        d.create_dataset("phase_id", data=np.array([phases.index(p) for p in buf["phase"]], np.int8))
        d.create_dataset("dones", data=np.eye(1, T, T - 1, dtype=np.uint8)[0])
        d.create_dataset("rewards", data=np.zeros(T, np.uint8))
        o = d.create_group("obs")
        for k in ["agentview_rgb", "eye_in_hand_rgb"] if args.img > 0 else []:
            o.create_dataset(k, data=np.array(buf[k], np.uint8), compression="lzf",
                             chunks=(1, args.img, args.img, 3))
        for k in ["ee_pos", "ee_ori", "gripper_states", "joint_states"]:
            o.create_dataset(k, data=np.array(buf[k], np.float32))
        o.create_dataset("ee_states", data=np.concatenate(
            [np.array(buf["ee_pos"]), np.array(buf["ee_ori"])], 1).astype(np.float32))
        # object-state vector length depends on the scene; stored per-episode
        o.create_dataset("object_states", data=np.array(buf["object_states"], np.float32))
    os.replace(tmp, out_h5)

    if args.videos and ep < args.videos and args.img > 0:
        import imageio
        frames = [np.concatenate([a[::-1], b[::-1]], 1)
                  for a, b in zip(buf["agentview_rgb"], buf["eye_in_hand_rgb"])]
        imageio.mimsave(os.path.join(args.out, "videos", f"episode_{ep:06d}.mp4"),
                        frames[::2], fps=20)
    env.close()
    return ep, "ok", dict(T=T, scene=cfg["scene"], fixture=cfg["fixture"], segments=segs)


def worker(args, eps, q):
    for ep in eps:
        t0 = time.time()
        try:
            r = run_episode(ep, args)
            q.put((r[0], r[1], r[2], time.time() - t0))
        except Exception:
            q.put((ep, "error", traceback.format_exc(), time.time() - t0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scene", default=None, help="kitchen / living_room / study (default: random)")
    p.add_argument("--fixture", default=None, help="force a fixture, e.g. flat_stove, microwave, white_cabinet")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img", type=int, default=128, help="camera resolution; 0 = no cameras (CPU-only nodes, MUJOCO_GL=glx)")
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--max_primitives", type=int, default=12)
    p.add_argument("--err_severity", type=float, default=0.15, help="fraction of injected errors that are large and obvious")
    p.add_argument("--p_err", type=float, default=0.3, help="fraction of actions with an injected near-failure")
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--videos", type=int, default=5, help="save mp4 for the first N episodes")
    p.add_argument("--mode", choices=["humanpath", "composed", "human", "scripted"], default="humanpath",
                   help="composed: keyframe splines styled by human statistics; human: retarget human demo segments; scripted: analytic primitives")
    p.add_argument("--explore", type=float, default=0.25, help="prob. of exploratory/perturbed attempts")
    p.add_argument("--clip_topk", type=int, default=3, help="sample among the k best-matching human clips")
    p.add_argument("--quality", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_quality.json"),
                   help="empirical clip transfer-success table (from validate runs)")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "bddl"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)

    import multiprocessing as mp
    # same hash seed in every worker: set iteration order (and so the episode)
    # is then reproducible from --seed / --start alone
    os.environ.setdefault("PYTHONHASHSEED", "0")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    eps = list(range(args.start, args.start + args.episodes))
    # every episode in a fresh process: nothing (caches, class state, the
    # simulator) carries over from one episode to the next, so an episode is
    # the same whether it runs alone or in a batch
    pending = list(eps)
    procs = []

    def launch():
        while pending and sum(pr.is_alive() for pr in procs) < args.workers:
            ep_ = pending.pop(0)
            pr = ctx.Process(target=worker, args=(args, [ep_], q))
            pr.start()
            pr.ep = ep_
            procs.append(pr)
    launch()
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t_start, done, steps = time.time(), 0, 0
    import queue as _queue
    reported = set()
    while done < len(eps):
        try:
            ep, status, info, dt = q.get(timeout=1.0)
        except _queue.Empty:
            # a worker that died without reporting (e.g. a simulator crash)
            dead = [pr for pr in procs if not pr.is_alive() and pr.exitcode not in (0, None) and pr.ep not in reported]
            if not dead:
                launch()
                continue
            ep, status, info, dt = dead[0].ep, "error", f"worker died, exit code {dead[0].exitcode}", 0.0
        reported.add(ep)
        launch()
        done += 1
        if status == "ok":
            steps += info["T"]
            ok = sum(s["success"] for s in info["segments"])
            print(f"[{done}/{len(eps)}] ep {ep} {info['scene']}/{info['fixture']} T={info['T']} "
                  f"segs={len(info['segments'])} ok={ok} {dt:.0f}s | "
                  f"{steps / (time.time() - t_start):.0f} steps/s total", flush=True)
            log.write(json.dumps(dict(ep=ep, **info), default=str) + "\n")
        else:
            print(f"[{done}/{len(eps)}] ep {ep} {status}: {str(info)[-800:]}", flush=True)
            log.write(json.dumps(dict(ep=ep, status=status, info=str(info))) + "\n")
        log.flush()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
