"""Render camera streams for play episodes collected without cameras (collect.py --img 0).

The scene is rebuilt from the episode's own BDDL (data/demo_0 attrs), every saved MuJoCo state is set and both
cameras are rendered, and obs/agentview_rgb + obs/eye_in_hand_rgb are written into the hdf5 in collect.py's format
(raw render, upside down as in LIBERO). Episodes that already have images are skipped.

Live rendering during collection draws body poses computed before mj_step's last position update, while the saved
state holds the updated qpos, so by default each state's qpos is stepped back one physics step (mj_integratePos with
-dt, the exact inverse of MuJoCo's Euler position update) before rendering. The scene comes from the BDDL when the
rebuilt model equals the stored MuJoCo XML, else from that XML (the collector sometimes edits the scene after
building it). Against live renders of the same episodes: identical images when the BDDL model matches, under 0.2%
of pixels different when the stored XML is needed.

  MUJOCO_GL=egl python render_from_states.py DIR_OR_FILES... [--img 128] [--workers 4]
Needs a GPU (EGL)."""
import argparse, glob, json, os, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def render_episode(path, img, prestep=1):
    import h5py, mujoco
    from libero.libero.envs import OffScreenRenderEnv
    with h5py.File(path, "r") as f:
        d = f["data/demo_0"]
        if "agentview_rgb" in d["obs"]:
            return path, "has images", 0.0
        bddl, states, xml = d.attrs["bddl"], d["states"][:], d.attrs["model_file"]
    t0 = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".bddl", delete=False) as tf:
        tf.write(bddl); bddl_path = tf.name
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=img, camera_widths=img, ignore_done=True, horizon=10 ** 6)
    env.reset(); os.unlink(bddl_path)
    # The collector can change the scene after building it from the BDDL (e.g. fixing sunk objects), so the
    # rebuilt model is used only when it equals the stored one; otherwise the stored MuJoCo XML is loaded.
    same_model = env.env.sim.model.get_xml() == xml
    if not same_model:
        env.env.reset_from_xml_string(xml)
    sim = env.env.sim; m, d = sim.model._model, sim.data._data
    if states.shape[1] != len(sim.get_state().flatten()):
        raise ValueError(f"state dim {states.shape[1]} != model's {len(sim.get_state().flatten())}")
    av, eh = [], []
    for s in states:
        sim.set_state_from_flattened(s)
        if prestep:
            q = d.qpos.copy(); mujoco.mj_integratePos(m, q, d.qvel, -prestep * m.opt.timestep); d.qpos[:] = q
        sim.forward()
        obs = env.env._get_observations(force_update=True)
        av.append(obs["agentview_image"]); eh.append(obs["robot0_eye_in_hand_image"])
    env.close()
    with h5py.File(path, "a") as f:
        o = f["data/demo_0/obs"]
        for k, v in (("agentview_rgb", av), ("eye_in_hand_rgb", eh)):
            o.create_dataset(k, data=np.array(v, np.uint8), compression="lzf", chunks=(1, img, img, 3))
        a = json.loads(f["data"].attrs.get("env_args", "{}")); a["camera_size"] = img; a["rendered_from_states"] = dict(prestep=prestep, model="bddl" if same_model else "stored_xml")
        f["data"].attrs["env_args"] = json.dumps(a)
    return path, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prestep", type=int, default=1, help="physics steps to step each state back before rendering (see above)")
    a = ap.parse_args()
    files = [f for p in a.inputs for f in (sorted(glob.glob(os.path.join(p, "episode_*.hdf5"))) if os.path.isdir(p) else [p])]
    for attempt in (1, 2):                              # a GL context occasionally dies; retry those once in a fresh pool
        failed = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(render_episode, f, a.img, a.prestep): f for f in files}
            for fu in as_completed(futs):
                try:
                    p, st, sec = fu.result(); print(f"{st:10s} {os.path.basename(p)} {sec:.1f}s", flush=True)
                except Exception as e:
                    failed.append(futs[fu])
                    print(f"{'retry' if attempt == 1 else 'FAIL'} {os.path.basename(futs[fu])}: {type(e).__name__}: {e}", flush=True)
        if not failed:
            break
        files = failed


if __name__ == "__main__":
    main()
