"""Replay human demo states to extract object/fixture poses alongside eef motion."""
import os, sys, json, numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

def make_env(npz_name):
    suite, task = os.path.basename(npz_name)[:-4].split("__")
    bddl = os.path.join(get_libero_path("bddl_files"), suite, task + ".bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64,
                             has_offscreen_renderer=False, use_camera_obs=False) if False else \
          OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=32, camera_widths=32)
    env.reset()
    return env

def drawer_info(env):
    e = env.env; m = e.sim.model._model
    out = []
    for fname in e.fixtures_dict:
        for lvl in ["top", "middle", "bottom"]:
            try:
                jid = m.joint(f"{fname}_{lvl}_level").id; bid = m.body(f"{fname}_cabinet_{lvl}").id
            except KeyError:
                continue
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
            h = min(geoms, key=lambda g: m.geom_pos[g][1])
            out.append((f"{fname}:{lvl}", m.jnt_qposadr[jid], bid, h))
    return out

if __name__ == "__main__":
    fn = sys.argv[1]
    z = np.load(fn, allow_pickle=True)
    env = make_env(fn); e = env.env; sim = e.sim; m = sim.model._model; d = sim.data._data
    drs = drawer_info(env)
    demos = sorted({k.split("/")[0] for k in z.files if k.startswith("demo_")}, key=lambda s: int(s.split("_")[1]))
    eef = m.site(e.robots[0].controller.eef_name).id
    for dm in demos[:int(sys.argv[2]) if len(sys.argv) > 2 else 10]:
        S = z[f"{dm}/states"]; A = z[f"{dm}/actions"]
        rec = []
        for t in range(len(S)):
            sim.set_state_from_flattened(S[t]); sim.forward()
            R = d.site_xmat[eef].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
            row = [d.site_xpos[eef].copy(), tilt, A[t, 6]]
            for name, adr, bid, h in drs:
                row.append((d.qpos[adr], d.geom_xpos[h].copy()))
            rec.append(row)
        # find the drawer that moved
        for k, (name, *_ ) in enumerate(drs):
            q = np.array([r[3 + k][0] for r in rec])
            if np.ptp(q) < 0.02:
                continue
            mv = np.where(np.abs(np.diff(q)) > 1e-3)[0]
            t0, t1 = mv[0], mv[-1]
            rel = np.array([rec[t][0] - rec[t][3 + k][1] for t in range(t0, t1 + 1)])
            tilts = np.array([rec[t][1] for t in range(t0, t1 + 1)])
            grip = np.array([rec[t][2] for t in range(t0, t1 + 1)])
            print(f"{dm} {name} q {q[0]:.3f}->{q[-1]:.3f} move t={t0}-{t1}/{len(S)} "
                  f"eef-handle at start {rel[0].round(3)} mid {rel[len(rel)//2].round(3)} "
                  f"tilt {tilts.mean():.0f}deg grip_closed {np.mean(grip>0):.2f}")
