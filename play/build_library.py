"""Replay human LIBERO demos and record, per step, eef pose + all object poses +
import os as _os
articulation joint values. Output: one pickle per task in libero_ref/replay/."""
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

REF = _os.environ.get("LIBERO_REF", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "libero_ref"))


def process(fn):
    out = os.path.join(REF, "replay", os.path.basename(fn)[:-4] + ".pkl")
    if os.path.exists(out):
        return fn, "exists"
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    suite, task = os.path.basename(fn)[:-4].split("__")
    bddl = os.path.join(get_libero_path("bddl_files"), suite, task + ".bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=32, camera_widths=32)
    env.reset()
    e = env.env
    sim = e.sim
    m, d = sim.model._model, sim.data._data
    eef = m.site(e.robots[0].controller.eef_name).id
    objs = list(e.objects_dict.keys())
    obj_bid = [m.body(e.objects_dict[o].root_body).id for o in objs]
    obj_cat = [e.objects_dict[o].category_name if hasattr(e.objects_dict[o], "category_name")
               else o.rsplit("_", 1)[0] for o in objs]
    fixtures = [f for f in e.fixtures_dict if not f.endswith("table") and f != "floor"]
    fix_bid = [m.body(e.fixtures_dict[f].root_body).id for f in fixtures]
    # articulated joints that belong to fixtures (non-free)
    joints = []
    for j in range(m.njnt):
        n = m.joint(j).name
        if m.jnt_type[j] in (2, 3) and any(n.startswith(f + "_") for f in fixtures):
            joints.append((n, m.jnt_qposadr[j], m.jnt_bodyid[j]))
    # drawer handle geoms (collision geom furthest -y in drawer body)
    handles = {}
    for n, adr, bid in joints:
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
        if geoms:
            handles[n] = min(geoms, key=lambda g: m.geom_pos[g][1])
    z = np.load(fn, allow_pickle=True)
    demos = sorted({k.split("/")[0] for k in z.files if k.startswith("demo_")},
                   key=lambda s: int(s.split("_")[1]))
    res = dict(task=task, suite=suite, bddl=bddl, objects=objs, categories=obj_cat,
               fixtures=fixtures, fixture_cats=[e.fixtures_dict[f].category_name
                                                if hasattr(e.fixtures_dict[f], "category_name")
                                                else f.rsplit("_", 1)[0] for f in fixtures],
               joints=[j[0] for j in joints], table_z=float(e.workspace_offset[2]),
               workspace_offset=np.array(e.workspace_offset), demos=[])
    for dm in demos:
        S, A = z[f"{dm}/states"], z[f"{dm}/actions"]
        T = len(S)
        rec = dict(actions=A, eef_pos=np.zeros((T, 3)), eef_mat=np.zeros((T, 3, 3)),
                   obj_pos=np.zeros((T, len(objs), 3)), obj_quat=np.zeros((T, len(objs), 4)),
                   fix_pos=np.zeros((T, len(fixtures), 3)), fix_quat=np.zeros((T, len(fixtures), 4)),
                   joint_q=np.zeros((T, len(joints))), handle_pos=np.zeros((T, len(joints), 3)),
                   handle_mat=np.zeros((T, len(joints), 3, 3)),
                   grip_q=z[f"{dm}/gripper_states"], joint_states=z[f"{dm}/joint_states"])
        for t in range(T):
            sim.set_state_from_flattened(S[t])
            sim.forward()
            rec["eef_pos"][t] = d.site_xpos[eef]
            rec["eef_mat"][t] = d.site_xmat[eef].reshape(3, 3)
            rec["obj_pos"][t] = d.xpos[obj_bid]
            rec["obj_quat"][t] = d.xquat[obj_bid]
            rec["fix_pos"][t] = d.xpos[fix_bid]
            rec["fix_quat"][t] = d.xquat[fix_bid]
            for k, (n, adr, bid) in enumerate(joints):
                rec["joint_q"][t, k] = d.qpos[adr]
                if n in handles:
                    rec["handle_pos"][t, k] = d.geom_xpos[handles[n]]
                    rec["handle_mat"][t, k] = d.xmat[bid].reshape(3, 3)
        for k in list(rec):
            if isinstance(rec[k], np.ndarray) and rec[k].dtype == np.float64:
                rec[k] = rec[k].astype(np.float32)
        res["demos"].append(rec)
    env.close()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    return fn, "ok"


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(REF, "lowdim", "*.npz")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        for fn, st in p.imap_unordered(process, files):
            print(st, os.path.basename(fn), flush=True)
