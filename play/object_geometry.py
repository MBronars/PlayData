"""Per-category canonical geometry: bbox (relative to body origin) at spawn orientation."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero.libero.envs.objects import get_object_dict
from scene_gen import write_bddl
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_geometry.json")
SKIP = {"target_zone"}
ARTIC = {"microwave", "slide_cabinet", "window", "faucet", "basin_faucet", "short_cabinet",
         "short_fridge", "wooden_cabinet", "white_cabinet", "flat_stove", "wooden_shelf",
         "wooden_two_layer_shelf", "wine_rack", "dining_set_group", "desk_caddy", "rack",
         "bowl_drainer", "white_storage_box", "dining_set_group"}

def bbox(m, d, bid):
    geoms = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == bid and m.geom_contype[g] != 0]
    pts = []
    for g in geoms:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pts.append(corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    pts = np.concatenate(pts)
    return pts.min(0) - d.xpos[bid], pts.max(0) - d.xpos[bid]

if __name__ == "__main__":
    from libero.libero.envs import OffScreenRenderEnv
    cats = [c for c in get_object_dict() if c not in SKIP and c not in ARTIC]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cats = [c for c in cats if c not in res and c not in {"cherries", "corn", "mayo"}]
    if len(sys.argv) > 1:
        cats = sys.argv[1].split(",")
    T = "/home/mbronars/.claude/jobs/642d8e55/tmp/s"
    for i in range(0, len(cats), 1):
        batch = cats[i:i + 1]
        cfg = dict(scene="kitchen", fixture=None, fixture_pose=None, objects=batch,
                   positions=[(-0.15, -0.2), (-0.15, 0.2), (0.1, -0.2), (0.1, 0.2)][:len(batch)])
        p = f"{T}/geom_{i}.bddl"
        names = write_bddl(cfg, p)
        try:
            env = OffScreenRenderEnv(bddl_file_name=p, camera_heights=32, camera_widths=32)
            env.reset()
        except Exception as ex:
            print("fail", batch, ex); continue
        e = env.env; m, d = e.sim.model._model, e.sim.data._data
        for _ in range(20): env.step(np.zeros(7))
        for n, c in zip(names, batch):
            bid = m.body(e.objects_dict[n].root_body).id
            lo, hi = bbox(m, d, bid)
            res[c] = dict(lo=lo.round(4).tolist(), hi=hi.round(4).tolist(),
                          quat=d.xquat[bid].round(4).tolist())
            print(c, res[c], flush=True)
        env.close()
    json.dump(res, open(OUT, "w"), indent=1)
