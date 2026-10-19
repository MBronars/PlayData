import os, sys, time
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/robotwin_play")
from collect_play import build_args, ROBOTWIN_ROOT
os.chdir(ROBOTWIN_ROOT); sys.path.insert(0, ROBOTWIN_ROOT)
from envs.play_table import play_table, _is_tool
from envs.utils import UnStableError
import numpy as np
args = build_args("play_clean", None, os.environ.get("PLAY_DBG_OUT", "/tmp/robotwin_play_dbg")); args["save_data"] = False
if os.environ.get("OBJECTS"):
    args["play"]["objects"] = os.environ["OBJECTS"].split(",")
    args["play"]["p_container"] = float(os.environ.get("P_CONTAINER", args["play"].get("p_container", 0.7)))
if os.environ.get("FIXTURE"):
    args["play"]["p_fixture"] = 1.0
    args["play"]["fixtures"] = [os.environ["FIXTURE"]]
seed, skill, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
def make_cands(t):
    objs, conts = t.objects, t.containers
    if skill == "put_cabinet":
        return [(a, f) for a in objs for f in t.fixtures]
    if skill == "stow":   # the drawer open, something to put in it
        for f in t.fixtures:
            j, i = t._fixture_joint(f); lo, hi = f.get_qlimits()[i]; q = f.get_qpos().copy(); q[i] = lo + 0.85 * (hi - lo); f.set_qpos(q)
        for _ in range(50): t.scene.step()
        return [(a, f) for a in objs for f in t.fixtures]
    if skill in ("open", "close", "toggle") and os.environ.get("FX_FRAC"):
        for f in t.fixtures:
            j, i = t._fixture_joint(f); lo, hi = f.get_qlimits()[i]; q = f.get_qpos().copy(); q[i] = lo + float(os.environ["FX_FRAC"]) * (hi - lo); f.set_qpos(q)
        for _ in range(50): t.scene.step()
        return [(f,) for f in t.fixtures]
    if skill in ("open", "close", "toggle"):
        if skill == "toggle" and os.environ.get("SWITCH_OFF"):
            for f in t.fixtures:
                j, i = t._fixture_joint(f); lo, hi = f.get_qlimits()[i]; q = f.get_qpos().copy(); q[i] = lo; f.set_qpos(q)
            for _ in range(50): t.scene.step()
        if skill == "close":   # start it open
            for f in t.fixtures:
                j, i = t._fixture_joint(f); lo, hi = f.get_qlimits()[i]; q = f.get_qpos().copy(); q[i] = lo + (0.6 if f.play_name == "015_laptop" else 0.8) * (hi - lo); f.set_qpos(q)
            for _ in range(50): t.scene.step()
        return [(f,) for f in t.fixtures]
    if skill in ("move", "push", "reorient", "handover", "press", "shake"):
        return [(a,) for a in objs]
    if skill == "put_in":
        return [(a, c) for a in objs for c in conts]
    if skill == "line_up":
        return [(a, b) for a in objs for b in objs if a is not b]
    if skill == "stack":
        from envs.play_table import STACK_BASES
        return [(a, b) for a in objs for b in objs + conts if a is not b and any(b.get_name().startswith(n) for n in STACK_BASES)]
    if skill == "knock_with":
        return [(a, b) for a in objs if _is_tool(a) for b in objs if b is not a]
    if skill == "dual_pick":
        return [(a, b) for a in objs for b in objs if a.get_pose().p[0] < 0 < b.get_pose().p[0]]
    if skill == "lift_together":
        return [(c,) for c in conts if len(list(c.iter_contact_points("list"))) >= 2]
    if skill == "pour":
        out = []
        from envs.play_table import OPEN_TOP
        for c in [c for c in conts if any(c.get_name().startswith(n) for n in OPEN_TOP)]:
            for o in t.objects:
                if not t._inside(o, c):
                    # put something in it first (teleport) so there is something to tip out
                    import sapien
                    lo, hi = t._aabb(c)
                    o.actor.set_pose(sapien.Pose([c.get_pose().p[0], c.get_pose().p[1], hi[2] + 0.02], o.get_pose().q))
                    for _ in range(100): t.scene.step()
                    if t._inside(o, c):
                        out.append((c,)); break
            if out: break
        return out
    return []
for k in range(12):
    t = play_table()
    try:
        t.setup_demo(now_ep_num=0, seed=int(os.environ["SCENE_SEED"]) if os.environ.get("SCENE_SEED") else seed * 31 + k, **args)
    except UnStableError:
        t.close_env(); continue
    t._done, t._touched = {}, {}
    cands = make_cands(t)
    if os.environ.get("ONLY"):   # only candidates whose first object is one of these
        cands = [c for c in cands if c[0].get_name() in os.environ["ONLY"].split(",")]
    if cands: break
    t.close_env()
objs, conts = t.objects, t.containers
print("scene", [a.get_name() for a in conts + objs])
rng = np.random.default_rng(seed); rng.shuffle(cands)
done = 0
for c in cands:
    if done >= n: break
    t0 = time.time(); t.plan_success = True
    if os.environ.get("FORCE_ERROR"):
        from envs.play_table import errors
        t.err = errors.sample_error(rng, skill, 1.0, 0.0); t.action_steps, t.action_len = 0, 2500
        print("  ERR", t.err)
    t._ignore = set(c)
    if os.environ.get("TRACK"):
        tracked = [a for a in t.objects + t.containers if a.get_name() == os.environ["TRACK"]] or [c[0]]
        t.save_freq = 5
        def _tp():
            a = tracked[0]
            print(f"      F{t.FRAME_IDX:4d} obj={np.round(a.get_pose().p, 3)} eeL={np.round(t._ee('left')[:3], 3)} eeR={np.round(t._ee('right')[:3], 3)} gripR={t.robot.get_right_arm_jointState()[-1]:.2f} tilt={t._tilt(a):.2f}", flush=True)
            t.FRAME_IDX += 1
        t._take_picture = _tp
    res = getattr(t, "p_" + skill)(*c); t.plan_success = True; t.err = None
    if res is None:
        print("  skipped", [a.get_name() for a in c], getattr(t, "_why", "")); continue
    done += 1
    for a in conts + objs:
        lo, hi = t._aabb(a); print(f"    {a.get_name():20s} p={np.round(a.get_pose().p,3)} aabb z {lo[2]:.3f}..{hi[2]:.3f} xy {np.round(lo[:2],3)}..{np.round(hi[:2],3)}")
    print(f"  {res['language']:60s} success={res['success']} arm={res['arm']} {time.time()-t0:.0f}s frames={t.FRAME_IDX} plans={t.n_plans} syncs={getattr(t,'n_syncs',0)}")
t.close_env()
