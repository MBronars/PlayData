"""Play whole episodes without recording or rendering and print what was done
(for checking the action mix; runs where the GPU can't render).

    python checks/dry_episode.py SEED [N] [--fixtures 036_cabinet] [--p_fixture 1] [--p_err 0]
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from collect_play import build_args, ROBOTWIN_ROOT
os.chdir(ROBOTWIN_ROOT)
sys.path.insert(0, ROBOTWIN_ROOT)
from envs.play_table import play_table
from envs.utils import UnStableError

p = argparse.ArgumentParser()
p.add_argument("seed", type=int)
p.add_argument("n", type=int, nargs="?", default=1)
p.add_argument("--fixtures", default=None)
p.add_argument("--p_fixture", type=float, default=None)
p.add_argument("--p_err", type=float, default=0.0)
a = p.parse_args()
args = build_args("play_clean", None, "/tmp/robotwin_play_dry")
args["save_data"] = False
args["play"]["p_err"] = a.p_err
if a.p_fixture is not None:
    args["play"]["p_fixture"] = a.p_fixture
if a.fixtures:
    args["play"]["fixtures"] = a.fixtures.split(",")
for ep in range(a.n):
    seed = (a.seed * 100003 + ep) * 31
    for k in range(10):
        t = play_table()
        try:
            t.setup_demo(now_ep_num=ep, seed=seed + k, **args)
            break
        except UnStableError:
            t.close_env()
    t0 = time.time()
    info = t.play_once()["info"]
    print(f"ep {ep} scene {[s['name'] for s in info['scene']]}  {time.time() - t0:.0f}s plans {info['plans']}", flush=True)
    for s in info["segments"]:
        print(f"   {'ok ' if s['success'] else 'no '} {s['primitive']:12s} {s['language'][:55]:55s} {s.get('strategy', '')}", flush=True)
    t.close_env()
