"""Compare play episodes of two collection runs with the same seed (e.g. with and without cameras, or on two
machines): actions and MuJoCo states must match, and, where both runs have images, how far the images differ.

  python checks/compare_episodes.py RUN_A RUN_B"""
import glob, os, sys

import h5py
import numpy as np


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a_dir, "episode_*.hdf5")))
    if not names:
        sys.exit(f"no episodes in {a_dir}")
    worst = 0.0
    for n in names:
        pb = os.path.join(b_dir, n)
        if not os.path.exists(pb):
            print(f"{n}: missing in {b_dir}"); worst = np.inf; continue
        with h5py.File(os.path.join(a_dir, n)) as fa, h5py.File(pb) as fb:
            da, db = fa["data/demo_0"], fb["data/demo_0"]
            Ta, Tb = len(da["actions"]), len(db["actions"])
            line = f"{n}: frames {Ta} vs {Tb}"
            if Ta == Tb:
                act = float(np.abs(da["actions"][:] - db["actions"][:]).max())
                st = float(np.abs(da["states"][:] - db["states"][:]).max())
                worst = max(worst, act, st)
                line += f" | max |d action| {act:.3g} | max |d state| {st:.3g}"
                if da.attrs["segments"] != db.attrs["segments"]:
                    line += " | segments differ"
                for k in ("agentview_rgb", "eye_in_hand_rgb"):
                    if k in da["obs"] and k in db["obs"]:
                        ia, ib = da["obs"][k][:].astype(np.int16), db["obs"][k][:].astype(np.int16)
                        diff = np.abs(ia - ib)
                        line += f" | {k}: mean |d| {diff.mean():.3f}, pixels differing {100 * (diff.max(-1) > 0).mean():.2f}%, max {diff.max()}"
            else:
                worst = np.inf
        print(line)
    print("IDENTICAL trajectories" if worst == 0 else f"max difference {worst:.3g}")


if __name__ == "__main__":
    main()
