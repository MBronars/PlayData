"""Compare recorded actions (joint targets per frame, released layout) of
RoboTwin play data against RoboTwin's released expert data.

    python checks/actdist.py PLAY_DIR [REF_GLOB]

PLAY_DIR/data_released/*.hdf5 vs. REF_GLOB (default: all downloaded
aloha-agilex clean_50 episodes under /data/user_data/mbronars/robotwin_ref).
In both, the action at frame t is the joint target vector of frame t+1."""
import glob
import sys

import h5py
import numpy as np

ARM = np.r_[0:6, 7:13]
GRIP = [6, 13]


def load(files):
    out = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            out.append(dict(q=f["joint_action/vector"][:], ee_l=f["endpose/left_endpose"][:, :3],
                            ee_r=f["endpose/right_endpose"][:, :3]))
    return out


def stats(E):
    D = np.concatenate([np.diff(e["q"], axis=0) for e in E])
    Q = np.concatenate([e["q"] for e in E])
    V = np.concatenate([np.r_[np.linalg.norm(np.diff(e["ee_l"], axis=0), axis=1), np.linalg.norm(np.diff(e["ee_r"], axis=0), axis=1)] for e in E])
    dq = np.abs(D[:, ARM])
    moving = dq.max(1) > 1e-4
    tog = sum(int(np.sum(np.abs(np.diff((e["q"][:, g] > 0.5).astype(int))))) for e in E for g in GRIP)
    return dict(
        frames=len(Q),
        dq_p50=np.percentile(dq[moving], 50), dq_p99=np.percentile(dq[moving], 99), dq_max=dq.max(),
        idle_frac=1 - moving.mean(),
        ee_step_p50=np.percentile(V[V > 1e-4], 50), ee_step_p99=np.percentile(V[V > 1e-4], 99),
        grip_toggles_per_100=100 * tog / len(Q),
        q_lo=np.percentile(Q[:, ARM], 0.5, axis=0), q_hi=np.percentile(Q[:, ARM], 99.5, axis=0),
        dq=dq, Q=Q)


def main():
    play = sorted(glob.glob(f"{sys.argv[1]}/data_released/*.hdf5"))
    ref = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else
                           "/data/user_data/mbronars/robotwin_ref/*/aloha-agilex_clean_50/**/data/*.hdf5", recursive=True))
    P, R = stats(load(play)), stats(load(ref))
    print(f"episodes: play {len(play)}, released expert {len(ref)}")
    for k in ("frames", "dq_p50", "dq_p99", "dq_max", "idle_frac", "ee_step_p50", "ee_step_p99", "grip_toggles_per_100"):
        print(f"  {k:22s} play {P[k]:10.4f}   expert {R[k]:10.4f}")
    # share of play frames whose joint targets / joint steps fall outside the expert range
    # (with 0.02 rad tolerance: joints resting exactly at a bound are not "outside")
    lo, hi = np.percentile(R["Q"][:, ARM], 0.1, axis=0) - 0.02, np.percentile(R["Q"][:, ARM], 99.9, axis=0) + 0.02
    out_q = np.mean(np.any((P["Q"][:, ARM] < lo) | (P["Q"][:, ARM] > hi), axis=1))
    lim = np.percentile(R["dq"], 99.9, axis=0)
    out_dq = np.mean(np.any(P["dq"] > lim, axis=1))
    ref_out_q = np.mean(np.any((R["Q"][:, ARM] < lo) | (R["Q"][:, ARM] > hi), axis=1))
    print(f"  frames with a joint target outside the expert 0.1-99.9% range: play {out_q:.4f} (expert itself {ref_out_q:.4f})")
    print(f"  frames with a joint step beyond the expert 99.9th percentile:   play {out_dq:.4f} (expert itself ~0.001)")


if __name__ == "__main__":
    main()
