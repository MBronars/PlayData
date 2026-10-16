"""Human-vs-play discriminator. Trains a 1D-CNN on short windows of motion
(actions, eef linear/angular velocity, gripper opening) and reports held-out
accuracy (split by episode). 50% = indistinguishable.

  python eval_realism.py --play /path/to/play_dir [--human_glob '*'] [--win 32]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_analysis import load_human, load_play, HZ  # noqa
from scipy.spatial.transform import Rotation


def features(t):
    a = t["actions"].astype(np.float32)
    p = t["ee_pos"].astype(np.float32)
    v = np.diff(p, axis=0, prepend=p[:1]) * HZ
    R = Rotation.from_rotvec(t["ee_ori"])
    w = np.zeros_like(v)
    w[1:] = (R[1:] * R[:-1].inv()).as_rotvec() * HZ
    g = t["gripper_states"][:, :1] - t["gripper_states"][:, 1:2]
    return np.concatenate([a, v * 5, w, g * 20], 1).astype(np.float32)


def windows(trajs, win, stride, max_per=None, rng=None):
    out = []
    for i, t in enumerate(trajs):
        f = features(t)
        idx = list(range(0, len(f) - win, stride))
        if max_per and len(idx) > max_per:
            idx = list(rng.choice(idx, max_per, replace=False))
        for j in idx:
            out.append((i, f[j:j + win]))
    return out


class Net(nn.Module):
    def __init__(self, c, win):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x.transpose(1, 2)).squeeze(-1)


def run(human, play, win=32, seed=0, epochs=15, verbose=True):
    rng = np.random.default_rng(seed)
    # split episodes 75/25
    def split(n):
        p = rng.permutation(n)
        k = int(0.75 * n)
        return set(p[:k]), set(p[k:])
    htr, hte = split(len(human))
    ptr, pte = split(len(play))
    hw = windows(human, win, 8, max_per=20, rng=rng)
    pw = windows(play, win, 8, max_per=None, rng=rng)
    n = min(len(hw), len(pw))
    hw = [hw[i] for i in rng.choice(len(hw), n, replace=False)]
    pw = [pw[i] for i in rng.choice(len(pw), n, replace=False)]
    def mk(ws, ids, y):
        X = np.stack([w for i, w in ws if i in ids]) if ws else np.zeros((0, win, 1))
        return X, np.full(len(X), y, np.float32)
    Xh_tr, yh_tr = mk(hw, htr, 0); Xh_te, yh_te = mk(hw, hte, 0)
    Xp_tr, yp_tr = mk(pw, ptr, 1); Xp_te, yp_te = mk(pw, pte, 1)
    Xtr = np.concatenate([Xh_tr, Xp_tr]); ytr = np.concatenate([yh_tr, yp_tr])
    Xte = np.concatenate([Xh_te, Xp_te]); yte = np.concatenate([yh_te, yp_te])
    # balance test set
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = Net(Xtr.shape[2], win).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    w_pos = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_pos, device=dev))
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = lossf(net(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = torch.sigmoid(net(torch.tensor(Xte, device=dev))).cpu().numpy()
    acc_h = float(((pr < 0.5) == (yte == 0))[yte == 0].mean())
    acc_p = float(((pr >= 0.5) == (yte == 1))[yte == 1].mean())
    bal = 0.5 * (acc_h + acc_p)
    if verbose:
        print(f"balanced acc {bal:.3f} (human {acc_h:.3f}, play {acc_p:.3f}) "
              f"train {len(Xtr)} test {len(Xte)}")
    return bal, net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", required=True)
    ap.add_argument("--human_glob", default="*")
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    human = load_human("/data/user_data/mbronars/AgenticExploration/libero_ref/lowdim",
                       pattern=args.human_glob)
    play = load_play(sorted(glob.glob(os.path.join(args.play, "episode_*.hdf5"))))
    print(len(human), "human demos,", len(play), "play episodes")
    accs = [run(human, play, args.win, seed=s)[0] for s in range(args.seeds)]
    print("MEAN balanced acc %.3f" % np.mean(accs))
