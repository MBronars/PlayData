"""Goal-conditioned human motor-style policy, trained on all human LIBERO demos.

Each step is labeled with the next keyframe the human was heading to: the eef
pose at the next gripper toggle (close / open) or the start / end of an
articulated-joint motion, or the demo end. The policy maps
  (goal - eef position, goal orientation error, eef pos rel. robot base,
   recent velocities, recent actions, gripper opening, keyframe type)
to the next action (mean + std for the 6 arm dims, logit for the gripper).
At play time, keyframes come from scene geometry, the policy supplies the
human-like motion between them."""
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
H = 4  # history length
KF_TYPES = ["close", "open", "artic_start", "artic_end", "end"]


def keyframes_for_demo(d, joint_q):
    A = d["actions"]
    T = len(A)
    g = np.sign(A[:, 6])
    kf = []
    for i in np.where(np.diff(g) != 0)[0] + 1:
        kf.append((int(i), "close" if g[i] > 0 else "open"))
    if joint_q is not None and joint_q.shape[1] > 0:
        for j in range(joint_q.shape[1]):
            q = joint_q[:, j]
            if np.ptp(q) < 0.02:
                continue
            mv = np.where(np.abs(np.diff(q)) > 5e-4)[0]
            if len(mv):
                kf.append((int(mv[0]), "artic_start"))
                kf.append((int(mv[-1] + 1), "artic_end"))
    kf.append((T - 1, "end"))
    kf = sorted(set(kf))
    return kf


def features(eef_pos, eef_mat, vel_hist, act_hist, grip_w, goal_pos, goal_mat, kf_type, base):
    dp = goal_pos - eef_pos
    dr = Rotation.from_matrix(goal_mat @ eef_mat.T).as_rotvec()
    # gripper symmetry: use the smaller of the two equivalent goal orientations
    dr2 = Rotation.from_matrix(goal_mat @ np.diag([-1.0, -1.0, 1.0]) @ eef_mat.T).as_rotvec()
    if np.linalg.norm(dr2) < np.linalg.norm(dr):
        dr = dr2
    oh = np.zeros(len(KF_TYPES)); oh[KF_TYPES.index(kf_type)] = 1
    return np.concatenate([dp * 5, dr, (eef_pos - base) * 2, eef_mat[:, 2], eef_mat[:, 0],
                           vel_hist.ravel() * 20, act_hist.ravel(), [grip_w * 10], oh,
                           [np.linalg.norm(dp) * 5]]).astype(np.float32)


def build_dataset(out_path):
    from transit import task_base
    X, Y, W = [], [], []
    files = sorted(glob.glob(os.path.join(HERE, "..", "libero_ref", "replay", "*.pkl")))
    for fn in files:
        R_ = pickle.load(open(fn, "rb"))
        base = task_base(R_)
        for d in R_["demos"]:
            A = d["actions"].astype(np.float64)
            P = d["eef_pos"].astype(np.float64)
            M = d["eef_mat"].astype(np.float64)
            gq = d["grip_q"]
            T = len(A)
            kf = keyframes_for_demo(d, d["joint_q"])
            V = np.diff(P, axis=0, prepend=P[:1])
            ki = 0
            for t in range(H, T - 1):
                while ki < len(kf) and kf[ki][0] <= t:
                    ki += 1
                if ki >= len(kf):
                    break
                tk, typ = kf[ki]
                x = features(P[t], M[t], V[t - H + 1:t + 1], A[t - H:t, :], gq[t][0] - gq[t][1],
                             P[tk], M[tk], typ, base)
                X.append(x); Y.append(A[t].astype(np.float32))
    X = np.array(X); Y = np.array(Y)
    np.savez_compressed(out_path, X=X, Y=Y)
    print("dataset", X.shape, Y.shape)
    return X, Y


class StyleNet(nn.Module):
    def __init__(self, din, hid=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                               nn.Linear(hid, hid), nn.GELU())
        self.mu = nn.Linear(hid, 6)
        self.logstd = nn.Linear(hid, 6)
        self.grip = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.f(x)
        return self.mu(h), self.logstd(h).clamp(-5, 0), self.grip(h).squeeze(-1)


def train(data_path, out_path, epochs=30, bs=4096, lr=1e-3):
    z = np.load(data_path)
    X, Y = torch.tensor(z["X"]), torch.tensor(z["Y"])
    n = len(X); idx = torch.randperm(n); nv = n // 20
    vi, ti = idx[:nv], idx[nv:]
    mean, std = X[ti].mean(0), X[ti].std(0) + 1e-3
    dev = "cuda"
    net = StyleNet(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xn = ((X - mean) / std).to(dev); Yd = Y.to(dev)
    for ep in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(ti))]
        tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs].to(dev)
            mu, ls, gl = net(Xn[b])
            nll = (0.5 * ((Yd[b, :6] - mu) / ls.exp()) ** 2 + ls).sum(-1).mean()
            gb = nn.functional.binary_cross_entropy_with_logits(gl, (Yd[b, 6] > 0).float())
            loss = nll + gb
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            b = vi.to(dev)
            mu, ls, gl = net(Xn[b])
            mae = (Yd[b, :6] - mu).abs().mean().item()
            gacc = ((gl > 0) == (Yd[b, 6] > 0)).float().mean().item()
        print(f"ep {ep} train {tot / len(ti):.3f} val mae {mae:.4f} grip acc {gacc:.4f}", flush=True)
    torch.save(dict(state=net.state_dict(), mean=mean, std=std, din=X.shape[1]), out_path)


class StylePolicy:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = StyleNet(ck["din"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.mean, self.std = ck["mean"].numpy(), ck["std"].numpy()

    def __call__(self, x, sample_scale=0.0, rng=None):
        xn = (x - self.mean) / self.std
        with torch.no_grad():
            mu, ls, gl = self.net(torch.tensor(xn[None]))
        mu = mu[0].numpy(); sd = ls[0].exp().numpy()
        if sample_scale > 0 and rng is not None:
            mu = mu + sample_scale * sd * rng.normal(size=6)
        return mu, sd, float(torch.sigmoid(gl[0]))


if __name__ == "__main__":
    dp = os.path.join(HERE, "..", "libero_ref", "style_data.npz")
    if sys.argv[1] == "data":
        build_dataset(dp)
    else:
        train(dp, os.path.join(HERE, "style_policy.pt"), epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 30)
