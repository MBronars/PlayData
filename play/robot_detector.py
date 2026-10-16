"""Robot-likeness detector: trained on human demos (0) vs scripted-controller
play (1); applied to held-out human windows and to new play data. Reports the
fraction of windows flagged 'robotic'. Content-controlled realism check."""
import sys, glob, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/data/user_data/mbronars/AgenticExploration/play")
import eval_realism as ER
from motion_analysis import load_human, load_play
L = "/data/user_data/mbronars/AgenticExploration/libero_ref/lowdim"

def train(human, robot, win=32, seed=0, epochs=20):
    rng = np.random.default_rng(seed)
    hw = ER.windows(human, win, 8, max_per=12, rng=rng)
    rw = ER.windows(robot, win, 4, rng=rng)
    n = min(len(hw), len(rw))
    hw = [hw[i][1] for i in rng.choice(len(hw), n, replace=False)]
    rw = [rw[i][1] for i in rng.choice(len(rw), n, replace=False)]
    X = torch.tensor(np.stack(hw + rw)); y = torch.tensor(np.r_[np.zeros(n), np.ones(n)], dtype=torch.float32)
    dev = "cuda"; torch.manual_seed(seed)
    net = ER.Net(X.shape[2], win).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    X, y = X.to(dev), y.to(dev); lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        p = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 256):
            b = p[i:i + 256]; loss = lf(net(X[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    return net

def flag_rate(net, trajs, win=32):
    w = ER.windows(trajs, win, 8, max_per=40, rng=np.random.default_rng(1))
    X = torch.tensor(np.stack([x for _, x in w])).cuda()
    with torch.no_grad():
        return float((torch.sigmoid(net(X)) > 0.5).float().mean())

if __name__ == "__main__":
    human = load_human(L)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(human)); cut = int(0.8 * len(human))
    h_tr = [human[i] for i in idx[:cut]]; h_te = [human[i] for i in idx[cut:]]
    robot = load_play(sorted(glob.glob(sys.argv[1] + "/episode_*.hdf5")))
    robot_te = robot[-max(1, len(robot) // 5):]; robot_tr = robot[:-max(1, len(robot) // 5)]
    tests = {p: load_play(sorted(glob.glob(p + "/episode_*.hdf5"))) for p in sys.argv[2:]}
    for seed in range(2):
        net = train(h_tr, robot_tr, seed=seed)
        out = {"heldout_human": flag_rate(net, h_te), "heldout_scripted": flag_rate(net, robot_te)}
        for p, T in tests.items():
            out[p.split("/")[-1]] = flag_rate(net, T)
        print("seed", seed, {k: round(v, 3) for k, v in out.items()}, flush=True)
