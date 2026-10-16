"""Dump a frame grid (agentview | wrist) for an episode step range, plus per-frame joint margins and tilt."""
import sys, h5py, numpy as np, json
from PIL import Image
from scipy.spatial.transform import Rotation
fn, a, b, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
QLO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]); QHI = -QLO.copy(); QHI[3] = -0.0698; QHI[5] = 3.7525
with h5py.File(fn) as f:
    g = f["data/demo_0"]
    av = g["obs/agentview_rgb"][a:b]; wr = g["obs/eye_in_hand_rgb"][a:b]; Q = g["obs/joint_states"][a:b]; O = g["obs/ee_ori"][a:b]
    ph = g["phase_id"][a:b]; names = json.loads(g.attrs["phases"])
idx = np.linspace(0, len(av) - 1, k).astype(int)
rows = [np.concatenate([av[i][::-1], wr[i][::-1]], 0) for i in idx]
Image.fromarray(np.concatenate(rows, 1)).save(out)
for i in idx:
    R = Rotation.from_rotvec(O[i]).as_matrix()
    m = np.minimum(Q[i] - QLO, QHI - Q[i])
    print(a + i, names[ph[i]], "tilt %.0f" % np.degrees(np.arccos(-R[2, 2])), "closing axis", R[:, 0].round(2), "joint margins", m.round(2))
