import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, json, numpy as np, h5py
from PIL import Image, ImageDraw
D, ep, a, b, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with h5py.File(f"{D}/episode_{ep:06d}.hdf5") as f:
    g = f["data/demo_0"]; av = g["obs/agentview_rgb"]; ph = g["phase_id"][:]; phases = json.loads(g.attrs["phases"])
    ts = list(range(a, min(b, len(ph)), step))
    ims = []
    for t in ts:
        im = Image.fromarray(av[t][::-1]); dr = ImageDraw.Draw(im); dr.text((2, 2), f"{t} {phases[ph[t]]}", fill=(255, 255, 0)); ims.append(np.array(im))
cols = 8
rows = [np.concatenate(ims[i:i + cols] + [np.zeros_like(ims[0])] * (cols - len(ims[i:i + cols])), 1) for i in range(0, len(ims), cols)]
Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[6])
