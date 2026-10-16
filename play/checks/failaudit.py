import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, h5py, glob, json, numpy as np, collections
for D in sys.argv[1:]:
    C = collections.Counter(); rows = []
    for fn in sorted(glob.glob(f'{D}/episode_*.hdf5')):
        g = h5py.File(fn)['data/demo_0']; ph = json.loads(g.attrs['phases']); pid = g['phase_id'][:]; P = g['obs/ee_pos'][:]; A = g['actions'][:]
        names = np.array([ph[i] for i in pid])
        last = -100
        for t in range(1, len(names)):
            if names[t] == 'regrasp' and names[t-1] != 'regrasp':
                tc = t - 1
                while tc > 0 and A[tc, 6] > 0: tc -= 1
                closed = t - tc - 1
                lift = P[tc+1:t+1, 2].max() - P[tc+1:t+1, 2].min() if closed > 0 else 0.0
                kind = 'no-close' if closed == 0 else ('closed-no-lift' if lift < 0.015 else 'real')
                C[kind] += 1
                if t - last < 40: C['rapid-repeat'] += 1
                last = t
                if kind != 'real': rows.append((fn[-9:-5], t, kind, closed, round(float(lift), 3)))
    print(D, dict(C)); [print('   ', r) for r in rows[:12]]
