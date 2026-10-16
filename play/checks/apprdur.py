import os as _os, sys as _sys
PLAY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PROJECT = _os.path.dirname(PLAY)
_sys.path.insert(0, PLAY)
import sys, glob, json, numpy as np, h5py, itertools
for D in sys.argv[1:]:
    L=[]
    for fn in glob.glob(f'{D}/episode_*.hdf5'):
        g=h5py.File(fn)['data/demo_0']; ph=json.loads(g.attrs['phases']); pid=g['phase_id'][:]; P=g['obs/ee_pos'][:]; A=g['actions'][:]
        for k,v in itertools.groupby(range(len(pid)), key=lambda i: ph[pid[i]]=='approach'):
            v=list(v)
            if k and v[-1]+1<len(pid) and ph[pid[v[-1]+1]] in ('grasp','correct'):
                L.append((len(v), np.linalg.norm(P[v[-1]]-P[v[0]]), np.linalg.norm(A[v,3:6],axis=1).mean()))
    L=np.array(L); print(D, 'approach steps p25/50/75', np.percentile(L[:,0],[25,50,75]), ' dist p50 %.2f'%np.median(L[:,1]), ' steps per 10cm', np.round(np.median(L[:,0]/(L[:,1]*10)),1), 'mean |rot| %.3f'%np.mean(L[:,2]), 'n', len(L))
