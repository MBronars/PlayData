"""Find unnatural jerks: largest step-to-step action changes and eef acceleration
spikes, with the phase transition they occur at (compared to human levels)."""
import sys, json, h5py, numpy as np, collections
fn = sys.argv[1]
with h5py.File(fn) as f:
    g = f["data/demo_0"]
    A = g["actions"][:]; P = g["obs/ee_pos"][:]
    ph = g["phase_id"][:]; names = json.loads(g.attrs["phases"])
da = np.linalg.norm(np.diff(A[:, :3], axis=0), axis=1)
dr = np.linalg.norm(np.diff(A[:, 3:6], axis=0), axis=1)
acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=1) * 400
print("human reference: |da_trans| p99 ~0.20, |da_rot| p99 ~0.06, accel p99 ~1.5 m/s^2")
print("this episode: |da_trans| p99 %.3f max %.3f | |da_rot| p99 %.3f max %.3f | accel p99 %.2f max %.2f" % (
    np.percentile(da, 99), da.max(), np.percentile(dr, 99), dr.max(), np.percentile(acc, 99), acc.max()))
spikes = collections.Counter()
for t in np.argsort(-da)[:25]:
    spikes[(names[ph[t]], names[ph[t + 1]])] += 1
print("top-25 translational action jumps by (phase before -> after):", spikes.most_common())
for t in np.argsort(-da)[:8]:
    print("  t=%d |da|=%.2f %s->%s a=%s -> %s" % (t, da[t], names[ph[t]], names[ph[t + 1]], A[t, :3].round(2), A[t + 1, :3].round(2)))
