"""Aggregate play logs into an empirical clip transfer-success table.
For each human source task / clip: how often did the retargeted clip achieve
its interaction (grasp for approaches, joint motion for articulations)."""
import json, sys, collections
tasks = collections.defaultdict(lambda: [0, 0]); clips = collections.defaultdict(lambda: [0, 0])
for f in sys.argv[2:]:
    for l in open(f):
        r = json.loads(l)
        for s in r.get("segments", []):
            p = s["primitive"]
            if p.startswith("pick_place"):
                ok = bool(s.get("grasped"))
            elif any(p.startswith(k) for k in ("drawer", "microwave", "stove")):
                ok = bool(s["success"])
            else:
                continue
            kind = "pick" if p.startswith("pick") else p
            t = s["source"].split("#")[0]
            tasks[f"{kind}|{t}"][0] += ok; tasks[f"{kind}|{t}"][1] += 1
            clips[f"{kind}|{s['source']}"][0] += ok; clips[f"{kind}|{s['source']}"][1] += 1
json.dump(dict(tasks=tasks, clips=clips), open(sys.argv[1], "w"))
tot = sum(v[1] for v in tasks.values()); succ = sum(v[0] for v in tasks.values())
print(f"{len(tasks)} task keys, {len(clips)} clip keys, {tot} trials, overall {succ/tot:.2f}")
for k, v in sorted(tasks.items(), key=lambda x: x[1][0] / x[1][1]):
    if v[1] >= 8: print(f"{v[0]/v[1]:.2f} {v[1]:3d} {k[:90]}")
