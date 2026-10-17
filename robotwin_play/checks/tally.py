"""Per-skill success counts and injected-error outcomes for play runs.

    python checks/tally.py RUN_DIR ...
"""
import collections, json, sys

for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{run}/log.jsonl")]
    eps = [r for r in rows if "segments" in r]
    bad = [r for r in rows if "segments" not in r]
    by_skill, by_err, clean = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0]), [0, 0]
    frames = 0
    for r in eps:
        frames += r["frames"]
        for s in r["segments"]:
            ok = bool(s.get("success"))
            by_skill[s["primitive"]][ok] += 1
            e = s.get("error")
            if e:
                for p in e.get("parts", [e["type"]]):
                    by_err[p + (" (severe)" if e.get("severe") else "")][ok] += 1
            else:
                clean[ok] += 1
    n = sum(v[0] + v[1] for v in by_skill.values())
    k = sum(v[1] for v in by_skill.values())
    print(f"{run}: {len(eps)} episodes ({len(bad)} failed), {frames} frames, {n} actions, {k} succeeded ({100 * k / max(n, 1):.0f}%)")
    print("  skills:   " + ", ".join(f"{s} {v[1]}/{v[0] + v[1]}" for s, v in sorted(by_skill.items(), key=lambda x: -sum(x[1]))))
    print(f"  no error: {clean[1]}/{clean[0] + clean[1]} succeeded")
    if by_err:
        print("  with error (succeeded/total): " + ", ".join(f"{s} {v[1]}/{v[0] + v[1]}" for s, v in sorted(by_err.items(), key=lambda x: -sum(x[1]))))
