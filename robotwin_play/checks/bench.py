"""Per-skill benchmark: run each skill on several random scenes (no
rendering, no recording) in parallel and report success rates and why
attempts failed or were skipped.

    python checks/bench.py OUTDIR SKILL[:FIXTURE] ... [--seeds 100-111] [--n 2] [--jobs 6]
"""
import argparse, collections, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/data/user_data/mbronars/.miniconda3/envs/robotwin/bin/python"
ROOT = "/data/user_data/mbronars/packages/RoboTwin"


def run(out, skill, fixture, seed, n, extra_env):
    env = dict(os.environ, PLAY_DEBUG="1", **extra_env)
    if fixture:
        env["FIXTURE"] = fixture
    log = os.path.join(out, f"{skill}{'_' + fixture if fixture else ''}_{seed}.log")
    with open(log, "w") as f:
        subprocess.run([PY, os.environ.get("BENCH_HARNESS", os.path.join(HERE, "dbg_skill.py")), str(seed), skill, str(n)], cwd=ROOT, env=env,
                       stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    res = []
    for line in open(log):
        m = re.search(r"^  (.*?)\s+success=(True|False) arm=(\S+)", line)
        if m:
            res.append(("ok" if m.group(2) == "True" else "fail", m.group(1).strip()))
        elif line.startswith("  skipped"):
            res.append(("skip", line.strip()[8:90]))
        elif "Traceback" in line:
            res.append(("crash", log))
    return skill + (":" + fixture if fixture else ""), seed, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("skills", nargs="+")
    p.add_argument("--seeds", default="100-109")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--env", default="", help="KEY=VAL,KEY=VAL for every run")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.seeds.split("-"))
    extra = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    jobs = [(s.split(":")[0], (s.split(":") + [None])[1], seed) for s in a.skills for seed in range(lo, hi + 1)]
    tally = collections.defaultdict(collections.Counter)
    with ThreadPoolExecutor(a.jobs) as ex:
        for key, seed, res in ex.map(lambda j: run(a.out, j[0], j[1], j[2], a.n, extra), jobs):
            for kind, what in res:
                tally[key][kind] += 1
    summary = {}
    for key, c in tally.items():
        tried = c["ok"] + c["fail"]
        summary[key] = dict(ok=c["ok"], tried=tried, skipped=c["skip"], crashed=c["crash"],
                            rate=round(c["ok"] / tried, 2) if tried else None)
        print(f"{key:22s} {c['ok']:3d}/{tried:<3d} ({100 * c['ok'] / max(tried, 1):3.0f}%)  skipped {c['skip']:3d}  crashed {c['crash']}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
