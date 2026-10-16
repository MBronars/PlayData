"""Merge episodes from other collection runs into one dataset directory.

    python merge_sets.py DST NEW_EP=SRC_DIR:SRC_EP [NEW_EP=SRC_DIR:SRC_EP ...]

Copies each source episode's HDF5 and BDDL into DST under the new episode
number and appends its log.jsonl record (with "ep" renumbered and the source
recorded in "merged_from")."""
import json
import os
import shutil
import sys


def main():
    dst, specs = sys.argv[1], sys.argv[2:]
    logs = [json.loads(l) for l in open(os.path.join(dst, "log.jsonl"))]
    have = {r["ep"] for r in logs}
    for spec in specs:
        new, src = spec.split("=")
        new = int(new)
        src_dir, src_ep = src.rsplit(":", 1)
        src_ep = int(src_ep)
        if new in have:
            raise SystemExit(f"episode {new} already in {dst}")
        shutil.copy(os.path.join(src_dir, f"episode_{src_ep:06d}.hdf5"), os.path.join(dst, f"episode_{new:06d}.hdf5"))
        shutil.copy(os.path.join(src_dir, "bddl", f"episode_{src_ep:06d}.bddl"), os.path.join(dst, "bddl", f"episode_{new:06d}.bddl"))
        rec = [json.loads(l) for l in open(os.path.join(src_dir, "log.jsonl")) if json.loads(l)["ep"] == src_ep][0]
        rec["ep"], rec["merged_from"] = new, f"{os.path.basename(os.path.normpath(src_dir))}:{src_ep}"
        logs.append(rec)
        have.add(new)
    logs.sort(key=lambda r: r["ep"])
    with open(os.path.join(dst, "log.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in logs)


if __name__ == "__main__":
    main()
