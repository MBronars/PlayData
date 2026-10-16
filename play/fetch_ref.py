"""Stream LIBERO demo files from HF, keep low-dim data (+ images for 2 demos/file), delete originals."""
import os as _os
import os, sys, numpy as np, h5py, json
from huggingface_hub import list_repo_files, hf_hub_download
OUT = _os.environ.get("LIBERO_REF", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "libero_ref"))
os.makedirs(f"{OUT}/lowdim", exist_ok=True)
files = [f for f in list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset") if f.endswith(".hdf5")]
for f in files:
    name = os.path.basename(f).replace("_demo.hdf5", "")
    suite = f.split("/")[0]
    out = f"{OUT}/lowdim/{suite}__{name}.npz"
    if os.path.exists(out):
        continue
    p = hf_hub_download("yifengzhu-hf/LIBERO-datasets", f, repo_type="dataset", local_dir=f"{OUT}/raw")
    with h5py.File(p, "r") as h:
        data = {}
        demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for i, k in enumerate(demos):
            d = h["data"][k]
            for key in ["actions", "states", "robot_states"]:
                if key in d: data[f"{k}/{key}"] = d[key][:].astype(np.float32)
            for key in ["ee_pos", "ee_ori", "gripper_states", "joint_states"]:
                data[f"{k}/{key}"] = d["obs"][key][:].astype(np.float32)
            if i < 2:
                data[f"{k}/agentview_rgb"] = d["obs"]["agentview_rgb"][:]
        meta = dict(problem_info=h["data"].attrs.get("problem_info", ""), bddl=h["data"].attrs.get("bddl_file_name", ""), n=len(demos))
    np.savez_compressed(out, meta=json.dumps(meta, default=str), **data)
    os.remove(p)
    print("done", f, flush=True)
