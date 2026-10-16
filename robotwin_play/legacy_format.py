"""Write an episode in the layout of RoboTwin 2.0's *released* datasets
(HuggingFace TianxingChen/RoboTwin2.0, dataset/<task>/<embodiment>_clean_50):

    joint_action/{left_arm, left_gripper, right_arm, right_gripper, vector}
    endpose/{left_endpose, left_gripper, right_endpose, right_gripper}
    observation/<head|left|right|front>_camera/{rgb, intrinsic_cv, extrinsic_cv, cam2world_gl}
    pointcloud

one entry per recorded frame (every frame, nothing dropped), images as
"legacy" JPEG bytes (cv2.imencode of the RGB array, read back with
cv2.imdecode). The current RoboTwin code writes a newer layout
(state/action/vision, last frame dropped); both come from the same per-frame
recordings, so they hold the same data."""
import os
import pickle

import cv2
import h5py
import numpy as np


def _frames(cache_dir):
    files = sorted((int(f[:-4]), f) for f in os.listdir(cache_dir) if f.endswith(".pkl") and f[:-4].isdigit())
    for i, (k, f) in enumerate(files):
        if k != i:
            raise ValueError(f"missing frame {i}.pkl in {cache_dir}")
        with open(os.path.join(cache_dir, f), "rb") as fh:
            yield pickle.load(fh)


def _collect(frames):
    out = None

    def walk(d, acc):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, acc.setdefault(k, {}))
            else:
                acc.setdefault(k, []).append(v)
    for fr in frames:
        if out is None:
            out = {}
        walk(fr, out)
    return out


def _encode(images):
    enc = [cv2.imencode(".jpg", np.asarray(im))[1].tobytes() for im in images]
    n = max(len(b) for b in enc)
    return np.array([b.ljust(n, b"\0") for b in enc], dtype=f"S{n}")


def _write(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _write(group.require_group(k), v)
        elif k == "rgb":
            group.create_dataset(k, data=_encode(v))
        else:
            arr = np.asarray(v)
            if arr.dtype == object:
                arr = np.asarray([np.asarray(x, dtype=np.float64) for x in v])
            group.create_dataset(k, data=arr)


def write_legacy(cache_dir, path):
    data = _collect(_frames(cache_dir))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        if not data.get("pointcloud") or all(len(np.asarray(p)) == 0 for p in data["pointcloud"]):
            n = len(data["joint_action"]["vector"])
            data["pointcloud"] = np.zeros((n, 0))
        _write(f, data)
    return path
