"""Injected near-failures and failures for RoboTwin play (the RoboTwin
counterpart of play/errors.py). Each action gets one with probability p_err;
with probability `severity` it is drawn from the large ranges (and, for
picks, combined into two to four mistakes). Recorded on the action as
info["error"] = {"type": ..., "parts": [...], "severe": bool, params}.

Grasp stage          grasp_offset   fingers take the object off-centre (along the finger plane)
                     grasp_high     fingertips only (hand stops short of the grasp point)
                     unstable_grasp one of the worst-ranked grasp points
                     close_early    fingers close before the hand arrives, then it pushes on
                     close_late     the hand overshoots into the object before closing
                     lift_early     a snatch: lifting while the fingers are still closing
Carry stage          carry_hit      swings the held object through a neighbour on the way
                     shaky          smooth noise on the joint targets for the whole action
                     jitter         a short burst of stronger smooth noise
Finish stage         place_offset   set down away from the target (misses a bin, half off a base)
                     drop_high      let go well above the surface
                     release_early  the hand opens part-way through the carry
                     knock_over     set down half on top of a neighbour so something topples
Other actions        push_offset    the push starts to one side (the object spins away)
                     press_miss     presses beside the thing instead of on it
                     tap_miss       the tool comes down beside the target
                     handover_miss  the receiving hand closes early / off the object
"""
import os

import numpy as np

PICK = ("move", "put_in", "stack", "line_up", "reorient", "shake", "knock_with", "pour")
PICK_ERRORS = {"grasp_offset": 1.0, "grasp_high": 0.7, "unstable_grasp": 0.8, "close_early": 0.8, "close_late": 0.8,
               "lift_early": 0.8, "place_offset": 1.2, "drop_high": 0.8, "release_early": 0.6, "shaky": 0.45, "jitter": 0.45}
OTHER = {"push": {"push_offset": 1.5, "shaky": 0.4, "jitter": 0.4},
         "press": {"press_miss": 1.5, "shaky": 0.4, "jitter": 0.4},
         "handover": {"handover_miss": 1.0, "grasp_offset": 0.5, "release_early": 0.5, "shaky": 0.4, "jitter": 0.4},
         "dual_pick": {"grasp_offset": 1.0, "close_early": 0.6, "place_offset": 0.8, "drop_high": 0.6, "shaky": 0.4, "jitter": 0.4},
         "lift_together": {"grasp_offset": 1.0, "close_early": 0.6, "shaky": 0.5, "jitter": 0.5}}
GRASP_STAGE = {"grasp_offset": 1.0, "grasp_high": 0.8, "close_early": 0.8, "lift_early": 0.6, "close_late": 0.6, "unstable_grasp": 1.2}
FINISH_STAGE = {"knock_over": 1.4, "release_early": 1.0, "drop_high": 1.0, "place_offset": 0.8}

# parameter ranges: mild / severe
RANGES = {
    "grasp_offset": ((0.012, 0.025), (0.03, 0.05)),
    "grasp_high": ((0.012, 0.022), (0.03, 0.045)),
    "close_early": ((0.025, 0.05), (0.07, 0.11)),
    "close_late": ((0.01, 0.02), (0.025, 0.04)),
    "place_offset": ((0.035, 0.06), (0.07, 0.13)),
    "drop_high": ((0.05, 0.09), (0.12, 0.2)),
    "release_early": ((0.6, 0.85), (0.25, 0.5)),
    "shaky": ((0.008, 0.015), (0.015, 0.03)),       # rad, smooth noise on joint targets
    "jitter": ((0.02, 0.035), (0.035, 0.06)),
    "push_offset": ((0.015, 0.03), (0.035, 0.06)),
    "press_miss": ((0.03, 0.05), (0.05, 0.08)),
    "tap_miss": ((0.03, 0.05), (0.06, 0.1)),
    "knock_over": ((0.02, 0.035), (0.02, 0.045)),
}


def _pick_one(rng, table):
    k = list(table)
    w = np.array([table[x] for x in k], dtype=float)
    return k[int(rng.choice(len(k), p=w / w.sum()))]


def _params(rng, t, severe):
    if t not in RANGES:
        return {}
    lo, hi = RANGES[t][1 if severe else 0]
    e = {}
    if t in ("grasp_offset", "push_offset"):
        e["offset"] = float(rng.choice([-1, 1]) * rng.uniform(lo, hi))
    elif t == "place_offset":
        a = rng.uniform(-np.pi, np.pi)
        e["offset_xy"] = [float(np.cos(a) * rng.uniform(lo, hi)), float(np.sin(a) * rng.uniform(lo, hi))]
    elif t in ("press_miss", "tap_miss"):
        a = rng.uniform(-np.pi, np.pi)
        d = rng.uniform(lo, hi)
        e["offset_xy"] = [float(np.cos(a) * d), float(np.sin(a) * d)]
    elif t == "release_early":
        e["frac"] = float(rng.uniform(lo, hi))
    elif t == "jitter":
        e["sigma"] = float(rng.uniform(lo, hi))
        e["frac"] = float(rng.uniform(0.1, 0.7))     # where in the action the burst starts
        e["len"] = float(rng.uniform(0.6, 1.5))      # seconds
    elif t == "knock_over":
        e["overlap"] = float(rng.uniform(lo, hi))
    elif t in ("unstable_grasp", "handover_miss", "lift_early"):
        pass
    else:
        e[{"grasp_high": "offset", "close_early": "early", "close_late": "overshoot", "drop_high": "drop",
           "shaky": "sigma"}[t]] = float(rng.uniform(lo, hi))
    return e


def sample_error(rng, kind, p_err=0.3, severity=0.15):
    forced = os.environ.get("FORCE_ERROR")
    noise_only = kind not in PICK and kind not in OTHER
    if not forced and rng.random() >= p_err * (0.35 if noise_only else 1.0):
        return None
    severe = bool(rng.random() < severity) or bool(os.environ.get("FORCE_SEVERE"))
    if forced:
        severe = severe or forced == "knock_over"
        e = {"type": forced, "parts": [forced], "severe": severe}
        e.update(_params(rng, forced, severe))
        return e
    if kind in PICK and severe:
        parts = []
        if rng.random() < 0.7:
            parts.append(_pick_one(rng, GRASP_STAGE))
        hit = rng.random() < 0.5
        shaky = rng.random() < 0.15
        finish = _pick_one(rng, FINISH_STAGE) if rng.random() < 0.85 else None
        if len(parts) + hit + (finish is not None) < 2:
            if finish is None:
                finish = _pick_one(rng, FINISH_STAGE)
            else:
                hit = True
        if hit:
            parts.append("carry_hit")
        if shaky:
            parts.append("shaky")
        if finish is not None:
            parts.append(finish)
        e = {"type": "+".join(parts), "parts": parts, "severe": True}
        for t in parts:
            if t in RANGES:
                e.update(_params(rng, t, True))
        return e
    table = PICK_ERRORS if kind in PICK else OTHER.get(kind, {"shaky": 1.0, "jitter": 1.0})
    t = _pick_one(rng, table)
    e = {"type": t, "parts": [t], "severe": severe}
    e.update(_params(rng, t, severe))
    if severe and kind in OTHER and t not in ("shaky", "jitter") and rng.random() < 0.2:
        e["parts"] = [t, "shaky"]
        e["type"] = f"{t}+shaky"
        e["sigma"] = float(rng.uniform(*RANGES["shaky"][0]))
    return e


def has(e, t):
    return bool(e) and t in e.get("parts", [e.get("type")])


def smooth_noise(rng, n, sigma, dims, hz=250.0, cutoff=1.5):
    """(n, dims) smooth zero-mean noise with std ~sigma: a wobbling hand, not white noise."""
    if n < 3:
        return np.zeros((n, dims))
    w = rng.normal(0, 1, (n + 400, dims))
    k = int(hz / cutoff)
    kern = np.hanning(k)
    kern /= kern.sum()
    s = np.stack([np.convolve(w[:, d], kern, mode="same") for d in range(dims)], 1)[200:200 + n]
    s -= s.mean(0)
    sd = s.std(0) + 1e-9
    return s / sd * sigma
