"""Natural near-failures, injected into a fraction of the play (for learning
what almost-working looks like). Each injected error is recorded on the
action as info["error"] = {"type": ..., params}.

  grasp_offset   grasp point 1.5-3 cm off along the closing axis (shallow / one-sided hold)
  grasp_high     grasp point 1.5-2.5 cm too high (tips of the fingers only)
  unstable_grasp one of the worst-ranked grasp candidates instead of a good one
  close_early    fingers start closing 3-6 cm before the hand arrives
  close_late     the hand overshoots the grasp point (nudging the object) before closing
  lift_early     lifting starts before the fingers have finished closing
  place_offset   set down 3.5-6 cm away from where it was meant to go (misses a bin)
  drop_high      let go 5-9 cm above the surface
  release_early  the hand opens while still carrying
  shaky          smooth (temporally correlated) noise on the arm actions for the whole action
  jitter         a short burst of stronger smooth noise somewhere in the action
  push_offset    the push starts 1.5-3 cm to the side (the object spins away)
  drawer_miss    the hook/pinch on the drawer handle is 1-2 cm off
  knock_over     (severe only) set down partly on top of a neighbour, which gets
                 knocked over or shoved

Severe errors on picks are combinations, one from each stage: a bad or
unstable grasp, then hitting something with the held object on the way and/or
a shaky hand, then a bad finish (let go mid-carry, set down half on a
neighbour, dropped from high, set down off target), e.g. an unstable grasp,
a knock against the mug, then the box slips out over the bin's rim. The parts
are listed in e["parts"] (use has(e, t) to test for one); e["type"] joins them
with "+".

With probability `severity` an error is drawn from much larger ranges
(e["severe"] = True): grasps 3.5-5.5 cm off or 3-5 cm high (usually a miss),
closing 7-12 cm early, overshooting 2.5-4 cm (shoving the object over), lifting
with the fingers still open, set down 7-13 cm off, dropped from 12-20 cm,
let go a quarter to half way through the carry, a stronger shake / jerk, pushes
3.5-6 cm off, drawer hooks 2.5-4 cm off. Severe errors also let the hand and
the held object plow through loose objects (up to 4 cm of overlap) on the way.
"""
import numpy as np

PICK = ("move", "place_rel", "put_in", "take_out", "stack", "reorient", "put_on_fixture", "put_in_drawer",
        "line_up", "tower", "bump_with", "push_with", "door_sweep")
SEVERE_PICK_EXTRA = {"knock_over": 1.4}
PICK_ERRORS = {"grasp_offset": 1.0, "grasp_high": 0.7, "unstable_grasp": 0.8, "close_early": 0.8,
               "close_late": 0.8, "lift_early": 0.8, "place_offset": 1.2, "drop_high": 0.8,
               "release_early": 0.6, "shaky": 0.45, "jitter": 0.45}
PUSH_ERRORS = {"push_offset": 1.5, "shaky": 0.4, "jitter": 0.4}
DRAWER_ERRORS = {"drawer_miss": 1.5, "shaky": 0.35, "jitter": 0.35}
OTHER_ERRORS = {"shaky": 1.0, "jitter": 1.0}


def sample_error(rng, kind, p_err=0.3, severity=0.0):
    import os
    forced = os.environ.get("FORCE_ERROR")
    noise_only = kind not in PICK and kind not in ("push", "push_into", "topple", "drawer")
    # actions that can only get noise (knobs, doors, reaches) get it less often:
    # noise could be added to anything, it shouldn't be on everything
    if not forced and rng.random() >= p_err * (0.35 if noise_only else 1.0):
        return None
    if kind in PICK:
        table = PICK_ERRORS
    elif kind in ("push", "push_into", "topple"):
        table = PUSH_ERRORS
    elif kind == "drawer":
        table = DRAWER_ERRORS
    else:
        table = OTHER_ERRORS
    severe = bool(rng.random() < severity)
    if severe and table is PICK_ERRORS:
        table = dict(PICK_ERRORS, **SEVERE_PICK_EXTRA)
    names = list(table)
    w = np.array([table[k] for k in names])
    t = names[int(rng.choice(len(names), p=w / w.sum()))]
    if forced:
        if forced == "knock_over" and table is PICK_ERRORS:
            names.append(forced)
        if forced not in names:
            return None
        t = forced
        severe = severe or forced == "knock_over" or bool(os.environ.get("FORCE_SEVERE"))
    e = {"type": t}
    if severe:
        e["severe"] = True
        if table is not PICK_ERRORS and "knock_over" not in table and not forced:
            return _severe_simple(rng, e, table)
        if not forced:
            return _combo(rng)
        return _severe(rng, e)
    if t == "grasp_offset":
        e["offset"] = float(rng.choice([-1, 1]) * rng.uniform(0.015, 0.03))
    elif t == "grasp_high":
        e["offset"] = float(rng.uniform(0.015, 0.025))
    elif t == "close_early":
        e["early"] = float(rng.uniform(0.03, 0.06))
    elif t == "close_late":
        e["overshoot"] = float(rng.uniform(0.01, 0.02))
    elif t == "place_offset":
        a = rng.uniform(-np.pi, np.pi)
        e["offset_xy"] = (np.r_[np.cos(a), np.sin(a)] * rng.uniform(0.035, 0.06)).round(4).tolist()
    elif t == "drop_high":
        e["drop"] = float(rng.uniform(0.05, 0.09))
    elif t == "release_early":
        e["frac"] = float(rng.uniform(0.6, 0.85))
    elif t == "shaky":
        e["sigma"] = float(rng.uniform(0.05, 0.1))
    elif t == "jitter":
        e["sigma"] = float(rng.uniform(0.15, 0.25))
        e["start"] = int(rng.integers(15, 90))
        e["len"] = int(rng.integers(8, 22))
    elif t == "push_offset":
        e["offset"] = float(rng.choice([-1, 1]) * rng.uniform(0.015, 0.03))
    elif t == "drawer_miss":
        e["offset"] = (rng.normal(0, 1, 2) / np.sqrt(2) * rng.uniform(0.01, 0.02)).round(4).tolist()
    return e


def _severe(rng, e):
    t = e["type"]
    if t == "grasp_offset":
        e["offset"] = float(rng.choice([-1, 1]) * rng.uniform(0.035, 0.055))
    elif t == "grasp_high":
        e["offset"] = float(rng.uniform(0.03, 0.05))
    elif t == "close_early":
        e["early"] = float(rng.uniform(0.07, 0.12))
    elif t == "close_late":
        e["overshoot"] = float(rng.uniform(0.025, 0.04))
    elif t == "place_offset":
        a = rng.uniform(-np.pi, np.pi)
        e["offset_xy"] = (np.r_[np.cos(a), np.sin(a)] * rng.uniform(0.07, 0.13)).round(4).tolist()
    elif t == "drop_high":
        e["drop"] = float(rng.uniform(0.12, 0.2))
    elif t == "release_early":
        e["frac"] = float(rng.uniform(0.25, 0.5))
    elif t == "shaky":
        e["sigma"] = float(rng.uniform(0.1, 0.2))
    elif t == "jitter":
        e["sigma"] = float(rng.uniform(0.25, 0.4))
        e["start"] = int(rng.integers(15, 90))
        e["len"] = int(rng.integers(12, 28))
    elif t == "push_offset":
        e["offset"] = float(rng.choice([-1, 1]) * rng.uniform(0.035, 0.06))
    elif t == "drawer_miss":
        e["offset"] = (rng.normal(0, 1, 2) / np.sqrt(2) * rng.uniform(0.025, 0.04)).round(4).tolist()
    elif t == "knock_over":
        e["overlap"] = float(rng.uniform(0.02, 0.045))
    return e


def has(e, t):
    """Does the injected error (possibly a combination) include type t?"""
    return bool(e) and (e.get("type") == t or t in e.get("parts", ()))


GRASP_STAGE = {"grasp_offset": 1.0, "grasp_high": 0.8, "close_early": 0.8, "lift_early": 0.6,
               "close_late": 0.6, "unstable_grasp": 1.2}
FINISH_STAGE = {"knock_over": 1.4, "release_early": 1.0, "drop_high": 1.0, "place_offset": 0.8}


def _pick(rng, table):
    k = list(table)
    w = np.array([table[x] for x in k])
    return k[int(rng.choice(len(k), p=w / w.sum()))]


def _combo(rng):
    """Two or three mistakes stacked in one pick-and-place."""
    parts = []
    if rng.random() < 0.7:
        parts.append(_pick(rng, GRASP_STAGE))
    hit = rng.random() < 0.5
    shaky = rng.random() < 0.15
    finish = _pick(rng, FINISH_STAGE) if rng.random() < 0.85 else None
    if len(parts) + hit + (finish is not None) < 2:
        # at least two mistakes - but not by padding it out with noise
        if finish is None:
            finish = _pick(rng, FINISH_STAGE)
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
        d = _severe(rng, {"type": t})
        d.pop("type")
        e.update(d)
    if "shaky" in parts and "unstable_grasp" not in parts:
        e["sigma"] = float(rng.uniform(0.08, 0.15))   # on top of other trouble
    if "close_late" in parts:
        e["overshoot"] = float(rng.uniform(0.04, 0.06))
    return e


def _severe_simple(rng, e, table):
    """Severe pushes / drawers / fixtures: the main mistake plus a shaky hand."""
    t = e["type"]
    e = _severe(rng, e)
    if t not in ("shaky", "jitter") and rng.random() < 0.2:
        e["parts"] = [t, "shaky"]
        e["type"] = f"{t}+shaky"
        e["sigma"] = float(rng.uniform(0.08, 0.15))
    return e
