# LIBERO play data

Generates unscripted "play" episodes in LIBERO: random scenes and objects, a
robot that explores them (moves, stacks, lines things up, pushes, knocks one
thing with another, opens drawers / the microwave, turns the stove on, grabs
things in different ways), with injected near-failures and failures, and
actions kept in the distribution of the LIBERO human demos (OSC_POSE deltas,
20 Hz). Each episode is 60 s (1200 steps) and is labelled with the actions it
contains.

## Setup

* Python 3.10 env with the packages in `env/pip_freeze.txt` (key versions in
  `env/versions.txt`: mujoco 3.3.7, robosuite 1.4.1, numpy 2.1.2).
* LIBERO at commit `8f1084e` and robosuite at `b9d8d3d` (see `env/versions.txt`),
  **with the local patches** `env/LIBERO.patch` and `env/robosuite.patch`
  applied (`git apply env/LIBERO.patch` in the LIBERO checkout, same for
  robosuite). These were part of the environment the data was made with.
* Headless rendering: `MUJOCO_GL=egl`.

## Reference data (human demos)

The generator samples human motion residuals and style from the LIBERO human
demos, stored in `../libero_ref` (not versioned, 1.5 GB):

1. `python fetch_ref.py` downloads the LIBERO datasets from Hugging Face
   (`yifengzhu-hf/LIBERO-datasets`) and keeps the low-dim parts in
   `libero_ref/lowdim/*.npz`.
2. `python replay_ref.py libero_ref/lowdim/<file>.npz` for each file writes
   `libero_ref/replay/<file>.pkl` (object / fixture poses along the demo).
3. Check against the copy used for the data: `cd ../libero_ref && sha256sum -c ../play/env/libero_ref.sha256`.

Set `LIBERO_REF=/path/to/libero_ref` to use another location.

## Collecting

```
cd play
MUJOCO_GL=egl python collect.py --mode composed --out OUT --episodes N --workers 8 --seed S
```

| flag | meaning |
|---|---|
| `--seed S`, `--start K` | episode `i` uses seed `S*100003 + i` (i = K..K+N-1); same seed → identical episode |
| `--scene kitchen\|living_room\|study`, `--fixture flat_stove\|microwave\|white_cabinet\|wooden_cabinet` | force a scene / fixture (default random) |
| `--p_err` (0.3) | fraction of actions with an injected error |
| `--err_severity` (0.15) | fraction of those that are large and obvious (combinations: bad grasp + knocking something on the way + drop / knock-over) |
| `--img` (128) | camera resolution |

Every episode runs in its own process with `PYTHONHASHSEED=0`, so an episode
is bit-identical whether it is collected alone or in a batch, with any
`--workers`.

Output: `OUT/episode_XXXXXX.hdf5` (+ `OUT/bddl/`, `OUT/log.jsonl`):

```
data/demo_0/
  actions            (T, 7)  OSC_POSE deltas + gripper ±1
  states             (T, S)  MuJoCo sim states
  obs/agentview_rgb, eye_in_hand_rgb (T, H, W, 3) uint8
  obs/ee_pos, ee_ori, ee_states, gripper_states, joint_states, object_states
  primitive_id, segment_id, phase_id  (T,)
  attrs: bddl, model_file, scene, seed, phases,
         segments: list of actions with start/end step, primitive, target,
                   language, success, error (injected error or null),
                   truncated (cut off by the 60 s end), abandoned (gave up partway),
                   diag (grasp type, distances, ...)
```

## Reproducing the showcase sets

```
PY=python bash make_showcase.sh ..
```

regenerates `../showcase_play` (21 episodes, default error settings; 16 random
scenes + 2 stove + 3 microwave) and `../showcase_hierr` (8 episodes, both error
knobs at 1.0) and runs the checks. Takes ~45 min on 8 cores.
`python artifact/build_page.py` rebuilds the showcase web page from them into
`../artifact_composed/`.

## Checks

```
PY=python bash checks/run_all.sh DIR [DIR ...]
```

* `actdist.py`: commanded actions vs. all LIBERO human demo steps (per-dim
  quantiles, rotation, step-to-step change, share of steps outside the human
  0.1–99.9% range). Current data: ~2% of steps outside (human data itself 0.7%),
  no step-to-step change beyond the human 99.9th percentile.
* `failaudit.py`: failed grasps should be real close-and-lift attempts.
* `freeaudit.py`, `gaps.py`: free-space time, time outside any labelled action.
* `apprdur.py`: reach durations vs. human.
* `retreatscan.py`: longest retreat / posture recovery per episode (flag > 60 steps).
* `sheet.py DIR EP A B STEP OUT.png`: contact sheet of frames A..B.

## Code map

| file | role |
|---|---|
| `collect.py` | episode loop, workers, HDF5 writing |
| `scene_gen.py` | random scenes (BDDL), object placement |
| `play_controller.py` | env wrapper: stepping, rate limit, injected noise, object queries |
| `compose.py` | keyframe splines → human-like actions (lag model, residuals, turn-rate limit, joint guard, stall stop) |
| `composed_play.py` | pick / place / carry / retreat planning, grasping, posture recovery, episode loop |
| `play_tasks.py` | task choice (variety, follow-ups), pushes, wall grasps for containers |
| `creative.py`, `interactions.py`, `fixtures_play.py` | line-ups / towers / push-with, knocks / door sweep, drawers / microwave / stove |
| `grasp_sampler.py` | grasps from collision geometry (incl. handle and side grasps) |
| `errors.py` | injected error types, severity, combinations |
| `collision.py`, `reach.py` | clearance queries, IK and joint margins |
| `merge_sets.py`, `make_showcase.sh`, `checks/`, `artifact/` | reproduction, checks, showcase page |

Other files (`human_play.py`, `human_path_play.py`, `style_*`, `retarget.py`,
`test_*`, ...) belong to earlier modes (`--mode humanpath|human|scripted`) and
experiments.

## Known limitations

* A few episodes still have a 3–4.5 s posture recovery after a fast wrist swing.
* Hand-tuned for LIBERO's Panda and table layout (comfortable reach 47–62 cm
  from the base, wrist range from the human demos, fixture poses).
* Reaches are ~1.3× slower than the humans' and translation commands somewhat
  smaller (median 0.36 vs 0.59).
