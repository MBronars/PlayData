# RoboTwin 2.0 play data (work in progress)

Play data for RoboTwin 2.0 in RoboTwin's own scripted-expert action format, so
it pairs with policies trained on RoboTwin's released data. Every motion goes
through RoboTwin's planner (cuRobo) and `move()` API, and episodes are recorded
by RoboTwin's recorder.

## Setup

* `bash env/install_robotwin.sh`: clones RoboTwin at the pinned commit to
  `/data/user_data/mbronars/packages/RoboTwin`, creates the conda env
  `robotwin`, installs RoboTwin's requirements and its two library patches,
  builds cuRobo, links `play_table.py` into `RoboTwin/envs/`.
* `python env/download_assets_and_reference.py` (any env with huggingface_hub):
  RoboTwin assets (15 GB) and released expert data for 10 tasks, aloha-agilex
  and franka clean_50 (8.4 GB, `/data/user_data/mbronars/robotwin_ref`).
* `python env/make_play_planner_cfg.py` (robotwin env): the planning-only
  robot configs with expert-range joint limits (also run by the install script).
* Versions: `env/versions.txt`, `env/pip_freeze.txt`.

## Collecting

```
/data/user_data/mbronars/.miniconda3/envs/robotwin/bin/python collect_play.py --out OUT --episodes N --workers 3 --seed S
```

Each episode runs in its own process (plan and record in one pass, ~5 min for
~60 s of play). Output:

* `OUT/data/episode_XXXXXXX.hdf5`: RoboTwin's current layout (`state/`, `action/`,
  `vision/`; action[t] = state[t+1]).
* `OUT/data_released/episodeN.hdf5`: the layout of RoboTwin's released datasets
  (`joint_action/vector`, `endpose/`, `observation/*_camera/rgb`, ...), same data.
* `OUT/_traj_data/`: the planned joint paths; `OUT/video/`: head-camera videos;
  `OUT/log.jsonl`: scene and labelled actions (primitive, language, success,
  start/end frame, arm, grasp point).

Keep `--workers` at about 3: rendering shares the GPU.

Errors: `--p_err P` is the fraction of actions that get an injected mistake
(default 0.3; `--p_err 0` for clean play) and `--err_severity S` the fraction
of those that are large or combined (default 0.15; `--p_err 1 --err_severity 1`
for a high-error set). The mistake is recorded on the action in `log.jsonl`
as `error: {type, parts, severe, ...params}`. Episodes are reproducible from
`--seed` and the episode index (one fresh process per episode).

## Play

`play_table.py`: 3-5 random objects from RoboTwin's library that have annotated
grasp points (often one of them a tool: hammer, mallet, stamp, pen), plus often
a container. Spawn spacing uses each model's real footprint.

Skills, one per family of RoboTwin's 50-task suite: move to another spot, put
in a container (passed between the arms when one arm can't reach both), stack
on a flat-topped thing, line up next to something, hand over between the
arms, push, turn around, press down on something (a click), shake, tap one
object with another (hammer on block, stamp on card), pick two things up at
the same time with both arms, lift a container with both hands, tip a
container out. Which one comes next is weighted by novelty (objects not yet
touched, kinds not yet done, not the same kind twice in a row).

Every motion is planned first with RoboTwin's planner (cuRobo), checked, and
then that same plan is executed through RoboTwin's `move()` in its replay
mode, so what was checked is what gets recorded. The planners use a
planning-only copy of the robot (`env/make_play_planner_cfg.py`) whose joint
limits are the released expert data's per-joint range + 0.06 rad
(`expert_joint_ranges_aloha-agilex.json`); RoboTwin's own URDF has placeholder
limits of +-10 rad, which let the planner wind the wrist to poses the expert
data never uses. Put-downs are retried with the object turned 90/180 deg
about the target when the wrist can't reach the first orientation.

## Errors

`errors.py` is the RoboTwin counterpart of the LIBERO module. Grasp stage:
off-centre grasp, fingertips only, a badly ranked grasp point, closing early
(then pushing on), overshooting into the object, snatching (lifting while
closing). Carry stage: swinging the held object through a neighbour, a
wobbling hand (smooth low-frequency noise on the joint targets, used
sparingly), a short jitter burst. Finish stage: setting down off target,
letting go too high, opening part-way through the carry, setting down half on
a neighbour so it topples. Other skills: a push that starts to one side, a
press or a tap beside the target, a receiving hand that closes early in a
hand-over. Severe draws combine two to four of these in one action, and half
of the failed actions are followed by a clean retry. The hooks live in
`play_table.py` next to the motion they change; every erroneous motion still
goes through the planner and the expert joint range.

## Checks

* `python checks/actdist.py OUT`: joint targets and steps compared with the
  released expert data (step percentiles, idle fraction, gripper toggles,
  frames outside the expert joint range).
* `python checks/tally.py OUT ...`: per-skill success counts and, per injected
  error type, how many of those actions still succeeded.
* `python checks/sheet.py OUT EP OUT.png [nseg]`: contact sheet of one episode
  (8 frames per action from the head-camera video).
* `PLAY_DEBUG=1 python checks/dbg_skill.py SEED SKILL N` (from the RoboTwin
  root): runs one skill N times on a random scene without recording and prints
  the planning trace (`FORCE_ERROR=type` injects an error, `TRACK=obj` prints
  the object and hand positions every few steps, `PLAY_NO_OBSTACLES=1` plans
  without the scene objects as obstacles).

## Obstacles

RoboTwin's cuRobo planners know only the table, so an arm on its way to the
next object could sweep a neighbour off the table. The play task loads the
scene objects into the planner as boxes before every plan (`_world`): the
objects the action handles are left out, so are small neighbours within 12 cm
of the plan's target (the hand has to get close to them anyway); big things
(containers) always stay.

## Showcase page

`python artifact/build_media.py RUN_DIR[:ep,ep] ...` (robotwin env) writes
2x2 camera-grid videos, `data.json` and `index.html` to
`../artifact_robotwin/` from `artifact/template.html`.
