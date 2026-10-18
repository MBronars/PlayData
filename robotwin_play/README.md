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

Each episode runs in its own process (plan and record in one pass, ~5-12 min
for ~60 s of play). Scenes and the opening actions replay identically from the
seed (the planner is seeded per plan), but small floating-point differences in
cuRobo's GPU kernels make later trajectories drift between runs: the recorded
files, not a re-run, are the exact data. Output:

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
grasp points (often one of them a tool: hammer, mallet, stamp, pen), often a
container (half the time near the middle, where both hands reach it), and in
about half the scenes one articulated fixture placed as in RoboTwin's own
tasks: a microwave (open_microwave), a cabinet drawer (put_object_cabinet), a
laptop (open_laptop) or a switch (turn_switch). Spawn spacing uses each
model's real footprint.

Skills, one per family of RoboTwin's 50-task suite: move to another spot, put
in a container (passed between the arms when one arm can't reach both), stack
on a flat-topped thing, line up next to something, hand over between the arms
(tall things: the other hand takes it below; long things: by the other end),
push, turn around, press down on something (a click), shake (bottle-like
things, a tilt like shake_bottle), tap one object with another (hammer on
block, stamp on card), pick two things up at the same time, lift a container
with both hands, tip a container out, open / close the microwave door, the
drawer and the laptop lid, flip the switch, put something into the open
drawer. Which one comes next is weighted by novelty (objects not yet touched,
kinds not yet done, not the same kind twice in a row); an action that can't be
planned is not drawn again in that episode.

Every motion is planned first with RoboTwin's planner (cuRobo), checked, and
then that same plan is executed through RoboTwin's `move()` in its replay
mode, so what was checked is what gets recorded. The planners use a
planning-only copy of the robot (`env/make_play_planner_cfg.py`) whose joint
limits are the released expert data's per-joint range + 0.06 rad
(`expert_joint_ranges_aloha-agilex.json`); RoboTwin's own URDF has placeholder
limits of +-10 rad, which let the planner wind the wrist to poses the expert
data never uses.

Details that matter:
* Put-downs keep the object as it is held: its middle over the spot, its
  bottom onto the surface. (RoboTwin's `get_place_pose` turns an object back
  to its model's upright pose about its origin; a pen lying on its side, whose
  origin is at one end, would be stood on end.) Positions and success checks
  use the middle of the object's box, not its origin.
* Grasps keep the fingertips 3.5 mm above the table: an annotated grasp on
  something low puts them at table height, and closing then rolls a pen.
* Long uniform things (pens, remote, toothpaste) can also be grasped anywhere
  along their length (the annotated grasp slid along the long axis).
* Fixtures: the hand grasps the annotated handle (or, for the laptop, the
  lid's top edge from above) and follows the joint's own arc or slide, in
  short planned segments; if the handle slips it grabs it again once; a lid
  left part-way is pressed shut from above.

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
hand-over, a door / drawer / lid handle slipping out of the hand part-way
(mild: grabbed again; severe: left), a switch press beside the rocker. Severe draws combine two to four of these in one action, and half
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
* `python checks/bench.py OUT SKILL[:FIXTURE] ... --seeds 100-109`: each skill
  on several random scenes in parallel, no rendering; success / skip counts.
* `PLAY_DEBUG=1 python checks/dbg_skill.py SEED SKILL N` (from the RoboTwin
  root): runs one skill N times on a random scene without recording and prints
  the planning trace (`FORCE_ERROR=type` injects an error, `TRACK=obj` prints
  the object and hand positions every few steps, `PLAY_NO_OBSTACLES=1` plans
  without the scene objects as obstacles, `FIXTURE=044_microwave` puts that
  fixture in the scene, `OBJECTS=a,b` / `ONLY=a` force objects / candidates).

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
