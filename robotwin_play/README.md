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

## Play

`play_table.py`: 3-5 random objects from RoboTwin's library that have annotated
grasp points, plus often a container. Actions: move to another spot, put in a
container (handed over between arms when one arm can't do both), stack, push,
reorient, hand over between arms. Before moving, every grasp and put-down is
checked with the planner (twice, since cuRobo is randomized) and rejected if its
joints would leave the range RoboTwin's expert data uses
(`expert_joint_ranges_aloha-agilex.json`).

## Checks

`python checks/actdist.py OUT` compares joint targets and steps with the
released expert data.
