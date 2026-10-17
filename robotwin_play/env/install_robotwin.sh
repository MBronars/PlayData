#!/bin/bash
# Install RoboTwin 2.0 (pinned commit) into a conda env "robotwin" for robotwin_play.
# Assets and reference data: env/download_assets_and_reference.py (run after this).
set -euxo pipefail
CONDA=/data/user_data/mbronars/.miniconda3; CONDA_BIN=/home/mbronars/miniconda3/bin/conda
ROOT=/data/user_data/mbronars/packages/RoboTwin
if [ ! -d $ROOT ]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git $ROOT
  cd $ROOT && git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
fi
cd $ROOT
[ -d $CONDA/envs/robotwin ] || $CONDA_BIN create -y -p $CONDA/envs/robotwin python=3.10
PY=$CONDA/envs/robotwin/bin/python
$PY -m pip install --upgrade pip
# simulation requirements (skip the description-generation / logging extras)
grep -v -E "^(azure|openai|wandb|#)" scripts/requirements.txt | grep -v '^$' > /tmp/rt_req_$$.txt
$PY -m pip install -r /tmp/rt_req_$$.txt --extra-index-url https://download.pytorch.org/whl/cu121
# their two library patches (same sed lines as scripts/_install.sh)
SAPIEN_LOCATION=$($PY -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN_LOCATION/wrapper/urdf_loader.py
MPLIB_LOCATION=$($PY -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB_LOCATION/planner.py
# cuRobo v0.7.8. This machine has no nvcc for CUDA 12.1 (torch's build); the
# 12.9 compiler works (same major version). L40S = compute capability 8.9.
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9"
cd envs
[ -d curobo ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
$PY -m pip install -e . --no-build-isolation
# cuRobo's install pulls newer scipy/warp; restore RoboTwin's pins
$PY -m pip install scipy==1.10.1 numpy==1.26.4 warp-lang==1.12.0 setuptools==69.5.1
cd $ROOT
# asset paths in the embodiment configs, then link the play task in
$PY scripts/update_embodiment_config_path.py
ln -sf "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/play_table.py" $ROOT/envs/play_table.py
# planning-only robot configs (needs the assets: run env/download_assets_and_reference.py first)
[ -d $ROOT/assets/embodiments/aloha-agilex ] && $PY "$(dirname "${BASH_SOURCE[0]}")/make_play_planner_cfg.py"
echo INSTALL_DONE
