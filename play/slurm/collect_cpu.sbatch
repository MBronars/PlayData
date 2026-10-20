#!/bin/bash
# LIBERO play collection on CPU-only nodes (no cameras; images are rendered later from the saved states with
# render_from_states.py on any GPU node). One array task = one block of PER_TASK episodes, collected by collect.py's
# own worker pool on this task's cores. Episodes are bit-identical to a camera run with the same seed (checked:
# checks/compare_episodes.py), and existing episodes are skipped, so a resubmission resumes.
# Seeds reproduce an episode only within one CPU family: on a node without AVX-512 (e.g. babel-x9-28) MuJoCo and
# numpy take other SIMD paths and the same seed gives a different (equally valid) trajectory. The constraint keeps
# every task on AVX-512 nodes (babel-l9-16, babel-l9-20, babel-z9-16), where it matches our GPU nodes bit for bit.
#
#   OUT=/data/.../play_v1/default SEED=11 PER_TASK=100 sbatch --array=0-14%10 slurm/collect_cpu.sbatch
#   OUT=/data/.../play_v1/hierr   SEED=12 PER_TASK=100 EXTRA="--p_err 1 --err_severity 1" sbatch --array=0-4%10 slurm/collect_cpu.sbatch
#SBATCH -p katef-cpu
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J play_cpu
#SBATCH --constraint=avx512f
#SBATCH -o %x_%A_%a.out
set -eu
: "${OUT:?set OUT}" "${SEED:?set SEED}"
PER_TASK=${PER_TASK:-100}; EXTRA=${EXTRA:-}
PY=${PY:-/data/user_data/mbronars/.miniconda3/envs/mcts/bin/python}
export LIBERO_REF=${LIBERO_REF:-/data/user_data/mbronars/AgenticExploration/libero_ref}
export MUJOCO_GL=glx PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")/.."
START=$((SLURM_ARRAY_TASK_ID * PER_TASK))
echo "host $(hostname) task $SLURM_ARRAY_TASK_ID episodes $START..$((START + PER_TASK - 1)) seed $SEED out $OUT $(date)"
$PY collect.py --mode composed --img 0 --videos 0 --out "$OUT" --seed "$SEED" --start "$START" --episodes "$PER_TASK" \
    --workers "$SLURM_CPUS_PER_TASK" $EXTRA
echo "done $(date)"
