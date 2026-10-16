#!/bin/bash
# Regenerate the two showcase datasets exactly (same seeds -> identical HDF5
# actions/states), then run the quality checks on them.
#
#   PY=/path/to/python bash make_showcase.sh [OUT_ROOT]     (default OUT_ROOT: ..)
#
# Produces OUT_ROOT/showcase_play   (default error settings: --p_err 0.3 --err_severity 0.15)
#          OUT_ROOT/showcase_hierr  (both error knobs at maximum)
# Each episode runs in its own process, so the --workers count does not change
# the data. ~45 min on 8 cores.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
ROOT=${1:-..}
export MUJOCO_GL=${MUJOCO_GL:-egl}
W=${WORKERS:-8}
run() { "$PY" collect.py --mode composed --workers "$W" "$@"; }

for d in showcase_play showcase_hierr _sc_fx_s _sc_fx_m _hi_fx; do
  if [ -e "$ROOT/$d" ]; then echo "$ROOT/$d exists; move it away first" >&2; exit 1; fi
done

# clean play: 16 random scenes + 2 stove + 3 microwave scenes
run --out "$ROOT/showcase_play" --episodes 16 --seed 21
run --out "$ROOT/_sc_fx_s" --episodes 2 --seed 41 --scene kitchen --fixture flat_stove
run --out "$ROOT/_sc_fx_m" --episodes 3 --seed 42 --scene kitchen --fixture microwave
"$PY" merge_sets.py "$ROOT/showcase_play" 16="$ROOT/_sc_fx_s:0" 17="$ROOT/_sc_fx_s:1" \
    18="$ROOT/_sc_fx_m:0" 19="$ROOT/_sc_fx_m:1" 20="$ROOT/_sc_fx_m:2"

# high error: 6 random scenes + 2 stove scenes, every action gets an error, all severe
run --out "$ROOT/showcase_hierr" --episodes 6 --seed 78 --p_err 1.0 --err_severity 1.0
run --out "$ROOT/_hi_fx" --episodes 2 --seed 43 --scene kitchen --fixture flat_stove --p_err 1.0 --err_severity 1.0
"$PY" merge_sets.py "$ROOT/showcase_hierr" 6="$ROOT/_hi_fx:0" 7="$ROOT/_hi_fx:1"

rm -rf "$ROOT/_sc_fx_s" "$ROOT/_sc_fx_m" "$ROOT/_hi_fx"
bash checks/run_all.sh "$ROOT/showcase_play" "$ROOT/showcase_hierr"
