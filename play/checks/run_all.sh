#!/bin/bash
# Quality checks on one or more collected dataset directories.
#   PY=/path/to/python bash checks/run_all.sh DIR [DIR ...]
#
# actdist      commanded actions vs. the LIBERO human demos (per-dim quantiles,
#              rotation, step-to-step change, share of steps outside the human range)
# failaudit    failed grasps: real close-and-lift attempts vs. back-offs
# freeaudit    time in free space / outside any labelled action, longest stretches
# gaps         what fills the time between labelled actions
# apprdur      approach durations (compare: human ~51 steps for ~28 cm)
# retreatscan  longest retreat / posture recovery per episode (flagged > 50 steps)
set -euo pipefail
DIRS=(); for d in "$@"; do DIRS+=("$(realpath "$d")"); done
set -- "${DIRS[@]}"
cd "$(dirname "$0")"
PY=${PY:-python}
echo "== action distribution vs human";  "$PY" actdist.py "$@" 2>&1 | grep -v Warning
echo "== failed grasp attempts";         "$PY" failaudit.py "$@"
echo "== free space";                    "$PY" freeaudit.py "$@"
echo "== time between actions";          "$PY" gaps.py "$@"
echo "== approach durations";            "$PY" apprdur.py "$@"
echo "== long retreats (>60 steps)";     "$PY" retreatscan.py "$@" | awk '{n=$6+0; if (n>60) print}'
