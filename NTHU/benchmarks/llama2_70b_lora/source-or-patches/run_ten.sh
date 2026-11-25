#!/bin/bash
# The ten runs of a submission, one after another, in the submission tree's layout:
#
#   nvidia/run_ten.sh [OUT_DIR]      # default $WORK_ROOT/runs/ten
#
# Ten distinct seeds; result_0.txt .. result_9.txt land in OUT_DIR/results, ready to be
# copied to TEAM/results/student_system/llama2_70b_lora/. Runs are sequential because
# each uses all 14 nodes, and each has its own run directory (see CLAUDE.md on why).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
. "$ROOT/configs/site.env"
OUT="${1:-$WORK_ROOT/runs/ten}"; mkdir -p "$OUT/results"
SEEDS=(${SEEDS:-1234 2468 4680 5678 5791 3579 6802 7913 8024 9135})
[ ${#SEEDS[@]} -eq 10 ] || { echo "need exactly 10 distinct seeds" >&2; exit 1; }
[ "$(printf '%s\n' "${SEEDS[@]}" | sort -u | wc -l)" -eq 10 ] || { echo "seeds must be distinct" >&2; exit 1; }
# A run that fails -- does not converge, or trips the compliance checker -- must not
# take the other nine with it: the rules allow discarding one failed result, and on the
# competition day there is no second window. Keep going, copy whatever log exists, and
# report at the end.
failed=()
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}
  SEED=$s RUN_DIR="$OUT/seed$s" "$HERE/run_final.sh" || failed+=("$i/seed$s")
  cp "$OUT/seed$s/result_seed$s.txt" "$OUT/results/result_$i.txt" 2>/dev/null \
    || { echo "no MLPerf log for seed $s" >&2; failed+=("$i/seed$s(nolog)"); }
done
echo "wrote $OUT/results/result_0..9.txt"
[ ${#failed[@]} -eq 0 ] || { echo "runs that failed: ${failed[*]}" >&2; exit 1; }
