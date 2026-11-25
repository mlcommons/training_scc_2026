#!/bin/bash
# Pull the MLPerf log out of a finished run's Slurm output, summarise it and lint it.
#
#   nvidia/extract_result.sh <run_dir> [dest.txt]
#
# Default dest is <run_dir>/result.txt. Each run has its own RUN_DIR holding exactly
# one slurm-<jobid>.out (never share one -- see CLAUDE.md on O_APPEND over NFS), so the
# job id does not have to be passed in. Prints "status / converged at / time to train"
# and the last lines of the compliance checker. Exits non-zero if the run did not reach
# the target or the log does not lint, which is what makes a failed run visible to
# run_final.sh and to run_competition.sh's collector job.
set -uo pipefail
RUN_DIR="${1:?usage: extract_result.sh <run_dir> [dest.txt]}"
DEST="${2:-$RUN_DIR/result.txt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

shopt -s nullglob
logs=("$RUN_DIR"/slurm-*.out)
shopt -u nullglob
[ ${#logs[@]} -eq 1 ] || { echo "expected one slurm-*.out in $RUN_DIR, found ${#logs[@]}" >&2; exit 1; }

# The trainer's progress bars leave bare CRs in the Slurm log.
tr '\r' '\n' < "${logs[0]}" | grep '^:::MLLOG' > "$DEST" || true
[ -s "$DEST" ] || { echo "no :::MLLOG lines in ${logs[0]}" >&2; exit 1; }

python3 - "$DEST" <<'PY'
import json, sys
ev = [json.loads(l[l.index("{"):]) for l in open(sys.argv[1])]
t0 = next((e["time_ms"] for e in ev if e["key"] == "run_start"), None)
stop = next((e for e in ev if e["key"] == "run_stop"), None)
evals = [(e["metadata"]["samples_count"], e["value"]) for e in ev if e["key"] == "eval_accuracy"]
conv = next((s for s, v in evals if v <= 0.925), None)
print("status:", stop["metadata"]["status"] if stop else "no run_stop")
print("converged at:", conv if conv else "-", "samples")
if stop and t0: print("time to train: %.2f min" % ((stop["time_ms"] - t0) / 60000))
sys.exit(0 if (stop and stop["metadata"].get("status") == "success" and conv) else 1)
PY
rc=$?

"$ROOT/scripts/check_compliance.sh" "$DEST" | tail -3 || rc=1
exit $rc
