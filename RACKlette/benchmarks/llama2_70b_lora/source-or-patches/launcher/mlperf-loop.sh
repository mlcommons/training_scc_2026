#!/bin/bash
# mlperf-loop.sh: the command the wrapper runs (once) in cert-mlperf.sbatch: 10 MLPerf trainings back to back in the
# same allocation, same cap and fans, seeds from $CERT_A/seeds.txt. Each run = gen/run-cert.sbatch (live run.sbatch
# with per-run LOGDIR, MASTER_PORT offset, hang guard ends only that run), output kept as runs/<job>/run_<i>.out.
# A failed run is recorded and the loop goes on; after a 2nd failure it stops (MLPerf allows 1).
# Writes cert-1-mlperf.tstamps: before line right before run 0, after line right after the last run (also on TERM).
set -u
: "${CERT_A:?}" "${SLURM_JOB_ID:?}"
D=$CERT_A/runs/$SLURM_JOB_ID; mkdir -p "$D"
T=$CERT_A/upload/cert1-submission1/mlperf/cert-1-mlperf.tstamps
TSV=$D/runs.tsv
mapfile -t SEEDS < "$CERT_A/seeds.txt"
[ ${#SEEDS[@]} -eq 10 ] || { echo "CERT_LOOP abort: ${#SEEDS[@]} seeds in seeds.txt"; exit 3; }
[ -e "$T" ] && { echo "CERT_LOOP abort: $T exists, not overwriting"; exit 3; }
mkdir -p "$(dirname "$T")"
ended=0
stamp_end() { [ $ended = 1 ] && return; ended=1; echo `TZ=America/Chicago date` >> "$T"; echo "CERT_LOOP end stamp: $(tail -n 1 "$T")"; }
on_term() {
  echo "CERT_LOOP TERM: stopping run ${i:-?}"
  [ -n "${cp:-}" ] && { pkill -TERM -P "$cp" 2>/dev/null; kill -TERM "$cp" 2>/dev/null; wait "$cp" 2>/dev/null; }
  stamp_end; exit 143; }
trap on_term TERM INT
echo -e "idx\tseed\trc\trun_stop\tstart\tend" > "$TSV"
echo `TZ=America/Chicago date` > "$T"; echo "CERT_LOOP begin stamp: $(cat "$T")"
fails=0
for i in 0 1 2 3 4 5 6 7 8 9; do
  export SEED=${SEEDS[$i]} RUN_IDX=$i
  s=$(date '+%F %T'); echo "CERT_LOOP run $i seed $SEED start $s"
  bash "$CERT_A/gen/run-cert.sbatch" > >(tee "$D/run_$i.out") 2>&1 & cp=$!
  wait "$cp"; rc=$?
  while kill -0 "$cp" 2>/dev/null; do wait "$cp"; rc=$?; done
  st=$(grep -a -o '"key": "run_stop".*"status": "[a-z]*"' "$D/run_$i.out" | grep -o '"status": "[a-z]*"' | tail -1 | cut -d'"' -f4)
  e=$(date '+%F %T'); echo -e "$i\t$SEED\t$rc\t${st:-missing}\t$s\t$e" >> "$TSV"
  echo "CERT_LOOP run $i seed $SEED rc=$rc run_stop=${st:-missing} end $e"
  if [ "$st" != success ]; then
    fails=$((fails + 1))
    [ $fails -ge 2 ] && { echo "CERT_LOOP stop: $fails failed runs (MLPerf allows 1), not starting more"; break; }
  fi
done
stamp_end
echo "CERT_LOOP done: $fails failed of $((i + 1)) started"
[ $fails -le 1 ] && [ "$i" = 9 ]
