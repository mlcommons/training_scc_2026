#!/bin/bash
# run_final.sh - the SCC26 final configuration in one command.
#
#   NEXP=10 bash ntu_scc/launcher/run_final.sh         # THE SUBMISSION: 10 runs, ONE JOB PER SEED, seeds base..base+9 with a RANDOM base,
#                                                      #   each log validated, then packaged + checked (SCC26 layout)
#   bash ntu_scc/launcher/run_final.sh                 # one test run, random seed
#   SEED=1240 bash ntu_scc/launcher/run_final.sh       # one test run with a chosen seed (rehearsals only, never for the submission)
#   CHECK_ONLY=1 bash ntu_scc/launcher/run_final.sh    # only apply/verify the node state, do not submit
#   CERT=1 NEXP=10 bash ntu_scc/launcher/run_final.sh  # also write cert-1-mlperf.* files (OFF by default, see 4.)
#
# What it does, in order:
#   1. node state on ntu-scc01 + ntu-scc02 (idempotent, prints the result): GPU caps 390 W x 8, SM clock lock 345-1050 MHz,
#      CPUs performance governor with 2.2 GHz max; checks the drop-caches systool is installed; reminds about the fan pin
#      (BMC, manual 40 %), which this script cannot set.
#   2. one persistent ssh session per compute node for 24 h (keeps /dev/shm alive: logind RemoveIPC), unless one exists.
#   3. submits the best config (run_best_18min.sh: kernel add-ons on, recompute off, cache clear + cache_clear record,
#      MLPERF_SUBMISSION_ORG/PLATFORM) and prints the job id, log path and how to watch it.
#   4. optional (CERT=<n>, default off): cert-$CERT-mlperf.{tstamps,input,rslts,run} in $CERTDIR (OUTSIDE the upload directory).
#      The SCC26 organizers said on 2026-11-16 that MLPerf needs NO manual cert files: the mllog timestamps in the result logs
#      are used, and extra files in the mlperf/ upload could interfere with the MLCommons tooling. Keep CERT=0 unless they ask.
#
# Seeds: MLPerf forbids cherry-picking; the reference draws SEED_BASE=$RANDOM and runs NEXP consecutive seeds, all of which are
# reported. Do that for the submission (NEXP=10). The checker only verifies the seeds are logged and distinct.
#
# Measured with this exact state (job 4853, seed 1240): TTT 17.27 min, step 2.396 s, PDU median 9.33 kW / max 9.69 kW, no 1-s
# sample at or above 10 kW. See ntu_scc/SEEDS.md.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
NODES=${NODES:-"ntu-scc01 ntu-scc02"}
CAP=${CAP:-390}; LOCK=${LOCK:-1050}; CPUMAX=${CPUMAX:-2.2}
SEED=${SEED:-$RANDOM}; NEXP=${NEXP:-1}     # reference run.sub behaviour: random base, run i uses base-1+i (consecutive); no seed selection
TAG=${TAG:-final_${CAP}W_$(date +%m%d_%H%M%S)}
CERT=${CERT:-0}; CERTDIR=${CERTDIR:-/work/benchmark/MLPerf/submission/scc26-cert}   # cert files OFF by default (organizers: mllog timestamps are used); never inside scc26/

ok=1
echo "== node state (${CAP} W x 8, SM lock 345-${LOCK} MHz, CPUs performance/${CPUMAX} GHz)"
for n in $NODES; do
  out=$(ssh -n -o ConnectTimeout=8 "$n" "
    BUDGET=$((CAP*8)) $REPO/ntu_scc/powercap/applycaps.sh $CAP $CAP $CAP $CAP $CAP $CAP $CAP $CAP >/dev/null 2>&1
    $REPO/ntu_scc/powercap/setclk.sh $LOCK >/dev/null 2>&1
    printf '5\n${CPUMAX}\ns\nyes\n' | /mnt/opt/systools/set-cpu-freq >/dev/null 2>&1
    gov=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); mx=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
    caps=\$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | sort -u | paste -sd,)
    util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -u | paste -sd,)
    tool=\$([ -u /mnt/opt/systools/drop-caches ] && echo ok || echo MISSING)
    fan=\$(cat /sys/class/hwmon/hwmon*/fan1_input 2>/dev/null | head -1)
    echo \"\$(hostname): caps \$caps W, lock 345-$LOCK (applied), cpu \$gov/\$mx kHz, gpu util \$util %, drop-caches \$tool\"
    [ \"\$caps\" = \"$CAP.00\" ] && [ \"\$gov\" = performance ] && [ \"\$mx\" = \"$(python3 -c "print(int($CPUMAX*1000000))")\" ] && [ \"\$tool\" = ok ] || echo \"   STATE NOT AS EXPECTED on \$(hostname)\"
  " 2>&1)
  echo "$out"; echo "$out" | grep -q 'NOT AS EXPECTED' && ok=0
done
echo "   fans: pin them at 40 % (manual) in each node's BMC before the run; this script cannot do it. Idle nodes: powersave."
[ "$ok" = 1 ] || { echo "!! fix the node state above, then re-run"; exit 2; }
[ "${CHECK_ONLY:-0}" = 1 ] && exit 0

echo "== persistent ssh sessions (24 h) so logind does not wipe /dev/shm during the job"
for n in $NODES; do
  pgrep -f "ssh -n .*$n sleep 86400" >/dev/null || { nohup ssh -n -o ServerAliveInterval=60 -o ConnectTimeout=8 "$n" 'sleep 86400' >/dev/null 2>&1 & disown; echo "   opened on $n"; }
done

if squeue -h -w "${NODES// /,}" 2>/dev/null | grep -q .; then echo "!! a job is already running on $NODES:"; squeue -w "${NODES// /,}"; exit 3; fi

cd "$REPO" || exit 1
TS=$CERTDIR/cert-$CERT-mlperf.tstamps
if [ "$CERT" != 0 ]; then
  mkdir -p "$CERTDIR"
  [ -s "$TS" ] && echo "   note: $TS already existed ($(wc -l < "$TS") lines) and is overwritten; use CERT=<n> for another attempt"
  echo "   note: the organizers do not want cert files for MLPerf (mllog timestamps are used); these stay in $CERTDIR, do not upload them"
  echo "$(TZ=America/Chicago date)" > "$TS"
  echo "== cert $CERT: start timestamp $(cat "$TS") -> $TS (end timestamp, .input/.rslts/.run follow automatically after the last job)"
fi
if [ "$NEXP" -gt 1 ]; then
  # one Slurm job per seed, back to back: seeds SEED..SEED+NEXP-1, each log validated (cache drop, cache_clear records, no
  # placeholders, compliance_checker), failed runs reported not retried, then the SCC26 package is built and checked.
  OUT=/work/benchmark/MLPerf/runs/submit_$TAG; mkdir -p "$OUT"
  echo "== queueing $NEXP runs now, one job each chained with --dependency=afterany, seeds $SEED..$((SEED+NEXP-1)) (random base $SEED), driver log $OUT/submit_seeds.log"
  QUEUE_ALL=1 SEED_BASE=$SEED NEXP=$NEXP OUT=$OUT CAP=$CAP LOCK=$LOCK CERT=$CERT CERTDIR=$CERTDIR nohup setsid bash ntu_scc/tools/submit_seeds.sh > "$OUT/submit_seeds.log" 2>&1 < /dev/null &
  sleep 45; grep -E "QUEUED|START|STOP|DRIFT|^  " "$OUT/submit_seeds.log"
  echo "   follow:  tail -f $OUT/submit_seeds.log      (START / RESULT / RETRY / FAILED-RUN / DRIFT / PACK lines; ~21 min per run)"
  echo "   results: $OUT/submit_seeds.results; package + checker output at the end of the log (/work/benchmark/MLPerf/submission/scc26/NTU)"
  [ "$CERT" != 0 ] && echo "   cert:    $CERTDIR/cert-$CERT-mlperf.{tstamps,input,rslts,run} (CERT line in the log when the end timestamp is written)"
  echo "   stop:    kill the 'bash ntu_scc/tools/submit_seeds.sh' process, then scancel the running job"
  exit 0
fi
echo "== submitting one run: seed $SEED, tag $TAG"
JID=$(SEED_BASE=$SEED NEXP=$NEXP TAG=$TAG bash ntu_scc/launcher/run_best_18min.sh 2>&1 | tail -1)
[[ "$JID" =~ ^[0-9]+$ ]] || { echo "!! sbatch failed: $JID"; exit 4; }
echo "   job $JID   log $REPO/slurm-$JID.out   results /work/benchmark/MLPerf/runs/$TAG"
if [ "$CERT" != 0 ]; then   # waiter: end timestamp the moment the job leaves the queue, then the cert files for this one log
  nohup setsid bash -c "while squeue -h -j $JID 2>/dev/null | grep -q .; do sleep 10; done; echo \"\$(TZ=America/Chicago date)\" >> '$TS'; CERT=$CERT CERTDIR='$CERTDIR' bash '$REPO/ntu_scc/submission/make_cert.sh' '$REPO/slurm-$JID.out'" > "$CERTDIR/cert-$CERT-waiter.log" 2>&1 < /dev/null &
  echo "   cert:    $TS gets the end timestamp when job $JID ends (waiter pid $!, log $CERTDIR/cert-$CERT-waiter.log), then .input/.rslts/.run"
fi
echo "   watch:   bash ntu_scc/tools/watchjob.sh $JID        (stalls -> scancel, parses TTT/convergence at the end)"
echo "   power:   bash ntu_scc/tools/power/plot_power.py $JID figs/power_$JID.png   (after the run; PDU series: cockpit_pdu_power_watts)"
echo "   expect:  run_start after ~3 min, ~17.3 min TTT at 3072 samples (~19.5 at 3456), PDU max ~9.7 kW"
