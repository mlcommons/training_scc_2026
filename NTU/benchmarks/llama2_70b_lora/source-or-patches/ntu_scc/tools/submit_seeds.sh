#!/bin/bash
# submit_seeds.sh - run the submission seeds one Slurm job per seed (390 W, SM lock 1050 MHz, CPUs performance/2.2 GHz),
# verify each log for submission validity (cache_clear records + real cache drop + compliance_checker), retry only
# infrastructure failures (no run_start / hang), keep genuine failed runs as reported runs, then build the SCC26 package and
# run the rcp / package / verify / summarizer checks.
# Usage (submission, via run_final.sh): QUEUE_ALL=1 SEED_BASE=<random> NEXP=10 OUT=<dir> nohup setsid bash submit_seeds.sh > $OUT/submit_seeds.log 2>&1 &
# Usage (rehearsal list):               SEEDS="..." TARGET=10 OUT=<dir> nohup setsid bash submit_seeds.sh > ... &
# Events (one line each) go to stdout: QUEUED, START, HUNG, STARTUP-HUNG, RETRY, SKIP, RESULT, FAILED-RUN, STOP, DRIFT, CERT, PACK, SUBMIT_SEEDS DONE.
S=${OUT:-/work/benchmark/MLPerf/runs/submit_seeds}; mkdir -p "$S"   # event log, results and compliance logs go here
D=/work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b
MLC=/work/benchmark/MLPerf/tools/mlc; PY=$MLC/venv/bin/python
CERT=${CERT:-0}; CERTDIR=${CERTDIR:-/work/benchmark/MLPerf/submission/scc26-cert}   # optional cert files, off by default (organizers: not needed for MLPerf)
if [ -n "${SEED_BASE:-}" ]; then SEEDS=$(seq -s " " "$SEED_BASE" $((SEED_BASE + ${NEXP:-10} - 1))); RESERVE=""; TARGET=${NEXP:-10}; fi   # submission mode: consecutive seeds, every run reported
SEEDS=${SEEDS:-"1235 1237 1240 1241 1242 1244 1245 1246 1247 1248"}
CAP=${CAP:-390}; LOCK=${LOCK:-1050}
RES=$S/submit_seeds.results; : > $RES
cd $D || exit 1

preflight() {   # re-apply the requested node state before every run (idempotent); report drift
  for h in ntu-scc01 ntu-scc02; do
    st=$(ssh -n -o ConnectTimeout=8 $h 'gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); mx=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); caps=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | sort -u | paste -sd,); echo "$gov $mx $caps"')
    read -r gov mx caps <<< "$st"
    [ "$caps" = "$CAP.00" ] || { echo "DRIFT $h GPU caps were $caps -> re-applying $CAP W $(date +%T)"; ssh -n $h "BUDGET=$((CAP*8)) $D/ntu_scc/powercap/applycaps.sh $CAP $CAP $CAP $CAP $CAP $CAP $CAP $CAP >/dev/null 2>&1"; }
    if [ "$gov" != performance ] || [ "$mx" != 2200000 ]; then echo "DRIFT $h CPU was $gov/$mx -> re-applying performance/2.2 GHz $(date +%T)"; ssh -n $h "printf '5\n2.2\ns\nyes\n' | /mnt/opt/systools/set-cpu-freq >/dev/null 2>&1"; fi
    ssh -n $h "$D/ntu_scc/powercap/setclk.sh $LOCK >/dev/null" || echo "WARN $h setclk $LOCK failed $(date +%T)"
  done
  [ -u /mnt/opt/systools/drop-caches ] || { echo "STOP drop-caches systool missing -> runs would not be compliant $(date +%T)"; exit 2; }
}

parse() {   # prints: status conv ttt step evals
python3 - "$1" <<'PY'
import re,sys,json,statistics
F=sys.argv[1]; st=[]; t={}; ev=[]; status="nostart"
for line in open(F, errors="ignore"):
    m=re.search(r':::MLLOG (\{.*\})', line)
    if not m: continue
    try: d=json.loads(m.group(1))
    except Exception: continue
    k=d.get("key")
    if k=="run_start": status="running"
    if k=="tracked_stats" and isinstance(d.get("value"),dict) and "train_step_time" in d["value"]: st.append(d["value"]["train_step_time"])
    if k in ("run_start","run_stop"): t[k]=d["time_ms"]
    if k=="run_stop": status=d.get("metadata",{}).get("status","?")
    if k=="eval_accuracy": ev.append((d["metadata"]["samples_count"], round(d["value"],4)))
ttt=f"{(t['run_stop']-t['run_start'])/60000:.2f}" if len(t)==2 else "nan"
step=f"{statistics.mean(st[10:]):.4f}" if len(st)>10 else "nan"
conv=str(ev[-1][0]) if ev and ev[-1][1]<=0.925 else "none"
print(status, conv, ttt, step, json.dumps(ev).replace(" ",""))
PY
}

validity() {   # prints "ok" or the reasons; checks the cache clear really happened and the file passes the compliance checker
  local F=$1 why=""
  local nrec ndrop nerr
  nrec=$(grep -c '"key": "cache_clear"' "$F"); ndrop=$(grep -c 'dropped caches:' "$F"); nerr=$(grep -cE 'drop-caches: |error: .*drop_caches' "$F")
  [ "$nrec" -ge 1 ] || why="$why no-cache_clear-record"
  [ "$ndrop" -ge 2 ] || why="$why drop-lines=$ndrop(expected 2)"
  [ "$nerr" -eq 0 ] || why="$why drop-errors=$nerr"
  grep -q 'SUBMISSION_ORG_PLACEHOLDER\|SUBMISSION_PLATFORM_PLACEHOLDER' "$F" && why="$why placeholders-present"
  if ! $PY -m mlperf_logging.compliance_checker --usage training --ruleset 6.0.0 "$F" > "$F.compliance.log" 2>&1; then why="$why compliance_checker-FAILED(see $F.compliance.log)"; fi
  [ -z "$why" ] && echo ok || echo "$why"
}

# Job still alive? squeue first; if it shows nothing, confirm with sacct (a transient squeue failure must not look like "finished").
active() {
  local j=$1 s
  s=$(squeue -h -j "$j" -o %T 2>/dev/null); [ -n "$s" ] && return 0
  s=$(sacct -j "$j" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  case "$s" in PENDING|RUNNING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED|RESIZING) return 0 ;; "") ;; *) return 1 ;; esac
  # neither squeue nor sacct answered: assume alive unless the log already has run_stop or has been silent for 10 min
  [ -f "$D/slurm-$j.out" ] && grep -q '"key": "run_stop"' "$D/slurm-$j.out" && return 1
  [ -f "$D/slurm-$j.out" ] && [ $(( $(date +%s) - $(stat -c %Y "$D/slurm-$j.out") )) -gt 600 ] && return 1
  return 0
}
wait_nodes_free() {   # the previous job may linger in COMPLETING on the nodes
  local n=0; while squeue -h -w ntu-scc01,ntu-scc02 2>/dev/null | grep -q .; do sleep 15; n=$((n+1)); [ $n -gt 80 ] && { echo "WARN nodes still busy after 20 min $(date +%T)"; break; }; done
}

TARGET=${TARGET:-10}
RESERVE=${RESERVE-"1249 1250 1251 1252 1253 1254"}    # only used if a listed seed keeps failing; convergence unknown. No colon: submission mode sets RESERVE="" (empty, not unset) so no extra seeds are queued
queue="$SEEDS $RESERVE"
# QUEUE_ALL=1: submit every seed now, chained with --dependency=afterany so they run in order; the loop below then only waits,
# validates, and re-applies the node state right after each job ends (the next job spends ~3 min in startup before training).
declare -A QJOB
if [ "${QUEUE_ALL:-0}" = 1 ]; then
  wait_nodes_free; preflight; prev=""
  for seed in $queue; do
    TAG=sub_${seed}_$(date +%H%M%S)
    JID=$(SBATCH_EXTRA="${prev:+--dependency=afterany:$prev}" NEXP=1 SEED_BASE=$seed TAG=$TAG bash ntu_scc/launcher/run_best_18min.sh 2>&1 | tail -1)
    [[ "$JID" =~ ^[0-9]+$ ]] || { echo "STOP sbatch failed for seed $seed: $JID $(date +%T)"; exit 2; }
    QJOB[$seed]=$JID; echo "QUEUED seed $seed job $JID$([ -n "$prev" ] && echo " (after job ${prev})") $(date +%T)"; prev=$JID
  done
  squeue -h -u "$USER" -o '   %i %T %r' | sed 's/^/  /'
fi

idx=0
for seed in $queue; do
  [ $idx -ge $TARGET ] && break
  retries=0
  while :; do
    if [ "${QUEUE_ALL:-0}" = 1 ] && [ -n "${QJOB[$seed]:-}" ]; then
      JID=${QJOB[$seed]}; unset "QJOB[$seed]"; TAG=queued
      while squeue -h -j $JID -o %T 2>/dev/null | grep -q PENDING; do sleep 30; done
      preflight
    else
      wait_nodes_free
      preflight
      TAG=sub_${seed}_$(date +%H%M%S)
      JID=$(NEXP=1 SEED_BASE=$seed TAG=$TAG bash ntu_scc/launcher/run_best_18min.sh 2>&1 | tail -1)
      [[ "$JID" =~ ^[0-9]+$ ]] || { echo "STOP sbatch failed for seed $seed: $JID $(date +%T)"; exit 2; }
    fi
    echo "START seed $seed job $JID idx $idx $(date +%T)"
    F=$D/slurm-$JID.out; hung=0; t_start=""; sleep 60
    while active $JID; do
      if [ -f $F ]; then
        [ -z "$t_start" ] && t_start=$(stat -c %Y $F)
        age=$(( $(date +%s) - $(stat -c %Y $F) ))
        if grep -q '"key": "run_start"' $F; then
          if [ $age -gt 360 ]; then echo "HUNG seed $seed job $JID: no log for ${age}s -> scancel $(date +%T)"; scancel $JID; hung=1; sleep 30; break; fi
        elif [ $(( $(date +%s) - t_start )) -gt 900 ] && [ $age -gt 300 ]; then
          echo "STARTUP-HUNG seed $seed job $JID: no run_start after $(( $(date +%s) - t_start )) s, log idle ${age}s -> scancel $(date +%T)"; scancel $JID; hung=1; sleep 30; break
        fi
      fi
      sleep 30
    done
    sleep 20
    read status conv ttt step evals <<< "$(parse "$F")"
    if [ "$status" = "success" ]; then
      v=$(validity "$F")
      if [ "$v" = ok ]; then
        echo "RESULT seed $seed job $JID: TTT $ttt min, converged@$conv, step $step, valid=ok, evals $evals"
        echo "$idx $seed $JID $ttt $conv $step" >> $RES; idx=$((idx+1)); break
      else
        echo "STOP seed $seed job $JID finished (TTT $ttt, conv@$conv) but the log is NOT submission-valid:$v $(date +%T)"; exit 3
      fi
    elif [ "$status" != "nostart" ] && [ "$status" != "running" ] && [ -n "${SEED_BASE:-}" ]; then
      # run_stop was logged with a non-success status (did not converge within MAX_STEPS): a genuine failed run, reported as such.
      # "running" (run_start but no run_stop: hang/cancel/node failure) and "nostart" are infrastructure failures -> retried below.
      echo "FAILED-RUN seed $seed job $JID: status=$status conv=$conv -> kept as a reported failed run (no cherry-picking) $(date +%T)"
      echo "$idx $seed $JID $ttt $conv $step FAILED" >> $RES; idx=$((idx+1)); break
    else
      reason=$(grep -m2 -E 'Error|error:|slurmstepd|CANCELLED|DUE TO TIME' "$F" 2>/dev/null | grep -vE 'docstring|cache_position' | tr '\n' ' ' | cut -c1-200)
      echo "RETRY seed $seed job $JID: status=$status hung=$hung conv=$conv step=$step (retry $retries) reason: ${reason:-none} $(date +%T)"
      retries=$((retries+1)); [ $retries -gt 2 ] && { echo "SKIP seed $seed after 3 attempts -> next seed $(date +%T)"; break; }
      sleep 60
    fi
  done
done
[ $idx -ge $TARGET ] || { echo "STOP only $idx valid runs after exhausting the seed queue $(date +%T)"; exit 5; }

if [ "$CERT" != 0 ]; then echo "$(TZ=America/Chicago date)" >> "$CERTDIR/cert-$CERT-mlperf.tstamps"; echo "CERT $CERT end timestamp appended to $CERTDIR/cert-$CERT-mlperf.tstamps $(date +%T)"; fi
echo "PACK building the SCC26 package from $RES $(date +%T)"
bash $D/ntu_scc/submission/make_package.sh $RES 6.0.0 2>&1 | sed 's/^/  /'
echo "SUBMIT_SEEDS DONE $(date +%T)"
