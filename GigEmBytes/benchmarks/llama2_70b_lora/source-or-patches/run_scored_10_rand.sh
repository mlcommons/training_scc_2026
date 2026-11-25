#!/bin/bash
# ONE file for the scored attempt: N Llama2-70B LoRA runs with clock-based random seeds,
# one flux job, everything on gpu1's local NVMe.
#
#   flux batch -N1 -n1 -g8 --exclusive -q gpu -t 540 run_scored_10.sh
#
# Env overrides, e.g. a 5-min-train smoke:
#   NUM_RUNS=1 MAX_STEPS=70 SMOKE=1 SEED_TIMEOUT=720 RESULTS=$WORK/results_smoke \
#     flux batch -N1 -n1 -g8 --exclusive -q gpu -t 14 run_scored_10.sh
#
# Config (closed division): LR=4e-4 WARMUP_STEPS=11 GBS=8 MAX_STEPS=550 TP4/DP2/MINIBS4.
# Results: $RESULTS/result_0.txt .. result_{N-1}.txt, each PASS-checked.
set -uo pipefail

WORK=${WORK:-/local/users/dkb/MLPerf}
JOB_SCRIPT=${JOB_SCRIPT:-$WORK/run_lora70b_triton_fix.sh}
RESULTS=${RESULTS:-$WORK/results}
LOGS=${LOGS:-$WORK/logs}
NUM_RUNS=${NUM_RUNS:-10}
SEED_TIMEOUT=${SEED_TIMEOUT:-5400}

export LR=${LR:-0.0004} WARMUP_STEPS=${WARMUP_STEPS:-11} GBS=${GBS:-8} MAX_STEPS=${MAX_STEPS:-550}
unset SKIP_EVALS FORCE_SUCCESS_STATUS

# clock-based seed in [1, 2147483646]; $1 = run index to guarantee uniqueness
gen_seed() {
 echo $(( ( $(date +%s%N) + RANDOM * 32768 + RANDOM + $1 * 7919 ) % 2147483646 + 1 ))
}

cd "$WORK" || exit 1
[ -f "$JOB_SCRIPT" ] || { echo "missing $JOB_SCRIPT" >&2; exit 2; }
mkdir -p "$LOGS" "$RESULTS"
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -eq 8 ] || { echo "need 8 GPUs, found $ngpu" >&2; exit 2; }

tsv=$RESULTS/seeds.tsv
[ -s "$tsv" ] || printf 'run\tseed\truntime_s\tstop_step\tfinal_loss\tresult_file\tstatus\n' > "$tsv"

status_of() { # <logdir> <stdout> <seed>
 local d=$1 o=$2 s=$3 log=$1/train.log r=""
 [ -f "$log" ] || { echo "FAILED:no_train_log"; return; }
 [ -d "$d" ] || { echo "FAILED:no_logdir"; return; }
 grep -qx 'TRAIN_EXIT:0' "$o" 2>/dev/null || r="$r,train_exit"
 if [ "${SMOKE:-0}" = 1 ]; then
   grep -q '"key": "tracked_stats"' "$log" || r="$r,no_steps"
   [ -z "$r" ] && echo PASS:smoke || echo "FAILED:${r#,}"
   return
 fi
 grep -q '"key": "cache_clear"' "$log" || r="$r,cache_clear"
 grep '"key": "run_stop"' "$log" | tail -1 | grep -q '"status": "success"' || r="$r,run_stop"
 grep '"key": "eval_accuracy"' "$log" |
   sed -E 's/.*"value": ([^,]+),.*/\1/' |
   awk -v t=0.925 '$1+0 <= t {f=1} END {exit !f}' || r="$r,eval_accuracy"
 grep -q " SEED=${s} " "$o" 2>/dev/null || r="$r,cfg_seed"
 [ -z "$r" ] && echo PASS || echo "FAILED:${r#,}"
}

failed=0
for ((i = 0; i < NUM_RUNS; i++)); do
 s=$(gen_seed "$i")
 echo "$i $s" >> "$RESULTS/seeds_used.txt"
 out=$LOGS/run${i}_seed${s}.out
 echo "=== run $i seed $s ($(date '+%F %T')) -> $out"
 before=$(date +%s)
 SEED=$s timeout "$SEED_TIMEOUT" bash "$JOB_SCRIPT" > "$out" 2>&1
 rc=$?
 runtime=$(( $(date +%s) - before ))

 logdir=$(sed -n 's/^LOGDIR://p' "$out" 2>/dev/null | tail -1)
 if [ ! -d "${logdir:-/none}" ]; then
   logdir=$(ls -1dt "$WORK"/logs/train_LR${LR}_WU${WARMUP_STEPS}_GBS${GBS}_S${MAX_STEPS}_seed${s}_* 2>/dev/null | head -1)
   [ -n "${logdir:-}" ] && [ "$(stat -c %Y "$logdir")" -lt $((before - 60)) ] && logdir=""
 fi
 st=$(status_of "${logdir:-/none}" "$out" "$s")
 [ "$rc" -eq 124 ] && st=FAILED:timeout

 samples=$(grep '"key": "run_stop"' "$logdir/train.log" 2>/dev/null | tail -1 |
           grep -o '"samples_count": [0-9]*' | head -1 | awk '{print $2}')
 stop=$(( ${samples:-0} / GBS ))
 acc=$(grep '"key": "eval_accuracy"' "$logdir/train.log" 2>/dev/null | tail -1 |
       sed -E 's/.*"value": ([^,]+),.*/\1/')

 result_file=""
 case "$st" in
   PASS) result_file=$RESULTS/result_${i}.txt
         sed -n 's/^.*\(:::MLLOG \)/\1/p' "$logdir/train.log" > "$result_file" ;;
   PASS:smoke) ;;
   *) failed=$((failed + 1)) ;;
 esac
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$s" "$runtime" "$stop" "${acc:-}" "$result_file" "$st" >> "$tsv"
 echo "    $st  runtime=${runtime}s  steps=$stop  acc=${acc:-n/a}"
done
echo "=== done: $(date '+%F %T'); result files in $RESULTS"
exit $(( failed > 0 ))
