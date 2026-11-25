#!/usr/bin/env bash
# MLPerf Training: Llama 2 70B LoRA. Submits configs/<config>.sh as a Slurm job (scripts/mlperf.sbatch).
#   ./run.sh <config> [runs] [VAR=value ...]   10 runs (a full result set) unless given. VAR=value changes the config for this job only.
#   for a in 16 14 12; do ./run.sh amd ACL=$a; done   one job per value; Slurm runs the jobs one after another.
# After 10 runs, scripts/mlperf.sbatch runs the competition's checks: package checker, result summarizer, repo checker.
# The results go to results/<MMDD-HHMM>_<config>[_VAR=value ...]_<N>runs/, with the config as used in config.sh.
# scripts/results.sh shows them.
cd "$(dirname "$(readlink -f "$0")")"
git config --global --get-all safe.directory | grep -qx "$PWD" || git config --global --add safe.directory "$PWD"   # shared repository: git works for every team member
die() { echo "ERROR: $*"; exit 1; }
if [ -z "$1" ]; then
  echo "usage: ./run.sh <config> [runs] [VAR=value ...]"
  echo "configs: $(ls configs | sed 's/\.sh$//' | tr '\n' ' ')"
  exit 1
fi
NAME=$(basename "$1" .sh)
shift
[ -f configs/$NAME.sh ] || die "no configs/$NAME.sh"
[ -f data/train.npy ] && [ -f data/model/model_config.yaml ] && [ -x .venv/bin/python ] || die "setup is not complete: run scripts/setup.sh"
RUNS=10
TAG=$NAME
CONFIG=$(cat configs/$NAME.sh)
for arg; do
  if [[ $arg =~ ^[1-9][0-9]*$ ]]; then RUNS=$arg; continue; fi
  [[ $arg =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || die "$arg: give a number of runs or VAR=value"
  var=${BASH_REMATCH[1]}
  val=${BASH_REMATCH[2]}
  new="export $var=$(printf %q "$val")"
  re="^export $var=([^[:space:]]*)(.*)$"
  old=$(grep -m1 "^export $var=" <<< "$CONFIG")
  if [[ $old =~ $re ]]; then
    echo "$var: ${BASH_REMATCH[1]} -> $val"
    CONFIG=$(while IFS= read -r l; do [[ $l =~ $re ]] && l=$new${BASH_REMATCH[2]}; printf '%s\n' "$l"; done <<< "$CONFIG")
  else
    echo "$var: not in configs/$NAME.sh, added"
    CONFIG+=$'\n'$new
  fi
  TAG+=_$var=${val//[^A-Za-z0-9.+-]/-}
done
GBS=$(eval "$CONFIG"; echo $(( MINIBS * 8 / (TP * PP) / CP )))
case $GBS in 8|16|32|128) ;; *) die "global batch size $GBS has no reference convergence point: use 8, 16, 32 or 128" ;; esac
OUT=results/$(date +%m%d-%H%M)_${TAG}_${RUNS}runs
mkdir -p results
mkdir $OUT 2>/dev/null || die "$OUT exists: wait one minute"
printf '%s\n' "$CONFIG" > $OUT/config.sh
sbatch --job-name=mlperf_$TAG --output=$OUT/log.txt scripts/mlperf.sbatch $OUT $RUNS || die "Slurm submission failed"
echo "results: $OUT"
