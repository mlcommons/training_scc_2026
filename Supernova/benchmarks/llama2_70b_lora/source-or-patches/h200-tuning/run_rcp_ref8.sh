#!/bin/bash
set -u
H=${MLPERF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export MLPERF_ROOT=$H
export CACHE_CLEAR=${CACHE_CLEAR:-1}
FIRST=${1:-12552}; LAST=${2:-12561}
MODEL_DIR=${MODEL_DIR:-$H/model/llama2_70b_megatron_bridge}
DATA_DIR=${DATA_DIR:-$H/data}
export NODELIST=${NODELIST:-$(hostname -s)}
export NGPUS=${NGPUS:-$(( $(echo "$NODELIST" | tr , '\n' | wc -l) * $(nvidia-smi -L | wc -l) ))}

source "${SUBMISSION_CONFIG:-$(dirname "$0")/submission/config.sh}"
echo "layout: ${LAYOUT:-?}, $NGPUS GPUs on $NODELIST"

mkdir -p $H/results
PFX=${RUNPFX:-rcp}
for SEED in $(seq $FIRST $LAST); do
  RUN=$PFX-s$SEED
  echo "=== [$(date '+%F %T')] launching $RUN (seed $SEED) ==="
  ENVSPEC=$(echo $COMMON $RECIPE SEED=$SEED | tr -s ' \n' ' ')
  echo "env: $ENVSPEC"
  ( cd $H && env $ENVSPEC bash h200-tuning/run_rcp_het.sh $RUN ) > $H/results/$RUN.launch.log 2>&1
  rc=$?
  echo "=== [$(date '+%F %T')] $RUN done rc=$rc ==="
done
echo "=== [$(date '+%F %T')] batch complete (seeds $FIRST..$LAST) ==="
if [ $((LAST - FIRST + 1)) -eq 10 ]; then
  python3 $H/h200-tuning/package_submission.py ${CERT:+--cert $CERT --run-script $0} \
    $(for SEED in $(seq $FIRST $LAST); do echo $H/results/$PFX-s$SEED; done) \
    || echo "WARNING: submission packaging failed" >&2
fi
