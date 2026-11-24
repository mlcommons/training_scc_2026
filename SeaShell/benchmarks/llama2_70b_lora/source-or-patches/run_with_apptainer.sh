#!/bin/bash

# this assumes that run_dir is a directory above code, the directory where this file sits
RUN_DIR=/mlperf
IMAGE=$RUN_DIR/amd-mlperf_llama2_70b_training_6.0_mi300x.sif
DATA=$RUN_DIR/data
RESULTS=$RUN_DIR/results
WORK=$RUN_DIR/work
LOCALCACHE=/scratch/$USER/mlperf_cache
CODE=$RUN_DIR/code


mkdir -p $LOCALCACHE $WORK/cache $WORK/output $WORK/logs $WORK/primus_output $RESULTS

if [ ! -f $WORK/overlay.img ]; then
  apptainer overlay create --size 51200 $WORK/overlay.img
fi

apptainer run \
  --overlay $WORK/overlay.img \
  --bind $CODE:/workspace/code \
  --bind $LOCALCACHE:/localcache \
  --bind $DATA:/data \
  --bind $RESULTS:/results \
  --bind $WORK/cache:/workspace/cache_persist \
  --bind $WORK/output:/workspace/output \
  --bind $WORK/logs:/workspace/Primus/logs \
  --bind $WORK/primus_output:/workspace/Primus/output \
  --env PRIMUS_CACHE_ROOT=/workspace/cache_persist \
  --env LOGDIR=/workspace/output \
  --env TRITON_CACHE_DIR=/localcache/triton \
  --env XDG_CACHE_HOME=/workspace/cache_persist/xdg \
  --env TORCHINDUCTOR_CACHE_DIR=/localcache/inductor \
  --env TMPDIR=/tmp \
  --rocm \
  $IMAGE \
  $CODE/apptainer_start_in_ct.sh
