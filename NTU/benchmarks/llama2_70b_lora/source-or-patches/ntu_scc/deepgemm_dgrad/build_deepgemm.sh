#!/bin/bash
# Build DeepGEMM 2.8.1 (commit 057ca59, "Public release 26/09/30") into a private --target dir, INSIDE the training
# container (same torch/CUDA ABI as the run). The image lacks libdw/libelf, which deep_gemm's extension links against,
# so Ubuntu 24.04's packages are unpacked into a private sysroot (no root needed) and used for both build and runtime
# (dg_dgrad.apply() preloads them).
#   G=/work/benchmark/MLPerf/gemmlib  ->  $G/pylib_dg (python package), $G/sysroot (libdw/libelf), $G/dg_cache (JIT cache)
set -euo pipefail
G=${G:-/work/benchmark/MLPerf/gemmlib}
CONT_ARGS="--container-name=llama2_70b_lora_persist_l2l-nemo --no-container-mount-home --container-mounts=$G:$G"
mkdir -p "$G/sysroot" "$G/debs" "$G/pylib_dg" "$G/logs"
[ -d "$G/deepgemm_src" ] || git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$G/deepgemm_src"
git -C "$G/deepgemm_src" checkout 057ca59 && git -C "$G/deepgemm_src" submodule update --init --recursive
srun -p cpu -N1 -n1 -c 8 -t 15 $CONT_ARGS bash -c "cd $G/debs && apt-get download libdw-dev libelf-dev libdw1t64 libelf1t64 && for d in *.deb; do dpkg -x \$d $G/sysroot; done"
S=$G/sysroot/usr
srun -p cpu -N1 -n1 -c 64 -t 30 $CONT_ARGS bash -c "cd $G/deepgemm_src && \
  CPLUS_INCLUDE_PATH=$S/include C_INCLUDE_PATH=$S/include LIBRARY_PATH=$S/lib/x86_64-linux-gnu LDFLAGS=-L$S/lib/x86_64-linux-gnu \
  MAX_JOBS=64 pip install --no-build-isolation --no-deps --target $G/pylib_dg . > $G/logs/deepgemm_install.log 2>&1; tail -2 $G/logs/deepgemm_install.log"
