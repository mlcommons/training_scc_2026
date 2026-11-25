#!/bin/bash
# Compile the FP8 flash-attention backward (hdim 128, causal, GQA) inside the training container.
set -euo pipefail
A=/work/benchmark/MLPerf/attnkernel; FA=$A/flash-attention; F=$A/fp8
# Flash_bwd_params needs one extra host float; add it to a private copy of flash.h (the FA3 tree stays untouched)
mkdir -p $F/inc
cp $FA/hopper/flash.h $F/inc/flash.h
srun -p cpu -N1 -n1 -c 16 --time=40 --container-name=llama2_70b_lora_persist_l2l-nemo --no-container-mount-home \
  --container-remap-root --container-mounts=$A:$A bash -c "cd $F && time nvcc -std=c++17 -O3 -DNDEBUG \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -gencode arch=compute_90a,code=sm_90a \
  -DFLASHATTENTION_DISABLE_VARLEN -DFLASHATTENTION_DISABLE_LOCAL -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_SM8x \
  -I$F/inc -I$F -I$FA/hopper -I$FA/csrc/cutlass/include -I/usr/local/cuda/include \
  ${EXTRA:-} -Xcompiler -fPIC -shared --threads 8 -o ${OUT:-libfa3_bwd_fp8.so} fa3_bwd_fp8.cu transpose_fp8.cu 2>&1 | grep -v -E 'warning #|^\s*$|Remark|note:' | head -${ERRLINES:-60}; ls -la ${OUT:-libfa3_bwd_fp8.so} 2>/dev/null || echo BUILD FAILED"
