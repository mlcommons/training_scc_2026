#!/bin/bash
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
W=${1:?usage: apply_all.sh <empty-workdir>}
mkdir -p "$W"

get() {
  local dir=$1 url=$2 rev=$3; shift 3
  git clone -q --filter=blob:none --no-checkout "$url" "$W/$dir"
  if [ $# -gt 0 ]; then
    git -C "$W/$dir" sparse-checkout set --no-cone "$@"
  fi
  git -C "$W/$dir" -c advice.detachedHead=false checkout -q "$rev"
  echo "$dir @ $(git -C "$W/$dir" rev-parse --short HEAD)"
}
subs() {
  local dir=$1; shift
  git -C "$W/$dir" submodule update --init --depth 1 -q -- "$@"
  echo "  submodules: $*"
}
apply() {
  git -C "$W/$1" apply --whitespace=nowarn "$2"
  echo "  applied $(basename "$2")"
}

IMPL=NVIDIA/benchmarks/llama2_70b_lora/implementations
get training_results_v6.0 https://github.com/mlcommons/training_results_v6.0.git \
    eabf23a07b2a0c60a289ff871dc3a46fff0d0421 "/$IMPL/nemo/" "/$IMPL/swiglu_fusion/"
apply training_results_v6.0 "$P/01-llama2_70b_lora-nemo.patch"

get Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge.git 634ec709062e643f0469610a315b0cfff62cfc85
apply Megatron-Bridge "$P/02-megatron-bridge.patch"

get TransformerEngine https://github.com/NVIDIA/TransformerEngine.git v2.20.2
subs TransformerEngine 3rdparty/cutlass 3rdparty/googletest 3rdparty/nccl-extensions
for p in "$P"/03-transformer-engine/*.patch; do apply TransformerEngine "$p"; done

get flash-attention https://github.com/Dao-AILab/flash-attention.git e9515d5dee6ade134a33d6020d38d01ef0596996
subs flash-attention csrc/cutlass
apply flash-attention "$P/04-flash-attention.patch"

get DeepGEMM https://github.com/deepseek-ai/DeepGEMM.git 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
subs DeepGEMM third-party/cutlass third-party/deep_jit
apply DeepGEMM "$P/05-deepgemm.patch"

get nccl https://github.com/NVIDIA/nccl.git v2.32.3-1
apply nccl "$P/06-nccl.patch"

echo "done: $W"
