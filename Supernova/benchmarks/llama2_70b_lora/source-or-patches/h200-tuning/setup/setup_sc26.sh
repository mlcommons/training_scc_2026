#!/bin/bash
set -x
set -o pipefail
H=$(cd "$(dirname "$0")/../.." && pwd)
V=$H/venv-train
PY=${SC26_PYTHON:-$(command -v python3.12)}
export UV_CACHE_DIR=$H/cache/uv
export TMPDIR=$H/tmp; mkdir -p "$TMPDIR" "$H/cache"
export CC=gcc CXX=g++ CUDAHOSTCXX=g++
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
JOBS=${JOBS:-140}
export MAX_JOBS=$JOBS CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
export NVCC_THREADS=${NVCC_THREADS:-2}
for _e in "$HOME/software/elfutils-dev/usr"; do
    [ -f "$_e/include/elfutils/libdwfl.h" ] || continue
    export CPATH=$_e/include${CPATH:+:$CPATH}
    export LIBRARY_PATH=$_e/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}
done
[ -f /usr/include/elfutils/libdwfl.h ] || echo "WARN: elfutils headers missing (apt install libdw-dev)"

stage() { echo "=== [$(date +%H:%M:%S)] STAGE: $1 ==="; }

[ -x $V/bin/python ] || uv venv --python $PY $V || exit 1

stage "torch nightly cu134"
uv pip install --python $V/bin/python \
  --index-url https://download.pytorch.org/whl/nightly/cu134 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.15.0.dev20261004+cu134" 2>&1 | tail -3 || exit 1

stage "nvidia wheel pins"
uv pip install --python $V/bin/python --override $H/h200-tuning/setup/venv-train-overrides.txt \
  nvidia-nvshmem-cu13==3.8.0 nvidia-nccl-cu13==2.32.3 \
  nvidia-cudnn-cu13==9.27.0.42 nvidia-cublas==13.8.1.7 2>&1 | tail -3 || exit 1

stage "benchmark/bridge deps"
uv pip install --python $V/bin/python \
  "omegaconf==2.3.0" "transformers==5.5.4" "hydra-core==1.3.2" \
  "peft>=0.18.1" "datasets>=2.20.0" accelerate einops tensorboard rich pyyaml tqdm \
  nvidia-resiliency-ext ninja cmake packaging pybind11 setuptools wheel nvidia-modelopt \
  nvidia-cudnn-frontend nvidia-mathdx==25.1.1 huggingface_hub requests \
  "git+https://github.com/mlcommons/logging.git@6.0.0-rc6" \
  "git+https://github.com/NVIDIA/mlperf-common.git@c02c93c0414071400a10716d9cd5cb8f668b78d4" 2>&1 | tail -3 || exit 1

stage "mcore editable"
(cd $H/Megatron-Bridge/3rdparty/Megatron-LM && \
  uv pip install --python $V/bin/python --no-build-isolation -e . 2>&1 | tail -3) || exit 1

stage "deepgemm editable + cutlass symlinks"
ln -sfn $H/deepgemm/third-party/cutlass/include/cutlass $H/deepgemm/deep_gemm/include/cutlass
ln -sfn $H/deepgemm/third-party/cutlass/include/cute $H/deepgemm/deep_gemm/include/cute
(. $H/h200-tuning/setup/venv_toolchain.sh && cd $H/deepgemm && \
  uv pip install --python $V/bin/python --no-build-isolation -e . 2>&1 | tail -3) || exit 1

stage "deepgemm JIT header freshness"
$V/bin/python - "$H/deepgemm" <<'PYEOF' || exit 1
import filecmp, importlib.util, os, sys
tree = sys.argv[1]
spec = importlib.util.find_spec("deep_gemm")
if spec is None or spec.origin is None:
    sys.exit("FATAL: deep_gemm is not importable from this venv")
pkg = os.path.dirname(os.path.abspath(spec.origin))
hdr = "deep_gemm/impls/sm90_fp8_gemm_per_tensor_legacy.cuh"
a, b = os.path.join(tree, "deep_gemm/include", hdr), os.path.join(pkg, "include", hdr)
if not os.path.exists(b):
    sys.exit(f"FATAL: installed DeepGEMM has no {hdr} ({pkg}/include)")
if not filecmp.cmp(a, b, shallow=False):
    sys.exit(f"FATAL: installed DeepGEMM JIT header is STALE.\n"
             f"  tree:      {a}\n  installed: {b}\n"
             f"The JIT compiles against the installed copy, so every tree edit\n"
             f"under deep_gemm/include/ is ignored until the package is\n"
             f"reinstalled. Reinstall it (wheel: rebuild+install; editable:\n"
             f"re-run the stage above) before running any DeepGEMM arm.")
print(f"deepgemm JIT headers fresh ({pkg}/include)")
PYEOF

stage "site-overlay (transformers 5.12.1 + megatron-energon)"
OB=$H/.site-overlay-build
rm -rf $OB; mkdir -p $OB
uv venv --python $PY $OB/ovenv
uv pip install --python $OB/ovenv/bin/python \
  "transformers==5.12.1" "tokenizers>=0.22.0,<0.23" megatron-energon \
  "safetensors==0.6.2" 2>&1 | tail -3 || exit 1
uv pip install --python $V/bin/python "safetensors==0.6.2" 2>&1 | tail -2 || exit 1
rm -rf $H/site-overlay; mkdir -p $H/site-overlay
rsync -a --exclude '/torch' --exclude '/torch-*' --exclude '/nvidia' --exclude '/nvidia_*' \
  --exclude '/triton' --exclude '/triton-*' --exclude '/functorch' \
  --exclude '__pycache__' --exclude '/bin' \
  $OB/ovenv/lib/python3.12/site-packages/ $H/site-overlay/
rm -rf $OB

stage "apex wgrad extension (fused_weight_gradient_mlp_cuda)"
APEX_SRC=${APEX_SRC:-$H/cache/apex-src}
if [ ! -f "$APEX_SRC/csrc/type_shim.h" ]; then
  git clone -q --depth 1 https://github.com/NVIDIA/apex.git "$APEX_SRC"
fi
bash $H/h200-tuning/setup/build_wgrad_ext.sh "$APEX_SRC" $V || exit 1

stage "TE patch series (v2.20.2 + every transformerengine/te*.patch, dependency-ordered)"
TE=$H/transformerengine
apply_te_patches() {
  local d=$1; shift
  local todo=($TE/te*.patch) p
  while [ ${#todo[@]} -gt 0 ]; do
    local next=() progress=0
    for p in "${todo[@]}"; do
      if git -C $d apply "$@" "$p" 2>/dev/null; then progress=1; else next+=("$p"); fi
    done
    [ $progress -eq 1 ] || { echo "TE patches unresolvable: ${next[*]}"; return 1; }
    todo=("${next[@]}")
  done
}
git -C $TE rev-parse --verify -q HEAD >/dev/null || { echo "transformerengine/ is not a TE v2.20.2 git checkout"; exit 1; }
if git -C $TE diff --quiet HEAD -- transformer_engine build_tools; then
  apply_te_patches $TE --index || { echo "TE patch series failed"; exit 1; }
else
  TW=$(mktemp -d)
  git -C $TE worktree add -q --detach $TW HEAD || exit 1
  apply_te_patches $TW || { git -C $TE worktree remove --force $TW; echo "TE patch series failed on HEAD"; exit 1; }
  git -C $TW add -A >/dev/null
  if ! diff -q <(git -C $TW diff --cached HEAD -- transformer_engine build_tools) \
              <(git -C $TE diff HEAD -- transformer_engine build_tools) >/dev/null; then
    git -C $TE worktree remove --force $TW
    echo "transformerengine/ differs from v2.20.2 + te*.patch series"; exit 1
  fi
  git -C $TE worktree remove --force $TW
fi

stage "TE build (build_te.sh)"
(cd $H/transformerengine && TRAIN_VENV=$V \
  ./build_te.sh 2>&1 | tail -25) || exit 1

stage "SwiGLU fusion library (sm90a; includes TE common headers, so after the TE build)"
mkdir -p $H/swiglu_fusion/build
$CUDA_HOME/bin/nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_90a --threads 1 \
  -I $H/transformerengine/transformer_engine/common \
  $H/swiglu_fusion/swiglu.cu -o $H/swiglu_fusion/build/libswiglu_fusion.so || exit 1

stage "flash-attention hopper editable (--no-deps: keeps the nvidia wheel pins)"
(cd $H/flash-attention/hopper && \
  PATH=$V/bin:$CUDA_HOME/bin:$PATH MAX_JOBS=$JOBS \
  FLASHATTENTION_ENABLE_BWD_FP8=TRUE FLASH_ATTENTION_ENABLE_VCOLMAJOR=TRUE \
  FLASH_ATTENTION_DISABLE_SM80=TRUE \
  FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  uv pip install --python $V/bin/python --no-build-isolation --no-deps -e . 2>&1 | tail -5) || exit 1

stage "patched NCCL (sm90 TMA symmetric kernels over the 2.32.3 wheel)"
bash $H/h200-tuning/setup/build_nccl.sh $V || exit 1

stage "smoke imports"
$V/bin/python -c "
import torch, transformer_engine, transformer_engine_torch as tex
import deep_gemm, flash_attn_interface, megatron.core
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('TE', transformer_engine.__version__)
" || exit 1
PYTHONPATH=$H/site-overlay $V/bin/python -c "import transformers; print('overlay transformers', transformers.__version__)" || exit 1

echo "=== SETUP_SC26 DONE ==="
