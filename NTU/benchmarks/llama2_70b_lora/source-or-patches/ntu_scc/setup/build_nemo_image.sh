#!/bin/bash
# Build the NVIDIA/Fujitsu llama2_70b_lora image: NGC pytorch:26.04-py3
# + cuDNN 9.21.1.3 + Megatron-Bridge 26.04-alpha.rc1 + their requirements.
# TE rebuild intentionally SKIPPED (their NVTE_CUDA_ARCHS=100a;103a is Blackwell;
# we are sm_90 H200, so the stock TE in the NGC image is correct).
# mlperf-logging pinned to 6.0.0-rc6 per standing project rule (they ship rc2).
set -euxo pipefail
export ENROOT_CACHE_PATH=/scratch/enroot-cache ENROOT_TEMP_PATH=/scratch/enroot-tmp
export ENROOT_DATA_PATH=/scratch/enroot-data ENROOT_RUNTIME_PATH=/scratch/enroot-runtime
mkdir -p $ENROOT_CACHE_PATH $ENROOT_TEMP_PATH $ENROOT_DATA_PATH $ENROOT_RUNTIME_PATH
NAME=l2l-nemo-build
enroot remove -f $NAME 2>/dev/null || true
enroot create --name $NAME /scratch/containers/pytorch2604.sqsh

enroot start --root --rw --mount /work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b:/host-impl $NAME bash -eux <<'INNER'
git config --global user.name a && git config --global user.email a
export PIP_CONSTRAINT=""
NVARCH=$(uname -m | grep x86_64 2>/dev/null || echo sbsa)
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list
apt-get update
# NOTE: installing ONLY libcudnn9-cuda-13 makes apt REMOVE libcudnn9-dev-cuda-13,
# which owns the unversioned /usr/lib/x86_64-linux-gnu/libcudnn.so symlink that
# transformer_engine's _load_cuda_library() requires -> 'cudnn shared object not found'.
# Install the matching dev+headers packages so the symlink survives at 9.21.1.3.
apt-get install -y libcudnn9-cuda-13=9.21.1.3-1 libcudnn9-dev-cuda-13=9.21.1.3-1 libcudnn9-headers-cuda-13=9.21.1.3-1
ldconfig
ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/x86_64-linux-gnu/libcudnn.so.9
python -c "import transformer_engine.common; print('TE cudnn load OK')"
rm -rf /var/lib/apt/lists/*

pip uninstall -y megatron-core || true
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge
cd /workspace/Megatron-Bridge
git checkout 26.04-alpha.rc1
git submodule update --init --recursive
git rev-parse HEAD > /MBRIDGE_COMMIT_HASH.env
cd 3rdparty/Megatron-LM
git rev-parse HEAD > /MCORE_COMMIT_HASH.env
pip install -e .

mkdir -p /workspace/ft-llm && cp -r /host-impl/. /workspace/ft-llm/
cd /workspace/ft-llm && pip install -r requirements.txt
echo 'export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"' >> /etc/bash.bashrc
python -c "import megatron.core as m; print('megatron-core', m.__version__)"
python -c "import mlperf_logging; print('mlperf_logging OK')"
INNER

enroot export --output /scratch/containers/l2l-nemo.sqsh $NAME
enroot remove -f $NAME
ls -lh /scratch/containers/
