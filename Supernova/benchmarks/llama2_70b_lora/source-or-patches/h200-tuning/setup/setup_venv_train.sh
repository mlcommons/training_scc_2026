#!/bin/bash
set -x
H=/home/alien/mlperf-sc26
PYBIN=/home/apps/spack/opt/spack/linux-zen4/python-3.12.12-ccmdqmn4a7kvz26euhyr5jlom4e6yoml/bin/python3.12
V=$H/venv-train
[ -x $V/bin/python ] || uv venv --python $PYBIN $V || exit 1
OV="--override $H/h200-tuning/setup/venv-train-overrides.txt"
uv pip install --python $V/bin/python $OV --index-url https://download.pytorch.org/whl/nightly/cu134 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.15.0.dev20261004+cu134" 2>&1 | tail -3 || exit 1
uv pip install --python $V/bin/python $OV \
  "omegaconf==2.3.0" "transformers==5.5.4" "hydra-core==1.3.2" \
  "peft>=0.18.1" "datasets>=2.20.0" accelerate einops tensorboard rich pyyaml tqdm \
  nvidia-resiliency-ext ninja cmake packaging pybind11 nvidia-modelopt \
  nvidia-cudnn-frontend nvidia-mathdx==25.1.1 huggingface_hub \
  "git+https://github.com/mlcommons/logging.git@6.0.0-rc6" \
  "git+https://github.com/NVIDIA/mlperf-common.git@c02c93c0414071400a10716d9cd5cb8f668b78d4" 2>&1 | tail -3 || exit 1
echo "VENV_TRAIN_BASE done"
