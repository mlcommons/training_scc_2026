#!/usr/bin/env bash
# Updates our scripts and configs (git pull), installs the competition's checkers (mlperf-logging 6.0.0-rc6) in .venv,
# then makes data/ if it is not complete:
# the dataset and Llama 2 70B, converted by scripts/prepare_data_and_model.sh in AMD's container.
# The first time takes hours (run it inside tmux) and asks for a Hugging Face token from an account with access to
# https://huggingface.co/meta-llama/Llama-2-70b-hf
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
git config --global --get-all safe.directory | grep -qx "$PWD" || git config --global --add safe.directory "$PWD"   # shared repository: git works for every team member
echo "=== git pull"
git pull -q --ff-only
echo "=== mlperf-logging 6.0.0-rc6 in .venv"
if [ ! -x .venv/bin/python ]; then
  python3 -c "import ensurepip" 2>/dev/null || sudo apt-get install -y -qq python3-venv
  python3 -m venv .venv
fi
.venv/bin/python -c "import mlperf_logging" 2>/dev/null || .venv/bin/pip install -q "git+https://github.com/mlperf/logging.git@6.0.0-rc6"
if [ -f data/train.npy ] && [ -f data/model/model_config.yaml ]; then
  echo "=== data/ is complete. run: ./run.sh <config> [runs]"
  exit
fi
eval "$(grep '^CONT=' scripts/mlperf.sbatch)"
[ -n "$HF_TOKEN" ] || { read -rsp "Hugging Face token (hidden): " HF_TOKEN; echo; }
export HF_TOKEN
mkdir -p data
echo "=== dataset and Llama 2 70B: download and convert (hours), log in data/prepare.log"
docker run --rm -e HF_TOKEN -e HF_HOME=/data/hf \
  -v "$PWD/data:/data" -v "$PWD/implementation/scripts:/workspace/code/scripts:ro" \
  -v "$PWD/scripts/prepare_data_and_model.sh:/workspace/prepare_data_and_model.sh:ro" \
  --net=host --uts=host --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined \
  $CONT bash /workspace/prepare_data_and_model.sh > data/prepare.log 2>&1 \
  || { echo "ERROR: see data/prepare.log. Run scripts/setup.sh again to retry (the download resumes)."; exit 1; }
echo "=== done. run: ./run.sh <config> [runs]"
