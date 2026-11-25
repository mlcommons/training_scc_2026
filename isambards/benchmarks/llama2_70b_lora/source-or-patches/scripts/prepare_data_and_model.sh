#!/bin/bash
# AMD's scripts/prepare_data_and_model.sh from the v5.1 container, changed for our layout (see git history). It runs inside
# the container, started by scripts/setup.sh. Changes: one /data mount (no /ckpt), NeMo-Run pinned, a clean start, and the
# result directly in /data: train.npy, validation.npy, model/.
set -euo pipefail

SCRIPT_DIR=/workspace/code/scripts

# Clean start: an earlier failed attempt can leave half-written folders. The model download in /data/hf resumes.
rm -rf /data/gov_report /data/model /data/ckpt /data/train.npy /data/validation.npy
cd /workspace/code
pip uninstall -y bitsandbytes   # AMD's instructions: remove it before this script

# Install Dependencies
pip install huggingface-hub==0.30.0
rm -rf /workspace/deps/NeMo && pip uninstall -y nemo-toolkit \
    && git clone https://github.com/NVIDIA/NeMo.git && cd NeMo && git checkout 25.04-alpha.rc1 && pip install -e ".[nlp]" && cd ..

rm -rf /workspace/deps/megatron_lm && pip uninstall -y megatron-core \
    && git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM && cd Megatron-LM && git checkout 25.04-alpha.rc1 \
    && pip install . && cd megatron/core/datasets && make && cd ../../../.. \
    && export PYTHONPATH="${PYTHONPATH}:/workspace/ft-llm/Megatron-LM"

pip install git+https://github.com/NVIDIA-NeMo/Run.git@be41071f72edb1216f5e38ed8df5fc37ba3393dd   # 9 Oct 2026; AMD's script takes the latest
pip install leptonai==0.27.3   # NeMo-Run imports leptonai.api.v1, which leptonai 0.47 (20 Oct 2026) no longer has

cd /workspace/code/

python ${SCRIPT_DIR}/download_dataset.py --data_dir /data/gov_report  # download dataset
python ${SCRIPT_DIR}/convert_dataset.py --data_dir /data/gov_report
python ${SCRIPT_DIR}/convert_model.py --output_path=/data/ckpt/

# Organize files for finetuning
mv /data/gov_report/train.npy /data/gov_report/validation.npy /data/
rmdir /data/gov_report
mv /data/ckpt/weights/ /data/ckpt/model_weights/
cd /data/ckpt/model_weights/
for d in module.*/; do mv -- "$d" "model.${d#module.}"; done
cp /data/ckpt/context/nemo_tokenizer/tokenizer.model /data/ckpt/tokenizer.model
cp /workspace/code/scripts/model_config.yaml /data/ckpt/model_config.yaml

mv /data/ckpt /data/model
rm -rf /data/hf   # the Hugging Face download (about 130 GB)

echo "Data and model preparation completed successfully"
