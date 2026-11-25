#!/bin/bash
# Download the Megatron-Bridge dist-checkpoint used by the NVIDIA/NeMo submissions.
# Public repo, named in NVIDIA's own scripts/download_model.py. ~138 GB.
set -uo pipefail
source /usr/share/modules/init/bash 2>/dev/null; module load conda/latest
export PATH=/mnt/home/unicorn/.conda/envs/llama2lora/bin:$PATH
unset HF_HOME; export HF_HUB_CACHE=/work/benchmark/MLPerf/tmp/hfcache
OUT=/work/benchmark/MLPerf/data/models/llama2_70b_megatron_bridge
mkdir -p "$OUT"
date -Is
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("michal1270/llama2_70b_megatron_bridge",
                      local_dir="/work/benchmark/MLPerf/data/models/llama2_70b_megatron_bridge",
                      max_workers=16)
print("DONE", p)
PY
echo "EXIT=$?"; date -Is; du -sh "$OUT"
