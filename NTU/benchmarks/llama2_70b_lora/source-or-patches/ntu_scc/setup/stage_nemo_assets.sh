#!/bin/bash
# Stage the NeMo llama2_70b_lora assets to THIS node's local /scratch.
# Everything is read many times per run; /work/benchmark is NFS over a single
# 1 GbE link, so local copies are mandatory for sane load times.
set -euo pipefail
SRC=/work/benchmark/MLPerf
DST=/scratch/benchmark/MLPerf
mkdir -p $DST/data $DST/containers
echo "=== $(hostname -s) staging start $(date -Is) ==="

# container image (skip if already byte-identical)
if [ ! -f /scratch/containers/l2l-nemo.sqsh ] || \
   [ "$(stat -c%s /scratch/containers/l2l-nemo.sqsh)" != "$(stat -c%s $SRC/containers/l2l-nemo.sqsh)" ]; then
  echo "--- image ---"; rsync -a --inplace $SRC/containers/l2l-nemo.sqsh /scratch/containers/
else echo "--- image already present ---"; fi

echo "--- npy dataset ---"
rsync -a $SRC/data/dataset/nemo_npy $DST/data/dataset/
echo "--- megatron checkpoint (129 GB) ---"
rsync -a $SRC/data/models/llama2_70b_megatron_bridge $DST/data/models/

echo "=== verify against expected-mounts.csv ==="
python3 - <<'PY'
import csv,os
ck='/scratch/benchmark/MLPerf/data/models/llama2_70b_megatron_bridge'
dd='/scratch/benchmark/MLPerf/data/dataset/nemo_npy'
ok=bad=0
for r in csv.DictReader(open('/work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b/expected-mounts.csv')):
    base = ck if r['key']=='CHECKPOINT' else dd
    p=os.path.join(base, r['relative_path'])
    good = os.path.exists(p) and os.path.getsize(p)==int(r['num_bytes'])
    ok+=good; bad+=(not good)
    if not good: print("  BAD", r['relative_path'])
print(f"  expected-mounts: {ok} OK, {bad} bad")
PY
echo "=== $(hostname -s) staging done $(date -Is) ==="
du -sh /scratch/containers/l2l-nemo.sqsh $DST/data/models/* $DST/data/dataset/nemo_npy
