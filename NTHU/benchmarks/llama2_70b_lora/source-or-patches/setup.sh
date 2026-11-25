#!/bin/bash
# Build everything the NVIDIA-code path needs, from nothing, on this cluster.
#
#   nvidia/setup.sh                 # all steps, in order
#   nvidia/setup.sh source assets   # just these
#
# Steps (each is idempotent and skips what already exists):
#   container  the Apptainer image; submits a ~40 min Slurm build (scripts/build_container.sh)
#              and stops, since the later steps need it. Re-run setup.sh afterwards.
#   pyextra    the (empty) directory launch.sh binds at /pyextra
#   source     NVIDIA's pristine v5.0 llama2_70b_lora code at a pinned commit, copied to
#              $IMPL (default $WORK_ROOT/nvidia_impl) and patched by nvidia/patch_nvidia_impl.py
#   assets     the dataset and the NeMo checkpoint, fetched by NVIDIA's own scripts run
#              inside the container (138 GB, up to ~30 min)
#
# Paths come from configs/site.env (WORK_ROOT, DATA_ROOT) and .build_dir (BUILD_DIR),
# both committed with this cluster's real values.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
[ -f "$ROOT/configs/site.env" ] && . "$ROOT/configs/site.env"
BUILD_DIR="${BUILD_DIR:-$(cat "$ROOT/.build_dir")}"
: "${DATA_ROOT:=$WORK_ROOT}"   # site.env.example leaves DATA_ROOT commented out; it then defaults to WORK_ROOT
SIF="${SIF:-$BUILD_DIR/containers/llama2_70b_lora_ngc25.04.sif}"
IMPL="${IMPL:-$WORK_ROOT/nvidia_impl}"
PRISTINE="${PRISTINE:-$WORK_ROOT/nvidia_v50/repo}"
DATADIR="${DATADIR:-$DATA_ROOT/nvidia_npy}"
CKPT="${CKPT:-$DATA_ROOT/nemo_torch_dist}"
# mlcommons/training_results_v5.0 at the commit this work was built and measured on.
REPO_URL=https://github.com/mlcommons/training_results_v5.0.git
REPO_COMMIT=d7e703fccc1c86e70cb31bac108f99072e34dba0
SUBDIR=NVIDIA/benchmarks/llama2_70b_lora/implementations/tyche_ngpu8_ngc25.04_nemo

steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=(container pyextra source assets)

for step in "${steps[@]}"; do
  echo "== $step"
  case "$step" in
    container)
      if [ -e "$SIF" ]; then echo "have $SIF"; else
        "$ROOT/scripts/build_container.sh"
        echo "Container build submitted. Wait for the job, then re-run nvidia/setup.sh."; exit 0
      fi ;;
    pyextra)
      mkdir -p "$BUILD_DIR/containers/pyextra"; echo "$BUILD_DIR/containers/pyextra" ;;
    source)
      if [ ! -d "$PRISTINE/.git" ]; then
        mkdir -p "$(dirname "$PRISTINE")"
        git clone --filter=blob:none --no-checkout "$REPO_URL" "$PRISTINE"
        git -C "$PRISTINE" sparse-checkout set "${SUBDIR%/*/*}"
      fi
      git -C "$PRISTINE" checkout -q "$REPO_COMMIT"
      rm -rf "$IMPL"; cp -a "$PRISTINE/$SUBDIR" "$IMPL"
      # The upstream script names a Hugging Face repo that no longer exists:
      # michal1270/llama2_70b_nemo_zarr was renamed to .../llama2_70b_nemo_torch_dist
      # some time after this code was pinned (confirmed by matching repo sha and
      # .gitattributes oid against the checkpoint already fetched under DATA_ROOT).
      sed -i 's/llama2_70b_nemo_zarr/llama2_70b_nemo_torch_dist/' "$IMPL/scripts/download_model.py"
      python3 "$HERE/patch_nvidia_impl.py" "$IMPL"
      echo "patched copy: $IMPL" ;;
    assets)
      [ -e "$SIF" ] || { echo "no container yet: run the container step first" >&2; exit 1; }
      mkdir -p "$DATADIR" "$CKPT"
      run() { apptainer exec --bind "$DATA_ROOT:/data" --pwd /workspace/ft-llm \
                --bind "$IMPL:/workspace/ft-llm" "$SIF" "$@"; }
      [ -e "$DATADIR/train.npy" ] || run python scripts/download_dataset.py --data_dir "/data/${DATADIR#$DATA_ROOT/}"
      [ -e "$CKPT/weights/common.pt" ] || run python scripts/download_model.py --model_dir "/data/${CKPT#$DATA_ROOT/}"
      # The Hugging Face repo (michal1270/llama2_70b_nemo_torch_dist, the replacement for the
      # renamed michal1270/llama2_70b_nemo_zarr) stores each ~69 GB weight shard pre-split into
      # two files, weights/__R_S.distcp0.part and weights/__R_S.distcp1.part, rather than the
      # single weights/__R_S.distcp file torch.distributed.checkpoint expects -- this is how the
      # repo itself is laid out, not a download bug, and download_model.py reports success
      # without checking for it. Rejoin them (verified byte-identical to a known-good checkpoint
      # in the 2026-09-22 reproduction check) if the joined file is not already there.
      shopt -s nullglob
      for part0 in "$CKPT"/weights/*.distcp0.part; do
        joined="${part0%0.part}"; part1="${joined}1.part"
        [ -e "$joined" ] && continue
        [ -e "$part1" ] || { echo "missing $part1 to rejoin $part0" >&2; exit 1; }
        echo "rejoining $(basename "$joined")"; cat "$part0" "$part1" > "$joined.tmp" && mv "$joined.tmp" "$joined"
      done
      shopt -u nullglob
      ls "$CKPT"/weights/*.distcp >/dev/null 2>&1 || { echo "no checkpoint *.distcp shard files after download+rejoin -- see nvidia/setup.sh" >&2; exit 1; }
      ls -la "$DATADIR" "$CKPT/weights" | head -20 ;;
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
