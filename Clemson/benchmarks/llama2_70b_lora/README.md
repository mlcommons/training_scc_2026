# Clemson: Llama 2 70B LoRA (MLPerf Training, SCC26)

## System
2 nodes, each with 4x NVIDIA A100-SXM4-40GB (NVLink), 2x AMD EPYC 7763, 1 TB RAM, and 2x InfiniBand HDR 200 Gb/s.
Details: `systems/student_system.json`.

## Software
MLCommons reference implementation (Hugging Face Accelerate + DeepSpeed ZeRO-3), in a container built from
`source-or-patches/Dockerfile.olympus` (NGC PyTorch 24.01, flash-attn 2.4.1, the reference's pinned libraries).
Launched with Slurm and Docker across both nodes (`source-or-patches/llama_final.sbatch` -> `run_and_time.sh`).

## How to run
1. Build the image on each node: `docker build -f Dockerfile.olympus -t mlperf-llama2-lora:olympus .`
2. Set paths and settings in `config.sh` (dataset, model, network interface, InfiniBand cards).
3. Submit one run: `sbatch --export=ALL,SEED=<seed> llama_final.sbatch`

## Changes from the reference
Training math is unchanged: same model, data, optimizer and hyperparameters (global batch 8, learning rate 4e-4).

1. **GPUDirect RDMA forced on (`NCCL_NET_GDR_LEVEL=SYS`).** On this system each GPU reaches its InfiniBand card
   through the CPU, so NCCL skipped GPUDirect by default even with `nvidia-peermem` loaded. Forcing it raised
   all-gather bandwidth between nodes from 7.6 to 23.6 GB/s and cut training step time by about 40% (37.8 -> 22.8 s).
2. **Evaluation delay aligned with the rules.** The reference trainer also evaluated at step 144 (`step >= delay`),
   which the MLPerf logger never records (`step > delay`) and the rules skip. Adding 1 to the delay removes that
   unused evaluation (~5 min per run). Logged evaluations are unchanged: every 384 samples, from step 192.
3. **Evaluation batch size 2 with per-sample loss averaging.** Batching halves the weight gathers per evaluation
   (315 -> 198 s). Because samples are scored on 75 to 2,206 tokens, plain batching changes the averaged loss, so a
   small trainer subclass averages each sample on its own, exactly reproducing the reference metric (unit test:
   4.416618 vs 4.416618). Patch: `source-or-patches/train_eval_fixes.patch`.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (memory allocator setting).
5. **Submission fields** in the MLPerf logger set to this team (`mlperf_logging_submission_fields.patch`).

Measured effect (practice, same convergence point): 263 min (reference setup) -> 249 min (changes 2-3) -> 152.5 min (all changes).
Changes tested that did not help: DeepSpeed memory settings and other NCCL options.

## Notes
Runs were made during the SCC26 benchmarking window (Nov 16, 2026, 9am-5pm CST). At about 135-155 min per run, only 2-3 runs fit in the window, so this submission has fewer than the 10 runs MLPerf requires for full credit. All runs are genuine, with random seeds, and none were omitted. Run 1 (job 7595) converged at 2,688 samples in 134.1 min; run 2 (job 7596) at 3,072 samples in 153.3 min. The SCC committee required three runs for certification; run 3 (job 7600) did not reach the target by its 3,072-sample evaluation and was cancelled at 16:55 CST to meet the 17:00 deadline; its unmodified partial log is included as result_2.txt as the one permitted failed run.
