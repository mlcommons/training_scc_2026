# NTU SCC: reproducing the 16.49-min llama2-70B LoRA run

Best run: **TTT 16.45 min** (job 3618, 2026-10-08, seed 1235). Reproducibility: 15 seeds at 400 W (2026-10-14/15) → 10 converge at 3072 samples in 16.72–16.77 min, 3 at 3456 (≈19.0), 2 at 3840 (≈21.1); per-seed table, lower-power-cap runs and the stock baseline in [SEEDS.md](SEEDS.md).
Previous best: 18.06 min (job 2908, 2026-10-02), the same settings without the kernel add-ons.

**Submission set (2026-10-17, jobs 4479–4488):** the 10 seeds re-run with every record a submission needs (page cache cleared and
logged, org/platform set), 400 W + SM clock lock 1200 MHz + CPUs 2.2 GHz: 7 × 3072 samples (16.68–16.75 min), 3 × 3456 (18.9);
official package checker SUCCESS, RCP PASS, summarizer score **17.27 min** under the v6.0.0 rules. Package recipe and checker
results: [`ntu_scc/submission/`](submission/) (`make_package.sh`, `VALIDATION.md`). Not yet valid under v6.1.0 (missing
disclosure keys, see Known issues). To rebuild all of this elsewhere, follow **Recreate on another machine** at the end.

| | value |
|---|---|
| hardware | 16x H200 NVL: `ntu-scc01` + `ntu-scc02`, 8 GPUs each, 4-GPU NVLink bridges (0-3, 4-7) |
| fabric | InfiniBand; NCCL/Gloo use `ibp101s0f0` |
| software | Slurm + pyxis/enroot; NVIDIA driver 580.173.02 |
| power | **400 W per GPU** |
| parallelism | TP=2 + sequence parallel, MBS 1, GBS 8, LR 5e-4 |
| GEMMs | FP8 (TE 2.14, delayed scaling, TE-ops MLP `USE_TE_OPS=1`); DeepGEMM for the FP8 dgrad GEMMs |
| kernels | NTU add-ons (`ntu_scc/addons/`): fused LoRA output, qkv-LoRA on the local shard, FP8 dy all-gather, dSwiGLU and dgrad operands quantized once to E4M3 1x128, Triton RoPE |
| memory | no activation recompute, made possible by FP8 weight transposes built on demand |
| step time | 2.365 s (2.512 s in job 2908) |
| startup | job start -> `run_start` 134-145 s |
| profile | `figs/barriers_16gpu_best_norecomp_400W.png` in the project dir (not in this repo) |

## What is where

| path | what |
|---|---|
| `configs/config_NTUSCC16TP2_2x8x1xtp2pp1cp1.sh` | the 16-GPU TP=2 config. It still sets recompute block/16; the launcher's `FORCE_ENV` turns recompute off |
| `run.sub` | opt-in startup fixes: `SKIP_SHINIT`, `SKIP_SYS_DIAG`, `CACHE_SYSINFO` |
| `ntu_scc/deepgemm_dgrad/` | DeepGEMM dgrad + `FP8W_T_ONDEMAND` (`dg_dgrad.py`), loaded by a site-packages add-on (`zz_dg_dgrad.pth`, `dg_dgrad_autoload.py`) so **the benchmark source stays stock**; `train_dgdgrad.py` is the older route; `build_deepgemm.sh` |
| `ntu_scc/launcher/run_nemo_l2l.sh` | the launcher, identical to the cluster's `tools/run_nemo_l2l.sh`. Holds the startup defaults and the patch mounts |
| `ntu_scc/launcher/run_best_18min.sh` | **the exact settings of the 16.49-min run** (name kept for continuity) |
| `ntu_scc/addons/` | the kernel add-ons (Triton + autograd re-bindings, no source change), their unit tests, and a README with per-add-on numerics |
| `ntu_scc/attn_bwd/` | hand-written Hopper FP8 flash-attention backward (FA3-derived, CUDA/CuTe), hook `NTU_ATTN_BWD=1`; step −3.8%, see its README |
| `ntu_scc/setup/` | building the image, downloading/converting data and checkpoint, staging to node-local `/scratch` |
| `ntu_scc/powercap/` | per-GPU caps through the site tool `/mnt/opt/systools/set-gpu-freq` (200-400 W) |
| `ntu_scc/gpubench/` | per-card benchmark suite + 2026-10-07 results (16 cards x 600/400/300 W) |
| `ntu_scc/gpu_recover/` | runbook + script to free a hung GPU without rebooting (then cold-reboot before real work) |
| `ntu_scc/ncclcheck/` | 2-minute cross-node NCCL correctness check (known patterns, GPUDirect RDMA on/off): run it after any GPU incident |
| `ntu_scc/systool/drop-caches/` | setuid-root helper (C launcher + bash, like the site's set-cpu-freq) that drops the page cache so run.sub can log the mandatory `cache_clear`; `make`, `make test`, `make install` (root) |
| `ntu_scc/submission/` | system JSON, `make_package.sh` (assembles the package from a runs list and runs rcp/package/summarizer checkers), `VALIDATION.md`, the 10 submission runs and their power lines |
| `ntu_scc/tools/` | `submit_seeds.sh` (runs seeds back to back, re-applies node settings, validates each log, packs), `watchjob.sh`, `govlog.py` (setting-drift log via Prometheus), `power/promwin.py` + `promother.py` (per-window power stats and the node power decomposition), `sweep_until_3072.sh`, `nccl_pair_bench.py` |

## One-time setup

All paths below are this cluster's; the scripts hard-code them.

1. **Image.** `ntu_scc/setup/build_nemo_image.sh` writes `/scratch/containers/l2l-nemo.sqsh`.
   - Base: NGC pytorch 26.04 + cuDNN 9.21.1.3 + Megatron-Bridge 26.04-alpha.rc1.
   - `mlperf-logging` is pinned to **6.0.0-rc6**.
   - Copy the image to `/work/benchmark/MLPerf/containers/` for staging.
2. **Data.**
   - `scripts/download_dataset.py` downloads to `/work/benchmark/MLPerf/data/dataset/OFFICIAL_scrolls`.
   - `ntu_scc/setup/make_npy_in_container.py` (run inside the image) writes `.../dataset/nemo_npy` and checks the
     official sha256.
3. **Checkpoint.** `ntu_scc/setup/fetch_megatron_ckpt.sh` fetches the Megatron-Bridge checkpoint, ~138 GB:
   `.../data/models/llama2_70b_megatron_bridge`.
4. **Stage to every node's local disk** (NFS is 1 GbE).
   - `ntu_scc/setup/stage_nemo_assets.sh` on one node.
   - `ntu_scc/setup/stage_node.sh <node>-ib...` copies it to the others over IB.
5. **DeepGEMM.** `ntu_scc/deepgemm_dgrad/build_deepgemm.sh` builds DeepGEMM 2.8.1 (057ca59) into
   `/work/benchmark/MLPerf/gemmlib/{pylib_dg,sysroot}`.
6. **Patches.** The launcher mounts them from `/work/benchmark/MLPerf/patches`:
   ```bash
   mkdir -p /work/benchmark/MLPerf/patches /work/benchmark/MLPerf/cache
   cp ntu_scc/deepgemm_dgrad/{dg_dgrad.py,dg_dgrad_autoload.py,zz_dg_dgrad.pth,train_dgdgrad.py} /work/benchmark/MLPerf/patches/
   cp ntu_scc/addons/{ntu_lora_fused.py,ntu_te_comm.py,ntu_dg_quant.py,ntu_rope.py} /work/benchmark/MLPerf/patches/
   ```
   The kernel add-ons (`ntu_scc/addons/README.md`) are on by default in `run_best_18min.sh`. Each has its own env flag.
7. **Cache-clear systool (needed for submittable logs).** Build as the user, install as root, once per node (or once if
   `/mnt/opt` is shared):
   ```bash
   cd ntu_scc/systool/drop-caches && make && make test        # user
   sudo make install                                           # root: /mnt/opt/systools/drop-caches (4755) + drop-caches.sh (0755)
   /mnt/opt/systools/drop-caches                               # smoke test: prints "dropped caches: Cached X kB -> Y kB"
   ```
   The launcher enables `CLEAR_CACHES=1 DROPCACHE_CMD=/mnt/opt/systools/drop-caches` automatically when the setuid file exists.
   Two things run.sub's cache_clear logging step needs in our layout (both already in the launcher / rootfs, see
   `submission/VALIDATION.md`): `/workspace/ft-llm` on `PYTHONPATH`, and a writable `/compile_cache` mountpoint inside the persisted
   container rootfs (`chmod 755 ~/.local/share/enroot/pyxis_llama2_70b_lora_persist_l2l-nemo/compile_cache`; enroot leaves it mode 000).
8. **Checker tooling** (for packaging): `ntu_scc/submission/make_package.sh` clones `mlcommons/logging` into
   `/work/benchmark/MLPerf/tools/mlc/logging` and installs it into a venv on first use (needs a python with pip, e.g. the
   `llama2lora` conda env).

## Each run

**Final SCC26 configuration, one command:** `NEXP=10 bash ntu_scc/launcher/run_final.sh` for the submission (random seed base,
10 consecutive seeds, one Slurm job per seed back to back, every run reported, each log validated, package built and checked); `bash ntu_scc/launcher/run_final.sh` for a
single test run; `CHECK_ONLY=1` to only set/verify the node state. It applies
and verifies the node state (390 W × 8, SM lock 345–1050 MHz, CPUs performance / 2.2 GHz, drop-caches systool present), opens the
persistent ssh sessions, submits the best config and prints the job id and how to watch it. Fans must be pinned at 40 % in the BMC
by hand (job 4853 with exactly this state: 17.27 min, step 2.396 s, PDU median 9.33 / max 9.69 kW, no 1-s sample ≥ 10 kW).
Certification timestamps: the SCC26 organizers stated (mail of 2026-11-16) that MLPerf needs **no** manual
`cert-N-mlperf.*` files; the judges use the mllog timestamps inside the result logs, and extra files in the `mlperf/` upload could
interfere with the MLCommons tooling. So upload only `NTU/`. `CERT=N bash ntu_scc/launcher/run_final.sh` can still write
`cert-N-mlperf.{tstamps,input,rslts,run}` for our own records, into `/work/benchmark/MLPerf/submission/scc26-cert/` (outside the
upload directory), via `ntu_scc/submission/make_cert.sh`; it is off by default. All nodes are NTP-synced to ntp2.ntu.edu.tw
(offsets < 20 ms), which the rules strongly encourage so the log timestamps line up with the 1-s PDU trace.
The manual steps below are what the script does.

```bash
# 1. node state, BEFORE launching (caps revert to 300 W at boot; other users change caps and CPU governors on this cluster):
for n in ntu-scc01 ntu-scc02; do
  ssh $n 'BUDGET=3200 /work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b/ntu_scc/powercap/applycaps.sh 400 400 400 400 400 400 400 400'   # 400 W x 8
  ssh $n '/work/benchmark/MLPerf/MLPerf-training-llama2-lora-70b/ntu_scc/powercap/setclk.sh 1050'                                           # SM lock 345-1050 MHz: keeps the cluster < 10 kW (+1.0 % step); 1200 = -0.4 % step but 9 % of PSU samples > 10 kW
  ssh $n "printf '5\n2.2\ns\nyes\n' | /mnt/opt/systools/set-cpu-freq"                                       # performance governor, 2.2 GHz max (0 %, -250 W)
done
# 2. keep one ssh session open on each compute node for the whole job (logind RemoveIPC wipes /dev/shm when your last session
#    ends -> dataloader hang):  nohup ssh -n -o ServerAliveInterval=60 ntu-scc02 'sleep 86400' &
# 3. launch (one sbatch over both nodes; results in /work/benchmark/MLPerf/runs/$TAG)
bash ntu_scc/launcher/run_best_18min.sh               # SEED_BASE=1234 NEXP=1 by default
# 3b. or a whole submission set: runs seeds back to back, re-applies the node state before each, validates every log
#     (cache drop on both nodes, cache_clear records, no placeholders, compliance_checker), then packs and checks
SEEDS="1235 1237 1240 1241 1242 1244 1245 1246 1247 1248" OUT=/work/benchmark/MLPerf/runs/submit_seeds \
  nohup setsid bash ntu_scc/tools/submit_seeds.sh > /work/benchmark/MLPerf/runs/submit_seeds/submit_seeds.log 2>&1 &
# 4. (re)build the package from a runs list and run the official checkers
bash ntu_scc/submission/make_package.sh ntu_scc/submission/submission_runs.txt 6.0.0
```

**What to expect:**
- Step time ~2.365 s, with 5 evals of ~24 s.
- Converges at 3072 samples (TTT ~17.0 min) or 3456 samples (~18.8 min, estimated).
- At exit every rank prints `[NTU_*] calls {...}` / `[DG_DGRAD] calls {...}`: all counters except `fallback` should be
  non-zero. In particular `prequant` should equal `deepgemm`. If they are 0, an add-on is inert (e.g. `USE_TE_OPS` unset). Which one is a seed / data-order coin flip, and
  the same seed is not bit-deterministic.
- Peak memory 128.4 GiB allocated / ~137 GiB reserved of 139.8: only ~3 GiB of headroom.

## Known issues

- **Compliance (solved 2026-10-17):** logs lacked `cache_clear` because `sudo sysctl vm.drop_caches=3` needs root. The setuid
  systool `ntu_scc/systool/drop-caches` fixes it (step 7 above); runs before that date are not submittable.
- **Ruleset:** the stack is pinned to the v6.0 reference and mlperf-logging 6.0.0-rc6 and validates under **6.0.0**. Under
  **6.1.0** every log fails for the nine new mandatory disclosure keys (`lowest_numerical_precision_in_{linear,attn,comm}`,
  `tensor/pipeline/context/expert_parallelism`, `micro_batch_size`, `config_filename`), which must be emitted by the training
  code at init (logging-only change) and require re-running the set. Open until the competition's rulebook is known.
- **Other users change node settings:** GPU caps were found reset to ~300 W/GPU (once to 600 W) and CPU governors/caps changed
  between runs; the drivers re-apply the caps / lock / 2.2 GHz before every job and print a `DRIFT` line when they had to.
- **10 kW power limit (SC26: PDU reading ≥ 10,000 W at any time is a penalty, ≥ 11,000 W a DQ; the PDU reads AC):** with 16 GPUs at
  400 W the cluster median is ~9.5 kW but the PSU meters catch sub-second bursts. The 100 ms trace (`ntu_scc/tools/power/`) shows
  every card overshooting its 400 W limit 25–50 % of the time in bursts ≤ 1 s (p99 ~430 W, max 445 W; the limiter regulates a
  ~1 s average), synchronized across the cluster by the step, plus fan ramps of +90–130 W per node (40 % → 46 % fan speed). With
  the 1200 MHz lock that put 9 % of samples over 10 kW (max 10.1–10.4 kW); **with the 1050 MHz lock nothing exceeded 10 kW on any
  meter at any averaging window** (max 9.8 kW in cockpit units = PSU output × 1.04) for +1.0 % step time (jobs 4707 vs 4742, SEEDS.md).
  Lower clock ceilings buy little more (the cards already sit on the power limit) and cost 2–7 % step; the next levers are a 390 W
  cap (−160 W cluster, ~+2 %) or taking idle scc03 (1.0–1.3 kW) off the circuit. CPUs are not involved (±20 W).
- **`PARALLEL_LOAD=True` hung once in 11 runs** (NCCL broadcast in the checkpoint exchange). `PARALLEL_LOAD=False` costs
  ~35 s of startup.
- **scc02 GPU 4 is ~4-5 % slower** at <=400 W: it holds a lower clock at the same power. It sets the 16-GPU pace (see
  `ntu_scc/gpubench/RESULTS.md`).
- **Do not run direct cross-socket GPU-to-GPU copies on scc02:** GPU1<->GPU4 hangs GPU 1 (Xid 119). NCCL does not use that
  path. If a GPU hangs: `ntu_scc/gpu_recover/` to free it, then a cold reboot and `ntu_scc/ncclcheck/` before real work.

## Recreate on another machine

Everything needed is in this repo except the data, the checkpoint and the two site setuid tools; the launcher hard-codes the
paths below, so either recreate them or edit `ntu_scc/launcher/run_nemo_l2l.sh` (top of file: `LOCAL`, `CONT`, `DATADIR`,
`MODEL`, `LOGDIR`, the enroot paths) and `ntu_scc/submission/make_package.sh` (`SUB`, `MLC`, `PYBASE`).

| needs | what we use | notes |
|---|---|---|
| scheduler | Slurm with pyxis + enroot, `--mpi=pmix`; 2 nodes × 8 GPUs in one partition (`gpu`) | enroot's data path comes from `/etc/enroot/enroot.conf`, not from the user's env: here it is NFS home, shared by both nodes |
| shared FS | `/work/benchmark/MLPerf/` (repo, `patches/`, `cache/`, `runs/`, `gemmlib/`, `tools/mlc/`, `submission/`) | 1 GbE NFS here, so data/checkpoint are staged node-locally |
| node-local | `/scratch/containers/*.sqsh`, `/scratch/benchmark/MLPerf/data/{dataset/nemo_npy,models/llama2_70b_megatron_bridge}` (0.2 GB + 129 GB) | `ntu_scc/setup/stage_*.sh` |
| image | `ntu_scc/setup/build_nemo_image.sh` → `l2l-nemo.sqsh` from NGC `pytorch:26.04-py3` (needs `pytorch2604.sqsh` imported first) | pins cuDNN 9.21.1.3, Megatron-Bridge 26.04-alpha.rc1, mlperf-logging 6.0.0-rc6 |
| data + checkpoint | steps 2–4 of One-time setup (official download + `make_npy_in_container.py`; `fetch_megatron_ckpt.sh`) | sha256 is checked by the npy script |
| DeepGEMM | step 5 (`build_deepgemm.sh`) | the FP8 attention backward `.so` is built by `ntu_scc/attn_bwd/build.sh` |
| patches dir | step 6 (`cp` of add-ons and `.pth` into `/work/benchmark/MLPerf/patches`) | the `.pth` autoload is what makes the add-ons apply without touching the benchmark source |
| root-level tools | `/mnt/opt/systools/set-gpu-freq`, `set-cpu-freq` (site, setuid; `ntu_scc/powercap/` wraps them) and `drop-caches` (ours, step 7) | without `set-gpu-freq`, `nvidia-smi -pl 400 -lgc 345,1200` as root does the same; without `set-cpu-freq`, write `scaling_governor` / `scaling_max_freq` as root |
| checker tooling | step 8 (`make_package.sh` bootstraps `mlcommons/logging` in a venv) | a python with `pip` is required; `/usr/bin/python3` here has none |
| telemetry (optional) | cockpit Prometheus on `ntu-scc04:9090` (tunnel `ssh -N -L 19090:localhost:9090 ntu-scc04`) for `tools/govlog.py` and `tools/power/*` | the cockpit REST API needs auth since 2026-10-17; Prometheus does not |

Order of a full rebuild: image → data/checkpoint → stage to nodes → DeepGEMM + attention `.so` → patches dir → systool install →
node state (caps/lock/governor) → one `run_best_18min.sh` run and check the log for `cache_clear` and the `[NTU_*] calls` counters →
`submit_seeds.sh` for the set → `make_package.sh`.
