# Isambards MLPerf

- **Model:** Llama 2 70B LoRA on GovReport using AMD's v5.1 NeMo container.
- **Hardware:** 8 MI300X GPUs.
- **Target:** validation loss ≤ 0.925.
- **Power limit:** 10 kW.
- **Submission name:** Isambards.
- **Implementation:** `implementation/` is AMD's published v5.1 MI300X code ([training_results_v5.1](https://github.com/mlcommons/training_results_v5.1/tree/main/AMD/benchmarks/llama2_70b_lora/implementations/MI300X_EPYC_9575F_pytorch_llama2_70b)).
- **Patches:** Two patches to the above code have been submitted for organiser review in [mlcommons/training_scc_2026](https://github.com/mlcommons/training_scc_2026).
- **Config:** Our config uses AMD's `config_MI300X_1x8x1.sh` with our submission name, `HSA_ENABLE_IPC_MODE_LEGACY=0` for cn01's driver and AMD's machine settings from `runtime_tunables.sh`.

### Setup

```bash
git clone https://github.com/UK-SCC/mlperf.git
cd mlperf
bash scripts/setup.sh
```

- Requires Docker and Slurm
- Setup installs the checkers and prepares `data/` (if missing). Downloading the model requires authorised Hugging Face access (and takes a very long time).
- Training runs inside the pinned Docker container

### Run

One run with the default config:
```bash
./run.sh amd 1
```

Override settings for one job:
```bash
./run.sh amd 1 LR=0.0005
```

- **Syntax:** `./run.sh CONFIG RUNS [VAR=value ...]`.
- **Config:** `configs/CONFIG.sh`. Overrides are saved with the results.
- **LR:** learning rate. Default `amd` value is `0.0004`.
- **ACL:** number of layers recomputed to save memory. Lower values can cause OOM.
- Source is copied into the results dir when the job starts so don't edit inputs used by queued jobs.

### Patches

[mlcommons/training_scc_2026](https://github.com/mlcommons/training_scc_2026)

`PATCHES` takes patch names from `patches/` without `.patch` and the job applies them to its copy of `implementation/`.

```bash
./run.sh amd 1 PATCHES=0001-fp8-activation-store-config-key ACL=0
```

- **0001:** corrects `FP8_ACTIVATION=True`, which stores the SwiGLU input in FP8 and frees enough memory for `ACL=0`.
- **0002:** lets the warm-up run with more than one micro-batch, allowing for `CP=2`.

[to-do.md](to-do.md) lists configs to test before if code changes approved and time allows.

### Competition submission

Run ten times with distinct seeds:
```bash
./run.sh amd 10
```

- `configs/amd.sh` without overrides is the baseline: AMD's settings, LR 0.0004 and 21 recomputed layers.
- The launcher flushes the host page cache before each run.
- The job runs until ten runs have converged. A run that stalls before training is stopped after 10 minutes and a run still going after 50 minutes is stopped. Their logs are kept as `not_started_<n>.txt` and `failed_<n>.txt` and the next seed takes the slot. Three such attempts in a row stop the job.
- `SEEDS="1 2 3"` names the seeds in order.
- After ten runs, it packages the results as `Isambards/` and runs the organisers' checks from mlperf-logging 6.0.0-rc6: package checker (including the RCP checker), result summarizer and repository checker.
- Before certification, update `systems/MI300X_EPYC_9654.json` to describe the competition system, including the GPU power cap. The cap must stay the same for all ten runs.

### Results

Show the queue, timings, and validation losses:
```bash
scripts/results.sh
```

Under `results/<timestamp>_<config>_.../`:

- `log.txt`: full job log.
- `config.sh`: settings used.
- `result_0.txt`, etc.: MLPerf result logs.
- `times.txt`: launch and finish times.
- `source/`: source snapshot.
- `package_checker.log`, `repo_checker.log`: check results for a ten-run batch.
- `submission/Isambards/`: submission package, including `student_system.json`, benchmark scripts/source, ten result logs and `scaling.json`.
