# DeepGEMM for the FP8 dgrad GEMMs + on-demand FP8 weight transposes (NTU SCC)

Sends Transformer Engine's FP8 **dgrad** GEMMs (`dx = dy @ W`) to DeepGEMM's `fp8_gemm_nt`. Every other GEMM stays on cuBLAS:
fprop (cuBLAS is faster there), eval, and anything that does not look like a plain dgrad.

The same wrapper can also free the cached FP8 weight transposes (`FP8W_T_ONDEMAND=1`, below). That frees enough memory to train
TP=2 without activation recompute.

**These two features are the only training-code changes in the NTU SCC setup, and they need NO change to the
benchmark source:** the default route loads them as a site-packages add-on (below). The stock `train.py` runs unchanged.


## No benchmark-source change: the site-packages add-on (`DG_DGRAD_ADDON=1`, default since 2026-10-07)

| file | role |
|---|---|
| `zz_dg_dgrad.pth` | one line, `import dg_dgrad_autoload`. Python's `site` module executes it at interpreter start |
| `dg_dgrad_autoload.py` | inert unless `DG_DGRAD=1`; then a meta-path hook waits for `transformer_engine.pytorch` to finish importing and calls `dg_dgrad.apply()` (plus the exit stats line) |

- **What the launcher does with `DG_DGRAD_ADDON=1`:**
  - mounts both files into the container's `/usr/local/lib/python3.12/dist-packages/`;
  - mounts `patches/` and `gemmlib/`;
  - does **not** mount `train_dgdgrad.py`.
- **Result:** the benchmark tree (`train.py`, `utils.py`, configs) is byte-identical to upstream. The only differences are the
  container environment (two extra files in site-packages) and env vars (`DG_DGRAD=1`, `FP8W_T_ONDEMAND=1`).
- **Numerics:** the same `dg_dgrad.apply()` runs on the same TE modules, so they are identical to the
  `train_dgdgrad.py` route.
- **Timing:** apply now runs as soon as TE is imported instead of after train.py's imports. apply re-binds module globals, so
  the timing does not matter.
- The old route (`DG_DGRAD_PATCH=1`, mounting `train_dgdgrad.py` over `train.py`) still works. The two are mutually exclusive.

## Files
| file | role |
|---|---|
| `dg_dgrad.py` | Runtime wrapper (DeepGEMM dgrad + optional on-demand weight transposes) around TE's `general_gemm` in `ops.basic.basic_linear`, `module.linear`, `module.layernorm_linear` and `module.layernorm_mlp`. It also holds a fused Triton kernel that re-quantizes `dy` from TE's E5M2 per-tensor scaling to E4M3 with 1x128 per-token-group scales (the DeepSeek-V3 recipe). `W^T` is TE's own E4M3 transpose plus a broadcast per-tensor block-scale grid, so it is not re-quantized. |
| `train_dgdgrad.py` | The stock `train.py` plus 8 lines: if `DG_DGRAD=1`, import `dg_dgrad` and call `apply()`, and print call counts at exit. |
| `build_deepgemm.sh` | Builds DeepGEMM 2.8.1 (commit `057ca59`) into a private `--target` dir inside the training container. It also builds a private libdw/libelf sysroot, because the image lacks them. |

Only calls that look exactly like a plain dgrad are rerouted: layout NN, `grad=True`, no accumulate/alpha/beta/bias/userbuffers,
no quantized output, bf16 out, FP8 storages. Everything else falls through to the original `general_gemm`.

## How it is wired (paths are this cluster's)
The launcher (`/work/benchmark/MLPerf/tools/run_nemo_l2l.sh`, flag `DG_DGRAD_PATCH=1`) does three things:
- mounts `train_dgdgrad.py` over `/workspace/ft-llm/train.py`;
- mounts `/work/benchmark/MLPerf/gemmlib` (`pylib_dg`, `sysroot`, `dg_cache`) and `/work/benchmark/MLPerf/patches` (`dg_dgrad.py`);
- sets `TRITON_CACHE_DIR`.

The runtime switch is `DG_DGRAD=1`. `apply()` preloads libdw/libelf from the sysroot, puts `pylib_dg` on `sys.path`, and sets
`DG_JIT_CACHE_DIR`. It cannot be combined with other patches that replace `train.py`.

## Results (16x H200 NVL, TP=2, block recompute 16, `USE_TE_OPS=1`)
| | value |
|---|---|
| dgrad GEMMs, sustained at the power cap | 300 W: 483-494 vs cuBLAS split-accumulator 431-460 TFLOPS (+7-15%). 400 W: 703-720 vs 639-680 (+6-13%) |
| hybrid layer GEMM time (cuBLAS fprop + DeepGEMM dgrad) | −5.3% (400 W) |
| training step | 4.015 vs 4.109 s at 300 W = **−2.3%** (2 runs per arm) |
| convergence, 2 runs per arm, 300 W | DeepGEMM 3456 / 3072 samples vs control 3072 / 3456. Mean per-eval delta +0.0012; same-arm noise ≤ 0.006 |
| numerics | dx rel. error 0.064 vs TE's 0.059, from the double rounding E5M2 → E4M3. Accumulation stays precise (promotion every 128 K). TE fast-accum dgrad, the alternative, broke convergence. |

Same-seed runs are NOT bit-identical (first loss differs by ~0.012), so convergence can only be judged over several runs per arm.

## On-demand FP8 weight transposes: training without activation recompute (`FP8W_T_ONDEMAND=1`)

**What occupies the memory.**
- The stock `utils.py` builds the model under `te.fp8_model_init`, so the frozen base weights are FP8-primary (no bf16 copy).
- The 64.3 GiB of weights per GPU at TP=2 is FP8 data (32 GiB) plus TE's cached FP8 **transpose** (32 GiB).
- Hopper FP8 GEMMs are TN-only, so dgrad uses `W^T`.

**What the flag does.**
- The wrapper drops a frozen FP8 weight's transpose (`update_usage(columnwise_usage=False)`) after each fprop (TN) and dgrad (NN)
  GEMM that uses it.
- TE's backward already calls `update_usage(columnwise_usage=True)` before every dgrad, which rebuilds it with `tex.fp8_transpose`,
  so at most one layer's transposes are alive.
- This frees ~32 GiB/GPU, enough to turn recompute off:
  `RECOMPUTE_GRANULARITY=null RECOMPUTE_METHOD=null RECOMPUTE_NUM_LAYERS=null` (the stock config sets block/16, so override after
  sourcing it).

Results (400 W, DeepGEMM dgrad on):

| | value |
|---|---|
| 1 node, 8 GPUs TP=2 MBS1, 70 steps | recompute 16: 2.635 s; transposes on demand + recompute 16: 2.666 s (+1.2%); **transposes on demand + no recompute: 2.447 s (−7.1%)** |
| 16 GPUs (2 nodes), full run, seed 1234 | step 2.683 → **2.512 s (−6.4%)**. Converged at 3072 samples. **TTT 19.15 → 18.06 min** |
| peak memory without recompute | 128.43 GiB allocated, ~137 GiB reserved of 139.8 GiB, identical on all ranks. No OOM, including the 5 eval passes |

The headroom is only ~3 GiB, so anything that adds memory per GPU will need some recompute back.
