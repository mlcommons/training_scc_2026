# source-or-patches

The SCC26 submission carries patches rather than a full source tree: the base is
NVIDIA's MLPerf v5.0 `llama2_70b_lora` implementation
(`benchmarks/llama2_70b_lora/implementations/tyche_ngpu8_ngc25.04_nemo` from
`mlcommons/training_results_v5.0`), used unmodified except for what is here.

`scripts/run_competition.sh` fills this directory at submission time by copying, from
the repository that produced the runs:

| file | what it is |
| --- | --- |
| `launch.sh` | sbatch + srun + `apptainer exec` launcher; applies the patches below |
| `run_final.sh` | one timed run: 14 nodes, 580 W cap, fans 30%, pre-cool, then `launch.sh` |
| `setup.sh` | clones NVIDIA v5.0 and applies the patches |
| `patch_nvidia_impl.py` | the patch set against NVIDIA's implementation |
| `cpu_offload_patch.py` | saved activations to Grace memory over NVLink-C2C |
| `fp8_transpose_patch.py` | FP8 transpose-cache fix at `TP=1` |
| `swiglu_patch.py` | fused SwiGLU saved-tensor tagging for the offload path |
| `eval_truncate.py` | evaluation batching |
| `manual_ckpt_load.py` | bf16 checkpoint restore into FP8 parameters |
| `ib_ifname.py` | pins NCCL to the one HCA with a link (`mlx5_0`) |
| `gpumon.sh` | per-run GPU power/clock/temperature sampling |
| `config_GH200_8x1x1xtp1pp1cp1.sh` | base config |
| `config_GH200_8x1x1xtp1pp1cp1_offload.sh` | the config the runs load |
| `config_common.sh`, … | NVIDIA's own configs, as loaded |

If you are reading this in a submission directory and the files are not beside it, the
copy step did not run — see `scripts/run_competition.sh`.
