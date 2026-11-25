#!/usr/bin/env python3
"""Adapt NVIDIA's MLPerf v5.0 llama2_70b_lora implementation to one GPU per node.

    nvidia/patch_nvidia_impl.py <path-to-tyche_ngpu8_ngc25.04_nemo copy>

The source is mlcommons/training_results_v5.0
NVIDIA/benchmarks/llama2_70b_lora/implementations/tyche_ngpu8_ngc25.04_nemo --
the code Krai's v6.0 GH200 submission is based on. That submission ran TP=4
inside nodes of four NVLinked GH200s. This cluster has one GH200 per node, so
TP has to be 1 (any TP group would cross InfiniBand on every layer), and at TP=1
a 70B only fits in 96GB with the three memory settings measured on this cluster
earlier (see the root README). NVIDIA's code hardcodes one of them to the wrong
value and does not expose the other two.

Each patch is exposed as an environment variable so the stock behaviour stays
reachable, and each is idempotent.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
train = root / "train.py"
t = train.read_text()
changed = []

# 1. disable_parameter_transpose_cache is hardcoded False. Measured on this
#    cluster: False makes TE cache a second, column-major FP8 copy of every
#    weight the first time each layer runs -- +1 byte per parameter, +63.7 GiB
#    for a 70B, OOM at about layer 35 of 80 on a 96GB GPU. At TP=4 that is 16 GiB
#    per GPU and fits, which is why NVIDIA and Krai never needed it.
old = "        disable_parameter_transpose_cache=False,\n"
new = ('        disable_parameter_transpose_cache=bool(int(os.environ.get("DISABLE_TRANSPOSE_CACHE", "1"))),'
       "  # local patch 1\n")
if old in t:
    t = t.replace(old, new, 1)
    changed.append("1 disable_parameter_transpose_cache from env (default on)")

# 2. Activation recomputation is not exposed at all. Measured: without it,
#    activations at seq 8192, MBS 1, TP=1 need more than 28 GiB on top of the
#    64.7 GiB of FP8 weights. NeMo's Llama2Config70B accepts the mcore fields
#    directly. ACG=none leaves it off (stock behaviour).
old = "        cpu_offloading=cfg.model.cpu_offloading,\n"
new = ('        recompute_granularity=None if os.environ.get("ACG", "full") == "none" else os.environ.get("ACG", "full"),'
       "  # local patch 2\n"
       '        recompute_method=os.environ.get("ACM", "uniform"),\n'
       '        recompute_num_layers=int(os.environ.get("ACL", "1")),\n'
       "        cpu_offloading=cfg.model.cpu_offloading,\n")
if "local patch 2" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("2 recompute_granularity/method/num_layers from ACG/ACM/ACL (default full/uniform/1)")

# 3. Megatron's CheckpointFunction is an autograd.Function, so under full
#    recompute it only gets a grad_fn if one of its *inputs* requires grad. LoRA
#    freezes the embedding, so the hidden states entering layer 0 do not, and
#    backward fails with "element 0 of tensors does not require grad". Measured
#    on this stack's predecessor; forcing the embedding output to require grad
#    is the fix (HF's enable_input_require_grads). Applied after the PEFT
#    transform, via a Lightning hook on the model. Also needed without
#    recompute when CPU offloading is on: TE's offload handler counts layer
#    groups with an autograd Function after every layer, and the one after
#    layer 0 only gets a backward if layer 0's input requires grad; otherwise
#    the group counter drifts by one per step. Always on, since it costs one
#    unused dgrad GEMM; EMBEDDING_REQUIRES_GRAD=0 disables.
old = "    trainer.callbacks.append(custom_callback)\n"
new = '''    trainer.callbacks.append(custom_callback)
    if os.environ.get("EMBEDDING_REQUIRES_GRAD", "1") == "1":  # local patch 3
        import lightning.pytorch as _pl

        class _EmbeddingRequiresGrad(_pl.Callback):
            def on_fit_start(self, trainer, pl_module):
                emb = pl_module.module
                while not hasattr(emb, "embedding") and hasattr(emb, "module"):
                    emb = emb.module
                emb.embedding.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

        trainer.callbacks.append(_EmbeddingRequiresGrad())
'''
if "local patch 3" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("3 embedding output requires_grad under recompute")

# 4. TE 2.2 keeps a column-major FP8 copy of every weight from the moment it is
#    created (see nvidia/fp8_transpose_patch.py). At TP=1 on 96GB the model OOMs
#    at 94.7 GiB inside configure_model. Free transposes during init, and bound
#    them to about one layer during training. FREE_FP8_TRANSPOSE=0 disables.
old = "torch.cuda.set_device(int(os.getenv(\"SLURM_LOCALID\", \"0\")))\n"
new = old + (
    "if os.environ.get(\"FREE_FP8_TRANSPOSE\", \"1\") == \"1\":  # local patch 4\n"
    "    import fp8_transpose_patch\n"
    "    fp8_transpose_patch.install_init_patch()\n"
)
if "local patch 4" not in t and old in t:
    t = t.replace(old, new, 1)
    # 8. Activation offload to host memory in place of recompute
    #    (nvidia/cpu_offload_patch.py). TE's handler class is replaced before
    #    the model is built; CPU_OFFLOADING itself is read by NVIDIA's config.
    t = t.replace(new, new + (
        "if os.environ.get(\"CPU_OFFLOADING\", \"False\").lower() in (\"true\", \"1\"):  # local patch 8\n"
        "    import cpu_offload_patch\n"
        "    cpu_offload_patch.install()\n"
    ), 1)
    old2 = "    trainer.callbacks.append(custom_callback)\n"
    new2 = old2 + (
        "    if os.environ.get(\"FREE_FP8_TRANSPOSE\", \"1\") == \"1\":  # local patch 4b\n"
        "        import lightning.pytorch as _pl4\n"
        "\n"
        "        class _TransposeHooks(_pl4.Callback):\n"
        "            def on_fit_start(self, trainer, pl_module):\n"
        "                n = fp8_transpose_patch.install_layer_hooks(pl_module)\n"
        "                print(f\"fp8_transpose_patch: hooks on {n} decoder layers\", flush=True)\n"
        "\n"
        "        trainer.callbacks.append(_TransposeHooks())\n"
    )
    t = t.replace(old2, new2, 1)
    old3 = "    trainer.callbacks.append(custom_callback)\n"
    new3 = old3 + (
        "    if os.environ.get(\"CPU_OFFLOADING\", \"False\").lower() in (\"true\", \"1\"):  # local patch 8b\n"
        "        import lightning.pytorch as _pl8\n"
        "\n"
        "        class _OffloadHooks(_pl8.Callback):\n"
        "            def on_fit_start(self, trainer, pl_module):\n"
        "                cpu_offload_patch.install_layer_hooks(pl_module)\n"
        "\n"
        "        trainer.callbacks.append(_OffloadHooks())\n"
    )
    t = t.replace(old3, new3, 1)
    changed.append("4 free FP8 weight transposes at init and between layers")
    changed.append("8 activation offload handler when CPU_OFFLOADING=True")

# 5. NeMo's checkpoint restore dequantises every FP8 weight to bf16 on the GPU
#    and holds all of them at once (see nvidia/manual_ckpt_load.py): OOM at
#    94.7 GiB in selective_restore at TP=1. Load manually instead, in
#    on_train_start, which runs after the LoRA transform and before the MLPerf
#    run_start. MANUAL_CKPT_LOAD=0 restores NeMo's path.
old = "    OmegaConf.resolve(cfg)\n"
new = old + (
    "    manual_ckpt = bool(cfg.load_ckpt) and os.environ.get(\"MANUAL_CKPT_LOAD\", \"1\") == \"1\"  # local patch 5\n"
    "    if manual_ckpt:\n"
    "        cfg.load_ckpt = False\n"
)
if "local patch 5" not in t and old in t:
    t = t.replace(old, new, 1)
    old2 = "    trainer.callbacks.append(custom_callback)\n"
    new2 = old2 + (
        "    if manual_ckpt:  # local patch 5b\n"
        "        import lightning.pytorch as _pl5\n"
        "        import manual_ckpt_load\n"
        "\n"
        "        class _ManualCkptLoad(_pl5.Callback):\n"
        "            def on_train_start(self, trainer, pl_module):\n"
        "                manual_ckpt_load.load(pl_module, f\"{cfg.ckpt_root}/weights\", cfg.model.num_layers)\n"
        "\n"
        "        # First, so it runs before custom_callback's warmup and run_start.\n"
        "        trainer.callbacks.insert(0, _ManualCkptLoad())\n"
    )
    t = t.replace(old2, new2, 1)
    changed.append("5 manual checkpoint load in on_train_start")

# 6. Lightning's strategy teardown calls `lightning_module.cpu()`, which
#    dequantises every Float8Tensor to bf16 on the GPU on its way off it. At
#    TP=1 that OOMs after the run has finished (jobs 165/166): the MLPerf log is
#    complete but the job exits non-zero. Nothing is used after training, so
#    make that move a no-op.
old = "    trainer.callbacks.append(custom_callback)\n"
new = old + (
    "    import lightning.pytorch as _pl6  # local patch 6\n"
    "\n"
    "    class _NoCpuTeardown(_pl6.Callback):\n"
    "        def on_train_end(self, trainer, pl_module):\n"
    "            pl_module.cpu = lambda *a, **k: pl_module\n"
    "\n"
    "    trainer.callbacks.append(_NoCpuTeardown())\n"
)
if "local patch 6" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("6 skip the teardown move to CPU")

# 7. Keys NVIDIA's code does not emit. `cache_clear` (6.0.0 common.yaml) comes
#    from NVIDIA's run.sub, which this cluster does not use; launch.sh tries the
#    drop and passes the outcome in CACHE_CLEARED. The nine run-config keys are
#    required by 6.1.0 only. Values: FP8 GEMMs, FP8 attention when FP8_DPA=1,
#    LoRA grads all-reduced in bf16 (grad_reduce_in_fp32=False).
old = ("        mllogger.mlperf_submission_log(\n"
       "            benchmark=\"llama2_70b_lora\", num_nodes=cfg.trainer.num_nodes\n"
       "        )\n")
new = old + (
    "        # local patch 7\n"
    "        mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=os.environ.get(\"CACHE_CLEARED\", \"0\") == \"1\")\n"
    "        _fp8 = bool(cfg.model.fp8)\n"
    "        for _k, _v in [\n"
    "            (\"lowest_numerical_precision_in_linear\", \"fp8\" if _fp8 else \"bfloat16\"),\n"
    "            (\"lowest_numerical_precision_in_attn\", \"fp8\" if _fp8 and bool(cfg.model.fp8_dot_product_attention) else \"bfloat16\"),\n"
    "            (\"lowest_numerical_precision_in_comm\", \"bfloat16\"),\n"
    "            (\"tensor_parallelism\", int(cfg.model.tensor_model_parallel_size)),\n"
    "            (\"pipeline_parallelism\", int(cfg.model.pipeline_model_parallel_size)),\n"
    "            (\"context_parallelism\", int(cfg.model.context_parallel_size)),\n"
    "            (\"expert_parallelism\", 1),\n"
    "            (\"micro_batch_size\", int(cfg.model.micro_batch_size)),\n"
    "            (\"config_filename\", os.environ.get(\"CONFIG_FILENAME\", \"unknown\")),\n"
    "        ]:\n"
    "            mllogger.event(key=_k, value=_v)\n"
)
if "local patch 7" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("7 cache_clear and the 6.1.0 run-config keys")

# 9. TORCH_PROFILE_STEP=N profiles one training step with torch.profiler
#    (nvidia/torch_profile.py) and writes the kernel breakdown to /results.
old = "    trainer.callbacks.append(custom_callback)\n"
new = old + (
    "    if os.environ.get(\"TORCH_PROFILE_STEP\"):  # local patch 9\n"
    "        import torch_profile\n"
    "        trainer.callbacks.append(torch_profile.make_callback(int(os.environ[\"TORCH_PROFILE_STEP\"])))\n"
)
if "local patch 9" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("9 one-step torch.profiler via TORCH_PROFILE_STEP")

# 10. EVAL_TRUNCATE=1 evaluates validation samples only up to their last loss
#     token (nvidia/eval_truncate.py). Installed at import, before the data
#     module builds its datasets.
old = "torch.cuda.set_device(int(os.getenv(\"SLURM_LOCALID\", \"0\")))\n"
new = old + (
    "if os.environ.get(\"EVAL_TRUNCATE\", \"0\") == \"1\":  # local patch 10\n"
    "    import eval_truncate\n"
    "    eval_truncate.install()\n"
)
if "local patch 10" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("10 validation samples cut after the last loss token (EVAL_TRUNCATE)")

# 11. Fuse the FP8 upcast into the SwiGLU backward (nvidia/swiglu_patch.py).
old = "torch.cuda.set_device(int(os.getenv(\"SLURM_LOCALID\", \"0\")))\n"
new = old + (
    "if os.environ.get(\"SWIGLU_FP8_FUSED\", \"1\") == \"1\":  # local patch 11\n"
    "    import swiglu_patch\n"
    "    swiglu_patch.install()\n"
)
if "local patch 11" not in t and old in t:
    t = t.replace(old, new, 1)
    changed.append("11 FP8 upcast fused into the SwiGLU backward")

train.write_text(t)
import shutil
shutil.copy(Path(__file__).with_name("swiglu_patch.py"), root / "swiglu_patch.py")
shutil.copy(Path(__file__).with_name("gpumon.sh"), root / "gpumon.sh")
shutil.copy(Path(__file__).with_name("eval_truncate.py"), root / "eval_truncate.py")
shutil.copy(Path(__file__).with_name("torch_profile.py"), root / "torch_profile.py")
shutil.copy(Path(__file__).with_name("manual_ckpt_load.py"), root / "manual_ckpt_load.py")
shutil.copy(Path(__file__).with_name("fp8_transpose_patch.py"), root / "fp8_transpose_patch.py")
shutil.copy(Path(__file__).with_name("ib_ifname.py"), root / "ib_ifname.py")
shutil.copy(Path(__file__).with_name("cpu_offload_patch.py"), root / "cpu_offload_patch.py")
for cfg in Path(__file__).parent.glob("config_GH200_*.sh"):
    shutil.copy(cfg, root / cfg.name)
print("\n".join(changed) if changed else "already patched")
