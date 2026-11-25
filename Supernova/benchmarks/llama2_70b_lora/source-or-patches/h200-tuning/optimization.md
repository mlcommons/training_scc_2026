# Optimizations for llama2_70b_lora on H200 NVL

All measurements in this document come from 8 H200 NVL GPUs unless the text says otherwise. The peak of the H200 NVL is 1670.5 TFLOPS for dense FP8, and the kernels are built for sm90a with nvcc from CUDA 13.4 release V13.4.92. TFLOPS is 2 x M x N x K divided by the time, and the percent of peak uses the 1670.5 TFLOPS above.

This document describes each change in the patch set: what the change does, which setting controls it, and what it gives. The [patch index](../patches/README.md) tells how to apply the patches and build the software.

The benchmark trains LoRA adapters on Llama 2 70B. The production recipe uses tensor parallelism 1, context parallelism 2 and data parallelism 8 across 16 GPUs (8 per node). Each GPU processes one sequence per microbatch with one microbatch per GPU, so the global batch is 8 sequences. A 12-GPU variant (TP1 x CP2 x DP6, global batch 12) runs the same recipe at a larger batch; it needs 13 % more samples to converge (reference 3605 against 3179) and measured 13.5 % higher time to train at the reference convergence point, so the 16-GPU GBS 8 configuration is the default.

Layouts with tensor parallelism, such as TP2 x CP2 and TP4 x CP1, remain supported. The sections marked **Only for TP 2 or more**, **Only for CP 2 or more** or **Runs under the userbuffer overlap** describe these layouts. They are off in this recipe, and their measurements come from TP2 x CP2 unless the text says otherwise.

Each section has the same parts.

- **Setting** gives the environment variable that controls the change and its default value in the code.
- **Result** gives the measured effect. Step times come from training runs with real weights, and a percentage is the change of the step time. The results of the fused kernels and of the memory changes come from 8 H200 SXM GPUs at a power limit of 700 W unless a section says otherwise.
- **Code** links to the patch hunks that make the change.

## Kernel measurements

The kernel tables give each added or changed kernel next to the original kernel that it replaces. Each kernel is measured with Nsight Compute on one H200 SXM at two power limits, 600 W and 400 W, where 400 W approximates the H200 NVL. A power limit applies to each GPU, so one GPU behaves like one GPU in a full node, and the kernel runs at the clock that the limit sets. A single short kernel does not draw enough power to reach the limit, so the program first runs the kernel continuously for a few seconds until the power and the clock are stable, and then marks one launch for the profiler. Nsight Compute runs with `--profile-from-start off` and `--launch-count 1`, so the setup and warmup launches are not profiled. It uses `--replay-mode application`, so that each pass repeats the warmup and measures at the stable clock, and `--clock-control none`, because the default control locks the clock at a value that a capped kernel never reaches. Each measurement records the SM clock, which at 400 W agrees with the SM clock of a training run at 400 W. The context parallel exchange reads the memory of the peer GPUs, so its tables use the four GPUs of one context parallel group.

The columns give the kernel time and the throughput of the SMs, the tensor pipe, the L2 cache and the DRAM as a percent of the peak. The attention rows use the causal FLOP count. The capture has the source counters on, so each report also holds the per-instruction and per-source-line attribution.

## Run settings

The recipe uses these values. The last column gives the value that the code uses when the variable is unset.

| Variable | This recipe | Code default |
|---|---|---|
| TP, CP | 1, 2 | 1, 1 |
| MBS, MINIBS | 1, 2 | 1, 1 |
| NVTE_FAV3_FP8_BWD | 1 | 0 |
| NVTE_FAV3_FP8_FWD | 1 | 0 |
| NVTE_FP8_DPA_BWD | 1 | 1 |
| NVTE_CP_A2A_HEAD_CHUNKS | off | off |
| CP_A2A_COMM | ce | nccl |
| CP_SHARD_WEIGHTS | auto | unset, equal shards |
| DEEPGEMM | 1 | 0 |
| DEEPGEMM_NUM_SMS | 128, fc2 keeps the full count | 0, the full count |
| USE_TE_OPS | 1 | follows DEEPGEMM |
| FP8_ACT | 1 | 0 |
| SWIGLU_FUSION | 1 | 1 |
| TP_COMM_OVERLAP | False | False |
| UB_FC1_FPROP_NUM_SM | off | 8 at TP 2 |
| NVTE_FUSED_POST_RS_NORM | 1, no effect at TP1 | 0 |
| LORA_A2A | 1 | 1 |
| LORA_LOCAL | 1, no effect at TP1 | 0 |
| MCORE_CUDA_GRAPH, CUDA_GRAPH_SCOPE | 1, full_iteration | 0, full_iteration |
| LORA_EVAL_MERGE | 1 | 0 |
| VAL_MBS, VAL_GBS | 2, 8 | 1, computed from VAL_MBS |
| EVAL_PAD_MICROBATCH | 1 | 0 |
| FP8, LOAD_CKPT | True, True | True, False |
| FP8_TRANSIENT_TRANSPOSE | 1 | 0 |
| WT_RESIDENT_MLP, PYTORCH_CUDA_ALLOC_CONF | fc2:42 at load, then set for each rank by the uneven context parallel shards, expandable_segments:True | empty, unset |
| LOAD_GROUP_LAYERS | 4 | 20 |
| LOAD_STREAM, PARALLEL_LOAD, LOAD_THREADS | 1, True, half the CPU count of the node | 1, False, half the CPU count of the node |
| FP8_AMAX_HISTORY, FP8_AMAX_ALGO, FP8_REDUCE_AMAX | 4, max, True | 128, most_recent, False |
| GPU power limit | 450 W | board maximum |
| SM clock pin (LGC) | 1200 MHz | unset, driver-controlled |
| vboost | 2 | 0 |
| MAX_STEPS, WARMUP_STEPS, LR_DECAY_STEPS | 800, 32 (16 GPUs; 0 at 8 GPUs), 800 | 800, 0, 800 |
| LR | 5e-4 | 5e-4 |
| VAL_CHECK_INTERVAL, SKIP_EVALS | 384, 3 | 384, 0 |

## 1. Attention

### FP8 attention backward with FlashAttention 3

TransformerEngine computes the FP8 attention backward with cuDNN, and this change sends the backward to a FlashAttention 3 kernel for FP8 inputs. The patch adds an E4M3 backward kernel for head dimension 128 to FlashAttention 3, built from the kernel templates of the existing BF16 backward in `hopper/flash_bwd_kernel_sm90.h` and `hopper/flash_bwd_launch_template.h`, and its FP8 register-layout conversion follows `convert_layout_acc_Aregs_fp8` of NVIDIA recsys-examples HSTU (`corelib/hstu/hopper/utils.h`). The kernel reads Q, K, V and dO in E4M3, quantizes the probabilities P to E4M3 and the score gradients dS to E5M2 in registers, and accumulates in FP32. The dQ, dK and dV outputs are BF16, or E5M2 when the TE gradient quantizer asks for E5M2 with grouped query attention. The kernel writes the dS amax into the TE dP quantizer, writes the amax and inverse scale of E5M2 outputs, and also accepts an E4M3 output O with its scale. To our knowledge, no open source implementation of a FlashAttention 3 backward kernel in FP8 exists today.

An adapter module connects the TE FP8 attention backward to the kernel. It converts the E5M2 dO of TE to E4M3 and makes the transposed operands of the kernel in one Triton pass. It accepts delayed scaling, head dimension 128, a sequence length that is a multiple of 16, equal query and key lengths, a query head count that is a multiple of the key head count, vanilla softmax, zero dropout and bias free attention, and it stops with an error for other configurations.

The backward kernels of FlashAttention 3 use programmatic dependent launch, so each kernel starts while the previous kernel finishes. The preprocess kernel also sets the dK and dV accumulators to zero, which replaces two separate fill operations for grouped query attention. In the producer warpgroup, which has three warps of which only two carried load before, the third warp now loads dO, its transpose and dPsum, while the first keeps Q, its transpose, LSE, K and V. A blocked Q stage then cannot delay a ready dO slot, and the reverse. The consumer credits, the storage and the arithmetic are unchanged, the split needs no setting, and shapes outside the dense FP8 backward keep the combined ring.

**Setting.** NVTE_FAV3_FP8_BWD=1, default 0. The change also needs NVTE_FP8_DPA_BWD=1, which keeps the FP8 attention backward of TE and its quantizers, and the TE patch sets it when the variable is unset. FlashAttention 3 must be built with FLASHATTENTION_ENABLE_BWD_FP8=TRUE, which builds the E4M3 backward kernel. The Triton pass of the adapter uses programmatic dependent launch.

**Result.**

| Measurement | FlashAttention 3 | cuDNN | Change |
|---|---|---|---|
| Step time | 3.585 s | 3.65 s | 1.8% less |
| Relative error of dV | 0.054 | 0.19 | 3.52 times less |

The errors are against an FP32 reference. The lower error leads to faster convergence, so the run needs fewer training samples at the same hyperparameters as the [Fujitsu configuration](https://github.com/mlcommons/training_results_v6.0/blob/main/Fujitsu/benchmarks/llama2_70b_lora/implementations/nemo/configs/config_PGCDI_1x8x4xtp4pp1cp1.sh), and with these kernels and the learning rate of that configuration the run reaches the target at 2688 samples.

At a batch of 1, a sequence length of 8192, 16 query heads and 2 key and value heads, the dQ bulk reduce of the backward makes 108.5 of the 180 million L2 sectors. The FP8 backward takes 10.5% of the GPU time in a trace of 41 steady steps.

The next table uses the shape of one rank of the TP1 x CP4 layout, which is a batch of 2, a sequence length of 8192, 16 query heads and 2 key and value heads, with causal masking. The TFLOPS of the mainloop uses the causal FLOP count.

| Kernel | Implementation | Power limit | Time in us | TFLOPS | SM | Tensor pipe | L2 | DRAM | Speedup over cuDNN |
|---|---|---|---|---|---|---|---|---|---|
| FP8 backward kernel | cuDNN | 600 W | 2274.9 | 604 | 44.2 | 36.1 | 60.6 | 27.6 | |
| FP8 backward kernel | cuDNN | 400 W | 4095.6 | 336 | 44.6 | 36.4 | 37.6 | 15.3 | |
| Backward mainloop | FlashAttention 3 | 600 W | 1672.5 | 822 | 44.1 | 44.4 | 51.8 | 6.3 | 1.36 times |
| Backward mainloop | FlashAttention 3 | 400 W | 1901.4 | 723 | 45.5 | 45.5 | 45.4 | 5.3 | 2.15 times |

**Design principles.** The backward kernel is faster than cuDNN mainly because it moves fewer bytes and keeps the tensor cores busier, and the tables show both effects: the DRAM throughput is 6.3% against 27.6%, the L2 throughput is 51.8% against 60.6%, and the tensor pipe is 44.4% against 36.1%. Four design choices produce this.

- **FP8 end to end inside the kernel.** Q, K, V and dO stay in E4M3 in shared memory. The probabilities P and the score gradients dS are quantized in registers, to E4M3 and E5M2, and feed the next GEMM directly, so no BF16 tile is written out and read back. The accumulators stay in FP32, and the scales are device-side tensors that the kernel reads, which keeps the launch graph safe.
- **Operands in the layout that the FP8 GEMM needs.** Hopper FP8 GEMMs need K-major operands, so one Triton pass makes the transposed Q, K and dO before the kernel, and the kernel uses the swapped S and dP layout with a register-fed dK and dV GEMM, with V held in registers. The layout work happens once per call and never inside the main loop.
- **Less memory traffic for the same FLOPs.** The dQ partial sums reduce into an FP32 buffer with bulk reductions that run beside the math. The tile scheduler groups the heads in L2 sized sections, so that the blocks which run together work on data that fits in the L2 cache. For the causal FP8 case, a pair of CTAs in a two CTA cluster exchanges the dS tile through distributed shared memory with bulk copies and mbarriers.
- **Keeping every unit fed.** The producer warps are split so that a blocked Q stage cannot delay a ready dO slot. The register budget gives each consumer warpgroup what it needs and the producer only a minimum, which lets four consumer warpgroups run in one CTA. The preprocess, main and postprocess kernels chain with programmatic dependent launch, so each starts its setup while the previous kernel finishes.

The gain is larger at 400 W because a capped GPU spends its power budget on the clock: a kernel that moves fewer bytes and idles less holds a higher clock than one that does more memory work for the same FLOPs.

**Code.** [backward mainloop](../patches/04-flash-attention.patch#L2153), [E4M3 kernel instance](../patches/04-flash-attention.patch#L2114), [Python entry](../patches/04-flash-attention.patch#L833), [launch chain](../patches/04-flash-attention.patch#L1172), [adapter](../patches/01-llama2_70b_lora-nemo.patch#L2865), [TE hook](../patches/03-transformer-engine/02-te-local-mods.patch#L522), [build option](../patches/04-flash-attention.patch#L3604)

### FP8 attention forward with FlashAttention 3

This change sends the TE FP8 attention forward to the FlashAttention 3 FP8 kernel. The FP8 forward mainloop is already open source in upstream FlashAttention 3, so unlike the backward, whose FP8 kernel is new, this change adds only the E4M3 output path described below and no kernel of its own. TE calls the kernel at two locations, the standard attention path of TE and the context parallel all to all path, and in the head chunk pipeline the kernel also quantizes the output O to E4M3 for the next exchange.

The output quantization happens in the forward epilogue of the kernel, which applies the scale to each FP32 value, writes E4M3 and updates the amax. This path needs a sequence length that is a multiple of 128.

**Setting.** NVTE_FAV3_FP8_FWD=1, default 0. TE uses the kernel for delayed scaling, head dimension 128, a query head count that is a multiple of the key head count, vanilla softmax, zero dropout and bias free attention. Other recipes and head layouts use cuDNN, and the adapter also needs equal query and key lengths that are multiples of 16.

The forward writes V in a token permuted column major layout, which a Triton kernel prepares once per forward call. This removes the transpose that the producer warpgroup repeated inside the kernel. The layout is an activation buffer of 4 MiB at the full shape and 2 MiB with one key and value head, so no weight is duplicated, and the backward keeps the original V. Shapes that the column major kernel does not support keep the row major layout.

**Result.** The main reason for this change is accuracy.

| Measurement | FlashAttention 3 | cuDNN | Change |
|---|---|---|---|
| Relative error of the output | 0.0229 | 0.0399 | 1.74 times less |
| Samples to target, olympic mean of 10 seeds | 2976 | 3168 | 6.1% less |

The error is against an FP32 reference at the shape of the benchmark.

| Kernel | Implementation | Power limit | Time in us | TFLOPS | SM | Tensor pipe | L2 | DRAM | Speedup over cuDNN |
|---|---|---|---|---|---|---|---|---|---|
| FP8 forward | FlashAttention 3 | 600 W | 595.6 | 923 | 52.5 | 51.9 | 30.5 | 3.0 | 1.01 times |
| FP8 forward | FlashAttention 3 | 400 W | 683.0 | 805 | 52.5 | 52.0 | 27.0 | 2.6 | 1.00 times |
| FP8 forward | cuDNN | 600 W | 599.1 | 917 | 54.2 | 51.4 | 25.8 | 2.1 | |
| FP8 forward | cuDNN | 400 W | 683.6 | 804 | 54.3 | 51.5 | 23.0 | 1.8 | |

**Code.** [forward epilogue](../patches/04-flash-attention.patch#L84), [adapter](../patches/01-llama2_70b_lora-nemo.patch#L2987), [TE hook](../patches/03-transformer-engine/02-te-local-mods.patch#L544), [context parallel hook](../patches/03-transformer-engine/02-te-local-mods.patch#L678)

### Fused RoPE and FP8 quantize for the context parallel exchange

Before the context parallel all to all exchange, the attention input goes through a chain of separate operations: a copy takes Q out of the fused QKV projection output, two rotary position embedding kernels rotate Q and K in BF16, a copy makes V contiguous, a concatenation joins Q, K and V, and a TE cast quantizes the result to E4M3 for the FP8 attention. The backward runs the inverse chain with a concatenation, a TE dequantize, two rotary backward kernels and a concatenation of the gradients.

This change replaces each chain with one Triton kernel. The forward kernel reads the fused QKV output once, applies the rotation in FP32 in the same operation order as the TE kernel, rounds to BF16, writes the amax into the TE quantizer of the attention and writes the E4M3 Q, K and V in the layout of the exchange, so the attention receives FP8 tensors and TE skips its own concatenation and cast. The backward kernel reads the E5M2 gradients from the exchange, dequantizes them, applies the inverse rotation and writes the BF16 gradient of the fused QKV output. A table of sine and cosine values, built once for each frequency tensor and used by all layers and steps, replaces the evaluation per element. Both kernels write each head row of 128 values with one contiguous store.

The kernels give bitwise identical results to the replaced operations in a unit test at the production shapes, in eager mode and in full iteration CUDA graphs.

**Setting.** The kernels are always on in training. They need the context parallel all to all exchange with the sbhd layout, FP8 delayed scaling with FP8 attention, BF16 activations, head dimension 128 and a head count that is a power of two. The evaluation uses the same kernels. Other configurations use the stock operations.

**Result.**

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 2.8245 s | 2.7829 s | 1.5% less |
| Energy per GPU per step | 1.944 kJ | 1.918 kJ | 1.3% less |

The next table gives the kernels per layer and microbatch at one rank of the TP1 x CP4 recipe, on one H200 SXM GPU with a 400 W power limit and CUDA graph replay, as the mean of two runs. The stock backward includes the E5M2 cast of the qkv output gradient, because the fused backward also writes that cast. The outputs of the two paths are bitwise identical.

| Kernels | Stock | Fused | Change |
|---|---|---|---|
| Forward | 249.7 us | 43.9 us | 5.7 times less |
| Backward with the E5M2 cast | 235.7 us | 63.5 us | 3.7 times less |

The next tables give the forward kernels measured with Nsight Compute, first at the training shape of one rank, a mixed qkv tensor of 2048 by 2 by 8 by 1280 values, and then at the evaluation microbatch of 4. The SM, L2 and DRAM columns give the throughput as a percent of the peak.

| Kernel | Implementation | Power limit | Time in us | SM | L2 | DRAM | Speedup over the stock kernels |
|---|---|---|---|---|---|---|---|
| RoPE and FP8 quantize, training | fused kernel | 600 W | 41.8 | 36.2 | 73.8 | 57.5 | 5.54 times |
| split copy of q | stock | 600 W | 77.6 | 55.5 | 44.7 | 31.2 | |
| TE RoPE of q | stock | 600 W | 59.9 | 58.9 | 57.7 | 41.5 | |
| TE RoPE of k | stock | 600 W | 12.0 | 50.1 | 38.1 | 16.6 | |
| copy of v | stock | 600 W | 11.9 | 44.7 | 35.6 | 14.8 | |
| concatenation | stock | 600 W | 39.5 | 7.6 | 88.7 | 76.1 | |
| FP8 cast | stock | 600 W | 30.7 | 52.7 | 86.1 | 76.2 | |
| all stock kernels | stock | 600 W | 231.6 | | | | |
| RoPE and FP8 quantize, training | fused kernel | 400 W | 41.8 | 36.3 | 74.1 | 57.5 | 5.56 times |
| split copy of q | stock | 400 W | 77.7 | 55.4 | 44.6 | 31.1 | |
| TE RoPE of q | stock | 400 W | 61.5 | 59.8 | 56.3 | 40.5 | |
| TE RoPE of k | stock | 400 W | 11.7 | 49.6 | 38.4 | 16.9 | |
| copy of v | stock | 400 W | 12.0 | 45.0 | 35.7 | 14.7 | |
| concatenation | stock | 400 W | 39.3 | 7.8 | 88.5 | 76.5 | |
| FP8 cast | stock | 400 W | 30.2 | 52.9 | 87.8 | 77.2 | |
| all stock kernels | stock | 400 W | 232.4 | | | | |

| Kernel | Implementation | Power limit | Time in us | SM | L2 | DRAM | Speedup over the stock kernels |
|---|---|---|---|---|---|---|---|
| RoPE and FP8 quantize, evaluation | fused kernel | 600 W | 76.5 | 38.5 | 81.2 | 65.5 | 5.87 times |
| split copy of q | stock | 600 W | 152.9 | 56.1 | 46.3 | 34.2 | |
| TE RoPE of q | stock | 600 W | 117.2 | 62.5 | 59.7 | 45.4 | |
| TE RoPE of k | stock | 600 W | 19.3 | 60.4 | 45.3 | 21.2 | |
| copy of v | stock | 600 W | 20.8 | 52.2 | 39.8 | 18.2 | |
| concatenation | stock | 600 W | 79.7 | 7.0 | 89.2 | 81.6 | |
| FP8 cast | stock | 600 W | 59.2 | 54.4 | 88.5 | 83.6 | |
| all stock kernels | stock | 600 W | 449.1 | | | | |
| RoPE and FP8 quantize, evaluation | fused kernel | 400 W | 78.0 | 39.3 | 79.7 | 64.3 | 5.77 times |
| split copy of q | stock | 400 W | 152.7 | 55.8 | 46.0 | 34.2 | |
| TE RoPE of q | stock | 400 W | 118.7 | 63.6 | 59.0 | 44.8 | |
| TE RoPE of k | stock | 400 W | 19.2 | 61.9 | 45.5 | 21.1 | |
| copy of v | stock | 400 W | 20.8 | 51.6 | 40.2 | 18.4 | |
| concatenation | stock | 400 W | 80.9 | 7.0 | 88.0 | 80.3 | |
| FP8 cast | stock | 400 W | 57.8 | 56.2 | 91.0 | 85.4 | |
| all stock kernels | stock | 400 W | 450.1 | | | | |

**Code.** [forward kernel](../patches/01-llama2_70b_lora-nemo.patch#L4568), [backward kernel](../patches/01-llama2_70b_lora-nemo.patch#L4655), [autograd function](../patches/01-llama2_70b_lora-nemo.patch#L4847), [install](../patches/01-llama2_70b_lora-nemo.patch#L4990), [train.py hook](../patches/01-llama2_70b_lora-nemo.patch#L8406)

### Fused RoPE writes the context parallel send layout

The context parallel exchange sends each rank a block of heads for all of its tokens. The send buffer therefore holds Q, K and V in the order [rank, sequence, batch, heads per rank, head dimension], while the attention input is in the order [sequence, batch, heads, head dimension]. In the stock path the exchange packs each of Q, K and V into its send buffer with a strided copy on the main stream, which takes about 20 us for Q and 4.4 us each for K and V per layer and microbatch.

The fused RoPE and FP8 quantize kernel writes Q, K and V directly into the send buffers in the send order. The forward first takes the three send buffers from the exchange in the order that the exchange uses them, so the buffer rotation and the synchronization with the other ranks stay unchanged, and TE receives each send buffer as a tensor of the attention input shape. Only the exchange reads these tensors, and it skips the packing copy for them.

The exchanged bytes, the amax and the scale are bitwise identical to the stock path in a unit test at the production shapes, in eager mode and in CUDA graphs.

**Setting.** Always on in training with the IPC exchange (CP_A2A_COMM=tma or ce) and head counts that divide by the context parallel size. Other configurations use a separate output buffer and the packing copy.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit, with a profile of three steps.

| Measurement | Before | After | Change |
|---|---|---|---|
| K and V packing copies per step | 320 | 0 | |
| Q packing copies per step | 160 | 0 | |
| Packing copy time per step | | | 2.24 ms less |
| Main stream time per step | | | 1.8 ms less |

**Code.** [store layout](../patches/01-llama2_70b_lora-nemo.patch#L4605), [send buffers in the forward](../patches/01-llama2_70b_lora-nemo.patch#L4776), [eligibility](../patches/01-llama2_70b_lora-nemo.patch#L4974), [send buffer claim](../patches/01-llama2_70b_lora-nemo.patch#L1012), [skipped packing copy](../patches/01-llama2_70b_lora-nemo.patch#L1279)

### Gradient of the attention output cast into the context parallel send layout

In the backward of the context parallel attention, TE receives the BF16 gradient of the attention output, casts it to E5M2 with its own kernel and hands the FP8 data to the exchange, which then packs it into its send buffer with a strided copy into the order [rank, sequence, batch, heads per rank, head dimension]. Each layer and microbatch therefore pays a cast and a copy of the full gradient on the main stream.

This change wraps the backward of the attention: it takes the send buffer from the exchange first, and one Triton kernel casts the gradient straight into that buffer in the send order. The cast follows the TE delayed scaling cast, with the BF16 value times the scale rounded to E5M2 with saturation, the largest absolute value added to the amax of the quantizer, and the inverse scale written for the tensor. The attention then receives an FP8 tensor, so TE skips its cast and the exchange skips its packing copy.

The exchanged bytes, the amax and the inverse scale are bitwise identical to the TE cast followed by the stock exchange in a unit test at the production shape, in eager mode and in CUDA graphs, including values that saturate.

**Setting.** Always on in training with the IPC exchange (CP_A2A_COMM=tma or ce), FP8 delayed scaling with the sbhd layout, and a head count that divides by the context parallel size. Other configurations use the stock cast and exchange.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit, with a profile of two steps.

| Measurement | Before calls | Before time | After calls | After time | Change |
|---|---|---|---|---|---|
| TE casts of the gradient per step | 160 | 4.43 ms | 0 | | |
| Packing copies of the gradient per step | 160 | 3.09 ms | 0 | | |
| Fused cast into the send buffer per step | | | 160 | 4.38 ms | |
| Kernel time per step | | | | | 3.1 ms less |

**Code.** [kernel](../patches/01-llama2_70b_lora-nemo.patch#L33), [send buffer and FP8 tensor](../patches/01-llama2_70b_lora-nemo.patch#L49), [eligibility](../patches/01-llama2_70b_lora-nemo.patch#L67), [backward wrapper](../patches/01-llama2_70b_lora-nemo.patch#L116), [train.py hook](../patches/01-llama2_70b_lora-nemo.patch#L8398)

### FP8 gradient of the qkv projection written by the RoPE backward

The fused RoPE backward writes the BF16 gradient of the fused QKV output, and the backward of the qkv projection then casts that gradient to E5M2 with its grad output quantizer for its FP8 dgrad GEMM, in a separate kernel over the full [tokens, QKV width] tensor.

This change makes the RoPE backward kernel also write the E5M2 data next to the BF16 gradient, with the scale of that quantizer, and add the largest absolute value to the amax of the quantizer and write the inverse scale as the cast does. The data waits in a store keyed by the quantizer, and the cast of the quantizer returns it for that tensor instead of casting again, while a cast of any other tensor with the same quantizer drops the stored entry and casts as before.

The BF16 gradient, the E5M2 data, the amax and the inverse scale are bitwise identical to the separate cast in a unit test at the production shape, in eager mode, in CUDA graphs and through the TE operations of the qkv layer over three iterations, including values that saturate.

**Setting.** Always on in training where the fused RoPE runs and the qkv layer is the fused LoRA layer with a delayed scaling E5M2 grad output quantizer.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit, with a profile of two steps.

| Measurement | Before calls | Before time | After calls | After time | Change |
|---|---|---|---|---|---|
| Casts of the qkv output gradient per step | 158 | 5.65 ms | 0 | | |

The next table gives the kernel times per layer and microbatch on one H200 SXM GPU with a 400 W power limit and CUDA graph replay, as the mean of two runs.

| Measurement | Time |
|---|---|
| E5M2 cast of the qkv output gradient, removed | 33.8 us |
| Time that the E5M2 write adds to the RoPE backward kernel | 18.1 us |
| Kernel time | 15.7 us less |

**Code.** [store helper](../patches/01-llama2_70b_lora-nemo.patch#L4654), [amax and inverse scale](../patches/01-llama2_70b_lora-nemo.patch#L4722), [payload buffers](../patches/01-llama2_70b_lora-nemo.patch#L4819), [hand-off to the store](../patches/01-llama2_70b_lora-nemo.patch#L4875), [quantizer lookup](../patches/01-llama2_70b_lora-nemo.patch#L4879), [keyed store](../patches/01-llama2_70b_lora-nemo.patch#L4250), [use in the cast](../patches/01-llama2_70b_lora-nemo.patch#L4258), [install](../patches/01-llama2_70b_lora-nemo.patch#L5030)

### FP8 attention output fed straight into the output projection

The attention quantizes its output O to E4M3, and the context parallel all to all exchange moves those FP8 bytes. In the stock path, TE then dequantizes the received O to BF16 and returns it, and the output projection quantizes the BF16 O to E4M3 again for its FP8 GEMM. Each layer and microbatch therefore pays one dequantize and one quantize of the full attention output, and the GEMM reads a value rounded twice.

This change asks the attention for an FP8 output. TE then returns the received payload as an FP8 tensor without running a kernel, and the TE ops linear layer of the output projection uses it as is, because it does not quantize an input that is already quantized. The LoRA branch of the projection reads the same payload, and the folded dropout and LoRA A GEMM dequantize it in their prologue. The scale and amax of the attention output quantizer are written as before, and the input quantizer of the projection is no longer used on this path.

The result is not bitwise identical, because the projection GEMM reads the attention output rounded once instead of twice. In a unit test at the production shape, the projection output matches an FP64 product of the FP8 operands within a normwise relative error of 2.6e-3, against 2.8e-3 for the stock path. The gradient into the attention output is bitwise identical. The LoRA weight gradients are as close to an FP64 reference as the stock ones.

**Setting.** The change is always on in training. It needs the fused RoPE and FP8 quantize path of the attention input, an E4M3 delayed scaling quantizer for the attention output, and an output projection that is the fused TE LoRA linear layer with its adapter on. Other configurations and the evaluation use the stock operations.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit. The kernel level evidence from an nsys profile comes from the combined run of the accepted changes. The kernel times are from a unit test on one GPU with CUDA graph replays at the production shape.

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 2.7724 s | 2.7611 s | 0.41% less |
| Energy per GPU per step | 1.864 kJ | 1.849 kJ | 0.8% less |
| Dequantize of the attention output | 31.9 us | none | removed |
| Quantize of the projection input | 29.7 us | none | removed |
| Attention output to projection and LoRA, forward | 641.2 us | 594.3 us | 7.3% less |
| Attention output to projection and LoRA, forward and backward | 1197.7 us | 1168.2 us | 2.5% less |

**Code.** [eligibility](../patches/01-llama2_70b_lora-nemo.patch#L3737), [output projection check](../patches/01-llama2_70b_lora-nemo.patch#L3676), [Megatron signature](../patches/01-llama2_70b_lora-nemo.patch#L3747), [FP8 output request](../patches/01-llama2_70b_lora-nemo.patch#L3761), [attention wrapper](../patches/01-llama2_70b_lora-nemo.patch#L3798), [install](../patches/01-llama2_70b_lora-nemo.patch#L3768), [train.py hook](../patches/01-llama2_70b_lora-nemo.patch#L8408)

### Head chunks for the context parallel exchange

**Only for CP 2 or more.**

With context parallelism, an all to all exchange moves the attention heads between the ranks before attention. A second exchange moves the output back after attention. This change divides the heads into groups and makes a pipeline.

- In the forward, the Q, K and V exchange of group 2 runs during the attention of group 1. The output exchange of group 1 runs during the attention of group 2.
- The backward uses the same pipeline for dO and for dQ, dK and dV.
- With 2 groups, each group holds 8 query heads and the one key and value head that they share.
- The exchanges use grouped NCCL send and receive operations into the final buffers. With the fused casts of the sections above, each received half of the load balanced sequence goes directly to its final position, and each send buffer is a contiguous slice of the output.
- The heads are independent, and the delayed scaling quantizers collect their amax over all groups. The loss of step 1 is bitwise identical to the single exchange.

**Setting.** NVTE_CP_A2A_HEAD_CHUNKS=2, default 1. The pipeline needs delayed scaling, NVTE_FAV3_FP8_FWD=1, NVTE_FAV3_FP8_BWD=1, NVTE_FP8_DPA_BWD=1, the sbhd layout, head dimension 128, vanilla softmax, zero dropout, bias free attention and head counts that divide by the context parallel size times the group count. Other configurations use the stock exchange.

**Result.**

| Change | Step time before | Step time after | Change |
|---|---|---|---|
| Head pipeline | 3.367 s | 3.349 s | 0.53% less |
| Send and receive exchange into the final buffers | 3.313 s | 3.296 s | 0.51% less |

**Code.** [head chunk pipeline](../patches/03-transformer-engine/05-te-cp-a2a-head-chunks.patch#L10)

## 2. Linear layers

### DeepGEMM FP8 GEMMs

The patch adds per tensor scaled FP8 GEMM kernels for sm90 to DeepGEMM. The kernels are derived from `sm90_fp8_gemm_1d2d.cuh` of DeepGEMM. They accumulate over all of K in the tensor core accumulator. This is the accumulation that TE requests for forward GEMMs.

The kernels serve two callers.

- A router replaces the TE GEMM function and sends the forward FP8 GEMMs of the dense layers to DeepGEMM. These are the GEMMs with E4M3 inputs, per tensor scales, BF16 output, and K and N that are multiples of 128, and all other GEMMs go to TE.
- A userbuffer chunk provider in TE computes the GEMM chunks of the tensor parallel overlap. It accepts the forward chunks, which use the accumulation that TE requests for forward GEMMs, with the math SM count of its userbuffer ring, and the backward chunks use cuBLAS.

A heuristic selects the kernel configuration for each shape. For M below 6144, a 128 row kernel can run. For larger M, a 192 row kernel with three warp groups can run.

The patch also adds E5M2 inputs, an E4M3 output with scale and amax, and an NN layout entry to the per tensor GEMM, and the GEMM accepts any positive SM count.

**Setting.** DEEPGEMM=1, default 0, and the chunk provider of TE reads the same variable. DEEPGEMM_NUM_SMS sets the persistent grid of the kernel and is 0, the full device count, by default. The recipe sets it to 128 and keeps the fc2 forward at the full count. DEEPGEMM=1 needs the TE ops fuser, which the configuration turns on when DEEPGEMM=1 and USE_TE_OPS is unset, and an explicit USE_TE_OPS=0 stops the run with an error. The TE build needs NVTE_UB_DEEPGEMM_ROOT to compile the chunk provider, as the patch index tells.

The persistent grid holds one block for each SM, and the kernel gives each block 128 x 256 tiles in turn. A tile count that does not divide the grid leaves the last wave partly full. At 132 SMs the qkv, proj and fc1 forward have 1280, 1024 and 7168 tiles, so their last waves hold 92, 100 and 40 of 132 blocks. At 128 SMs every one of those counts divides exactly, with 10, 8 and 56 waves. The fc2 forward keeps 132 because its 1024 tiles run over a reduction length of 28672, and the smaller grid then costs more than the tail wave saves.

**Result.**

| Measurement | DeepGEMM | cuBLASLt | Change |
|---|---|---|---|
| Step time | | | 0.6% less |
| Maximum relative error on the real forward shapes | 0.00368 | 0.00735 | 2.0 times less |

The errors are against an FP32 reference. The DeepGEMM kernel has a relative error of 0.0026 at 4096 x 8192 x 8192.

The tables compare the DeepGEMM kernel with cuBLASLt where a cuBLASLt time exists. M is the token count of the GEMM, K is the reduction length and N is the output width.

The TP1 x CP4 layout runs unsharded shapes, so each dimension that TP2 shards is twice as large. The router selects the DeepGEMM configuration 128,256,1,1,4,2 at all four shapes, and one entry for each shape serves the training step and the eval step.

The four configurations together remove about 15 ms per step on one node.

The next table gives the training shapes and the eval shapes of the TP1 x CP4 layout at 600 W. The fc1 rows use the DeepGEMM kernel with the amax epilogue that the SwiGLU kernel reads. The outputs of the two kernels are bitwise identical at all four shapes.

| GEMM | M, K, N | Kernel | Time in us | TFLOPS | SM | Tensor pipe | L2 | DRAM | Speedup over cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|
| qkv | 4096, 8192, 10240 | DeepGEMM | 459.4 | 1496 | 92.9 | 97.2 | 50.4 | 19.3 | 1.09 times |
| qkv | 4096, 8192, 10240 | cuBLASLt | 500.9 | 1372 | 91.1 | 93.4 | 58.0 | 20.7 |  |
| proj | 4096, 8192, 8192 | DeepGEMM | 365.1 | 1506 | 92.6 | 96.9 | 49.1 | 19.8 | 1.08 times |
| proj | 4096, 8192, 8192 | cuBLASLt | 393.9 | 1396 | 90.8 | 93.0 | 60.5 | 20.7 |  |
| fc1 | 4096, 8192, 57344 | DeepGEMM | 2639.0 | 1458 | 96.2 | 98.5 | 61.1 | 16.6 | 1.08 times |
| fc1 | 4096, 8192, 57344 | cuBLASLt | 2853.2 | 1349 | 93.3 | 93.9 | 63.9 | 17.8 |  |
| fc2 | 4096, 28672, 8192 | DeepGEMM | 1293.8 | 1487 | 95.1 | 99.2 | 50.2 | 17.8 | 1.09 times |
| fc2 | 4096, 28672, 8192 | cuBLASLt | 1406.1 | 1368 | 95.7 | 97.8 | 59.6 | 23.8 |  |
| qkv | 8192, 8192, 10240 | DeepGEMM | 946.0 | 1453 | 93.8 | 98.2 | 52.4 | 19.0 | 1.07 times |
| qkv | 8192, 8192, 10240 | cuBLASLt | 1011.2 | 1359 | 92.1 | 94.1 | 59.2 | 19.2 |  |
| proj | 8192, 8192, 8192 | DeepGEMM | 751.4 | 1463 | 93.6 | 98.0 | 49.8 | 19.7 | 1.06 times |
| proj | 8192, 8192, 8192 | cuBLASLt | 799.3 | 1376 | 91.8 | 93.7 | 61.0 | 19.4 |  |
| fc1 | 8192, 8192, 57344 | DeepGEMM | 5277.4 | 1458 | 97.2 | 98.5 | 60.4 | 17.7 | 1.08 times |
| fc1 | 8192, 8192, 57344 | cuBLASLt | 5720.7 | 1345 | 93.8 | 94.1 | 65.3 | 17.6 |  |
| fc2 | 8192, 28672, 8192 | DeepGEMM | 2627.9 | 1464 | 95.2 | 99.4 | 53.2 | 19.0 | 1.09 times |
| fc2 | 8192, 28672, 8192 | cuBLASLt | 2854.9 | 1348 | 96.2 | 98.0 | 60.6 | 26.2 |  |

The next table gives the same shapes at 400 W.

| GEMM | M, K, N | Kernel | Time in us | TFLOPS | SM | Tensor pipe | L2 | DRAM | Speedup over cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|
| qkv | 4096, 8192, 10240 | DeepGEMM | 577.8 | 1189 | 92.9 | 97.7 | 42.4 | 15.3 | 1.09 times |
| qkv | 4096, 8192, 10240 | cuBLASLt | 630.4 | 1090 | 91.5 | 93.6 | 47.5 | 16.1 |  |
| proj | 4096, 8192, 8192 | DeepGEMM | 456.3 | 1205 | 92.7 | 97.4 | 41.0 | 15.7 | 1.09 times |
| proj | 4096, 8192, 8192 | cuBLASLt | 496.9 | 1106 | 90.8 | 93.0 | 48.2 | 16.2 |  |
| fc1 | 4096, 8192, 57344 | DeepGEMM | 3340.1 | 1152 | 95.7 | 98.6 | 48.9 | 13.0 | 1.16 times |
| fc1 | 4096, 8192, 57344 | cuBLASLt | 3862.0 | 996 | 93.4 | 94.0 | 49.2 | 13.1 |  |
| fc2 | 4096, 28672, 8192 | DeepGEMM | 1633.6 | 1178 | 94.7 | 99.3 | 41.8 | 14.1 | 1.15 times |
| fc2 | 4096, 28672, 8192 | cuBLASLt | 1881.2 | 1023 | 95.7 | 97.9 | 45.4 | 18.0 |  |
| qkv | 8192, 8192, 10240 | DeepGEMM | 1215.9 | 1130 | 93.8 | 98.4 | 46.8 | 14.6 | 1.06 times |
| qkv | 8192, 8192, 10240 | cuBLASLt | 1291.5 | 1064 | 92.6 | 94.2 | 47.0 | 15.2 |  |
| proj | 8192, 8192, 8192 | DeepGEMM | 952.2 | 1155 | 93.7 | 98.3 | 43.0 | 15.5 | 1.12 times |
| proj | 8192, 8192, 8192 | cuBLASLt | 1066.5 | 1031 | 91.9 | 93.7 | 47.5 | 14.3 |  |
| fc1 | 8192, 8192, 57344 | DeepGEMM | 6708.5 | 1147 | 96.7 | 98.6 | 44.8 | 14.1 | 1.14 times |
| fc1 | 8192, 8192, 57344 | cuBLASLt | 7676.1 | 1003 | 93.9 | 94.1 | 50.7 | 13.1 |  |
| fc2 | 8192, 28672, 8192 | DeepGEMM | 3395.9 | 1133 | 94.9 | 99.5 | 41.5 | 17.3 | 1.17 times |
| fc2 | 8192, 28672, 8192 | cuBLASLt | 3960.1 | 972 | 96.3 | 98.1 | 45.6 | 19.9 |  |

**Design principles at a power limit.** Under a power limit the GPU sits at the limit for the whole GEMM, so the kernel time follows the energy per GEMM and not the issue rate. Two choices in the kernel lower the energy of each cycle without changing the work done per cycle, and the clock that the limit allows rises in turn.

- **One ready barrier for each stage.** Math warp group 0 waits for the TMA completion of a stage and then releases it to warp group 1. Warp group 1 waits on that release only, because the release already implies the completed load, so each stage costs one barrier wait for each warp group instead of up to two.
- **Hardware amax reduction.** The amax epilogue of the fc1 GEMM reduces each warp with one `__reduce_max_sync` on the bits of the non-negative floats, whose integer order is the same as their float order, in place of five shuffle and max steps.

**Result.** The fc1 forward on one H200 SXM at 400 W, vboost 0, CUDA graph replays, medians of 5 interleaved rounds:

| Measurement | Without the two changes | With them | Change |
|---|---|---|---|
| Time per call | 4584.2 us | 4235.3 us | 7.61% less |
| Energy per call | 1.8355 J | 1.6872 J | 8.08% less |
| SM clock | 843 MHz | 921 MHz | 9.25% higher |

The clock ranges of the two kernels do not overlap (827 to 867 MHz against 894 to 926 MHz). At a common fixed clock the two take the same time (4381.5 us against 4422.6 us), so the kernel does the same work per cycle with less power per cycle, and the power limit then allows a higher clock. The qkv, proj and fc2 forward have no amax epilogue and a shorter reduction length, and their change is inside the run spread. At 700 W, where the board runs about 10 W below its limit, an 8 GPU training run shows no change in step time (2.6169 s against 2.6064 s, inside the same-node spread). The outputs are bitwise identical on all four training shapes.

**Code.** [kernels](../patches/05-deepgemm.patch#L540), [ready barrier](../patches/05-deepgemm.patch#L954), [amax reduction](../patches/05-deepgemm.patch#L596), [host entry](../patches/05-deepgemm.patch#L167), [router](../patches/01-llama2_70b_lora-nemo.patch#L2576), [chunk provider](../patches/03-transformer-engine/02-te-local-mods.patch#L915), [forward chunk gate](../patches/03-transformer-engine/02-te-local-mods.patch#L76), [install in train.py](../patches/01-llama2_70b_lora-nemo.patch#L8366), [interfaces](../patches/05-deepgemm.patch#L1)

### TE ops fuser

The TE ops fuser is upstream. It builds the MLP from fused TE operations and gives fused casts, a fused gated activation and fused rotary embeddings. The patch adds three things to it.

- An FP8 userbuffer for the reduce scatter. The fc1 dgrad userbuffer uses this path.
- A reset of the recipe state when the quantizer count of an operation changes.
- A default that follows DEEPGEMM, because the DeepGEMM router patches the GEMM that only the fused path calls.

**Setting.** USE_TE_OPS=1. DEEPGEMM=1 needs it.

**Code.** [configuration](../patches/01-llama2_70b_lora-nemo.patch#L588), [FP8 reduce scatter](../patches/03-transformer-engine/02-te-local-mods.patch#L1109)

### Fused residual add, norm and quantize

Each transformer layer has two residual adds per microbatch. In the stock path, Megatron's bias dropout add writes the BF16 sum of the residual and the branch output, and the RMSNorm of the next fused TE operation then reads that sum and writes the E4M3 input of the next GEMM. In the backward, autograd adds the gradients of the norm branch and the residual branch, and the linear layer that produced the branch output casts that sum to E5M2.

This change replaces each pair with one Triton kernel. The fused residual add and RMSNorm pattern follows the allreduce, residual add and RMSNorm fusion family of HPC-Ops (Tencent, `github.com/Tencent/hpc-ops`). In the forward, the bias dropout add only records its two inputs, and the next RMSNorm call reads them once and writes the BF16 sum, which is still needed as the next residual, and the E4M3 norm output with the delayed scaling quantizer of the next operation. It also writes the amax, the inverse scale and the reciprocal standard deviation in the TE layout. When the quantizer also needs columnwise data, the FP8 transpose is made from the same output, as in TE's norm. In the backward, an autograd function at the norm input adds the two gradients. The same kernel casts the sum to E5M2 with the grad output quantizer of the producing linear layer, and that layer's cast returns this result.

The sum is bitwise identical to the stock add. The FP32 reduction order of the norm is different from cuDNN, so the norm output can change by one BF16 unit in the last place. In a unit test at the production shape, 28 of 33.5 million E4M3 values differ by one step. The backward sum and its E5M2 cast, amax and inverse scale are bitwise identical to the stock path.

**Setting.** The kernels are always on in training. They need FP8 delayed scaling, BF16 activations, no bias, hidden dropout 0, no FP32 residual, no recompute or offload of the norms and one pipeline stage. The evaluation uses the forward kernel. The last layer's MLP add, which feeds the final norm, uses the stock operations.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit. The kernel times show where the saving comes from.

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 2.7724 s | 2.7402 s | 1.2% less |
| Energy per GPU per step | 1.864 kJ | 1.856 kJ | 0.4% less |
| Forward add, norm and cast per site | 78.6 us | 62.0 us | 21% less |
| Backward add and cast per site | 72.9 us | 59.1 us | 19% less |

The next table gives the evaluation forward at the evaluation shape, 8192 tokens by 8192 values, measured with Nsight Compute.

| Kernel | Implementation | Power limit | Time in us | SM | L2 | DRAM | Speedup over the stock kernels |
|---|---|---|---|---|---|---|---|
| add, norm and cast, evaluation | fused kernel | 600 W | 114.9 | 27.0 | 89.9 | 81.3 | 1.52 times |
| BF16 add | stock | 600 W | 91.2 | 16.5 | 91.4 | 88.0 | |
| TE RMSNorm with FP8 output | stock | 600 W | 82.9 | 60.8 | 56.9 | 47.0 | |
| all stock kernels | stock | 600 W | 174.1 | | | | |
| add, norm and cast, evaluation | fused kernel | 400 W | 115.2 | 27.9 | 89.9 | 81.0 | 1.53 times |
| BF16 add | stock | 400 W | 90.8 | 17.2 | 92.1 | 88.3 | |
| TE RMSNorm with FP8 output | stock | 400 W | 86.0 | 59.4 | 54.7 | 45.2 | |
| all stock kernels | stock | 400 W | 176.8 | | | | |

The next table gives the add, norm and cast per evaluation iteration on rank 0 in a full run with 8 H200 SXM GPUs, a 400 W power limit and an SM clock of 1150 MHz.

| Kernels | Time per evaluation iteration |
|---|---|
| stock add and RMSNorm, 160 calls | 37.1 ms |
| fused kernel, 161 calls | 29.7 ms |

The evaluation forward runs with autograd enabled. The check for an evaluation layer does not depend on the grad mode.

**Code.** [forward kernel](../patches/01-llama2_70b_lora-nemo.patch#L3885), [backward kernel](../patches/01-llama2_70b_lora-nemo.patch#L3915), [deferred add](../patches/01-llama2_70b_lora-nemo.patch#L4094), [residual fork](../patches/01-llama2_70b_lora-nemo.patch#L4115), [norm hook](../patches/01-llama2_70b_lora-nemo.patch#L4179), [cast hook](../patches/01-llama2_70b_lora-nemo.patch#L4256), [install](../patches/01-llama2_70b_lora-nemo.patch#L4427), [train.py hook](../patches/01-llama2_70b_lora-nemo.patch#L8418)

### RMSNorm backward folded into the residual gradient add and cast

In the backward, each residual fork adds the gradient that comes back through the next RMSNorm to the residual gradient and casts the sum to E5M2 for the GEMM that consumes it. Before, this took three kernels per fork: the RMSNorm backward of TE, the finalize kernel of its weight gradient, and the add and cast kernel of the fused residual path.

The weights of these norms are frozen under LoRA, so the norm needs no weight gradient. This change defers the backward of a norm that reads the newest residual fork output and has a frozen weight, checked on the parameter itself. The fork's backward then runs one Triton kernel per row. The kernel computes the input gradient of the norm in FP32 from the saved inverse RMS, rounds it to BF16 as TE's norm backward stores it, adds the residual gradient in BF16, and writes the E5M2 data with the scale of the consumer's quantizer, its amax and the inverse scale. Any other norm runs the stock backward, and a deferred backward that does not reach its fork raises an error.

**Setting.** Always on in training with the fused residual path. It needs a frozen RMSNorm weight and an E5M2 delayed scaling quantizer on the consumer of the residual gradient.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit, with a profile of two steps.

| Measurement | Before calls | Before time | After calls | After time | Change |
|---|---|---|---|---|---|
| RMSNorm backward kernels per step | 320 | 19.8 ms | 2 | | |
| Weight gradient finalize kernels per step | 320 | 1.9 ms | 2 | | |
| Add and cast kernels per step | 318 | 19.3 ms | 0 | | |
| Fused kernels per step | | | 318 | 25.6 ms, 80.6 us per call | |
| Kernel time per step | | | | | 15.3 ms less |

**Code.** [kernel](../patches/01-llama2_70b_lora-nemo.patch#L4005), [launch](../patches/01-llama2_70b_lora-nemo.patch#L4037), [fork backward](../patches/01-llama2_70b_lora-nemo.patch#L4129), [frozen weight check](../patches/01-llama2_70b_lora-nemo.patch#L4215), [deferred norm backward](../patches/01-llama2_70b_lora-nemo.patch#L4229)

### Rowwise-only qkv input cast and a pre-zeroed fc1 amax slot

The qkv layer with its LoRA adapter is a chain of fused TE operations: RMSNorm, a quantize operation, a fork to the LoRA branch and the linear layer. The RMSNorm writes FP8 with the quantizer of that quantize operation. Nothing sets the usage of this quantizer, so it keeps the default of rowwise and columnwise data. On Hopper, FP8 GEMMs only take the TN layout, so the norm also writes a transposed copy of the [tokens, hidden] activation with a separate transpose kernel. Nothing reads that copy. The base weight is frozen, so the linear layer needs no weight gradient and does not keep the transposed input, and the LoRA branch reads the rowwise data.

This change makes the forward quantizer of the quantize operation rowwise-only before each forward. If a later consumer needs the columnwise data, it asks the tensor for it and TE builds the transpose at that point, as it does for a weight gradient.

The fc1 GEMM writes the amax of its output from its epilogue, and the SwiGLU kernel uses that amax. Before, each fc1 GEMM got a new zero-filled amax buffer, which costs a memset and an idle gap before the GEMM. Now the slot is persistent. The fused residual add and norm kernel that makes the fc1 input sets it to zero in the same launch. The GEMM uses the slot only when it has been set to zero since its last use. Otherwise it allocates a buffer as before.

Both changes give bitwise identical results. In a unit test of the qkv branch at the production shape, the output, the input gradient and both LoRA weight gradients are bitwise identical over three iterations, with plain and fused norm inputs. The amax value is unchanged.

**Setting.** Always on in training. The quantizer change applies to a delayed scaling FP8 quantizer with columnwise usage on a forward quantize operation. The fc1 slot needs the fc1 amax epilogue of the DeepGEMM kernel and the fused residual norm kernel.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit, with a profile of two steps.

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 2.7291 s | 2.7051 s | 0.9% less |
| Energy per GPU per step | 1.826 kJ | 1.850 kJ | 1.3% more |
| Transposes of the qkv input per step | 160 | 0 | 3.24 ms less |
| Memsets before the fc1 GEMM per step | 160 | 0 | |
| Idle gap before each fc1 GEMM | 5.85 us | 0.30 us | 0.89 ms per step less |
| Kernel and idle time per step | | | 4.3 ms less |

**Code.** [quantizer usage](../patches/01-llama2_70b_lora-nemo.patch#L6711), [fc1 slot hookup](../patches/01-llama2_70b_lora-nemo.patch#L6722), [zero store in the norm kernel](../patches/01-llama2_70b_lora-nemo.patch#L3910), [fc1 slot](../patches/01-llama2_70b_lora-nemo.patch#L2355), [fc1 GEMM use](../patches/01-llama2_70b_lora-nemo.patch#L2559), [train.py hook](../patches/01-llama2_70b_lora-nemo.patch#L8424)

## 3. MLP activation

### SwiGLU fusion kernels

This change replaces the SwiGLU forward and backward of the TE ops path with CUDA kernels. The kernels are for plain SwiGLU with the FP8 cached input that FP8_ACT=1 gives.

The forward kernels run in training. In evaluation, a forward kernel without the input cache writes the E4M3 output and its amax, and its output is bitwise the output of the TE kernel. The first kernel computes SwiGLU and writes the output in E4M3 with the delayed scale of the next operation. It also writes the amax of the output and the amax of the input. The second kernel quantizes the saved input to E4M3 with the current scaling rule of TE.

In the backward, one kernel reads the saved E4M3 input and rounds each dequantized value to BF16 in registers. This is the same BF16 value that the TE dequantize gives. The kernel then applies the TE SiLU math and writes the E5M2 gradient with its amax and its inverse scale. The GEMM accumulation is the same as before. Other data types and layouts use the TE path.

The kernel source is in the swiglu_fusion folder next to the benchmark. Build it after TE, because it includes the TE common headers.

**Setting.** SWIGLU_FUSION=1, which is also the default. SWIGLU_FUSION=0 turns the kernels off.

**Result.** The table gives the measurements of the SwiGLU fusion. The two isolated kernel rows use M 8192 with gate and up of 14336 columns each. The per-call table below uses the TP1 training shape.

| Test | Power limit | Before | After | Step or kernel time | Energy |
|---|---|---|---|---|---|
| Forward kernel alone | 400 W | 755.72 us | 566.01 us | 25.1% less | 24.2% less |
| Backward kernel alone | 600 W | 483.39 us | 288.57 us | 40.3% less | |

A real weight run with the fusion reaches the target at 3072 samples, with eval score 0.9230.

At TP1 the fc1 GEMM provides the input amax, so one forward kernel writes the output and the input cache from a single read of the input. This follows the fused epilogue approach of SonicMoE, arXiv 2512.14080. Both kernels issue the sigmoid reciprocals of a 16-element vector before the rare exact-division fallback, convert to FP8 two values at a time and step through the tensor without an integer division per step. The outputs are bitwise identical to the stock kernels. The forward kernel takes two vectors per thread per step on 2112 blocks, and the backward kernel one vector on 1056 blocks.

The next table gives the time per call at the TP1 shape, M 4096 with gate and up of 28672 columns each, on one H200 SXM GPU with a 400 W power limit and CUDA graph replay, as the mean of two runs. The stock kernels are the TE SwiGLU forward with the E4M3 cast and the TE SwiGLU backward on the BF16 input. The fused forward also writes the E4M3 input cache that the fused backward reads.

| Kernel | Stock (us) | Fused (us) | Change |
|---|---|---|---|
| Forward, one pass (swiglu_fwd_cache_known_amax) | 246.0 | 252.3 | 2.6% more |
| Backward (swiglu_cached_fp8_bwd) | 397.3 | 219.0 | 44.9% less |
| Forward and backward | 643.3 | 471.3 | 26.7% less |

The fused forward takes 2.6% more time than the TE forward because it also writes the E4M3 input cache. The fused backward reads this cache instead of the BF16 input, so the two together take 26.7% less time.

The next table gives the evaluation forward at the evaluation shape, M 8192 with gate and up of 28672 columns each, measured with Nsight Compute.

| Kernel | Implementation | Power limit | Time in us | SM | L2 | DRAM | Speedup over the stock kernels |
|---|---|---|---|---|---|---|---|
| SwiGLU forward, evaluation | fused kernel | 600 W | 289.1 | 60.2 | 89.5 | 83.9 | 1.33 times |
| TE gated activation | stock | 600 W | 385.8 | 71.5 | 73.4 | 62.9 | |
| SwiGLU forward, evaluation | fused kernel | 400 W | 305.2 | 64.5 | 87.4 | 79.5 | 1.33 times |
| TE gated activation | stock | 400 W | 405.2 | 73.3 | 70.4 | 59.9 | |

**Code.** [kernels](../patches/01-llama2_70b_lora-nemo.patch#L9451), [Python module](../patches/01-llama2_70b_lora-nemo.patch#L6724)

## 4. Communication

### Context parallel exchange on the IPC transport

This change replaces the NCCL all to all of the context parallel exchange with a bulk copy over memory that the peers map through CUDA IPC. It covers all eight exchanges: Q, K, V and the output in the forward, and dO, dQ, dK and dV in the backward.

- For each payload size, each rank allocates two send slots and a counter block that its peers read through CUDA IPC, so the ranks of a context parallel group must be on the same node. The slots are created before the CUDA graph capture.
- The producer copies its payload into a slot and publishes a counter. Each consumer pulls its rows from the peer slot on a side stream and increments its done counter.
- Single block spin kernels read the counters on the GPU, so the whole sequence stays in the full iteration CUDA graph. A spin holds one SM, so at most `cp_size` of the 132 SMs are held.
- The producer of a call needs only its consumers' call two places back, and a consumer needs only the producers' current call, so the cross rank lead is bounded at two calls.
- The data and the amax are byte for byte the same as the NCCL exchange.

The TMA pull kernel of the `tma` carrier is the same T-CCL derived primitive as the reduce scatter below, used as a plain peer copy.

**Setting.** CP_A2A_COMM selects the carrier. `nccl`, the code default, keeps the stock exchange. `tma` pulls each peer's slot with the `tma_copy_blocks` entry of libtma_reduce.so, from TMA_REDUCE_LIB or the build of this checkout. `ce` copies the same bytes with the copy engines on the same side streams and needs no library. This recipe names `ce`. The transport needs the sbhd layout; other layouts use the stock exchange. The CTAs of each pull come from a per GPU budget of 12 split over the peers, not from a setting.

**Result.** The transport removes the NCCL exchange buffers, so the peak memory falls.

| Measurement | Stock NCCL | IPC transport | Change |
|---|---|---|---|
| Peak memory | 139,350 MiB | 137,812 MiB | 1.5 GiB less |

This recipe uses `ce` at every power limit. The carriers were compared on one node of 8 H200 SXM GPUs with a single NVSwitch domain, seed 12552, 40 training steps and one evaluation, median over steps 15 to 40 without step 25.

| Measurement | Power limit | nccl | tma | ce |
|---|---|---|---|---|
| Step time | 600 W | 2.8101 s | 2.7829 s | 2.8087 s |
| Step time | 400 W | 4.3081 s | 4.2560 s | 4.2533 s |
| One evaluation | 600 W | 30.0 s | 29.8 s | 29.9 s |
| One evaluation | 400 W | 43.4 s | 42.8 s | 42.9 s |

At 400 W the copy engine carrier takes 1.27 percent less step time than the NCCL exchange and about the same time as the TMA carrier. The exchange moves pure permutations with no reduction, which is the case the copy engines serve directly, and the copy engines add no SM work to the capped power budget.

**Code.** [transport](../patches/01-llama2_70b_lora-nemo.patch#L600), [install](../patches/01-llama2_70b_lora-nemo.patch#L1426)

### Wide copies for the context parallel exchange

The reorder copies of the context parallel exchange move the same bytes with a wider element type, and the FlashAttention 3 forward writes its output in the layout that the quantizer needs. Both changes are always on. Together they cut the copy kernels from 87.0 ms to 16.6 ms per step and shorten the critical path of the attention regions by about 60.5 ms per step. The results are bitwise identical to the stock path.

### SM clock pin under the power cap

The GPU governor allocates SM clock below the power limit, so the per-chip clocks of a node disperse. Pinning
the clock sets a ceiling instead: the governor still allocates below it, and the pin removes the spread at the
top. The ceiling helps while it sits above what the limit allows and costs time once it binds in place of the
limit, so the sweep has a minimum.

**Setting.** `LGC` in MHz, default 1200, applied on every node by `run_rcp.sh` and `run_rcp_het.sh` (which
`run_rcp_ref8.sh` calls) through `h200-tuning/lgc.sh`, which verifies the readback (the driver quantizes the pin to
its clock steps, so 1250 reads back 1260; a value within one step is accepted). `LGC=0` or empty keeps the
driver's own control.

**Result.** 16 GPUs at 420 W with vboost 2, the production recipe, 40 training steps and one evaluation per
pin, mean step time over the run:

| pin | readback | step (s) | eval (s) | TTT (s) |
|---|---|---|---|---|
| none | - | 2.0380 | 18.61 | 101.0 |
| 1150 | 1155 | 2.0089 | 18.18 | 99.3 |
| **1200** | 1200 | **1.9993** | 18.23 | **98.9** |
| 1250 | 1260 | 2.0002 | 18.25 | 99.0 |
| 1300 | 1305 | 2.0163 | 18.43 | 99.8 |

1200 is 2.1 % faster than the driver's own control, and 1300 is past the knee (+0.9 % against 1200). A
single-node sweep at 400 W had its minimum at 1150; at 420 W the power limit binds 20 W later, so the ceiling
stops binding at a higher clock. The 1200 against 1250 difference (0.7 ms) is inside the run spread; the
signal is 1200 against no pin.

At 455 W the minimum moves up: 40 training steps and one evaluation per pin, median step time of steps 6-40
(the first five include the graph capture), the same cached split:

| pin | power limit | readback | step (s) | last 10 steps (s) |
|---|---|---|---|---|
| 1200 | 455 W | 1200 | 1.8911 | 1.9124 |
| **1250** | 455 W | 1260 | **1.8706** | **1.9039** |
| 1300 | 450 W | 1305 | 1.8981 | 1.9194 |

1250 is 1.1 % faster than 1200. These are the first 40 steps: in a full run the step time rises from about
1.89 s to a plateau near 2.00-2.03 s over the first 150 steps as the GPUs heat up at 30 % fan and the
power limit takes more of the clock, so the steady-state ranking can differ from the cold one.

At 450 W (the submission limit, vboost 2) the 1250/1200 gap is gone. Same protocol, seed 12552, one run per
pin, 1250 first, both on the 450 W split calibrated with the pin at 1250
(`results/lgc450-p{1250,1200}`, `results/steptimes.py`):

| pin | power limit | step median (s) | step mean (s) | eval (s) | run (s) |
|---|---|---|---|---|---|
| 1250 | 450 W | 1.8849 | 1.8798 | 17.5 | 93.2 |
| **1200** | 450 W | **1.8806** | 1.8803 | 17.5 | 93.3 |

The difference (0.2 % median, 0.03 % mean) is inside the run spread. The submission uses 1200 MHz at 450 W.

### Uneven context parallel shards

At a power limit the GPUs of one node run at different clocks. Each GPU reaches its limit at a different clock, and the spread is 885 to 960 MHz at 400 W. The ranks of a context parallel group wait for the slowest GPU at each exchange, so the slowest GPU sets the step time. This change gives the faster GPU of a group more tokens, so all ranks of the group finish together.

- The batch slicing, the embedding, the output layer and the loss keep the stock equal shards. At the start of the decoder one all to all moves the boundary tokens so that rank r holds two chunks of c_r tokens. At the end of the decoder the inverse moves them back. The move has an autograd backward.
- Each rank still holds two contiguous chunks in the global order. The context parallel exchange takes a different size from each peer and puts each chunk at its global offset. The send slots are sized for the largest shard.
- Attention runs on the full sequence with the heads split evenly, so its cost does not change. The linear, MLP and elementwise work moves with the tokens.
- The LoRA dropout masks use the global position of each token, so they do not change.
- Each consumer gate of the exchange adds its spin time to a counter on the GPU. The lateness of a rank is the time that its peers wait for it.
- With CP_SHARD_WEIGHTS set to auto, the warmup finds the split before the CUDA graph capture. Each iteration sets a split, captures the full iteration graph, replays it and runs one evaluation step, and reads the lateness of each rank. Tokens move from late ranks to early ranks. The split with the smallest lateness spread is kept. The calibration keeps the same step count on all ranks, so the collectives of the data parallel group stay in step. The calibration batches use random token ids, so the power draw is near the power draw of the training data.
- Every shard is a multiple of 128 tokens.
- Each rank fits a peak memory model to the peaks that it measures in the calibration. The model gives the memory of each token and of each resident fc2 weight transpose. Each rank gets a token cap and as many resident fc2 transposes as fit beside its tokens, with 3 GiB kept free. The first calibration step runs with no resident transposes at all: the load-time count has no measured memory point behind it, and at 16 GPUs with 42 resident the step ran out of memory before the model existed. A rank with more tokens keeps fewer resident transposes, and a rank with fewer tokens keeps more.
- The split and the resident transpose count of each rank go into a cache file. The cache key is the host, the physical GPUs of the group, their power limits and vboost levels, the GPU counts, the sequence length, the context parallel size, the granularity, the memory margin and the load-time resident MLP transposes. A run that finds its key uses the split and runs no calibration step.
- Training never runs in the process that calibrated. When a run calibrates, it writes the cache and every rank exits before the training warmup; the launcher (run_rcp.sh, run_rcp_het.sh) sees the marker line in the log, moves that log to `<run>-cal` and starts the run again once, on the cache. The calibration steps change TransformerEngine state that no restore reproduced. Measured at 16 GPUs, 455 W, the same code: a run that calibrated in-process gave a first-step gradient norm of 56711, 91898 and 98809 and diverged (evaluation loss 9.6 and 12.1 after 40 steps), one gave 59 and converged, and a run on the cached split of a diverged run gave 49 and an evaluation loss of 1.052, the same as the 420 W cached run (49, 1.050). The weights, the frozen FP8 data and the LoRA adapters were unchanged by the calibration (checksums); the FP8 state was not: at the start of training the weight quantizer scales, the amax histories and the scales differed from the cached run, also after a restore of every TE module and weight quantizer from a snapshot. An extra model load runs only when the cache misses.
- Without the memory model, the largest shard is at most CP_SHARD_MAX times the equal shard. The default is 1.03125.
- With equal shards the module does nothing, and the exchange is bitwise the same as before.

**Setting.** CP_SHARD_WEIGHTS set to auto calibrates and uses the cache. A list of weights, one for each physical GPU of the node, sets a fixed split. Unset, `off` or equal weights keep the equal shards. The change needs the IPC transport, either `ce` or `tma`, and the FlashAttention 3 FP8 attention, and it does not run with head chunks. It is active only with a context parallel size of 2 or more. The TP1 x CP2 and TP1 x CP4 recipes set auto. The first run on a machine and power setting calibrates during the warmup, before the timed region, and the later runs use the cache. CP_SHARD_CACHE_DIR sets the cache directory, and CP_SHARD_RECALIBRATE set to 1 ignores the cache.

**Result.** The step 1 loss is bitwise equal to the equal shards, and the evaluation loss at step 40 is in the same band. Measured on one node of 8 H200 SXM GPUs at 400 W with vboost 2 and an SM clock pin of 1150 MHz, tensor parallel 1, context parallel 2, seed 12552, 40 training steps and one evaluation, median over steps 15 to 40 without step 25. The calibration gives 4224 and 3968 tokens to two pairs and equal shards to the other two, with 19 to 38 resident fc2 transposes on each rank. The equal shards keep the resident fc2 transposes of the first 16 layers on every rank.

| Measurement | Equal shards | Uneven shards |
|---|---|---|
| Step time | 4.1672 s | 4.1240 s |

The uneven shards take 1.04 percent less step time.

**Code.** [shards and calibration](../patches/01-llama2_70b_lora-nemo.patch#L1668), [exchange sizes and gate timers](../patches/01-llama2_70b_lora-nemo.patch#L954), [warmup](../patches/01-llama2_70b_lora-nemo.patch#L449)

### Userbuffer overlap

The userbuffer overlap of TE runs the tensor parallel all gather and reduce scatter in chunks during the GEMMs. train.py wires the overlap for any tensor parallelism above 1 with sequence parallelism on. The table and the communication budgets below are the TP2 x CP2 values, and a tensor parallelism of 4 or more uses its own per-op configuration, described after the table.

| Userbuffer | Method | Communication SMs | Buffer |
|---|---|---|---|
| qkv dgrad | pipeline, 4 splits | 8 | BF16 |
| proj dgrad | ring, SM copy | 8 | FP8 |
| fc1 fprop | preset | 8, or UB_FC1_FPROP_NUM_SM | preset |
| fc2 fprop | preset | 4 | BF16 |
| proj fprop | preset | preset | BF16 |

The bulk wgrad and dgrad overlaps are off, because the frozen base weights never run the wgrad GEMM that hides the bulk reduction, and the reduce scatter of the dgrad runs in the pipeline instead. NCCL bootstraps the userbuffers. The H200 has 132 SMs, so fc1 fprop keeps 124 math SMs with 8 communication SMs and 128 math SMs with 4, and fc2 fprop keeps 128 math SMs.

At a tensor parallelism of 4 or more, the qkv fprop and proj dgrad rings copy with 16 SMs, and the fc1 fprop, fc2 fprop and fc2 dgrad rings use the copy engines. The qkv dgrad (16 SMs, 8 splits) and fc1 dgrad (8 SMs, 4 splits) reduce scatters run inside the pipelined GEMMs, and the proj fprop reduce scatter is a pipeline with 16 SMs and 4 splits. A ring that sends with SM copies signals once per block, and the TE patch makes the receive wait for that many signals. The reduce scatter buffer of proj fprop stays BF16, because an FP8 buffer raised the eval loss by about 0.007 at TP4. UB_RS_BF16_EXTRA names further operations that keep a BF16 buffer where TE honors the flag.

The TE patch also lets the userbuffers change their token count for each phase: the eval microbatch of 4 needs buffers for 16384 tokens while training uses 8192, so the Megatron Bridge patch allocates the buffers for the larger shape and TE uses the token count of each call.

**Setting.** TP_COMM_OVERLAP=True, default False. train.py stops with an error unless tensor parallelism is above 1 and sequence parallelism is on. UB_FC1_FPROP_NUM_SM sets the communication SMs of fc1 fprop, and the TP2 x CP2 recipe sets it to 4.

**Result.**

| Measurement | Value |
|---|---|
| Step time with 4 communication SMs for fc1 fprop, 600 W | 10.4 ms less |
| Step time with 4 communication SMs for fc1 fprop, 350 W | 32.8 ms less |

**Code.** [ring settings](../patches/01-llama2_70b_lora-nemo.patch#L8687), [token count per phase](../patches/03-transformer-engine/02-te-local-mods.patch#L100), [buffer size](../patches/02-megatron-bridge.patch#L61)

### qkv all gather on the userbuffer ring

**Runs under the userbuffer overlap.**

The qkv layer has LoRA, so it returns its norm output to the adapter, and upstream TE then turns off the userbuffer all gather of the qkv forward and runs an NCCL all gather. This change keeps the userbuffer all gather on when the layer returns the local norm output, and the NCCL gather runs only when the layer returns the gathered output. At TP 2 the qkv forward uses the preset ring with 1 SM and the copy engine, and at a tensor parallelism of 4 or more it uses the ring with 16 SMs.

**Setting.** This change is part of TP_COMM_OVERLAP=True.

**Result.** The step time decreases by 0.86%, 3.367 s against 3.396 s. The step 1 loss is bitwise identical to the NCCL gather.

**Code.** [condition](../patches/03-transformer-engine/04-te-lora-qkv-ub-ag.patch#L10)

### Fused norm, quantize and userbuffer fill

**Runs under the userbuffer overlap at any TP.**

At the qkv layer, one Triton kernel replaces three operations after the reduce scatter. These operations are the RMSNorm forward, the FP8 delayed scaling quantize and the copy into the all gather userbuffer. The kernel writes four outputs.

1. The BF16 norm output for the LoRA adapter.
2. The E4M3 data for the weight gradient GEMM. This copy is separate, because later microbatches use the userbuffer before the backward runs.
3. The same E4M3 data into the local chunk of the userbuffer.
4. The reciprocal standard deviation in the TE layout for the backward.

This decreases the memory passes from six to four. The FP32 reduction order of the norm is different from cuDNN, which can change the norm output by one BF16 unit in the last place. The quantize step and the amax update are bitwise identical to TE. The backward is the stock RMSNorm backward.

The fused path runs in training only. The first call on each userbuffer, calls with a different token count and the evaluation use the stock chain.

**Setting.** NVTE_FUSED_POST_RS_NORM=1, default 0. The fused path needs RMSNorm, a delayed scaling FP8 quantizer, the userbuffer all gather and a layer that returns its norm output.

**Result.**

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 3.2551 s | 3.2333 s | 0.7% less |
| Energy per GPU per step | 1.740 kJ | 1.699 kJ | 2.4% less |
| Step time with a 350 W power limit | | | 0.7% less |

**Code.** [kernel](../patches/01-llama2_70b_lora-nemo.patch#L3548), [TE hook](../patches/03-transformer-engine/01-te-fused-post-rs-norm.patch#L25)

### TMA reduce scatter on the userbuffer ring

**Only for TP 2 or more.**

`UB_TRANSPORT=tccl` replaces the ring of the BF16 reduce scatter userbuffers. Each rank reads the partials of its peers through the IPC mapping of the userbuffers. A TMA kernel from TMA_REDUCE_LIB adds them into the output. The kernel follows the reduce scatter design of T-CCL, arXiv 2610.07098. The output is bitwise identical to the ring. The all gather uses the same primitive.

`UB_TRANSPORT` also selects `symk_nccl`, which allocates the collective buffers from a NCCL symmetric memory window so that NCCL runs its symmetric kernels in place of the ring kernels.

**Setting.** `UB_TRANSPORT=tccl` with TMA_REDUCE_LIB set to libtma_reduce.so. The code default is `nccl`, the ring, at every TP size. The CTAs of each TMA kernel come from a per-GPU budget of 12 split over the peers, not from a setting.

**Code.** [reduce scatter](../patches/03-transformer-engine/09-te-ub-symk.patch#L1)

### TP collective transport

The tensor parallel collectives run inside the GEMM on the userbuffer overlap. `TP_COMM_MODE=tccl` or `symk` moves them to the transports above instead.

With the TE ops fuser the transformer layers are TE modules, which reach NCCL through TE's own distributed helpers and not through the Megatron mappings. The takeover therefore wraps `all_gather_into_tensor` and `reduce_scatter_tensor`, where every one of those helpers ends, so it does not depend on the call site. `ub` is the default. With the overlap on, the collectives run inside the GEMM and there is nothing for a Python switch to intercept, so another mode stops the run unless UB_TRANSPORT selects a transport for the overlap itself.

**Setting.** TP_COMM_MODE=nccl, tccl or symk. This recipe runs the default `ub`.

## 5. LoRA adapters

### Fused adapter

**Only for TP 2 or more.**

The fused adapter patches the stock LoRA linear. Megatron Bridge keeps the op fuser off when the tensor parallel size is above 1, so at TP 2 or more the adapter is the stock `LoRALinear` and these patches run. The local path below builds on it.

At TP1 the module installs but its patches never run. The TE ops fuser is on there, so Megatron Bridge wraps the qkv and proj layers in the fused TE linear, which has its own forward and builds its own dropout op from the adapter's probability. The dropout at TP1 is handled by "LoRA dropout folded into the LoRA-A GEMMs" below.

This change is always on. It replaces the adapter dropout with a Triton dropout. The dropout takes one Philox seed per call from the CUDA generator and makes the mask again in the backward, so it keeps the seed in memory in place of the mask. The change also replaces the LoRA forward.

For the proj adapter, the input is sharded on the hidden dimension. The adapter output needs an all to all exchange to the sequence layout. With LORA_A2A=1, the exchange uses one contiguous buffer, and a Triton kernel adds the received data to the residual. The backward is one permute kernel and one exchange. The result is bitwise identical to the stock adapter.

**Setting.** The fused adapter is always on. LORA_A2A=1, default 1, selects the all to all exchange.

**Code.** [fused dropout](../patches/01-llama2_70b_lora-nemo.patch#L5505), [all to all add](../patches/01-llama2_70b_lora-nemo.patch#L5657), [setting](../patches/01-llama2_70b_lora-nemo.patch#L8570)

### Local adapter path

**Only for TP 2 or more.**

The base layers are sharded over the tensor parallel ranks, and the stock adapter follows this sharding with its own exchanges for each adapter, each microbatch and each direction. The local path removes these exchanges: each rank computes a partial LoRA result from the data that it holds, and a collective that the base layer runs already adds the partial results.

At TP1 there is no sharding and nothing to remove, so the fused TE linear of Megatron Bridge runs. The recipe keeps LORA_LOCAL=1 so that one environment serves all layouts.

The local path keeps all parameters, their sharding, the optimizer state and the checkpoint format. The change is in how the adapter runs.

**Column half, qkv layer.** The qkv layer is column parallel. Its input is the sequence local norm output, and its output covers the full sequence.

- Each rank holds a full copy of factor A and its own shard of factor B.
- Each rank multiplies its local rows by A, so this GEMM has only the local share of the rows of the stock GEMM, which is half of them at TP 2.
- One all gather collects the small result of 16 columns over the sequence. The stock path gathers the full norm output of 8192 columns.
- Each rank multiplies the gathered result by its shard of B and adds the result to the base output.
- In the backward, one reduce scatter sends the gradient of the small result back to the sequence shards.

**Row half, proj layer.** The proj layer is row parallel, and TE splits its output GEMM into chunks and sends each chunk to the reduce scatter during the next GEMM chunk. The add before the reduce scatter exists only under the userbuffer overlap, so the local row path falls back to the stock adapter forward when the base layer runs without the overlap.

- Each rank holds its own shard of factor A and a full copy of factor B.
- Each rank computes its partial LoRA result for all tokens.
- The TE patch adds this partial result to each GEMM chunk before the reduce scatter sends the chunk, or into the userbuffer chunk that the pipeline reduce scatter reads. The adapter result thus uses the reduce scatter of the base layer, and the next chunk GEMM hides the add.
- With DEEPGEMM=1, the chunk GEMM adds the partial result in its epilogue before it stores the tile. The result is bitwise identical to a separate BF16 add. A chunk that the DeepGEMM provider declines uses a separate add kernel.
- In the backward, TE returns the gathered output gradient of its dgrad as the gradient of the partial result. Autograd then computes the adapter gradients from it.

**Shared factor collectives.** The full copies of the factors are A of all qkv adapters and B of all proj adapters. One tensor parallel all gather per microbatch collects them at the first adapter call. One reduce scatter in the backward returns their gradients to the shards. The stock adapter runs 4 collectives for these factors in each layer and each microbatch. The model has 80 layers and each step has 2 microbatches, so the stock adapter runs 4 x 80 x 2 = 640 collectives in each step. The local path runs 1 all gather and 1 reduce scatter for each microbatch, which is 4 collectives in each step. The adapter gradients accumulate into main_grad, as in the stock adapter. The local path reads the adapter weights directly, so it runs the forward pre hooks of the adapter layers. These hooks finish the parameter all gather of the distributed optimizer.

**Dropout.** The local path calls the same dropout modules on the same tensors in the same order as the stock path. Thus the random generator state and the masks are the same. Each token row of qkv and each element of proj goes to one rank, so each rank applies its own mask.

**Numerics.** In exact arithmetic, the local path gives the same result as the stock path. The rounding is different in four places.

1. The proj B GEMM runs on each rank before the sum. The stock path sums first and rounds to BF16. The base layer uses the same order for its own partial results.
2. The qkv A GEMM produces all 16 output columns on each rank, where the stock path produces its share of them, which is 8 at TP 2. The reduction length is the same.
3. The proj adapter gradients use the FP8 gathered output gradient of the dgrad. The stock path uses a BF16 gradient. The main dgrad GEMM uses the same FP8 operand.
4. The new gradient reductions of the shared factors use a different summation order.

**Setting.** LORA_LOCAL=1, default 0. The path runs in training and in evaluation.

- The row half needs the pipeline, atomic or ring reduce scatter overlap of TP_COMM_OVERLAP=True. Without the overlap the row half uses the fused adapter and the column half stays local. Under the cuBLASMp overlap, TE stops with an error at the first LoRA partial result.
- With LORA_LOCAL=0, or at TP1 where there is no sharding, the fused adapter runs for both halves.

**Result.**

| Configuration | Step time | Change |
|---|---|---|
| stock adapter path | 3.4703 s | |
| column half local, row half stock | 3.4012 s | 1.99% less |
| both halves, LORA_LOCAL=1 | 3.3181 s | 4.39% less |

**Code.** [local adapter](../patches/01-llama2_70b_lora-nemo.patch#L6159), [row half](../patches/01-llama2_70b_lora-nemo.patch#L6062), [shared factors](../patches/01-llama2_70b_lora-nemo.patch#L5969), [TE join](../patches/03-transformer-engine/03-te-lora-prers-join.patch#L457), [join kernel](../patches/03-transformer-engine/03-te-lora-prers-join.patch#L182), [epilogue add](../patches/03-transformer-engine/06-te-lora-join-fold.patch#L1), [DeepGEMM addend](../patches/05-deepgemm.patch#L1821)

### LoRA dropout folded into the LoRA-A GEMMs

At TP1 the qkv and proj adapters run as a branch of fused TE operations: dropout with probability 0.1, the LoRA A linear layer, the LoRA B linear layer, the scale and the add into the main output. In the stock path, the dropout kernel reads the adapter input and writes a BF16 copy with the dropped values set to zero, plus a bit mask. The A GEMM then reads that copy. In the backward, the A dgrad GEMM writes its output, a second kernel reads it again with the mask, and the A wgrad GEMM reads the saved BF16 copy.

This change replaces the dropout and the A linear layer with one TE operation built from two Triton kernels. The forward kernel makes the dropout mask in its prologue, applies it and the scale 1/(1-p) to each input tile, rounds the tile to BF16 and multiplies it by A in a split K GEMM. The qkv input arrives in E4M3 and is dequantized in the kernel with its inverse scale. The proj input also arrives in E4M3 since the change "FP8 attention output fed straight into the output projection" in section 1; the proj kernel times below were measured with a BF16 input. The backward kernel makes the same mask again. It writes the input gradient, the masked and scaled product of the output gradient and A, and it accumulates the A weight gradient from the masked input into the main gradient. The operation saves the input and an 8 byte seed for the backward, in place of the BF16 copy and the mask.

The mask follows the TE dropout kernel. One Philox4x32-10 call covers 16 elements, an element is dropped when its random byte is below floor(256 p), and the kept values are scaled by 1/(1-p). The seed comes from the default CUDA generator, as in TE, and it is drawn on the device, so the CUDA graph capture and the seeds of each rank are unchanged. The random stream is different from the one of TE, so the result is not bitwise identical. With the same mask, the input gradient is bitwise identical to the stock kernels, and the output and the weight gradient differ only in the GEMM summation order. In a unit test at the production shape, the keep rate is 0.902367 against 0.902344 for TE, and the noise of the output and of both gradients matches the stock dropout within 1%.

**Setting.** The fold is always on in training. Adapter branches with the dropout after the adapter, tensor parallelism, a rank other than 16, 32 or 64, an input size that is not a multiple of 128 or a weight that is not BF16 keep the stock operations.

**Result.** These measurements come from 8 H200 SXM GPUs with a 700 W power limit. The kernel level evidence from an nsys profile comes from the combined run of the accepted changes. The kernel times are from a unit test on one GPU with CUDA graph replays at the production shape.

| Measurement | Before | After | Change |
|---|---|---|---|
| Step time | 2.7724 s | 2.7488 s | 0.85% less |
| Energy per GPU per step | 1.864 kJ | 1.868 kJ | 0.2% more |
| Dropout and A GEMM forward, qkv (E4M3 input) | 66.2 us | 28.5 us | 57% less |
| Dropout and A GEMM forward, proj (BF16 input) | 72.1 us | 41.6 us | 42% less |
| A dgrad, mask and A wgrad backward, qkv | 87.5 us | 63.5 us | 27% less |
| A dgrad, mask and A wgrad backward, proj | 87.6 us | 61.3 us | 30% less |
| Adapter branch forward and backward, qkv | 336.4 us | 283.4 us | 15.8% less |
| Adapter branch forward and backward, proj | 307.1 us | 254.9 us | 17.0% less |

**Code.** [mask](../patches/01-llama2_70b_lora-nemo.patch#L5094), [forward kernel](../patches/01-llama2_70b_lora-nemo.patch#L5178), [backward kernel](../patches/01-llama2_70b_lora-nemo.patch#L5208), [TE operation](../patches/01-llama2_70b_lora-nemo.patch#L5340), [branch rewrite](../patches/01-llama2_70b_lora-nemo.patch#L5421), [install](../patches/01-llama2_70b_lora-nemo.patch#L5439), [utils.py hook](../patches/01-llama2_70b_lora-nemo.patch#L9416)

## 6. Dataloader

### Dataloader reuse

The patch builds the training and eval dataloaders one time and keeps them. The warmup builds them before the timed region and starts their iterators. The first training pass and the first evaluation use the started iterators, so they keep the batches that the warmup loaded. The batch sampler counts those batches when the warmup loads them, so a reset at this point would start the epoch after them and skip them. The eval dataloader uses at most 2 workers.

**Setting.** The reuse is always on.

**Code.** [dataloader cache](../patches/01-llama2_70b_lora-nemo.patch#L9015), [iterator reuse](../patches/01-llama2_70b_lora-nemo.patch#L8853)

### Validation iterator reset

The validation loader uses persistent workers. The stock cyclic iterator starts a new epoch only when the next evaluation asks for a batch. At that moment the loader resets, and the workers prepare the first batches from the start. At the start of each evaluation after the first, the host waits for the first batch while the GPUs are idle.

The patch replaces the cyclic iterator. After each batch, it checks two conditions. The loader has no task in flight, and every dispatched task has been handed out. When both are true, the epoch is complete and the iterator resets the loader at once. The workers then prepare the first batches of the next epoch during the training steps. Both conditions are necessary. With two workers, the results can arrive out of order and wait in a buffer. A reset at that moment drops a buffered batch. The batch order of every evaluation is the same as the stock order.

**Setting.** The reset is always on. It applies to every loader that the cyclic iterator wraps.

**Result.** The measurements come from rank 0 of 8 H200 SXM GPUs with a 400 W power limit.

| Measurement | Stock | Patch |
|---|---|---|
| Idle time of the GPU between the last training kernel and the first evaluation graph | 2.09 s | 0.01 s |

**Code.** [iterator reset](../patches/01-llama2_70b_lora-nemo.patch#L2262), [install](../patches/01-llama2_70b_lora-nemo.patch#L8898)

## 7. Evaluation

### LoRA merge for evaluation

For each evaluation, the code adds the LoRA product A x B into the base weights and runs the eval forward on the merged weights. The FP8 base weights get the new values with stochastic rounding to E4M3 and their existing scale. The generator of the rounding has a fixed seed. After the evaluation, the code restores the original weights.

The qkv and proj weights keep a resident FP8 transpose for their dgrad (section 8). The evaluation forward neither reads nor rebuilds it, so after the evaluation the code transposes it back into the weight on the GPU. This restores the original bytes exactly, and the transpose stays valid for the next dgrad. If the transpose buffer was rewritten during the evaluation, the restore raises an error. A weight without a valid resident transpose saves its original FP8 data in a pinned host cache one time. The base weights are frozen, so their data is the same at each evaluation. Before each merge, a byte checksum on the GPU compares the weights with the cache. When the data changed, the code saves a new copy. The restore of these weights uses pinned asynchronous copies.

One Triton kernel merges each FP8 weight in place. It reads the E4M3 data and the scale, computes the product of the two adapter factors, adds it to the weight and rounds the sum to E4M3 with stochastic rounding. The product is rounded to BF16 twice, as the torch matrix product and the scaling round it. The random numbers come from an integer hash of the element index and a seed that is fixed for each weight, so the merge is the same at each evaluation. The kernel takes any adapter rank. It pads the rank to a power of two and masks the extra rows. For a tensor parallel size above 1, the code first gathers the sharded factor, as the merge of Megatron Bridge does. Weights that are not FP8 use the generic torch path.

**Setting.** LORA_EVAL_MERGE=1, default 0. The kernel has no setting.

**Result.** The next table gives the kernel time on one H200 SXM GPU, measured with CUDA events. The measurement in a full evaluation is 31.0 ms for all 160 weights.

| Weight | Shape | Kernel time |
|---|---|---|
| qkv | 10240 x 8192 | 215.7 us |
| projection | 8192 x 8192 | 175.1 us |

The next table gives the qkv merge kernel measured with Nsight Compute, as the other kernels in this document. The kernel does not use sufficient power to reach either limit, so the SM clock stays near 1.79 GHz at both limits and the two rows agree. The SM throughput is the limit of the kernel and the DRAM is not.

| Kernel | Power limit | Time in us | SM | L2 | DRAM |
|---|---|---|---|---|---|
| LoRA merge, qkv | 600 W | 136.3 | 82.6 | 28.7 | 21.7 |
| LoRA merge, qkv | 400 W | 137.2 | 82.6 | 28.5 | 21.5 |

**Code.** [merge kernel](../patches/01-llama2_70b_lora-nemo.patch#L6251), [factor gather](../patches/01-llama2_70b_lora-nemo.patch#L6287), [kernel launch](../patches/01-llama2_70b_lora-nemo.patch#L6314), [use in the evaluation context](../patches/01-llama2_70b_lora-nemo.patch#L9374), [restore from the resident transpose](../patches/01-llama2_70b_lora-nemo.patch#L9402), [stochastic rounding of the generic path](../patches/01-llama2_70b_lora-nemo.patch#L9290)

### Eval microbatch 4 with padding

Each GPU processes eval microbatches of 4 sequences. The stock loader gives a short last microbatch. The patch pads the last microbatch with copies of the last row and sets their loss mask to zero. The same wrapper removes the dense causal attention mask from each eval microbatch. The causal kernels apply the same pattern, and the training batches also use the causal kernels directly. This removes a host to device copy of 256 MiB for each eval microbatch.

**Setting.** VAL_MBS=4, VAL_GBS=8 and EVAL_PAD_MICROBATCH=1. VAL_GBS is a new variable that sets the eval global batch. Its default comes from VAL_MBS.

**Result.**

| Measurement | Value |
|---|---|
| Eval time against VAL_MBS=1 | 26.6% less |
| Mask removal, each evaluation | 0.25 s less |
| One evaluation with VAL_GBS=8 | 33.7 s |
| One evaluation with VAL_MBS=1 | 105.4 s |

With VAL_GBS=8, the eval runs fewer iterations. An eval microbatch of 8 needs about 143 GB, which is more than the GPU memory.

**Code.** [padding and mask removal](../patches/01-llama2_70b_lora-nemo.patch#L9191), [VAL_GBS](../patches/01-llama2_70b_lora-nemo.patch#L573)

### Attention mask cache

The Megatron Bridge patch builds the causal attention mask of the training dataset one time and keeps it. Each later batch of the same length uses the kept mask. A batch with a new length builds a new mask and keeps it.

**Code.** [mask cache](../patches/02-megatron-bridge.patch#L1)

## 8. Checkpoint load

### Calibrated FP8 checkpoint load

The model keeps its base weights in FP8. The load does three things.

1. It stages the checkpoint shards in high precision on the CPU. This replaces the upstream patch of force_all_tensors_to_non_fp8.
2. After it assembles each weight, it calibrates the amax and the scale of that weight. Thus the quantization uses the real range of the weight.
3. It builds the transposed FP8 copy of each weight after the load.

The load runs in groups of LOAD_GROUP_LAYERS transformer layers, default 20. For each group, the code stages the shards on the GPU, merges them, calibrates them, copies them into the FP8 weights and releases the memory. The arithmetic is the same as a load of the full model, and the step 1 loss is bitwise identical. The host memory stays low, so the fully parallel load with PARALLEL_LOAD=True fits. The TP1 x CP4 recipe stages 4 layers per group, because 20 layers of BF16 weights are about 32 GiB on each rank. LOAD_STREAM=0 loads the full model through host memory, and LOAD_THREADS sets the torch threads during the load.

**Setting.** FP8=True and LOAD_CKPT=True turn on the calibrated load. LOAD_STREAM=1, default 1. LOAD_GROUP_LAYERS=4 at TP1 x CP4, default 20. LOAD_THREADS is half the CPU count of the node, set in the recipe and also the default of the code. FP8_RECIPE selects the FP8 recipe of the weights, default delayed.

**Result.** The load is outside the timed region.

| Load | Load time | Peak RAM used |
|---|---|---|
| full model through host memory | 129 s | about 850 GB |
| groups | 54 s | 145 GB |
| groups, PARALLEL_LOAD=True | 32 s | 142 GB |
| groups, PARALLEL_LOAD=True, LOAD_THREADS=128 | 27 s | 145 GB |

**Code.** [calibration](../patches/01-llama2_70b_lora-nemo.patch#L3086), [group load](../patches/01-llama2_70b_lora-nemo.patch#L3296), [install](../patches/01-llama2_70b_lora-nemo.patch#L8966)

### Transient weight transpose

A Hopper tensor core takes both FP8 operands in the TN layout, which is a column major weight. The dgrad GEMM therefore needs the transpose of the weight. TransformerEngine builds that copy at the first use and keeps it for the whole run, so each weight costs twice its FP8 size for the life of the process.

At TP1 the whole weight set lives on each rank, and the second copy does not fit in the GPU memory.

This change builds the transpose inside the dgrad call and drops it when the GEMM returns. The transposed copy then exists only while one GEMM runs, and the steady state drops by half of the FP8 weight bytes. This makes TP1 x CP4 fit on four H200 GPUs, which removes the heavy communication of tensor parallelism.

**Setting.** FP8_TRANSIENT_TRANSPOSE=1, default 0. The setting covers the delayed scaling and current scaling FP8 quantizers. Other quantizers keep the persistent transpose.

**Result.**

| Measurement | Persistent transpose | Transient transpose | Change |
|---|---|---|---|
| Peak GPU memory, TP1 x CP4 | OOM | 125,202 MiB | fits |

**Code.** [TE quantizer usage](../patches/03-transformer-engine/08-te-transient-transpose.patch#L1), [load path](../patches/01-llama2_70b_lora-nemo.patch#L3103)

### Resident transpose for the qkv and proj weights

With the transient weight transpose, the dgrad of every FP8 linear layer builds the transposed weight, runs the GEMM and drops it, once per microbatch. The base weights are frozen in LoRA fine tuning, so the transposed copy is the same at every call.

This change keeps the transpose of the attention qkv and output projection weights for the whole run. The checkpoint load builds it once from the final FP8 payload, and the dgrad finds it valid and keeps it. If an in place weight update invalidates it, the dgrad rebuilds it into the same buffer before the GEMM, so the CUDA graphs that captured its address stay valid. The 160 resident transposes take 11.25 GiB on each rank.

The MLP weights can also keep a resident transpose, built and kept the same way: WT_RESIDENT_MLP lists, for fc1 and fc2, how many leading layers keep theirs (for example fc1:8,fc2:16). The recipe keeps fc2 resident, for three reasons. First, L2 locality: the transient transpose runs between the kernel that writes the dgrad's input gradient and the dgrad GEMM, and streams the whole weight through the 50 MB L2. For fc2 that input gradient is 32 MB of FP8 (4096 x 8192), written just before by the residual-grad add and FP8 cast, so without the transpose the dgrad can read it from L2. For fc1 the input gradient is the SwiGLU backward output, 235 MB, which does not fit in L2 in either case. The weight transpose itself is far larger than L2 for both, so the GEMM reads it from HBM whether it is transient or resident. Second, per byte the fc2 transpose is slightly slower (235 MB in 136.7 us, 3.44 TB/s, against fc1 470 MB in 264.8 us, 3.55 TB/s), so a resident byte of fc2 saves slightly more time. Third, the fc2 unit is half the size, so it fills the remaining memory more closely. Which layers are resident does not matter: the memory is static and the backward has no communication at TP1 for either transpose to hide behind. The TP1 x CP4 recipe sets fc2:42, 9.2 GiB more on each rank, which is the largest fc2 count that runs training and evaluation at TP1 x CP4 without running out of memory. The TP1 x CP2 recipe loads with fc2:42, the most that fit at equal shards, and the uneven context parallel shards then lower the resident fc2 count of each rank to what its memory model allows. It needs the PyTorch allocator's expandable segments (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True): 42 layers do not fit with the default allocator. The other fc2 weights and all fc1 weights keep the transient transpose.

The dgrad result is bitwise identical. In a unit test at the production shapes, the dgrad output with the resident transpose matches the transient path in every element, and the transpose built at load matches the TE transpose byte for byte.

**Setting.** The qkv and proj part is always on when FP8_TRANSIENT_TRANSPOSE=1. WT_RESIDENT_MLP=fc2:42 and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, defaults empty and unset, add the MLP part. Without FP8_TRANSIENT_TRANSPOSE=1 every weight keeps its transpose and the change does nothing.

**Result.** These measurements come from nsys profiles of rank 0 on 8 H200 SXM GPUs with a 700 W power limit, two training steps per window, one run with the change and one without it. The removed transposes save 13.4 ms per step, and the dgrad GEMMs are unchanged. Training and evaluation run without running out of memory.

| Measurement | Before | After | Change |
|---|---|---|---|
| qkv weight transpose per call | 45.7 us | none | 7.2 ms per step removed |
| proj weight transpose per call | 38.5 us | none | 6.2 ms per step removed |
| qkv dgrad GEMM per call | 528.2 us | 531.4 us | unchanged |
| proj dgrad GEMM per call | 430.8 us | 432.1 us | unchanged |
| Resident transposes per rank | none | 11.25 GiB | 11.25 GiB more |
| Peak GPU memory, 40 steps | | 142,755 MiB | no out of memory in training or evaluation |

**Code.** [weight selection](../patches/01-llama2_70b_lora-nemo.patch#L3192), [MLP layer selection](../patches/01-llama2_70b_lora-nemo.patch#L3202), [setting check](../patches/01-llama2_70b_lora-nemo.patch#L3171), [load time build](../patches/01-llama2_70b_lora-nemo.patch#L3111), [dgrad wrapper](../patches/01-llama2_70b_lora-nemo.patch#L3277)

## 9. Other changes

These changes keep the benchmark correct and easy to operate. Their effect on the step time is about zero, except for the CUDA graphs and the three host-sync removals.

- **Full iteration CUDA graphs.** A training step launches about 94 thousand kernels, and the launch overhead of that many kernels is a large part of the step. A run with the graphs off takes 34% longer per step. The patch sets the graph implementation to `full_iteration`, which captures the whole training iteration, including the attention, the collectives and the optimizer, into one CUDA graph and replays it. Megatron Bridge upstream captures one layer at a time with the `local` scope. A single memory pool is shared between the training graph and the evaluation graph, so the two graphs fit in the memory at the same time. [Code](../patches/01-llama2_70b_lora-nemo.patch#L8511)
- **Megatron Bridge 26.09 port.** The benchmark uses the dataset, packing and batch interfaces of Megatron Bridge 26.09. The patch keeps the sequential batch order and sets the loss mask to zero on end of sequence tokens in the packed batches. It pads each sequence to the maximum length, passes the sampler seed and the drop last setting to the samplers, and uses a separate data parallel group for evaluation. The gradient accumulation count comes from the global batch, the microbatch and the data parallel size. The sample count of the schedule covers the larger of MAX_STEPS and LR_DECAY_STEPS. The training loss is the mean over the batch, and the eval loss is the sum. FP8_RECIPE also accepts tensorwise and blockwise. The validation step turns off offline packing while it runs. [Code](../patches/01-llama2_70b_lora-nemo.patch#L8988)
- **Fused qkv and proj at TP1.** With tensor parallelism 1 Megatron Bridge wraps the qkv and proj layers in a fused LoRA linear whose training forward is a TE ops main branch. With the adapter disabled, which is the case in evaluation under the LoRA merge, the fused branch is kept instead of falling back to the stock layer. The stock fallback writes the norm output in BF16 and casts it afterwards, which rounds twice, and it uses its own quantizer. The fused branch rounds once, as in training. The step 1 loss is bitwise equal to the training path, and the evaluation peak falls by 13.2 GiB. [Code](../patches/01-llama2_70b_lora-nemo.patch#L6631)
- **Optimizer checks.** The optimizer checks that each step succeeds, and it synchronizes the gradient norm across the model parallel ranks. Both checks return the input directly when the model parallel group holds one rank. At tensor parallelism 1 the group holds one rank, and the reduction has nothing to agree with, but its host sync waits for the whole CUDA stream. The early return removes this wait. The first step after an evaluation falls from 4.58 to 3.87 s. [Code](../patches/02-megatron-bridge.patch)
- **Deferred step logging.** The step-end callback copies the loss and the gradient norm to pinned host memory without blocking and logs them one step later, and takes the step time from CUDA events, so the host queues the next step while the GPU runs the current one instead of waiting on a `.item()` sync; the step before an evaluation and the last step still sync, so `eval_start` and `run_stop` follow the GPU. [Code](../patches/01-llama2_70b_lora-nemo.patch#L186)
- **No sync per evaluation iteration.** Megatron Bridge synchronizes the device after each replay of the full iteration evaluation graph; the patch removes it, since the next batch is copied into the static buffers on a stream that waits for the previous replay, so the host prepares the next batch while the GPU runs. [Code](../patches/02-megatron-bridge.patch#L40)
- **No dense attention mask.** The packed collate no longer builds the [MBS, 1, S, S] causal mask (64 MiB per training microbatch, 256 MiB per eval batch) when Megatron Bridge's batch step would discard it, which removes its build, pinning and host to device copy from every training step and evaluation batch. [Code](../patches/01-llama2_70b_lora-nemo.patch#L9048)
- **NVLink placement check.** At startup, each rank finds the physical GPUs of its tensor parallel and context parallel peers. Each pair of peers must have an NVLink connection, and the check stops the run with an error for other placements. [Code](../patches/01-llama2_70b_lora-nemo.patch#L6472)
- **Evaluation start.** The ranks meet at a barrier before each evaluation so that the evaluation collectives launch in step. [Code](../patches/01-llama2_70b_lora-nemo.patch#L9427)
- **Warmup.** The warmup uses batches of the training shape, saves and restores the learning rate also when it is a tensor, and sizes the eval warmup from the data parallel group. [Code](../patches/01-llama2_70b_lora-nemo.patch#L314)
- **Training end.** The logging callback leaves the release of the CUDA graphs at the end of training to Megatron Bridge 26.09. [Code](../patches/01-llama2_70b_lora-nemo.patch#L146)
- **Debug statistics.** The debug statistics callback loads when STAT_CALLBACK_FNAME is set. [Code](../patches/01-llama2_70b_lora-nemo.patch#L8786)
- **Process control.** The process group closes with abort at the end, and the process exits with the return code of the run. The rank variables also accept RANK and LOCAL_RANK. [Code](../patches/01-llama2_70b_lora-nemo.patch#L8326)
## 10. RCP

The runs in this section use global batch 12 on 12 H200 GPUs at a power limit of 700 W, on three nodes with four GPUs each. The layout is tensor parallel 1, context parallel 4 and data parallel 3, and the values of the Run settings table apply with the learning rate schedule of global batch 12, which has a learning rate of 5e-4, a warmup of 40 steps and a cosine decay over 533 steps.

| Seed | Samples | TTT in s | Median step in s |
|---|---|---|---|
| 12552 | 3456 | 942 | 2.807 |
| 12553 | 3456 | 940 | 2.792 |
| 12554 | 3456 | 945 | 2.810 |
| 12555 | 3840 | 1057 | 2.795 |
| 12556 | 3456 | 944 | 2.813 |
| 12557 | 3840 | 1055 | 2.799 |
| 12558 | 3840 | 1055 | 2.815 |
| 12559 | 3456 | 942 | 2.792 |
| 12560 | 3456 | 943 | 2.814 |
| 12561 | 3840 | 1049 | 2.796 |

## Environment variables

| Variable | What it does |
|---|---|
| `TP_COMM_MODE` | Selects the TP collective transport. `ub` is the default and runs TE's UserBuffers overlap inside the GEMM. `nccl`, `symk` and `tccl` intercept the Python collectives instead. |
| `UB_TRANSPORT` | Selects what carries the ring-exchange reduce-scatter. `nccl` is the stock ring, `symk_nccl` a symmetric NCCL window, `tccl` our TMA reduce over the IPC-mapped peer buffer. |
| `CP_A2A_COMM` | Carries the CP attention exchange at any cp. `nccl` is the default and keeps the stock exchange, `tma` pulls each peer's slot with a TMA kernel, `ce` with the copy engines. |
| `CP_SHARD_WEIGHTS` | Uneven context parallel shards. `auto` finds the split in the warmup and caches it. A list of weights, one for each physical GPU of the node, sets a fixed split. Unset or `off` keeps equal shards. |
| `CP_SHARD_MAX` | The largest shard as a multiple of the equal shard. Without it the memory model of the calibration sets the cap of each rank. |
| `CP_SHARD_CACHE_DIR` | Directory of the shard calibration cache. Unset uses `cp_shard_cache` in MLPERF_ROOT. |
| `CP_SHARD_RECALIBRATE` | `1` ignores the cached split and calibrates again. |
| `DEEPGEMM` | Routes the dense forward FP8 GEMMs and the UserBuffers chunk GEMMs to DeepGEMM instead of cuBLAS. |
| `DEEPGEMM_NUM_SMS` | Persistent grid of the DeepGEMM per-tensor kernel. Unset keeps one block per SM. `128` makes the tile counts of the qkv, proj and fc1 forward divide the grid, and the fc2 forward keeps the full count. |
| `SWIGLU_FUSION` | Uses the cached-FP8 SwiGLU kernels for the activation forward and backward. |
| `NVTE_CP_A2A_HEAD_CHUNKS` | Head groups the CP all-to-all attention is split into. Unset runs the stock exchange. |
| `NVTE_FUSED_POST_RS_NORM` | Uses the fused post-residual RMSNorm, quantize and userbuffer fill kernel. |
| `FP8_TRANSIENT_TRANSPOSE` | Builds each weight transpose in the dgrad and drops it after the GEMM, instead of keeping it for the whole run. The qkv and proj weights keep a resident transpose. |
| `WT_RESIDENT_MLP` | `fc1:N,fc2:M`: the fc1 weights of the first N layers and the fc2 weights of the first M layers also keep a resident transpose. |
| `TMA_REDUCE_LIB` | Path to the TMA reduce library. Unset uses the checkout's build. |
| `TCCL_PEER_ROWS` | Overrides the staging buffer row count for the TP tccl path. Unset sizes it from the model config. |
| `TCCL_PEER_COLS` | Overrides the staging buffer column count for the TP tccl path. Unset sizes it from the hidden size. |
| `TCCL_CTAS` | CTAs per TMA launch on the TP tccl path. The CP path computes its own budget. |
| `UB_RS_BF16_EXTRA` | Names further userbuffer operations, comma separated, whose reduce-scatter buffer stays BF16. It acts only where TE honors the flag. |
| `NCCL_SYM_CTAS` | CTAs for NCCL's symmetric kernels. Only read when the symmetric pool is on. |
| `UB_FC1_FPROP_NUM_SM` | Comm SMs reserved for the FC1 forward all-gather ring. `4` leaves 128 math SMs. |
| `NVTE_FAV3_FP8_FWD` | Sends the FP8 attention forward to the FlashAttention 3 kernel, which also writes the FP8 output for the next exchange. |
| `LORA_LOCAL` | Uses the local adapter path, which removes the dedicated adapter exchanges of the tensor parallel layers. It has no effect at TP1. |
| `LORA_A2A` | Exchanges the proj adapter output with one all-to-all on a contiguous buffer. |
| `LORA_EVAL_MERGE` | Merges the adapter into the base weights for each evaluation, then restores the original weights. |
| `EVAL_PAD_MICROBATCH` | Pads the last evaluation microbatch so every rank runs the same shape. |
| `LOAD_GROUP_LAYERS` | Number of transformer layers the FP8 checkpoint load stages at one time. |
