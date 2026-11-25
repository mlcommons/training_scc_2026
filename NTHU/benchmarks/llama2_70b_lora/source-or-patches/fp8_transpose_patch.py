"""Keep at most one layer's FP8 weight transposes alive, for TE 2.x on Hopper.

Imported by the patched train.py (nvidia/patch_nvidia_impl.py, patch 4).

Why this exists. On GPUs below compute capability 10.0 -- Hopper, including
GH200 -- Transformer Engine 2.x cannot run an FP8 GEMM on a transposed weight
(`non_tn_fp8_gemm_supported()` is False), so the backward pass of every FP8
Linear needs a second, column-major uint8 copy of its weight: the Float8Tensor's
`_transpose`. That is one more byte per parameter.

TE 2.2 allocates that copy eagerly, when the weight is first created under
`fp8_model_init`, and keeps it. Measured in the NGC 25.04 container on a GH200:
a 10-layer Llama2-70B slice takes 1.69-1.79 GiB per layer against 0.80 expected,
and the full model OOMs at 94.7 GiB during `configure_model`, before a single
step. On TP=4 nodes, as in NVIDIA's and Krai's submissions, a quarter of that per
GPU fits and nobody notices. At TP=1 on 96GB it does not.

TE 1.x offered `disable_parameter_transpose_cache` to recompute the transpose per
use instead of caching it. In TE 2.2 that mcore flag no longer reaches the
weight. What TE 2.2 does provide is `Float8Tensor.update_usage(columnwise_usage=
False)`, which frees `_transpose`, and the layer forward re-creates it whenever
the weight's quantizer asks for column-wise usage (i.e. whenever grad is
enabled). This module uses those two facts:

1. `TransformerEngineBaseModule.reset_parameters` is wrapped so each module
   frees its transposes as soon as it has built its weights. Init then peaks at
   one module's transposes.

2. A forward pre-hook on every decoder layer frees the transposes of every
   *other* layer that ran since the last hook, in any grad mode. The no-grad
   forward that checkpointing runs first DOES create them: `_LayerNormLinear`
   sets `columnwise=True` on its weight quantizer unconditionally, and `Linear`
   does so whenever fp8 activation recompute is on and not in the recompute
   phase (TE pre-creates them for the recompute). Job 164 OOMed at 90 GiB in
   that forward because the hook used to skip no-grad passes. Under full
   recompute (uniform, one layer), layer i's backward is preceded by its own
   recompute forward, which re-creates its transposes; the next thing to run is
   layer i-1's recompute forward, whose pre-hook frees layer i's. So both passes
   hold about one layer's transposes at a time.

MEM_TRACE=N prints allocated memory before each of the first N layer calls.

The cost is one `fp8_transpose` per FP8 weight per step, a memory-bound kernel.
"""
import os

import torch

_installed = False


def _float8_params(module):
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    for p in module.parameters(recurse=True):
        if isinstance(p, Float8Tensor):
            yield p


def _free(module):
    freed = 0
    for p in _float8_params(module):
        if p._transpose is not None:
            p.update_usage(columnwise_usage=False)
            freed += 1
    return freed


def install_init_patch():
    """Free transposes module by module while the model is being built."""
    global _installed
    if _installed:
        return
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    original = TransformerEngineBaseModule.reset_parameters

    def reset_parameters(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        for p in self.parameters(recurse=False):
            if getattr(p, "_transpose", None) is not None:
                p.update_usage(columnwise_usage=False)
        return out

    TransformerEngineBaseModule.reset_parameters = reset_parameters
    _installed = True


def install_layer_hooks(model):
    """Bound backward-pass transposes to about one decoder layer.

    `model` is anything with a `decoder.layers` ModuleList somewhere below it;
    wrappers exposing `.module` are unwrapped.
    """
    m = model
    while not hasattr(m, "decoder") and hasattr(m, "module"):
        m = m.module
    layers = list(m.decoder.layers)
    touched = set(range(len(layers)))
    trace = {"left": int(os.environ.get("MEM_TRACE", "0"))}

    def make_hook(index):
        def pre_hook(mod, args):
            for j in touched:
                if j != index:
                    _free(layers[j])
            touched.clear()
            touched.add(index)
            if trace["left"] > 0:
                trace["left"] -= 1
                print(f"mem_trace: layer {index} grad={torch.is_grad_enabled()} "
                      f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
        return pre_hook

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_hook(i))
    return len(layers)
