"""Fuse the FP8 upcast into Megatron's SwiGLU backward (patch 11).

With FP8_ACT=1 (`activation_func_fp8_input_store`) Megatron stores the fc1
output for backward as float8_e4m3fn and, in `SwiGLUFunction.backward`, first
upcasts it with `input.to(bf16)` and then calls the torch.compile'd
`swiglu_back`. The upcast is a separate 940 MB copy kernel: 1.7 ms per layer,
0.13 s per step in job 250's profile (the compiled cat kernel is another
0.9 ms). Moving the upcast inside the compiled function lets inductor fuse it
into the same kernel, which then reads FP8 and writes bf16 once.

SWIGLU_FP8_FUSED=0 disables. Same math, so no change to results.
"""
import torch
import torch.nn.functional as F

_installed = False


def install():
    global _installed
    if _installed:
        return
    from megatron.core.fusions import fused_bias_swiglu as m
    from megatron.core.jit import jit_fuser

    @jit_fuser
    def swiglu_back_from_fp8(g, y8):
        y = y8.to(g.dtype)
        y_1, y_2 = torch.chunk(y, 2, -1)
        return torch.cat(
            (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
        )

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        if ctx.fp8_input_store:
            return swiglu_back_from_fp8(grad_output, input), None
        return m.swiglu_back(grad_output, input), None

    m.SwiGLUFunction.backward = backward
    _installed = True
    print("swiglu_patch: FP8 upcast fused into the SwiGLU backward kernel", flush=True)
