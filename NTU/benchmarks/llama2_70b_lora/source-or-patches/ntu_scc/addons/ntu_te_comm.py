"""TE ops: FP8 instead of bf16 for the all-gather in the backward of the MLP's ReduceScatter op (NTU_FP8_DY_AG=1).

Megatron's TEFusedMLP (USE_TE_OPS=1) builds  RMSNorm -> BasicLinear(fc1, column, SP) -> SwiGLU -> BasicLinear(fc2, no TP)
-> ReduceScatter.  FC1's BasicLinear gathers its input itself, in FP8. FC2's dy all-gather is done by the separate
ReduceScatter op, whose backward knows nothing about quantization: it gathers bf16 [S, H] (64 MB/rank), then fc2's
BasicLinear quantizes the full gathered tensor. Profile (16 GPU, 400 W): AG 671 us + quantize 72 us per layer.

Here: the ReduceScatter op keeps fc2's grad-output quantizer (the fuser passes it to op_forward as
prev_op_grad_output_quantizer) and its backward does quantize(local) -> FP8 all-gather, exactly what TE's own
row-parallel Linear does. Delayed scaling: the scale is the same tensor either way and the cast is elementwise, so the
gathered FP8 values are bit-identical; only the local amax is recorded (max over the TP group after amax reduction =
the same global amax).
"""
import os
import torch

_STATS = {"fp8_ag": 0, "e4m3_1x128_ag": 0, "bf16_ag": 0}
_GRADQ = os.environ.get("NTU_DG_GRADQ", "0") == "1"
_ORIG = {}


def apply():
    from transformer_engine.pytorch.ops.basic.reduce_scatter import ReduceScatter
    from transformer_engine.pytorch.distributed import gather_along_first_dim
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    if "fwd" in _ORIG:
        return
    _ORIG["fwd"], _ORIG["bwd"] = ReduceScatter.op_forward, ReduceScatter.op_backward

    def op_forward(self, ctx, input_, prev_op_grad_output_quantizer, next_op_input_quantizer):
        q = prev_op_grad_output_quantizer
        ctx.ntu_dy_quantizer = (q if (FP8GlobalStateManager.is_fp8_enabled() and type(q) is Float8Quantizer
                                      and self.process_group_size > 1) else None)
        return _ORIG["fwd"](self, ctx, input_, prev_op_grad_output_quantizer, next_op_input_quantizer)

    def op_backward(self, ctx, grad_output):
        q = getattr(ctx, "ntu_dy_quantizer", None)
        if q is None or not isinstance(grad_output, torch.Tensor) or type(grad_output) is not torch.Tensor:
            _STATS["bf16_ag"] += 1
            return _ORIG["bwd"](self, ctx, grad_output)
        if _GRADQ and not q.columnwise_usage and grad_output.dtype == torch.bfloat16 and grad_output.shape[-1] % 128 == 0:
            import ntu_dg_quant as DQ                  # E4M3 1x128 DeepGEMM operand, gathered as data + scales
            dy2 = grad_output.contiguous().view(-1, grad_output.shape[-1])
            DQ._maybe_dump("fc2_dy", dy=dy2)
            qf, sf = DQ.quant_gather_e4m3_1x128(dy2, self.process_group, self.process_group_size)
            _STATS["e4m3_1x128_ag"] += 1
            shape = list(grad_output.shape); shape[0] *= self.process_group_size
            return DQ.make_dg_operand(qf, sf, shape), ()
        dy, _ = gather_along_first_dim(grad_output.contiguous(), self.process_group, quantizer=q)
        _STATS["fp8_ag"] += 1
        return dy, ()

    ReduceScatter.op_forward, ReduceScatter.op_backward = op_forward, op_backward
    print("[NTU_FP8_DY_AG] te.ops.ReduceScatter backward: quantize -> FP8 all-gather", flush=True)


def stats():
    return dict(_STATS)
