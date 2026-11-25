"""Grad-output quantization fused straight into DeepGEMM's operand format (FP8 E4M3, 1x128 per-token-group scales).

Today, every DeepGEMM dgrad pays for TWO quantizations of dy: TE casts it to E5M2 with a per-tensor scale (2-bit mantissa),
then dg_dgrad.requant() re-quantizes that to E4M3 1x128. These kernels produce the E4M3 1x128 operand in ONE pass from
the high-precision source, so the extra pass disappears and the operand is more accurate (one rounding, 3-bit mantissa).

  dswiglu_e4m3_1x128(dy, x)   SwiGLU backward (TE convention: x = [a | b], y = silu(a) * b) + quantize.
                              Replaces tex.dswiglu (E5M2 out, 478 us) + requant (161 us) for the FC1 dgrad.
  quant_e4m3_1x128(x)         bf16 [M, N] -> E4M3 + scales.

The result is handed to dg_dgrad as a Float8TensorStorage (E4M3 data, scale_inv = 1) carrying `_ntu_sf` (fp32 scales,
column-major [M, N/128] = DeepGEMM's sfa layout); dg_dgrad then skips its requant. Such a tensor must only ever reach a
DeepGEMM dgrad: dg_dgrad raises if one would fall back to TE's GEMM (its scale_inv is not the real scale).

NTU_DSWIGLU_DG=1 (needs DG_DGRAD=1) patches te.ops.SwiGLU.op_backward.
NTU_DG_GRADQ=1   (needs DG_DGRAD=1) does the same for the other three dgrads (frozen weights, so dy feeds only the dgrad):
  linear_qkv  (TE LayerNormLinear, column): quant_e4m3_1x128(dy)                     replaces cast_transpose E5M2 + requant
  linear_proj (TE Linear, row + SP):        quant(local dy) -> all-gather data + scales   (same FP8 bytes on NVLink)
  linear_fc2  (te.ops ReduceScatter bwd):   quant(local dy) -> all-gather data + scales   (with NTU_FP8_DY_AG's hook)
"""
import os
import torch
import triton
import triton.language as tl

_STATS = {"dswiglu_fused": 0, "dswiglu_fallback": 0, "gradq_local": 0, "gradq_gather": 0, "gradq_fallback": 0}


# NTU_DG_M2=1: round to a 2-bit mantissa (via E5M2, RNE) before storing E4M3, so the lowest E4M3 mantissa bit is
# always 0. Fewer operand bits toggle in the tensor cores -> less power -> +1.3-2.0% sustained dgrad throughput at
# 400 W with real gradients (gemmlib/cand_dg_lowmant.py). Precision = stock TE's E5M2 gradient mantissa, but with
# 1x128 scales instead of one per-tensor scale.
_M2 = os.environ.get("NTU_DG_M2", "0") == "1"


@triton.jit
def _e4m3(y, M2: tl.constexpr):
    if M2:
        y = y.to(tl.float8e5).to(tl.float32)
    return y.to(tl.float8e4nv)


@triton.jit
def _dswiglu_q_kernel(x_ptr, dy_ptr, q_ptr, sf_ptr, M, N: tl.constexpr, BM: tl.constexpr, M2: tl.constexpr):
    # x [M, 2N] bf16 = [a | b], dy [M, N] bf16  ->  q [M, 2N] e4m3 = [da | db],  sf [2N/128, M] fp32 (dequant scales)
    pm, g = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = g * 128 + tl.arange(0, 128)
    rm = r[:, None] < M
    xa = x_ptr + r[:, None].to(tl.int64) * (2 * N) + c[None, :]
    a = tl.load(xa, mask=rm, other=0.0).to(tl.float32)
    b = tl.load(xa + N, mask=rm, other=0.0).to(tl.float32)
    d = tl.load(dy_ptr + r[:, None].to(tl.int64) * N + c[None, :], mask=rm, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-a))
    db = d * (a * sig)                                   # d/db  = dy * silu(a)
    da = d * b * (sig * (1.0 + a * (1.0 - sig)))         # d/da  = dy * b * silu'(a)
    sa = 448.0 / tl.maximum(tl.max(tl.abs(da), axis=1), 1e-30)
    sb = 448.0 / tl.maximum(tl.max(tl.abs(db), axis=1), 1e-30)
    qa = q_ptr + r[:, None].to(tl.int64) * (2 * N) + c[None, :]
    tl.store(qa, _e4m3(da * sa[:, None], M2), mask=rm)
    tl.store(qa + N, _e4m3(db * sb[:, None], M2), mask=rm)
    rr = r < M
    tl.store(sf_ptr + g * M + r, 1.0 / sa, mask=rr)
    tl.store(sf_ptr + (N // 128 + g) * M + r, 1.0 / sb, mask=rr)


@triton.jit
def _dswiglu_q_kernel_v2(x_ptr, dy_ptr, q_ptr, sf_ptr, M, N: tl.constexpr, BM: tl.constexpr, GPP: tl.constexpr):
    # same math as _dswiglu_q_kernel; each program walks GPP column groups so loads can be software-pipelined
    pm, gb = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    rm = r[:, None] < M
    rr = r < M
    r64 = r[:, None].to(tl.int64)
    for gi in range(GPP):
        g = gb * GPP + gi
        c = g * 128 + tl.arange(0, 128)
        xa = x_ptr + r64 * (2 * N) + c[None, :]
        a = tl.load(xa, mask=rm, other=0.0).to(tl.float32)
        b = tl.load(xa + N, mask=rm, other=0.0).to(tl.float32)
        d = tl.load(dy_ptr + r64 * N + c[None, :], mask=rm, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-a))
        db = d * (a * sig)
        da = d * b * (sig * (1.0 + a * (1.0 - sig)))
        sa = 448.0 / tl.maximum(tl.max(tl.abs(da), axis=1), 1e-30)
        sb = 448.0 / tl.maximum(tl.max(tl.abs(db), axis=1), 1e-30)
        qa = q_ptr + r64 * (2 * N) + c[None, :]
        tl.store(qa, (da * sa[:, None]).to(tl.float8e4nv), mask=rm)
        tl.store(qa + N, (db * sb[:, None]).to(tl.float8e4nv), mask=rm)
        tl.store(sf_ptr + g * M + r, 1.0 / sa, mask=rr)
        tl.store(sf_ptr + (N // 128 + g) * M + r, 1.0 / sb, mask=rr)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, sf_ptr, M, N: tl.constexpr, BM: tl.constexpr, M2: tl.constexpr):
    pm, g = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = g * 128 + tl.arange(0, 128)
    rm = r[:, None] < M
    off = r[:, None].to(tl.int64) * N + c[None, :]
    x = tl.load(x_ptr + off, mask=rm, other=0.0).to(tl.float32)
    s = 448.0 / tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    tl.store(q_ptr + off, _e4m3(x * s[:, None], M2), mask=rm)
    tl.store(sf_ptr + g * M + r, 1.0 / s, mask=r < M)


def dswiglu_e4m3_1x128(dy2, x2, bm=32, warps=4, m2=None):
    """dy2 [M, N] bf16, x2 [M, 2N] bf16 -> (E4M3 [M, 2N], fp32 sf column-major [M, 2N/128])."""
    M, N = dy2.shape
    q = torch.empty((M, 2 * N), device=dy2.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((2 * N // 128, M), device=dy2.device, dtype=torch.float32)
    _dswiglu_q_kernel[(triton.cdiv(M, bm), N // 128)](x2, dy2, q, sf, M, N, bm, _M2 if m2 is None else m2, num_warps=warps)
    return q, sf.t()


def dswiglu_e4m3_1x128_v2(dy2, x2, bm=32, gpp=4, warps=4, stages=3):
    M, N = dy2.shape
    q = torch.empty((M, 2 * N), device=dy2.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((2 * N // 128, M), device=dy2.device, dtype=torch.float32)
    _dswiglu_q_kernel_v2[(triton.cdiv(M, bm), N // 128 // gpp)](x2, dy2, q, sf, M, N, bm, gpp, num_warps=warps, num_stages=stages)
    return q, sf.t()


def quant_e4m3_1x128(x2, bm=32, warps=4, m2=None):
    M, N = x2.shape
    q = torch.empty((M, N), device=x2.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((N // 128, M), device=x2.device, dtype=torch.float32)
    _quant_kernel[(triton.cdiv(M, bm), N // 128)](x2, q, sf, M, N, bm, _M2 if m2 is None else m2, num_warps=warps)
    return q, sf.t()


_ONE = {}

# NTU_DUMP_DY_DIR=<dir>: save the high-precision dgrad inputs once (call number NTU_DUMP_AT of each kind), from global
# ranks 0 and 1 (one TP pair), for offline GEMM-energy experiments with real gradients. Off unless the env var is set.
import os as _os
_DUMP_DIR = _os.environ.get("NTU_DUMP_DY_DIR", "")
_DUMP_AT = int(_os.environ.get("NTU_DUMP_AT", "2000"))
_DUMP_N = {}


def _maybe_dump(kind, **tensors):
    if not _DUMP_DIR:
        return
    n = _DUMP_N.get(kind, 0) + 1
    _DUMP_N[kind] = n
    if n != _DUMP_AT:
        return
    r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if r > 1:
        return
    _os.makedirs(_DUMP_DIR, exist_ok=True)
    torch.save({k: v.detach().clone().cpu() for k, v in tensors.items()}, f"{_DUMP_DIR}/{kind}_rank{r}.pt")
    print(f"[NTU_DUMP] {kind} rank {r}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in tensors.items()), flush=True)


def make_dg_operand(q, sf, shape):
    """Wrap (E4M3 data, 1x128 scales) as the Float8TensorStorage TE passes around; dg_dgrad reads `_ntu_sf`."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
    one = _ONE.get(q.device)
    if one is None:
        one = _ONE[q.device] = torch.ones(1, dtype=torch.float32, device=q.device)
    t = Float8TensorStorage(data=q.view(torch.uint8).view(shape), fp8_scale_inv=one, fp8_dtype=tex.DType.kFloat8E4M3,
                            fake_dtype=torch.bfloat16, quantizer=None)
    t._ntu_sf = sf
    return t


def quant_gather_e4m3_1x128(dy_loc, group, tp):
    """Quantize this rank's rows, then all-gather data + scales along the rows. -> (E4M3 [tp*M, N], sf col-major [tp*M, N/128])."""
    import torch.distributed as dist
    q, sf = quant_e4m3_1x128(dy_loc)                     # sf: view of a [N/128, M] buffer
    M, N = q.shape
    qf = torch.empty((tp * M, N), dtype=torch.uint8, device=q.device)
    sff = torch.empty((tp * (N // 128), M), dtype=torch.float32, device=q.device)
    dist.all_gather_into_tensor(qf, q.view(torch.uint8), group=group)
    dist.all_gather_into_tensor(sff, sf.t(), group=group)
    sf_full = sff.view(tp, N // 128, M).permute(1, 0, 2).reshape(N // 128, tp * M)   # [N/128, tp*M]  (2 MB copy)
    return qf.view(torch.float8_e4m3fn), sf_full.t()


def _gradq_ok(dy, quantizer):
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    # (LayerNormLinear sets columnwise usage even for frozen weights; the caller checks ctx.requires_wgrad instead)
    return (type(quantizer) is Float8Quantizer and type(dy) is torch.Tensor
            and dy.dtype == torch.bfloat16 and dy.shape[-1] % 128 == 0)


def apply_gradq():
    import transformer_engine.pytorch.module.base as B
    TEB = B.TransformerEngineBaseModule
    if getattr(TEB, "_ntu_orig_gop", None) is not None:
        return
    orig = TEB.grad_output_preprocess
    TEB._ntu_orig_gop = orig

    def grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
        ok = (getattr(ctx, "fp8", False) and not getattr(ctx, "debug", False) and not getattr(ctx, "ub_overlap_ag", False)
              and not getattr(ctx, "use_bias", True) and getattr(ctx, "requires_wgrad", True) is False
              and getattr(ctx, "requires_dgrad", True) and _gradq_ok(grad_output, quantizer))
        if not ok:
            _STATS["gradq_fallback"] += 1
            return orig(ctx, grad_output, row_parallel_mode, quantizer)
        dy = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        _maybe_dump(f"gop_{'row' if row_parallel_mode else 'col'}_{dy.shape[1]}", dy=dy)
        if row_parallel_mode and ctx.sequence_parallel:
            tp = torch.distributed.get_world_size(ctx.tp_group)
            q, sf = quant_gather_e4m3_1x128(dy, ctx.tp_group, tp)
            _STATS["gradq_gather"] += 1
        else:
            q, sf = quant_e4m3_1x128(dy)
            _STATS["gradq_local"] += 1
        return make_dg_operand(q, sf, q.shape), None

    TEB.grad_output_preprocess = staticmethod(grad_output_preprocess)
    print("[NTU_DG_GRADQ] TE Linear/LayerNormLinear grad output -> E4M3 1x128 DeepGEMM operand (gather: data + scales)", flush=True)


def apply_dswiglu():
    from transformer_engine.pytorch.ops.basic.swiglu import SwiGLU
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.utils import clear_tensor_data
    if getattr(SwiGLU, "_ntu_orig_bwd", None) is not None:
        return
    orig = SwiGLU._ntu_orig_bwd = SwiGLU.op_backward

    def op_backward(self, ctx, grad_output):
        (x,) = ctx.saved_tensors
        ok = (type(self) is SwiGLU and self.glu_interleave_size is None and type(ctx.prev_op_grad_output_quantizer) is Float8Quantizer
              and type(x) is torch.Tensor and type(grad_output) is torch.Tensor and x.dtype == torch.bfloat16
              and grad_output.dtype == torch.bfloat16 and x.shape[-1] == 2 * grad_output.shape[-1]
              and grad_output.shape[-1] % 128 == 0)
        if not ok:
            _STATS["dswiglu_fallback"] += 1
            return orig(self, ctx, grad_output)
        x2 = x.contiguous().view(-1, x.shape[-1])
        dy2 = grad_output.contiguous().view(-1, grad_output.shape[-1])
        _maybe_dump("fc1_dswiglu", dy=dy2, x=x2)
        shape = x.shape
        q, sf = dswiglu_e4m3_1x128(dy2, x2)
        clear_tensor_data(x)                       # (empties x: take the shape first)
        _STATS["dswiglu_fused"] += 1
        return make_dg_operand(q, sf, shape), ()

    SwiGLU.op_backward = op_backward
    print("[NTU_DSWIGLU_DG] te.ops.SwiGLU backward -> fused dSwiGLU + E4M3 1x128 quant (DeepGEMM operand)", flush=True)


def stats():
    return dict(_STATS)
