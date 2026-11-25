"""Triton drop-in for TE's fused QKV RoPE (tex.fused_qkv_rope_forward / _backward), the path Megatron takes with
FUSED_SINGLE_QKV_ROPE=1. TE's kernels run at ~1.2 TB/s (fwd 142 us, bwd 171 us per layer at 400 W); these stream at
HBM speed. Same arithmetic as TE (fp32 sincos of the angle, x*cos + rotate_half(x)*sin, round once to bf16).

Covered: sbhd, non-interleaved, rotary dim == head dim, cp_size == 1, no start_positions; anything else -> TE.
NTU_ROPE=1 patches transformer_engine.pytorch.attention.rope.FusedQKVRoPEFunc.{forward,backward}."""
import torch
import triton
import triton.language as tl

_STATS = {"fwd": 0, "bwd": 0, "fallback": 0, "quant_nocat": 0, "dequant_nocat": 0, "cat_fallback": 0}


@triton.jit
def _rot(x, c, r, s, MODE: tl.constexpr):
    # x*c + r*s with a fixed rounding pattern
    if MODE == 0:
        return x * c + r * s
    elif MODE == 1:
        return r * s + x * c
    elif MODE == 2:
        return tl.fma(x, c, r * s)
    else:
        return tl.fma(r, s, x * c)


@triton.jit
def _rope_fwd_kernel(qkv_ptr, f_ptr, fk_ptr, q_ptr, k_ptr, v_ptr, SB, B: tl.constexpr, G: tl.constexpr, QPG: tl.constexpr,
                     D: tl.constexpr, BR: tl.constexpr, FMA_ORDER: tl.constexpr, K_ORDER: tl.constexpr):
    # qkv [S*B, G, (QPG+2)*D]; f [S, D] fp32 angles; q [S*B, G*QPG, D], k/v [S*B, G, D]
    pr, g = tl.program_id(0), tl.program_id(1)
    HD: tl.constexpr = D // 2
    rows = pr * BR + tl.arange(0, BR)
    rm = rows < SB
    srow = rows // B
    d = tl.arange(0, HD)
    f_lo = tl.load(f_ptr + srow[:, None] * D + d[None, :], mask=rm[:, None], other=0.0)
    f_hi = tl.load(f_ptr + srow[:, None] * D + HD + d[None, :], mask=rm[:, None], other=0.0)
    c_lo, s_lo = tl.cos(f_lo), tl.sin(f_lo)
    c_hi, s_hi = tl.cos(f_hi), tl.sin(f_hi)
    W: tl.constexpr = (QPG + 2) * D
    base = qkv_ptr + rows[:, None, None].to(tl.int64) * (G * W) + g * W
    j = tl.arange(0, QPG)
    m3 = rm[:, None, None]
    x_lo = tl.load(base + j[None, :, None] * D + d[None, None, :], mask=m3, other=0.0).to(tl.float32)
    x_hi = tl.load(base + j[None, :, None] * D + HD + d[None, None, :], mask=m3, other=0.0).to(tl.float32)
    o_lo = _rot(x_lo, c_lo[:, None, :], -x_hi, s_lo[:, None, :], FMA_ORDER)
    o_hi = _rot(x_hi, c_hi[:, None, :], x_lo, s_hi[:, None, :], FMA_ORDER)
    qo = q_ptr + rows[:, None, None].to(tl.int64) * (G * QPG * D) + (g * QPG + j[None, :, None]) * D + d[None, None, :]
    tl.store(qo, o_lo.to(tl.bfloat16), mask=m3)
    tl.store(qo + HD, o_hi.to(tl.bfloat16), mask=m3)
    # k (its own angle table)
    fk_lo = tl.load(fk_ptr + srow[:, None] * D + d[None, :], mask=rm[:, None], other=0.0)
    fk_hi = tl.load(fk_ptr + srow[:, None] * D + HD + d[None, :], mask=rm[:, None], other=0.0)
    c_lo, s_lo = tl.cos(fk_lo), tl.sin(fk_lo)
    c_hi, s_hi = tl.cos(fk_hi), tl.sin(fk_hi)
    base2 = qkv_ptr + rows[:, None].to(tl.int64) * (G * W) + g * W
    kb = base2 + QPG * D
    k_lo = tl.load(kb + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(kb + HD + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    ko_lo = _rot(k_lo, c_lo, -k_hi, s_lo, K_ORDER)
    ko_hi = _rot(k_hi, c_hi, k_lo, s_hi, K_ORDER)
    ko = k_ptr + rows[:, None].to(tl.int64) * (G * D) + g * D + d[None, :]
    tl.store(ko, ko_lo.to(tl.bfloat16), mask=rm[:, None])
    tl.store(ko + HD, ko_hi.to(tl.bfloat16), mask=rm[:, None])
    # v: copy
    dd = tl.arange(0, D)
    vv = tl.load(base2 + (QPG + 1) * D + dd[None, :], mask=rm[:, None], other=0.0)
    tl.store(v_ptr + rows[:, None].to(tl.int64) * (G * D) + g * D + dd[None, :], vv, mask=rm[:, None])


@triton.jit
def _rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, f_ptr, fk_ptr, dqkv_ptr, SB, B: tl.constexpr, G: tl.constexpr, QPG: tl.constexpr,
                     D: tl.constexpr, BR: tl.constexpr, FMA_ORDER: tl.constexpr, K_ORDER: tl.constexpr):
    # inverse rotation: dx_lo = dy_lo*cos_lo + dy_hi*sin_hi ; dx_hi = dy_hi*cos_hi - dy_lo*sin_lo
    pr, g = tl.program_id(0), tl.program_id(1)
    HD: tl.constexpr = D // 2
    rows = pr * BR + tl.arange(0, BR)
    rm = rows < SB
    srow = rows // B
    d = tl.arange(0, HD)
    f_lo = tl.load(f_ptr + srow[:, None] * D + d[None, :], mask=rm[:, None], other=0.0)
    f_hi = tl.load(f_ptr + srow[:, None] * D + HD + d[None, :], mask=rm[:, None], other=0.0)
    c_lo, s_lo = tl.cos(f_lo), tl.sin(f_lo)
    c_hi, s_hi = tl.cos(f_hi), tl.sin(f_hi)
    W: tl.constexpr = (QPG + 2) * D
    ob = dqkv_ptr + rows[:, None, None].to(tl.int64) * (G * W) + g * W
    j = tl.arange(0, QPG)
    m3 = rm[:, None, None]
    qi = dq_ptr + rows[:, None, None].to(tl.int64) * (G * QPG * D) + (g * QPG + j[None, :, None]) * D + d[None, None, :]
    y_lo = tl.load(qi, mask=m3, other=0.0).to(tl.float32)
    y_hi = tl.load(qi + HD, mask=m3, other=0.0).to(tl.float32)
    x_lo = _rot(y_lo, c_lo[:, None, :], y_hi, s_hi[:, None, :], FMA_ORDER)
    x_hi = _rot(y_hi, c_hi[:, None, :], -y_lo, s_lo[:, None, :], FMA_ORDER)
    tl.store(ob + j[None, :, None] * D + d[None, None, :], x_lo.to(tl.bfloat16), mask=m3)
    tl.store(ob + j[None, :, None] * D + HD + d[None, None, :], x_hi.to(tl.bfloat16), mask=m3)
    fk_lo = tl.load(fk_ptr + srow[:, None] * D + d[None, :], mask=rm[:, None], other=0.0)
    fk_hi = tl.load(fk_ptr + srow[:, None] * D + HD + d[None, :], mask=rm[:, None], other=0.0)
    c_lo, s_lo = tl.cos(fk_lo), tl.sin(fk_lo)
    c_hi, s_hi = tl.cos(fk_hi), tl.sin(fk_hi)
    ki = dk_ptr + rows[:, None].to(tl.int64) * (G * D) + g * D + d[None, :]
    yk_lo = tl.load(ki, mask=rm[:, None], other=0.0).to(tl.float32)
    yk_hi = tl.load(ki + HD, mask=rm[:, None], other=0.0).to(tl.float32)
    xk_lo = _rot(yk_lo, c_lo, yk_hi, s_hi, K_ORDER)
    xk_hi = _rot(yk_hi, c_hi, -yk_lo, s_lo, K_ORDER)
    ob2 = dqkv_ptr + rows[:, None].to(tl.int64) * (G * W) + g * W
    kb = ob2 + QPG * D
    tl.store(kb + d[None, :], xk_lo.to(tl.bfloat16), mask=rm[:, None])
    tl.store(kb + HD + d[None, :], xk_hi.to(tl.bfloat16), mask=rm[:, None])
    dd = tl.arange(0, D)
    vv = tl.load(dv_ptr + rows[:, None].to(tl.int64) * (G * D) + g * D + dd[None, :], mask=rm[:, None], other=0.0)
    tl.store(ob2 + (QPG + 1) * D + dd[None, :], vv, mask=rm[:, None])


def _dims(qkv, split):
    S, B, G, W = qkv.shape
    D = split[1]
    return S, B, G, split[0] // D, D


def rope_fwd(qkv, freqs, kfreqs, split, br=4, warps=4, fma_order=2, k_order=2):
    S, B, G, QPG, D = _dims(qkv, split)
    # q, k, v live back to back in ONE buffer: TE's FP8 attention flattens + concatenates them before quantizing
    # (combine_and_quantize); with them adjacent, the patched combine skips that 84 MB copy.
    nq, nk = S * B * G * QPG * D, S * B * G * D
    buf = torch.empty(nq + 2 * nk, dtype=qkv.dtype, device=qkv.device)
    q = buf[:nq].view(S, B, G * QPG, D)
    k = buf[nq:nq + nk].view(S, B, G, D)
    v = buf[nq + nk:].view(S, B, G, D)
    _rope_fwd_kernel[(triton.cdiv(S * B, br), G)](qkv, freqs, kfreqs, q, k, v, S * B, B, G, QPG, D, br, fma_order, k_order, num_warps=warps)
    return q, k, v


def rope_bwd(dq, dk, dv, freqs, kfreqs, split, br=4, warps=4, fma_order=2, k_order=2):
    S, B, H, D = dq.shape
    G = dk.shape[2]
    QPG = H // G
    dqkv = torch.empty((S, B, G, (QPG + 2) * D), dtype=dq.dtype, device=dq.device)
    _rope_bwd_kernel[(triton.cdiv(S * B, br), G)](dq, dk, dv, freqs, kfreqs, dqkv, S * B, B, G, QPG, D, br, fma_order, k_order, num_warps=warps)
    return dqkv


def _adjacent(ts):
    """Contiguous tensors of one dtype that sit back to back in one storage."""
    st = ts[0].untyped_storage().data_ptr()
    p = ts[0].data_ptr()
    for t in ts:
        if not t.is_contiguous() or t.dtype != ts[0].dtype or t.untyped_storage().data_ptr() != st or t.data_ptr() != p:
            return False
        p += t.numel() * t.element_size()
    return True


def _flat(ts):
    return ts[0].as_strided((sum(t.numel() for t in ts),), (1,))


def _patch_combine():
    """Same as TE's combine_and_(de)quantize case 3 (separate q/k/v), minus the torch.cat when they are adjacent."""
    from transformer_engine.pytorch.attention.dot_product_attention import backends as BK
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
    if getattr(BK, "_ntu_orig_combine", None) is not None:
        return
    oq, od = BK.combine_and_quantize, BK.combine_and_dequantize
    BK._ntu_orig_combine = (oq, od)

    def _split(flat, ts):
        out, o = [], 0
        for t in ts:
            out.append(flat[o:o + t.numel()].view(t.shape)); o += t.numel()
        return out

    def combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer):
        if len(qkv_layout.replace("paged_kv_", "").split("_")) == 3 and type(q) is torch.Tensor and _adjacent((q, k, v)):
            qkv_fp8 = qkv_quantizer(_flat((q, k, v)))
            _STATS["quant_nocat"] += 1
            return tuple(Float8Tensor.make_like(qkv_fp8, data=x, dtype=q.dtype) for x in _split(qkv_fp8._data, (q, k, v)))
        _STATS["cat_fallback"] += 1
        return oq(qkv_layout, q, k, v, qkv_quantizer)

    def combine_and_dequantize(qkv_layout, q_fp8, k_fp8, v_fp8, src_nominal_dtype=None, des_nominal_dtype=None):
        ds = [getattr(x, "_data", None) for x in (q_fp8, k_fp8, v_fp8)]
        if (len(qkv_layout.replace("paged_kv_", "").split("_")) == 3 and all(isinstance(x, Float8Tensor) for x in (q_fp8, k_fp8, v_fp8))
                and all(d is not None for d in ds) and _adjacent(ds)):
            src = q_fp8.dtype
            des = des_nominal_dtype if des_nominal_dtype is not None else src
            qkv = Float8Tensor.make_like(q_fp8, data=_flat(ds), dtype=src).dequantize(dtype=des)
            _STATS["dequant_nocat"] += 1
            return tuple(_split(qkv, ds))
        _STATS["cat_fallback"] += 1
        return od(qkv_layout, q_fp8, k_fp8, v_fp8, src_nominal_dtype=src_nominal_dtype, des_nominal_dtype=des_nominal_dtype)

    BK.combine_and_quantize, BK.combine_and_dequantize = combine_and_quantize, combine_and_dequantize


def _ok(qkv, q_freqs, k_freqs, split, start_positions, tensor_format, interleaved, cp_size):
    if tensor_format != "sbhd" or interleaved or cp_size != 1 or start_positions is not None:
        return False
    if qkv.dtype != torch.bfloat16 or qkv.dim() != 4 or not qkv.is_contiguous() or len(split) != 3:
        return False
    D = split[1]
    if split[2] != D or split[0] % D or qkv.shape[-1] != sum(split) or D not in (64, 128):
        return False
    QPG = split[0] // D
    if QPG & (QPG - 1):
        return False
    return all(f.shape[-1] == D and f.numel() == qkv.shape[0] * D and f.is_contiguous() for f in (q_freqs, k_freqs))


def apply():
    from transformer_engine.pytorch.attention import rope as R
    F = R.FusedQKVRoPEFunc
    if getattr(F, "_ntu_orig", None) is not None:
        return
    F._ntu_orig = (F.forward, F.backward)
    orig_fwd, orig_bwd = F.forward, F.backward

    def forward(ctx, qkv, q_freqs, k_freqs, qkv_split_arg_list, start_positions=None, tensor_format="sbhd",
                interleaved=False, cp_size=1, cp_rank=0):
        qf = q_freqs if q_freqs.dtype == torch.float32 else q_freqs.float()
        kf = k_freqs if k_freqs.dtype == torch.float32 else k_freqs.float()
        if not _ok(qkv, qf, kf, list(qkv_split_arg_list), start_positions, tensor_format, interleaved, cp_size):
            _STATS["fallback"] += 1
            ctx.ntu = False
            return orig_fwd(ctx, qkv, q_freqs, k_freqs, qkv_split_arg_list, start_positions, tensor_format,
                            interleaved, cp_size, cp_rank)
        _STATS["fwd"] += 1
        ctx.ntu = True
        ctx.save_for_backward(qf, kf)
        ctx.split = list(qkv_split_arg_list)
        return rope_fwd(qkv, qf, kf, ctx.split)

    def backward(ctx, gq, gk, gv):
        if not getattr(ctx, "ntu", False):
            return orig_bwd(ctx, gq, gk, gv)
        qf, kf = ctx.saved_tensors
        _STATS["bwd"] += 1
        return rope_bwd(gq.contiguous(), gk.contiguous(), gv.contiguous(), qf, kf, ctx.split), None, None, None, None, None, None, None, None

    F.forward, F.backward = staticmethod(forward), staticmethod(backward)
    _patch_combine()
    print("[NTU_ROPE] FusedQKVRoPEFunc -> Triton RoPE (sbhd, non-interleaved); FP8 attention q/k/v combine without cat", flush=True)


def stats():
    return dict(_STATS)
