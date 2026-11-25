"""NTU_ATTN_BWD=1: hand-written Hopper FP8 flash-attention backward (attnkernel/fp8) in place of cuDNN's, for the
training's attention shape (sbhd, causal, hdim 128, GQA, FP8 delayed scaling, no dropout).

Hooks transformer_engine.pytorch.attention.dot_product_attention.backends.fused_attn_bwd. For the FP8 path TE hands over
Float8Tensors q/k/v/o/dO (data + per-tensor scale_inv) and the S/dP/dQKV quantizers; the kernel takes the quantizer
scales directly, quantizes P and dS exactly where cuDNN does (E4M3, per-tensor), writes amax_dP and amax_dQKV into the
quantizers (delayed scaling bookkeeping), and returns dq/dk/dv in bf16 -- the dtype TE would have dequantized to anyway
(the training's q/k/v are bf16), so TE's dequantize kernels are skipped too. Anything else falls back to TE."""
import ctypes
import os
import torch

_LIB_PATH = os.environ.get("NTU_ATTN_BWD_LIB", "/work/benchmark/MLPerf/attnkernel/fp8/libfa3_bwd_fp8.so")
_DQ32 = os.environ.get("NTU_ATTN_BWD_DQ32", "0") == "1"   # the .so was built with -DFP8_DQ_FP32 (fp32 dQ partial sums)
_STATS = {"ntu": 0, "fallback": 0}
_lib = None
_TRANSPOSE = None
_P = lambda t: ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def _load():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.fa3_bwd_fp8_hdim128.restype = ctypes.c_int
        _lib.fa3_transpose_sbhd_to_bhds.restype = ctypes.c_int
        global _TRANSPOSE
        _TRANSPOSE = _lib.fa3_transpose_sbhd_to_bhds
        if hasattr(_lib, "fa3_bwd_fp8_perm16") and _lib.fa3_bwd_fp8_perm16() == 1:   # kernel wants perm16-ordered Q^T/dO^T/K^T
            _TRANSPOSE = _lib.fa3_transpose_sbhd_to_bhds_p16
        _TRANSPOSE.restype = ctypes.c_int
    return _lib


def _shape(t):
    return tuple(t._data.shape)


def _ok(q, k, v, o, d_o, aux, qkv_layout, attn_bias_type, attn_mask_type, softmax_type, window_size, dropout):
    from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
    if not all(isinstance(t, Float8TensorStorage) and getattr(t, "_data", None) is not None and getattr(t, "_scale_inv", None) is not None
               for t in (q, k, v, o, d_o)):
        return False
    if qkv_layout != "sbhd_sbhd_sbhd" or attn_bias_type != "no_bias" or softmax_type != "vanilla" or dropout != 0.0:
        return False
    if attn_mask_type not in ("causal", "padding_causal") or tuple(window_size) not in ((-1, -1), (-1, 0)):
        return False
    if len(_shape(q)) != 4 or len(_shape(k)) != 4:
        return False
    S, B, H, D = _shape(q)
    Sk, Bk, Hk, Dk = _shape(k)
    if D != 128 or Dk != 128 or Sk != S or Bk != B or H % Hk != 0 or S % 128 != 0:
        return False
    if _shape(v) != (Sk, Bk, Hk, Dk) or _shape(o) != (S, B, H, D) or _shape(d_o) != (S, B, H, D):
        return False
    if not (len(aux) >= 1 and aux[0].dtype == torch.float32 and aux[0].numel() == B * H * S):
        return False
    return all(t._data.is_contiguous() for t in (q, k, v, o, d_o))


def _bwd(max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, q, k, v, o, d_o, fake_dtype, dqkv_dtype, aux_ctx_tensors,
         fused_attention_backend, cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None, s_quantizer=None, dp_quantizer=None,
         dqkv_quantizer=None, attn_scale=None, dropout=0.0, fast_zero_fill=True, qkv_layout="sbh3d", attn_bias_type="no_bias",
         attn_mask_type="padding", softmax_type="vanilla", window_size=(-1, -1), bottom_right_diagonal=None, deterministic=False,
         cuda_graph=False):
    if not _ok(q, k, v, o, d_o, aux_ctx_tensors, qkv_layout, attn_bias_type, attn_mask_type, softmax_type, window_size, dropout) \
            or s_quantizer is None or dp_quantizer is None or dqkv_quantizer is None:
        _STATS["fallback"] += 1
        if os.environ.get("NTU_ATTN_BWD_DEBUG", "0") == "1" and _STATS["fallback"] <= 2:
            def sh(t):
                try:
                    d = getattr(t, "_data", None)
                    return (type(t).__name__, tuple(t.shape), None if d is None else bool(d.is_contiguous()))
                except Exception as e:
                    return (type(t).__name__, repr(e)[:40])
            try:
                print(f"[NTU_ATTN_BWD] fallback: q {sh(q)} k {sh(k)} v {sh(v)} o {sh(o)} d_o {sh(d_o)} aux0 {tuple(aux_ctx_tensors[0].shape)} {aux_ctx_tensors[0].dtype} "
                      f"layout {qkv_layout} bias {attn_bias_type} mask {attn_mask_type} softmax {softmax_type} window {window_size} dropout {dropout} "
                      f"quantizers {s_quantizer is not None} {dp_quantizer is not None} {dqkv_quantizer is not None}", flush=True)
            except Exception as e:
                print(f"[NTU_ATTN_BWD] fallback (debug print failed: {e!r})", flush=True)
        return _ORIG(max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, q, k, v, o, d_o, fake_dtype, dqkv_dtype, aux_ctx_tensors,
                     fused_attention_backend, cu_seqlens_q_padded, cu_seqlens_kv_padded, s_quantizer, dp_quantizer, dqkv_quantizer,
                     attn_scale, dropout, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type, softmax_type, window_size,
                     bottom_right_diagonal, deterministic, cuda_graph)
    lib = _load()
    S, B, H, D = _shape(q)
    Hk = _shape(k)[2]
    dev = q._data.device
    if attn_scale is None:
        attn_scale = 1.0 / (D ** 0.5)
    # pre-transposed operands (FP8 bytes): [B, H, D, S], via the .so's tiled transpose
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    qt = torch.empty(B, H, D, S, device=dev, dtype=torch.uint8)
    dot = torch.empty(B, H, D, S, device=dev, dtype=torch.uint8)
    kt = torch.empty(B, Hk, D, S, device=dev, dtype=torch.uint8)
    for src, dst, hh in ((q._data, qt, H), (d_o._data, dot, H), (k._data, kt, Hk)):
        rc = _TRANSPOSE(_P(src), _P(dst), S, B, hh, D, stream)
        if rc != 0:
            raise RuntimeError(f"[NTU_ATTN_BWD] transpose failed: {rc}")
    lse = aux_ctx_tensors[0]
    f32 = torch.float32
    sqr, skr = (S + 63) // 64 * 64, (S + 127) // 128 * 128
    dq_accum = torch.empty(B * H * sqr * 128, device=dev, dtype=torch.float32 if _DQ32 else torch.float16)
    dk_accum = torch.empty(B * Hk * skr * 128, device=dev, dtype=f32)
    dv_accum = torch.empty(B * Hk * skr * 128, device=dev, dtype=f32)
    softmax_d = torch.empty(B * H * sqr, device=dev, dtype=f32)
    lse_log2 = torch.empty(B * H * sqr, device=dev, dtype=f32)
    sem = torch.zeros(16, device=dev, dtype=torch.int32)        # tile-count semaphore; dq/dk/dv semaphores null (non-deterministic)
    inv_scale_dqkv = torch.reciprocal(dqkv_quantizer.scale)
    dq = torch.empty(S, B, H, D, device=dev, dtype=fake_dtype)
    dk = torch.empty(S, B, Hk, D, device=dev, dtype=fake_dtype)
    dv = torch.empty(S, B, Hk, D, device=dev, dtype=fake_dtype)
    num_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    import transformer_engine_torch as tex
    grad_e5m2 = 1 if d_o._fp8_dtype == tex.DType.kFloat8E5M2 else 0
    if dp_quantizer.dtype != d_o._fp8_dtype:
        raise RuntimeError(f"[NTU_ATTN_BWD] dP quantizer dtype {dp_quantizer.dtype} != dO dtype {d_o._fp8_dtype}")
    # fp16 dQ partial sums are pre-scaled by the dQKV output scale (fp8_max/amax); normalize to a 4096 target so partials fit fp16
    dq_prescale = 4096.0 / (57344.0 if dqkv_quantizer.dtype == tex.DType.kFloat8E5M2 else 448.0)
    rc = lib.fa3_bwd_fp8_hdim128(
        _P(q._data), _P(k._data), _P(v._data), _P(o._data), _P(d_o._data), _P(qt), _P(dot), _P(kt), _P(lse),
        _P(q._scale_inv), _P(k._scale_inv), _P(v._scale_inv), _P(o._scale_inv), _P(d_o._scale_inv),
        _P(s_quantizer.scale), _P(dp_quantizer.scale), _P(dqkv_quantizer.scale), _P(inv_scale_dqkv),
        _P(dp_quantizer.amax), _P(dqkv_quantizer.amax), 1, grad_e5m2, ctypes.c_float(dq_prescale),
        _P(dq), _P(dk), _P(dv), _P(dq_accum), _P(dk_accum), _P(dv_accum), _P(softmax_d), _P(lse_log2),
        _P(None), _P(None), _P(None), _P(sem),
        B, S, S, H, Hk, D, ctypes.c_float(attn_scale), num_sm, stream)
    if rc != 0:
        raise RuntimeError(f"[NTU_ATTN_BWD] kernel failed: cuda error {rc}")
    _STATS["ntu"] += 1
    return [dq, dk, dv]


_ORIG = None


def apply():
    global _ORIG
    from transformer_engine.pytorch.attention.dot_product_attention import backends as BK
    if _ORIG is None:
        _ORIG = BK.fused_attn_bwd
        BK.fused_attn_bwd = _bwd
    try:   # fingerprint the library actually loaded (NFS can serve a stale view right after a rebuild)
        import hashlib, time
        st = os.stat(_LIB_PATH); lib = _load()
        p16 = lib.fa3_bwd_fp8_perm16() if hasattr(lib, "fa3_bwd_fp8_perm16") else -1
        fp = hashlib.md5(open(_LIB_PATH, "rb").read()).hexdigest()[:8]
        info = f"size {st.st_size} mtime {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))} md5 {fp} perm16 {p16}"
    except Exception as e:
        info = f"fingerprint failed: {e!r}"
    print(f"[NTU_ATTN_BWD] FP8 attention backward -> {_LIB_PATH} ({info}; TE fused_attn_bwd hooked; bf16 grads, amax_dP/dQKV written)", flush=True)


def stats():
    return dict(_STATS)
