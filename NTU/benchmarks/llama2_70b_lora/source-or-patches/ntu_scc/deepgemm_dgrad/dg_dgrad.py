"""NTU-SCC patch: route TE's FP8 dgrad GEMM (dx = dy @ W) to DeepGEMM (DG_DGRAD=1).

Why: at the 300 W cap, DeepGEMM's promoted-accumulation FP8 GEMM sustains 483-494 TFLOPS on the four llama2-70B
TP=2 dgrad shapes vs cuBLAS split-accumulator 431-460 (gemmlib, 2026-10-01). Accumulation stays precise
(promotion every 128 K), unlike TE fast-accum dgrad which broke convergence.

What changes numerically: dy (TE: E5M2, per-tensor scale) is re-quantized to E4M3 with 1x128 per-token-group scales
(DeepSeek-V3 recipe; finer than TE's per-tensor). W^T is TE's own E4M3 per-tensor transpose, passed with a
broadcast 128x128 block-scale grid (bit-identical values, no requant).

FP8W_T_ONDEMAND=1 (memory): TE keeps a cached FP8 TRANSPOSE of every FP8-primary (fp8_model_init) weight, because Hopper
FP8 GEMMs need both operands K-major and dgrad uses W^T. For frozen LoRA base weights that is ~32 GiB/GPU at TP=2. With the
flag, the transpose of a frozen FP8 weight is dropped right after each fprop (TN) and dgrad (NN) GEMM that uses it; TE's
backward re-creates it via update_usage(columnwise_usage=True) just before the dgrad (tex.fp8_transpose), so at most one
layer's transposes are alive. That frees enough memory to train without activation recompute.

Only calls that look exactly like a plain dgrad (layout NN, grad=True, no accumulate/alpha/beta/bias/ub/quantized
output, bf16 out, FP8 storages) are rerouted; everything else goes to the original general_gemm.
"""
import os, sys, torch
import triton, triton.language as tl

_STATS = {"deepgemm": 0, "prequant": 0, "fallback": 0, "t_dropped": 0, "fwd_t_skipped": 0}
_T_ONDEMAND = os.environ.get("FP8W_T_ONDEMAND", "0") == "1"
# FP8W_T_NOFWD=1 (with FP8W_T_ONDEMAND): TE Linear/LayerNormLinear also BUILD the transpose of a frozen FP8 weight in the
# forward (get_weight_workspace -> update_usage(columnwise)), which ONDEMAND then drops right after the fprop GEMM and the
# backward rebuilds anyway. Skip building it in the forward (get_weight_workspace is only called from forward passes).
_T_NOFWD = _T_ONDEMAND and os.environ.get("FP8W_T_NOFWD", "0") == "1"
_ORIG = {}


@triton.jit
def _requant_e4m3_1x128(x_ptr, sinv_ptr, y_ptr, sf_ptr, M, N, BM: tl.constexpr, POW2: tl.constexpr):
    pm = tl.program_id(0); g = tl.program_id(1)
    r = pm * BM + tl.arange(0, BM); c = g * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + r[:, None] * N + c[None, :]).to(tl.float32) * tl.load(sinv_ptr)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    if POW2:   # power-of-two group scale: E5M2 values (2-bit mantissa) map EXACTLY into E4M3 (3-bit) -> no double rounding
        s = tl.exp2(tl.floor(tl.log2(448.0 / amax)))
    else:
        s = 448.0 / amax
    tl.store(y_ptr + r[:, None] * N + c[None, :], (x * s[:, None]).to(tl.float8e4nv))
    tl.store(sf_ptr + g * M + r, 1.0 / s)                 # [N/128, M] storage == column-major [M, N/128]


def requant(dy8, sinv):
    """dy8: [M, N] fp8 (e5m2 or e4m3) with per-tensor dequant scale sinv -> (E4M3 [M, N], fp32 sf column-major [M, N/128])."""
    M, N = dy8.shape
    y = torch.empty((M, N), device=dy8.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((N // 128, M), device=dy8.device, dtype=torch.float32)
    _requant_e4m3_1x128[(M // 64, N // 128)](dy8, sinv, y, sf, M, N, 64, os.environ.get("DG_POW2", "0") == "1")
    return y, sf.t()


def _as_fp8(raw, te_dtype):
    import transformer_engine_torch as tex
    if raw.dtype in (torch.float8_e4m3fn, torch.float8_e5m2): return raw
    return raw.view(torch.float8_e5m2 if te_dtype == tex.DType.kFloat8E5M2 else torch.float8_e4m3fn)


# FP8W_T_KEEP_GB=<x> (with ONDEMAND): keep the transposes of the first weights seen resident, up to x GiB, instead of
# rebuilding them every backward (uses memory freed elsewhere, e.g. by NTU_LORA_QKV_LOCAL).
_KEEP_BUDGET = float(os.environ.get("FP8W_T_KEEP_GB", "0")) * 2**30
_KEPT = set()


def _drop_weight_transpose(w):
    """Free the cached FP8 transpose of a FROZEN FP8 weight (A operand of TN fprop / NN dgrad)."""
    if (_T_ONDEMAND and getattr(w, "_transpose", None) is not None and not getattr(w, "_transpose_invalid", True)
            and getattr(w, "requires_grad", True) is False and hasattr(w, "update_usage")):
        if id(w) in _KEPT:
            return
        nb = w._transpose.numel() * w._transpose.element_size()
        if _STATS.get("t_kept_bytes", 0) + nb <= _KEEP_BUDGET:
            _KEPT.add(id(w))
            _STATS["t_kept_bytes"] = _STATS.get("t_kept_bytes", 0) + nb
            _STATS["t_kept"] = len(_KEPT)
            return
        w.update_usage(columnwise_usage=False)
        _STATS["t_dropped"] += 1


def _make_patched(orig):
    import deep_gemm
    from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage

    def patched(A, B, out_dtype=None, quantization_params=None, gelu=False, gelu_in=None, alpha=1.0, beta=None,
                accumulate=False, layout="TN", out=None, bias=None, use_split_accumulator=False, grad=False,
                ub=None, ub_type=None, extra_output=None, bulk_overlap=False):
        ok = (layout == "NN" and grad and quantization_params is None and not accumulate and not gelu and bias is None
              and ub is None and extra_output is None and not bulk_overlap and (alpha is None or alpha == 1.0)
              and (beta is None or beta == 0) and out_dtype in (None, torch.bfloat16)
              and isinstance(A, Float8TensorStorage) and isinstance(B, Float8TensorStorage)
              and getattr(A, "_transpose", None) is not None and getattr(B, "_data", None) is not None)
        if ok:
            wT = _as_fp8(A._transpose, A._fp8_dtype)                     # [K_in, N_out] E4M3
            dy = _as_fp8(B._data, B._fp8_dtype)
            dy2 = dy.reshape(-1, dy.shape[-1]); M, N = dy2.shape; K = wT.shape[0]
            ok = (wT.dtype == torch.float8_e4m3fn and wT.shape[1] == N and M % 64 == 0 and N % 128 == 0 and K % 128 == 0
                  and wT.is_contiguous() and dy2.is_contiguous() and (out is None or (out.dtype == torch.bfloat16 and out.is_contiguous())))
        if not ok:
            if getattr(B, "_ntu_sf", None) is not None:
                raise RuntimeError(f"[DG_DGRAD] a pre-quantized E4M3 1x128 operand reached a non-DeepGEMM GEMM (layout {layout})")
            _STATS["fallback"] += 1
            res = orig(A, B, out_dtype=out_dtype, quantization_params=quantization_params, gelu=gelu, gelu_in=gelu_in,
                       alpha=alpha, beta=beta, accumulate=accumulate, layout=layout, out=out, bias=bias,
                       use_split_accumulator=use_split_accumulator, grad=grad, ub=ub, ub_type=ub_type,
                       extra_output=extra_output, bulk_overlap=bulk_overlap)
            if layout in ("TN", "NN"):
                _drop_weight_transpose(A)
            return res
        sf = getattr(B, "_ntu_sf", None)
        if sf is not None:                                             # already E4M3 1x128 (ntu_dg_quant): no requant
            a8, sfa = dy2, sf
            _STATS["prequant"] += 1
        else:
            a8, sfa = requant(dy2, B._scale_inv.float())
        sfb = A._scale_inv.float().reshape(1, 1).expand(K // 128, N // 128).contiguous()
        dx = out.view(M, K) if out is not None else torch.empty((M, K), device=dy.device, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt((a8, sfa), (wT, sfb), dx, disable_ue8m0_cast=True)
        _STATS["deepgemm"] += 1
        del wT
        _drop_weight_transpose(A)
        if out is None and dy.dim() > 2: dx = dx.view(*dy.shape[:-1], K)
        return dx, None, None, None
    return patched


def apply():
    import importlib, ctypes
    sysroot = "/work/benchmark/MLPerf/gemmlib/sysroot/usr/lib/x86_64-linux-gnu"     # libdw/libelf for deep_gemm's .so
    for lib in ("libelf.so.1", "libdw.so.1"):
        try: ctypes.CDLL(f"{sysroot}/{lib}", mode=ctypes.RTLD_GLOBAL)
        except OSError as e: print(f"[DG_DGRAD] preload {lib} failed: {e}", flush=True)
    if "/work/benchmark/MLPerf/gemmlib/pylib_dg" not in sys.path: sys.path.insert(0, "/work/benchmark/MLPerf/gemmlib/pylib_dg")
    os.environ.setdefault("DG_JIT_CACHE_DIR", "/work/benchmark/MLPerf/gemmlib/dg_cache")
    mods = ["transformer_engine.pytorch.ops.basic.basic_linear", "transformer_engine.pytorch.module.linear",
            "transformer_engine.pytorch.module.layernorm_linear", "transformer_engine.pytorch.module.layernorm_mlp"]
    done = []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "general_gemm") and m not in _ORIG:
                _ORIG[m] = mod.general_gemm; mod.general_gemm = _make_patched(_ORIG[m]); done.append(m.rsplit(".", 1)[1])
        except Exception as e:
            print(f"[DG_DGRAD] skip {m}: {e}", flush=True)
    if _T_NOFWD:
        import transformer_engine.pytorch.module.base as B
        from transformer_engine.pytorch.tensor import QuantizedTensor
        TEB = B.TransformerEngineBaseModule
        if getattr(TEB, "_ntu_orig_gww", None) is None:
            orig_gww = TEB._ntu_orig_gww = TEB.get_weight_workspace

            def get_weight_workspace(self, *, tensor=None, quantizer=None, **kw):
                if (isinstance(tensor, QuantizedTensor) and tensor.requires_grad is False and quantizer is not None
                        and getattr(quantizer, "columnwise_usage", False) and hasattr(tensor, "_transpose")):
                    tensor.update_usage(rowwise_usage=True if quantizer.rowwise_usage else None)
                    _STATS["fwd_t_skipped"] += 1
                    return tensor
                return orig_gww(self, tensor=tensor, quantizer=quantizer, **kw)
            TEB.get_weight_workspace = get_weight_workspace
        done.append("no-fwd-transpose")
    print(f"[DG_DGRAD] dgrad GEMMs -> DeepGEMM in: {', '.join(done)}", flush=True)
    return done


def stats():
    r = dict(_STATS)
    if torch.cuda.is_available():
        r["max_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        r["max_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    return r
