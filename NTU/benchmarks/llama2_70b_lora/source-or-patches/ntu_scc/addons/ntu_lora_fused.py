"""Fused LoRA output path for Megatron-Bridge's LoRALinear + ParallelLinearAdapter (TP>1, sequence parallel, a2a).

Replaces, per adapter, `base + scale * linear_out(h)` (+ the 64 MB hidden->sequence all-to-all for linear_proj) with one
cuBLAS GEMM that folds the scale and the add (addmm, beta=1, alpha=scale):

  linear_qkv  (column base, gathered input):  out = addmm(base, h, B_shard^T)                 dgrad all-reduce of dh (as before)
  linear_proj (row base, a2a_experimental):   out = addmm(base, h[my seq half], B_full^T)     B_full = all_gather(B_shard), 128 KB
                                             (no all-to-all; backward: all_gather(dh rows) + reduce_scatter(dB))

Dropout, linear_in and its communication are untouched (same RNG use). Weight grads go into weight.main_grad in fp32
exactly like Megatron's gradient_accumulation_fusion path. Enable with NTU_FUSED_LORA=1 (via ntu_addons).

NTU_LORA_QKV_LOCAL=1 (on top): linear_qkv's LoRA runs on this rank's SEQUENCE SHARD instead of the gathered LN output.
  Stock: TE LayerNormLinear(return_layernorm_output_gathered=True) all-gathers the LN output in bf16 (LoRA needs it),
  then quantizes the gathered copy for the GEMM; and TE's backward never reads the gradient of that gathered output, so
  the LoRA dx GEMM + its all-reduce + the dropout backward are computed and thrown away.
  Here: TE returns the local bf16 LN output and all-gathers in FP8 (bit-identical GEMM input: same scale, elementwise
  cast). LoRA: h_loc = drop(x_loc) @ A_full^T (A_full = all_gather(A_shard), 256 KB), h = all_gather(h_loc) (256 KB),
  out = base + scale * h @ B_shard^T. Backward: dB as before, dh_loc = reduce_scatter(scale * g @ B_shard),
  dA_shard = reduce_scatter(dh_loc^T @ x_loc); no dx (same as stock). Dropout runs on the local shard under the
  tensor-parallel RNG tracker (independent masks per rank; same distribution as stock)."""
import os
import torch
import torch.distributed as dist

_STATS = {"qkv": 0, "qkv_local": 0, "proj": 0, "fallback": 0}
_QKV_LOCAL = os.environ.get("NTU_LORA_QKV_LOCAL", "0") == "1"


def _tp_group():
    from megatron.core import parallel_state as ps
    return ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size(), ps.get_tensor_model_parallel_rank()


def _accumulate_wgrad(weight, dw_fp32, dtype):
    """Mirror Megatron's LinearWithGradAccumulationAndAsyncCommunication wgrad handling."""
    if hasattr(weight, "main_grad"):
        weight.main_grad.add_(dw_fp32)
        if hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True
            if getattr(weight, "zero_out_wgrad", False):
                return torch.zeros(weight.main_grad.shape, dtype=dtype, device=weight.device)
            return torch.empty(weight.main_grad.shape, dtype=dtype, device=weight.device)
        return None
    return dw_fp32.to(dtype)


def _mm_fp32(a, b):
    """a @ b with bf16 inputs and fp32 output (cuBLAS fp32 accumulate, no bf16 rounding of the result)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a.float(), b.float())


class _QKVOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, h, w, scale, group):
        ctx.save_for_backward(h, w)
        ctx.scale, ctx.group = scale, group
        out = torch.addmm(base.reshape(-1, base.shape[-1]), h.reshape(-1, h.shape[-1]), w.t(), beta=1.0, alpha=scale)
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, g):
        h, w = ctx.saved_tensors
        s = ctx.scale
        g2, h2 = g.reshape(-1, g.shape[-1]), h.reshape(-1, h.shape[-1])
        dh = torch.mm(g2, w).mul_(s)                          # partial over the TP-sharded output dim
        work = dist.all_reduce(dh, group=ctx.group, async_op=True)
        dw = _mm_fp32(g2.t(), h2).mul_(s)                     # overlaps the all-reduce
        gw = _accumulate_wgrad(w, dw, w.dtype)
        work.wait()
        return g, dh.view(h.shape), gw, None, None


class _QKVLocal(torch.autograd.Function):
    """out = base + scale * all_gather_seq(x_loc @ all_gather(A_shard)^T) @ B_shard^T ; x_loc carries no gradient."""
    @staticmethod
    def forward(ctx, base, x_loc, a, w, scale, group, tp):
        x2 = x_loc.reshape(-1, x_loc.shape[-1])
        a_full = torch.empty((a.shape[0] * tp, a.shape[1]), dtype=a.dtype, device=a.device)
        dist.all_gather_into_tensor(a_full, a.contiguous(), group=group)          # rank order = linear_in column order
        h_loc = torch.mm(x2, a_full.t())
        h = torch.empty((h_loc.shape[0] * tp, h_loc.shape[1]), dtype=h_loc.dtype, device=h_loc.device)
        dist.all_gather_into_tensor(h, h_loc, group=group)                         # sequence order
        ctx.save_for_backward(x2, h, a, w)
        ctx.scale, ctx.group, ctx.tp = scale, group, tp
        out = torch.addmm(base.reshape(-1, base.shape[-1]), h, w.t(), beta=1.0, alpha=scale)
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, g):
        x2, h, a, w = ctx.saved_tensors
        s, group, tp = ctx.scale, ctx.group, ctx.tp
        g2 = g.reshape(-1, g.shape[-1])
        dh = torch.mm(g2, w).mul_(s)                                               # partial over TP output shards
        dh_loc = torch.empty((dh.shape[0] // tp, dh.shape[1]), dtype=dh.dtype, device=dh.device)
        work = dist.reduce_scatter_tensor(dh_loc, dh, group=group, async_op=True)
        dw = _mm_fp32(g2.t(), h).mul_(s)                                           # overlaps the reduce-scatter
        gw = _accumulate_wgrad(w, dw, w.dtype)
        work.wait()
        daf = _mm_fp32(dh_loc.t(), x2)                                             # [r, H], partial over sequence shards
        das = torch.empty((a.shape[0], a.shape[1]), dtype=torch.float32, device=a.device)
        dist.reduce_scatter_tensor(das, daf, group=group)
        ga = _accumulate_wgrad(a, das, a.dtype)
        return g, None, ga, gw, None, None, None


class _ProjOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, h, w, scale, group, tp, rank):
        S = h.shape[0]
        hs = h[rank * (S // tp):(rank + 1) * (S // tp)].contiguous()          # my sequence half, [S/tp, b, r]
        w_full = torch.empty((w.shape[0] * tp, w.shape[1]), dtype=w.dtype, device=w.device)
        dist.all_gather_into_tensor(w_full, w.contiguous(), group=group)       # B_full in rank (= column) order
        ctx.save_for_backward(hs, w_full, w)
        ctx.scale, ctx.group, ctx.tp, ctx.hshape = scale, group, tp, h.shape
        out = torch.addmm(base.reshape(-1, base.shape[-1]), hs.reshape(-1, hs.shape[-1]), w_full.t(), beta=1.0, alpha=scale)
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, g):
        hs, w_full, w = ctx.saved_tensors
        s, group, tp = ctx.scale, ctx.group, ctx.tp
        g2, hs2 = g.reshape(-1, g.shape[-1]), hs.reshape(-1, hs.shape[-1])
        dhs = torch.mm(g2, w_full).mul_(s)                                    # rows of my sequence half, complete
        dh = torch.empty(ctx.hshape, dtype=dhs.dtype, device=dhs.device)
        w1 = dist.all_gather_into_tensor(dh, dhs.view(hs.shape), group=group, async_op=True)
        dwf = _mm_fp32(g2.t(), hs2).mul_(s)                                   # [H, r] partial over sequence halves
        dws = torch.empty((w.shape[0], w.shape[1]), dtype=torch.float32, device=w.device)
        w2 = dist.reduce_scatter_tensor(dws, dwf, group=group, async_op=True)
        w1.wait(); w2.wait()
        gw = _accumulate_wgrad(w, dws, w.dtype)
        return g, dh, gw, None, None, None, None


def _mark_ln_grad_unused(ln):
    """The local LN output only feeds LoRA under no_grad, so its gradient is identically zero. Tell TE's autograd node:
    don't materialize that zero tensor, and (via the patched backward) don't add it into dgrad. Bit-identical."""
    node = getattr(ln, "grad_fn", None)
    for _ in range(3):
        if node is None:
            return
        if type(node).__name__.endswith("_LayerNormLinearBackward"):
            node._ntu_ln_grad_unused = True
            node.set_materialize_grads(False)
            _STATS["ln_grad_skipped"] = _STATS.get("ln_grad_skipped", 0) + 1
            return
        nf = getattr(node, "next_functions", ())
        node = nf[0][0] if nf else None


def _patch_lnlinear_backward():
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
    if getattr(_LayerNormLinear, "_ntu_orig_bwd", None) is not None:
        return
    orig = _LayerNormLinear._ntu_orig_bwd = _LayerNormLinear.backward

    def backward(ctx, *grad_outputs):
        if getattr(ctx, "_ntu_ln_grad_unused", False) and len(grad_outputs) > 1 and grad_outputs[1] is None:
            ctx.return_layernorm_output = False                 # skips `dgrad + grad_outputs[1]` (a zero tensor)
            try:
                return orig(ctx, *grad_outputs)
            finally:
                ctx.return_layernorm_output = True
        return orig(ctx, *grad_outputs)

    _LayerNormLinear.backward = staticmethod(backward)


def _mode(adapter):
    from megatron.bridge.peft.utils import ParallelLinearAdapter
    if not isinstance(adapter, ParallelLinearAdapter) or adapter.is_expert or adapter.dropout_position != "pre":
        return None
    if not isinstance(adapter.activation, torch.nn.Identity) or getattr(adapter.linear_out, "bias", None) is not None:
        return None
    if getattr(adapter.config, "cpu_offloading", False) or adapter.linear_out.gather_output:
        return None
    _, tp, _ = _tp_group()
    if tp == 1:
        return None
    if adapter.input_is_parallel and adapter.use_a2a and not adapter.disable_sequence_parallel_comm:
        return "proj"
    if not adapter.input_is_parallel and adapter.disable_sequence_parallel_comm:
        return "qkv"
    return None


_ORIG_FORWARD = None


def _forward(self, x, *args, **kwargs):
    ad = self.adapter
    mode = _mode(ad) if self._adapter_enabled else None
    if mode is None:
        _STATS["fallback"] += 1
        return _ORIG_FORWARD(self, x, *args, **kwargs)
    group, tp, rank = _tp_group()
    if mode == "qkv" and _QKV_LOCAL:
        te_mod = self.to_wrap
        if getattr(te_mod, "return_layernorm_output_gathered", False):
            te_mod.return_layernorm_output_gathered = False      # TE now returns the local LN output, gathers in FP8
        linear_output, bias, ln_loc = self.base_linear_forward(x, *args, **kwargs)
        if ln_loc.shape[0] * tp != linear_output.shape[0]:
            raise RuntimeError(f"NTU_LORA_QKV_LOCAL: unexpected LN output {tuple(ln_loc.shape)} vs {tuple(linear_output.shape)}")
        _mark_ln_grad_unused(ln_loc)
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        with torch.no_grad(), get_cuda_rng_tracker().fork():
            x_loc = ad.dropout(ln_loc.contiguous())
        _STATS["qkv_local"] += 1
        return _QKVLocal.apply(linear_output, x_loc, ad.linear_in.weight, ad.linear_out.weight, ad.alpha / ad.dim,
                               group, tp), bias
    linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
    xin = ad.dropout(layernorm_output.contiguous())
    h, _ = ad.linear_in(xin)
    scale = ad.alpha / ad.dim
    _STATS[mode] += 1
    if mode == "qkv":
        out = _QKVOut.apply(linear_output, h, ad.linear_out.weight, scale, group)
    else:
        out = _ProjOut.apply(linear_output, h, ad.linear_out.weight, scale, group, tp, rank)
    return out, bias


def apply():
    global _ORIG_FORWARD
    from megatron.bridge.peft import lora_layers
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = lora_layers.LoRALinear.forward
        lora_layers.LoRALinear.forward = _forward
        if _QKV_LOCAL:
            _patch_lnlinear_backward()
    print("[NTU_FUSED_LORA] LoRALinear.forward -> fused output path (qkv: addmm; proj: addmm + B all-gather, no a2a)"
          + ("; qkv LoRA on the local sequence shard, FP8 LN all-gather" if _QKV_LOCAL else ""), flush=True)


def stats():
    return dict(_STATS)
