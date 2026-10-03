"""Bipolar SC attention matmuls (Q·Kᵀ and softmax·V) — scmp_vit port.

Mirrors ``scmp_vit/model/sc_vit.py``'s ``_sc_qk`` / ``_sc_av``:

* Q·Kᵀ at ``sc_runtime`` ``qk_granularity`` (``per_row`` by default — each
  query row its own range, which is what makes per-row MP exact; ``per_head``
  is the legacy scope).
* attn·V at ``per_row``, one batched call for all (B, H).
* Both run at the run's uniform stream length (the same one the linears use),
  or dispatch per row through the calibrated table when ``qk`` / ``av`` are MP
  ops. The MP metric is pooled GLOBALLY over B*H*N rows, as the calibrator
  records it.

The block index comes from ``scmp_kernels.mp.set_current_block_idx`` (set by
IRASim's Attention before the calls), so the public signatures stay what the
naive-int8 baseline patches over.
"""
from __future__ import annotations

import torch

from scmp_kernels import sc_matmul
from scmp_kernels import trace as sc_trace
from scmp_kernels.mp import (
    adaptive_classify_rows,
    compute_row_metric,
    get_current_block_idx,
    AutoMPBudgetLogger,
)

from . import sc_runtime as rt


def _mm3(a3, b3, granularity, sl, sc_prec):
    return sc_matmul(a3, b3, granularity=granularity, mode="bipolar",
                     sc_prec=sc_prec, group_a=1, group_b=1, stoc_len=int(sl),
                     halve_bipolar_stoc_len=rt.state()["halve"])


def _padded_level_gather(level_of_row, sl, BH, N):
    mask = (level_of_row == sl).reshape(BH, N)
    counts = mask.sum(dim=1)
    M = int(counts.max().item())
    if M == 0:
        return None, None, 0
    order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)
    gather_idx = order[:, :M].contiguous()
    ar = torch.arange(M, device=level_of_row.device).unsqueeze(0)
    valid = ar < counts.unsqueeze(1)
    first = gather_idx[:, :1].expand(-1, M)
    gather_idx = torch.where(valid, gather_idx, first)
    return gather_idx, valid, M


def _batched_level_matmul(a3, b3, level_of_row, levels, *, sc_prec, out_dim):
    """Per-row MP over a 3D batch, one batched per_row call per rung."""
    BH, N, K = a3.shape
    out = torch.zeros(BH, N, out_dim, dtype=torch.float32, device=a3.device)
    row_cycles = 0.0
    slice_base = torch.arange(BH, device=a3.device).unsqueeze(1) * N
    for sl in levels:
        sl = int(sl)
        if sl <= 0:
            continue
        gather_idx, valid, M = _padded_level_gather(level_of_row, sl, BH, N)
        if gather_idx is None:
            continue
        a_sel = a3.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, K))
        out_sel = _mm3(a_sel.contiguous(), b3, "per_row", sl, sc_prec)
        dst = (slice_base + gather_idx).reshape(-1)
        vflat = valid.reshape(-1)
        out.reshape(BH * N, out_dim)[dst[vflat]] = out_sel.reshape(BH * M, out_dim)[vflat]
        row_cycles += float(int(valid.sum().item()) * sl)
    return out, row_cycles


def _mp_rows(op, a3, b3, cfg, sc_prec, out_dim):
    BH, N, K = a3.shape
    blk = get_current_block_idx()
    tb = rt.state()["total_blocks"]
    name, sign = cfg.get_dispatch_metric(op)
    metric = rt.maybe_shuffle(compute_row_metric(a3.reshape(BH * N, K), name) * sign)
    assignment = adaptive_classify_rows(metric, cfg, operator=op, block_idx=blk,
                                        total_blocks=tb, timestep=0, total_timesteps=1)
    levels = cfg.classify_level_values(operator=op, block_idx=blk, total_blocks=tb,
                                       timestep=0, total_timesteps=1)
    lv = torch.zeros(BH * N, dtype=torch.long, device=a3.device)
    for sl, rows in assignment.level_row_indices.items():
        if rows.numel():
            lv[rows] = int(sl)
    out, row_cycles = _batched_level_matmul(a3, b3, lv, levels, sc_prec=sc_prec,
                                            out_dim=out_dim)
    max_sl = max(int(s) for s in levels)
    AutoMPBudgetLogger.record(blk, op, float(BH * N) * out_dim * K * max_sl,
                              row_cycles * out_dim * K)
    rt.record_cost(op, float(BH * N) * out_dim * K, row_cycles / max(BH * N, 1))
    return out


def _sc_attn_op(op, a3, b3, granularity, sc_prec, stoc_len):
    """``a3 @ b3.T`` for one attention op under the run's SC policy."""
    BH, N, K = a3.shape
    out_dim = b3.shape[1]
    blk = get_current_block_idx()
    for hook in rt.attn_hooks():
        hook(op, blk, a3, b3)
    if not rt.sc_enabled():
        return torch.matmul(a3, b3.transpose(-2, -1))
    if sc_trace._ENABLED:
        sc_trace.set_context(op, blk, None)
    gmap = rt.group_stoclen()
    cfg = rt.mp_cfg_for(op)
    if gmap is not None:
        sl = int(gmap.get((op, blk), rt.resolve_stoc_len(stoc_len, sc_prec)))
        rt.record_cost(op, float(BH * N) * out_dim * K, max(sl, 0))
        if sl <= 0:
            return torch.zeros(BH, N, out_dim, device=a3.device)
        return _mm3(a3, b3, granularity, sl, sc_prec)
    if cfg is not None:
        if granularity != "per_row":
            raise ValueError(f"{op} MP needs per_row granularity (got {granularity}); "
                             f"calibrate and deploy qk at per_row")
        return _mp_rows(op, a3, b3, cfg, sc_prec, out_dim)
    sl = rt.resolve_stoc_len(stoc_len, sc_prec)
    rt.record_cost(op, float(BH * N) * out_dim * K, sl)
    return _mm3(a3, b3, granularity, sl, sc_prec)


def sc_qk_matmul(q_scaled: torch.Tensor, k: torch.Tensor,
                 sc_prec: int | None = None, stoc_len: int | None = None) -> torch.Tensor:
    """SC Q·Kᵀ. ``q_scaled``/``k``: (B, H, N, D) -> (B, H, N, N) logits.

    Pre-scaling Q by 1/sqrt(D) does not change any SC result: per_row and
    per_head ranges scale with it, so the quantized integers are identical.
    """
    B, H, N, D = q_scaled.shape
    sc_prec = rt.state()["sc_prec"] if sc_prec is None else int(sc_prec)
    qf = q_scaled.float().reshape(B * H, N, D).contiguous()
    kf = k.float().reshape(B * H, N, D).contiguous()
    out = _sc_attn_op("qk", qf, kf, rt.state()["qk_granularity"], sc_prec, stoc_len)
    return out.reshape(B, H, N, N).to(q_scaled.dtype)


def sc_av_matmul(attn: torch.Tensor, v: torch.Tensor,
                 sc_prec: int | None = None, stoc_len: int | None = None) -> torch.Tensor:
    """SC softmax·V. ``attn``: (B, H, N, Nk), ``v``: (B, H, Nk, D) -> (B, H, N, D)."""
    B, H, N, Nk = attn.shape
    D = v.shape[-1]
    sc_prec = rt.state()["sc_prec"] if sc_prec is None else int(sc_prec)
    a3 = attn.float().reshape(B * H, N, Nk).contiguous()
    v3 = v.float().transpose(-1, -2).reshape(B * H, D, Nk).contiguous()
    out = _sc_attn_op("av", a3, v3, "per_row", sc_prec, stoc_len)
    return out.reshape(B, H, N, D).to(attn.dtype)
