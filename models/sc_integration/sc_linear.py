"""Bipolar SC replacement for nn.Linear forward — scmp_vit ``SCLinear`` port.

Same datapath as ``scmp_vit/model/sc_vit.py``'s SCLinear, in functional form
so IRASim's Attention / SCMlp keep calling ``sc_linear_forward``:

* ``granularity="per_row"`` with ``chunk_d`` (128) on every linear.
* Uniform: one call at ``sc_runtime.resolve_stoc_len()``.
* MP (``SC_MP_TABLE``): protected input channels split off along the
  contraction dim at the table's protected stream length, residual rows
  dispatched by the table's per-op metric/thresholds (``AdaptiveMPConfig``),
  one kernel call per rung; or per-(row, chunk) rungs when the table has a
  ``per_row_chunk`` section.

IRASim-specific (kept from the pre-alignment integration): SmoothQuant scales
attached as ``linear._sc_smooth_scales`` are applied as an explicit
pre-transform ``(x/s, w*s)`` BEFORE dispatch, so the MP metric, the protected
channels and the calibration sigmas all see the operands the kernel quantizes.
This is exactly what ``sc_matmul(smooth_scales=...)`` does internally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scmp_kernels import sc_matmul
from scmp_kernels import trace as sc_trace
from scmp_kernels.mp import (
    adaptive_classify_rows,
    compute_row_metric,
    AutoMPBudgetLogger,
)

from . import sc_runtime as rt

_CHUNK_WARNED: set = set()


def _mm(x2, w, stoc_len, sc_prec):
    return sc_matmul(
        x2, w, granularity="per_row", mode="bipolar", sc_prec=sc_prec,
        stoc_len=int(stoc_len), group_a=1, group_b=1,
        chunk_d=rt.chunk_d_for(x2.shape[-1]),
        halve_bipolar_stoc_len=rt.state()["halve"])


def _channel_index_tensor(indices, width, device):
    vals = sorted({int(i) for i in (indices or []) if 0 <= int(i) < width})
    if not vals:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _complement(width, selected, device):
    if selected.numel() == 0:
        return torch.arange(width, dtype=torch.long, device=device)
    mask = torch.ones(width, dtype=torch.bool, device=device)
    mask[selected] = False
    return mask.nonzero(as_tuple=True)[0]


def per_row_chunk_rungs(x, chunk_d, levels, thresholds):
    """(N, n_chunks) rung table from the per-(row, chunk) absmax, min-max
    normalized over all pairs of the call (scmp_vit ``per_row_chunk_rungs``)."""
    N, D = x.shape
    n_chunks = (D + chunk_d - 1) // chunk_d
    if N == 0:
        return torch.zeros((0, n_chunks), dtype=torch.int32, device=x.device)
    pad = n_chunks * chunk_d - D
    xa = x.abs()
    if pad:
        xa = F.pad(xa, (0, pad))
    m = xa.view(N, n_chunks, chunk_d).amax(-1)
    lo, hi = m.min(), m.max()
    mn = (m - lo) / (hi - lo).clamp_min(1e-8)
    th = torch.as_tensor([float(t) for t in thresholds], dtype=mn.dtype, device=mn.device)
    if th.numel() != len(levels) - 1:
        raise ValueError(f"per-(row, chunk) needs {len(levels) - 1} thresholds, "
                         f"got {th.numel()}")
    return torch.bucketize(mn, th).to(torch.int32)


def _forward_row_mp(x2, w, op, block_idx, cfg, sc_prec):
    tb = rt.state()["total_blocks"]
    prot = cfg.get_protected_channels(operator=op, block_idx=block_idx)
    prot_idx = _channel_index_tensor(prot, x2.shape[1], x2.device)
    rest_idx = _complement(x2.shape[1], prot_idx, x2.device) if prot_idx.numel() else None

    prc = None
    get_prc = getattr(cfg, "get_per_row_chunk", None)
    if get_prc is not None:
        prc = get_prc(op, block_idx, tb, timestep=0, total_timesteps=1)

    width = max(x2.shape[1], 1)
    prot_scale = float(prot_idx.numel()) / float(width)
    rest_scale = 1.0 - prot_scale
    macs_per_row = x2.shape[1] * w.shape[0]

    if prc is not None:
        levels = [int(v) for v in prc[0]]
    else:
        levels = cfg.classify_level_values(operator=op, block_idx=block_idx,
                                           total_blocks=tb, timestep=0,
                                           total_timesteps=1)
    max_sl = max(int(s) for s in levels)
    baseline = x2.shape[0] * macs_per_row * max_sl
    actual = 0.0

    y = torch.zeros(x2.shape[0], w.shape[0], dtype=torch.float32, device=x2.device)
    if prot_idx.numel() > 0:
        prot_sl = int(getattr(cfg, "protected_channel_stoc_len", None) or max_sl)
        y += _mm(x2.index_select(1, prot_idx).contiguous(),
                 w.index_select(1, prot_idx).contiguous(), prot_sl, sc_prec)
        actual += x2.shape[0] * macs_per_row * prot_scale * prot_sl
        x_disp = x2.index_select(1, rest_idx).contiguous()
        w_disp = w.index_select(1, rest_idx).contiguous()
    else:
        x_disp, w_disp = x2, w

    if rest_scale > 0.0 and x_disp.shape[1] > 0:
        if prc is not None:
            cd = rt.state()["chunk_d"]
            if not (cd > 0 and x_disp.shape[1] > cd):
                raise ValueError(f"sc_linear({op}, b{block_idx}): per_row_chunk "
                                 f"needs 0 < chunk_d < width ({x_disp.shape[1]})")
            rung = per_row_chunk_rungs(x_disp, cd, levels, prc[1])
            y += sc_matmul(x_disp, w_disp, granularity="per_row", mode="bipolar",
                           sc_prec=sc_prec, stoc_len=max_sl, group_a=1, group_b=1,
                           chunk_d=cd, halve_bipolar_stoc_len=rt.state()["halve"],
                           rung_table=rung, level_lens=levels)
            n_ch = rung.shape[1]
            cnt = torch.bincount(rung.reshape(-1).to(torch.int64), minlength=len(levels))
            for r, sl in enumerate(levels):
                if int(cnt[r]):
                    actual += (int(cnt[r]) / n_ch) * macs_per_row * rest_scale * sl
        else:
            name, sign = cfg.get_dispatch_metric(op)
            metric = rt.maybe_shuffle(compute_row_metric(x_disp, name) * sign)
            assignment = adaptive_classify_rows(
                metric, cfg, operator=op, block_idx=block_idx, total_blocks=tb,
                timestep=0, total_timesteps=1)
            for sl, idx in assignment.level_row_indices.items():
                sl = int(sl)
                if idx.numel() == 0 or sl <= 0:     # rung 0 == pruned row
                    continue
                actual += idx.numel() * macs_per_row * rest_scale * sl
                out_sub = _mm(x_disp.index_select(0, idx).contiguous(), w_disp, sl, sc_prec)
                if prot_idx.numel() > 0:
                    y[idx] += out_sub
                else:
                    y[idx] = out_sub
    AutoMPBudgetLogger.record(block_idx, op, baseline, actual)
    rt.record_cost(op, x2.shape[0] * macs_per_row, actual / max(x2.shape[0] * macs_per_row, 1))
    return y


def sc_linear_forward(
    x: torch.Tensor,
    linear: nn.Linear,
    sc_prec: int | None = None,
    stoc_len: int | None = None,
    op: str | None = None,
    block_idx: int | None = None,
) -> torch.Tensor:
    """``y = SC(x @ Wᵀ) + bias``. ``op`` is the controller name (qkv / proj /
    mlp_fc1 / mlp_fc2); tables and hooks see the scmp_vit name."""
    top = rt.CTRL_TO_TABLE_OP.get(op, op)
    sc_prec = rt.state()["sc_prec"] if sc_prec is None else int(sc_prec)
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).float().contiguous()
    w = linear.weight.float().contiguous()
    bias = linear.bias.float() if linear.bias is not None else None
    smooth = getattr(linear, "_sc_smooth_scales", None)
    if smooth is not None:
        s = smooth.to(device=x2.device, dtype=torch.float32)
        x2 = (x2 / s).contiguous()
        w = (w * s).contiguous()

    if (rt.state()["chunk_d"] <= 0 and x2.shape[1] >= rt.state()["chunk_d_warn_width"]
            and (top, x2.shape[1]) not in _CHUNK_WARNED):
        _CHUNK_WARNED.add((top, x2.shape[1]))
        import warnings
        warnings.warn(f"sc_linear({top}, D_in={x2.shape[1]}) with chunk_d=0: "
                      f"unchunked SC on wide layers collapses (scmp_vit: 0% top-1).",
                      RuntimeWarning, stacklevel=2)

    macs_per_row = float(x2.shape[1]) * float(w.shape[0])
    for hook in rt.linear_hooks():
        hook(top, block_idx, x2, w, bias, macs_per_row)

    if not rt.sc_enabled() or x2.shape[0] == 0:
        y = F.linear(x2, w, bias)
        return y.reshape(*orig_shape[:-1], -1).to(x.dtype)
    if sc_trace._ENABLED:
        sc_trace.set_context(top, block_idx, None)

    gmap = rt.group_stoclen()
    cfg = rt.mp_cfg_for(top)
    if gmap is not None:
        sl = int(gmap.get((top, block_idx), rt.resolve_stoc_len(stoc_len, sc_prec)))
        y = (torch.zeros(x2.shape[0], w.shape[0], device=x2.device)
             if sl <= 0 else _mm(x2, w, sl, sc_prec))
        rt.record_cost(top, x2.shape[0] * macs_per_row, max(sl, 0))
    elif cfg is not None:
        y = _forward_row_mp(x2, w, top, block_idx, cfg, sc_prec)
    else:
        sl = rt.resolve_stoc_len(stoc_len, sc_prec)
        y = _mm(x2, w, sl, sc_prec)
        rt.record_cost(top, x2.shape[0] * macs_per_row, sl)
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], -1).to(x.dtype)
