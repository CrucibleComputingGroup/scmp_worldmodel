"""Deterministic INT / BitMoD baselines for IRASim, scmp_vit protocol.

Two recipes, both swapped in over the SC entry points (``sc_linear_forward``,
``sc_qk_matmul``, ``sc_av_matmul``) exactly like the naive-int8 baseline, so
they quantize the SAME (op, block) set the SC cells run on SC (``--skip``
groups stay FP):

* ``int``   — scmp_vit ``model/int_vit.py`` W=A RTN at ``bits``, asymmetric
  (the Table V INT recipe): linears per-row x 128-wide input chunk for both
  activations and weights (the SC ``per_row`` + ``chunk_d=128`` scope),
  Q@K^T and attn@V per row (the SC ``per_row`` attention scope).
* ``bitmod`` — scmp_vit ``cls/eval_wq.py`` with ``--aq_bits 8``: weights in a
  BitMoD-HPCA-25 datatype (``mixed_bitmod`` by default; ``int4_asym``,
  ``mx_fp4`` ... also accepted) at group 128, activations INT8 asymmetric
  per-row x 128 chunk, attention INT8 per row.

SmoothQuant: scmp_vit's rule is "both off or both on" for the INT and SC
cells. IRASim's SC cells keep SmoothQuant, so when a linear carries
``_sc_smooth_scales`` the INT linear quantizes ``(x/s, w*s)`` too.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CHUNK = 128


def fake_quant(x: torch.Tensor, bits: int, sym: bool, dim: int = -1) -> torch.Tensor:
    """scmp_vit ``int_vit.fake_quant`` (verbatim semantics)."""
    if bits >= 16:
        return x
    xf = x.float()
    if sym:
        qmax = float(2 ** (bits - 1) - 1)
        qmin = -float(2 ** (bits - 1))
        amax = xf.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        deq = torch.clamp(torch.round(xf / scale), qmin, qmax) * scale
    else:
        qmax = float(2 ** bits - 1)
        xmin = xf.amin(dim=dim, keepdim=True)
        xmax = xf.amax(dim=dim, keepdim=True)
        scale = (xmax - xmin).clamp_min(1e-8) / qmax
        zp = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(xf / scale) + zp, 0.0, qmax)
        deq = (q - zp) * scale
    return deq.to(x.dtype)


def fake_quant_chunks(x: torch.Tensor, bits: int, sym: bool,
                      chunk_size: int = CHUNK) -> torch.Tensor:
    """scmp_vit ``int_vit.fake_quant_chunks``."""
    if bits >= 16:
        return x
    if chunk_size <= 0 or x.shape[-1] <= chunk_size:
        return fake_quant(x, bits, sym, dim=-1)
    return torch.cat([fake_quant(x[..., s:s + chunk_size], bits, sym, dim=-1)
                      for s in range(0, x.shape[-1], chunk_size)], dim=-1)


_CFG = {"a_bits": 8, "w_bits": 8, "sym": False, "w_datatype": None, "attn_bits": 8}
_WCACHE: dict = {}


def _weight_q(linear, w_s):
    key = id(linear)
    hit = _WCACHE.get(key)
    if hit is not None:
        return hit
    if _CFG["w_datatype"] is None:
        wq = fake_quant_chunks(w_s, _CFG["w_bits"], _CFG["sym"])
    else:
        from evaluate.bitmod_wq import quantize_weight
        wq = quantize_weight(w_s, _CFG["w_bits"], _CFG["w_datatype"], CHUNK)
    _WCACHE[key] = wq.to(w_s.device)
    return _WCACHE[key]


def int_linear_forward(x, linear, sc_prec=8, stoc_len=None, op=None, block_idx=None):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    w = linear.weight.float()
    s = getattr(linear, "_sc_smooth_scales", None)
    if s is not None:
        s = s.to(device=x2.device, dtype=torch.float32)
        x2, w = x2 / s, w * s
    a = fake_quant_chunks(x2, _CFG["a_bits"], _CFG["sym"])
    y = F.linear(a, _weight_q(linear, w),
                 linear.bias.float() if linear.bias is not None else None)
    return y.reshape(*shape[:-1], -1).to(x.dtype)


def int_qk(q_scaled, k, sc_prec=8, stoc_len=None):
    b = _CFG["attn_bits"]
    qd = fake_quant(q_scaled.float(), b, _CFG["sym"], dim=-1)
    kd = fake_quant(k.float(), b, _CFG["sym"], dim=-1)
    return torch.matmul(qd, kd.transpose(-1, -2)).to(q_scaled.dtype)


def int_av(attn, v, sc_prec=8, stoc_len=None):
    b = _CFG["attn_bits"]
    ad = fake_quant(attn.float(), b, _CFG["sym"], dim=-1)
    # SC quantizes V^T per row (per output channel over the key axis).
    vd = fake_quant(v.float().transpose(-1, -2), b, _CFG["sym"], dim=-1).transpose(-1, -2)
    return torch.matmul(ad, vd).to(attn.dtype)


def install(kind: str, bits: int, w_datatype: str | None = None) -> str:
    """Patch the SC entry points. Returns a one-line description."""
    if kind == "int":
        _CFG.update(a_bits=bits, w_bits=bits, attn_bits=bits, sym=False, w_datatype=None)
        desc = f"INT W{bits}A{bits} asym (scmp_vit int_vit): linears per-row x {CHUNK} chunk, attention per-row"
    elif kind == "bitmod":
        dt = w_datatype or "mixed_bitmod"
        _CFG.update(a_bits=8, w_bits=bits, attn_bits=8, sym=False, w_datatype=dt)
        desc = (f"BitMoD W{bits} {dt} g{CHUNK} + A8 asym per-row x {CHUNK} chunk + INT8 attention "
                f"per-row (scmp_vit eval_wq --aq_bits 8)")
    else:
        raise ValueError(f"unknown INT baseline {kind!r}")
    _WCACHE.clear()
    import models.sc_integration as root
    import models.sc_integration.sc_attention as sa
    import models.sc_integration.sc_linear as sl
    import models.sc_integration.sc_mlp as sm
    for mod in (root, sa):
        mod.sc_qk_matmul = int_qk
        mod.sc_av_matmul = int_av
    for mod in (root, sl, sm):
        mod.sc_linear_forward = int_linear_forward
    return desc + "; SmoothQuant applied where attached"
