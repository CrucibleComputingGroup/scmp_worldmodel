"""Calibrate per-operator, per-block SC mixed-precision thresholds for IRASim.

IRASim port of ``scmp_vit/cls/calibrate_mp_thresholds.py`` (itself the ViT
port of scmp_llm's calibrator). Everything between the two ``IRASim
adapters`` banners below is copied VERBATIM from scmp_vit -- the metric /
error utilities, the global-lambda budget solver, ``ThresholdCalibrator`` and
its export -- so a table produced here has the same schema, method tags and
semantics as a scmp_vit table and is consumed unchanged by
``scmp_kernels.mp.AdaptiveMPConfig`` (``SC_MP_TABLE`` at eval time).

What is IRASim-specific:

* **Calibration unit = one generated video.** Each calibration key runs the
  full sampler (PNDM-50 by default, classifier-free guidance as at eval); the
  hooks record on every ``--record_every_k_steps``-th denoising step (uniform
  over the trajectory, so the single timestep bucket stays unbiased). Keys are
  drawn from the Bridge test split EXCLUDING ``--eval_keys_file``, and the
  exact keys are recorded in the table (``calib_keys``) so the eval refuses to
  score them.
* **Hooks, not modules.** IRASim's SC path is functional
  (``models/sc_integration``), so the linear / attention recorders register
  through ``sc_runtime.add_linear_hook`` / ``add_attn_hook`` and receive the
  operands the kernel quantizes (SmoothQuant already applied).
* **Kept IRASim choices.** ``--skip`` blocks stay FP (never recorded, never in
  the SC budget) and ``SC_SMOOTH_SCALES`` stays on; both are recorded in the
  table and the eval refuses a table under a different skip / SQ deployment.
* **qk / av are calibrated** (scmp_vit's formal ``ALLOPS``), at
  ``--qk_granularity per_row``. The measured-dLoss probe
  (``--cross-layer-weight measured*``) is not ported.

Levels/budget are in HALVED space when ``--halve 1``.

Usage (the scmp_vit formal recipe, ``cls/run_l7_10k.sh``):

  SC_SMOOTH_SCALES=results/smoothquant_scales.pt PYTHONPATH=. \\
  python evaluate/calibrate_mp_thresholds.py \\
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml \\
    --skip "$(python -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")" \\
    --mp_levels 64,48,32,24,16 --budget_ratio 0.5 --budget_ref_stoc_len 64 \\
    --halve 1 --sc_prec 8 --chunk_d 128 --qk_granularity per_row \\
    --metric-select amax --objective sigma --cross-layer-weight macs --argmin-pricing macs \\
    --protect-channel-frac 0.02 --protect-channel-metric act --protect-channel-stoc-len 0 \\
    --eval_keys_file results/diverse_300.json --n_calib_samples 4 \\
    --output_json results/mp_calib/mp_32.json

  SC_HALVE=1 SC_MP_TABLE=results/mp_calib/mp_32_wrapper.json ... \\
    python evaluate/eval_local_n_samples.py ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent   # scmp_worldmodel
for _p in (str(_ROOT / "evaluate"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mp_objectives import objective_errors as _objective_errors  # noqa: E402
from scmp_kernels import sc_matmul  # noqa: E402
from models.sc_integration import sc_runtime as rt  # noqa: E402
from models.sc_integration.sc_controller import get_current_step  # noqa: E402

DEFAULT_OPERATORS = "qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av"
_LINEAR_OPS = {"qkv_proj": ("attn", "qkv"), "out_proj": ("attn", "proj"),
               "mlp_fc1": ("mlp", "fc1"), "mlp_fc2": ("mlp", "fc2")}
_TABLE_TO_CTRL = {v: k for k, v in rt.CTRL_TO_TABLE_OP.items()}


# =====================================================================
# IRASim adapters: build the SC model + a calibration forward iterator.
# =====================================================================


def _build_sc_model(args, device):
    """IRASim (frame-ada) with every SC op enabled except ``--skip`` blocks,
    the deployment's SmoothQuant scales attached, and the run-wide SC knobs
    set to the values the table will be deployed at."""
    from diffusers.models import AutoencoderKL
    from evaluate.eval_local_n_samples import (build_args, load_model, make_pipe,
                                               parse_skip)
    from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks

    rt.set_sc_options(sc_prec=int(args.sc_prec), halve=bool(args.halve),
                      chunk_d=int(args.chunk_d), qk_granularity=args.qk_granularity,
                      uniform_stoc_len=int(args.attn_stoc_len), step_schedule=None)
    cfg = build_args(args.config, args.inference_steps)
    if not str(cfg.attention_mode).startswith("sc_int8"):
        raise SystemExit(f"--config attention_mode={cfg.attention_mode!r} is not an "
                         f"SC preset")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(cfg, device)
    sq_path = os.environ.get("SC_SMOOTH_SCALES")
    if sq_path:
        payload = torch.load(sq_path, map_location=device, weights_only=False)
        named = dict(model.named_modules())
        n_att = 0
        for name, s in payload["scales"].items():
            if name in named:
                named[name]._sc_smooth_scales = s.to(device).float()
                n_att += 1
        print(f"[calib] smoothquant: attached {n_att} scale vectors", flush=True)
    reconfigure(cfg.attention_mode)
    clear_skip_blocks()
    skip_map = parse_skip(args.skip)
    for op, blocks in skip_map.items():
        set_skip_blocks(op, blocks)
    rt.set_sc_options(total_blocks=len(model.blocks))
    pipe = make_pipe(cfg, vae, model, args.scheduler)
    print(f"[calib] model {cfg.attention_mode}, {len(model.blocks)} blocks, "
          f"skip={skip_map}, steps={args.inference_steps} {args.scheduler}",
          flush=True)
    return model, pipe, cfg, skip_map


def _build_calib_iter(args, device, pipe, cfg):
    """Calibration keys from the Bridge test split, EXPLICITLY disjoint from the
    keys PSNR/SSIM is later reported on (``--eval_keys_file``)."""
    import numpy as _np
    from evaluate.eval_local_n_samples import (GT_LATENT_DIR, ANNOT_DIR, SEQUENCE_LENGTH,
                                               C_ACT_SCALER, compute_actions_for_slice,
                                               gen_one, parse_key)
    excluded = set()
    if args.eval_keys_file:
        excluded = set(json.load(open(args.eval_keys_file)))
    pool = []
    for fn in sorted(os.listdir(GT_LATENT_DIR)):
        if not fn.endswith(".pt") or fn[:-3] in excluded:
            continue
        eid, cam, start = parse_key(fn)
        ann_path = os.path.join(ANNOT_DIR, f"{eid}.json")
        if os.path.exists(ann_path):
            pool.append(fn)
    random.Random(args.calib_seed).shuffle(pool)
    sel, lo = [], args.calib_offset
    for fn in pool[lo:]:
        if len(sel) >= args.n_calib_samples:
            break
        eid, cam, start = parse_key(fn)
        with open(os.path.join(ANNOT_DIR, f"{eid}.json")) as f:
            ann = json.load(f)
        if start + SEQUENCE_LENGTH <= len(ann["state"]):
            sel.append((fn, ann))
    if len(sel) < args.n_calib_samples:
        raise ValueError(f"only {len(sel)} calibration keys available after "
                         f"excluding {len(excluded)} eval keys")
    print(f"[calib] calib pool {len(pool)} keys ({len(excluded)} eval keys "
          f"excluded), using {len(sel)}: {[fn[:-3] for fn, _ in sel]}", flush=True)

    def run_forward(_model, item):
        fn, ann = item
        eid, cam, start = parse_key(fn)
        gt_lat = torch.load(os.path.join(GT_LATENT_DIR, fn), weights_only=False,
                            map_location=device)
        arm = _np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = _np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        actions = torch.from_numpy(compute_actions_for_slice(arm, grip) * C_ACT_SCALER
                                   ).float().unsqueeze(0)
        gen_one(pipe, gt_lat[0:1].unsqueeze(0), actions, cfg, device)

    return sel, run_forward, [fn[:-3] for fn, _ in sel]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _recording_step(k: int) -> bool:
    i, _n = get_current_step()
    return k <= 1 or i % k == 0


# =====================================================================
# Metric / error utilities + budget solver + ThresholdCalibrator:
# VERBATIM from scmp_vit/cls/calibrate_mp_thresholds.py (HPCA_VIT c85fefd)
# =====================================================================


def _normalize_metric(metric: torch.Tensor) -> torch.Tensor:
    metric = metric.float()
    if metric.numel() == 0:
        return metric
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        return torch.ones_like(metric, dtype=torch.float32)
    return (metric - m_min) / (m_max - m_min)


def _metric_candidates(x_rows: torch.Tensor) -> dict:
    """act_global_v2 candidate dispatch metrics for one call, per-call
    NORMALIZED like the primary amax metric. ``x_rows`` is the SAME row-source
    tensor amax was computed from, flattened to (rows, D). Formulas mirror
    ``scmp_kernels.mp.compute_row_metric`` exactly (runtime parity):
    l2 = ‖row‖₂, crest = ‖row‖_inf / ‖row‖₂ (eps 1e-12)."""
    xf = x_rows.float().reshape(-1, x_rows.shape[-1])
    l2 = xf.norm(dim=-1)
    amax = xf.abs().amax(dim=-1)
    crest = amax / (l2 + 1e-12)
    return {"l2": _normalize_metric(l2), "crest": _normalize_metric(crest)}


def _relative_l2_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float().reshape(pred.shape[0], -1)
    target_f = target.float().reshape(target.shape[0], -1)
    denom = target_f.norm(dim=-1).clamp_min(1e-8)
    return (pred_f - target_f).norm(dim=-1) / denom


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


# =====================================================================
# Budget allocation (Lagrangian binary search) — verbatim from the LLM
# =====================================================================


def _cost_assignments(
    errors: np.ndarray,
    costs: np.ndarray,
    budget_total: float,
) -> np.ndarray:
    """Per-group budget: pick a level per row minimizing sum(error) under
    sum(cost) <= budget (used only for --budget-scope per_bucket)."""
    n_units = errors.shape[0]
    min_cost = float(n_units * costs[-1])
    max_cost = float(n_units * costs[0])
    if budget_total <= min_cost:
        return np.full(n_units, len(costs) - 1, dtype=np.int64)
    if budget_total >= max_cost:
        return np.zeros(n_units, dtype=np.int64)

    def solve_for_lambda(lmbd: float):
        objective = errors + lmbd * costs[None, :]
        assignment = objective.argmin(axis=1)
        total_cost = float(costs[assignment].sum())
        return assignment, total_cost

    lo, hi = 0.0, 1.0
    _, cost_hi = solve_for_lambda(hi)
    while cost_hi > budget_total and hi < 1e6:
        hi *= 2.0
        _, cost_hi = solve_for_lambda(hi)

    best = np.zeros(n_units, dtype=np.int64)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cand, cost_mid = solve_for_lambda(mid)
        best = cand
        if cost_mid > budget_total:
            lo = mid
        else:
            hi = mid
    return best


def _global_lambda(
    weighted_errors: list,
    costs: np.ndarray,
    budget_ratio: float,
    ref: float,
    row_counts: list,
    prices: Optional[list] = None,
) -> float:
    """ONE shared Lagrange multiplier across all groups for a global,
    R_g-weighted budget — the cross-layer allocation core. ``row_counts[g]``
    is group g's TRUE per-forward weight (rows, or MACs under
    --budget-weight macs, in which case iso-budget = iso-compute).
    ``prices[g]`` is the per-stored-row cycle PRICE in the argmin. Default (None)
    is ``rep_g = R_g / n_g`` — each stored (subsampled) row stands for rep_g true
    rows. ``--argmin-pricing macs`` passes ``mac_per_row[op]`` instead: the
    KKT-consistent price where the subsample ratio cancels between benefit and
    cost (the extra true/stored factor in rep_g otherwise over-prices heavily
    subsampled ops and starves them — the LLM's attention/dense mispricing)."""
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    if prices is None:
        prices = [float(R) / max(int(e.shape[0]), 1)
                  for e, R in zip(weighted_errors, row_counts)]
    reps = prices
    global_budget = budget_ratio * ref * total_rows
    min_cost = float(total_rows * costs[-1])
    max_cost = float(total_rows * costs[0])
    if global_budget >= max_cost:
        return 0.0
    if global_budget <= min_cost:
        return 1e12

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, rep in zip(weighted_errors, row_counts, reps):
            assign = (e + lmbd * rep * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


def _global_lambda_multi(
    weighted_errors: list,
    group_costs: list,
    budget_ratio: float,
    ref: float,
    row_counts: list,
    prices: list,
) -> float:
    """``_global_lambda`` with a DIFFERENT cost ladder per group.

    Needed once per-group ladders exist: the shared-``costs`` version prices
    every group against the global ladder, so a group that was re-fitted onto
    its own rungs would be solved at one ladder and deployed at another. The
    budget semantics are unchanged — one shared λ, cost measured in the
    ``budget_weight`` currency (MACs under ``--budget-weight macs``).
    """
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    global_budget = budget_ratio * ref * total_rows
    min_cost = sum(float(R) * float(c[-1]) for R, c in zip(row_counts, group_costs))
    max_cost = sum(float(R) * float(c[0]) for R, c in zip(row_counts, group_costs))
    if global_budget >= max_cost:
        return 0.0
    if global_budget <= min_cost:
        return 1e12

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, rep, costs in zip(weighted_errors, row_counts, prices,
                                    group_costs):
            assign = (e + lmbd * rep * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


def _select_group_ladder(err_obj: np.ndarray, costs: np.ndarray,
                         lam: float, rep: float, weight: float,
                         k: int) -> np.ndarray:
    """Pick the ``k`` rungs of ``costs`` this group actually uses.

    A calibrated table historically carried ONE ladder shared by every
    (op x layer-bucket) bucket, with only the thresholds varying. Measured
    occupancy shows that wastes roughly half the rungs: on ViT-L/14 the
    ``out_proj`` buckets pin at the ceiling and put ZERO mass on the bottom
    rungs while ``mlp_fc1`` lives entirely at the bottom — the two populations
    sit at opposite ends of a shared ladder, so each gets about half the
    available resolution.

    The fix is to calibrate sigma on a DENSE candidate ladder and let each
    group keep only the rungs its own argmin puts mass on. Returns INDICES into
    ``costs``, ascending, so the caller can slice both costs and errors.

    Selection keeps the two ENDPOINTS of the group's occupied range and fills
    the rest by occupancy. Ranking purely by mass is wrong here and was measured
    to be wrong: the first version dropped the high rungs (few rows, by
    definition — the importance distribution is heavy-tailed) and capped whole
    `mlp_fc1` buckets at 24 cycles, so no row in them could reach high precision
    however fragile it was. That cost 1.66 pt vs the shared ladder at eff 64
    (`s32_glk` 0.7407 vs `s32_base` 0.7573). The top rung is exactly the one a
    mass criterion discards and exactly the one outlier rows need.
    """
    assign = ((weight * err_obj) + lam * rep * costs[None, :]).argmin(axis=1)
    mass = np.bincount(assign, minlength=costs.size).astype(np.float64)
    k = max(2, min(int(k), int(costs.size)))
    occupied = np.flatnonzero(mass > 0)
    if occupied.size <= k:
        keep = occupied
    else:
        # Endpoints of the occupied range are non-negotiable: the top rung is
        # the headroom, the bottom rung is the cheap end the budget is paid
        # from. Middle slots go by occupancy, ties toward the longer stream.
        forced = {int(occupied[0]), int(occupied[-1])}
        order = np.lexsort((np.arange(costs.size), -mass))
        for idx in order:
            if len(forced) >= k:
                break
            if mass[idx] > 0:
                forced.add(int(idx))
        keep = np.sort(np.fromiter(forced, dtype=np.int64))
    if keep.size < 2:
        # Degenerate: everything on one rung. Keep a neighbour so the group
        # still has something to trade against — a 1-rung "ladder" is uniform,
        # and AdaptiveMPConfig rejects it anyway.
        only = int(keep[0]) if keep.size else 0
        nb = only + 1 if only + 1 < costs.size else only - 1
        keep = np.sort(np.array([only, max(nb, 0)], dtype=np.int64))
    return keep


def _normalize_group_weights(raw: dict, floor_frac: float = 0.1) -> dict:
    """Floor ΔL at a small positive fraction of the mean, then scale to mean 1.

    The floor keeps measurement noise (or a genuinely negative ΔL) from starving
    a group outright; the mean-1 scaling keeps the global budget interpretation
    (realized avg ≈ budget_ratio·ref) unchanged, since a uniform scale on W_g is
    absorbed by λ. Ported from scmp_llm's ``_normalize_group_weights``.
    """
    vals = np.asarray(list(raw.values()), dtype=np.float64)
    if vals.size == 0:
        return {}
    pos = np.clip(vals, 0.0, None)
    mean_pos = float(pos.mean()) if pos.mean() > 0 else 1.0
    floor = floor_frac * mean_pos
    floored = {k: max(float(v), floor) for k, v in raw.items()}
    m = float(np.mean(list(floored.values()))) or 1.0
    return {k: v / m for k, v in floored.items()}



# =====================================================================
# IRASim adapters: protected-channel statistics
# =====================================================================


@torch.no_grad()
def _collect_channel_act_scales(model, batches, run_forward) -> dict:
    """Per-input-channel activation amax for every SC linear (scmp_vit
    semantics), measured on the operand the kernel quantizes — i.e. AFTER
    SmoothQuant — over an FP trajectory. Keyed ``(table_op, block)``; the
    entry's length is the layer's D_in."""
    scales: dict = {}

    def hook(op, block_idx, x2, _w, _b, _macs):
        a = x2.abs().amax(dim=0).float().cpu()
        key = (op, block_idx)
        prev = scales.get(key)
        scales[key] = a if prev is None else torch.maximum(prev, a)

    remove = rt.add_linear_hook(hook)
    rt.set_sc_enabled(False)
    try:
        for batch in batches:
            run_forward(model, batch)
    finally:
        rt.set_sc_enabled(True)
        remove()
    return scales


def _smoothed_weight(model, op, block_idx):
    parent, name = _LINEAR_OPS[op]
    lin = getattr(getattr(model.blocks[block_idx], parent), name)
    w = lin.weight.detach().float()
    s = getattr(lin, "_sc_smooth_scales", None)
    return w * s.to(w.device).float() if s is not None else w


def _select_protected_channels(model, act_scales, frac, metric):
    """Top-``frac`` salient input channels per SC linear -> {(op, block): idx}
    (scmp_vit ``_select_protected_channels``; weights are the SmoothQuant-
    transformed ones the kernel sees)."""
    if frac <= 0.0:
        return {}, {}
    if not 0.0 < frac < 1.0:
        raise SystemExit(f"[calib] --protect-channel-frac must be in (0, 1), got {frac}")
    indices, dims = {}, {}
    for key, a in sorted(act_scales.items()):
        op, block_idx = key
        if op not in _LINEAR_OPS:
            continue
        w = _smoothed_weight(model, op, block_idx)
        D = int(w.shape[1])
        n_keep = int(round(frac * D))
        if n_keep <= 0:
            continue
        a = a.to(w.device).float()
        if metric == "act":
            score = a
        elif metric == "act_weight":
            score = a.pow(2) * w.pow(2).sum(dim=0)
        elif metric == "weight":
            score = w.pow(2).sum(dim=0)
        else:
            raise SystemExit(f"[calib] unknown --protect-channel-metric {metric!r}")
        top = torch.topk(score, n_keep).indices.sort().values
        indices[key] = [int(i) for i in top.cpu().tolist()]
        dims[key] = D
    return indices, dims


def _thresholds_from_counts(metrics: np.ndarray, counts: np.ndarray) -> list:
    if metrics.size == 0:
        return []
    sorted_metrics = np.sort(metrics)[::-1]
    thresholds: list = []
    offset = 0
    for count in counts[:-1]:
        offset += int(count)
        if offset <= 0:
            thresholds.append(1.0)
        elif offset >= len(sorted_metrics):
            thresholds.append(0.0)
        else:
            thresholds.append(
                float(0.5 * (sorted_metrics[offset - 1] + sorted_metrics[offset]))
            )
    return thresholds


# =====================================================================
# Calibrator (slim act_global port; W_g from --cross-layer-weight)
# =====================================================================


class ThresholdCalibrator:
    def __init__(
        self,
        levels: list,
        operators: Iterable[str],
        total_blocks: int,
        layer_buckets: int,
        budget_ratio: float,
        budget_ref_stoc_len: Optional[int],
        max_units_per_call: int,
        min_bucket_units: int,
        rng_seed: int,
        budget_scope: str = "global",
        budget_weight: str = "macs",
        metric_select: str = "amax",
        metric_select_margin: float = 0.05,
        argmin_pricing: str = "rep",
        objective: str = "sigma",
        cross_layer_weight: str = "uniform",
        group_ladder_k: int = 0,
        escape_gate_k: Optional[float] = None,
        escape_stoc_len: int = 0,
    ):
        # Escape gate. Deployed by the WRAPPER, but the calibrator has to know
        # about it: the gate hands rows to a high rung AFTER the Lagrangian has
        # allocated, so an ungated solve overshoots the budget by whatever the
        # gate takes. Measured on ViT that is 13-23% of rows, i.e. not a
        # rounding error. See _gated_mac_avg.
        self.escape_gate_k = (float(escape_gate_k)
                              if escape_gate_k is not None else None)
        self.escape_stoc_len = int(escape_stoc_len or max(levels))
        # Per-group ladders: 0 = off (one shared ladder, the historical
        # behaviour); k>0 = calibrate sigma on the dense --mp_levels ladder but
        # deploy only the k rungs each (op x layer-bucket) actually uses.
        self.group_ladder_k = int(group_ladder_k)
        # Protected input channels: {(op, block): [channel indices]} run at a
        # fixed high stream length for every row, split off along the
        # contraction dim. Empty unless --protect-channel-frac > 0.
        self.protected_channel_indices: dict = {}
        self.protect_channel_frac = 0.0
        self.protect_channel_stoc_len = 0
        self.protect_channel_metric = "none"
        self.protect_compensate_budget = False
        self.protected_channel_dims: dict = {}
        # Allocation objective currency (mp_objectives.objective_errors) and the
        # per-group importance weight W_g. Defaults reproduce act_global exactly.
        self.objective = objective
        self.cross_layer_weight = cross_layer_weight
        self._w_norm: Optional[float] = None
        # Phase-2 measured sensitivity, normalized to mean 1 and expanded to
        # per-BLOCK so it can be averaged onto any table bucketing.
        # {(operator, block_idx): W} — empty unless a measured probe ran.
        self.block_weights: dict = {}
        self.measure_meta: dict = {}
        # act_global_v2 dispatch-metric selection: "amax" (legacy, byte-
        # identical) or "auto" (capture l2/crest candidates, pick per operator
        # the signed candidate with best |Spearman ρ| vs σ-benefit).
        self.metric_select = metric_select
        self.metric_select_margin = metric_select_margin
        self.dispatch_metric_selection: dict = {}
        # act_global_v2 argmin cost pricing: "rep" (legacy rep_g = R_g/n_g,
        # includes the subsample ratio) or "macs" (KKT-consistent mac_per_row
        # only; use with --budget-weight macs + global scope).
        self.argmin_pricing = argmin_pricing
        self.levels = levels
        self.operators = set(operators)
        self.total_blocks = total_blocks
        self.timestep_buckets = 1
        self.layer_buckets = layer_buckets
        self.budget_ratio = budget_ratio
        self.budget_ref_stoc_len = (int(budget_ref_stoc_len)
                                    if budget_ref_stoc_len else max(levels))
        self.max_units_per_call = max_units_per_call
        self.min_bucket_units = min_bucket_units
        self.budget_scope = budget_scope
        self.budget_weight = budget_weight
        self.costs = np.asarray(levels, dtype=np.float64)
        self.records: dict = defaultdict(
            lambda: {"metrics": [], "errors": [], "alts": defaultdict(list)})
        # True (un-subsampled) per-forward row / MAC counts per group. MACs
        # accumulate rows*macs_per_row per call, so groups that pool blocks
        # with different channel widths are priced by their true mean.
        self.true_counts: dict = defaultdict(int)
        self.true_macs: dict = defaultdict(float)
        self._rng = np.random.default_rng(rng_seed)
        # Dispatch unit for the linear operators ("row" | "chunk"); set from
        # --dispatch-unit by main(). chunk_unit_ops collects the operators whose
        # records are (row, chunk) PAIRS so export() writes them a
        # per_row_chunk section.
        self.dispatch_unit = "row"
        self.chunk_threshold_fit = "counts"
        self.chunk_unit_ops: set = set()
        # Reference table payload for --bucket-budget-from (None = off).
        self.bucket_budget_ref: Optional[dict] = None

    def use_operator(self, op: str) -> bool:
        return op in self.operators

    def add(
        self,
        operator: str,
        block_idx: int,
        metric_norm: torch.Tensor,
        errors_by_level: list,
        mac_per_row: float,
        alts: Optional[dict] = None,
    ):
        if not self.use_operator(operator):
            return
        metrics = metric_norm.detach().float().reshape(-1).cpu().numpy()
        err = torch.stack(
            [e.detach().float().reshape(-1) for e in errors_by_level], dim=-1
        )
        errors = err.cpu().numpy()
        if metrics.size == 0:
            return
        # act_global_v2 candidate metrics (per-call NORMALIZED, aligned with the
        # amax rows/errors) captured only under --metric-select auto.
        alt_np = ({k: v.detach().float().reshape(-1).cpu().numpy()
                   for k, v in alts.items()}
                  if (self.metric_select == "auto" and alts) else {})
        true_size = int(metrics.size)
        if self.max_units_per_call > 0 and metrics.size > self.max_units_per_call:
            keep = self._rng.choice(metrics.size, self.max_units_per_call,
                                    replace=False)
            metrics = metrics[keep]
            errors = errors[keep]
            alt_np = {k: (v[keep] if v.size == true_size else v)
                      for k, v in alt_np.items()}
        l_bucket = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
        key = (operator, 0, l_bucket)
        self.records[key]["metrics"].append(metrics)
        self.records[key]["errors"].append(errors)
        for k, v in alt_np.items():
            self.records[key]["alts"][k].append(v)
        self.true_counts[key] += true_size
        self.true_macs[key] += true_size * float(mac_per_row)

    # -- budget weights ---------------------------------------------------
    def _weight_count(self, key, n_fallback=None) -> float:
        """Per-group budget weight R_g: rows or MACs per self.budget_weight."""
        n = (n_fallback if n_fallback is not None
             else sum(int(m.size) for m in self.records[key]["metrics"]))
        if self.budget_weight == "macs":
            return float(self.true_macs.get(key, float(n)))
        return float(self.true_counts.get(key, n))

    def _rep_for_key(self, key) -> float:
        n = sum(int(m.size) for m in self.records[key]["metrics"])
        return self._weight_count(key) / max(n, 1)

    def _rep_for_op(self, operator) -> float:
        n = R = 0.0
        for key, rec in self.records.items():
            if key[0] != operator:
                continue
            ng = sum(int(m.size) for m in rec["metrics"])
            n += ng
            R += self._weight_count(key, ng)
        return (R / max(n, 1)) if n else 1.0

    def _mac_per_row(self, key) -> float:
        """True MACs per row for a group = true_macs / true_rows (subsample-
        ratio-free, unlike rep_g = true_macs / n_stored)."""
        tc = float(self.true_counts.get(key, 0))
        return (float(self.true_macs.get(key, 0.0)) / tc) if tc > 0 else 1.0

    def _mac_per_row_op(self, operator) -> float:
        tm = tc = 0.0
        for key in self.records:
            if key[0] == operator:
                tm += float(self.true_macs.get(key, 0.0))
                tc += float(self.true_counts.get(key, 0))
        return (tm / tc) if tc > 0 else 1.0

    def _price_for_key(self, key) -> float:
        """Per-row cycle PRICE for the argmin. 'macs' = mac_per_row (KKT); 'rep'
        = rep_g = R_g/n_g (legacy, includes the subsample ratio)."""
        if self.argmin_pricing == "macs":
            return self._mac_per_row(key)
        return self._rep_for_key(key)

    def _price_for_op(self, operator) -> float:
        if self.argmin_pricing == "macs":
            return self._mac_per_row_op(operator)
        return self._rep_for_op(operator)

    # -- cross-layer importance weight W_g ---------------------------------
    def _weight_norm(self) -> float:
        """Scale so max(W_g) == 1. A GLOBAL scale on W_g is absorbed by lambda
        (the bisection just lands elsewhere), so this only keeps the exported
        global_lambda in a readable range — it changes no assignment."""
        if self._w_norm is None:
            vals = [self._mac_per_row(k) for k in self.records]
            self._w_norm = max(max(vals), 1e-12) if vals else 1.0
        return self._w_norm

    def _weight_for_key(self, key) -> float:
        """Per-group importance weight W_g for the global cross-layer solve.

        ``uniform`` (act_global, and scmp_llm's default) is W_g = 1: the
        objective is then Sigma over STORED rows of sigma while the cost side is
        MAC-weighted, so an operator that is cheap per row buys error reduction
        cheaply and the Lagrangian pins it at the ladder ceiling. Measured on
        DINOv2 ViT-L/14 (MP_REPORT.md): ``out_proj`` — 1.05 M MACs/row, 8 % of
        SC MACs — sits at 60 of a possible 64 cycles while ``mlp_fc1`` (4.19 M
        MACs/row, 32 % of MACs) is starved to 22, below its own quality cliff,
        and MP goes NEGATIVE against uniform SC at that budget.

        ``macs`` sets W_g = MACs/row, putting benefit and cost in one currency:
        minimize MAC-weighted Sigma sigma subject to MAC-weighted cycles <= B.
        Because ``rep_g`` already carries the same MACs/row factor, the per-row
        argmin then reduces to ``sigma + lambda' * cycles`` for EVERY operator —
        a single global cycles-per-sigma exchange rate, with no cost-side
        favouritism toward narrow layers. Rows still compete on their own sigma
        curves, so a genuinely fragile operator can still win budget.

        ``measured`` / ``macs_measured`` fill the same slot from the phase-2
        knock-down probe (see ``_measure_group_sensitivity``). Note that
        ``measured`` ALONE re-introduces the cost-side favouritism ``macs``
        removes: ``rep_g`` already carries a MACs/row factor, so the per-row
        argmin is ``(W_g / mac_g)·σ + λ′·cycles``, and only ``W_g ∝ mac_g``
        cancels it. ``macs_measured`` = ``mac_g × ΔL̂_g`` therefore keeps the
        currency aligned AND adds the measured tilt; plain ``measured`` is kept
        for ablation only.
        """
        clw = self.cross_layer_weight
        if clw == "macs":
            return self._mac_per_row(key) / self._weight_norm()
        if clw in ("measured", "macs_measured"):
            w = self._measured_for_key(key)
            if clw == "measured":
                return w
            return (self._mac_per_row(key) / self._weight_norm()) * w
        return 1.0

    def _measured_for_key(self, key) -> float:
        """Mean measured W over the blocks this table bucket covers.

        The probe runs on its own (coarser) bucketing, so it is stored per block
        and averaged onto whatever bucketing the threshold table uses. Falls back
        to 1.0 when the probe never covered this operator.
        """
        operator, _t, l_bucket = key
        vals = [w for (op, b), w in self.block_weights.items()
                if op == operator
                and _bucket_index(b, self.total_blocks, self.layer_buckets) == l_bucket]
        return float(np.mean(vals)) if vals else 1.0

    def _op_weight(self, operator) -> float:
        """Mean W_g over an operator's buckets (for the operator_default fit)."""
        clw = self.cross_layer_weight
        if clw == "uniform":
            return 1.0
        mac = (self._mac_per_row_op(operator) / self._weight_norm()
               if clw in ("macs", "macs_measured") else 1.0)
        if clw in ("measured", "macs_measured"):
            vals = [w for (op, _b), w in self.block_weights.items() if op == operator]
            mac *= float(np.mean(vals)) if vals else 1.0
        return mac

    # -- diagnostics --------------------------------------------------------
    def metric_fidelity_rho(self) -> dict:
        """Spearman rho(metric, sigma-benefit) per operator — how well the
        dispatch metric ranks rows by their true need for cycles. Low/negative
        rho => a better metric (not more levels) is the lever."""
        per_op = defaultdict(lambda: {"m": [], "b": []})
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            m = np.concatenate(rec["metrics"], axis=0)
            eo = _objective_errors(
                np.concatenate(rec["errors"], axis=0), self.objective)
            benefit = eo[:, -1] - eo[:, 0]
            per_op[key[0]]["m"].append(m)
            per_op[key[0]]["b"].append(benefit)
        out = {}
        for op, d in per_op.items():
            m = np.concatenate(d["m"]); b = np.concatenate(d["b"])
            out[op] = {"rho": _spearman(m, b), "n": int(m.size),
                       "benefit_mean": float(b.mean())}
        return out

    def select_dispatch_metrics(self) -> dict:
        """act_global_v2 (--metric-select auto): per-operator SIGNED dispatch-
        metric selection. For each operator, pool all buckets and compute
        Spearman ρ(candidate, objective-benefit) for amax + every captured alt;
        pick the candidate with the largest |ρ| (negative ρ ⇒ deploy with
        sign −1, which inverts the ranking). Switch away from raw amax only when
        the winner clears the incumbent by ``metric_select_margin`` (noise
        guard). The benefit is measured in the SAME currency the allocator uses
        (``--objective``), so metric selection and allocation never disagree
        about what a row gains from more cycles.

        On a switch the stored per-bucket metric arrays are REWRITTEN to the
        selected candidate (sign −1 ⇒ 1−m, exactly what the runtime's negated-
        metric min–max normalization produces), so thresholds / summaries / the
        post-selection ρ all reflect the deployed metric. The choice is exported
        via payload["dispatch_metrics"]. Assignments/budget are untouched — the
        metric only maps counts to thresholds."""
        if self.metric_select != "auto":
            return {}
        per_op: dict = defaultdict(lambda: defaultdict(lambda: {"m": [], "b": []}))
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            e = _objective_errors(
                np.concatenate(rec["errors"], axis=0), self.objective)
            b = e[:, -1] - e[:, 0]                 # obj(min L) − obj(max L) ≥ 0
            cands = {"amax": np.concatenate(rec["metrics"], axis=0)}
            for name, lst in rec["alts"].items():
                if lst:
                    cands[name] = np.concatenate(lst, axis=0)
            for name, m in cands.items():
                if m.shape[0] != b.shape[0]:
                    # Partial capture — never kill a long calibration over a
                    # diagnostic candidate; amax always stays available.
                    print(f"[calib] WARN metric-select: candidate '{name}' "
                          f"misaligned for group {key} "
                          f"({m.shape[0]} vs {b.shape[0]} rows) — skipped.")
                    continue
                per_op[key[0]][name]["m"].append(m)
                per_op[key[0]][name]["b"].append(b)
        chosen: dict = {}
        for op, cands in per_op.items():
            rhos = {name: _spearman(np.concatenate(d["m"]),
                                    np.concatenate(d["b"]))
                    for name, d in cands.items()}
            base = float(rhos.get("amax", 0.0))
            best_name = max(rhos, key=lambda n: abs(rhos[n]))
            best_rho = float(rhos[best_name])
            sign = 1.0 if best_rho >= 0.0 else -1.0
            if best_name == "amax" and sign > 0.0:
                continue                            # incumbent stands
            if abs(best_rho) < max(base, 0.0) + self.metric_select_margin:
                continue                            # not clearly better
            if best_name != "amax":
                # A switched candidate must cover EVERY stored row of the op.
                covered = all(
                    sum(int(a.size) for a in rec["alts"][best_name])
                    == sum(int(m.size) for m in rec["metrics"])
                    for k, rec in self.records.items() if k[0] == op)
                if not covered:
                    print(f"[calib] WARN metric-select: '{best_name}' wins on "
                          f"{op} but has partial capture — keeping amax.")
                    continue
            for k, rec in self.records.items():
                if k[0] != op:
                    continue
                src = (rec["metrics"] if best_name == "amax"
                       else rec["alts"][best_name])
                rec["metrics"] = [(1.0 - m) if sign < 0.0 else m for m in src]
            chosen[op] = {"metric": best_name, "sign": sign,
                          "rho_signed": abs(best_rho), "rho_amax": base}
        self.dispatch_metric_selection = chosen
        return chosen

    def _gated_mac_avg(self, lam: float, ladder_keep: dict) -> float:
        """MAC-weighted mean stream length INCLUDING escaped rows.

        Reproduces exactly what the runtime does: per group, take the
        Lagrangian argmin on that group's own ladder, then override every row
        whose normalized metric exceeds ``mu_g + k*sigma_g`` to
        ``escape_stoc_len``. mu/sigma are the same per-bucket statistics
        ``_fit_group`` exports and ``AdaptiveMPConfig`` reads back, so the
        threshold here IS the deployed threshold.
        """
        num = den = 0.0
        for key in self.records:
            metrics = np.concatenate(self.records[key]["metrics"], axis=0)
            errors = np.concatenate(self.records[key]["errors"], axis=0)
            if metrics.size == 0:
                continue
            err_obj = _objective_errors(errors, self.objective)
            costs = self.costs
            keep = ladder_keep.get(key)
            if keep is not None:
                costs = self.costs[keep]
                err_obj = err_obj[:, keep]
            assign = ((self._weight_for_key(key) * err_obj)
                      + lam * self._price_for_key(key) * costs[None, :]
                      ).argmin(axis=1)
            cost_row = costs[assign].astype(np.float64)
            t_esc = float(metrics.mean()) + self.escape_gate_k * float(metrics.std())
            if t_esc < 1.0:
                cost_row = np.where(metrics > t_esc,
                                    float(self.escape_stoc_len), cost_row)
            R = self._weight_count(key, metrics.size)
            num += R * float(cost_row.mean())
            den += R
        return (num / den) if den else 0.0

    def _global_protected_fraction(self) -> float:
        """MAC-weighted share of the model that runs on protected channels.

        Protected channels are a slice of the CONTRACTION dim, so a module with
        ``n_prot`` of ``D`` input channels spends ``n_prot/D`` of its MACs at
        the protected stream length. Weighting by each group's true MACs turns
        the per-module fractions into the one number the budget compensation
        needs.
        """
        num = den = 0.0
        for key, macs in self.true_macs.items():
            op, _t, l_bucket = key
            fracs = [len(v) / max(self.protected_channel_dims.get((o, b), 1), 1)
                     for (o, b), v in self.protected_channel_indices.items()
                     if o == op
                     and _bucket_index(b, self.total_blocks,
                                       self.layer_buckets) == l_bucket]
            if not fracs:
                den += float(macs)
                continue
            num += float(macs) * (sum(fracs) / len(fracs))
            den += float(macs)
        return (num / den) if den > 0 else 0.0

    # -- solve + export ------------------------------------------------------
    def _fit_group(self, metrics: np.ndarray, errors: np.ndarray,
                   lam: Optional[float] = None, rep: float = 1.0,
                   weight: float = 1.0,
                   keep: Optional[np.ndarray] = None,
                   budget_avg: Optional[float] = None) -> dict:
        # The ASSIGNMENT is solved in the objective currency; every reported
        # error stays raw relative-L2 so summaries stay comparable across
        # --objective settings.
        err_obj = _objective_errors(errors, self.objective)
        # ``keep`` restricts this group to its OWN ladder (per-group ladders).
        # Slicing costs AND errors together keeps rung r's sigma attached to
        # rung r's cycle count; the exported "stoc_len_levels" then tells the
        # runtime which rungs these thresholds were fitted against.
        costs = self.costs
        levels = self.levels
        if keep is not None:
            costs = self.costs[keep]
            levels = [self.levels[int(i)] for i in keep]
            err_obj = err_obj[:, keep]
            errors = errors[:, keep]
        if lam is None:
            avg = (float(budget_avg) if budget_avg is not None
                   else self.budget_ratio * self.budget_ref_stoc_len)
            budget_total = avg * metrics.size
            assignment = _cost_assignments(err_obj, costs, budget_total)
        else:
            objective = (weight * err_obj) + lam * rep * costs[None, :]
            assignment = objective.argmin(axis=1)
        counts = np.bincount(assignment, minlength=len(levels))
        thresholds = _thresholds_from_counts(metrics, counts)
        threshold_fit = None
        if self.dispatch_unit == "chunk" and self.chunk_threshold_fit == "ordered":
            from fit_group_thresholds import fit_ordered_thresholds
            thresholds, assignment, threshold_fit = fit_ordered_thresholds(
                metrics, err_obj, levels, thresholds)
            counts = np.bincount(assignment, minlength=len(levels))
        avg_cost = float(costs[assignment].mean()) if assignment.size else 0.0
        avg_error = (float(errors[np.arange(errors.shape[0]), assignment].mean())
                     if assignment.size else 0.0)
        out = {
            "num_units": int(metrics.size),
            "counts": counts.tolist(),
            "fractions": (counts / max(metrics.size, 1)).tolist(),
            "thresholds": thresholds,
            "avg_stoc_len": avg_cost,
            "avg_error": avg_error,
            "level_mean_error": [float(errors[:, i].mean())
                                 for i in range(errors.shape[1])],
            "metric_mean": float(metrics.mean()) if metrics.size else 0.0,
            "metric_std": float(metrics.std()) if metrics.size else 0.0,
        }
        if keep is not None:
            out["stoc_len_levels"] = [int(x) for x in levels]
        if threshold_fit is not None:
            out["threshold_fit"] = threshold_fit
        return out

    def export(self):
        summary_rows: list = []
        payload = {
            "stoc_len_levels": self.levels,
            "budget_ratio": self.budget_ratio,
            "budget_ref_stoc_len": self.budget_ref_stoc_len,
            "timestep_buckets": self.timestep_buckets,
            "layer_buckets": self.layer_buckets,
            "budget_scope": self.budget_scope,
            "budget_weight": self.budget_weight,
            "argmin_pricing": self.argmin_pricing,
            "objective": self.objective,
            "cross_layer_weight": self.cross_layer_weight,
            "operator_defaults": {},
            "buckets": {},
        }
        if self.protected_channel_indices:
            # Protected channels are charged at protect_stoc_len for EVERY row,
            # so they raise the realized average above the nominal target
            # unless the residual budget is lowered to pay for them. p is the
            # MAC-weighted protected fraction; the residual target solves
            #   p*protect_sl + (1-p)*rest = target.
            p = self._global_protected_fraction()
            target = float(self.budget_ratio) * float(self.budget_ref_stoc_len)
            protected = {
                "stoc_len": int(self.protect_channel_stoc_len),
                "frac": float(self.protect_channel_frac),
                "metric": self.protect_channel_metric,
                "compensate_budget": bool(self.protect_compensate_budget),
                "global_mac_weighted_frac": float(p),
                "indices": {f"{op}:b{b}": [int(i) for i in v]
                            for (op, b), v in
                            sorted(self.protected_channel_indices.items())},
                "dims": {f"{op}:b{b}": int(v) for (op, b), v in
                         sorted(self.protected_channel_dims.items())},
            }
            if self.protect_compensate_budget and p > 0.0:
                rest = (target - p * float(self.protect_channel_stoc_len)) / max(
                    1.0 - p, 1e-12)
                if rest <= 0.0:
                    raise SystemExit(
                        f"[calib] protected-channel compensation produced a "
                        f"non-positive residual target {rest:.4f}: the "
                        f"protected fraction ({p:.4f}) at stoc_len "
                        f"{self.protect_channel_stoc_len} already exceeds the "
                        f"whole budget {target:.2f}. Lower "
                        f"--protect-channel-frac or --protect-channel-stoc-len.")
                protected["original_budget_ratio"] = float(self.budget_ratio)
                protected["residual_target_avg_stoc_len"] = float(rest)
                self.budget_ratio = float(rest) / float(self.budget_ref_stoc_len)
                protected["residual_budget_ratio"] = float(self.budget_ratio)
                payload["budget_ratio"] = self.budget_ratio
            payload["protected_channels"] = protected

        if self.bucket_budget_ref is not None:
            return self._export_matched_to_reference(payload, summary_rows)

        global_lam = None
        ladder_keep: dict = {}
        if self.budget_scope == "global":
            keys = list(self.records.keys())
            werrs, row_counts, prices = [], [], []
            for key in keys:
                rec = self.records[key]
                e = _objective_errors(
                    np.concatenate(rec["errors"], axis=0), self.objective)
                werrs.append(self._weight_for_key(key) * e)
                row_counts.append(self._weight_count(key, e.shape[0]))
                prices.append(self._price_for_key(key))
            global_lam = _global_lambda(
                werrs, self.costs, self.budget_ratio,
                float(self.budget_ref_stoc_len), row_counts, prices=prices)

            # ---- per-group ladders: pass 1 selects, pass 2 re-prices --------
            # Pass 1's lambda is solved against the DENSE ladder. Restricting a
            # group to k rungs removes options from its argmin, so its realized
            # cost moves and the budget no longer holds at that lambda. Pass 2
            # re-solves ONE shared lambda against the per-group ladders that
            # will actually be deployed, which is what puts the realized
            # MAC-weighted average back on target.
            if 0 < self.group_ladder_k < len(self.levels):
                group_costs = []
                for key, e in zip(keys, werrs):
                    raw = _objective_errors(
                        np.concatenate(self.records[key]["errors"], axis=0),
                        self.objective)
                    keep = _select_group_ladder(
                        raw, self.costs, global_lam,
                        self._price_for_key(key), self._weight_for_key(key),
                        self.group_ladder_k)
                    ladder_keep[key] = keep
                    group_costs.append(self.costs[keep])
                werrs2 = [w[:, ladder_keep[k]] for w, k in zip(werrs, keys)]
                global_lam = _global_lambda_multi(
                    werrs2, group_costs, self.budget_ratio,
                    float(self.budget_ref_stoc_len), row_counts, prices)
                payload["group_ladder_k"] = int(self.group_ladder_k)
                payload["group_ladders"] = {
                    f"{op}:l{l}": [int(self.levels[int(i)])
                                   for i in ladder_keep[(op, t, l)]]
                    for (op, t, l) in sorted(ladder_keep)}
            # ---- escape-gate budget compensation ---------------------------
            # The gate fires at RUNTIME, after the allocator has spent the
            # budget, so an ungated solve overshoots by whatever the gate
            # takes (measured: 13-23% of rows on ViT, worth ~+2.6% realized
            # cycles at k=2). Shrink the target until the GATED average lands
            # on the nominal budget, so a gated cell is iso-compute with an
            # ungated one instead of quietly buying accuracy with cycles.
            if self.escape_gate_k is not None:
                target = float(self.budget_ratio) * float(self.budget_ref_stoc_len)
                ratio = float(self.budget_ratio)
                gated = self._gated_mac_avg(global_lam, ladder_keep)
                payload["ungated_gated_avg_stoc_len"] = float(gated)
                for _ in range(8):
                    if gated <= 0 or abs(gated - target) <= 0.005 * target:
                        break
                    ratio *= target / gated
                    lam_try = _global_lambda(
                        werrs, self.costs, ratio,
                        float(self.budget_ref_stoc_len), row_counts,
                        prices=prices)
                    if ladder_keep:
                        lam_try = _global_lambda_multi(
                            [w[:, ladder_keep[k]] for w, k in zip(werrs, keys)],
                            [self.costs[ladder_keep[k]] for k in keys],
                            ratio, float(self.budget_ref_stoc_len),
                            row_counts, prices)
                    global_lam = lam_try
                    gated = self._gated_mac_avg(global_lam, ladder_keep)
                self.budget_ratio = ratio
                payload["budget_ratio"] = ratio
                payload["escape_gate_k"] = self.escape_gate_k
                payload["escape_stoc_len"] = int(self.escape_stoc_len)
                payload["escape_compensated_budget_ratio"] = float(ratio)
                payload["expected_gated_avg_stoc_len"] = float(gated)
                print(f"[calib] escape gate k={self.escape_gate_k}: "
                      f"compensated budget_ratio -> {ratio:.4f}, "
                      f"gated MAC-avg {gated:.2f} (target {target:.2f})",
                      flush=True)

            payload["global_lambda"] = float(global_lam)
            if self.cross_layer_weight != "uniform":
                payload["group_weights"] = {
                    f"{op}:l{l}": self._weight_for_key((op, t, l))
                    for (op, t, l) in sorted(self.records)}

        # Operator-level fallback (concat all layer buckets per op).
        for operator in sorted(self.operators):
            op_metrics, op_errors = [], []
            for (op, _, _), rec in self.records.items():
                if op != operator:
                    continue
                op_metrics.extend(rec["metrics"])
                op_errors.extend(rec["errors"])
            if op_metrics:
                op_err = np.concatenate(op_errors, axis=0)
                op_keep = None
                if ladder_keep:
                    # The operator default is the fallback for any bucket the
                    # table has no entry for, so it gets its own pooled ladder
                    # rather than inheriting the dense one — otherwise a bucket
                    # miss would silently deploy k dense rungs where every real
                    # bucket deploys k selected ones.
                    op_keep = _select_group_ladder(
                        _objective_errors(op_err, self.objective), self.costs,
                        global_lam, self._price_for_op(operator),
                        self._op_weight(operator), self.group_ladder_k)
                fitted = self._fit_group(
                    np.concatenate(op_metrics, axis=0), op_err,
                    lam=global_lam, rep=self._price_for_op(operator),
                    weight=self._op_weight(operator), keep=op_keep,
                )
                payload["operator_defaults"][operator] = fitted
                summary_rows.append({
                    "scope": "operator_default", "operator": operator,
                    "t_bucket": -1, "l_bucket": -1,
                    **_flatten_summary(fitted),
                })

        # Per-bucket entries.
        for key in sorted(self.records.keys()):
            metrics = np.concatenate(self.records[key]["metrics"], axis=0)
            errors = np.concatenate(self.records[key]["errors"], axis=0)
            if metrics.size < self.min_bucket_units:
                continue
            operator, t_bucket, l_bucket = key
            fitted = self._fit_group(metrics, errors, lam=global_lam,
                                     rep=self._price_for_key(key),
                                     weight=self._weight_for_key(key),
                                     keep=ladder_keep.get(key))
            bucket_key = f"{operator}:t{t_bucket}:l{l_bucket}"
            payload["buckets"][bucket_key] = fitted
            summary_rows.append({
                "scope": "bucket", "operator": operator,
                "t_bucket": t_bucket, "l_bucket": l_bucket,
                **_flatten_summary(fitted),
            })

        # Per-(row, chunk) dispatch: the same fitted buckets, re-emitted in the
        # kernel's per_row_chunk schema (ASCENDING levels and thresholds; vit's
        # thresholds are the descending-order cuts, metric above th[0] ->
        # levels[0], so both simply reverse). The per-row "buckets" entries
        # for these operators stay as the fallback / for the budget summary;
        # the runtime prefers per_row_chunk when it is present.
        if getattr(self, "dispatch_unit", "row") == "chunk" and self.chunk_unit_ops:
            prc_buckets = {}
            for bucket_key, fitted in payload["buckets"].items():
                if bucket_key.split(":")[0] not in self.chunk_unit_ops:
                    continue
                lv = [int(x) for x in fitted.get("stoc_len_levels", self.levels)]
                th = [float(x) for x in fitted["thresholds"]]
                if any(v <= 0 for v in lv):
                    raise SystemExit("[calib] per_row_chunk has no pruned (0) rung")
                prc_buckets[bucket_key] = {"levels": lv[::-1],
                                           "thresholds": th[::-1]}
            payload["per_row_chunk"] = {"buckets": prc_buckets}
            payload["dispatch_unit"] = "chunk"
            payload["chunk_unit_ops"] = sorted(self.chunk_unit_ops)

        # Predicted realized averages. Row-weighted (diagnostic) + MAC-weighted
        # (the true iso-compute check — must land near budget_ratio*ref).
        num = den = fnum = fden = 0.0
        for bkey, fitted in payload["buckets"].items():
            operator, _t, lpart = bkey.split(":")
            l = int(lpart[1:])
            key = (operator, 0, l)
            R = float(self.true_counts.get(key, fitted["num_units"]))
            Rm = float(self.true_macs.get(key, R))
            avg = float(fitted["avg_stoc_len"])
            num += R * avg
            den += R
            fnum += Rm * avg
            fden += Rm
        payload["expected_avg_stoc_len"] = (num / den) if den else 0.0
        payload["expected_flop_avg_stoc_len"] = (fnum / fden) if fden else 0.0
        # Informational: mean MACs/row per bucket (static for CNNs).
        payload["mac_per_row"] = {
            f"{op}:l{l}": self.true_macs[(op, 0, l)] / max(self.true_counts[(op, 0, l)], 1)
            for (op, _t, l) in self.records
        }
        return payload, summary_rows


def _export_matched_to_reference(self, payload: dict, summary_rows: list):
    """--bucket-budget-from: hold the reference table's per-(op, layer-bucket)
    average stream length and its non-chunk operators' entries fixed; re-solve
    only the chunk-unit operators' within-bucket assignment over (row, chunk)
    pairs. Iso-compute per bucket by construction (up to the tail chunk being
    priced at the mean width), and the attention dispatch is byte-identical to
    the reference's."""
    ref = self.bucket_budget_ref
    ref_buckets = ref.get("buckets", {})
    ref_defaults = ref.get("operator_defaults", {})
    payload["budget_matched_to"] = ref.get("calib_command", "<reference>")
    payload["budget_ratio"] = ref.get("budget_ratio", self.budget_ratio)
    if "protected_channels" in ref and "protected_channels" in payload:
        # same selection (verified by the caller's diff); carry the reference's
        # compensation bookkeeping so downstream summaries read the same way.
        for k in ("original_budget_ratio", "residual_target_avg_stoc_len",
                  "residual_budget_ratio"):
            if k in ref["protected_channels"]:
                payload["protected_channels"][k] = ref["protected_channels"][k]
    chunk_ops = set(self.chunk_unit_ops)
    # non-chunk operators: verbatim from the reference. If this table's ladder
    # differs from the reference's (e.g. a wider pair ladder), pin the copied
    # entries to the REFERENCE ladder per bucket so their thresholds keep
    # meaning the rungs they were fitted against.
    ref_levels = [int(x) for x in ref.get("stoc_len_levels", self.levels)]
    pin = ref_levels != [int(x) for x in self.levels]
    def _copy(entry):
        entry = dict(entry)
        if pin and "stoc_len_levels" not in entry:
            entry["stoc_len_levels"] = list(ref_levels)
        return entry
    for operator in sorted(self.operators):
        if operator in chunk_ops:
            continue
        if operator in ref_defaults:
            payload["operator_defaults"][operator] = _copy(ref_defaults[operator])
        for bkey, fitted in ref_buckets.items():
            if bkey.split(":")[0] == operator:
                payload["buckets"][bkey] = _copy(fitted)
    if pin:
        payload["reference_stoc_len_levels"] = ref_levels
    # chunk operators: per-bucket solve to the reference bucket's average
    for operator in sorted(chunk_ops):
        op_metrics, op_errors = [], []
        for (op, _, _), rec in self.records.items():
            if op == operator:
                op_metrics.extend(rec["metrics"]); op_errors.extend(rec["errors"])
        if op_metrics and operator in ref_defaults:
            fitted = self._fit_group(
                np.concatenate(op_metrics, axis=0), np.concatenate(op_errors, axis=0),
                lam=None, budget_avg=float(ref_defaults[operator]["avg_stoc_len"]))
            payload["operator_defaults"][operator] = fitted
            summary_rows.append({"scope": "operator_default", "operator": operator,
                                 "t_bucket": -1, "l_bucket": -1, **_flatten_summary(fitted)})
    n_matched = 0
    for key in sorted(self.records.keys()):
        operator, t_bucket, l_bucket = key
        if operator not in chunk_ops:
            continue
        bucket_key = f"{operator}:t{t_bucket}:l{l_bucket}"
        if bucket_key not in ref_buckets:
            raise SystemExit(f"[calib] reference table has no bucket {bucket_key}")
        metrics = np.concatenate(self.records[key]["metrics"], axis=0)
        errors = np.concatenate(self.records[key]["errors"], axis=0)
        fitted = self._fit_group(metrics, errors, lam=None,
                                 budget_avg=float(ref_buckets[bucket_key]["avg_stoc_len"]))
        fitted["reference_avg_stoc_len"] = float(ref_buckets[bucket_key]["avg_stoc_len"])
        payload["buckets"][bucket_key] = fitted
        summary_rows.append({"scope": "bucket", "operator": operator,
                             "t_bucket": t_bucket, "l_bucket": l_bucket,
                             **_flatten_summary(fitted)})
        n_matched += 1
    print(f"[calib] budget matched to reference: {n_matched} chunk-unit buckets "
          f"re-solved at their reference avg_stoc_len; "
          f"{sum(1 for k in payload['buckets'] if k.split(':')[0] not in chunk_ops)} "
          f"buckets copied verbatim", flush=True)
    # per_row_chunk section for the chunk operators (ascending schema)
    prc_buckets = {}
    for bucket_key, fitted in payload["buckets"].items():
        if bucket_key.split(":")[0] not in chunk_ops:
            continue
        lv = [int(x) for x in fitted.get("stoc_len_levels", self.levels)]
        th = [float(x) for x in fitted["thresholds"]]
        prc_buckets[bucket_key] = {"levels": lv[::-1], "thresholds": th[::-1]}
    payload["per_row_chunk"] = {"buckets": prc_buckets,
                               "strict": self.chunk_threshold_fit == "ordered",
                               "operators": sorted(chunk_ops)}
    payload["chunk_threshold_fit"] = self.chunk_threshold_fit
    payload["dispatch_unit"] = "chunk"
    payload["chunk_unit_ops"] = sorted(chunk_ops)
    # predicted averages: chunk ops from this fit (true_macs), reference ops
    # from the reference's own bookkeeping
    num = den = fnum = fden = 0.0
    ref_macs = ref.get("mac_per_row", {})
    for bkey, fitted in payload["buckets"].items():
        operator, _t, lpart = bkey.split(":")
        l = int(lpart[1:]); key = (operator, 0, l)
        if operator in chunk_ops:
            R = float(self.true_counts.get(key, fitted["num_units"]))
            Rm = float(self.true_macs.get(key, R))
        else:
            R = float(fitted.get("num_units", 0))
            Rm = R * float(ref_macs.get(f"{operator}:l{l}", 1.0))
        avg = float(fitted["avg_stoc_len"])
        num += R * avg; den += R; fnum += Rm * avg; fden += Rm
    payload["expected_avg_stoc_len"] = (num / den) if den else 0.0
    payload["expected_flop_avg_stoc_len"] = (fnum / fden) if fden else 0.0
    payload["mac_per_row"] = {
        f"{op}:l{l}": self.true_macs[(op, 0, l)] / max(self.true_counts[(op, 0, l)], 1)
        for (op, _t, l) in self.records}
    return payload, summary_rows


ThresholdCalibrator._export_matched_to_reference = _export_matched_to_reference


def _flatten_summary(fitted: dict) -> dict:
    row = {
        "num_units": fitted["num_units"],
        "avg_stoc_len": fitted["avg_stoc_len"],
        "avg_error": fitted["avg_error"],
        "metric_mean": fitted["metric_mean"],
        "metric_std": fitted["metric_std"],
    }
    row["counts"] = ",".join(str(int(x)) for x in fitted["counts"])
    row["fractions"] = ",".join(f"{float(x):.6f}" for x in fitted["fractions"])
    row["thresholds"] = ",".join(f"{float(x):.6f}" for x in fitted["thresholds"])
    row["level_mean_error"] = ",".join(
        f"{float(x):.6f}" for x in fitted["level_mean_error"])
    return row


def _write_summary_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




# =====================================================================
# IRASim adapters: calibration hooks (scmp_vit hook bodies, functional form)
# =====================================================================


def _make_attn_hook(calibrator, levels, sc_prec, halve, qk_granularity, every_k):
    """Record qk / av calibration data at the granularity the runtime
    dispatches at (scmp_vit ``make_calib_attention_forward``):

    * qk ``per_row``: unit = one query row, pooled globally over B*H*N;
      ``per_head``: unit = one head pooled over the batch.
    * av: unit = one attention row, pooled globally over B*H*N.

    The forward itself continues through the normal SC path at the uniform
    calibration stream length, so downstream blocks see the deployed noisy
    trajectory (``--calib-on-sc``).
    """
    def sc(a3, b3, gran, sl):
        return sc_matmul(a3, b3, granularity=gran, mode="bipolar", group_a=1,
                         group_b=1, sc_prec=sc_prec, stoc_len=int(sl),
                         halve_bipolar_stoc_len=halve)

    def hook(op, block_idx, a3, b3):
        if (block_idx is None or not calibrator.use_operator(op)
                or not _recording_step(every_k)):
            return
        BH, N, K = a3.shape
        M = b3.shape[1]
        with torch.no_grad():
            teacher = torch.matmul(a3, b3.transpose(-2, -1))
            gran = qk_granularity if op == "qk" else "per_row"
            if gran == "per_row":
                rows2 = a3.reshape(BH * N, K)
                metric = _normalize_metric(rows2.abs().amax(dim=-1))
                alts = (_metric_candidates(rows2)
                        if calibrator.metric_select == "auto" else None)
                t2 = teacher.reshape(BH * N, M)
                tnorm = t2.norm(dim=-1).clamp_min(1e-8)
                errs = []
                for sl in levels:
                    s2 = (sc(a3, b3, gran, sl).reshape(BH * N, M) if sl > 0
                          else torch.zeros_like(t2))
                    errs.append((s2 - t2).norm(dim=-1) / tnorm)
                calibrator.add(op, block_idx, metric, errs,
                               mac_per_row=float(M) * K, alts=alts)
            else:
                # Legacy per-head qk: the kernel's per_head range is one
                # (max, min) per (b*h) slice; IRASim's B already folds frames
                # or tokens, so heads are pooled over that leading dim.
                H = calibrator.n_heads
                q4 = a3.reshape(-1, H, N, K)
                metric = _normalize_metric(q4.abs().amax(dim=(0, 2, 3)))
                t4 = teacher.reshape(-1, H, N, M)
                tnorm = t4.permute(1, 0, 2, 3).reshape(H, -1).norm(dim=-1).clamp_min(1e-8)
                errs = []
                for sl in levels:
                    e = (sc(a3, b3, gran, sl).reshape(-1, H, N, M) - t4 if sl > 0 else -t4)
                    errs.append(e.permute(1, 0, 2, 3).reshape(H, -1).norm(dim=-1) / tnorm)
                calibrator.add(op, block_idx, metric, errs,
                               mac_per_row=float(q4.shape[0]) * N * M * K, alts=None)
    return hook


def _make_linear_hook(calibrator, levels, sc_prec, halve, chunk_d, every_k):
    """Record, per SC linear call: the per-row dispatch metric, the per-level SC
    reconstruction error, and the layer's MACs/row (scmp_vit hook body; the
    operands arrive SmoothQuant-transformed, so the teacher ``x2 @ w.T + b``
    equals the layer's FP output)."""
    def hook(op, block_idx, x2, w, bias, mac_per_row):
        if (op is None or block_idx is None or not calibrator.use_operator(op)
                or not _recording_step(every_k) or x2.shape[0] == 0):
            return
        with torch.no_grad():
            xf = x2.to(torch.float32).contiguous()
            w = w.to(torch.float32).contiguous()
            teacher = F.linear(xf, w, bias)

            prot = calibrator.protected_channel_indices.get((op, block_idx))
            if prot:
                pi = torch.as_tensor(prot, dtype=torch.long, device=xf.device)
                mask = torch.ones(xf.shape[1], dtype=torch.bool, device=xf.device)
                mask[pi] = False
                ri = mask.nonzero(as_tuple=True)[0]
                x_p, w_p = xf.index_select(1, pi).contiguous(), w.index_select(1, pi).contiguous()
                x_r, w_r = xf.index_select(1, ri).contiguous(), w.index_select(1, ri).contiguous()
                base = sc_matmul(
                    x_p, w_p, granularity="per_row", mode="bipolar",
                    sc_prec=sc_prec,
                    stoc_len=int(calibrator.protect_channel_stoc_len),
                    chunk_d=(chunk_d if (chunk_d > 0
                                         and x_p.shape[-1] > chunk_d) else 0),
                    halve_bipolar_stoc_len=halve)
            else:
                x_r, w_r, base = xf, w, None

            if getattr(calibrator, "dispatch_unit", "row") == "chunk":
                N_r, D_r = x_r.shape
                nch = (D_r + chunk_d - 1) // chunk_d
                pad = nch * chunk_d - D_r
                xa = x_r.abs()
                if pad:
                    xa = F.pad(xa, (0, pad))
                m = xa.view(N_r, nch, chunk_d).amax(-1)
                lo_m, hi_m = m.min(), m.max()
                metric = ((m - lo_m) / (hi_m - lo_m).clamp_min(1e-8)).reshape(-1)
                denom = teacher.norm(dim=-1).clamp_min(1e-8)
                per_level = [torch.zeros(N_r, nch, device=xf.device)
                             for _ in levels]
                for c in range(nch):
                    c0, c1 = c * chunk_d, min((c + 1) * chunk_d, D_r)
                    xc = x_r[:, c0:c1].contiguous()
                    wc = w_r[:, c0:c1].contiguous()
                    fpc = xc @ wc.t()
                    for li, sl in enumerate(levels):
                        if sl <= 0:
                            err = fpc.norm(dim=-1)
                        else:
                            scc = sc_matmul(
                                xc, wc, granularity="per_row", mode="bipolar",
                                sc_prec=sc_prec, stoc_len=int(sl), chunk_d=0,
                                halve_bipolar_stoc_len=halve)
                            err = (scc - fpc).norm(dim=-1)
                        per_level[li][:, c] = err / denom
                level_errors = [e.reshape(-1) for e in per_level]
                mac_scale = float(D_r) / float(max(xf.shape[1], 1))
                calibrator.add(op, block_idx, metric, level_errors,
                               mac_per_row=mac_per_row * mac_scale / nch,
                               alts=None)
                calibrator.chunk_unit_ops.add(op)
                return

            metric = _normalize_metric(x_r.abs().amax(dim=-1))
            alts = (_metric_candidates(x_r)
                    if calibrator.metric_select == "auto" else None)
            level_errors = []
            for sl in levels:
                if sl <= 0:
                    sc_rows = (torch.zeros_like(teacher) if base is None
                               else base + (bias if bias is not None else 0))
                else:
                    sc_rows = sc_matmul(
                        x_r, w_r,
                        granularity="per_row", mode="bipolar",
                        sc_prec=sc_prec, stoc_len=int(sl),
                        chunk_d=(chunk_d if (chunk_d > 0
                                             and x_r.shape[-1] > chunk_d) else 0),
                        halve_bipolar_stoc_len=halve,
                    )
                    if base is not None:
                        sc_rows = sc_rows + base
                    if bias is not None:
                        sc_rows = sc_rows + bias
                level_errors.append(_relative_l2_rows(sc_rows, teacher))
            mac_scale = float(x_r.shape[1]) / float(max(xf.shape[1], 1))
            calibrator.add(op, block_idx, metric, level_errors,
                           mac_per_row=mac_per_row * mac_scale,
                           alts=alts)
    return hook


# =====================================================================
# Args + main
# =====================================================================


def _expand_levels(value: str) -> list:
    levels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 stoc_len levels, got {levels}")
    for i in range(len(levels) - 1):
        if levels[i] <= levels[i + 1]:
            raise ValueError(f"Levels must be strictly descending, got {levels}")
    return levels


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="IRASim eval config with an SC attention_mode "
                        "(e.g. configs/evaluation/bridge/frame_ada_sc_full.yaml).")
    p.add_argument("--skip", default="",
                   help="FP-kept (op, block) groups, eval_local_n_samples syntax "
                        "('mlp_fc1=4,25;qkv=3'). MUST match the eval --skip.")
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--scheduler", choices=["PNDM", "DPM"], default="PNDM")
    p.add_argument("--record_every_k_steps", type=int, default=5,
                   help="Record calibration units on every k-th denoising step "
                        "(the forward still runs every step). Uniform over the "
                        "trajectory, so the single timestep bucket is unbiased.")
    p.add_argument("--mp_levels", required=True,
                   help="Descending comma-separated stoc_len levels "
                        "(halved space when --halve 1), e.g. '128,96,64,48'.")
    p.add_argument("--budget_ratio", type=float, default=0.5,
                   help="Target avg stoc_len / budget_ref_stoc_len.")
    p.add_argument("--budget_ref_stoc_len", type=int, default=None,
                   help="Reference stoc_len. Defaults to max(mp_levels) — so "
                        "the ratio is relative to THIS ladder's top rung, not "
                        "an absolute scale.")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--halve", type=int, default=1,
                   help="halve_bipolar_stoc_len. Must match deployment.")
    p.add_argument("--chunk_d", type=int, default=128,
                   help="chunk_d for SC linear calls. Must match deployment; "
                        "0 collapses wide layers (scmp_vit: 0% top-1).")
    p.add_argument("--operators", default=DEFAULT_OPERATORS)
    p.add_argument("--attn_stoc_len", type=int, default=0,
                   help="Uniform stoc_len every SC op runs at during the "
                        "calibration forward (and qk/av when they are not in "
                        "--operators). 0 -> the deployment average.")
    p.add_argument("--n_calib_samples", type=int, default=4,
                   help="Generated videos to calibrate on (one sampler run each).")
    p.add_argument("--calib_seed", type=int, default=1234,
                   help="Shuffle seed for the calibration key pool.")
    p.add_argument("--calib_offset", type=int, default=0,
                   help="Start offset into the shuffled calibration pool.")
    p.add_argument("--eval_keys_file", default="results/diverse_300.json",
                   help="Keys PSNR/SSIM is reported on; removed from the "
                        "calibration pool. Empty disables the exclusion.")
    p.add_argument("--protect-channel-stat-samples",
                   dest="protect_channel_stat_samples", type=int, default=1,
                   help="Videos (FP trajectory) for the per-channel activation-"
                        "amax pre-pass that selects protected channels.")
    p.add_argument("--layer_buckets", type=int, default=0,
                   help="0 (default) = one bucket per block.")
    p.add_argument("--max_units_per_call", type=int, default=512)
    p.add_argument("--min_bucket_units", type=int, default=64)
    p.add_argument("--dispatch-unit", dest="dispatch_unit",
                   choices=("row", "chunk"), default="row",
                   help="Dispatch granularity for the LINEAR operators. 'row' "
                        "(default): one stream length per input row, the "
                        "historical per-row MP. 'chunk': one length per "
                        "(row, chunk_d-wide chunk) pair of the residual input; "
                        "the metric is the pair's own quantization scale "
                        "(absmax), the error is the chunk's SC partial-product "
                        "error in units of the row's teacher norm, and the "
                        "table gains a 'per_row_chunk' section that "
                        "model.sc_vit.SCLinear dispatches through the kernel "
                        "rung table. qk/av stay per-row.")
    p.add_argument("--bucket-budget-from", dest="bucket_budget_from", default="",
                   help="Path to a REFERENCE calibrated table. Every (op, layer "
                        "bucket) of the operators recorded at the chunk unit is "
                        "solved to the reference bucket's avg_stoc_len (per-bucket "
                        "budget), and every OTHER operator's bucket / "
                        "operator_default entry is copied from the reference "
                        "verbatim. Isolates the dispatch-granularity change: the "
                        "per-(op, block) compute split and the attention dispatch "
                        "are exactly the reference's; only the within-bucket "
                        "distribution over (row, chunk) pairs is re-solved. "
                        "Requires --dispatch-unit chunk.")
    p.add_argument("--chunk-threshold-fit", choices=("counts", "ordered"), default="counts",
                   help="ordered: improve deployed per-group thresholds at the same "
                        "sampled cycle budget; requires chunk dispatch and bucket-budget-from")
    p.add_argument("--budget-scope", dest="budget_scope",
                   choices=["per_bucket", "global"], default="global")
    p.add_argument("--group-ladder-k", dest="group_ladder_k", type=int,
                   default=0,
                   help="Per-group ladders. 0 (default) = off: every bucket "
                        "shares the --mp_levels ladder. k>0 = calibrate sigma "
                        "on the FULL (dense) --mp_levels ladder but deploy only "
                        "the k rungs each (op x layer-bucket) actually puts "
                        "mass on, then re-solve one shared lambda against those "
                        "ladders so the budget still holds. Use a dense "
                        "--mp_levels (8-12 rungs) with k=3 or 4.")
    p.add_argument("--escape-gate-k", dest="escape_gate_k", type=float,
                   default=None,
                   help="Absolute escape gate: rows whose NORMALIZED dispatch "
                        "metric exceeds mu_b + k*sigma_b run at "
                        "--escape-stoc-len regardless of the band thresholds. "
                        "mu/sigma are the per-bucket metric stats this "
                        "calibrator already exports, so the gate is a wrapper-"
                        "only setting. Unset (default) = off.")
    p.add_argument("--escape-stoc-len", dest="escape_stoc_len", type=int,
                   default=0,
                   help="Stream length escaped rows run at. 0 -> the ladder "
                        "ceiling (max of --mp_levels).")
    p.add_argument("--protect-channel-frac", dest="protect_channel_frac",
                   type=float, default=0.0,
                   help="Fraction of INPUT channels per linear to pin at "
                        "--protect-channel-stoc-len for every row, split off "
                        "along the contraction dim (exact: a matmul is "
                        "additive over that dim). 0 (default) = off. On ViT "
                        "the target is mlp_fc2, whose post-GELU rows carry "
                        "sparse outliers over a near-zero bulk.")
    p.add_argument("--protect-channel-metric", dest="protect_channel_metric",
                   choices=["act", "act_weight", "weight"], default="act",
                   help="Channel salience score. 'act' (default): the "
                        "per-channel activation amax — the channel that SETS "
                        "the shared per-row quantization scale. 'act_weight': "
                        "a_c^2*||W[:,c]||^2, the output-energy proxy.")
    p.add_argument("--protect-channel-stoc-len",
                   dest="protect_channel_stoc_len", type=int, default=0,
                   help="Stream length for protected channels. 0 -> the "
                        "ladder ceiling.")
    p.add_argument("--protect-compensate-budget",
                   dest="protect_compensate_budget", type=int, default=1,
                   help="1 (default): lower the residual per-row budget so the "
                        "MAC-weighted average still lands on the nominal "
                        "target once protected channels are charged. 0: let "
                        "protection overshoot the budget (NOT iso-compute).")
    p.add_argument("--qk_granularity", dest="qk_granularity",
                   choices=["per_head", "per_row"], default="per_row",
                   help="Quantization scope for Q@K^T, and hence the MP "
                        "dispatch unit. 'per_head' (legacy) gives only 16 "
                        "units per block on ViT-L/14. 'per_row' matches the "
                        "av convention and scmp_llm. MUST match the eval "
                        "invocation.")
    p.add_argument("--budget-weight", dest="budget_weight",
                   choices=["rows", "macs"], default="macs",
                   help="'macs' (default): iso-budget = iso-compute, the "
                        "shipped configuration. 'rows' reproduces the "
                        "row-weighted budget bug, for ablation only.")
    p.add_argument("--calib-on-sc", dest="calib_on_sc", type=int, default=1,
                   help="1 (default): calibration forward runs in SC at a "
                        "uniform deployment-average stoc_len, so metric and "
                        "sigma are measured on the noisy activation "
                        "distribution actually deployed. 0: FP trajectory "
                        "(ablation).")
    p.add_argument("--calib-sc-stoclen", dest="calib_sc_stoclen", type=int,
                   default=0,
                   help="stoc_len for the --calib-on-sc forward. 0 -> "
                        "round(budget_ratio * budget_ref_stoc_len).")
    p.add_argument("--metric-select", dest="metric_select",
                   default="amax", choices=["amax", "auto"],
                   help="act_global_v2 per-operator dispatch-metric selection: "
                        "'amax' (legacy, byte-identical) or 'auto' (pick the "
                        "signed amax/l2/crest candidate with best Spearman rho "
                        "vs sigma-benefit; exports 'dispatch_metrics').")
    p.add_argument("--metric-select-margin", dest="metric_select_margin",
                   type=float, default=0.05,
                   help="min |rho| improvement over amax required to switch.")
    p.add_argument("--argmin-pricing", dest="argmin_pricing",
                   default="rep", choices=["rep", "macs"],
                   help="Per-row cost pricing in the argmin. 'macs' was tested "
                        "and REJECTED on scmp_cnn (-17 pt with auto metrics); "
                        "keep 'rep' unless ablating. NOTE on a ViT this knob is "
                        "a NO-OP: every op sees the same rows/forward (257*B), "
                        "so rep_g and mac_per_row differ by one global constant "
                        "that lambda absorbs.")
    p.add_argument("--objective", dest="objective", default="sigma",
                   choices=["sigma", "sigma2", "delta_sigma2"],
                   help="Allocation objective (mp_objectives.py, shared with "
                        "scmp_llm). 'sigma' (default): minimize sum of "
                        "relative-L2 recon error — byte-identical to the "
                        "pre-2026-07-25 calibrator. 'sigma2': minimize sum of "
                        "sigma^2, a propagated-loss surrogate that penalizes "
                        "the low-precision cliff superlinearly. "
                        "'delta_sigma2': same, but priced on the INCREASE over "
                        "each row's error at the top rung, so irreducible error "
                        "at max stoc_len is not chased.")
    p.add_argument("--cross-layer-weight", dest="cross_layer_weight",
                   default="uniform",
                   choices=["uniform", "macs"],
                   help="Per-group importance weight W_g for --budget-scope "
                        "global. 'uniform' (default): W_g=1, i.e. act_global — "
                        "benefit counted per ROW while cost is per MAC, which "
                        "pins the cheap out_proj at the ladder ceiling and "
                        "starves mlp_fc1 (MP_REPORT.md). 'macs': W_g=MACs/row, "
                        "so benefit and cost share one currency and every "
                        "operator faces the same cycles-per-sigma exchange "
                        "rate. 'macs_measured': phase-2 — that, times a measured "
                        "per-group dLoss from a knock-down probe. 'measured': "
                        "dLoss alone, which RE-INTRODUCES the cost-side "
                        "favouritism macs removes (rep_g already carries mac_g) "
                        "— ablation only. Implies --budget-scope global.")
    p.add_argument("--variants", default="",
                   help="Export SEVERAL tables from ONE sigma collection, so an "
                        "objective/W_g comparison carries zero calibration-"
                        "sample noise. Comma-separated "
                        "'<objective>[:<uniform|macs>]' entries, or the literal "
                        "'all' for every combination. Each table is written to "
                        "<output_json stem>__<objective>_w<W_g>.json with its "
                        "own wrapper. Empty (default) = one table at "
                        "--output_json from --objective/--cross-layer-weight.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_summary_csv", default=None)
    return p





def main():
    args = _build_parser().parse_args()
    if args.chunk_threshold_fit == "ordered" and (
            args.dispatch_unit != "chunk" or not args.bucket_budget_from):
        raise SystemExit("--chunk-threshold-fit ordered requires --dispatch-unit chunk and --bucket-budget-from")

    # ---- pure-config validation, BEFORE the CUDA gate ----
    # A bad flag combination should be reportable from the login node; nothing
    # here touches a device, so there is no reason to make the user burn a Slurm
    # allocation to find out the ladder or the scope is wrong.
    levels = _expand_levels(args.mp_levels)
    halve = bool(args.halve)
    sc_prec = int(args.sc_prec)
    halved_cap = 2 ** (sc_prec - 1) if halve else 2 ** sc_prec
    if max(levels) > halved_cap:
        raise ValueError(
            f"max(mp_levels)={max(levels)} exceeds cap {halved_cap} "
            f"for sc_prec={sc_prec} halve={halve}.")
    if (args.cross_layer_weight != "uniform"
            and args.budget_scope != "global"):
        raise SystemExit(
            f"--cross-layer-weight {args.cross_layer_weight} requires "
            f"--budget-scope global: the per_bucket solve pins every group to "
            f"the same average independently, so W_g cannot move budget across "
            f"operators and would be silently ignored.")

    if not torch.cuda.is_available():
        raise RuntimeError("Calibration requires CUDA (sc kernels are GPU-only).")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    seed_all(args.seed)

    calib_on_sc = bool(args.calib_on_sc)
    ref_sl = args.budget_ref_stoc_len or max(levels)
    calib_sc_sl = int(args.calib_sc_stoclen) or max(
        1, int(round(args.budget_ratio * ref_sl)))
    if not args.attn_stoc_len:
        args.attn_stoc_len = calib_sc_sl

    # Build the SC model, then decide whether the forward TRAJECTORY is SC or
    # FP. The modules stay patched either way so the hooks still fire; only
    # what flows between blocks changes.
    model, pipe, cfg, skip_map = _build_sc_model(args, device)
    rt.set_sc_enabled(calib_on_sc)

    total = rt.state()["total_blocks"]
    if not total:
        raise RuntimeError("model exposes no blocks — nothing to calibrate.")

    operators = {x.strip() for x in args.operators.split(",") if x.strip()}
    attn_ops = operators & {"qk", "av"}
    from models.sc_integration import get_config
    _ctrl = get_config()
    present = {top for ctrl, top in rt.CTRL_TO_TABLE_OP.items()
               if getattr(_ctrl, f"enable_{ctrl}")}
    missing = operators - present
    if missing:
        raise ValueError(
            f"--operators names {sorted(missing)} but the patched model only "
            f"exposes {sorted(present)}.")

    layer_buckets = int(args.layer_buckets) or total
    print(f"[calib] blocks={total} operators={sorted(operators)} "
          f"layer_buckets={layer_buckets} levels={levels} "
          f"budget={args.budget_ratio:.4f}x{ref_sl}="
          f"{args.budget_ratio * ref_sl:.1f} "
          f"scope={args.budget_scope} weight={args.budget_weight} "
          f"objective={args.objective} W_g={args.cross_layer_weight} "
          f"halve={halve} sc_prec={sc_prec} metric={args.metric_select} "
          f"calib_on_sc={calib_on_sc}"
          + (f" calib_sc_stoclen={calib_sc_sl}" if calib_on_sc else ""),
          flush=True)

    loader, run_forward, calib_keys = _build_calib_iter(args, device, pipe, cfg)

    calibrator = ThresholdCalibrator(
        levels=levels,
        operators=operators,
        total_blocks=total,
        layer_buckets=layer_buckets,
        budget_ratio=args.budget_ratio,
        budget_ref_stoc_len=args.budget_ref_stoc_len,
        max_units_per_call=args.max_units_per_call,
        min_bucket_units=args.min_bucket_units,
        rng_seed=args.seed,
        budget_scope=args.budget_scope,
        budget_weight=args.budget_weight,
        metric_select=args.metric_select,
        metric_select_margin=args.metric_select_margin,
        argmin_pricing=args.argmin_pricing,
        objective=args.objective,
        cross_layer_weight=args.cross_layer_weight,
        group_ladder_k=args.group_ladder_k,
        escape_gate_k=args.escape_gate_k,
        escape_stoc_len=args.escape_stoc_len,
    )
    calibrator.n_heads = int(model.blocks[0].attn.num_heads)

    # ---- protected input channels (selected BEFORE the sigma pass) ---------
    # Order matters: the dispatch metric and every per-level sigma are measured
    # on the RESIDUAL channels, so the protected set has to be known first.
    if args.bucket_budget_from:
        if args.dispatch_unit != "chunk":
            raise SystemExit("[calib] --bucket-budget-from requires --dispatch-unit chunk")
        with open(args.bucket_budget_from) as f:
            calibrator.bucket_budget_ref = json.load(f)
        protection = calibrator.bucket_budget_ref.get("protected_channels")
        if protection:
            def unpack(values):
                return {(key.rsplit(":b", 1)[0], int(key.rsplit(":b", 1)[1])): value
                        for key, value in values.items()}
            idxs = unpack(protection["indices"])
            dims = unpack(protection["dims"])
            for key in idxs:
                op_, b_ = key
                d_in = int(_smoothed_weight(model, op_, b_).shape[1])
                if (dims.get(key) != d_in or len(set(idxs[key])) != len(idxs[key])
                        or any(i < 0 or i >= d_in for i in idxs[key])):
                    raise ValueError(f"reference protection does not match module {key}")
            calibrator.protected_channel_indices = idxs
            calibrator.protected_channel_dims = dims
            calibrator.protect_channel_frac = float(protection["frac"])
            calibrator.protect_channel_stoc_len = int(protection["stoc_len"])
            calibrator.protect_channel_metric = protection["metric"]
            calibrator.protect_compensate_budget = bool(protection["compensate_budget"])
            print(f"[calib] frozen protected channels: {len(idxs)} modules from "
                  f"{args.bucket_budget_from}; selection pre-pass skipped", flush=True)
        elif args.protect_channel_frac > 0:
            raise ValueError("bucket-budget-from reference has no protected channels; cannot add them")
    elif args.protect_channel_frac > 0.0:
        prot_sl = int(args.protect_channel_stoc_len or max(levels))
        if prot_sl > max(levels):
            raise SystemExit(
                f"[calib] --protect-channel-stoc-len {prot_sl} exceeds the "
                f"ladder ceiling {max(levels)}; protected channels would run "
                f"longer streams than any rung on the grid.")
        n_stat = max(1, int(args.protect_channel_stat_samples))
        stat_batches = [b for i, b in enumerate(loader) if i < n_stat]
        t_p = time.time()
        act_scales = _collect_channel_act_scales(model, stat_batches, run_forward)
        idxs, dims = _select_protected_channels(
            model, act_scales, args.protect_channel_frac,
            args.protect_channel_metric)
        calibrator.protected_channel_indices = idxs
        calibrator.protected_channel_dims = dims
        calibrator.protect_channel_frac = float(args.protect_channel_frac)
        calibrator.protect_channel_stoc_len = prot_sl
        calibrator.protect_channel_metric = args.protect_channel_metric
        calibrator.protect_compensate_budget = bool(args.protect_compensate_budget)
        n_ch = sum(len(v) for v in idxs.values())
        print(f"[calib] protected channels: {len(idxs)} modules, {n_ch} channels "
              f"({args.protect_channel_frac:.1%} each) at stoc_len={prot_sl}, "
              f"metric={args.protect_channel_metric}, "
              f"compensate={bool(args.protect_compensate_budget)} "
              f"[{time.time() - t_p:.0f}s]", flush=True)
        del stat_batches, act_scales

    calibrator.dispatch_unit = args.dispatch_unit
    calibrator.chunk_threshold_fit = args.chunk_threshold_fit
    if args.bucket_budget_from:
        print(f"[calib] per-bucket budgets and non-chunk operators taken from "
              f"{args.bucket_budget_from}", flush=True)
    if args.dispatch_unit == "chunk":
        if args.metric_select != "amax":
            raise SystemExit("[calib] --dispatch-unit chunk dispatches on the "
                             "pair absmax; --metric-select must be amax")
        if args.group_ladder_k or args.escape_gate_k is not None:
            raise SystemExit("[calib] --dispatch-unit chunk does not combine "
                             "with per-group ladders or the escape gate")
        if int(args.chunk_d) <= 0:
            raise SystemExit("[calib] --dispatch-unit chunk needs chunk_d > 0")
        print(f"[calib] dispatch unit: per-(row, {int(args.chunk_d)}-chunk) "
              f"pairs for the linear operators", flush=True)

    k_rec = int(args.record_every_k_steps)
    hooks = [rt.add_linear_hook(_make_linear_hook(calibrator, levels, sc_prec, halve,
                                                  int(args.chunk_d), k_rec))]
    print(f"[calib] linear hook installed (recording every {k_rec}-th of "
          f"{args.inference_steps} steps)", flush=True)

    if attn_ops:
        hooks.append(rt.add_attn_hook(_make_attn_hook(
            calibrator, levels, sc_prec, halve, args.qk_granularity, k_rec)))
        print(f"[calib] attention hook installed for {sorted(attn_ops)} "
              f"(qk {args.qk_granularity}, av per_row)", flush=True)

    t0 = time.time()
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                run_forward(model, batch)
                print(f"[calib] sample {i + 1}/{len(loader)} {batch[0][:-3]} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    finally:
        for remove in hooks:
            remove()

    # ---- solve + export, once per requested (objective, W_g) variant ----
    # The sigma collection above is INDEPENDENT of the objective and of W_g:
    # only the Lagrangian solve reads them. So N variants exported here share
    # ONE sigma draw, which removes the calibration-sample noise (~0.33 pt on
    # top-1 at n_calib=256, MP_REPORT.md) from any comparison between them —
    # every difference is attributable to the objective alone. Independent
    # calibrations per variant would confound the two.
    variants = _parse_variants(args, calibrator)

    snapshot = {k: list(rec["metrics"]) for k, rec in calibrator.records.items()}
    out_root = Path(args.output_json)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for objective, clw in variants:
        # Restore the as-collected amax metrics: select_dispatch_metrics()
        # REBINDS rec["metrics"] to the chosen candidate, and the choice is
        # objective-dependent, so each variant must start from the raw draw.
        for k, metrics in snapshot.items():
            calibrator.records[k]["metrics"] = list(metrics)
        calibrator.objective = objective
        calibrator.cross_layer_weight = clw
        calibrator.dispatch_metric_selection = {}

        tag = f"{objective}_w{clw}"
        out_json = (out_root if len(variants) == 1
                    else out_root.with_name(f"{out_root.stem}__{tag}.json"))
        print(f"\n[calib] === variant objective={objective} W_g={clw} "
              f"-> {out_json.name} ===", flush=True)

        rho = calibrator.metric_fidelity_rho()
        print("[calib] metric-fidelity rho(metric, objective-benefit) per op "
              "(~1 good; low/neg => metric misranks -> headroom):")
        for op in sorted(rho, key=lambda o: rho[o]["rho"]
                         if rho[o]["rho"] == rho[o]["rho"] else 9):
            d = rho[op]
            print(f"    {op:<11} rho={d['rho']:+.3f}  n={d['n']:>8}  "
                  f"benefit_mean={d['benefit_mean']:.4f}")

        # act_global_v2: per-operator signed dispatch-metric selection. Rewrites
        # the stored metric to the deployed candidate so thresholds build on it.
        chosen = calibrator.select_dispatch_metrics()
        if chosen:
            print(f"[calib] metric-select auto -> switched metric on "
                  f"{len(chosen)} op(s):")
            for op in sorted(chosen):
                c = chosen[op]
                print(f"    {op:<11} metric={c['metric']:<5} "
                      f"sign={c['sign']:+.0f} |rho|={c['rho_signed']:.3f} "
                      f"(amax rho={c['rho_amax']:+.3f})")
            rho_post = calibrator.metric_fidelity_rho()
            print("[calib] post-selection rho(deployed metric, benefit):")
            for op in sorted(rho_post, key=lambda o: rho_post[o]["rho"]
                             if rho_post[o]["rho"] == rho_post[o]["rho"] else 9):
                d = rho_post[op]
                print(f"    {op:<11} rho={d['rho']:+.3f}  n={d['n']:>8}")
        elif calibrator.metric_select == "auto":
            print("[calib] metric-select auto -> no op cleared the margin; "
                  "keeping amax everywhere.")

        payload, summary_rows = calibrator.export()
        payload["metric_fidelity_rho"] = {op: rho[op]["rho"] for op in rho}
        if chosen:
            payload["dispatch_metrics"] = {
                op: {"metric": c["metric"], "sign": c["sign"]}
                for op, c in chosen.items()}
        payload["task"] = "video_gen"
        payload["model"] = "irasim_" + str(cfg.attention_mode)
        payload["sc_prec"] = sc_prec
        payload["halve_bipolar_stoc_len"] = halve
        payload["operators"] = sorted(operators)
        payload["n_calib_samples"] = args.n_calib_samples
        payload["inference_steps"] = int(args.inference_steps)
        payload["scheduler"] = args.scheduler
        payload["record_every_k_steps"] = k_rec
        payload["skip"] = {k: sorted(v) for k, v in skip_map.items() if v}
        _sq = os.environ.get("SC_SMOOTH_SCALES")
        payload["smoothquant_scales"] = os.path.abspath(_sq) if _sq else None
        payload["total_blocks"] = int(total)
        payload["seed"] = int(args.seed)
        payload["calib_seed"] = int(args.calib_seed)
        payload["eval_excluded"] = {"keys_file": args.eval_keys_file}
        payload["calib_keys"] = calib_keys
        payload["mp_levels"] = args.mp_levels
        payload["chunk_d"] = int(args.chunk_d)
        # Ops that consume budget but carry no MP table (see module docstring).
        payload["uniform_ops"] = {op: int(args.attn_stoc_len)
                                  for op in ("qk", "av")
                                  if op not in operators}
        method = "act_global" if args.budget_scope == "global" else "act"
        if args.budget_weight == "macs":
            method += "_fw"
        if args.argmin_pricing == "macs":
            method += "_kkt"
        if clw != "uniform":
            method += f"_w{clw}"
        if objective != "sigma":
            method += f"_{objective}"
        if calib_on_sc:
            method += "_onsc"
            payload["calib_on_sc"] = True
            payload["calib_sc_stoclen"] = calib_sc_sl
        if args.group_ladder_k > 0:
            method += f"_gl{args.group_ladder_k}"
        if args.protect_channel_frac > 0.0:
            method += f"_pc{args.protect_channel_frac:g}"
        if args.escape_gate_k is not None:
            method += f"_esc{args.escape_gate_k:g}"
        if args.qk_granularity != "per_head":
            method += f"_qk{args.qk_granularity}"
        payload["qk_granularity"] = args.qk_granularity
        payload["method"] = method
        payload["calib_command"] = " ".join(sys.argv)
        if len(variants) > 1:
            # One sigma draw, many tables: say so in the bundle itself, so a
            # cell can never be mistaken for an independent calibration.
            payload["variant"] = {"tag": tag, "shared_sigma_draw": True,
                                  "siblings": [f"{o}_w{c}" for o, c in variants]}

        print(f"[calib] expected_avg_stoc_len (row-weighted) = "
              f"{payload['expected_avg_stoc_len']:.2f}")
        print(f"[calib] expected_flop_avg_stoc_len (iso-compute check) = "
              f"{payload['expected_flop_avg_stoc_len']:.2f} "
              f"(target {args.budget_ratio * calibrator.budget_ref_stoc_len:.2f})")

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[calib] wrote {out_json}")

        # Deployment wrapper: what model.sc_vit.load_adaptive_mp() consumes.
        # threshold_table_path is stored RELATIVE so a calibration bundle can be
        # moved or copied between machines without rewriting it.
        wrapper = out_json.with_name(out_json.stem + "_wrapper.json")
        wrapper_spec = {
            "type": "AdaptiveMPConfig",
            "stoc_len_levels": levels,
            "threshold_table_path": out_json.name,
        }
        if args.escape_gate_k is not None:
            wrapper_spec["escape_gate_k"] = float(args.escape_gate_k)
            wrapper_spec["escape_stoc_len"] = int(args.escape_stoc_len
                                                  or max(levels))
        with open(wrapper, "w") as f:
            json.dump(wrapper_spec, f, indent=2)
        print(f"[calib] wrote {wrapper}  <- pass this to --adaptive_mp_table")

        summary_csv = (args.output_summary_csv if len(variants) == 1
                       else None) or (str(out_json.with_suffix("")) + "_summary.csv")
        _write_summary_csv(summary_csv, summary_rows)
        print(f"[calib] wrote {summary_csv}")
        written.append((tag, out_json, payload))

    if len(written) > 1:
        _print_variant_comparison(written, calibrator,
                                  args.budget_ratio * calibrator.budget_ref_stoc_len)




def _parse_variants(args, calibrator) -> list:
    """``--variants`` -> [(objective, cross_layer_weight), ...].

    Empty (default) yields the single pair from ``--objective`` /
    ``--cross-layer-weight``, so the one-table behaviour is unchanged.
    ``all`` expands to every (objective x W_g) combination.
    """
    spec = (args.variants or "").strip()
    if not spec:
        return [(args.objective, args.cross_layer_weight)]
    objectives = ["sigma", "sigma2", "delta_sigma2"]
    weights = ["uniform", "macs"]
    if spec == "all":
        # 'all' stays the objective x {uniform, macs} grid: the measured weights
        # need a probe, so pulling them in silently would change the run's cost.
        return [(o, w) for w in ("uniform", "macs") for o in objectives]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        obj, _, clw = item.partition(":")
        clw = clw or "uniform"
        if obj not in objectives or clw not in weights:
            raise SystemExit(
                f"--variants entry {item!r} must be "
                f"'<objective>[:<W_g>]' with objective in {objectives} and "
                f"W_g in {weights} (or the literal 'all').")
        if (obj, clw) not in out:
            out.append((obj, clw))
    if not out:
        raise SystemExit("--variants parsed to nothing")
    if any(w != "uniform" for _o, w in out) and args.budget_scope != "global":
        raise SystemExit("--variants with W_g=macs requires --budget-scope global")
    return out


def _print_variant_comparison(written: list, calibrator, target: float):
    """Per-operator MAC-weighted allocation across variants of ONE sigma draw.

    This is the cheap preview of what the eval will show: if two variants
    allocate the same cycles per operator, running both is wasted GPU time.
    """
    ops = sorted({k[0] for k in calibrator.records})
    hdr = f'{"variant":<26}' + "".join(f"{o:>11}" for o in ops) + f'{"realized":>10}'
    print(f"\n[calib] === allocation across variants (one shared sigma draw, "
          f"target {target:.1f}) ===")
    print(hdr)
    print("-" * len(hdr))
    for tag, _path, payload in written:
        per, num, den = {}, 0.0, 0.0
        for bkey, fit in payload["buckets"].items():
            op, _t, lpart = bkey.split(":")
            mac = payload["mac_per_row"][f"{op}:{lpart}"]
            w = fit["num_units"] * mac
            e = per.setdefault(op, [0.0, 0.0])
            e[0] += fit["avg_stoc_len"] * w
            e[1] += w
            num += fit["avg_stoc_len"] * w
            den += w
        row = f"{tag:<26}" + "".join(
            f'{per[o][0] / per[o][1]:>11.1f}' if o in per else f'{"-":>11}'
            for o in ops)
        print(row + f'{num / den if den else float("nan"):>10.2f}')


if __name__ == "__main__":
    main()
