"""Run-wide SC state for IRASim, aligned with scmp_vit's ``model/sc_vit.py``.

Every SC knob that is not per-call lives here: the quantization grid
(``sc_prec``), stream halving, D-chunking, the Q@K^T quantization scope, the
uniform stream length (or per-step schedule), the calibrated MP table, and the
calibration hooks. ``sc_linear`` / ``sc_attention`` read it at call time.

Defaults are the scmp_vit FORMAL operating point (``cls/run_l7_10k.sh``):

=====================  ===================  =================================
knob                   default              env override
=====================  ===================  =================================
sc_prec                8                    ``SC_PREC``
halve                  off                  ``SC_HALVE=1``
linear granularity     per_row (fixed)      —
chunk_d                128                  ``SC_CHUNK_D``
qk granularity         per_row              ``SC_QK_GRANULARITY``
av granularity         per_row (fixed)      —
uniform stoc_len       grid cap             ``SC_UNIFORM_STOC_LEN``
MP                     off                  ``SC_MP_TABLE=<wrapper.json>``
=====================  ===================  =================================

``chunk_d=128`` is a correctness requirement on wide layers, not a speed knob
(scmp_vit measured unchunked SC on its D_in=4096 fc2 at 0% top-1; IRASim's
fc2 has D_in=4608). The uniform stream length and the per-step schedule apply
to ALL SC ops, attention included, so a ladder rung names the cost of the
whole SC datapath rather than of the linears alone.

The pre-alignment homebrew MP knobs (``SC_MP_CONFIG``, ``SC_MP_PER_MODULE``,
``SC_MP_INVERT``) and ``SC_LINEAR_GRANULARITY=per_tensor`` are removed and
raise, so an old command line cannot silently run a different configuration.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path


from scmp_kernels.mp import AdaptiveMPConfig

# Controller op names (sc_controller / --skip) -> table operator names. The
# table uses scmp_vit / scmp_llm's taxonomy so calibration bundles read the
# same across the SC applications.
CTRL_TO_TABLE_OP = {"qkv": "qkv_proj", "proj": "out_proj",
                    "mlp_fc1": "mlp_fc1", "mlp_fc2": "mlp_fc2",
                    "qk": "qk", "av": "av"}
TABLE_OPS = tuple(CTRL_TO_TABLE_OP.values())
LINEAR_TABLE_OPS = ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _reject_removed_env():
    removed = {
        "SC_MP_CONFIG": "use SC_MP_TABLE=<wrapper.json> from evaluate/calibrate_mp_thresholds.py",
        "SC_MP_PER_MODULE": "per-module fractions are replaced by the calibrated threshold table (SC_MP_TABLE)",
        "SC_MP_INVERT": "the table carries each operator's dispatch metric and sign",
    }
    for k, why in removed.items():
        if os.environ.get(k):
            raise RuntimeError(f"{k} was removed in the scmp_vit alignment: {why}.")
    g = os.environ.get("SC_LINEAR_GRANULARITY")
    if g and g != "per_row":
        raise RuntimeError(
            f"SC_LINEAR_GRANULARITY={g!r}: linears are per_row only (scmp_vit "
            f"SCLinear); per_tensor cannot be D-chunked.")
    if os.environ.get("SC_MP_FIXED_PREC") == "0":
        raise RuntimeError(
            "SC_MP_FIXED_PREC=0 was removed: every rung runs at the one sc_prec "
            "grid (rungs are cycle counts, as in scmp_vit / scmp_llm).")


_reject_removed_env()

_STATE = {
    "sc_prec": _env_int("SC_PREC", 8),
    "halve": os.environ.get("SC_HALVE") == "1",
    "chunk_d": _env_int("SC_CHUNK_D", 128),
    "chunk_d_warn_width": 1024,
    "qk_granularity": os.environ.get("SC_QK_GRANULARITY") or "per_row",
    "uniform_stoc_len": _env_int("SC_UNIFORM_STOC_LEN", None),
    "step_schedule": None,
    "mp_cfg": None,            # AdaptiveMPConfig shared by every MP op
    "mp_ops": frozenset(),     # table operator names that dispatch through it
    "mp_table": None,          # provenance of the loaded table
    "total_blocks": None,
}
if os.environ.get("SC_STEP_SCHEDULE"):
    _STATE["step_schedule"] = [int(x) for x in json.loads(os.environ["SC_STEP_SCHEDULE"])]
if _STATE["qk_granularity"] not in ("per_head", "per_row"):
    raise RuntimeError(f"SC_QK_GRANULARITY must be per_head|per_row, got "
                       f"{_STATE['qk_granularity']!r}")

_SC_ENABLED = [True]
_GROUP_STOCLEN = [None]
# Calibration hooks: called with already-smoothed operands (see sc_linear).
_LINEAR_HOOKS: list = []
_ATTN_HOOKS: list = []
# MAC-weighted cycle accounting: {table_op: [macs, macs*cycles]}.
_COST = defaultdict(lambda: [0.0, 0.0])


def state() -> dict:
    return dict(_STATE)


def set_sc_options(**kw):
    """Override run-wide knobs (``sc_prec``, ``halve``, ``chunk_d``,
    ``qk_granularity``, ``uniform_stoc_len``, ``step_schedule``,
    ``total_blocks``)."""
    for k, v in kw.items():
        if k not in _STATE or k in ("mp_cfg", "mp_ops", "mp_table"):
            raise KeyError(f"unknown SC option {k!r}")
        _STATE[k] = v
    if _STATE["qk_granularity"] not in ("per_head", "per_row"):
        raise ValueError("qk_granularity must be per_head|per_row")


def grid_cap(sc_prec: int | None = None) -> int:
    p = _STATE["sc_prec"] if sc_prec is None else int(sc_prec)
    return 2 ** (p - 1) if _STATE["halve"] else 2 ** p


def resolve_stoc_len(stoc_len=None, sc_prec: int | None = None) -> int:
    """Effective UNIFORM stream length for this call (scmp_vit
    ``_resolve_stoc_len``): explicit request > per-step schedule >
    ``SC_UNIFORM_STOC_LEN`` > grid cap, always clamped to the cap. Under halve
    the cap is ``2**(sc_prec-1)``; a longer stream buys no resolution."""
    if stoc_len is None:
        sched = _STATE["step_schedule"]
        if sched is not None:
            from .sc_controller import get_current_step
            i, n = get_current_step()
            stoc_len = sched[min(i * len(sched) // max(n, 1), len(sched) - 1)]
        else:
            stoc_len = _STATE["uniform_stoc_len"]
    cap = grid_cap(sc_prec)
    return int(cap) if stoc_len is None else int(min(int(stoc_len), cap))


def chunk_d_for(width: int) -> int:
    cd = int(_STATE["chunk_d"])
    return cd if (cd > 0 and width > cd) else 0


# ---- SC on/off + knock-down probe (calibration helpers) --------------------

def set_sc_enabled(flag: bool):
    """Bypass SC everywhere (FP) while keeping the calibration hooks firing."""
    _SC_ENABLED[0] = bool(flag)


def sc_enabled() -> bool:
    return _SC_ENABLED[0]


def set_group_stoclen(mapping: dict | None):
    """Force ``{(table_op, block): stoc_len}`` uniformly, MP bypassed."""
    _GROUP_STOCLEN[0] = dict(mapping) if mapping else None


def group_stoclen():
    return _GROUP_STOCLEN[0]


# ---- calibration hooks -----------------------------------------------------

def add_linear_hook(fn):
    """``fn(op, block_idx, x2, w, bias, mac_per_row)`` per SC linear call.
    ``x2``/``w`` are float32 and already SmoothQuant-transformed."""
    _LINEAR_HOOKS.append(fn)
    return lambda: _LINEAR_HOOKS.remove(fn)


def add_attn_hook(fn):
    """``fn(op, block_idx, a3, b3)`` per SC attention call (``a3 @ b3.T``)."""
    _ATTN_HOOKS.append(fn)
    return lambda: _ATTN_HOOKS.remove(fn)


def linear_hooks():
    return _LINEAR_HOOKS


def attn_hooks():
    return _ATTN_HOOKS


# ---- cost accounting -------------------------------------------------------

def record_cost(op: str, macs: float, cycles: float):
    """Charge ``macs`` MACs at ``cycles`` stream length to ``op``."""
    c = _COST[op]
    c[0] += float(macs)
    c[1] += float(macs) * float(cycles)


def cost_summary(clear: bool = False) -> dict:
    """MAC-weighted average cycles per op and over all SC ops — the
    iso-compute number (what scmp_vit's trace-based iso_check reports)."""
    out = {op: (c[1] / c[0] if c[0] else 0.0) for op, c in sorted(_COST.items())}
    tot_m = sum(c[0] for c in _COST.values())
    tot_c = sum(c[1] for c in _COST.values())
    out["ALL"] = tot_c / tot_m if tot_m else 0.0
    out["mac_share"] = {op: (c[0] / tot_m if tot_m else 0.0)
                        for op, c in sorted(_COST.items())}
    if clear:
        _COST.clear()
    return out


# ---- MP table --------------------------------------------------------------

def _resolve_mp_table_path(path: str):
    wrapper_path = Path(path).expanduser().resolve()
    with open(wrapper_path) as f:
        spec = json.load(f)
    table = spec.get("threshold_table_path")
    table_path = None
    if table:
        table_path = Path(table)
        if not table_path.is_absolute():
            table_path = wrapper_path.parent / table_path
    return wrapper_path, table_path, spec


def read_mp_table_provenance(path: str) -> dict:
    _w, table_path, _s = _resolve_mp_table_path(path)
    if table_path is None:
        return {}
    with open(table_path) as f:
        payload = json.load(f)
    keys = ("sc_prec", "halve_bipolar_stoc_len", "chunk_d", "stoc_len_levels",
            "method", "total_blocks", "objective", "cross_layer_weight",
            "qk_granularity", "operators", "calib_keys", "skip",
            "smoothquant_scales")
    return {k: payload[k] for k in keys if k in payload}


def load_mp_table(path: str, ops=None) -> AdaptiveMPConfig:
    """Load a calibrated wrapper (scmp_vit ``load_adaptive_mp`` contract).

    The table's ``sc_prec`` / ``halve`` / ``chunk_d`` / ``qk_granularity``
    are authoritative: halve and qk_granularity are ADOPTED when the run did
    not set them, and any explicit disagreement RAISES — thresholds and rungs
    do not transfer across those changes.
    """
    wrapper_path, table_path, spec = _resolve_mp_table_path(path)
    if table_path is None:
        raise ValueError(f"{wrapper_path}: needs threshold_table_path (the "
                         f"kernel has no closed-form threshold mode).")
    prov = read_mp_table_provenance(str(wrapper_path))
    levels = [int(x) for x in spec["stoc_len_levels"]]

    if "halve_bipolar_stoc_len" in prov:
        tab = bool(prov["halve_bipolar_stoc_len"])
        if "SC_HALVE" in os.environ and _STATE["halve"] != tab:
            raise ValueError(f"{table_path.name}: calibrated with halve={tab} "
                             f"but SC_HALVE={int(_STATE['halve'])}.")
        _STATE["halve"] = tab
    if "qk_granularity" in prov:
        tab = prov["qk_granularity"]
        if os.environ.get("SC_QK_GRANULARITY") and _STATE["qk_granularity"] != tab:
            raise ValueError(f"{table_path.name}: calibrated with qk_granularity="
                             f"{tab!r} but SC_QK_GRANULARITY="
                             f"{_STATE['qk_granularity']!r}.")
        _STATE["qk_granularity"] = tab
    for name, key in (("sc_prec", "sc_prec"), ("chunk_d", "chunk_d")):
        if key in prov and int(prov[key]) != int(_STATE[name]):
            raise ValueError(f"{table_path.name}: calibrated with {name}="
                             f"{prov[key]} but this run deploys {name}="
                             f"{_STATE[name]}.")
    if max(levels) > grid_cap():
        raise ValueError(f"{table_path.name}: top rung {max(levels)} exceeds "
                         f"the grid cap {grid_cap()}.")

    cfg = AdaptiveMPConfig(stoc_len_levels=levels,
                           enable_pruning=bool(spec.get("enable_pruning", True)),
                           target_fractions=spec.get("target_fractions"))
    cfg.load_threshold_table(str(table_path))
    if spec.get("escape_gate_k") is not None:
        cfg.escape_gate_k = float(spec["escape_gate_k"])
        cfg.escape_stoc_len = int(spec.get("escape_stoc_len", max(levels)))
        cfg.threshold_table_path = str(table_path)
        cfg.load_threshold_table(str(table_path))

    table_ops = set(prov.get("operators") or TABLE_OPS)
    want = set(ops) if ops else table_ops
    unknown = want - table_ops
    if unknown:
        raise ValueError(f"{table_path.name}: ops {sorted(unknown)} have no "
                         f"calibrated thresholds (table covers {sorted(table_ops)}).")
    _STATE["mp_cfg"] = cfg
    _STATE["mp_ops"] = frozenset(want)
    _STATE["mp_table"] = dict(prov, path=str(wrapper_path))
    if prov.get("total_blocks"):
        _STATE["total_blocks"] = int(prov["total_blocks"])
    return cfg


def mp_cfg_for(table_op: str):
    if _STATE["mp_cfg"] is not None and table_op in _STATE["mp_ops"]:
        return _STATE["mp_cfg"]
    return None


def configure_from_env(model=None, verbose: bool = True) -> dict:
    """Finish run-wide setup: record the block count and load ``SC_MP_TABLE``
    (optionally restricted by ``SC_MP_OPS``). Returns ``state()``."""
    if model is not None:
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            n = len(blocks)
            if _STATE["total_blocks"] not in (None, n):
                raise ValueError(f"MP table has total_blocks="
                                 f"{_STATE['total_blocks']} but model has {n}")
            _STATE["total_blocks"] = n
    table = os.environ.get("SC_MP_TABLE")
    if table:
        ops = [o.strip() for o in os.environ.get("SC_MP_OPS", "").split(",") if o.strip()]
        load_mp_table(table, ops or None)
        if model is not None and _STATE["mp_table"].get("total_blocks"):
            n = len(getattr(model, "blocks", []))
            if n and n != int(_STATE["mp_table"]["total_blocks"]):
                raise ValueError(f"MP table total_blocks="
                                 f"{_STATE['mp_table']['total_blocks']} != model {n}")
    if verbose:
        s = state()
        from ._bootstrap import kernels_provenance
        print(f"[env] scmp_kernels {kernels_provenance()}", flush=True)
        print(f"[sc] sc_prec={s['sc_prec']} halve={s['halve']} chunk_d={s['chunk_d']} "
              f"qk={s['qk_granularity']} av=per_row linear=per_row "
              f"uniform_stoc_len={resolve_stoc_len() if s['step_schedule'] is None else 'schedule'} "
              f"mp={'off' if s['mp_cfg'] is None else sorted(s['mp_ops'])}"
              + (f" table={s['mp_table']['path']} method={s['mp_table'].get('method')}"
                 if s['mp_table'] else ""), flush=True)
    return state()

