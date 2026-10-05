"""Read, check and summarize per-group precision traces (cls/group_trace.py).

Port of scmp_llm's benchmark/ppl/group_trace_check.py (feat/group-trace 78b7a93); the same
file serves scmp_vit, scmp_cnn and scmp_worldmodel. An attention op whose description says
trace_unit: null (IRASim) is matched against a summary traced per call, not per head.

    python evaluate/group_trace_check.py --exact DIR [--fp DIR] [--sc-trace FILE]
        [--archived-trace FILE] [--out report.json]

1. Identity (exact capture). The groups, re-aggregated per (block, op, unit, stoc_len, d_in,
   d_out, smoothed, rng_levels) the way scmp_kernels/trace.py aggregates what the kernels
   ran -- a per-(row, chunk) call by rung and chunk width, a per-row / uniform call by length
   at the whole dispatched width, the protected slice as one call, attention per head, a
   depthwise conv as one batched call -- must equal the scmp-trace-summary written for the
   same windows: rows, MACs and row-cycles of every group, exactly.
2. FP vs exact. How many groups (and MACs) the FP trajectory puts on the same length, how
   far the others move (ladder rungs), per operator and per block, and what that does to the
   trace cost and to tile-max avg_sl (tile_cycle_macs, R rows per tile).
3. Context. Per-operator avg_sl of the captured windows next to the archived full-eval
   trace of the same cell (a few-window sample is not the full eval).

load_group_trace / sc_calls / family / summary_from_groups / tile_avg_sl are the reader API.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TILE_ROWS = (1, 8, 16, 32, 64)


def load_group_trace(path):
    """(meta, arrays, ops): the json, the lazily loaded npz, {(op, block, unit): desc}."""
    path = Path(path)
    meta = json.loads((path / "group_trace.json").read_text())
    if meta.get("schema") != "scmp-group-trace-v1":
        raise ValueError(f"{path}: not a scmp-group-trace-v1 trace")
    arrays = np.load(path / "group_trace.npz")
    ops = {(o["op"], o["block"], o["unit"]): o for o in meta["ops"]}
    return meta, arrays, ops


def sc_calls(meta):
    """(window, call) for every SC call, in execution order."""
    for w in sorted(meta["calls"], key=int):
        for c in meta["calls"][w]:
            if c["backend"] == "sc":
                yield int(w), c


def family(desc):
    """linear: [rows, chunks] maps; attention: [batch*heads, rows], traced per head;
    depthwise: [batch*channels, pixels], one batched call (unit None)."""
    if "heads" not in desc:
        return "linear"
    return "depthwise" if desc.get("slice_kind") == "channel" else "attention"


def _chunk_classes(lengths, desc):
    """Split a [rows, chunks] map into (sub-map, chunk width) as trace.py does."""
    widths, chunk_d = desc["chunk_widths"], desc["chunk_d"]
    if widths[-1] == chunk_d:
        return [(lengths, chunk_d)]
    return [(lengths[:, :-1], chunk_d), (lengths[:, -1:], widths[-1])]


def summary_from_groups(meta, arrays, ops):
    """{(block, op, unit, stoc_len, d_in, d_out, smoothed, rng_levels): [rows, macs,
    row_cycles]} -- the scmp-trace-summary-v1 aggregation of the SC calls, exact integers."""
    rng = meta["header"]["sc"]["rng_levels"]
    acc = defaultdict(lambda: [0, 0, 0])

    def add(c, unit, sl, rows, d_in, smoothed):
        a = acc[(c["block"], c["op"], unit, sl, d_in, c["d_out"], smoothed, rng)]
        a[0] += rows
        a[1] += rows * d_in * c["d_out"]
        a[2] += rows * sl

    for _, c in sc_calls(meta):
        desc = ops[(c["op"], c["block"], c["unit"])]
        fam = family(desc)
        if fam == "linear":
            if c["groups"] is not None:
                lengths = arrays[c["groups"]]
                if desc.get("dispatch", "rung_table") == "rung_table":
                    for sub, width in _chunk_classes(lengths, desc):
                        vals, cnt = np.unique(sub, return_counts=True)
                        for v, k in zip(vals.tolist(), cnt.tolist()):
                            add(c, c["unit"], v, k, width, desc["smoothed"])
                else:          # one call per length on whole rows of the dispatched input
                    if (lengths != lengths[:, :1]).any():
                        raise ValueError(f"{c['groups']}: row dispatch with unequal chunks")
                    vals, cnt = np.unique(lengths[:, 0], return_counts=True)
                    for v, k in zip(vals.tolist(), cnt.tolist()):
                        add(c, c["unit"], v, k, sum(desc["chunk_widths"]), desc["smoothed"])
            if desc["protected_cols"]:
                add(c, c["unit"], desc["protected_stoc_len"], c["rows"],
                    desc["protected_cols"], desc["smoothed"])
        elif fam == "attention" and desc.get("trace_unit", "head") == "head":
            lengths, heads = arrays[c["groups"]], c["heads"]
            for h in range(heads):
                vals, cnt = np.unique(lengths[h::heads], return_counts=True)
                for v, k in zip(vals.tolist(), cnt.tolist()):
                    add(c, h, v, k, c["d_in"], desc["smoothed"])
        else:          # depthwise, or attention the kernels traced per call (unit None)
            vals, cnt = np.unique(arrays[c["groups"]], return_counts=True)
            for v, k in zip(vals.tolist(), cnt.tolist()):
                add(c, c["unit"], v, k, c["d_in"], desc["smoothed"])
    return dict(acc)


def summary_from_trace(path):
    t = json.loads(Path(path).read_text())
    if t.get("schema") != "scmp-trace-summary-v1":
        raise ValueError(f"{path}: not a scmp-trace-summary-v1 trace")
    acc = defaultdict(lambda: [0, 0, 0])
    for g in t["groups"]:
        if g["dims_vary"]:
            raise ValueError(f"{path}: dims_vary group, shapes are not exact")
        a = acc[(g["block"], g["op"], g["unit"], g["stoc_len"], g["d_in"], g["d_out"],
                 g["smoothed"], g["rng_levels"])]
        a[0] += g["rows"]
        a[1] += g["macs"]
        a[2] += g["row_cycles"]
    return t["header"], dict(acc)


def identity(groups, trace):
    keys = set(groups) | set(trace)
    bad = sorted((k for k in keys if groups.get(k) != trace.get(k)), key=str)
    return dict(ok=not bad, n_keys=len(keys), n_mismatch=len(bad),
                macs_groups=sum(v[1] for v in groups.values()),
                macs_trace=sum(v[1] for v in trace.values()),
                first_mismatches=[dict(key=list(k), groups=groups.get(k), trace=trace.get(k))
                                  for k in bad[:20]])


def _op_families(meta):
    return {o["op"]: family(o) for o in meta["ops"] if o.get("backend") == "sc"}


def per_op_avg_sl(summary, fams):
    """MAC-weighted mean stoc_len per op, plus 'all' and per family -- the trace cost
    definition (every SC group, protected slice included)."""
    num, den = defaultdict(int), defaultdict(int)
    for key, (_, macs, _) in summary.items():
        op, sl = key[1], key[3]
        for k in (op, "all", fams.get(op, "attention" if op in ("qk", "av") else "linear")):
            num[k] += macs * sl
            den[k] += macs
    return {k: num[k] / den[k] for k in den}


def tile_cycle_macs(lengths, row_macs, tile_rows=TILE_ROWS):
    """Cycle-MACs when every R-row tile runs at its longest row (scmp_llm
    benchmark/ppl/tile_cost.py, verbatim).

    lengths: (G, N, C) stream lengths of G independent row sequences of N rows, one
    column per contraction chunk. row_macs: (C,) MACs one row spends in each chunk.
    Returns {R: sum over tiles and chunks of max length * MACs of the tile's real rows}.
    """
    import torch
    g, n, c = lengths.shape
    lf = lengths.to(torch.float64)
    w = row_macs.to(torch.float64)
    out = {}
    for r in tile_rows:
        t = (n + r - 1) // r
        pad = t * r - n
        # Zero padding never wins the max: stream lengths are positive and the
        # last tile always holds at least one real row.
        lp = torch.nn.functional.pad(lf, (0, 0, 0, pad)) if pad else lf
        tile_max = lp.view(g, t, r, c).amax(2)
        real = torch.full((t,), float(r), dtype=torch.float64, device=lf.device)
        real[-1] = r - pad
        out[r] = (tile_max * real[None, :, None] * w[None, None, :]).sum()
    return out


def _group_macs(c, desc, shape):
    if family(desc) == "linear":
        return np.broadcast_to(np.asarray(desc["chunk_widths"], np.float64) * c["d_out"], shape)
    return np.full(shape, float(c["d_in"] * c["d_out"]))


def tile_avg_sl(meta, arrays, ops, tile_rows=TILE_ROWS):
    """Flat (R=1) and tile-max avg_sl of the captured windows: an R-row tile runs at its
    longest row, per 128-chunk (tile_cost.py's hardware model). Attention / depthwise: the
    rows of one (image, head|channel) slice, R consecutive query rows | pixels per tile."""
    import torch
    cyc, macs = defaultdict(float), defaultdict(float)
    fixed_macs = fixed_cyc = 0.0
    fams = set()
    for _, c in sc_calls(meta):
        desc = ops[(c["op"], c["block"], c["unit"])]
        fam = family(desc)
        fams.add(fam)
        if fam == "linear":
            if desc["protected_cols"]:
                m = c["rows"] * desc["protected_cols"] * c["d_out"]
                fixed_macs += m
                fixed_cyc += m * desc["protected_stoc_len"]
            if c["groups"] is None:
                continue
            lengths = torch.as_tensor(arrays[c["groups"]].astype(np.int64))[None]
            row_macs = torch.as_tensor(desc["chunk_widths"], dtype=torch.float64) * c["d_out"]
        else:
            lengths = torch.as_tensor(arrays[c["groups"]].astype(np.int64))[..., None]
            row_macs = torch.tensor([float(c["d_in"] * c["d_out"])], dtype=torch.float64)
        for r, v in tile_cycle_macs(lengths, row_macs, tile_rows).items():
            cyc[(fam, r)] += float(v)
        macs[fam] += float(row_macs.sum()) * lengths.shape[0] * lengths.shape[1]
    macs["linear"] += fixed_macs
    res = {}
    for r in tile_rows:
        cy = {f: cyc[(f, r)] for f in fams}
        if "linear" in cy:
            cy["linear"] += fixed_cyc
        tot_c, tot_m = sum(cy.values()), sum(macs[f] for f in fams)
        res[str(r)] = dict(all=tot_c / tot_m, **{f: (cy[f] / macs[f] if macs[f] else None)
                                                 for f in sorted(fams)})
    return res


def compare(fp, exact):
    """Group-by-group agreement of the FP-trajectory capture with the exact one (dense calls
    pair row by row). The dispatched_* lengths cover the compared groups only (no protected
    slice); the trace-cost comparison is per_op_avg_sl of each capture."""
    (meta_f, arr_f, _), (meta_e, arr_e, ops_e) = fp, exact
    struct = lambda m: {w: [(c["block"], c["op"], c["backend"], c["rows"], c["d_in"],
                             c["d_out"]) for c in cs] for w, cs in m["calls"].items()}
    stats = defaultdict(lambda: defaultdict(float))
    by_block = defaultdict(lambda: [0, 0])
    unmatched = defaultdict(int)
    for w, c in sc_calls(meta_e):
        key = c["groups"]
        if key is None:
            continue
        e = arr_e[key]
        if key not in arr_f.files:
            unmatched["exact_only_groups"] += e.size
            continue
        f = arr_f[key]
        if e.shape != f.shape:
            raise ValueError(f"{key}: call with different shapes in the two captures")
        desc = ops_e[(c["op"], c["block"], c["unit"])]
        macs = _group_macs(c, desc, e.shape)
        ladder = np.asarray(sorted(set(desc["ladder"])))
        dr = np.abs(np.searchsorted(ladder, e) - np.searchsorted(ladder, f))
        ef, ff = e.astype(np.float64), f.astype(np.float64)
        for k in (c["op"], "all", family(desc)):
            s = stats[k]
            s["groups"] += e.size
            s["same"] += int((dr == 0).sum())
            s["within_1_rung"] += int((dr <= 1).sum())
            s["macs"] += macs.sum()
            s["macs_same"] += macs[dr == 0].sum()
            s["cycles_exact"] += (macs * ef).sum()
            s["cycles_fp"] += (macs * ff).sum()
            s["fp_longer_macs"] += macs[f > e].sum()
            s["fp_shorter_macs"] += macs[f < e].sum()
        by_block[c["block"]][0] += e.size
        by_block[c["block"]][1] += int((dr == 0).sum())
    for w, c in sc_calls(meta_f):
        if c["groups"] is not None and c["groups"] not in arr_e.files:
            unmatched["fp_only_groups"] += arr_f[c["groups"]].size
    out = {}
    for k, s in stats.items():
        out[k] = dict(groups=int(s["groups"]), same_frac=s["same"] / s["groups"],
                      within_1_rung_frac=s["within_1_rung"] / s["groups"],
                      mac_same_frac=s["macs_same"] / s["macs"],
                      dispatched_avg_sl_exact=s["cycles_exact"] / s["macs"],
                      dispatched_avg_sl_fp=s["cycles_fp"] / s["macs"],
                      dispatched_rel_err=s["cycles_fp"] / s["cycles_exact"] - 1,
                      mac_frac_fp_longer=s["fp_longer_macs"] / s["macs"],
                      mac_frac_fp_shorter=s["fp_shorter_macs"] / s["macs"])
    return dict(call_structure_identical=struct(meta_f) == struct(meta_e),
                unmatched=dict(unmatched), per_op=out,
                per_block_same_frac={b: v[1] / v[0] for b, v in sorted(by_block.items())})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact", required=True, help="group trace captured on the SC path")
    ap.add_argument("--fp", help="group trace captured on the FP trajectory")
    ap.add_argument("--sc-trace", help="scmp-trace-summary of the exact capture's windows")
    ap.add_argument("--archived-trace", help="archived full-eval trace of the same cell")
    ap.add_argument("--out", help="write the report here (json)")
    args = ap.parse_args()

    exact = load_group_trace(args.exact)
    fams = _op_families(exact[0])
    report = dict(exact=args.exact, fp=args.fp, windows=exact[0]["header"]["windows"])
    groups = summary_from_groups(*exact)
    report["avg_sl_exact"] = per_op_avg_sl(groups, fams)
    report["tile_avg_sl_exact"] = tile_avg_sl(*exact)
    ok = True
    if args.sc_trace:
        _, trace = summary_from_trace(args.sc_trace)
        report["identity"] = identity(groups, trace)
        ok &= report["identity"]["ok"]
        print(f"[check] identity vs {args.sc_trace}: ok={report['identity']['ok']} "
              f"keys={report['identity']['n_keys']} mismatches={report['identity']['n_mismatch']} "
              f"macs {report['identity']['macs_groups']} vs {report['identity']['macs_trace']}")
    if args.fp:
        fp = load_group_trace(args.fp)
        report["avg_sl_fp"] = per_op_avg_sl(summary_from_groups(*fp), fams)
        report["tile_avg_sl_fp"] = tile_avg_sl(*fp)
        report["fp_vs_exact"] = cmp_all = compare(fp, exact)
        ok &= cmp_all["call_structure_identical"]
        print(f"[check] fp vs exact: call structure identical = "
              f"{cmp_all['call_structure_identical']}, unmatched {cmp_all['unmatched']}")
    if args.archived_trace:
        _, arch = summary_from_trace(args.archived_trace)
        report["avg_sl_archived_full_eval"] = per_op_avg_sl(arch, fams)
        a, e = report["avg_sl_archived_full_eval"], report["avg_sl_exact"]
        print("[check] avg_sl (captured windows, exact) vs archived full eval: " + ", ".join(
            f"{k} {e[k]:.2f}/{a[k]:.2f}" for k in e if k in a and k in
            ("all", "linear", "attention", "depthwise")))
    t = report["tile_avg_sl_exact"]
    print("[check] tile-max avg_sl (exact): " + ", ".join(
        f"R={r} {t[str(r)]['all']:.2f}" for r in TILE_ROWS))
    report["ok"] = bool(ok)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"[check] wrote {args.out}")
    print(f"[check] ok={ok}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
