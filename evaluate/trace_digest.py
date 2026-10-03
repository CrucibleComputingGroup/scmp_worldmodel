#!/usr/bin/env python3
"""(scmp_vit cls/trace_digest.py @ 89c5917; + IRASim result lookup: <result_dir>/<tag>/summary_shard_0.json mean_psnr.)

Digest of scmp_kernels precision traces (SC_MP_TRACE summaries written by
scmp_kernels/trace.py via the sc/matmul.py hook, kernels PR #25).

For every `<cell>_trace.json` in --trace_dir:
  * per-cell: images, kernel calls recorded, MACs/img, row-cycles/img,
    MAC-weighted and row-weighted mean stream length, SC energy (Σ macs·L/128),
    MAC share per stream length, per-op MAC share + mean L
  * one CSV with every (cell, op, block, unit, L, granularity, chunk_d, d_in, d_out,
    calls, rows, macs, row_cycles) group of every cell.

Trace semantics (see scmp_kernels/trace.py): one record per sc_matmul call; a
"group" aggregates calls with identical (block, op, unit, stoc_len, prec, halve,
mode, granularity, chunk_d, d_in, d_out).  rows = Σ rows dispatched at that L,
macs = Σ rows·d_in·d_out (true MACs of those rows), row_cycles = rows·L.
Per-(row,chunk) cells show up as per_row groups whose d_in is the chunk width
(the rung-table kernel is traced chunk by chunk).
"""
import argparse, json, csv, glob, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--trace_dir", required=True)
ap.add_argument("--result_dir", default=None, help="where <cell>.json / <cell>.log with accuracy live (default trace_dir)")
ap.add_argument("--acc_key", default="top1")
ap.add_argument("--n_images", type=int, default=0, help="override images/trace (ViT headers lack eval_images)")
ap.add_argument("--cells", default="", help="comma list of cell tags (default: all *_trace.json)")
ap.add_argument("--out_md", required=True); ap.add_argument("--out_csv", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--ref", default="", help="uniform cell tag: price every (op,block) at its TRUE MACs from this trace (ViT MP traces carry attention padding)")
args = ap.parse_args()
rdir = args.result_dir or args.trace_dir

def acc_of(tag, header):
    for k in ("top1", "mIoU"):
        if k in header: return float(header[k]) * 100, k
    p = f"{rdir}/{tag}/summary_shard_0.json"          # IRASim eval_local_n_samples
    if os.path.exists(p):
        d = json.load(open(p))
        return float(d["mean_psnr"]), "PSNR"
    p = f"{rdir}/{tag}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        if args.acc_key in d: return float(d[args.acc_key]) * 100, args.acc_key
    p = f"{rdir}/{tag}.log"
    if os.path.exists(p):
        m = re.findall(r"mIoU=([0-9.]+)", open(p).read())
        if m: return float(m[-1]) * 100, "mIoU"
    return float("nan"), "?"

tags = args.cells.split(",") if args.cells else sorted(
    os.path.basename(p)[:-len("_trace.json")] for p in glob.glob(f"{args.trace_dir}/*_trace.json"))

ref = {}
if args.ref:
    for g in json.load(open(f"{args.trace_dir}/{args.ref}_trace.json"))["groups"]:
        k = (g["op"], g["block"]); ref[k] = ref.get(k, 0) + g["macs"]
rows_csv = []; md = [f"# {args.title}".rstrip(), ""]
md.append(f"trace dir: `{args.trace_dir}` — one `<cell>_trace.json` per SC cell (schema scmp-trace-summary-v1, "
          "kernels PR #25 `trace.py`). Energy = Σ macs·L/128 (INT8-MAC equivalents), SC part only; "
          "hybrid cells' INT modules are not in the trace (coverage = sc_matmul only)."
          + (f" `E true` re-prices each (op,block) at the true MACs of the `{args.ref}` trace (removes MP attention padding)." if ref else ""))
md.append("")
md.append("| cell | acc | imgs | calls | groups | MACs/img (G) | row-cycles/img (G) | L̄ MAC-wtd | L̄ row-wtd | SC energy/img (G INT8-MAC) | E true | MAC share by L |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
per_op_md = []
for tag in tags:
    tp = f"{args.trace_dir}/{tag}_trace.json"
    if not os.path.exists(tp): md.append(f"| {tag} | (no trace) |"); continue
    d = json.load(open(tp)); h = d["header"]; G = d["groups"]
    n_img = args.n_images or h.get("eval_images") or 0
    acc, ak = acc_of(tag, h)
    macs = sum(g["macs"] for g in G); rows = sum(g["rows"] for g in G); cyc = sum(g["row_cycles"] for g in G)
    calls = sum(g["calls"] for g in G)
    L_mac = sum(g["macs"] * g["stoc_len"] for g in G) / macs
    L_row = cyc / rows
    energy = sum(g["macs"] * g["stoc_len"] for g in G) / 128
    e_true = float("nan")
    if ref:
        acc_ = {}
        for g in G:
            k = (g["op"], g["block"]); e = acc_.setdefault(k, [0.0, 0.0]); e[0] += g["macs"] * g["stoc_len"]; e[1] += g["macs"]
        e_true = sum(ref[k] * num / den / 128 for k, (num, den) in acc_.items() if k in ref and den > 0)
    byL = defaultdict(float)
    for g in G: byL[g["stoc_len"]] += g["macs"]
    byL_s = " ".join(f"L{L}:{v / macs * 100:.0f}%" for L, v in sorted(byL.items()))
    md.append(f"| {tag} | {acc:.2f} | {n_img} | {calls} | {len(G)} | {macs / n_img / 1e9:.3f} | {cyc / n_img / 1e9:.3f} | "
              f"{L_mac:.1f} | {L_row:.1f} | {energy / n_img / 1e9:.3f} | {e_true / n_img / 1e9:.3f} | {byL_s} |")
    # per-op
    op = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # macs, macs*L, rows, groups
    for g in G:
        o = op[g["op"]]; o[0] += g["macs"]; o[1] += g["macs"] * g["stoc_len"]; o[2] += g["rows"]; o[3] += 1
    per_op_md.append(f"\n### {tag}  ({ak} {acc:.2f})\n\n| op | MAC share | L̄ MAC-wtd | rows/img (M) | groups | Ls used |\n|---|---|---|---|---|---|")
    for o, (m, mL, r, ng) in sorted(op.items(), key=lambda kv: -kv[1][0]):
        Ls = sorted({g["stoc_len"] for g in G if g["op"] == o})
        per_op_md.append(f"| {o} | {m / macs * 100:.1f}% | {mL / m:.1f} | {r / n_img / 1e6:.2f} | {ng} | {','.join(map(str, Ls))} |")
    for g in G:
        rows_csv.append(dict(cell=tag, op=g["op"], block=g["block"], unit=g["unit"], stoc_len=g["stoc_len"],
                             granularity=g["granularity"], chunk_d=g["chunk_d"], d_in=g["d_in"], d_out=g["d_out"],
                             dims_vary=g["dims_vary"], calls=g["calls"], rows=g["rows"], macs=g["macs"],
                             row_cycles=g["row_cycles"], macs_per_img=g["macs"] / n_img,
                             row_cycles_per_img=g["row_cycles"] / n_img))
md.append("\n## per-op breakdown"); md.extend(per_op_md)
md.append(f"\nfull per-group dump: `{args.out_csv}` ({len(rows_csv)} rows)")
open(args.out_md, "w").write("\n".join(md) + "\n")
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys())); w.writeheader(); w.writerows(rows_csv)
print("\n".join(md[:len(tags) + 6]))
print(f"\nwrote {args.out_md} and {args.out_csv}")
