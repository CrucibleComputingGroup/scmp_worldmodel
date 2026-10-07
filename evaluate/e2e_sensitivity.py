#!/usr/bin/env python3
"""End-to-end-sensitivity cross-layer allocation for IRASim SC (no row mixing).

Why: on IRASim the scmp_vit allocator's currency (per-matmul relative error
sigma, MAC-weighted) points the wrong way -- the MP tables lower deployed sigma
by 10% yet lose PSNR -- and mixing rungs inside a group costs ~0.65 dB through
the convexity of the error in L. So this allocator (a) measures each group's
sensitivity END TO END and (b) keeps every group uniform (one L per
(op, block)), moving cycles only BETWEEN groups.

Groups = 6 ops x 4 layer buckets (7 blocks each). Probe = uniform at L0 with
ONE group knocked down to L0 - step; sensitivity = increase of the final-latent
MSE to the FP output (same seed / initial noise per key, so the difference is
deterministic per key), averaged over the calibration keys.

Allocation (LP within a trust region, the measured range): each group may move
by -step..+step in multiples of 8; cycles flow from the least to the most
sensitive groups per MAC-cycle until the MAC-weighted budget is balanced.

  python evaluate/e2e_sensitivity.py plan    --out results/e2e --L0 48 --step 16
  (run the probes: scripts/run_e2e_probe.sh)
  python evaluate/e2e_sensitivity.py collect --out results/e2e --mac_trace <uniform trace>
  (MODE=val: candidates + uniform on the validation keys)
  python evaluate/e2e_sensitivity.py select  --out results/e2e --eval_root <...>
  (MODE=eval: the selected e2e_<T>.json on the test keys)

Selection gate: uniform is a candidate and wins ties, so the deployed
allocation is >= uniform on the validation keys by construction (MP contains
uniform as a special case; the measured allocation only deviates from it when
the end-to-end metric says so).
"""
import argparse, glob, json, os, random
from collections import defaultdict

OPS = ["qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2", "qk", "av"]


def buckets(n_blocks, n_buckets):
    per = n_blocks // n_buckets
    return [list(range(i * per, n_blocks if i == n_buckets - 1 else (i + 1) * per))
            for i in range(n_buckets)]


def cmd_plan(a):
    os.makedirs(f"{a.out}/probes", exist_ok=True)
    gt = os.path.join(os.environ.get("BRIDGE_ROOT", "robotdata/opensource_robotdata/bridge"),
                      "evaluation_latent_videos/test_sample_latent_videos")
    ann = os.path.join(os.environ.get("BRIDGE_ROOT", "robotdata/opensource_robotdata/bridge"),
                       "annotation/test")
    excl = set(json.load(open(a.exclude_keys)))
    pool = sorted(f[:-3] for f in os.listdir(gt) if f.endswith(".pt") and f[:-3] not in excl
                  and os.path.exists(os.path.join(ann, "_".join(f[:-3].split("_")[:-2]) + ".json")))
    random.Random(a.seed).shuffle(pool)
    ok = []
    for k in pool:
        eid, start = "_".join(k.split("_")[:-2]), int(k.split("_")[-1])
        if start + 16 <= len(json.load(open(os.path.join(ann, eid + ".json")))["state"]):
            ok.append(k)
        if len(ok) == a.n_keys + a.n_val:
            break
    keys, val = ok[:a.n_keys], ok[a.n_keys:]
    json.dump(keys, open(f"{a.out}/calib_keys.json", "w"))
    json.dump(val, open(f"{a.out}/val_keys.json", "w"))
    names = []
    for op in OPS:
        for bi, bl in enumerate(buckets(a.n_blocks, a.n_buckets)):
            name = f"{op}_l{bi}"
            json.dump({"group_stoclen": {f"{op}:b{b}": a.L0 - a.step for b in bl},
                       "probe": {"op": op, "bucket": bi, "blocks": bl, "L0": a.L0,
                                 "L": a.L0 - a.step}},
                      open(f"{a.out}/probes/{name}.json", "w"))
            names.append(name)
    json.dump({"L0": a.L0, "step": a.step, "n_blocks": a.n_blocks, "n_buckets": a.n_buckets,
               "keys": keys, "val_keys": val, "probes": names, "seed": a.seed,
               "exclude_keys": a.exclude_keys}, open(f"{a.out}/plan.json", "w"), indent=1)
    print(f"{len(names)} probes, calib keys {keys}, validation keys {val}")


def _mse(x, y):
    import torch
    return float(((x.float() - y.float()) ** 2).mean())


def cmd_collect(a):
    import torch
    plan = json.load(open(f"{a.out}/plan.json"))
    L0, step = plan["L0"], plan["step"]
    root = a.eval_root

    def lat(tag, k):
        return torch.load(f"{root}/{tag}/latents/{k}.pt")

    keys = plan["keys"]
    fp = {k: lat("e2e_fp", k) for k in keys}
    base = {k: _mse(lat(f"e2e_base_{L0}", k), fp[k]) for k in keys}
    sens = {}
    for name in plan["probes"]:
        d = [_mse(lat(f"e2e_{name}", k), fp[k]) - base[k] for k in keys]
        sens[name] = sum(d) / len(d)
    # MACs per (op, block) from a padding-free uniform trace of this model
    macs = defaultdict(float)
    for g in json.load(open(a.mac_trace))["groups"]:
        macs[(g["op"], g["block"])] += g["macs"]
    bl = buckets(plan["n_blocks"], plan["n_buckets"])
    groups = []
    for name in plan["probes"]:
        op, bi = name.rsplit("_l", 1)
        bi = int(bi)
        m = sum(macs.get((op, b), 0.0) for b in bl[bi])
        if m <= 0:
            continue                       # all blocks skipped (FP): no SC cost
        groups.append({"name": name, "op": op, "bucket": bi, "blocks": bl[bi], "macs": m,
                       "dL": sens[name], "s": sens[name] / (m * step)})
    tot = sum(g["macs"] for g in groups)
    report = {"L0": L0, "step": step, "base_mse": base, "groups": groups, "allocations": {}}
    jobs = [(int(t), int(m)) for t in a.targets.split(",") for m in a.moves.split(",")]
    for T, mv in jobs:
        st = mv * (2 if T >= 96 else 1)
        lo, hi = max(8, T - st), min(128, T + st)
        L = {g["name"]: T for g in groups}
        order = sorted(groups, key=lambda g: g["s"])       # least sensitive first
        i, j = 0, len(order) - 1
        cost = lambda: sum(g["macs"] * L[g["name"]] for g in groups) / tot
        # move 8 cycles at a time: up on the most sensitive, down on the least,
        # keeping the MAC-weighted average <= T
        while i < j:
            up, dn = order[j], order[i]
            if up["s"] <= dn["s"]:
                break
            if L[up["name"]] + 8 > hi:
                j -= 1; continue
            if L[dn["name"]] - 8 < lo:
                i += 1; continue
            L[up["name"]] += 8
            while cost() > T + 1e-9 and i < j:
                if L[dn["name"]] - 8 < lo:
                    i += 1; dn = order[i]
                    if i >= j:
                        break
                    continue
                L[dn["name"]] -= 8
            if cost() > T + 1e-9:          # could not pay for it: undo
                L[up["name"]] -= 8
                break
        # spend the leftover budget: +8 on the most sensitive groups that still fit
        for g in sorted(groups, key=lambda g: -g["s"]):
            while L[g["name"]] + 8 <= hi:
                L[g["name"]] += 8
                if cost() > T + 1e-9:
                    L[g["name"]] -= 8
                    break
        table = {"group_stoclen": {f"{g['op']}:b{b}": L[g["name"]] for g in groups for b in g["blocks"]},
                 "target": T, "realized_mac_avg": cost(), "method": "e2e_knockdown_lp",
                 "probe_L0": L0, "probe_step": step, "move": st}
        json.dump(table, open(f"{a.out}/e2e_{T}_m{mv}.json", "w"), indent=1)
        report["allocations"][f"{T}_m{mv}"] = {g["name"]: L[g["name"]] for g in groups}
        report["allocations"][f"{T}_m{mv}_realized"] = cost()
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
    print(f"base MSE vs FP: {base}")
    tags = [f"{T}_m{mv}" for T, mv in jobs]
    print(f"{'group':14s} {'MAC%':>6s} {'dL':>10s} {'dL/MACcyc':>10s}  " + " ".join(f"{t:>8s}" for t in tags))
    for g in sorted(groups, key=lambda g: -g["s"]):
        print(f"{g['name']:14s} {g['macs']/tot*100:6.2f} {g['dL']:+10.3e} {g['s']:+10.3e}  " +
              " ".join(f"{report['allocations'][t][g['name']]:8d}" for t in tags))
    for t in tags:
        print(f"candidate {t}: realized MAC-avg {report['allocations'][t + '_realized']:.2f}")


def cmd_select(a):
    """Pick, per target, the best of {uniform} U {e2e_<T>_m<move>} on the
    VALIDATION keys (paired, seeded), by mean PSNR. Uniform wins ties, so the
    deployed allocation is never worse than uniform on validation by construction."""
    import shutil
    plan = json.load(open(f"{a.out}/plan.json"))
    vk = plan["val_keys"]

    def psnr(tag):
        v = []
        for k in vk:
            p = f"{a.eval_root}/{tag}/metrics/{k}.json"
            if not os.path.exists(p):
                return None
            v.append(json.load(open(p))["psnr"])
        return sum(v) / len(v)

    out = {}
    for T in [int(t) for t in a.targets.split(",")]:
        uni = psnr(f"e2eval_uni_{T}")
        best, best_p = "uniform", uni
        cands = {}
        for m in a.moves.split(","):
            pv = psnr(f"e2eval_{T}_m{m}")
            cands[f"m{m}"] = pv
            if pv is not None and uni is not None and pv > best_p + a.margin:
                best, best_p = f"m{m}", pv
        out[T] = {"uniform": uni, **cands, "selected": best}
        dst = f"{a.out}/e2e_{T}.json"
        if best == "uniform":
            json.dump({"group_stoclen": {}, "target": T, "selected": "uniform",
                       "validation_psnr": out[T]}, open(dst, "w"), indent=1)
        else:
            t = json.load(open(f"{a.out}/e2e_{T}_{best}.json"))
            t["selected"] = best; t["validation_psnr"] = out[T]
            json.dump(t, open(dst, "w"), indent=1)
        print(f"T={T}: " + "  ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in out[T].items()))
    json.dump(out, open(f"{a.out}/selection.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default="results/e2e"); p.add_argument("--L0", type=int, default=48)
    p.add_argument("--step", type=int, default=16); p.add_argument("--n_blocks", type=int, default=28)
    p.add_argument("--n_buckets", type=int, default=4); p.add_argument("--n_keys", type=int, default=3)
    p.add_argument("--n_val", type=int, default=4)
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--exclude_keys", default="results/diverse_300.json")
    c = sub.add_parser("collect")
    c.add_argument("--out", default="results/e2e"); c.add_argument("--eval_root", required=True)
    c.add_argument("--mac_trace", required=True); c.add_argument("--targets", default="96,48,40")
    c.add_argument("--moves", default="8,16,24",
                   help="trust-region sizes (+-cycles; doubled for T>=96), one candidate each")
    v = sub.add_parser("select")
    v.add_argument("--out", default="results/e2e"); v.add_argument("--eval_root", required=True)
    v.add_argument("--targets", default="96,48,40"); v.add_argument("--moves", default="8,16,24")
    v.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    {"plan": cmd_plan, "collect": cmd_collect, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
