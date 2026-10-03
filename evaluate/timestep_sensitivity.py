#!/usr/bin/env python3
"""Timestep-axis allocation for IRASim SC (per-denoising-step stream length).

The layer axis has no exploitable heterogeneity on IRASim (row / module
sensitivities nearly flat; every layer-axis allocation tied or lost to
uniform). The denoising-step axis does show a direction (PR #4 log E11:
late-vs-early 50/50 split differs by 0.56 dB), and needs no row ranking.

Windows = the sampler's steps split into W equal windows. Probe = uniform L0
with ONE window knocked to L0 - step (via SC_STEP_SCHEDULE, a W-entry list the
runtime resamples onto the step index); sensitivity = increase of the
final-latent MSE to the FP output on the calibration keys (seeded noise).
Allocation: every step costs the same MACs, so the budget is the plain mean of
the per-window lengths; cycles move from the least to the most sensitive
windows within +-move (multiples of 8). Candidates then go through the same
validation gate as e2e_sensitivity.py (uniform is a candidate and wins ties).

  python evaluate/timestep_sensitivity.py plan    --out results/e2t --W 10
  MODE=tprobe ... scripts/run_e2e_probe.sh
  python evaluate/timestep_sensitivity.py collect --out results/e2t --eval_root results/local_n_eval_vit
"""
import argparse, json, os


def cmd_plan(a):
    os.makedirs(a.out, exist_ok=True)
    e2e = json.load(open(f"{a.e2e}/plan.json"))
    probes = {}
    for w in range(a.W):
        sched = [a.L0] * a.W
        sched[w] = a.L0 - a.step
        probes[f"w{w}"] = sched
    json.dump({"L0": a.L0, "step": a.step, "W": a.W, "keys": e2e["keys"], "val_keys": e2e["val_keys"],
               "probes": probes, "fp_tag": "e2e_fp", "base_tag": f"e2e_base_{a.L0}"},
              open(f"{a.out}/plan.json", "w"), indent=1)
    print(f"{a.W} window probes; keys {e2e['keys']}; reuse e2e_fp / e2e_base_{a.L0}")


def cmd_collect(a):
    import torch
    plan = json.load(open(f"{a.out}/plan.json"))
    keys = plan["keys"]

    def lat(tag, k):
        return torch.load(f"{a.eval_root}/{tag}/latents/{k}.pt").float()

    mse = lambda x, y: float(((x - y) ** 2).mean())
    fp = {k: lat(plan["fp_tag"], k) for k in keys}
    base = {k: mse(lat(plan["base_tag"], k), fp[k]) for k in keys}
    sens = {}
    for name in plan["probes"]:
        d = [mse(lat(f"e2t_{name}", k), fp[k]) - base[k] for k in keys]
        sens[name] = (sum(d) / len(d), d)
    W, L0 = plan["W"], plan["L0"]
    order = sorted(range(W), key=lambda w: sens[f"w{w}"][0])     # least sensitive first
    out = {"sens": {k: v[0] for k, v in sens.items()}, "per_key": {k: v[1] for k, v in sens.items()},
           "candidates": {}}
    for T in [int(t) for t in a.targets.split(",")]:
        for mv in [int(m) for m in a.moves.split(",")]:
            mv2 = mv * (2 if T >= 96 else 1)
            lo, hi = max(8, T - mv2), min(128, T + mv2)
            L = [T] * W
            i, j = 0, W - 1
            while i < j:
                up, dn = order[j], order[i]
                if sens[f"w{up}"][0] <= sens[f"w{dn}"][0]:
                    break
                if L[up] + 8 > hi:
                    j -= 1; continue
                if L[dn] - 8 < lo:
                    i += 1; continue
                L[up] += 8; L[dn] -= 8          # equal per-step MACs: 1:1 trade
            assert sum(L) == T * W
            name = f"{T}_m{mv}"
            json.dump(L, open(f"{a.out}/sched_{name}.json", "w"))
            out["candidates"][name] = L
    json.dump(out, open(f"{a.out}/report.json", "w"), indent=1)
    print("base MSE vs FP:", {k: round(v, 5) for k, v in base.items()})
    for w in range(W):
        s, d = sens[f"w{w}"]
        print(f"window {w}: dL {s:+.3e}  per-key {[f'{x:+.2e}' for x in d]}")
    for n, L in out["candidates"].items():
        print(f"candidate {n}: {L}")


def cmd_select(a):
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

    sel = {}
    for T in [int(t) for t in a.targets.split(",")]:
        uni = psnr(f"e2eval_uni_{T}")
        best, bp = "uniform", uni
        row = {"uniform": uni}
        for m in a.moves.split(","):
            pv = psnr(f"e2tval_{T}_m{m}")
            row[f"m{m}"] = pv
            if pv is not None and uni is not None and pv > bp + a.margin:
                best, bp = f"m{m}", pv
        row["selected"] = best
        sel[T] = row
        L = [T] * plan["W"] if best == "uniform" else json.load(open(f"{a.out}/sched_{T}_{best}.json"))
        json.dump(L, open(f"{a.out}/sched_{T}.json", "w"))
        print(f"T={T}: " + "  ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in row.items()))
    json.dump(sel, open(f"{a.out}/selection.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default="results/e2t"); p.add_argument("--e2e", default="results/e2e")
    p.add_argument("--L0", type=int, default=48); p.add_argument("--step", type=int, default=16)
    p.add_argument("--W", type=int, default=10)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="results/e2t"); c.add_argument("--eval_root", required=True)
    c.add_argument("--targets", default="96,48,40"); c.add_argument("--moves", default="8,16,24")
    s = sub.add_parser("select")
    s.add_argument("--out", default="results/e2t"); s.add_argument("--eval_root", required=True)
    s.add_argument("--targets", default="96,48,40"); s.add_argument("--moves", default="8,16,24")
    s.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    {"plan": cmd_plan, "collect": cmd_collect, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
