"""Per-group precision trace for the IRASim (frame_ada, Bridge) SC eval -- the stream
length of EVERY computation group, in execution order, as input for the energy/latency
simulator.

Port of scmp_llm's benchmark/ppl/group_trace.py (CrucibleComputingGroup/scmp_llm,
feat/group-trace 78b7a93); siblings: scmp_vit cls/group_trace.py, scmp_cnn group_trace.py.
Same schema, keys and array layouts, same observe-only approach (hooks wrap the dispatch
entry points; nothing touches dispatch or numerics), same early stop, same identity check
against the kernels' own scmp-trace-summary of the captured windows
(evaluate/group_trace_check.py).

A window is ONE DiT forward (one denoising step of one video; PNDM-50 runs ~59 forwards per
video). Lengths depend on the step, so the capture keeps every GTRACE_STRIDE-th forward of
the first video, GTRACE_WINDOWS of them (default 5 x 12: forwards 0, 5, ..., 55); forwards in
between run normally but are recorded nowhere, and are kept out of the check trace too.
Window ids in the trace are the forward indices.

  group_trace.npz
    w<w>/b<block>/<op>   uint8 stream length of every group of one call
        linears  [rows, n_chunks] (qkv_proj, out_proj, mlp_fc1, mlp_fc2). Row = token of the
                 DiT input (frames * patches, 5120 rows). Chunk = 128 consecutive columns of
                 the DISPATCHED input (protected columns removed, ascending, short tail
                 last). Group MACs = chunk width * d_out. ops[].dispatch: rung_table | rows,
                 as in scmp_vit.
        qk, av   [batch*heads, rows], row index = b*heads + h: one length per (head, query
                 row). Spatial blocks attend over the 320 patches of a frame (batch =
                 frames), temporal blocks over the 16 frames of a patch (batch = patches).
                 Group MACs = d_in * d_out (dense). The kernel summary traces IRASim
                 attention per call, not per head (ops[].trace_unit = null).
    protected/b<block>/<op>   int32 protected input columns of an SC linear
    w<w>/timestep   the diffusion timestep(s) t of this forward (window identity)
  group_trace.json
    header / ops / calls exactly as scmp_llm.

Lengths are SC cycles in the code's halved space (nominal = 2x). SmoothQuant is a
pre-transform (x/s, w*s) applied before dispatch, so the kernels see -- and the trace
records -- smoothed=false. Blocks skipped by --skip run FP and are absent, as are FP ops
(embedders, norms, adaLN modulation, softmax, final layer), as in the existing trace.

Usage -- the env + eval_local_n_samples.py command line of the cell, after this script:
    GTRACE_OUT=/dir [GTRACE_WINDOWS=12] [GTRACE_STRIDE=5] [GTRACE_CHECK_TRACE=/path.json] \\
    [GTRACE_CELL=<id>] SC_HALVE=1 SC_PREC=8 ... python -u evaluate/group_trace.py --config ...
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = "scmp-group-trace-v1"
LINEARS = ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")
COVERAGE = ("SC-MP datapath operator calls only: SC calls with per-group lengths -- FP ops "
            "(embedders, norms, adaLN, softmax, final layer) and --skip blocks are NOT "
            "recorded")
SEMANTICS = {
    "stoc_len": "SC cycles per group, halved (sign-magnitude) space = stoc_len of "
                "scmp-trace-summary-v1; nominal = 2x",
    "window": "one DiT forward (denoising step) of the first video; ids = forward index",
    "linear_group": "(row, 128-chunk of the dispatched input); MACs = chunk width * d_out",
    "linear_rows": "token of the DiT input: frames * patches",
    "linear_dispatch": "rung_table = per-(row, chunk) rung table; rows = one call per length "
                       "on that length's rows (all chunks of a row share its length)",
    "protected": "every row's protected columns run at protected_stoc_len; "
                 "MACs = rows * n_protected * d_out",
    "attention_group": "(head, query row), row index b*heads + h; MACs = d_in * d_out (dense)",
    "smoothquant": "pre-transform before dispatch; smoothed=false in the kernels' records",
}


class _CaptureDone(BaseException):
    """Raised on the first forward after the last kept window."""


def _chunk_widths(width, chunk_d):
    if chunk_d <= 0 or width <= chunk_d:
        return [int(width)]
    n = (width + chunk_d - 1) // chunk_d
    return [int(chunk_d)] * (n - 1) + [int(width - chunk_d * (n - 1))]


class GroupTraceCollector:
    def __init__(self, n_windows, stride=1, sc_trace=None):
        self.keep = set(range(0, int(n_windows) * int(stride), int(stride)))
        self.last = max(self.keep)
        self.window = -1          # forward index
        self.sc_trace = sc_trace  # check trace: enabled only on kept windows
        self.maps, self.protected, self.ops, self.calls = {}, {}, {}, {}
        self.timesteps = {}
        self._ctx = None

    def _keep(self):
        return self.window in self.keep

    def _forward(self, t):
        self.window += 1
        if self.window > self.last:
            raise _CaptureDone()
        if self.sc_trace is not None:
            self.sc_trace._ENABLED = self._keep()
        if self._keep():
            import torch
            tt = t if torch.is_tensor(t) else torch.as_tensor(t)
            self.timesteps[self.window] = tt.detach().reshape(-1).float().cpu().numpy()

    def _op(self, key, **desc):
        if self.ops.setdefault(key, desc) != desc:
            raise ValueError(f"static description of {key} changed between calls")

    def _call(self, **rec):
        calls = self.calls.setdefault(self.window, [])
        calls.append(dict(seq=len(calls), **rec))

    def _store(self, key, lengths):
        import torch
        if lengths.numel() and (int(lengths.max()) > 255 or int(lengths.min()) <= 0):
            raise ValueError(f"stream length outside uint8 range at {key}")
        if key in self.maps:
            raise ValueError(f"two dispatches for one group map: {key}")
        self.maps[key] = lengths.to(torch.uint8).cpu().numpy()

    @contextmanager
    def installed(self, pkg, lin, att, mlp, rt, model_cls):
        import torch
        col = self
        o = dict(lin_mm=lin.sc_matmul, lin_cls=lin.adaptive_classify_rows,
                 att_mm=att.sc_matmul, att_cls=att.adaptive_classify_rows,
                 lin_fwd=lin.sc_linear_forward, qk=att.sc_qk_matmul, av=att.sc_av_matmul,
                 pkg_lin=pkg.sc_linear_forward, pkg_qk=pkg.sc_qk_matmul,
                 pkg_av=pkg.sc_av_matmul, mlp_lin=mlp.sc_linear_forward,
                 model=model_cls.forward)

        def mm(orig):
            def run(a, b, **kw):
                ctx = col._ctx
                if ctx is not None:
                    rt_ = kw.get("rung_table")
                    ctx["mm"].append(dict(
                        rows=int(a.shape[-2]), d_in=int(a.shape[-1]),
                        stoc_len=int(kw["stoc_len"]) if kw.get("stoc_len") is not None else None,
                        rung_table=None if rt_ is None else rt_.detach().clone(),
                        level_lens=(None if kw.get("level_lens") is None
                                    else [int(v) for v in kw["level_lens"]])))
                return orig(a, b, **kw)
            return run

        def classify(orig):
            def run(metric, config, *args, **kw):
                result = orig(metric, config, *args, **kw)
                ctx = col._ctx
                if ctx is None or ctx["cls"] is not None or kw.get("operator") != ctx["op"]:
                    raise ValueError(f"unrecognized dispatch context for {kw.get('operator')}")
                values = config.classify_level_values(
                    operator=ctx["op"], block_idx=kw.get("block_idx"),
                    total_blocks=kw.get("total_blocks"),
                    **{k: kw[k] for k in ("timestep", "total_timesteps") if k in kw})
                ctx["cls"] = ([int(v) for v in values], result.row_levels.detach().clone())
                return result
            return run

        def linear(x, linear_mod, sc_prec=None, stoc_len=None, op=None, block_idx=None):
            top = rt.CTRL_TO_TABLE_OP.get(op, op)
            if not rt.sc_enabled() or x.numel() == 0:
                return o["lin_fwd"](x, linear_mod, sc_prec, stoc_len, op, block_idx)
            if rt.group_stoclen() is not None or top not in LINEARS:
                raise ValueError(f"group trace: knock-down probes / unknown op {top} unsupported")
            rows, d_in = x.numel() // x.shape[-1], int(x.shape[-1])
            d_out = int(linear_mod.out_features)
            cfg = rt.mp_cfg_for(top)
            pidx = torch.empty(0, dtype=torch.long)
            if cfg is not None:
                pidx = lin._channel_index_tensor(
                    cfg.get_protected_channels(operator=top, block_idx=block_idx), d_in, "cpu")
            n_prot = int(pidx.numel())
            width = d_in - n_prot
            previous = col._ctx
            ctx = col._ctx = dict(op=top, block=block_idx, mm=[], cls=None)
            try:
                y = o["lin_fwd"](x, linear_mod, sc_prec, stoc_len, op, block_idx)
                calls = ctx["mm"]
                pcalls = [c for c in calls if n_prot and c["d_in"] == n_prot
                          and c["rows"] == rows and c["rung_table"] is None]
                prot_sl = None
                if n_prot:
                    if len(pcalls) != 1:
                        raise ValueError(f"{top}/{block_idx}: expected one protected call")
                    prot_sl = pcalls[0]["stoc_len"]
                main = [c for c in calls if c not in pcalls]
                prc = [c for c in main if c["rung_table"] is not None]
                if prc:
                    if len(prc) != 1 or len(main) != 1:
                        raise ValueError(f"{top}/{block_idx}: mixed rung-table dispatch")
                    cd = int(rt.state()["chunk_d"])
                    widths = _chunk_widths(width, cd)
                    c = prc[0]
                    lad = c["level_lens"]
                    lengths = torch.as_tensor(lad, device=c["rung_table"].device)[
                        c["rung_table"].long()]
                    if list(lengths.shape) != [rows, len(widths)]:
                        raise ValueError(f"{top}/{block_idx}: rung table {tuple(lengths.shape)}")
                    dispatch = "rung_table"
                else:
                    cd = int(rt.state()["chunk_d"])
                    widths = _chunk_widths(width, rt.chunk_d_for(width))
                    if ctx["cls"] is not None:
                        lad, lv = ctx["cls"]
                        per_row = torch.as_tensor(lad, device=lv.device)[lv.long()]
                    else:
                        if len(main) != 1:
                            raise ValueError(f"{top}/{block_idx}: uniform linear with "
                                             f"{len(main)} calls")
                        lad = [main[0]["stoc_len"]]
                        per_row = torch.full((rows,), lad[0], dtype=torch.long)
                    got = Counter()
                    for c in main:
                        if c["d_in"] != width:
                            raise ValueError(f"{top}/{block_idx}: width {c['d_in']} != {width}")
                        got[c["stoc_len"]] += c["rows"]
                    if got != Counter(per_row.tolist()):
                        raise ValueError(f"{top}/{block_idx}: rows per length {dict(got)} != "
                                         f"classified {dict(Counter(per_row.tolist()))}")
                    lengths = per_row[:, None].expand(rows, len(widths))
                    dispatch = "rows"
                if col._keep():
                    key = f"w{col.window}/b{block_idx}/{top}"
                    col._store(key, lengths)
                    pkey = None
                    if n_prot:
                        pkey = f"protected/b{block_idx}/{top}"
                        if pkey not in col.protected:
                            col.protected[pkey] = pidx.to(torch.int32).numpy()
                    col._op((top, block_idx, None), backend="sc", d_in=d_in, d_out=d_out,
                            chunk_d=cd, chunk_widths=widths, dispatch=dispatch, ladder=lad,
                            smoothed=False, protected_cols=n_prot, protected_key=pkey,
                            protected_stoc_len=int(prot_sl) if n_prot else None)
                    col._call(block=block_idx, op=top, unit=None, backend="sc", rows=rows,
                              d_in=d_in, d_out=d_out, groups=key)
                return y
            finally:
                col._ctx = previous

        def attention(op, orig):
            def run(a, b, sc_prec=None, stoc_len=None):
                if not rt.sc_enabled():
                    return orig(a, b, sc_prec, stoc_len)
                from scmp_kernels.mp import get_current_block_idx
                block = get_current_block_idx()
                bsz, heads, n = (int(v) for v in a.shape[:3])
                k, m = (int(a.shape[-1]), n) if op == "qk" else (n, int(b.shape[-1]))
                if op == "av":
                    k = int(a.shape[-1])
                if rt.group_stoclen() is not None:
                    raise ValueError("group trace: knock-down probes unsupported")
                previous = col._ctx
                ctx = col._ctx = dict(op=op, block=block, mm=[], cls=None)
                try:
                    y = orig(a, b, sc_prec, stoc_len)
                    if ctx["cls"] is not None:
                        lad, lv = ctx["cls"]
                        lengths = torch.as_tensor(lad, device=lv.device)[lv.long()]
                        ran = {c["stoc_len"] for c in ctx["mm"]}
                        if ran != set(lengths.unique().tolist()):
                            raise ValueError(f"{op}/{block}: kernels ran {sorted(ran)}")
                    else:
                        if len(ctx["mm"]) != 1 or ctx["mm"][0]["rows"] != n:
                            raise ValueError(f"{op}/{block}: uniform attention with "
                                             f"{len(ctx['mm'])} calls")
                        lad = [ctx["mm"][0]["stoc_len"]]
                        lengths = torch.full((bsz * heads * n,), lad[0], dtype=torch.long)
                    if col._keep():
                        key = f"w{col.window}/b{block}/{op}"
                        col._store(key, lengths.view(bsz * heads, n))
                        col._op((op, block, None), backend="sc", heads=heads,
                                slice_kind="head", trace_unit=None, d_in=k, d_out=m,
                                ladder=lad, smoothed=False)
                        col._call(block=block, op=op, unit=None, backend="sc", batch=bsz,
                                  heads=heads, rows=n, d_in=k, d_out=m, groups=key)
                    return y
                finally:
                    col._ctx = previous
            return run

        def model_forward(mod, x, t, *a, **kw):
            col._forward(t)
            return o["model"](mod, x, t, *a, **kw)

        lin.sc_matmul, lin.adaptive_classify_rows = mm(o["lin_mm"]), classify(o["lin_cls"])
        att.sc_matmul, att.adaptive_classify_rows = mm(o["att_mm"]), classify(o["att_cls"])
        qk, av = attention("qk", o["qk"]), attention("av", o["av"])
        lin.sc_linear_forward = pkg.sc_linear_forward = mlp.sc_linear_forward = linear
        att.sc_qk_matmul = pkg.sc_qk_matmul = qk
        att.sc_av_matmul = pkg.sc_av_matmul = av
        model_cls.forward = model_forward
        try:
            yield self
        finally:
            lin.sc_matmul, lin.adaptive_classify_rows = o["lin_mm"], o["lin_cls"]
            att.sc_matmul, att.adaptive_classify_rows = o["att_mm"], o["att_cls"]
            lin.sc_linear_forward, att.sc_qk_matmul, att.sc_av_matmul = (
                o["lin_fwd"], o["qk"], o["av"])
            pkg.sc_linear_forward, pkg.sc_qk_matmul, pkg.sc_av_matmul = (
                o["pkg_lin"], o["pkg_qk"], o["pkg_av"])
            mlp.sc_linear_forward = o["mlp_lin"]
            model_cls.forward = o["model"]

    def save(self, out_dir, header):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.maps)
        arrays.update(self.protected)
        for w, t in self.timesteps.items():
            if w in self.calls:
                arrays[f"w{w}/timestep"] = t
        np.savez_compressed(out_dir / "group_trace.npz", **arrays)
        windows = sorted(self.calls)
        ops = [dict(op=op, block=b, unit=u, **d) for (op, b, u), d in sorted(
            self.ops.items(), key=lambda t: (t[0][1], t[0][0]))]
        meta = dict(schema=SCHEMA,
                    header=dict(header, windows=windows, n_maps=len(self.maps),
                                n_groups=int(sum(v.size for v in self.maps.values()))),
                    ops=ops, calls={str(w): self.calls[w] for w in windows})
        (out_dir / "group_trace.json").write_text(json.dumps(meta, indent=1))


def _sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def _git(repo):
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(status) if status is not None else None)


def provenance(repo, argv, rt):
    wrapper = os.environ.get("SC_MP_TABLE")
    table = None
    try:
        table = json.loads(Path(wrapper).read_text()).get("threshold_table_path")
        if table and not os.path.isabs(table):
            table = str(Path(wrapper).parent / table)
    except (OSError, TypeError, ValueError):
        pass
    st = rt.state()
    sc_prec, halve = int(st["sc_prec"]), bool(st["halve"])
    sources = ("models/sc_integration/sc_linear.py", "models/sc_integration/sc_attention.py",
               "models/sc_integration/sc_runtime.py", "models/irasim.py",
               "kernels/scmp_kernels/mp/config.py", "kernels/scmp_kernels/sc/matmul.py",
               "evaluate/eval_local_n_samples.py", "evaluate/group_trace.py")
    return dict(
        model="IRASim frame_ada (bridge)", task="video_prediction",
        cell=os.environ.get("GTRACE_CELL"), argv=argv, mp_config_json=wrapper,
        threshold_table=table, smooth_scales=os.environ.get("SC_SMOOTH_SCALES"),
        sha256={k: _sha256(v) for k, v in (("mp_config_json", wrapper),
                                           ("threshold_table", table))},
        sc=dict(sc_prec=sc_prec, halve=halve, chunk_d=st["chunk_d"],
                qk_granularity=st["qk_granularity"],
                uniform_stoc_len=st.get("uniform_stoc_len"),
                rng_levels=2 ** (sc_prec - 1) if halve else 2 ** sc_prec,
                **{k: os.environ.get(k) for k in ("SC_OWEN_MODE", "SC_SCRAMBLE_MASKS")}),
        code=dict(scmp_worldmodel=_git(repo), kernels=_git(repo / "kernels"),
                  sha256={s: _sha256(repo / s) for s in sources}),
    )


def main() -> int:
    out = os.environ.get("GTRACE_OUT")
    if not out:
        sys.exit("[gtrace] GTRACE_OUT (an output directory) is required")
    n_windows = int(os.environ.get("GTRACE_WINDOWS", "12"))
    stride = int(os.environ.get("GTRACE_STRIDE", "5"))
    check_trace = os.environ.get("GTRACE_CHECK_TRACE")
    if os.environ.get("SC_MP_TRACE"):
        sys.exit("[gtrace] unset SC_MP_TRACE: a truncated capture must not write an eval "
                 "trace (GTRACE_CHECK_TRACE writes the capture's own)")
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    sys.path[:0] = [str(repo)]
    import models.sc_integration as pkg
    import models.sc_integration.sc_linear as lin
    import models.sc_integration.sc_attention as att
    import models.sc_integration.sc_mlp as mlp
    import models.sc_integration.sc_runtime as rt
    from models.irasim import IRASim
    sc_trace = None
    if check_trace:
        from scmp_kernels import trace as sc_trace
        sc_trace.reset()
        sc_trace.enable(check_trace, mode="summary")
        sc_trace._ENABLED = False          # switched on per kept window
    col = GroupTraceCollector(n_windows, stride, sc_trace)
    script = repo / "evaluate" / "eval_local_n_samples.py"
    argv = sys.argv[1:]
    finished = "eval_completed"
    t0 = time.time()
    with col.installed(pkg, lin, att, mlp, rt, IRASim):
        sys.argv = [str(script), *argv]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except _CaptureDone:
            finished = f"stopped_after_forward_{col.last}"
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
            finished = f"eval_exit_{e.code}"
    secs = time.time() - t0
    windows = sorted(col.calls)
    if sc_trace is not None:
        sc_trace._ENABLED = True
        path = sc_trace.flush(check_trace, header_extra=dict(
            truncated_capture=True, windows=windows, finished=finished,
            mp_config_json=os.environ.get("SC_MP_TABLE"),
            note="scmp-trace-summary of the group-trace capture windows only; NOT a "
                 "full eval trace"))
        sc_trace.disable()
        print(f"[gtrace] check trace -> {path}")
    header = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), finished=finished,
                  fp_matmul=False, trajectory="exact: the deployed SC run (first video)",
                  n_windows_requested=n_windows, window_stride=stride,
                  capture_seconds=round(secs, 1), coverage=COVERAGE, semantics=SEMANTICS,
                  check_trace=check_trace, **provenance(repo, argv, rt))
    col.save(out, header)
    n_groups = sum(v.size for v in col.maps.values())
    print(f"[gtrace] {finished} in {secs:.0f}s: windows {windows}, {len(col.maps)} group "
          f"maps, {n_groups} groups, {sum(len(c) for c in col.calls.values())} calls -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
