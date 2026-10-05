"""CPU checks for the per-group precision trace (no model/GPU required).

    PYTHONPATH=kernels:. python -m unittest tests.test_group_trace -v   (from scmp_worldmodel/)

Port of scmp_llm benchmark/ppl/test_group_trace.py. The real sc_linear_forward (adaptive
per-row with protected channels, per-(row, chunk) with a tail chunk, uniform), the real
per_row qk MP dispatch and uniform av, called through the names IRASim resolves
(models.sc_integration.* and sc_mlp.sc_linear_forward), and the real sc_matmul entry
(validation + scmp_kernels.trace recording) run; only the Triton kernels behind sc_matmul
are replaced by FP matmuls. Windows are model forwards, kept with a stride.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "kernels")]


class GroupTraceTests(unittest.TestCase):
    def test_capture_matches_kernels_and_trace(self):
        import torch
        from unittest.mock import patch
        import models.sc_integration as pkg
        import models.sc_integration.sc_linear as lin
        import models.sc_integration.sc_attention as att
        import models.sc_integration.sc_mlp as mlp
        import models.sc_integration.sc_runtime as rt
        import scmp_kernels.sc.matmul as scmm
        from scmp_kernels import trace as sc_trace
        from scmp_kernels.mp import set_current_block_idx
        from scmp_kernels.mp.config import AdaptiveMPConfig
        from evaluate.group_trace import GroupTraceCollector, _CaptureDone
        from evaluate import group_trace_check as chk

        mp = AdaptiveMPConfig([32, 16, 8])
        mp.per_row_chunk = {}
        mp.bucket_thresholds = {("qkv_proj", 0, 0): [.75, .25], ("qk", 0, 0): [.75, .25]}
        mp._load_per_row_chunk({"buckets": {"mlp_fc1:t0:l0": {
            "levels": [8, 16, 32], "thresholds": [.25, .75]}}})
        mp.protected_channel_indices = {("mlp_fc1", 0, None): [0, 1, 2, 3],
                                        ("qkv_proj", 0, None): [5, 9]}
        mp.protected_channel_stoc_len = 64
        saved = dict(rt._STATE)
        rt._STATE.update(halve=True, sc_prec=8, chunk_d=128, qk_granularity="per_row",
                         uniform_stoc_len=24, total_blocks=2, mp_cfg=mp,
                         mp_ops=frozenset({"qkv_proj", "mlp_fc1", "qk"}))
        lin_ = lambda i, o: torch.nn.Linear(i, o, bias=False)
        qkv, fc1, fc2 = lin_(150, 5), lin_(261, 5), lin_(300, 4)
        g = torch.Generator().manual_seed(0)
        windows = [dict(x=torch.randn(6, 150, generator=g) * torch.linspace(0.2, 3.0, 150),
                        x1=torch.randn(6, 261, generator=g) * torch.linspace(0.2, 3.0, 261),
                        x2=torch.randn(6, 300, generator=g),
                        q=torch.randn(1, 2, 5, 17, generator=g)
                        * torch.tensor([.2, 1., 3., .5, 2.])[:, None],
                        k=torch.randn(1, 2, 5, 17, generator=g),
                        p=torch.softmax(torch.randn(1, 2, 5, 5, generator=g), -1),
                        v=torch.randn(1, 2, 5, 17, generator=g)) for _ in range(5)]

        class Model:
            def forward(self, x, t):
                w = windows[x]
                set_current_block_idx(0)
                return [pkg.sc_linear_forward(w["x"], qkv, op="qkv", block_idx=0),
                        pkg.sc_qk_matmul(w["q"], w["k"]), pkg.sc_av_matmul(w["p"], w["v"]),
                        pkg.sc_linear_forward(w["x1"], fc1, op="mlp_fc1", block_idx=0),
                        mlp.sc_linear_forward(w["x2"], fc2, op="mlp_fc2", block_idx=0)]

        executed = []

        def mlp_kernel(a, b, **o):
            if o.get("rung_table") is not None:
                executed.append(("prc", torch.tensor(o["level_lens"])[o["rung_table"].long()]))
            else:
                executed.append(("fixed", o["stoc_len"]))
            return a @ b.T

        def row_kernel(a, b, **o):
            executed.append(("row", o["stoc_len"]))
            return a @ b.T

        def batched_kernel(a, b, **o):
            executed.append(("batched", o["stoc_len"]))
            return a @ b.transpose(-1, -2)

        def run(idx):
            m = Model()
            return [o for i in idx for o in Model.forward(m, i, torch.tensor([999 - 10 * i]))]

        hooks = lambda: (lin.sc_matmul, lin.adaptive_classify_rows, att.sc_matmul,
                         att.adaptive_classify_rows, lin.sc_linear_forward, att.sc_qk_matmul,
                         att.sc_av_matmul, pkg.sc_linear_forward, pkg.sc_qk_matmul,
                         pkg.sc_av_matmul, mlp.sc_linear_forward, Model.forward)
        old = hooks()
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(scmm, "_sc_matmul_per_row_mlp", mlp_kernel), \
                    patch.object(scmm, "_sc_matmul_per_row", row_kernel), \
                    patch.object(scmm, "_sc_matmul_per_row_batched", batched_kernel), \
                    torch.no_grad():
                plain = run(range(3))
                executed.clear()
                trace_path = str(Path(tmp) / "sc_trace.json")
                sc_trace.reset()
                sc_trace.enable(trace_path, mode="summary")
                sc_trace._ENABLED = False
                col = GroupTraceCollector(2, stride=2, sc_trace=sc_trace)
                try:
                    with col.installed(pkg, lin, att, mlp, rt, Model):
                        hooked = run(range(3))          # forwards 0, 1, 2; 0 and 2 kept
                        n_exec = len(executed)
                        with self.assertRaises(_CaptureDone):
                            run([3])
                        self.assertEqual(len(executed), n_exec)
                finally:
                    sc_trace._ENABLED = True
                    sc_trace.flush(trace_path)
                    sc_trace.disable()
                self.assertEqual(old, hooks())
                self.assertTrue(all(torch.equal(p, h) for p, h in zip(plain, hooked)))
                self.assertEqual(sorted(col.calls), [0, 2])
                prc = [e for e in executed if e[0] == "prc"]
                self.assertEqual(len(prc), 3)
                for w, i in ((0, 0), (2, 2)):
                    self.assertTrue(torch.equal(
                        torch.as_tensor(col.maps[f"w{w}/b0/mlp_fc1"]).long(), prc[i][1]))
                m = col.maps["w0/b0/qkv_proj"]
                self.assertEqual(m.shape, (6, 2))
                self.assertGreater(len(set(m[:, 0].tolist())), 1)
                self.assertTrue((col.maps["w2/b0/mlp_fc2"] == 24).all())
                self.assertTrue((col.maps["w0/b0/av"] == 24).all())
                self.assertGreater(len(set(col.maps["w0/b0/qk"].reshape(-1).tolist())), 1)
                self.assertEqual([c["op"] for c in col.calls[2]],
                                 ["qkv_proj", "qk", "av", "mlp_fc1", "mlp_fc2"])
                self.assertEqual(float(col.timesteps[2][0]), 979.0)
                desc = col.ops[("mlp_fc1", 0, None)]
                self.assertEqual((desc["chunk_widths"], desc["protected_cols"],
                                  desc["protected_stoc_len"], desc["dispatch"]),
                                 ([128, 128, 1], 4, 64, "rung_table"))

                col.save(Path(tmp) / "gt", dict(model="stub", sc=dict(rng_levels=128)))
                meta, arrays, ops = chk.load_group_trace(Path(tmp) / "gt")
                groups = chk.summary_from_groups(meta, arrays, ops)
                _, trace = chk.summary_from_trace(trace_path)
                ident = chk.identity(groups, trace)
                self.assertTrue(ident["ok"], json.dumps(ident, indent=1, default=str))
                self.assertGreater(ident["n_keys"], 5)
                flat = chk.tile_avg_sl(meta, arrays, ops)["1"]["all"]
                self.assertAlmostEqual(
                    flat, chk.per_op_avg_sl(groups, chk._op_families(meta))["all"], places=9)
                arrays.close()
        finally:
            rt._STATE.clear()
            rt._STATE.update(saved)


if __name__ == "__main__":
    unittest.main()
