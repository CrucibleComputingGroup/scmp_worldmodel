"""GPU check that the MP dispatch path is a strict superset of uniform SC.

If every group is assigned the uniform length, the MP datapath must reproduce the
uniform call bit for bit: (1) a per-(row, chunk) rung-table call whose table points at
the 48 rung, (2) per-row dispatch = row-subset calls at 48 stitched back, (3) attention
per-row MP = gathered row subsets of a 3D batched call. Any difference means the MP path
injects a different noise realization than uniform (a confound for MP-vs-uniform).

    PYTHONPATH=kernels:. python tests/gpu_mp_degenerate.py
"""
import torch
from scmp_kernels import sc_matmul

torch.manual_seed(0)
dev = "cuda"
lad = [16, 24, 32, 48, 64, 96, 128]
kw = dict(granularity="per_row", mode="bipolar", sc_prec=8, group_a=1, group_b=1,
          halve_bipolar_stoc_len=True)
x = (torch.randn(1024, 1152, device=dev) * torch.linspace(.1, 3, 1152, device=dev))
w = torch.randn(3456, 1152, device=dev) * .05
yu = sc_matmul(x, w, stoc_len=48, chunk_d=128, **kw)
rung = torch.full((1024, 9), lad.index(48), dtype=torch.int32, device=dev)
yr = sc_matmul(x, w, stoc_len=128, chunk_d=128, rung_table=rung, level_lens=lad, **kw)
idx = torch.randperm(1024, device=dev)
a, b = idx[:400].sort().values, idx[400:].sort().values
ys = torch.empty_like(yu)
ys[a] = sc_matmul(x[a].contiguous(), w, stoc_len=48, chunk_d=128, **kw)
ys[b] = sc_matmul(x[b].contiguous(), w, stoc_len=48, chunk_d=128, **kw)
q = torch.randn(256, 320, 72, device=dev)
k = torch.randn(256, 320, 72, device=dev)
au = sc_matmul(q, k, stoc_len=48, **kw)
g = torch.argsort(torch.rand(256, 320, device=dev), dim=1)[:, :150].sort(1).values
qa = q.gather(1, g[..., None].expand(-1, -1, 72)).contiguous()
ag = sc_matmul(qa, k, stoc_len=48, **kw)
ref = au.gather(1, g[..., None].expand(-1, -1, 320))
rel = lambda p, r: float((p - r).abs().max() / r.abs().max())
print(f"[degenerate] rung-table all-48 vs uniform : max rel diff {rel(yr, yu):.3e}  "
      f"bit-exact={torch.equal(yr, yu)}")
print(f"[degenerate] row subsets @48 vs uniform    : max rel diff {rel(ys, yu):.3e}  "
      f"bit-exact={torch.equal(ys, yu)}")
print(f"[degenerate] attn row subsets vs uniform   : max rel diff {rel(ag, ref):.3e}  "
      f"bit-exact={torch.equal(ag, ref)}")
fp = x @ w.T
print(f"[degenerate] (scale) uniform vs FP rel err : {float((yu - fp).norm() / fp.norm()):.3e}")
