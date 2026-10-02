"""(Copied verbatim from scmp_vit cls/bitmod_wq.py @ fcf56ea.)

Deterministic weight quantizers from BitMoD-HPCA-25 (abdelfattah-lab), wrapped
for arbitrary [K, C] weights: INT (sym/asym), FP5/FP6, MX, and BitMoD's mixed
datatypes (per-group search among FP*_ER / FP*_EA). Group of ``group`` input
columns per output row (BitMoD default 128); a tail narrower than ``group`` is
its own group; C < group means one group per output row. No stochastic
computing anywhere: pure round-to-nearest fake quantization."""
import os, sys
import torch
_BITMOD = os.environ.get("BITMOD_ROOT",
                         "/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/BitMoD-HPCA-25/bitmod_quant")
if _BITMOD not in sys.path:
    sys.path.insert(0, _BITMOD)
from quant_utils.quant_weight import (quant_int, quant_int_asym, quant_datatype,  # noqa: E402
                                      search_datatype, quant_mx)

GRID = {   # bits -> datatypes evaluated (BitMoD repo's own menu per precision)
    3: ["int3_asym", "mixed_bitmod", "mx_fp3"],
    4: ["int4_asym", "mixed_bitmod", "mx_fp4"],
    5: ["int5_asym", "fp5_e2m2"],
    6: ["int6_asym", "fp6_e2m3"],
    8: ["int8_asym"],
}


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int, datatype: str, group: int = 128) -> torch.Tensor:
    """w: [K, C] float -> dequantized float32 of the same shape."""
    if datatype in ("fp16", "fp32", "fp", None):
        return w.float()
    w16 = w.detach().to(torch.float16)
    K, C = w16.shape
    g = 32 if datatype.startswith("mx") else int(group)
    out = []
    for s in range(0, C, g):
        blk = w16[:, s:s + g].contiguous()
        if datatype.startswith("int") and "asym" in datatype:
            q = quant_int_asym(blk, wq_bits=bits, group_size=None)
        elif datatype.startswith("int"):
            q = quant_int(blk, wq_bits=bits, group_size=None)
        elif datatype.startswith("mx"):
            q = quant_mx(blk, wq_bits=bits, datatype=datatype, group_size=blk.shape[1])
        elif datatype.startswith("mixed"):
            q = search_datatype(blk, wq_bits=bits, datatype=datatype, group_size=None)
        else:
            q = quant_datatype(blk, wq_bits=bits, datatype=datatype, group_size=None)
        out.append(q.float())
    return torch.cat(out, dim=1)
