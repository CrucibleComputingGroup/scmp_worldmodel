"""SC (Stochastic Computing) replacement ops for IRASim.

The actual SC Triton kernels live in the ``scmp_kernels`` package, vendored as
a git submodule at ``kernels/`` (repo root) and installed editable with
``pip install -e ./kernels``. This package only holds the IRASim-specific glue
(attention / linear / MLP drop-ins + the global SC controller); the kernels
themselves are shared with the other SC applications, so kernel updates
propagate here automatically by bumping the submodule.

The SC datapath and the mixed-precision mechanism follow scmp_vit
(``model/sc_vit.py`` + ``cls/calibrate_mp_thresholds.py``); run-wide knobs
live in :mod:`.sc_runtime`.

If imports below fail with ``ModuleNotFoundError: scmp_kernels``, the submodule
is not installed — run::

    git submodule update --init --recursive
    pip install -e ./kernels
"""
from ._bootstrap import kernels_provenance  # noqa: F401  (must run first)
from .sc_attention import sc_qk_matmul, sc_av_matmul
from .sc_linear import sc_linear_forward
from .sc_mlp import SCMlp
from .sc_controller import (
    configure,
    reconfigure,
    get_config,
    is_op_enabled,
    set_skip_blocks,
    clear_skip_blocks,
)
from .sc_runtime import (
    configure_from_env,
    cost_summary,
    load_mp_table,
    read_mp_table_provenance,
    set_sc_options,
    state as sc_state,
)

__all__ = [
    "sc_qk_matmul",
    "sc_av_matmul",
    "sc_linear_forward",
    "SCMlp",
    "configure",
    "reconfigure",
    "get_config",
    "is_op_enabled",
    "set_skip_blocks",
    "clear_skip_blocks",
    "configure_from_env",
    "cost_summary",
    "load_mp_table",
    "read_mp_table_provenance",
    "set_sc_options",
    "sc_state",
]
