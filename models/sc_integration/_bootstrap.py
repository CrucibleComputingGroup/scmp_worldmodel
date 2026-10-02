"""Make the repo's ``kernels/`` submodule the scmp_kernels this package imports
(scmp_vit ``model/_bootstrap.py``).

Machines here carry other editable installs of scmp_kernels (e.g. an env whose
``pip show scmp_kernels`` points at a sibling repo's older vendored copy), so a
bare ``import scmp_kernels`` can silently resolve to the wrong kernels. This
prepends ``<repo>/kernels`` to ``sys.path`` before the first import. Set
``SCMP_USE_INSTALLED_KERNELS=1`` to opt out.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "kernels"


def ensure_kernels_on_path():
    if os.environ.get("SCMP_USE_INSTALLED_KERNELS") == "1":
        return None
    if not (KERNELS_DIR / "scmp_kernels" / "__init__.py").is_file():
        raise RuntimeError(f"kernels submodule missing at {KERNELS_DIR}; run "
                           f"`git submodule update --init kernels`")
    loaded = sys.modules.get("scmp_kernels")
    if loaded is not None:
        got = Path(loaded.__file__).resolve().parents[1]
        if got != KERNELS_DIR.resolve():
            raise RuntimeError(f"scmp_kernels was already imported from {got}, "
                               f"not {KERNELS_DIR}; import models.sc_integration "
                               f"first")
    p = str(KERNELS_DIR)
    if sys.path[:1] != [p]:
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return KERNELS_DIR


def kernels_provenance() -> str:
    import scmp_kernels
    got = Path(scmp_kernels.__file__).resolve().parents[1]
    tag = "repo-local" if got == KERNELS_DIR.resolve() else "EXTERNAL"
    return f"{tag}: {scmp_kernels.__file__}"


ensure_kernels_on_path()
