"""Single work-root and frozen-kit import guard for X-OPT resume v2."""
import importlib
import os
from pathlib import Path
import sys

ROOT = str(Path(os.environ.get("XOPT_WORK_ROOT", "/ceph/dwong/hcp_xopt_20261107")).resolve())
EXP = Path(ROOT) / "EXP"
assert Path(__file__).resolve().is_relative_to(EXP), "XOPT code outside configured EXP"
# Retain interpreter/library paths; all project imports must come from this EXP.
sys.path[:] = [str(EXP), str(EXP / "src")] + [x for x in sys.path if x and "/ceph/dwong/" not in x or x.startswith("/ceph/dwong/meg_env/")]
for name in (
    "experiments.mlst2026.hcp_rnoise_20261019.learning",
    "experiments.mlst2026.hcp_rnoise_20261019.signal",
    "experiments.mlst2026.meg_z_20261013.fit",
    "experiments.mlst2026.t25_production_20260930.science",
):
    mod = importlib.import_module(name)
    location = Path(mod.__file__).resolve()
    assert location.is_relative_to(EXP), f"kit import outside EXP: {name}: {location}"
    print(f"KIT_IMPORT {name} {location}", flush=True)
print(f"XOPT_WORK_ROOT {ROOT}", flush=True)
