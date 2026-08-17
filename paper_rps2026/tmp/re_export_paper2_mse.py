"""D7 re-export: full-precision held-out metrics from the archived Paper 2 run.

Source of truth: src/NFPA/nfpa_demo.py in
/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction (deterministic,
np.random.seed(42)). The archived paper table truncated IsoAE/IsoPCA raw MSE to
< 10^-5; this script re-imports that exact archived run and prints every
held-out metric at full precision for transcription into
tables/results_summary.tex.

Run from the paper repo with the noise-weighted-subspace-reconstruction venv:
  source /Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/.venv/bin/activate
  python tmp/re_export_paper2_mse.py
"""
import importlib.util
import sys
from pathlib import Path

DEMO = (
    Path("/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction")
    / "src" / "NFPA" / "nfpa_demo.py"
)

spec = importlib.util.spec_from_file_location("nfpa_demo_archive", DEMO)
mod = importlib.util.module_from_spec(spec)
sys.modules["nfpa_demo_archive"] = mod
spec.loader.exec_module(mod)

R = mod.R
print("Held-out metrics at full precision (archived run, seed 42):")
for method, d in R.items():
    print(f"  {method:<8} chi2={d['chi2']:.9e}  raw_mse={d['mse']:.9e}")
