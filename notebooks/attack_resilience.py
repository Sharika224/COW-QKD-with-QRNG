"""
Attack_Resilience.py
Recreates the Attack_Resilience.ipynb as a runnable script.
It runs a larger simulation to produce reduction plots.
"""
import sys, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYEXE = sys.executable

(REPO / "data" / "results").mkdir(parents=True, exist_ok=True)
(REPO / "data" / "keys").mkdir(parents=True, exist_ok=True)
(REPO / "results").mkdir(parents=True, exist_ok=True)

cmd = [PYEXE, "-m", "src.cow_qkd",
       "--keys", "10000",
       "--attack", "pns",
       "--seed", "42",
       "--out-prefix", "exp"]
print(">>> Running:", " ".join(cmd))
res = subprocess.run(cmd, cwd=str(REPO))
if res.returncode != 0:
    raise SystemExit(res.returncode)

print("Done. Check:")
print(" -", REPO/"data/results/exp_symbols.csv")
print(" -", REPO/"data/results/exp_symbols_metrics.json")
print(" -", REPO/"data/keys/exp_sifted_key.bin")
print(" -", REPO/"results/photon_counts.png")
print(" -", REPO/"results/pns_reduction.png")