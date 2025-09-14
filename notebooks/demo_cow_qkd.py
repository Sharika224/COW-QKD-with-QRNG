"""
Demo_COW_QKD.py
Recreates the Demo_COW_QKD.ipynb as a runnable script.
It runs a small simulation and leaves outputs in data/ and results/.
"""
import sys, subprocess
from pathlib import Path

# Resolve repo root (folder that contains src/ and data/)
REPO = Path(__file__).resolve().parents[1]
PYEXE = sys.executable

# Ensure directories exist
(REPO / "data" / "results").mkdir(parents=True, exist_ok=True)
(REPO / "data" / "keys").mkdir(parents=True, exist_ok=True)
(REPO / "results").mkdir(parents=True, exist_ok=True)

# Run the simulator (equivalent to notebook cell)
cmd = [PYEXE, "-m", "src.cow_qkd",
       "--keys", "2000",
       "--attack", "pns",
       "--seed", "123",
       "--out-prefix", "demo"]
print(">>> Running:", " ".join(cmd))
res = subprocess.run(cmd, cwd=str(REPO))
if res.returncode != 0:
    raise SystemExit(res.returncode)

print("Done. Check:")
print(" -", REPO/"data/results/demo_symbols.csv")
print(" -", REPO/"data/results/demo_symbols_metrics.json")
print(" -", REPO/"data/keys/demo_sifted_key.bin")
print(" -", REPO/"results/photon_counts.png")
print(" -", REPO/"results/pns_reduction.png")