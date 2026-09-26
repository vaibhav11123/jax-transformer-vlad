"""Chinchilla — scaling plot from logged sweep data only.

Reads results/sweep_results.csv produced by sweep.py and writes
figures/scaling_plot.pdf. This is the reproducibility requirement:

  configs → sweep → CSV → this script → figure → (later) PDF

Do not rebuild the figure by pasting numbers into Excel.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
CSV_PATH = _HERE / "results" / "sweep_results.csv"
FIG_DIR = _HERE / "figures"

rows = list(csv.DictReader(CSV_PATH.open()))
if not rows:
    raise SystemExit(f"no rows in {CSV_PATH}; run sweep.py first")

# X: estimated compute C≈6ND from each training run
# Y: final validation loss for that (N, D) configuration
C = [float(r["C"]) for r in rows]
loss = [float(r["final_val_loss"]) for r in rows]

FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(5, 4))
plt.scatter(C, loss)
plt.xscale("log")  # compute spans orders of magnitude across the grid
plt.xlabel("C ≈ 6ND (approx. training FLOPs)")
plt.ylabel("final val loss")
plt.title("Chinchilla: loss vs compute (from sweep CSV)")
plt.tight_layout()
plt.savefig(FIG_DIR / "scaling_plot.pdf")
plt.savefig(FIG_DIR / "scaling_plot.png", dpi=150)
print(f"wrote {FIG_DIR / 'scaling_plot.pdf'}  ({len(rows)} points)")
