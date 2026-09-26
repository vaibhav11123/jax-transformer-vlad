"""Chinchilla — multi-scale training sweep (rectangular N×D sanity check).

Goal
----
Empirically check scaling behavior on the digit-addition Transformer
by training several model sizes (N) at several token budgets (D).

Pipeline
--------
  grid in this file
      → train.main(cfg)   # logs N, D, C≈6ND, final_val_loss
      → results/sweep_results.csv
      → analyze_scaling.py
      → figures/scaling_plot.pdf

No numbers are typed into Excel by hand; the plot must come from this CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import Config
from train import main

_HERE = Path(__file__).resolve().parent
CSV_PATH = _HERE / "results" / "sweep_results.csv"

# --- Vary N (capacity): change width / depth. Keep n_embd % n_head == 0. ---
MODELS = [
    {"n_embed": 64, "n_head": 4, "n_layer": 2},
    {"n_embed": 128, "n_head": 4, "n_layer": 4},
    {"n_embed": 256, "n_head": 4, "n_layer": 4},
]

# --- Vary D (data): change max_iters only. ---
# batch_size and n_digits stay fixed so block_size (sequence length) is constant;
# then D = max_iters * batch_size * block_size scales linearly with max_iters.
MAX_ITERS_LIST = [500, 1500, 3000]

# Columns written for analyze_scaling.py / later PDF tables
FIELDS = [
    "n_embed", "n_layer", "n_head", "batch_size", "max_iters",
    "N", "D", "C", "final_val_loss", "wandb_run_name",
]


def main_sweep():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()

    # Append mode: re-running sweep adds rows; delete the CSV to start clean.
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()

        for m in MODELS:
            for steps in MAX_ITERS_LIST:
                cfg = Config()
                cfg.wandb_project = "vlad_exercise_2"
                cfg.n_digits = 2  # match committed Exp1 CSV (2-digit runs)
                cfg.n_embd = m["n_embed"]
                cfg.n_head = m["n_head"]
                cfg.n_layer = m["n_layer"]
                cfg.batch_size = 64
                cfg.max_iters = steps
                cfg.wandb_run_name = f"n{cfg.n_embd}_L{cfg.n_layer}_steps{steps}"

                row = main(cfg)
                w.writerow({k: row[k] for k in FIELDS})
                f.flush()


if __name__ == "__main__":
    main_sweep()
