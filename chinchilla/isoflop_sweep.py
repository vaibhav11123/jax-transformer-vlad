"""Chinchilla — Experiment 2: fixed-compute (IsoFLOP) allocation sweep.

Hoffmann / Chinchilla asks: given fixed compute C ≈ 6 N D, how should we
split that budget between model size N and tokens D?

This script holds C approximately constant within each budget_id and varies
(N, D) with ND ≈ C/6. The best loss in each budget estimates (N*, D*).

Pipeline
--------
  BUDGETS grid below
      → reuse Exp1 CSV row when exact match, else train.main(cfg)
      → results/isoflop_results.csv
      → analyze_isoflop.py
      → figures/isoflop_loss_vs_N.pdf, figures/Nstar_Dstar_vs_C.pdf

Experiment 1 (rectangular N×D) stays in results/sweep_results.csv — do not overwrite.
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
EXP1_CSV = _HERE / "results" / "sweep_results.csv"
ISO_CSV = _HERE / "results" / "isoflop_results.csv"

# Columns for IsoFLOP results (Steps = max_iters; Tokens = D — keep them separate)
FIELDS = [
    "budget_id",
    "C_target",
    "n_embed",
    "n_layer",
    "n_head",
    "batch_size",
    "max_iters",  # optimizer steps (NOT Chinchilla D)
    "N",          # parameters
    "D",          # training tokens = max_iters * batch_size * block_size
    "C",          # measured ≈ 6 N D
    "final_val_loss",
    "wandb_run_name",
    "source",     # "reuse_exp1" or "train"
]

# ---------------------------------------------------------------------------
# Fixed-compute budgets. Within each budget, ND ≈ C_target/6.
# batch_size=64, n_digits=2 → block_size=8 → D = max_iters * 512.
#
# Budget A ≈ 1.54e11 FLOPs, B ≈ 1.20e12, C ≈ 4.98e12.
# Two points match Experiment 1 exactly and will be reused from Exp1 CSV.
# ---------------------------------------------------------------------------
BUDGETS = [
    {
        "budget_id": "A",
        "C_target": 1.54e11,
        "points": [
            # Reuse Exp1: n64_L2_steps500
            {"n_embed": 64, "n_head": 4, "n_layer": 2, "max_iters": 500},
            {"n_embed": 64, "n_head": 4, "n_layer": 4, "max_iters": 248},
            {"n_embed": 128, "n_head": 4, "n_layer": 2, "max_iters": 125},
        ],
    },
    {
        "budget_id": "B",
        "C_target": 1.20e12,
        "points": [
            {"n_embed": 128, "n_head": 4, "n_layer": 2, "max_iters": 976},
            # Reuse Exp1: n128_L4_steps500
            {"n_embed": 128, "n_head": 4, "n_layer": 4, "max_iters": 500},
            {"n_embed": 256, "n_head": 4, "n_layer": 2, "max_iters": 246},
        ],
    },
    {
        "budget_id": "C",
        "C_target": 4.98e12,
        "points": [
            {"n_embed": 160, "n_head": 4, "n_layer": 3, "max_iters": 1739},
            {"n_embed": 256, "n_head": 4, "n_layer": 3, "max_iters": 682},
            {"n_embed": 256, "n_head": 4, "n_layer": 4, "max_iters": 512},
        ],
    },
]


def load_exp1_index() -> dict[tuple, dict]:
    """Map (n_embed, n_layer, n_head, max_iters) → Exp1 row for reuse."""
    idx = {}
    if not EXP1_CSV.exists():
        return idx
    with EXP1_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (
                int(row["n_embed"]),
                int(row["n_layer"]),
                int(row["n_head"]),
                int(row["max_iters"]),
            )
            idx[key] = row
    return idx


def append_iso_row(row: dict) -> None:
    ISO_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not ISO_CSV.exists()
    with ISO_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})
        f.flush()


def run_point(budget_id: str, C_target: float, point: dict, exp1: dict) -> dict:
    key = (point["n_embed"], point["n_layer"], point["n_head"], point["max_iters"])

    # Reuse Experiment 1 if this (N architecture, steps) was already trained
    if key in exp1:
        r = exp1[key]
        print(f"[reuse Exp1] {budget_id} {key}")
        return {
            "budget_id": budget_id,
            "C_target": C_target,
            "n_embed": int(r["n_embed"]),
            "n_layer": int(r["n_layer"]),
            "n_head": int(r["n_head"]),
            "batch_size": int(r["batch_size"]),
            "max_iters": int(r["max_iters"]),
            "N": int(r["N"]),
            "D": int(r["D"]),
            "C": int(r["C"]),
            "final_val_loss": float(r["final_val_loss"]),
            "wandb_run_name": r["wandb_run_name"],
            "source": "reuse_exp1",
        }

    # Otherwise train a new IsoFLOP point
    cfg = Config()
    cfg.wandb_project = "vlad_exercise_2"
    cfg.n_digits = 2
    cfg.n_embd = point["n_embed"]
    cfg.n_head = point["n_head"]
    cfg.n_layer = point["n_layer"]
    cfg.batch_size = 64
    cfg.max_iters = point["max_iters"]
    # Shorter eval cadence on short runs so we still get a last-step val loss
    cfg.eval_interval = max(50, cfg.max_iters // 10)
    cfg.eval_iters = min(20, cfg.eval_iters)
    cfg.wandb_run_name = (
        f"iso_{budget_id}_n{cfg.n_embd}_L{cfg.n_layer}_steps{cfg.max_iters}"
    )

    print(f"[train] {cfg.wandb_run_name}")
    metrics = main(cfg)
    return {
        "budget_id": budget_id,
        "C_target": C_target,
        **{k: metrics[k] for k in [
            "n_embed", "n_layer", "n_head", "batch_size", "max_iters",
            "N", "D", "C", "final_val_loss", "wandb_run_name",
        ]},
        "source": "train",
    }


def main_isoflop() -> None:
    # Start clean so analyze_isoflop always sees exactly this grid
    if ISO_CSV.exists():
        ISO_CSV.unlink()

    exp1 = load_exp1_index()
    print(f"Writing IsoFLOP results to {ISO_CSV.resolve()}")

    for budget in BUDGETS:
        bid = budget["budget_id"]
        Ct = budget["C_target"]
        print("\n" + "=" * 60)
        print(f"Budget {bid}: C_target ≈ {Ct:.3e}  (hold ND ≈ C/6)")
        print("=" * 60)
        for point in budget["points"]:
            row = run_point(bid, Ct, point, exp1)
            append_iso_row(row)
            # Sanity: measured C should be close to target within each budget
            rel = abs(row["C"] - Ct) / Ct
            print(
                f"  N={row['N']}  steps={row['max_iters']}  D_tokens={row['D']}  "
                f"C={row['C']:.3e}  rel_err={rel:.2%}  loss={row['final_val_loss']:.4f}  "
                f"[{row['source']}]"
            )

    print(f"\nDone. IsoFLOP CSV: {ISO_CSV}")


if __name__ == "__main__":
    main_isoflop()
