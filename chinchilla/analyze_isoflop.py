"""Chinchilla — analyze IsoFLOP (fixed-compute) results.

Reads results/isoflop_results.csv only (no Excel) and writes:

  figures/isoflop_loss_vs_N.pdf
      For each compute budget, loss vs N. Marks N* = argmin loss.

  figures/Nstar_Dstar_vs_C.pdf
      Across budgets: log N* and log D* vs log C*.
      Fits N* ∝ C^a and D* ∝ C^b.
      Chinchilla / equal α,β predicts a ≈ b ≈ 0.5.

Terminology
-----------
  max_iters = optimizer steps
  D         = training tokens  (not steps)
  C         = ≈ 6 N D
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
ISO_CSV = _HERE / "results" / "isoflop_results.csv"
FIG_DIR = _HERE / "figures"


def load_rows() -> list[dict]:
    if not ISO_CSV.exists():
        raise SystemExit(f"missing {ISO_CSV}; run isoflop_sweep.py first")
    rows = list(csv.DictReader(ISO_CSV.open()))
    if not rows:
        raise SystemExit(f"no rows in {ISO_CSV}")
    for r in rows:
        r["C_target"] = float(r["C_target"])
        r["n_embed"] = int(r["n_embed"])
        r["n_layer"] = int(r["n_layer"])
        r["n_head"] = int(r["n_head"])
        r["batch_size"] = int(r["batch_size"])
        r["max_iters"] = int(r["max_iters"])
        r["N"] = int(r["N"])
        r["D"] = int(r["D"])
        r["C"] = int(r["C"])
        r["final_val_loss"] = float(r["final_val_loss"])
    return rows


def pick_stars(rows: list[dict]) -> list[dict]:
    """For each budget_id, the row with lowest final_val_loss → (N*, D*, C*)."""
    by_budget = defaultdict(list)
    for r in rows:
        by_budget[r["budget_id"]].append(r)
    stars = []
    for bid in sorted(by_budget):
        best = min(by_budget[bid], key=lambda x: x["final_val_loss"])
        stars.append(best)
    return stars


def main() -> None:
    rows = load_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Print clean table (Steps vs Tokens D) for the PDF later -----
    print("IsoFLOP runs (D = training tokens, max_iters = steps):")
    print(
        f"{'bud':3s} {'N':>10s} {'steps':>7s} {'D_tokens':>10s} "
        f"{'C':>12s} {'loss':>8s} {'src':>10s}"
    )
    for r in sorted(rows, key=lambda x: (x["budget_id"], x["N"])):
        print(
            f"{r['budget_id']:3s} {r['N']:10d} {r['max_iters']:7d} {r['D']:10d} "
            f"{r['C']:12.3e} {r['final_val_loss']:8.4f} {r['source']:>10s}"
        )

    # ----- Figure A: loss vs N within each IsoFLOP budget -----
    by_budget = defaultdict(list)
    for r in rows:
        by_budget[r["budget_id"]].append(r)

    fig, ax = plt.subplots(figsize=(6, 4))
    for bid in sorted(by_budget):
        pts = sorted(by_budget[bid], key=lambda x: x["N"])
        Ns = [p["N"] for p in pts]
        losses = [p["final_val_loss"] for p in pts]
        ax.plot(Ns, losses, "o-", label=f"budget {bid}")
        # Mark compute-optimal N* for this fixed C
        best = min(pts, key=lambda x: x["final_val_loss"])
        ax.scatter([best["N"]], [best["final_val_loss"]], marker="*", s=160, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("N (parameters)")
    ax.set_ylabel("final val loss")
    ax.set_title("IsoFLOP: loss vs N at fixed C (star = N*)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "isoflop_loss_vs_N.pdf")
    fig.savefig(FIG_DIR / "isoflop_loss_vs_N.png", dpi=150)
    print(f"wrote {FIG_DIR / 'isoflop_loss_vs_N.pdf'}")

    # ----- Figure B: N*, D* vs C and Chinchilla-style slope fits -----
    stars = pick_stars(rows)
    print("\nCompute-optimal picks per budget:")
    for s in stars:
        print(
            f"  {s['budget_id']}: N*={s['N']}  D*={s['D']}  C*={s['C']:.3e}  "
            f"loss={s['final_val_loss']:.4f}"
        )

    C_star = np.array([s["C"] for s in stars], dtype=float)
    N_star = np.array([s["N"] for s in stars], dtype=float)
    D_star = np.array([s["D"] for s in stars], dtype=float)

    # log N* = a log C + const  ⇒  N* ∝ C^a
    # log D* = b log C + const  ⇒  D* ∝ C^b
    # Theory (α≈β): a ≈ b ≈ 0.5
    a, _ = np.polyfit(np.log(C_star), np.log(N_star), 1)
    b, _ = np.polyfit(np.log(C_star), np.log(D_star), 1)
    print(f"\nFitted exponents:  a={a:.3f}  (N*∝C^a),  b={b:.3f}  (D*∝C^b)")
    print("Chinchilla equal α,β prediction: a ≈ b ≈ 0.5")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    ax.scatter(C_star, N_star, s=80)
    # Guide line with fitted slope a through geometric mean
    C_line = np.array([C_star.min(), C_star.max()])
    N_line = np.exp(np.log(N_star.mean()) + a * (np.log(C_line) - np.log(C_star.mean())))
    ax.plot(C_line, N_line, "k--", alpha=0.5, label=f"fit a={a:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C* ≈ 6 N* D*")
    ax.set_ylabel("N*")
    ax.set_title("N* vs C")
    ax.legend()

    ax = axes[1]
    ax.scatter(C_star, D_star, s=80)
    D_line = np.exp(np.log(D_star.mean()) + b * (np.log(C_line) - np.log(C_star.mean())))
    ax.plot(C_line, D_line, "k--", alpha=0.5, label=f"fit b={b:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C* ≈ 6 N* D*")
    ax.set_ylabel("D* (tokens)")
    ax.set_title("D* vs C")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Nstar_Dstar_vs_C.pdf")
    fig.savefig(FIG_DIR / "Nstar_Dstar_vs_C.png", dpi=150)
    print(f"wrote {FIG_DIR / 'Nstar_Dstar_vs_C.pdf'}")


if __name__ == "__main__":
    main()
