"""Smoke-test ragged_dot, check vs reference, time baseline, append CSV."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from moe_pallas.ragged_baseline import moe_ragged, ragged_projections
from moe_pallas.reference import moe_reference
from moe_pallas.routing import make_routing, restore_token_order, sort_tokens

RESULTS = Path(__file__).resolve().parent / "results" / "baseline.csv"
PROFILE_ROOT = Path(__file__).resolve().parent / "results"

# Tiny correctness (CPU/TPU). Headline profile: E > D (assignment F > D).
CORRECT = dict(T=32, E=4, D=8, H=16, dtype=jnp.float32)
HEADLINE = dict(T=8192, E=256, D=128, H=512, dtype=jnp.bfloat16)
# CSV: Ex2-style names; include backend; run_id separates repeated appends.
FIELDS = [
    "run_id",
    "impl",
    "token_count",
    "expert_count",
    "model_dim",
    "hidden_dim",
    "E_gt_D",
    "dtype",
    "routing",
    "activation",
    "backend",
    "device_kind",
    "jax_version",
    "median_ms",
    "std_ms",
    "scope",
]
SCOPES = ("full", "proj", "sort")


def _device_meta():
    d0 = jax.devices()[0]
    return {
        "device_kind": getattr(d0, "device_kind", str(d0)),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }


def smoke_ragged_dot(dtype=jnp.bfloat16):
    lhs = jnp.ones((6, 4), dtype=dtype)
    rhs = jnp.ones((2, 4, 3), dtype=dtype)
    sizes = jnp.array([2, 4], dtype=jnp.int32)
    out = jax.jit(jax.lax.ragged_dot)(lhs, rhs, sizes)
    out.block_until_ready()
    np.testing.assert_allclose(np.asarray(out), 4.0, rtol=0, atol=0)


def make_case(T, E, D, H, dtype, seed=0):
    keys = jax.random.split(jax.random.key(seed), 4)
    scale_up = jnp.asarray(D**-0.5, dtype=dtype)
    scale_down = jnp.asarray(H**-0.5, dtype=dtype)
    x = jax.random.normal(keys[0], (T, D), dtype=dtype)
    w_up = jax.random.normal(keys[1], (E, D, H), dtype=dtype) * scale_up
    w_down = jax.random.normal(keys[2], (E, H, D), dtype=dtype) * scale_down
    gate = jax.random.uniform(keys[3], (T,), minval=0.5, maxval=1.0, dtype=dtype)
    return x, w_up, w_down, gate


def check_correctness(distribution="balanced"):
    # TPU default matmul precision is not float32-exact vs dense einsum.
    # Restore afterward: "highest" breaks bf16 ragged_dot Mosaic on TPU.
    prev = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    try:
        cfg = CORRECT
        x, w_up, w_down, gate = make_case(**cfg, seed=1)
        expert_ids, order, sizes = make_routing(
            cfg["T"], cfg["E"], seed=2, distribution=distribution
        )
        got = moe_ragged(x, w_up, w_down, gate, order, sizes)
        exp = moe_reference(x, w_up, w_down, gate, expert_ids)
        got.block_until_ready()
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(exp), rtol=1e-4, atol=1e-4
        )
    finally:
        jax.config.update("jax_default_matmul_precision", prev)
        jax.clear_caches()


def _median_ms(fn, warmup=10, repeats=50):
    # jax.block_until_ready handles pytrees (sort scope returns a tuple).
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.std(samples))


def _sort_restore(x, gate, order):
    # Rearrangement only — no GEMMs. Restores x and gate to original order.
    x_s, g_s = sort_tokens(x, gate, order)
    x_back = restore_token_order(x_s, order)
    g_back = restore_token_order(g_s[:, None], order)[:, 0]
    return x_back, g_back


def time_case(cfg, distribution, scope="full"):
    x, w_up, w_down, gate = make_case(**cfg, seed=3)
    _, order, sizes = make_routing(
        cfg["T"], cfg["E"], seed=4, distribution=distribution
    )
    # JIT the exact timed path (inputs closed over). Warm up before median.
    if scope == "full":
        run = jax.jit(
            lambda: moe_ragged(x, w_up, w_down, gate, order, sizes)
        )
    elif scope == "proj":
        x_s, g_s = sort_tokens(x, gate, order)
        run = jax.jit(
            lambda: ragged_projections(x_s, w_up, w_down, g_s, sizes)
        )
    elif scope == "sort":
        run = jax.jit(lambda: _sort_restore(x, gate, order))
    else:
        raise ValueError(f"unknown scope: {scope}")

    jax.block_until_ready(run())  # compile + warm up; excluded from median
    med, std = _median_ms(run)
    meta = _device_meta()
    return {
        "impl": "ragged_dot",
        "token_count": cfg["T"],
        "expert_count": cfg["E"],
        "model_dim": cfg["D"],
        "hidden_dim": cfg["H"],
        "E_gt_D": int(cfg["E"] > cfg["D"]),
        "dtype": jnp.dtype(cfg["dtype"]).name,
        "routing": distribution,
        "activation": "relu",
        "backend": meta["backend"],
        "device_kind": meta["device_kind"],
        "jax_version": meta["jax_version"],
        "median_ms": med,
        "std_ms": std,
        "scope": scope,
    }


def append_row(row):
    # Append-only CSV. run_id is set by main() on each row (not enforced here).
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def _profile_proj(cfg=HEADLINE, distribution="balanced"):
    # Best-effort trace for balanced proj; skip cleanly if tooling fails.
    x, w_up, w_down, gate = make_case(**cfg, seed=3)
    _, order, sizes = make_routing(
        cfg["T"], cfg["E"], seed=4, distribution=distribution
    )
    x_s, g_s = sort_tokens(x, gate, order)
    run = jax.jit(
        lambda: ragged_projections(x_s, w_up, w_down, g_s, sizes)
    )
    jax.block_until_ready(run())
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROFILE_ROOT / f"profile_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with jax.profiler.trace(str(out_dir)):
            for _ in range(5):
                jax.block_until_ready(run())
        print(f"profile wrote → {out_dir}")
    except Exception as e:
        print(f"profile skipped: {type(e).__name__}: {e}")


def _run_timing_grid(run_id):
    for dist in ("balanced", "skewed"):
        for scope in SCOPES:
            row = time_case(HEADLINE, dist, scope=scope)
            row["run_id"] = run_id
            append_row(row)
            print(
                f"{dist}/{scope}: median={row['median_ms']:.3f} ms "
                f"std={row['std_ms']:.3f} E>D={row['E_gt_D']}"
            )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="repeat the timing grid with a fresh run_id each time",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="best-effort jax.profiler.trace on balanced proj",
    )
    args = p.parse_args(argv)

    meta = _device_meta()
    print(
        f"backend={meta['backend']} device={meta['device_kind']} "
        f"jax={meta['jax_version']} sessions={args.sessions}"
    )
    smoke_ragged_dot()
    print("smoke_ragged_dot: ok")
    check_correctness("balanced")
    check_correctness("skewed")
    print("correctness: ok")

    # bf16 ragged_dot Mosaic needs default precision (not "highest" from checks).
    jax.config.update("jax_default_matmul_precision", "default")
    jax.clear_caches()

    if args.profile:
        _profile_proj()

    for _ in range(max(1, args.sessions)):
        run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        print(f"run_id={run_id}")
        _run_timing_grid(run_id)


if __name__ == "__main__":
    main()
