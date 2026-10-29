"""Smoke-test ragged_dot, check vs reference, time baseline, append CSV."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from moe_pallas.ragged_baseline import moe_ragged, ragged_projections
from moe_pallas.reference import moe_reference
from moe_pallas.routing import make_routing, sort_tokens

RESULTS = Path(__file__).resolve().parent / "results" / "baseline.csv"

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
    for _ in range(warmup):
        fn().block_until_ready()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn().block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.std(samples))


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
    else:
        x_s, g_s = sort_tokens(x, gate, order)
        run = jax.jit(
            lambda: ragged_projections(x_s, w_up, w_down, g_s, sizes)
        )

    run().block_until_ready()  # compile + warm up; excluded from median
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


def main():
    import datetime as _dt
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = _device_meta()
    print(
        f"run_id={run_id} backend={meta['backend']} "
        f"device={meta['device_kind']} jax={meta['jax_version']}"
    )
    smoke_ragged_dot()
    print("smoke_ragged_dot: ok")
    check_correctness("balanced")
    check_correctness("skewed")
    print("correctness: ok")

    # bf16 ragged_dot Mosaic needs default precision (not "highest" from checks).
    jax.config.update("jax_default_matmul_precision", "default")
    jax.clear_caches()

    for dist in ("balanced", "skewed"):
        for scope in ("full", "proj"):
            row = time_case(HEADLINE, dist, scope=scope)
            row["run_id"] = run_id
            append_row(row)
            print(
                f"{dist}/{scope}: median={row['median_ms']:.3f} ms "
                f"std={row['std_ms']:.3f} E>D={row['E_gt_D']}"
            )


if __name__ == "__main__":
    main()
