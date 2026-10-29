# Exercise 3 — MoE `ragged_dot` baseline (October)

Unfused top-1 MoE MLP via two `jax.lax.ragged_dot` projections + ReLU. Fused Pallas comes in November.

Vlad’s writeup uses **F** for expert count; in this repo **E ≡ F**. Code and CSV use only `E` / `expert_count`. The regime of interest is **E > D** (more experts than model channels), e.g. `E=256, D=128`.

## Shapes

| Role | T | E | D | H | dtype |
|------|---|---|---|---|-------|
| Correctness / unit tests | 32 | 4 | 8 | 16 | float32 |
| Headline profile | 8192 | 256 | 128 | 512 | bfloat16 |

Headline satisfies `E > D`. First locked TPU baseline: **`run_id=20261029T091150Z`** (Colab `--tpu v5e1`, device `TPU v5 lite`, jax `0.11.2`). Later `run_id`s are reproducibility / `sort`-scope follow-ups; the CSV is append-only.

## Math (idealized estimates)

Top-1 MoE MLP: up `D→H`, ReLU, down `H→D`. Balanced load: \(T_e \approx T/E\).

**FLOPs** (2 per MAC): \(4TDH\).

**Unfused intermediate** (write+read of \(T\times H\)): \(\Delta B = 2bTH\) under the usual assumptions — fusion can remove that traffic.

Idealized HBM traffic / arithmetic intensity (weights read **once** — optimistic; real `ragged_dot` may reload tiles):

\[
B_{\mathrm{unfused}} \approx b(2TD + 2TH + 2EDH),\quad
AI_{\mathrm{unfused}} \approx \frac{2/b}{1/H+1/D+E/T}
\]

\[
B_{\mathrm{fused}} \approx b(2TD + 2EDH),\quad
AI_{\mathrm{fused}} \approx \frac{2/b}{1/H+E/T}
\]

These are **estimates**, not measured HBM. \(E>D\) can lower estimated AI via smaller per-expert GEMMs and a larger \(E/T\) term — a **hypothesis to test**, not a proof that the kernel is memory-bound.

**SRAM / tiling caveat:** for the headline, \(2EDH = 33{,}554{,}432\) elements ≈ **64 MiB** at bf16 (all experts’ up+down). A practical fused kernel must **tile** a working set into VMEM/SRAM; November owns that design.

### Headline AI (bf16, \(b=2\)) — reproducible

With \(T=8192,\,E=256,\,D=128,\,H=512\):

\[
AI_{\mathrm{unfused}}
= \frac{1}{1/512 + 1/128 + 256/8192}
= \frac{1}{1/512 + 1/128 + 1/32}
= \frac{1}{0.041015625}
\approx 24.4 \text{ FLOPs/byte}
\]

\[
AI_{\mathrm{fused}}
= \frac{1}{1/512 + 1/32}
= \frac{1}{0.033203125}
\approx 30.1 \text{ FLOPs/byte}
\]

FLOPs \(= 4TDH = 2.147\times 10^9\).

## Layout

```text
moe_pallas/
  routing.py          # assign / sort / group_sizes / restore
  reference.py        # dense einsum oracle (tiny cases only)
  ragged_baseline.py  # two ragged_dot + ReLU + gate
  benchmark.py        # smoke, check, time, append CSV
  results/baseline.csv
```

From repo root:

```bash
python -m pytest tests/test_moe.py -q
python -m moe_pallas.benchmark
python -m moe_pallas.benchmark --sessions 3
python -m moe_pallas.benchmark --profile   # best-effort; may skip on some images
```

Timing scopes in CSV:

- **`proj`** — jitted projections + gate (tokens already sorted)
- **`full`** — jitted end-to-end (device sort/restore + projections)
- **`sort`** — jitted sort + restore only (no GEMMs); for balanced vs skewed rearrangement cost

`make_routing` / expert-id generation is **outside** the timed region. Each timing grid stamps a UTC `run_id`. `--sessions N` repeats the grid with a fresh `run_id` each time (within-kernel variance). Fresh Colab VMs separate across-session variance.

Float32 oracle checks set matmul precision to `highest`, then restore `default` before bf16 timing — Mosaic `ragged_dot` fails to compile bf16 under `highest`.

## Colab CLI (no git clone) — verified 2026-10-29

Same pattern as Exercise 1: session → **mkdir before upload** → upload → exec → download. Flags: `colab new --help` / `colab exec -h` / `colab download --help`.

This Colab CLI build has **no** `colab exec -c`; pass a **local** script with `-f`.

```bash
# 1) TPU session
colab new -s moe-tpu --tpu v5e1

# 2) Parent dirs FIRST
cat > /tmp/_moe_mkdir.py <<'PY'
import os
os.makedirs("/content/moe_pallas/results", exist_ok=True)
os.makedirs("/content/moe_pallas/figures", exist_ok=True)
os.makedirs("/content/tests", exist_ok=True)
print("mkdirs ok")
PY
colab exec -s moe-tpu -f /tmp/_moe_mkdir.py --timeout 60

# 3) Upload package + tests + existing CSV (append-only; keep first run_id)
colab upload -s moe-tpu moe_pallas/__init__.py /content/moe_pallas/__init__.py
colab upload -s moe-tpu moe_pallas/routing.py /content/moe_pallas/routing.py
colab upload -s moe-tpu moe_pallas/reference.py /content/moe_pallas/reference.py
colab upload -s moe-tpu moe_pallas/ragged_baseline.py /content/moe_pallas/ragged_baseline.py
colab upload -s moe-tpu moe_pallas/benchmark.py /content/moe_pallas/benchmark.py
colab upload -s moe-tpu moe_pallas/results/baseline.csv /content/moe_pallas/results/baseline.csv
colab upload -s moe-tpu tests/test_moe.py /content/tests/test_moe.py

# 4) Deps
colab install -s moe-tpu pytest

# 5) Benchmark (add --sessions 3 and/or --profile as needed)
cat > /tmp/_moe_bench.py <<'PY'
import os, sys
os.chdir("/content")
sys.argv = ["moe_pallas.benchmark", "--sessions", "1", "--profile"]
import runpy
runpy.run_module("moe_pallas.benchmark", run_name="__main__")
PY
colab exec -s moe-tpu -f /tmp/_moe_bench.py --timeout 900

# 6) Fetch CSV
colab download -s moe-tpu /content/moe_pallas/results/baseline.csv moe_pallas/results/baseline.csv
```

For a second **fresh** VM, `colab new -s moe-tpu-b --tpu v5e1` and repeat upload/exec/download (merge by keeping append-only CSV).

## First headline results (`run_id=20261029T091150Z`)

TPU v5e · bf16 · `T=8192, E=256, D=128, H=512` (scopes `full` / `proj` only in this first run):

| routing | scope | median_ms | std_ms |
|---------|-------|-----------|--------|
| balanced | full | 0.416 | 0.010 |
| balanced | proj | 0.418 | 0.018 |
| skewed | full | 0.454 | 0.015 |
| skewed | proj | 0.421 | 0.010 |

### What these measurements show

Fastest median: **0.416 ms** (balanced / full). Slowest: **0.454 ms** (skewed / full).

Routing distribution:

- Full: skewed is **+0.038 ms** (~**9.1%**) vs balanced.
- Proj: skewed is **+0.003 ms** (~**0.7%**) vs balanced.

So the distribution shows up more in `full` than in `proj` on this run — an **observation**, not yet a proven cause.

Within-run std is small vs the medians; that does not measure variance across separate sessions, compilations, or device conditions (see later `run_id`s).

### Why `full` and `proj` look almost identical

| Routing  | Full     | Proj     | Difference |
| -------- | -------- | -------- | ---------- |
| Balanced | 0.416 ms | 0.418 ms | −0.002 ms  |
| Skewed   | 0.454 ms | 0.421 ms | +0.033 ms  |

Balanced: `full` is **0.002 ms faster** than `proj`. Do **not** read that as “sorting speeds up the compute” — the delta is tiny, and separately compiled paths can differ. It is not a reliable sort-overhead estimate.

Skewed: `full` is **+0.033 ms** (~7.8% of proj). That gap is worth investigating with the `sort` scope and multi-session repeats.

### Effective throughput (~5.14 TFLOP/s)

\[
4TDH = 4(8192)(128)(512) = 2.147\times 10^9 \text{ FLOPs}
\]

Balanced `proj` median 0.418 ms:

\[
P = \frac{2.147\times 10^9}{0.418\times 10^{-3}} \approx 5.14\times 10^{12}
\quad\Rightarrow\quad 5.14\ \mathrm{TFLOP/s}
\]

Skewed `proj` at 0.421 ms ≈ **5.10 TFLOP/s**. This is an effective estimate for the two matmuls, not measured TPU utilization, and not by itself a compute- vs memory-bound verdict.

## Follow-up runs (sort scope + multi-session)

Append-only CSV after the first headline. Within-kernel repeats: `20261029T092502Z`, `…507Z`, `…512Z` (`moe-tpu-a`, `--sessions 3 --profile`). Fresh VM: `20261029T092820Z` (`moe-tpu-b`). Profile trace wrote to `results/profile_20261029T092501Z` on session A.

Representative medians (ms):

| run_id | bal full | bal proj | bal sort | skew full | skew proj | skew sort |
|--------|----------|----------|----------|-----------|-----------|-----------|
| 20261029T091150Z (first) | 0.416 | 0.418 | — | 0.454 | 0.421 | — |
| 20261029T092502Z | 0.432 | 0.420 | 0.228 | 0.456 | 0.430 | 0.194 |
| 20261029T092507Z | 0.420 | 0.418 | 0.193 | 0.460 | 0.441 | 0.201 |
| 20261029T092512Z | 0.440 | 0.389 | 0.219 | 0.451 | 0.407 | 0.244 |
| 20261029T092820Z (fresh VM) | 0.406 | 0.388 | 0.202 | 0.462 | 0.424 | 0.166 |

Observations (not proven mechanisms):

- **Skewed `full` slower than balanced `full` reproduces** across within-kernel and fresh-VM runs (~0.45–0.46 vs ~0.41–0.44 ms).
- **`sort` alone is ~0.17–0.23 ms** — large vs the ~0.03 ms skewed `full`−`proj` gap on the first run. Do **not** add `sort`+`proj` to explain `full`: separately compiled paths are not additive.
- **Skewed `sort` is often similar to or faster than balanced `sort`**, so the skewed-`full` penalty is **not** explained by “skewed routing makes rearrange slower” in these measurements.
- Within-session and across-session medians for `proj` sit roughly in **0.39–0.44 ms**; the first-run ~0.42 ms headline remains a fair center.

## Bottleneck assessment (cautious)

Evidence in hand:

1. Idealized \(AI_{\mathrm{unfused}} \approx 24.4\) FLOPs/byte (weights-read-once model).
2. Measured latency ≈ **0.39–0.44 ms** on the projection-only path across runs (~**5.1 TFLOP/s** effective at 0.418 ms).
3. A `jax.profiler.trace` directory exists for one balanced `proj` call (`profile_20261029T092501Z`) — raw traces only; not yet a ridge / stall analysis.

**Not yet enough** to claim the baseline is memory-bound. A defensible claim needs comparing that AI to TPU v5e peak BF16 compute vs HBM bandwidth (ridge point), and reading the profiler for memory stalls / compute util / traffic. **`E > D` alone does not prove memory-bound.**

November fused+tiled Pallas remains a **design target** (remove unfused \(\Delta B = 2bTH\), improve weight reuse under SRAM limits) — not a proven speedup from this CSV alone. The reproducible routing effect on `full` is a separate open question for kernel design.

`--profile` writes under `moe_pallas/results/profile_*/` when `jax.profiler.trace` works; if the Colab image rejects it, the benchmark prints `profile skipped` and continues.
