# Exercise 3 — MoE `ragged_dot` baseline (October)

Unfused top-1 MoE MLP via two `jax.lax.ragged_dot` projections + ReLU. Fused Pallas comes in November.

Vlad’s writeup uses **F** for expert count; in this repo **E ≡ F**. Code and CSV use only `E` / `expert_count`. The regime of interest is **E > D** (more experts than model channels), e.g. `E=256, D=128`.

## Shapes

| Role | T | E | D | H | dtype |
|------|---|---|---|---|-------|
| Correctness / unit tests | 32 | 4 | 8 | 16 | float32 |
| Headline profile (locked after TPU) | 8192 | 256 | 128 | 512 | bfloat16 |

Headline satisfies `E > D`. Validated on Colab TPU v5e (`--tpu v5e1`).

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

These are **estimates**, not measured HBM. \(E>D\) can lower estimated AI via smaller per-expert GEMMs and a larger \(E/T\) term — a **hypothesis to test**, not a proof that the kernel is memory-bound. \(E/T\), \(H\), dtype, device, and the actual implementation matter.

**SRAM / tiling caveat:** for the headline, \(2EDH = 33{,}554{,}432\) elements ≈ **64 MiB** at bf16 (all experts’ up+down). A practical fused kernel must **tile** a working set into VMEM/SRAM; November owns that design. \(B_{\mathrm{fused}}\) is an idealized lower bound, not a claim that all weights fit on-chip at once.

### Headline AI (bf16, \(b=2\))

With \(T=8192,\,E=256,\,D=128,\,H=512\):

- \(E/T = 1/32\)
- \(AI_{\mathrm{unfused}} \approx 23.8\) FLOPs/byte
- \(AI_{\mathrm{fused}} \approx 30.1\) FLOPs/byte
- FLOPs \(= 4TDH \approx 2.15\times 10^9\)

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
```

Timing scopes in CSV:

- **`proj`** — jitted projections only (tokens already sorted)
- **`full`** — jitted end-to-end (device sort/restore + projections)

`make_routing` / expert-id generation is **outside** the timed region. Each `main()` run stamps a UTC `run_id` on every appended row. CSV also records `backend`, `device_kind`, `jax_version`.

Float32 oracle checks set matmul precision to `highest` (TPU default is not float32-exact), then restore `default` before bf16 timing — Mosaic `ragged_dot` fails to compile bf16 under `highest`.

## Colab CLI (no git clone) — verified 2026-10-29

Same pattern as Exercise 1: session → **mkdir before upload** → upload → exec → download. Flags checked with `colab new --help` / `colab exec -h` / `colab download --help`.

Note: this Colab CLI build has **no** `colab exec -c`; pass a **local** script with `-f` (the CLI reads the file locally and runs it remotely).

```bash
# 1) TPU session
colab new -s moe-tpu --tpu v5e1

# 2) Parent dirs FIRST (uploads do not mkdir)
cat > /tmp/_moe_mkdir.py <<'PY'
import os
os.makedirs("/content/moe_pallas/results", exist_ok=True)
os.makedirs("/content/moe_pallas/figures", exist_ok=True)
os.makedirs("/content/tests", exist_ok=True)
print("mkdirs ok")
PY
colab exec -s moe-tpu -f /tmp/_moe_mkdir.py --timeout 60

# 3) Upload package + tests (from repo root)
colab upload -s moe-tpu moe_pallas/__init__.py /content/moe_pallas/__init__.py
colab upload -s moe-tpu moe_pallas/routing.py /content/moe_pallas/routing.py
colab upload -s moe-tpu moe_pallas/reference.py /content/moe_pallas/reference.py
colab upload -s moe-tpu moe_pallas/ragged_baseline.py /content/moe_pallas/ragged_baseline.py
colab upload -s moe-tpu moe_pallas/benchmark.py /content/moe_pallas/benchmark.py
colab upload -s moe-tpu tests/test_moe.py /content/tests/test_moe.py

# 4) Deps
colab install -s moe-tpu pytest

# 5) Tests + benchmark
cat > /tmp/_moe_pytest.py <<'PY'
import os
os.chdir("/content")
import jax
jax.config.update("jax_default_matmul_precision", "highest")
jax.clear_caches()
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_moe.py"]))
PY
cat > /tmp/_moe_bench.py <<'PY'
import os
os.chdir("/content")
import runpy
runpy.run_module("moe_pallas.benchmark", run_name="__main__")
PY
colab exec -s moe-tpu -f /tmp/_moe_pytest.py --timeout 180
colab exec -s moe-tpu -f /tmp/_moe_bench.py --timeout 900

# 6) Fetch CSV
colab download -s moe-tpu /content/moe_pallas/results/baseline.csv moe_pallas/results/baseline.csv
```

Optional W&B: CSV is the source of truth for latency. If logging habit from Ex1/Ex2: project `vlad_exercise_3`.

## TPU results (`run_id=20261029T091150Z`, TPU v5 lite, jax 0.11.2)

Headline `T=8192, E=256, D=128, H=512`, bf16:

| routing | scope | median_ms | std_ms |
|---------|-------|-----------|--------|
| balanced | full | 0.416 | 0.010 |
| balanced | proj | 0.418 | 0.018 |
| skewed | full | 0.454 | 0.015 |
| skewed | proj | 0.421 | 0.010 |

Effective ~5.1 TFLOP/s on `proj` (\(4TDH / t\)). `full` ≈ `proj` on balanced (sort/restore is cheap vs the two GEMMs).

## Bottleneck note

On this TPU v5e run, the unfused baseline looks **memory-bound / bandwidth-limited relative to peak compute**: idealized \(AI_{\mathrm{unfused}}\approx 24\) FLOPs/byte sits well below typical TPU ridge points, and achieved ~5 TFLOP/s is a small fraction of bf16 peak. `full` and `proj` medians nearly match, so the gap to close in November is the two unfused `ragged_dot`s (and the \(T\times H\) intermediate), not host-side routing. Treat this as a **measured-supported hypothesis**, not a profiler-grade HBM trace — November fused+tiled kernel should target removing \(\Delta B=2bTH\) and improving weight reuse.
