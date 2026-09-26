# JAX Transformer for Digit Addition

Vlad Feinberg exercises — built from scratch using JAX, Flax NNX, and Optax.

- **Exercise 1:** digit-addition Transformer (repo root)
- **Exercise 2:** Chinchilla scaling empirics in `chinchilla/`  
  Walkthrough: https://youtu.be/ji2_hHj-UI4  
  Report PDF: https://drive.google.com/file/d/1I59jP0TeVZS3I9O9pTMRt_lG1YfIEnZ7/view?usp=sharing

## What it does
Trains a ~10M parameter transformer to perform up to 3-digit addition.
Input format: `345+678=1023` (fixed-length, space-padded).
Vocab: digits 0–9, space, +, = (13 tokens).

## Model
- Architecture: Causal transformer (decoder-only)
- Parameters: ~10.6M
- Config: n_embd=384, n_layer=6, n_head=6, head_size=64
- Sequence length: 12 (block_size=11)
- Loss: cross-entropy on answer digits only (positions after =)

## Results
- Val loss: **0.0006**
- AR exact match: **100.00%** (512 held-out sums, greedy decode)

Sample outputs after training:
```
345+678=1023  (true 345+678=1023) ✓
  7+ 42=  49  (true   7+ 42=  49) ✓
999+999=1998  (true 999+999=1998) ✓
100+200= 300  (true 100+200= 300) ✓
503+497=1000  (true 503+497=1000) ✓
 12+  3=  15  (true  12+  3=  15) ✓
```

## Recordings
All files written from scratch, screen-recorded.

Playlist: https://www.youtube.com/playlist?list=PLQHXISqfzAx8

Individual videos:
- config.py: https://youtu.be/LfWly21u04c
- data.py: https://youtu.be/EVKmKqjJ_oU
- model.py part 1: https://youtu.be/a2JgQ5qNmWE
- model.py part 2: https://youtu.be/0nbTllcPwUQ
- train.py: https://youtu.be/8WT4H3YY_Ws
- Scaling + Training: https://youtu.be/HTu7leIrPyE

## Repo structure

### Exercise 1 (core model) — repo root
```
config.py           # all hyperparameters, no hardcoded constants
data.py             # vocab, dataset generator, loss mask
model.py            # TokenEmbedding, PositionalEmbedding,
                    # SelfAttentionHead, MultiHeadAttention,
                    # FeedForward, Block, DigitTransformer
train.py            # training loop, Optax AdamW, W&B logging, generate()
                    # also logs N, D, C≈6ND for Chinchilla sweeps
tests/
  test_shapes.py    # shape assertions for every component
requirements.txt
```

### Exercise 2 — folder `chinchilla/`
Walkthrough: https://youtu.be/ji2_hHj-UI4  
Report PDF: https://drive.google.com/file/d/1I59jP0TeVZS3I9O9pTMRt_lG1YfIEnZ7/view?usp=sharing

```
chinchilla/
  sweep.py, analyze_scaling.py          # rectangular N×D sanity check
  isoflop_sweep.py, analyze_isoflop.py  # fixed-C IsoFLOP allocation
  results/                              # sweep_results.csv, isoflop_results.csv
  figures/                              # scaling + IsoFLOP plots
```

## Reproduce
```bash
git clone https://github.com/vaibhav11123/jax-transformer-vlad
cd jax-transformer-vlad
pip install -r requirements.txt
python train.py --wandb_mode disabled
```

On Colab T4/TPU:
```bash
# Runtime → Change runtime type → T4 GPU (or TPU)
git clone https://github.com/vaibhav11123/jax-transformer-vlad
cd jax-transformer-vlad
pip install -r requirements.txt
python train.py --wandb_mode disabled
```

## Exercise 2 — Chinchilla scaling

Walkthrough / screen recording: https://youtu.be/ji2_hHj-UI4  
Report PDF: https://drive.google.com/file/d/1I59jP0TeVZS3I9O9pTMRt_lG1YfIEnZ7/view?usp=sharing

Exercise 1 trains one model. Exercise 2 reuses that trainer to study scaling:

1. **Sanity check (rectangular N×D):** vary model size and token budget independently.  
   Shows that more N or more D lowers val loss (but compute C also changes).
2. **IsoFLOP (fixed C):** hold `C ≈ 6ND` roughly constant and find `(N*, D*)` per budget,  
   then fit `N* ∝ C^a`, `D* ∝ C^b` (theory for α≈β: `a ≈ b ≈ 0.5`).

### Reproduce Experiment 1 (N×D sweep)
```bash
cd chinchilla
python3 sweep.py
python3 analyze_scaling.py
# → results/sweep_results.csv, figures/scaling_plot.pdf
```

### Reproduce Experiment 2 (IsoFLOP)
```bash
cd chinchilla
python3 isoflop_sweep.py
python3 analyze_isoflop.py
# → results/isoflop_results.csv
# → figures/isoflop_loss_vs_N.pdf, figures/Nstar_Dstar_vs_C.pdf
```

### W&B projects
- Exercise 1: `jax-transformer-vlad` (default in `config.py`)
- Exercise 2: `vlad_exercise_2` (set by chinchilla sweep scripts)

## Dependencies
Core: jax, flax, optax, wandb, numpy, matplotlib.
Full list in requirements.txt.
