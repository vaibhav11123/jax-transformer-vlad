# JAX Transformer for Digit Addition

Vlad Feinberg Exercise 1 — built from scratch using JAX, Flax NNX, and Optax.

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
```
config.py           # all hyperparameters, no hardcoded constants
data.py             # vocab, dataset generator, loss mask
model.py            # TokenEmbedding, PositionalEmbedding,
                    # SelfAttentionHead, MultiHeadAttention,
                    # FeedForward, Block, DigitTransformer
train.py            # training loop, Optax AdamW, W&B logging, generate()
tests/
  test_shapes.py    # shape assertions for every component
requirements.txt
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

## Dependencies
Core: jax, flax, optax, wandb, numpy.
Full list in requirements.txt.
