from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx

from config import Config
import data
from model import DigitTransformer


def compute_loss(logits, targets, mask):
    B, T, V = logits.shape
    per_token = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V), targets.reshape(B * T)
    ).reshape(B, T)
    mask = mask.astype(per_token.dtype)
    valid_count = mask.sum() * B
    return (per_token * mask).sum() / valid_count


def generate(model: nnx.Module, a: int, b: int, *, n_digits: int) -> int:
    """Greedy decode the sum for a+b; returns predicted integer."""
    c = a + b
    full = f"{a:>{n_digits}}+{b:>{n_digits}}={c:>{n_digits+1}}"
    prompt_len = 2 * n_digits + 2  # through '='
    answer_len = n_digits + 1
    idx = jnp.array(data.encode(full[:prompt_len]), dtype=jnp.int32)[None, :]
    for _ in range(answer_len):
        logits = model(idx, training=False)
        next_id = jnp.argmax(logits[0, -1])
        idx = jnp.concatenate([idx, next_id[None, None]], axis=-1)
    answer = data.decode(idx[0, prompt_len:].tolist())
    return int(answer.replace(" ", ""))


def main(cfg: Config):
    rngs = nnx.Rngs(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    model = DigitTransformer(cfg, rngs=rngs)

    # Chinchilla accounting (Hoffmann et al. 2022): N params, D tokens, C≈6ND
    N = int(sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))))
    D = int(cfg.max_iters * cfg.batch_size * cfg.block_size)
    C = int(6 * N * D)
    print(f"{N / 1e6:.4f}M parameters")
    print(f"N={N}  D={D}  C≈6ND={C}")

    optimizer = nnx.Optimizer(model, optax.adamw(cfg.learning_rate), wrt=nnx.Param)
    mask = jnp.array(data.loss_mask(cfg.n_digits, cfg.block_size))

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cast(Literal["online", "disabled", "shared", "offline"], cfg.wandb_mode),
        config=asdict(cfg),
    )

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            logits = model(x, training=True)
            return compute_loss(logits, y, mask)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, x, y):
        logits = model(x, training=False)
        return compute_loss(logits, y, mask)

    def estimate_val_loss():
        losses = jnp.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = data.get_batch(np_rng, cfg.n_digits, cfg.batch_size)
            losses = losses.at[k].set(eval_step(model, jnp.array(x), jnp.array(y)))
        return losses.mean()

    def print_sample_generations():
        print("\nSample generations:")
        rng_gen = np.random.default_rng(0)
        max_val = 10**cfg.n_digits
        for _ in range(5):
            a = int(rng_gen.integers(0, max_val))
            b = int(rng_gen.integers(0, max_val))
            try:
                pred = generate(model, a, b, n_digits=cfg.n_digits)
                correct = "✓" if pred == a + b else "✗"
                print(f"  {a}+{b}={pred} {correct} (true {a + b})")
            except ValueError:
                print(f"  {a}+{b}=? ✗ (true {a + b})  [non-numeric decode]")
        print()

    final_val_loss = None
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            val_loss = estimate_val_loss()
            final_val_loss = float(val_loss)
            print(f"step {it}: val_loss={val_loss:.4f}")
            wandb.log({"val_loss": final_val_loss}, step=it)
            print_sample_generations()

        x, y = data.get_batch(np_rng, cfg.n_digits, cfg.batch_size)
        loss = train_step(model, optimizer, jnp.array(x), jnp.array(y))
        wandb.log({"train_loss": float(loss)}, step=it)

    wandb.summary["N"] = N
    wandb.summary["D"] = D
    wandb.summary["C"] = C
    wandb.summary["final_val_loss"] = final_val_loss
    wandb.finish()

    # Metrics dict for chinchilla/ sweeps (CSV column n_embed maps to cfg.n_embd)
    return {
        "n_embed": cfg.n_embd,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "batch_size": cfg.batch_size,
        "max_iters": cfg.max_iters,
        "N": N,
        "D": D,
        "C": C,
        "final_val_loss": final_val_loss,
        "wandb_run_name": cfg.wandb_run_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_mode", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.wandb_mode is not None:
        cfg.wandb_mode = cast(
            Literal["online", "offline", "disabled", "shared"], args.wandb_mode
        )
    main(cfg)
