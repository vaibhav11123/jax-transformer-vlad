"""Baseline MoE: two jax.lax.ragged_dot projections + ReLU (unfused)."""
import jax
import jax.numpy as jnp

from moe_pallas.routing import restore_token_order, sort_tokens


def ragged_projections(x_sorted, w_up, w_down, gate_sorted, group_sizes):
    # Same group_sizes for up [E,D,H] and down [E,H,D].
    hidden = jax.nn.relu(jax.lax.ragged_dot(x_sorted, w_up, group_sizes))
    y_sorted = jax.lax.ragged_dot(hidden, w_down, group_sizes)
    return y_sorted * gate_sorted[:, None]


def moe_ragged(x, w_up, w_down, gate, sorted_order, group_sizes):
    x_sorted, gate_sorted = sort_tokens(x, gate, sorted_order)
    y_sorted = ragged_projections(
        x_sorted, w_up, w_down, gate_sorted, group_sizes
    )
    return restore_token_order(y_sorted, sorted_order)
