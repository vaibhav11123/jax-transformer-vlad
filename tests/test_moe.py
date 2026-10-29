import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

# TPU default matmul precision is not float32-exact vs dense einsum.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np

from moe_pallas.ragged_baseline import moe_ragged
from moe_pallas.reference import moe_reference
from moe_pallas.routing import make_routing, restore_token_order, sort_tokens


def _tiny_inputs(seed):
    T, E, D, H = 32, 4, 8, 16
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (T, D), dtype=jnp.float32)
    w_up = jax.random.normal(keys[1], (E, D, H), dtype=jnp.float32)
    w_down = jax.random.normal(keys[2], (E, H, D), dtype=jnp.float32)
    gate = jax.random.uniform(keys[3], (T,), dtype=jnp.float32)
    return T, E, x, w_up, w_down, gate


def test_balanced_invariants():
    T, E = 32, 4
    expert_ids, order, sizes = make_routing(T, E, seed=0, distribution="balanced")
    assert int(sizes.sum()) == T
    assert sizes.shape == (E,)
    sorted_ids = np.asarray(expert_ids)[np.asarray(order)]
    assert np.all(sorted_ids[1:] >= sorted_ids[:-1])


def test_sort_restore_roundtrip_skewed():
    expert_ids, order, sizes = make_routing(
        64, 8, seed=7, distribution="skewed"
    )
    assert int(sizes.sum()) == 64
    assert sizes.shape == (8,)
    x = jnp.arange(64 * 3).reshape(64, 3).astype(jnp.float32)
    gate = jnp.ones((64,), dtype=jnp.float32)
    xs, _ = sort_tokens(x, gate, order)
    back = restore_token_order(xs, order)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(x))


def test_ragged_matches_reference_balanced():
    T, E, x, w_up, w_down, gate = _tiny_inputs(0)
    expert_ids, order, sizes = make_routing(T, E, seed=1, distribution="balanced")
    got = moe_ragged(x, w_up, w_down, gate, order, sizes)
    exp = moe_reference(x, w_up, w_down, gate, expert_ids)
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-4, atol=1e-4)


def test_ragged_matches_reference_skewed():
    T, E, x, w_up, w_down, gate = _tiny_inputs(5)
    expert_ids, order, sizes = make_routing(T, E, seed=3, distribution="skewed")
    got = moe_ragged(x, w_up, w_down, gate, order, sizes)
    exp = moe_reference(x, w_up, w_down, gate, expert_ids)
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-4, atol=1e-4)
