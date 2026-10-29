"""Dense per-token MoE oracle for tiny correctness cases (not for timing)."""
import jax
import jax.numpy as jnp


def moe_reference(x, w_up, w_down, gate, expert_ids):
    # x: [T, D], w_up: [E, D, H], w_down: [E, H, D], gate: [T], expert_ids: [T]
    hidden = jax.nn.relu(jnp.einsum("td,tdh->th", x, w_up[expert_ids]))
    out = jnp.einsum("th,thd->td", hidden, w_down[expert_ids])
    return out * gate[:, None]
