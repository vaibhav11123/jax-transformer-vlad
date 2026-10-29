"""Top-1 expert assignment, sort order, and group sizes for ragged_dot."""
import numpy as np
import jax.numpy as jnp


def make_routing(token_count, expert_count, seed=0, distribution="balanced"):
    if token_count <= 0 or expert_count <= 0:
        raise ValueError("token_count and expert_count must be positive")

    rng = np.random.default_rng(seed)

    if distribution == "balanced":
        # Benchmark convenience: equal tokens per expert.
        if token_count % expert_count:
            raise ValueError("balanced routing requires T divisible by E")
        expert_ids = np.tile(
            np.arange(expert_count), token_count // expert_count
        )
        rng.shuffle(expert_ids)
    elif distribution == "skewed":
        if expert_count < 2:
            raise ValueError("skewed routing requires E >= 2")
        probabilities = np.full(expert_count, 0.5 / (expert_count - 1))
        probabilities[0] = 0.5
        expert_ids = rng.choice(
            expert_count, size=token_count, p=probabilities
        )
    else:
        raise ValueError("distribution must be 'balanced' or 'skewed'")

    # Stable sort so equal expert ids keep relative order.
    sorted_order = np.argsort(expert_ids, kind="stable")
    group_sizes = np.bincount(
        expert_ids, minlength=expert_count
    ).astype(np.int32)
    assert int(group_sizes.sum()) == token_count

    return (
        jnp.asarray(expert_ids),
        jnp.asarray(sorted_order),
        jnp.asarray(group_sizes),
    )


def sort_tokens(x, gate, sorted_order):
    return x[sorted_order], gate[sorted_order]


def restore_token_order(y_sorted, sorted_order):
    return jnp.zeros_like(y_sorted).at[sorted_order].set(y_sorted)
