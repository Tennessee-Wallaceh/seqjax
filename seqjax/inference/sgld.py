from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
from typing import Callable

from seqjax.model.base import ParametersType
from seqjax.model.typing import Batched, SampleAxis
from typing import Generic


class SGLDConfig(eqx.Module, Generic[ParametersType]):
    """Configuration for :func:`run_sgld`."""

    step_size: float | ParametersType = 1e-3
    num_iters: int = 100


def _tree_randn_like(key: PRNGKeyArray, tree: ParametersType) -> ParametersType:  # type: ignore[misc]
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, len(leaves))
    new_leaves = [jrandom.normal(k, shape=jnp.shape(leaf)) for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def run_sgld(
    grad_estimator: Callable[[ParametersType, PRNGKeyArray], ParametersType],
    key: PRNGKeyArray,
    initial_parameters: ParametersType,
    *,
    config: SGLDConfig = SGLDConfig(),
) -> Batched[ParametersType, SampleAxis | int]:
    """Run SGLD updates using ``grad_estimator``."""

    n_iters = config.num_iters
    split_keys = jrandom.split(key, 2 * n_iters)
    grad_keys = split_keys[:n_iters]
    noise_keys = split_keys[n_iters:]

    if jax.tree_util.tree_structure(config.step_size) == jax.tree_util.tree_structure(initial_parameters):  # type: ignore[operator]
        step_sizes = config.step_size
    else:
        step_sizes = jax.tree_util.tree_map(lambda _: config.step_size, initial_parameters)

    def step(params: ParametersType, inp: tuple[PRNGKeyArray, PRNGKeyArray]):
        g_key, n_key = inp
        grad = grad_estimator(params, g_key)
        noise = _tree_randn_like(n_key, params)
        updates = jax.tree_util.tree_map(
            lambda g, n, s: 0.5 * s * g + jnp.sqrt(s) * n,
            grad,
            noise,
            step_sizes,
        )
        params = eqx.apply_updates(params, updates)
        return params, params

    _, samples = jax.lax.scan(step, initial_parameters, (grad_keys, noise_keys))
    return samples
