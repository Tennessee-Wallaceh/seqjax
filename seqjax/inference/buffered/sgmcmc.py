from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
    ParameterPrior,
    HyperParametersType,
)
from seqjax.model.typing import Batched, SequenceAxis
from seqjax.inference.particlefilter import SMCSampler
from .buffered import _run_segment


class BufferedSGLDConfig(eqx.Module):
    """Configuration for :func:`run_buffered_sgld`."""

    step_size: float | ParametersType = 1e-3
    num_iters: int = 100
    buffer_size: int = 0
    batch_size: int = 1
    particle_filter: (
        SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType] | None
    ) = None
    parameter_prior: ParameterPrior[ParametersType, HyperParametersType] | None = None
    hyperparameters: HyperParametersType | None = None


def _tree_randn_like(key: PRNGKeyArray, tree: ParametersType) -> ParametersType:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, len(leaves))
    new_leaves = [
        jrandom.normal(k, shape=jnp.shape(leaf)) for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def run_buffered_sgld(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: BufferedSGLDConfig,
) -> Batched[ParametersType, SequenceAxis | int]:
    """Run buffered SGLD updates over ``observations``."""

    smc = config.particle_filter
    if smc is None:
        raise ValueError("particle_filter must be provided in config")
    if smc.target is not target:
        smc = eqx.tree_at(lambda m: m.target, smc, target)
    prior = config.parameter_prior
    if prior is None:
        raise ValueError("parameter_prior must be provided in config")

    seq_len = jax.tree_util.tree_leaves(observations)[0].shape[0]
    if config.batch_size > seq_len:
        raise ValueError("batch_size must not exceed sequence length")

    start_max = seq_len - config.batch_size + 1
    n_iters = config.num_iters
    split_keys = jrandom.split(key, 2 * n_iters + 1)
    start_key, pf_keys, noise_keys = (
        split_keys[0],
        split_keys[1 : n_iters + 1],
        split_keys[n_iters + 1 :],
    )
    starts = jrandom.randint(start_key, shape=(n_iters,), minval=0, maxval=start_max)

    if jax.tree_util.tree_structure(config.step_size) == jax.tree_util.tree_structure(
        parameters
    ):  # type: ignore[operator]
        step_sizes = config.step_size
    else:
        step_sizes = jax.tree_util.tree_map(lambda _: config.step_size, parameters)

    def step(params: ParametersType, inp: tuple[PRNGKeyArray, PRNGKeyArray, jax.Array]):
        pf_key, noise_key, start = inp

        def log_post(p: ParametersType) -> jax.Array:
            log_mps = _run_segment(
                start,
                smc,
                pf_key,
                p,
                observations,
                condition_path,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
            )
            return prior.log_prob(p, config.hyperparameters) + jnp.sum(log_mps)

        grad = jax.grad(log_post)(params)
        noise = _tree_randn_like(noise_key, params)
        updates = jax.tree_util.tree_map(
            lambda g, n, s: 0.5 * s * g + jnp.sqrt(s) * n,
            grad,
            noise,
            step_sizes,
        )
        params = eqx.apply_updates(params, updates)
        return params, params

    _, samples = jax.lax.scan(step, parameters, (pf_keys, noise_keys, starts))
    return samples
