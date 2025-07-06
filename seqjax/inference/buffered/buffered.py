from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Any

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
)
from seqjax.model.typing import Batched, SequenceAxis
from seqjax.util import (
    dynamic_index_pytree_in_dim,
    dynamic_slice_pytree,
)
from seqjax.inference.particlefilter import SMCSampler, run_filter


def _pad_edges_pytree(tree: Any, pad: int) -> Any:
    """Pad ``tree`` on the leading and trailing sequence axes using edge values."""

    def _pad(x: jax.Array) -> jax.Array:
        pad_width = [(pad, pad)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_width, mode="edge")

    return jax.tree_util.tree_map(_pad, tree)


class BufferedConfig(eqx.Module):
    """Configuration for :func:`run_buffered_filter`."""

    buffer_size: int = 0
    batch_size: int = 1
    particle_filter: (
        SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType] | None
    ) = None


def _run_segment(
    start: int | jax.Array,
    smc: SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observations: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None,
    *,
    buffer_size: int,
    batch_size: int,
) -> jax.Array:
    """Run filtering over a buffered segment starting at ``start``."""

    slice_size = batch_size + 2 * buffer_size

    obs_padded = _pad_edges_pytree(observations, buffer_size)
    if condition_path is not None:
        cond_padded = _pad_edges_pytree(condition_path, buffer_size)
    else:
        cond_padded = None

    obs_slice = dynamic_slice_pytree(obs_padded, start, slice_size)
    if condition_path is not None:
        cond_slice = dynamic_slice_pytree(cond_padded, start, slice_size)
    else:
        cond_slice = None

    if smc.target.prior.order > 0:
        init_conds = tuple(
            dynamic_index_pytree_in_dim(cond_padded, start + i, 0)
            if cond_padded is not None
            else None
            for i in range(smc.target.prior.order)
        )
    else:
        init_conds = ()

    _, _, log_mp_hist, _, _, _ = run_filter(
        smc,
        key,
        parameters,
        obs_slice,
        cond_slice,
        initial_conditions=init_conds,
    )

    return log_mp_hist[buffer_size : buffer_size + batch_size]


def run_buffered_filter(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: BufferedConfig,
) -> jax.Array:
    """Run buffered particle filtering over ``observations``."""

    smc = config.particle_filter
    if smc is None:
        raise ValueError("particle_filter must be provided in config")
    if smc.target is not target:
        smc = eqx.tree_at(lambda m: m.target, smc, target)
    seq_len = jax.tree_util.tree_leaves(observations)[0].shape[0]
    starts = jnp.arange(0, seq_len, config.batch_size)
    keys = jax.random.split(key, len(starts))

    def step(_, inp):
        s, k = inp
        logmps = _run_segment(
            s,
            smc,
            k,
            parameters,
            observations,
            condition_path,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
        )
        return None, logmps

    _, seg_logmps = jax.lax.scan(step, None, (starts, keys))

    return jnp.concatenate(seg_logmps)[:seq_len]
