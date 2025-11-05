from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.util import dynamic_index_pytree_in_dim as index_tree
from seqjax.model.typing import Latent as ParticleType

Resampler = Callable[
    [PRNGKeyArray, Array, tuple[ParticleType, ...], Scalar, int],
    tuple[tuple[ParticleType, ...], Array, Array],
]


def multinomial_resample_from_log_weights(
    key: PRNGKeyArray,
    raw_log_weights: Array,
    particles: tuple[ParticleType, ...],
    _ess_e: Scalar,
    num_resample: int,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample particles using standard multinomial sampling."""
    # jax.random.categorical requires unormalized logits
    particle_ix = jrandom.categorical(key, raw_log_weights, shape=(num_resample,))
    resampled_particles = jax.vmap(index_tree, in_axes=[None, 0, None])(
        particles,
        particle_ix,  # type: ignore[arg-type]
        0,
    )
    new_log_weights = jax.vmap(index_tree, in_axes=[None, 0, None])(
        raw_log_weights,
        particle_ix,  # type: ignore[arg-type]
        0,
    )
    return resampled_particles, new_log_weights, particle_ix


def conditional_resample(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    ess_e: Scalar,
    num_resample: int,
    *,
    resampler: Resampler,
    esse_threshold: float,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample only when the ESS efficiency falls below ``esse_threshold``."""

    def _resample(p):
        return resampler(key, log_weights, p, ess_e, num_resample)

    def _noresample(p):
        return p, log_weights, jnp.arange(log_weights.shape[0])

    return jax.lax.cond(ess_e < esse_threshold, _resample, _noresample, particles)
