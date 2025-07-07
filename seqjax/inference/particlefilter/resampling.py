from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.util import dynamic_index_pytree_in_dim as index_tree
from seqjax.model.base import ParticleType

Resampler = Callable[
    [PRNGKeyArray, Array, tuple[ParticleType, ...], Scalar],
    tuple[tuple[ParticleType, ...], Array, Array],
]


def gumbel_resample_from_log_weights(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    _ess_e: Scalar,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample particles using the Gumbel-max trick."""
    gumbels = -jnp.log(
        -jnp.log(jrandom.uniform(key, (log_weights.shape[0], log_weights.shape[0])))
    )
    particle_ix = jnp.argmax(log_weights + gumbels, axis=1).reshape(-1)
    resampled_particles = jax.vmap(index_tree, in_axes=[None, 0, None])(
        particles, particle_ix, 0  # type: ignore[arg-type]
    )
    new_log_weights = jnp.full_like(log_weights, -jnp.log(log_weights.shape[0]))
    return resampled_particles, new_log_weights, particle_ix


def multinomial_resample_from_log_weights(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    _ess_e: Scalar,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample particles using standard multinomial sampling."""
    particle_ix = jrandom.categorical(
        key, log_weights, shape=(log_weights.shape[0],)
    )
    resampled_particles = jax.vmap(index_tree, in_axes=[None, 0, None])(
        particles,
        particle_ix,  # type: ignore[arg-type]
        0,
    )
    new_log_weights = jnp.full_like(log_weights, -jnp.log(log_weights.shape[0]))
    return resampled_particles, new_log_weights, particle_ix


def stratified_resample_from_log_weights(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    _ess_e: Scalar,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample particles using stratified resampling."""
    weights = jax.nn.softmax(log_weights)
    n = weights.shape[0]
    u = jrandom.uniform(key, shape=(n,))
    positions = (jnp.arange(n) + u) / n
    cumulative = jnp.cumsum(weights)
    particle_ix = jnp.searchsorted(cumulative, positions, side="right")
    resampled_particles = jax.vmap(index_tree, in_axes=[None, 0, None])(
        particles,
        particle_ix,  # type: ignore[arg-type]
        0,
    )
    new_log_weights = jnp.full_like(log_weights, -jnp.log(n))
    return resampled_particles, new_log_weights, particle_ix


def systematic_resample_from_log_weights(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    _ess_e: Scalar,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample particles using systematic resampling."""
    weights = jax.nn.softmax(log_weights)
    n = weights.shape[0]
    u0 = jrandom.uniform(key, minval=0.0, maxval=1.0 / n)
    positions = u0 + jnp.arange(n) / n
    cumulative = jnp.cumsum(weights)
    particle_ix = jnp.searchsorted(cumulative, positions, side="right")
    resampled_particles = jax.vmap(index_tree, in_axes=[None, 0, None])(
        particles,
        particle_ix,  # type: ignore[arg-type]
        0,
    )
    new_log_weights = jnp.full_like(log_weights, -jnp.log(n))
    return resampled_particles, new_log_weights, particle_ix


def conditional_resample(
    key: PRNGKeyArray,
    log_weights: Array,
    particles: tuple[ParticleType, ...],
    ess_e: Scalar,
    *,
    resampler: Resampler,
    esse_threshold: float,
) -> tuple[tuple[ParticleType, ...], Array, Array]:
    """Resample only when the ESS efficiency falls below ``esse_threshold``."""
    def _resample(p):
        return resampler(key, log_weights, p, ess_e)

    def _noresample(p):
        num = log_weights.shape[0]
        return p, log_weights, jnp.arange(num)

    return jax.lax.cond(ess_e < esse_threshold, _resample, _noresample, particles)
